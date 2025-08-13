# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from xformers.ops import RMSNorm
from xformers.ops.fmha.attn_bias import (
    BlockDiagonalCausalWithOffsetPaddedKeysMask as AttnBias,
)

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

from bitnet_vllm.layers.rotary_embedding import get_rope
from bitnet_vllm.utils.context import get_context
from bitnet_vllm.utils.kvcache import store_kvcache
# import torch.distributed as dist


import ctypes
bitnet_lib = ctypes.CDLL('bitnet_kernels/libbitnet.so')
gemm_lib = ctypes.CDLL('bitnet_kernels/libgemm.so')

import numpy as np

def bitnet_int8xint2_linear_gemv(input0, input1, s, ws):
    out_shape = list(input0.shape)
    out_shape[-1] = input1.shape[0]

    stream = torch.cuda.current_stream()

    M = input0.shape[0]
    if len(out_shape) == 3: 
        M *= input0.shape[1]
    N = input1.shape[0]
    K = input1.shape[1] * 4

    ret = torch.zeros(*out_shape, dtype=torch.bfloat16, device=input0.device)

    bitnet_lib.bitlinear_int8xint2(*[ctypes.c_void_p(input0.data_ptr()), ctypes.c_void_p(input1.data_ptr()), ctypes.c_void_p(ret.data_ptr()), ctypes.c_void_p(s.data_ptr()), ctypes.c_void_p(ws.data_ptr()), ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K), ctypes.c_void_p(stream.cuda_stream)])

    return ret

def bitnet_int8xint2_linear_gemm(input0, input1, s, ws):
    out_shape = list(input0.shape)
    out_shape[-1] = input1.shape[0]

    stream = torch.cuda.current_stream()

    M = input0.shape[0]
    if len(out_shape) == 3: 
        M *= input0.shape[1]
    N = input1.shape[0]
    K = input1.shape[1] * 4

    ret = torch.zeros(*out_shape, dtype=torch.int32, device=input0.device)

    gemm_lib.bitlinear_int8xint2(*[ctypes.c_void_p(input0.data_ptr()), ctypes.c_void_p(input1.data_ptr()), ctypes.c_void_p(ret.data_ptr()), ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K), ctypes.c_void_p(stream.cuda_stream)])
    ret = ret.to(torch.bfloat16)
    ret = ret / s
    if N == 3840 and K == 2560:
        #split last dim to 6 parts evenly
        ret = ret.reshape(*ret.shape[:-1], 6, -1)
        # devide each part by first 6 coresponding weight scale
        ret = ret * ws[:6].reshape(1, 6, 1)
    elif (N == 2560 and K == 2560):
        # 1 part
        ret = ret* ws[:1].reshape(1, 1, 1, 1)
    elif (N == 13824 and K == 2560):
        # 2 parts
        ret = ret.reshape(*ret.shape[:-1], 2, -1)
        # devide each part by first 2 coresponding weight scale
        ret = ret * ws[:2].reshape(1, 1, 2, 1)
    elif (N == 2560 and K == 6912):
        # 1 part
        ret = ret * ws[:1].reshape(1, 1, 1, 1)

    return ret.reshape(*out_shape)

@dataclass
class ModelArgs:
    dim: int = 2560
    n_layers: int = 30
    n_heads: int = 20
    n_kv_heads: int = 5
    vocab_size: int = 128256
    ffn_dim: int = 6912
    norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    use_kernel: bool = False
    max_position: int = 4096

# LayerCache = Tuple[torch.Tensor, torch.Tensor]

class BitLinearKernel(nn.Module):
    in_features: int
    out_features: int
    weight: torch.Tensor
    weight_scale: torch.Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = torch.nn.Parameter(torch.zeros(out_features, in_features//4, dtype=torch.int8), requires_grad=False)
        self.weight_scale = torch.nn.Parameter(torch.zeros(6, dtype=torch.bfloat16), requires_grad=False)

    @torch.compile
    def quant_input(self, input):
        s = 127 / input.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        return (input * s).round().clamp(-128, 127).to(torch.int8), s

    def forward(self, input):
        input, s = self.quant_input(input)
        if input.shape[0] == 1:
            return bitnet_int8xint2_linear_gemv(input, self.weight, s, self.weight_scale)
        else:
            return bitnet_int8xint2_linear_gemm(input, self.weight, s, self.weight_scale)

class BitLinear(nn.Linear):
    @torch.compile
    def quant_input(self, input):
        s = 127 / input.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        return (input * s).round().clamp(-128, 127) / s

    def forward(self, input):
        input = self.quant_input(input)
        return F.linear(input, self.weight)

class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        head_dim: int,
        n_heads: int,
        n_kv_heads: int,
        rope_theta: float,
        norm_eps: float,
        max_position: int,
        use_kernel: bool
    ):
        super().__init__()

        self.head_dim = head_dim
        self.rope_theta = rope_theta

        self.n_local_heads = n_heads
        self.n_local_kv_heads = n_kv_heads
        self.scale = self.head_dim**-0.5
        
        self.k_cache = self.v_cache = torch.tensor([])

        Linear = BitLinearKernel if use_kernel else BitLinear

        self.wqkv = Linear(
            dim,
            (self.n_local_heads + 2 * self.n_local_kv_heads) * head_dim,
            bias=False,
        )
        self.wo = Linear(
            self.n_local_heads * head_dim,
            dim,
            bias=False,
        )
        
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta
        )

        self.attn_sub_norm = RMSNorm(dim, norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        xqkv = self.wqkv(x)
        xq = xqkv[:, : (self.n_local_heads * self.head_dim)]
        xkv = xqkv[:, (self.n_local_heads * self.head_dim) :]
        xk, xv = xkv.chunk(2, 1)

        output_shape = xq.shape
        heads_per_group = self.n_local_heads // self.n_local_kv_heads

        xq = xq.view(xq.shape[0], self.n_local_kv_heads * heads_per_group, self.head_dim)
        xk = xk.view(xk.shape[0], self.n_local_kv_heads, self.head_dim)
        xv = xv.view(xv.shape[0], self.n_local_kv_heads, self.head_dim)
        xq, xk = self.rotary_emb(positions, xq, xk)

        context = get_context()

        cache_k, cache_v = self.k_cache, self.v_cache
            
        if cache_k.numel() and cache_v.numel():
            store_kvcache(xk, xv, cache_k, cache_v, context.slot_mapping)

        if context.is_prefill:
            if context.block_tables is not None: 
                xk, xv = cache_k, cache_v
            output = flash_attn_varlen_func(xq, xk, xv,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            output = flash_attn_with_kvcache(xq.unsqueeze(1), cache_k, cache_v,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables,
                                        softmax_scale=self.scale, causal=True)

        output = output.reshape(output_shape)
        output = self.attn_sub_norm(output)
        output = self.wo(output)

        return output

@torch.compile
def squared_relu(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x) ** 2

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        norm_eps: float,
        use_kernel: bool,
    ):
        super().__init__()

        Linear = BitLinearKernel if use_kernel else BitLinear

        self.w13 = Linear(
            dim,
            2 * hidden_dim,
            bias=False,
        )
        self.w2 = Linear(
            hidden_dim,
            dim,
            bias=False,
        )
        self.ffn_sub_norm = RMSNorm(hidden_dim, norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x13 = self.w13(x)
        x1, x3 = x13.chunk(2, -1)
        inner = self.ffn_sub_norm(squared_relu(x1) * x3)
        output = self.w2(inner)
        return output


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        assert args.dim % args.n_heads == 0
        head_dim = args.dim // args.n_heads
        if args.n_kv_heads is not None:
            n_kv_heads = args.n_kv_heads
        else:
            n_kv_heads = args.n_heads

        assert args.n_heads % n_kv_heads == 0

        self.attention = Attention(
            dim=args.dim,
            head_dim=head_dim,
            n_heads=args.n_heads,
            n_kv_heads=n_kv_heads,
            rope_theta=args.rope_theta,
            norm_eps=args.norm_eps,
            max_position=args.max_position,
            use_kernel=args.use_kernel
        )
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=args.ffn_dim,
            norm_eps=args.norm_eps,
            use_kernel=args.use_kernel,
        )
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        h = x + self.attention.forward(
            self.attention_norm(x),
            positions
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        assert args.vocab_size > 0

        self.tok_embeddings = nn.Embedding(
            num_embeddings=args.vocab_size,
            embedding_dim=args.dim,
        )

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(TransformerBlock(args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        
        self.output = nn.Linear(
            args.dim,
            args.vocab_size,
            bias=False,
        )

    @torch.no_grad()
    def forward(
        self,
        token_values: torch.Tensor,
        positions: torch.Tensor
    ) -> torch.Tensor:
        h = self.tok_embeddings(token_values)

        for _, layer in enumerate(self.layers):
            h = layer(h, positions)

        return h
    
    @torch.no_grad()
    def compute_logits(
        self,
        h: torch.Tensor    
    ) -> torch.Tensor:
        h = self.norm(h)
        context = get_context()
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            h = h[last_indices].contiguous()
        logits = self.output(h)
        return logits.float()
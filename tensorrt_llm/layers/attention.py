# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from typing import List, Optional

import numpy as np
import tensorrt as trt

from .._common import default_net, precision
from .._utils import numpy_fp32_to_bf16, trt_dtype_to_np
from ..functional import (AttentionMaskType, PositionEmbeddingType,
                          RotaryScalingType, Tensor, bert_attention, cast, clip,
                          concat, constant, embedding, expand_dims, expand_mask,
                          generate_alibi_biases, generate_alibi_slopes,
                          gpt_attention, matmul, repeat_interleave, round,
                          shape, slice, softmax, split, view, where)
from ..module import Module
from ..parameter import Parameter
from ..quantization import QuantMode
from ..quantization.functional import dequantize, quantize
from ..quantization.layers import FP8Linear, FP8RowLinear
from .linear import ColumnLinear, RowLinear
from .lora import Lora


class RopeEmbeddingUtils:

    @staticmethod
    def create_sinusoidal_positions(num_pos: int,
                                    dim: int,
                                    theta: float = 10000.0,
                                    dtype=np.float32):
        inv_freq = 1.0 / (theta**(np.arange(0, dim, 2) / dim)).astype(dtype)
        sinusoid_inp = np.einsum("i , j -> i j",
                                 np.arange(num_pos, dtype=dtype),
                                 inv_freq,
                                 dtype=dtype)
        concat = np.concatenate((np.sin(sinusoid_inp), np.cos(sinusoid_inp)),
                                axis=1)
        return np.expand_dims(concat, axis=0).astype(np.float32)

    @staticmethod
    def rotate_every_two(tensor: Tensor) -> Tensor:
        assert tensor.ndim() == 4

        shape_tensor = concat([
            shape(tensor, i) / 2 if i == (tensor.ndim() -
                                          1) else shape(tensor, i)
            for i in range(tensor.ndim())
        ])
        x1 = slice(tensor, [0, 0, 0, 0], shape_tensor, [1, 1, 1, 2])
        x2 = slice(tensor, [0, 0, 0, 1], shape_tensor, [1, 1, 1, 2])
        x1 = expand_dims(x1, 4)
        x2 = expand_dims(x2, 4)
        zero = constant(
            np.ascontiguousarray(np.zeros([1],
                                          dtype=trt_dtype_to_np(x2.dtype))))
        x2 = zero - x2
        x = concat([x2, x1], 4)
        return view(
            x, concat([shape(x, 0),
                       shape(x, 1),
                       shape(x, 2),
                       shape(x, 3) * 2]))

    @staticmethod
    def rotate_half(tensor: Tensor) -> Tensor:
        # [bs, num_attention_kv_heads, seqlen, attention_head_size]
        assert tensor.ndim() == 4
        shape_tensor = concat([
            shape(tensor, i) / 2 if i == (tensor.ndim() -
                                          1) else shape(tensor, i)
            for i in range(tensor.ndim())
        ])
        last_dim = shape(tensor, tensor.ndim() - 1) / 2
        x1 = slice(tensor, [0, 0, 0, 0], shape_tensor, [1, 1, 1, 1])
        x2 = slice(tensor, concat([0, 0, 0, last_dim]), shape_tensor,
                   [1, 1, 1, 1])
        zero = constant(
            np.ascontiguousarray(np.zeros([1],
                                          dtype=trt_dtype_to_np(x2.dtype))))
        x2 = zero - x2
        x = concat([x2, x1], 3)
        return x

    @staticmethod
    def apply_rotary_pos_emb(
        tensor: Tensor,
        position_embedding: List[Tensor] = None,
        pos_emb_type: PositionEmbeddingType = PositionEmbeddingType.rope_gptj
    ) -> Tensor:

        rotate_func = None
        if pos_emb_type == PositionEmbeddingType.rope_gpt_neox:
            assert len(position_embedding) == 2
            cos, sin = position_embedding
            sin = expand_dims(sin, 2)
            cos = expand_dims(cos, 2)
            sin = concat([sin, sin], 3)
            cos = concat([cos, cos], 3)
            rotate_func = RopeEmbeddingUtils.rotate_half
        elif pos_emb_type == PositionEmbeddingType.rope_gptj:
            assert len(position_embedding) == 2
            cos, sin = position_embedding
            sin = expand_dims(sin, 2)
            cos = expand_dims(cos, 2)
            sin = repeat_interleave(sin, 2, 3)
            cos = repeat_interleave(cos, 2, 3)
            rotate_func = RopeEmbeddingUtils.rotate_every_two
        elif pos_emb_type == PositionEmbeddingType.chatglm:
            assert len(position_embedding) == 4
            cos0, cos1, sin0, sin1 = position_embedding
            if default_net().strongly_typed and tensor.dtype != cos0.dtype:
                tensor = cast(tensor, cos0.dtype)
            shape_tensor = concat([
                shape(tensor, i) / 2 if i == (tensor.ndim() -
                                              1) else shape(tensor, i)
                for i in range(tensor.ndim())
            ])
            last_dim = shape(tensor, tensor.ndim() - 1) / 2
            x_part0 = slice(tensor, [0, 0, 0, 0], shape_tensor, [1, 1, 1, 1])
            x_part1 = slice(tensor, concat([0, 0, 0, last_dim]), shape_tensor,
                            [1, 1, 1, 1])

            y_part0 = (x_part0 *
                       cos0) + (RopeEmbeddingUtils.rotate_half(x_part0) * sin0)
            y_part1 = (x_part1 *
                       cos1) + (RopeEmbeddingUtils.rotate_half(x_part1) * sin1)

            result = concat([y_part0, y_part1], dim=3)
            return result.view(shape(tensor))

        else:
            raise ValueError('The PositionEmbeddingType is not RoPE')
        return (tensor * cos) + (rotate_func(tensor) * sin)

    @staticmethod
    def apply_rotary_pos_emb_chatglm(
        qkv,
        position_embedding,
        num_attention_heads,
        attention_head_size,
        max_position_embeddings,
        rotary_embedding_scale,
    ) -> Tensor:

        half_head_size = attention_head_size // 2
        qkv_shape = shape(qkv)
        qkv = qkv.view(
            concat([
                shape(qkv, 0),
                shape(qkv, 1),
                num_attention_heads,
                3,
                attention_head_size,
            ]))
        query, key, value = split(qkv, 1, dim=3)
        q_shape = concat([
            shape(qkv, 0),
            shape(qkv, 1),
            num_attention_heads,
            attention_head_size,
        ])
        query = query.view(q_shape)
        key = key.view(q_shape)
        value = value.view(q_shape)

        embedding_weight = RopeEmbeddingUtils.create_sinusoidal_positions(
            max_position_embeddings, half_head_size)
        embedding_weight /= rotary_embedding_scale
        embedding_weight = np.split(embedding_weight.squeeze(0), 2, axis=1)
        embedding_weight = np.concatenate(
            [
                embedding_weight[0],
                embedding_weight[0],
                embedding_weight[1],
                embedding_weight[1],
            ],
            axis=1,
        )

        embedding_weight = constant(embedding_weight)
        position_embedding = embedding(position_embedding, embedding_weight)
        position_embedding, block_embedding = split(
            position_embedding,
            1,
            dim=1,
        )
        sin0, cos0 = split(position_embedding, half_head_size, dim=3)
        sin1, cos1 = split(block_embedding, half_head_size, dim=3)

        new_shape = concat([
            shape(qkv, 0),
            shape(qkv, 1),
            1,
            half_head_size,
        ])
        position_embedding = [
            tensor.view(new_shape) for tensor in [cos0, cos1, sin0, sin1]
        ]

        query = RopeEmbeddingUtils.apply_rotary_pos_emb(
            tensor=query,
            position_embedding=position_embedding,
            pos_emb_type=PositionEmbeddingType.chatglm)
        key = RopeEmbeddingUtils.apply_rotary_pos_emb(
            tensor=key,
            position_embedding=position_embedding,
            pos_emb_type=PositionEmbeddingType.chatglm)

        if default_net().strongly_typed:
            if query.dtype != value.dtype:
                query = cast(query, value.dtype)
            if key.dtype != value.dtype:
                key = cast(key, value.dtype)

        qkv = concat([query, key, value], dim=2)
        qkv = qkv.view(qkv_shape)

        return qkv


class AttentionParams(object):

    def __init__(self,
                 sequence_length: Tensor = None,
                 context_lengths: Tensor = None,
                 host_context_lengths: Tensor = None,
                 max_context_length: int = None,
                 host_request_types: Tensor = None,
                 encoder_input_lengths: Tensor = None,
                 encoder_max_input_length: Tensor = None):
        self.sequence_length = sequence_length
        self.context_lengths = context_lengths
        self.host_context_lengths = host_context_lengths
        # max allowed context length. Required to
        # compute scratch memory size.
        self.max_context_length = max_context_length
        self.host_request_types = host_request_types

        self.encoder_input_lengths = encoder_input_lengths
        self.encoder_max_input_length = encoder_max_input_length

    def is_valid_cross_attn(self, do_cross_attention):
        if do_cross_attention:
            if self.encoder_input_lengths is None:
                return False
            if self.encoder_max_input_length is None:
                return False
        return True

    def is_valid(self, gpt_attention_plugin, remove_input_padding):
        if gpt_attention_plugin:
            if self.sequence_length is None:
                return False
            if self.context_lengths is None:
                return False
            if self.host_request_types is None:
                return False
            if self.max_context_length is None:
                return False

        if remove_input_padding:
            if self.host_context_lengths is None:
                return False
            if not gpt_attention_plugin:
                return False

        return True


class KeyValueCacheParams:

    def __init__(self,
                 past_key_value: List[Tensor] = None,
                 host_past_key_value_lengths: Tensor = None,
                 host_max_kv_cache_lengths: List[Tensor] = None,
                 kv_cache_block_pointers: List[Tensor] = None,
                 host_kv_cache_block_pointers: List[Tensor] = None,
                 cache_indirection: Tensor = None,
                 past_key_value_length: Tensor = None):
        self.past_key_value = past_key_value
        self.host_past_key_value_lengths = host_past_key_value_lengths
        self.host_max_kv_cache_lengths = host_max_kv_cache_lengths
        self.kv_cache_block_pointers = kv_cache_block_pointers
        self.host_kv_cache_block_pointers = host_kv_cache_block_pointers
        self.cache_indirection = cache_indirection
        # self.past_key_value_length = past_key_value_length

    def get_first_past_key_value(self):
        if self.past_key_value is None:
            return None
        return self.past_key_value[0]

    def get_first_kv_cache_block_pointers(self):
        if self.kv_cache_block_pointers is None:
            return None
        return self.kv_cache_block_pointers[0]

    def get_first_host_kv_cache_block_pointers(self):
        if self.host_kv_cache_block_pointers is None:
            return None
        return self.host_kv_cache_block_pointers[0]

    def fill_none_tensor_list(self, list_size):
        if self.past_key_value is None:
            self.past_key_value = tuple([None] * list_size)
        if self.host_max_kv_cache_lengths is None:
            self.host_max_kv_cache_lengths = tuple([None] * list_size)

    def is_valid(self, gpt_attention_plugin):
        if gpt_attention_plugin:
            if self.host_past_key_value_lengths is None:
                return False
            if self.host_max_kv_cache_lengths is None:
                return False
            if self.cache_indirection is None:
                return False

        return True


class Attention(Module):

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        num_kv_heads=None,
        max_position_embeddings=1024,
        num_layers=1,
        apply_query_key_layer_scaling=False,
        attention_head_size=None,
        attention_mask_type=AttentionMaskType.padding,
        bias=True,
        dtype=None,
        position_embedding_type=PositionEmbeddingType.learned_absolute,
        rotary_embedding_base=10000.0,
        rotary_embedding_scaling=None,
        use_int8_kv_cache=False,
        rotary_embedding_percentage=1.0,
        tp_group=None,
        tp_size=1,
        tp_rank=0,
        quant_mode: QuantMode = QuantMode(0),
        q_scaling=1.0,
        cross_attention=False,
        relative_attention=False,
        max_distance=0,
        num_buckets=0,
        instance_id: int = 0,
        dense_bias=None,
    ):
        super().__init__()

        self.cross_attention = cross_attention
        self.attention_mask_type = attention_mask_type
        self.attention_head_size = hidden_size // num_attention_heads if attention_head_size is None else attention_head_size
        assert num_attention_heads % tp_size == 0, \
        "num_attention_heads must be divisible by tp_size"
        self.num_attention_heads = num_attention_heads // tp_size
        self.num_attention_kv_heads = (
            num_kv_heads + tp_size - 1
        ) // tp_size if num_kv_heads is not None else self.num_attention_heads
        self.hidden_size = hidden_size // tp_size
        self.max_position_embeddings = max_position_embeddings
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.dtype = dtype
        if dense_bias is None:
            dense_bias = bias

        self.num_layers = num_layers
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.norm_factor = math.sqrt(self.attention_head_size)
        self.q_scaling = q_scaling
        if self.apply_query_key_layer_scaling:
            self.norm_factor *= self.num_layers
            self.q_scaling *= self.num_layers
        # Whether to scale ALiBi bias. Mathematically, it's equivalent to
        # normalizing QK after adding bias.
        #   - False, inv_sqrt_Dh * Q*K^T + alibi_bias
        #   - True,  inv_sqrt_Dh * Q*K^T + inv_sqrt_Dh * alibi_bias
        self.scale_alibi_bias = position_embedding_type == PositionEmbeddingType.alibi_with_scale
        self.position_embedding_type = position_embedding_type
        self.relative_attention = relative_attention
        self.max_distance = max_distance
        self.rotary_embedding_base = rotary_embedding_base
        self.rotary_embedding_scale_type = RotaryScalingType.none
        self.rotary_embedding_scale = 1.0
        if rotary_embedding_scaling is not None:
            assert rotary_embedding_scaling["type"] in ["linear", "dynamic"]
            self.rotary_embedding_scale_type = RotaryScalingType.linear if rotary_embedding_scaling[
                "type"] == "linear" else RotaryScalingType.dynamic
            self.rotary_embedding_scale = rotary_embedding_scaling["factor"]
            assert self.rotary_embedding_scale > 1.0

        self.embed_positions = None
        self.rotary_enabled = False
        self.rotary_embedding_dim = 0

        if self.position_embedding_type.is_rope():
            self.rotary_embedding_dim = int(self.attention_head_size *
                                            rotary_embedding_percentage)
            self.rotary_enabled = True
            self.embed_positions = RopeEmbeddingUtils.create_sinusoidal_positions(
                self.max_position_embeddings,
                self.rotary_embedding_dim,
            )

        self.quant_mode = quant_mode
        if use_int8_kv_cache:
            # TODO: remove use_int8_kv_cache as can be replaced by quant_mode.has_kv_cache_quant()
            # Merge int8 setting into quant_mode
            self.quant_mode = self.quant_mode.set_int8_kv_cache()
        self.use_int8_kv_cache = use_int8_kv_cache
        if self.quant_mode.has_kv_cache_quant():
            self.kv_orig_quant_scale = Parameter(shape=(1, ), dtype='float32')
            self.kv_quant_orig_scale = Parameter(shape=(1, ), dtype='float32')
        else:
            self.register_parameter('kv_orig_quant_scale', None)
            self.register_parameter('kv_quant_orig_scale', None)

        # The output feature size is therefore (h/tp + 2*kvh/tp) * d, where h is num_heads,
        # d is head_size, kvh is the num_kv_heads and tp is tensor_parallel_size.
        # In ColumnLinear op, the output dim is calculated by (h + 2*kvh) * d / tp,
        # which matches the desired output size (h/tp + 2*kvh/tp) * d after splitting

        self.use_fp8_qdq = self.quant_mode.has_fp8_qdq()
        if self.use_fp8_qdq:
            self.qkv = FP8Linear(
                hidden_size,
                tp_size * self.num_attention_heads * self.attention_head_size +
                (2 * tp_size * self.num_attention_kv_heads *
                 self.attention_head_size),
                bias=bias,
                dtype=dtype,
                tp_group=tp_group,
                tp_size=tp_size,
                gather_output=False)
            self.dense = FP8RowLinear(hidden_size,
                                      hidden_size,
                                      bias=dense_bias,
                                      dtype=dtype,
                                      tp_group=tp_group,
                                      tp_size=tp_size,
                                      instance_id=instance_id)
        else:
            # out dim is not necessarily hidden_size + kv specific size (in MQA/GQA), but num_heads * heads_size
            # example: d_model != num_heads * head_size in Flan-T5
            self.qkv = ColumnLinear(
                hidden_size,
                tp_size * self.num_attention_heads * self.attention_head_size +
                (2 * tp_size * self.num_attention_kv_heads *
                 self.attention_head_size),
                bias=bias,
                dtype=dtype,
                tp_group=tp_group,
                tp_size=tp_size,
                gather_output=False)
            self.dense = RowLinear(tp_size * self.num_attention_heads *
                                   self.attention_head_size,
                                   hidden_size,
                                   bias=dense_bias,
                                   dtype=dtype,
                                   tp_group=tp_group,
                                   tp_size=tp_size,
                                   instance_id=instance_id)

        # per-layer relative attention table
        if relative_attention:
            self.rel_attn_table = Parameter(shape=(num_attention_heads //
                                                   tp_size, num_buckets),
                                            dtype=dtype)

        self.q_lora = Lora(
            in_hidden_size=hidden_size,
            out_hidden_size=self.num_attention_heads * self.attention_head_size,
            max_low_rank=min(
                hidden_size,
                self.num_attention_heads * self.attention_head_size),
        )

        self.k_lora = Lora(
            in_hidden_size=hidden_size,
            out_hidden_size=self.num_attention_kv_heads *
            self.attention_head_size,
            max_low_rank=min(
                hidden_size,
                self.num_attention_kv_heads * self.attention_head_size),
        )

        self.v_lora = Lora(
            in_hidden_size=hidden_size,
            out_hidden_size=self.num_attention_kv_heads *
            self.attention_head_size,
            max_low_rank=min(
                hidden_size,
                self.num_attention_kv_heads * self.attention_head_size),
        )

    def forward(self,
                hidden_states: Tensor,
                attention_mask=None,
                use_cache=False,
                kv_cache_params=None,
                attention_params=None,
                encoder_output: Optional[Tensor] = None,
                workspace=None,
                position_embedding=None,
                norm_before_bmm1=False,
                lora_param=None):

        assert isinstance(hidden_states, Tensor)

        alibi_slopes = None
        if self.position_embedding_type.is_alibi():
            dtype = trt.float32
            if default_net().plugin_config.gpt_attention_plugin:
                dtype = hidden_states.dtype
            alibi_scale = 1. / self.norm_factor if self.scale_alibi_bias else 1.
            alibi_slopes = generate_alibi_slopes(self.num_attention_heads *
                                                 self.tp_size,
                                                 dtype=dtype,
                                                 tp_size=self.tp_size,
                                                 tp_rank=self.tp_rank,
                                                 alibi_scale=alibi_scale)

        qkv_lora_param = None
        if lora_param is not None:
            qkv_lora_param = lora_param.get_runtime_params(0, "attn_qkv")
        qkv = self.qkv(hidden_states, qkv_lora_param)

        if default_net().plugin_config.lora_plugin and lora_param is not None:
            q_lora_param = lora_param.get_runtime_params(0, "attn_q")
            k_lora_param = lora_param.get_runtime_params(0, "attn_k")
            v_lora_param = lora_param.get_runtime_params(0, "attn_v")

            assert (q_lora_param is not None and k_lora_param is not None and v_lora_param is not None) or \
                (q_lora_param is None and k_lora_param is None and v_lora_param is None), "q_lora_param, k_lora_param and v_lora_param should be all enabled or all disabled at the same time."
            if q_lora_param is not None and k_lora_param is not None and v_lora_param is not None:
                assert qkv_lora_param is None, "Cannot enable qkv_lora_param and split q_lora_parm, k_lora_param, v_lora_param at the same time."
                q_lora = self.q_lora(hidden_states,
                                     lora_runtime_param=q_lora_param)
                k_lora = self.k_lora(hidden_states,
                                     lora_runtime_param=k_lora_param)
                v_lora = self.v_lora(hidden_states,
                                     lora_runtime_param=v_lora_param)
                qkv_lora = concat([q_lora, k_lora, v_lora], dim=2)
                qkv = qkv + qkv_lora

        if self.position_embedding_type == PositionEmbeddingType.chatglm:
            qkv = RopeEmbeddingUtils.apply_rotary_pos_emb_chatglm(
                qkv,
                position_embedding,
                self.num_attention_heads,
                self.attention_head_size,
                self.max_position_embeddings,
                self.rotary_embedding_scale,
            )
            self.rotary_embedding_scale_type = RotaryScalingType.none
            self.rotary_embedding_scale = 1.0

        paged_kv_cache = default_net().plugin_config.paged_kv_cache

        assert attention_params is None or attention_params.is_valid(
            default_net().plugin_config.gpt_attention_plugin,
            default_net().plugin_config.remove_input_padding)
        assert kv_cache_params is None or kv_cache_params.is_valid(
            default_net().plugin_config.gpt_attention_plugin)

        past_key_value = None if kv_cache_params is None else kv_cache_params.get_first_past_key_value(
        )
        if self.cross_attention and (past_key_value is not None):
            past_key_value = kv_cache_params.past_key_value[1]

        # if cross attention, cross QKV only needs to be calculated once in the
        # 1st decoding step --> write to cross KV cache --> remains constant
        # during the entire decoding. 1st and >1 steps are distinguished by
        # whether past_key_value exists or not
        # also, cross KV cache max length is set from encoder output seqlen,
        # this maps to the max context length concept in decoder-only models
        cross_qkv = None
        # get length data in every run
        if encoder_output:
            assert isinstance(encoder_output, Tensor)
        # but only do projection once at 1st decoding step
        if self.cross_attention and encoder_output:
            cross_qkv = self.qkv(encoder_output)

        if default_net().plugin_config.gpt_attention_plugin:
            assert self.attention_mask_type in [
                AttentionMaskType.causal, AttentionMaskType.bidirectional,
                AttentionMaskType.bidirectionalglm
            ], 'Plugin only support masked MHA.'
            kv_orig_quant_scale = self.kv_orig_quant_scale.value if self.quant_mode.has_kv_cache_quant(
            ) else None
            kv_quant_orig_scale = self.kv_quant_orig_scale.value if self.quant_mode.has_kv_cache_quant(
            ) else None
            context, past_key_value = gpt_attention(
                qkv=qkv,
                past_key_value=past_key_value,
                sequence_length=attention_params.sequence_length,
                host_past_key_value_lengths=kv_cache_params.
                host_past_key_value_lengths,
                host_max_kv_cache_lengths=kv_cache_params.
                host_max_kv_cache_lengths,
                context_lengths=attention_params.context_lengths,
                cache_indirection=kv_cache_params.cache_indirection,
                host_request_types=attention_params.host_request_types,
                num_heads=self.num_attention_heads,
                num_kv_heads=self.num_attention_kv_heads,
                hidden_size_per_head=self.attention_head_size,
                q_scaling=self.q_scaling,
                rotary_embedding_dim=self.rotary_embedding_dim,
                rotary_embedding_base=self.rotary_embedding_base,
                rotary_embedding_scale_type=self.rotary_embedding_scale_type,
                rotary_embedding_scale=self.rotary_embedding_scale,
                rotary_embedding_max_positions=self.max_position_embeddings,
                position_embedding_type=self.position_embedding_type,
                kv_orig_quant_scale=kv_orig_quant_scale,
                kv_quant_orig_scale=kv_quant_orig_scale,
                kv_cache_quant_mode=self.quant_mode,
                max_context_length=attention_params.max_context_length,
                mask_type=self.attention_mask_type,
                alibi_slopes=alibi_slopes,
                tp_size=self.tp_size,
                tp_rank=self.tp_rank,
                kv_cache_block_pointers=kv_cache_params.
                get_first_kv_cache_block_pointers(),
                host_kv_cache_block_pointers=kv_cache_params.
                get_first_host_kv_cache_block_pointers(),
                do_cross_attention=self.cross_attention,
                cross_qkv=cross_qkv,
                cross_qkv_length=attention_params.encoder_max_input_length,
                encoder_input_lengths=attention_params.encoder_input_lengths,
                relative_attention_bias=self.rel_attn_table.value
                if self.relative_attention else None,
                max_distance=self.max_distance,
                host_context_lengths=attention_params.host_context_lengths,
                use_cache=use_cache,
            )

        else:
            # plain TensorRT mode
            assert paged_kv_cache == False
            past_key_value = None if kv_cache_params is None else kv_cache_params.get_first_past_key_value(
            )

            def transpose_for_scores(x,
                                     rotary: bool = False,
                                     is_kv: bool = False):
                _num_attention_heads = self.num_attention_kv_heads if is_kv else self.num_attention_heads
                new_x_shape = concat([
                    shape(x, 0),
                    shape(x, 1), _num_attention_heads, self.attention_head_size
                ])
                if rotary:
                    return x.view(new_x_shape)
                else:
                    return x.view(new_x_shape).permute([0, 2, 1, 3])

            # qkv after projection is of shape
            #   [bs, seqlen, (num_attention_heads + 2 * num_attention_kv_heads), attention_head_size].
            # The projected and split qkv after transpose_for_scores():
            #   Q[bs, num_attention_heads, seqlen, attention_head_size]
            #   K[bs, num_attention_kv_heads, seqlen, attention_head_size]
            #   V[bs, num_attention_kv_heads, seqlen, attention_head_size]
            kv_size = self.attention_head_size * self.num_attention_kv_heads
            query, key, value = split(qkv, [self.hidden_size, kv_size, kv_size],
                                      dim=2)

            # in cross attention mode, replace kv by encoder_output
            if self.cross_attention and encoder_output is not None:
                encoder_qkv = self.qkv(encoder_output)
                _, key, value = split(encoder_qkv,
                                      [self.hidden_size, kv_size, kv_size],
                                      dim=2)

            query = transpose_for_scores(query, rotary=self.rotary_enabled)
            key = transpose_for_scores(key,
                                       is_kv=True,
                                       rotary=self.rotary_enabled)
            value = transpose_for_scores(value, is_kv=True)

            if self.rotary_enabled:
                if self.dtype == trt.bfloat16:
                    embed_positions = numpy_fp32_to_bf16(
                        self.embed_positions.astype(np.float32))
                    embed_positions = constant(embed_positions)
                else:
                    embed_positions = constant(self.embed_positions)

                if default_net().strongly_typed and (embed_positions.dtype !=
                                                     value.dtype):
                    embed_positions = cast(embed_positions, value.dtype)

                if self.rotary_embedding_dim is not None:
                    # When shape(hidden_states, 1) > 1(Context phase), the embedding start from 0,
                    # otherwise (Generation phase) move start to position
                    start = where(
                        shape(hidden_states, 1) > 1, 0,
                        shape(past_key_value, 3))
                    size = where(
                        shape(hidden_states, 1) > 1, shape(hidden_states, 1), 1)
                    sincos = slice(embed_positions, concat([0, start, 0]),
                                   concat([1, size, self.rotary_embedding_dim]))
                    sin, cos = split(sincos,
                                     self.rotary_embedding_dim // 2,
                                     dim=-1)

                    key_rot_size = concat([
                        shape(key, 0),
                        shape(key, 1),
                        shape(key, 2), self.rotary_embedding_dim
                    ])
                    query_rot_size = concat([
                        shape(query, 0),
                        shape(query, 1),
                        shape(query, 2), self.rotary_embedding_dim
                    ])
                    remaining = shape(key, 3) - self.rotary_embedding_dim
                    key_pass_size = concat([
                        shape(key, 0),
                        shape(key, 1),
                        shape(key, 2), remaining
                    ])
                    query_pass_size = concat([
                        shape(query, 0),
                        shape(query, 1),
                        shape(query, 2), remaining
                    ])
                    k_rot = slice(key, [0, 0, 0, 0], key_rot_size)
                    k_pass = slice(key, [0, 0, 0, self.rotary_embedding_dim],
                                   key_pass_size)

                    q_rot = slice(query, [0, 0, 0, 0], query_rot_size)
                    q_pass = slice(query, [0, 0, 0, self.rotary_embedding_dim],
                                   query_pass_size)

                    k_rot = RopeEmbeddingUtils.apply_rotary_pos_emb(
                        k_rot, [cos, sin], self.position_embedding_type)
                    q_rot = RopeEmbeddingUtils.apply_rotary_pos_emb(
                        q_rot, [cos, sin], self.position_embedding_type)

                    key = concat([k_rot, k_pass], dim=3)
                    query = concat([q_rot, q_pass], dim=3)
                else:
                    key = RopeEmbeddingUtils.apply_rotary_pos_emb(
                        key, [cos, sin], self.position_embedding_type)
                    query = RopeEmbeddingUtils.apply_rotary_pos_emb(
                        query, [cos, sin], self.position_embedding_type)

                key = key.permute([0, 2, 1, 3])
                query = query.permute([0, 2, 1, 3])

            past_key_value = None if kv_cache_params is None else kv_cache_params.get_first_past_key_value(
            )
            if past_key_value is not None:

                def dequantize_tensor(x, scale):
                    # Cast from int8 to dtype
                    casted_x = cast(x, self.dtype)
                    return casted_x * scale

                if self.use_int8_kv_cache:
                    past_key_value = dequantize_tensor(
                        past_key_value, self.kv_quant_orig_scale.value)

                if self.use_fp8_qdq and self.quant_mode.has_kv_cache_quant():
                    past_key_value = dequantize(past_key_value,
                                                self.kv_quant_orig_scale.value)

                # past_key_value [bs, 2, num_heads, max_seq_len, head_dim]
                past_key, past_value = split(past_key_value, 1, dim=1)

                key_shape = concat([
                    shape(past_key, 0),
                    shape(past_key, 2),
                    shape(past_key, 3),
                    shape(past_key, 4)
                ])
                past_key = past_key.view(key_shape, zero_is_placeholder=False)
                past_value = past_value.view(key_shape,
                                             zero_is_placeholder=False)

                key = concat([past_key, key], dim=2).cast(self.dtype)
                value = concat([past_value, value], dim=2).cast(self.dtype)

            if use_cache:
                key_inflated_shape = concat([
                    shape(key, 0), 1,
                    shape(key, 1),
                    shape(key, 2),
                    shape(key, 3)
                ])
                inflated_key = key.view(key_inflated_shape,
                                        zero_is_placeholder=False)
                inflated_value = value.view(key_inflated_shape,
                                            zero_is_placeholder=False)
                past_key_value = concat([inflated_key, inflated_value], dim=1)

                if self.use_int8_kv_cache:

                    def quantize_tensor(x, scale):
                        scaled = x * scale
                        rounded = round(scaled)
                        clipped = clip(rounded, -128, 127)
                        quantized = cast(clipped, 'int8')
                        return quantized

                    past_key_value = quantize_tensor(
                        past_key_value, self.kv_orig_quant_scale.value)

                if self.use_fp8_qdq and self.quant_mode.has_kv_cache_quant():
                    past_key_value = quantize(past_key_value,
                                              self.kv_orig_quant_scale.value,
                                              dtype='fp8')

            # MQA broadcast
            if self.num_attention_heads // self.num_attention_kv_heads > 1:
                key = repeat_interleave(
                    key,
                    self.num_attention_heads // self.num_attention_kv_heads, 1)
                value = repeat_interleave(
                    value,
                    self.num_attention_heads // self.num_attention_kv_heads, 1)

            key_length = shape(key, 2)

            # The following code creates a 2D tensor with 0s in the lower triangular (including the diagonal) and
            # +INF in the upper triangular parts. This bias tensor will be added to the output of the Q*K^T matrix
            # multiplication (BMM1). The +INF elements will be transformed to 0s by the Softmax operator that
            # follows. The elements that corresponds to 0s in the bias are unaffected by the bias tensor.
            #
            # Note that when we added to another bias tensor B (for example, with AliBi), the values in the lower-
            # triangular part of the B tensor are not affected and the upper-triangular ones are set to +INF.
            if self.attention_mask_type == AttentionMaskType.causal:
                query_length = shape(query, 2)
                starts = concat([0, 0, key_length - query_length, 0])
                sizes = concat([1, 1, query_length, key_length])
                select_buf = np.expand_dims(
                    np.tril(
                        np.ones((self.max_position_embeddings,
                                 self.max_position_embeddings))).astype(bool),
                    (0, 1))

                select_buf = np.logical_not(select_buf)
                mask_buf = np.zeros_like(select_buf, np.float32)
                mask_buf[select_buf] = float('-inf')
                buffer = constant(mask_buf)
                generated_mask = slice(buffer, starts, sizes)

            elif self.attention_mask_type == AttentionMaskType.bidirectional:
                query_length = shape(query, 2)
                zero_buf = np.expand_dims(
                    np.zeros((self.max_position_embeddings,
                              self.max_position_embeddings),
                             dtype=np.float32), (0, 1))

                zero_buf[:, :, :-1, -1] = 1
                zero_buf *= -10000

                mask = constant(zero_buf)

                # context phase, query_length
                mask_size = where(query_length > 1, query_length, 1)
                mask_start = where(query_length > 1,
                                   self.max_position_embeddings - mask_size, 1)
                start = concat([0, 0, mask_start, mask_start])
                size = concat([1, 1, mask_size, mask_size])
                generated_mask = slice(mask, start, size)

            if attention_mask is not None:
                attention_mask = expand_mask(attention_mask, shape(query, 2))
            bias = attention_mask
            if self.position_embedding_type.is_alibi():
                alibi_biases = generate_alibi_biases(alibi_slopes, key_length)
                bias = alibi_biases if bias is None else bias + alibi_biases

            key = key.permute([0, 1, 3, 2])
            with precision('float32'):
                if norm_before_bmm1:
                    # Apply norm on query earlier to prevent matmul fp16 overflow.
                    query /= self.norm_factor
                attention_scores = matmul(cast(query, 'float32'),
                                          cast(key, 'float32'))
                if not norm_before_bmm1:
                    attention_scores = attention_scores / self.norm_factor

                if self.attention_mask_type in [
                        AttentionMaskType.causal,
                        AttentionMaskType.bidirectional
                ]:
                    bias = generated_mask if bias is None else bias + generated_mask

                if bias is not None and not self.cross_attention:
                    attention_scores = attention_scores + bias

            attention_probs = softmax(attention_scores, dim=-1)

            if default_net().strongly_typed and (attention_probs.dtype !=
                                                 value.dtype):
                attention_probs = cast(attention_probs, value.dtype)

            context = matmul(attention_probs, value).permute([0, 2, 1, 3])
            context = context.view(
                concat([shape(context, 0),
                        shape(context, 1), self.hidden_size]))

        dense_lora_param = None
        if lora_param is not None:
            dense_lora_param = lora_param.get_runtime_params(0, "attn_dense")
        context = self.dense(context,
                             workspace,
                             lora_runtime_param=dense_lora_param)

        if use_cache:
            return (context, past_key_value)
        else:
            return context


class BertAttention(Module):

    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 max_position_embeddings=1024,
                 num_layers=1,
                 attention_head_size=None,
                 num_kv_heads=None,
                 q_scaling=1.0,
                 apply_query_key_layer_scaling=False,
                 bias=True,
                 dtype=None,
                 tp_group=None,
                 tp_size=1,
                 tp_rank=0,
                 relative_attention=False,
                 max_distance=0,
                 num_buckets=0):
        super().__init__()

        self.attention_head_size = hidden_size // num_attention_heads if attention_head_size is None else attention_head_size
        self.num_attention_heads = num_attention_heads // tp_size
        self.num_attention_kv_heads = (
            num_kv_heads + tp_size - 1
        ) // tp_size if num_kv_heads is not None else self.num_attention_heads
        self.hidden_size = hidden_size // tp_size
        self.max_position_embeddings = max_position_embeddings
        self.norm_factor = math.sqrt(self.attention_head_size)
        self.tp_size = tp_size
        self.tp_rank = tp_rank

        self.num_layers = num_layers
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.norm_factor = math.sqrt(self.attention_head_size)
        self.q_scaling = q_scaling
        if self.apply_query_key_layer_scaling:
            self.norm_factor *= self.num_layers
            self.q_scaling *= self.num_layers

        self.dtype = dtype

        self.relative_attention = relative_attention
        self.max_distance = max_distance

        # out dim is not necessarily hidden_size + kv specific size (in MQA/GQA), but num_heads * heads_size
        # example: d_model != num_heads * head_size in Flan-T5
        self.qkv = ColumnLinear(
            hidden_size,
            tp_size * self.num_attention_heads * self.attention_head_size +
            (2 * tp_size * self.num_attention_kv_heads *
             self.attention_head_size),
            bias=bias,
            dtype=dtype,
            tp_group=tp_group,
            tp_size=tp_size,
            gather_output=False)
        self.dense = RowLinear(tp_size * self.num_attention_heads *
                               self.attention_head_size,
                               hidden_size,
                               bias=bias,
                               dtype=dtype,
                               tp_group=tp_group,
                               tp_size=tp_size)

        # per-layer relative attention table
        if relative_attention:
            self.rel_attn_table = Parameter(shape=(num_attention_heads //
                                                   tp_size, num_buckets),
                                            dtype=dtype)

    def forward(self,
                hidden_states: Tensor,
                attention_mask=None,
                input_lengths=None,
                workspace=None,
                max_input_length=None):
        assert isinstance(hidden_states, Tensor)

        qkv = self.qkv(hidden_states)

        if default_net().plugin_config.bert_attention_plugin:
            # TRT plugin mode
            assert input_lengths is not None
            context = bert_attention(
                qkv,
                input_lengths,
                self.num_attention_heads,
                self.attention_head_size,
                q_scaling=self.q_scaling,
                relative_attention=self.relative_attention,
                max_distance=self.max_distance,
                relative_attention_bias=self.rel_attn_table.value
                if self.relative_attention else None,
                max_input_length=max_input_length)
        else:
            # plain TRT mode
            def transpose_for_scores(x):
                new_x_shape = concat([
                    shape(x, 0),
                    shape(x, 1), self.num_attention_heads,
                    self.attention_head_size
                ])
                return x.view(new_x_shape).permute([0, 2, 1, 3])

            query, key, value = split(qkv, self.hidden_size, dim=2)
            query = transpose_for_scores(query)
            key = transpose_for_scores(key)
            value = transpose_for_scores(value)

            key = key.permute([0, 1, 3, 2])
            attention_scores = matmul(query, key)
            attention_scores = attention_scores / self.norm_factor

            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask

            attention_probs = softmax(attention_scores, dim=-1)

            context = matmul(attention_probs, value).permute([0, 2, 1, 3])
            context = context.view(
                concat([shape(context, 0),
                        shape(context, 1), self.hidden_size]))

        context = self.dense(context, workspace)

        return context

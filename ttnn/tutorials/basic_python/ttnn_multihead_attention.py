# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import os
import time
import torch
import ttnn
from loguru import logger

from ttnn.model_preprocessing import (
    preprocess_linear_bias,
    preprocess_linear_weight,
)


def main():
    torch.manual_seed(0)

    device_id = 0
    device = ttnn.open_device(device_id=device_id)

    def multi_head_attention(
        hidden_states,
        attention_mask,
        query_weight,
        query_bias,
        key_weight,
        key_bias,
        value_weight,
        value_bias,
        output_weight,
        output_bias,
        *,
        num_heads,
    ):
        fallback_reshape = ttnn.get_fallback_function(ttnn.reshape)

        batch_size, sequence_size, hidden_size = hidden_states.shape
        head_size = hidden_size // num_heads

        query = hidden_states @ query_weight
        query = query + query_bias
        query = ttnn.to_layout(query, layout=ttnn.ROW_MAJOR_LAYOUT)
        query = fallback_reshape(query, (batch_size, sequence_size, num_heads, head_size))
        query = ttnn.to_layout(query, layout=ttnn.TILE_LAYOUT)
        query = ttnn.permute(query, (0, 2, 1, 3))

        key = hidden_states @ key_weight
        key = key + key_bias
        key = ttnn.to_layout(key, layout=ttnn.ROW_MAJOR_LAYOUT)
        key = fallback_reshape(key, (batch_size, sequence_size, num_heads, head_size))
        key = ttnn.to_layout(key, layout=ttnn.TILE_LAYOUT)
        key = ttnn.permute(key, (0, 2, 3, 1))

        value = hidden_states @ value_weight
        value = value + value_bias
        value = ttnn.to_layout(value, layout=ttnn.ROW_MAJOR_LAYOUT)
        value = fallback_reshape(value, (batch_size, sequence_size, num_heads, head_size))
        value = ttnn.to_layout(value, layout=ttnn.TILE_LAYOUT)
        value = ttnn.permute(value, (0, 2, 1, 3))

        attention_scores = query @ key
        attention_scores = attention_scores * (1 / (head_size**0.5))
        attention_scores += attention_mask
        attention_probs = ttnn.softmax(attention_scores, dim=-1, numeric_stable=False)

        context_layer = attention_probs @ value
        context_layer = ttnn.permute(context_layer, (0, 2, 1, 3))
        context_layer = ttnn.to_layout(context_layer, layout=ttnn.ROW_MAJOR_LAYOUT)
        context_layer = fallback_reshape(context_layer, (batch_size, sequence_size, hidden_size))
        context_layer = ttnn.to_layout(context_layer, layout=ttnn.TILE_LAYOUT)

        self_output = context_layer @ output_weight
        self_output = self_output + output_bias

        return self_output

    batch_size = 6
    sequence_size = 384
    num_heads = 16
    head_size = 64
    hidden_size = num_heads * head_size

    torch_hidden_states = torch.randn((batch_size, sequence_size, hidden_size), dtype=torch.bfloat16)
    torch_attention_mask = torch.randn((batch_size, 1, 1, sequence_size), dtype=torch.bfloat16)
    torch_query_weight = torch.randn((hidden_size, hidden_size), dtype=torch.bfloat16)
    torch_query_bias = torch.randn((hidden_size,), dtype=torch.bfloat16)
    torch_key_weight = torch.randn((hidden_size, hidden_size), dtype=torch.bfloat16)
    torch_key_bias = torch.randn((hidden_size,), dtype=torch.bfloat16)
    torch_value_weight = torch.randn((hidden_size, hidden_size), dtype=torch.bfloat16)
    torch_value_bias = torch.randn((hidden_size,), dtype=torch.bfloat16)
    torch_output_weight = torch.randn((hidden_size, hidden_size), dtype=torch.bfloat16)
    torch_output_bias = torch.randn((hidden_size,), dtype=torch.bfloat16)

    hidden_states = ttnn.from_torch(torch_hidden_states, layout=ttnn.TILE_LAYOUT, device=device)
    attention_mask = ttnn.from_torch(torch_attention_mask, layout=ttnn.TILE_LAYOUT, device=device)
    query_weight = ttnn.from_torch(torch_query_weight, layout=ttnn.TILE_LAYOUT, device=device)
    query_bias = ttnn.from_torch(
        torch_query_bias, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    key_weight = ttnn.from_torch(torch_key_weight, layout=ttnn.TILE_LAYOUT, device=device)
    key_bias = ttnn.from_torch(
        torch_key_bias, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    value_weight = ttnn.from_torch(torch_value_weight, layout=ttnn.TILE_LAYOUT, device=device)
    value_bias = ttnn.from_torch(
        torch_value_bias, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    output_weight = ttnn.from_torch(torch_output_weight, layout=ttnn.TILE_LAYOUT, device=device)
    output_bias = ttnn.from_torch(
        torch_output_bias, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    start = time.time()
    multi_head_attention(
        hidden_states,
        attention_mask,
        query_weight,
        query_bias,
        key_weight,
        key_bias,
        value_weight,
        value_bias,
        output_weight,
        output_bias,
        num_heads=num_heads,
    )
    end = time.time()
    duration = end - start

    logger.info(f"Multi-head attention ran in {duration} seconds for the first iteration")

    start = time.time()
    output = multi_head_attention(
        hidden_states,
        attention_mask,
        query_weight,
        query_bias,
        key_weight,
        key_bias,
        value_weight,
        value_bias,
        output_weight,
        output_bias,
        num_heads=num_heads,
    )
    end = time.time()
    duration = end - start

    logger.info(
        f"Multi-head attention ran in {duration} seconds for the subsequent iteration because of the program cache"
    )

    def optimized_multi_head_attention(
        hidden_states,
        attention_mask,
        fused_qkv_weight,
        fused_qkv_bias,
        self_output_weight,
        self_output_bias,
        *,
        num_heads,
        num_cores_x=12,
    ):
        batch_size, _, hidden_size = hidden_states.shape
        head_size = hidden_size // num_heads

        hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)

        fused_qkv_output = ttnn.linear(
            hidden_states,
            fused_qkv_weight,
            bias=fused_qkv_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            core_grid=ttnn.CoreGrid(y=batch_size, x=num_cores_x),
        )

        (
            query,
            key,
            value,
        ) = ttnn.transformer.split_query_key_value_and_split_heads(
            fused_qkv_output,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            num_heads=num_heads,
        )
        ttnn.deallocate(fused_qkv_output)

        attention_scores = ttnn.matmul(
            query,
            key,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            core_grid=ttnn.CoreGrid(y=batch_size, x=num_cores_x),
        )
        ttnn.deallocate(query)
        ttnn.deallocate(key)

        attention_probs = ttnn.transformer.attention_softmax_(
            attention_scores, attention_mask=attention_mask, head_size=head_size
        )

        context_layer = ttnn.matmul(
            attention_probs,
            value,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            core_grid=ttnn.CoreGrid(y=batch_size, x=num_cores_x),
        )
        ttnn.deallocate(attention_probs)

        context_layer_after_concatenate_heads = ttnn.transformer.concatenate_heads(
            context_layer,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(context_layer)

        self_output = ttnn.linear(
            context_layer_after_concatenate_heads,
            self_output_weight,
            bias=self_output_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            core_grid=ttnn.CoreGrid(y=batch_size, x=num_cores_x),
        )
        ttnn.deallocate(context_layer_after_concatenate_heads)

        return self_output

    torch_qkv_weight = torch.cat([torch_query_weight, torch_key_weight, torch_value_weight], dim=-1)
    torch_qkv_bias = torch.cat([torch_query_bias, torch_key_bias, torch_value_bias], dim=-1)

    qkv_weight = preprocess_linear_weight(torch_qkv_weight.T, dtype=ttnn.bfloat16)
    qkv_bias = preprocess_linear_bias(torch_qkv_bias, dtype=ttnn.bfloat16)
    output_weight = preprocess_linear_weight(torch_output_weight.T, dtype=ttnn.bfloat16)
    output_bias = preprocess_linear_bias(torch_output_bias, dtype=ttnn.bfloat16)

    qkv_weight = ttnn.to_device(qkv_weight, device)
    qkv_bias = ttnn.to_device(qkv_bias, device, memory_config=ttnn.L1_MEMORY_CONFIG)
    output_weight = ttnn.to_device(output_weight, device)
    output_bias = ttnn.to_device(output_bias, device, memory_config=ttnn.L1_MEMORY_CONFIG)

    start = time.time()
    hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)
    optimized_output = optimized_multi_head_attention(
        hidden_states,
        attention_mask,
        qkv_weight,
        qkv_bias,
        output_weight,
        output_bias,
        num_heads=num_heads,
    )
    end = time.time()
    duration = end - start

    logger.info(f"Optimized multi-head attention ran in {duration} seconds for the first iteration")

    start = time.time()
    optimized_output = optimized_multi_head_attention(
        hidden_states,
        attention_mask,
        qkv_weight,
        qkv_bias,
        output_weight,
        output_bias,
        num_heads=num_heads,
    )
    end = time.time()
    duration = end - start

    logger.info(
        f"Optimized multi-head attention ran in {duration} seconds for the subsequent iteration because of the program cache"
    )

    torch_output = ttnn.to_torch(output)
    torch_optimized_output = ttnn.to_torch(optimized_output)

    assert torch.allclose(torch_output, torch_optimized_output)

    ttnn.close_device(device)


if __name__ == "__main__":
    main()

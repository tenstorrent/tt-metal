// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::pool::upsample {

bool is_integer_scale(float scale);

// Output extent for a float scale, as torch computes it: floor(in * scale) in double. A float32 scale that is the
// rounded ratio out/in (e.g. 384.0f / 17) is snapped back to that ratio so the floor lands on out.
double compute_upsample_output_extent(uint32_t in_extent, float scale);

// float32 reciprocal torch uses for nearest indexing: (float)(1.0 / scale) with the same snapped scale.
float compute_upsample_nearest_reciprocal(uint32_t in_extent, float scale);

uint32_t compute_num_cores_nhw(const tt::tt_metal::ShardSpec& shard_spec, tt::tt_metal::TensorMemoryLayout mem_layout);

enum class UpsamplePath { INTEGER_OPTIMIZED, FLOAT_GENERAL, UNSUPPORTED };

UpsamplePath select_upsample_path(const Tensor& input, float scale_h, float scale_w, const std::string& mode);

// For ND sharded tensors (float path only)
tt::tt_metal::MemoryConfig compute_nd_output_mem_config(
    const tt::tt_metal::MemoryConfig& input_mem_config, float scale_h, float scale_w);

// For integer path sharded (HEIGHT/BLOCK sharded, nearest or bilinear)
tt::tt_metal::MemoryConfig compute_integer_output_mem_config(
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const Tensor& input,
    const std::string& mode,
    float scale_h,
    float scale_w,
    uint32_t out_n,
    uint32_t out_h,
    uint32_t out_w);

// For float path sharded (all standard sharding layouts)
tt::tt_metal::MemoryConfig compute_float_output_mem_config(
    const tt::tt_metal::MemoryConfig& input_mem_config, uint32_t out_n, uint32_t out_h, uint32_t out_w);

std::string generate_unsupported_config_message(
    const Tensor& input, float scale_h, float scale_w, const std::string& mode);

}  // namespace ttnn::operations::pool::upsample

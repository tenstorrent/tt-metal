// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <cstdint>
#include <span>

namespace ttnn::operations::data_movement::transpose {

bool is_native_transpose_sharding(const TensorSpec& input_spec, const tt::tt_metal::MemoryConfig& output_memory_config);

tt::tt_metal::ShardSpec adjust_shard_spec_to_shape(
    const tt::tt_metal::ShardSpec& shard_spec, const ttnn::Shape& from_shape, const ttnn::Shape& to_shape);

tt::tt_metal::ShardSpec generate_transpose_shard_spec(
    const Tensor& input_tensor, const ttnn::Shape& padded_out_shape, tt::tt_metal::TensorMemoryLayout memory_layout);

// Copies the RuntimeTensorShape common args for `buffer` into `dst`. Uses std::span so the
// destination length is carried alongside the pointer and bounds-checked against the element
// count produced by `TensorAccessorArgs::get_common_runtime_args()`.
void copy_transpose_common_runtime_args(const tt::tt_metal::Buffer& buffer, std::span<std::uint32_t> dst);

// Refreshes the RuntimeTensorShape common args on both reader and writer kernels. Used by the
// four interleaved program factories in `override_runtime_arguments` on program-cache hits.
void refresh_transpose_common_runtime_args(
    tt::tt_metal::Program& program,
    tt::tt_metal::KernelHandle reader_kernel_id,
    tt::tt_metal::KernelHandle writer_kernel_id,
    const tt::tt_metal::Buffer& input_buffer,
    const tt::tt_metal::Buffer& output_buffer);

}  // namespace ttnn::operations::data_movement::transpose

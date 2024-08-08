/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <optional>
#include <variant>
#include <vector>

#include "tt_metal/impl/buffers/circular_buffer_types.hpp"
#include "tt_metal/impl/kernels/kernel_types.hpp"
#include "ttnn/cpp/ttnn/tensor/tensor.hpp"

namespace ttnn {
namespace operations {

inline bool is_dram(const Tensor &tensor) { return tensor.memory_config().buffer_type == BufferType::DRAM; }
inline bool is_dram(const std::optional<const Tensor> tensor) {
    return tensor.has_value() ? is_dram(tensor.value()) : true;
}
inline bool is_dram(const std::optional<std::reference_wrapper<const Tensor>> tensor) {
    return tensor.has_value() ? is_dram(tensor->get()) : true;
}
inline bool is_dram(const Buffer *buffer) { return buffer->buffer_type() == BufferType::DRAM; }

struct ComputeKernelArg {
    const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec;
    uint32_t num_tile_per_core_group;
    const std::vector<uint32_t> &compile_args = {};
};

[[maybe_unused]] tt::tt_metal::KernelHandle CreateReadKernel(
    Program &program,
    const std::string &file_name,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec,
    const std::vector<uint32_t> &compile_args = {},
    std::map<string, string> defines = {});

[[maybe_unused]] tt::tt_metal::KernelHandle CreateWriteKernel(
    Program &program,
    const std::string &file_name,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec,
    const std::vector<uint32_t> &compile_args = {},
    std::map<string, string> defines = {});

struct CircularBufferArg {
    uint32_t buffer_index = 0;
    uint32_t num_tiles = 0;
    tt::DataFormat data_format = tt::DataFormat::Invalid;
    std::optional<std::variant<CoreCoord, CoreRange, CoreRangeSet>> core_range = std::nullopt;
    tt::tt_metal::Buffer *buffer_ptr = nullptr;
};

[[maybe_unused]] std::vector<tt::tt_metal::KernelHandle> CreateComputeKernel(
    Program &program,
    const std::string &file_name,
    std::vector<ComputeKernelArg> args,
    std::map<std::string, std::string> defines = {},
    MathFidelity math_fidelity = MathFidelity::HiFi4,
    bool fp32_dest_acc_en = false,
    bool math_approx_mode = false,
    bool preserve_fp32_precision = false);

[[maybe_unused]] tt::tt_metal::KernelHandle CreateComputeKernel(
    Program &program,
    const std::string &file_name,
    ComputeKernelArg arg,
    std::map<std::string, std::string> defines = {},
    MathFidelity math_fidelity = MathFidelity::HiFi4,
    bool fp32_dest_acc_en = false,
    bool math_approx_mode = false,
    bool preserve_fp32_precision = false);

[[maybe_unused]] std::vector<CBHandle> CreateCircularBuffer(
    Program &program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_range,
    tt::DataFormat data_format,
    std::vector<CircularBufferArg> args);

[[maybe_unused]] CBHandle CreateCircularBuffer(
    Program &program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_range,
    tt::DataFormat data_format,
    CircularBufferArg arg);

void UpdateCircularBuffer(
    Program &program,
    tt::DataFormat data_format,
    std::vector<CBHandle> cb_handles,
    std::vector<CircularBufferArg> args);

void UpdateCircularBuffer(Program &program, tt::DataFormat data_format, CBHandle cb_handle, CircularBufferArg arg);

bool is_hw_dim(uint32_t dim, uint32_t rank);

uint32_t compute_inner(tt::tt_metal::Shape shape, uint32_t dim);

uint32_t compute_outer(tt::tt_metal::Shape shape, uint32_t dim);

std::tuple<CoreRangeSet, CoreRangeSet, CoreRangeSet> add_core_offset(
    CoreRangeSet all_cores, CoreRangeSet core_group_1, CoreRangeSet core_group_2, uint32_t offset_x, uint32_t offset_y);

std::tuple<uint32_t, CoreRangeSet, CoreRangeSet, CoreRangeSet, uint32_t, uint32_t> split_work_to_cores(
    CoreRange core_range, uint32_t units_to_divide);

}  // namespace operations
}  // namespace ttnn

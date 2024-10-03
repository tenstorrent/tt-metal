/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <optional>
#include <variant>

#include "tt_metal/host_api.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt {
namespace operations {
namespace primary {

using namespace tt::tt_metal;

inline bool is_dram(const Tensor &tensor) { return tensor.memory_config().buffer_type == BufferType::DRAM; }
inline bool is_dram(const std::optional<const Tensor> tensor) {
    return tensor.has_value() ? is_dram(tensor.value()) : true;
}
inline bool is_dram(const std::optional<std::reference_wrapper<const Tensor>> tensor) {
    return tensor.has_value() ? is_dram(tensor->get()) : true;
}
inline bool is_dram(const Buffer *buffer) { return buffer->buffer_type() == BufferType::DRAM; }

inline bool is_scalar(const Tensor &tensor) {
    // TODO(dongjin): current impl requires finding a scalar in a 2d shape
    const auto &shape = tensor.get_logical_shape();
    const uint32_t rank = shape.rank();

    // TODO(dongjin): refactor dot op
    if (rank == 4) {
        return (shape[0] == 1 && shape[1] == 1 && shape[2] == 1 && shape[3] == 1);
    }
    return (rank == 2 && shape[-1] == 1 && shape[-2] == 1);
}

inline bool is_1d_tensor(const Tensor &tensor) {
    // TODO(dongjin): current impl requires finding a 1d in a 2d shape
    const auto &shape = tensor.get_logical_shape();
    const uint32_t rank = shape.rank();

    // TODO(dongjin): refactor dot op
    if (rank == 4) {
        return (shape[0] == 1 && shape[1] == 1 && shape[2] == 1);
    }
    return (rank == 2 && shape[-2] == 1);
}

inline bool is_same_shape(const Tensor &tensor_a, const Tensor &tensor_b) {
    const auto &tensor_a_shape = tensor_a.get_logical_shape();
    const auto &tensor_b_shape = tensor_b.get_logical_shape();
    return (tensor_a_shape == tensor_b_shape);
}

std::tuple<CoreRangeSet, CoreRangeSet, CoreRangeSet> add_core_offset(
    CoreRangeSet all_cores, CoreRangeSet core_group_1, CoreRangeSet core_group_2, uint32_t offset_x, uint32_t offset_y);

std::tuple<uint32_t, CoreRangeSet, CoreRangeSet, CoreRangeSet, uint32_t, uint32_t> split_work_to_cores(
    CoreRange core_range, uint32_t units_to_divide);

[[maybe_unused]] KernelHandle CreateReadKernel(
    Program &program,
    const std::string &file_name,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec,
    const std::vector<uint32_t> &compile_args = {},
    std::map<string, string> defines = {});

[[maybe_unused]] KernelHandle CreateWriteKernel(
    Program &program,
    const std::string &file_name,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec,
    const std::vector<uint32_t> &compile_args = {},
    std::map<string, string> defines = {});

struct ComputeKernelArg {
    const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec;
    uint32_t num_tile_per_core_group;
    const std::vector<uint32_t> &compile_args = {};
};

struct ComputeKernelConfig {
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    bool fp32_dest_acc_en = false;
    vector<UnpackToDestMode> unpack_to_dest_mode;
    bool math_approx_mode = false;
    std::map<std::string, std::string> defines;
};

[[maybe_unused]] std::vector<KernelHandle> CreateComputeKernel(
    Program &program,
    const std::string &file_name,
    std::vector<ComputeKernelArg> args,
    std::map<std::string, std::string> defines = {},
    MathFidelity math_fidelity = MathFidelity::HiFi4,
    bool fp32_dest_acc_en = false,
    bool math_approx_mode = false,
    vector<UnpackToDestMode> unpack_to_dest_mode = {});

[[maybe_unused]] KernelHandle CreateComputeKernel(
    Program &program,
    const std::string &file_name,
    ComputeKernelArg arg,
    std::map<std::string, std::string> defines = {},
    MathFidelity math_fidelity = MathFidelity::HiFi4,
    bool fp32_dest_acc_en = false,
    bool math_approx_mode = false,
    vector<UnpackToDestMode> unpack_to_dest_mode = {});

[[maybe_unused]] std::vector<KernelHandle> CreateComputeKernel(
    Program &program, const std::string &file_name, std::vector<ComputeKernelArg> args, ComputeKernelConfig config);

[[maybe_unused]] KernelHandle CreateComputeKernel(
    Program &program, const std::string &file_name, ComputeKernelArg arg, ComputeKernelConfig config);

struct CircularBufferArg {
    uint32_t buffer_index;
    uint32_t num_tiles;
    tt::DataFormat data_format;
    std::optional<std::variant<CoreCoord, CoreRange, CoreRangeSet>> core_range = std::nullopt;

    CircularBufferArg(uint32_t buffer_index, uint32_t num_tiles) : buffer_index(buffer_index), num_tiles(num_tiles) {
        data_format = tt::DataFormat::Invalid;
    }
    CircularBufferArg(uint32_t buffer_index, uint32_t num_tiles, tt::DataFormat data_format) :
        buffer_index(buffer_index), num_tiles(num_tiles), data_format(data_format) {}
};

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

void check_tensor(
    const Tensor &tensor,
    const std::string &op_name,
    const std::string &tensor_name,
    const std::initializer_list<DataType> &data_types = {DataType::BFLOAT16},
    Layout layout = Layout::TILE,
    bool check_dtype = true,
    bool check_layout = true);

void check_tensor(
    std::optional<Tensor> tensor,
    const std::string &op_name,
    const std::string &tensor_name,
    const std::initializer_list<DataType> &data_types = {DataType::BFLOAT16},
    Layout layout = Layout::TILE,
    bool check_dtype = true,
    bool check_layout = true);

struct CallbackArgMap {
    std::map<uint32_t, uint32_t> input;
    std::map<uint32_t, uint32_t> optional_input;
    std::map<uint32_t, uint32_t> output;
};

using Tensors = std::vector<Tensor>;
using OptionalConstTensors = std::vector<std::optional<const Tensor>>;

// To use this function, the arguments in the reader kernel must always be sorted in the order of input followed by
// optional_input. Furthermore, input and output tensors must always start from the 0th argument.
template <typename OutputTensors = Tensors>
auto create_override_runtime_arguments_callback(
    KernelHandle reader_kernel_id, KernelHandle writer_kernel_id, uint32_t num_cores, uint32_t core_h) {
    return [reader_kernel_id = reader_kernel_id, writer_kernel_id = writer_kernel_id, num_cores, core_h](
               const void *operation,
               Program &program,
               const Tensors &input_tensors,
               const OptionalConstTensors &optional_input_tensors,
               const OutputTensors &output_tensors) -> void {
        for (uint32_t icore = 0; icore < num_cores; icore++) {
            CoreCoord core = {icore / core_h, icore % core_h};

            // readers
            {
                uint32_t rt_idx = 0;
                auto &runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
                for (uint32_t idx = 0; idx < input_tensors.size(); idx++) {
                    runtime_args[rt_idx++] = input_tensors.at(idx).buffer()->address();
                }
                for (uint32_t idx = 0; idx < optional_input_tensors.size(); idx++) {
                    auto optional_input_tensor = optional_input_tensors.at(idx);
                    runtime_args[rt_idx++] =
                        optional_input_tensor.has_value() ? optional_input_tensor.value().buffer()->address() : 0;
                }
            }

            // writer
            {
                auto &runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
                for (uint32_t idx = 0; idx < output_tensors.size(); idx++) {
                    runtime_args[idx] = output_tensors.at(idx).buffer()->address();
                }
            }
        }
    };
}

// Using this structure is not recommended because directly setting the callback argument map doesn't significantly
// reduce the amount of code.
template <typename OutputTensors = Tensors>
auto create_override_runtime_arguments_callback(
    KernelHandle reader_kernel_id,
    KernelHandle writer_kernel_id,
    uint32_t num_cores,
    uint32_t core_h,
    CallbackArgMap arg_map) {
    return [reader_kernel_id = reader_kernel_id, writer_kernel_id = writer_kernel_id, arg_map, num_cores, core_h](
               const void *operation,
               Program &program,
               const Tensors &input_tensors,
               const OptionalConstTensors &optional_input_tensors,
               const OutputTensors &output_tensors) -> void {
        for (uint32_t icore = 0; icore < num_cores; icore++) {
            CoreCoord core = {icore / core_h, icore % core_h};

            // readers
            {
                auto &runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
                for (const auto &pair : arg_map.input) {
                    runtime_args[pair.first] = input_tensors.at(pair.second).buffer()->address();
                }
                for (const auto &pair : arg_map.optional_input) {
                    auto optional_input_tensor = optional_input_tensors.at(pair.second);
                    runtime_args[pair.first] =
                        optional_input_tensor.has_value() ? optional_input_tensor.value().buffer()->address() : 0;
                }
            }

            // writer
            {
                auto &runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
                for (const auto &pair : arg_map.output) {
                    runtime_args[pair.first] = output_tensors.at(pair.second).buffer()->address();
                }
            }
        }
    };
}

// To use this function, the arguments in the reader kernel must always be sorted in the order of input followed by
// optional_input. Furthermore, input and output tensors must always start from the 0th argument.
template <typename OutputTensors = Tensors>
auto create_override_addresses_callback(
    KernelHandle reader_kernel_id, KernelHandle writer_kernel_id, uint32_t num_cores, uint32_t core_h) {
    return [reader_kernel_id = reader_kernel_id, writer_kernel_id = writer_kernel_id, num_cores, core_h](
               const Program &program,
               const std::vector<Buffer *> &input_buffers,
               const std::vector<Buffer *> &output_buffers) -> void {
        for (uint32_t icore = 0; icore < num_cores; icore++) {
            CoreCoord core = {icore / core_h, icore % core_h};

            // readers
            {
                auto &runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
                for (uint32_t idx = 0; idx < input_buffers.size(); idx++) {
                    auto buffer = input_buffers.at(idx);
                    if (buffer != nullptr) {
                        runtime_args[idx] = buffer->address();
                    }
                }
            }

            // writer
            {
                auto &runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
                for (uint32_t idx = 0; idx < output_buffers.size(); idx++) {
                    auto buffer = output_buffers.at(idx);
                    if (buffer != nullptr) {
                        runtime_args[idx] = buffer->address();
                    }
                }
            }
        }
    };
}

bool is_hw_dim(uint32_t dim, uint32_t rank);

uint32_t compute_inner(tt::tt_metal::LegacyShape shape, uint32_t dim);

uint32_t compute_outer(tt::tt_metal::LegacyShape shape, uint32_t dim);

void expand_to_max_dim(std::vector<uint32_t> &dim, const tt::tt_metal::LegacyShape &shape);

void validate_input_with_dim(const Tensor &input, const int64_t &dim);

void validate_output_with_keepdim(const Tensor &input, const Tensor &output, const int64_t &dim, const bool &keep_dim);

void initialize_dims_with_range(std::vector<int64_t> &dims, uint32_t input_rank);

std::vector<int64_t> get_dim(
    const std::optional<std::variant<int64_t, std::vector<int64_t>>> &dim, uint32_t input_rank);

std::tuple<uint32_t, uint32_t, uint32_t> extract_spatial_dims(const ttnn::SimpleShape& shape);

std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> extract_and_scale_spatial_dims(const ttnn::SimpleShape& shape, uint32_t dim);

}  // namespace primary
}  // namespace operations
}  // namespace tt

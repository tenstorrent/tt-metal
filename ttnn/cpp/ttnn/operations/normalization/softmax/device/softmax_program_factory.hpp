// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "softmax_operation_types.hpp"

#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/work_split.hpp>
#include <ttnn/device_operation.hpp>

namespace ttnn::operations::normalization::softmax::program {
//
// General-purpose softmax with arbitrary dimension support
//
//
struct SoftmaxProgramFactoryGeneral {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle unary_reader_kernel_id{};
        tt::tt_metal::KernelHandle unary_writer_kernel_id{};
        std::size_t num_cores{};
        std::size_t num_cores_y{};
    };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;
    static void override_runtime_arguments(
        cached_program_t&, const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
};

struct SoftmaxProgramFactoryGeneralWSmall : SoftmaxProgramFactoryGeneral {
    static cached_program_t create(const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
};
struct SoftmaxProgramFactoryGeneralWLarge : SoftmaxProgramFactoryGeneral {
    static cached_program_t create(const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
};
struct SoftmaxProgramFactoryGeneralHSmall : SoftmaxProgramFactoryGeneral {
    static cached_program_t create(const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
};
struct SoftmaxProgramFactoryGeneralHLarge : SoftmaxProgramFactoryGeneral {
    static cached_program_t create(const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
};
struct SoftmaxProgramFactoryGeneralCLarge : SoftmaxProgramFactoryGeneral {
    static cached_program_t create(const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
};

//
// Optimized for transformer attention patterns
//
// Interleaved memory
struct SoftmaxProgramFactoryAttentionOptimized {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader_kernels_id{};
        tt::tt_metal::KernelHandle writer_kernels_id{};
        tt::tt_metal::KernelHandle softmax_kernels_id{};
        CoreCoord grid_size{};
        bool fp32_dest_acc_en{};
        uint32_t scalar_tile_size{}, in0_tile_size{}, im_tile_size{}, out0_tile_size{}, mask_tile_size{};
        tt::tt_metal::CBHandle cb_in0_id{}, cb_out0_id{}, cb_intermed1_id{}, cb_in2_id{}, cb_intermed0_id{};
        std::optional<tt::tt_metal::CBHandle> cb_intermed3_id{}, cb_in3_id{}, cb_in4_id{}, cb_intermed2_id{},
            cb_intermed4_id{};
    };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;
    static cached_program_t create(const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
    static void override_runtime_arguments(
        cached_program_t&, const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
};
// Sharded memory
struct SoftmaxShardedProgramFactoryAttentionOptimized {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader_kernels_id{};
        tt::tt_metal::CBHandle cb_in0_id{}, cb_out0_id{};
        std::optional<tt::tt_metal::CBHandle> cb_in3_id{};
        uint32_t num_cores{};
        CoreCoord grid_size{};
    };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;
    static cached_program_t create(const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
    static void override_runtime_arguments(
        cached_program_t&, const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
};
}  // namespace ttnn::operations::normalization::softmax::program

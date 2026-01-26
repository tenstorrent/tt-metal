// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "softmax_operation_types.hpp"

#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <ttnn/device_operation.hpp>

namespace ttnn::prim {
//
// Optimized for transformer attention patterns
//
// Interleaved memory
struct SoftmaxProgramFactoryAttentionOptimized {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader_kernels_id{};
        tt::tt_metal::KernelHandle writer_kernels_id{};
        tt::tt_metal::KernelHandle softmax_kernels_id{};
        CoreCoord grid_size;
        bool fp32_dest_acc_en{};
        uint32_t scalar_tile_size{}, in0_tile_size{}, im_tile_size{}, out0_tile_size{}, mask_tile_size{};
        tt::tt_metal::CBHandle cb_in0_id{}, cb_out0_id{}, cb_intermed1_id{}, cb_in2_id{}, cb_intermed0_id{};
        std::optional<tt::tt_metal::CBHandle> cb_intermed3_id, cb_in3_id, cb_in4_id, cb_intermed2_id, cb_intermed4_id;
    };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;
    static cached_program_t create(const SoftmaxParams&, const SoftmaxInputs&, Tensor&);
    static void override_runtime_arguments(cached_program_t&, const SoftmaxParams&, const SoftmaxInputs&, Tensor&);
};

}  // namespace ttnn::prim

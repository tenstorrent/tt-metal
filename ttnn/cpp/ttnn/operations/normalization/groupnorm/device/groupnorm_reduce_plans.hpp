// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/kernel_lib/host/reduce_host.hpp"
#include "groupnorm_program_utils.hpp"
#include "kernels/groupnorm_constants.hpp"
#include <algorithm>

namespace ttnn::prim {

struct GroupNormReducePlans {
    std::vector<uint32_t> calls;
    ttnn::kernel_lib::host::ReduceAuxiliaryPlan local_auxiliary{2, {}};
    ttnn::kernel_lib::host::ReduceAuxiliaryPlan global_auxiliary{4, {}};

    void append_auxiliary_to(std::vector<uint32_t>& args) const {
        ttnn::kernel_lib::host::ReduceAuxiliaryArgs(local_auxiliary).append_to(args);
        ttnn::kernel_lib::host::ReduceAuxiliaryArgs(global_auxiliary).append_to(args);
    }
};

// Local shapes describe masked tiles. HW reduction cannot mask both the row
// and column padding, so group selection and spatial masks stay in compute.
inline GroupNormReducePlans make_groupnorm_reduce_plans(
    uint32_t first_rows,
    uint32_t first_columns,
    uint32_t second_rows,
    uint32_t second_columns,
    uint32_t global_tiles,
    float local_scalar,
    float global_scalar,
    tt::tt_metal::DataType dtype,
    const ttnn::kernel_lib::host::ReduceHardwareConfig& hardware,
    compute_kernel_lib::ReduceInputPolicy second_policy = compute_kernel_lib::ReduceInputPolicy::NoWaitNoPop) {
    using namespace tt::tt_metal;
    namespace rh = ttnn::kernel_lib::host;
    const TensorLayout layout(dtype, PageConfig(Layout::TILE), MemoryConfig{});
    const TensorSpec output(Shape{1, 1}, layout);
    GroupNormReducePlans result;
    auto append = [&](uint32_t rows,
                      uint32_t columns,
                      float scalar,
                      compute_kernel_lib::ReduceInputPolicy policy,
                      rh::ReduceAuxiliaryPlan& auxiliary) {
        auto plan = rh::make_reduce_plan(
            TensorSpec(Shape{rows * 32, columns * 32}, layout),
            output,
            ReduceOpMath::SUM,
            ReduceOpDim::HW,
            scalar,
            ReduceFp32Mode::Fast,
            hardware);
        plan.input_policy = policy;
        const rh::ReduceCallPlan call{
            .input_cb_id = 0,
            .auxiliary_cb_id = 1,
            .auxiliary_tile_offset = static_cast<uint32_t>(auxiliary.tiles.size()),
            .output_cb_id = 2,
            .accumulator_cb_id = std::nullopt,
            .plan = plan};
        rh::ReduceCallArgs(call).append_to(result.calls);
        auxiliary.tiles.insert(auxiliary.tiles.end(), plan.auxiliary_tiles.begin(), plan.auxiliary_tiles.end());
    };
    append(
        first_rows,
        first_columns,
        local_scalar,
        compute_kernel_lib::ReduceInputPolicy::NoWaitNoPop,
        result.local_auxiliary);
    append(second_rows, second_columns, local_scalar, second_policy, result.local_auxiliary);
    append(
        global_tiles,
        1,
        global_scalar,
        compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
        result.global_auxiliary);
    return result;
}

inline GroupNormReducePlans make_interleaved_groupnorm_reduce_plans(
    uint32_t block_h,
    uint32_t block_w,
    uint32_t num_out_blocks,
    uint32_t num_cores,
    uint32_t single_tile_size,
    uint32_t reduce_factor,
    const GroupNormPadCorrection& pad,
    tt::tt_metal::DataType dtype,
    const ttnn::kernel_lib::host::ReduceHardwareConfig& hardware) {
    const uint32_t normal_rows = block_h / num_out_blocks;
    uint32_t last_rows = normal_rows;
    uint32_t padded_blocks = num_out_blocks;
    if (block_h % num_out_blocks != 0) {
        const uint32_t residual = block_h - num_out_blocks * normal_rows;
        padded_blocks += residual / normal_rows + 1;
        last_rows = residual % normal_rows;
    }
    const uint32_t global_tiles =
        (padded_blocks * num_cores * dfb_ex_external_slot_pitch_bytes + single_tile_size - 1) / single_tile_size;
    const float divisor =
        static_cast<float>(reduce_factor) * (pad.active ? static_cast<float>(pad.logical_hw) / pad.padded_hw : 1.0F);
    // An empty final block emits a zero tile in compute; its unused descriptor
    // must still describe a nonempty tensor.
    return make_groupnorm_reduce_plans(
        normal_rows,
        block_w,
        std::max(1u, last_rows),
        block_w,
        global_tiles,
        1.0F / divisor,
        1.0F / num_cores,
        dtype,
        hardware);
}

}  // namespace ttnn::prim

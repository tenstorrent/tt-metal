// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/kernel_lib/host/reduce_host.hpp"
#include "groupnorm_program_utils.hpp"
#include "kernels/groupnorm_constants.hpp"
#include <optional>
#include <cstdint>

namespace ttnn::prim {

struct GroupNormReducePlans {
    std::vector<std::uint32_t> calls;
    // Ordinary blocks use offset 0; the final block uses offset 3. Both
    // offsets select the same compile-time call and its preplanned variants.
    std::vector<std::uint32_t> local_runtime_args;
    ttnn::kernel_lib::host::ReduceAuxiliaryPlan local_auxiliary{2, {}};
    ttnn::kernel_lib::host::ReduceAuxiliaryPlan global_auxiliary{4, {}};

    void append_auxiliary_to(std::vector<std::uint32_t>& args) const {
        ttnn::kernel_lib::host::ReduceAuxiliaryArgs(local_auxiliary).append_to(args);
        ttnn::kernel_lib::host::ReduceAuxiliaryArgs(global_auxiliary).append_to(args);
    }
};

// Local shapes describe masked tiles. HW reduction cannot mask both the row
// and column padding, so group selection and spatial masks stay in compute.
// Sharded callers use two independent local calls. Interleaved callers set
// second_is_tail to describe a runtime alternative within one local call.
inline GroupNormReducePlans make_groupnorm_reduce_plans(
    std::uint32_t first_rows,
    std::uint32_t first_columns,
    std::uint32_t second_rows,
    std::uint32_t second_columns,
    std::uint32_t global_tiles,
    std::uint32_t first_input_cb_tiles,
    std::uint32_t second_input_cb_tiles,
    std::uint32_t global_input_cb_tiles,
    float local_scalar,
    float global_scalar,
    tt::tt_metal::DataType dtype,
    const ttnn::kernel_lib::host::ReduceHardwareConfig& hardware,
    compute_kernel_lib::ReduceInputPolicy second_policy = compute_kernel_lib::ReduceInputPolicy::NoWaitNoPop,
    compute_kernel_lib::ReduceDataFormatReconfigMode first_native_reconfig =
        compute_kernel_lib::ReduceDataFormatReconfigMode::NONE,
    bool second_is_tail = false,
    compute_kernel_lib::ReduceInputPolicy first_policy = compute_kernel_lib::ReduceInputPolicy::NoWaitNoPop) {
    using namespace tt::tt_metal;
    namespace rh = ttnn::kernel_lib::host;
    GroupNormReducePlans result;
    auto append = [&](std::uint32_t rows,
                      std::uint32_t columns,
                      std::uint32_t input_cb_tiles,
                      float scalar,
                      compute_kernel_lib::ReduceInputPolicy policy,
                      rh::ReduceAuxiliaryPlan& auxiliary,
                      compute_kernel_lib::ReduceDataFormatReconfigMode native_reconfig =
                          compute_kernel_lib::ReduceDataFormatReconfigMode::NONE,
                      std::optional<std::uint32_t> tail_rows = std::nullopt) {
        auto block = rh::ReduceBlockSpec::tiled(rows * 32, columns * 32, dtype, dtype);
        block.input_cb_tiles = input_cb_tiles;
        if (tail_rows.has_value()) {
            block.tail = rh::ReduceTailConfig{{*tail_rows * 32, columns * 32, 1}};
        }
        auto plan = rh::make_reduce_plan(
            block, ReduceOpMath::SUM, ReduceOpDim::HW, scalar, ReduceFp32Mode::Fast, hardware, policy);
        // Existing native calls retain caller-owned format state. The sharded
        // mean follows masking and must restore its reduction operands instead.
        // Add consumes two inputs instead of input/scaler and configures that pair.
        auto configure = [&](rh::ReducePlan& variant) {
            variant.reconfig_mode = variant.algorithm == compute_kernel_lib::ReduceAlgorithm::ReduceTile
                                        ? native_reconfig
                                        : compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT;
        };
        configure(plan);
        if (plan.tail_plan) {
            configure(*plan.tail_plan);
            // Full blocks read the marker at offset 0; the final block reads the tail at offset 1.
            plan.append_runtime_args(result.local_runtime_args, false);
            plan.append_runtime_args(result.local_runtime_args);
        }
        const rh::ReduceCallPlan call{
            .input_cb_id = 0,
            .auxiliary_cb_id = 1,
            .auxiliary_tile_offset = static_cast<std::uint32_t>(auxiliary.tiles.size()),
            .output_cb_id = 2,
            .accumulator_cb_id = std::nullopt,
            .plan = plan};
        rh::ReduceCallArgs(call).append_to(result.calls);
        auxiliary.tiles.insert(auxiliary.tiles.end(), plan.auxiliary_tiles.begin(), plan.auxiliary_tiles.end());
    };
    TT_FATAL(
        !second_is_tail || (second_rows <= first_rows && second_columns == first_columns),
        "Groupnorm tail must use at most the full rows and the same width");
    append(
        first_rows,
        first_columns,
        first_input_cb_tiles,
        local_scalar,
        first_policy,
        result.local_auxiliary,
        first_native_reconfig,
        second_is_tail && second_rows < first_rows ? std::optional{second_rows} : std::nullopt);
    if (!second_is_tail) {
        append(second_rows, second_columns, second_input_cb_tiles, local_scalar, second_policy, result.local_auxiliary);
    }
    append(
        global_tiles,
        1,
        global_input_cb_tiles,
        global_scalar,
        compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
        result.global_auxiliary);
    return result;
}

inline GroupNormReducePlans make_interleaved_groupnorm_reduce_plans(
    std::uint32_t block_h,
    std::uint32_t block_w,
    std::uint32_t num_out_blocks,
    std::uint32_t num_cores,
    std::uint32_t single_tile_size,
    std::uint32_t reduce_factor,
    std::uint32_t local_input_cb_tiles,
    std::uint32_t global_input_cb_tiles,
    const GroupNormPadCorrection& pad,
    tt::tt_metal::DataType dtype,
    const ttnn::kernel_lib::host::ReduceHardwareConfig& hardware) {
    TT_FATAL(num_out_blocks > 0 && block_h >= num_out_blocks, "Groupnorm reduction blocks must contain rows");
    const std::uint32_t normal_rows = block_h / num_out_blocks;
    std::uint32_t last_rows = normal_rows;
    std::uint32_t padded_blocks = num_out_blocks;
    if (block_h % num_out_blocks != 0) {
        const std::uint32_t residual = block_h - num_out_blocks * normal_rows;
        padded_blocks += residual / normal_rows + 1;
        last_rows = residual % normal_rows;
    }
    TT_FATAL(last_rows <= normal_rows, "Groupnorm final reduction block must match its full/tail/empty dispatch");
    const std::uint32_t global_tiles =
        (padded_blocks * num_cores * dfb_ex_external_slot_pitch_bytes + single_tile_size - 1) / single_tile_size;
    const float divisor =
        static_cast<float>(reduce_factor) * (pad.active ? static_cast<float>(pad.logical_hw) / pad.padded_hw : 1.0F);
    // An empty final block emits a zero tile in compute and needs no tail plan.
    return make_groupnorm_reduce_plans(
        normal_rows,
        block_w,
        last_rows == 0 ? normal_rows : last_rows,
        block_w,
        global_tiles,
        local_input_cb_tiles,
        local_input_cb_tiles,
        global_input_cb_tiles,
        1.0F / divisor,
        1.0F / num_cores,
        dtype,
        hardware,
        compute_kernel_lib::ReduceInputPolicy::NoWaitNoPop,
        compute_kernel_lib::ReduceDataFormatReconfigMode::NONE,
        true);
}

}  // namespace ttnn::prim

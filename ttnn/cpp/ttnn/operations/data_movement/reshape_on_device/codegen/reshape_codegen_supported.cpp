// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/reshape_on_device/codegen/reshape_codegen_supported.hpp"

#include <cstdint>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/hal.hpp>

#include "ttnn/operations/data_movement/reshape_on_device/codegen/reshape_codegen_program_factory.hpp"

namespace ttnn::operations::data_movement::reshape_codegen {

bool supported_by_codegen(const Tensor& input, uint32_t out_last_dim_elements, const MemoryConfig& output_mem_config) {
    if (input.layout() != ttnn::ROW_MAJOR_LAYOUT) {
        return false;
    }
    // Sharded input or output: the transport has no unshard-to-interleaved hop, so those stay on
    // native.
    if (input.memory_config().is_sharded() || output_mem_config.is_sharded()) {
        return false;
    }
    // Cross-placement calls are out of scope: the reader/writer size their CB slots and NoC
    // transfer bounds from each buffer's own alignment, and the gate below only bounds the plan
    // for the input's own buffer type. A different output buffer type is a different alignment
    // the same plan was never checked against.
    if (output_mem_config.buffer_type() != input.memory_config().buffer_type()) {
        return false;
    }
    // Scalar dtypes only: the transport is a pure byte mover keyed off element_size(), which has
    // no meaning for a block-float shared-exponent format.
    switch (input.dtype()) {
        case tt::tt_metal::DataType::BFLOAT16:
        case tt::tt_metal::DataType::FLOAT32:
        case tt::tt_metal::DataType::INT32:
        case tt::tt_metal::DataType::UINT32:
        case tt::tt_metal::DataType::UINT16: break;
        default: return false;
    }
    if (input.logical_shape().rank() < 1) {
        return false;
    }
    if (out_last_dim_elements == 0) {
        return false;
    }
    if (input.logical_shape().volume() == 0) {
        // Zero-volume reshapes are metadata-only in the free function, before any kernel routing
        // is considered; nothing here needs to serve them.
        return false;
    }
    if (input.storage_type() != ttnn::StorageType::DEVICE) {
        // Not yet on device (e.g. host-side probing); nothing to bound against, and every other
        // condition above answers for a host tensor too.
        return true;
    }

    const uint32_t elem_size = input.element_size();
    const uint32_t old_stick_bytes = static_cast<uint32_t>(input.logical_shape()[-1]) * elem_size;
    const uint32_t new_stick_bytes = out_last_dim_elements * elem_size;
    // out_last_dim_elements is the only output-shape fact this predicate is handed; the transport's
    // plan only needs total output-stick count for its work-unit count, which is the (equal)
    // logical volume divided by the new last dim.
    const uint64_t total_elements = input.logical_shape().volume();
    if (total_elements % out_last_dim_elements != 0) {
        return false;
    }
    const uint32_t num_new_sticks = static_cast<uint32_t>(total_elements / out_last_dim_elements);

    auto* device = input.device();
    const uint32_t input_alignment = device->allocator()->get_alignment(input.memory_config().buffer_type());
    const uint32_t output_alignment = device->allocator()->get_alignment(output_mem_config.buffer_type());
    const uint32_t noc_max_burst_bytes = tt::tt_metal::hal::get_noc_max_burst_size_bytes();
    const uint32_t usable_l1_bytes =
        device->l1_size_per_core() - device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);

    const auto plan = ttnn::prim::plan_reshape_rm_arbitrary_transport(
        old_stick_bytes,
        new_stick_bytes,
        num_new_sticks,
        input_alignment,
        output_alignment,
        noc_max_burst_bytes,
        usable_l1_bytes);
    // The factory's CB footprint scales with the stick widths (region_stride, slab_slot_bytes);
    // plan_reshape_rm_arbitrary_transport() reports nabatch == 0 when even the minimum staging
    // area does not fit L1, which the factory would otherwise TT_FATAL on. Reject here so `auto`
    // falls back to native instead of routing to a program that cannot be created.
    return plan.nabatch > 0;
}

bool is_demoted(const Tensor& input, uint32_t out_last_dim_elements, const MemoryConfig& output_mem_config) {
    // A last-dim-preserving ROW_MAJOR reshape is a zero-cost metadata view on the native path (see
    // ttnn::reshape's `this_is_view` -- unchanged last dim plus matching sharded/L1 placement,
    // which is guaranteed here because supported_by_codegen() already requires an unsharded output
    // in the input's own buffer type). Codegen has no such fast path; it always dispatches a
    // program. Demote so `auto` keeps the free view instead of paying for a real kernel launch to
    // reproduce it byte-for-byte.
    if (input.logical_shape().rank() >= 1 && out_last_dim_elements == input.logical_shape()[-1]) {
        return true;
    }

    // Measured on `perf_nightly`: wall-clock loses to native specifically on (BFLOAT16, L1) -- the
    // same combination regresses at every shape sampled in that dtype's stratum, and the opposite
    // combination for the same dtype (BFLOAT16+DRAM) wins at the same shapes. `device_vs_native`
    // does not track this split (it goes either way), so the cost is host/dispatch overhead, not
    // the generated kernel; the transport's per-unit region width (region_stride, derived from
    // old_stick_bytes and input_alignment in plan_reshape_rm_arbitrary_transport) is the piece that
    // differs across this dtype/buffer-type pairing and is the plausible source, but no verify run
    // has isolated it further than the regression's own boundary.
    //
    // A blanket (FLOAT32, DRAM) rule was tried here too, on the same reasoning as the BFLOAT16/L1
    // one. It over-demoted: `routing.demoted_but_faster` on the following verify named
    // codegen_reshape[40] and codegen_reshape[43] -- both FLOAT32+DRAM, both measurably faster than
    // native under forced codegen -- so the dtype/buffer-type pairing alone does not predict the
    // FLOAT32 regression the way it does for BFLOAT16. The one confirmed FLOAT32+DRAM regression in
    // that measurement (codegen_reshape[38]) is not distinguished from the wins by dtype/buffer-type
    // alone; the large-stick-count rule below already catches the other FLOAT32 regression in scope
    // (codegen_reshape[47]). Removed here rather than narrowed further, since no verify run yet
    // isolates what does distinguish 38 from 40/43.
    const bool bfloat16_on_l1 = input.dtype() == tt::tt_metal::DataType::BFLOAT16 &&
                                 output_mem_config.buffer_type() == tt::tt_metal::BufferType::L1;
    if (bfloat16_on_l1) {
        return true;
    }

    // Measured on `perf_nightly`: the two largest-output-stick-count shapes in scope (8880 and
    // 12642 new sticks) regress even in the otherwise-winning (FLOAT32, L1) combination above --
    // dispatching that many work units evidently pays enough host-side overhead (one runtime-arg
    // set per unit) to erase the per-unit device win. 5000 sits with wide margin above every
    // passing case's stick count (max 536 measured) and wide margin below both failing cases, so it
    // separates the measured data without being tuned to an exact case.
    constexpr uint64_t kLargeStickCountThreshold = 5000;
    if (input.storage_type() == ttnn::StorageType::DEVICE && out_last_dim_elements != 0) {
        const uint64_t total_new_sticks = input.logical_shape().volume() / out_last_dim_elements;
        if (total_new_sticks > kLargeStickCountThreshold) {
            return true;
        }
    }

    return false;
}

}  // namespace ttnn::operations::data_movement::reshape_codegen

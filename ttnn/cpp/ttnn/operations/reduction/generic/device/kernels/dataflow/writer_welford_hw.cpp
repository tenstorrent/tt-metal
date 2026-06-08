// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Welford HW-reduction writer kernel.
//
// Phase 1 (per output): Reads Wt partial (mean, var) tile pairs from
// cb_partial (written by the compute kernel using
// welford_finalize_to_row), combines them across W using the parallel
// Welford merge formula, applies Bessel's correction, and writes the
// combined scalar into cb_combined for the compute kernel to apply
// sqrtf (if std) and re-pack in the output format. cb_combined is
// normally fp32, but for variance output to bf16 the program
// factory may declare it as bf16 to save SRAM with no precision loss
// since data is packed to bf16 output anyways and there is no math before
// the final pack. combined_is_bf16 compile-time arg selects the path.
//
// Phase 2 (per output): Waits for the compute kernel to pack the
// output tile into cb_out (in the correct output data format), then
// NOC-writes it to DRAM.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/numeric/bfloat16.h"
#include "api/tensor/noc_traits.h"
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/normalization/groupnorm/device/kernels/dataflow/welford_combine.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t NC_per_core = get_arg_val<uint32_t>(1);
    const uint32_t output_tile_start_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t W = get_compile_time_arg_val(1);
    constexpr uint32_t tile_width = get_compile_time_arg_val(2);
    constexpr uint32_t H = get_compile_time_arg_val(3);
    constexpr bool correction = get_compile_time_arg_val(4) != 0;
    constexpr uint32_t reduce_batch_size = get_compile_time_arg_val(5);
    constexpr bool combined_is_bf16 = get_compile_time_arg_val(6) != 0;

    constexpr auto cb_partial = tt::CBIndex::c_21;
    // cb_combined: combined scalar tile written by this kernel, read back by
    // compute for repacking into the output data format. Format is Float32 by
    // default; bf16 when combined_is_bf16 is true (variance-to-bf16 path).
    constexpr auto cb_combined = tt::CBIndex::c_22;
    // cb_out: output tile packed by compute in the correct data format.
    constexpr auto cb_out = tt::CBIndex::c_16;

    constexpr auto dst_args = TensorAccessorArgs<7>();

    // welford_finalize_to_row stores 32 per-column values in tile row 0.
    // In tile format, row 0 spans Face 0 (columns 0-15) and Face 1 (columns 16-31).
    // Each face has FACE_W rows × FACE_W columns elements.
    constexpr uint32_t FACE_W = tt::constants::FACE_WIDTH;
    constexpr uint32_t FACE_ELEMENTS = FACE_W * FACE_W;
    constexpr uint32_t last_tile_cols = (W % tile_width == 0) ? tile_width : W % tile_width;

    const uint32_t partial_tile_size_bytes = get_tile_size(cb_partial);
    const uint32_t out_tile_size_bytes = get_tile_size(cb_out);

    Noc noc;
    CircularBuffer cb_partial_obj(cb_partial);
    CircularBuffer cb_combined_obj(cb_combined);
    CircularBuffer cb_out_obj(cb_out);

    const auto tensor_out = TensorAccessor(dst_args, dst_addr);

    // NC_per_core is the total number of NC slices assigned to this core.
    // Each output element is produced by combining reduce_batch_size
    // consecutive NC slices (each contributing Wt partial tile pairs).
    uint32_t num_outputs = NC_per_core / reduce_batch_size;

    for (uint32_t out = 0; out < num_outputs; ++out) {
        // --- Phase 1: W-combine all per-column partials into one scalar ---
        WelfordStats<float> running = {0.0f, 0.0f, 0};

        for (uint32_t b = 0; b < reduce_batch_size; ++b) {
            for (uint32_t wt = 0; wt < Wt; ++wt) {
                cb_partial_obj.wait_front(2);

                auto means_addr = cb_partial_obj.get_read_ptr();
                auto vars_addr = means_addr + partial_tile_size_bytes;

                // cb_partial is Float32: each element is 4 bytes.
                auto* means_ptr = reinterpret_cast<volatile float*>(means_addr);
                auto* vars_ptr = reinterpret_cast<volatile float*>(vars_addr);

                uint32_t num_cols = (wt < Wt - 1) ? tile_width : last_tile_cols;
                for (uint32_t c = 0; c < num_cols; ++c) {
                    // In tile row format, columns 0-15 are in Face 0 and
                    // columns 16-31 are in Face 1 (offset by FACE_ELEMENTS).
                    uint32_t idx = (c < FACE_W) ? c : (FACE_ELEMENTS + c - FACE_W);
                    WelfordStats<float> partial;
                    partial.mean = means_ptr[idx];
                    partial.variance = vars_ptr[idx];
                    partial.count = H;
                    running = combine(running, partial);
                }

                cb_partial_obj.pop_front(2);
            }
        }

        float final_var = running.variance;
        if constexpr (correction) {
            uint32_t N = running.count;
            final_var = final_var * static_cast<float>(N) / static_cast<float>(N - 1);
        }

        // Write the combined scalar into a tile in cb_combined.  The compute
        // kernel will unpack this and re-pack into cb_out in the correct
        // output data format (using the packer hardware).
        //
        // Only Face 0 row 0 (FACE_W elements) needs zeroing.  The scalar
        // lives at position [0,0]; the remaining FACE_W-1 elements in
        // the same row share a BFP exponent group, so they must be zero
        // to avoid corrupting the scalar's mantissa precision in
        // BFLOAT8_B output.  Other face rows have independent exponents
        // and are never read (the output is a single scalar), so stale
        // L1 contents there are harmless.
        cb_combined_obj.reserve_back(1);
        if constexpr (combined_is_bf16) {
            auto* combined_ptr = reinterpret_cast<uint16_t*>(cb_combined_obj.get_write_ptr());
            for (uint32_t i = 0; i < FACE_W; ++i) {
                combined_ptr[i] = 0;
            }
            // fp32_to_bf16 applies round-to-nearest-even, matching the packer
            // hardware so the output is bit-identical to a packer-produced bf16.
            combined_ptr[0] = fp32_to_bf16(final_var);
        } else {
            auto* combined_ptr = reinterpret_cast<float*>(cb_combined_obj.get_write_ptr());
            for (uint32_t i = 0; i < FACE_W; ++i) {
                combined_ptr[i] = 0.0f;
            }
            combined_ptr[0] = final_var;
        }
        cb_combined_obj.push_back(1);

        // --- Phase 2: NOC-write the output tile (packed by compute) to DRAM ---
        cb_out_obj.wait_front(1);
        uint32_t out_tile_id = output_tile_start_id + out;
        noc.async_write(cb_out_obj, tensor_out, out_tile_size_bytes, {}, {.page_id = out_tile_id});
        noc.async_writes_flushed();
        cb_out_obj.pop_front(1);
    }

    noc.async_write_barrier();
}

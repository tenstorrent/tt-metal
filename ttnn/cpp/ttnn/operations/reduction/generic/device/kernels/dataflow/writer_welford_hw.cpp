// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 Welford HW-reduction writer kernel.
//
// Phase 1 (per output): Reads Wt partial (mean, var) tile pairs from cb_partial
// (written by the compute kernel), combines them across W using the parallel
// Welford merge formula, applies Bessel's correction, and writes the combined
// scalar into a Float32 tile in cb_combined for the compute kernel to apply
// sqrtf (if std) and re-pack in the output format.
// Phase 2 (per output): Waits for the compute kernel to pack the output tile
// into cb_out, then NOC-writes it to DRAM.
//
// Migration notes: the output tensor is bound by name (ta::output_tensor); the
// host declares a TensorParameter and supplies the MeshTensor via
// ProgramRunParams::TensorArg, so the kernel constructs a TensorAccessor
// directly without needing is_dram or aligned_page_size as named CTAs.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/dataflow_buffer.h"
#include "experimental/noc.h"
#include "experimental/tensor.h"
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/normalization/groupnorm/device/kernels/dataflow/welford_combine.h"

void kernel_main() {
    // Per-node runtime arguments.
    const uint32_t NC_per_core = get_arg(args::NC_per_core);
    const uint32_t output_tile_start_id = get_arg(args::output_tile_start_id);

    // Compile-time arguments.
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t W = get_arg(args::W);
    constexpr uint32_t tile_width = get_arg(args::tile_width);
    constexpr uint32_t H = get_arg(args::H);
    constexpr bool correction = get_arg(args::correction) != 0;
    constexpr uint32_t reduce_batch_size = get_arg(args::reduce_batch_size);

    experimental::DataflowBuffer dfb_partial(dfb::partial);
    experimental::DataflowBuffer dfb_combined(dfb::combined);
    experimental::DataflowBuffer dfb_output(dfb::output);

    const uint32_t cb_partial = dfb_partial.get_id();

    // welford_finalize_to_row stores 32 per-column values in tile row 0. In tile
    // format, row 0 spans Face 0 (cols 0–15) and Face 1 (cols 16–31). Each face
    // is FACE_W rows × FACE_W columns elements.
    constexpr uint32_t FACE_W = tt::constants::FACE_WIDTH;
    constexpr uint32_t FACE_ELEMENTS = FACE_W * FACE_W;
    constexpr uint32_t last_tile_cols = (W % tile_width == 0) ? tile_width : W % tile_width;

    const uint32_t partial_tile_size_bytes = get_tile_size(cb_partial);
    const uint32_t out_tile_size_bytes = get_tile_size(dfb_output.get_id());

    TensorAccessor output_accessor(ta::output_tensor);

    experimental::Noc noc;

    // NC_per_core is the total number of NC slices assigned to this core. Each
    // output element is produced by combining reduce_batch_size consecutive NC
    // slices (each contributing Wt partial tile pairs).
    uint32_t num_outputs = NC_per_core / reduce_batch_size;

    for (uint32_t out = 0; out < num_outputs; ++out) {
        // Phase 1: W-combine all per-column partials into one scalar.
        WelfordStats<float> running = {0.0f, 0.0f, 0};

        for (uint32_t b = 0; b < reduce_batch_size; ++b) {
            for (uint32_t wt = 0; wt < Wt; ++wt) {
                dfb_partial.wait_front(2);

                auto means_addr = dfb_partial.get_read_ptr();
                auto vars_addr = means_addr + partial_tile_size_bytes;

                auto* means_ptr = reinterpret_cast<volatile float*>(means_addr);
                auto* vars_ptr = reinterpret_cast<volatile float*>(vars_addr);

                uint32_t num_cols = (wt < Wt - 1) ? tile_width : last_tile_cols;
                for (uint32_t c = 0; c < num_cols; ++c) {
                    // In tile row format, columns 0–15 are in Face 0 and columns 16–31
                    // are in Face 1 (offset by FACE_ELEMENTS).
                    uint32_t idx = (c < FACE_W) ? c : (FACE_ELEMENTS + c - FACE_W);
                    WelfordStats<float> partial;
                    partial.mean = means_ptr[idx];
                    partial.variance = vars_ptr[idx];
                    partial.count = H;
                    running = combine(running, partial);
                }

                dfb_partial.pop_front(2);
            }
        }

        float final_var = running.variance;
        if constexpr (correction) {
            uint32_t N = running.count;
            final_var = final_var * static_cast<float>(N) / static_cast<float>(N - 1);
        }

        // Write combined scalar into a Float32 tile in cb_combined. Only Face 0 row
        // 0 (FACE_W floats) is zeroed; the scalar lives at [0,0]; the remaining
        // FACE_W-1 elements share a BFP exponent group so they must be zero to
        // avoid corrupting the scalar's mantissa precision in BFLOAT8_B output.
        dfb_combined.reserve_back(1);
        auto* combined_ptr = reinterpret_cast<float*>(dfb_combined.get_write_ptr());
        for (uint32_t i = 0; i < FACE_W; ++i) {
            combined_ptr[i] = 0.0f;
        }
        combined_ptr[0] = final_var;
        dfb_combined.push_back(1);

        // Phase 2: NOC-write the output tile (packed by compute) to DRAM.
        dfb_output.wait_front(1);
        const uint32_t out_tile_id = output_tile_start_id + out;
        noc.async_write(
            dfb_output, output_accessor, out_tile_size_bytes, {.offset_bytes = 0}, {.page_id = out_tile_id});
        noc.async_writes_flushed();
        dfb_output.pop_front(1);
    }

    noc.async_write_barrier();
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/experimental/quasar/matmul/device/matmul_device_operation_types.hpp"

namespace ttnn::prim::qsr {

// Derived from a MatmulUnifiedProgramConfig + operand shapes by one function shared by validation,
// output-spec derivation and the factory, so they cannot disagree (all config constraints TT_FATAL there).
struct UnifiedMatmulPlan {
    uint32_t M_tiles = 0;
    uint32_t K_tiles = 0;
    uint32_t N_tiles = 0;
    uint32_t batch_size = 0;  // batches of A (and of C)
    // Tiles from one batch of B to the next: K_tiles * N_tiles when B has a [K x N] per batch, 0 when B is a
    // single [K x N] (or [1 x K x N]) that every batch of A multiplies. A's stride is always M_tiles * K_tiles.
    uint32_t B_batch_stride_tiles = 0;

    // Blocking, after the config's auto fields are resolved.
    // The C slices of one batch are walked across N then down M; C_slices_across_N is the walk's row
    // length, which the factory's per-core start positions and the sharded output layout both follow.
    uint32_t C_slices_across_N = 0;
    // Valid element columns in A's last K tile; 0 when K is a tile multiple (reader zeroes the rest).
    uint32_t A_last_K_tile_valid_columns = 0;
    // DST capacity (and the unpack modes the factory programs) assumed this accumulation mode.
    bool fp32_dest_acc_en = false;
    uint32_t C_slice_M_tiles = 0;
    uint32_t C_slice_N_tiles = 0;
    uint32_t K_chunk_tiles = 0;
    uint32_t num_K_chunks = 0;  // K_tiles / K_chunk_tiles
    uint32_t subblock_M_tiles = 0;
    uint32_t subblock_N_tiles = 0;
    // C slice dims rounded up to subblock multiples (equal when the subblock divides the slice).
    // Buffers and kernel loops use these; overshoot rows/columns are clipped on write.
    uint32_t C_slice_M_padded_tiles = 0;
    uint32_t C_slice_N_padded_tiles = 0;

    // Compute threads per core (NEOs). The C slice's subblocks, numbered across N then down M, are assigned
    // round-robin to the threads, which all see the whole A and B slices. A thread's share of the C slice
    // (ceil(subblocks / threads) subblocks, back to back) is C_entries_per_thread entries of C_slice and
    // C_partials, the credits it moves per K chunk: one entry with several threads (the thread's tile
    // counter then stays at one address, which block packs and unpacks need), one entry per tile with one.
    uint32_t num_compute_threads = 1;
    uint32_t C_entries_per_thread = 0;

    // C slice assignment: one batch's C slices, walked across N then down M, split into contiguous
    // runs per active core (the factory derives the per-core RTAs).
    uint32_t C_slices_per_batch = 0;
    tt::tt_metal::ShardOrientation orientation = tt::tt_metal::ShardOrientation::ROW_MAJOR;
    std::vector<tt::tt_metal::CoreCoord> cores;
    uint32_t max_C_slices_per_core = 0;  // sizes the DFBs and gates partials aliasing

    // Borrowed operand: its L1 shard on each active core is bound as the DFB itself, no copy.
    // Needs batch 1, one C slice per core, and a shard grid in assignment order. C also needs one
    // compute thread: with several, each thread owns every N-th C_slice entry, not the shard's tile order.
    bool borrow_A = false;
    bool borrow_B = false;
    bool borrow_C = false;

    // DFB sizing; entry sizes are in bytes. An A_slice / B_slice entry holds one tile; C_slice / C_partials
    // hold C_entries_per_thread entries per thread (see above).
    bool packer_l1_acc_en = false;
    tt::DataFormat A_format{};
    tt::DataFormat B_format{};
    tt::DataFormat C_format{};
    tt::DataFormat C_partials_format{};
    uint32_t A_entry_bytes = 0;
    uint32_t B_entry_bytes = 0;
    uint32_t C_entry_bytes = 0;
    uint32_t C_partials_entry_bytes = 0;
    uint32_t A_slice_entries = 0;
    uint32_t B_slice_entries = 0;
    uint32_t C_slice_entries = 0;
    uint32_t C_partials_entries = 0;
    // C_partials shares C_slice's L1; only safe when partials are never live while C_slice holds unread data.
    bool alias_C_partials_onto_C_slice = false;
    uint64_t l1_bytes = 0;  // total DFB footprint per core

    // Only meaningful for a sharded output: the layout implied by how the C slices tile C.
    tt::tt_metal::TensorMemoryLayout sharded_output_layout{};
};

namespace detail {

// Max-volume DST-filling subblock among the shapes the caller's fits predicate accepts (L1 fit and
// borrow preservation stay external): the C slice is rounded up to subblock multiples and the
// overshoot is clipped on write. Ties prefer the least padding waste; 1x1 (no padding) if nothing
// is accepted, and the caller's DFB sizing FATALs with the full breakdown. With several compute threads
// the subblocks are assigned round-robin, so the least work on the busiest thread comes first.
template <typename FitsSubblock>
std::pair<uint32_t, uint32_t> maximize_subblock_size(
    uint32_t C_slice_M_tiles,
    uint32_t C_slice_N_tiles,
    uint32_t dst_capacity_tiles,
    uint32_t num_compute_threads,
    const FitsSubblock& fits) {
    std::pair<uint32_t, uint32_t> best{1, 1};
    uint64_t best_busiest_thread_tiles = UINT64_MAX;
    uint64_t best_volume = 0;
    uint64_t best_padded_area = UINT64_MAX;
    for (uint32_t h = 1; h <= dst_capacity_tiles; ++h) {
        for (uint32_t w = 1; h * w <= dst_capacity_tiles; ++w) {
            const uint64_t volume = h * w;
            const uint64_t num_subblocks = (uint64_t)tt::div_up(C_slice_M_tiles, h) * tt::div_up(C_slice_N_tiles, w);
            const uint64_t padded_area = num_subblocks * volume;
            // Padding included; 0 with one thread, which leaves the volume-then-padding rule.
            const uint64_t busiest_thread_tiles =
                num_compute_threads > 1 ? tt::div_up(num_subblocks, (uint64_t)num_compute_threads) * volume : 0;
            const bool better = busiest_thread_tiles != best_busiest_thread_tiles
                                    ? busiest_thread_tiles < best_busiest_thread_tiles
                                : volume != best_volume ? volume > best_volume
                                                        : padded_area < best_padded_area;
            if (better && fits(h, w)) {
                best = {h, w};
                best_busiest_thread_tiles = busiest_thread_tiles;
                best_volume = volume;
                best_padded_area = padded_area;
            }
        }
    }
    return best;
}

}  // namespace detail

// `output` is the C tensor when the caller supplied one (or the op already created it); it decides whether
// C can be packed in place. Without it the op allocates C from the plan, which matches by construction.
UnifiedMatmulPlan plan_unified_matmul(
    const tt::tt_metal::IDevice& device,
    const ttnn::Tensor& input_tensor_a,
    const ttnn::Tensor& input_tensor_b,
    const operations::experimental::quasar::matmul::MatmulUnifiedProgramConfig& config,
    const MatmulParams& attributes,
    const std::optional<ttnn::Tensor>& output);

struct MatmulUnifiedProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const MatmulParams& operation_attributes,
        const MatmulInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value);
};

}  // namespace ttnn::prim::qsr

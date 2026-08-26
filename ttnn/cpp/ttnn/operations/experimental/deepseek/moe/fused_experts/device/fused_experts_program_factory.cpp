// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_experts_device_operation.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <numeric>
#include <string>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::experimental::deepseek::moe::fused_experts {

using namespace tt;
using namespace tt::tt_metal;

namespace {
constexpr std::string_view kKernelDir =
    "ttnn/cpp/ttnn/operations/experimental/deepseek/moe/fused_experts/device/kernels";

// DRAM weight layout: 64 shards, one per original 8x8 core. H = 64 * 64 columns.
constexpr uint32_t kNumWeightShards = 64;
constexpr uint32_t kGridY = 8;

// Each weight shard owns kOutTilesPerCore tiles (64 columns) of the H-dim output row, so a
// down shard is [I, 64] and H must be kNumWeightShards * 64.
constexpr uint32_t kOutTilesPerCore = 2;

// 6-expert path: 16 cores per expert on a 12x8 = 96-core grid (2 columns x 8 rows per expert).
constexpr uint32_t kParallelExperts = 6;
constexpr uint32_t kCoresPerExpertParallel = 16;

// The SwiGLU I dim is split across DRAM shards (rather than the 64-columns-per-core the output
// uses) so every NoC port is busy fetching gate_up weights during phase 1. Shard width stays one
// 32-column I-tile (a [gate_32 | up_32] DRAM shard). At I == 2048 that is 64 shards; TP slices I
// (e.g. 512 -> 16 shards) so the 16 cores of a group still each own at least one shard rather
// than four cores covering the whole I dim and the rest sitting idle.
uint32_t swiglu_tiles_per_shard_for(uint32_t i_tiles) { return std::max<uint32_t>(1u, i_tiles / kNumWeightShards); }

uint32_t align_up_32(uint32_t x) { return (x + 31u) & ~31u; }
}  // namespace

// The B (<= 32) token rows of a batch share a single tile row, so every tile count, matmul and NoC
// transfer below is the same as it is for one token: a "row" in this description is the whole
// [B, ...] tile row. Batch enters only through the routing weights -- the hit set is the union of
// the rows' selections, and each expert's down output is scaled by a per-token weight column
// instead of one scalar.
//
// The selected experts are processed in BLOCKS of `experts_block` (operation_attributes
// .experts_block_size), and the two phases below run once per block. Only a block's activations are
// resident, so L1 is sized by the block rather than by the number of selected experts -- which is
// what lets a batch whose tokens select disjoint experts (up to 32 * top_k of them) run at all. Each
// expert is still fetched exactly once and the arithmetic is unchanged; a block costs one
// gather/broadcast synchronization. A single block (the default) reproduces the original pipeline,
// including a single-slot cb_act.
//
// Pipeline (per block of experts: two phases with one synchronization between):
//   - {0,0} (NoC 0) reads the routing ids and scores, computes/broadcasts the selected ("hit")
//     expert ids (ascending) and their per-token weights, and acts as the activation-gather leader.
//   - {1,0} (NoC 1) reads the activation tile row and broadcasts it to every
//     core's L1 (cb_input).
//   - PHASE 1 -- gate_up + SwiGLU for the block's experts: each SwiGLU core fetches its
//     [K, 64*swiglu_tiles_per_core] gate_up shard per expert (one NoC read -- a per-core
//     [gate | up] block) and produces its slice of each expert's activation act[B, I]. The I
//     dim is spread over all 64 cores (one tile each at I == 2048), so every core's NoC port
//     contributes to this DRAM-bound phase. Each core's writer scatters the block's expert j act
//     tiles to {0,0}'s cb_act slot at tile offset (j*i_tiles + idx*swiglu_tiles_per_core).
//   - SYNC -- gather + broadcast: once {0,0} has the block's chunks from every SwiGLU core
//     (num_producers per expert, via sem_gather), it multicasts the whole block of activations
//     back to every core in one shot (sem_bcast, whose value is the number of blocks broadcast
//     so far). Within a block cb_act is never reused, so no per-expert back-pressure is needed;
//     between blocks it is double-buffered, and the leader reserves the next slot before
//     broadcasting -- so a core that has received block j's broadcast knows the leader's slot for
//     block j+1 is free and may scatter into it. Symmetrically, a producer's scatter for block j+1
//     proves its own compute is past block j, which is what makes the leader's multicast into every
//     core's slot safe. Non-producer cores (only when I < 32*64) have no scatter to prove that, so
//     they bump sem_gather once per block instead.
//   - PHASE 2 -- DOWN matmul for the block's experts: each of the 64 cores fetches its [I, H/64]
//     down shard per expert (one NoC read) and multiplies it by that expert's activation to
//     produce its 2-tile slice of the output row[B, H]. The compute kernel scales each
//     expert's slice by the tokens' routing weights for it (a per-token weight tile) and
//     accumulates across all blocks, so the writer writes a single [1, B, H] DRAM output
//     tile row (the weighted sum).
ProgramDescriptor FusedExpertsDeviceOperation::MultiCore::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    // Routing arrives as the router produced it: each token's selected expert ids plus the score
    // row they index. Only the {0,0} sender kernel reads the two, and only to populate cb_bcast;
    // everything downstream of that consumes cb_bcast.
    const auto& routing_tensor = tensor_args.routing_indices;
    const auto& input_tensor = tensor_args.input_tensor;
    auto& output_tensor = tensor_return_value;

    auto* routing_buffer = routing_tensor.buffer();
    auto* score_buffer = tensor_args.routing_scores.buffer();
    auto* input_buffer = input_tensor.buffer();
    auto* out_buffer = output_tensor.buffer();
    auto* device = routing_tensor.device();

    const auto grid = device->compute_with_storage_grid_size();
    const uint32_t num_weights = static_cast<uint32_t>(tensor_args.gate_up_weights.size());
    const uint32_t num_active = operation_attributes.num_experts;
    const bool parallel_experts = num_active == kParallelExperts;
    const uint32_t GRID_X = parallel_experts ? (kParallelExperts * 2u) : 8u;
    const uint32_t GRID_Y = kGridY;
    const uint32_t cores_per_expert = parallel_experts ? kCoresPerExpertParallel : kNumWeightShards;
    const uint32_t num_expert_groups = parallel_experts ? kParallelExperts : 1u;
    // Down/H shards per core: H is always 64-way sharded (4096 cols), so 16 cores cover 4 shards each.
    const uint32_t shards_per_core = kNumWeightShards / cores_per_expert;
    TT_FATAL(
        grid.x >= GRID_X && grid.y >= GRID_Y,
        "fused_experts: expected at least {}x{} compute grid, got {}x{}",
        GRID_X,
        GRID_Y,
        grid.x,
        grid.y);

    // The op takes all experts' weights and uses the routing weights to select which
    // ones to run. `num_weights` is the total provided (and the routing-row width);
    // `num_active` is the routing-selected count that drives the fetch / compute /
    // writer loops and the number of output rows.
    const uint32_t sentinel = num_weights;  // "no expert" marker for unused id slots

    // Experts run in blocks of `experts_block`: phase 1, the gather/broadcast sync and phase 2 all
    // run once per block, and only one block's activations are resident. The last block is short
    // when num_active is not a multiple of the block size. The 6-expert / 96-core path assigns one
    // expert per 16-core group, so it always runs a single block of one expert.
    const uint32_t experts_block =
        parallel_experts ? 1u : std::min(operation_attributes.experts_block_size, num_active);
    const uint32_t num_blocks = parallel_experts ? 1u : (num_active + experts_block - 1u) / experts_block;
    // With more than one block, cb_act is double-buffered: the leader reserves block j+1's slot
    // before broadcasting block j, which is how the other cores learn that the leader's next slot is
    // free (see the pipeline description above). One block needs no such handoff, so it keeps the
    // original single slot -- and so the original (twice as large) usable block size.
    const uint32_t act_slots = num_blocks > 1u ? 2u : 1u;

    // Token rows computed together. B <= 32 (validated), so all rows live in ONE tile row and the
    // whole tile-level pipeline below -- tile counts, matmuls, gather, output pages -- is exactly
    // what it is for a single token. Batch only shows up in the routing path: there are B routing
    // rows to scan for hits, and each expert's routing weight becomes a per-row column vector
    // instead of one scalar.
    const uint32_t batch = static_cast<uint32_t>(input_tensor.logical_shape()[-2]);

    // gate_up weights are [K=H, N=2I] per expert (TILE layout), reshaped+permuted on the
    // host into per-shard [gate_32 | up_32] blocks. Each DRAM shard is one I-tile of gate
    // plus its paired up tile. All experts share the same layout, so one TensorAccessorArgs
    // (from weight 0) is reused.
    const auto& gate_up0 = tensor_args.gate_up_weights.front();
    auto* gate_up0_buffer = gate_up0.buffer();
    constexpr uint32_t TILE_DIM = 32;
    const uint32_t k_tiles = static_cast<uint32_t>(gate_up0.logical_shape()[-2]) / TILE_DIM;
    const uint32_t n_tiles = static_cast<uint32_t>(gate_up0.logical_shape()[-1]) / TILE_DIM;  // 2I / 32
    const uint32_t i_tiles = n_tiles / 2u;  // SwiGLU output tile cols (I / 32)

    // Token-row tile shape. Every CB whose tiles hold "one tile row of B tokens" -- cb_input,
    // cb_out, cb_mm, cb_act, cb_rscalar, cb_down_out, cb_acc, cb_wtmp -- and the DRAM output
    // tensor use this tile. Weights (gate_up, down) and routing (indices/scores/bcast) stay at
    // 32x32: they don't carry per-token rows and their kernel-side layout math is bound to
    // 16x16 faces. Width must be 32 (kernels index tile columns as 32-wide); height can be any
    // supported tiny value (1, 2, 4, 8, 16, 32). Bfp8_b at tiny heights is now valid tt-llk
    // support, so cb_act / cb_out keep the Bfp8_b format their L1 budget was sized for.
    const auto& input_tile = input_tensor.tensor_spec().tile();
    const uint32_t input_tile_h = input_tile.get_height();
    const uint32_t input_tile_hw = input_tile.get_tile_hw();
    // Tiny-tile face layout: face_r_dim = min(tile_h, 16), num_face_rows = 1 for tile_h <= 16
    // else 2. Width is always 32 => 2 face columns per row of faces.
    const uint32_t face_r_dim = std::min<uint32_t>(input_tile_h, 16u);
    const uint32_t num_face_rows = (input_tile_h + 15u) / 16u;
    const TileDescriptor input_tile_desc(input_tile);
    const uint32_t weight_tile_bytes = static_cast<uint32_t>(gate_up0_buffer->page_size());
    // Number of SwiGLU cores and each one's share of the I dim (see swiglu_tiles_per_shard_for).
    const uint32_t swiglu_tiles_per_core = swiglu_tiles_per_shard_for(i_tiles);
    const uint32_t num_producers = parallel_experts ? cores_per_expert : (i_tiles / swiglu_tiles_per_core);
    // Gate_up/I shards per core: I is 64-way only at the full 2048 width. TP slices it (e.g. I=512
    // -> 16 tiles), so 16 cores cover one I-tile each rather than four.
    const uint32_t i_shards_per_core = parallel_experts ? (i_tiles / cores_per_expert) : 1u;
    TT_FATAL(
        !parallel_experts || (i_tiles % cores_per_expert == 0 && i_shards_per_core >= 1u),
        "fused_experts: 16-core-per-expert path needs I/32 ({}) divisible by cores_per_expert ({})",
        i_tiles,
        cores_per_expert);
    TT_FATAL(
        i_tiles == (parallel_experts ? cores_per_expert * i_shards_per_core : num_producers) * swiglu_tiles_per_core,
        "fused_experts: I/32 ({}) must equal the product of SwiGLU cores/shards ({}) and tiles_per_shard ({})",
        i_tiles,
        parallel_experts ? cores_per_expert * i_shards_per_core : num_producers,
        swiglu_tiles_per_core);
    // Each core's weight slice is its gate tiles + paired up tiles per k-row.
    const uint32_t weight_slice_tiles = k_tiles * (2u * swiglu_tiles_per_core);
    // Double-buffer the weight slice so the reader can hold one expert's slice ready in L1
    // while compute consumes the previous expert's, overlapping data movement with
    // computation (see tech_reports/Saturating_DRAM_bandwidth -- "In0 and in1 shards are
    // also double buffered, to overlap the data movement with computation"). Down weights
    // reuse this CB, so it must be at least as large as the (also double-buffered) down slice.
    // This is the floor; it is rounded up below so the CB holds a whole number of slices of
    // both phases, and how many down slices it ends up holding sets the prefetch depth.
    const uint32_t min_weights_cb_bytes = 2u * weight_slice_tiles * weight_tile_bytes;

    // down weights are [I, H] per expert (TILE layout), DRAM ND-sharded into [I, H/64]
    // column blocks (one per core). Core idx owns the H output cols [idx*64, idx*64+64) ->
    // its 2 output tiles, and needs the full I (== gate_up output) contraction dim, so its
    // shard is [i_tiles, 2] tiles. All experts share one layout, so weight 0's accessor is
    // reused (the fetch indexes by shard id == this core's index).
    const auto& down0 = tensor_args.down_weights.front();
    auto* down0_buffer = down0.buffer();
    const uint32_t down_slice_tiles = i_tiles * kOutTilesPerCore;  // [I, 64] = i_tiles * 2 tiles
    const uint32_t down_tile_bytes = static_cast<uint32_t>(down0_buffer->page_size());
    const uint32_t down_slice_bytes = down_slice_tiles * down_tile_bytes;
    const uint32_t down_cb_bytes = 2u * down_slice_bytes;  // double-buffered
    const tt::DataFormat down_df = datatype_to_dataformat_converter(down0.dtype());

    const tt::DataFormat gate_up_df = datatype_to_dataformat_converter(gate_up0.dtype());
    const tt::DataFormat routing_df = datatype_to_dataformat_converter(routing_tensor.dtype());
    const tt::DataFormat out_df = datatype_to_dataformat_converter(output_tensor.dtype());
    const tt::DataFormat input_df = datatype_to_dataformat_converter(input_tensor.dtype());

    constexpr uint32_t out_elem_bytes = 4;  // uint32 expert ids (broadcast scratch)
    // The ids are one TILE page covering every token row at once (B <= 32 and top_k <= 16 both fit
    // inside a single 32x32 tile), so there is exactly one page to read. It lands in cb_routing at
    // the buffer's *aligned* page stride, so the read's L1 destination shares the alignment of the
    // DRAM page it comes from.
    const uint32_t routing_page_bytes = static_cast<uint32_t>(routing_buffer->page_size());
    const uint32_t routing_row_stride = static_cast<uint32_t>(routing_buffer->aligned_page_size());
    // The score row lives in its own tile row of E/32 pages, read whole and then indexed in L1 (the
    // kernel needs at most top_k scattered elements per token, but reading tiles keeps every NoC
    // transfer page-aligned).
    const uint32_t score_page_bytes = static_cast<uint32_t>(score_buffer->page_size());
    const uint32_t score_page_stride = static_cast<uint32_t>(score_buffer->aligned_page_size());
    const uint32_t score_pages = static_cast<uint32_t>(score_buffer->num_pages());
    const uint32_t top_k = operation_attributes.top_k;
    // topk emits uint16 ids; ttnn.embedding can only gather from a bfloat16 table, so a
    // table-driven router delivers the same ids as bf16 (exact below 256). Both are 2-byte
    // elements in the same tile geometry -- only the decode differs.
    const bool index_is_bf16 = routing_tensor.dtype() == tt::tt_metal::DataType::BFLOAT16;
    // cb_bcast carries the compacted expert ids (num_weights uint32, ascending hit ids padded
    // with the sentinel) followed by the active experts' routing weights (num_active * batch fp32
    // bit patterns, hit-major then token row), broadcast to every core in one multicast.
    const uint32_t bcast_page_bytes = (num_weights + num_active * batch) * out_elem_bytes;

    // Activation is TILE layout [1,1,B,H] with B <= 32 -> Kt == k_tiles tiles (one tile-row).
    const uint32_t input_page_size = static_cast<uint32_t>(input_buffer->page_size());
    const uint32_t input_num_pages = static_cast<uint32_t>(input_buffer->num_pages());

    // Output is TILE [1, B, H] bf16 (the per-token routing-weighted sum of every active expert's
    // down matmul): each core writes its 2 output tiles (its 64-column H slice) of the single tile
    // row, which covers all B tokens.
    const uint32_t out_tile_bytes = static_cast<uint32_t>(out_buffer->page_size());

    // The gathered activation is stored as Bfp8_b (not bf16) to keep the resident
    // [num_active, I] block -- the dominant L1 consumer -- within the L1 budget. The SwiGLU
    // output (cb_out) is packed in the same format so the writer can scatter it byte-for-byte
    // into the leader's cb_act, and the down matmul reads it as its bf8 in0 (paired with the
    // bf4 down weights). The down output stays bf16 to match the DRAM output tensor.
    // Sized from the input tile: tiny-tile bfp8 is now supported by tt-llk, so a (tile_h, 32)
    // bfp8 tile packs to input_tile.get_tile_size(Bfp8_b) bytes (num_faces * face_r_dim * 16 +
    // num_faces * 16 exponent bytes for the block-float format).
    const tt::DataFormat act_df = tt::DataFormat::Bfp8_b;
    const uint32_t act_tile_bytes = input_tile.get_tile_size(act_df);

    // SwiGLU clamp limit, passed to the compute kernel as a bit-cast float (the kernel
    // derives -limit internally).
    const uint32_t limit_bits = std::bit_cast<uint32_t>(operation_attributes.swiglu_limit);

    // cb_routing is the {0,0} sender's private scratch for turning the routing input into the
    // cb_bcast id/weight list: the single id page, then the score tile row, then the selection
    // scratch -- a num_weights-bit "was this expert selected" bitmap, a num_weights-entry uint16
    // table mapping an expert id to its position in the compacted hit list, and the batch's decoded
    // ids. All three are O(E) or O(B*k) and too large for the RISC's stack, so they are carved out
    // of this CB instead.
    const uint32_t score_l1_offset = align_up_32(routing_row_stride);
    const uint32_t scratch_l1_offset = align_up_32(score_l1_offset + score_pages * score_page_stride);
    const uint32_t bitmap_bytes = align_up_32(((num_weights + 31u) / 32u) * 4u);
    const uint32_t rank_bytes = align_up_32(num_weights * 2u);
    const uint32_t scratch_bytes = bitmap_bytes + rank_bytes + align_up_32(batch * top_k * 2u);
    const uint32_t routing_cb_bytes = std::max<uint32_t>(scratch_l1_offset + scratch_bytes, 32u);
    const uint32_t bcast_cb_bytes = std::max<uint32_t>(align_up_32(bcast_page_bytes), 32u);
    const uint32_t input_cb_bytes = input_num_pages * input_page_size;
    // Double-buffer the matmul output so compute can run ahead of the writer. Each core
    // produces swiglu_tiles_per_core SwiGLU output tiles (its slice of the I dim) per expert.
    // cb_out holds the bf8 SwiGLU activation (== act_df) so the writer can scatter it
    // directly into cb_act.
    const uint32_t out_cb_bytes = parallel_experts ? i_shards_per_core * swiglu_tiles_per_core * act_tile_bytes
                                                   : 2u * swiglu_tiles_per_core * act_tile_bytes;
    // Matmul staging buffer (fp32 for full precision before the SwiGLU SFPU pass): phase 1
    // stages 2*swiglu_tiles_per_core tiles (gate | up) and phase 2 reuses it for the
    // kOutTilesPerCore down-matmul tiles, so it is sized for whichever is larger. The DEST
    // tile llk-matmul produces has in0's face_r_dim (== the input tile's), so the fp32 pack
    // writes input_tile_hw * 4 bytes per tile -- tiny at tiny input heights, full-sized at 32.
    const uint32_t mm_tile_bytes = input_tile_hw * 4u;
    const uint32_t mm_cb_bytes = std::max(2u * swiglu_tiles_per_core, kOutTilesPerCore) * mm_tile_bytes;

    // Gathered activation: ONE BLOCK of experts ([experts_block, B, I] == experts_block * i_tiles
    // tiles) per slot. Filled by the gather on {0,0} (the block's chunks) and by the block's
    // broadcast on every other core, then consumed by the down matmul for the block's experts.
    // Sized for a whole block so the down phase needs no per-expert synchronization.
    //
    // NOTE: this is the dominant L1 consumer -- act_slots * experts_block * i_tiles * act_tile_bytes
    // bytes on EVERY core (e.g. experts_block=6, I=2048 -> 6*64*1088 = 408 KB per slot) -- and it is
    // the reason experts_block exists: it, not num_active, is what has to fit, so the op can run far
    // more experts than L1 could hold at once by streaming them through in blocks. The batch rides
    // along inside each tile and costs nothing here: what bounds a block is the number of DISTINCT
    // experts in it, not the token count.
    const uint32_t act_cb_bytes = act_slots * experts_block * i_tiles * act_tile_bytes;
    // Per-core down output: the single accumulated [B, H] output tile-row slice (kOutTilesPerCore
    // tiles), double-buffered.
    const uint32_t down_out_tiles = parallel_experts ? shards_per_core * kOutTilesPerCore : kOutTilesPerCore;
    const uint32_t down_out_cb_bytes = (parallel_experts ? 1u : 2u) * down_out_tiles * out_tile_bytes;
    // Routing-weight tiles (one per expert in the current block) for the bf16 multiply that scales
    // each expert's down output before accumulation. Built per core by the reader, once per block:
    // row b of a tile holds that expert's routing weight for token b, splatted across the row, so one
    // elementwise multiply applies every token's own weight. Rows past `batch` are zero. Held per
    // block rather than for all experts because at a full 32-token disjoint batch the whole set would
    // be hundreds of KB, and the source weights stay resident in cb_bcast anyway.
    const tt::DataFormat scalar_df = tt::DataFormat::Float16_b;
    const uint32_t scalar_tile_bytes = input_tile.get_tile_size(scalar_df);
    const uint32_t rscalar_cb_bytes = experts_block * scalar_tile_bytes;
    // Per-core running accumulator for the weighted down-output sum (kOutTilesPerCore tiles),
    // double-buffered so the compute kernel can ping-pong the partial sum across experts.
    const uint32_t acc_tiles = parallel_experts ? shards_per_core * kOutTilesPerCore : kOutTilesPerCore;
    const uint32_t acc_cb_bytes = (parallel_experts ? 1u : 2u) * acc_tiles * out_tile_bytes;
    // Per-core staging for one expert's weighted down output. Serial: kOutTilesPerCore tiles,
    // double-buffered. Parallel: shards_per_core * 2 tiles for the reduce ping-pong.
    const uint32_t wtmp_cb_bytes = (parallel_experts ? 1u : 2u) * acc_tiles * out_tile_bytes;
    // Group-0 reduce scratch: the other (num_expert_groups-1) groups unicast their H-slice here.
    // Allocated on every core so L1 addresses match; unused on non-root groups.
    const uint32_t reduce_cb_bytes =
        parallel_experts ? (num_expert_groups - 1u) * acc_tiles * out_tile_bytes : std::max(32u, out_tile_bytes);

    // Each fetch pulls a whole slice in one contiguous NoC read into the CB's write pointer, so
    // the weight CB must be a whole number of slices for BOTH phases -- otherwise a read
    // straddles the wrap and runs off the end of the buffer. Size it in units of the
    // lcm(gate_up, down) slice.
    const uint32_t gate_up_slice_bytes = weight_slice_tiles * weight_tile_bytes;
    const uint32_t slice_unit_bytes = std::lcm(gate_up_slice_bytes, down_slice_bytes);
    const uint32_t weights_cb_bytes =
        (min_weights_cb_bytes + slice_unit_bytes - 1u) / slice_unit_bytes * slice_unit_bytes;

    // Down slices a receiver fetches *before* the gather/broadcast barrier, to keep DRAM busy
    // through it. Profiling showed cores idling tens of microseconds in the barrier with DRAM
    // doing nothing, while the weight reads themselves already run close to the DRAM roofline;
    // the down weights depend only on the expert ids, which every core already has, so fetching
    // them early is always legal.
    //
    // The depth is what the weight CB holds minus one. The spare slot is a liveness requirement:
    // compute cannot drain a down slice until the activation broadcast lands, and the broadcast
    // is only published to compute by the reader itself once it is past the barrier, so a reader
    // that filled the CB would block in reserve_back forever. Going deeper than this by enlarging
    // the CB measured *slower* (308 us vs 296 us at num_active=6): the extra pre-barrier read
    // traffic competes with the leader's activation multicast, which gates every core.
    // Pages each phase reserves in the shared weight CB per slice. Unblocked, the two phases are
    // separated by the single sync and each fills the CB starting from a slice-aligned pointer, so
    // each reserves exactly its own slice -- which is what packs four down slices into the CB and
    // gives the prefetch below its depth. Blocked, the phases ALTERNATE in that CB (block j's down
    // slices are followed by block j+1's gate_up slices), and a pointer left mid-CB by one phase can
    // sit closer to the end than the other phase's slice is long -- whose single contiguous NoC read
    // would then run off the end. Reserving a uniform stride for both phases keeps the pointer at a
    // multiple of the larger slice, which the CB size is a multiple of, so no read can ever straddle
    // the wrap. The unused tail of a padded slot costs nothing but prefetch depth.
    const uint32_t weight_slot_tiles = std::max(weight_slice_tiles, down_slice_tiles);
    const bool blocked = num_blocks > 1u;
    const uint32_t gate_up_reserve_tiles = blocked ? weight_slot_tiles : weight_slice_tiles;
    const uint32_t down_reserve_tiles = blocked ? weight_slot_tiles : down_slice_tiles;

    // Only the current block's down slices can be prefetched: compute consumes them in block order,
    // so going past the block would stall the reader before its next gather.
    const uint32_t down_slots = weights_cb_bytes / (down_reserve_tiles * down_tile_bytes);
    const uint32_t down_prefetch =
        std::min(parallel_experts ? shards_per_core : experts_block, down_slots > 1u ? down_slots - 1u : 0u);

    // CB reuse: the gather/broadcast sync is a hard barrier between Phase 1 (gate_up) and Phase 2
    // (down) of a block, so Phase-1 buffers are dead during Phase 2 and can host Phase-2 buffers in
    // the same L1 (across blocks the two just alternate in it, each waiting on the other's pages
    // through the CB's own flow control) -- but ONLY when both share the same producer->consumer
    // (a CB index with two different producers/consumers corrupts its page-sync counters) AND
    // the same page size (a shared-region CB's total size must be divisible by every page size).
    //   - down weights reuse cb_weights: both reader -> compute, both Bfp4_b same page, and the
    //     gate_up weight CB is already >= the down weight slice, so it is reused in place.
    //   - the down output keeps its own CB (see cb_down_out below): it is compute -> writer (so
    //     it cannot share any reader -> compute buffer) and bf16, while the only compute ->
    //     writer buffer (cb_out) is Bfp8_b, so neither constraint is satisfiable.
    TT_FATAL(
        gate_up_df == down_df && weight_tile_bytes == down_tile_bytes && weights_cb_bytes >= down_cb_bytes,
        "fused_experts: down weights reuse cb_weights, which requires a matching Bfp4_b format/page "
        "and a gate_up weight CB at least as large as the down weight slice");

    // Core sets: full grid, the two senders {0,0} (expert ids) and {1,0} (activations),
    // and the 62 receivers.
    const CoreCoord sender{0, 0};
    const CoreCoord input_sender{1, 0};
    const CoreRange all_range({0, 0}, {GRID_X - 1, GRID_Y - 1});
    const CoreRangeSet all_cores{all_range};
    const CoreRangeSet sender_set{CoreRange{sender, sender}};
    const CoreRangeSet input_sender_set{CoreRange{input_sender, input_sender}};
    // Receivers = full grid minus {0,0} and {1,0}: row 0 (x=2..7) plus rows 1..7 (all x).
    const CoreRangeSet receiver_cores{std::vector<CoreRange>{
        CoreRange{{2, 0}, {GRID_X - 1, 0}},
        CoreRange{{0, 1}, {GRID_X - 1, GRID_Y - 1}},
    }};
    // Writers on the DM processor not used by each core's reader: {1,0}'s reader is
    // NoC 1, so its writer is NoC 0; everyone else's reader is NoC 0, writer NoC 1.
    const CoreRangeSet writer_noc1_cores{std::vector<CoreRange>{
        CoreRange{sender, sender},
        CoreRange{{2, 0}, {GRID_X - 1, 0}},
        CoreRange{{0, 1}, {GRID_X - 1, GRID_Y - 1}},
    }};
    const CoreRangeSet writer_noc0_cores{CoreRange{input_sender, input_sender}};

    ProgramDescriptor desc;

    // Two broadcast-ready semaphores on ALL cores: expert ids ({0,0}) and activations ({1,0}).
    constexpr uint32_t sem_id = 0;
    constexpr uint32_t sem_input_id = 1;
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = sem_id,
        .core_type = CoreType::WORKER,
        .core_ranges = all_cores,
        .initial_value = 0,
    });
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = sem_input_id,
        .core_type = CoreType::WORKER,
        .core_ranges = all_cores,
        .initial_value = 0,
    });
    // Down-phase semaphores (all on {0,0}, but allocated on every core for a uniform id).
    // Within a block cb_act holds every expert's activation at once and is never reused, so a block
    // needs no per-expert back-pressure -- just these two, both counting monotonically across blocks
    // so no core ever has to reset them (which would race with a peer still reading the old value):
    //   sem_gather : SwiGLU cores bump it after scattering each expert's activation chunk to {0,0};
    //                {0,0} waits for the running total, num_producers chunks per expert processed so
    //                far (plus one bump per block from any non-producer core, which has no chunk to
    //                send but must still report that its cb_act slot is free).
    //   sem_bcast  : {0,0} sets it to the number of blocks it has broadcast; every core waits for
    //                its block's value.
    constexpr uint32_t sem_gather_id = 2;
    constexpr uint32_t sem_bcast_id = 3;
    constexpr uint32_t sem_reduce_id = 4;
    for (uint32_t s : {sem_gather_id, sem_bcast_id, sem_reduce_id}) {
        desc.semaphores.push_back(SemaphoreDescriptor{
            .id = s,
            .core_type = CoreType::WORKER,
            .core_ranges = all_cores,
            .initial_value = 0,
        });
    }

    // CBs are allocated identically on all cores so the broadcast CBs land at the same
    // L1 address everywhere (required for the multicast writes to be valid).
    constexpr uint32_t cb_routing = CBIndex::c_0;
    desc.cbs.push_back(CBDescriptor{
        .total_size = routing_cb_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cb_routing,
            .data_format = routing_df,
            .page_size = routing_cb_bytes,
        }}},
    });

    constexpr uint32_t cb_bcast = CBIndex::c_1;
    desc.cbs.push_back(CBDescriptor{
        .total_size = bcast_cb_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cb_bcast,
            .data_format = tt::DataFormat::UInt32,
            .page_size = bcast_cb_bytes,
        }}},
    });

    // Activation tiles (page = one tile) so the matmul can index them tile-by-tile.
    // The token-row-shaped tile (input_tile) is attached so JIT get_tile_size(cb_input) and
    // unpack strides match the real tile geometry rather than the default 32x32.
    constexpr uint32_t cb_input = CBIndex::c_2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = input_cb_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cb_input,
            .data_format = input_df,
            .page_size = input_page_size,
            .tile = input_tile_desc,
        }}},
    });

    // Per-core gate_up weight slice ([K, 128] = k_tiles x 4 tiles: gate 0,1 | up 2,3),
    // double-buffered.
    constexpr uint32_t cb_weights = CBIndex::c_3;
    desc.cbs.push_back(CBDescriptor{
        .total_size = weights_cb_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cb_weights,
            .data_format = gate_up_df,
            .page_size = weight_tile_bytes,
        }}},
    });

    // Per-core SwiGLU output (kOutTilesPerCore tiles per expert), double-buffered. Stored as
    // Bfp8_b (act_df) so it can be scattered byte-for-byte into the bf8 cb_act.
    constexpr uint32_t cb_out = CBIndex::c_4;
    desc.cbs.push_back(CBDescriptor{
        .total_size = out_cb_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cb_out,
            .data_format = act_df,
            .page_size = act_tile_bytes,
            .tile = input_tile_desc,
        }}},
    });

    // Per-core matmul staging buffer (fp32): the compute kernel packs the gate/up matmul
    // results here (gate 0,1 | up 2,3), then reloads them for the SwiGLU SFPU pass. Its DEST
    // tile inherits face_r_dim from cb_input (in0), so the pack writes an input-tile-shaped
    // fp32 tile -- attach the input tile descriptor so unpack strides for the SwiGLU SFPU
    // reload match.
    constexpr uint32_t cb_mm = CBIndex::c_5;
    desc.cbs.push_back(CBDescriptor{
        .total_size = mm_cb_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cb_mm,
            .data_format = tt::DataFormat::Float32,
            .page_size = mm_tile_bytes,
            .tile = input_tile_desc,
        }}},
    });

    // Gathered activation, act_slots slots of one expert block each (experts_block * i_tiles tiles).
    // Allocated identically on all cores so the gather scatter / broadcast land at the same L1
    // address, and every core reserves/pushes exactly one slot per block so the slots stay in
    // lockstep chip-wide.
    constexpr uint32_t cb_act = CBIndex::c_6;
    desc.cbs.push_back(CBDescriptor{
        .total_size = act_cb_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cb_act,
            .data_format = act_df,
            .page_size = act_tile_bytes,
            .tile = input_tile_desc,
        }}},
    });

    // Reuse (Proposal 1): the per-core down weight slice ([I, 64] = i_tiles x 2 tiles) shares
    // the gate_up weight CB. gate_up (Phase 1) is fully consumed before the sync and down is
    // fetched only afterwards (Phase 2), so the two never coexist; both are Bfp4_b with the same
    // page and cb_weights' double-buffered region is larger than the down slice needs.
    constexpr uint32_t cb_down_weights = cb_weights;

    // Per-core down output (kOutTilesPerCore tiles per expert), double-buffered. NOTE: this is
    // NOT merged into another CB. The down output is produced by compute and consumed by the
    // writer (compute -> writer), so it cannot safely alias any reader -> compute buffer
    // (cb_input/cb_act/cb_weights) -- a CB index can only have one producer or its page-sync
    // counters corrupt. The only compatible compute -> writer buffer, cb_out, is Bfp8_b
    // (1088 B page) while this output is bf16 (2048 B page), and a shared-region CB's total
    // size must be divisible by every page size (LCM(1088, 2048) = 34816 B), which would use
    // MORE L1 than keeping them separate. So the down output keeps its own small CB (c_8).
    constexpr uint32_t cb_down_out = CBIndex::c_8;
    desc.cbs.push_back(CBDescriptor{
        .total_size = down_out_cb_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cb_down_out,
            .data_format = out_df,
            .page_size = out_tile_bytes,
            .tile = input_tile_desc,
        }}},
    });

    // Routing-weight tiles (one per expert in a block, bf16) consumed by the down-output multiply.
    // The reader refills row b of each tile with that expert's routing weight for token b per block.
    // The tile shape matches the input's so mul_tiles(cb_mm, cb_rscalar) has agreeing face_r_dim.
    constexpr uint32_t cb_rscalar = CBIndex::c_7;
    desc.cbs.push_back(CBDescriptor{
        .total_size = rscalar_cb_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cb_rscalar,
            .data_format = scalar_df,
            .page_size = scalar_tile_bytes,
            .tile = input_tile_desc,
        }}},
    });

    // Running accumulator for the weighted down-output sum (compute-internal ping-pong).
    constexpr uint32_t cb_acc = CBIndex::c_9;
    desc.cbs.push_back(CBDescriptor{
        .total_size = acc_cb_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cb_acc,
            .data_format = out_df,
            .page_size = out_tile_bytes,
            .tile = input_tile_desc,
        }}},
    });

    // Staging for one expert's weighted down output (compute-internal).
    constexpr uint32_t cb_wtmp = CBIndex::c_10;
    desc.cbs.push_back(CBDescriptor{
        .total_size = wtmp_cb_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cb_wtmp,
            .data_format = out_df,
            .page_size = out_tile_bytes,
            .tile = input_tile_desc,
        }}},
    });

    constexpr uint32_t cb_reduce = CBIndex::c_11;
    desc.cbs.push_back(CBDescriptor{
        .total_size = reduce_cb_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cb_reduce,
            .data_format = out_df,
            .page_size = out_tile_bytes,
            .tile = input_tile_desc,
        }}},
    });

    // Multicast rectangle (NoC coords) covering the whole grid. Non-loopback
    // multicast excludes the sender, so num_dests = total cores - 1.
    const auto corner_a = device->worker_core_from_logical_core(CoreCoord{0, 0});
    const auto corner_b = device->worker_core_from_logical_core(CoreCoord{GRID_X - 1, GRID_Y - 1});
    const uint32_t mcast_start_x = std::min<uint32_t>(corner_a.x, corner_b.x);
    const uint32_t mcast_start_y = std::min<uint32_t>(corner_a.y, corner_b.y);
    const uint32_t mcast_end_x = std::max<uint32_t>(corner_a.x, corner_b.x);
    const uint32_t mcast_end_y = std::max<uint32_t>(corner_a.y, corner_b.y);
    const uint32_t num_dests = GRID_X * GRID_Y - 1;

    auto mcast_rect = [&](uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1) {
        const auto a = device->worker_core_from_logical_core(CoreCoord{x0, y0});
        const auto b = device->worker_core_from_logical_core(CoreCoord{x1, y1});
        return std::array<uint32_t, 5>{
            std::min<uint32_t>(a.x, b.x),
            std::min<uint32_t>(a.y, b.y),
            std::max<uint32_t>(a.x, b.x),
            std::max<uint32_t>(a.y, b.y),
            (x1 - x0 + 1) * (y1 - y0 + 1) - 1,
        };
    };
    auto group_rect = [&](uint32_t g) {
        if (!parallel_experts) {
            return std::array<uint32_t, 5>{mcast_start_x, mcast_start_y, mcast_end_x, mcast_end_y, num_dests};
        }
        const uint32_t x0 = 2u * g;
        return mcast_rect(x0, 0, x0 + 1, GRID_Y - 1);
    };
    auto group_leader_noc = [&](uint32_t g) {
        const uint32_t x = parallel_experts ? 2u * g : 0u;
        const auto c = device->worker_core_from_logical_core(CoreCoord{x, 0});
        return std::pair<uint32_t, uint32_t>{static_cast<uint32_t>(c.x), static_cast<uint32_t>(c.y)};
    };
    const auto group0_rect = group_rect(0);

    // Every per-core index is derived from the core's flat grid index idx: its gate_up and down
    // DRAM shard ids are both idx, its SwiGLU output tiles are
    // [idx*swiglu_tiles_per_core, +swiglu_tiles_per_core) and its H output tiles are
    // [idx*kOutTilesPerCore, +kOutTilesPerCore). The kernels take idx and reconstruct the rest
    // from the (compile-time) per-core tile counts.
    //
    // The index is column-major (x*GRID_Y + y) rather than the obvious row-major. The weights are
    // round-robined over the 8 DRAM banks by shard id, so idx % 8 picks the bank a core reads
    // from: column-major puts the 8 cores that share a bank in a grid *row* instead of a grid
    // *column*, which spreads their return traffic over different NoC paths. Worth ~10 us.
    auto core_index_for = [](const CoreCoord& c) -> uint32_t { return c.x * GRID_Y + c.y; };
    // Base address of every expert's gate_up weight, in expert-id order. All
    // experts are passed so the fetch can index by routing-selected hit id.
    std::vector<uint32_t> gate_up_addrs;
    gate_up_addrs.reserve(num_weights);
    for (const auto& w : tensor_args.gate_up_weights) {
        gate_up_addrs.push_back(static_cast<uint32_t>(w.buffer()->address()));
    }
    // down weight base addresses, in expert-id order (indexed by routing-selected hit id).
    std::vector<uint32_t> down_addrs;
    down_addrs.reserve(num_weights);
    for (const auto& w : tensor_args.down_weights) {
        down_addrs.push_back(static_cast<uint32_t>(w.buffer()->address()));
    }
    // The weight addresses are compile-time args: appended (gate_up then down, expert-id order)
    // to each reader kernel's compile_time_args right after its TensorAccessorArgs. The kernels
    // index the resident kernel_compile_time_args array by the runtime-selected expert id.
    auto append_addrs_ct = [&](std::vector<uint32_t>& ct_args) {
        for (uint32_t a : gate_up_addrs) {
            ct_args.push_back(a);
        }
        for (uint32_t a : down_addrs) {
            ct_args.push_back(a);
        }
    };

    // Core {0,0} NoC coordinates (virtual; usable on either NoC) — the gather scatter target
    // and the home of the down-phase semaphores.
    const uint32_t leader_noc_x = corner_a.x;
    const uint32_t leader_noc_y = corner_a.y;

    // ---- Expert-id sender kernel on {0,0} (NoC 0). ----
    std::vector<uint32_t> sender_ct_args = {
        num_weights,
        num_active,
        sentinel,
        cb_routing,
        cb_bcast,
        routing_page_bytes,
        bcast_page_bytes,
        sem_id,
        cb_weights,
        k_tiles,
        i_tiles,
        weight_tile_bytes,
        sem_input_id,
        cb_input,
        cb_down_weights,
        cb_act,
        down_slice_tiles,
        down_tile_bytes,
        act_tile_bytes,
        num_producers,
        sem_gather_id,
        sem_bcast_id,
        cb_rscalar,
        down_prefetch,
        batch,
        experts_block,
        gate_up_reserve_tiles,
        down_reserve_tiles,
        // ---- routing ----
        top_k,
        index_is_bf16 ? 1u : 0u,
        std::bit_cast<uint32_t>(operation_attributes.routed_scaling_factor),
        std::bit_cast<uint32_t>(operation_attributes.routing_eps),
        score_pages,
        score_page_bytes,
        score_page_stride,
        score_l1_offset,
        scratch_l1_offset,
        bitmap_bytes,
        rank_bytes,
        // Routing-scalar tile geometry (bf16, width 32, height == input tile height).
        input_tile_h,
        face_r_dim,
        num_face_rows,
        scalar_tile_bytes,
        cores_per_expert,
        shards_per_core,
        i_shards_per_core,
        num_expert_groups,
        sem_reduce_id,
        cb_reduce,
    };
    TensorAccessorArgs(*routing_buffer).append_to(sender_ct_args);
    TensorAccessorArgs(*score_buffer).append_to(sender_ct_args);
    TensorAccessorArgs(*gate_up0_buffer).append_to(sender_ct_args);
    TensorAccessorArgs(*down0_buffer).append_to(sender_ct_args);
    append_addrs_ct(sender_ct_args);

    KernelDescriptor sender_desc;
    sender_desc.kernel_source = std::string(kKernelDir) + "/dataflow/compute_expert_ids.cpp";
    sender_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    sender_desc.core_ranges = sender_set;
    sender_desc.compile_time_args = sender_ct_args;
    sender_desc.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::NOC_0,
    };
    // Pass the routing buffers as BufferBindings (not raw addresses) so the framework
    // patches the addresses on program-cache hits instead of rebuilding the descriptor.
    sender_desc.emplace_runtime_args(
        sender,
        {routing_buffer,
         mcast_start_x,
         mcast_start_y,
         mcast_end_x,
         mcast_end_y,
         num_dests,
         core_index_for(sender),
         score_buffer,
         group0_rect[0],
         group0_rect[1],
         group0_rect[2],
         group0_rect[3],
         group0_rect[4]});
    desc.kernels.push_back(std::move(sender_desc));

    // ---- Input-broadcaster kernel on {1,0} (NoC 1). ----
    std::vector<uint32_t> input_ct_args = {
        cb_input,
        input_page_size,
        input_num_pages,
        sem_input_id,
        sem_id,
        num_active,
        cb_weights,
        k_tiles,
        i_tiles,
        weight_tile_bytes,
        cb_bcast,
        cb_down_weights,
        cb_act,
        down_slice_tiles,
        down_tile_bytes,
        act_tile_bytes,
        num_producers,
        sem_gather_id,
        sem_bcast_id,
        num_weights,
        cb_rscalar,
        down_prefetch,
        batch,
        experts_block,
        gate_up_reserve_tiles,
        down_reserve_tiles,
        // Routing-scalar tile geometry (bf16, width 32, height == input tile height).
        input_tile_h,
        face_r_dim,
        num_face_rows,
        scalar_tile_bytes,
        cores_per_expert,
        shards_per_core,
        i_shards_per_core,
        num_expert_groups,
        sem_reduce_id,
        cb_reduce,
    };
    TensorAccessorArgs(*input_buffer).append_to(input_ct_args);
    TensorAccessorArgs(*gate_up0_buffer).append_to(input_ct_args);
    TensorAccessorArgs(*down0_buffer).append_to(input_ct_args);
    append_addrs_ct(input_ct_args);

    KernelDescriptor input_sender_desc;
    input_sender_desc.kernel_source = std::string(kKernelDir) + "/dataflow/broadcast_input.cpp";
    input_sender_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    input_sender_desc.core_ranges = input_sender_set;
    input_sender_desc.compile_time_args = input_ct_args;
    input_sender_desc.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = NOC::NOC_1,
    };
    // NoC 1 multicasts traverse from high to low coordinates, so swap start/end.
    // input_buffer is a BufferBinding so the framework patches its address on cache hits.
    input_sender_desc.emplace_runtime_args(
        input_sender,
        {input_buffer,
         mcast_end_x,
         mcast_end_y,
         mcast_start_x,
         mcast_start_y,
         num_dests,
         core_index_for(input_sender),
         leader_noc_x,
         leader_noc_y});
    desc.kernels.push_back(std::move(input_sender_desc));

    // ---- Receiver reader kernel on the other 62 cores (NoC 0). ----
    std::vector<uint32_t> receiver_ct_args = {
        sem_id,
        sem_input_id,
        num_active,
        cb_input,
        cb_weights,
        k_tiles,
        i_tiles,
        weight_tile_bytes,
        cb_bcast,
        cb_down_weights,
        cb_act,
        down_slice_tiles,
        down_tile_bytes,
        act_tile_bytes,
        num_producers,
        sem_gather_id,
        sem_bcast_id,
        num_weights,
        cb_rscalar,
        down_prefetch,
        batch,
        experts_block,
        gate_up_reserve_tiles,
        down_reserve_tiles,
        // Routing-scalar tile geometry (bf16, width 32, height == input tile height).
        input_tile_h,
        face_r_dim,
        num_face_rows,
        scalar_tile_bytes,
        cores_per_expert,
        shards_per_core,
        i_shards_per_core,
        num_expert_groups,
        sem_reduce_id,
        cb_reduce,
    };
    TensorAccessorArgs(*gate_up0_buffer).append_to(receiver_ct_args);
    TensorAccessorArgs(*down0_buffer).append_to(receiver_ct_args);
    append_addrs_ct(receiver_ct_args);

    KernelDescriptor receiver_desc;
    receiver_desc.kernel_source = std::string(kKernelDir) + "/dataflow/wait_expert_ids.cpp";
    receiver_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    receiver_desc.core_ranges = receiver_cores;
    receiver_desc.compile_time_args = receiver_ct_args;
    receiver_desc.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::NOC_0,
    };
    for (const auto& cr : receiver_cores.ranges()) {
        for (const auto& core : cr) {
            const uint32_t idx = core_index_for(core);
            const uint32_t g = parallel_experts ? idx / cores_per_expert : 0u;
            const uint32_t local = parallel_experts ? idx % cores_per_expert : idx;
            const auto g_rect = group_rect(g);
            const auto g_leader = group_leader_noc(g);
            receiver_desc.runtime_args.emplace_back(
                core,
                KernelDescriptor::CoreRuntimeArgs{
                    idx,
                    g_leader.first,
                    g_leader.second,
                    g_rect[0],
                    g_rect[1],
                    g_rect[2],
                    g_rect[3],
                    g_rect[4],
                    (parallel_experts && local == 0) ? 1u : 0u});
        }
    }
    desc.kernels.push_back(std::move(receiver_desc));

    // ---- Compute (gate_up matmul) kernel on all 64 cores. ----
    std::vector<uint32_t> compute_ct_args = {
        num_active,
        k_tiles,
        i_tiles,
        cb_input,
        cb_weights,
        cb_mm,
        cb_out,
        limit_bits,
        cb_act,
        cb_down_weights,
        cb_down_out,
        cb_rscalar,
        cb_acc,
        cb_wtmp,
        num_producers,
        experts_block,
        gate_up_reserve_tiles,
        down_reserve_tiles,
        cores_per_expert,
        shards_per_core,
        i_shards_per_core,
        num_expert_groups,
        cb_reduce,
    };
    KernelDescriptor compute_desc;
    compute_desc.kernel_source = std::string(kKernelDir) + "/compute/matmul_gate_up.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_cores;
    compute_desc.compile_time_args = compute_ct_args;
    compute_desc.config = ComputeConfigDescriptor{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en = true,
    };
    for (uint32_t y = 0; y < GRID_Y; ++y) {
        for (uint32_t x = 0; x < GRID_X; ++x) {
            const CoreCoord core{x, y};
            compute_desc.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs{core_index_for(core)});
        }
    }
    desc.kernels.push_back(std::move(compute_desc));

    // ---- Writer kernel on all 64 cores (two processor groups). ----
    std::vector<uint32_t> writer_ct_args = {
        num_active,
        i_tiles,  // I/32: activation stride between experts in the gathered block
        cb_out,
        cb_down_out,
        cb_act,
        act_tile_bytes,
        out_tile_bytes,
        sem_gather_id,
        num_producers,  // SwiGLU-core guard for the gather scatter
        experts_block,
        cores_per_expert,
        shards_per_core,
        i_shards_per_core,
        num_expert_groups,
        cb_reduce,
        sem_reduce_id,
    };
    TensorAccessorArgs(*out_buffer).append_to(writer_ct_args);

    auto make_writer = [&](const CoreRangeSet& cores, DataMovementProcessor proc, NOC noc) {
        KernelDescriptor writer_desc;
        writer_desc.kernel_source = std::string(kKernelDir) + "/dataflow/write_gate_up.cpp";
        writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        writer_desc.core_ranges = cores;
        writer_desc.compile_time_args = writer_ct_args;
        writer_desc.config = DataMovementConfigDescriptor{.processor = proc, .noc = noc};
        for (const auto& cr : cores.ranges()) {
            for (const auto& core : cr) {
                const uint32_t idx = core_index_for(core);
                const uint32_t g = parallel_experts ? idx / cores_per_expert : 0u;
                const uint32_t local = parallel_experts ? idx % cores_per_expert : idx;
                const auto g_leader = group_leader_noc(g);
                // Group-0 counterpart: same local index in columns 0-1 (or {0,0} on the 64-core path).
                const CoreCoord reduce_logical{parallel_experts ? (core.x % 2) : 0u, parallel_experts ? core.y : 0u};
                const auto reduce_noc = device->worker_core_from_logical_core(reduce_logical);
                (void)local;
                writer_desc.emplace_runtime_args(
                    core,
                    {out_buffer,
                     idx,
                     g_leader.first,
                     g_leader.second,
                     static_cast<uint32_t>(reduce_noc.x),
                     static_cast<uint32_t>(reduce_noc.y)});
            }
        }
        desc.kernels.push_back(std::move(writer_desc));
    };
    make_writer(writer_noc1_cores, DataMovementProcessor::RISCV_1, NOC::NOC_1);
    make_writer(writer_noc0_cores, DataMovementProcessor::RISCV_0, NOC::NOC_0);

    return desc;
}

}  // namespace ttnn::operations::experimental::deepseek::moe::fused_experts

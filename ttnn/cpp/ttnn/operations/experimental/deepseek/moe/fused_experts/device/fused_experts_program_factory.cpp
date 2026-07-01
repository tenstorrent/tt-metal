// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_experts_device_operation.hpp"

#include <algorithm>
#include <bit>
#include <string>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::experimental::deepseek::moe::fused_experts {

using namespace tt;
using namespace tt::tt_metal;

namespace {
constexpr std::string_view kKernelDir =
    "ttnn/cpp/ttnn/operations/experimental/deepseek/moe/fused_experts/device/kernels";

// Compute grid for the broadcast / matmul: 8x8 = 64 cores.
constexpr uint32_t GRID_X = 8;
constexpr uint32_t GRID_Y = 8;

// Each active core owns kTilesPerCore SwiGLU output tiles (64 columns). Its gate_up
// shard is 2*kTilesPerCore tiles wide ([gate_64 | up_64]) and its first output tile
// (== col_start_tile) is compute_index * kTilesPerCore.
constexpr uint32_t kTilesPerCore = 2;

uint32_t align_up_32(uint32_t x) { return (x + 31u) & ~31u; }
}  // namespace

// Pipeline (two phases over all selected experts, with a single synchronization between):
//   - {0,0} (NoC 0) reads routing weights, computes/broadcasts the selected ("hit")
//     expert ids (ascending), and acts as the activation-gather leader.
//   - {1,0} (NoC 1) reads the decode activation row and broadcasts it to every
//     core's L1 (cb_input).
//   - PHASE 1 -- gate_up + SwiGLU for ALL experts: each of the I/64 SwiGLU cores fetches
//     its [K, 128] gate_up shard per expert (one NoC read -- a per-core [gate_64 | up_64]
//     block) and produces its 2-tile slice of each expert's activation act[1, I]. Each
//     core's writer scatters expert e's 2 act tiles to {0,0}'s cb_act at tile offset
//     (e*i_tiles + col_start).
//   - SINGLE SYNC -- gather + broadcast: once {0,0} has every expert's chunk from every
//     SwiGLU core (num_producers * num_active in total, via sem_gather), it multicasts the
//     whole [num_active, I] activation block back to every core in one shot (sem_bcast).
//     Because cb_act holds all experts at once and is never reused, no per-expert
//     back-pressure is needed.
//   - PHASE 2 -- DOWN matmul for ALL experts: each of the 64 cores fetches its [I, H/64]
//     down shard per expert (one NoC read) and multiplies it by that expert's activation to
//     produce its 2-tile slice of the output row[1, H]. The compute kernel scales each
//     expert's slice by its routing weight (SCALAR broadcast) and accumulates across all
//     experts, so the writer writes a single [1, 1, H] DRAM output row (the weighted sum).
ProgramDescriptor FusedExpertsDeviceOperation::MultiCore::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& routing_weights = tensor_args.routing_weights;
    const auto& input_tensor = tensor_args.input_tensor;
    auto& output_tensor = tensor_return_value;

    auto* routing_buffer = routing_weights.buffer();
    auto* input_buffer = input_tensor.buffer();
    auto* out_buffer = output_tensor.buffer();
    auto* device = routing_weights.device();

    const auto grid = device->compute_with_storage_grid_size();
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
    const uint32_t num_weights = static_cast<uint32_t>(tensor_args.gate_up_weights.size());
    const uint32_t num_active = operation_attributes.num_experts;
    const uint32_t sentinel = num_weights;  // "no expert" marker for unused id slots

    // gate_up weights are [K=H, N=2I] per expert (TILE layout), reshaped+permuted on the
    // host into per-core [gate_64 | up_64] blocks. Each core fetches its [K, 128]
    // (k_tiles x 4-tile) column slice -- one DRAM shard (gate cols 0,1 | up cols 2,3) --
    // for every selected expert. All experts share the same layout, so one
    // TensorAccessorArgs (from weight 0) is reused.
    const auto& gate_up0 = tensor_args.gate_up_weights.front();
    auto* gate_up0_buffer = gate_up0.buffer();
    constexpr uint32_t TILE_DIM = 32;
    const uint32_t k_tiles = static_cast<uint32_t>(gate_up0.logical_shape()[-2]) / TILE_DIM;
    const uint32_t n_tiles = static_cast<uint32_t>(gate_up0.logical_shape()[-1]) / TILE_DIM;  // 2I / 32
    const uint32_t i_tiles = n_tiles / 2u;  // SwiGLU output tile cols (I / 32)
    const uint32_t weight_tile_bytes = static_cast<uint32_t>(gate_up0_buffer->page_size());
    // Each core's weight slice is its 2 gate tiles + 2 paired up tiles per k-row.
    const uint32_t weight_slice_tiles = k_tiles * (2u * kTilesPerCore);
    // Double-buffer the weight slice so the reader can prefetch the next expert.
    const uint32_t weights_cb_bytes = weight_slice_tiles * weight_tile_bytes;

    // down weights are [I, H] per expert (TILE layout), DRAM ND-sharded into [I, H/64]
    // column blocks (one per core). Core idx owns the H output cols [idx*64, idx*64+64) ->
    // its 2 output tiles, and needs the full I (== gate_up output) contraction dim, so its
    // shard is [i_tiles, 2] tiles. All experts share one layout, so weight 0's accessor is
    // reused (the fetch indexes by shard id == this core's index).
    const auto& down0 = tensor_args.down_weights.front();
    auto* down0_buffer = down0.buffer();
    const uint32_t down_slice_tiles = i_tiles * kTilesPerCore;  // [I, 64] = i_tiles * 2 tiles
    const uint32_t down_tile_bytes = static_cast<uint32_t>(down0_buffer->page_size());
    const uint32_t down_cb_bytes = 2u * down_slice_tiles * down_tile_bytes;  // double-buffered
    const tt::DataFormat down_df = datatype_to_dataformat_converter(down0.dtype());

    // Number of SwiGLU cores (each produces one [1, 64] activation chunk per expert).
    const uint32_t num_producers = i_tiles / kTilesPerCore;  // I / 64

    const tt::DataFormat gate_up_df = datatype_to_dataformat_converter(gate_up0.dtype());
    const tt::DataFormat routing_df = datatype_to_dataformat_converter(routing_weights.dtype());
    const tt::DataFormat out_df = datatype_to_dataformat_converter(output_tensor.dtype());
    const tt::DataFormat input_df = datatype_to_dataformat_converter(input_tensor.dtype());

    constexpr uint32_t routing_elem_bytes = 2;  // bfloat16
    constexpr uint32_t out_elem_bytes = 4;      // uint32 expert ids (broadcast scratch)
    // Routing row and the id broadcast span all provided experts.
    const uint32_t routing_page_bytes = num_weights * routing_elem_bytes;
    // cb_bcast carries the compacted expert ids (num_weights uint32, ascending hit ids padded
    // with the sentinel) followed by the active experts' routing-weight scalars (num_active
    // fp32 bit patterns, in the same hit order), broadcast to every core in one multicast.
    const uint32_t bcast_page_bytes = (num_weights + num_active) * out_elem_bytes;

    // Activation is TILE layout [1,1,1,H] -> Kt == k_tiles tiles (one tile-row).
    const uint32_t input_page_size = static_cast<uint32_t>(input_buffer->page_size());
    const uint32_t input_num_pages = static_cast<uint32_t>(input_buffer->num_pages());

    // Output is TILE [1, 1, H] bf16 (the routing-weighted sum of every active expert's down
    // matmul): each core writes its 2 output tiles (its 64-column H slice) of the single row.
    const uint32_t out_tile_bytes = static_cast<uint32_t>(out_buffer->page_size());

    // The gathered activation is stored as Bfp8_b (not bf16) to keep the resident
    // [num_active, I] block -- the dominant L1 consumer -- within the L1 budget. The SwiGLU
    // output (cb_out) is packed in the same format so the writer can scatter it byte-for-byte
    // into the leader's cb_act, and the down matmul reads it as its bf8 in0 (paired with the
    // bf4 down weights). The down output stays bf16 to match the DRAM output tensor.
    const tt::DataFormat act_df = tt::DataFormat::Bfp8_b;
    const uint32_t act_tile_bytes = tt::tile_size(act_df);

    // SwiGLU clamp limit, passed to the compute kernel as a bit-cast float (the kernel
    // derives -limit internally).
    const uint32_t limit_bits = std::bit_cast<uint32_t>(operation_attributes.swiglu_limit);

    const uint32_t routing_cb_bytes = std::max<uint32_t>(align_up_32(routing_page_bytes), 32u);
    const uint32_t bcast_cb_bytes = std::max<uint32_t>(align_up_32(bcast_page_bytes), 32u);
    const uint32_t input_cb_bytes = input_num_pages * input_page_size;
    // Double-buffer the matmul output so compute can run ahead of the writer. Each core
    // produces kOutTilesPerCore SwiGLU output tiles (its 64-column I slice) per expert.
    // cb_out holds the bf8 SwiGLU activation (== act_df) so the writer can scatter it
    // directly into cb_act.
    constexpr uint32_t kOutTilesPerCore = 2;
    const uint32_t out_cb_bytes = 2u * kOutTilesPerCore * act_tile_bytes;
    // Matmul staging buffer (fp32 for full precision before the SwiGLU SFPU pass):
    // 2*kOutTilesPerCore tiles per expert (gate 0,1 | up 2,3), single-buffered.
    const uint32_t mm_tile_bytes = TILE_DIM * TILE_DIM * 4u;
    const uint32_t mm_cb_bytes = 2u * kOutTilesPerCore * mm_tile_bytes;

    // Gathered activation: the WHOLE [num_active, I] block (num_active * i_tiles tiles),
    // single-buffered. Filled by the gather on {0,0} (all experts' chunks) and by the single
    // broadcast on every other core, then consumed by the down matmul for every expert. Sized
    // for all experts at once so the down phase needs no per-expert synchronization.
    // NOTE: this is the dominant L1 consumer -- num_active * i_tiles * act_tile_bytes bytes on
    // EVERY core (e.g. num_active=6, I=2048 -> 6*64*2KB = 768 KB) -- so large num_active / I
    // can exceed the L1 budget.
    const uint32_t act_cb_bytes = num_active * i_tiles * act_tile_bytes;
    // Per-core down output: the single accumulated [1, H] output row slice (kOutTilesPerCore
    // tiles), double-buffered.
    const uint32_t down_out_cb_bytes = 2u * kOutTilesPerCore * out_tile_bytes;
    // Routing-weight scalar tiles (one per active expert) for the bf16 SCALAR broadcast that
    // scales each expert's down output before accumulation. Built per core by the reader.
    const tt::DataFormat scalar_df = tt::DataFormat::Float16_b;
    const uint32_t scalar_tile_bytes = tt::tile_size(scalar_df);
    const uint32_t rscalar_cb_bytes = num_active * scalar_tile_bytes;
    // Per-core running accumulator for the weighted down-output sum (kOutTilesPerCore tiles),
    // double-buffered so the compute kernel can ping-pong the partial sum across experts.
    const uint32_t acc_cb_bytes = 2u * kOutTilesPerCore * out_tile_bytes;
    // Per-core staging for one expert's weighted down output (kOutTilesPerCore tiles),
    // double-buffered. Holds routing_w[e] * down_e between the SCALAR-broadcast multiply and the
    // add into the accumulator (kept separate so each compute block uses a single op type).
    const uint32_t wtmp_cb_bytes = 2u * kOutTilesPerCore * out_tile_bytes;

    // CB reuse: the single gather/broadcast sync is a hard barrier between Phase 1 (gate_up) and
    // Phase 2 (down), so Phase-1-only buffers are dead during Phase 2 and can host Phase-2-only
    // buffers in the same L1 -- but ONLY when both share the same producer->consumer RISC pair
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
    // The two-phase structure (all gate_up, single gather+broadcast, all down) means cb_act
    // holds every expert's activation at once and is never reused, so only two semaphores are
    // needed -- no per-expert back-pressure:
    //   sem_gather : SwiGLU cores bump it after scattering each expert's activation chunk to
    //                {0,0}; {0,0} waits for all num_producers * num_active chunks (single sync).
    //   sem_bcast  : {0,0} sets it once after broadcasting the whole [num_active, I] block.
    constexpr uint32_t sem_gather_id = 2;
    constexpr uint32_t sem_bcast_id = 3;
    for (uint32_t s : {sem_gather_id, sem_bcast_id}) {
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
    constexpr uint32_t cb_input = CBIndex::c_2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = input_cb_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cb_input,
            .data_format = input_df,
            .page_size = input_page_size,
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
        }}},
    });

    // Per-core matmul staging buffer (fp32): the compute kernel packs the gate/up matmul
    // results here (gate 0,1 | up 2,3), then reloads them for the SwiGLU SFPU pass.
    constexpr uint32_t cb_mm = CBIndex::c_5;
    desc.cbs.push_back(CBDescriptor{
        .total_size = mm_cb_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cb_mm,
            .data_format = tt::DataFormat::Float32,
            .page_size = mm_tile_bytes,
        }}},
    });

    // Gathered activation (full [num_active, I] block, num_active * i_tiles tiles),
    // single-buffered. Allocated identically on all cores so the gather scatter / broadcast
    // land at the same L1 address.
    constexpr uint32_t cb_act = CBIndex::c_6;
    desc.cbs.push_back(CBDescriptor{
        .total_size = act_cb_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cb_act,
            .data_format = act_df,
            .page_size = act_tile_bytes,
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
        }}},
    });

    // Routing-weight scalar tiles (one per active expert, bf16) consumed by the down-output
    // SCALAR broadcast multiply. The reader splats each expert's scalar into [0,0] of its tile.
    constexpr uint32_t cb_rscalar = CBIndex::c_7;
    desc.cbs.push_back(CBDescriptor{
        .total_size = rscalar_cb_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cb_rscalar,
            .data_format = scalar_df,
            .page_size = scalar_tile_bytes,
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

    // Each core's first SwiGLU output tile is idx * kTilesPerCore (it owns the 2 I-dim
    // output tiles [idx*2, idx*2 + 1]), idx = y*GRID_X + x. The fetch derives the DRAM
    // shard id from this as col_start_tile / kTilesPerCore == idx.
    auto col_start_tile_for = [](const CoreCoord& c) -> uint32_t { return (c.y * GRID_X + c.x) * kTilesPerCore; };
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
        num_weights,      num_active,    sentinel,        cb_routing,   cb_bcast,         routing_page_bytes,
        bcast_page_bytes, sem_id,        cb_weights,      k_tiles,      i_tiles,          weight_tile_bytes,
        sem_input_id,     cb_input,      cb_down_weights, cb_act,       down_slice_tiles, down_tile_bytes,
        act_tile_bytes,   num_producers, sem_gather_id,   sem_bcast_id, cb_rscalar,
    };
    TensorAccessorArgs(*routing_buffer).append_to(sender_ct_args);
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
    {
        KernelDescriptor::CoreRuntimeArgs args{
            routing_buffer->address(),
            mcast_start_x,
            mcast_start_y,
            mcast_end_x,
            mcast_end_y,
            num_dests,
            col_start_tile_for(sender),
        };
        sender_desc.runtime_args.emplace_back(sender, std::move(args));
    }
    desc.kernels.push_back(std::move(sender_desc));

    // ---- Input-broadcaster kernel on {1,0} (NoC 1). ----
    std::vector<uint32_t> input_ct_args = {
        cb_input,     input_page_size,  input_num_pages, sem_input_id,      sem_id,        num_active,
        cb_weights,   k_tiles,          i_tiles,         weight_tile_bytes, cb_bcast,      cb_down_weights,
        cb_act,       down_slice_tiles, down_tile_bytes, act_tile_bytes,    num_producers, sem_gather_id,
        sem_bcast_id, num_weights,      cb_rscalar,
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
    {
        KernelDescriptor::CoreRuntimeArgs args{
            input_buffer->address(),
            mcast_end_x,
            mcast_end_y,
            mcast_start_x,
            mcast_start_y,
            num_dests,
            col_start_tile_for(input_sender),
        };
        input_sender_desc.runtime_args.emplace_back(input_sender, std::move(args));
    }
    desc.kernels.push_back(std::move(input_sender_desc));

    // ---- Receiver reader kernel on the other 62 cores (NoC 0). ----
    std::vector<uint32_t> receiver_ct_args = {
        sem_id,        sem_input_id,     num_active,        cb_input,       cb_weights,
        k_tiles,       i_tiles,          weight_tile_bytes, cb_bcast,       cb_down_weights,
        cb_act,        down_slice_tiles, down_tile_bytes,   act_tile_bytes, num_producers,
        sem_gather_id, sem_bcast_id,     num_weights,       cb_rscalar,
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
            receiver_desc.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs{col_start_tile_for(core)});
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
            compute_desc.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs{col_start_tile_for(core)});
        }
    }
    desc.kernels.push_back(std::move(compute_desc));

    // ---- Writer kernel on all 64 cores (two processor groups). ----
    std::vector<uint32_t> writer_ct_args = {
        num_active,
        i_tiles,  // I/32: SwiGLU-core guard for the gather scatter
        cb_out,
        cb_down_out,
        cb_act,
        act_tile_bytes,
        out_tile_bytes,
        sem_gather_id,
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
                writer_desc.runtime_args.emplace_back(
                    core,
                    KernelDescriptor::CoreRuntimeArgs{
                        out_buffer->address(), col_start_tile_for(core), leader_noc_x, leader_noc_y});
            }
        }
        desc.kernels.push_back(std::move(writer_desc));
    };
    make_writer(writer_noc1_cores, DataMovementProcessor::RISCV_1, NOC::NOC_1);
    make_writer(writer_noc0_cores, DataMovementProcessor::RISCV_0, NOC::NOC_0);

    return desc;
}

}  // namespace ttnn::operations::experimental::deepseek::moe::fused_experts

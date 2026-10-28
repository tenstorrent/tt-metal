// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/core_local_mem.h"

namespace {

constexpr uint32_t kFaceRowBytes = 16 * 2;              // 16 bf16 columns: one face row
constexpr uint32_t kRowChunkBytes = 2 * kFaceRowBytes;  // 32 bf16 columns: one tile width of a row

// Byte offset of row r's 16-column half h inside a (tile_h x 32) bf16 tile of face_r_dim-row faces:
// faces are raster ordered (0, 1 on top, 2, 3 below when tile_h == 32), each row-major.
FORCE_INLINE uint32_t face_row_offset(uint32_t r, uint32_t h, uint32_t face_r_dim) {
    return ((((r / face_r_dim) * 2u) + h) * face_r_dim + (r % face_r_dim)) * kFaceRowBytes;
}

}  // namespace

// Writer / gather-scatter kernel (runs on every compute core), structured in two phases:
//
//   INPUT (INPUT_RM_ROWS only): several ROW_MAJOR token rows are not a run of tiles, so before
//      anything else this core restripes them into its own cb_input as (tile_h x 32) tiles and
//      publishes it -- from its L1 replica in place (INPUT_REPLICATED), or from DRAM through a
//      double-buffered staging CB. The writer is otherwise idle until compute produces, so this
//      runs alongside the routing and the first gate_up weight fetch.
//
//   PHASE 1 -- GATHER (SwiGLU cores only): for every expert, copies this core's
//      swiglu_tiles-tile SwiGLU activation slice (cb_out) into the cb_act slot of the hub that
//      owns that slice, at tile offset (j*i_tiles + col_start) -- a single NoC write -- then bumps
//      BOTH hubs' gather semaphores. The I dim is split at `split_col`: a slice starting below it
//      goes to hub0, at or above it to hub1. Hub0/hub1 are the two corners of the gather's
//      multicast rectangle -- {0,0} and {GRID_X-1, GRID_Y-1} on the 64-core path (col_start is
//      core_index*swiglu_tiles), and the first/last core of this expert's 16-core group on the
//      6-expert / 96-core path (col_start is local_idx*i_shards_per_core, the core's I-tiles
//      written together -- 1 at TP I=512, 4 at I=2048).
//
//   PHASE 2 -- WRITE OUTPUT:
//      64-core path: the compute kernel has accumulated every expert into cb_down_out, so this
//      core writes its 2-tile H-slice to DRAM.
//      6-expert path: each group holds one expert's H-slice. Non-root groups unicast their
//      shards_per_core*2 tiles to the matching core in group 0; group 0's compute reduces them
//      and this writer then drains the sum to DRAM.
//
// Compile-time args:
//   0: num_active     (routing-selected experts to run)
//   1: i_tiles        (I/32; activation stride between experts in the gathered block)
//   2: cb_out         (this core's SwiGLU activation tiles per expert)
//   3: cb_down_out    (this core's accumulated output tiles)
//   4: cb_act         (gathered activation; used only to locate the leader's L1 address)
//   5: act_tile_bytes (bytes per activation tile)
//   6: out_tile_bytes (bytes per output tile)
//   7: sem_gather     (leader's gather semaphore)
//   8: num_producers  (number of SwiGLU cores in the gather group)
//   9: experts_block  (experts per block; cb_act holds one block per slot)
//   10: cores_per_expert
//   11: shards_per_core (down/H shards this core covers)
//   12: i_shards_per_core (gate_up/I shards this core covers)
//   13: num_expert_groups (1 = 64-core path, 6 = 96-core path)
//   14: cb_reduce
//   15: sem_reduce
//   16: dst_tiles_per_page (compute tiles packed into one output TensorAccessor page;
//       TILE == 1, ROW_MAJOR stick/shard may hold several 1x32 faces)
//   17: split_col (first act tile of each expert that hub1 owns; == i_tiles when there is one hub)
//   18: num_hubs  (1, or 2 for the two-hub gather/broadcast)
//   19+: TensorAccessorArgs(output)
//   then: shared_i_tiles (6-expert path's shared expert I-tiles; 0 without one)
//   then (used by INPUT_RM_ROWS): rows, k_tiles, cb_input, input_tile_bytes, face_r_dim,
//         input_row_stride (bytes between a replica's rows), cb_rm_stage, stage_chunk_tiles
//         (32-column chunks per staged DRAM read), then TensorAccessorArgs(input)
//
// Runtime args:
//   0: output base address
//   1: core_index (this core's flat grid index, x*GRID_Y + y)
//   2: hub0_noc_x     3: hub0_noc_y (this core's group's hub0, the low-I gather target)
//   4: reduce_noc_x   5: reduce_noc_y (group-0 counterpart, 6-expert path only)
//   6: hub1_noc_x     7: hub1_noc_y (this core's group's hub1, the high-I gather target)
//   8: down_local     (6-expert path: this core's slot among its group's 16 down cores, picking its
//                      H columns; 0xFFFFFFFF on a gate_up-only 9th-row core, which writes no output)
//   9: input base address (INPUT_RM_ROWS only)
//
// A producer scatters its I slice to whichever hub owns it (the slice's first column decides: below
// `split_col` goes to hub0, at or above it to hub1) and bumps BOTH hubs' gather semaphores. Bumping
// the hub that did not receive the chunk matters: that bump is the evidence the other hub needs that
// this core's compute has moved past the previous block, and it is the only such evidence for a core
// whose chunk that hub never gathers.
void kernel_main() {
    constexpr uint32_t num_active = get_compile_time_arg_val(0);
    constexpr uint32_t i_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out_id = get_compile_time_arg_val(2);
    constexpr uint32_t cb_down_out_id = get_compile_time_arg_val(3);
    constexpr uint32_t cb_act_id = get_compile_time_arg_val(4);
    constexpr uint32_t act_tile_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t out_tile_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t sem_gather_id = get_compile_time_arg_val(7);
    constexpr uint32_t num_producers = get_compile_time_arg_val(8);
    constexpr uint32_t experts_block = get_compile_time_arg_val(9);
    constexpr uint32_t cores_per_expert = get_compile_time_arg_val(10);
    constexpr uint32_t shards_per_core = get_compile_time_arg_val(11);
    constexpr uint32_t i_shards_per_core = get_compile_time_arg_val(12);
    constexpr uint32_t num_expert_groups = get_compile_time_arg_val(13);
    constexpr uint32_t cb_reduce_id = get_compile_time_arg_val(14);
    constexpr uint32_t sem_reduce_id = get_compile_time_arg_val(15);
    constexpr uint32_t dst_tiles_per_page = get_compile_time_arg_val(16);
    constexpr uint32_t split_col = get_compile_time_arg_val(17);
    constexpr uint32_t num_hubs = get_compile_time_arg_val(18);

    constexpr auto out_args = TensorAccessorArgs<19>();
    constexpr uint32_t kSharedArg = out_args.next_compile_time_args_offset();
    constexpr uint32_t shared_i_tiles = get_compile_time_arg_val(kSharedArg);
    constexpr uint32_t rows = get_compile_time_arg_val(kSharedArg + 1);
    constexpr uint32_t k_tiles = get_compile_time_arg_val(kSharedArg + 2);
    constexpr uint32_t cb_input_id = get_compile_time_arg_val(kSharedArg + 3);
    constexpr uint32_t input_tile_bytes = get_compile_time_arg_val(kSharedArg + 4);
    constexpr uint32_t face_r_dim = get_compile_time_arg_val(kSharedArg + 5);
    constexpr uint32_t input_row_stride = get_compile_time_arg_val(kSharedArg + 6);
    constexpr uint32_t cb_rm_stage_id = get_compile_time_arg_val(kSharedArg + 7);
    constexpr uint32_t stage_chunk_tiles = get_compile_time_arg_val(kSharedArg + 8);
    constexpr auto input_args = TensorAccessorArgs<kSharedArg + 9>();

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t core_index = get_arg_val<uint32_t>(1);
    const uint32_t hub0_noc_x = get_arg_val<uint32_t>(2);
    const uint32_t hub0_noc_y = get_arg_val<uint32_t>(3);
    const uint32_t reduce_noc_x = get_arg_val<uint32_t>(4);
    const uint32_t reduce_noc_y = get_arg_val<uint32_t>(5);
    const uint32_t hub1_noc_x = get_arg_val<uint32_t>(6);
    const uint32_t hub1_noc_y = get_arg_val<uint32_t>(7);
    const uint32_t down_local = get_arg_val<uint32_t>(8);
    const bool down_core = down_local != 0xFFFFFFFFu;
    (void)down_core;

    constexpr uint32_t kOutTiles = 2;
    constexpr uint32_t swiglu_tiles =
        num_expert_groups > 1 ? (i_tiles / (cores_per_expert * i_shards_per_core)) : (i_tiles / num_producers);
    const uint32_t local_idx = core_index % cores_per_expert;
    const uint32_t expert_group = core_index / cores_per_expert;
    const bool swiglu_core = local_idx < num_producers;
    const uint32_t swiglu_col_start =
        num_expert_groups > 1 ? local_idx * i_shards_per_core * swiglu_tiles : core_index * swiglu_tiles;
    const uint32_t out_col_start =
        num_expert_groups > 1 ? down_local * shards_per_core * kOutTiles : core_index * kOutTiles;
    // This core's chunk belongs to hub1 if it starts at or past the I split; otherwise to hub0. With
    // one hub `split_col == i_tiles`, so every chunk goes to hub0.
    const bool chunk_is_hub1 = swiglu_col_start >= split_col;
    const uint32_t chunk_hub_x = chunk_is_hub1 ? hub1_noc_x : hub0_noc_x;
    const uint32_t chunk_hub_y = chunk_is_hub1 ? hub1_noc_y : hub0_noc_y;

    Noc noc;
    const auto out = TensorAccessor(out_args, out_addr);
    CircularBuffer cb_out(cb_out_id);
    CircularBuffer cb_down_out(cb_down_out_id);
    CircularBuffer cb_act(cb_act_id);
    CircularBuffer cb_reduce(cb_reduce_id);

#ifdef INPUT_RM_ROWS
    {
        // Lays `num_tiles` 32-column chunks of the rows (row r at src + r * src_row_stride) out as
        // consecutive input tiles from dst. Tile rows past `rows` are left as the CB held them: matmul
        // and SwiGLU keep rows independent, and those rows are never written back out.
        auto rows_to_tiles = [](uint32_t src, uint32_t src_row_stride, uint32_t dst, uint32_t num_tiles) {
            constexpr uint32_t kHalfWords = kFaceRowBytes / sizeof(uint32_t);
            for (uint32_t r = 0; r < rows; ++r) {
                auto* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src + r * src_row_stride);
                const uint32_t lo = face_row_offset(r, 0, face_r_dim);
                const uint32_t hi = face_row_offset(r, 1, face_r_dim);
                for (uint32_t t = 0; t < num_tiles; ++t) {
                    auto* d_lo = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst + t * input_tile_bytes + lo);
                    auto* d_hi = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst + t * input_tile_bytes + hi);
                    for (uint32_t w = 0; w < kHalfWords; ++w) {
                        d_lo[w] = s[w];
                    }
                    for (uint32_t w = 0; w < kHalfWords; ++w) {
                        d_hi[w] = s[kHalfWords + w];
                    }
                    s += 2 * kHalfWords;
                }
            }
        };

        const uint32_t input_addr = get_arg_val<uint32_t>(9);
        CircularBuffer cb_input(cb_input_id);
        cb_input.reserve_back(k_tiles);
        const uint32_t tiles_l1 = cb_input.get_write_ptr();
#ifdef INPUT_REPLICATED
        // The replica's rows sit in this core's L1 at the same address on every core.
        (void)input_args;
        (void)cb_rm_stage_id;
        (void)stage_chunk_tiles;
        rows_to_tiles(input_addr, input_row_stride, tiles_l1, k_tiles);
#else
        // A DRAM read keeps 64-byte alignment between source and destination, which a 32-byte face
        // row does not have, so each chunk of every row lands contiguously in a staging slot first.
        // While one slot is restriped the next chunk is in flight into the other.
        (void)input_row_stride;
        const auto input = TensorAccessor(input_args, input_addr);
        CircularBuffer cb_stage(cb_rm_stage_id);
        const uint32_t stage_l1 = (cb_stage.get_write_ptr() + 63u) & ~63u;
        constexpr uint32_t stage_row_bytes = stage_chunk_tiles * kRowChunkBytes;
        constexpr uint32_t slot_bytes = rows * stage_row_bytes;
        constexpr uint32_t num_chunks = (k_tiles + stage_chunk_tiles - 1) / stage_chunk_tiles;
        auto chunk_tiles = [](uint32_t c) {
            const uint32_t left = k_tiles - c * stage_chunk_tiles;
            return left < stage_chunk_tiles ? left : stage_chunk_tiles;
        };
        auto read_chunk = [&](uint32_t c) {
            const uint32_t slot = stage_l1 + (c % 2u) * slot_bytes;
            for (uint32_t r = 0; r < rows; ++r) {
                noc.async_read(
                    input,
                    CoreLocalMem<uint32_t>(slot + r * stage_row_bytes),
                    chunk_tiles(c) * kRowChunkBytes,
                    {.page_id = r, .offset_bytes = c * stage_row_bytes},
                    {});
            }
        };
        read_chunk(0);
        for (uint32_t c = 0; c < num_chunks; ++c) {
            noc.async_read_barrier();
            if (c + 1 < num_chunks) {
                read_chunk(c + 1);
            }
            rows_to_tiles(
                stage_l1 + (c % 2u) * slot_bytes,
                stage_row_bytes,
                tiles_l1 + c * stage_chunk_tiles * input_tile_bytes,
                chunk_tiles(c));
        }
#endif
        cb_input.push_back(k_tiles);
    }
#else
    (void)rows;
    (void)k_tiles;
    (void)cb_input_id;
    (void)input_tile_bytes;
    (void)input_row_stride;
    (void)cb_rm_stage_id;
    (void)stage_chunk_tiles;
    (void)input_args;
#endif

    // Drains `num_tiles` output tiles from the front of cb_down_out into H columns from
    // out_col_start. Restriped rows go back as one face row per (row, half), because each row of a
    // ROW_MAJOR output is its own page run; the 32-byte writes only need DRAM's 16-byte write alignment.
    auto write_output = [&](uint32_t num_tiles) {
        for (uint32_t t = 0; t < num_tiles; ++t) {
            const uint32_t global_tile = out_col_start + t;
#ifdef INPUT_RM_ROWS
            constexpr uint32_t pages_per_row = k_tiles / dst_tiles_per_page;
            for (uint32_t r = 0; r < rows; ++r) {
                for (uint32_t h = 0; h < 2; ++h) {
                    noc.async_write(
                        cb_down_out,
                        out,
                        kFaceRowBytes,
                        {.offset_bytes = t * out_tile_bytes + face_row_offset(r, h, face_r_dim)},
                        {.page_id = r * pages_per_row + global_tile / dst_tiles_per_page,
                         .offset_bytes = (global_tile % dst_tiles_per_page) * kRowChunkBytes + h * kFaceRowBytes});
                }
            }
#else
            noc.async_write(
                cb_down_out,
                out,
                out_tile_bytes,
                {.offset_bytes = t * out_tile_bytes},
                {.page_id = global_tile / dst_tiles_per_page,
                 .offset_bytes = (global_tile % dst_tiles_per_page) * out_tile_bytes});
#endif
        }
        noc.async_write_barrier();
    };

    // cb_act / cb_reduce are allocated identically on every core, so every hub's staging region
    // shares this core's BASE address -- read here, before anything can have advanced the write
    // pointer.
    const uint32_t act_slot_base = cb_act.get_write_ptr();
    const uint32_t reduce_base = cb_reduce.get_write_ptr();
    constexpr uint32_t kNumBlocks = (num_active + experts_block - 1) / experts_block;
    const uint32_t block_act_tiles = experts_block * i_tiles;
    const uint32_t act_slots = kNumBlocks > 1 ? 2 : 1;

    Semaphore<> sem_gather(sem_gather_id);

    // The gather bump goes to BOTH hubs: the owning hub needs it to know the chunk landed, and the
    // other hub needs it as the only evidence that this core is done with the previous block's slot
    // (that hub never gathers this core's chunk, so it has no scatter of its own to observe).
    auto bump_gather = [&]() {
        sem_gather.up(noc, hub0_noc_x, hub0_noc_y, 1);
        if constexpr (num_hubs > 1) {
            sem_gather.up(noc, hub1_noc_x, hub1_noc_y, 1);
        }
    };

    if constexpr (num_expert_groups > 1) {
        const uint32_t scatter_tiles = i_shards_per_core * swiglu_tiles;
        if (swiglu_core) {
            cb_out.wait_front(scatter_tiles);
            noc.async_write(
                cb_out,
                UnicastEndpoint{},
                scatter_tiles * act_tile_bytes,
                {.offset_bytes = 0},
                {.noc_x = chunk_hub_x,
                 .noc_y = chunk_hub_y,
                 .addr = act_slot_base + swiglu_col_start * act_tile_bytes});
            noc.async_write_barrier();
            bump_gather();
            cb_out.pop_front(scatter_tiles);
        }
        // Shared-expert tile j of this group runs on local core j + 1 and lands right after the routed
        // activation, in the last hub's range (see run_reader_loop).
        const uint32_t shared_tiles = expert_group < shared_i_tiles
                                          ? (shared_i_tiles - expert_group + num_expert_groups - 1u) / num_expert_groups
                                          : 0u;
        if (local_idx >= 1u && local_idx < 1u + shared_tiles) {
            cb_out.wait_front(1);
            noc.async_write(
                cb_out,
                UnicastEndpoint{},
                act_tile_bytes,
                {.offset_bytes = 0},
                {.noc_x = num_hubs > 1 ? hub1_noc_x : hub0_noc_x,
                 .noc_y = num_hubs > 1 ? hub1_noc_y : hub0_noc_y,
                 .addr = act_slot_base + (i_tiles + local_idx - 1u) * act_tile_bytes});
            noc.async_write_barrier();
            bump_gather();
            cb_out.pop_front(1);
        }
        if (!down_core) {
            noc.async_atomic_barrier();
            return;
        }

        constexpr uint32_t kLocalTiles = shards_per_core * kOutTiles;
        cb_down_out.wait_front(kLocalTiles);
        if (expert_group == 0) {
            write_output(kLocalTiles);
        } else {
            const uint32_t slot = (expert_group - 1u) * kLocalTiles * out_tile_bytes;
            noc.async_write(
                cb_down_out,
                UnicastEndpoint{},
                kLocalTiles * out_tile_bytes,
                {.offset_bytes = 0},
                {.noc_x = reduce_noc_x, .noc_y = reduce_noc_y, .addr = reduce_base + slot});
            noc.async_write_barrier();
            Semaphore<>(sem_reduce_id).up(noc, reduce_noc_x, reduce_noc_y, 1);
        }
        cb_down_out.pop_front(kLocalTiles);
    } else {
        if (swiglu_core) {
            for (uint32_t blk = 0; blk < kNumBlocks; ++blk) {
                const uint32_t first_expert = blk * experts_block;
                const uint32_t remaining = num_active - first_expert;
                const uint32_t block_experts = remaining < experts_block ? remaining : experts_block;
                const uint32_t slot_base = act_slot_base + (blk % act_slots) * block_act_tiles * act_tile_bytes;

                for (uint32_t j = 0; j < block_experts; ++j) {
                    cb_out.wait_front(swiglu_tiles);
                    const uint32_t hub_addr = slot_base + (j * i_tiles + swiglu_col_start) * act_tile_bytes;
                    noc.async_write(
                        cb_out,
                        UnicastEndpoint{},
                        swiglu_tiles * act_tile_bytes,
                        {.offset_bytes = 0},
                        {.noc_x = chunk_hub_x, .noc_y = chunk_hub_y, .addr = hub_addr});
                    noc.async_write_barrier();
                    bump_gather();
                    cb_out.pop_front(swiglu_tiles);
                }
            }
        }

        // The compute kernel has already summed routing_w[e] * down_e over all experts into
        // cb_down_out, so the writer drains it once into output pages idx*2 + {0, 1}.
        cb_down_out.wait_front(kOutTiles);
        write_output(kOutTiles);
        cb_down_out.pop_front(kOutTiles);
    }
    // Every Semaphore::up above is a non-posted atomic; async_write_barrier does not drain those.
    noc.async_atomic_barrier();
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/core_local_mem.h"

#include "fetch_gate_up.h"

// Expert-id sender + activation-gather leader kernel (runs on core {0,0}).
//
// 1. Reads the routing input -- the router's k selected expert ids per token plus the score row
//    they index -- and computes the selected ("hit") expert ids on device, compacted ascending at
//    the front of cb_bcast and padded with the sentinel. The id list is the DEDUPLICATED UNION of
//    the tokens' selections: an expert that several tokens picked appears exactly once and is
//    therefore fetched exactly once by every core, with one matmul serving all of those tokens.
//    Alongside the ids it publishes each hit's per-token routing weights -- the selected scores
//    renormalized and scaled here rather than by the caller -- which is what keeps the tokens
//    distinguishable after the dedup.
//
//    The dedup and the ascending sort come out of one E-bit bitmap pass, so the whole thing is
//    O(B x k): the caller hands over its selection untouched instead of scattering it out to E
//    columns that this kernel would only scan straight back down to k. The ids arrive either as
//    raw uint16 (a topk output) or as bf16 values (an embedding gather, which is how a
//    table-driven router produces them); `index_is_bf16` picks the decode.
// 2. Multicasts the ids buffer to all other compute cores' L1 (cb_bcast).
// 3. Sets + multicasts a semaphore (sem_id) to signal the other cores.
// 4. Waits for the activation broadcast and publishes it to this core's compute.
// 5. Runs the per-expert reader loop as the LEADER: fetches this core's gate_up + down
//    slices, gathers every SwiGLU core's activation chunk into the local cb_act, and
//    broadcasts the full activation back to every core for the down matmul.
//
// Compile-time args:
//   0: num_weights (total experts whose weights are provided; routing-row width)
//   1: num_active  (routing-selected experts to run)
//   2: sentinel value for unused id slots (= num_weights)
//   3: cb_routing  (L1 scratch for the ids, the score row and the selection tables)
//   4: cb_bcast    (L1 buffer holding the expert ids; broadcast to all cores)
//   5: routing_page_bytes
//   6: bcast_page_bytes
//   7: sem_id      (expert-ids-ready / sequencing semaphore)
//   8: cb_weights  (this core's per-expert gate_up slice)
//   9: k_tiles     (H / 32)
//   10: i_tiles    (I / 32)
//   11: gate_up_tile_bytes
//   12: sem_input_id (input-ready semaphore)
//   13: cb_input     (activation tiles, published to compute)
//   14: cb_down_w    (this core's per-expert down slice)
//   15: cb_act       (gathered activation)
//   16: down_slice_tiles
//   17: down_tile_bytes
//   18: act_tile_bytes
//   19: num_producers (number of SwiGLU cores == I/64)
//   20: sem_gather
//   21: sem_bcast
//   22: cb_rscalar
//   23: down_prefetch (down slices to fetch before the gather/broadcast sync)
//   24: batch        (token rows, B <= 32; one id row each)
//   25: experts_block (experts per block; the activation block held in L1 at once)
//   26: gate_up_reserve_tiles (pages a gate_up slice reserves in cb_weights)
//   27: down_reserve_tiles    (pages a down slice reserves in cb_weights)
//   28: top_k          (ids per token row)
//   29: index_is_bf16  (1 = ids arrive as bf16 values, 0 = raw uint16)
//   30: scaling_bits   (routed_scaling_factor, as fp32 bits)
//   31: eps_bits       (routing_eps, as fp32 bits)
//   32: score_pages    (E/32 tile pages in the score row)
//   33: score_page_bytes   34: score_page_stride
//   35: score_l1_offset    (where the score tiles land in cb_routing)
//   36: scratch_l1_offset  (where the bitmap + rank + id cache live in cb_routing)
//   37: bitmap_bytes       (bitmap size, i.e. the rank table's offset past it)
//   38: rank_bytes         (rank table size, i.e. the id cache's offset past it)
//   39+: TensorAccessorArgs(routing), TensorAccessorArgs(scores), TensorAccessorArgs(gate_up),
//        TensorAccessorArgs(down)
//   then: gate_up base addresses (one per expert), then down base addresses (one per expert)
//
// Runtime args:
//   0: routing base address (the selected-id tensor)
//   1: mcast_start_x   2: mcast_start_y
//   3: mcast_end_x     4: mcast_end_y
//   5: num_dests       (number of receiver cores = total cores - 1)
//   6: core_index      (this core's flat grid index, x*8 + y)
//   7: score base address

namespace {

constexpr uint32_t kFaceBytes = 512;    // 16x16 face of 2-byte elements
constexpr uint32_t kFaceRowBytes = 32;  // one face row (16 elements)

// Byte offset of element (row, col) inside a 32x32 tile of 2-byte elements. A tile is 4 faces --
// faces 0,1 cover rows 0-15 and faces 2,3 rows 16-31, each face 16 columns wide -- which is how a
// token's row is reached in the topk ids and the score row without untilizing either of them.
FORCE_INLINE uint32_t tile_elem_offset(uint32_t row, uint32_t col) {
    const uint32_t face = ((row >> 4) << 1) + (col >> 4);
    return (face * kFaceBytes) + ((row & 15u) * kFaceRowBytes) + ((col & 15u) * 2u);
}

// Widen bf16 to fp32 (exact: bf16 is a subset). Soft-float on the data-movement RISC-V core, which
// is affordable here because only top_k values per token are ever touched.
FORCE_INLINE float bf16_to_f32(uint16_t v) {
    const uint32_t bits = static_cast<uint32_t>(v) << 16;
    float out;
    __builtin_memcpy(&out, &bits, sizeof(out));
    return out;
}

// Round fp32 to nearest-even bf16, returned widened back to fp32 bits. cb_bcast weight slots are
// consumed by build_routing_scalars as a `>> 16` truncation, so rounding here is what makes the
// delivered weight the correctly-rounded bf16 that a dense caller would have supplied.
FORCE_INLINE uint32_t f32_to_bf16_bits(float v) {
    uint32_t bits;
    __builtin_memcpy(&bits, &v, sizeof(bits));
    return (bits + 0x7FFFu + ((bits >> 16) & 1u)) & 0xFFFF0000u;
}

}  // namespace

void kernel_main() {
    constexpr uint32_t num_weights = get_compile_time_arg_val(0);
    constexpr uint32_t num_active = get_compile_time_arg_val(1);
    constexpr uint32_t sentinel = get_compile_time_arg_val(2);
    constexpr uint32_t cb_routing_id = get_compile_time_arg_val(3);
    constexpr uint32_t cb_bcast_id = get_compile_time_arg_val(4);
    constexpr uint32_t routing_page_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t bcast_page_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t sem_id = get_compile_time_arg_val(7);
    constexpr uint32_t cb_weights_id = get_compile_time_arg_val(8);
    constexpr uint32_t k_tiles = get_compile_time_arg_val(9);
    constexpr uint32_t i_tiles = get_compile_time_arg_val(10);
    constexpr uint32_t gate_up_tile_bytes = get_compile_time_arg_val(11);
    constexpr uint32_t sem_input_id = get_compile_time_arg_val(12);
    constexpr uint32_t cb_input_id = get_compile_time_arg_val(13);
    constexpr uint32_t cb_down_w_id = get_compile_time_arg_val(14);
    constexpr uint32_t cb_act_id = get_compile_time_arg_val(15);
    constexpr uint32_t down_slice_tiles = get_compile_time_arg_val(16);
    constexpr uint32_t down_tile_bytes = get_compile_time_arg_val(17);
    constexpr uint32_t act_tile_bytes = get_compile_time_arg_val(18);
    constexpr uint32_t num_producers = get_compile_time_arg_val(19);
    constexpr uint32_t sem_gather_id = get_compile_time_arg_val(20);
    constexpr uint32_t sem_bcast_id = get_compile_time_arg_val(21);
    constexpr uint32_t cb_rscalar_id = get_compile_time_arg_val(22);
    constexpr uint32_t down_prefetch = get_compile_time_arg_val(23);
    constexpr uint32_t batch = get_compile_time_arg_val(24);
    constexpr uint32_t experts_block = get_compile_time_arg_val(25);
    constexpr uint32_t gate_up_reserve_tiles = get_compile_time_arg_val(26);
    constexpr uint32_t down_reserve_tiles = get_compile_time_arg_val(27);
    constexpr uint32_t top_k = get_compile_time_arg_val(28);
    constexpr bool index_is_bf16 = get_compile_time_arg_val(29) == 1;
    constexpr uint32_t scaling_bits = get_compile_time_arg_val(30);
    constexpr uint32_t eps_bits = get_compile_time_arg_val(31);
    constexpr uint32_t score_pages = get_compile_time_arg_val(32);
    constexpr uint32_t score_page_bytes = get_compile_time_arg_val(33);
    constexpr uint32_t score_page_stride = get_compile_time_arg_val(34);
    constexpr uint32_t score_l1_offset = get_compile_time_arg_val(35);
    constexpr uint32_t scratch_l1_offset = get_compile_time_arg_val(36);
    constexpr uint32_t bitmap_bytes = get_compile_time_arg_val(37);
    constexpr uint32_t rank_bytes = get_compile_time_arg_val(38);
    // Routing-scalar tile geometry (bf16 tile matching the input's height; width fixed at 32).
    constexpr uint32_t rscalar_tile_h = get_compile_time_arg_val(39);
    constexpr uint32_t rscalar_face_r_dim = get_compile_time_arg_val(40);
    constexpr uint32_t rscalar_num_face_rows = get_compile_time_arg_val(41);
    constexpr uint32_t rscalar_tile_bytes = get_compile_time_arg_val(42);
    constexpr uint32_t cores_per_expert = get_compile_time_arg_val(43);
    constexpr uint32_t shards_per_core = get_compile_time_arg_val(44);
    constexpr uint32_t i_shards_per_core = get_compile_time_arg_val(45);
    constexpr uint32_t num_expert_groups = get_compile_time_arg_val(46);
    constexpr uint32_t sem_reduce_id = get_compile_time_arg_val(47);
    constexpr uint32_t cb_reduce_id = get_compile_time_arg_val(48);

    constexpr auto routing_args = TensorAccessorArgs<49>();
    constexpr auto score_args = TensorAccessorArgs<routing_args.next_compile_time_args_offset()>();
    constexpr auto gate_up_args = TensorAccessorArgs<score_args.next_compile_time_args_offset()>();
    constexpr auto down_args = TensorAccessorArgs<gate_up_args.next_compile_time_args_offset()>();
    // The gate_up then down weight base addresses (one per expert) follow the accessor args
    // in the compile-time args, indexed by the runtime-selected expert id.
    constexpr uint32_t kGateUpAddrBase = down_args.next_compile_time_args_offset();
    constexpr uint32_t kDownAddrBase = kGateUpAddrBase + num_weights;

    const uint32_t routing_addr = get_arg_val<uint32_t>(0);
    const uint32_t mcast_start_x = get_arg_val<uint32_t>(1);
    const uint32_t mcast_start_y = get_arg_val<uint32_t>(2);
    const uint32_t mcast_end_x = get_arg_val<uint32_t>(3);
    const uint32_t mcast_end_y = get_arg_val<uint32_t>(4);
    const uint32_t num_dests = get_arg_val<uint32_t>(5);
    const uint32_t core_index = get_arg_val<uint32_t>(6);
    const uint32_t score_addr = get_arg_val<uint32_t>(7);
    const uint32_t group_mcast_start_x = get_arg_val<uint32_t>(8);
    const uint32_t group_mcast_start_y = get_arg_val<uint32_t>(9);
    const uint32_t group_mcast_end_x = get_arg_val<uint32_t>(10);
    const uint32_t group_mcast_end_y = get_arg_val<uint32_t>(11);
    const uint32_t group_num_dests = get_arg_val<uint32_t>(12);

    // Pin the expert-id sender to NoC 0; the input broadcaster on {1,0} uses NoC 1.
    Noc noc(0);
    const auto routing = TensorAccessor(routing_args, routing_addr);

    CircularBuffer cb_routing(cb_routing_id);
    CircularBuffer cb_bcast(cb_bcast_id);

    // ---- 1. Read the routing input + compute expert ids into cb_bcast. ----
    // Pages are placed at the buffer's aligned page stride so each read's L1 destination shares the
    // alignment of the DRAM page it comes from.
    cb_routing.reserve_back(1);
    // One TILE page of ids (B <= 32 rows and top_k <= 16 columns both fit a single 32x32 tile),
    // then the score row's E/32 tile pages after it.
    noc.async_read(routing, cb_routing, routing_page_bytes, {.page_id = 0}, {.offset_bytes = 0});
    const auto scores = TensorAccessor(score_args, score_addr);
    for (uint32_t p = 0; p < score_pages; ++p) {
        noc.async_read(
            scores,
            cb_routing,
            score_page_bytes,
            {.page_id = p},
            {.offset_bytes = score_l1_offset + p * score_page_stride});
    }
    noc.async_read_barrier();

    cb_bcast.reserve_back(1);
    const uint32_t bcast_l1 = cb_bcast.get_write_ptr();

    const uint32_t routing_l1 = cb_routing.get_write_ptr();
    CoreLocalMem<volatile uint16_t> rw(routing_l1);
    CoreLocalMem<volatile uint32_t> ids(bcast_l1);

    // ids[0..num_weights)                : compacted ascending hit ids, padded (see below).
    // ids[num_weights..+num_active*batch): each hit's per-token routing weights as fp32 bit
    //   patterns (a bf16 value widened), hit-major then token row, for the down-output multiply.
    //
    // A hit is recorded ONCE for the whole batch, so the id list is the deduplicated union of the
    // tokens' selections: experts shared by several tokens are fetched and multiplied once, and the
    // tokens stay distinguishable purely through their weight column (which is zero for the tokens
    // that did not select the expert).
    uint32_t n = 0;
    // A `num_weights`-bit bitmap does the dedup and the ascending sort in one pass: setting a
    // bit per selected id collapses duplicates, and walking the words in order then yields the
    // union already sorted, without ever comparing ids against each other.
    constexpr uint32_t bitmap_words = (num_weights + 31u) / 32u;
    CoreLocalMem<volatile uint32_t> bitmap(routing_l1 + scratch_l1_offset);
    // rank[e] is expert e's position in the compacted hit list, so a token can find where to
    // deposit its weight for e without searching the list.
    CoreLocalMem<volatile uint16_t> rank(routing_l1 + scratch_l1_offset + bitmap_bytes);
    // The ids are needed again by the weight pass below, so decode them once here into
    // `sel_ids` (row-major [batch, top_k]) rather than re-reading -- and, for a bf16 producer,
    // re-converting -- them out of the tile.
    CoreLocalMem<volatile uint16_t> sel_ids(routing_l1 + scratch_l1_offset + bitmap_bytes + rank_bytes);
    constexpr uint16_t kDropped = 0xFFFFu;

    for (uint32_t w = 0; w < bitmap_words; ++w) {
        bitmap[w] = 0;
    }
    for (uint32_t b = 0; b < batch; ++b) {
        for (uint32_t j = 0; j < top_k; ++j) {
            const uint16_t raw = rw[tile_elem_offset(b, j) >> 1];
            // topk hands over the id as a plain integer; an embedding gather hands over the
            // same id as the bf16 value it had to be stored as (exact for E <= 256).
            uint32_t e;
            if constexpr (index_is_bf16) {
                e = static_cast<uint32_t>(bf16_to_f32(raw));
            } else {
                e = raw;
            }
            // An id past the expert count would index off the end of the bitmap and corrupt
            // L1, so bound it here rather than trusting the producer. A repeat of an id this
            // same token already picked is dropped too: a table-driven router can hand over
            // one, and the reference collapses it (it scatters the selection into a one-hot
            // mask), so counting it twice in the token's sum would not match. Both cases are
            // parked on the sentinel, which the weight pass then skips.
            bool drop = e >= num_weights;
            for (uint32_t p = 0; p < j && !drop; ++p) {
                drop = sel_ids[b * top_k + p] == e;
            }
            if (!drop) {
                bitmap[e >> 5] |= 1u << (e & 31u);
            } else {
                e = num_weights;
            }
            sel_ids[b * top_k + j] = static_cast<uint16_t>(e);
        }
    }
    // Zero the whole weight region up front: a token contributes to only its own k experts, and
    // every other (hit, token) slot must read as zero.
    for (uint32_t i = 0; i < num_active * batch; ++i) {
        ids[num_weights + i] = 0;
    }
    for (uint32_t w = 0; w < bitmap_words; ++w) {
        uint32_t word = bitmap[w];
        while (word != 0u) {
            const uint32_t bit = static_cast<uint32_t>(__builtin_ctz(word));
            word &= word - 1u;
            const uint32_t e = (w << 5) + bit;
            // A union larger than the caller's bound overflows the fetch loop, which only runs
            // [0, num_active). Drop the surplus rather than writing past it; an under-sized
            // num_experts is the caller's error, and it costs those experts' contributions.
            if (n < num_active) {
                ids[n] = e;
                rank[e] = static_cast<uint16_t>(n);
            } else {
                rank[e] = kDropped;
            }
            ++n;
        }
    }

    // Weights: each token's selected scores renormalized to sum to 1 and scaled. Equivalent to
    // the caller-side `scale * (s * mask) / (sum(s * mask) + eps)`, but over the k selected
    // scores instead of an E-wide masked row.
    constexpr float scaling = __builtin_bit_cast(float, scaling_bits);
    constexpr float eps = __builtin_bit_cast(float, eps_bits);
    constexpr uint32_t sel_slots = top_k > 0u ? top_k : 1u;
    for (uint32_t b = 0; b < batch; ++b) {
        float sel[sel_slots];
        float sum = 0.0f;
        for (uint32_t j = 0; j < top_k; ++j) {
            const uint32_t e = sel_ids[b * top_k + j];
            if (e >= num_weights) {
                sel[j] = 0.0f;
                continue;
            }
            // Expert e's score sits in tile e/32 of the score row, column e%32 of token row b.
            const uint32_t off = score_l1_offset + ((e >> 5) * score_page_stride) + tile_elem_offset(b, e & 31u);
            sel[j] = bf16_to_f32(rw[off >> 1]);
            sum += sel[j];
        }
        const float inv = scaling / (sum + eps);
        for (uint32_t j = 0; j < top_k; ++j) {
            const uint32_t e = sel_ids[b * top_k + j];
            if (e >= num_weights) {
                continue;
            }
            const uint32_t pos = rank[e];
            if (pos < num_active) {
                ids[num_weights + pos * batch + b] = f32_to_bf16_bits(sel[j] * inv);
            }
        }
    }

    // `num_active` may over-provision the union (its exact size is data dependent once B > 1, while
    // the program -- and any trace containing it -- is compiled for a fixed expert count). Unused
    // slots inside the loop bound therefore have to be *harmless* rather than merely unread: they
    // name expert 0 (a valid weight address, so the fetch cannot run off the compile-time address
    // table) with an all-zero weight column, contributing nothing to any token. They do cost one
    // redundant weight fetch each, so passing the exact union size is still the fast path.
    for (uint32_t i = n; i < num_active; ++i) {
        ids[i] = 0;  // the weight region for these slots was zeroed above
    }
    for (uint32_t i = (n > num_active ? n : num_active); i < num_weights; ++i) {
        ids[i] = sentinel;  // never read by the fetch loop; keeps the broadcast buffer well defined
    }

    // ---- 2. Broadcast the ids to all other cores' L1 (same cb_bcast address). ----
    noc.async_write_multicast(
        CoreLocalMem<uint32_t>(bcast_l1),
        MulticastEndpoint{},
        bcast_page_bytes,
        num_dests,
        {.offset_bytes = 0},
        {.noc_x_start = mcast_start_x,
         .noc_y_start = mcast_start_y,
         .noc_x_end = mcast_end_x,
         .noc_y_end = mcast_end_y,
         .addr = bcast_l1},
        /*linked=*/false);
    noc.async_write_barrier();

    // ---- 3. Signal the other cores via the expert-ids-ready semaphore. ----
    Semaphore<> sem(sem_id);
    sem.set(1);
    sem.set_multicast(noc, mcast_start_x, mcast_start_y, mcast_end_x, mcast_end_y, num_dests, /*linked=*/false);

    // ---- 4. Wait for the activation broadcast, then publish it to compute. ----
    Semaphore<>(sem_input_id).wait(1);
    publish_input(cb_input_id, k_tiles);

    // ---- 5. Blocked reader loop (leader role): per block, gate_up, gather+broadcast, down. ----
    run_reader_loop(
        noc,
        num_active,
        core_index,
        i_tiles,
        k_tiles,
        gate_up_tile_bytes,
        down_slice_tiles,
        down_tile_bytes,
        act_tile_bytes,
        num_producers,
        cb_bcast_id,
        cb_weights_id,
        cb_down_w_id,
        cb_act_id,
        sem_gather_id,
        sem_bcast_id,
        group_mcast_start_x,
        group_mcast_start_y,
        group_mcast_end_x,
        group_mcast_end_y,
        group_num_dests,
        gate_up_args,
        kGateUpAddrBase,
        down_args,
        kDownAddrBase,
        cb_rscalar_id,
        num_weights,
        down_prefetch,
        batch,
        experts_block,
        gate_up_reserve_tiles,
        down_reserve_tiles,
        // The leader gathers into its own L1, so it never sends itself a slot-free ack.
        /*leader_noc_x=*/0,
        /*leader_noc_y=*/0,
        rscalar_tile_h,
        rscalar_face_r_dim,
        rscalar_num_face_rows,
        rscalar_tile_bytes,
        cores_per_expert,
        shards_per_core,
        i_shards_per_core,
        num_expert_groups,
        /*is_group_leader=*/true,
        sem_reduce_id,
        cb_reduce_id);
}

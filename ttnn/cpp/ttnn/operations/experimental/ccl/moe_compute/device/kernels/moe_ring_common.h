// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#include "../hostdevcommon/config.hpp"

// Per-chunk device profiler zones of the tilize and ring kernels: a study build (the host defines MOE_ZONES when
// TTNN_MOE_COMPUTE_ZONES is set), the served kernels carry none. The profiler's L1 buffer holds 250 markers per RISC
// per launch (PROFILER_L1_OPTIONAL_MARKER_COUNT) and a zone is two, so the zones record a window of MOE_ZONES_CHUNKS
// chunks from MOE_ZONES_FIRST_CHUNK (in feed order) rather than the whole call. Without the device profiler
// (PROFILE_KERNEL) the zones are nothing even with MOE_ZONES defined.
#if defined(MOE_ZONES) && defined(PROFILE_KERNEL)
#include "tools/profiler/kernel_profiler.hpp"
#ifndef MOE_ZONES_FIRST_CHUNK
#define MOE_ZONES_FIRST_CHUNK 48
#endif
#ifndef MOE_ZONES_CHUNKS
#define MOE_ZONES_CHUNKS 16
#endif
namespace moe_ring::zones {
constexpr bool in_window(uint32_t chunk) {
    return chunk >= MOE_ZONES_FIRST_CHUNK && chunk < MOE_ZONES_FIRST_CHUNK + MOE_ZONES_CHUNKS;
}
// kernel_profiler::profileScope recorded only when `on` (the same marker protocol: start, end, stack depth).
template <uint32_t timer_id>
struct ChunkZone {
    bool start_marked = false;
    inline __attribute__((always_inline)) explicit ChunkZone(bool on) {
        if (on && kernel_profiler::bufferHasRoom()) {
            kernel_profiler::stackSize += kernel_profiler::PROFILER_L1_MARKER_UINT32_SIZE;
            start_marked = true;
            kernel_profiler::mark_time_at_index_inlined(kernel_profiler::wIndex, timer_id);
            kernel_profiler::wIndex += kernel_profiler::PROFILER_L1_MARKER_UINT32_SIZE;
        }
    }
    inline __attribute__((always_inline)) ~ChunkZone() {
        if (start_marked) {
            kernel_profiler::mark_time_at_index_inlined(
                kernel_profiler::wIndex, kernel_profiler::get_const_id(timer_id, kernel_profiler::ZONE_END));
            kernel_profiler::wIndex += kernel_profiler::PROFILER_L1_MARKER_UINT32_SIZE;
            kernel_profiler::stackSize -= kernel_profiler::PROFILER_L1_MARKER_UINT32_SIZE;
        }
    }
};
}  // namespace moe_ring::zones
#define MOE_ZONE_IF(on, name)                                                           \
    DO_PRAGMA(message(PROFILER_MSG_NAME(name)));                                        \
    auto constexpr moe_zone_hash = kernel_profiler::Hash16_CT(PROFILER_MSG_NAME(name)); \
    moe_ring::zones::ChunkZone<moe_zone_hash> moe_zone(on);
#else
#define MOE_ZONE_IF(on, name) (void)(on)
namespace moe_ring::zones {
constexpr bool in_window(uint32_t) { return false; }
}  // namespace moe_ring::zones
#endif

// Study delays: MOE_STUDY_DELAY_<POINT> = spin iterations inserted at one handoff of the feed protocol (the host
// defines them from TTNN_MOE_COMPUTE_STUDY_DELAYS; every one is 0 in the served kernels, where MOE_STUDY_DELAY compiles
// to nothing). One point at a time widens the window between a signal and the data it covers, or slows one party of
// a handoff, to make a race show in a soak. The points (W_ tilize writer, R_ tilize reader, O_ dm1, D_ dm0, M_
// compute):
//   W_BEFORE_MCAST         multicaster: its slot is gathered, before its multicast into the ring cores' half
//   W_BEFORE_SIGNAL        drain: the chunk is multicast (both halves), before the ready signal to the ring cores
//   W_BEFORE_GO            drain: after the ready signal, before feed_go to the gathering tilize cores
//   W_BEFORE_POP           every tilize core: after its send, before it pops its staging slot
//   W_GATHER_BEFORE_WRITE  gathering core: feed_go seen, before its sub-chunk write into the multicaster's slot
//   W_GATHER_BEFORE_INC    gathering core: its write barriered, before the gather count increment
//   W_SECONDARY_BEFORE_INC secondary multicaster: its multicast barriered, before the drain's second-half count
//   R_BEFORE_READS         tilize reader: before a chunk's row reads (a slower feed)
//   O_BEFORE_CREDIT        dm1: the chunk's rows written and popped, before the half credit to the drain
//   O_A2A_BEFORE_INC       dm1: an a2a step's tiles written, before the neighbour's semaphore increment
//   O_BEFORE_ROWS          dm1: the a2a done, before waiting for compute's W2 output (a slower ring core)
//   D_BEFORE_SLICE         dm0: before issuing a weight slice (slower weights)
//   M_BEFORE_W0W1          compute: the chunk's ready signal seen, before its first read of the input half
//   O_A2A_LAG              dm1 of ONE ring position spins before every a2a step, so that core consumes its ring
//                          buffers late while its predecessor runs ahead (an asymmetric lag)
//   O_A2A_LAG_POS          the ring position O_A2A_LAG applies to (a parameter, 0 admitted)
// Fault points (X_, value 1 = on): deliberate breaks of the protocol, the soak's sensitivity controls -- each MUST
// produce mismatches under the matching delay, or the soak proves nothing about that handoff:
//   X_GATHER_NO_GO_WAIT    gathering cores skip the feed_go wait (write into a slot the multicast may still read)
//   X_DRAIN_NO_FLUSH       the drain pops its slot without flushing its multicast (its tilize may overwrite it)
//   X_DRAIN_NO_HALF_WAIT   the drain skips the half credit (multicasts into a half the ring cores may still read)
#ifndef MOE_STUDY_DELAY_W_BEFORE_MCAST
#define MOE_STUDY_DELAY_W_BEFORE_MCAST 0
#endif
#ifndef MOE_STUDY_DELAY_W_BEFORE_SIGNAL
#define MOE_STUDY_DELAY_W_BEFORE_SIGNAL 0
#endif
#ifndef MOE_STUDY_DELAY_W_BEFORE_GO
#define MOE_STUDY_DELAY_W_BEFORE_GO 0
#endif
#ifndef MOE_STUDY_DELAY_W_BEFORE_POP
#define MOE_STUDY_DELAY_W_BEFORE_POP 0
#endif
#ifndef MOE_STUDY_DELAY_W_GATHER_BEFORE_WRITE
#define MOE_STUDY_DELAY_W_GATHER_BEFORE_WRITE 0
#endif
#ifndef MOE_STUDY_DELAY_W_GATHER_BEFORE_INC
#define MOE_STUDY_DELAY_W_GATHER_BEFORE_INC 0
#endif
#ifndef MOE_STUDY_DELAY_W_SECONDARY_BEFORE_INC
#define MOE_STUDY_DELAY_W_SECONDARY_BEFORE_INC 0
#endif
#ifndef MOE_STUDY_DELAY_R_BEFORE_READS
#define MOE_STUDY_DELAY_R_BEFORE_READS 0
#endif
#ifndef MOE_STUDY_DELAY_O_BEFORE_CREDIT
#define MOE_STUDY_DELAY_O_BEFORE_CREDIT 0
#endif
#ifndef MOE_STUDY_DELAY_O_A2A_BEFORE_INC
#define MOE_STUDY_DELAY_O_A2A_BEFORE_INC 0
#endif
#ifndef MOE_STUDY_DELAY_O_BEFORE_ROWS
#define MOE_STUDY_DELAY_O_BEFORE_ROWS 0
#endif
#ifndef MOE_STUDY_DELAY_D_BEFORE_SLICE
#define MOE_STUDY_DELAY_D_BEFORE_SLICE 0
#endif
#ifndef MOE_STUDY_DELAY_M_BEFORE_W0W1
#define MOE_STUDY_DELAY_M_BEFORE_W0W1 0
#endif
#ifndef MOE_STUDY_DELAY_O_A2A_LAG
#define MOE_STUDY_DELAY_O_A2A_LAG 0
#endif
#ifndef MOE_STUDY_DELAY_O_A2A_LAG_POS
#define MOE_STUDY_DELAY_O_A2A_LAG_POS 0
#endif
#ifndef MOE_STUDY_DELAY_X_GATHER_NO_GO_WAIT
#define MOE_STUDY_DELAY_X_GATHER_NO_GO_WAIT 0
#endif
#ifndef MOE_STUDY_DELAY_X_DRAIN_NO_FLUSH
#define MOE_STUDY_DELAY_X_DRAIN_NO_FLUSH 0
#endif
#ifndef MOE_STUDY_DELAY_X_DRAIN_NO_HALF_WAIT
#define MOE_STUDY_DELAY_X_DRAIN_NO_HALF_WAIT 0
#endif
#define MOE_STUDY_FAULT(point) (MOE_STUDY_DELAY_##point > 0)
#define MOE_STUDY_PARAM(point) (MOE_STUDY_DELAY_##point)
namespace moe_ring::study {
inline __attribute__((always_inline)) void spin(uint32_t iterations) {
    // a volatile counter the compiler cannot fold away (the header is also compiled by the host, -Werror on a
    // volatile increment, hence the explicit load-store form)
    volatile uint32_t counter = 0;
    while (counter < iterations) {
        counter = counter + 1;
    }
}
}  // namespace moe_ring::study
#define MOE_STUDY_DELAY(point)                              \
    do {                                                    \
        if constexpr (MOE_STUDY_DELAY_##point > 0) {        \
            moe_ring::study::spin(MOE_STUDY_DELAY_##point); \
        }                                                   \
    } while (0)

namespace moe_ring {

namespace detail {
constexpr uint32_t div_up(const uint32_t a, const uint32_t b) { return (a + b - 1) / b; }

template <uint32_t a, uint32_t b>
constexpr uint32_t div_up() {
    return (a + b - 1) / b;
}

}  // namespace detail

constexpr uint32_t W0_W1_TXNS_PER_BLOCK = 2;
// Tokens per expert chunk: one tile row of activations (the kernels' "tokens_per_chunk" compile arg).
constexpr uint32_t TOKENS_PER_CHUNK = 32;
// The DRAM transaction size in tiles is a per-shape compile-time parameter (tiles_per_txn_for_shape below; the host
// passes it to the kernels as the "tiles_per_txn" named compile arg and they build MoeRingConfig with it). Every block
// height derives from it (MoeRingConfig::block_tiles_h / half_block_tiles_h): there are deliberately no fixed
// 14-tile *_TILES_H / *_TILES_PER_TXN constants, so code written for the fixed geometry fails to compile rather than
// read 28-tile blocks from a 20-tile stream.
constexpr uint32_t DEFAULT_TILES_PER_TXN = 14;  // 8,064 B BF4; 28-tile blocks, 4 wide x 7 K rows
constexpr uint32_t ALT_TILES_PER_TXN = 20;      // 11,520 B BF4 (one NoC packet); 40-tile blocks, 4 wide x 10 K rows

// probably don't need this for W0_W1 and W2 because it's the same
constexpr uint32_t W0_W1_BLOCK_TILES_W = 4;

// Half block-column: the odd gate/up column of a ring core that owns an odd column count is stored as
// (W0 c, W1 c) only. A block is still 2 transactions but 2 tiles wide and twice as many K rows high; each
// stored 4-tile row holds two consecutive K rows (W0 k, W1 k, W0 k+1, W1 k+1).
constexpr uint32_t W0_W1_HALF_BLOCK_TILES_W = W0_W1_BLOCK_TILES_W / 2;

// Let's call this a constant
constexpr uint32_t W2_TILES_PER_A2A_ITER_W = 4;
// Half-width last a2a iteration (ALT_TILES_PER_TXN only, when every core has 1 or 2 output tiles left for it):
// 2 output tiles wide, twice the K rows per block, two consecutive K rows per stored 4-tile row.
constexpr uint32_t W2_HALF_A2A_ITER_TILES_W = W2_TILES_PER_A2A_ITER_W / 2;

//-----------------------------------------------------------------------------
// Shard distribution functions (hardware-agnostic).
// Identical to the Python equivalents in ttnn/ttnn/_experimental/moe_compute_utils.py.
//-----------------------------------------------------------------------------

constexpr bool is_big_w0w1(uint32_t core_id, uint32_t n_big, uint32_t n_cores) {
    return n_big > 0 && (core_id * n_big) % n_cores < n_big;
}

constexpr uint32_t shard_tiles(uint32_t n_tiles, uint32_t core_id, uint32_t n_cores) {
    const uint32_t n_big = n_tiles % n_cores;
    const uint32_t small = n_tiles / n_cores;
    return small + (is_big_w0w1(core_id, n_big, n_cores) ? 1u : 0u);
}

constexpr uint32_t w2_shard_tiles(uint32_t Ht, uint32_t core_id, uint32_t Nt, uint32_t n_cores) {
    const uint32_t n_big_nt = Nt % n_cores;
    const uint32_t n_big_ht = Ht % n_cores;
    const uint32_t small_ht = Ht / n_cores;
    if (n_big_nt + n_big_ht == n_cores) {
        return is_big_w0w1(core_id, n_big_nt, n_cores) ? small_ht : small_ht + 1u;
    }
    return shard_tiles(Ht, core_id, n_cores);
}

template <uint32_t N>
struct ShardLUT {
    uint32_t data[N];
    constexpr uint32_t operator[](uint32_t i) const { return data[i]; }
};

template <uint32_t n_tiles, uint32_t n_cores>
constexpr ShardLUT<n_cores> make_shard_lut() {
    ShardLUT<n_cores> lut{};
    for (uint32_t c = 0; c < n_cores; ++c) {
        lut.data[c] = shard_tiles(n_tiles, c, n_cores);
    }
    return lut;
}

constexpr uint32_t even_stride_at_least_a2a_width(uint32_t tiles) {
    const uint32_t even_tiles = (tiles + 1) & ~1u;
    return even_tiles < W2_TILES_PER_A2A_ITER_W ? W2_TILES_PER_A2A_ITER_W : even_tiles;
}

// Per-shape choice of the W0/W1 column layout. Compact (each ring core stores only its own shard_tiles(Nt, c)
// columns) when that shortens the critical path: the busiest core owns fewer columns than the uniform even stride
// (1-3 columns, or an odd count). Otherwise (e.g. 6/5 or 8/7 columns) the busiest core does the same work either way
// and every core keeps the uniform stride, byte-identical to the per-core stride layout: compacting such shapes only
// added cross-bank reads and half-width passes (GLM-4.5-Air 4096/1408, Nemotron-3-Nano 2688/1856 measured ~1 % slower).
constexpr bool w0_w1_compact_for_shape(uint32_t Nt, uint32_t n_cores) {
    const uint32_t max_cols = (Nt + n_cores - 1) / n_cores;
    return max_cols < even_stride_at_least_a2a_width(max_cols);
}

// Gate/up columns ring core c stores (and produces for a routed expert): its own columns when compact, else the
// uniform stride (its own columns followed by zero columns).
constexpr uint32_t w0_w1_stored_cols(uint32_t Nt, uint32_t core_id, uint32_t n_cores) {
    return w0_w1_compact_for_shape(Nt, n_cores) ? shard_tiles(Nt, core_id, n_cores)
                                                : even_stride_at_least_a2a_width((Nt + n_cores - 1) / n_cores);
}

// The W2 exchange's handshake iterations: dm1 runs the ring (one rendezvous over cb_w2c_rdy and one ring-semaphore
// step per shard) for this many of Cfg::num_a2a_iters. The partials travel in iteration 0 only: it fills a2a buffers
// 1..num_cores-1 once per chunk, every later W2 iteration of the compute reads the same resident buffers, and the
// credit that frees them follows the chunk's W2 output. One iteration is therefore the whole exchange: the later
// iterations' wait / increment pairs are dropped on every core alike (the ring semaphore accounting stays
// consistent) and the compute reads the resident buffers without a rendezvous. Bitwise on every form (decode rows,
// the 128-token chunk, the 2048-row slab); -10 % per moe_compute launch on the 1x4 p150 line (2026-09-26). Part of
// every moe_compute kernel's compile through this header, so the form is in the kernel hash.
constexpr uint32_t a2a_handshake_iters = 1;

// Width in tiles of the in2 slice every ring core hands to the a2a ring: with the compact layout the largest per-core
// column count (at least 2), else the uniform stride.
constexpr uint32_t a2a_exchange_tiles(uint32_t Nt, uint32_t n_cores) {
    const uint32_t max_cols = (Nt + n_cores - 1) / n_cores;
    if (!w0_w1_compact_for_shape(Nt, n_cores)) {
        return even_stride_at_least_a2a_width(max_cols);
    }
    return max_cols < 2 ? 2 : max_cols;
}

// Entry c = block offset of ring core c's compact W0/W1 slice in one (layer, expert) stream; entry n_cores = the
// stream's total blocks. Cfg is a MoeRingConfig.
template <typename Cfg, uint32_t n_cores>
constexpr ShardLUT<n_cores + 1> make_w0_w1_block_offset_lut() {
    ShardLUT<n_cores + 1> lut{};
    for (uint32_t c = 0; c <= n_cores; ++c) {
        lut.data[c] = Cfg::w0_w1_core_block_offset(c);
    }
    return lut;
}

template <uint32_t Ht, uint32_t Nt, uint32_t n_cores>
constexpr ShardLUT<n_cores> make_w2_shard_lut() {
    ShardLUT<n_cores> lut{};
    for (uint32_t c = 0; c < n_cores; ++c) {
        lut.data[c] = w2_shard_tiles(Ht, c, Nt, n_cores);
    }
    return lut;
}

template <uint32_t Ht, uint32_t Nt, uint32_t n_cores>
constexpr ShardLUT<n_cores> make_w2_offset_lut() {
    ShardLUT<n_cores> lut{};
    uint32_t offset = 0;
    for (uint32_t c = 0; c < n_cores; ++c) {
        lut.data[c] = offset;
        offset += w2_shard_tiles(Ht, c, Nt, n_cores);
    }
    return lut;
}

// Compact W0/W1 layout helpers (see MoeRingConfig): blocks for `cols` gate/up columns, and the block offset of
// ring core `core_id`'s slice when all cores' slices of one (layer, expert) are laid back to back.
constexpr uint32_t w0_w1_blocks_for_cols(uint32_t cols, uint32_t blocks_per_col, uint32_t blocks_per_half_col) {
    return (cols / 2) * blocks_per_col + (cols % 2) * blocks_per_half_col;
}

constexpr uint32_t w0_w1_core_block_offset(
    uint32_t Nt, uint32_t core_id, uint32_t n_cores, uint32_t blocks_per_col, uint32_t blocks_per_half_col) {
    uint32_t offset = 0;
    for (uint32_t c = 0; c < core_id; ++c) {
        offset += w0_w1_blocks_for_cols(w0_w1_stored_cols(Nt, c, n_cores), blocks_per_col, blocks_per_half_col);
    }
    return offset;
}

// Block geometry for a transaction size of tiles_per_txn tiles: a block is W0_W1_TXNS_PER_BLOCK transactions,
// 4 tiles wide (W0/W1 block-column, W2 a2a iteration) or 2 tiles wide (half block-column, half a2a iteration).
constexpr uint32_t block_tiles_h(uint32_t tiles_per_txn) {
    return W0_W1_TXNS_PER_BLOCK * tiles_per_txn / W0_W1_BLOCK_TILES_W;
}
constexpr uint32_t half_block_tiles_h(uint32_t tiles_per_txn) {
    return W0_W1_TXNS_PER_BLOCK * tiles_per_txn / W0_W1_HALF_BLOCK_TILES_W;
}

// The per-shape DRAM transaction size (tiles) of both weight streams. 20-tile transactions store the
// Qwen3.8-Flash-Next expert (hidden 2560 = 80 tiles, intermediate 640 = 20 tiles, no bias) on the 8-bank ring, where
// the layout has no padding at all (gate/up K 80 = 8 x 10 and 4 x 20, W2 K 20 = 2 x 10 and 1 x 20; 80 blocks =
// 10 per bank; W2 a2a iterations 4 + 4 + a half). Other rings and every other shape keep the 14-tile layout: 14
// tiles is one 8 KB Wormhole NoC packet, and the 20-tile blocks with padded bank pieces and no half-width W2
// iteration (12 cores: 7 blocks per bank + 4 pad, W2 4 + 3) produced wrong outputs on a Wormhole chip. Mirrored by
// moe_compute_utils.py::_tiles_per_txn.
constexpr uint32_t tiles_per_txn_for_shape(uint32_t Ht, uint32_t Nt, bool has_bias, uint32_t num_cores) {
    return (Ht == 80 && Nt == 20 && !has_bias && num_cores == 8) ? ALT_TILES_PER_TXN : DEFAULT_TILES_PER_TXN;
}

// Blocks the weight CB (c_3) holds: as many as fit in the 3-block budget of 14-tile transactions (84 tiles), and at
// least 3 (3 for 14-tile and for 20-tile transactions, 4 for 10). dm0 keeps all but one of them in flight as DRAM
// reads, so smaller transactions keep about the same bytes in flight. The streaming decode ring adds a fourth block
// (the factory's "weight_slots" named arg, moe_compute_program_factory.cpp): the prefill ring forms' L1 has no room
// for it (their feed halves are larger), the decode ring's has.
constexpr uint32_t weight_cb_slots(uint32_t tiles_per_txn) {
    const uint32_t fit = (3 * W0_W1_TXNS_PER_BLOCK * DEFAULT_TILES_PER_TXN) / (W0_W1_TXNS_PER_BLOCK * tiles_per_txn);
    return fit < 3 ? 3 : fit;
}

// Blocks one DRAM bank holds per (layer, expert) in the compact W0/W1 layout, for a stored K height of
// k_dram_tiles (hidden tiles, plus one with bias): the cores' slices back to back, cut into num_banks equal
// pieces of whole blocks. Runtime form of MoeRingConfig::w0_w1_bank_blocks_per_expert (host side).
constexpr uint32_t w0_w1_bank_blocks_per_expert(
    uint32_t k_dram_tiles, uint32_t Nt, uint32_t n_cores, uint32_t num_banks, uint32_t tiles_per_txn) {
    const uint32_t blocks_per_col = detail::div_up(k_dram_tiles, block_tiles_h(tiles_per_txn));
    const uint32_t blocks_per_half_col = detail::div_up(k_dram_tiles, half_block_tiles_h(tiles_per_txn));
    const uint32_t expert_blocks = w0_w1_core_block_offset(Nt, n_cores, n_cores, blocks_per_col, blocks_per_half_col);
    return detail::div_up(expert_blocks, num_banks);
}

// W2 a2a iterations: ceil(max W2 output tiles per core / 4). With ALT_TILES_PER_TXN the last one is half width
// when it has 1 or 2 output tiles; 14-tile layouts keep 4-wide iterations only (unchanged).
constexpr uint32_t w2_num_a2a_iters(uint32_t Ht, uint32_t n_cores) {
    return detail::div_up(detail::div_up(Ht, n_cores), W2_TILES_PER_A2A_ITER_W);
}
constexpr bool w2_last_a2a_iter_half(uint32_t Ht, uint32_t n_cores, uint32_t tiles_per_txn) {
    const uint32_t last = detail::div_up(Ht, n_cores) % W2_TILES_PER_A2A_ITER_W;
    return tiles_per_txn != DEFAULT_TILES_PER_TXN && last != 0 && last <= W2_HALF_A2A_ITER_TILES_W;
}

// W2 blocks one ring core reads per (layer, expert): each full a2a iteration is ceil(K / block_tiles_h) blocks of
// 4 x block_tiles_h, a half last iteration ceil(K / half_block_tiles_h) blocks of 2 x half_block_tiles_h, with
// K = w2_dram_tiles_h (intermediate tiles, plus one with bias).
constexpr uint32_t w2_core_blocks_per_expert(
    uint32_t Ht, uint32_t w2_dram_tiles_h, uint32_t n_cores, uint32_t tiles_per_txn) {
    const uint32_t iters = w2_num_a2a_iters(Ht, n_cores);
    const uint32_t half = w2_last_a2a_iter_half(Ht, n_cores, tiles_per_txn) ? 1u : 0u;
    return (iters - half) * detail::div_up(w2_dram_tiles_h, block_tiles_h(tiles_per_txn)) +
           half * detail::div_up(w2_dram_tiles_h, half_block_tiles_h(tiles_per_txn));
}

//-----------------------------------------------------------------------------
// Derived ring constants — single source of truth for compute, dm0, dm1.
//-----------------------------------------------------------------------------
// TilesPerTxn: the kernels pass their "tiles_per_txn" named compile arg (tiles_per_txn_for_shape). The block
// geometry, the W0/W1 block counts and the W2 block counts are only valid for a Cfg built with it; in2_tiles_per_step,
// num_a2a_iters and w2_tiles_per_expert_w do not depend on it.
template <
    uint32_t Ht,
    uint32_t Nt,
    uint32_t num_cores,
    bool has_bias,
    uint32_t SharedExpertTp = 1,
    uint32_t TilesPerTxn = DEFAULT_TILES_PER_TXN>
struct MoeRingConfig {
    // DRAM transaction geometry (both weight streams): a block is 2 transactions of tiles_per_txn tiles,
    // consumed 4 wide x block_tiles_h high (or 2 wide x half_block_tiles_h high for the half variants).
    static constexpr uint32_t tiles_per_txn = TilesPerTxn;
    static constexpr uint32_t txns_per_block = W0_W1_TXNS_PER_BLOCK;
    static constexpr uint32_t tiles_per_block = txns_per_block * tiles_per_txn;        // 28 (14-tile txn) or 40 (20)
    static constexpr uint32_t block_tiles_h = moe_ring::block_tiles_h(tiles_per_txn);  // 7 or 10
    static constexpr uint32_t half_block_tiles_h = moe_ring::half_block_tiles_h(tiles_per_txn);  // 14 or 20
    static_assert(tiles_per_block % W0_W1_BLOCK_TILES_W == 0, "a block must be whole 4-tile rows");
    static constexpr uint32_t weight_cb_slots = moe_ring::weight_cb_slots(tiles_per_txn);  // 3 (14- or 20-tile)

    // W0/W1
    static constexpr uint32_t w0_w1_dram_tiles_h = has_bias ? Ht + 1 : Ht;
    static constexpr uint32_t w0_w1_blocks_per_col = (w0_w1_dram_tiles_h + block_tiles_h - 1) / block_tiles_h;
    static constexpr uint32_t in2_tiles_per_step = a2a_exchange_tiles(Nt, num_cores);

    // W0/W1 layout: ring core c stores w0_w1_stored_cols(Nt, c) columns (its own shard_tiles(Nt, c) columns for a
    // compact shape, else the uniform stride) -- floor(cols / 2) block-columns of w0_w1_blocks_per_col blocks
    // (4 wide x block_tiles_h high) and, for an odd count, one half block-column of w0_w1_blocks_per_half_col blocks
    // (2 wide x half_block_tiles_h high), in that order. The in2 slice it hands to the a2a ring is in2_tiles_per_step
    // (a2a_exchange_tiles) wide; tiles past its own columns are never read (the W2 walk takes shard_tiles(src) tiles
    // from each source's slice).
    static constexpr uint32_t w0_w1_blocks_per_half_col =
        (w0_w1_dram_tiles_h + half_block_tiles_h - 1) / half_block_tiles_h;
    // Block offset of core c's slice inside one (layer, expert) stream (the cores' slices back to back).
    static constexpr uint32_t w0_w1_core_block_offset(uint32_t core_id) {
        return moe_ring::w0_w1_core_block_offset(
            Nt, core_id, num_cores, w0_w1_blocks_per_col, w0_w1_blocks_per_half_col);
    }
    // The (layer, expert) stream is cut into num_banks equal pieces of whole blocks (zero-padded at the end);
    // piece b is stored in bank b, so every bank holds the same bytes per expert and core c reads its slice
    // from its own bank plus, where its slice crosses a piece boundary, the next one.
    static constexpr uint32_t w0_w1_bank_blocks_per_expert(uint32_t num_banks) {
        return moe_ring::w0_w1_bank_blocks_per_expert(w0_w1_dram_tiles_h, Nt, num_cores, num_banks, tiles_per_txn);
    }

    // Shared-expert (TpNt) variants: the intermediate dim is TP-split to TpNt = ceil(Nt/tp).
    // After add_shared_expert_weights front-packs each core's real TpNt slice to the front of its
    // full-Nt shard, the kernel reads/produces only the real prefix (in2_tiles_per_step_shared per
    // core) and zero-fills the rest of the core's columns; the full W2 walk then contracts
    // real×real in the prefix and zero×zero past it.
    static constexpr uint32_t TpNt = detail::div_up<Nt, SharedExpertTp>();
    static constexpr uint32_t in2_tiles_per_step_shared =
        even_stride_at_least_a2a_width((TpNt + num_cores - 1) / num_cores);
    // Columns core c produces for an expert: all of its columns for a routed expert; for a shared expert the
    // front-packed prefix, capped at the columns it stores (an even prefix shorter than cols reads pairs only).
    static constexpr uint32_t w0_w1_prod_cols(uint32_t core_id, bool is_shared_expert) {
        const uint32_t cols = w0_w1_stored_cols(Nt, core_id, num_cores);
        return (is_shared_expert && in2_tiles_per_step_shared < cols) ? in2_tiles_per_step_shared : cols;
    }

    // W2
    static constexpr uint32_t max_w2_tiles_per_core = (Ht + num_cores - 1) / num_cores;
    static constexpr uint32_t num_a2a_iters =
        (max_w2_tiles_per_core + W2_TILES_PER_A2A_ITER_W - 1) / W2_TILES_PER_A2A_ITER_W;
    static constexpr uint32_t w2_tiles_per_expert_w = num_a2a_iters * W2_TILES_PER_A2A_ITER_W;
    // With ALT_TILES_PER_TXN the last a2a iteration is half width (2 output tiles, w2_blocks_per_half_a2a_iter
    // blocks of 2 wide x half_block_tiles_h) when every core's last iteration has at most 2 output tiles. The
    // output rows keep the w2_tiles_per_expert_w pitch: the half iteration still packs 4 DEST tiles, the two it does
    // not compute are zero (the packer clears the DEST half at every tile_regs_release) and dm1 copies only
    // w2_shard_tiles(c) tiles of each row.
    static constexpr uint32_t w2_dram_tiles_h = has_bias ? Nt + 1 : Nt;
    static constexpr bool w2_last_iter_half = w2_last_a2a_iter_half(Ht, num_cores, tiles_per_txn);
    static constexpr uint32_t w2_blocks_per_a2a_iter = (w2_dram_tiles_h + block_tiles_h - 1) / block_tiles_h;
    static constexpr uint32_t w2_blocks_per_half_a2a_iter =
        (w2_dram_tiles_h + half_block_tiles_h - 1) / half_block_tiles_h;
    // Blocks per expert that dm0 reads and compute consumes: NOT w2_blocks_per_a2a_iter * num_a2a_iters when the
    // last iteration is half width -- consume per iteration.
    static constexpr uint32_t w2_blocks_per_expert =
        w2_core_blocks_per_expert(Ht, w2_dram_tiles_h, num_cores, tiles_per_txn);
    static_assert(
        w2_blocks_per_expert == (num_a2a_iters - (w2_last_iter_half ? 1u : 0u)) * w2_blocks_per_a2a_iter +
                                    (w2_last_iter_half ? w2_blocks_per_half_a2a_iter : 0u),
        "W2 blocks per expert must be the full iterations' blocks plus the half iteration's");
};

//-----------------------------------------------------------------------------
// Packed (token, k) lists of the local output path (tilize_reader / tilize_writer produce, dm1 consumes).
//-----------------------------------------------------------------------------
// One page: `header_words` words of segment starts (`offsets[0..experts_per_device]`, the rest padding), then the
// entries of every local expert back to back, expert e's `count_e` entries at `offsets[e]`. An entry is
// `(k_slot << TOKEN_BITS) | token_id`. Segment starts are aligned to SEGMENT_ALIGN_ENTRIES entries (64 B) so that
// dm1's per-chunk read of `tokens_per_chunk` entries at `(header_words + offsets[e] + tokens_per_chunk * c) * 4` B
// starts on a DRAM-aligned address; the capacity keeps one chunk of tail (the bound needs half of one) so that read
// never leaves the page. Mirrored by ttnn/ttnn/_experimental/moe_compute_utils.py (token_list_*).
namespace token_list {

constexpr uint32_t TOKEN_BITS = 24;
constexpr uint32_t TOKEN_MASK = (1u << TOKEN_BITS) - 1;
constexpr uint32_t MAX_K_SLOTS = 1u << (32 - TOKEN_BITS);
constexpr uint32_t SEGMENT_ALIGN_ENTRIES = 16;
constexpr uint32_t ENTRY_BYTES = 4;
// A producer's pair: word 0 the token id, word 1 the local expert index in the high half and the k slot in the low.
constexpr uint32_t PAIR_BYTES = 8;
constexpr uint32_t PAIR_EXPERT_SHIFT = 16;

constexpr uint32_t pack_entry(uint32_t token_id, uint32_t k_slot) { return (k_slot << TOKEN_BITS) | token_id; }
constexpr uint32_t entry_token(uint32_t entry) { return entry & TOKEN_MASK; }
constexpr uint32_t entry_k_slot(uint32_t entry) { return entry >> TOKEN_BITS; }

constexpr uint32_t align_entries(uint32_t entries) {
    return detail::div_up(entries, SEGMENT_ALIGN_ENTRIES) * SEGMENT_ALIGN_ENTRIES;
}
// Start of the segment after one that starts at `start` and holds `count` entries.
constexpr uint32_t next_segment_start(uint32_t start, uint32_t count) { return align_entries(start + count); }
constexpr uint32_t header_words(uint32_t experts_per_device) { return align_entries(experts_per_device + 1); }
// Every (token, k slot) is listed at most once; every segment may carry SEGMENT_ALIGN_ENTRIES - 1 entries of padding.
constexpr uint32_t entry_capacity(
    uint32_t tokens, uint32_t selected_experts_k, uint32_t experts_per_device, uint32_t tokens_per_chunk) {
    return align_entries(tokens * selected_experts_k + (SEGMENT_ALIGN_ENTRIES - 1) * experts_per_device) +
           tokens_per_chunk;
}
constexpr uint32_t page_words(
    uint32_t tokens, uint32_t selected_experts_k, uint32_t experts_per_device, uint32_t tokens_per_chunk) {
    return header_words(experts_per_device) +
           entry_capacity(tokens, selected_experts_k, experts_per_device, tokens_per_chunk);
}

}  // namespace token_list

//-----------------------------------------------------------------------------
// Chunk ownership over R replicated rings (the prefill mode's rings; R = 1 is today's single ring).
//-----------------------------------------------------------------------------
// The chunk stream is expert-major in id order and every role knows the per-expert counts, so each core derives
// the same owner table from the counts alone. An expert with at least R chunks is split round-robin over all rings,
// starting after the previous owner; a smaller expert goes whole to the least-loaded ring, ties to the lowest index,
// never the previous owner (R >= 2). So no run of R consecutive same-owner chunks in feed order: the single drain,
// R chunks ahead of the slowest owner, keeps every ring fed. Nothing here assumes a token count: 2048-row slabs and
// 1-chunk decode experts take the same rule.
namespace rings {

// The ring cores' chunk input buffer (c_0) holds this many chunks, one per "half": two today (the drain runs one chunk
// ahead of the ring), R + 1 with R prefill rings (R chunks ahead of the slowest owner = the owner table's run bound).
// The drain keeps one credit semaphore per half; chunk g lands in, and frees, half g % chunk_halves.
constexpr uint32_t MAX_CHUNK_HALVES = 5;
constexpr uint32_t MAX_RINGS = MAX_CHUNK_HALVES - 1;
constexpr uint32_t chunk_halves(uint32_t rings) { return rings < 2 ? 2u : rings + 1; }
// The feed's chunk slots under the a2a pipeline (the streaming ring, rings < 2): the compute holds chunk c's input
// until W2(c) runs after W0/W1(c + 1), so chunk c + 2 must land in a third slot while c and c + 1 are in flight (with
// two, the feed of c + 2 waited for the rows of c: the pipeline's first measurement, 13.8 us per expert against the
// serial 12.9, 2026-09-26).
// Three under the pipeline: the fourth half (160 KB) does not fit beside the slab-mode server's L1 buffers (measured:
// its chunked prefill's static CBs clash with them by 46 KB), and after dm1's exchange-ahead order the third half
// already keeps the feed off the critical path.
constexpr uint32_t feed_halves(uint32_t rings, bool a2a_pipeline) { return a2a_pipeline ? 3u : chunk_halves(rings); }

template <uint32_t R>
struct ChunkOwners {
    uint32_t load[R] = {};
    uint32_t prev_owner = R;  // no expert yet
    uint32_t first_owner = 0;
    bool split = false;

    // Once per expert in id order (zero-chunk experts included); then owner(c) for c in [0, chunks).
    constexpr void begin_expert(uint32_t chunks) {
        split = false;
        if (chunks == 0) {
            return;
        }
        split = chunks >= R;
        if (split) {
            first_owner = prev_owner == R ? 0u : (prev_owner + 1) % R;
            for (uint32_t r = 0; r < R; ++r) {
                load[r] += chunks / R;
            }
            for (uint32_t c = 0; c < chunks % R; ++c) {
                load[(first_owner + c) % R] += 1;
            }
            prev_owner = (first_owner + chunks - 1) % R;
            return;
        }
        uint32_t best = R;
        for (uint32_t r = 0; r < R; ++r) {
            if (r == prev_owner) {
                continue;
            }
            if (best == R || load[r] < load[best]) {
                best = r;
            }
        }
        if (best == R) {  // R == 1: every expert on the one ring
            best = 0;
        }
        first_owner = best;
        load[best] += chunks;
        prev_owner = best;
    }
    constexpr uint32_t owner(uint32_t chunk) const { return split ? (first_owner + chunk) % R : first_owner; }
};

}  // namespace rings

}  // namespace moe_ring

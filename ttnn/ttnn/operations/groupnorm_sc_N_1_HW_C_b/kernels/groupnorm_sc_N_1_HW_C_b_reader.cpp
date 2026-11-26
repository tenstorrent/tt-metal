// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// groupnorm_sc_N_1_HW_C_b reader (NCRISC; NoC0, NoC1 on the DRAM odd-column cores: program descriptor "NoC pairing").
//
// Once per kernel, after block 0's last pass-1 push: group map and the affine map ring zeroed by NoC writes (one
// tile from MEM_ZEROS, then whole-tile self-copies of it), then the map's one-hot halfword stores. On unit roots,
// MEM_ZEROS writes over cb_group_shift at the first pilot step (only its row 0 is rewritten per block).
// Per block (one unit = image n, channel slice s, rows [row_start, row_start + rows_core)):
//   pilot : image row 0 of the slice -> staging -> cb_pilot_rows (row 0: -k, k the shift of each lane's group;
//           pass 1 accumulates x + (-k)) and, on the unit root, cb_group_shift (lane g = +k of group g, row 0
//           only, pushed at once); read together with chunk 0, built after a pass-1 credit push
//   pass 1: chunks 0..B-1 -> the input ring (TILE: both views, lockstep) or cb_input_rm sticks (RM);
//           compute keeps the last K chunks on the keep view
//   affine: gamma / beta slices -> staging (read issued after the last pass-1 push, ahead of the map build) -> the
//           expand matmul's gamma map (gamma[c] at row g(c), column c's lane) + beta map (row 0 = -beta[c]),
//           cb_gamma_rows (stall shadow of the combine)
//   pass 2: re-read chunks 0..B-K-1 -> the input ring (cb_input_rm for RM)
//   output share (DRAM interleaved): compute packs each pass-2 chunk for whichever DM RISC has a free output slot
//           (the writer's cb_output_tiles preferred, else this RISC's cb_output_b) and names the chunk's pass-2
//           index q on that consumer's sequence CB (cb_out_seq_b here; END closes the block). This RISC writes its
//           share to DRAM on its own NoC: it drains cb_output_b while a pass-2 re-read waits for ring room, then
//           to END after the block's last re-read. Both NoCs carry pass-2 writes, in proportion to how fast each
//           drains (the pass-2 writer on NOC0 was ~6x slower per chunk than on NOC1 in DRAM-congested units).
// Every chunk pushes the nominal T pages (ragged tail: only rows_valid tile-rows transferred).
// SHARDED: the input CB (TILE: the ring, RM: cb_input_rm) is backed on this core's own shard, so a chunk is
// a credit push with no data movement; the pilot comes from the unit root's shard (row 0 of the image).
//
// A slice lane v (0 .. W*32) holds channel s*P + (v mod P), P = MAP_PERIOD: P = W*32 except a sharded RM
// slice narrower than a tile multiple, which compute tilizes as k = W*32/P sticks per tile row.
// C % 32 != 0 (Refinement 6, interleaved): the last slice holds only VC_LAST channels in ceil(VC_LAST / 32) tiles;
// pilot, stick, tile and affine reads and the output writes cover just those (RM: the stick's last vc * elem bytes).
//
// Formats (Refinement 4): the input is bf16 or Float32 (IN_TILE_BYTES; DRAM interleaved only for Float32), gamma /
// beta bf16 or Float32 (AFF_ELEM), ROW_MAJOR or TILE (AFF_TILE: row 0 of each (1, C) tile carries the values, read
// as one 64 B row-0 run of faces 0 / 1 per channel tile). The pilot k is always bf16: a Float32 sample's upper
// halfword (its bf16 truncation, a valid shift). The group map, pilot rows and -1 tile stay bf16; the affine maps
// take gamma / beta's format (cb_gamma_rows page = AMAP_TILE_BYTES).
//
// raw_api rows (op_design.md API Mapping): map / affine-map one-hot stores and pilot row-0 copies have no helper
// (generate_index_tile writes a ramp, fill_tile_column writes all 32 rows); DRAM reads are TensorAccessor.
// The prologue's zero fills are stateful NoC *writes* from MEM_ZEROS (not Noc::async_write_zeros, whose loopback
// reads share the read barrier with the in-flight gamma / beta DRAM read), and the affine maps are 8-word
// load-then-store runs (no helper copies a strided L1 word run).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

namespace {
constexpr uint32_t cb_input_rm = 0;
constexpr uint32_t cb_input_tiles = 2;  // input ring, pass-1 view (alias group A)
constexpr uint32_t cb_input_keep = 3;   // input ring, pass-2 view (same bytes, pushed in lockstep)
constexpr uint32_t cb_group_map = 12;
constexpr uint32_t cb_gamma_rows = 13;  // the expand matmul's in1: gamma map (+ beta map)
constexpr uint32_t cb_affine_staging = 21;
constexpr uint32_t cb_pilot_rows = 22;
constexpr uint32_t cb_group_shift = 24;
constexpr uint32_t cb_rm_stage_in = 25;  // sharded RM: staged partial / image-mixed tile rows (+ k-fill template)
constexpr uint32_t cb_output_b = 27;   // DRAM interleaved: this RISC's share of the pass-2 output
constexpr uint32_t cb_out_seq_b = 29;  // pass-2 index of each chunk compute packed into cb_output_b (16 B pages)
constexpr uint32_t OUT_SEQ_END = 0xFFFFFFFFu;

constexpr uint32_t BF16_TILE_BYTES = 2048;  // the group map (always bf16)
constexpr uint32_t F32_TILE_BYTES = 4096;
constexpr uint32_t BF16_ONE = 0x3F80;
constexpr uint32_t ALIGN = 64;               // DRAM read alignment (BH)
constexpr uint32_t ROW0_READ_BYTES = 64;     // TILE pilot / gamma: one read of a face's row 0 (bf16: rows 0-1)
constexpr uint32_t MAX_LANES = 32;           // group lanes in one tile row
constexpr uint32_t PILOT_SLOT_BYTES = 16;    // sharded TILE pilot: one 16 B read per group lane
// bfloat8_b tile (Refinement 7): a 64 B exponent section - one shared exponent byte per 16-lane face row, face-major
// (face f row r at f * 16 + r) - then one sign | 7-bit magnitude byte per element (hidden bit at bit 6), face-major.
constexpr uint32_t BFP8_EXP_BYTES = 64;
constexpr uint32_t BFP8_FACE_BYTES = 256;

// bfloat8_b element m (sign | magnitude) at shared exponent e -> bf16 bits, exact (<= 7 significant bits): the
// magnitude is normalized to its leading one (tt-metal's host decode, convert_bfp_to_u32).
FORCE_INLINE uint32_t bfp8_to_bf16(uint32_t m, uint32_t e) {
    uint32_t mag = m & 0x7F;
    if (mag == 0) {
        return 0;
    }
    uint32_t sh = 0;
    while ((mag & 0x40) == 0) {
        mag <<= 1;
        ++sh;
    }
    if (e <= sh) {
        return 0;  // below bf16's normal range
    }
    return ((m & 0x80) << 8) | ((e - sh) << 7) | ((mag & 0x3F) << 1);
}

// bf16 bits b -> bfloat8_b element at shared exponent e (>= b's exponent), truncated: |value| = mag * 2^(e - 133).
FORCE_INLINE uint32_t bf16_to_bfp8(uint32_t b, uint32_t e) {
    const uint32_t be = (b >> 7) & 0xFF;
    const uint32_t s = e - be + 1;
    if (be == 0 || s >= 8) {
        return 0;
    }
    const uint32_t mag = (0x80 | (b & 0x7F)) >> s;
    return mag == 0 ? 0 : (((b >> 8) & 0x80) | mag);
}

// Element offset of tile-row-0 lane l (0..31) inside a tile: faces 0 / 1 hold lanes 0..15 / 16..31.
FORCE_INLINE uint32_t row0_lane(uint32_t l) { return (l >> 4) * 256 + (l & 15); }

// Element offset of slice lane v (0 .. W*32) in row 0 of its tile (1024 elements per tile). The lanes of one half
// row (16-aligned run) are contiguous elements.
FORCE_INLINE uint32_t lane_elem(uint32_t v) { return (v >> 5) * 1024 + (v & 16) * 16 + (v & 15); }

// Row-0 lanes [v, v1) <- val, one tight store loop per contiguous half-row run (the shift-rows hot loop:
// NCRISC is in-order, so per-lane index math - not the stores - set its cost). E = 1: one element per lane;
// E = 2 (bf16 lane pairs in one word): v and v1 even.
template <uint32_t E, typename T>
FORCE_INLINE void fill_lanes(volatile tt_l1_ptr T* base, uint32_t v, uint32_t v1, T val) {
    while (v < v1) {
        const uint32_t run_end = ((v | 15) + 1) < v1 ? ((v | 15) + 1) : v1;
        volatile tt_l1_ptr T* p = base + lane_elem(v) / E;
        const uint32_t n = (run_end - v) / E;
        if (n == 16 / E) {
            // A whole half row: straight-line stores (a looped store costs the in-order core ~3 instructions).
#pragma GCC unroll 16
            for (uint32_t i = 0; i < 16 / E; ++i) {
                p[i] = val;
            }
        } else {
            for (uint32_t i = 0; i < n; ++i) {
                p[i] = val;
            }
        }
        v = run_end;
    }
}

// n (<= 16) words src -> dst, each xor x. Whole 8-word runs (a bf16 half row; half of a Float32 one) load every
// word before storing any, so the in-order core overlaps the L1 load latencies instead of paying one per word.
FORCE_INLINE void copy_run(volatile tt_l1_ptr uint32_t* d, volatile tt_l1_ptr uint32_t* s, uint32_t n, uint32_t x) {
    for (; n >= 8; n -= 8, d += 8, s += 8) {
        const uint32_t a0 = s[0], a1 = s[1], a2 = s[2], a3 = s[3], a4 = s[4], a5 = s[5], a6 = s[6], a7 = s[7];
        d[0] = a0 ^ x;
        d[1] = a1 ^ x;
        d[2] = a2 ^ x;
        d[3] = a3 ^ x;
        d[4] = a4 ^ x;
        d[5] = a5 ^ x;
        d[6] = a6 ^ x;
        d[7] = a7 ^ x;
    }
    for (uint32_t i = 0; i < n; ++i) {
        d[i] = s[i] ^ x;
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t W = get_compile_time_arg_val(0);         // Wt_core: channel tiles per block
    constexpr uint32_t CHUNK_HT = get_compile_time_arg_val(1);  // h: tile rows per chunk
    constexpr bool IS_RM = get_compile_time_arg_val(2) != 0;
    constexpr bool HAS_GAMMA = get_compile_time_arg_val(3) != 0;
    constexpr bool HAS_BETA = get_compile_time_arg_val(4) != 0;
    // CT arg 5: R, the gather tile rows (compute / writer only since the ones tiles were dropped).
    constexpr uint32_t CG = get_compile_time_arg_val(6);
    constexpr uint32_t WT_TOTAL = get_compile_time_arg_val(7);
    constexpr uint32_t HT = get_compile_time_arg_val(8);
    constexpr uint32_t GX = get_compile_time_arg_val(9);
    constexpr uint32_t IN_PAGE_BYTES = get_compile_time_arg_val(10);
    constexpr uint32_t P = get_compile_time_arg_val(11);  // MAP_PERIOD: channels per slice (sharded: the shard width)
    constexpr bool SHARDED = get_compile_time_arg_val(12) != 0;
    // Per-chunk zones only when the host found every chunk's zone executions inside the marker budget.
    constexpr bool CHUNK_ZONES = get_compile_time_arg_val(13) != 0;
    // Sharded RM (Refinement 3): partial / image-mixed tile rows are staged here (k-filled) for compute to tilize.
    constexpr bool HAS_STAGE = get_compile_time_arg_val(14) != 0;
    // Sharded, fully resident: the shard's pieces are walked once (no pass-2 re-read credits).
    constexpr bool OPS = get_compile_time_arg_val(15) != 0;
    // Formats (Refinement 4): the input's tile bytes, gamma / beta's element bytes, layout and page bytes.
    constexpr uint32_t IN_TILE_BYTES = get_compile_time_arg_val(16);
    constexpr uint32_t AFF_ELEM = get_compile_time_arg_val(17);
    constexpr bool AFF_TILE = get_compile_time_arg_val(18) != 0;
    constexpr uint32_t AFF_PAGE_BYTES = get_compile_time_arg_val(19);
    // Refinement 5: an image's true row count (RM stick paging). HT = ceil(HW / 32) tile rows per image.
    constexpr uint32_t HW = get_compile_time_arg_val(20);
    // Refinement 6 (c_non_aligned, interleaved): valid channels of the last column slice (P when C % 32 == 0). That
    // slice is partial: VC_LAST channels (whole groups) in ceil(VC_LAST / 32) tiles. Only those are read and written;
    // its pad lanes (RM: never read; TILE: unspecified padding) are lane-masked by compute after the column collapse.
    constexpr uint32_t VC_LAST = get_compile_time_arg_val(21);
    // Refinement 7: bfloat8_b input (TILE, interleaved) / gamma / beta (TILE). Their staged row-0 values (the pilot k,
    // the affine maps' source) are decoded to bf16; the k-fill writes the pad rows in the input's shared-exponent form.
    constexpr bool IN_BFP8 = get_compile_time_arg_val(22) != 0;
    constexpr bool AFF_BFP8 = get_compile_time_arg_val(23) != 0;
    constexpr auto in_args = TensorAccessorArgs<24>();
    constexpr auto gamma_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    constexpr auto beta_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();
    // The output (written for this RISC's pass-2 share; DRAM interleaved only).
    constexpr uint32_t OUT_PAGE_BYTES = get_compile_time_arg_val(beta_args.next_compile_time_args_offset());
    constexpr auto out_args = TensorAccessorArgs<beta_args.next_compile_time_args_offset() + 1>();

    // Input element bytes (bf16 2, Float32 4; bfloat8_b: its 1-byte mantissa, the exponents sit in the tile header).
    constexpr uint32_t IN_ELEM = IN_BFP8 ? 1 : IN_TILE_BYTES / 1024;
    constexpr uint32_t IN_HDR_BYTES = IN_BFP8 ? BFP8_EXP_BYTES : 0;  // ahead of face 0's elements
    constexpr uint32_t IN_FACE_BYTES = IN_BFP8 ? BFP8_FACE_BYTES : IN_TILE_BYTES / 4;
    constexpr uint32_t TILE_SEGMENT_BYTES = 32 * IN_ELEM;  // 32 channels of one stick
    // Pilot staging per input tile (host: Formats.pilot_tile_stage): row-0 runs of faces 0 / 1, + bfloat8_b's
    // exponent section.
    constexpr uint32_t PILOT_TILE_STAGE = (IN_BFP8 ? 3 : 2) * ROW0_READ_BYTES;
    constexpr uint32_t AMAP_TILE_BYTES = 1024 * AFF_ELEM;  // one affine map tile (the map's format, bf16 for bfp8)
    constexpr uint32_t AFF_HDR_BYTES = AFF_BFP8 ? BFP8_EXP_BYTES : 0;
    constexpr uint32_t AFF_FACE_BYTES = AFF_BFP8 ? BFP8_FACE_BYTES : 256 * AFF_ELEM;
    constexpr uint32_t AFF_TILE_STAGE = (AFF_BFP8 ? 3 : 2) * ROW0_READ_BYTES;  // host: Formats.aff_tile_stage
    static_assert(
        IN_BFP8 ? (IN_TILE_BYTES == BFP8_EXP_BYTES + 1024 && !IS_RM && !SHARDED) : (IN_ELEM == 2 || IN_ELEM == 4),
        "bf16 or Float32 input, or bfloat8_b TILE interleaved");
    static_assert(AFF_ELEM == 2 || AFF_ELEM == 4, "bf16 or Float32 affine maps");
    static_assert(!AFF_BFP8 || (AFF_TILE && AFF_ELEM == 2), "bfloat8_b gamma / beta: TILE, decoded into bf16 maps");
    static_assert(
        !SHARDED || (IN_ELEM == 2 && AFF_ELEM == 2 && !AFF_TILE),
        "sharded: bf16 input with bf16 ROW_MAJOR gamma / beta (the op's EXCLUSIONS)");
    constexpr uint32_t T = CHUNK_HT * W;
    // hw_non_aligned (Refinement 5): valid rows of each image's last tile row (0 = whole tile rows). Its pad rows are
    // k-filled in pass 1 (x - k = 0 and x * (x - k) = 0 add nothing to either moment); RM reads only the valid sticks.
    constexpr uint32_t HR = HW % 32;
    static_assert(HT * 32 == HW + (HR ? 32 - HR : 0), "HT = ceil(HW / 32)");
    constexpr uint32_t FACE_ROW_BYTES = 16 * IN_ELEM;  // one face row of an input tile
    constexpr uint32_t SLICE_BYTES = W * TILE_SEGMENT_BYTES;
    // Per-tensor affine staging (host: affine_stage_bytes). RM: the slice's P channels from a 64 B-aligned start;
    // TILE: row 0 of faces 0 / 1 of each of the slice's W channel tiles, 64 B each (+ bfloat8_b's exponents).
    constexpr uint32_t AFFINE_STAGE_BYTES =
        AFF_TILE ? W * AFF_TILE_STAGE : ((P * AFF_ELEM + ALIGN - 1) / ALIGN) * ALIGN + ALIGN;

    uint32_t arg = 0;
    const uint32_t in_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t beta_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t unit_start = get_arg_val<uint32_t>(arg++);
    const uint32_t num_blocks = get_arg_val<uint32_t>(arg++);
    const uint32_t row_start = get_arg_val<uint32_t>(arg++);
    const uint32_t rows_core = get_arg_val<uint32_t>(arg++);
    const uint32_t resident_chunks = get_arg_val<uint32_t>(arg++);
    const uint32_t is_root = get_arg_val<uint32_t>(arg++);
    const uint32_t root_x = get_arg_val<uint32_t>(arg++);  // SHARDED: unit root (virtual), first core row of the band
    const uint32_t root_y = get_arg_val<uint32_t>(arg++);
    const uint32_t out_addr = get_arg_val<uint32_t>(arg++);
    // Sharded slice geometry (interleaved: derived per block below). Lanes count the statistic unit's groups.
    const uint32_t sh_xs = get_arg_val<uint32_t>(arg++);    // first channel of the slice
    const uint32_t sh_vc = get_arg_val<uint32_t>(arg++);    // its valid channels (padded last column: < P)
    const uint32_t sh_l0 = get_arg_val<uint32_t>(arg++);    // group lane of the first channel
    const uint32_t sh_off0 = get_arg_val<uint32_t>(arg++);  // its offset inside that group
    const uint32_t g_unit = get_arg_val<uint32_t>(arg++);   // group lanes of the unit
    const uint32_t rm_credit_pages = get_arg_val<uint32_t>(arg++);
    const uint32_t n_cols = get_arg_val<uint32_t>(arg++);  // the unit's core columns (pilot sources), virtual x
    const uint32_t cols_arg = arg;
    arg += n_cols;
    const uint32_t n_staged = get_arg_val<uint32_t>(arg++);  // RM staged rows: (shard stick, position, sticks)
    const uint32_t staged_arg = arg;

    const auto in_acc = TensorAccessor(in_args, in_addr, IN_PAGE_BYTES);
    [[maybe_unused]] const auto out_acc = TensorAccessor(out_args, out_addr, OUT_PAGE_BYTES);
    [[maybe_unused]] const auto gamma_acc = TensorAccessor(gamma_args, gamma_addr, AFF_PAGE_BYTES);
    [[maybe_unused]] const auto beta_acc = TensorAccessor(beta_args, beta_addr, AFF_PAGE_BYTES);

    const uint32_t num_chunks = (rows_core + CHUNK_HT - 1) / CHUNK_HT;
    const uint32_t K = resident_chunks < num_chunks ? resident_chunks : num_chunks;
    const uint32_t first_resident = num_chunks - K;
    const uint32_t staging = get_write_ptr(cb_affine_staging);
    volatile tt_l1_ptr uint16_t* sv = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(staging);
    // k per group lane (bf16 bits): pilot rows, group shift and (RM staged) the k stick are built from it.
    uint32_t kv[MAX_LANES];

    // Every run of slice lanes [v, ve) inside one half row (16 lanes) whose channels ch.. (relative to the slice's
    // first) belong to one group lane: f(v, ve, lane, ch). Periods of P lanes repeat the slice's vc valid channels
    // (sharded RM: k sticks per virtual row); lanes past vc in a period (a padded last column) are never visited.
    auto for_each_run = [&](uint32_t l0, uint32_t off0, uint32_t vc, auto&& f) {
        for (uint32_t p0 = 0; p0 < W * 32; p0 += P) {
            uint32_t a = 0, lane = l0, gend = CG - off0;
            while (a < vc) {
                const uint32_t b = gend < vc ? gend : vc;
                for (uint32_t v = p0 + a, ve = p0 + b, ch = a; v < ve;) {
                    const uint32_t re = ((v | 15) + 1) < ve ? ((v | 15) + 1) : ve;
                    f(v, re, lane, ch);
                    ch += re - v;
                    v = re;
                }
                a = b;
                gend += CG;
                ++lane;
            }
        }
    };

    // One output chunk at the front of cb_output_b -> DRAM: rows_valid tile rows from image tile row row0g
    // (n * HT + row0), the writer's page walk; the slice's w_blk valid tiles of each W-tile ring row.
    auto write_b_chunk = [&](uint32_t row0g, uint32_t rows_valid, uint32_t wt0, uint32_t w_blk) {
        cb_wait_front(cb_output_b, T);
        const uint32_t src0 = get_read_ptr(cb_output_b);
        for (uint32_t r = 0; r < rows_valid; ++r) {
            const uint32_t tile0 = (row0g + r) * WT_TOTAL + wt0;
            uint32_t src = src0 + r * W * OUT_PAGE_BYTES;
            for (uint32_t wt = 0; wt < w_blk; ++wt) {
                noc_async_write_page(tile0 + wt, out_acc, src);
                src += OUT_PAGE_BYTES;
            }
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_output_b, T);
    };

    // Group map M (W tiles): element (slice lane r, group lane l) = 1.0 iff slice lane r holds a channel of group
    // lane l (padding lanes: no one). Block-invariant (interleaved slices start on group boundaries; a sharded core
    // runs one block): built once per kernel. Compute first reads it at the collapse, so it is built off the
    // critical path, after the last pass-1 chunk push of block 0.
    auto build_map = [&](uint32_t l0, uint32_t off0, uint32_t vc) {
        MaybeDeviceZoneScope("r_map_setup");
        cb_reserve_back(cb_group_map, W);
        // Zeros: one tile from MEM_ZEROS, then whole-tile self-copies of it into the map's other W - 1 tiles and
        // (gamma) the whole affine map ring, whose off-position elements then stay zero across blocks (the one-hot
        // positions are block-invariant and the ring returns to its base). NoC writes, one state per size (each
        // issue programs only the two addresses): the write barrier does not wait on the gamma / beta read.
        const uint32_t map_base = get_write_ptr(cb_group_map);
        noc_async_write_one_packet_set_state(get_noc_addr(map_base), MEM_ZEROS_SIZE);
        for (uint32_t off = 0; off < BF16_TILE_BYTES; off += MEM_ZEROS_SIZE) {
            noc_async_write_one_packet_with_state(MEM_ZEROS_BASE, map_base + off);
        }
        noc_async_write_barrier();
        noc_async_write_one_packet_set_state(get_noc_addr(map_base), BF16_TILE_BYTES);
        for (uint32_t t = 1; t < W; ++t) {
            noc_async_write_one_packet_with_state(map_base, map_base + t * BF16_TILE_BYTES);
        }
        if constexpr (HAS_GAMMA) {
            // The affine maps in 2 KB steps of the zero tile (a Float32 map tile is two of them).
            constexpr uint32_t AMAP_TILES = (HAS_BETA ? 2 : 1) * W;
            cb_reserve_back(cb_gamma_rows, AMAP_TILES);
            const uint32_t am_base = get_write_ptr(cb_gamma_rows);
            for (uint32_t off = 0; off < AMAP_TILES * AMAP_TILE_BYTES; off += BF16_TILE_BYTES) {
                noc_async_write_one_packet_with_state(map_base, am_base + off);
            }
        }
        noc_async_write_barrier();
        volatile tt_l1_ptr uint16_t* map = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(map_base);
        // Row r = v & 31 of tile v >> 5 has its one at column `lane`: per run, one strided store loop (16 elements
        // per face row).
        for_each_run(l0, off0, vc, [&](uint32_t v, uint32_t ve, uint32_t lane, uint32_t) {
            const uint32_t r = v & 31;
            volatile tt_l1_ptr uint16_t* p =
                map + (v >> 5) * 1024 + ((r >> 4) * 2 + (lane >> 4)) * 256 + (r & 15) * 16 + (lane & 15);
            for (uint32_t i = 0, n = ve - v; i < n; ++i) {
                p[i * 16] = BF16_ONE;
            }
        });
        cb_push_back(cb_group_map, W);
    };

    // ---------------- Shift rows from kv: lane v -> k of its group lane ----------------
    // cb_pilot_rows (bf16): -k (sign bit flipped: exact), the srcB operand of pass 1's x + (-k), a row broadcast:
    // the unpacker reads only row 0 (faces 0/1), so rows 1..31 are never zeroed. Padding lanes keep whatever they
    // held (compute masks / skips them). cb_group_shift (root, Float32): +k per group lane in row 0, 0 past the
    // unit's lanes (rows 1..31 stay zero).
    auto build_shift_rows = [&](uint32_t l0, uint32_t off0, uint32_t vc, uint32_t lanes, uint32_t blk) {
        {
            MaybeDeviceZoneScope("r_shift_rows");
            cb_reserve_back(cb_pilot_rows, W);
            const uint32_t pbase = get_write_ptr(cb_pilot_rows);
            for_each_run(l0, off0, vc, [&](uint32_t v, uint32_t ve, uint32_t lane, uint32_t) {
                const uint32_t nk16 = kv[lane] ^ 0x8000u;  // -k
                if constexpr (CG % 2 == 0) {
                    // Even groups: lanes (v, v + 1), v even, share a group and one word of a face row.
                    fill_lanes<2>(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pbase), v, ve, nk16 | (nk16 << 16));
                } else {
                    fill_lanes<1>(
                        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(pbase), v, ve, static_cast<uint16_t>(nk16));
                }
            });
            cb_push_back(cb_pilot_rows, W);  // compute's pass-1 moments walk starts here
        }
        if (is_root) {
            cb_reserve_back(cb_group_shift, 1);
            const uint32_t gs = get_write_ptr(cb_group_shift);
            if (blk == 0) {
                // Compute forms -k on every row as (-1 tile) x [k row; zeros], so rows 1..31 must be zero
                // (once: only row 0 is rewritten per block) and nothing is replicated. NoC writes from
                // MEM_ZEROS, as in build_map (no gamma / beta read is in flight yet).
                noc_async_write_one_packet_set_state(get_noc_addr(gs), MEM_ZEROS_SIZE);
                for (uint32_t off = 0; off < F32_TILE_BYTES; off += MEM_ZEROS_SIZE) {
                    noc_async_write_one_packet_with_state(MEM_ZEROS_BASE, gs + off);
                }
                noc_async_write_barrier();
            }
            volatile tt_l1_ptr uint32_t* gv = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(gs);
            for (uint32_t g = 0; g < 32; ++g) {
                gv[row0_lane(g)] = g < lanes ? kv[g] << 16 : 0;
            }
            cb_push_back(cb_group_shift, 1);
        }
    };

    // gamma / beta slices [xs, xs + vc) -> staging. RM: one read from a 64 B-aligned start (never past the slice's
    // valid channels); TILE (DRAM interleaved: xs a tile multiple): the row-0 run of faces 0 / 1 of each channel
    // tile, 64 B each, staged 128 B per tile.
    auto issue_affine_reads = [&](uint32_t xs, uint32_t vc) {
        if constexpr (HAS_GAMMA) {
            if constexpr (AFF_TILE) {
                // Per tile (AFF_TILE_STAGE): faces 0 / 1 at +0 / +64; bfloat8_b: the exponent section at +128.
                const uint32_t t0 = xs >> 5;
                const uint32_t nt = ((xs & 31) + vc + 31) >> 5;
                for (uint32_t t = 0; t < nt; ++t) {
                    for (uint32_t f = 0; f < (AFF_BFP8 ? 3u : 2u); ++f) {
                        const uint32_t dst = staging + t * AFF_TILE_STAGE + f * ROW0_READ_BYTES;
                        const uint32_t off = f < 2 ? AFF_HDR_BYTES + f * AFF_FACE_BYTES : 0;
                        noc_async_read(gamma_acc.get_noc_addr(t0 + t, off), dst, ROW0_READ_BYTES);
                        if constexpr (HAS_BETA) {
                            noc_async_read(
                                beta_acc.get_noc_addr(t0 + t, off), dst + AFFINE_STAGE_BYTES, ROW0_READ_BYTES);
                        }
                    }
                }
            } else {
                const uint32_t lo = (xs * AFF_ELEM) & ~(ALIGN - 1);
                const uint32_t len = ((xs + vc) * AFF_ELEM - lo + ALIGN - 1) & ~(ALIGN - 1);
                noc_async_read(gamma_acc.get_noc_addr(0, lo), staging, len);
                if constexpr (HAS_BETA) {
                    noc_async_read(beta_acc.get_noc_addr(0, lo), staging + AFFINE_STAGE_BYTES, len);
                }
            }
        }
    };

    // Staged element (AFF_ELEM bytes) index of slice channel xs + ch (issue_affine_reads' layout).
    auto aff_idx = [](uint32_t xs, uint32_t ch) -> uint32_t {
        if constexpr (AFF_TILE) {
            const uint32_t cl = (xs & 31) + ch;
            return ((cl >> 5) * AFF_TILE_STAGE + ((cl >> 4) & 1) * ROW0_READ_BYTES) / AFF_ELEM + (cl & 15);
        } else {
            return ((xs * AFF_ELEM) & (ALIGN - 1)) / AFF_ELEM + ch;
        }
    };

    // ---------------- Affine expand maps ----------------
    // cb_gamma_rows holds the expand matmul's in1, [group lane x channel] per tile (not transposed): W "gamma map"
    // tiles, element (g(c), lane of c) = gamma[c], then (beta) W "beta map" tiles with row 0 = -beta[c]. compute:
    // a = rstd x gmap, b' = [mean*rstd, e] x [gmap; bmap] (e = 1), so b' = mean*rstd*gamma - beta. Adjacent channels
    // of one group are adjacent halfwords of one row, so both maps are word copies of the staged slice (even
    // groups; every Float32 run). Padding lanes stay zero (the ring was zeroed once by build_map).
    auto build_affine = [&](uint32_t xs, uint32_t l0, uint32_t off0, uint32_t vc) {
        if constexpr (HAS_GAMMA) {
            MaybeDeviceZoneScope("r_affine");
            constexpr uint32_t AMAP_TILES = (HAS_BETA ? 2 : 1) * W;
            cb_reserve_back(cb_gamma_rows, AMAP_TILES);
            noc_async_read_barrier();
            if constexpr (AFF_BFP8) {
                // bfloat8_b gamma / beta: each staged face's 16 row-0 mantissas -> 16 bf16 values over the same 64 B
                // run (loaded before the stores), with the face row's shared exponent (+128); the map build below
                // then reads bf16 (AFF_ELEM = 2).
                const uint32_t nt = ((xs & 31) + vc + 31) >> 5;
                for (uint32_t s = 0; s < (HAS_BETA ? 2u : 1u); ++s) {
                    for (uint32_t t = 0; t < nt; ++t) {
                        const uint32_t tb = staging + s * AFFINE_STAGE_BYTES + t * AFF_TILE_STAGE;
                        for (uint32_t f = 0; f < 2; ++f) {
                            const uint32_t e =
                                reinterpret_cast<volatile tt_l1_ptr uint8_t*>(tb + 2 * ROW0_READ_BYTES)[f * 16];
                            volatile tt_l1_ptr uint32_t* w =
                                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tb + f * ROW0_READ_BYTES);
                            const uint32_t m[4] = {w[0], w[1], w[2], w[3]};
                            for (uint32_t i = 0; i < 8; ++i) {
                                const uint32_t q = m[i >> 1] >> ((i & 1) * 16);
                                w[i] = bfp8_to_bf16(q & 0xFF, e) | (bfp8_to_bf16((q >> 8) & 0xFF, e) << 16);
                            }
                        }
                    }
                }
            }
            const uint32_t am = get_write_ptr(cb_gamma_rows);
            if constexpr (AFF_ELEM == 4 || CG % 2 == 0) {
                // Runs of one group inside one half row (<= 16 lanes; bf16: 8 words - the run starts, the skew and
                // CG are even, so a word never straddles a group; Float32: one word per lane): gamma -> row g(c) of
                // the gamma map, -beta -> row 0 of the beta map, both at the run's own face-row positions.
                constexpr uint32_t EPW = 4 / AFF_ELEM;  // elements per word
                constexpr uint32_t NEG = AFF_ELEM == 4 ? 0x80000000u : 0x80008000u;
                volatile tt_l1_ptr uint32_t* gs = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(staging);
                volatile tt_l1_ptr uint32_t* bs =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(staging + AFFINE_STAGE_BYTES);
                volatile tt_l1_ptr uint32_t* gm = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(am);
                volatile tt_l1_ptr uint32_t* bm =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(am + W * AMAP_TILE_BYTES);
                for_each_run(l0, off0, vc, [&](uint32_t v, uint32_t ve, uint32_t lane, uint32_t ch) {
                    const uint32_t nw = (ve - v) / EPW;
                    const uint32_t w = aff_idx(xs, ch) / EPW;
                    // Word of (tile v >> 5, half (v >> 4) & 1, row 0, lane v & 15).
                    const uint32_t base = ((v >> 5) * 1024 + ((v >> 4) & 1) * 256 + (v & 15)) / EPW;
                    copy_run(gm + base + ((lane >> 4) * 512 + (lane & 15) * 16) / EPW, gs + w, nw, 0);
                    if constexpr (HAS_BETA) {
                        copy_run(bm + base, bs + w, nw, NEG);  // -beta
                    }
                });
            } else {
                volatile tt_l1_ptr uint16_t* gs = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(staging);
                volatile tt_l1_ptr uint16_t* bs =
                    reinterpret_cast<volatile tt_l1_ptr uint16_t*>(staging + AFFINE_STAGE_BYTES);
                volatile tt_l1_ptr uint16_t* gm = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(am);
                volatile tt_l1_ptr uint16_t* bm =
                    reinterpret_cast<volatile tt_l1_ptr uint16_t*>(am + W * AMAP_TILE_BYTES);
                for_each_run(l0, off0, vc, [&](uint32_t v, uint32_t ve, uint32_t lane, uint32_t ch) {
                    for (uint32_t i = 0; v + i < ve; ++i) {
                        const uint32_t col = ((v + i) >> 5) * 1024 + row0_lane((v + i) & 31);
                        const uint32_t si = aff_idx(xs, ch + i);
                        gm[col + (lane >> 4) * 512 + (lane & 15) * 16] = gs[si];
                        if constexpr (HAS_BETA) {
                            bm[col] = bs[si] ^ 0x8000;  // -beta
                        }
                    }
                });
            }
            cb_push_back(cb_gamma_rows, AMAP_TILES);
        }
    };

    if constexpr (SHARDED) {
        // ================= Sharded: one block, the core's own shard =================
        // ---------------- Pilot: k of each group lane, x[band row 0, first channel of the group] ----------------
        // Read from the unit core holding the group's first channel (the band's first core row, whose shard row 0 is
        // an image's row 0). Lane l's first channel is l * CG past the unit's first column. The unit root reads
        // every lane (its group shift), others only their own. RM: one read per source column of the 16 B-aligned
        // span of stick 0 holding its lanes' channels (packed from the staging base); TILE: one 16 B slot per lane
        // (tile row 0 of faces 0/1: not contiguous across tiles). kv[l] = the lane's halfword in the staging until
        // the barrier. The pilot gates pass 1's moments (the unpacker waits on it right after its tilize), so the
        // reads go out first and the pass-1 credit push below covers their flight (measured: perf_experiments/
        // r3_ragged: per-lane RM reads cost ~200 ns of issue on a 4-lane core).
        const uint32_t n_own = (sh_off0 + sh_vc + CG - 1) / CG;
        const uint32_t la = is_root ? 0 : sh_l0;
        const uint32_t lb = is_root ? g_unit : sh_l0 + n_own;
        {
            MaybeDeviceZoneScopeIf(CHUNK_ZONES, "r_pilot_issue");
            uint32_t l = la, c = la * CG;
            if constexpr (IS_RM) {
                uint32_t pos = 0;
                while (l < lb) {
                    const uint32_t col = c / P;  // (a group wider than a column skips columns)
                    const uint32_t col_base = col * P;
                    const uint32_t lc = l;
                    for (; l < lb && c < col_base + P; ++l, c += CG) {
                        kv[l] = c - col_base;  // local channel in the source stick
                    }
                    const uint32_t b0 = (kv[lc] * 2) & ~15u;
                    const uint32_t b1 = (kv[l - 1] * 2 + 2 + 15) & ~15u;
                    const uint32_t src_x = get_arg_val<uint32_t>(cols_arg + col);
                    noc_async_read(get_noc_addr(src_x, root_y, in_addr + b0), staging + pos, b1 - b0);
                    for (uint32_t j = lc; j < l; ++j) {
                        kv[j] = (pos + kv[j] * 2 - b0) >> 1;
                    }
                    pos += b1 - b0;
                }
            } else {
                uint32_t col = c / P, col_end = (col + 1) * P, state_col = ~0u;
                for (; l < lb; ++l, c += CG) {
                    while (c >= col_end) {
                        ++col;
                        col_end += P;
                    }
                    const uint32_t cl = c - (col_end - P);  // local channel in the source shard
                    const uint32_t off =
                        (cl >> 5) * IN_TILE_BYTES + ((cl >> 4) & 1) * IN_FACE_BYTES + (cl & 15) * IN_ELEM;
                    kv[l] = l * (PILOT_SLOT_BYTES / 2) + ((off & 15) >> 1);
                    if (col != state_col) {
                        noc_async_read_one_packet_set_state(
                            get_noc_addr(get_arg_val<uint32_t>(cols_arg + col), root_y, in_addr), PILOT_SLOT_BYTES);
                        state_col = col;
                    }
                    noc_async_read_one_packet_with_state(in_addr + (off & ~15u), staging + l * PILOT_SLOT_BYTES);
                }
            }
        }

        // Pass-1 credits: the chunks are resident in this core's shard, so a credit push is all compute needs.
        if constexpr (!IS_RM) {
            cb_reserve_back(cb_input_tiles, num_chunks * T);
            cb_reserve_back(cb_input_keep, num_chunks * T);
            cb_push_back(cb_input_tiles, num_chunks * T);
            cb_push_back(cb_input_keep, num_chunks * T);
        } else {
            const uint32_t pages = OPS ? rm_credit_pages : num_chunks * T;
            cb_reserve_back(cb_input_rm, pages);
            cb_push_back(cb_input_rm, pages);
        }

        {
            MaybeDeviceZoneScopeIf(CHUNK_ZONES, "r_pilot_wait");
            noc_async_read_barrier();
        }
        for (uint32_t l = la; l < lb; ++l) {
            kv[l] = sv[kv[l]];
        }
        build_shift_rows(sh_l0, sh_off0, sh_vc, g_unit, 0);

        if constexpr (HAS_STAGE) {
            // ---------------- Staged tile rows (RM): partial tail / image-mixed rows ----------------
            // A staged row = the segment's sticks of one tile-row position, every other stick the k stick (x - k = 0
            // and x * (x - k) = 0: contributes nothing to the moments). The k stick (channel c -> k of its group, 0 on
            // padding channels) is built over the pilot slots, doubled into a full template row after the staged
            // rows, then each staged row = template prefix + shard sticks + template suffix (local NoC copies).
            MaybeDeviceZoneScope("r_stage_rows");
            constexpr uint32_t STICK = P * IN_ELEM;
            constexpr uint32_t TR_BYTES = W * IN_TILE_BYTES;  // 32k sticks
            {
                uint32_t a = 0, lane = sh_l0, gend = CG - sh_off0;
                while (a < sh_vc) {
                    const uint32_t b = gend < sh_vc ? gend : sh_vc;
                    for (uint32_t ch = a; ch < b; ++ch) {
                        sv[ch] = kv[lane];
                    }
                    a = b;
                    gend += CG;
                    ++lane;
                }
                for (uint32_t ch = sh_vc; ch < P; ++ch) {
                    sv[ch] = 0;
                }
            }
            cb_reserve_back(cb_rm_stage_in, n_staged * W);
            const uint32_t stage_base = get_write_ptr(cb_rm_stage_in);
            const uint32_t tmpl = stage_base + n_staged * TR_BYTES;
            noc_async_write(staging, get_noc_addr(tmpl), STICK);
            noc_async_write_barrier();
            for (uint32_t m = STICK; m < TR_BYTES; m *= 2) {
                noc_async_write(tmpl, get_noc_addr(tmpl + m), m < TR_BYTES - m ? m : TR_BYTES - m);
                noc_async_write_barrier();
            }
            for (uint32_t i = 0; i < n_staged; ++i) {
                const uint32_t src = get_arg_val<uint32_t>(staged_arg + 3 * i) * STICK;
                const uint32_t pos = get_arg_val<uint32_t>(staged_arg + 3 * i + 1) * STICK;
                const uint32_t len = get_arg_val<uint32_t>(staged_arg + 3 * i + 2) * STICK;
                const uint32_t dst = stage_base + i * TR_BYTES;
                if (pos > 0) {
                    noc_async_write(tmpl, get_noc_addr(dst), pos);
                }
                // The shard's sticks: a read of this core's own L1 by NoC address, so the transfer's local side is
                // the stage row (the shard tail past its whole tile rows lies beyond the shard-backed CB).
                noc_async_read(get_noc_addr(in_addr + src), dst + pos, len);
                if (pos + len < TR_BYTES) {
                    noc_async_write(tmpl + pos + len, get_noc_addr(dst + pos + len), TR_BYTES - pos - len);
                }
            }
            noc_async_write_barrier();
            noc_async_read_barrier();
            cb_push_back(cb_rm_stage_in, n_staged * W);
        }

        // After the pilot work (which frees the staging): the gamma / beta slices' DRAM reads land behind the map
        // build instead of in front of the affine build.
        issue_affine_reads(sh_xs, sh_vc);
        build_map(sh_l0, sh_off0, sh_vc);
        build_affine(sh_xs, sh_l0, sh_off0, sh_vc);

        if constexpr (IS_RM && !OPS) {
            // Pass-2 re-read credits (partial residency): the shard FIFO wraps to the non-resident head; each
            // reserve waits for compute to free the ring slot.
            for (uint32_t j = 0; j < first_resident; ++j) {
                cb_reserve_back(cb_input_rm, T);
                cb_push_back(cb_input_rm, T);
            }
        }
        return;
    }

    // k of group lane l < g_blk (the slice's groups; slices start on group boundaries) = the staged pilot sample of
    // its first channel, as bf16: a Float32 sample's upper halfword. TILE staging holds 64 B of row 0 of faces 0 / 1
    // per tile. Lanes >= g_blk are never read: the pilot / k rows walk only the slice's vc valid channels and the
    // group shift writes k = 0 there (a partial last slice's spare lanes see zero sums, so finite statistics).
    // bfloat8_b: the sample decoded with its face row's shared exponent (exact in bf16). With HR != 0 every k is then
    // truncated to one common exponent k_exp (the largest), so the k-fill can write any pad face row - whose 16 lanes
    // may span several groups - with a single shared exponent and every k exactly; k is only a shift, so the
    // quantized k' is used everywhere k is (pilot rows, group shift, pad rows) and the moments stay exact.
    uint32_t k_exp = 0;
    auto extract_kv = [&](uint32_t g_blk) {
        for (uint32_t l = 0; l < g_blk; ++l) {
            const uint32_t c = l * CG;
            if constexpr (IN_BFP8) {
                const uint32_t tb = staging + (c >> 5) * PILOT_TILE_STAGE;
                const uint32_t f = (c >> 4) & 1;
                volatile tt_l1_ptr uint8_t* b8 = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(tb);
                kv[l] = bfp8_to_bf16(b8[f * ROW0_READ_BYTES + (c & 15)], b8[2 * ROW0_READ_BYTES + f * 16]);
            } else {
                const uint32_t byte =
                    IS_RM ? c * IN_ELEM
                          : (c >> 5) * PILOT_TILE_STAGE + ((c >> 4) & 1) * ROW0_READ_BYTES + (c & 15) * IN_ELEM;
                kv[l] = sv[(byte + IN_ELEM - 2) >> 1];
            }
        }
        if constexpr (IN_BFP8 && HR != 0) {
            k_exp = 0;
            for (uint32_t l = 0; l < g_blk; ++l) {
                const uint32_t e = (kv[l] >> 7) & 0xFF;
                k_exp = e > k_exp ? e : k_exp;
            }
            for (uint32_t l = 0; l < g_blk; ++l) {
                kv[l] = bfp8_to_bf16(bf16_to_bfp8(kv[l], k_exp), k_exp);
            }
        }
    };

    // hw_non_aligned (Refinement 5): the pass-1 copy of an image's last tile row (ring tile row `lr` of the chunk at
    // `base`) gets k in its pad rows HR..31, so they add nothing to S1' = Sigma (x - k) or T2 = Sigma x * (x - k) and
    // n stays HW * Cg (TILE padding is unspecified; RM never read those sticks). The same mechanism as the sharded RM
    // staged rows (Refinement 3); the moments program itself is unchanged. The k row (lane v -> k of its group, in
    // the input's format) is built in the staging, free between the pilot (step 0) and the affine reads issued after
    // the last pass-1 push; local NoC writes then copy it over the pad: RM, one per pad stick; TILE, one per pad face
    // row (16 lanes) per tile. The writes complete before the chunk's push.
    [[maybe_unused]] auto k_fill_last_row = [&](uint32_t base, uint32_t lr, uint32_t vc, uint32_t w_blk) {
        MaybeDeviceZoneScope("r_k_fill");
        for_each_run(0, 0, vc, [&](uint32_t v, uint32_t ve, uint32_t lane, uint32_t) {
            if constexpr (IN_BFP8) {
                // Mantissa bytes at the common exponent k_exp (exact: kv was truncated to it by extract_kv).
                volatile tt_l1_ptr uint8_t* kr = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(staging);
                const uint8_t kb = static_cast<uint8_t>(bf16_to_bfp8(kv[lane], k_exp));
                for (uint32_t i = v; i < ve; ++i) {
                    kr[i] = kb;
                }
            } else if constexpr (IN_ELEM == 4) {
                volatile tt_l1_ptr uint32_t* kr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(staging);
                for (uint32_t i = v; i < ve; ++i) {
                    kr[i] = kv[lane] << 16;
                }
            } else {
                for (uint32_t i = v; i < ve; ++i) {
                    sv[i] = static_cast<uint16_t>(kv[lane]);
                }
            }
        });
        if constexpr (IS_RM) {
            for (uint32_t s = HR; s < 32; ++s) {
                noc_async_write(staging, get_noc_addr(base + (lr * 32 + s) * SLICE_BYTES), SLICE_BYTES);
            }
        } else {
            noc_async_write_one_packet_set_state(get_noc_addr(base), FACE_ROW_BYTES);
            for (uint32_t wt = 0; wt < w_blk; ++wt) {
                const uint32_t tile = base + (lr * W + wt) * IN_TILE_BYTES;
                for (uint32_t r = HR; r < 32; ++r) {
                    const uint32_t row = tile + IN_HDR_BYTES + (r >> 4) * 2 * IN_FACE_BYTES + (r & 15) * FACE_ROW_BYTES;
                    noc_async_write_one_packet_with_state(staging + wt * 32 * IN_ELEM, row);
                    noc_async_write_one_packet_with_state(staging + (wt * 32 + 16) * IN_ELEM, row + IN_FACE_BYTES);
                    if constexpr (IN_BFP8) {
                        // The two pad face rows' shared exponents (bytes of the header: no NoC-aligned run).
                        volatile tt_l1_ptr uint8_t* ex = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(tile);
                        const uint32_t f0 = (r >> 4) * 2;
                        ex[f0 * 16 + (r & 15)] = static_cast<uint8_t>(k_exp);
                        ex[(f0 + 1) * 16 + (r & 15)] = static_cast<uint8_t>(k_exp);
                    }
                }
            }
        }
        noc_async_write_barrier();
    };

    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        MaybeDeviceZoneScope("r_block");
        const uint32_t unit = unit_start + blk;
        const uint32_t n = unit / GX;
        const uint32_t c_start = (unit % GX) * P;  // first channel of the slice
        const uint32_t wt0 = c_start / 32;         // its tile column (P = W*32 whenever the input is tiled in DRAM)
        // Refinement 6: the last slice of a C % 32 != 0 input holds VC_LAST channels (g_blk whole groups) in w_blk
        // tiles; every other slice is P channels, W tiles.
        const uint32_t vc = unit % GX == GX - 1 ? VC_LAST : P;
        const uint32_t w_blk = (vc + 31) / 32;
        const uint32_t g_blk = vc / CG;
        // Output share (DRAM interleaved).
        bool b_done = false;
        // One cb_output_b chunk if its index and data are both there (block: wait for them); END ends the share.
        auto try_b = [&](bool block) {
            if (!block && !cb_pages_available_at_front(cb_out_seq_b, 1)) {
                return;
            }
            cb_wait_front(cb_out_seq_b, 1);
            invalidate_l1_cache();
            const uint32_t q = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_out_seq_b));
            if (q == OUT_SEQ_END) {
                cb_pop_front(cb_out_seq_b, 1);
                b_done = true;
                return;
            }
            if (!block && !cb_pages_available_at_front(cb_output_b, T)) {
                return;
            }
            cb_pop_front(cb_out_seq_b, 1);
            const uint32_t j = q < K ? first_resident + q : q - K;  // pass-2 order: resident tail, then the head
            const uint32_t rows_left = rows_core - j * CHUNK_HT;
            write_b_chunk(n * HT + row_start + j * CHUNK_HT, rows_left < CHUNK_HT ? rows_left : CHUNK_HT, wt0, w_blk);
        };

        // One loop, one site per NoC pattern (kernel text must fit the kernel-config ring under --dev):
        // steps 0..B-1 are pass 1, step B emits the affine rows (stall shadow of the combine), and steps
        // B+1..B+first_resident re-read the non-resident head for pass 2.
        for (uint32_t step = 0; step <= num_chunks + first_resident; ++step) {
            if (step == num_chunks) {
                build_affine(c_start, 0, 0, vc);
                continue;
            }

            // ---------------- One chunk (nominal T pages) ----------------
            const uint32_t j = step < num_chunks ? step : step - num_chunks - 1;
            const uint32_t row0 = row_start + j * CHUNK_HT;
            const uint32_t rows_left = rows_core - j * CHUNK_HT;
            const uint32_t rows_valid = rows_left < CHUNK_HT ? rows_left : CHUNK_HT;

            if (step == 0) {
                // Pilot source (image row 0 of the slice) -> staging; its barrier is the chunk's.
                // TILE: row 0 of faces 0 / 1 of each tile (64 B reads: bf16 rows 0-1, Float32 row 0), staged 128 B
                // per tile. RM: the slice of stick 0, staged contiguously.
                if constexpr (IS_RM) {
                    noc_async_read(in_acc.get_noc_addr(n * HW, wt0 * TILE_SEGMENT_BYTES), staging, vc * IN_ELEM);
                } else {
                    // Per tile (PILOT_TILE_STAGE): faces 0 / 1 at +0 / +64; bfloat8_b: the exponent section at +128.
                    for (uint32_t wt = 0; wt < w_blk; ++wt) {
                        for (uint32_t f = 0; f < (IN_BFP8 ? 3u : 2u); ++f) {
                            const uint32_t off = f < 2 ? IN_HDR_BYTES + f * IN_FACE_BYTES : 0;
                            const uint64_t src = in_acc.get_noc_addr(n * HT * WT_TOTAL + wt0 + wt, off);
                            noc_async_read(src, staging + wt * PILOT_TILE_STAGE + f * ROW0_READ_BYTES, ROW0_READ_BYTES);
                        }
                    }
                }
            }

            {
                // Back-pressure: compute has not freed the ring slot yet.
                MaybeDeviceZoneScopeIf(CHUNK_ZONES, "r_reserve");
                // Pass-2 re-read: compute may be blocked on cb_output_b (this RISC's share), so write the share
                // out while the ring has no room for the next chunk.
                if (step > num_chunks) {
                    for (;;) {
                        const bool room = IS_RM ? cb_pages_reservable_at_back(cb_input_rm, T)
                                                : cb_pages_reservable_at_back(cb_input_tiles, T) &&
                                                      cb_pages_reservable_at_back(cb_input_keep, T);
                        if (room) {
                            break;
                        }
                        if (!b_done) {
                            try_b(false);
                        }
                    }
                }
                if constexpr (IS_RM) {
                    cb_reserve_back(cb_input_rm, T);
                } else {
                    // Both views must have room: the keep view still holds the resident tail until pass 2.
                    cb_reserve_back(cb_input_tiles, T);
                    cb_reserve_back(cb_input_keep, T);
                }
            }
            const uint32_t chunk_base = IS_RM ? get_write_ptr(cb_input_rm) : get_write_ptr(cb_input_tiles);
            if constexpr (IS_RM) {
                MaybeDeviceZoneScopeIf(CHUNK_ZONES, "r_issue");
                uint32_t page = n * HW + row0 * 32;
                uint32_t dst = chunk_base;
                // The image's last tile row holds HR valid sticks (the sticks after it are the next image's).
                const uint32_t sticks = rows_valid * 32 - (HR != 0 && row0 + rows_valid == HT ? 32 - HR : 0);
                // The slice's valid channels of each stick (a partial last slice ends at the stick's end, C channels).
                const uint32_t seg_bytes = vc * IN_ELEM;
                for (uint32_t s = 0; s < sticks; ++s) {
                    noc_async_read(in_acc.get_noc_addr(page, wt0 * TILE_SEGMENT_BYTES), dst, seg_bytes);
                    ++page;
                    dst += SLICE_BYTES;
                }
            } else {
                MaybeDeviceZoneScopeIf(CHUNK_ZONES, "r_issue");
                for (uint32_t r = 0; r < rows_valid; ++r) {
                    const uint32_t tile0 = (n * HT + row0 + r) * WT_TOTAL + wt0;
                    uint32_t dst = chunk_base + r * W * IN_TILE_BYTES;
                    for (uint32_t wt = 0; wt < w_blk; ++wt) {
                        noc_async_read_page(tile0 + wt, in_acc, dst);
                        dst += IN_TILE_BYTES;
                    }
                }
            }

            {
                MaybeDeviceZoneScopeIf(CHUNK_ZONES, "r_barrier");
                noc_async_read_barrier();
            }

            if constexpr (HR != 0) {
                if (step == 0) {
                    extract_kv(g_blk);  // before the push: chunk 0 may hold the image's last tile row
                }
                if (step < num_chunks && row0 + rows_valid == HT) {
                    k_fill_last_row(chunk_base, rows_valid - 1, vc, w_blk);
                }
            }

            if constexpr (IS_RM) {
                cb_push_back(cb_input_rm, T);
            } else {
                cb_push_back(cb_input_tiles, T);
                cb_push_back(cb_input_keep, T);
            }

            if (step == 0) {
                if constexpr (HR == 0) {
                    extract_kv(g_blk);
                }
                build_shift_rows(0, 0, vc, g_blk, blk);
            }
            if (step == num_chunks - 1) {
                // After the last pass-1 push and the pilot work (which frees the staging): the gamma / beta
                // slices' DRAM reads (one hot bank: every core of a slice reads the same page) land behind the
                // map build instead of in front of the affine build.
                issue_affine_reads(c_start, vc);
                if (blk == 0) {
                    // After the last pass-1 push (compute first reads the map after reducing this chunk). Its zero
                    // fill is NoC writes, so the gamma / beta reads just issued do not hold it. Built over all P
                    // lanes (block-invariant): a partial last slice's pad lanes are zeroed before the collapse.
                    build_map(0, 0, P);
                }
            }
        }
        // The rest of this block's output share: whatever compute packs for this RISC after the last re-read was
        // issued (every chunk of the share when nothing is re-read), up to END.
        while (!b_done) {
            try_b(true);
        }
    }
    noc_async_write_barrier();
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// groupnorm_sc_N_1_HW_C — reader (NoC0).
//
// Per core: builds the row-0-valid gamma/beta tiles for this core's K channel tiles once
// (cb_scaler is allocated but never written — compute's AccumulateViaAdd reduce never reads it), then makes the core's
// `core_hw_tiles x K` tile rectangle of every image available to compute:
//   interleaved, resident : TensorAccessor pages, once per image (all three passes re-read L1)
//   interleaved, streaming: once per pass, in Q-tile-row chunks (three passes; TWO when `two_pass`:
//                           compute takes both statistics from one read of each chunk)
//   block-sharded, TILE   : the shard IS the block and already sits in L1 — cb_input_tiles is
//                           placed on the shard buffer, so the reader only pushes the Hs*K
//                           credits per image (zero-copy: no NoC read of the local shard)
//   block-sharded, RM     : the shard is stick pages of shard_w elements; the tilize needs
//                           32-stick blocks at a K*64 B stride, so each tile-row's valid sticks
//                           are staged L1 -> L1 into cb_input_sticks (pad lanes / pad sticks
//                           zeroed) and compute tilizes them once into the resident tiled CB
//   block-sharded, RM, direct view: the shard IS the row-major block of width
//                           lcm(shard_w, 32) (a [2048,40] shard is a [512,160] block), placed as
//                           cb_input_sticks with tile pages — credits only, no staging; block lane j
//                           is channel c0 + j % c_period (membership rows, gamma/beta rows)
// TILE inputs are moved as tile pages (raw TensorAccessor reads — no dataflow
// helper wraps tile-page moves); interleaved ROW_MAJOR inputs go through
// read_sticks_for_tilize into cb_input_sticks and are tilized by compute.
//
// Non-tile-aligned shapes (interleaved): the core's channel block is K tiles with
// c_valid <= K*32 valid lanes (the last column block of a C % 32 != 0 tensor), and the image's
// last tile-row holds row_hi <= 32 valid sticks (HW % 32 != 0). TILE pages already carry zero
// padding; the ROW_MAJOR reader reads exactly c_valid*elem bytes of each of the sb valid sticks
// into a stick CB whose pad lanes were zeroed once (nothing else ever writes them) and whose pad
// sticks are re-zeroed before a partial tile-row — so pass 1 sums zeros and pass 2's masked mean
// (writer / compute) gives (0 - 0)^2 = 0 on them. In the streaming regime the pass-2 chunk
// sequence follows compute's [head][body][tail] segments (geometry.hpp pass2_segments).
//
// CB quantum contract (shared with compute/writer): every streaming chunk is
// a NOMINAL Q*K pages (a ragged tail chunk of q < Q rows still pushes Q*K —
// the last (Q-q)*K pages carry no data and are popped unread by compute), so
// the CB write pointer only ever advances in multiples of Q*K and no linear
// block ever straddles the ring boundary. In the resident regime the block is
// padded to Hmax*K per image for the same reason.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"
#include "ttnn/ttnn/operations/groupnorm_sc_N_1_HW_C/kernels/groupnorm_sc_N_1_HW_C_geometry.hpp"

namespace {

// DRAM -> L1 reads must keep the source's residue modulo the DRAM read alignment (64 B on
// Blackhole, 32 B on Wormhole; 64 is the common superset).
constexpr uint32_t DRAM_READ_ALIGN = 64;

// ONE NoC read body for the (cold, once-per-program) affine fill — kernel-config ring budget. Every read
// here is <= one face row (<= 128 B), far under NOC_MAX_BURST_SIZE, so it takes the one-packet path (the
// generic-length noc_async_read is ~4x the code in the --dev build); noinline pins the small body once.
// The hot loops (interleaved DRAM tiles, L1 -> L1 stick staging) keep inline reads.
__attribute__((noinline, noclone)) void noc_read_bytes(uint64_t src, uint32_t dst, uint32_t bytes) {
    ASSERT(bytes <= NOC_MAX_BURST_SIZE);
    noc_async_read<NOC_MAX_BURST_SIZE>(src, dst, bytes);
}

// Bfp8_b tile byte layout: a 64 B exponent header (one shared exponent per 16-lane face row, index
// face*16 + row), then the 1 B mantissas face-major (64 + face*256 + row*16 + col).
constexpr uint32_t BFP8_EXP_BYTES = 64;

// copy_affine_run: one run of `nv` consecutive source lanes -> row 0 lanes [l, l + nv) of the
// destination tile (`elem` bytes per destination lane: the rows CB's element). Two source shapes:
//   plain (bf16 / fp32 weights): nv*elem contiguous bytes, read with ONE NoC read into the tile's
//     face-2/3 scratch at the source's 64 B residue and moved into place with whole-word stores. An
//     odd bf16 lane count (C = 17, 47, ...) ends mid-word; the word's other half is scratch the read
//     never touched: it is masked to zero — exactly what pad lane l + nv must hold. Runs start on even lanes for
//     bf16 (c0 and c_period are even — the host requires c_period*elem % 4 == 0 and every shard width
//     is a multiple of 8 — and a 16-lane face split is even), so every word store is word-aligned;
//     fp32 lanes are always word-aligned.
//   bf8b TILE weights: `nv` mantissa bytes at `src` and the face row's shared exponent
//     byte at `exp_src`; each lane is decoded to bf16 bits (sign | exponent | normalized 7-bit
//     mantissa, the host's convert_bfp_to_u32 truncated to bf16 — exact, bf8b has a 7-bit mantissa)
//     and stored as a halfword. The rows CB is bf16 for bf8b weights (elem == 2).
// Faces 2/3 (rows 16..31) are scratch and stay whatever they were: only row 0 of a rows tile is ever consumed.
// noinline + noclone keeps this body out of fill_affine_rows (code size: the K = 2 staged RM hw_mask
// shards sit close to the kernel-config ring in the --dev build). The bf8b decode is compiled only into
// programs with bf8b weights for the same reason.
template <bool bf8_src>
__attribute__((noinline, noclone)) void copy_affine_run(
    uint64_t src, uint64_t exp_src, uint32_t tile, uint32_t scratch, uint32_t l, uint32_t nv, uint32_t elem) {
    const uint32_t half_row_bytes = 16 * elem;  // 16 lanes of one face row
    const uint32_t face_bytes = 256 * elem;
    // L1 address of byte `lane_byte` of row 0: face 0 (lanes 0..15) or face 1 (lanes 16..31).
    auto lane_addr = [&](uint32_t lane_byte) {
        return lane_byte < half_row_bytes ? tile + lane_byte : tile + face_bytes + (lane_byte - half_row_bytes);
    };
    if constexpr (!bf8_src) {
        const uint32_t dst = scratch + ((static_cast<uint32_t>(src) - scratch) & (DRAM_READ_ALIGN - 1));
        noc_read_bytes(src, dst, nv * elem);
        noc_async_read_barrier();
        volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
        const uint32_t bytes = nv * elem;
        const uint32_t words = (bytes + 3) / 4;
        for (uint32_t w = 0; w < words; ++w) {
            uint32_t v = s[w];
            if (w + 1 == words && (bytes & 3) != 0) {
                v &= 0xFFFFu;  // odd bf16 lane count: the word's high half is scratch, the pad lane must read 0
            }
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(lane_addr(l * elem + 4 * w)) = v;
        }
    } else {
        // (An explicit else: code after a `return` inside `if constexpr` is still compiled.)
        // bf8b: the mantissas land in the scratch's first 64 B residue window, the exponent in the next.
        const uint32_t mdst = scratch + ((static_cast<uint32_t>(src) - scratch) & (DRAM_READ_ALIGN - 1));
        const uint32_t edst =
            scratch + DRAM_READ_ALIGN + ((static_cast<uint32_t>(exp_src) - scratch) & (DRAM_READ_ALIGN - 1));
        noc_read_bytes(src, mdst, nv);
        noc_read_bytes(exp_src, edst, 1);
        noc_async_read_barrier();
        volatile tt_l1_ptr uint8_t* m = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(mdst);
        volatile tt_l1_ptr uint8_t* e = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(edst);
        const uint32_t shared_exp = *e;
        for (uint32_t i = 0; i < nv; ++i) {
            const uint32_t d = m[i];
            uint32_t man = d & 0x7f;
            uint32_t bits = 0;
            if (man != 0) {
                uint32_t ex = shared_exp;
                while ((man & 0x40) == 0) {  // normalize: leading 1 into bit 6 ...
                    man <<= 1;
                    --ex;
                }
                man = (man << 1) & 0x7f;  // ... then drop it (the hidden bit)
                bits = ((d >> 7) << 15) | (ex << 7) | man;
            }
            *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(lane_addr((l + i) * elem)) = bits;
        }
    }
}

// Build K row-0-valid weight tiles: block lane j = 32T + l holds the weight of channel
// c0 + (j % c_period) when (j % c_period) < c_valid, else zero — in face-0 row 0 (lanes 0..15) and
// face-1 row 0 (lanes 16..31); the rest of row 0 is zero, rows 1..31 are unspecified (never consumed).
// c_period >= K*32 is the identity map
// (lane j = channel c0 + j, the interleaved / staged paths: one run per tile); c_period = shard_w on
// the RM direct view, where a tile spans up to ceil(32/c_period) + 1 runs of consecutive channels.
// Source addressing through `pages`, the NoC addresses the
// caller resolved with the weight's TensorAccessor (one accessor construction and one get_noc_addr
// site in the kernel, whatever the weight count — kernel-config ring budget):
//   ROW_MAJOR weight: pages[0] = the (1,1,1,C) stick page; channel c is at byte c*elem.
//   TILE weight     : pages[t] = tile page tile0 + t (tile0 = c0/32; the core's channels span at most
//                     K + 1 tiles); channel c is row 0, lane c%16 of face (c%32)/16 of tile c/32 (a
//                     (1,1,1,C) TILE tensor is padded to 32 rows with zeros). A run is split at the
//                     16-lane face rows so every NoC read stays contiguous. bf8b tiles are decoded
//                     lane by lane into a bf16 rows CB — the same body serves every geometry
//                     (periodic c_period, c0 not a tile multiple, c_valid-clipped last block).
// noinline+noclone, one instantiation per program (gamma and beta share it).
template <bool tile_src, bool bf8_src>
__attribute__((noinline, noclone)) void fill_affine_rows(
    const uint64_t* pages,
    uint32_t tile0,
    uint32_t cb_rows,
    uint32_t tile_bytes,
    uint32_t K,
    uint32_t c0,
    uint32_t c_valid,
    uint32_t c_period) {
    const uint32_t elem = tile_bytes / 1024;  // the rows CB's element (== the weight's, bf16 for bf8b)
    const uint32_t face_bytes = 256 * elem;

    CircularBuffer rows(cb_rows);
    rows.reserve_back(K);
    const uint32_t base = rows.get_write_ptr();
    // Only row 0 (faces 0 and 1) of a rows tile is ever consumed — scale_rows_block / shift_full_block feed
    // BroadcastDim::Row and unary_bcast<Row> — so only those 32 lanes are zeroed (4*elem word stores per face
    // row, not a NoC tile fill, which is much larger code in the --dev build); rows 1..31 are unspecified.
    for (uint32_t T = 0; T < K; ++T) {
        const uint32_t tile = base + T * tile_bytes;
        volatile tt_l1_ptr uint32_t* r0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile);
        volatile tt_l1_ptr uint32_t* r1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile + face_bytes);
        for (uint32_t w = 0; w < 4 * elem; ++w) {
            r0[w] = 0;
            r1[w] = 0;
        }
    }
    for (uint32_t T = 0; T < K; ++T) {
        const uint32_t tile = base + T * tile_bytes;
        const uint32_t scratch = tile + 2 * face_bytes;
        for (uint32_t l = 0; l < 32;) {
            const uint32_t p = (32 * T + l) % c_period;  // channel offset inside the period
            const uint32_t run = (c_period - p) < (32 - l) ? (c_period - p) : (32 - l);
            if (p >= c_valid) {
                l += run;  // pad lanes of this period: stay zero
                continue;
            }
            uint32_t nv = (c_valid - p) < run ? (c_valid - p) : run;
            const uint32_t c = c0 + p;
            uint64_t src;
            uint64_t exp_src = 0;
            if constexpr (tile_src) {
                nv = nv < (16 - (c & 15)) ? nv : (16 - (c & 15));  // stay inside one face row
                const uint64_t page = pages[(c >> 5) - tile0];
                const uint32_t face = (c >> 4) & 1;
                if constexpr (bf8_src) {
                    src = page + BFP8_EXP_BYTES + face * 256 + (c & 15);
                    exp_src = page + face * 16;
                } else {
                    src = page + face * face_bytes + (c & 15) * elem;
                }
            } else {
                src = pages[0] + c * elem;
            }
            copy_affine_run<bf8_src>(src, exp_src, tile, scratch, l, nv, elem);
            l += nv;
        }
    }
    rows.push_back(K);
}

}  // namespace

void kernel_main() {
    // ---- compile-time knobs / CB ids ------------------------------------
    constexpr bool is_rm = get_compile_time_arg_val(0) == 1;
    constexpr bool input_resident = get_compile_time_arg_val(1) == 1;
    constexpr uint32_t K = get_compile_time_arg_val(2);  // block_c_tiles
    constexpr uint32_t Q = get_compile_time_arg_val(3);  // chunk_hw_tiles
    constexpr bool has_gamma = get_compile_time_arg_val(4) == 1;
    constexpr bool has_beta = get_compile_time_arg_val(5) == 1;
    constexpr bool sharded = get_compile_time_arg_val(6) == 1;
    // hw_mask: some tile-row is only partly inside its image -> compute runs pass 2 in
    // [head][body][tail] segments; a streaming TILE reader must chunk pass 2 the same way.
    constexpr bool hw_mask = get_compile_time_arg_val(7) == 1;
    // rm_direct (host-derived): a ROW_MAJOR block shard consumed IN PLACE as the row-major
    // block of width lcm(shard_w, 32) = K*32 elements (m sticks per block row): the shard itself is the
    // tilize's input CB and the output shard the untilize's output CB (tile-sized pages), and block
    // lane j is channel c0 + j % c_period. No stick staging, no stick write-back.
    constexpr bool rm_direct = get_compile_time_arg_val(8) == 1;
    // two_pass (host-derived): a streaming program computes both statistics from ONE read of
    // each chunk (pass A) and applies in pass B — two input passes instead of three.
    constexpr bool two_pass = get_compile_time_arg_val(9) == 1;
    constexpr uint32_t cb_input_tiles = get_compile_time_arg_val(10);
    constexpr uint32_t cb_input_sticks = get_compile_time_arg_val(11);
    [[maybe_unused]] constexpr uint32_t cb_scaler = get_compile_time_arg_val(12);
    constexpr uint32_t cb_gamma_rows = get_compile_time_arg_val(13);
    constexpr uint32_t cb_beta_rows = get_compile_time_arg_val(14);
    [[maybe_unused]] constexpr uint32_t cb_input_shard = get_compile_time_arg_val(15);
    // gamma/beta layout: TILE weights are lane-gathered from row 0 of their tile pages
    // (bf8b tiles decoded to bf16); ROW_MAJOR weights are stick slices. Reader-only knobs.
    constexpr bool affine_tile = get_compile_time_arg_val(16) == 1;
    constexpr bool affine_bf8 = get_compile_time_arg_val(17) == 1;
    constexpr uint32_t accessor_ct_base = 18;

    constexpr auto input_args = TensorAccessorArgs<accessor_ct_base>();
    // One args block for BOTH weights (equal placement CT args, host-asserted): one accessor type, one
    // fill_affine_rows instantiation — the kernel-config ring budget.
    [[maybe_unused]] constexpr auto affine_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();

    // ---- runtime args: grid-wide constants are COMMON, per-core geometry per core ----
    const uint32_t input_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t gamma_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t beta_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t N = get_common_arg_val<uint32_t>(3);
    const uint32_t HWt = get_common_arg_val<uint32_t>(4);
    const uint32_t Ct = get_common_arg_val<uint32_t>(5);
    const uint32_t HW = get_common_arg_val<uint32_t>(6);
    const uint32_t Hmax = get_common_arg_val<uint32_t>(7);
    // The input's aligned page (buffer_aligned_page_size): RM shard stick / RM full-row stick / tile.
    const uint32_t page_bytes = get_common_arg_val<uint32_t>(8);
    [[maybe_unused]] const uint32_t hw_stride = get_common_arg_val<uint32_t>(9);  // stick pitch between images
    const uint32_t c_period = get_common_arg_val<uint32_t>(10);  // channel period of a block lane (>= K*32: identity)
    // The weights' aligned page (buffer_aligned_page_size): the (1,1,1,C) stick, or a weight tile.
    [[maybe_unused]] const uint32_t affine_page_bytes = get_common_arg_val<uint32_t>(11);
    const uint32_t is_active = get_arg_val<uint32_t>(0);
    const uint32_t r0 = get_arg_val<uint32_t>(1);
    const uint32_t H_core = get_arg_val<uint32_t>(2);
    const uint32_t t0 = get_arg_val<uint32_t>(3);
    const uint32_t c0 = get_arg_val<uint32_t>(4);
    const uint32_t c_valid = get_arg_val<uint32_t>(5);
    [[maybe_unused]] const uint32_t s0 = get_arg_val<uint32_t>(6);
    [[maybe_unused]] const uint32_t sticks_valid = get_arg_val<uint32_t>(7);
    const uint32_t row_hi = get_arg_val<uint32_t>(8);  // valid sticks of this block's last tile-row (interleaved)

    if (is_active == 0) {
        return;
    }

    constexpr uint32_t input_tile_bytes = get_tile_size(cb_input_tiles);
    constexpr uint32_t input_elem = input_tile_bytes / 1024;

    // cb_scaler is a template parameter of compute's reduce<..., AccumulateViaAdd>, but that datapath never
    // waits on or reads it: reduce_helpers_compute.inl waits the scaler only for a partial 0/1 mask (this op
    // reduces whole tiles) or the CopySeedZeroPair reload (the op uses the default CopySeedPairs), and its
    // init is copy_init, not reduce_init. So it is never filled; the CB stays allocated (1 bf16 page) so the
    // helper's template contract holds.

    // Weights are (1,1,1,C): ROW_MAJOR -> one stick page of C elements (page 0 + byte offset);
    // TILE -> Ct tile pages, channel c in row 0 of tile c/32. The page NoC addresses this core's
    // channels touch (one stick page, or the <= K + 1 weight tiles from c0/32) are resolved HERE, in
    // one loop over the two weights (one accessor construction, one get_noc_addr site, one call —
    // the program must fit the kernel-config ring in the --dev build), and handed to fill_affine_rows.
    if constexpr (has_gamma || has_beta) {
        uint64_t pages[K + 1];
        const uint32_t tile0 = c0 >> 5;
        for (uint32_t w = 0; w < 2; ++w) {
            if (w == 0 ? !has_gamma : !has_beta) {
                continue;
            }
            const uint32_t cb_rows = w == 0 ? cb_gamma_rows : cb_beta_rows;
            const auto acc = TensorAccessor(affine_args, w == 0 ? gamma_addr : beta_addr, affine_page_bytes);
            const uint32_t num_pages = affine_tile ? (tile0 + K + 1 < Ct ? K + 1 : Ct - tile0) : 1;
            for (uint32_t t = 0; t < num_pages; ++t) {
                pages[t] = acc.get_noc_addr(affine_tile ? tile0 + t : 0);
            }
            fill_affine_rows<affine_tile, affine_bf8>(
                pages, tile0, cb_rows, get_tile_size(cb_rows), K, c0, c_valid, c_period);
        }
    }

    if constexpr (sharded) {
        // ---- block_sharded_resident: the shard is the block, already in L1 ----
        if constexpr (rm_direct) {
            // RM shard, direct view: cb_input_sticks IS the shard, re-paged as tile-sized
            // pages (K pages = one 32-row block of the lcm(shard_w, 32)-wide view). Image n's rows are
            // consecutive in the shard and images are processed in order, so compute's tilize consumes
            // the region front to back: hand it exactly image n's rows*K credits (zero-copy — no NoC
            // read of the local shard, no staging).
            for (uint32_t image = 0; image < N; ++image) {
                const auto w = groupnorm_geometry::image_work_sharded(image, s0, sticks_valid, HW, hw_stride);
                if (w.active) {
                    cb_reserve_back(cb_input_sticks, w.rows * K);
                    cb_push_back(cb_input_sticks, w.rows * K);
                }
            }
        } else if constexpr (!is_rm) {
            // TILE shard: cb_input_tiles is placed on the shard buffer. Hand compute the whole
            // shard's credits once per image IT HAS WORK IN (compute pops Hs*K at that image's
            // release, so the ring pointer is back at base — the block offsets are shard-absolute;
            // an image the shard misses is skipped by compute and must not be credited).
            for (uint32_t image = 0; image < N; ++image) {
                if (!groupnorm_geometry::image_work_sharded(image, s0, sticks_valid, HW, hw_stride).active) {
                    continue;
                }
                cb_reserve_back(cb_input_tiles, Hmax * K);
                cb_push_back(cb_input_tiles, Hmax * K);
            }
        } else {
            // RM shard: stage image n's tile-rows stick by stick (L1 -> L1, own core) into the
            // tilize staging CB at the K*64 B stick stride the tilize expects; sticks outside
            // the image / lanes beyond c_valid are zero (pass 1 sums them harmlessly; pass 2 masks
            // the partial rows — see the writer's cb_masked_mean).
            constexpr uint32_t stride = K * 32 * input_elem;
            const uint32_t copy_bytes = c_valid * input_elem;
            // Lanes [c_valid, K*32) are never written by the stick copies: zero them whenever they
            // exist — also when c_valid is tile-aligned but short of the K*32 block (the last column
            // shard of C = 1920 / 2560 under a 176- / 240-wide shard has c_valid = 160 < 192 / 256),
            // so pass 1 never sums stale L1 (a NaN bit pattern times a zero membership row is NaN).
            const bool lane_pad = c_valid < K * 32;
            const uint32_t shard_base = get_read_ptr(cb_input_shard);
            CircularBuffer sticks(cb_input_sticks);
            Noc noc;
            for (uint32_t image = 0; image < N; ++image) {
                const auto w = groupnorm_geometry::image_work_sharded(image, s0, sticks_valid, HW, hw_stride);
                for (uint32_t i = 0; i < w.rows; ++i) {
                    uint32_t sa, sb;
                    groupnorm_geometry::row_sticks(w, i, sa, sb);
                    sticks.reserve_back(K);
                    if (lane_pad || sa != 0 || sb != 32) {
                        noc.async_write_zeros(sticks, K * input_tile_bytes);
                        noc.write_zeros_l1_barrier();
                    }
                    const uint32_t l1 = sticks.get_write_ptr();
                    const uint32_t src0 = shard_base + ((w.row_off + i) * 32) * page_bytes;
                    // One-packet reads (a stick is <= K*32*elem <= 2 KB, under the burst size), issued inline:
                    // this loop is NoC-issue-bound and a per-stick noinline call measured slower.
                    for (uint32_t s = sa; s < sb; ++s) {
                        noc_async_read<NOC_MAX_BURST_SIZE>(
                            get_noc_addr(my_x[noc_index], my_y[noc_index], src0 + s * page_bytes),
                            l1 + s * stride,
                            copy_bytes);
                    }
                    noc_async_read_barrier();
                    sticks.push_back(K);
                }
            }
        }
        return;
    }

    // Input accessor: TILE -> tile pages; ROW_MAJOR -> sticks of C elements (aligned page).
    const auto input = TensorAccessor(input_args, input_addr, page_bytes);
    // Every image is the same [0, H_core) sub-block; only the last tile-row can be ragged.
    const auto w = groupnorm_geometry::image_work_interleaved(H_core, row_hi);

    if constexpr (is_rm) {
        // Lanes beyond c_valid are never written by the stick reads: zero the whole stick ring
        // ONCE (it is empty here, so the reserve returns at base) and they stay zero for the
        // program. Pad sticks of a ragged last tile-row are re-zeroed per row below.
        if ((c_valid & 31) != 0) {
            CircularBuffer sticks(cb_input_sticks);
            Noc noc;
            const uint32_t cap = get_local_cb_interface(cb_input_sticks).fifo_num_pages;
            sticks.reserve_back(cap);
            noc.async_write_zeros(sticks, cap * input_tile_bytes);
            noc.write_zeros_l1_barrier();
        }
    }

    // read_tile_rows: q tile-rows starting at block row `row` of `image`, pushed as `nominal` pages.
    auto read_tile_rows = [&](uint32_t image, uint32_t row, uint32_t q, uint32_t nominal) {
        cb_reserve_back(cb_input_tiles, nominal);
        uint32_t l1 = get_write_ptr(cb_input_tiles);
        for (uint32_t r = 0; r < q; ++r) {
            const uint32_t page_row = (image * HWt + r0 + row + r) * Ct + t0;
            for (uint32_t col = 0; col < K; ++col) {
                noc_async_read(input.get_noc_addr(page_row + col), l1, input_tile_bytes);
                l1 += input_tile_bytes;
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_input_tiles, nominal);
    };

    // Streaming: one input pass per algorithmic pass — three (sum, centered squares, apply), or TWO when
    // compute takes both statistics from the resident chunk (two_pass: pass A, then the apply pass B).
    constexpr uint32_t passes = input_resident ? 1 : (two_pass ? 2 : 3);

    for (uint32_t image = 0; image < N; ++image) {
        for (uint32_t pass = 0; pass < passes; ++pass) {
            if constexpr (is_rm) {
                // One tile-row (its sb valid sticks, c_valid*elem bytes each at the K*64 B tilize
                // stride) per call; the helper pushes K tile-sized pages per tile-row, so no chunk
                // boundary has to agree with compute here.
                for (uint32_t i = 0; i < H_core; ++i) {
                    uint32_t sa, sb;
                    groupnorm_geometry::row_sticks(w, i, sa, sb);
                    if (sb != 32) {
                        CircularBuffer sticks(cb_input_sticks);
                        Noc noc;
                        sticks.reserve_back(K);  // same slot the helper reserves next (no push here)
                        noc.async_write_zeros(sticks, K * input_tile_bytes);
                        noc.write_zeros_l1_barrier();
                    }
                    dataflow_kernel_lib::read_sticks_for_tilize<cb_input_sticks>(
                        input, sb, c_valid * input_elem, image * HW + 32 * (r0 + i), c0 * input_elem);
                }
            } else {
                // Resident: exact q*K per chunk (the image block is padded to Hmax*K below).
                // Streaming: nominal Q*K per chunk; pass 2 of an hw_mask program follows compute's
                // [head][body][tail] segment order so every chunk boundary agrees.
                const auto seg = groupnorm_geometry::pass2_segments(w, hw_mask && pass == 1);
                const uint32_t nominal_q = input_resident ? 0 : Q * K;  // 0 -> exact
                auto push = [&](uint32_t row, uint32_t q) {
                    read_tile_rows(image, row, q, nominal_q ? nominal_q : q * K);
                };
                if (seg.head) {
                    push(0, 1);
                }
                for (uint32_t r = seg.body_start; r < seg.body_end; r += Q) {
                    push(r, (seg.body_end - r) < Q ? (seg.body_end - r) : Q);
                }
                if (seg.tail) {
                    push(H_core - 1, 1);
                }
            }
            if constexpr (input_resident && !is_rm) {
                // Pad the resident block to the uniform Hmax*K capacity so the
                // ring pointer returns to base every image (compute pops Hmax*K).
                const uint32_t pad = (Hmax - H_core) * K;
                if (pad > 0) {
                    cb_reserve_back(cb_input_tiles, pad);
                    cb_push_back(cb_input_tiles, pad);
                }
            }
        }
    }
}

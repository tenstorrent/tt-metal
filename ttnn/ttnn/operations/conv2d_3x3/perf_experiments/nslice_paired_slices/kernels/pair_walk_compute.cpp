// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Isolated conv2d_3x3 N-slice-outer matmul walk (single core, compute only). Bench-only
// (perf_experiments/nslice_paired_slices).
//
// The op multiplies a resident [bm, K] in0 block by one owner N-slice of weights ([K, out_n]) at a time, whole K in
// one DEST accumulation, packs the fp32 [bm, out_n] slice once and pushes it to the CB of the RISC that sends it:
// cb_slice_r (reader, the +d steps j = 0, 2, 4, ...) or cb_matmul_partials (writer, the -d steps and the own slice,
// the last step). Here both slice CBs hold all of their RISC's slices (no consumer), and compute itself pushes each
// weight round (the writer's stand-in) right before it consumes it.
//
// impl 0 base:        the op's exact call, one ckl::matmul_block per slice ([bm/sbh x out_n/sbw] subblocks of
//                     sbh x sbw, Short init, Out, TileRowMajor, in0 WaitAndRetainOnLastBlock, in1 WaitAndPopPerKBlock).
//                     Weights stacked: slice j = [K, out_n] row-major.
// impl 1 pair_il:     two consecutive slices per call, weights interleaved per pair: [K, 2 * out_n] row-major (k-row =
//                     slice A's out_n tiles then slice B's), so one LLK matmul_block with ct_dim = 2 * out_n streams a
//                     whole subblock row; each DEST tile packs to its own slice's CB (A: tiles 0..out_n-1 of a row).
// impl 2 pair_2call:  pairs on the op's stacked weights ([K, out_n] then [K, out_n]): per k two LLK calls of
//                     ct_dim = out_n into DEST 0.. and out_n.. (no writer change), packs as impl 1.
// impl 3 pair_1cb:    ckl::matmul_block on the interleaved pair weights, ONE output CB of [bm, 2 * out_n] tiles per
//                     pair (tile-row-major): the helper-only form (senders would read their half strided).
// impl 4 base_raw:    impl 0's walk, raw (whole-slice reserve / push, init + reconfig once per slice): attributes the
//                     helper's own per-subblock overhead.
//
//   RAW LLK / HELPER BYPASS (impls 1, 2, 4): ckl::matmul_block takes one output CB per call and packs a subblock's
//   tiles row-strided into it; a 1 x (2 * out_n) subblock whose left half belongs to one RISC's slice CB and right
//   half to the other's cannot be expressed. The raw walk performs the helper's LLK sequence (same per-tile
//   matmul_block calls over K in the same order, pack_tile<true> to absolute offsets) with per-tile pack targets.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

namespace ckl = compute_kernel_lib;

namespace {

constexpr uint32_t CB_IN0 = 0;
constexpr uint32_t CB_W = 1;
constexpr uint32_t CB_OUT_W = 4;     // writer-sent slices (cb_matmul_partials)
constexpr uint32_t CB_OUT_PAIR = 5;  // impl 3 only
constexpr uint32_t CB_OUT_R = 10;    // reader-sent slices (cb_slice_r)

template <uint32_t nslices>
constexpr bool reader_sends(uint32_t j) {
    return j != nslices - 1 && j % 2 == 0;
}

// Raw pair walk (impls 1, 2): slices ja = 2p and jb = 2p + 1; h rows per subblock, the whole pair width per row.
template <uint32_t bm, uint32_t K, uint32_t out_n, uint32_t h, bool interleaved>
ALWI void raw_pair(uint32_t cb_a, uint32_t cb_b) {
    constexpr uint32_t pw = 2 * out_n;
    constexpr uint32_t pair_w_tiles = K * pw;
    constexpr uint32_t slice_tiles = bm * out_n;
    static_assert(bm % h == 0 && h * pw <= 4);
    cb_reserve_back(CB_W, pair_w_tiles);
    cb_push_back(CB_W, pair_w_tiles);
    reconfig_data_format(CB_W, CB_IN0);
    PACK((pack_reconfig_data_format(cb_a)));
    ckernel::matmul_block_init(CB_IN0, CB_W, false, interleaved ? pw : out_n, h, K);
    cb_wait_front(CB_W, pair_w_tiles);
    cb_reserve_back(cb_a, slice_tiles);
    cb_reserve_back(cb_b, slice_tiles);
    for (uint32_t r0 = 0; r0 < bm; r0 += h) {
        tile_regs_acquire();
        for (uint32_t k = 0; k < K; ++k) {
            if constexpr (interleaved) {
                ckernel::matmul_block(CB_IN0, CB_W, r0 * K + k, k * pw, 0, false, pw, h, K);
            } else {
                // DEST holds [h, pw] row-major only when h == 1 (ct_dim = out_n blocks would interleave rows).
                static_assert(h == 1, "stacked pairs: one row per subblock");
                ckernel::matmul_block(CB_IN0, CB_W, r0 * K + k, k * out_n, 0, false, out_n, h, K);
                ckernel::matmul_block(CB_IN0, CB_W, r0 * K + k, K * out_n + k * out_n, out_n, false, out_n, h, K);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t rr = 0; rr < h; ++rr) {
            for (uint32_t c = 0; c < out_n; ++c) {
                pack_tile<true>(rr * pw + c, cb_a, (r0 + rr) * out_n + c);
            }
            for (uint32_t c = 0; c < out_n; ++c) {
                pack_tile<true>(rr * pw + out_n + c, cb_b, (r0 + rr) * out_n + c);
            }
        }
        tile_regs_release();
    }
    cb_push_back(cb_a, slice_tiles);
    cb_push_back(cb_b, slice_tiles);
    cb_pop_front(CB_W, pair_w_tiles);
}

// Raw single-slice walk (impl 4): sbh x sbw subblocks, whole-slice reserve / push.
template <uint32_t bm, uint32_t K, uint32_t out_n, uint32_t sbh, uint32_t sbw>
ALWI void raw_single(uint32_t cb) {
    constexpr uint32_t w_tiles = K * out_n;
    constexpr uint32_t slice_tiles = bm * out_n;
    cb_reserve_back(CB_W, w_tiles);
    cb_push_back(CB_W, w_tiles);
    reconfig_data_format(CB_W, CB_IN0);
    PACK((pack_reconfig_data_format(cb)));
    ckernel::matmul_block_init(CB_IN0, CB_W, false, sbw, sbh, K);
    cb_wait_front(CB_W, w_tiles);
    cb_reserve_back(cb, slice_tiles);
    for (uint32_t r0 = 0; r0 < bm; r0 += sbh) {
        for (uint32_t c0 = 0; c0 < out_n; c0 += sbw) {
            tile_regs_acquire();
            for (uint32_t k = 0; k < K; ++k) {
                ckernel::matmul_block(CB_IN0, CB_W, r0 * K + k, k * out_n + c0, 0, false, sbw, sbh, K);
            }
            tile_regs_commit();
            tile_regs_wait();
            ckl::pack_subblock_row_strided(0, cb, r0 * out_n + c0, out_n, sbh, sbw);
            tile_regs_release();
        }
    }
    cb_push_back(cb, slice_tiles);
    cb_pop_front(CB_W, w_tiles);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t bm = get_compile_time_arg_val(0);
    constexpr uint32_t K = get_compile_time_arg_val(1);
    constexpr uint32_t out_n = get_compile_time_arg_val(2);
    constexpr uint32_t nslices = get_compile_time_arg_val(3);
    constexpr uint32_t impl = get_compile_time_arg_val(4);
    constexpr uint32_t sbh = get_compile_time_arg_val(5);  // impls 0 / 4: slice subblock; impls 1-3: pair rows
    constexpr uint32_t sbw = get_compile_time_arg_val(6);
    static_assert(nslices % 2 == 0, "bench: whole pairs");

    compute_kernel_hw_startup<SrcOrder::Reverse>(CB_IN0, CB_W, impl == 3 ? CB_OUT_PAIR : CB_OUT_W);
    // The resident in0 block (the op tilizes it once per M-block before the walk).
    cb_reserve_back(CB_IN0, bm * K);
    cb_push_back(CB_IN0, bm * K);

    CircularBuffer in0(CB_IN0);
    CircularBuffer weights(CB_W);

    {
        MaybeDeviceZoneScope("compute_matmul_walk");
        if constexpr (impl == 0) {
            CircularBuffer slice_r(CB_OUT_R);
            CircularBuffer partials(CB_OUT_W);
            for (uint32_t j = 0; j < nslices; ++j) {
                cb_reserve_back(CB_W, K * out_n);
                cb_push_back(CB_W, K * out_n);
                CircularBuffer& slice_cb = reader_sends<nslices>(j) ? slice_r : partials;
                ckl::matmul_block<
                    /*transpose=*/false,
                    /*packer_l1_acc=*/false,
                    ckl::LastBlockTarget::Out,
                    ckl::OutputCBLayout::TileRowMajor,
                    ckl::matmul_config::InitMode::Short,
                    ckl::InputPolicy::WaitAndRetainOnLastBlock,
                    ckl::InputPolicy::WaitAndPopPerKBlock>(
                    in0,
                    weights,
                    slice_cb,
                    slice_cb,
                    ckl::MatmulBlockShape::of(bm / sbh, out_n / sbw, sbh, sbw, K, 1),
                    {},
                    {},
                    /*in1_per_core_w=*/out_n,
                    /*out_row_width=*/out_n);
            }
        } else if constexpr (impl == 1 || impl == 2) {
            for (uint32_t p = 0; p < nslices / 2; ++p) {
                // Step 2p is the reader's (+d), 2p + 1 the writer's (-d, or the own slice last).
                raw_pair<bm, K, out_n, sbh, impl == 1>(CB_OUT_R, CB_OUT_W);
            }
        } else if constexpr (impl == 3) {
            CircularBuffer pair_out(CB_OUT_PAIR);
            for (uint32_t p = 0; p < nslices / 2; ++p) {
                cb_reserve_back(CB_W, K * 2 * out_n);
                cb_push_back(CB_W, K * 2 * out_n);
                ckl::matmul_block<
                    /*transpose=*/false,
                    /*packer_l1_acc=*/false,
                    ckl::LastBlockTarget::Out,
                    ckl::OutputCBLayout::TileRowMajor,
                    ckl::matmul_config::InitMode::Short,
                    ckl::InputPolicy::WaitAndRetainOnLastBlock,
                    ckl::InputPolicy::WaitAndPopPerKBlock>(
                    in0,
                    weights,
                    pair_out,
                    pair_out,
                    ckl::MatmulBlockShape::of(bm / sbh, 1, sbh, 2 * out_n, K, 1),
                    {},
                    {},
                    /*in1_per_core_w=*/2 * out_n,
                    /*out_row_width=*/2 * out_n);
            }
        } else {
            for (uint32_t j = 0; j < nslices; ++j) {
                raw_single<bm, K, out_n, sbh, sbw>(reader_sends<nslices>(j) ? CB_OUT_R : CB_OUT_W);
            }
        }
    }
}

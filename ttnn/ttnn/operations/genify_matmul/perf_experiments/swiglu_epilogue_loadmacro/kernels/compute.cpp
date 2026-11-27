// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Isolated bench (round 2) of genify_matmul's swiglu epilogue: one fronted Float32 accumulator block
// (block_m x block_n tiles, TileRowMajor, gate half then up half per row -- or natural (gate, up) pairs)
// -> block_m x block_n/2 output tiles, out = silu(gate) * up. Every variant is the op's eltwise_chain;
// only the SFPU element differs.
//   0 baseline      the op's SwiGluMul, included verbatim from kernels/genify_matmul_swiglu_sfpu.hpp
//   1 floor         CopyTile g -> D0, CopyTile u -> D1, PackTile D0 (no SFPU)
//   2 sfpu_x2       baseline with SwiGluMul run twice (delta vs baseline = one SFPU pass)
//   3 nosfpu_store  floor + the SwiGluMul init (no SFPU exec)
//   4 lm            SwiGluMulLM: SFPLOADMACRO-pipelined body (swiglu_lm_sfpu.hpp)
//   5 tuned         SwiGluMulT: instruction-sequence tuned body, no load macros (swiglu_lm_sfpu.hpp)
//   6, 7            lm2 / lm3 experimental
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/pack.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/ttnn/operations/genify_matmul/kernels/genify_matmul_swiglu_sfpu.hpp"
#include "swiglu_lm_sfpu.hpp"

namespace ckl = compute_kernel_lib;

constexpr uint32_t cb_partials = get_compile_time_arg_val(0);
constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(1);
constexpr uint32_t swiglu_rows = get_compile_time_arg_val(2);
constexpr uint32_t swiglu_cols = get_compile_time_arg_val(3);
constexpr uint32_t swiglu_stride = get_compile_time_arg_val(4);
constexpr uint32_t swiglu_up_base = get_compile_time_arg_val(5);
constexpr uint32_t epilogue_block_tiles = get_compile_time_arg_val(6);
constexpr uint32_t variant = get_compile_time_arg_val(7);
constexpr bool silu_fp32_precision = get_compile_time_arg_val(8) != 0;
constexpr uint32_t block_tiles = get_compile_time_arg_val(9);
constexpr bool fp32_grade = silu_fp32_precision && DST_ACCUM_MODE;

#define PARTIALS_IN(...)                      \
    ckl::CopyTile<                            \
        ckl::input(                           \
            cb_partials,                      \
            ckl::WaitPolicy::None,            \
            ckl::PopPolicy::None,             \
            ckl::InputTileMapping::Block,     \
            ckl::DataFormatReconfig::Enabled, \
            ckl::TileAddressing::Strided),    \
        __VA_ARGS__>
#define OUT_PACK(SLOT) \
    ckl::PackTile<ckl::output(cb_output_tiles, ckl::ReservePolicy::PerOuter, ckl::PushPolicy::PerOuter), SLOT>

// SwiGluMul init only (variant 3).
struct InitOnly : ckl::BinaryOp<InitOnly, ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D0> {
    static ALWI void init() { ckl::SwiGluMul<fp32_grade>::init(); }
    static ALWI void exec_impl(uint32_t) {}
};

ALWI void run_epilogue(DataflowBuffer& partials_dfb, DataflowBuffer& out_dfb) {
    using namespace ckl;
    const auto shape = IterationShape::grid(swiglu_rows, swiglu_cols).block_size(epilogue_block_tiles);
    const StridedTileRange gr{0, swiglu_stride}, ur{swiglu_up_base, swiglu_stride};
    if constexpr (variant == 0) {
        eltwise_chain(
            shape,
            PARTIALS_IN(Dst::D0){partials_dfb, gr},
            PARTIALS_IN(Dst::D1){partials_dfb, ur},
            SwiGluMul<fp32_grade, Dst::D0, Dst::D1, Dst::D0>{},
            OUT_PACK(Dst::D0){out_dfb});
    } else if constexpr (variant == 1) {
        eltwise_chain(
            shape,
            PARTIALS_IN(Dst::D0){partials_dfb, gr},
            PARTIALS_IN(Dst::D1){partials_dfb, ur},
            OUT_PACK(Dst::D0){out_dfb});
    } else if constexpr (variant == 2) {
        eltwise_chain(
            shape,
            PARTIALS_IN(Dst::D0){partials_dfb, gr},
            PARTIALS_IN(Dst::D1){partials_dfb, ur},
            SwiGluMul<fp32_grade, Dst::D0, Dst::D1, Dst::D0>{},
            SwiGluMul<fp32_grade, Dst::D0, Dst::D1, Dst::D0>{},
            OUT_PACK(Dst::D0){out_dfb});
    } else if constexpr (variant == 3) {
        eltwise_chain(
            shape,
            PARTIALS_IN(Dst::D0){partials_dfb, gr},
            PARTIALS_IN(Dst::D1){partials_dfb, ur},
            InitOnly{},
            OUT_PACK(Dst::D0){out_dfb});
    } else {
        // candidates: SwiGluLM<impl> writes its result to SwiGluLMOut<impl> (D0 or D1)
        constexpr uint32_t impl = variant - 4;
        eltwise_chain(
            shape,
            PARTIALS_IN(Dst::D0){partials_dfb, gr},
            PARTIALS_IN(Dst::D1){partials_dfb, ur},
            SwiGluLM<impl, fp32_grade>{},
            OUT_PACK((SwiGluLM<impl, fp32_grade>::out_slot)){out_dfb});
    }
}

void kernel_main() {
    const uint32_t reps = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(cb_partials, cb_partials, cb_output_tiles);
    CircularBuffer partials_buf(cb_partials);
    DataflowBuffer partials_dfb(cb_partials), out_dfb(cb_output_tiles);

    // Resident sharded block: mark it available once (the op's matmul pushes it).
    partials_buf.reserve_back(block_tiles);
    partials_buf.push_back(block_tiles);
    partials_buf.wait_front(block_tiles);
    {
        MaybeDeviceZoneScope("swg_epilogue");
        for (uint32_t r = 0; r < reps; ++r) {
            run_epilogue(partials_dfb, out_dfb);
        }
    }
    partials_buf.pop_front(block_tiles);
}

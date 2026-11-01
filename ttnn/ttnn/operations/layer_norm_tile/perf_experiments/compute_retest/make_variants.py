"""Builds kernels/{stats,affine,combo}/layer_norm_tile_compute.cpp from kernels/base (a copy of the op's current compute)."""

from pathlib import Path

HERE = Path(__file__).parent / "kernels"
BASE = (HERE / "base" / "layer_norm_tile_compute.cpp").read_text()


def sub(text, old, new):
    assert text.count(old) == 1, old[:80]
    return text.replace(old, new)


STATS_HELPERS = r"""#ifdef TRISC_MATH
#include "ckernel_sfpu_sqrt.h"
#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel::sfpu {
// rsqrt(x * scale + addend) on the SFPU loads holding tile column 0 only: a load covers 4 rows x 8
// columns (dest address bit 1 picks even / odd columns) and dst_reg[k] is address 2k, so column 0 is
// dst_reg[0], [2], [4], [6]. Half the rsqrt work of a VectorMode::C pass; normalize's col broadcast
// reads only column 0. Same rsqrt body as rsqrt_tile.
template <bool APPROXIMATION_MODE, bool fp32_dest_acc_en, uint32_t scale_bits>
inline void calculate_add_rsqrt_col0(uint32_t addend_bits) {
    const sfpi::vFloat scale = Converter::as_float(scale_bits);
    const sfpi::vFloat addend = Converter::as_float(addend_bits);
#pragma GCC unroll 4
    for (int group = 0; group < 4; ++group) {
        sfpi::vFloat y =
            _calculate_sqrt_body_<APPROXIMATION_MODE, true, false>(sfpi::dst_reg[2 * group] * scale + addend);
        if constexpr (!fp32_dest_acc_en) {
            y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[2 * group] = y;
    }
}
}  // namespace ckernel::sfpu
#endif

namespace {

using compute_kernel_lib::DataFormatReconfig;

// c (rows x Wt, resident at the cb_centered front, not popped) -> inv_std (one tile per row, column 0
// valid). Per row one DEST window: S = sum_w c * c (mul, acc_to_dest), sfpu_reduce ROW collapse, then
// rsqrt(S * 1/W + eps) on column 0.
template <
    uint32_t cb_centered,
    uint32_t cb_inv_std,
    uint32_t tensor_w_tiles,
    uint32_t inv_w_bits,
    uint32_t epsilon_bits>
ALWI void row_stats_block(uint32_t rows, DataflowBuffer& centered, DataflowBuffer& inv_std) {
    constexpr DataFormat dst_fmt = DST_ACCUM_MODE ? DataFormat::Float32 : DataFormat::Float16_b;
    centered.wait_front(rows * tensor_w_tiles);
    reconfig_data_format(cb_centered, cb_centered);
    pack_reconfig_data_format(cb_inv_std);
    for (uint32_t row = 0; row < rows; ++row) {
        const uint32_t base = row * tensor_w_tiles;
        tile_regs_acquire();
        mul_init(cb_centered, cb_centered, true);  // a freshly acquired DEST reads 0: the first mul seeds S
        for (uint32_t col = 0; col < tensor_w_tiles; ++col) {
            mul_tiles(cb_centered, cb_centered, base + col, base + col, 0);
        }
        sfpu_reduce_init<PoolType::SUM, dst_fmt>();
        sfpu_reduce<PoolType::SUM, dst_fmt, ReduceDim::REDUCE_ROW>(0, 1, 1);
        rsqrt_tile_init();
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_add_rsqrt_col0,
            (APPROX, DST_ACCUM_MODE, inv_w_bits),
            0,
            VectorMode::C,
            epsilon_bits));
        tile_regs_commit();
        inv_std.reserve_back(1);
        tile_regs_wait();
        pack_tile(0, cb_inv_std);
        tile_regs_release();
        inv_std.push_back(1);
    }
}
"""

STATS_HEAD = """// row_stats_block uses raw LLK: eltwise_chain's PerRow DEST accumulation packs the accumulated tile at
// row end with no post-accumulation hook, and reduce() reads its input only from a CB, so collapsing
// and finalizing S inside the accumulating DEST window (no fp32 sumsq CB round trip, no copy_tile
// reload, one acquire per row) needs the window written out by hand.
"""


def stats(text):
    text = sub(
        text,
        "//   row_sumsq_block    : S = sum_w c * c, DEST-accumulated   -> row_sumsq (c stays resident)\n"
        "//   row_inv_std_block  : rsqrt(collapse(S) / W + eps)        -> inv_std\n",
        "//   row_stats_block    : S = sum_w c * c DEST-accumulated, then in the same DEST window the row\n"
        "//                        collapse and rsqrt(S / W + eps)     -> inv_std  (c stays resident)\n",
    )
    text = sub(
        text,
        "// one is page i of the other. Both reduces use AccumulateViaAdd on tile-aligned W, which never reads\n"
        "// the scaler slot (bound to cb_gamma_tiles). The inv_std post-op uses raw SFPU calls because\n"
        "// reduce()'s post_reduce_op hook takes a DEST-index callable, not chain elements.\n",
        "// one is page i of the other. row_mean uses AccumulateViaAdd on tile-aligned W, which never reads\n"
        "// the scaler slot (bound to cb_gamma_tiles).\n//\n" + STATS_HEAD,
    )
    text = sub(text, '#include "api/compute/eltwise_unary/binop_with_scalar.h"\n', "")
    text = sub(text, "namespace {\n\nusing compute_kernel_lib::DataFormatReconfig;\n", STATS_HELPERS)
    text = sub(
        text,
        "    constexpr uint32_t cb_row_sumsq = get_compile_time_arg_val(7);\n",
        "    // compile-time arg 7 (cb_row_sumsq) is unused: S never leaves DEST.\n",
    )
    text = sub(text, "    DataflowBuffer row_sumsq(cb_row_sumsq);\n", "")
    start = text.index("    constexpr InputSpec centered_resident =")
    end = text.index("    for (uint32_t block_row = 0;")
    text = text[:start] + text[end:]
    start = text.index("        // row_sumsq_block (c stays resident for normalize_block)")
    end = text.index("        // normalize_block")
    text = (
        text[:start] + "        // row_stats_block (c stays resident for normalize_block)\n"
        "        {\n"
        '            MaybeDeviceZoneScope("compute_row_stats");\n'
        "            row_stats_block<cb_row_work_b, cb_inv_std, tensor_w_tiles, inv_w_bits, epsilon_bits>(\n"
        "                rows, work_b, inv_std);\n"
        "        }\n\n" + text[end:]
    )
    text = sub(text, "        const auto block_shape = IterationShape::grid(rows, tensor_w_tiles);\n", "")
    return text


AFFINE_BLOCK = """        // affine_block: DEST lanes seeded with the beta row broadcast; ELWMUL accumulates n * gamma onto
        // them. beta and gamma both unpack through srcB with one page format, and work_a keeps srcA's
        // format from normalize: one srcB reconfig here and none in the chain, whose two elements
        // alternate inits every DEST group and would re-emit an enabled reconfig each time.
        {
            MaybeDeviceZoneScope("compute_affine");
            reconfig_data_format_srcb(cb_inv_std, cb_gamma_tiles);
            eltwise_chain(
                lane_shape,
                UnaryBcast<
                    BroadcastDim::Row,
                    input(cb_beta_tiles, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Row, same_work_fmt)>{
                    beta_tiles},
                BinaryFpu<
                    BinaryFpuOp::Mul,
                    lane_input(cb_row_work_a, same_work_fmt),
                    input(
                        cb_gamma_tiles,
                        BroadcastDim::Row,
                        WaitPolicy::Upfront,
                        PopPolicy::None,
                        InputTileMapping::Row,
                        same_work_fmt)>{work_a, gamma_tiles},
                PackTile<lane_output(cb_output_tiles)>{output_tiles});
        }
    }
"""


def affine(text):
    text = sub(
        text,
        "//   affine_scale_block : * gamma (row bcast)                 -> work_b\n"
        "//   affine_shift_block : + beta (row bcast)                  -> output\n",
        "//   affine_block       : beta (row bcast) + n * gamma         -> output\n"
        "// affine_block seeds each DEST lane with the beta row broadcast and lets ELWMUL (a DEST\n"
        "// multiply-accumulate) add n * gamma onto it, so the shift needs no pass of its own.\n",
    )
    text = sub(
        text,
        '#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"\n',
        '#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"\n'
        '#include "ttnn/cpp/ttnn/kernel_lib/eltwise/broadcast/bcast.hpp"\n',
    )
    start = text.index("        // affine_scale_block")
    end = text.index("    gamma_tiles.pop_front(tensor_w_tiles);")
    return text[:start] + AFFINE_BLOCK + "\n" + text[end:]


for name, text in {"stats": stats(BASE), "affine": affine(BASE), "combo": affine(stats(BASE))}.items():
    (HERE / name / "layer_norm_tile_compute.cpp").write_text(text)
    print(name, "ok")

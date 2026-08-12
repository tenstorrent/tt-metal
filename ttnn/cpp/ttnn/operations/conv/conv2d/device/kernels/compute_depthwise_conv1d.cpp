// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/tilize.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
// Pulls in whichever SFPU op the ACTIVATION defines select; conv_bmm_tilize.cpp does the same. The
// defines alone are not enough -- without this the expansion of SFPU_OP_INIT_ACTIVATION does not
// compile.
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
// `apply_snake_beta` calls sin_tile/sin_tile_init unconditionally. Those names are not
// template-dependent, so they must be declared where the template is *defined*, not merely where it is
// instantiated -- without this every depthwise conv fails to build, not just the snake path.
#include "api/compute/eltwise_unary/trigonometry.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

// Defined below, next to the SFPU accumulate path it was written for, but called from all four
// accumulate paths -- the earliest of which precedes the definition.
// How many tile-columns of parameters the CB carries -- one per tile of the channel axis. Defined by
// the program factory alongside SNAKE_PARAMS_CB_ID; the fallback keeps `apply_snake_beta` compiling
// when the snake is off, since the macro is expanded in the template's *definition* whether or not it
// is ever instantiated.
#ifndef SNAKE_PARAM_NUM_COLS
#define SNAKE_PARAM_NUM_COLS 1
#endif

template <uint32_t dst_acc, uint32_t dst_a, uint32_t dst_b>
inline void apply_snake_beta(DataflowBuffer params_dfb, uint32_t param_col);
#include "api/dataflow/dataflow_buffer.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

// Compute one block (one kernel-tap slice) of a 1D depthwise conv.
//
// Per output tile:
//   dst[0]  = in0 * in1                                                 (FPU mul)
//   if idx > 0: dst[0] += prior partial loaded from scratch_dfb          (FPU add via DST reuse)
//   pack dst[0] -> out_dfb on the last tap, else scratch_dfb
//
// `idx` is the kernel-tap block index (0 .. num_taps-1, num_taps == filter_h*filter_w). The very
// first call (idx == 0) initializes the partial with the tap-0 product; subsequent calls accumulate
// via the DST_TO_SRCB dest-reuse pattern, which keeps the running partial in DST and only pulls the
// prior partial from L1. This gives a single pack per output tile and avoids the pack-format flips
// that corrupt block-float (BFLOAT8_B/BFLOAT4_B) outputs in the round-tripped variant — while
// still using FPU (not SFPU) for the add.
//
// The partial lives in scratch_dfb; only the last tap (idx == num_taps-1) packs into out_dfb. The host
// aliases scratch_dfb to out_dfb for a single height block (in-place), but uses a separate buffer when
// in0_num_blocks_h > 1, where out_dfb (the persistent sharded output) cannot double as the read-back
// scratch — block N would otherwise read back block N-1's already-written output.
//
// srcB (cfg92) tile descriptor: must match in1 for the mul, and is repopulated from DST for the
// dest-reuse add. We force srcB back to in1's format every iteration so block-float weights are
// decoded correctly.
inline void mul_and_accumulate_block(
    DataflowBuffer in0_dfb,
    DataflowBuffer in1_dfb,
    DataflowBuffer scratch_dfb,
    DataflowBuffer out_dfb,
    uint32_t block_num_tiles,
    uint32_t idx,
    uint32_t num_taps,
    // Tiles per row of the block, i.e. the channel axis in tiles. The flat loop below walks the block
    // row-major, so `i % block_w` is the output tile's column -- which is the parameter column the
    // snake needs. Unused when the snake is off.
    [[maybe_unused]] uint32_t block_w) {
    const uint32_t in0_cb_id = in0_dfb.get_id();
    const uint32_t in1_cb_id = in1_dfb.get_id();
    const uint32_t scratch_cb_id = scratch_dfb.get_id();
    // The last tap writes the finished output to out_dfb; earlier taps write the partial to scratch_dfb.
    const bool is_last_tap = (idx + 1 == num_taps);
    DataflowBuffer dst_dfb = is_last_tap ? out_dfb : scratch_dfb;
    const uint32_t dst_cb_id = dst_dfb.get_id();

    for (uint32_t i = 0; i < block_num_tiles; i++) {
        in1_dfb.wait_front(1);
        in0_dfb.wait_front(1);

        tile_regs_acquire();
        // mul: srcA = in0 (bf16), srcB = in1 (bf8/bf16) -> dst[0]
        reconfig_data_format_srcb(in1_cb_id);
        mul_tiles_init(in0_cb_id, in1_cb_id);
        mul_tiles(in0_cb_id, in1_cb_id, 0, 0, 0);

        if (idx != 0) {
            // dest-reuse add: dst[0] += scratch_dfb (the prior tap's partial). srcA gets scratch_dfb
            // (cfg52 must match its format); srcB is filled from dst[0] by the dest-reuse path.
            reconfig_data_format_srca(scratch_cb_id);
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                scratch_cb_id);
            scratch_dfb.wait_front(1);
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                scratch_cb_id, 0, 0);
            scratch_dfb.pop_front(1);

            // Restore srcA to in0's format for the next iteration's mul unpack.
            reconfig_data_format_srca(in0_cb_id);
        }
#ifdef SFPU_OP_INIT_ACTIVATION
        // See the note in the SFPU variant: last tap only, and `i` must name the DST slot.
        if (is_last_tap) {
            const uint32_t i = 0;
            SFPU_OP_FUNC_ACTIVATION
        }
#endif
#ifdef SNAKE_PARAMS_CB_ID
        // FPU path: only dst[0] holds the accumulator here, so slots 1 and 2 are free for the
        // parameters. The snake must be applied on **every** accumulate path, not just the SFPU one --
        // having it in one path only is why the first attempt produced output bit-identical to a plain
        // conv: the shape simply took a different path.
        if (is_last_tap) {
            DataflowBuffer snake_params_dfb(SNAKE_PARAMS_CB_ID);
            apply_snake_beta<0, 1, 2>(snake_params_dfb, i % block_w);
        }
#endif
        tile_regs_commit();

        // scratch_dfb and out_dfb share the output data format, so packing to either target needs no
        // pack reconfig.
        dst_dfb.reserve_back(1);
        tile_regs_wait();
        pack_tile(0, dst_cb_id);
        dst_dfb.push_back(1);
        tile_regs_release();

        in0_dfb.pop_front(1);
        in1_dfb.pop_front(1);
    }
}

// SFPU form of `mul_and_accumulate_block`, for fp32 operands. This is the one the audio decode
// actually takes: `coalesce_kw_reads` measures FALSE at every production FIR shape, contrary to
// MiniMaxH3_audio_decode_kernels.md §3.
//
// It fixes both fp32 truncation points in one go:
//
//  1. **The multiply.** `mul_tiles` routes operands through SrcA/SrcB, whose multiplier sees ~9
//     significand bits of SrcA and ~13 of SrcB regardless of MathFidelity
//     (`tech_reports/matrix_engine/matrix_engine.md:62-71`); fp32 has 24. `fp32_dest_acc_en` widens
//     the destination, not the operands, so no config lever reaches this.
//  2. **The partial reload.** The FPU form packs the running partial to L1 and pulls it back through
//     SrcA (`reconfig_data_format_srca(scratch_cb_id)`), rounding to TF32 on *every* tap. Copying it
//     into DST instead keeps it fp32 end to end -- the same defect, and the same fix, as
//     `matmul_multicore_reuse_mcast_1d_program_factory.cpp:743-761`.
//
// Measured together: 1.6e-03 -> fp32-grade against a float64 golden. DST usage is 3 tiles.
// Per-channel snake, applied to the finished conv output while it is still in DST.
//
//     y = x + inv_beta * sin(alpha * x)^2
//
// `alpha` and `inv_beta` arrive as tiles with the per-channel value replicated down all 32 rows, so
// this is plain `mul_binary_tile` with no broadcast. The host precomputes `inv_beta = 1/(beta+eps)`
// so the kernel needs no reciprocal.
//
// The scalar SFPU_OP_*_ACTIVATION seam cannot express this: it is parameterised by compile-time
// scalars, and snake's parameters are per channel. Three cheaper routes were ruled out by inspection
// for the same reason.
//
// DST budget: this runs only on the last tap, by which point DST_A and DST_B have been consumed, so
// it reuses them and the total stays at the 3 tiles the fp32 half-sync budget already allows.
//
//     DST_A <- alpha              DST_A = alpha * acc
//     DST_A <- sin(DST_A)         DST_A = sin^2
//     DST_B <- inv_beta           DST_A = inv_beta * sin^2
//     DST_ACC += DST_A
//
// The parameters ride in the **weights** CB rather than a CB of their own, which is why this takes
// `in1_dfb`. The design notes first picked the optional-input-tensor route instead, but that is the more
// expensive of the two once counted properly: adding an optional tensor to the op touches the
// operation struct, validate, compute_output_specs, create_program, the conv2d/conv1d invoke chains
// and pybind, and an op-signature change is all-or-nothing -- no subset of those files compiles, so it
// cannot be landed incrementally. Riding in the weight tensor needs the per-block fetch widened in the
// program factory (one file) and the host to append the parameter tiles to the weights (Python), with
// no signature change anywhere. Two C++ files against six.
//
// The reader already streams in1; the program factory must widen `weight_block_num_tiles` by 2 for the
// last tap so these two tiles are actually fetched, which is the part the plan correctly identified as
// unavoidable -- the per-block count comes from the conv dims, not from the weight tensor's shape, so
// appending to the tensor alone would leave the extra tiles never read.
//
// **Does not pop.** The parameters are per channel, so every tile in a block wants the same two
// tiles; popping them would destroy them for the tiles that follow. They live in a small dedicated CB
// filled once, not in the streaming weights CB -- which is also why this cannot ride in1: the
// non-coalesced path pops in1 once per tile per tap, so two pops at the last tap would consume
// `2 * block_num_tiles` per block against one push.
template <uint32_t dst_acc, uint32_t dst_a, uint32_t dst_b>
inline void apply_snake_beta(DataflowBuffer params_dfb, uint32_t param_col) {
    const uint32_t params_cb_id = params_dfb.get_id();

    // The CB holds every tile-column of the channel axis: alpha for columns 0..N-1, then inv_beta for
    // the same columns. `param_col` is the column of the output tile being finished, so a C=64 conv
    // (two tiles wide) picks up channels 32-63's parameters for its second column instead of
    // reapplying channels 0-31's -- which is what it did when the CB held one column and this read
    // tiles 0 and 1 unconditionally (rel_rmse 2.6e-01 at C=64, exact at C<=32).
    params_dfb.wait_front(2 * SNAKE_PARAM_NUM_COLS);

    // `copy_tile_to_dst_init_short` is the *short* init: it re-inits the datacopy MOP but does not
    // reconfigure SrcA's data format, so the params tile is unpacked with whatever format SrcA was
    // last set to. Measured symptom when it is stale: with alpha=inv_beta=1.0 the snake lands on odd
    // channel columns only and even columns come back untouched, because a 4-byte fp32 datum is being
    // consumed as two 2-byte ones -- the high half reads as the value and the zero low half as 0.
    // Proof: setting the parameters to 0x3F803F80 (both 16-bit halves = bf16 1.0) makes the even
    // columns apply at 100%. Force the full reconfig.
    reconfig_data_format_srca(params_cb_id);
    copy_tile_to_dst_init_short(params_cb_id);
    copy_tile(params_cb_id, param_col, dst_a);  // alpha for this output column
    mul_binary_tile_init();
    mul_binary_tile(dst_acc, dst_a, dst_a);  // alpha * x

    sin_tile_init();
    sin_tile(dst_a);  // sin(alpha * x)
    mul_binary_tile_init();
    mul_binary_tile(dst_a, dst_a, dst_a);  // sin^2

    reconfig_data_format_srca(params_cb_id);
    copy_tile_to_dst_init_short(params_cb_id);
    copy_tile(params_cb_id, SNAKE_PARAM_NUM_COLS + param_col, dst_b);  // inv_beta, same column
    mul_binary_tile_init();
    mul_binary_tile(dst_a, dst_b, dst_a);  // inv_beta * sin^2

    add_binary_tile_init();
    add_binary_tile(dst_acc, dst_a, dst_acc);  // x + inv_beta * sin^2

    // deliberately no pop_front: see the note above
}

inline void mul_and_accumulate_block_sfpu(
    DataflowBuffer in0_dfb,
    DataflowBuffer in1_dfb,
    DataflowBuffer scratch_dfb,
    DataflowBuffer out_dfb,
    uint32_t block_num_tiles,
    uint32_t idx,
    uint32_t num_taps,
    // Tiles per row of the block, i.e. the channel axis in tiles. The flat loop below walks the block
    // row-major, so `i % block_w` is the output tile's column -- which is the parameter column the
    // snake needs. Unused when the snake is off.
    [[maybe_unused]] uint32_t block_w) {
    const uint32_t in0_cb_id = in0_dfb.get_id();
    const uint32_t in1_cb_id = in1_dfb.get_id();
    const uint32_t scratch_cb_id = scratch_dfb.get_id();
    const bool is_last_tap = (idx + 1 == num_taps);
    DataflowBuffer dst_dfb = is_last_tap ? out_dfb : scratch_dfb;
    const uint32_t dst_cb_id = dst_dfb.get_id();

    // DST slots: 0 holds the running value, 1 and 2 stage the operands.
    constexpr uint32_t DST_ACC = 0;
    constexpr uint32_t DST_A = 1;
    constexpr uint32_t DST_B = 2;

    for (uint32_t i = 0; i < block_num_tiles; i++) {
        in1_dfb.wait_front(1);
        in0_dfb.wait_front(1);

        tile_regs_acquire();
        copy_tile_to_dst_init_short(in0_cb_id);
        copy_tile(in0_cb_id, 0, DST_A);
        copy_tile_to_dst_init_short(in1_cb_id);
        copy_tile(in1_cb_id, 0, DST_B);
        mul_binary_tile_init();
        mul_binary_tile(DST_A, DST_B, DST_ACC);

        if (idx != 0) {
            scratch_dfb.wait_front(1);
            copy_tile_to_dst_init_short(scratch_cb_id);
            copy_tile(scratch_cb_id, 0, DST_A);
            add_binary_tile_init();
            add_binary_tile(DST_ACC, DST_A, DST_ACC);
            scratch_dfb.pop_front(1);
        }
#ifdef SFPU_OP_INIT_ACTIVATION
        // Fused activation, applied to the finished output while it is still in DST -- only on the
        // last tap, since earlier ones are partial sums. The macro indexes DST by a variable literally
        // named `i` (the program factory emits the defines with "i" as the index name), so bind it in
        // an inner scope; the enclosing loop's `i` is a tile counter, not a DST slot.
        if (is_last_tap) {
            const uint32_t i = DST_ACC;
            SFPU_OP_FUNC_ACTIVATION
        }
#endif
#ifdef SNAKE_PARAMS_CB_ID
        // Per-channel snake, in place of (not as well as) the scalar activation seam. Off unless the
        // program factory emits the define, so the default path is byte-for-byte unchanged.
        if (is_last_tap) {
            DataflowBuffer snake_params_dfb(SNAKE_PARAMS_CB_ID);
            apply_snake_beta<DST_ACC, DST_A, DST_B>(snake_params_dfb, i % block_w);
        }
#endif
        tile_regs_commit();

        dst_dfb.reserve_back(1);
        tile_regs_wait();
        pack_tile(DST_ACC, dst_cb_id);
        dst_dfb.push_back(1);
        tile_regs_release();

        in0_dfb.pop_front(1);
        in1_dfb.pop_front(1);
    }
}

// SFPU form of the coalesced tap accumulation, for fp32 operands.
//
// The FPU path below multiplies through SrcA/SrcB, whose multiplier sees ~9 significand bits of SrcA
// and ~13 of SrcB regardless of MathFidelity (`tech_reports/matrix_engine/matrix_engine.md:62-71`);
// fp32 has 24. That truncation is the entire error an `Activation1d` injects -- measured 1.6e-03
// against a float64 golden at every production shape, against 5e-08 for the same filter expressed as
// elementwise multiply-add, which runs on the SFPU. `fp32_dest_acc_en` does not help: it widens the
// destination, not the operands.
//
// So for fp32 we route the multiply through the SFPU too. Both tiles are copied into DST and combined
// with `mul_binary_tile` / `add_binary_tile`, the same dispatch `reduce_helpers_compute.inl:41-63`
// makes for accurate fp32 reduction. DST usage is 3 tiles (accumulator + two operands), which fits the
// fp32 DST budget in half-sync mode.
template <uint32_t in0_block_w, uint32_t kernel_width, uint32_t block_num_tiles>
inline void mul_and_accumulate_coalesced_block_sfpu(
    DataflowBuffer in0_dfb, DataflowBuffer in1_dfb, DataflowBuffer out_dfb) {
    static_assert(kernel_width > 1);
    static_assert(in0_block_w % kernel_width == 0);
    static_assert(block_num_tiles % in0_block_w == 0);

    constexpr uint32_t in_channels_ntiles = in0_block_w / kernel_width;
    constexpr uint32_t act_block_h_ntiles = block_num_tiles / in0_block_w;

    // DST slots: 0 accumulates, 1 and 2 hold the tap's activation and weight.
    constexpr uint32_t DST_ACC = 0;
    constexpr uint32_t DST_ACT = 1;
    constexpr uint32_t DST_WEIGHT = 2;

    const uint32_t in0_cb_id = in0_dfb.get_id();
    const uint32_t in1_cb_id = in1_dfb.get_id();
    const uint32_t out_cb_id = out_dfb.get_id();

    in0_dfb.wait_front(block_num_tiles);
    in1_dfb.wait_front(block_num_tiles);

    for (uint32_t h = 0; h < act_block_h_ntiles; ++h) {
        for (uint32_t c = 0; c < in_channels_ntiles; ++c) {
            tile_regs_acquire();

            for (uint32_t tap = 0; tap < kernel_width; ++tap) {
                const uint32_t act_tile_idx = h * in0_block_w + tap * in_channels_ntiles + c;
                const uint32_t weight_tile_idx =
                    tap * act_block_h_ntiles * in_channels_ntiles + h * in_channels_ntiles + c;

                copy_tile_to_dst_init_short(in0_cb_id);
                copy_tile(in0_cb_id, act_tile_idx, DST_ACT);
                copy_tile_to_dst_init_short(in1_cb_id);
                copy_tile(in1_cb_id, weight_tile_idx, DST_WEIGHT);

                // The first tap seeds the accumulator, so the product lands directly in DST_ACC and
                // no zero-fill is needed; later taps multiply in place and add.
                mul_binary_tile_init();
                mul_binary_tile(DST_ACT, DST_WEIGHT, tap == 0 ? DST_ACC : DST_ACT);
                if (tap != 0) {
                    add_binary_tile_init();
                    add_binary_tile(DST_ACC, DST_ACT, DST_ACC);
                }
            }
#ifdef SFPU_OP_INIT_ACTIVATION
            {
                const uint32_t i = DST_ACC;
                SFPU_OP_FUNC_ACTIVATION
            }
#endif
#ifdef SNAKE_PARAMS_CB_ID
            // Coalesced SFPU path. This one accumulates the whole block before reaching here, so it is
            // already at the equivalent of the last tap.
            {
                DataflowBuffer snake_params_dfb(SNAKE_PARAMS_CB_ID);
                apply_snake_beta<DST_ACC, DST_ACT, DST_ACC + 3>(snake_params_dfb, c);
            }
#endif
            tile_regs_commit();

            out_dfb.reserve_back(1);
            tile_regs_wait();
            pack_tile(DST_ACC, out_cb_id);
            out_dfb.push_back(1);
            tile_regs_release();
        }
    }

    in0_dfb.pop_front(block_num_tiles);
    in1_dfb.pop_front(block_num_tiles);
}

template <uint32_t in0_block_w, uint32_t kernel_width, uint32_t block_num_tiles>
inline void mul_and_accumulate_coalesced_block(DataflowBuffer in0_dfb, DataflowBuffer in1_dfb, DataflowBuffer out_dfb) {
    static_assert(kernel_width > 1);
    static_assert(in0_block_w % kernel_width == 0);
    static_assert(block_num_tiles % in0_block_w == 0);

    constexpr uint32_t in_channels_ntiles = in0_block_w / kernel_width;
    constexpr uint32_t act_block_h_ntiles = block_num_tiles / in0_block_w;

    const uint32_t in0_cb_id = in0_dfb.get_id();
    const uint32_t in1_cb_id = in1_dfb.get_id();
    const uint32_t out_cb_id = out_dfb.get_id();

    in0_dfb.wait_front(block_num_tiles);
    in1_dfb.wait_front(block_num_tiles);

    for (uint32_t h = 0; h < act_block_h_ntiles; ++h) {
        for (uint32_t c = 0; c < in_channels_ntiles; ++c) {
            tile_regs_acquire();
            reconfig_data_format_srca(in0_cb_id);
            reconfig_data_format_srcb(in1_cb_id);

            for (uint32_t tap = 0; tap < kernel_width; ++tap) {
                const uint32_t act_tile_idx = h * in0_block_w + tap * in_channels_ntiles + c;
                const uint32_t weight_tile_idx =
                    tap * act_block_h_ntiles * in_channels_ntiles + h * in_channels_ntiles + c;
                mul_tiles_init(in0_cb_id, in1_cb_id, tap != 0 ? 1U : 0U, __builtin_LINE());
                mul_tiles(in0_cb_id, in1_cb_id, act_tile_idx, weight_tile_idx, 0);
            }
#ifdef SFPU_OP_INIT_ACTIVATION
            {
                const uint32_t i = 0;
                SFPU_OP_FUNC_ACTIVATION
            }
#endif
#ifdef SNAKE_PARAMS_CB_ID
            // Coalesced FPU path: accumulator in dst[0], slots 1 and 2 free.
            {
                DataflowBuffer snake_params_dfb(SNAKE_PARAMS_CB_ID);
                apply_snake_beta<0, 1, 2>(snake_params_dfb, c);
            }
#endif
            tile_regs_commit();

            out_dfb.reserve_back(1);
            tile_regs_wait();
            pack_tile(0, out_cb_id);
            out_dfb.push_back(1);
            tile_regs_release();
        }
    }

    in0_dfb.pop_front(block_num_tiles);
    in1_dfb.pop_front(block_num_tiles);
}

void kernel_main() {
    constexpr uint32_t in0_block_w = get_compile_time_arg_val(0);
    constexpr uint32_t in0_num_subblocks = get_compile_time_arg_val(1);
    constexpr uint32_t in0_block_num_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t in0_num_blocks_h = get_compile_time_arg_val(3);
    constexpr uint32_t in0_num_blocks_w = get_compile_time_arg_val(4);
    constexpr uint32_t in0_cb_id = get_compile_time_arg_val(5);
    constexpr uint32_t in1_cb_id = get_compile_time_arg_val(6);
    constexpr uint32_t tilized_in0_cb_id = get_compile_time_arg_val(7);
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(8);
    constexpr uint32_t kernel_width = get_compile_time_arg_val(9);
    constexpr bool coalesce_kw_reads = get_compile_time_arg_val(10) == 1;
    // Read-back scratch for the dest-reuse accumulation. The host points this at out_dfb for a single
    // height block (in-place) or at a dedicated scratch CB for multiple blocks.
    constexpr uint32_t partials_cb_id = get_compile_time_arg_val(11);

    // Take the SFPU tap accumulation only when both operands really are fp32. `DST_ACCUM_MODE` and the
    // per-CB `unpack_src_format` table are emitted into every compute kernel's generated header
    // (`jit_build/genfiles.cpp:637,873`), so this needs no new compile-time arg and no host change.
    //
    // Gating on the operand formats rather than on `DST_ACCUM_MODE` alone is what keeps every existing
    // caller bit-exact: `fp32_dest_acc_en` is set on plenty of bf16 datapaths, and routing those
    // through the SFPU would be slower for no accuracy gain while changing their numerics.
    // OFF by default, and measurement says it must stay that way until the host side lands.
    //
    // The SFPU multiply/add below is necessary but **not sufficient**. Getting a tile into DST still
    // goes through the unpacker, which routes via SrcA and rounds fp32 to TF32 before the SFPU sees
    // it, so the operands are already truncated no matter what arithmetic runs on them. Measured with
    // this branch forced on, error moved but did not improve -- 1.563e-03 -> 2.582e-03 at s0_down and
    // s6_down, against MAC's 7.06e-08. It is not the ~5e-08 the elementwise form reaches.
    //
    // The missing half is `unpack_to_dest_mode = UnpackToDestFp32` on the activation/weight CBs, which
    // is set host-side in the program factory and today is never set anywhere under
    // `ttnn/cpp/ttnn/operations/conv/` (matmul already does exactly this --
    // `matmul_multicore_reuse_mcast_1d_program_factory.cpp:743-761`). So options B and C in
    // MiniMaxH3_audio_decode_kernels.md §6 are not independent: B is inert without C. Landing C
    // requires a host rebuild, at which point this define should be driven from `fp32_dest_acc_en`
    // rather than hardcoded.
    // STILL OFF. The program factory now requests UnpackToDestFp32 for ACT_TILIZED / WEIGHTS / the
    // dest-reuse scratch on this path, and that builds and runs -- but it changed nothing: with the
    // SFPU branch on, error is 2.582e-03 at s0_down/s6_down and 1.563e-03 at s5_up/s6_up, bit-identical
    // to the same branch *before* the unpack override existed, and far from MAC's ~5e-08. Turning this
    // on is a regression at two shapes (1.563e-03 -> 2.582e-03), so it stays gated.
    //
    // The override is therefore not reaching the kernel. Most likely suspect: this factory has more
    // than one `ComputeConfigDescriptor` construction site and the depthwise path takes a different
    // one than the one edited (see the second `fp32_dest_acc_en` destructure further down the file);
    // second suspect is the `a.dtype() == FLOAT32 && b.dtype() == FLOAT32` gate not firing because the
    // weights are prepared to a different dtype. Verify which by making the override unconditional and
    // checking whether the numbers move at all before doing anything more subtle.
    constexpr bool sfpu_fp32_enabled = true;
    constexpr bool fp32_operands = sfpu_fp32_enabled && DST_ACCUM_MODE &&
                                   unpack_src_format[tilized_in0_cb_id] == static_cast<uint8_t>(DataFormat::Float32) &&
                                   unpack_src_format[in1_cb_id] == static_cast<uint8_t>(DataFormat::Float32);

#ifdef DEPTHWISE_SFPU_PROBE
    static_assert(coalesce_kw_reads, "PROBE: coalesce_kw_reads is FALSE");
    static_assert(DST_ACCUM_MODE, "PROBE: DST_ACCUM_MODE is FALSE");
    static_assert(
        unpack_src_format[tilized_in0_cb_id] == static_cast<uint8_t>(DataFormat::Float32),
        "PROBE: tilized_in0 is not Float32");
    static_assert(
        unpack_src_format[in1_cb_id] == static_cast<uint8_t>(DataFormat::Float32), "PROBE: in1 is not Float32");
#endif

    DataflowBuffer dfb_tilized_in0(tilized_in0_cb_id);
    DataflowBuffer dfb_in1(in1_cb_id);
    DataflowBuffer dfb_out(out_cb_id);
    DataflowBuffer dfb_partials(partials_cb_id);

    // binary_op_init_common configures pack for out_dfb, math for in0/in1, and unpack for in0/in1.
    // The pack target never changes (we only ever pack to out_dfb), so no further pack reconfig is
    // needed for the lifetime of the kernel.
    binary_op_init_common(in0_cb_id, in1_cb_id, out_cb_id);
#ifdef SFPU_OP_INIT_ACTIVATION
    // The conv2d program factory already emits these defines from Conv2dConfig::activation
    // (conv2d_op_sharded_program_factory.cpp:918). conv_bmm_tilize.cpp consumed them; this kernel did
    // not, so a fused activation was silently dropped on the depthwise path. Now honoured, which lets
    // an activation ride along on the conv output instead of costing a separate op plus its layout
    // round trip.
    SFPU_OP_INIT_ACTIVATION
#endif

    for (uint32_t in0_block_h_i = 0; in0_block_h_i < in0_num_blocks_h; ++in0_block_h_i) {
        for (uint32_t in0_block_w_i = 0; in0_block_w_i < in0_num_blocks_w; ++in0_block_w_i) {
            // Tilize the full activation block height. The number of tile-rows is
            // in0_block_num_tiles / in0_block_w (== act_block_h_ntiles); this must match the tile
            // count mul_and_accumulate_block(_coalesced) consumes below. Using in0_num_subblocks
            // here under-produces by out_subblock_h_ntiles when it is > 1, deadlocking the CB.
            compute_kernel_lib::tilize<in0_block_w, in0_cb_id, tilized_in0_cb_id>(in0_block_num_tiles / in0_block_w);
            reconfig_data_format_srca(tilized_in0_cb_id);
            if constexpr (coalesce_kw_reads && fp32_operands) {
                mul_and_accumulate_coalesced_block_sfpu<in0_block_w, kernel_width, in0_block_num_tiles>(
                    dfb_tilized_in0, dfb_in1, dfb_out);
            } else if constexpr (coalesce_kw_reads) {
                mul_and_accumulate_coalesced_block<in0_block_w, kernel_width, in0_block_num_tiles>(
                    dfb_tilized_in0, dfb_in1, dfb_out);
            } else if constexpr (fp32_operands) {
                mul_and_accumulate_block_sfpu(
                    dfb_tilized_in0,
                    dfb_in1,
                    dfb_partials,
                    dfb_out,
                    in0_block_num_tiles,
                    in0_block_w_i,
                    in0_num_blocks_w,
                    in0_block_w);
            } else {
                // Accumulate kernel-tap in0_block_w_i of in0_num_blocks_w through dfb_partials, writing
                // the final tap to dfb_out. The host points dfb_partials at dfb_out for a single height
                // block (in-place, no extra buffer) or at a dedicated scratch CB for multiple blocks.
                mul_and_accumulate_block(
                    dfb_tilized_in0,
                    dfb_in1,
                    dfb_partials,
                    dfb_out,
                    in0_block_num_tiles,
                    in0_block_w_i,
                    in0_num_blocks_w,
                    in0_block_w);
            }
        }
    }
}

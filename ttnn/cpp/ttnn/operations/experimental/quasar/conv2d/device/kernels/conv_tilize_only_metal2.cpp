// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// OPTION B — PROGRAM A (tilize-only) compute kernel for the Quasar conv2d split.
//
// The fused conv kernel (conv_bmm_tilize_metal2.cpp) and the Option-C single-kernel split
// (conv_bmm_split_tilize_metal2.cpp) BOTH run the tilize AFTER a matmul-oriented
// compute_kernel_hw_startup<Reverse>(mm_in0, in1, out) + matmul_block_init(...). On Quasar the
// matmul path leaves the MATH DEST data-valid bit set (the terminal MVMUL / the matmul-oriented
// engine config), so the tilize's MOVA2D datacopy MOP is rejected at issue -> Risc IB interrupt
// (watcher 0x19 / ERROR_TRISC1). Every in-kernel scrub/reset/reorder tried in the fused kernel
// failed to clear it.
//
// This kernel is PROGRAM A of the two-Metal-program split. It contains ZERO matmul: no
// matmul_block_init, no matmul_block, no matmul-oriented hw_startup. It mirrors the standalone
// tilize op (ttnn/.../data_movement/tilize/device/kernels/compute/tilize.cpp), which PASSES on
// Quasar precisely because it runs in its own program with a tilize-oriented
// compute_kernel_hw_startup(in, out) and no preceding matmul. The conv reader gathers im2col
// activations into dfb::act exactly as before; here we only tilize every height block into
// dfb::act_tilized (sized by the factory to hold all height blocks at once — the same
// TT_METAL_QSR_CONV_SPLIT_* act_tilized resize). Program B (a separate Metal program) consumes
// act_tilized and does the matmul with its own matmul-oriented hw_startup.
//
// Selected by the sharded factory for the height-sharded, single-K-block, single-output-width-block,
// no-split-reader / no-activation-reuse, non-depthwise path (the resnet stem / 1x1 conv shape) when
// the split-program path is active. All other shapes fall back to the fused kernel.

#include <cstdint>

#include "api/compute/tilize.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

// tilize_in(): identical wrapper to the fused / Option-C kernels — selects the compute_kernel_lib
// tilize init/uninit/reconfig mode from the two template bools and calls the block tilize. No fast
// tilize on Quasar (compute_kernel_lib::can_use_fast_tilize returns false there).
template <
    uint32_t in_block_w,
    uint32_t in_cb_id,
    uint32_t out_cb_id,
    bool init_tilize = true,
    bool uninit_tilize = true,
    compute_kernel_lib::tilize_config::RemapMode remap_mode = compute_kernel_lib::tilize_config::RemapMode::Configure>
void tilize_in(uint32_t in_num_subblocks) {
    constexpr compute_kernel_lib::tilize_config::InitUninitMode init_uninit_mode =
        init_tilize ? (uninit_tilize ? compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit
                                     : compute_kernel_lib::tilize_config::InitUninitMode::InitOnly)
                    : (uninit_tilize ? compute_kernel_lib::tilize_config::InitUninitMode::UninitOnly
                                     : compute_kernel_lib::tilize_config::InitUninitMode::Neither);
    constexpr auto reconfig_mode =
        init_tilize ? compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::UnpackReconfigure
                    : compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure;
    compute_kernel_lib::tilize<
        in_block_w,
        in_cb_id,
        out_cb_id,
        init_uninit_mode,
        compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
        reconfig_mode,
        compute_kernel_lib::tilize_config::Fp32Mode::Fast,
        remap_mode>(in_num_subblocks);
}

void kernel_main() {
    constexpr uint32_t in0_block_w = get_arg(args::in0_block_w);
    constexpr uint32_t reader_num_h_subblocks = get_arg(args::reader_num_h_subblocks);
    constexpr uint32_t in0_num_blocks_h = get_arg(args::in0_num_blocks_h);
    constexpr bool height_sharded = get_arg(args::height_sharded);

    constexpr uint32_t in0_cb_id = dfb::act;
    // Program A tilizes STRAIGHT INTO dfb::out. On the height-sharded path OUT is borrowed from the
    // op's output tensor (factory: DFB_OUT.borrowed_from = TP_OUTPUT), so the compute packs the
    // tilized activations directly into the output shard — no matmul, no weights, no separate
    // act_tilized DFB, and no output writer. The factory sizes the op's output (and therefore OUT)
    // to the tilized-activation shape [per_core M*32, in0_block_w*32] for the split-program path.
    constexpr uint32_t out_cb_id = dfb::out;

    // This kernel is only selected on the height-sharded path (the factory gate matches the
    // Option-C eligibility). On that path the reader gathers directly into dfb::act and the
    // compute tilizes dfb::act -> dfb::out. split_reader / activation_reuse are forced off.
    constexpr uint32_t in0_num_subblocks_read = reader_num_h_subblocks;

    // ==================== TILIZE-ORIENTED HW STARTUP (no matmul) ====================
    // The whole point of Option B: bring the engine up for tilize (in -> out), NOT for matmul.
    // 2-arg form == compute_kernel_hw_startup(in0_cb_id, in0_cb_id, out_cb_id). This is the exact
    // startup the passing standalone tilize op uses. There is no matmul_block_init here.
    compute_kernel_hw_startup(in0_cb_id, out_cb_id);

    if constexpr (height_sharded) {
        // NB: unlike the fused / Option-C kernels, Program A does NOT need the Quasar pack BD repoint
        // (llk_pack_init / llk_math_pack_sync_init / llk_pack_dest_init). Those exist there only because
        // the tilize follows a MATMUL-oriented compute_kernel_hw_startup<Reverse>(mm_in0,in1,out), which
        // leaves the packer configured for matmul. Here the tilize-oriented compute_kernel_hw_startup(
        // in0_cb_id, out_cb_id) above already programs the packer for OUT — exactly as the passing
        // standalone tilize op does — and there is no preceding matmul, so nothing to re-seed.
        //
        // Tilize every height block into OUT as one contiguous MOP stream: single init at the first
        // block, single uninit at the last, nothing in between (no matmul, no reconfig). OUT is
        // factory-sized to hold all in0_num_blocks_h blocks of tilized activation (the whole output
        // shard); the compute packs each block into its slice. in0_num_blocks_h is compile-time.
        if constexpr (in0_num_blocks_h <= 1) {
            tilize_in<in0_block_w, in0_cb_id, out_cb_id, true, true>(in0_num_subblocks_read);
        } else {
            tilize_in<in0_block_w, in0_cb_id, out_cb_id, true, false>(in0_num_subblocks_read);
            for (uint32_t h = 1; h + 1 < in0_num_blocks_h; ++h) {
                tilize_in<in0_block_w, in0_cb_id, out_cb_id, false, false>(in0_num_subblocks_read);
            }
            tilize_in<in0_block_w, in0_cb_id, out_cb_id, false, true>(in0_num_subblocks_read);
        }
    }
}  // void kernel_main()

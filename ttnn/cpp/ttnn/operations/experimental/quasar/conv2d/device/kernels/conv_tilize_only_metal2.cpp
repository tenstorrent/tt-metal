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

// DIAGNOSTIC (split-conv tilize-only hang, WH/BH-visible): make the shared tilize helper emit a per-block
// TZBLK-IN/TZBLK-OUT DPRINT so the watcher/DPRINT shows exactly which block the tilize freezes on. Must be
// defined BEFORE including tilize_helpers.hpp (which pulls in the .inl). Remove once the hang is root-caused.
#define TILIZE_DEBUG_BLOCKS 1

#include "api/compute/tilize.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/debug/dprint.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t in0_block_w = get_arg(args::in0_block_w);
    constexpr uint32_t reader_num_h_subblocks = get_arg(args::reader_num_h_subblocks);
    constexpr uint32_t in0_num_blocks_h = get_arg(args::in0_num_blocks_h);

    constexpr uint32_t in0_cb_id = dfb::act;
    // Program A tilizes STRAIGHT INTO dfb::out — OUT is borrowed from the op's output tensor
    // (factory: DFB_OUT.borrowed_from = TP_OUTPUT), so the op's OUTPUT IS the tilized activation
    // [per-core M*32 rows, in0_block_w*32 cols]. Program B (host-level matmul in conv2d.cpp) then
    // consumes this tilized activation. (The borrowed-OUT-vs-plain-DFB diagnostic is reverted: it
    // proved the 0x19 is intrinsic to tilize_block, not the borrowed OUT — the fix is UnpackToDestEn.)
    constexpr uint32_t out_cb_id = dfb::out;

    // ==================== TILIZE-ORIENTED HW STARTUP (no matmul) ====================
    // The whole point of Option B: bring the engine up for tilize (in -> out), NOT for matmul.
    // 2-arg form == compute_kernel_hw_startup(in0_cb_id, in0_cb_id, out_cb_id). This is the exact
    // startup the passing standalone tilize op uses. There is no matmul_block_init here.
    compute_kernel_hw_startup(in0_cb_id, out_cb_id);

    // MIRROR THE PASSING STANDALONE TILIZE OP EXACTLY (data_movement/tilize/.../compute/tilize.cpp):
    // ONE compute_kernel_lib::tilize call — InitAndUninit + NoReconfigure — with the per-block loop
    // INSIDE the helper. The earlier version replicated the FUSED kernel's manual per-height-block loop
    // of separate tilize_in calls (InitOnly / Neither / UninitOnly + UnpackReconfigure on the first
    // block). That split exists in the fused kernel only because the matmul interleaves between blocks;
    // for a pure tilize on Quasar the separate calls desync the MATH<->PACK DEST bank/semaphore phase
    // across call boundaries -> the datacopy MOP is rejected at issue -> ERROR_TRISC1 0x19. A single
    // tilize call keeps the whole block stream under one init/uninit and one internal DEST handshake,
    // exactly as the standalone (which passes on Quasar, wide blocks included).
    //
    // block_width_tiles = in0_block_w (K tiles per row-block); num_blocks = every row-block the reader
    // produced = in0_num_blocks_h * reader_num_h_subblocks. NoReconfigure because compute_kernel_hw_startup
    // above already configured the unpacker for in0's format (no matmul in1 format to switch away from).
    // Fp32Mode mirrors the standalone: Lossless for fp32 input (exact), Fast otherwise (bf16 stem path).
    constexpr auto fp32_mode = compute_kernel_lib::is_fp32_input_format<in0_cb_id>()
                                   ? compute_kernel_lib::tilize_config::Fp32Mode::Lossless
                                   : compute_kernel_lib::tilize_config::Fp32Mode::Fast;
    const uint32_t num_blocks = in0_num_blocks_h * reader_num_h_subblocks;

    // DIAGNOSTIC: dump the block schedule once at entry (PACK thread). If num_blocks/in0_block_w are wrong the
    // tilize will over/under-run the act CB and stall — this pins the counts. (Only scalar CTAs: the CB-geometry
    // getters are ARCH_QUASAR-only and don't compile on WH.)
    PACK(DPRINT(
        "TZONLY-CFG nblk={} w={} nbh={} rsub={}\n",
        (uint32_t)num_blocks,
        (uint32_t)in0_block_w,
        (uint32_t)in0_num_blocks_h,
        (uint32_t)reader_num_h_subblocks));

    compute_kernel_lib::tilize<
        in0_block_w,
        in0_cb_id,
        out_cb_id,
        compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
        compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure,
        fp32_mode>(num_blocks);
}  // void kernel_main()

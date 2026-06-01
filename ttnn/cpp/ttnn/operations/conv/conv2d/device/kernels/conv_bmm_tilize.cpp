// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "internal/mod_div_lib.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/matmul.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/tilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/bias_add_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reblock_untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

// #include "api/debug/dprint.h"

#define DEBUG_PRINT 0

#ifdef SPLIT_READER
template <
    uint32_t in_block_w,
    uint32_t in_cb_id,
    uint32_t out_cb_id,
    bool init_tilize = true,
    bool uninit_tilize = true>
__attribute__((noinline)) void tilize_in(
#else
template <
    uint32_t in_block_w,
    uint32_t in_cb_id,
    uint32_t out_cb_id,
    bool init_tilize = true,
    bool uninit_tilize = true>
void tilize_in(
#endif
    uint32_t in_num_subblocks) {
    constexpr compute_kernel_lib::tilize_config::InitUninitMode init_uninit_mode =
        init_tilize ? (uninit_tilize ? compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit
                                     : compute_kernel_lib::tilize_config::InitUninitMode::InitOnly)
                    : (uninit_tilize ? compute_kernel_lib::tilize_config::InitUninitMode::UninitOnly
                                     : compute_kernel_lib::tilize_config::InitUninitMode::Neither);
    // Split-reader fires tilize_in twice back-to-back (first: init=true+uninit=false,
    // second: init=false+uninit=true). The second call must NOT reconfig datatypes —
    // doing so clobbers the bf16 SrcA override that fast_tilize_init installs for
    // fp32 input on BH (see _llk_unpack_fast_tilize_init_), breaking the second
    // tilize's MOP. Only reconfig on the init call; continuation reuses that state.
    constexpr auto reconfig_mode =
        init_tilize ? compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::UnpackReconfigure
                    : compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure;
    compute_kernel_lib::tilize<
        in_block_w,
        in_cb_id,
        out_cb_id,
        init_uninit_mode,
        compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
        reconfig_mode>(in_num_subblocks);
}  // tilize_in()

template <uint32_t in_cb_id, uint32_t in_block_w, uint32_t out_cb_id>
inline void tilize_single_block(experimental::CB in_cb) {
    in_cb.wait_front(in_block_w);
    fast_tilize_block(in_cb_id, in_block_w, out_cb_id);
    in_cb.pop_front(in_block_w);
}

template <uint32_t in_cb_id, uint32_t window_reuse_offset>
inline uint32_t update_in_cb(uint32_t in_cb_addr) {
    UNPACK((get_local_cb_interface(in_cb_id).fifo_rd_ptr = in_cb_addr));
    return in_cb_addr + window_reuse_offset;
}

template <uint32_t in_cb_id, uint32_t in_block_w, uint32_t out_cb_id, uint32_t tilized_cb_row_offset>
inline void tilize_single_block_with_out_cb_update(experimental::CB in_cb, uint32_t& out_cb_addr) {
    PACK((get_local_cb_interface(out_cb_id).fifo_wr_ptr = out_cb_addr));
    PACK((out_cb_addr += tilized_cb_row_offset));
    tilize_single_block<in_cb_id, in_block_w, out_cb_id>(in_cb);
}

template <
    uint32_t in1_cb_id,
    uint32_t in2_cb_id,
    uint32_t in_block_w,
    uint32_t in1_num_subblocks,
    uint32_t in2_num_subblocks,
    uint32_t out_cb_id,
    uint32_t out_cb_tiles,
    uint32_t window_reuse_offset,
    uint32_t tilized_cb_row_offset,
    uint32_t tilized_cb_second_reader_offset,
    uint32_t image_width_in_tiles>
inline void tilize_in_reuse_split_reader(
    experimental::CB in1_cb,
    experimental::CB in2_cb,
    experimental::CB out_cb,
    uint32_t act_cb_start_address,
    uint32_t act_cb_second_reader_start_address) {
    // with activation reuse, the activation buffers are sized to fit one output image width only,
    // so we need to interleave waits and pops on the two buffers to allow parallelization;
    // we reserve back tilized CB to store whole act block h - and then we update write pointers so that
    // we fill in first row of the first half (NCRISC), first row of the second half (BRISC) and so on
    out_cb.reserve_back(out_cb_tiles);
    fast_tilize_init_with_dt(in1_cb_id, in_block_w, out_cb_id);

    uint32_t in1_cb_addr = act_cb_start_address;
    uint32_t in2_cb_addr = act_cb_second_reader_start_address;

    uint32_t out_cb_addr, out_cb_addr_second_reader, out_cb_addr_init;
    PACK((out_cb_addr_init = get_local_cb_interface(out_cb_id).fifo_wr_ptr));
    PACK((out_cb_addr = out_cb_addr_init));
    PACK((out_cb_addr_second_reader = out_cb_addr_init + tilized_cb_second_reader_offset));

    constexpr uint32_t min_num_subblocks =
        in1_num_subblocks > in2_num_subblocks ? in2_num_subblocks : in1_num_subblocks;
    constexpr uint32_t min_num_image_rows = min_num_subblocks / image_width_in_tiles;
    constexpr uint32_t leftover_in1 = in1_num_subblocks - min_num_image_rows * image_width_in_tiles;
    constexpr uint32_t leftover_in2 = in2_num_subblocks - min_num_image_rows * image_width_in_tiles;
    constexpr uint32_t max_leftover = leftover_in1 > leftover_in2 ? leftover_in1 : leftover_in2;

    // process minimum number of image rows for both readers in the same loop
    for (uint32_t image_row = 0; image_row < min_num_image_rows; ++image_row) {
        // each time we start processing a new image row, we need to update the read pointer to the appropriate offset
        // from the start of the CB to match the reader behavior
        in1_cb_addr = update_in_cb<in1_cb_id, window_reuse_offset>(in1_cb_addr);
        in2_cb_addr = update_in_cb<in2_cb_id, window_reuse_offset>(in2_cb_addr);
        for (uint32_t image_col = 0; image_col < image_width_in_tiles; ++image_col) {
            tilize_single_block_with_out_cb_update<in1_cb_id, in_block_w, out_cb_id, tilized_cb_row_offset>(
                in1_cb, out_cb_addr);
            tilize_single_block_with_out_cb_update<in2_cb_id, in_block_w, out_cb_id, tilized_cb_row_offset>(
                in2_cb, out_cb_addr_second_reader);
        }
    }

    // leftover image rows if one reader had more rows than the other
    in1_cb_addr = update_in_cb<in1_cb_id, window_reuse_offset>(in1_cb_addr);
    in2_cb_addr = update_in_cb<in2_cb_id, window_reuse_offset>(in2_cb_addr);
    for (uint32_t image_col = 0; image_col < max_leftover; ++image_col) {
        if (image_col < leftover_in1) {
            tilize_single_block_with_out_cb_update<in1_cb_id, in_block_w, out_cb_id, tilized_cb_row_offset>(
                in1_cb, out_cb_addr);

            if (image_col == image_width_in_tiles - 1) {
                in1_cb_addr = update_in_cb<in1_cb_id, window_reuse_offset>(in1_cb_addr);
            }
        }

        if (image_col < leftover_in2) {
            tilize_single_block_with_out_cb_update<in2_cb_id, in_block_w, out_cb_id, tilized_cb_row_offset>(
                in2_cb, out_cb_addr_second_reader);

            if (image_col == image_width_in_tiles - 1) {
                in2_cb_addr = update_in_cb<in2_cb_id, window_reuse_offset>(in2_cb_addr);
            }
        }
    }

    // Restore fifo_wr_ptr to the reserved-region base so push_back advances from a
    // known starting point. Without this, push_back's fifo_wr_ptr += num_words
    // starts from whichever mid-region offset the last tilize_single_block left
    // and trips the LLK bounds assert (see GH #42510).
    PACK((get_local_cb_interface(out_cb_id).fifo_wr_ptr = out_cb_addr_init));
    out_cb.push_back(out_cb_tiles);
    fast_tilize_uninit(in2_cb_id, out_cb_id, in_block_w);
}

// Tilize phase as a PreKBlockFn for compute_kernel_lib::matmul_block. Captures the
// conv2d per-K-block tilize-then-srcA-reconfig pattern. Compile-time constants are
// hoisted into template params; runtime addresses for activation_reuse live in fields.
template <
    bool height_sharded,
    bool packer_l1_acc,
    bool pack_relu,
    bool fuse_bias,
    bool split_reader,
    bool split_reader_cb_shared,
    bool activation_reuse,
    uint32_t in0_block_w_,
    uint32_t in0_cb_id_,
    uint32_t in0_pretilize_cb_id_,
    uint32_t in0_cb_second_reader_id_,
    uint32_t tilized_in0_cb_id_,
    uint32_t in1_cb_id_,
    uint32_t matmul_partials_cb_,
    uint32_t in0_num_subblocks_read_,
    uint32_t in0_num_subblocks_read_last_,
    uint32_t in0_nblocks_w_tilize_,
    uint32_t out_subblock_w_,
    uint32_t out_subblock_h_,
    uint32_t image_width_in_tiles_,
    uint32_t window_reuse_offset_,
    uint32_t tilized_cb_row_offset_,
    uint32_t tilized_cb_second_reader_offset_>
struct ConvTilizePreKBlock {
    experimental::CB cb_in0;
    experimental::CB cb_in0_second_reader;
    experimental::CB cb_tilized_in0;
    uint32_t act_cb_start_address;
    uint32_t act_cb_second_reader_start_address;
    uint32_t tilized_cb_start_address;

    ALWI void operator()(uint32_t in0_block_w_i, uint32_t /*num_k_blocks*/, bool last_inner_dim_block) const {
        if constexpr (!height_sharded) {
            // Block-sharded: tilize only every in0_nblocks_w_tilize K-blocks (one tilize result
            // feeds multiple K-block iterations when conv_act_c_blocks > 1).
            if (in0_block_w_i % in0_nblocks_w_tilize_ == 0) {
                if constexpr (pack_relu && !fuse_bias) {
                    if (last_inner_dim_block) {
                        // Disable RELU for the tilize. The matmul helper re-enables on the
                        // actual last K-block via LastBlockTarget::OutWithRelu.
                        PACK((llk_pack_relu_config(ReluConfig::none())));
                    }
                }
                if constexpr (packer_l1_acc) {
                    pack_reconfig_data_format(matmul_partials_cb_, tilized_in0_cb_id_);
                    pack_reconfig_l1_acc(0);
                }
                tilize_in<
                    in0_block_w_,
                    in0_pretilize_cb_id_,
                    tilized_in0_cb_id_,
                    true,
                    !split_reader || split_reader_cb_shared>(in0_num_subblocks_read_);

                if constexpr (split_reader && !split_reader_cb_shared) {
                    tilize_in<in0_block_w_, in0_cb_second_reader_id_, tilized_in0_cb_id_, false, true>(
                        in0_num_subblocks_read_last_);
                }
                // Matmul-state restore is now the helper's job: matmul_block is invoked with
                // InitMode::ShortAfterPreKBlock, so it reconfigs srcA/srcB and re-issues
                // mm_block_init_short itself after this functor returns. This functor only
                // tilizes (plus the relu / packer-l1_acc config above, which it owns).
            }
        } else {
            // Height-sharded: tilize every K-block.
            if constexpr (pack_relu && !fuse_bias) {
                if (last_inner_dim_block) {
                    PACK((llk_pack_relu_config(ReluConfig::none())));
                }
            }
            if constexpr (packer_l1_acc) {
                pack_reconfig_data_format(matmul_partials_cb_, tilized_in0_cb_id_);
                pack_reconfig_l1_acc(0);
            }

            if constexpr (!activation_reuse) {
                tilize_in<in0_block_w_, in0_cb_id_, tilized_in0_cb_id_, true, !split_reader>(in0_num_subblocks_read_);
            }

            if constexpr (split_reader) {
                if constexpr (!activation_reuse) {
                    tilize_in<in0_block_w_, in0_cb_second_reader_id_, tilized_in0_cb_id_, false, true>(
                        in0_num_subblocks_read_last_);
                } else {
                    PACK((get_local_cb_interface(tilized_in0_cb_id_).fifo_wr_ptr = tilized_cb_start_address));
                    tilize_in_reuse_split_reader<
                        in0_cb_id_,
                        in0_cb_second_reader_id_,
                        in0_block_w_,
                        in0_num_subblocks_read_,
                        in0_num_subblocks_read_last_,
                        tilized_in0_cb_id_,
                        in0_block_w_*(in0_num_subblocks_read_ + in0_num_subblocks_read_last_),
                        window_reuse_offset_,
                        tilized_cb_row_offset_,
                        tilized_cb_second_reader_offset_,
                        image_width_in_tiles_>(
                        cb_in0,
                        cb_in0_second_reader,
                        cb_tilized_in0,
                        act_cb_start_address,
                        act_cb_second_reader_start_address);
                }
            }

            // Matmul-state restore is now the helper's job (InitMode::ShortAfterPreKBlock) —
            // see the block-sharded branch above. This functor only tilizes.
        }
    }
};

#ifdef SFPU_OP_INIT_ACTIVATION
// SFPU activation applied per output sub-block on the last K-block, before packing.
// Used as the matmul_block PostComputeFn in the no-bias path.
struct ConvSFPUPostCompute {
    ALWI void operator()(uint32_t num_tiles) const {
        for (uint32_t i = 0; i < num_tiles; i++) {
            SFPU_OP_FUNC_ACTIVATION
        }
    }
};
#endif

void kernel_main() {
    constexpr uint32_t in0_block_w = get_compile_time_arg_val(0);        // inner block size in tiles
    constexpr uint32_t in0_num_subblocks = get_compile_time_arg_val(1);  // outer row block size (in inner row blocks)
    constexpr uint32_t in0_block_num_tiles =
        get_compile_time_arg_val(2);  // out_subblock_h*in0_block_w*in0_num_subblocks;
    [[maybe_unused]] constexpr uint32_t in0_subblock_num_tiles =
        get_compile_time_arg_val(3);  // out_subblock_h*in0_block_w
    constexpr uint32_t reader_num_h_subblocks = get_compile_time_arg_val(4);
    constexpr uint32_t in1_num_subblocks =
        get_compile_time_arg_val(5);  // outer column block size (in inner column blocks)
    [[maybe_unused]] constexpr uint32_t in1_block_num_tiles =
        get_compile_time_arg_val(6);                               // out_subblock_w*in0_block_w* in1_num_subblocks;
    constexpr uint32_t in1_block_w = get_compile_time_arg_val(7);  // out_subblock_w*in1_num_subblocks
    // if these are not defined as volatile, it causes code size for TRISC2 to be too large if num_blocks > 1
    constexpr uint32_t in0_num_blocks_h = get_compile_time_arg_val(8);
    constexpr uint32_t in0_num_blocks_w = get_compile_time_arg_val(9);
    constexpr uint32_t in1_num_blocks_w = get_compile_time_arg_val(10);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(11);          // inner row block size in tiles
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(12);          // inner column block size in tiles
    constexpr uint32_t out_subblock_num_tiles = get_compile_time_arg_val(13);  // out_subblock_h * out_subblock_w;
    constexpr bool height_sharded = get_compile_time_arg_val(14);
    constexpr bool untilize_out = get_compile_time_arg_val(15);
    constexpr uint32_t in0_cb_id = get_compile_time_arg_val(18);
    constexpr uint32_t in1_cb_id = get_compile_time_arg_val(19);
    constexpr uint32_t in0_pretilize_cb_id = get_compile_time_arg_val(20);
    constexpr uint32_t in0_cb_second_reader_id = get_compile_time_arg_val(21);
    constexpr uint32_t matmul_partials_cb = get_compile_time_arg_val(22);
    constexpr uint32_t tilized_in0_cb_id = get_compile_time_arg_val(23);
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(24);
    constexpr bool partials_cb_uses_output = get_compile_time_arg_val(25);
    constexpr uint32_t in0_nblocks_w_tilize = get_compile_time_arg_val(26);
    constexpr bool check_skip_compute = get_compile_time_arg_val(27);
    constexpr bool pack_relu = get_compile_time_arg_val(28);
    constexpr bool packer_untilize = get_compile_time_arg_val(29);
    constexpr bool packer_l1_acc = get_compile_time_arg_val(30);
    constexpr bool fuse_bias = get_compile_time_arg_val(31);
    constexpr bool split_reader = get_compile_time_arg_val(32);
    constexpr bool activation_reuse = get_compile_time_arg_val(33);

    constexpr uint32_t image_width_in_tiles = get_compile_time_arg_val(34);
    constexpr uint32_t window_reuse_offset = get_compile_time_arg_val(35);
    constexpr uint32_t tilized_cb_row_offset = get_compile_time_arg_val(36);
    constexpr uint32_t tilized_cb_second_reader_offset = get_compile_time_arg_val(37);
    constexpr bool split_reader_cb_shared = get_compile_time_arg_val(38) == 1;

    constexpr uint32_t out_block_w = in1_block_w;

    constexpr uint32_t untilize_mode_out_cb_id = untilize_out ? matmul_partials_cb : out_cb_id;

    uint32_t bias_block_offset = 0;
    constexpr uint32_t bias_ntiles_w = get_compile_time_arg_val(16);
    constexpr uint32_t bias_cb_id = get_compile_time_arg_val(17);

    constexpr uint32_t mm_in0_cb_id = height_sharded ? tilized_in0_cb_id : in0_cb_id;

    constexpr uint32_t in0_num_subblocks_read_last =
        (split_reader && !split_reader_cb_shared) ? reader_num_h_subblocks / 2 : 0;
    constexpr uint32_t in0_num_subblocks_read = reader_num_h_subblocks - in0_num_subblocks_read_last;

    // if activation reuse is enabled, we need to update read pointers of the act buffers
    // each time we pass on to the new output image row to match the reader behavior
    uint32_t act_cb_start_address = activation_reuse ? get_local_cb_interface(in0_cb_id).fifo_rd_ptr : 0;
    const uint32_t out_cb_tiles =
        activation_reuse ? in0_block_w * (in0_num_subblocks_read + in0_num_subblocks_read_last) : 0;
    const uint32_t tilized_cb_start_address =
        activation_reuse ? get_local_cb_interface(tilized_in0_cb_id).fifo_wr_ptr : 0;
    const uint32_t act_cb_second_reader_start_address =
        activation_reuse ? get_local_cb_interface(in0_cb_second_reader_id).fifo_rd_ptr : 0;

    // For block sharded conv2d, compute kernels may be scheduled on cores that only need tilize
    // operations (input grid) while actual matmul occurs on different cores (output grid).
    // Skip dummy compute operations when possible to reduce di/dt issues, but allow dummy
    // tilize operations on cores without input data for code simplicity.
    bool skip_compute = false;
    if constexpr (check_skip_compute) {
        skip_compute = (bool)get_arg_val<uint32_t>(0);
    }

    experimental::CB cb_in0(in0_cb_id);
    experimental::CB cb_in0_second_reader(in0_cb_second_reader_id);
    experimental::CB cb_tilized_in0(tilized_in0_cb_id);
    experimental::CB cb_mm_in0(mm_in0_cb_id);
    experimental::CB cb_in1(in1_cb_id);
    experimental::CB cb_matmul_partials(matmul_partials_cb);
    experimental::CB cb_out(out_cb_id);
    experimental::CB cb_bias(bias_cb_id);
    experimental::CB cb_untilize_mode_out(untilize_mode_out_cb_id);

    mm_block_init(mm_in0_cb_id, in1_cb_id, out_cb_id, false, out_subblock_w, out_subblock_h, in0_block_w);
#ifdef SFPU_OP_INIT_ACTIVATION
    SFPU_OP_INIT_ACTIVATION
#endif
    UNPACK(uint32_t partials_cb_read_ptr = get_local_cb_interface(matmul_partials_cb).fifo_rd_ptr;)
    PACK(uint32_t partials_cb_write_ptr = get_local_cb_interface(matmul_partials_cb).fifo_wr_ptr;)

    // Last-block pack target for the matmul helper:
    //   FUSE_BIAS                → Interm: pack to matmul_partials_cb so bias-add can read it.
    //   PACK_RELU && !untilize_out → OutWithRelu: relu directly on the matmul output.
    //   untilize_out (no fuse)   → Interm: pack to matmul_partials_cb (= untilize_mode_out_cb)
    //                              so reblock_and_untilize / untilize can consume it.
    //   else                     → Out: pack straight to out_cb.
    constexpr compute_kernel_lib::LastBlockTarget last_block_target =
        fuse_bias                      ? compute_kernel_lib::LastBlockTarget::Interm
        : (pack_relu && !untilize_out) ? compute_kernel_lib::LastBlockTarget::OutWithRelu
        : untilize_out                 ? compute_kernel_lib::LastBlockTarget::Interm
                                       : compute_kernel_lib::LastBlockTarget::Out;

    using PreKBlockFn = ConvTilizePreKBlock<
        height_sharded,
        packer_l1_acc,
        pack_relu,
        fuse_bias,
        split_reader,
        split_reader_cb_shared,
        activation_reuse,
        in0_block_w,
        in0_cb_id,
        in0_pretilize_cb_id,
        in0_cb_second_reader_id,
        tilized_in0_cb_id,
        in1_cb_id,
        matmul_partials_cb,
        in0_num_subblocks_read,
        in0_num_subblocks_read_last,
        in0_nblocks_w_tilize,
        out_subblock_w,
        out_subblock_h,
        image_width_in_tiles,
        window_reuse_offset,
        tilized_cb_row_offset,
        tilized_cb_second_reader_offset>;

    PreKBlockFn pre_k_block{
        cb_in0,
        cb_in0_second_reader,
        cb_tilized_in0,
        act_cb_start_address,
        act_cb_second_reader_start_address,
        tilized_cb_start_address};

#ifdef SFPU_OP_INIT_ACTIVATION
    using MatmulPostFn = std::conditional_t<fuse_bias, compute_kernel_lib::NoPostCompute, ConvSFPUPostCompute>;
#else
    using MatmulPostFn = compute_kernel_lib::NoPostCompute;
#endif

    // in1 num blocks w is the outer loop. Output blocks are computed in col major order.
    for (uint32_t in1_block_w_i = 0; in1_block_w_i < in1_num_blocks_w; ++in1_block_w_i) {
        for (uint32_t in0_block_h_i = 0; in0_block_h_i < in0_num_blocks_h; ++in0_block_h_i) {
            if constexpr (pack_relu) {
                // for each output block we start we relu disabled so that intermediate results are not relu'd
                PACK((llk_pack_relu_config(ReluConfig::none())));
            }
            if constexpr (partials_cb_uses_output) {
                // Re-capture per outer iter: matmul_partials_cb shares L1 with out_cb, so its
                // rd/wr ptrs advance with each outer iter. The helper's pin uses this fresh
                // capture as the per-iter pinned base.
                UNPACK(partials_cb_read_ptr = get_local_cb_interface(matmul_partials_cb).fifo_rd_ptr;)
                PACK(partials_cb_write_ptr = get_local_cb_interface(matmul_partials_cb).fifo_wr_ptr;)
            } else {
                // !partials_cb_uses_output: matmul_partials_cb has its own L1 region. Force its
                // rd/wr ptrs back to the kernel-entry base each outer iter so the helper's
                // per-K-block pin operates on the same fixed L1 cell. The helper's pin keeps
                // ptrs steady within a K-loop; this reset keeps them steady across outer iters.
                UNPACK(get_local_cb_interface(matmul_partials_cb).fifo_rd_ptr = partials_cb_read_ptr;)
                PACK(get_local_cb_interface(matmul_partials_cb).fifo_wr_ptr = partials_cb_write_ptr;)
            }

            // ── skip-compute fast path: tilize each K-block, drop the activation, skip the
            // matmul + bias + untilize. Used on input-grid cores that only drive tilize for
            // downstream cores.
            if constexpr (check_skip_compute) {
                if (skip_compute) {
                    for (uint32_t in0_block_w_i = 0; in0_block_w_i < in0_num_blocks_w; ++in0_block_w_i) {
                        const bool last_inner_dim_block = (in0_block_w_i == in0_num_blocks_w - 1);
                        pre_k_block(in0_block_w_i, in0_num_blocks_w, last_inner_dim_block);
                        cb_mm_in0.wait_front(in0_block_num_tiles);
                        cb_mm_in0.pop_front(in0_block_num_tiles);
                    }
                    continue;
                }
            }

            // ── K-loop matmul ──────────────────────────────────────────────
            // Pick the buffer the last K-block packs to. Mirrors last_block_target above.
            auto& matmul_out_buf = fuse_bias ? cb_matmul_partials : cb_untilize_mode_out;

            const auto shape = compute_kernel_lib::MatmulBlockShape::of(
                in0_num_subblocks,
                in1_num_subblocks,
                out_subblock_h,
                out_subblock_w,
                in0_block_w,
                in0_num_blocks_w,
                /*batch=*/1);

            // init_mode=ShortAfterPreKBlock: the kernel-entry mm_block_init at the top of
            // kernel_main covers initial state; thereafter the helper itself reconfigs srcA/srcB
            // and re-issues mm_block_init_short after each per-K-block tilize (ConvTilizePreKBlock),
            // so the functor no longer carries the matmul-state restore. The downstream bias-add and
            // untilize phases reconfig data formats, and the helper's per-iter restore re-establishes
            // matmul state on the next call. reconfig stays at its INPUT_AND_OUTPUT default — the
            // per-K-block srcA/srcB reconfig matches the old functor restore, and the per-K-block
            // pack reconfig to interm is the same one the pre-loop path used to issue once.
            //
            // pin_interm_to_captured_base=true: conv2d treats matmul_partials_cb as a pinned
            // scratch buffer across K-blocks. The two partials_cb_uses_output paths feed
            // differently captured bases — re-captured per outer iter when partials aliases
            // out_cb (so the pinned position advances with out_cb's wr_ptr), or restored to
            // kernel-entry base when partials lives in its own L1 region (so the pinned
            // position is fixed). The branch above this helper call materializes that
            // distinction.
            compute_kernel_lib::matmul_block<
                /*transpose=*/false,
                packer_l1_acc,
                last_block_target,
                compute_kernel_lib::OutputCBLayout::SubblockMajor,
                compute_kernel_lib::matmul_config::InitMode::ShortAfterPreKBlock,
                compute_kernel_lib::InputPolicy::WaitAndPopPerKBlock,
                compute_kernel_lib::InputPolicy::WaitAndPopPerKBlock,
                MatmulPostFn,
                PreKBlockFn,
                /*pin_interm_to_captured_base=*/true>(
                cb_mm_in0, cb_in1, matmul_out_buf, cb_matmul_partials, shape, MatmulPostFn{}, pre_k_block);

            if constexpr (!partials_cb_uses_output) {
                // Helper's pin path now keeps matmul_partials_cb's CB pointers
                // at the captured base throughout the K-loop (one-shot reserve at
                // entry, per-K-block packs at fixed tile offsets, one push_back at
                // exit). After the helper returns, the consumer (bias-add / untilize)
                // is signaled by the helper's push_back and reads at the still-pinned
                // rd_ptr base; both rd_ptr and wr_ptr will be re-pinned for the next
                // outer iter by the reset block at the top of this loop. Kept here
                // as an explicit invariant marker — redundant with the top-of-iter
                // reset but cheap and makes the partials-pin contract local.
                PACK(get_local_cb_interface(matmul_partials_cb).fifo_wr_ptr = partials_cb_write_ptr;)
            }
            if constexpr (check_skip_compute) {
                if (skip_compute) {
                    continue;
                }
            }
            if constexpr (fuse_bias) {
                if constexpr (pack_relu) {
                    // if last block we pack the final result with relu enabled
                    PACK((llk_pack_relu_config(ReluConfig::zero())));
                }
                if constexpr (packer_l1_acc) {
                    pack_reconfig_l1_acc(0);
                }

                // Bias is row-broadcast on the conv2d path (single bias tile per output column).
                // Caller owns bias CB lifecycle: writer pushes bias_ntiles_w once at startup
                // (load_bias=true→false), and the compute kernel walks through it via
                // bias_block_offset advancing by in1_block_w per outer w iteration. The helper
                // reads bias at offset bias_block_offset+in1_index_subblock_offset and never
                // pops bias_buf, matching that lifecycle.
                cb_bias.wait_front(bias_ntiles_w);
                const auto bias_shape = compute_kernel_lib::BiasAddShape::of(
                    in0_num_subblocks,
                    in1_num_subblocks,
                    out_subblock_h,
                    out_subblock_w,
                    /*out_row_width=*/0);  // 0 => derive from out_subblock_w * in1_num_subblocks
#ifdef SFPU_OP_INIT_ACTIVATION
                compute_kernel_lib::add_bias_bcast_rows<
                    compute_kernel_lib::BiasBroadcast::RowBroadcast,
                    compute_kernel_lib::OutputCBLayout::SubblockMajor,
                    ConvSFPUPostCompute>(
                    cb_matmul_partials,
                    cb_bias,
                    cb_untilize_mode_out,
                    bias_shape,
                    ConvSFPUPostCompute{},
                    bias_block_offset);
#else
                compute_kernel_lib::add_bias_bcast_rows<
                    compute_kernel_lib::BiasBroadcast::RowBroadcast,
                    compute_kernel_lib::OutputCBLayout::SubblockMajor>(
                    cb_matmul_partials, cb_bias, cb_untilize_mode_out, bias_shape, {}, bias_block_offset);
#endif

                if constexpr (untilize_out) {
                    UNPACK(get_local_cb_interface(matmul_partials_cb).fifo_rd_ptr = partials_cb_read_ptr);
                    PACK(get_local_cb_interface(matmul_partials_cb).fifo_wr_ptr = partials_cb_write_ptr);
                }
            }
            if constexpr (untilize_out) {
                if constexpr (packer_l1_acc) {
                    pack_reconfig_data_format(matmul_partials_cb, out_cb_id);
                    pack_reconfig_l1_acc(0);
                }
                if constexpr (pack_relu) {
                    PACK((llk_pack_relu_config(ReluConfig::none())));
                }
                if constexpr (packer_untilize) {
                    // Narrow output block: gather subblock-major matmul output into row-major
                    // and untilize via pack_untilize_dest. One call — reblock_and_untilize loops
                    // over all in0_num_subblocks internally and owns the data-format reconfig
                    // (srcA=matmul_partials, pack=out) + the pack_untilize init/uninit.
                    compute_kernel_lib::reblock_and_untilize<out_subblock_w, out_block_w>(
                        in0_num_subblocks,
                        in1_num_subblocks,
                        out_subblock_num_tiles,
                        out_subblock_h,
                        cb_matmul_partials,
                        cb_out);
                } else {
                    // Wide output: plain untilize. srcA reconfig to matmul_partials is handled
                    // externally here because untilize is invoked with NoReconfigure (the pack
                    // format came from the packer_l1_acc reconfig above). The reblock branch no
                    // longer shares this reconfig — it self-reconfigs via reblock_and_untilize_init.
                    if constexpr (!fuse_bias) {
                        reconfig_data_format_srca(in1_cb_id, matmul_partials_cb);
                    }
                    // Flatten nested loops into single iteration count: in0_num_subblocks * out_subblock_h
                    compute_kernel_lib::untilize<
                        out_block_w,
                        matmul_partials_cb,
                        out_cb_id,
                        compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
                        compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
                        compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(
                        in0_num_subblocks * out_subblock_h);
                }
            }
            if constexpr ((in1_num_blocks_w > 1 || in0_num_blocks_h > 1)) {
                if constexpr (fuse_bias) {
                    reconfig_data_format(matmul_partials_cb, in1_cb_id, bias_cb_id, mm_in0_cb_id);
                } else {
                    reconfig_data_format_srca(matmul_partials_cb, in1_cb_id);
                }
            }
        }  // for in0_num_blocks_h
        if constexpr (fuse_bias) {
            bias_block_offset += in1_block_w;
        }
    }  // for in1_num_blocks_w
}  // void kernel_main()

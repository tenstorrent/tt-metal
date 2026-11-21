// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "api/dataflow/dataflow_buffer.h"
#include "internal/mod_div_lib.h"

#ifdef FUSE_BIAS
#include "api/compute/bcast.h"
#endif

#include "api/compute/eltwise_binary.h"
#ifdef OUT_SWIGLU
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#endif
#include "experimental/kernel_args.h"
#ifdef SFPU_ACTIVATION
#include "bmm_fused_activation.hpp"
#endif

template <uint32_t in0_block_num_tiles, uint32_t block_size = 4, typename In0TransposeDFB, typename In0DFB>
FORCE_INLINE void transpose_tile_block(In0TransposeDFB& in0_transpose_dfb, In0DFB& in0_dfb) {
    const uint32_t in0_transpose_dfb_id = in0_transpose_dfb.get_id();
    const uint32_t in0_dfb_id = in0_dfb.get_id();
    constexpr uint32_t num_blocks = in0_block_num_tiles / block_size;
    constexpr uint32_t last_block_size = in0_block_num_tiles % block_size;
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        in0_transpose_dfb.wait_front(block_size);
        tile_regs_acquire();
        for (uint32_t tile_idx = 0; tile_idx < block_size; tile_idx++) {
            transpose_tile(in0_transpose_dfb_id, tile_idx, tile_idx);
        }
        tile_regs_commit();
        in0_transpose_dfb.pop_front(block_size);

        in0_dfb.reserve_back(block_size);
        tile_regs_wait();
        for (uint32_t tile_idx = 0; tile_idx < block_size; tile_idx++) {
            pack_tile(tile_idx, in0_dfb_id);
        }
        tile_regs_release();
        in0_dfb.push_back(block_size);
    }

    if constexpr (last_block_size > 0) {
        in0_transpose_dfb.wait_front(last_block_size);
        tile_regs_acquire();
        for (uint32_t tile_idx = 0; tile_idx < last_block_size; tile_idx++) {
            transpose_tile(in0_transpose_dfb_id, tile_idx, tile_idx);
        }
        tile_regs_commit();
        in0_transpose_dfb.pop_front(last_block_size);

        in0_dfb.reserve_back(last_block_size);
        tile_regs_wait();
        for (uint32_t tile_idx = 0; tile_idx < last_block_size; tile_idx++) {
            pack_tile(tile_idx, in0_dfb_id);
        }
        tile_regs_release();
        in0_dfb.push_back(last_block_size);
    }
}

template <typename MmPartialsDFB, typename MmPartialsReloadDFB>
FORCE_INLINE void reload_from_dfb_to_dst(
    MmPartialsDFB& mm_partials_dfb,
    MmPartialsReloadDFB& mm_partials_reload_dfb,
    uint32_t in0_dfb_id,
    uint32_t in1_dfb_id,
    bool in1_transpose_tile,
    uint32_t out_subblock_num_tiles,
    uint32_t out_subblock_w,
    uint32_t out_subblock_h,
    uint32_t in0_block_w) {
    const uint32_t mm_partials_dfb_id = mm_partials_dfb.get_id();
    const uint32_t mm_partials_reload_dfb_id = mm_partials_reload_dfb.get_id();
    reconfig_data_format_srca(in1_dfb_id, mm_partials_reload_dfb_id);
    copy_init(mm_partials_reload_dfb_id);
    mm_partials_dfb.wait_front(static_cast<uint16_t>(out_subblock_num_tiles));

#ifdef MM_PARTIALS_RELOAD_ALIAS
    if (mm_partials_reload_dfb_id != mm_partials_dfb_id) {
        UNPACK((mm_partials_reload_dfb.evil_set_read_ptr(mm_partials_dfb.get_read_ptr())));
    }
#endif

    const uint32_t start_dst_index = 0;
    const uint32_t start_tile_index = 0;
    copy_block(mm_partials_reload_dfb_id, start_tile_index, start_dst_index, out_subblock_num_tiles);

    mm_partials_dfb.pop_front(static_cast<uint16_t>(out_subblock_num_tiles));
    reconfig_data_format_srca(mm_partials_reload_dfb_id, in1_dfb_id);
    matmul_block_init(
        in0_dfb_id, in1_dfb_id, static_cast<uint32_t>(in1_transpose_tile), out_subblock_w, out_subblock_h, in0_block_w);
}

template <uint32_t out_subblock_w, uint32_t out_block_w, typename IntermDFB, typename OutDFB>
inline void reblock_and_untilize(
    IntermDFB& interm_dfb,
    OutDFB& out_dfb,
    uint32_t num_out_subblocks_in_col,
    uint32_t out_subblock_num_tiles,
    uint32_t out_subblock_h) {
    const uint32_t interm_dfb_id = interm_dfb.get_id();
    const uint32_t out_dfb_id = out_dfb.get_id();
    const uint32_t num_tiles_in_row_of_subblocks = mulsi3(out_subblock_num_tiles, num_out_subblocks_in_col);
    interm_dfb.wait_front(static_cast<uint16_t>(num_tiles_in_row_of_subblocks));

    uint32_t within_block_index = 0;
    for (uint32_t h = 0; h < out_subblock_h; h++) {
        uint32_t block_offset = 0;

        out_dfb.reserve_back(out_block_w);
        for (uint32_t n = 0; n < num_out_subblocks_in_col; n++) {
            tile_regs_acquire();
            for (uint32_t w = 0; w < out_subblock_w; w++) {
                const uint32_t tile_index = block_offset + within_block_index + w;
                copy_tile(interm_dfb_id, tile_index, w);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_untilize_dest<out_subblock_w, out_block_w>(out_dfb_id, 1, n);
            tile_regs_release();
            block_offset += out_subblock_num_tiles;
        }
        out_dfb.push_back(out_block_w);

        within_block_index += out_subblock_w;
    }
    interm_dfb.pop_front(static_cast<uint16_t>(num_tiles_in_row_of_subblocks));
}

void kernel_main() {
#ifdef MATMUL_DRAM_SHARDED
    const bool is_worker_core = get_arg(args::is_worker_core) == 1;
    if (not is_worker_core) {
        return;
    }
#endif

    constexpr auto in0_block_w = get_arg(args::in0_block_w);
    constexpr auto in0_num_subblocks = get_arg(args::in0_num_subblocks);
    constexpr auto in0_block_num_tiles = get_arg(args::in0_block_num_tiles);
    constexpr auto in0_subblock_num_tiles = get_arg(args::in0_subblock_num_tiles);
    constexpr auto in1_num_subblocks = get_arg(args::in1_num_subblocks);
    constexpr auto in1_block_num_tiles = get_arg(args::in1_block_num_tiles);
    constexpr auto in1_block_w = get_arg(args::in1_block_w);
    constexpr auto num_blocks_inner_dim = get_arg(args::num_blocks_inner_dim);
    constexpr auto num_blocks_w_dim = get_arg(args::num_blocks_w_dim);
    constexpr auto num_blocks_h_dim = get_arg(args::num_blocks_h_dim);
    constexpr auto out_subblock_h = get_arg(args::out_subblock_h);
    constexpr auto out_subblock_w = get_arg(args::out_subblock_w);
    constexpr auto out_subblock_num_tiles = get_arg(args::out_subblock_num_tiles);
    constexpr auto batch = get_arg(args::batch);
    constexpr auto out_block_num_tiles = get_arg(args::out_block_num_tiles);
    constexpr bool untilize_out = get_arg(args::untilize_out) != 0;
    constexpr bool get_batch_from_reader = static_cast<bool>(get_arg(args::get_batch_from_reader));
#ifdef IN0_TRANSPOSE_TILE
    constexpr bool in0_transpose_tile = true;
#else
    constexpr bool in0_transpose_tile = false;
#endif

    constexpr uint32_t out_block_w = out_subblock_w * in1_num_subblocks;

#ifdef IN0_TRANSPOSE_TILE
    constexpr uint32_t in0_dfb_id = dfb::in0_transposed;
#else
    constexpr uint32_t in0_dfb_id = dfb::in0;
#endif
    constexpr uint32_t in1_dfb_id = dfb::in1;
    constexpr uint32_t out_dfb_id = dfb::out;
    constexpr uint32_t mm_partials_dfb_id = dfb::intermed0;
#ifdef MM_PARTIALS_RELOAD_ALIAS
    constexpr uint32_t mm_partials_reload_dfb_id = dfb::intermed0_reload_alias;
#else
    constexpr uint32_t mm_partials_reload_dfb_id = mm_partials_dfb_id;
#endif
    constexpr uint32_t untilize_mode_out_dfb_id = untilize_out ? mm_partials_dfb_id : out_dfb_id;
    constexpr uint32_t in0_transpose_dfb_id = dfb::in0;

    DataflowBuffer in0_dfb(in0_dfb_id);
#ifdef IN0_TRANSPOSE_TILE
    DataflowBuffer in0_transpose_dfb(in0_transpose_dfb_id);
#else
    auto& in0_transpose_dfb = in0_dfb;
#endif
    DataflowBuffer in1_dfb(dfb::in1);
    DataflowBuffer mm_partials_dfb(mm_partials_dfb_id);
#ifdef MM_PARTIALS_RELOAD_ALIAS
    DataflowBuffer mm_partials_reload_dfb(mm_partials_reload_dfb_id);
#else
    auto& mm_partials_reload_dfb = mm_partials_dfb;
#endif
    DataflowBuffer out_dfb(out_dfb_id);
    auto& untilize_mode_out_dfb = untilize_out ? mm_partials_dfb : out_dfb;

#ifdef FUSE_BIAS
    constexpr uint32_t bias_dfb_id = dfb::bias;
    constexpr auto bias_ntiles = get_arg(args::bias_ntiles);
#ifdef BIAS_PER_GROUP
    constexpr bool bias_per_group = true;
#else
    constexpr bool bias_per_group = false;
#endif
    constexpr uint32_t mm_out_dfb_id = mm_partials_dfb_id;
    constexpr bool row_broadcast_bias = static_cast<bool>(get_arg(args::row_broadcast_bias));
    DataflowBuffer bias_dfb(bias_dfb_id);
    auto& mm_out_dfb = mm_partials_dfb;
#else
    constexpr uint32_t mm_out_dfb_id = untilize_mode_out_dfb_id;
    auto& mm_out_dfb = untilize_mode_out_dfb;
#endif

#ifdef MATMUL_DRAM_SHARDED
    constexpr auto last_subblock_w_valid = get_arg(args::last_subblock_w_valid);
#else
    constexpr uint32_t last_subblock_w_valid = out_subblock_w;
#endif
    constexpr bool last_subblock_padded = last_subblock_w_valid < out_subblock_w;
#ifdef ARCH_QUASAR
    static_assert(!last_subblock_padded, "a narrowed last in1 subblock (padded per_core_N) is not supported on Quasar");
#endif

#ifdef SFPU_ACTIVATION
    constexpr KernelActivation activation_type = static_cast<KernelActivation>(get_arg(args::activation_type));
    constexpr auto activation_param0 = get_arg(args::activation_param0);
    constexpr auto activation_param1 = get_arg(args::activation_param1);
    constexpr auto activation_param2 = get_arg(args::activation_param2);

    ActivationInitHelper<activation_type, activation_param0, activation_param1>::init();
#endif

#ifdef IN1_TRANSPOSE_TILE
    constexpr uint32_t in1_transpose_tile = 1;
#else
    constexpr uint32_t in1_transpose_tile = 0;
#endif

    constexpr bool spill = num_blocks_inner_dim > 1;
#ifdef OUT_PINGPONG
#if defined(FUSE_BIAS) || defined(MM_PARTIALS_RELOAD_ALIAS)
#error "OUT_PINGPONG only supports the no-bias path"
#endif
    DataflowBuffer partials_b_dfb(dfb::intermed0_b);
    DataflowBuffer out_b_dfb(dfb::out_b);
#ifdef OUT_SWIGLU
    static_assert(out_subblock_h == 1 && out_subblock_w % 2 == 0, "OUT_SWIGLU needs a 1 x even subblock");
    DataflowBuffer swiglu_out_dfb(dfb::swiglu_out);
#endif
#endif

    compute_kernel_hw_startup<SrcOrder::Reverse>(in0_dfb_id, in1_dfb_id, mm_partials_dfb_id);
    matmul_block_init(in0_dfb_id, in1_dfb_id, in1_transpose_tile, out_subblock_w, out_subblock_h, in0_block_w);
    for (uint32_t b = 0; b < batch; b++) {
        if constexpr (get_batch_from_reader) {
#ifdef ARCH_QUASAR
            static_assert(
                !get_batch_from_reader,
                "get_batch_from_reader (batch sparsity) is unsupported on Quasar until the DM->TRISC "
                "is_batch_valid mailbox handoff is implemented; do not enable it on Quasar.");
            const bool is_batch_valid = true;
#else
            bool is_batch_valid = false;
            UNPACK(is_batch_valid = static_cast<bool>(mailbox_read(ckernel::ThreadId::BriscThreadId));)
            MATH(is_batch_valid = static_cast<bool>(mailbox_read(ckernel::ThreadId::BriscThreadId));)
            PACK(is_batch_valid = static_cast<bool>(mailbox_read(ckernel::ThreadId::BriscThreadId));)
#endif
            if (!is_batch_valid) {
                continue;
            }
        }

        for (uint32_t bh = 0; bh < num_blocks_h_dim; ++bh) {
            for (uint32_t bw = 0; bw < num_blocks_w_dim; ++bw) {
                bool enable_reload = false;
#ifdef OUT_PINGPONG
                const bool odd_block = (bw & 1u) != 0;
                auto& partials_sel = odd_block ? partials_b_dfb : mm_partials_dfb;
                const uint32_t partials_sel_id =
                    odd_block ? static_cast<uint32_t>(dfb::intermed0_b) : mm_partials_dfb_id;
                auto& out_sel = odd_block ? out_b_dfb : mm_out_dfb;
                const uint32_t out_sel_id = odd_block ? static_cast<uint32_t>(dfb::out_b) : mm_out_dfb_id;
                auto& partials_reload_sel = partials_sel;
#else
                auto& partials_sel = mm_partials_dfb;
                constexpr uint32_t partials_sel_id = mm_partials_dfb_id;
                auto& out_sel = mm_out_dfb;
                constexpr uint32_t out_sel_id = mm_out_dfb_id;
                auto& partials_reload_sel = mm_partials_reload_dfb;
#endif

#ifdef PACK_RELU
                if constexpr (batch > 1 || num_blocks_h_dim > 1 || num_blocks_w_dim > 1) {
                    PACK((llk_pack_relu_config(ReluConfig::none())));
                }
#endif

                if constexpr (batch > 1 || num_blocks_h_dim > 1 || num_blocks_w_dim > 1) {
                    PACK((pack_reconfig_data_format(partials_sel_id)));
#ifdef ARCH_QUASAR
                    pack_init(partials_sel_id);
#endif
                }

                for (uint32_t block = 0; block < num_blocks_inner_dim; block++) {
                    const bool last_out = block == (num_blocks_inner_dim - 1);

                    if constexpr (in0_transpose_tile) {
                        reconfig_data_format_srca(in1_dfb_id, in0_transpose_dfb_id);
                        transpose_init(in0_transpose_dfb_id);
                        PACK((pack_reconfig_data_format(in0_dfb_id)));
#ifdef ARCH_QUASAR
                        pack_init(in0_dfb_id);
#endif
#ifdef PACKER_L1_ACC
                        PACK((llk_pack_reconfig_l1_acc(0)));
#endif
                        transpose_tile_block<in0_block_num_tiles>(in0_transpose_dfb, in0_dfb);
                        reconfig_data_format_srca(in0_transpose_dfb_id, in1_dfb_id);
                        matmul_block_init(
                            in0_dfb_id, in1_dfb_id, in1_transpose_tile, out_subblock_w, out_subblock_h, in0_block_w);
                        PACK((pack_reconfig_data_format(partials_sel_id)));
#ifdef ARCH_QUASAR
                        pack_init(partials_sel_id);
#endif
                    }

#if not defined FUSE_BIAS and defined PACK_RELU
                    if (last_out) {
                        PACK((llk_pack_relu_config(ReluConfig::zero())));
                    }
#endif

#ifdef ARCH_QUASAR
                    if (last_out) {
                        if constexpr (out_sel_id != partials_sel_id) {
                            pack_init(out_sel_id);
                        }
                    }
#endif

#ifdef IN0_RESIDENT
                    if (bw == 0) {
                        in0_dfb.wait_front((block + 1) * in0_block_num_tiles);
                    }
                    in1_dfb.wait_front(in1_block_num_tiles);

                    uint32_t in0_index_subblock_offset = block * in0_block_num_tiles;
#else
                    in0_dfb.wait_front(in0_block_num_tiles);
                    in1_dfb.wait_front(in1_block_num_tiles);

                    uint32_t in0_index_subblock_offset = 0;
#endif
                    for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; in0_subblock++) {
                        uint32_t in1_index_subblock_offset = 0;
                        for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; in1_subblock++) {
                            const bool is_last_in1_subblock_padded =
                                last_subblock_padded && (in1_subblock == in1_num_subblocks - 1);
                            const uint32_t effective_subblock_w =
                                is_last_in1_subblock_padded ? last_subblock_w_valid : out_subblock_w;

                            tile_regs_acquire();
                            if (enable_reload) {
                                reload_from_dfb_to_dst(
                                    partials_sel,
                                    partials_reload_sel,
                                    in0_dfb_id,
                                    in1_dfb_id,
                                    in1_transpose_tile != 0,
                                    out_subblock_num_tiles,
                                    out_subblock_w,
                                    out_subblock_h,
                                    in0_block_w);
                            }

#ifndef SKIP_COMPUTE
                            const uint32_t dst_index = 0;
                            uint32_t in0_index = in0_index_subblock_offset;
                            uint32_t in1_index = in1_index_subblock_offset;
                            for (uint32_t inner_dim_idx = 0; inner_dim_idx < in0_block_w; ++inner_dim_idx) {
                                matmul_block(
                                    in0_dfb_id,
                                    in1_dfb_id,
                                    in0_index,
                                    in1_index,
                                    dst_index,
                                    in1_transpose_tile,
                                    effective_subblock_w,
                                    out_subblock_h,
                                    in0_block_w);
                                in0_index++;
                                in1_index += in1_block_w;
                            }

#endif

                            if (last_out) {
                                tile_regs_commit();
                                out_sel.reserve_back(out_subblock_num_tiles);

#if defined SFPU_ACTIVATION and not defined FUSE_BIAS
                                apply_activation_from_pack<
                                    activation_type,
                                    activation_param0,
                                    activation_param1,
                                    activation_param2>(out_subblock_num_tiles);
#else
                                tile_regs_wait();
#endif

#if defined FP32_DEST_ACC_EN or defined PACKER_L1_ACC
                                PACK((pack_reconfig_data_format(out_sel_id)));
#endif

#ifdef PACKER_L1_ACC
#ifdef FUSE_BIAS
                                if (block == 0) {
                                    PACK((llk_pack_reconfig_l1_acc(0)));
                                } else {
                                    PACK((llk_pack_reconfig_l1_acc(1)));
                                }
#else
                                PACK((llk_pack_reconfig_l1_acc(0)));
#endif
#endif
                                const uint32_t start_dst_index = 0;
                                pack_block(start_dst_index, out_sel_id, out_subblock_num_tiles);

                                tile_regs_release();
                                out_sel.push_back(out_subblock_num_tiles);
#ifdef OUT_SWIGLU
                                out_sel.wait_front(out_subblock_num_tiles);
                                swiglu_out_dfb.reserve_back(out_subblock_num_tiles / 2);
                                reconfig_data_format_srca(in1_dfb_id, out_sel_id);
                                PACK((pack_reconfig_data_format(static_cast<uint32_t>(dfb::swiglu_out))));
                                copy_init(out_sel_id);
                                for (uint32_t p = 0; p < out_subblock_num_tiles / 2; ++p) {
                                    tile_regs_acquire();
                                    copy_tile(out_sel_id, 2 * p, 0);
                                    copy_tile(out_sel_id, 2 * p + 1, 1);
                                    silu_tile_init();
                                    silu_tile(0);
                                    mul_binary_tile_init();
                                    mul_binary_tile(0, 1, 0);
                                    tile_regs_commit();
                                    tile_regs_wait();
                                    pack_tile(0, static_cast<uint32_t>(dfb::swiglu_out));
                                    tile_regs_release();
                                }
                                swiglu_out_dfb.push_back(out_subblock_num_tiles / 2);
                                out_sel.pop_front(out_subblock_num_tiles);
                                reconfig_data_format_srca(out_sel_id, in1_dfb_id);
                                matmul_block_init(
                                    in0_dfb_id,
                                    in1_dfb_id,
                                    in1_transpose_tile,
                                    out_subblock_w,
                                    out_subblock_h,
                                    in0_block_w);
                                PACK((pack_reconfig_data_format(out_sel_id)));
#endif

                            } else {
                                tile_regs_commit();
                                partials_sel.reserve_back(out_subblock_num_tiles);
                                tile_regs_wait();

#ifdef PACKER_L1_ACC
                                if (block == 0) {
                                    PACK((llk_pack_reconfig_l1_acc(0)));
                                } else if (block == 1) {
                                    PACK((llk_pack_reconfig_l1_acc(1)));
                                } else if (in0_transpose_tile) {
                                    PACK((llk_pack_reconfig_l1_acc(1)));
                                }
#endif

                                const uint32_t start_dst_index = 0;
                                pack_block(start_dst_index, partials_sel_id, out_subblock_num_tiles);

                                tile_regs_release();
                                partials_sel.push_back(out_subblock_num_tiles);
                            }

                            in1_index_subblock_offset += out_subblock_w;
                        }
                        in0_index_subblock_offset += in0_subblock_num_tiles;
                    }

#ifdef PACKER_L1_ACC
#ifdef FUSE_BIAS
                    if (block < num_blocks_inner_dim - 1) {
                        for (uint32_t s = 0; s < out_block_num_tiles; s += out_subblock_num_tiles) {
                            partials_sel.wait_front(out_subblock_num_tiles);
#ifdef ARCH_QUASAR
                            dummy_unpack(partials_sel_id);
#endif
                            partials_sel.pop_front(out_subblock_num_tiles);
                        }
                    }
                    enable_reload = false;
#else
                    if (block < num_blocks_inner_dim - 2) {
                        for (uint32_t s = 0; s < out_block_num_tiles; s += out_subblock_num_tiles) {
                            partials_sel.wait_front(out_subblock_num_tiles);
#ifdef ARCH_QUASAR
                            dummy_unpack(partials_sel_id);
#endif
                            partials_sel.pop_front(out_subblock_num_tiles);
                        }
                    }
                    if (block == num_blocks_inner_dim - 2) {
                        enable_reload = true;
                    }
#endif
#else
                    if constexpr (spill) {
                        enable_reload = true;
                    }
#endif

#ifdef IN0_RESIDENT
                    if (bw == num_blocks_w_dim - 1 && block == num_blocks_inner_dim - 1) {
                        in0_dfb.pop_front(num_blocks_inner_dim * in0_block_num_tiles);
                    }
#else
                    in0_dfb.pop_front(in0_block_num_tiles);
#endif
                    in1_dfb.pop_front(in1_block_num_tiles);
                }

#ifdef FUSE_BIAS
#ifdef PACK_RELU
                PACK((llk_pack_relu_config(ReluConfig::zero())));
#endif
#if defined FP32_DEST_ACC_EN or defined PACKER_L1_ACC
                PACK((pack_reconfig_data_format(out_dfb_id)));
#endif
#ifdef PACKER_L1_ACC
                PACK((llk_pack_reconfig_l1_acc(0)));
#endif
                reconfig_data_format(in1_dfb_id, partials_sel_id, in0_dfb_id, bias_dfb_id);
                if constexpr (row_broadcast_bias) {
                    add_bcast_rows_init(partials_sel_id, bias_dfb_id);
                } else {
                    add_init(partials_sel_id, bias_dfb_id);
                }
                if (bias_per_group || (b == 0 && bh == 0) || num_blocks_w_dim > 1) {
                    bias_dfb.wait_front(bias_ntiles);
                }
#ifdef ARCH_QUASAR
                if constexpr (untilize_mode_out_dfb_id != partials_sel_id) {
                    pack_init(untilize_mode_out_dfb_id);
                }
#endif
                for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; in0_subblock++) {
                    uint32_t in1_index_subblock_offset = 0;
                    for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; in1_subblock++) {
                        const bool is_last_in1_subblock_padded =
                            last_subblock_padded && (in1_subblock == in1_num_subblocks - 1);
                        partials_sel.wait_front(out_subblock_num_tiles);
                        tile_regs_acquire();
                        for (uint32_t i = 0, j = 0; j < out_subblock_h; j++) {
#ifdef BIAS_FULL_BLOCK
                            const uint32_t m_tile = in0_subblock * out_subblock_h + j;
                            uint32_t bias_tile_idx = m_tile * in1_block_w + in1_index_subblock_offset;
#else
                            uint32_t bias_tile_idx = in1_index_subblock_offset;
#endif
                            for (uint32_t k = 0; k < out_subblock_w; k++, i++) {
                                const uint32_t safe_bias_tile_idx =
                                    (is_last_in1_subblock_padded && k >= last_subblock_w_valid) ? 0u : bias_tile_idx;

                                if constexpr (row_broadcast_bias) {
                                    add_tiles_bcast_rows(partials_sel_id, bias_dfb_id, i, safe_bias_tile_idx, i);
                                } else {
                                    add_tiles(partials_sel_id, bias_dfb_id, i, safe_bias_tile_idx, i);
                                }
                                bias_tile_idx++;
                            }
                        }
                        tile_regs_commit();

                        partials_sel.pop_front(out_subblock_num_tiles);

                        untilize_mode_out_dfb.reserve_back(out_subblock_num_tiles);

#ifdef SFPU_ACTIVATION
                        PACK(TTI_SEMWAIT(
                            p_stall::STALL_TDMA | p_stall::STALL_CFG,
                            semaphore::t6_sem(semaphore::MATH_PACK),
                            p_stall::STALL_ON_ZERO));

                        PACK(TT_SETC16(
                            DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));

                        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                            ActivationApplyHelper<
                                activation_type,
                                activation_param0,
                                activation_param1,
                                activation_param2>::apply(i);
                        }

                        PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
#else
                        tile_regs_wait();
#endif
                        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                            pack_tile(i, untilize_mode_out_dfb_id);
                        }
                        tile_regs_release();
                        untilize_mode_out_dfb.push_back(out_subblock_num_tiles);

                        in1_index_subblock_offset += out_subblock_w;
                    }
                }
                if constexpr (bias_per_group || num_blocks_w_dim > 1) {
                    bias_dfb.pop_front(bias_ntiles);
                }
#endif
                if constexpr (untilize_out) {
#ifdef PACK_RELU
                    PACK((llk_pack_relu_config(ReluConfig::none())));
#endif
#ifndef FUSE_BIAS
                    reconfig_data_format_srca(in1_dfb_id, partials_sel_id);
#if defined FP32_DEST_ACC_EN or defined PACKER_L1_ACC
                    PACK((pack_reconfig_data_format(out_dfb_id)));
#endif
#ifdef PACKER_L1_ACC
                    PACK((llk_pack_reconfig_l1_acc(0)));
#endif
#endif
                    pack_untilize_dest_init<out_subblock_w, out_block_w>(out_dfb_id);
                    copy_init(partials_sel_id);
                    for (uint32_t in0_subblock_i = 0; in0_subblock_i < in0_num_subblocks; ++in0_subblock_i) {
                        reblock_and_untilize<out_subblock_w, out_block_w>(
                            partials_sel, out_dfb, in1_num_subblocks, out_subblock_num_tiles, out_subblock_h);
                    }
                    pack_untilize_uninit(partials_sel_id);
                }
                if constexpr (batch > 1 || num_blocks_w_dim > 1 || num_blocks_h_dim > 1) {
#ifdef FUSE_BIAS
                    reconfig_data_format(partials_sel_id, in1_dfb_id, bias_dfb_id, in0_dfb_id);
#else
                    reconfig_data_format_srca(partials_sel_id, in1_dfb_id);
#endif
                    matmul_block_init(
                        in0_dfb_id, in1_dfb_id, in1_transpose_tile, out_subblock_w, out_subblock_h, in0_block_w);
                }
            }
        }
    }
#ifdef FUSE_BIAS
    if constexpr (!bias_per_group && num_blocks_w_dim == 1) {
        bias_dfb.pop_front(bias_ntiles);
    }
#endif
}

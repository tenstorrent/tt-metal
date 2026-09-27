// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/matmul.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack.h"
#include "api/debug/assert.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/dfb_helpers_compute.hpp"

namespace compute_kernel_lib {
template <bool WithBias>
FORCE_INLINE void MatmulResult<WithBias>::prepare_untilize_input() const {
    if constexpr (!WithBias) {
        reconfig_data_format_srca(in1_cb_id, interm_cb_id);
    }
}

template <bool WithBias>
FORCE_INLINE void MatmulResult<WithBias>::restore_input_formats() const {
    if constexpr (WithBias) {
        reconfig_data_format(interm_cb_id, in1_cb_id, bias_cb_id, in0_cb_id);
    } else {
        reconfig_data_format_srca(interm_cb_id, in1_cb_id);
    }
}

template <
    KernelActivation Act,
    uint32_t Param0,
    uint32_t Param1,
    uint32_t Param2,
    bool PackRelu,
    ActivationThread Thread>
FORCE_INLINE void MatmulActivation<Act, Param0, Param1, Param2, PackRelu, Thread>::init() {
    if constexpr (Act != KernelActivation::NONE) {
        detail::ActivationInitHelper<Act, Param0, Param1, Thread>::init();
    }
}

template <
    KernelActivation Act,
    uint32_t Param0,
    uint32_t Param1,
    uint32_t Param2,
    bool PackRelu,
    ActivationThread Thread>
FORCE_INLINE void MatmulActivation<Act, Param0, Param1, Param2, PackRelu, Thread>::before_commit(uint32_t num_tiles) {
    if constexpr (Thread == ActivationThread::Math && Act != KernelActivation::NONE) {
        for (uint32_t i = 0; i < num_tiles; ++i) {
            detail::ActivationApplyHelper<Act, Param0, Param1, Param2, Thread>::apply(i);
        }
    }
}

template <
    KernelActivation Act,
    uint32_t Param0,
    uint32_t Param1,
    uint32_t Param2,
    bool PackRelu,
    ActivationThread Thread>
FORCE_INLINE void MatmulActivation<Act, Param0, Param1, Param2, PackRelu, Thread>::after_commit(uint32_t num_tiles) {
    if constexpr (Thread == ActivationThread::Pack && Act != KernelActivation::NONE) {
        detail::apply_activation_from_pack<Act, Param0, Param1, Param2>(num_tiles);
    } else {
        tile_regs_wait();
    }
}

template <
    bool TransposeIn1,
    bool PackerL1Acc,
    matmul_config::InitMode InitMode,
    matmul_config::InputPolicy InputPolicy,
    matmul_config::DataFormatReconfig Reconfig,
    typename Activation,
    bool WithBias,
    MatmulBiasMode BiasMode,
    typename PreKBlockFn,
    typename Shape>
ALWI MatmulResult<WithBias> matmul(
    uint32_t in0_cb_id,
    uint32_t in1_cb_id,
    uint32_t out_cb_id,
    uint32_t interm_cb_id,
    const Shape& shape,
    PreKBlockFn pre_k_block,
    MatmulBias bias) {
    constexpr bool needs_postprocess = !WithBias && Activation::enabled;

    const uint32_t matmul_out_cb_id = WithBias ? interm_cb_id : out_cb_id;
    DataflowBuffer in0_buf(in0_cb_id), in1_buf(in1_cb_id), out_buf(matmul_out_cb_id), interm_buf(interm_cb_id);
    const bool output_is_interm = matmul_out_cb_id == interm_cb_id;
    const uint32_t reload_cb_id =
        shape.partials_reload_cb_id == UINT32_MAX ? interm_cb_id : shape.partials_reload_cb_id;

    const bool reload_last = !output_is_interm || needs_postprocess;

    ASSERT(shape.in0_block_k > 0);
    ASSERT(shape.in0_num_subblocks > 0);
    ASSERT(shape.in1_num_subblocks > 0);
    ASSERT(shape.num_k_blocks > 0);
    ASSERT(shape.out_subblock_h > 0);
    ASSERT(shape.out_subblock_w > 0);
    ASSERT(shape.last_in1_subblock_w_valid <= shape.out_subblock_w);
    ASSERT(shape.batch > 0);
    ASSERT(in0_cb_id != out_cb_id);
    ASSERT(in1_cb_id != out_cb_id);
    ASSERT(in0_cb_id != matmul_out_cb_id);
    ASSERT(in1_cb_id != matmul_out_cb_id);
    ASSERT(shape.out_subblock_h * shape.out_subblock_w <= compute_kernel_lib::DEST_AUTO_LIMIT);
    if constexpr (WithBias) {
        ASSERT(shape.batch == 1);
        ASSERT(bias.num_tiles > 0);
    }
    if constexpr (Reconfig == matmul_config::DataFormatReconfig::InputAndOutput) {
        // Matmul convention: srca takes in1, srcb takes in0.
        reconfig_data_format(in1_cb_id, in0_cb_id);
        PACK((pack_reconfig_data_format(interm_cb_id)));
    }
    if constexpr (InitMode == matmul_config::InitMode::Initialize) {
        matmul_block_init(
            in0_cb_id, in1_cb_id, TransposeIn1, shape.out_subblock_w, shape.out_subblock_h, shape.in0_block_k);
    }

    const uint32_t out_subblock_num_tiles = shape.out_subblock_h * shape.out_subblock_w;
    const auto effective_subblock_width = [&](uint32_t in1_subblock) {
        return (shape.last_in1_subblock_w_valid != 0 && in1_subblock == shape.in1_num_subblocks - 1)
                   ? shape.last_in1_subblock_w_valid
                   : shape.out_subblock_w;
    };
    const uint32_t in0_subblock_num_tiles = shape.out_subblock_h * shape.in0_block_k;
    const uint32_t in0_block_num_tiles = in0_subblock_num_tiles * shape.in0_num_subblocks;
    uint32_t in1_per_core_w = shape.in1_per_core_w;
    if (in1_per_core_w == 0) {
        in1_per_core_w = shape.out_subblock_w * shape.in1_num_subblocks;
    }
    const uint32_t in1_block_num_tiles = in1_per_core_w * shape.in0_block_k;
    const uint32_t out_block_num_tiles = out_subblock_num_tiles * shape.in0_num_subblocks * shape.in1_num_subblocks;
    if constexpr (WithBias) {
        // num_tiles is the resident prefix of the bias CB, not the size of this output block.
        const uint32_t last_n_tile = (shape.in1_num_subblocks - 1) * shape.out_subblock_w +
                                     effective_subblock_width(shape.in1_num_subblocks - 1) - 1;
        uint32_t last_bias_index = bias.offset + last_n_tile;
        if constexpr (BiasMode == MatmulBiasMode::FullBlockElementwise) {
            const uint32_t last_m_tile = shape.in0_num_subblocks * shape.out_subblock_h - 1;
            last_bias_index += last_m_tile * in1_per_core_w;
        }
        ASSERT(last_bias_index < bias.num_tiles);
    }

    if constexpr (WithBias && !PackerL1Acc) {
        if (shape.num_k_blocks > 1 && shape.partials_alias_output_cb_id != UINT32_MAX) {
            // Bias redirects matmul to interm, so its external output may still
            // be draining even though the intermediate CB itself has space.
            DataflowBuffer(shape.partials_alias_output_cb_id).reserve_back(out_block_num_tiles);
        }
    }

    for (uint32_t b = 0; b < shape.batch; b++) {
        for (uint32_t block = 0; block < shape.num_k_blocks; block++) {
            const bool last_out = block == (shape.num_k_blocks - 1);
            const bool enable_reload = block != 0 && (!PackerL1Acc || (last_out && reload_last));

            pre_k_block(block, shape.num_k_blocks, last_out);

            in0_buf.wait_front(in0_block_num_tiles);
            in1_buf.wait_front(in1_block_num_tiles);

            DataflowBuffer& dst_buf = last_out ? out_buf : interm_buf;
            const uint32_t dst_cb_id = dst_buf.get_id();

#ifdef ARCH_QUASAR
            // The final K block can switch from partials to output; format
            // reconfiguration alone does not update Quasar's pack descriptor.
            if (last_out && dst_cb_id != interm_cb_id) {
                pack_init(dst_cb_id);
            }
#endif

            if constexpr (PackerL1Acc) {
                PACK((pack_reconfig_data_format(dst_cb_id)));
                PACK((llk_pack_reconfig_l1_acc(block != 0 && !enable_reload)));
            }

            if constexpr (!WithBias && Activation::pack_relu) {
                if (last_out) {
                    PACK((llk_pack_relu_config(ReluConfig::zero())));
                }
            }

            int in0_index_subblock_offset = 0;
            for (uint32_t in0_subblock = 0; in0_subblock < shape.in0_num_subblocks; in0_subblock++) {
                int in1_index_subblock_offset = 0;
                for (uint32_t in1_subblock = 0; in1_subblock < shape.in1_num_subblocks; in1_subblock++) {
                    tile_regs_acquire();

                    // Narrow the final FMA width without changing the packed subblock size.
                    const uint32_t effective_subblock_w = effective_subblock_width(in1_subblock);

                    if (enable_reload) {
                        reconfig_data_format_srca(in1_cb_id, reload_cb_id);
                        copy_init(reload_cb_id);
                        if (reload_cb_id != interm_cb_id) {
                            DataflowBuffer source(interm_cb_id), reload(reload_cb_id);
                            UNPACK((reload.evil_set_read_ptr(source.get_read_ptr())));
                        }

                        interm_buf.wait_front(out_subblock_num_tiles);
                        copy_block(reload_cb_id, 0, 0, out_subblock_num_tiles);
                        interm_buf.pop_front(out_subblock_num_tiles);

#ifndef ARCH_QUASAR
                        reconfig_data_format_srca(reload_cb_id, in1_cb_id);
                        matmul_block_init(
                            in0_cb_id,
                            in1_cb_id,
                            TransposeIn1,
                            shape.out_subblock_w,
                            shape.out_subblock_h,
                            shape.in0_block_k);
#endif
                    }

                    uint32_t in0_index = in0_index_subblock_offset;
                    uint32_t in1_index = in1_index_subblock_offset;
                    for (uint32_t inner_dim = 0; inner_dim < shape.in0_block_k; inner_dim++) {
#ifndef SKIP_COMPUTE
                        ckernel::matmul_block(
                            in0_cb_id,
                            in1_cb_id,
                            in0_index,
                            in1_index,
                            0,
                            TransposeIn1,
                            effective_subblock_w,
                            shape.out_subblock_h,
                            shape.in0_block_k);
#else
                        (void)in0_index;
                        (void)in1_index;
#endif
                        in0_index++;
                        in1_index += in1_per_core_w;
                    }

                    if (last_out) {
                        if constexpr (!WithBias) {
                            Activation::before_commit(out_subblock_num_tiles);
                        }
                    }
                    tile_regs_commit();
                    if constexpr (!PackerL1Acc && get_fp32_dest_acc_enabled()) {
                        PACK((pack_reconfig_data_format(dst_cb_id)));
                    }
                    // Partials may alias output still being consumed by the writer.
                    if (!output_is_interm && block == 0 && !last_out) {
                        const uint32_t tiles_to_wait =
                            (in0_subblock * shape.in1_num_subblocks + in1_subblock + 1) * out_subblock_num_tiles;
                        out_buf.reserve_back(tiles_to_wait);
                    }
                    dst_buf.reserve_back(out_subblock_num_tiles);
                    if (last_out && !WithBias) {
                        Activation::after_commit(out_subblock_num_tiles);
                    } else {
                        tile_regs_wait();
                    }
                    pack_block(0, dst_cb_id, out_subblock_num_tiles);

                    tile_regs_release();
                    dst_buf.push_back(out_subblock_num_tiles);

                    in1_index_subblock_offset += shape.out_subblock_w;
                }

                in0_index_subblock_offset += in0_subblock_num_tiles;
            }

            if constexpr (!WithBias && Activation::pack_relu) {
                if (last_out) {
                    PACK((llk_pack_relu_config(ReluConfig::none())));
                }
            }

            if constexpr (PackerL1Acc) {
                const bool keep_partials = last_out || (reload_last && block + 2 == shape.num_k_blocks);
                if (!keep_partials) {
                    for (uint32_t off = 0; off < out_block_num_tiles; off += out_subblock_num_tiles) {
                        interm_buf.wait_front(out_subblock_num_tiles);
#ifdef ARCH_QUASAR
                        // Order a bare partials drain without consuming accumulator data.
                        dummy_unpack(interm_cb_id);
#endif
                        interm_buf.pop_front(out_subblock_num_tiles);
                    }
                }
            }

            if (InputPolicy == matmul_config::InputPolicy::WaitAndPopPerKBlock || !last_out) {
                in0_buf.pop_front(in0_block_num_tiles);
                in1_buf.pop_front(in1_block_num_tiles);
            }
        }
        if constexpr (PackerL1Acc) {
            PACK((llk_pack_reconfig_l1_acc(0)));
        }
    }

    if constexpr (WithBias) {
        DataflowBuffer bias_buf(bias.cb_id);
        DataflowBuffer bias_out(out_cb_id);
        bias_buf.wait_front(bias.num_tiles);

        reconfig_data_format_srca(interm_cb_id);
        reconfig_data_format_srcb(bias.cb_id);
        pack_reconfig_data_format(out_cb_id);
#ifdef ARCH_QUASAR
        if (out_cb_id != interm_cb_id) {
            pack_init(out_cb_id);
        }
#endif
        if constexpr (Activation::pack_relu) {
            pack_relu_config(ReluConfig::zero());
        }
        if constexpr (BiasMode == MatmulBiasMode::RowBroadcast) {
            add_bcast_rows_init(interm_cb_id, bias.cb_id);
        } else {
            add_init(interm_cb_id, bias.cb_id);
        }

        for (uint32_t in0_subblock = 0; in0_subblock < shape.in0_num_subblocks; in0_subblock++) {
            for (uint32_t in1_subblock = 0; in1_subblock < shape.in1_num_subblocks; in1_subblock++) {
                const uint32_t effective_subblock_w = effective_subblock_width(in1_subblock);
                interm_buf.wait_front(out_subblock_num_tiles);
                tile_regs_acquire();

                uint32_t i = 0;
                for (uint32_t h = 0; h < shape.out_subblock_h; h++) {
                    const uint32_t m_tile = in0_subblock * shape.out_subblock_h + h;
                    for (uint32_t w = 0; w < shape.out_subblock_w; w++, i++) {
                        const uint32_t n_tile = in1_subblock * shape.out_subblock_w + w;
                        const bool padded = w >= effective_subblock_w;
                        uint32_t bias_idx = bias.offset + n_tile;
                        if constexpr (BiasMode == MatmulBiasMode::FullBlockElementwise) {
                            bias_idx += m_tile * in1_per_core_w;
                        }
                        if (padded) {
                            // A padded output column may have no corresponding bias tile.
                            bias_idx = bias.offset;
                        }
                        if constexpr (BiasMode == MatmulBiasMode::RowBroadcast) {
                            add_tiles_bcast_rows(interm_cb_id, bias.cb_id, i, bias_idx, i);
                        } else {
                            add_tiles(interm_cb_id, bias.cb_id, i, bias_idx, i);
                        }
                    }
                }

                Activation::before_commit(out_subblock_num_tiles);
                tile_regs_commit();
                interm_buf.pop_front(out_subblock_num_tiles);

                bias_out.reserve_back(out_subblock_num_tiles);
                Activation::after_commit(out_subblock_num_tiles);
                pack_block(0, out_cb_id, out_subblock_num_tiles);
                tile_regs_release();
                bias_out.push_back(out_subblock_num_tiles);
            }
        }
        if constexpr (Activation::pack_relu) {
            pack_relu_config(ReluConfig::none());
        }
    }
    return {in0_cb_id, in1_cb_id, interm_cb_id, bias.cb_id};
}

}  // namespace compute_kernel_lib

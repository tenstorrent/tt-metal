// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(ARCH_BLACKHOLE) || defined(ARCH_WORMHOLE)
#include "api/compute/experimental/matmul_custom.h"
#include "api/compute/experimental/sdpa_sub_custom.h"
#endif
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/dataflow/circular_buffer.h"
#include "api/debug/kernel_profiler.hpp"

#ifdef SDPA_MATMUL_FIDELITY
constexpr bool sdpa_matmul_fidelity_set = true;
constexpr MathFidelity sdpa_matmul_fidelity = static_cast<MathFidelity>(SDPA_MATMUL_FIDELITY);
#else
constexpr bool sdpa_matmul_fidelity_set = false;
constexpr MathFidelity sdpa_matmul_fidelity = MathFidelity::LoFi;
#endif
#ifdef SDPA_COMPUTE_LOFI
constexpr bool sdpa_compute_lofi = true;
#else
constexpr bool sdpa_compute_lofi = false;
#endif

ALWI void sdpa_mm_init(
    uint32_t in0_cb, uint32_t in1_cb, bool transpose, uint32_t ct_dim, uint32_t rt_dim, uint32_t kt_dim) {
    if constexpr (sdpa_matmul_fidelity_set) {
        mm_no_mop_init_short_fidelity<sdpa_matmul_fidelity>(in0_cb, in1_cb, transpose, ct_dim, rt_dim, kt_dim);
    } else {
        mm_no_mop_init_short(in0_cb, in1_cb, transpose, ct_dim, rt_dim, kt_dim);
    }
}

ALWI void sdpa_mm_reinit(
    uint32_t in0_cb, uint32_t in1_cb, bool transpose, uint32_t ct_dim, uint32_t rt_dim, uint32_t kt_dim) {
    if constexpr (sdpa_matmul_fidelity_set) {
        mm_no_mop_reinit_short_fidelity<sdpa_matmul_fidelity>(in0_cb, in1_cb, transpose, ct_dim, rt_dim, kt_dim);
    } else {
        mm_no_mop_reinit_short(in0_cb, in1_cb, transpose, ct_dim, rt_dim, kt_dim);
    }
}

ALWI void sdpa_pv_mm_setup(
    uint32_t in0_cb, uint32_t in1_cb, bool transpose, uint32_t ct_dim, uint32_t rt_dim, uint32_t kt_dim) {
    if constexpr ((sdpa_matmul_fidelity_set && sdpa_matmul_fidelity == MathFidelity::LoFi) || sdpa_compute_lofi) {
        sdpa_mm_init(in0_cb, in1_cb, transpose, ct_dim, rt_dim, kt_dim);
    } else {
        sdpa_mm_reinit(in0_cb, in1_cb, transpose, ct_dim, rt_dim, kt_dim);
    }
}

ALWI void sdpa_matmul_block_no_mop(
    uint32_t in0_cb,
    uint32_t in1_cb,
    uint32_t in0_index,
    uint32_t in1_index,
    uint32_t idst,
    bool transpose,
    uint32_t ct_dim,
    uint32_t rt_dim,
    uint32_t kt_dim) {
    if constexpr (sdpa_matmul_fidelity_set) {
        matmul_block_no_mop_fidelity<sdpa_matmul_fidelity>(
            in0_cb, in1_cb, in0_index, in1_index, idst, transpose, ct_dim, rt_dim, kt_dim);
    } else {
        matmul_block_no_mop(in0_cb, in1_cb, in0_index, in1_index, idst, transpose, ct_dim, rt_dim, kt_dim);
    }
}

#if defined(PROFILE_STREAMING)
#define MaybeDeviceZoneScopedN(ENABLED, name)
#elif defined(PROFILE_KERNEL)
template <bool Enabled, uint32_t timer_id>
struct MaybeProfileScope {
    inline __attribute__((always_inline)) MaybeProfileScope() {}
    inline __attribute__((always_inline)) ~MaybeProfileScope() {}
};
template <uint32_t timer_id>
struct MaybeProfileScope<true, timer_id> : kernel_profiler::profileScope<timer_id> {};

#define MaybeDeviceZoneScopedN(ENABLED, name)                                  \
    DO_PRAGMA(message(PROFILER_MSG_NAME(name)));                               \
    auto constexpr hash = kernel_profiler::Hash16_CT(PROFILER_MSG_NAME(name)); \
    MaybeProfileScope<ENABLED, hash> zone;
#else
#define MaybeDeviceZoneScopedN(ENABLED, name)
#endif

#ifdef TRISC_PACK
__attribute__((noinline, noclone)) static void sdpa_pack_tile_ooo(uint32_t dst, uint32_t cb, uint32_t idx) {
    llk_pack<DST_ACCUM_MODE, true, PackMode::Default>(dst, cb, idx);
}
#else
ALWI void sdpa_pack_tile_ooo(uint32_t, uint32_t, uint32_t) {}
#endif

static __attribute__((noinline, noclone)) void sdpa_cb_pop_front_out_of_line(uint32_t cb_id, uint32_t num_tiles) {
    CircularBuffer(cb_id).pop_front(num_tiles);
}

ALWI void cb_push_back_hold_wr_ptr(uint32_t cb_id, uint32_t num_tiles) {
    CircularBuffer(cb_id).push_back(num_tiles);
    PACK(({
        auto& intf = get_local_cb_interface(cb_id);
        intf.fifo_wr_ptr -= num_tiles * intf.fifo_page_size;
        uint32_t fifo_start = intf.fifo_limit - intf.fifo_size;
        if (intf.fifo_wr_ptr < fifo_start) {
            intf.fifo_wr_ptr += intf.fifo_size;
        }
    }));
}

constexpr uint32_t INVALID_CB = 32;

template <uint32_t old_cb, uint32_t new_cb>
ALWI void sdpa_maybe_pack_reconfig_data_format() {
#ifdef TRISC_PACK
    if constexpr (pack_dst_format[old_cb] != pack_dst_format[new_cb]) {
        pack_reconfig_data_format(old_cb, new_cb);
    }
#endif
}

template <uint32_t old_cb, uint32_t new_cb>
constexpr bool sdpa_unpack_format_changed() {
#if defined(TRISC_UNPACK) || defined(TRISC_MATH)
    return unpack_src_format[old_cb] != unpack_src_format[new_cb] ||
           unpack_dst_format[old_cb] != unpack_dst_format[new_cb] ||
           unpack_tile_face_r_dim[old_cb] != unpack_tile_face_r_dim[new_cb] ||
           unpack_tile_num_faces[old_cb] != unpack_tile_num_faces[new_cb];
#else
    return false;
#endif
}

template <uint32_t srca_old_cb, uint32_t srca_new_cb, uint32_t srcb_old_cb, uint32_t srcb_new_cb>
ALWI void sdpa_maybe_reconfig_data_format() {
#if defined(TRISC_UNPACK) || defined(TRISC_MATH)
    if constexpr (
        sdpa_unpack_format_changed<srca_old_cb, srca_new_cb>() ||
        sdpa_unpack_format_changed<srcb_old_cb, srcb_new_cb>()) {
        reconfig_data_format(srca_old_cb, srca_new_cb, srcb_old_cb, srcb_new_cb);
    }
#endif
}

static __attribute__((noinline, noclone)) void configure_pack_width(uint32_t cb, uint32_t pack_width) {
    PACK((llk_pack_init<ckernel::PackMode::Default, false, true, true>(cb, pack_width)));
}

ALWI void configure_single_tile_pack(uint32_t cb) { configure_pack_width(cb, 1); }

ALWI bool configure_row_pack_width(uint32_t cb, uint32_t pack_width) {
    const bool use_blocked_pack_width = should_use_blocked_pack_width(pack_width);
    configure_pack_width(cb, use_blocked_pack_width ? pack_width : 1);
    return use_blocked_pack_width;
}

template <uint32_t kPackDoneMax = 2>
ALWI void init_sdpa_streaming_semaphores() {
    static_assert(kPackDoneMax >= 1 && kPackDoneMax <= 15, "SEMINIT's max value is a 4-bit field");
    PACK((t6_semaphore_init(semaphore::FPU_SFPU, 0, 1)));
    PACK((t6_semaphore_init(semaphore::UNPACK_MATH_DONE, 0, 1)));
    PACK((t6_semaphore_init(semaphore::PACK_DONE, 0, kPackDoneMax)));
}

#if defined(TRISC_MATH) && defined(ARCH_WORMHOLE)
ALWI void recip_tile_first_column_wh_idst0_direct() {
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
    math::set_addr_mod_base();

#pragma GCC unroll 0
    for (int face = 0; face < 2; face++) {
        ckernel::sfpu::calculate_recip_first_column<DST_ACCUM_MODE>();
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
    }

    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU);
    math::clear_addr_mod_base();
}
#endif

ALWI void pack_contiguous_rows_nocfg(
    uint32_t out_cb,
    uint32_t row_base,
    uint32_t row_count,
    uint32_t row_stride,
    uint32_t col_base,
    uint32_t pack_width) {
    uint32_t dst_index = 0;
    const bool use_blocked_pack_width = should_use_blocked_pack_width(pack_width);
    for (uint32_t row = 0; row < row_count; ++row) {
        uint32_t out_tile_index = (row_base + row) * row_stride + col_base;
        if (use_blocked_pack_width) {
            sdpa_pack_tile_ooo(dst_index, out_cb, out_tile_index);
            dst_index += pack_width;
        } else {
            for (uint32_t col = 0; col < pack_width; ++col) {
                sdpa_pack_tile_ooo(dst_index++, out_cb, out_tile_index + col);
            }
        }
    }
}

ALWI void pack_contiguous_rows(
    uint32_t out_cb,
    uint32_t row_base,
    uint32_t row_count,
    uint32_t row_stride,
    uint32_t col_base,
    uint32_t pack_width) {
    configure_row_pack_width(out_cb, pack_width);
    pack_contiguous_rows_nocfg(out_cb, row_base, row_count, row_stride, col_base, pack_width);
}

template <uint32_t sbh_t, uint32_t sbw_t, uint32_t dst_size>
void salad_correct_fused(
    uint32_t out_in_cb,
    uint32_t sum_in_cb,
    uint32_t bcast_cb,
    uint32_t out_out_cb,
    uint32_t sum_out_cb,
    uint32_t ob_q_subblock,
    uint32_t sum_q_subblock,
    uint32_t write_q_subblock) {
    constexpr uint32_t tiles_per_row = sbh_t;
    constexpr uint32_t tiles_per_column = sbw_t;
    constexpr uint32_t col_batch = (dst_size / sbh_t < sbw_t) ? dst_size / sbh_t : sbw_t;
    constexpr uint32_t last_out_cols = (sbw_t % col_batch == 0) ? col_batch : (sbw_t % col_batch);
    constexpr bool can_fuse_last = (last_out_cols * sbh_t + sbh_t <= dst_size);

    const uint32_t ob_row_base = ob_q_subblock * tiles_per_row;
    const uint32_t sum_row_base = sum_q_subblock * tiles_per_row;
    const uint32_t write_row_base = write_q_subblock * tiles_per_row;

    mul_bcast_cols_init(out_in_cb, bcast_cb);

    CircularBuffer(out_in_cb).wait_front((ob_q_subblock + 1) * tiles_per_row * tiles_per_column);
    CircularBuffer(sum_in_cb).wait_front((sum_q_subblock + 1) * tiles_per_row);
    CircularBuffer(bcast_cb).wait_front((ob_q_subblock + 1) * tiles_per_row);

    constexpr uint32_t last_batch_rem = tiles_per_column % col_batch;
    for (uint32_t col_base = 0; col_base < tiles_per_column; col_base += col_batch) {
        const uint32_t cur_cols =
            (col_base + col_batch <= tiles_per_column) ? col_batch : (last_batch_rem > 0 ? last_batch_rem : col_batch);
        const bool is_last_out_batch = (col_base + cur_cols >= tiles_per_column);
        const bool fuse_sum_here = can_fuse_last && is_last_out_batch;

        tile_regs_acquire();
        uint32_t dst_index = 0;
        for (uint32_t i = 0; i < tiles_per_row; i++) {
            for (uint32_t j = 0; j < cur_cols; j++) {
                uint32_t in0_tile_index = (ob_row_base + i) * tiles_per_column + col_base + j;
                mul_tiles_bcast_cols(out_in_cb, bcast_cb, in0_tile_index, ob_row_base + i, dst_index++);
            }
        }
        if (fuse_sum_here) {
            for (uint32_t i = 0; i < tiles_per_row; i++) {
                mul_tiles_bcast_cols(sum_in_cb, bcast_cb, sum_row_base + i, ob_row_base + i, dst_index++);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_contiguous_rows(out_out_cb, write_row_base, tiles_per_row, tiles_per_column, col_base, cur_cols);
        dst_index = tiles_per_row * cur_cols;
        if (fuse_sum_here) {
            configure_single_tile_pack(sum_out_cb);
            for (uint32_t i = 0; i < tiles_per_row; i++) {
                sdpa_pack_tile_ooo(dst_index++, sum_out_cb, write_row_base + i);
            }
        }
        tile_regs_release();
    }

    if constexpr (!can_fuse_last) {
        tile_regs_acquire();
        for (uint32_t i = 0; i < tiles_per_row; i++) {
            mul_tiles_bcast_cols(sum_in_cb, bcast_cb, sum_row_base + i, ob_row_base + i, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        configure_single_tile_pack(sum_out_cb);
        for (uint32_t i = 0; i < tiles_per_row; i++) {
            sdpa_pack_tile_ooo(i, sum_out_cb, write_row_base + i);
        }
        tile_regs_release();
    }
}

static inline void l1_acc_tile_run(
    uint32_t mask_cb, uint32_t mask_base, uint32_t out_cb, uint32_t out_base, uint32_t count) {
    constexpr uint32_t kBatch = compute_kernel_lib::DEST_AUTO_LIMIT;
    for (uint32_t i = 0; i < count; i += kBatch) {
        const uint32_t batch = (count - i) < kBatch ? (count - i) : kBatch;
        tile_regs_acquire();
        for (uint32_t j = 0; j < batch; j++) {
            copy_tile(mask_cb, mask_base + i + j, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < batch; j++) {
            pack_tile<true>(j, out_cb, out_base + i + j);
        }
        tile_regs_release();
    }
}

template <bool reconfig_dt>
static inline void begin_mask_l1_accumulate(uint32_t cb_qkt_im, uint32_t cb_mask_in) {
    configure_single_tile_pack(cb_qkt_im);
    if constexpr (reconfig_dt) {
        reconfig_data_format_srca(cb_qkt_im, cb_mask_in);
        copy_init(cb_mask_in);
    } else {
        copy_init(cb_mask_in);
    }
    PACK((llk_pack_reconfig_l1_acc(1)));
}

static inline void end_mask_l1_accumulate() { PACK((llk_pack_reconfig_l1_acc(0))); }

template <
    bool profiling_enabled,
    uint32_t head_dim_t_,
    uint32_t dst_size,
    uint32_t col_identity_cb,
    uint32_t scratch_cb,
    uint32_t normalized_out_cb,
    uint32_t scale_fp32 = 0,
    bool use_attention_sink = false,
    uint32_t cb_attention_sink = INVALID_CB,
    bool sink_per_row = false>
static __attribute__((noinline, noclone)) void normalize_row_streaming(
    uint32_t cur_sum_cb,
    uint32_t cur_out_cb,
    uint32_t sbh,
    [[maybe_unused]] uint32_t cur_max_cb_rt = 0,
    [[maybe_unused]] uint32_t sink_row_offset = 0) {
    if constexpr (use_attention_sink) {
        CircularBuffer(cb_attention_sink).wait_front(sink_per_row ? sink_row_offset + sbh : 1);
        CircularBuffer(cur_max_cb_rt).wait_front(sink_row_offset + sbh);
    }
    configure_single_tile_pack(scratch_cb);
    for (uint32_t s = 0; s < sbh; s++) {
        {
            MaybeDeviceZoneScopedN(profiling_enabled, "NORM_MATMUL_RECIP");
            constexpr uint32_t N = 1;
            matmul_block_init(cur_sum_cb, col_identity_cb, 0, N, 1, N);
            sdpa_maybe_reconfig_data_format<normalized_out_cb, col_identity_cb, normalized_out_cb, scratch_cb>();
            sdpa_maybe_pack_reconfig_data_format<normalized_out_cb, scratch_cb>();

            CircularBuffer(col_identity_cb).wait_front(N);
            CircularBuffer(cur_sum_cb).wait_front(1);

            CircularBuffer(scratch_cb).reserve_back(1);
            tile_regs_acquire();
            matmul_block(cur_sum_cb, col_identity_cb, 0, 0, 0, 0, N, 1, N);
            if constexpr (use_attention_sink) {
                sdpa_maybe_reconfig_data_format<col_identity_cb, scratch_cb, scratch_cb, cb_attention_sink>();
                if constexpr (sink_per_row) {
                    sub_init(cur_max_cb_rt, cb_attention_sink);
                    sub_tiles(cur_max_cb_rt, cb_attention_sink, sink_row_offset + s, sink_row_offset + s, 1);
                } else {
                    sub_bcast_scalar_init(cur_max_cb_rt, cb_attention_sink);
                    sub_tiles_bcast_scalar(cur_max_cb_rt, cb_attention_sink, sink_row_offset + s, 0, 1);
                }
                sdpa_maybe_reconfig_data_format<scratch_cb, col_identity_cb, cb_attention_sink, scratch_cb>();
                MATH((llk_math_eltwise_unary_sfpu_init<SfpuType::exponential, DST_ACCUM_MODE>()));
                constexpr uint16_t scale_bf16 = scale_fp32 >> 16;
                constexpr uint16_t negated_scale_bf16 = scale_bf16 ^ 0x8000;
                MATH((exp_tile_first_column<EXP_APPROX_MODE, negated_scale_bf16>(1)));
                add_binary_tile_init();
                add_binary_tile(0, 1, 0);
            }
#ifdef ARCH_BLACKHOLE
            recip_tile_init();
            MATH((recip_tile(0, VectorMode::C)));
#else
            recip_tile_init();
            MATH((recip_tile_first_column_wh_idst0_direct()));
#endif
            tile_regs_commit();

            tile_regs_wait();
            pack_tile(0, scratch_cb);
            tile_regs_release();
            CircularBuffer(scratch_cb).push_back(1);

            CircularBuffer(cur_sum_cb).pop_front(1);
        }

        {
            MaybeDeviceZoneScopedN(profiling_enabled, "NORM_MUL_BCAST");
            constexpr uint32_t batch = (head_dim_t_ < dst_size) ? head_dim_t_ : dst_size;
            mul_bcast_cols_init(cur_out_cb, scratch_cb);
            sdpa_maybe_pack_reconfig_data_format<scratch_cb, normalized_out_cb>();
            CircularBuffer(cur_out_cb).wait_front(head_dim_t_);
            CircularBuffer(scratch_cb).wait_front(1);

            CircularBuffer(normalized_out_cb).reserve_back(head_dim_t_);
            for (uint32_t base = 0; base < head_dim_t_; base += batch) {
                constexpr uint32_t last_batch = head_dim_t_ % batch;
                const uint32_t cur_batch = (base + batch <= head_dim_t_) ? batch : last_batch;
                tile_regs_acquire();
                for (uint32_t j = 0; j < cur_batch; ++j) {
                    mul_tiles_bcast_cols(cur_out_cb, scratch_cb, base + j, 0, j);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t j = 0; j < cur_batch; ++j) {
                    pack_tile(j, normalized_out_cb);
                }
                tile_regs_release();
            }
            CircularBuffer(normalized_out_cb).push_back(head_dim_t_);

            CircularBuffer(scratch_cb).pop_front(1);
            CircularBuffer(cur_out_cb).pop_front(head_dim_t_);
        }
    }
    sdpa_maybe_pack_reconfig_data_format<normalized_out_cb, scratch_cb>();
}

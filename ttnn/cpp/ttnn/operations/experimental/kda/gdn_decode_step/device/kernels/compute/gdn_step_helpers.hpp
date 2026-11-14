// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Tile-level helpers shared by the gdn_decode_step compute kernels (plain and fused-conv variants).
#pragma once
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/softplus.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose_dest.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

namespace gdn_step {

// out[i] = a[i] * b[i]  (plain eltwise; b_index_fixed -> b[0] for every i)
template <bool b_fixed>
inline void multiply_tiles(uint32_t a, uint32_t b, uint32_t out, DataflowBuffer& out_dfb, uint32_t n) {
    out_dfb.reserve_back(n);
    pack_reconfig_data_format(out);
    reconfig_data_format(a, b);
    mul_init(a, b);
    for (uint32_t i = 0; i < n; ++i) {
        tile_regs_acquire();
        mul_tiles(a, b, i, b_fixed ? 0 : i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out, i);
        tile_regs_release();
    }
    out_dfb.push_back(n);
}

// out[i] = a[i] - b[i]
inline void subtract_tiles(uint32_t a, uint32_t b, uint32_t out, DataflowBuffer& out_dfb, uint32_t n) {
    out_dfb.reserve_back(n);
    pack_reconfig_data_format(out);
    reconfig_data_format(a, b);
    sub_init(a, b);
    for (uint32_t i = 0; i < n; ++i) {
        tile_regs_acquire();
        sub_tiles(a, b, i, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out, i);
        tile_regs_release();
    }
    out_dfb.push_back(n);
}

// out[i] = a[i] + b[i]
inline void add_tiles_n(uint32_t a, uint32_t b, uint32_t out, DataflowBuffer& out_dfb, uint32_t n) {
    out_dfb.reserve_back(n);
    pack_reconfig_data_format(out);
    reconfig_data_format(a, b);
    add_init(a, b);
    for (uint32_t i = 0; i < n; ++i) {
        tile_regs_acquire();
        add_tiles(a, b, i, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out, i);
        tile_regs_release();
    }
    out_dfb.push_back(n);
}

// out[i] = a[i] scaled per row by column 0 of col_tile (tile 0 of b)
inline void scale_rows(uint32_t a, uint32_t b, uint32_t out, DataflowBuffer& out_dfb, uint32_t n) {
    out_dfb.reserve_back(n);
    pack_reconfig_data_format(out);
    reconfig_data_format(a, b);
    mul_bcast_cols_init(a, b);
    for (uint32_t i = 0; i < n; ++i) {
        tile_regs_acquire();
        mul_tiles_bcast_cols(a, b, i, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out, i);
        tile_regs_release();
    }
    out_dfb.push_back(n);
}

// out[i] = a[i] scaled per column by row 0 of b[i]
inline void scale_cols(uint32_t a, uint32_t b, uint32_t out, DataflowBuffer& out_dfb, uint32_t n) {
    out_dfb.reserve_back(n);
    pack_reconfig_data_format(out);
    reconfig_data_format(a, b);
    mul_bcast_rows_init(a, b);
    for (uint32_t i = 0; i < n; ++i) {
        tile_regs_acquire();
        mul_tiles_bcast_rows(a, b, i, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out, i);
        tile_regs_release();
    }
    out_dfb.push_back(n);
}

// out[i] = a[i] * a[i]  (bf16 or fp32 in, fp32 out)
inline void square_tiles(uint32_t a, uint32_t out, DataflowBuffer& out_dfb, uint32_t n) {
    out_dfb.reserve_back(n);
    pack_reconfig_data_format(out);
    reconfig_data_format(a, a);
    mul_init(a, a);
    for (uint32_t i = 0; i < n; ++i) {
        tile_regs_acquire();
        mul_tiles(a, a, i, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out, i);
        tile_regs_release();
    }
    out_dfb.push_back(n);
}

// l2 variant: inv = (rsqrt(stats + eps) * post) * mask   (stats: row sums of squares in column 0)
inline void inverse_l2(
    uint32_t stats,
    uint32_t eps,
    uint32_t mask,
    uint32_t scratch,
    DataflowBuffer& scratch_dfb,
    uint32_t inv,
    DataflowBuffer& inv_dfb,
    uint32_t post_scale_bits) {
    scratch_dfb.reserve_back(1);
    pack_reconfig_data_format(scratch);
    reconfig_data_format(stats, eps);
    add_init(stats, eps);
    tile_regs_acquire();
    add_tiles(stats, eps, 0, 0, 0);
    rsqrt_tile_init();
    rsqrt_tile(0);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, post_scale_bits);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, scratch, 0);
    tile_regs_release();
    scratch_dfb.push_back(1);
    scratch_dfb.wait_front(1);
    inv_dfb.reserve_back(1);
    pack_reconfig_data_format(inv);
    reconfig_data_format(scratch, mask);
    mul_init(scratch, mask);
    tile_regs_acquire();
    mul_tiles(scratch, mask, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, inv, 0);
    tile_regs_release();
    inv_dfb.push_back(1);
    scratch_dfb.pop_front(1);
}

// rms variant: inv = rsqrt(stats * inv_n + eps)
inline void inverse_rms(
    uint32_t stats,
    uint32_t eps,
    uint32_t scratch,
    DataflowBuffer& scratch_dfb,
    uint32_t inv,
    DataflowBuffer& inv_dfb,
    uint32_t inv_n_bits) {
    scratch_dfb.reserve_back(1);
    pack_reconfig_data_format(scratch);
    reconfig_data_format_srca(stats);
    copy_init(stats);
    tile_regs_acquire();
    copy_tile(stats, 0, 0);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, inv_n_bits);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, scratch, 0);
    tile_regs_release();
    scratch_dfb.push_back(1);
    scratch_dfb.wait_front(1);
    inv_dfb.reserve_back(1);
    pack_reconfig_data_format(inv);
    reconfig_data_format(scratch, eps);
    add_init(scratch, eps);
    tile_regs_acquire();
    add_tiles(scratch, eps, 0, 0, 0);
    rsqrt_tile_init();
    rsqrt_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, inv, 0);
    tile_regs_release();
    inv_dfb.push_back(1);
    scratch_dfb.pop_front(1);
}

// out[j] = sum_i a[i] @ b[i*Nt + j]   (a: 1 x Kt row of tiles, b: Kt x Nt tiles)
inline void row_times_matrix(uint32_t a, uint32_t b, uint32_t out, DataflowBuffer& out_dfb, uint32_t Kt, uint32_t Nt) {
    out_dfb.reserve_back(Nt);
    pack_reconfig_data_format(out);
    reconfig_data_format<SrcOrder::Reverse>(a, b);
    matmul_init(a, b);
    for (uint32_t j = 0; j < Nt; ++j) {
        tile_regs_acquire();
        for (uint32_t i = 0; i < Kt; ++i) {
            matmul_tiles(a, b, i, i * Nt + j, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out, j);
        tile_regs_release();
    }
    out_dfb.push_back(Nt);
}

// out[i*Nt + j] = a[i] @ b[j]   (outer product of a column block and a row block, K = 1 tile)
inline void outer_product(uint32_t a, uint32_t b, uint32_t out, DataflowBuffer& out_dfb, uint32_t Kt, uint32_t Nt) {
    out_dfb.reserve_back(Kt * Nt);
    pack_reconfig_data_format(out);
    reconfig_data_format<SrcOrder::Reverse>(a, b);
    matmul_init(a, b);
    for (uint32_t i = 0; i < Kt; ++i) {
        for (uint32_t j = 0; j < Nt; ++j) {
            tile_regs_acquire();
            matmul_tiles(a, b, i, j, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, out, i * Nt + j);
            tile_regs_release();
        }
    }
    out_dfb.push_back(Kt * Nt);
}

// out[i] = transpose(a[i])  (32-bit in-DST transpose)
inline void transpose_tiles(uint32_t a, uint32_t out, DataflowBuffer& out_dfb, uint32_t n) {
    out_dfb.reserve_back(n);
    pack_reconfig_data_format(out);
    reconfig_data_format_srca(a);
    for (uint32_t i = 0; i < n; ++i) {
        // transpose_dest_init reprograms the math pipeline, so the datacopy must be re-initialised per tile
        copy_init(a);
        tile_regs_acquire();
        copy_tile(a, i, 0);
        transpose_dest_init<true>(a);
        transpose_dest<true>(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out, i);
        tile_regs_release();
    }
    out_dfb.push_back(n);
}

// out[i] = a[i]
inline void copy_tiles(uint32_t a, uint32_t out, DataflowBuffer& out_dfb, uint32_t n) {
    out_dfb.reserve_back(n);
    pack_reconfig_data_format(out);
    reconfig_data_format_srca(a);
    copy_init(a);
    for (uint32_t i = 0; i < n; ++i) {
        tile_regs_acquire();
        copy_tile(a, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out, i);
        tile_regs_release();
    }
    out_dfb.push_back(n);
}

// out = exp(a[0])
inline void exp_tile_copy(uint32_t a, uint32_t out, DataflowBuffer& out_dfb) {
    out_dfb.reserve_back(1);
    pack_reconfig_data_format(out);
    reconfig_data_format_srca(a);
    copy_init(a);
    tile_regs_acquire();
    copy_tile(a, 0, 0);
    exp_tile_init<false>();
    exp_tile<false>(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, out, 0);
    tile_regs_release();
    out_dfb.push_back(1);
}

}  // namespace gdn_step

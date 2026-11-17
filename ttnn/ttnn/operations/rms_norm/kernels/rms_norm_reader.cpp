// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// rms_norm reader (NCRISC).
//
// Per core: scaler (1.0) once; gamma (resident: once, streamed: per row block, pass-2 order);
// then for every row block of Rb tile-rows, x segments of Rb x Sb tiles (twice when not resident:
// pass 1 = sum of squares, pass 2 = normalize). W/H pads are zeroed here (this kernel is the CB
// producer): TILE input zeroes cols >= valid_w of the last W tile; RM input zero-fills missing
// sticks and the W tail of each stick segment.
//
// Raw dataflow (no helper covers per-stick byte-offset segment reads with H/W zero-fill or the
// gamma stick -> row-0-of-tile scatter); the scaler uses the reduce dataflow helper.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

namespace {
constexpr uint32_t TILE_DIM = 32;
constexpr uint32_t FACE_DIM = 16;
constexpr uint32_t FACE_ELEMS = FACE_DIM * FACE_DIM;

template <typename T>
FORCE_INLINE void zero_w_pad_of_tile(uint32_t tile_addr, uint32_t valid_w) {
    volatile tt_l1_ptr T* p = reinterpret_cast<volatile tt_l1_ptr T*>(tile_addr);
    for (uint32_t f = 0; f < 4; ++f) {
        const uint32_t col0 = (f & 1) * FACE_DIM;
        if (valid_w >= col0 + FACE_DIM) {
            continue;
        }
        const uint32_t j0 = valid_w > col0 ? valid_w - col0 : 0;
        for (uint32_t row = 0; row < FACE_DIM; ++row) {
            for (uint32_t j = j0; j < FACE_DIM; ++j) {
                p[f * FACE_ELEMS + row * FACE_DIM + j] = 0;
            }
        }
    }
}

template <typename T>
FORCE_INLINE void zero_elems(uint32_t addr, uint32_t first_elem, uint32_t end_elem) {
    volatile tt_l1_ptr T* p = reinterpret_cast<volatile tt_l1_ptr T*>(addr);
    for (uint32_t e = first_elem; e < end_elem; ++e) {
        p[e] = 0;
    }
}
}  // namespace

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t g_addr = get_arg_val<uint32_t>(1);
    const uint32_t first_tile_row = get_arg_val<uint32_t>(2);
    const uint32_t rows_core = get_arg_val<uint32_t>(3);
    const uint32_t scaler_bits = get_arg_val<uint32_t>(4);
    const uint32_t slice_tile0 = get_arg_val<uint32_t>(5);  // first W tile of this core's slice (W-split)
    const bool is_last_slice = get_arg_val<uint32_t>(6) != 0;
    const uint32_t first_stick = get_arg_val<uint32_t>(7);  // RM shard: global index of the shard's first stick
    const uint32_t core_sticks = get_arg_val<uint32_t>(8);  // RM shard: valid sticks in this core's shard

    constexpr bool IS_RM = get_compile_time_arg_val(0) != 0;
    constexpr uint32_t ES_X = get_compile_time_arg_val(1);
    constexpr uint32_t Wt = get_compile_time_arg_val(2);
    constexpr uint32_t Sb = get_compile_time_arg_val(3);
    constexpr uint32_t Rb = get_compile_time_arg_val(4);
    constexpr bool RESIDENT = get_compile_time_arg_val(5) != 0;
    constexpr bool HAS_GAMMA = get_compile_time_arg_val(6) != 0;
    constexpr uint32_t ES_G = get_compile_time_arg_val(7);
    constexpr bool GAMMA_RESIDENT = get_compile_time_arg_val(8) != 0;
    constexpr uint32_t W = get_compile_time_arg_val(9);
    constexpr uint32_t H = get_compile_time_arg_val(10);
    constexpr uint32_t Ht_img = get_compile_time_arg_val(11);
    constexpr uint32_t cb_x_rm = get_compile_time_arg_val(12);
    constexpr uint32_t cb_x = get_compile_time_arg_val(13);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(14);
    constexpr uint32_t cb_gamma = get_compile_time_arg_val(15);
    constexpr uint32_t cb_gamma_rm = get_compile_time_arg_val(16);
    constexpr bool GAMMA_TILE = get_compile_time_arg_val(17) != 0;
    constexpr uint32_t Wt_full = get_compile_time_arg_val(18);  // Wt is the per-core slice width
    constexpr bool IN_SHARDED = get_compile_time_arg_val(19) != 0;  // x shard resident in cb_x (zero-copy)
    constexpr bool ROW_FLAT = get_compile_time_arg_val(20) != 0;    // RM shard: tile-row = 32 consecutive sticks
    constexpr bool IN_SHARD_RM = get_compile_time_arg_val(21) != 0;  // RM x sticks live in the local shard
    constexpr uint32_t IN_STRIDE = get_compile_time_arg_val(22);     // shard stick pitch in bytes
    constexpr auto x_args = TensorAccessorArgs<23>();
    constexpr auto g_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();

    constexpr uint32_t num_seg = Wt / Sb;
    constexpr uint32_t SEG_ELEMS = Sb * TILE_DIM;
    constexpr uint32_t SEG_BYTES_X = SEG_ELEMS * ES_X;
    constexpr uint32_t SEG_BYTES_G = SEG_ELEMS * ES_G;
    const uint32_t slice_off_x = slice_tile0 * TILE_DIM * ES_X;  // byte offset of the slice in an RM stick
    const uint32_t slice_off_g = slice_tile0 * TILE_DIM * ES_G;
    constexpr uint32_t valid_w_last = W - TILE_DIM * (Wt_full - 1);  // 1..32

    Noc noc;
    DataflowBuffer dfb_x(cb_x);
    DataflowBuffer dfb_x_rm(cb_x_rm);
    DataflowBuffer dfb_gamma(cb_gamma);
    DataflowBuffer dfb_gamma_rm(cb_gamma_rm);

    const uint32_t x_tile_bytes = IS_RM ? 0 : dfb_x.get_tile_size();
    const auto x_acc = TensorAccessor(x_args, x_addr, IS_RM ? W * ES_X : x_tile_bytes);
    const uint32_t g_page_bytes = (HAS_GAMMA && GAMMA_TILE) ? dfb_gamma.get_tile_size() : W * ES_G;
    const auto g_acc = TensorAccessor(g_args, g_addr, g_page_bytes);

    // ---- one x segment: rb tile-rows x Sb tiles, pushed to cb_x (TILE) / cb_x_rm (RM) ----
    auto load_x_segment = [&](uint32_t tr0, uint32_t rb, uint32_t seg) {
        if constexpr (IN_SHARDED) {
            return;  // the whole shard was published once up front
        }
        if constexpr (IS_RM) {
            for (uint32_t rr = 0; rr < rb; ++rr) {
                const uint32_t tr = tr0 + rr;
                uint32_t valid_rows, stick0;
                if constexpr (ROW_FLAT) {
                    const uint32_t left = core_sticks - tr * TILE_DIM;
                    valid_rows = left < TILE_DIM ? left : TILE_DIM;
                    stick0 = first_stick + tr * TILE_DIM;
                } else {
                    const uint32_t img = tr / Ht_img;
                    const uint32_t r = tr % Ht_img;
                    valid_rows = (H - r * TILE_DIM) < TILE_DIM ? (H - r * TILE_DIM) : TILE_DIM;
                    stick0 = img * H + r * TILE_DIM;
                }
                dfb_x_rm.reserve_back(TILE_DIM);
                const uint32_t base = dfb_x_rm.get_write_ptr();
                for (uint32_t k = valid_rows; k < TILE_DIM; ++k) {
                    noc.async_write_zeros(dfb_x_rm, SEG_BYTES_X, {.offset_bytes = k * SEG_BYTES_X});
                }
                const uint32_t off = slice_off_x + seg * SEG_BYTES_X;
                const uint32_t avail = W * ES_X - off;
                const uint32_t bytes = avail < SEG_BYTES_X ? avail : SEG_BYTES_X;
                for (uint32_t k = 0; k < valid_rows; ++k) {
                    if constexpr (IN_SHARD_RM) {  // local shard: stick (stick0 - first_stick + k) at pitch IN_STRIDE
                        noc_async_read(
                            get_noc_addr(x_addr + (stick0 - first_stick + k) * IN_STRIDE + off),
                            base + k * SEG_BYTES_X,
                            bytes);
                    } else {
                        noc_async_read(x_acc.get_noc_addr(stick0 + k, off), base + k * SEG_BYTES_X, bytes);
                    }
                }
                if (valid_rows < TILE_DIM) {
                    noc.write_zeros_l1_barrier();
                }
                noc_async_read_barrier();
                if (bytes < SEG_BYTES_X) {  // W tail (only the last segment)
                    for (uint32_t k = 0; k < valid_rows; ++k) {
                        if constexpr (ES_X == 2) {
                            zero_elems<uint16_t>(base + k * SEG_BYTES_X, bytes / ES_X, SEG_ELEMS);
                        } else {
                            zero_elems<uint32_t>(base + k * SEG_BYTES_X, bytes / ES_X, SEG_ELEMS);
                        }
                    }
                }
                dfb_x_rm.push_back(TILE_DIM);
            }
        } else {
            dfb_x.reserve_back(rb * Sb);
            const uint32_t base = dfb_x.get_write_ptr();
            for (uint32_t rr = 0; rr < rb; ++rr) {
                const uint32_t tr = tr0 + rr;
                for (uint32_t c = 0; c < Sb; ++c) {
                    noc_async_read_tile(
                        tr * Wt_full + slice_tile0 + seg * Sb + c, x_acc, base + (rr * Sb + c) * x_tile_bytes);
                }
            }
            noc_async_read_barrier();
            if constexpr (valid_w_last < TILE_DIM) {
                if (is_last_slice && seg == num_seg - 1) {
                    for (uint32_t rr = 0; rr < rb; ++rr) {
                        const uint32_t t = base + (rr * Sb + Sb - 1) * x_tile_bytes;
                        if constexpr (ES_X == 2) {
                            zero_w_pad_of_tile<uint16_t>(t, valid_w_last);
                        } else {
                            zero_w_pad_of_tile<uint32_t>(t, valid_w_last);
                        }
                    }
                }
            }
            dfb_x.push_back(rb * Sb);
        }
    };

    // ---- one gamma segment: Sb tiles, stick elements scattered into row 0 of each tile ----
    auto load_gamma_segment = [&](uint32_t seg) {
        if constexpr (GAMMA_TILE) {
            // TILE gamma: tile column t holds the gamma slice in row 0; read tiles directly, no scatter.
            dfb_gamma.reserve_back(Sb);
            const uint32_t gbase = dfb_gamma.get_write_ptr();
            const uint32_t gtb = dfb_gamma.get_tile_size();
            for (uint32_t c = 0; c < Sb; ++c) {
                noc_async_read_tile(slice_tile0 + seg * Sb + c, g_acc, gbase + c * gtb);
            }
            noc_async_read_barrier();
            dfb_gamma.push_back(Sb);
            return;
        }
        const uint32_t g_tile_bytes = dfb_gamma.get_tile_size();
        const uint32_t scratch = dfb_gamma_rm.get_write_ptr();
        noc.async_write_zeros(dfb_gamma_rm, SEG_BYTES_G);
        noc.write_zeros_l1_barrier();
        const uint32_t off = slice_off_g + seg * SEG_BYTES_G;
        const uint32_t avail = W * ES_G - off;
        noc_async_read(g_acc.get_noc_addr(0, off), scratch, avail < SEG_BYTES_G ? avail : SEG_BYTES_G);
        noc_async_read_barrier();
        dfb_gamma.reserve_back(Sb);
        const uint32_t tiles = dfb_gamma.get_write_ptr();
        noc.async_write_zeros(dfb_gamma, Sb * g_tile_bytes);
        noc.write_zeros_l1_barrier();
        for (uint32_t t = 0; t < Sb; ++t) {
            if constexpr (ES_G == 2) {
                volatile tt_l1_ptr uint16_t* src = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch) + t * TILE_DIM;
                volatile tt_l1_ptr uint16_t* dst = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tiles + t * g_tile_bytes);
                for (uint32_t j = 0; j < FACE_DIM; ++j) {
                    dst[j] = src[j];
                    dst[FACE_ELEMS + j] = src[FACE_DIM + j];
                }
            } else {
                volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch) + t * TILE_DIM;
                volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tiles + t * g_tile_bytes);
                for (uint32_t j = 0; j < FACE_DIM; ++j) {
                    dst[j] = src[j];
                    dst[FACE_ELEMS + j] = src[FACE_DIM + j];
                }
            }
        }
        dfb_gamma.push_back(Sb);
    };

    // ---- schedule ----
    dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
        __builtin_bit_cast(float, scaler_bits));

    if constexpr (HAS_GAMMA) {
        if constexpr (!GAMMA_TILE) {
            dfb_gamma_rm.reserve_back(1);  // private scratch, never pushed
        }
        if constexpr (GAMMA_RESIDENT) {
            for (uint32_t seg = 0; seg < num_seg; ++seg) {
                load_gamma_segment(seg);
            }
        }
    }

    if constexpr (IN_SHARDED) {
        // Native sharding: cb_x is backed by this core's input shard. Publish it in place (no NoC read);
        // zero the W pad of the last tile column in place so it cannot enter the reduction.
        const uint32_t total = rows_core * Wt;
        if (total > 0) {
            dfb_x.reserve_back(total);
            if constexpr (!IS_RM && valid_w_last < TILE_DIM) {
                const uint32_t base = dfb_x.get_write_ptr();
                for (uint32_t rr = 0; rr < rows_core; ++rr) {
                    const uint32_t t = base + (rr * Wt + Wt - 1) * x_tile_bytes;
                    if constexpr (ES_X == 2) {
                        zero_w_pad_of_tile<uint16_t>(t, valid_w_last);
                    } else {
                        zero_w_pad_of_tile<uint32_t>(t, valid_w_last);
                    }
                }
            }
            dfb_x.push_back(total);
        }
    }

    for (uint32_t b = 0; b < rows_core; b += Rb) {
        const uint32_t rb = (rows_core - b) < Rb ? (rows_core - b) : Rb;
        const uint32_t tr0 = first_tile_row + b;
        // pass 1 (Σx²): R1 keeps these tiles resident for pass 2.
        for (uint32_t seg = 0; seg < num_seg; ++seg) {
            load_x_segment(tr0, rb, seg);
        }
        if constexpr (RESIDENT) {
            if constexpr (HAS_GAMMA && !GAMMA_RESIDENT) {
                for (uint32_t seg = 0; seg < num_seg; ++seg) {
                    load_gamma_segment(seg);
                }
            }
        } else {
            // pass 2: re-read x segments (+ stream gamma segment alongside)
            for (uint32_t seg = 0; seg < num_seg; ++seg) {
                load_x_segment(tr0, rb, seg);
                if constexpr (HAS_GAMMA && !GAMMA_RESIDENT) {
                    load_gamma_segment(seg);
                }
            }
        }
    }
}

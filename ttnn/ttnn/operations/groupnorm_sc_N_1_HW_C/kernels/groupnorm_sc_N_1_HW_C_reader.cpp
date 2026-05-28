// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// groupnorm_sc_N_1_HW_C — Reader kernel (NCRISC)
//
// Responsibilities (per batch n):
//   1. One-shot startup: build scaler_one (1.0 reduce scaler), inv_N_scalar
//      (1/N_per_g at (0,0)), eps_scalar (eps at (0,0)).
//   2. Phase R: for each (g, T_in_g_span, r): push one mask tile to
//      cb_mask_stream and one input tile to cb_input_tiles_R (or 32 sticks to
//      cb_input_rm_R for RM input).
//   3. Phase A: per output T, push HAS_GAMMA gamma sticks (replicated 32x),
//      HAS_BETA beta sticks (replicated 32x), then for each g touching T push
//      a mask tile to cb_mask_stream, then for each r push an input tile.
//
// Iteration order matches the compute kernel exactly.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

// -----------------------------------------------------------------------------
// CB indices — must match groupnorm_sc_N_1_HW_C_program_descriptor.py
// -----------------------------------------------------------------------------
constexpr uint32_t CB_INPUT_RM_R = 0;
constexpr uint32_t CB_INPUT_RM_A = 1;
constexpr uint32_t CB_INPUT_TILES_R = 2;
constexpr uint32_t CB_INPUT_TILES_A = 3;
constexpr uint32_t CB_GAMMA_RM = 4;
constexpr uint32_t CB_BETA_RM = 5;
constexpr uint32_t CB_SCALER_ONE = 8;
constexpr uint32_t CB_MASK_STREAM = 9;
constexpr uint32_t CB_INV_N_SCALAR = 10;

constexpr uint32_t TILE_DIM = 32;
constexpr uint32_t FACE_DIM = 16;
constexpr uint32_t FACE_HW = FACE_DIM * FACE_DIM;  // 256 bf16 = 512 bytes
constexpr uint32_t TILE_HW = TILE_DIM * TILE_DIM;  // 1024 bf16 = 2048 bytes
constexpr uint16_t BF16_ONE = 0x3F80;              // 1.0 in bf16 (top 16 bits of 0x3F800000)
constexpr uint16_t BF16_ZERO = 0x0000;
constexpr uint32_t CHUNK_BYTES = TILE_DIM * 2;  // 64 bytes — one tile-column chunk per stick (bf16)

// -----------------------------------------------------------------------------
// Helpers — mask construction, scalar tile fill, L1 zeroing.
// -----------------------------------------------------------------------------

// Convert a float to its bf16 bit pattern (round-toward-zero by truncation).
FORCE_INLINE uint16_t float_to_bf16_bits(float f) {
    uint32_t bits;
    __builtin_memcpy(&bits, &f, sizeof(bits));
    return static_cast<uint16_t>(bits >> 16);
}

// Zero a region of L1 (in 4-byte words). `n_bytes` must be a multiple of 4.
FORCE_INLINE void zero_l1(uint32_t l1_addr, uint32_t n_bytes) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_addr);
    uint32_t n = n_bytes >> 2;
    for (uint32_t i = 0; i < n; ++i) {
        p[i] = 0;
    }
}

// Write a row-replicated mask tile to L1 at l1_addr.
//   lane_start, lane_end ∈ [0, 32], lane_start ≤ lane_end.
//   Pattern[c] = 1.0 if lane_start ≤ c < lane_end else 0.0
// Tile face layout: 4 faces of 16x16 bf16 each, stored as:
//   face 0 (rows 0-15, cols 0-15) | face 1 (rows 0-15, cols 16-31) |
//   face 2 (rows 16-31, cols 0-15) | face 3 (rows 16-31, cols 16-31)
FORCE_INLINE void write_mask_tile(uint32_t l1_addr, uint32_t lane_start, uint32_t lane_end) {
    // Build the 32-element row pattern as 16 uint16_t pairs in 8 uint32_t words.
    uint16_t pattern[TILE_DIM];
    for (uint32_t c = 0; c < TILE_DIM; ++c) {
        pattern[c] = (c >= lane_start && c < lane_end) ? BF16_ONE : BF16_ZERO;
    }

    volatile tt_l1_ptr uint16_t* tile = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_addr);

    // Face 0: rows 0-15, cols 0-15 (pattern[0..15]) — replicate pattern row 16 times.
    for (uint32_t r = 0; r < FACE_DIM; ++r) {
        uint32_t face_row_base = r * FACE_DIM;
        for (uint32_t c = 0; c < FACE_DIM; ++c) {
            tile[face_row_base + c] = pattern[c];
        }
    }
    // Face 1: rows 0-15, cols 16-31 (pattern[16..31])
    for (uint32_t r = 0; r < FACE_DIM; ++r) {
        uint32_t face_row_base = FACE_HW + r * FACE_DIM;
        for (uint32_t c = 0; c < FACE_DIM; ++c) {
            tile[face_row_base + c] = pattern[FACE_DIM + c];
        }
    }
    // Face 2: rows 16-31, cols 0-15 — same pattern as face 0.
    for (uint32_t r = 0; r < FACE_DIM; ++r) {
        uint32_t face_row_base = 2 * FACE_HW + r * FACE_DIM;
        for (uint32_t c = 0; c < FACE_DIM; ++c) {
            tile[face_row_base + c] = pattern[c];
        }
    }
    // Face 3: rows 16-31, cols 16-31 — same as face 1.
    for (uint32_t r = 0; r < FACE_DIM; ++r) {
        uint32_t face_row_base = 3 * FACE_HW + r * FACE_DIM;
        for (uint32_t c = 0; c < FACE_DIM; ++c) {
            tile[face_row_base + c] = pattern[FACE_DIM + c];
        }
    }
}

// Write a scalar tile (value at (0,0), zero elsewhere) to L1 at l1_addr.
FORCE_INLINE void write_scalar_tile_bf16(uint32_t l1_addr, float value) {
    // Zero the full tile first.
    zero_l1(l1_addr, TILE_HW * 2);
    // Set position (0,0) — face 0, row 0, col 0 — to bf16(value).
    volatile tt_l1_ptr uint16_t* tile = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_addr);
    tile[0] = float_to_bf16_bits(value);
}

// Push one row-replicated mask tile to cb_mask_stream.
FORCE_INLINE void push_mask_tile(uint32_t lane_start, uint32_t lane_end) {
    cb_reserve_back(CB_MASK_STREAM, 1);
    uint32_t l1_addr = get_write_ptr(CB_MASK_STREAM);
    write_mask_tile(l1_addr, lane_start, lane_end);
    cb_push_back(CB_MASK_STREAM, 1);
}

// -----------------------------------------------------------------------------
// kernel_main
// -----------------------------------------------------------------------------

void kernel_main() {
    // ----- CT args -----
    constexpr uint32_t INPUT_LAYOUT_CODE = get_compile_time_arg_val(0);  // 0=TILE, 1=RM
    constexpr uint32_t HAS_GAMMA = get_compile_time_arg_val(1);
    constexpr uint32_t HAS_BETA = get_compile_time_arg_val(2);
    constexpr uint32_t N = get_compile_time_arg_val(3);
    constexpr uint32_t HW = get_compile_time_arg_val(4);
    constexpr uint32_t C = get_compile_time_arg_val(5);
    constexpr uint32_t G = get_compile_time_arg_val(6);
    constexpr uint32_t Cg = get_compile_time_arg_val(7);
    constexpr uint32_t Ht = get_compile_time_arg_val(8);
    constexpr uint32_t Ct = get_compile_time_arg_val(9);
    constexpr uint32_t EPS_BITS = get_compile_time_arg_val(10);
    constexpr uint32_t STICK_BYTES = get_compile_time_arg_val(11);

    // TensorAccessorArgs — declared unconditionally with placeholders for absent
    // tensors, chained via next_compile_time_args_offset().
    constexpr auto input_args = TensorAccessorArgs<12>();
    [[maybe_unused]] constexpr auto gamma_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto beta_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();

    // ----- RT args -----
    uint32_t input_addr = get_arg_val<uint32_t>(0);
    [[maybe_unused]] uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    [[maybe_unused]] uint32_t beta_addr = get_arg_val<uint32_t>(2);

    // ----- Constants derived from CT args -----
    constexpr uint32_t TILE_BYTES = TILE_HW * 2;  // bf16 tile size = 2048 bytes
    // Convert EPS_BITS (uint32 IEEE-754 single bit pattern) → float at runtime.
    float eps_f;
    {
        uint32_t bits = EPS_BITS;
        __builtin_memcpy(&eps_f, &bits, sizeof(eps_f));
    }
    const float INV_N = 1.0f / static_cast<float>(HW * Cg);

    // Accessors. Use the args' default AlignedPageSize so RM input (stick-sized
    // pages) and TILE input (tile-sized pages) both work.
    const auto input_accessor = TensorAccessor(input_args, input_addr);
    // We construct gamma/beta accessors only if needed at runtime — but declare
    // them at compile time so chain stays consistent. Build them inside the
    // `if constexpr (HAS_*)` blocks.

    // ============================================================
    // ONE-SHOT STARTUP: build scaler/constant tiles
    // ============================================================

    // cb_scaler_one — 1.0 reduce scaler (SUM REDUCE_SCALAR, row-0 fill).
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<CB_SCALER_ONE, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_SCALAR>();

    // cb_inv_N_scalar — manual: 1/N_per_g at position (0,0).
    // (eps is bit-packed into the compute kernel's CT args via SfpuAddScalar —
    // no scalar CB needed on the compute side.)
    {
        cb_reserve_back(CB_INV_N_SCALAR, 1);
        uint32_t l1 = get_write_ptr(CB_INV_N_SCALAR);
        write_scalar_tile_bf16(l1, INV_N);
        cb_push_back(CB_INV_N_SCALAR, 1);
    }
    (void)eps_f;  // retained for potential future use; eps currently flows via CT arg

    // ============================================================
    // PER-BATCH MAIN LOOP
    // ============================================================
    for (uint32_t n = 0; n < N; ++n) {
        // ------------------------------------------------------------
        // PHASE R: per group g, per tile T_in_g_span, per HW-row r,
        //   push 1 mask + 1 input tile (or 32 input RM sticks).
        // ------------------------------------------------------------
        for (uint32_t g = 0; g < G; ++g) {
            // Channel range for group g: [g*Cg, min((g+1)*Cg, C))
            const uint32_t g_start_ch = g * Cg;
            const uint32_t g_end_ch_raw = (g + 1) * Cg;
            const uint32_t g_end_ch = g_end_ch_raw < C ? g_end_ch_raw : C;
            const uint32_t T_first = g_start_ch / TILE_DIM;
            const uint32_t T_last = (g_end_ch - 1) / TILE_DIM;  // inclusive

            for (uint32_t T = T_first; T <= T_last; ++T) {
                // Lane range within tile T for group g.
                const uint32_t T_base = T * TILE_DIM;
                const uint32_t lane_start = (g_start_ch > T_base) ? (g_start_ch - T_base) : 0;
                const uint32_t lane_end_raw = (g_end_ch > T_base + TILE_DIM) ? TILE_DIM : (g_end_ch - T_base);
                // Cap by valid C within tile (handles partial last C-tile).
                const uint32_t c_in_tile = (T_base + TILE_DIM <= C) ? TILE_DIM : (C - T_base);
                const uint32_t lane_end = lane_end_raw < c_in_tile ? lane_end_raw : c_in_tile;

                for (uint32_t r = 0; r < Ht; ++r) {
                    // Push the mask tile for (g, T)
                    push_mask_tile(lane_start, lane_end);

                    // Push the input tile (n, r, T)
                    if constexpr (INPUT_LAYOUT_CODE == 0) {
                        // TILE_LAYOUT input — direct tile read
                        const uint32_t page_id = n * Ht * Ct + r * Ct + T;
                        cb_reserve_back(CB_INPUT_TILES_R, 1);
                        uint32_t l1 = get_write_ptr(CB_INPUT_TILES_R);
                        noc_async_read_tile(page_id, input_accessor, l1);
                        noc_async_read_barrier();
                        cb_push_back(CB_INPUT_TILES_R, 1);
                    } else {
                        // ROW_MAJOR_LAYOUT input — read 32 sticks (one tile-row block) of
                        // the c-tile column T. Valid bytes in this chunk may be < 64 when
                        // C is not tile-aligned (partial last C-tile).
                        const uint32_t valid_ch = (T_base + TILE_DIM <= C) ? TILE_DIM : (C - T_base);
                        const uint32_t valid_bytes = valid_ch * 2;
                        const uint32_t start_stick = n * HW + r * TILE_DIM;
                        const uint32_t byte_offset = T * CHUNK_BYTES;

                        cb_reserve_back(CB_INPUT_RM_R, TILE_DIM);
                        uint32_t l1_base = get_write_ptr(CB_INPUT_RM_R);
                        // If partial last-C-tile, zero-fill the buffer so padding lanes are 0.
                        if (valid_bytes < CHUNK_BYTES) {
                            zero_l1(l1_base, TILE_DIM * CHUNK_BYTES);
                        }
                        for (uint32_t row = 0; row < TILE_DIM; ++row) {
                            uint64_t noc_addr = input_accessor.get_noc_addr(start_stick + row, byte_offset);
                            noc_async_read(noc_addr, l1_base + row * CHUNK_BYTES, valid_bytes);
                        }
                        noc_async_read_barrier();
                        cb_push_back(CB_INPUT_RM_R, TILE_DIM);
                    }
                }
            }
        }

        // ------------------------------------------------------------
        // PHASE A: per output T, push gamma/beta (if any), then for each g
        //   touching T push a mask tile, then for each r push an input tile.
        //
        // Compute interleaves expansion + apply per T (see compute kernel).
        // The expected push sequence per T is:
        //   1. (Compute does NOT consume anything new for the means/rcp_std
        //      expansion EXCEPT the masks below.)
        //   2. for g touching T (in increasing order): push 1 mask tile.
        //   3. If HAS_GAMMA: push 32 sticks of gamma (chunk for T) to cb_gamma_rm.
        //   4. If HAS_BETA: push 32 sticks of beta (chunk for T) to cb_beta_rm.
        //   5. for r in 0..Ht-1: push 1 input tile.
        // ------------------------------------------------------------
        for (uint32_t T = 0; T < Ct; ++T) {
            const uint32_t T_base = T * TILE_DIM;
            const uint32_t c_in_tile = (T_base + TILE_DIM <= C) ? TILE_DIM : (C - T_base);

            // Push masks for every group g touching T.
            for (uint32_t g = 0; g < G; ++g) {
                const uint32_t g_start_ch = g * Cg;
                const uint32_t g_end_ch_raw = (g + 1) * Cg;
                const uint32_t g_end_ch = g_end_ch_raw < C ? g_end_ch_raw : C;
                // Does group g touch tile T? Intersection [g_start_ch, g_end_ch) ∩ [T_base, T_base+TILE_DIM)
                if (g_start_ch >= T_base + TILE_DIM) {
                    continue;
                }
                if (g_end_ch <= T_base) {
                    continue;
                }
                const uint32_t lane_start = (g_start_ch > T_base) ? (g_start_ch - T_base) : 0;
                const uint32_t lane_end_raw = (g_end_ch > T_base + TILE_DIM) ? TILE_DIM : (g_end_ch - T_base);
                const uint32_t lane_end = lane_end_raw < c_in_tile ? lane_end_raw : c_in_tile;
                push_mask_tile(lane_start, lane_end);
            }

            // Push gamma sticks (replicated 32×) — chunk for T.
            if constexpr (HAS_GAMMA) {
                const auto gamma_accessor = TensorAccessor(gamma_args, gamma_addr);
                const uint32_t valid_bytes = c_in_tile * 2;
                const uint32_t byte_offset = T * CHUNK_BYTES;

                cb_reserve_back(CB_GAMMA_RM, TILE_DIM);
                uint32_t l1_base = get_write_ptr(CB_GAMMA_RM);
                if (valid_bytes < CHUNK_BYTES) {
                    zero_l1(l1_base, TILE_DIM * CHUNK_BYTES);
                }
                // Gamma is shape (1, 1, 1, C) — only 1 stick. Read it 32 times into successive L1 rows.
                for (uint32_t row = 0; row < TILE_DIM; ++row) {
                    uint64_t noc_addr = gamma_accessor.get_noc_addr(0, byte_offset);
                    noc_async_read(noc_addr, l1_base + row * CHUNK_BYTES, valid_bytes);
                }
                noc_async_read_barrier();
                cb_push_back(CB_GAMMA_RM, TILE_DIM);
            }

            // Push beta sticks (replicated 32×) — chunk for T.
            if constexpr (HAS_BETA) {
                const auto beta_accessor = TensorAccessor(beta_args, beta_addr);
                const uint32_t valid_bytes = c_in_tile * 2;
                const uint32_t byte_offset = T * CHUNK_BYTES;

                cb_reserve_back(CB_BETA_RM, TILE_DIM);
                uint32_t l1_base = get_write_ptr(CB_BETA_RM);
                if (valid_bytes < CHUNK_BYTES) {
                    zero_l1(l1_base, TILE_DIM * CHUNK_BYTES);
                }
                for (uint32_t row = 0; row < TILE_DIM; ++row) {
                    uint64_t noc_addr = beta_accessor.get_noc_addr(0, byte_offset);
                    noc_async_read(noc_addr, l1_base + row * CHUNK_BYTES, valid_bytes);
                }
                noc_async_read_barrier();
                cb_push_back(CB_BETA_RM, TILE_DIM);
            }

            // Push input tiles for the inner r-loop.
            for (uint32_t r = 0; r < Ht; ++r) {
                if constexpr (INPUT_LAYOUT_CODE == 0) {
                    const uint32_t page_id = n * Ht * Ct + r * Ct + T;
                    cb_reserve_back(CB_INPUT_TILES_A, 1);
                    uint32_t l1 = get_write_ptr(CB_INPUT_TILES_A);
                    noc_async_read_tile(page_id, input_accessor, l1);
                    noc_async_read_barrier();
                    cb_push_back(CB_INPUT_TILES_A, 1);
                } else {
                    const uint32_t valid_bytes = c_in_tile * 2;
                    const uint32_t start_stick = n * HW + r * TILE_DIM;
                    const uint32_t byte_offset = T * CHUNK_BYTES;

                    cb_reserve_back(CB_INPUT_RM_A, TILE_DIM);
                    uint32_t l1_base = get_write_ptr(CB_INPUT_RM_A);
                    if (valid_bytes < CHUNK_BYTES) {
                        zero_l1(l1_base, TILE_DIM * CHUNK_BYTES);
                    }
                    for (uint32_t row = 0; row < TILE_DIM; ++row) {
                        uint64_t noc_addr = input_accessor.get_noc_addr(start_stick + row, byte_offset);
                        noc_async_read(noc_addr, l1_base + row * CHUNK_BYTES, valid_bytes);
                    }
                    noc_async_read_barrier();
                    cb_push_back(CB_INPUT_RM_A, TILE_DIM);
                }
            }
        }
    }
}

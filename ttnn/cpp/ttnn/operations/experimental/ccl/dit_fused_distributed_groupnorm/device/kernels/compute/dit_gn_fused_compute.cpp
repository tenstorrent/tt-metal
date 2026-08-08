// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Fused distributed GroupNorm compute kernel.
 *
 * PRE:  soft-accumulate per-group sum/sumsq from bf16 TILE face layout into L1
 *       fp32 arrays; write one stick (num_groups*16 B) to stats_local_cb
 *       (or stats_gathered_cb when is_local).
 * AG:   writer/forwarder — compute waits on stats_gathered_cb.
 * POST: merge ring sticks → μ, inv_std; normalize each input tile into the
 *       output CB in L1 face layout and push to output_cb.
 *
 * v1: bf16 input, no input_mask (C%32==0, C%num_groups==0). Groups may be
 * sub-tile (e.g. C=128,G=32 → 4 ch/group), so a whole-tile LLK reduce cannot
 * isolate a group without an input_mask + GN-packed weights. Instead we keep a
 * plain [1,1,1,C] γ/β contract and do correctly-synchronized soft-float.
 *
 * TRISC threading model (why this is not a plain C loop):
 *   kernel_main() runs on all three TRISC threads (UNPACK/MATH/PACK). Any code
 *   NOT wrapped in MATH()/UNPACK()/PACK() executes on ALL of them. The CB flow-
 *   control primitives are single-thread (cb_wait_front/cb_pop_front → UNPACK;
 *   cb_reserve_back/cb_push_back → PACK). So:
 *     • All soft-float L1 mutation is gated to MATH({...}) — one writer.
 *     • L1 byte addresses are obtained via ckernel::get_tile_address (read ptr)
 *       and get_write_tile_address (write ptr) — the proven mailbox+<<4 broadcast
 *       from tt_metal/hw/inc/api/compute/cb_api.h. These are UNGATED so all three
 *       threads participate in the mailbox handshake; that handshake also
 *       synchronizes MATH to UNPACK's cb_wait_front / PACK's cb_reserve_back.
 *     • Do NOT use free get_read_ptr/get_write_ptr or CircularBuffer::get_*_ptr:
 *       on TRISC those return raw 16B-word, per-thread-stale fifo pointers.
 *     • math_done_signal() is the reverse handshake: MATH announces it is done
 *       touching L1 so the UNPACK(pop)/PACK(push) owners release the CB only
 *       after the reads/writes retire. On the write path the token carries an L1
 *       read-back (RAW fence) so MATH's stores land before the signal (fence is a
 *       nop on these cores).
 */

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/dataflow/circular_buffer.h"

namespace {

union F32Bits {
    float f;
    uint32_t u;
};

ALWI float bf16_to_f32(uint16_t bf) {
    F32Bits b;
    b.u = static_cast<uint32_t>(bf) << 16;
    return b.f;
}

ALWI uint16_t f32_to_bf16(float f) {
    F32Bits b;
    b.f = f;
    b.u += 0x7fffu + ((b.u >> 16) & 1u);
    return static_cast<uint16_t>(b.u >> 16);
}

ALWI float soft_rsqrt(float x) {
    F32Bits b;
    b.f = x;
    b.u = 0x5f3759dfu - (b.u >> 1);
    float y = b.f;
    y = y * (1.5f - 0.5f * x * y * y);
    y = y * (1.5f - 0.5f * x * y * y);
    return y;
}

// bf16 TILE face index for element (r,c) in a 32x32 tile.
ALWI uint32_t tile_face_offset(uint32_t r, uint32_t c) {
    const uint32_t face_r = r >> 4;
    const uint32_t face_c = c >> 4;
    const uint32_t within_r = r & 15u;
    const uint32_t within_c = c & 15u;
    const uint32_t face = (face_r << 1) + face_c;  // 0..3
    return face * 256u + within_r * 16u + within_c;
}

// Write-pointer analogue of ckernel::get_tile_address (cb_api.h). PACK owns
// fifo_wr_ptr (llk_wait_for_free_tiles / llk_push_tiles advance it, in 16B-word
// units), so PACK computes the byte address (<<4) and fans it out to MATH and
// UNPACK over the RISC↔RISC mailboxes. Must be called AFTER cb_reserve_back on
// the target CB, and UNGATED so all threads reach the matching mailbox_read.
//   Channels: T2→T1 (PACK→MATH), T2→T0 (PACK→UNPACK); 1 write / 1 read each.
ALWI uint32_t get_write_tile_address(uint32_t cb_id, uint32_t tile_index) {
    uint32_t address = 0;
    PACK({
        // get_operand_id(cb_id) == cb_id (llk_operands.h); use cb_id directly so
        // this compiles identically on every TRISC thread.
        uint32_t base_address = get_local_cb_interface(cb_id).fifo_wr_ptr;
        uint32_t offset_address = get_local_cb_interface(cb_id).fifo_page_size * tile_index;
        address = (base_address + offset_address) << 4;  // 16B-word ptr -> byte address
        ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, address);
        ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, address);
    })
    MATH(address = ckernel::mailbox_read(ckernel::ThreadId::PackThreadId);)
    UNPACK(address = ckernel::mailbox_read(ckernel::ThreadId::PackThreadId);)
    return address;
}

// Reverse handshake: MATH signals it finished the current CB region so UNPACK
// (which owns cb_pop_front) and PACK (which owns cb_push_back) release it only
// after MATH's accesses retire. `token` should carry a read-back of the last L1
// store on write paths (RAW dependency → drains MATH's store buffer before the
// mailbox write, since `fence` is a nop and the L/S unit reorders); pass 1u on
// pure read paths.
//   Channels: T1→T0 (MATH→UNPACK), T1→T2 (MATH→PACK); 1 write / 1 read each.
ALWI void math_done_signal(uint32_t token) {
    MATH({
        ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, token);
        ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, token);
    })
    UNPACK((void)ckernel::mailbox_read(ckernel::ThreadId::MathThreadId);)
    PACK((void)ckernel::mailbox_read(ckernel::ThreadId::MathThreadId);)
}

}  // namespace

void kernel_main() {
    constexpr uint32_t input_cb = get_compile_time_arg_val(0);
    constexpr uint32_t stats_local_cb = get_compile_time_arg_val(1);
    constexpr uint32_t stats_gathered_cb = get_compile_time_arg_val(2);
    constexpr uint32_t weight_cb = get_compile_time_arg_val(3);
    constexpr uint32_t bias_cb = get_compile_time_arg_val(4);
    constexpr uint32_t epsilon_cb = get_compile_time_arg_val(5);
    constexpr uint32_t output_cb = get_compile_time_arg_val(6);
    constexpr uint32_t num_tile_cols = get_compile_time_arg_val(7);
    constexpr uint32_t block_size = get_compile_time_arg_val(8);
    constexpr uint32_t num_groups = get_compile_time_arg_val(9);
    constexpr uint32_t channels_per_group = get_compile_time_arg_val(10);
    constexpr uint32_t stick_bytes = get_compile_time_arg_val(11);
    constexpr uint32_t ring_size = get_compile_time_arg_val(12);
    constexpr uint32_t is_local = get_compile_time_arg_val(13);
    constexpr uint32_t has_weight = get_compile_time_arg_val(14);
    constexpr uint32_t has_bias = get_compile_time_arg_val(15);
    constexpr uint32_t weight_is_tile = get_compile_time_arg_val(16);
    constexpr uint32_t bias_is_tile = get_compile_time_arg_val(17);
    constexpr uint32_t C = get_compile_time_arg_val(18);
    constexpr uint32_t count_bits = get_compile_time_arg_val(19);
    constexpr uint32_t weight_is_fp32 = get_compile_time_arg_val(20);
    constexpr uint32_t bias_is_fp32 = get_compile_time_arg_val(21);

    const uint32_t num_tile_rows = get_arg_val<uint32_t>(0);

    constexpr uint32_t TILE_W = 32u;
    constexpr uint32_t TILE_HW = 1024u;
    constexpr uint32_t FACE_HW = 256u;

    // Soft-float accumulators / stats. These live on MATH's stack (only touched
    // inside MATH({...})); the copies on UNPACK/PACK are unused.
    float sum[num_groups];
    float sumsq[num_groups];
    MATH({
        for (uint32_t g = 0; g < num_groups; g++) {
            sum[g] = 0.f;
            sumsq[g] = 0.f;
        }
    })
    F32Bits count_bits_u;
    count_bits_u.u = count_bits;
    const float count_local = count_bits_u.f;

    // ===================== PRE =====================
    for (uint32_t tile_row = 0; tile_row < num_tile_rows; tile_row++) {
        for (uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
            const uint32_t tiles_in_block =
                ((num_tile_cols - col_tile) >= block_size) ? block_size : (num_tile_cols - col_tile);
            cb_wait_front(input_cb, tiles_in_block);
            // UNGATED: mailbox broadcast of the read base; MATH blocks here until
            // UNPACK has passed cb_wait_front (data ready).
            const uint32_t in_addr = ckernel::get_tile_address(input_cb, 0);
            MATH({
                volatile tt_l1_ptr uint16_t* tiles = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(in_addr);
                for (uint32_t i = 0; i < tiles_in_block; i++) {
                    const uint32_t c0 = (col_tile + i) * TILE_W;
                    volatile tt_l1_ptr uint16_t* tile = tiles + i * TILE_HW;
                    for (uint32_t r = 0; r < 32u; r++) {
                        for (uint32_t c = 0; c < 32u; c++) {
                            const float x = bf16_to_f32(tile[tile_face_offset(r, c)]);
                            const uint32_t g = (c0 + c) / channels_per_group;
                            sum[g] += x;
                            sumsq[g] += x * x;
                        }
                    }
                }
            })
            math_done_signal(1u);  // read-only: values consumed before signal
            cb_pop_front(input_cb, tiles_in_block);
        }
    }

    // ===================== Emit local stats stick =====================
    constexpr uint32_t stats_dest_cb = (is_local != 0) ? stats_gathered_cb : stats_local_cb;
    {
        cb_reserve_back(stats_dest_cb, 1);
        const uint32_t stick_addr = get_write_tile_address(stats_dest_cb, 0);
        uint32_t stick_token = 1u;
        MATH({
            volatile tt_l1_ptr float* stick = reinterpret_cast<volatile tt_l1_ptr float*>(stick_addr);
            for (uint32_t g = 0; g < num_groups; g++) {
                stick[g * 4u + 0u] = sum[g];
                stick[g * 4u + 1u] = sumsq[g];
                stick[g * 4u + 2u] = count_local;
                stick[g * 4u + 3u] = 0.f;
            }
            // read-back fence: force the last store to drain before we signal.
            stick_token = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(&stick[(num_groups - 1u) * 4u + 3u]);
        })
        math_done_signal(stick_token);
        cb_push_back(stats_dest_cb, 1);
    }

    // ===================== Wait gathered + eps → μ, inv_std =====================
    cb_wait_front(stats_gathered_cb, ring_size);
    cb_wait_front(epsilon_cb, 1);
    const uint32_t eps_addr = ckernel::get_tile_address(epsilon_cb, 0);
    const uint32_t gathered_addr = ckernel::get_tile_address(stats_gathered_cb, 0);

    float mean[num_groups];
    float inv_std[num_groups];
    MATH({
        F32Bits eps_u;
        eps_u.u = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(eps_addr);
        const float eps = eps_u.f;
        volatile tt_l1_ptr float* gathered = reinterpret_cast<volatile tt_l1_ptr float*>(gathered_addr);
        constexpr uint32_t floats_per_stick = stick_bytes / sizeof(float);
        for (uint32_t g = 0; g < num_groups; g++) {
            float S = 0.f;
            float Q = 0.f;
            float N = 0.f;
            for (uint32_t d = 0; d < ring_size; d++) {
                volatile tt_l1_ptr float* st = gathered + d * floats_per_stick + g * 4u;
                S += st[0];
                Q += st[1];
                N += st[2];
            }
            const float mu = S / N;
            mean[g] = mu;
            inv_std[g] = soft_rsqrt(Q / N - mu * mu + eps);
        }
    })
    math_done_signal(1u);  // read-only
    cb_pop_front(stats_gathered_cb, ring_size);
    // epsilon_cb stays resident (produced once by the writer); never popped.

    // ===================== Side inputs (γ/β) — resident for POST =====================
    volatile tt_l1_ptr uint16_t* weight_u16 = nullptr;
    volatile tt_l1_ptr float* weight_f32 = nullptr;
    volatile tt_l1_ptr uint16_t* bias_u16 = nullptr;
    volatile tt_l1_ptr float* bias_f32 = nullptr;
    if constexpr (has_weight) {
        cb_wait_front(weight_cb, weight_is_tile ? num_tile_cols : 1u);
        const uint32_t w_addr = ckernel::get_tile_address(weight_cb, 0);
        if constexpr (weight_is_fp32) {
            weight_f32 = reinterpret_cast<volatile tt_l1_ptr float*>(w_addr);
        } else {
            weight_u16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(w_addr);
        }
    }
    if constexpr (has_bias) {
        cb_wait_front(bias_cb, bias_is_tile ? num_tile_cols : 1u);
        const uint32_t b_addr = ckernel::get_tile_address(bias_cb, 0);
        if constexpr (bias_is_fp32) {
            bias_f32 = reinterpret_cast<volatile tt_l1_ptr float*>(b_addr);
        } else {
            bias_u16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(b_addr);
        }
    }

    // γ/β readers (called inside MATH({...}) only). Plain [1,1,1,C] contract:
    // TILE layout stores channel `ch` at row 0 of tile ch/32, face-split at c=16.
    auto load_gamma = [&](uint32_t ch) -> float {
        if constexpr (!has_weight) {
            return 1.f;
        }
        if constexpr (weight_is_tile) {
            const uint32_t t = ch / TILE_W;
            const uint32_t c = ch % TILE_W;
            if constexpr (weight_is_fp32) {
                volatile tt_l1_ptr float* tile = weight_f32 + t * TILE_HW;
                if (c < 16u) {
                    return tile[c];
                }
                return tile[FACE_HW + (c - 16u)];
            } else {
                volatile tt_l1_ptr uint16_t* tile = weight_u16 + t * TILE_HW;
                if (c < 16u) {
                    return bf16_to_f32(tile[c]);
                }
                return bf16_to_f32(tile[FACE_HW + (c - 16u)]);
            }
        } else if constexpr (weight_is_fp32) {
            return weight_f32[ch];
        } else {
            return bf16_to_f32(weight_u16[ch]);
        }
    };
    auto load_beta = [&](uint32_t ch) -> float {
        if constexpr (!has_bias) {
            return 0.f;
        }
        if constexpr (bias_is_tile) {
            const uint32_t t = ch / TILE_W;
            const uint32_t c = ch % TILE_W;
            if constexpr (bias_is_fp32) {
                volatile tt_l1_ptr float* tile = bias_f32 + t * TILE_HW;
                if (c < 16u) {
                    return tile[c];
                }
                return tile[FACE_HW + (c - 16u)];
            } else {
                volatile tt_l1_ptr uint16_t* tile = bias_u16 + t * TILE_HW;
                if (c < 16u) {
                    return bf16_to_f32(tile[c]);
                }
                return bf16_to_f32(tile[FACE_HW + (c - 16u)]);
            }
        } else if constexpr (bias_is_fp32) {
            return bias_f32[ch];
        } else {
            return bf16_to_f32(bias_u16[ch]);
        }
    };

    // ===================== POST =====================
    // Normalize each input tile (reader re-streams the input) into the output CB.
    for (uint32_t tile_row = 0; tile_row < num_tile_rows; tile_row++) {
        for (uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
            const uint32_t tiles_in_block =
                ((num_tile_cols - col_tile) >= block_size) ? block_size : (num_tile_cols - col_tile);
            cb_wait_front(input_cb, tiles_in_block);
            cb_reserve_back(output_cb, block_size);

            // UNGATED broadcasts: MATH blocks until UNPACK (input ready) and PACK
            // (output reserved) reach the matching mailbox writes.
            const uint32_t in_addr = ckernel::get_tile_address(input_cb, 0);
            const uint32_t out_addr = get_write_tile_address(output_cb, 0);

            uint32_t out_token = 1u;
            MATH({
                volatile tt_l1_ptr uint16_t* in_tiles = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(in_addr);
                volatile tt_l1_ptr uint16_t* out_tiles = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(out_addr);
                for (uint32_t i = 0; i < tiles_in_block; i++) {
                    const uint32_t c0 = (col_tile + i) * TILE_W;
                    volatile tt_l1_ptr uint16_t* in_t = in_tiles + i * TILE_HW;
                    volatile tt_l1_ptr uint16_t* out_t = out_tiles + i * TILE_HW;
                    for (uint32_t r = 0; r < 32u; r++) {
                        for (uint32_t c = 0; c < 32u; c++) {
                            const uint32_t off = tile_face_offset(r, c);
                            const uint32_t ch = c0 + c;
                            const uint32_t g = ch / channels_per_group;
                            float x = bf16_to_f32(in_t[off]);
                            x = (x - mean[g]) * inv_std[g];
                            x = x * load_gamma(ch) + load_beta(ch);
                            out_t[off] = f32_to_bf16(x);
                        }
                    }
                }
                // Zero-fill unused block slots so the writer can pop block_size safely.
                for (uint32_t i = tiles_in_block; i < block_size; i++) {
                    volatile tt_l1_ptr uint16_t* out_t = out_tiles + i * TILE_HW;
                    for (uint32_t e = 0; e < TILE_HW; e++) {
                        out_t[e] = 0;
                    }
                }
                // read-back fence: last element of the last block slot is always
                // a store above (normalized or zero-filled) → drains before signal.
                out_token = out_tiles[(block_size - 1u) * TILE_HW + (TILE_HW - 1u)];
            })
            math_done_signal(out_token);
            cb_pop_front(input_cb, tiles_in_block);
            cb_push_back(output_cb, block_size);
        }
    }

    if constexpr (has_weight) {
        cb_pop_front(weight_cb, weight_is_tile ? num_tile_cols : 1u);
    }
    if constexpr (has_bias) {
        cb_pop_front(bias_cb, bias_is_tile ? num_tile_cols : 1u);
    }
    (void)C;
}

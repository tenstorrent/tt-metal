// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// A fused K = 512 row of two or more chunks on Blackhole, split across threads: every SFPU instruction of a chunk on
// PACK, the copy and the face transposes on MATH, two chunks in flight (ckernel_sfpu_topk_xl.h, "Split K = 512 fused
// chunk"). Chunk c runs its stage j at step 9c + 2j, so consecutive steps belong to different chunks:
//
//   stage    MATH                          PACK
//   0        copy into the chunk's tile    stamp, sort up to the first transpose
//   1..6     transpose the chunk's tile    sort pass after that transpose (6: last pass, then merge into tile 0)
//   7, 8     transpose tile 0              rebuild build pass, rebuild column pass
//
// Chunk 0 is sorted in place in tile 0 and stops after stage 6; chunk c > 0 goes to tile 1 (odd c) or 2 (even c).
// MATH posts F2S after each stage's FPU part; PACK takes one before each SFPU part and posts S2F after it. MATH starts
// a stage once every SFPU part at least two steps back is done. After the row PACK splits the indices out and marks
// the -inf ones, then posts one more S2F, after which MATH transposes the index tile and commits the section.

#include <cstdint>

namespace topk_large_indices_split {

constexpr std::uint8_t F2S = ckernel::semaphore::FPU_SFPU;
constexpr std::uint8_t S2F = ckernel::semaphore::UNPACK_MATH_DONE;
constexpr std::uint32_t STEPS_PER_CHUNK = 9;
constexpr std::uint32_t STAGES_FIRST_CHUNK = 7;
constexpr std::uint32_t STAGES_MERGING_CHUNK = 9;

inline bool decode_step(
    const std::uint32_t c1,
    const std::uint32_t r,
    const std::uint32_t num_chunks,
    std::uint32_t& chunk,
    std::uint32_t& stage) {
    if ((r & 1) == 0) {
        chunk = c1;
        stage = r >> 1;
    } else {
        if (c1 == 0) {
            return false;
        }
        chunk = c1 - 1;
        stage = (r + STEPS_PER_CHUNK) >> 1;
    }
    return chunk < num_chunks && stage < (chunk == 0 ? STAGES_FIRST_CHUNK : STAGES_MERGING_CHUNK);
}

constexpr std::uint32_t chunk_tile(const std::uint32_t chunk) { return chunk == 0 ? 0 : 2 - (chunk & 1); }

#ifdef TRISC_MATH

inline void take_pack_token() {
    ckernel::t6_semaphore_wait_on_zero<ckernel::p_stall::STALL_SYNC>(S2F);
    ckernel::t6_semaphore_get(S2F);
}

// Once per kernel, before PACK issues any SFPU instruction: seeds the tokens, then hands PACK a MATH_PACK token as
// the start signal.
inline void start() {
    ckernel::t6_semaphore_init(F2S, 0, 15);
    ckernel::t6_semaphore_init(S2F, 0, 15);
    ckernel::t6_semaphore_post<ckernel::p_stall::MATH | ckernel::p_stall::WAIT_SFPU>(ckernel::semaphore::MATH_PACK);
}

inline void release_src() { TTI_SETRWC(ckernel::p_setrwc::CLR_AB, 0, 0, 0, 0, ckernel::p_setrwc::SET_ABD); }

// copy_chunk(chunk, tile) issues the chunk's copy. The SrcA/SrcB releases keep the single-thread order (after the copy,
// halfway through the chunk, before the next copy), and every release and copy runs with the transpose CFG block
// closed, as it does there.
template <typename CopyChunk>
inline void math_row(const std::uint32_t num_chunks, CopyChunk&& copy_chunk) {
    std::uint32_t posted = 0, taken = 0, half_release_step = 0;
    bool prev_step_busy = false, cfg_open = false, half_release_due = false;

    for (std::uint32_t c1 = 0; c1 <= num_chunks; c1++) {
        for (std::uint32_t r = 0; r < STEPS_PER_CHUNK; r++) {
            std::uint32_t chunk, stage;
            if (!decode_step(c1, r, num_chunks, chunk, stage)) {
                prev_step_busy = false;
                continue;
            }
            const std::uint32_t step = c1 * STEPS_PER_CHUNK + r;
            const std::uint32_t need = posted - (prev_step_busy ? 1 : 0);
            for (; taken < need; taken++) {
                take_pack_token();
            }

            if (stage == 0) {
                if (cfg_open) {
                    ckernel::sfpu::leave_transpose_cfg_block();
                    cfg_open = false;
                }
                if (chunk > 0) {
                    release_src();
                    half_release_due = true;
                    half_release_step = step + 5;
                }
                copy_chunk(chunk, chunk_tile(chunk));
            } else {
                if (half_release_due && step >= half_release_step) {
                    if (cfg_open) {
                        ckernel::sfpu::leave_transpose_cfg_block();
                        cfg_open = false;
                    }
                    release_src();
                    half_release_due = false;
                }
                if (!cfg_open) {
                    ckernel::sfpu::enter_transpose_cfg_block();
                    cfg_open = true;
                }
                ckernel::sfpu::_topk_xl_split_transpose_512_(chunk_tile(stage >= STAGES_FIRST_CHUNK ? 0 : chunk) << 6);
            }

            ckernel::t6_semaphore_post<ckernel::p_stall::MATH>(F2S);
            posted++;
            prev_step_busy = true;
        }
    }

    if (cfg_open) {
        ckernel::sfpu::leave_transpose_cfg_block();
    }
    release_src();
    for (; taken < posted; taken++) {
        take_pack_token();
    }
    // The epilogue's token: PACK has split the indices out and marked the -inf ones.
    take_pack_token();
}

#endif  // TRISC_MATH

#ifdef TRISC_PACK

inline void take_math_token() {
    ckernel::t6_semaphore_wait_on_zero<ckernel::p_stall::STALL_SYNC>(F2S);
    ckernel::t6_semaphore_get(F2S);
}

// Once per kernel: waits for MATH's start signal.
inline void wait_start() {
    _llk_packer_wait_for_math_done_();
    _llk_packer_set_math_semaphore_<ckernel::p_stall::NONE>();
}

inline __attribute__((noinline)) void sfpu_stage(const std::uint32_t chunk, const std::uint32_t stage) {
    using namespace ckernel::sfpu;
    const std::uint32_t tile_offset = chunk_tile(chunk) << 6;
    const bool ascending = chunk > 0;
    switch (stage) {
        case 0:
            _topk_xl_split_stamp_512_(tile_offset, chunk);
            _topk_xl_split_sort_head_512_(tile_offset, ascending);
            break;
        case 1: _topk_xl_split_stride2_512_<4>(tile_offset, ascending); break;
        case 2: _topk_xl_split_columns_512_<0x5050>(tile_offset, ascending); break;
        case 3: _topk_xl_split_stride2_512_<8>(tile_offset, ascending); break;
        case 4: _topk_xl_split_columns_512_<0x5500>(tile_offset, ascending); break;
        case 5: _topk_xl_split_stride2_512_<16>(tile_offset, ascending); break;
        case 6:
            _topk_xl_split_columns_512_<0>(tile_offset, ascending);
            if (chunk > 0) {
                if (chunk & 1) {
                    _topk_xl_split_merge_512_<64>(0);
                } else {
                    _topk_xl_split_merge_512_<128>(0);
                }
            }
            break;
        case 7: _topk_xl_split_rebuild_build_512_(0, false); break;
        default: _topk_xl_split_columns_512_<0>(0, false); break;
    }
}

// Every SFPU part of one row, survivor left fused in tile 0; then the row-major global index split of tile 0 into
// tiles 0 (values) and 1 (indices) and the -inf marking, after which MATH may transpose the index tile.
template <typename MarkNeginf>
inline void pack_row(const std::uint32_t num_chunks, MarkNeginf&& mark_neginf) {
    ckernel::sfpu::_topk_xl_split_sfpu_init_();
    TTI_STALLWAIT(ckernel::p_stall::STALL_SFPU, ckernel::p_stall::MATH);
    for (std::uint32_t c1 = 0; c1 <= num_chunks; c1++) {
        for (std::uint32_t r = 0; r < STEPS_PER_CHUNK; r++) {
            std::uint32_t chunk, stage;
            if (!decode_step(c1, r, num_chunks, chunk, stage)) {
                continue;
            }
            take_math_token();
            sfpu_stage(chunk, stage);
            ckernel::t6_semaphore_post<ckernel::p_stall::WAIT_SFPU>(S2F);
        }
    }

    ckernel::sfpu::_topk_xl_separate_indices_row_major_global_init_();
    ckernel::sfpu::_topk_xl_split_begin_(0);
    ckernel::sfpu::_topk_xl_separate_indices_row_major_global_<512>();
    TTI_SETRWC(ckernel::p_setrwc::CLR_NONE, 0, 0, 0, 0, ckernel::p_setrwc::SET_D);
    mark_neginf();
    TTI_SETRWC(ckernel::p_setrwc::CLR_NONE, 0, 0, 0, 0, ckernel::p_setrwc::SET_D);
    ckernel::t6_semaphore_post<ckernel::p_stall::WAIT_SFPU>(S2F);
}

#endif  // TRISC_PACK

}  // namespace topk_large_indices_split

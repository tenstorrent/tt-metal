// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// A row of two or more K = 512 or K = 2048 chunks on Blackhole, split across threads: every SFPU instruction of a
// chunk on PACK, the copy and the face transposes on MATH, two chunks in flight (ckernel_sfpu_topk_xl.h, "Split K = 512
// and K = 2048 fused chunks"). Chunk c runs its stage j at step 9c + 2j, so consecutive steps belong to different
// chunks:
//
//   stage    MATH                          PACK
//   0        copy into the chunk's tile    stamp, sort up to the first transpose
//   1..6     transpose the chunk's tile    sort pass after that transpose (6: last pass, then the merge)
//   7, 8     transpose the survivor        rebuild build pass, rebuild column pass
//
// Fused body (K = 512 and K = 2048 rows of up to 32 chunks): a chunk is one sequence of fused keys, one tile at K = 512
// and two at K = 2048; chunk 0 is sorted in place in sequence 0, chunk c > 0 goes to sequence 1 (odd c) or 2 (even c).
// Classic body (K = 512 rows of any width): stage 6 also splits the chunk's indices into the next tile, and the merge
// and rebuild are the unfused ones; the survivor's values and indices sit in tiles 0 and 1, chunk c > 0 in tiles 2 and
// 3 (odd c) or 4 and 5 (even c). MATH posts F2S after each stage's FPU part; PACK takes one before each SFPU part and
// posts S2F after it. MATH starts a stage once every SFPU part at least two steps back is done. After the row PACK runs
// the epilogue (the fused body's index split, the -inf marking) and posts one more S2F, after which MATH transposes the
// index tiles and commits the section.

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

// Tiles of one chunk: a fused sequence, or a Classic chunk's value and index tiles.
template <std::uint32_t K, bool classic>
constexpr std::uint32_t chunk_tiles = classic ? 2 : (K == 2048 ? 2 : 1);

template <std::uint32_t K, bool classic>
constexpr bool supported_split = (K == 512 || (K == 2048 && !classic));

template <std::uint32_t K, bool classic>
constexpr std::uint32_t chunk_tile(const std::uint32_t chunk) {
    return chunk_tiles<K, classic> * (chunk == 0 ? 0 : 2 - (chunk & 1));
}

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
template <std::uint32_t K, bool classic, typename CopyChunk>
inline void math_row(const std::uint32_t num_chunks, CopyChunk&& copy_chunk) {
    static_assert(supported_split<K, classic>, "the split takes fused K = 512 and 2048 rows and Classic K = 512 rows");
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
                copy_chunk(chunk, chunk_tile<K, classic>(chunk));
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
                const std::uint32_t tile = chunk_tile<K, classic>(stage >= STAGES_FIRST_CHUNK ? 0 : chunk);
                if constexpr (classic) {
                    if (stage >= STAGES_FIRST_CHUNK) {
                        ckernel::sfpu::_topk_xl_split_transpose_unfused_512_(0);
                    } else {
                        ckernel::sfpu::_topk_xl_split_transpose_<K>(tile << 6);
                    }
                } else {
                    ckernel::sfpu::_topk_xl_split_transpose_<K>(tile << 6);
                }
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
    // The epilogue's token.
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

template <std::uint32_t K, bool classic>
inline __attribute__((noinline)) void sfpu_stage(const std::uint32_t chunk, const std::uint32_t stage) {
    static_assert(supported_split<K, classic>, "the split takes fused K = 512 and 2048 rows and Classic K = 512 rows");
    using namespace ckernel::sfpu;
    constexpr int chunk_rows = 64 * chunk_tiles<K, classic>;
    const std::uint32_t tile_offset = chunk_tile<K, classic>(chunk) << 6;
    const bool ascending = chunk > 0;
    if constexpr (classic) {
        // The other chunk's unfused segments leave index tracking on.
        if (stage < STAGES_FIRST_CHUNK) {
            _init_sfpu_config_reg();
        }
    }
    switch (stage) {
        case 0:
            if constexpr (classic) {
                addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 4}}.set(ADDR_MOD_3);
            }
            _topk_xl_split_stamp_<K>(tile_offset, classic ? 0 : chunk);
            if constexpr (K == 2048) {
                _topk_xl_split_sort_head_2048_(tile_offset, ascending);
            } else {
                _topk_xl_split_sort_head_512_(tile_offset, ascending);
            }
            break;
        case 1: _topk_xl_split_stride2_<K, 4>(tile_offset, ascending); break;
        case 2: _topk_xl_split_columns_<K, 0x5050>(tile_offset, ascending); break;
        case 3: _topk_xl_split_stride2_<K, 8>(tile_offset, ascending); break;
        case 4: _topk_xl_split_columns_<K, 0x5500>(tile_offset, ascending); break;
        case 5: _topk_xl_split_stride2_<K, 16>(tile_offset, ascending); break;
        case 6:
            _topk_xl_split_columns_<K, 0>(tile_offset, ascending);
            if constexpr (classic) {
                _topk_xl_split_separate_512_(tile_offset);
                if (chunk > 0) {
                    if (chunk & 1) {
                        _topk_xl_split_merge_unfused_512_<chunk_rows>(0);
                    } else {
                        _topk_xl_split_merge_unfused_512_<2 * chunk_rows>(0);
                    }
                }
            } else if (chunk > 0) {
                if (chunk & 1) {
                    _topk_xl_split_merge_<K, chunk_rows>(0);
                } else {
                    _topk_xl_split_merge_<K, 2 * chunk_rows>(0);
                }
            }
            break;
        case 7:
            if constexpr (classic) {
                _topk_xl_split_rebuild_build_unfused_512_(0, false);
            } else {
                _topk_xl_split_rebuild_build_<K>(0, false);
            }
            break;
        default:
            if constexpr (classic) {
                _topk_xl_split_rebuild_columns_unfused_512_(0, false);
            } else {
                _topk_xl_split_columns_<K, 0>(0, false);
            }
            break;
    }
}

// Every SFPU part of one row, survivor left in tile 0, then epilogue(). PACK takes the SFPU over (its state and
// row_init()) after its first token, once MATH's last SFPU instruction has run.
template <std::uint32_t K, bool classic, typename RowInit, typename Epilogue>
inline void pack_row(const std::uint32_t num_chunks, RowInit&& row_init, Epilogue&& epilogue) {
    bool first = true;
    for (std::uint32_t c1 = 0; c1 <= num_chunks; c1++) {
        for (std::uint32_t r = 0; r < STEPS_PER_CHUNK; r++) {
            std::uint32_t chunk, stage;
            if (!decode_step(c1, r, num_chunks, chunk, stage)) {
                continue;
            }
            take_math_token();
            if (first) {
                ckernel::sfpu::_topk_xl_split_sfpu_init_<K>();
                row_init();
                TTI_STALLWAIT(ckernel::p_stall::STALL_SFPU, ckernel::p_stall::MATH);
                first = false;
            }
            sfpu_stage<K, classic>(chunk, stage);
            ckernel::t6_semaphore_post<ckernel::p_stall::WAIT_SFPU>(S2F);
        }
    }

    epilogue();
    ckernel::t6_semaphore_post<ckernel::p_stall::WAIT_SFPU>(S2F);
}

#endif  // TRISC_PACK

}  // namespace topk_large_indices_split

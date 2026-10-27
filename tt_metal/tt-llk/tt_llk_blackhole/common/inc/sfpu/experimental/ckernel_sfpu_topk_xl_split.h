// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Generic topk_xl MATH/PACK co-scheduling: Tensix semaphores, TTI instructions and ADDR_MOD programming. The
// caller's op glue decides from how many chunks a row runs here.
//
// A row of K = 512, 1024 or 2048 chunks: copies and face transposes on MATH, SFPU work on PACK. Chunk c runs
// stage j at step 9c + 2j, so MATH transposes one chunk while PACK sorts the other; a K 1024 or K 2048 row of
// two or more chunks opens with chunks 0 and 1 side by side (see paired_head), so no chunk starts alone.

// The thread guards accept both spellings: the metal JIT defines TRISC_MATH / TRISC_PACK, the tt-llk test
// harness defines LLK_TRISC_MATH / LLK_TRISC_PACK.

#include <cstdint>

// Self-contained: these supply ckernel::semaphore, t6_semaphore_*, the TTI_* macros and the topk_xl SFPU
// routines this schedules. Without them the header compiles only when the includer happens to pull them first.
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/experimental/ckernel_sfpu_topk_xl.h"

namespace topk_xl_split
{

constexpr std::uint8_t F2S                   = ckernel::semaphore::FPU_SFPU;
constexpr std::uint8_t S2F                   = ckernel::semaphore::UNPACK_MATH_DONE;
// The schedule runs ahead of PACK, so the shared semaphores need more than their default depth of 1.
constexpr std::uint8_t TOKEN_DEPTH             = 15;
constexpr std::uint8_t DEFAULT_SEMAPHORE_DEPTH = 1;
constexpr std::uint32_t STEPS_PER_CHUNK      = 9;
constexpr std::uint32_t STAGES_FIRST_CHUNK   = 7;
constexpr std::uint32_t STAGES_MERGING_CHUNK = 9;

inline bool decode_step(const std::uint32_t c1, const std::uint32_t r, const std::uint32_t num_chunks, std::uint32_t& chunk, std::uint32_t& stage)
{
    if ((r & 1) == 0)
    {
        chunk = c1;
        stage = r >> 1;
    }
    else
    {
        if (c1 == 0)
        {
            return false;
        }
        chunk = c1 - 1;
        stage = (r + STEPS_PER_CHUNK) >> 1;
    }
    return chunk < num_chunks && stage < (chunk == 0 ? STAGES_FIRST_CHUNK : STAGES_MERGING_CHUNK);
}

template <std::uint32_t K>
constexpr bool supported_split = K == 512 || K == 1024 || K == 2048;

// Chunks alternate DEST banks (bank 1 starts at tile 4): the chunk MATH transposes and the chunk PACK sorts sit in
// different tiles, and the next row's first chunk can be copied into bank 1 while this row finishes.
template <std::uint32_t K>
constexpr std::uint32_t slot_tile(const std::uint32_t from_last)
{
    return ((from_last & 1) ? 4 : 0) + ((from_last >> 1) & 1) * (K == 2048 ? 2 : 1);
}

template <std::uint32_t K>
constexpr std::uint32_t chunk_tile(const std::uint32_t chunk, const std::uint32_t num_chunks)
{
    // K 512 rows of two chunks keep the one-thread layout (survivor in tile 0, chunk 1 in tile 1): the index split
    // writes half of tile 1 and the rest of it then holds what the one-thread kernel leaves there.
    if constexpr (K == 512)
    {
        if (num_chunks == 2)
        {
            return chunk;
        }
    }
    return slot_tile<K>(num_chunks - 1 - chunk);
}

// Where chunk `chunk`'s stages 7 and 8 find the survivor.
template <std::uint32_t K>
constexpr std::uint32_t survivor_tile(const std::uint32_t chunk, const std::uint32_t num_chunks)
{
    return (K == 512 && num_chunks == 2) ? 0 : chunk_tile<K>(chunk, num_chunks);
}

// Rows from a fused chunk's tiles to the survivor of the chunk before it, modulo the 1024-row Dst address; slot_parity
// is the chunk's from_last & 3.
template <std::uint32_t K>
constexpr int survivor_distance(const std::uint32_t slot_parity)
{
    return static_cast<int>(((slot_tile<K>(slot_parity + 1) - slot_tile<K>(slot_parity)) & 15) << 6);
}

// Whether the next row's first fused chunk can be copied while this row finishes: it lands in bank 1, which the
// survivor and index tiles never use.
template <std::uint32_t K>
constexpr bool can_prefetch(const std::uint32_t num_chunks)
{
    return (num_chunks & 1) == 0 && !(K == 512 && num_chunks == 2);
}

// K 1024 and 2048 rows of two or more chunks run chunks 0 and 1 side by side up to stage 6 (HEAD_STEPS); slot 2 on
// follows, with chunk 1's stages 7 and 8 two steps early. K 512 rows keep decode_step's order. A K 1024 row runs here
// from three chunks (compute.cpp), so only K 2048 takes the paired head with exactly two.
constexpr std::uint32_t HEAD_STEPS = 14;

template <std::uint32_t K>
constexpr bool paired_head(const std::uint32_t num_chunks)
{
    return K != 512 && num_chunks >= 2;
}

#if defined(TRISC_MATH) || defined(LLK_TRISC_MATH)

inline void take_pack_token()
{
    ckernel::t6_semaphore_wait_on_zero<ckernel::p_stall::STALL_SYNC>(S2F);
    ckernel::t6_semaphore_get(S2F);
}

// Raise the two tokens to the depth the schedule needs. Runs before PACK issues any SFPU instruction.
inline void init_tokens()
{
    ckernel::t6_semaphore_init(F2S, 0 /*min_value*/, TOKEN_DEPTH /*max_value*/);
    ckernel::t6_semaphore_init(S2F, 0 /*min_value*/, TOKEN_DEPTH /*max_value*/);
}

// PACK's start signal. Separate from init_tokens so a caller can run work in between -- a dest scrub has to
// finish before PACK is let go.
inline void start_pack()
{
    ckernel::t6_semaphore_post<ckernel::p_stall::MATH | ckernel::p_stall::WAIT_SFPU>(ckernel::semaphore::MATH_PACK);
}

// Once per kernel, for callers with nothing to do between the two.
inline void start()
{
    init_tokens();
    start_pack();
}

// Once per kernel, after the last row: F2S and S2F back to value 0 and max 1, which every other kernel on
// this core assumes on entry.
inline void finish()
{
    ckernel::t6_semaphore_init(F2S, 0 /*min_value*/, DEFAULT_SEMAPHORE_DEPTH /*max_value*/);
    ckernel::t6_semaphore_init(S2F, 0 /*min_value*/, DEFAULT_SEMAPHORE_DEPTH /*max_value*/);
}

// A token that carries no step: MATH's row set-up is done.
inline void post_token()
{
    ckernel::t6_semaphore_post<ckernel::p_stall::MATH | ckernel::p_stall::WAIT_SFPU>(F2S);
}

inline void release_src()
{
    TTI_SETRWC(ckernel::p_setrwc::CLR_AB, 0, 0, 0, 0, ckernel::p_setrwc::SET_ABD);
}

// The SrcA/SrcB releases keep the single-thread order (after the copy, halfway through the chunk, before the next
// copy), and every release and copy runs with the transpose CFG block closed, as in the single-thread kernel.
//
// first_prefetched says the row's first chunk was already copied while the previous row finished, and
// prefetch_next asks for the next row's first chunk at the end of this one; the two must agree between
// consecutive calls. With first_prefetched the row's opening post is skipped, so PACK first sees the previous
// row's prefetch post, which stalls on MATH alone. A caller that runs MATH-side SFPU work between prefetched
// rows must therefore post its own token after that work and have PACK take one, or the two threads race for
// the SFPU. math_row ends by taking PACK's last token, so PACK must post one past its epilogue.
template <std::uint32_t K, typename CopyChunk>
inline void math_row(const std::uint32_t num_chunks, CopyChunk&& copy_chunk, const bool first_prefetched = false, const bool prefetch_next = false)
{
    static_assert(supported_split<K>, "the split takes fused K = 512, 1024 and 2048 rows");
    // A zero-chunk row posts no token, so the take at the end would hang both threads.
    LLK_ASSERT(num_chunks >= 1, "math_row needs at least one chunk");
    std::uint32_t posted = 0, taken = 0, half_release_step = 0;
    bool prev_step_busy = false, cfg_open = false, half_release_due = false;

    const bool paired = paired_head<K>(num_chunks);
    // Slot 2 of a paired row starts at step HEAD_STEPS: chunk 1's stages 5 and 6 are done, so its stages 7 and 8 come
    // two steps early.
    const std::uint32_t slot_step0 = paired ? HEAD_STEPS - 2 * STEPS_PER_CHUNK : 0;

    for (std::uint32_t c1 = 0; c1 <= num_chunks; c1++)
    {
        if (paired && c1 == 1)
        {
            continue;
        }
        const bool head = paired && c1 == 0;
        for (std::uint32_t r = 0, steps = head ? HEAD_STEPS : STEPS_PER_CHUNK; r < steps; r++)
        {
            std::uint32_t chunk, stage;
            if (head)
            {
                chunk = r & 1;
                stage = r >> 1;
            }
            else
            {
                if (!decode_step(c1, r, num_chunks, chunk, stage))
                {
                    prev_step_busy = false;
                    continue;
                }
                if (paired && c1 == 2 && (r & 1))
                {
                    stage += 2;
                    if (stage >= STAGES_MERGING_CHUNK)
                    {
                        prev_step_busy = false;
                        continue;
                    }
                }
            }
            const std::uint32_t step = head ? r : c1 * STEPS_PER_CHUNK + r + slot_step0;
            // Every SFPU part two or more steps back must be done before this copy touches its tiles.
            // Equivalent to posted - (prev_step_busy ? 1 : 0); it cannot wrap, because prev_step_busy is only
            // set right after posted++
            // below. The two cannot co-occur today, but a wrap here is a hang, not a wrong answer.
            const std::uint32_t need = ((paired && step == 1) || !prev_step_busy) ? posted : posted - 1;
            for (; taken < need; taken++)
            {
                take_pack_token();
            }

            if (first_prefetched && step == 0)
            {
                posted++;
                prev_step_busy = true;
                continue;
            }
            if (stage == 0)
            {
                if (cfg_open)
                {
                    ckernel::sfpu::leave_transpose_cfg_block();
                    cfg_open = false;
                }
                if (chunk > 0)
                {
                    if (paired && step == 1)
                    {
                        // Copy 0's valid flags may not be set yet, and the release would clear them first.
                        TTI_STALLWAIT(ckernel::p_stall::STALL_MATH, ckernel::p_stall::SRCA_VLD | ckernel::p_stall::SRCB_VLD);
                    }
                    release_src();
                    half_release_due  = true;
                    half_release_step = step + 5;
                }
                copy_chunk(chunk, chunk_tile<K>(chunk, num_chunks));
            }
            else
            {
                if (half_release_due && step >= half_release_step)
                {
                    if (cfg_open)
                    {
                        ckernel::sfpu::leave_transpose_cfg_block();
                        cfg_open = false;
                    }
                    release_src();
                    half_release_due = false;
                }
                if (!cfg_open)
                {
                    ckernel::sfpu::enter_transpose_cfg_block();
                    cfg_open = true;
                }
                const std::uint32_t tile = stage >= STAGES_FIRST_CHUNK ? survivor_tile<K>(chunk, num_chunks) : chunk_tile<K>(chunk, num_chunks);
                ckernel::sfpu::_topk_xl_split_transpose_<K>(tile << 6);
            }

            // PACK may take the SFPU over only after MATH's SFPU init, so the row's first token waits for it.
            if (posted == 0)
            {
                ckernel::t6_semaphore_post<ckernel::p_stall::MATH | ckernel::p_stall::WAIT_SFPU>(F2S);
            }
            else
            {
                ckernel::t6_semaphore_post<ckernel::p_stall::MATH>(F2S);
            }
            posted++;
            prev_step_busy = true;
        }
    }

    if (cfg_open)
    {
        ckernel::sfpu::leave_transpose_cfg_block();
    }
    release_src();
    if (prefetch_next)
    {
        // Stage 6 of the last chunk has read the other chunk's tiles; stages 7 and 8 stay in the survivor's bank.
        for (; taken + 2 < posted; taken++)
        {
            take_pack_token();
        }
        copy_chunk(0, chunk_tile<K>(0, num_chunks));
        ckernel::t6_semaphore_post<ckernel::p_stall::MATH>(F2S);
    }
    for (; taken < posted; taken++)
    {
        take_pack_token();
    }
    take_pack_token();
}

#endif // TRISC_MATH || LLK_TRISC_MATH

#if defined(TRISC_PACK) || defined(LLK_TRISC_PACK)

// _llk_packer_wait_for_math_done_ / _llk_packer_set_math_semaphore_ live here, and the includes at the top of
// this file do not reach it.
#include "llk_pack_common.h"

inline void take_math_token()
{
    ckernel::t6_semaphore_wait_on_zero<ckernel::p_stall::STALL_SYNC>(F2S);
    ckernel::t6_semaphore_get(F2S);
}

inline void wait_start()
{
    _llk_packer_wait_for_math_done_();
    _llk_packer_set_math_semaphore_<ckernel::p_stall::NONE>();
}

// The row-major global index split of sequence 0 into values (sequence 0) and indices (the tiles after it).
// seg_base 0 means the row starts at chunk 0. stall_after holds PACK until the SFPU drains; callers that follow
// this with more SFPU work in the same epilogue do not need it.
template <std::uint32_t K, bool stall_after = false>
inline void split_indices(const std::uint32_t seg_base)
{
    ckernel::sfpu::_topk_xl_separate_indices_row_major_global_init_();
    ckernel::sfpu::_topk_xl_split_begin_(0);
    if (seg_base != 0)
    {
        ckernel::sfpu::_topk_xl_separate_indices_row_major_global_base_<K>(seg_base);
    }
    else
    {
        ckernel::sfpu::_topk_xl_separate_indices_row_major_global_<K>();
    }
    TTI_SETRWC(ckernel::p_setrwc::CLR_NONE, 0, 0, 0, 0, ckernel::p_setrwc::SET_D);
    if constexpr (stall_after)
    {
        TTI_STALLWAIT(ckernel::p_stall::STALL_PACK, ckernel::p_stall::WAIT_SFPU);
    }
}

// Releases Dst to MATH, clearing only bank 0: bank 1 may already hold the next row's first chunk.
template <bool fp32>
inline void release_dest_bank0()
{
    TTI_STALLWAIT(ckernel::p_stall::STALL_MATH, ckernel::p_stall::PACK);
    TTI_ZEROACC(ckernel::p_zeroacc::CLR_HALF, fp32, 0, ckernel::ADDR_MOD_1, 0);
    _llk_packer_set_math_semaphore_<ckernel::p_stall::NONE>();
}

// rebuild_dir: the direction of stages 7 and 8 (ascending for the last chunk of a mirrored segment).
template <std::uint32_t K>
inline __attribute__((always_inline)) void sfpu_stage_body(
    const std::uint32_t chunk, const std::uint32_t stage, const std::uint32_t num_chunks, const bool rebuild_dir)
{
    static_assert(supported_split<K>, "the split takes fused K = 512, 1024 and 2048 rows");
    namespace sfpu                  = ckernel::sfpu;
    const std::uint32_t tile_offset = chunk_tile<K>(chunk, num_chunks) << 6;
    const bool ascending            = chunk > 0;
    switch (stage)
    {
        case 0:
            sfpu::_topk_xl_split_stamp_<K>(tile_offset, chunk);
            if constexpr (K == 2048)
            {
                sfpu::_topk_xl_split_sort_head_2048_(tile_offset, ascending);
            }
            else if constexpr (K == 1024)
            {
                sfpu::_topk_xl_split_sort_head_1024_(tile_offset, ascending);
            }
            else
            {
                sfpu::_topk_xl_split_sort_head_512_<true /*macros*/>(tile_offset, ascending);
            }
            break;
        case 1:
            sfpu::_topk_xl_split_stride2_<K, 4 /*sort_len*/, true /*macros*/>(tile_offset, ascending);
            break;
        case 2:
            sfpu::_topk_xl_split_columns_<K, 0x5050 /*lane_mask*/>(tile_offset, ascending);
            break;
        case 3:
            sfpu::_topk_xl_split_stride2_<K, 8 /*sort_len*/, true /*macros*/>(tile_offset, ascending);
            break;
        case 4:
            sfpu::_topk_xl_split_columns_<K, 0x5500 /*lane_mask*/>(tile_offset, ascending);
            break;
        case 5:
            sfpu::_topk_xl_split_stride2_<K, 16 /*sort_len*/, true /*macros*/>(tile_offset, ascending);
            break;
        case 6:
            sfpu::_topk_xl_split_columns_<K, 0 /*lane_mask*/>(tile_offset, ascending);
            if (chunk > 0)
            {
                if constexpr (K == 512)
                {
                    if (num_chunks == 2)
                    {
                        sfpu::_topk_xl_split_merge_<K, 64 /*distance*/, true /*macros*/>(0);
                        break;
                    }
                }
                switch ((num_chunks - 1 - chunk) & 3)
                {
                    case 1:
                        sfpu::_topk_xl_split_merge_<K, survivor_distance<K>(1), true /*macros*/>(tile_offset);
                        break;
                    case 3:
                        sfpu::_topk_xl_split_merge_<K, survivor_distance<K>(3), true /*macros*/>(tile_offset);
                        break;
                    default:
                        sfpu::_topk_xl_split_merge_<K, survivor_distance<K>(0), true /*macros*/>(tile_offset);
                        break;
                }
            }
            break;
        case 7:
            sfpu::_topk_xl_split_rebuild_build_<K>(survivor_tile<K>(chunk, num_chunks) << 6, rebuild_dir);
            break;
        default:
            sfpu::_topk_xl_split_columns_<K, 0 /*lane_mask*/>(survivor_tile<K>(chunk, num_chunks) << 6, rebuild_dir);
            break;
    }
}

// One SFPU part of the row. K 1024 and 2048 take it as a call, so that each stage body exists once whatever the
// schedule (their pack code fills TRISC2); K 512 keeps every step inline in the straight-line slot.
// mirror_last is a parameter, not a template argument: sfpu_stage_body takes the direction as a value, so
// templating on it duplicates every stage body in PACK instruction memory for identical code.
template <std::uint32_t K, std::uint32_t stage>
inline __attribute__((always_inline)) void sfpu_step_inline(const std::uint32_t chunk, const std::uint32_t num_chunks, const bool mirror_last, const bool take)
{
    if (take)
    {
        take_math_token();
    }
    sfpu_stage_body<K>(chunk, stage, num_chunks, mirror_last && chunk + 1 == num_chunks);
    ckernel::t6_semaphore_post<ckernel::p_stall::WAIT_SFPU>(S2F);
}

template <std::uint32_t K, std::uint32_t stage>
inline __attribute__((noinline, noclone)) void sfpu_step_call(
    const std::uint32_t chunk, const std::uint32_t num_chunks, const bool mirror_last, const bool take)
{
    sfpu_step_inline<K, stage>(chunk, num_chunks, mirror_last, take);
}

template <std::uint32_t K, std::uint32_t stage>
inline __attribute__((always_inline)) void sfpu_step(const std::uint32_t chunk, const std::uint32_t num_chunks, const bool mirror_last, const bool take = true)
{
    if constexpr (K == 512)
    {
        sfpu_step_inline<K, stage>(chunk, num_chunks, mirror_last, take);
    }
    else
    {
        sfpu_step_call<K, stage>(chunk, num_chunks, mirror_last, take);
    }
}

// PACK's steps of chunk slot c1 in decode_step's order: chunk c1's stages 0 to 4 between chunk c1 - 1's stages 5 to 8
// (7 and 8 only for a merged chunk). pack_row takes slot 0's first token.
template <std::uint32_t K>
inline __attribute__((noinline)) void pack_slot(const std::uint32_t c1, const std::uint32_t num_chunks, const bool mirror_last)
{
    const bool cur = c1 < num_chunks, prev = c1 > 0, prev_merged = c1 > 1;
    if (cur)
    {
        sfpu_step<K, 0>(c1, num_chunks, mirror_last, c1 > 0);
    }
    if (prev)
    {
        sfpu_step<K, 5>(c1 - 1, num_chunks, mirror_last);
    }
    if (cur)
    {
        sfpu_step<K, 1>(c1, num_chunks, mirror_last);
    }
    if (prev)
    {
        sfpu_step<K, 6>(c1 - 1, num_chunks, mirror_last);
    }
    if (cur)
    {
        sfpu_step<K, 2>(c1, num_chunks, mirror_last);
    }
    if (prev_merged)
    {
        sfpu_step<K, 7>(c1 - 1, num_chunks, mirror_last);
    }
    if (cur)
    {
        sfpu_step<K, 3>(c1, num_chunks, mirror_last);
    }
    if (prev_merged)
    {
        sfpu_step<K, 8>(c1 - 1, num_chunks, mirror_last);
    }
    if (cur)
    {
        sfpu_step<K, 4>(c1, num_chunks, mirror_last);
    }
}

template <std::uint32_t K>
inline __attribute__((noinline)) void pack_head(const std::uint32_t num_chunks, const bool mirror_last)
{
    sfpu_step<K, 0>(0, num_chunks, mirror_last, false /*take*/);
    sfpu_step<K, 0>(1, num_chunks, mirror_last);
    sfpu_step<K, 1>(0, num_chunks, mirror_last);
    sfpu_step<K, 1>(1, num_chunks, mirror_last);
    sfpu_step<K, 2>(0, num_chunks, mirror_last);
    sfpu_step<K, 2>(1, num_chunks, mirror_last);
    sfpu_step<K, 3>(0, num_chunks, mirror_last);
    sfpu_step<K, 3>(1, num_chunks, mirror_last);
    sfpu_step<K, 4>(0, num_chunks, mirror_last);
    sfpu_step<K, 4>(1, num_chunks, mirror_last);
    sfpu_step<K, 5>(0, num_chunks, mirror_last);
    sfpu_step<K, 5>(1, num_chunks, mirror_last);
    sfpu_step<K, 6>(0, num_chunks, mirror_last);
    sfpu_step<K, 6>(1, num_chunks, mirror_last);
    const bool cur = num_chunks > 2;
    if (cur)
    {
        sfpu_step<K, 0>(2, num_chunks, mirror_last);
    }
    sfpu_step<K, 7>(1, num_chunks, mirror_last);
    if (cur)
    {
        sfpu_step<K, 1>(2, num_chunks, mirror_last);
    }
    sfpu_step<K, 8>(1, num_chunks, mirror_last);
    if (cur)
    {
        sfpu_step<K, 2>(2, num_chunks, mirror_last);
        sfpu_step<K, 3>(2, num_chunks, mirror_last);
        sfpu_step<K, 4>(2, num_chunks, mirror_last);
    }
}

// Every SFPU part of one row, survivor left in tile 0 (ascending if mirror_last), then epilogue(). PACK takes the SFPU
// over (its state) after its first token, once MATH's last SFPU instruction has run.
template <std::uint32_t K, typename Epilogue>
inline void pack_row(const std::uint32_t num_chunks, const bool mirror_last, Epilogue&& epilogue)
{
    static_assert(supported_split<K>, "the split takes fused K = 512, 1024 and 2048 rows");
    LLK_ASSERT(num_chunks >= 1, "pack_row needs at least one chunk");
    // Only stages 7 and 8 read the rebuild direction, and a single-chunk row never reaches them, so it would
    // end descending whatever mirror_last says.
    LLK_ASSERT(num_chunks >= 2 || !mirror_last, "a single-chunk row cannot honour mirror_last");
    take_math_token();
    ckernel::sfpu::_topk_xl_split_sfpu_init_<K, true /*macros*/>();
    TTI_STALLWAIT(ckernel::p_stall::STALL_SFPU, ckernel::p_stall::MATH);
    std::uint32_t c1 = 0;
    if (paired_head<K>(num_chunks))
    {
        pack_head<K>(num_chunks, mirror_last);
        c1 = 3;
    }
    for (; c1 <= num_chunks; c1++)
    {
        pack_slot<K>(c1, num_chunks, mirror_last);
    }
    epilogue();
    ckernel::t6_semaphore_post<ckernel::p_stall::WAIT_SFPU>(S2F);
}

// A segmented row's later segment: after MATH's token (its unfused merge and rebuild done, or the previous segment's
// epilogue), move_acc() moves the accumulator out of Dst; MATH copies into tile 0 after the token that follows.
template <typename MoveAcc>
inline void hold_accumulator(MoveAcc&& move_acc)
{
    take_math_token();
    move_acc();
    ckernel::t6_semaphore_post<ckernel::p_stall::PACK>(S2F);
}

#endif // TRISC_PACK || LLK_TRISC_PACK

#if defined(TRISC_MATH) || defined(LLK_TRISC_MATH)

inline void release_accumulator()
{
    ckernel::t6_semaphore_post<ckernel::p_stall::MATH | ckernel::p_stall::WAIT_SFPU>(F2S);
    take_pack_token();
}

#endif // TRISC_MATH || LLK_TRISC_MATH

} // namespace topk_xl_split

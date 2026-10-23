// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

// TTI_REPLAY comes from the includer's ckernel.h (a TT-Metal kernel) or the deployment's headers.
namespace ckernel::sfpu::bf16::softplus
{
// An SFPU call owns replay slots 0-15: matmul and reduce record theirs from replay_buf_offset = 16 at init
// and replay them across the call. A row that would record past slot 15 is issued, or records only a head that fits
// (record_split).
#ifndef TTPOLY_SFPU_REPLAY_SLOTS
#define TTPOLY_SFPU_REPLAY_SLOTS 16
#endif
inline constexpr std::uint32_t kSfpuReplaySlots = TTPOLY_SFPU_REPLAY_SLOTS;
template <std::uint32_t Start, std::uint32_t Slots>
inline constexpr bool replay_fits = Start + Slots <= kSfpuReplaySlots;

// A row too long for the window records its first Head slots, the head: each replay plays them and issues the
// tail itself. Head is the head's exact slot count. Execute false records without running, as record_row.
template <std::uint32_t Start, std::uint32_t Slots, std::uint32_t Head, bool Execute = true, typename HeadRow, typename TailRow>
inline void record_split(HeadRow&& head, TailRow&& tail)
{
    static_assert(Head > 0 && Head <= Slots);
    if constexpr (replay_fits<Start, Slots> && Execute)
    {
        load_replay_buf<Exec>(
            Start,
            Slots,
            [&]
            {
                head();
                tail();
            });
    }
    if constexpr (replay_fits<Start, Slots> && !Execute)
    {
        load_replay_buf(
            Start,
            Slots,
            [&]
            {
                head();
                tail();
            });
    }
    if constexpr (!replay_fits<Start, Slots> && replay_fits<Start, Head> && Execute)
    {
        load_replay_buf<Exec>(
            Start,
            Head,
            [&]
            {
                head();
                tail();
            });
    }
    if constexpr (!replay_fits<Start, Slots> && replay_fits<Start, Head> && !Execute)
    {
        load_replay_buf(Start, Head, [&] { head(); });
    }
    if constexpr (!replay_fits<Start, Slots> && !replay_fits<Start, Head> && Execute)
    {
        head();
        tail();
    }
}

template <std::uint32_t Start, std::uint32_t Slots, std::uint32_t Head, typename HeadRow, typename TailRow>
inline void replay_split(HeadRow&& head, TailRow&& tail)
{
    if constexpr (replay_fits<Start, Slots>)
    {
        TTI_REPLAY(Start, Slots, 0, 0);
    }
    else if constexpr (replay_fits<Start, Head>)
    {
        TTI_REPLAY(Start, Head, 0, 0);
        tail();
    }
    else
    {
        head();
        tail();
    }
}
} // namespace ckernel::sfpu::bf16::softplus

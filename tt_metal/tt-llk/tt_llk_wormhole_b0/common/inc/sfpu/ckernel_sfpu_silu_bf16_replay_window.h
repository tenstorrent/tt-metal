// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

// TTI_REPLAY comes from the includer's ckernel.h (a TT-Metal kernel) or the deployment's headers.
namespace ckernel::sfpu::bf16::silu
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

// The row's instructions, recorded into slots Start.. while they execute (Execute) or without executing them, when
// they fit; otherwise executed (Execute) or nothing, since each replay_row then issues the row itself.
template <std::uint32_t Start, std::uint32_t Slots, bool Execute = true, typename Row>
inline void record_row(Row&& row)
{
    // One record per block: the Blackhole export brackets each record and its row with load_replay_buf.
    if constexpr (replay_fits<Start, Slots> && Execute)
    {
        TTI_REPLAY(Start, Slots, 1, 1);
        row();
    }
    if constexpr (replay_fits<Start, Slots> && !Execute)
    {
        TTI_REPLAY(Start, Slots, 0, 1);
        row();
    }
    if constexpr (!replay_fits<Start, Slots> && Execute)
    {
        row();
    }
}

// One replay of the row recorded at Start, or the row issued where it did not fit the window.
template <std::uint32_t Start, std::uint32_t Slots, typename Row>
inline void replay_row(Row&& row)
{
    if constexpr (replay_fits<Start, Slots>)
    {
        TTI_REPLAY(Start, Slots, 0, 0);
    }
    else
    {
        row();
    }
}
} // namespace ckernel::sfpu::bf16::silu

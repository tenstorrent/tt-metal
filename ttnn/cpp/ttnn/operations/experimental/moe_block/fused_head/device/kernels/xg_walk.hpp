// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The x all-gather's walk over chunks, shards and links.
// A fragment the including kernel inlines: it relies on what that kernel declared before it (included once a
// kernel).
#pragma once
constexpr uint32_t PPS = (KB + RP - 1) / RP, RPR = SPS * PPS;  // (packets a K step, a row tile)
// chunk m of shard q, link v: row tile R of the chunk (link (R / 2) % NV; part 1 / 2: even / odd R), step j, first
// tile kk0 within the step, n tiles; idx: the run's index in the link's full (part 0) walk of the chunk's shard
template <typename F>
FORCE_INLINE void walk(uint32_t m, uint32_t q, uint32_t part, uint32_t v, F&& f) {
    uint32_t cnt = 0;
    for (uint32_t R = m * CR; R < (m + 1) * CR; ++R) {
        const uint32_t lr = R - m * CR;  // (link and ring half by the chunk-local row: the host counts packets so)
        if ((lr / 2) % NV != v) {
            continue;
        }
        if (part == 0 || (lr & 1) == part - 1) {
            for (uint32_t st = 0; st < SPS; ++st) {
                for (uint32_t kk0 = 0, i = 0; kk0 < KB; kk0 += RP, ++i) {
                    f(R, q * SPS + st, st * KB + kk0, kk0, KB - kk0 < RP ? KB - kk0 : RP, cnt * RPR + st * PPS + i);
                }
            }
        }
        ++cnt;
    }
}
// x step j of chunk m, row tile R on a chip: its processing position there (the chip's shard rank, then the step
// within the shard), the position's owner core (NoC) and x-own address (se_engine3's layout). args at T: the chip's
// shard ranks [G]; fam0[S0] cols, fam1[S1] cols; per column: nown, family-1 base slot; column NoC x [GX]; row y [GY]
struct Dst {
    uint32_t x, y, addr;
};
FORCE_INLINE Dst dst_of(uint32_t T, uint32_t xown, uint32_t m, uint32_t R, uint32_t j, uint32_t kk0) {
    const uint32_t pos = get_arg_val<uint32_t>(T + j / SPS) * SPS + j % SPS;
    T += G;
    const uint32_t f = pos & 1, h = pos / 2;
    const uint32_t col = f ? get_arg_val<uint32_t>(T + S0 + h % S1) : get_arg_val<uint32_t>(T + h % S0);
    const uint32_t t = T + S0 + S1;
    const uint32_t nown = get_arg_val<uint32_t>(t + 2 * col), b1 = get_arg_val<uint32_t>(t + 2 * col + 1);
    const uint32_t slot = f ? b1 + h / S1 : h / S0;
    const uint32_t lr = R - m * CR;
    const uint32_t addr = xown + (((m * nown + slot) * MT + lr % MT) * KB + kk0) * 2048;
    return {get_arg_val<uint32_t>(t + 2 * GX + col), get_arg_val<uint32_t>(t + 3 * GX + lr / MT), addr};
}

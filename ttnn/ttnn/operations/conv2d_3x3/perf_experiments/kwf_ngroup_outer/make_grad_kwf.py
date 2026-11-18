# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only generator: op_grad/kernels/conv2d_3x3_kw_fold.hpp = HEAD's header comment (updated) + the measured copy's
run() (op_copy/) with the all-planes branch (ng_outer = false) and the plane stride removed: n-groups outer only."""

from pathlib import Path

HERE = Path(__file__).resolve().parent


def rep(text, old, new):
    assert old in text, old[:80]
    return text.replace(old, new)


src = (HERE / "op_copy/kernels/conv2d_3x3_kw_fold.hpp").read_text()
s = (HERE / "op_head/kernels/conv2d_3x3_kw_fold.hpp").read_text()

s = rep(
    s,
    """// Q_2[o + d]. The weights stream in column rounds, kw = 1 first, then 0, then 2: round kw is W'[(kh, c), kw, n]
// (3 C_in x C_out tiles, (kh, c, n) order), the prepared [K_p, C_out] tensor read tile-permuted (slot (kh, c, n) is
// prepared tile ((kh * 3 + kw) Ct + c, n)). Each Q_kw tile-row accumulates all its in-image kernel rows in DEST and is
// packed once.
//
// The shift: each Q_kw tile-row is pack-untilized (fp32 rows) into one fp32 accumulator ACC (cb_acc) at row
// rowbase(t) + (1 - kw) d, so the kw shift sums in fp32 in the packer (K in fp32 DEST). ACC leaves a d-row gap after
// every image row: the wrapped rows of Q_0 (w >= W - d) and Q_2 (w < d) land there and are never read (the masks cost
// nothing). Q_1 covers every valid ACC row exactly once and its round comes first, so it packs as an overwrite (packer
// L1 accumulation off) and Q_0 / Q_2, which spill d rows into the neighbouring row regions, accumulate onto it. Gap and
// halo rows take accumulations onto garbage, and nothing reads them. With a full-width subblock (sbw == N) each
// subblock is one packer stream (one close, not one per row). Last, each output tile is tilized from ACC (srcA, tf32)
// plus the bias (srcB) and packed with ReLU. The bias therefore has no reader seed and no wait before the first pack:
// the writer reads it after the weights and the reader decodes it into fp32 rows (cb_bias_rows) once its fill is done.""",
    """// Q_2[o + d]. The output columns are walked in n-groups of sbw tiles (the subblock width), one pass each; within a
// pass the columns kw go 1 first, then 0, then 2. The weights of column kw are W'[(kh, c), kw, n] (3 C_in x C_out
// tiles, (kh, c, n) order), the prepared [K_p, C_out] tensor read tile-permuted (slot (kh, c, n) is prepared tile
// ((kh * 3 + kw) Ct + c, n)). K-blocked, they stream in sub-rounds of one column and one n-group (3 C_in x sbw tiles),
// in pass order; whole-K, the three column rounds stay resident. Each Q_kw tile-row accumulates all its in-image
// kernel rows in DEST and is packed once.
//
// The shift: each Q_kw tile-row of the pass's n-group is pack-untilized (fp32 rows) into one fp32 accumulator plane
// ACC (cb_acc, [rows, sbw * 32]) at row rowbase(t) + (1 - kw) d, so the kw shift sums in fp32 in the packer (K in fp32
// DEST). ACC leaves a d-row gap after every image row: the wrapped rows of Q_0 (w >= W - d) and Q_2 (w < d) land there
// and are never read (the masks cost nothing). Q_1 covers every valid ACC row exactly once and its round comes first,
// so it packs as an overwrite (packer L1 accumulation off) and Q_0 / Q_2, which spill d rows into the neighbouring row
// regions, accumulate onto it. Gap and halo rows take accumulations onto garbage, and nothing reads them. A subblock
// covers the plane's full width, so each subblock is one packer stream (one close, not one per row). Last, each output
// tile of the n-group is tilized from ACC (srcA, tf32) plus the bias (srcB) and packed with ReLU; the next pass's Q_1
// overwrites the plane. The bias therefore has no reader seed and no wait before the first pack: the writer reads it
// after the first pass's weights and the reader decodes it into fp32 rows (cb_bias_rows) once its fill is done.
// One ACC plane (not N / sbw) and one sub-round per weight slot let kw-fold run sem res2.fuse_conv.0 / .1 (C_out 256)
// in one M-block per core. Measured (Blackhole p150b, LoFi, against the K-blocked tap-direct walk they ran before):
// 135.9 -> 100.4 and 108.8 -> 89.7 us; all planes resident measured the same (89.4 us) at 215 KB more L1.""",
)

i0 = s.index("template <\n    uint32_t cb_in,")
j0 = s.index("}  // namespace conv2d_3x3::kw_fold")
i1 = src.index("template <\n    uint32_t cb_in,")
j1 = src.index("}  // namespace conv2d_3x3::kw_fold")
body = src[i1:j1]
body = rep(body, "    uint32_t plane_bytes,\n    bool noinline,\n    bool ng_outer>", "    bool noinline>")
body = rep(
    body,
    """    constexpr uint32_t R = d * tpr;
    // ACC is column planes of [rows, sbw * 32] fp32: a subblock covers the full width of its plane, so every subblock
    // pack is one contiguous stream.
    constexpr uint32_t PITCH = sbw * 32 * 4;
    constexpr uint32_t PLANE = plane_bytes;
    static_assert(N % sbw == 0 && sbw * sbh <= 4 && d >= 1 && d <= 32);
    // n-groups: NG column groups of sbw output column tiles, one ACC plane each. ng_outer: the n-groups are the outer
    // loop, one pass (column rounds kw = 1, 0, 2, then a final pass) per n-group and one ACC plane resident; else one
    // pass over all NG planes. K-blocked weights stream in sub-rounds of one column kw and one n-group, W'[(kh, c), kw,
    // n-group] (3 Ct x sbw tiles, (kh, c, n) order), in pass order: sub-round s = (g = s / 3, kw = KW_ROUND_ORDER[s %
    // 3]) with ng_outer, else (kw = KW_ROUND_ORDER[s / NG], g = s % NG). Whole-K keeps the three column rounds
    // (3 Ct x N tiles each) resident. With N == sbw (NG 1) every form is the column rounds.
    constexpr uint32_t NG = N / sbw;
    constexpr uint32_t RW = kblocked ? sbw : N;       // output column tiles of one weight round
    constexpr uint32_t ROUND_TILES = 3 * Ct * RW;
    constexpr uint32_t PASSES = ng_outer ? NG : 1;   // matmul + final passes per M-block
    constexpr uint32_t PASS_GROUPS = NG / PASSES;    // n-groups per pass
    constexpr uint32_t PASS_N = PASS_GROUPS * sbw;   // output column tiles per pass
""",
    """    constexpr uint32_t R = d * tpr;
    // ACC is one plane of [rows, sbw * 32] fp32: a subblock covers its full width, so every subblock pack is one
    // contiguous stream.
    constexpr uint32_t PITCH = sbw * 32 * 4;
    static_assert(N % sbw == 0 && sbw * sbh <= 4 && d >= 1 && d <= 32);
    // One pass per n-group of sbw output column tiles: its columns kw = 1, 0, 2 onto the ACC plane, then its final
    // pass. K-blocked weights stream in sub-rounds of one column and one n-group, sub-round s = (n-group s / 3, kw =
    // KW_ROUND_ORDER[s % 3]): 3 Ct x sbw tiles, (kh, c, n) order. Whole-K keeps the three column rounds (3 Ct x N tiles
    // each) resident. With N == sbw (one n-group) both are the column rounds.
    constexpr uint32_t NG = N / sbw;
    constexpr uint32_t RW = kblocked ? sbw : N;  // output column tiles of one weight round
    constexpr uint32_t ROUND_TILES = 3 * Ct * RW;
""",
)
body = rep(
    body, "        for (uint32_t pass = 0; pass < PASSES; ++pass) {", "        for (uint32_t g = 0; g < NG; ++g) {"
)
body = rep(
    body,
    """                // One walk per weight sub-round, in round order kw = 1, 0, 2 (per n-group), as the sub-round lands: Q_kw's
                // rows each accumulate every in-image kernel row in DEST and are packed onto ACC once (Q_1: the valid
                // rows, as an overwrite; Q_0: tile-rows t0 - 1 .. t_end - 1 and Q_2: t0 .. t_end, accumulating).
                // K-blocked: one sub-round at a time (double-buffered, popped after its walk). Whole-K: the sub-rounds
                // stay resident across M-blocks; sub-round s waits only for sub-rounds 0 .. s.""",
    """                // n-group g's columns kw = 1, 0, 2, each as its weights land: Q_kw's rows each accumulate every
                // in-image kernel row in DEST and are packed onto ACC once (Q_1: the valid rows, as an overwrite; Q_0:
                // tile-rows t0 - 1 .. t_end - 1 and Q_2: t0 .. t_end, accumulating). K-blocked: one sub-round at a time
                // (double-buffered, popped after its walk). Whole-K: the rounds stay resident across M-blocks; column
                // step waits only for rounds 0 .. step.""",
)
body = rep(
    body,
    """                // Rows [i0, i0 + h) (tile-row t0 - 1 + i), one kw, n-group g (columns [g sbw, g sbw + sbw)), its
                // sub-round's tiles from in1 index w0.
                auto subblock_body = [&](uint32_t i0, uint32_t h, uint32_t kw, uint32_t g, uint32_t w0) {""",
    """                // Rows [i0, i0 + h) (tile-row t0 - 1 + i), one kw, the n-group's columns from in1 index w0 (row
                // stride RW).
                auto subblock_body = [&](uint32_t i0, uint32_t h, uint32_t kw, uint32_t w0) {""",
)
body = rep(
    body,
    """                    // One stream per run of rows that are consecutive in ACC (an image-row gap splits it).
                    const uint32_t plane = ng_outer ? 0 : g;
                    for (uint32_t r = 0; r < h;) {""",
    """                    // One stream per run of rows that are consecutive in ACC (an image-row gap splits it).
                    for (uint32_t r = 0; r < h;) {""",
)
body = rep(
    body,
    """                        PACK((pack_stream<sbw>(
                            acc_base + plane * PLANE + (rowbase<d, tpr>(t0, i0 + r) + d - kw * d) * PITCH, r, e - r)));""",
    """                        PACK((pack_stream<sbw>(
                            acc_base + (rowbase<d, tpr>(t0, i0 + r) + d - kw * d) * PITCH, r, e - r)));""",
)
body = rep(
    body,
    """                auto subblock_out_of_line = [&](uint32_t i0, uint32_t h, uint32_t kw, uint32_t g, uint32_t w0)
                                                __attribute__((noinline)) { subblock_body(i0, h, kw, g, w0); };
                auto subblock = [&](uint32_t i0, uint32_t h, uint32_t kw, uint32_t g, uint32_t w0) {
                    if constexpr (noinline) {
                        subblock_out_of_line(i0, h, kw, g, w0);
                    } else {
                        subblock_body(i0, h, kw, g, w0);
                    }
                };""",
    """                auto subblock_out_of_line = [&](uint32_t i0, uint32_t h, uint32_t kw, uint32_t w0)
                                                __attribute__((noinline)) { subblock_body(i0, h, kw, w0); };
                auto subblock = [&](uint32_t i0, uint32_t h, uint32_t kw, uint32_t w0) {
                    if constexpr (noinline) {
                        subblock_out_of_line(i0, h, kw, w0);
                    } else {
                        subblock_body(i0, h, kw, w0);
                    }
                };""",
)
body = rep(
    body,
    """                    for (uint32_t gi = 0; gi < PASS_GROUPS; ++gi) {
                        const uint32_t g = pass * PASS_GROUPS + gi;
                        {
                            // K-blocked: this sub-round at the front (the previous one was popped), so its in1 tiles
                            // start at index 0. Whole-K: rounds 0 .. step.
                            MaybeDeviceZoneScope("compute_kwf_weight_wait");
                            w.wait_front(kblocked ? ROUND_TILES : (step + 1) * ROUND_TILES);
                        }
                        const uint32_t w0 = kblocked ? 0 : step * ROUND_TILES + g * sbw;
                        for (uint32_t i0 = i_lo; i0 < i_end; i0 += sbh) {
                            const uint32_t h = i_end - i0 < sbh ? i_end - i0 : sbh;
                            subblock(i0, h, kw, g, w0);
                        }
                        if constexpr (kblocked) {
                            w.pop_front(ROUND_TILES);
                        }
                    }""",
    """                    {
                        // K-blocked: this sub-round at the front (the previous one was popped), so its in1 tiles start
                        // at index 0. Whole-K: rounds 0 .. step.
                        MaybeDeviceZoneScope("compute_kwf_weight_wait");
                        w.wait_front(kblocked ? ROUND_TILES : (step + 1) * ROUND_TILES);
                    }
                    const uint32_t w0 = kblocked ? 0 : step * ROUND_TILES + g * sbw;
                    for (uint32_t i0 = i_lo; i0 < i_end; i0 += sbh) {
                        const uint32_t h = i_end - i0 < sbh ? i_end - i0 : sbh;
                        subblock(i0, h, kw, w0);
                    }
                    if constexpr (kblocked) {
                        w.pop_front(ROUND_TILES);
                    }""",
)
body = rep(
    body,
    """                // every ACC write. The next pass's (or M-block's) Q_1 overwrite of the plane is packed after its matmul,
                // whose operands the unpacker reads after every read of this final pass.""",
    """                // every ACC write. The next n-group's (or M-block's) Q_1 overwrite of the plane is packed after its
                // matmul, whose operands the unpacker reads after every read of this final pass.""",
)
body = rep(
    body,
    "            // ---- output tile = tilize(ACC rows) + bias, ReLU on the pack: this pass's PASS_N output columns.",
    "            // ---- output tile = tilize(ACC rows) + bias, ReLU on the pack: the n-group's sbw output columns.",
)
body = rep(body, "                if (blk == 0 && pass == 0) {", "                if (blk == 0 && g == 0) {")
body = rep(
    body,
    """                if (pass == 0) {
                    out.reserve_back(bm * N);
                }
                const uint32_t n_first = pass * PASS_N;
                const uint32_t total = valid * PASS_N;""",
    """                if (g == 0) {
                    out.reserve_back(bm * N);
                }
                const uint32_t n_first = g * sbw;
                const uint32_t total = valid * sbw;""",
)
body = rep(
    body,
    """                    for (uint32_t j = 0; j < cnt; ++j) {
                        const uint32_t u = (f0 + j) / PASS_N;
                        const uint32_t nn = (f0 + j) - u * PASS_N;  // column within the pass
                        UNPACK((unpack_tilize_bias_at<cb_acc>(
                            acc_rd + (nn / sbw) * PLANE + rowbase<d, tpr>(t0, u + 1) * PITCH,
                            nn % sbw,
                            rows_rd + (n_first + nn) * ROWS_BYTES)));""",
    """                    for (uint32_t j = 0; j < cnt; ++j) {
                        const uint32_t u = (f0 + j) / sbw;
                        const uint32_t nn = (f0 + j) - u * sbw;  // column tile within the n-group
                        UNPACK((unpack_tilize_bias_at<cb_acc>(
                            acc_rd + rowbase<d, tpr>(t0, u + 1) * PITCH, nn, rows_rd + (n_first + nn) * ROWS_BYTES)));""",
)
body = rep(
    body,
    """                    for (uint32_t j = 0; j < cnt; ++j) {
                        const uint32_t u = (f0 + j) / PASS_N;
                        const uint32_t nn = (f0 + j) - u * PASS_N;
                        PACK((llk_pack<DST_ACCUM_MODE, true, PackMode::Default>(j, cb_out, u * N + n_first + nn)));
                    }""",
    """                    for (uint32_t j = 0; j < cnt; ++j) {
                        const uint32_t u = (f0 + j) / sbw;
                        const uint32_t nn = (f0 + j) - u * sbw;
                        PACK((llk_pack<DST_ACCUM_MODE, true, PackMode::Default>(j, cb_out, u * N + n_first + nn)));
                    }""",
)
body = rep(
    body,
    """                if (pass + 1 == PASSES) {
                    out.push_back(bm * N);
                }""",
    """                if (g + 1 == NG) {
                    out.push_back(bm * N);
                }""",
)
for tok in ["PASS_N", "PASS_GROUPS", "PASSES", "ng_outer", "PLANE", "pass ="]:
    assert tok not in body, tok
s = s[:i0] + body + s[j0:]
(HERE / "op_grad/kernels/conv2d_3x3_kw_fold.hpp").write_text(s)
print("wrote op_grad/kernels/conv2d_3x3_kw_fold.hpp")

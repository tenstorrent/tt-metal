# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Regenerates the reader variants of the `per_channel_boot_overlap` bake-off from
# the SHIPPED reader, so every variant differs from the baseline by exactly the
# transform named below and nothing else.  Run with plain python3 (no device).
#
#   python3 ttnn/ttnn/operations/rms_norm_ttnn/perf_experiments/per_channel_boot_overlap/patch_kernels.py
#
# k_base       shipped + the TRIM == 3 granularity branch (dead unless the host
#              knob asks for it) -- the honest baseline.
# k_stag       k_base + the per-core ROTATION of the per-channel tile loop.
# k_split      k_base + the per-channel ISSUE / BARRIER split.
# k_split_stag both.
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHIPPED = HERE.parent.parent / "kernels"
READER = "rms_norm_ttnn_reader.cpp"

SRC = (SHIPPED / READER).read_text()

# --------------------------------------------------------------------------
# (0) TRIM == 3: ONE read per tile covering BOTH face-rows.
# --------------------------------------------------------------------------
TRIM3_ANCHOR = """            } else if constexpr (TRIM == 1) {
                // Half the page from offset 0 == faces 0 and 1 in EVERY tiled
                // format, block-float included; needs no face-stride alignment."""
TRIM3_NEW = """            } else if constexpr (TRIM == 3) {
                // PERF 2 `per_channel_boot_overlap`, granularity option (3): ONE
                // read from offset 0 spanning face 0's row 0 THROUGH face 1's row
                // 0 -- `tile_bytes / 4 + 32 * elem` bytes.  Same two face-rows
                // TRIM == 2 fetches, same destination offsets (it is a prefix of
                // the tile, copied verbatim), at HALF the transaction count and
                // 4.5x the bytes of TRIM == 2 (576 B vs 2 x 64 B at bf16).  The
                // dead middle is face 0's rows 1..15, which the BroadcastDim::Row
                // consumer never reads -- exactly the bytes TRIM == 2 leaves
                // stale.  Needs NO face-stride alignment (single read, offset 0).
                constexpr uint32_t ROW_BYTES = TILE_DIM * ELEM_BYTES;
                // BLOCK-FLOAT SAFETY (the host `_trim_for` is where this gate
                // BELONGS -- it is here so the bench cannot mis-measure a format
                // it would truncate).  A bfloat8_b tile is a 64 B exponent header
                // plus 4 x 256 B mantissa faces, so face 1's row 0 sits at
                // 64 + 256 == tile_bytes / 4 + 48, NOT at tile_bytes / 4: the
                // prefix would stop 32 B short.  `tile_bytes / 4 % 64 != 0` is
                // exactly D23's `legal_2` test, and the fall-back is D23's own
                // answer for those formats (the half page).
                const uint32_t face = tile_bytes / 4;
                const uint32_t len = (face % 64 == 0) ? (face + ROW_BYTES) : (tile_bytes / 2);
                noc_async_read(get_noc_addr(tile_id, acc), l1_addr, len);
            } else if constexpr (TRIM == 1) {
                // Half the page from offset 0 == faces 0 and 1 in EVERY tiled
                // format, block-float included; needs no face-stride alignment."""
assert SRC.count(TRIM3_ANCHOR) == 1
BASE = SRC.replace(TRIM3_ANCHOR, TRIM3_NEW)

BASE_HEAD = """// ============================================================================
// PERF EXPERIMENT `per_channel_boot_overlap` -- k_base (the honest BASELINE)
// ============================================================================
// Byte-identical to ttnn/ttnn/operations/rms_norm_ttnn/kernels/rms_norm_ttnn_reader.cpp
// except for one DEAD compile-time branch (`TRIM == 3`), which the host only ever
// selects when the bake-off's granularity knob asks for it.  Its agreement with
// the `shipped` variant is a control in the results table.
// ============================================================================
"""

# --------------------------------------------------------------------------
# (1) STAGGER: rotate the per-channel tile loop by a per-core offset.
# --------------------------------------------------------------------------
STAG_OLD = """        const uint32_t tile_bytes = get_tile_size(CB_TILES);
        cb_reserve_back(CB_TILES, WT_CHUNK);
        uint32_t l1_addr = get_write_ptr(CB_TILES);
        for (uint32_t w = 0; w < WT_CHUNK; ++w) {
            const uint32_t wt = first_wt + w;"""
STAG_NEW = """        const uint32_t tile_bytes = get_tile_size(CB_TILES);
        cb_reserve_back(CB_TILES, WT_CHUNK);
        const uint32_t l1_base = get_write_ptr(CB_TILES);
        // PERF 2 `per_channel_boot_overlap`, option STAGGER.  Every core reads the
        // SAME WT_CHUNK per-channel pages, and it reads them in the SAME order --
        // so 110 cores pile onto the bank holding page `first_wt` at the same
        // instant and drain it in lock-step (measured spread: 67 us mean vs 158 us
        // max occupancy on the focus shape).  Rotating the START of the loop by a
        // per-core offset fans the first request of each core across the banks
        // instead.  It changes NO transaction, NO byte and NO destination address
        // -- only the ISSUE ORDER -- so the staged tile is bit-identical.
        //
        // The seed uses only runtime args the kernel already has: `row_start` is
        // this core's distinct index under the row schemes (w_start == 0), and
        // `w_start / WT_CHUNK` is it under the width schemes (row_start == 0).
        const uint32_t rot = stagger % WT_CHUNK;
        for (uint32_t i = 0; i < WT_CHUNK; ++i) {
            const uint32_t w = (i + rot < WT_CHUNK) ? (i + rot) : (i + rot - WT_CHUNK);
            const uint32_t l1_addr = l1_base + w * tile_bytes;
            const uint32_t wt = first_wt + w;"""

STAG_TAIL_OLD = """            }
#endif
            l1_addr += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(CB_TILES, WT_CHUNK);"""
STAG_TAIL_NEW = """            }
#endif
        }
        noc_async_read_barrier();
        cb_push_back(CB_TILES, WT_CHUNK);"""

# signature + call sites gain the runtime `stagger` argument
SIG_OLD = "FORCE_INLINE void stage_per_channel_chunk(const Acc& acc, uint32_t first_wt) {"
SIG_NEW = "FORCE_INLINE void stage_per_channel_chunk(const Acc& acc, uint32_t first_wt, uint32_t stagger) {"
CALL_G_OLD = "                PC_NARROW>(g_acc, first_wt);"
CALL_G_NEW = "                PC_NARROW>(g_acc, first_wt, pc_stagger);"
CALL_B_OLD = "                PC_NARROW>(b_acc, first_wt);"
CALL_B_NEW = "                PC_NARROW>(b_acc, first_wt, pc_stagger);"
LAMBDA_OLD = """    auto stage_per_channel = [&](uint32_t c) {
        const uint32_t first_wt = w_start + c * WT_CHUNK;"""
LAMBDA_NEW = """    const uint32_t pc_stagger = row_start + (WT_CHUNK ? (w_start / WT_CHUNK) : 0u);
    auto stage_per_channel = [&](uint32_t c) {
        const uint32_t first_wt = w_start + c * WT_CHUNK;"""


def apply_stagger(text):
    for old, new in (
        (STAG_OLD, STAG_NEW),
        (STAG_TAIL_OLD, STAG_TAIL_NEW),
        (SIG_OLD, SIG_NEW),
        (CALL_G_OLD, CALL_G_NEW),
        (CALL_B_OLD, CALL_B_NEW),
        (LAMBDA_OLD, LAMBDA_NEW),
    ):
        assert text.count(old) == 1, old[:70]
        text = text.replace(old, new)
    return text


STAG_HEAD = """// ============================================================================
// PERF EXPERIMENT `per_channel_boot_overlap` -- STAGGER
// ============================================================================
// k_base with ONE change: the per-channel operand's TILE read loop starts at a
// per-core rotation instead of at tile 0.  Same reads, same bytes, same
// destinations, same barrier -- only the order in which the 110 cores hit the
// operand's shared DRAM pages.  See the comment at the loop.
// ============================================================================
"""

# --------------------------------------------------------------------------
# (2) SPLIT: per-channel ISSUE at boot, BARRIER + push under the first x read.
# --------------------------------------------------------------------------
SPLIT_HELPERS = """
// ---- PERF 2 `per_channel_boot_overlap`: the per-channel read, ISSUE ONLY -----
// Byte-for-byte the transactions `stage_per_channel_chunk` issues for the TILE
// and the ROW_MAJOR *blocked* forms; it just RETURNS BEFORE THE BARRIER so the
// caller can put the first activation chunk in front of the DRAM wait.
//
// RAW-NoC / HELPER-BYPASS JUSTIFICATION (required by the raw-path rule).
// Helper bypassed: `stage_per_channel_chunk` above (and, beneath its ROW_MAJOR
// flat form, `dataflow_kernel_lib::read_sticks_for_tilize`).  Both own
// `reserve -> issue -> noc_async_read_barrier() -> cb_push_back` as ONE
// indivisible unit and expose no parameter, overload or template knob that
// returns after the issue but before the barrier.  Gap kind: CAPABILITY, not
// ergonomics -- an issue/finish split is INEXPRESSIBLE through them, so the only
// way to get one is to re-spell the reads.  Only the two forms whose issue is a
// plain `noc_async_read` loop are split (TILE, and ROW_MAJOR *blocked*); the
// ROW_MAJOR flat and NARROW forms fall back to the unsplit helper call in its
// shipped position, which `PC_SPLIT` gates.
template <
    uint32_t CB_STICKS,
    uint32_t CB_TILES,
    uint32_t WT,
    uint32_t WT_CHUNK,
    uint32_t ELEM_BYTES,
    uint32_t TRIM,
    bool IS_RM,
    typename Acc>
FORCE_INLINE void stage_per_channel_chunk_issue(const Acc& acc, uint32_t first_wt, uint32_t stagger) {
    constexpr uint32_t TILE_COL_BYTES = TILE_DIM * ELEM_BYTES;
    if constexpr (IS_RM) {
        // BLOCKED, non-narrow only (the caller's PC_SPLIT gate guarantees it).
        cb_reserve_back(CB_STICKS, WT_CHUNK);
        const uint32_t l1_base = get_write_ptr(CB_STICKS);
        for (uint32_t w = 0; w < WT_CHUNK; ++w) {
            const uint32_t wt = first_wt + w;
            const uint32_t page = (wt < WT) ? wt : (WT - 1);
            noc_async_read(acc.get_noc_addr(page, 0), l1_base + w * TILE_COL_BYTES, TILE_COL_BYTES);
        }
    } else {
        const uint32_t tile_bytes = get_tile_size(CB_TILES);
        cb_reserve_back(CB_TILES, WT_CHUNK);
        const uint32_t l1_base = get_write_ptr(CB_TILES);
        const uint32_t rot = stagger % WT_CHUNK;
        for (uint32_t i = 0; i < WT_CHUNK; ++i) {
            const uint32_t w = (i + rot < WT_CHUNK) ? (i + rot) : (i + rot - WT_CHUNK);
            const uint32_t l1_addr = l1_base + w * tile_bytes;
            const uint32_t wt = first_wt + w;
            const uint32_t tile_id = (wt < WT) ? wt : (WT - 1);
            if constexpr (TRIM == 2) {
                constexpr uint32_t ROW_BYTES = TILE_DIM * ELEM_BYTES;
                const uint64_t base = get_noc_addr(tile_id, acc);
                const uint32_t face = tile_bytes / 4;
                noc_async_read(base, l1_addr, ROW_BYTES);
                noc_async_read(base + face, l1_addr + face, ROW_BYTES);
            } else if constexpr (TRIM == 3) {
                constexpr uint32_t ROW_BYTES = TILE_DIM * ELEM_BYTES;
                // BLOCK-FLOAT SAFETY (the host `_trim_for` is where this gate
                // BELONGS -- it is here so the bench cannot mis-measure a format
                // it would truncate).  A bfloat8_b tile is a 64 B exponent header
                // plus 4 x 256 B mantissa faces, so face 1's row 0 sits at
                // 64 + 256 == tile_bytes / 4 + 48, NOT at tile_bytes / 4: the
                // prefix would stop 32 B short.  `tile_bytes / 4 % 64 != 0` is
                // exactly D23's `legal_2` test, and the fall-back is D23's own
                // answer for those formats (the half page).
                const uint32_t face = tile_bytes / 4;
                const uint32_t len = (face % 64 == 0) ? (face + ROW_BYTES) : (tile_bytes / 2);
                noc_async_read(get_noc_addr(tile_id, acc), l1_addr, len);
            } else if constexpr (TRIM == 1) {
                noc_async_read(get_noc_addr(tile_id, acc), l1_addr, tile_bytes / 2);
            } else {
                noc_async_read_tile(tile_id, acc, l1_addr);
            }
        }
    }
}

template <uint32_t CB_STICKS, uint32_t CB_TILES, uint32_t WT_CHUNK, bool IS_RM>
FORCE_INLINE void stage_per_channel_chunk_push() {
    if constexpr (IS_RM) {
        cb_push_back(CB_STICKS, WT_CHUNK);
    } else {
        cb_push_back(CB_TILES, WT_CHUNK);
    }
}

"""
SPLIT_HELPERS_ANCHOR = "// ---- native activations: publish a resident shard's pages, once -------------"

SPLIT_PROLOGUE_OLD = """    // Resident per-channel operands are staged ONCE per core, for every chunk the
    // row is cut into (NUM_W_CHUNKS == 1 in the RESIDENT regime, so this is one
    // call there).  In STREAM they are re-staged per pass-B chunk of every
    // row-block instead -- which for a prefill profile is as many DRAM bytes as x.
    if constexpr (X_RESIDENT) {
        for (uint32_t c = 0; c < NUM_W_CHUNKS; ++c) {
            stage_per_channel(c);
        }
    }
"""
SPLIT_PROLOGUE_NEW = """    // ==== PERF 2: ISSUE the per-channel read, then BARRIER under the first x ====
    // The per-channel operand is a SERIAL PROLOGUE with an empty pipeline behind
    // it: `stage_per_channel` blocks the reader on DRAM before a single x byte has
    // been requested.  Splitting it lets the operand's latency drain UNDER the
    // first activation chunk, and costs no extra barrier -- `noc_async_read_barrier()`
    // fences NoC-0 GLOBALLY, which is the same property the x + residual streams
    // already share one barrier on.
    //
    // Splittable only where the issue is a plain non-blocking read loop AND no
    // later step on this NoC borrows the write command buffer while the reads are
    // still in flight:
    //   * the ROW_MAJOR flat / NARROW forms go through helpers (or a per-page
    //     barrier) that own their own barrier -- INEXPRESSIBLE as a split;
    //   * a ragged width chunk (WT_PAD) and the BAND scheme both issue
    //     `async_write_zeros` between the reserve and the read, and that engine is
    //     released only by its OWN barrier, so in-flight reads must not straddle it;
    //   * NUM_W_CHUNKS > 1 (ROW_RESIDENT) would need two un-pushed reserves on the
    //     same ring, which alias to the same write pointer.
    // Everything that does not qualify keeps the shipped ordering exactly.
    constexpr bool PC_FORM_SPLITTABLE =
        (!PC_RM) || (!PC_NARROW && (!HAS_G || GAMMA_BLOCKED != 0) && (!HAS_B || BIAS_BLOCKED != 0));
    constexpr bool PC_SPLIT =
        X_RESIDENT && (HAS_G || HAS_B) && PC_FORM_SPLITTABLE && !HAS_WPAD && !BAND_X && (NUM_W_CHUNKS == 1);

    [[maybe_unused]] auto issue_per_channel = [&](uint32_t c) {
        const uint32_t first_wt = w_start + c * WT_CHUNK;
        if constexpr (HAS_G) {
            MaybeDeviceZoneScope("reader_issue_gamma");
            const auto g_acc = TensorAccessor(gamma_args, gamma_addr);
            stage_per_channel_chunk_issue<
                cb_gamma_sticks,
                cb_gamma_tiles,
                WT,
                WT_CHUNK,
                GAMMA_ELEM_BYTES,
                GAMMA_TRIM,
                PC_RM>(g_acc, first_wt, pc_stagger);
        }
        if constexpr (HAS_B) {
            MaybeDeviceZoneScope("reader_issue_bias");
            const auto b_acc = TensorAccessor(bias_args, bias_addr);
            stage_per_channel_chunk_issue<
                cb_bias_sticks,
                cb_bias_tiles,
                WT,
                WT_CHUNK,
                BIAS_ELEM_BYTES,
                BIAS_TRIM,
                PC_RM>(b_acc, first_wt, pc_stagger);
        }
    };
    [[maybe_unused]] bool pc_pending = false;
    [[maybe_unused]] auto finish_per_channel = [&]() {
        if (!pc_pending) {
            return;
        }
        pc_pending = false;
        MaybeDeviceZoneScope("reader_pc_barrier");
        noc_async_read_barrier();
        if constexpr (HAS_G) {
            stage_per_channel_chunk_push<cb_gamma_sticks, cb_gamma_tiles, WT_CHUNK, PC_RM>();
        }
        if constexpr (HAS_B) {
            stage_per_channel_chunk_push<cb_bias_sticks, cb_bias_tiles, WT_CHUNK, PC_RM>();
        }
    };

    if constexpr (PC_SPLIT) {
        issue_per_channel(0);
        pc_pending = true;
    }
    if constexpr (X_RESIDENT && !PC_SPLIT) {
        for (uint32_t c = 0; c < NUM_W_CHUNKS; ++c) {
            stage_per_channel(c);
        }
    }
"""

SPLIT_LOOP_OLD = """                stage_activations_chunk(r0, rows, c);
                // STREAM: the per-channel operands are chunked and re-read for
                // every pass-B chunk."""
SPLIT_LOOP_NEW = """                stage_activations_chunk(r0, rows, c);
                // PERF 2: the FIRST activation chunk has now issued AND drained a
                // barrier of its own, so the per-channel reads had the whole of it
                // to complete.  One extra `noc_async_read_barrier()` on a queue that
                // is already empty is a handful of cycles; on the interleaved path
                // the x read's own barrier covered these reads too, so it is free.
                // Deferring FURTHER (past the second block) would deadlock: the next
                // `cb_reserve_back(cb_input_tiles)` would wait on a compute kernel
                // that is itself waiting on the un-pushed per-channel CB.
                finish_per_channel();
                // STREAM: the per-channel operands are chunked and re-read for
                // every pass-B chunk."""

SPLIT_TAIL_OLD = """                if constexpr (!X_RESIDENT) {
                    if (pass == 1) {
                        stage_per_channel(c);
                    }
                }
            }
        }
    }
}
"""
SPLIT_TAIL_NEW = """                if constexpr (!X_RESIDENT) {
                    if (pass == 1) {
                        stage_per_channel(c);
                    }
                }
            }
        }
    }
    // Belt and braces: `num_rows == 0` returned long ago, so the loop above always
    // runs at least once and this is already a no-op -- but a pending reserve that
    // is never pushed is a HANG, so the invariant is spelled out rather than argued.
    finish_per_channel();
}
"""

SPLIT_HEAD = """// ============================================================================
// PERF EXPERIMENT `per_channel_boot_overlap` -- SPLIT (issue / finish)
// ============================================================================
// k_base with the per-channel operand's DRAM read split into its NoC ISSUE (a
// reserve plus the same reads, non-blocking) and its BARRIER + push.  The issue
// runs where the shipped read runs; the barrier moves to just after the FIRST
// activation chunk, which already owns a barrier -- so the operand's DRAM latency
// drains under the first x read instead of in front of it.  See the RAW-NoC
// justification on `stage_per_channel_chunk_issue`.
// ============================================================================
"""


def apply_split(text, stagger_seed_present):
    assert text.count(SPLIT_HELPERS_ANCHOR) == 1
    text = text.replace(SPLIT_HELPERS_ANCHOR, SPLIT_HELPERS.lstrip("\n") + SPLIT_HELPERS_ANCHOR)
    if not stagger_seed_present:
        # the split's issue path always takes a stagger argument; without the
        # STAGGER transform the seed is a constant 0 (shipped issue order).
        assert text.count(LAMBDA_OLD) == 1
        text = text.replace(LAMBDA_OLD, "    const uint32_t pc_stagger = 0;\n" + LAMBDA_OLD)
    for old, new in (
        (SPLIT_PROLOGUE_OLD, SPLIT_PROLOGUE_NEW),
        (SPLIT_LOOP_OLD, SPLIT_LOOP_NEW),
        (SPLIT_TAIL_OLD, SPLIT_TAIL_NEW),
    ):
        assert text.count(old) == 1, old[:70]
        text = text.replace(old, new, 1)
    text = text.replace(
        "    auto stage_per_channel = [&](uint32_t c) {",
        "    [[maybe_unused]] auto stage_per_channel = [&](uint32_t c) {",
    )
    return text


def emit(name, text, head):
    d = HERE / name
    d.mkdir(exist_ok=True)
    for f in ("perf_instrumentation.hpp", "rms_norm_ttnn_compute.cpp", "rms_norm_ttnn_writer.cpp"):
        shutil.copy(SHIPPED / f, d / f)
    (d / READER).write_text(head + text)
    print(f"wrote {d / READER} ({len(text.splitlines())} lines)")


emit("k_base", BASE, BASE_HEAD)
emit("k_stag", apply_stagger(BASE), STAG_HEAD)
emit("k_split", apply_split(BASE, False), SPLIT_HEAD)
emit("k_split_stag", apply_split(apply_stagger(BASE), True), SPLIT_HEAD + STAG_HEAD)


# --------------------------------------------------------------------------
# (3) SPLIT-LATE: the per-channel ISSUE moves BEHIND the first x chunk.
# --------------------------------------------------------------------------
# `split` measured a REGRESSION on the focus shape (194860 vs 186551 ns): sharing
# one barrier makes the FIRST x push wait for max(gamma, x0) rather than letting
# x0 land first, and x0 is what unblocks the compute kernel.  `split_late` keeps
# the hiding but moves it one stage down -- gamma is issued AFTER x block 0's
# barrier and finished after x block 1's -- so the operand drains under a chunk
# that nothing is waiting on.
LATE_BOOT_OLD = """    if constexpr (PC_SPLIT) {
        issue_per_channel(0);
        pc_pending = true;
    }
"""
LATE_BOOT_NEW = """    [[maybe_unused]] bool pc_issued = false;
"""
LATE_LOOP_OLD = """                finish_per_channel();
                // STREAM: the per-channel operands are chunked and re-read for"""
LATE_LOOP_NEW = """                if constexpr (PC_SPLIT) {
                    // Block 0: issue (x0 has ALREADY landed, so nothing that the
                    // compute kernel is waiting on is delayed).  Block 1: its
                    // barrier drains the operand for free.  A one-block core
                    // issues and finishes at the belt-and-braces call below.
                    if (!pc_issued) {
                        pc_issued = true;
                        issue_per_channel(0);
                        pc_pending = true;
                    } else {
                        finish_per_channel();
                    }
                }
                // STREAM: the per-channel operands are chunked and re-read for"""
LATE_TAIL_OLD = """    finish_per_channel();
}
"""
LATE_TAIL_NEW = """    if constexpr (PC_SPLIT) {
        if (!pc_issued) {
            issue_per_channel(0);
            pc_pending = true;
        }
    }
    finish_per_channel();
}
"""

LATE_HEAD = """// ============================================================================
// PERF EXPERIMENT `per_channel_boot_overlap` -- SPLIT-LATE
// ============================================================================
// k_split with the ISSUE moved BEHIND the first activation chunk's barrier, so
// the operand's DRAM latency drains under x block 1 instead of contending with
// x block 0 -- the chunk the compute kernel is actually blocked on.
// ============================================================================
"""


def apply_late(text):
    for old, new in ((LATE_BOOT_OLD, LATE_BOOT_NEW), (LATE_LOOP_OLD, LATE_LOOP_NEW), (LATE_TAIL_OLD, LATE_TAIL_NEW)):
        assert text.count(old) == 1, old[:60]
        text = text.replace(old, new)
    return text


emit("k_split_late", apply_late(apply_split(BASE, False)), LATE_HEAD)
emit(
    "k_ablate",
    BASE,
    BASE_HEAD
    + "#define RMS_ABLATE_PER_CHANNEL 1\n// ABLATION ONLY -- the per-channel NoC PAYLOAD is stubbed while the loop,\n// reserve, barrier and push are kept.  Numerically WRONG on purpose: it is the\n// floor any read-granularity change can reach.\n",
)

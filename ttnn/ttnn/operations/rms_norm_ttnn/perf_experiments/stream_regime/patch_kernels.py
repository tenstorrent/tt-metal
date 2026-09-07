# Builds k_cand/ from the shipped kernels.  Re-runnable from the pristine source,
# so the experiment's device edits stay a readable diff.
import pathlib
import shutil
import sys

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE.parents[1] / "kernels"
DST = HERE / "k_cand"
DST.mkdir(exist_ok=True)
for f in SRC.iterdir():
    shutil.copy(f, DST / f.name)

nedits = [0]


class F:
    def __init__(self, p):
        self.p = p
        self.s = p.read_text()

    def sub(self, old, new, count=1):
        assert self.s.count(old) == count, f"{self.p.name}: expected {count} of:\n{old[:300]}\ngot {self.s.count(old)}"
        self.s = self.s.replace(old, new)
        nedits[0] += 1

    def save(self):
        self.p.write_text(self.s)


# ===========================================================================
# READER
# ===========================================================================
r = F(DST / "rms_norm_ttnn_reader.cpp")

r.sub(
    "constexpr uint32_t cb_bias_tiles = 23;",
    """constexpr uint32_t cb_bias_tiles = 23;
// ---- perf_experiments/stream_regime: the COMPACT per-channel caches ---------
// Reader-private scratch: this kernel fills them once at boot and copies out of
// them per chunk, so they are never pushed and never popped (get_write_ptr on a
// CB that was never pushed IS its base).  See `reader_pc_compact_boot` below.
constexpr uint32_t cb_gamma_compact = 24;
constexpr uint32_t cb_bias_compact = 25;""",
)

# ---- CT args --------------------------------------------------------------
r.sub(
    "    constexpr uint32_t WT_PAD = get_compile_time_arg_val(29);",
    """    constexpr uint32_t WT_PAD = get_compile_time_arg_val(29);
    // ---- stream_regime: THE COMPACT PER-CHANNEL HOLD -----------------------
    // PC_COMPACT: cache the two face-rows D23's TRIM == 2 already fetches -- the
    // WHOLE of a (1,1,1,W) operand's information, 1/16 of its tiled bytes -- for
    // the core's entire row, once, and re-materialize each chunk's tiles by a
    // LOCAL L1 copy.  PC_HOLD_WT is the cache's width in tiles.  PC_CHUNK says the
    // per-channel TILE ring is a WT_CHUNK window popped per chunk rather than a
    // held whole row (it was implicitly `!X_RESIDENT`; the cache decouples them).
    constexpr uint32_t PC_COMPACT = get_compile_time_arg_val(30);
    constexpr uint32_t PC_HOLD_WT = get_compile_time_arg_val(31);
    constexpr uint32_t PC_CHUNK = get_compile_time_arg_val(32);""",
)
r.sub("constexpr auto x_args = TensorAccessorArgs<30>();", "constexpr auto x_args = TensorAccessorArgs<33>();")

r.sub(
    "    constexpr bool PC_NARROW = (NARROW_PC_STAGE != 0);",
    """    constexpr bool PC_NARROW = (NARROW_PC_STAGE != 0);
    // 0 off / 1 EAGER (fill the whole cache at boot) / 2 LAZY (fill chunk c the
    // first time chunk c is staged).  See the fill below for the measurement.
    constexpr bool PC_COMPACT_HOLD = (PC_COMPACT != 0);
    constexpr bool PC_LAZY = (PC_COMPACT == 2);
    constexpr bool PC_CHUNKED = (PC_CHUNK != 0);
    static_assert(!PC_COMPACT_HOLD || !PC_NARROW, "rms_norm_ttnn: the compact hold is the TILE form, not D30's");
    // The cache stores exactly what TRIM == 2 fetches, so the descriptor may only
    // turn it on where BOTH operands admit the face-row granularity.
    static_assert(!PC_COMPACT_HOLD || HAS_GAMMA == 0 || GAMMA_TRIM == 2, "rms_norm_ttnn: compact hold needs TRIM 2");
    static_assert(!PC_COMPACT_HOLD || HAS_BIAS == 0 || BIAS_TRIM == 2, "rms_norm_ttnn: compact hold needs TRIM 2");""",
)

# ---- the compact boot + the local re-materialize ---------------------------
r.sub(
    "    // ---- the per-channel operands: one chunk's worth of tiles (or sticks) ----",
    """    // ---- stream_regime: THE COMPACT PER-CHANNEL CACHE ----------------------
    //
    // WHAT.  For each per-channel operand, ONE pass over the core's whole width
    // reading the same two face-rows D23's TRIM == 2 already reads, packed at a
    // 2 * TILE_DIM * elem stride into a reader-private CB.  Then every later
    // `stage_per_channel` is an L1 -> L1 copy instead of a DRAM read.
    //
    // WHY (measured, blackhole p150b, (1,1,8192,7168) bf16 gamma+bias+residual
    // fp32_dest_acc_en=True, the op's perf case #15).  The shipped STREAM reader
    // re-stages both operands per pass-B chunk of EVERY row-block -- per core
    // 2.33 blocks x 4 chunks x 56 tiles x 2 reads x 2 operands = 2088 tiny DRAM
    // reads, profiled at 115,916 ns (gamma) + 104,584 ns (bias) = 220,500 ns of a
    // 892,618 ns reader span.  Deleting the read entirely (RMS_ABLATE_PER_CHANNEL)
    // took the wall from 1,580,377 to 1,350,072 ns, so 230,305 ns is the whole
    // prize and the cache collects it for 2 * 224 * 128 = 57 kB of L1.
    //
    // The SECOND, larger consequence is a HOST one: with the hold priced at the
    // compact size instead of the tiled one, ROW_RESIDENT fits where it did not,
    // and the regime change deletes pass B's re-read of x AND of the residual.
    //
    // RAW-API JUSTIFICATION.  The fill is TensorAccessor + noc_async_read at a
    // face offset -- byte-for-byte the reads `stage_per_channel_chunk`'s TRIM == 2
    // branch already issues, just landing compacted; no dataflow helper covers a
    // sub-page tiled read (see this file's helper-usage note).  The expand is a
    // plain L1 word copy: `noc_async_read` to one's own L1 would put a NoC
    // transaction (and its barrier) on a 64-byte move that the RISC-V does in ~20
    // cycles, and `l1_helpers.hpp` offers zero_tile / prepare_zero_tile only --
    // nothing that scatters a compact vector into two face-row slots of a tile.
    // Gap kind: ERGONOMICS (the mechanism is ordinary L1 addressing; what is
    // missing is a named helper for "vector -> row 0 of a tile-row").
    constexpr uint32_t PC_ROW_BYTES_G = TILE_DIM * GAMMA_ELEM_BYTES;
    constexpr uint32_t PC_ROW_BYTES_B = TILE_DIM * BIAS_ELEM_BYTES;

    // Copy `n` bytes L1 -> L1.  Both ends are 32-byte-aligned CB addresses and n is
    // a multiple of 4 for every non-block-float dtype, so a word loop is exact.
    auto l1_copy = [](uint32_t dst, uint32_t src, uint32_t n) {
        volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
        volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
        for (uint32_t i = 0; i < n / 4; ++i) {
            d[i] = s[i];
        }
    };

    // Fill tiles [w0, w0 + n) of one operand's cache.
    auto pc_compact_fill = [&](uint32_t cb, uint32_t row_bytes, uint32_t tile_bytes, const auto& acc, uint32_t w0,
                               uint32_t n) {
        const uint32_t base = get_write_ptr(cb);
        const uint32_t face = tile_bytes / 4;
        for (uint32_t w = w0; w < w0 + n; ++w) {
            const uint32_t wt = w_start + w;
            const uint32_t tile_id = (wt < WT) ? wt : (WT - 1);
            const uint64_t src = get_noc_addr(tile_id, acc);
            const uint32_t dst = base + w * 2 * row_bytes;
            noc_async_read(src, dst, row_bytes);
            noc_async_read(src + face, dst + row_bytes, row_bytes);
        }
        noc_async_read_barrier();
    };

    auto pc_fill_chunk = [&](uint32_t c) {
        const uint32_t w0 = c * WT_CHUNK;
        if constexpr (HAS_G) {
            const auto g_acc = TensorAccessor(gamma_args, gamma_addr);
            pc_compact_fill(cb_gamma_compact, PC_ROW_BYTES_G, get_tile_size(cb_gamma_tiles), g_acc, w0, WT_CHUNK);
        }
        if constexpr (HAS_B) {
            const auto b_acc = TensorAccessor(bias_args, bias_addr);
            pc_compact_fill(cb_bias_compact, PC_ROW_BYTES_B, get_tile_size(cb_bias_tiles), b_acc, w0, WT_CHUNK);
        }
    };

    // LAZY vs EAGER, and it is worth 1.16x on the target case (1,099,830 ->
    // 949,000 ns) for a one-line change of WHEN.  Eager fills the whole cache at
    // boot: every core issues 2 x PC_HOLD_WT tiny DRAM reads into the same few
    // hundred kB of the weight tensor at t = 0, and the profiled zone comes back
    // at 290,385 ns per core MEAN but 653,014 ns MAX -- pure arbitration spread,
    // and it sits in front of the first x read so nothing overlaps it.  Lazy
    // fetches chunk c the first time chunk c is staged, which for the resident
    // regimes is inside the block-0 per-channel loop -- exactly where this thread
    // is otherwise BLOCKED on the compute kernel's pass-B pops.  Same bytes, same
    // transactions, issued into a hole instead of a burst.
    uint32_t pc_filled = 0;
    if constexpr (PC_COMPACT_HOLD && !PC_LAZY) {
        MaybeDeviceZoneScope("reader_pc_compact_boot");
        for (uint32_t c = 0; c < NUM_W_CHUNKS; ++c) {
            pc_fill_chunk(c);
        }
        pc_filled = NUM_W_CHUNKS;
    }

    // Re-materialize chunk `c` of one operand out of its cache: WT_CHUNK tiles,
    // each getting its two face-rows written where TRIM == 2 would have put them.
    // Everything else in the page keeps whatever was there -- the same invariant
    // the trimmed DRAM read already relies on (row 0 is the only row the
    // BroadcastDim::Row consumer reads; measured BIT-IDENTICAL with rows 1..31
    // seeded 1e5x wrong).
    auto pc_compact_expand = [&](uint32_t cb_tiles, uint32_t cb_compact, uint32_t row_bytes, uint32_t c) {
        const uint32_t tile_bytes = get_tile_size(cb_tiles);
        const uint32_t face = tile_bytes / 4;
        const uint32_t src_base = get_write_ptr(cb_compact) + c * WT_CHUNK * 2 * row_bytes;
        cb_reserve_back(cb_tiles, WT_CHUNK);
        uint32_t dst = get_write_ptr(cb_tiles);
        for (uint32_t w = 0; w < WT_CHUNK; ++w) {
            const uint32_t src = src_base + w * 2 * row_bytes;
            l1_copy(dst, src, row_bytes);
            l1_copy(dst + face, src + row_bytes, row_bytes);
            dst += tile_bytes;
        }
        cb_push_back(cb_tiles, WT_CHUNK);
    };

    // ---- the per-channel operands: one chunk's worth of tiles (or sticks) ----""",
)

r.sub(
    """    auto stage_per_channel = [&](uint32_t c) {
        const uint32_t first_wt = w_start + c * WT_CHUNK;
        if constexpr (HAS_G) {""",
    """    auto stage_per_channel = [&](uint32_t c) {
        const uint32_t first_wt = w_start + c * WT_CHUNK;
        if constexpr (PC_COMPACT_HOLD) {
            if constexpr (PC_LAZY) {
                // Chunks are always visited in order, so ONE watermark is the whole
                // bookkeeping: the first block pays the DRAM read, every later one
                // (and every later chunk of this one) finds it already in L1.
                if (c >= pc_filled) {
                    MaybeDeviceZoneScope("reader_pc_compact_boot");
                    pc_fill_chunk(c);
                    pc_filled = c + 1;
                }
            }
            // No NoC read at all: the bytes are already in this core's L1.
            if constexpr (HAS_G) {
                MaybeDeviceZoneScope("reader_read_gamma");
                pc_compact_expand(cb_gamma_tiles, cb_gamma_compact, PC_ROW_BYTES_G, c);
            }
            if constexpr (HAS_B) {
                MaybeDeviceZoneScope("reader_read_bias");
                pc_compact_expand(cb_bias_tiles, cb_bias_compact, PC_ROW_BYTES_B, c);
            }
            return;
        }
        if constexpr (HAS_G) {""",
)

# ---- the loop nest --------------------------------------------------------
r.sub(
    """    if constexpr (X_RESIDENT) {
        for (uint32_t c = 0; c < NUM_W_CHUNKS; ++c) {
            stage_per_channel(c);
        }
    }""",
    """    // stream_regime: `PC_CHUNKED` is what used to be `!X_RESIDENT`.  A resident
    // build whose per-channel ring is a CHUNK stages inside the row-block loop
    // below instead, because the ring only has room for one chunk at a time.
    if constexpr (X_RESIDENT && !PC_CHUNKED) {
        for (uint32_t c = 0; c < NUM_W_CHUNKS; ++c) {
            stage_per_channel(c);
        }
    }""",
)

r.sub(
    """                stage_activations_chunk(r0, rows, c);
                // STREAM: the per-channel operands are chunked and re-read for
                // every pass-B chunk.
                if constexpr (!X_RESIDENT) {
                    if (pass == 1) {
                        stage_per_channel(c);
                    }
                }
            }
        }
    }
}""",
    """                stage_activations_chunk(r0, rows, c);
                // STREAM: the per-channel operands are chunked and re-staged for
                // every pass-B chunk.
                if constexpr (!X_RESIDENT) {
                    if (pass == 1) {
                        stage_per_channel(c);
                    }
                }
            }
        }
        // stream_regime, ROW_RESIDENT with a CHUNKED per-channel ring: the chunks
        // are pushed in a SECOND loop, after every activation chunk of this block
        // is on its way.
        //
        // THE ORDER IS THE DEADLOCK ARGUMENT, not a preference.  The ring holds ONE
        // chunk, so a push of chunk c blocks until the compute kernel pops chunk
        // c-1 -- which happens in PASS B.  Interleaving the two loops would stall
        // this thread inside pass A's feed (compute waiting on activation chunk
        // c+1, this kernel waiting on a pass-B pop) and hang.  Pushing all of pass
        // A's activations FIRST means the only thing left to wait for is a pass-B
        // pop, which pass A's completion always delivers.  Compute needs no
        // per-channel tile before pass B, so nothing is starved by the order.
        if constexpr (X_RESIDENT && PC_CHUNKED) {
            for (uint32_t c = 0; c < NUM_W_CHUNKS; ++c) {
                stage_per_channel(c);
            }
        }
    }
}""",
)
r.save()

# ===========================================================================
# COMPUTE
# ===========================================================================
c = F(DST / "rms_norm_ttnn_compute.cpp")

c.sub(
    "    constexpr uint32_t RES_FUSE_CT = get_compile_time_arg_val(25);",
    """    constexpr uint32_t RES_FUSE_CT = get_compile_time_arg_val(25);
    // stream_regime: the per-channel TILE CB is a CHUNKED window (WT_CHUNK pages,
    // popped after every chunk) rather than a whole row held for the core's life.
    // This WAS `!X_RESIDENT`; the reader's compact per-channel cache decouples the
    // two, so a ROW_RESIDENT build can hold x while chunking gamma/bias.
    constexpr uint32_t PC_CHUNK_CT = get_compile_time_arg_val(26);""",
)

c.sub(
    "    constexpr bool ROW_RESIDENT = X_RESIDENT && (NUM_W_CHUNKS > 1);",
    """    constexpr bool ROW_RESIDENT = X_RESIDENT && (NUM_W_CHUNKS > 1);
    constexpr bool PC_CHUNKED = (PC_CHUNK_CT != 0);
    static_assert(PC_CHUNKED || X_RESIDENT, "rms_norm_ttnn: a streamed per-channel ring is chunked by definition");
    // The compact cache is a TILE-operand mechanism; a ROW_MAJOR operand is staged
    // through cb_*_sticks and tilized, which the resident boot does once per core.
    static_assert(
        !PC_CHUNKED || !X_RESIDENT || !PC_RM, "rms_norm_ttnn: a chunked resident per-channel ring is TILE-only");""",
)

c.sub(
    """            if constexpr (HAS_G && !X_RESIDENT) {
                cb_pop_front(cb_gamma_tiles, WT_CHUNK);
            }
            if constexpr (HAS_B && !X_RESIDENT) {
                cb_pop_front(cb_bias_tiles, WT_CHUNK);
            }""",
    """            if constexpr (HAS_G && PC_CHUNKED) {
                cb_pop_front(cb_gamma_tiles, WT_CHUNK);
            }
            if constexpr (HAS_B && PC_CHUNKED) {
                cb_pop_front(cb_bias_tiles, WT_CHUNK);
            }""",
)
c.sub(
    """    if constexpr (HAS_G && X_RESIDENT) {
        cb_pop_front(cb_gamma_tiles, X_HOLD_WT);
    }
    if constexpr (HAS_B && X_RESIDENT) {
        cb_pop_front(cb_bias_tiles, X_HOLD_WT);
    }""",
    """    if constexpr (HAS_G && !PC_CHUNKED) {
        cb_pop_front(cb_gamma_tiles, X_HOLD_WT);
    }
    if constexpr (HAS_B && !PC_CHUNKED) {
        cb_pop_front(cb_bias_tiles, X_HOLD_WT);
    }""",
)

# the per-channel operands' TILE OFFSET: 0 when the ring is one chunk.
c.sub(
    "            const uint32_t hold_base = ROW_RESIDENT ? (c * WT_CHUNK) : 0;",
    """            const uint32_t hold_base = ROW_RESIDENT ? (c * WT_CHUNK) : 0;
            // stream_regime: gamma/bias index from 0 when their ring IS the chunk.
            const uint32_t pc_base = PC_CHUNKED ? 0u : hold_base;""",
)
c.sub(
    "ckl::input(G_IN, ckl::BroadcastDim::Row)>{0u, hold_base},",
    "ckl::input(G_IN, ckl::BroadcastDim::Row)>{0u, pc_base},",
    2,
)
c.sub(
    "ckl::input(B_IN, ckl::BroadcastDim::Row)>{0u, hold_base},",
    "ckl::input(B_IN, ckl::BroadcastDim::Row)>{0u, pc_base},",
)
c.save()

print(f"k_cand written, {nedits[0]} edits", file=sys.stderr)

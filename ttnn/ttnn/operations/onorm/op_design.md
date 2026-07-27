# Operation Design: onorm

Kimi-Linear KDA **s6** tail: gated RMSNorm + head-flatten, fused on-chip.

    out = flatten_heads( RMSNorm_over_V(o) * weight ) * sigmoid(gate)

---

## 1. Blocking Model  (decided FIRST — everything below is downstream of it)

### 1.1 Axes of this op

`o` is `[B, T, HV, V]` head-major (TILE layout tiles the last two dims → each
`(b,t)` is a `[HV=32, V=128]` image = **1 x 4 tiles**).
`gate` / `out` are `[B, T, HV*V]` flat token-major (TILE tiles `(T, FLAT)` → per
batch a `ceil(T/32) x 128` tile grid).

| Axis | Character | Block-factor knob | Phase-1 value | Core-assignment | Later unlock |
|------|-----------|-------------------|---------------|-----------------|--------------|
| **B** (batch) | **independent** — no term of the math crosses batches | folded into `TOKEN_BLOCK` (below); no separate knob | — | flattened with T into the `(b, token-block)` work index and spread over the grid | knob-turn |
| **T** (tokens) | **independent above a 32-token floor** — each token normalizes and gates on its own, but the head→flat re-tile fuses exactly 32 tokens into one output tile-row, so 32 tokens is the atomic unit | `TOKENS_PER_BLOCK` | **32** (= one output tile-row; the re-tile granularity floor) | `split_work_to_cores(grid, B*ceil(T/32), row_wise=True)` — **blocks spread across the whole grid in phase 1** | knob-turn (raise `TOKENS_PER_BLOCK` to 64/96…) |
| **T** within a core's block (normalize sub-loop) | independent | `NORM_CHUNK_TOKENS` | **8** (coarsest that fits the L1 budget in §6.2 — *not* 1) | single core (the core owns the whole block) | knob-turn |
| **HV** (value-heads, tiled row axis of `o`) | **dependent — re-tile-coupled**: all 32 heads of a token collapse into that token's single flat feature row, so an output tile draws one *row* from each of 32 input tiles | `V_TILES`-side; heads are never sub-blocked | whole `HV=32` = exactly one tile height | single core — never split | **scheme-change** (all-to-all cross-core row exchange) |
| **V** (head_dim, RMSNorm reduction axis) | **dependent** — `mean(o²)` spans V; within a core it is a cheap sequential/DEST accumulate over `V_TILES` tiles | `V_TILES = ceil(V/32)` | **4** (whole reduction resident — no cross-call accumulate needed) | single core — never split | **scheme-change** (cross-core partial-sum combine) |
| **FLAT** (`HV*V`, flat feature axis of `gate`/`out`) | **dependent — re-tile-coupled** (it *is* HV x V re-indexed): one flat output tile-row needs the full 4096-wide row-major stripe, because the tilize address generator's row stride **is** the block width | `FLAT_TILES = HV*V/32` | **128** (full width — forced, see §6.1) | single core | **scheme-change** (strided tilize / head-block untilize) |
| **weight** `[V]` | **reuse-shared** — the same 4 tiles feed every `(b,t,h)` on every core | `V_TILES` | 4 tiles, read once per core, held for the whole kernel | each core reads its own copy from DRAM (4 tiles = 8 KB/core; <1 % of traffic) | **scheme-change** (mcast from one core) |

### 1.2 Buffer-depth knobs (per streaming CB)

| Knob | Applies to | Phase-1 value | Why |
|------|------------|---------------|-----|
| `DM_BLOCK_TILES` | tiles per `noc_async_read`/`write` group (one barrier per group) | **4** | measured sweet spot 4–8; `block=1` is the 6.5 GB/s trap (`examples/double_buffer/report.md:31,36,96-102`) |
| `DM_DEPTH` | `cb_gate_tiles`, `cb_out_tiles` depth (in `DM_BLOCK_TILES` units) | **2** | depth-2 is sufficient in every measured cell (17.9 GB/s/core = the single-core NoC ceiling for 2 KB transactions); deeper is unmeasured (`double_buffer/report.md:39-42,119-120`) |
| `O_DEPTH` | `cb_o_tiles` depth (in `NORM_CHUNK_TOKENS`-chunk units) | **2** | reader fills chunk *i+1* while compute holds chunk *i* (the sum-of-squares pass must keep `o` resident for the later normalize pass) |
| `GATE_CHUNK_TILES` | output tiles per gate-chain invocation (phase C block factor) | **64** = `FLAT_TILES/2` | the gate chain is two *sequential* helpers, so its intermediate `cb_gate_sig` must hold a whole chunk (§8). 64 keeps that at 128 KB while costing only 4 helper calls per block; `FLAT_TILES` (one call) would cost 256 KB, `32` would cost 8 calls. Must divide `FLAT_TILES`. |

### 1.3 The scheme phase-1 commits to

Work unit = one **token-block** = 32 tokens = **128 `o` tiles in, 128 `gate` tiles
in, 128 `out` tiles out** (all three are 128 *consecutive* tile ids — see §4).
Phase 1 spreads `B*ceil(T/32)` such blocks across the whole compute grid with
`row_wise=True`. There is **no single-core phase**. Every knob above
(`TOKENS_PER_BLOCK`, `NORM_CHUNK_TOKENS`, `V_TILES`, `FLAT_TILES`,
`DM_BLOCK_TILES`, `DM_DEPTH`, `O_DEPTH`, the grid) is a **named compile-time or
runtime parameter with exactly one source of truth** on the host; every CB page
count and loop bound in §5/§7 is *derived* from those parameters — none is
restated as a second literal.

### 1.4 Bandwidth ranking of the candidate splits (qualitative, no ns)

| Candidate split | Bytes moved per unit of output | Verdict |
|---|---|---|
| **`(b, token-block)`** — the chosen one | exactly `2 x 128 + 128` tiles read/written per 128 output tiles, plus 4 weight tiles once per core. **Zero re-read, zero combine, fully contiguous 128-tile runs on all three streams.** | **primary split** |
| `HV`/`FLAT` (head or flat-column groups across cores) | the tilize row stride is the block width, so a core producing a column slice still needs *every* `o` tile of the block ⇒ **4x–32x read amplification on `o`**; and pack_untilize cannot emit a row subset, so it is not even expressible | rejected — more DRAM bytes for more cores is a net loss on a DRAM-bound op |
| `V` (split the reduction across cores) | V is only 4 tiles; a cross-core partial-sum combine would move more semaphore/NoC traffic than the 8 KB it parallelizes | rejected (lamp) |
| `weight` mcast instead of per-core re-read | saves `8 KB x num_cores`; total traffic is ~15.7 MB at the T=640 profiling shape ⇒ ~1 % | not worth it in phase 1 (lamp) |

**Grid-fill sanity.** `B*ceil(T/32)` is 1 / 2 / 4 / 20 / 4 blocks for the five
`INPUTS` shapes, so on a 56–64-core grid the *block count*, not the grid, is the
limit. That is acceptable and deliberate: the op is DRAM-bandwidth-bound
(≈1 flop/byte; 768 KB moved per 128 output tiles), and per-core NoC ceiling is
~17.9 GB/s measured against a ~190 GB/s device DRAM peak
(`double_buffer/report.md:81-88`), so ~10–12 cores already saturate DRAM. Adding
cores by re-reading `o` (the only way to go finer than 32 tokens without an
all-to-all) would *increase* the bytes that actually bound the op. The catalog
states this outcome directly: "Once DRAM-bandwidth-bound, none of the levers
matter" (`examples/master.md:45,48-49`).

### 1.5 Lamp — scheme-changes phase-1 deliberately leaves reachable

| Lamp | What it unlocks | Why phase-1 does not foreclose it |
|---|---|---|
| **Cross-core re-tile (all-to-all row-major exchange)** | parallelism finer than 32 tokens: each core untilizes its own tokens and NoC-writes 4096-byte row-major slices into the owning core's `cb_rm_flat_rows`, so the 32 rows of an output tile-row can come from 32 different cores | `cb_rm_flat_rows` is already a *plain row-major L1 stripe addressed by token row*; a remote writer filling row `t` at byte offset `t*FLAT*2` is the same contract the local untilize already honours. Needs `mcast_pipe.hpp` (`ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp`) sender/receiver + one semaphore per block. |
| **`weight` mcast** | removes the per-core 4-tile re-read | `cb_weight` is already a standalone reader-produced, compute-held CB with no other coupling; swapping its producer for `SenderPipe::send()` / `ReceiverPipe::receive()` (`mcast_pipe.hpp`) is a reader-kernel change only. |
| **Cross-core V combine** | would let the reduction be split if `V` ever grows past L1 | the reduce is already expressed through `reduce_mean(...)`, whose `Accumulate` / `AccumulateViaAdd` path (`reduce_helpers_compute.hpp:307-348`) is the cross-call accumulator the cross-core version needs. |
| **Physical sharding** (`HEIGHT_SHARDED` `o`/`gate`/`out`) | a caller that pre-places the token-block in L1 | the work-split *is* a logical height-shard of the `(b, token)` axis: pinning the block geometry + core-assignment to a caller-supplied shard spec and swapping `cb_o_tiles`/`cb_gate_tiles`/`cb_out_tiles` for `ttnn.cb_descriptor_from_sharded_tensor` (zero-copy, **no NoC read**) is a placement change, not a new algorithm. Not in TARGET, so not built. |
| **Per-NoC gate/out rebalance** | if a *low*-core-count run turns out per-core-NoC-bound rather than DRAM-bound | a second `cb_gate_tiles_hi` CB owned by the writer + a `gate_tiles_from_reader` split constant. **Deliberately not phase-1**: reads issued on NoC1 measured **4.8x slower** than on NoC0 (`examples/noc_placement/README.md:60-70`), so all reads stay on the reader. |
| **`RECONFIG_MODE` = off** | up to 1.19x on the compute phases (`compute_block_size/README.md:128-143`) | **every CB in §5 is `Float16_b`**, so the dtype never changes anywhere in the kernel and flipping each helper's reconfig template arg to `NoReconfigure` / `NONE` / `None` is a legal one-line knob-turn. Phase 1 keeps reconfig **on** (safe) because the op is DRAM-bound, where the catalog says the lever does not pay. |

---

## 2. Overview

| Field | Value |
|-------|-------|
| Classification | fused (reduction + eltwise + in-kernel re-tile) |
| Goal | Fuse KDA's `rms_norm -> reshape -> multiply(sigmoid(gate))` tail into one on-chip kernel so `o_norm` and `sigmoid(gate)` never round-trip through DRAM. Only `o`, `gate`, `weight` are read and the flat output written. |
| Math | `n[b,t,h,:] = o[b,t,h,:] * rsqrt(mean(o[b,t,h,:]^2) + eps) * weight[:]`; `out[b,t,h*V+c] = n[b,t,h,c] * sigmoid(gate[b,t,h*V+c])` |
| Mode | Derivative (fused replacement for `ttnn.rms_norm` + `ttnn.reshape` + `ttnn.multiply(ttnn.sigmoid(...))`) |
| Shape-changer | **yes** — input `o` is `[B,T,HV,V]`, output is `[B,T,HV*V]` |
| References | `models/experimental/kimi_delta_attention/tt/ttnn_kda.py` (s6 block); `eval/golden_tests/onorm/feature_spec.py`; `.claude/references/generic_op_template/`; `ttnn/ttnn/operations/examples/master.md` |

### 2.1 Fixed geometry (TARGET pins these — they are constants, not axes)

| Symbol | Value | Meaning |
|--------|-------|---------|
| `HV` | 32 | value-heads (TP=1) — exactly one tile height |
| `V` | 128 | head_dim = RMSNorm reduction width |
| `V_TILES` | `ceil(V/32)` = 4 | column tiles per head-major image |
| `FLAT` | `HV*V` = 4096 | flat feature width |
| `FLAT_TILES` | `ceil(FLAT/32)` = 128 | column tiles per flat tile-row |
| `TOKENS_PER_BLOCK` | 32 | tokens per output tile-row (= `TILE_HEIGHT`) |
| `Tt` | `ceil(T/32)` | token tile-rows per batch |

All tile counts use `ceil` and are per-image, even though the contract makes
every one of them exact (`HV=32`, `V=128`, `T % 32 == 0`, `FLAT=4096`).

## 3. Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `o` | `ttnn.Tensor` | yes | `[B,T,HV,V]`, bf16, TILE, interleaved | — | tensor |
| `gate` | `ttnn.Tensor` | yes | `[B,T,HV*V]`, bf16, TILE, interleaved, **pre-sigmoid** | — | tensor |
| `weight` | `ttnn.Tensor` | yes | `[1,1,1,V]`, bf16, TILE, interleaved | — | tensor |
| `epsilon` | `float` | no | `> 0` | `1e-5` | **RT** to compute (fp32 bit pattern via `struct.unpack("I", struct.pack("f", eps))[0]`) |
| `compute_kernel_config` | `ttnn.DeviceComputeKernelConfig` | no | — | `default_compute_kernel_config()` | host-only → `ComputeConfigDescriptor` |
| `TOKENS_PER_BLOCK` | knob (CT) | — | multiple of 32 | 32 | CT to all three kernels |
| `NORM_CHUNK_TOKENS` | knob (CT) | — | divides `TOKENS_PER_BLOCK` | 8 | CT to compute |
| `GATE_CHUNK_TILES` | knob (CT) | — | divides `FLAT_TILES` | 64 | CT to compute |
| `DM_BLOCK_TILES` | knob (CT) | — | 1..8 | 4 | CT to reader/writer |
| `DM_DEPTH`, `O_DEPTH` | knob (host) | — | >= 2 | 2 | host CB sizing only |

`default_compute_kernel_config()` is a **single exported factory** in
`ttnn/ttnn/operations/onorm/onorm.py`; the entry point resolves `None` through it
and nowhere else. It must set all four fields:

| Field | Value | Why this exact value |
|-------|-------|----------------------|
| `fp32_dest_acc_en` | **`True`** | the sum-of-squares accumulates in fp32 DEST (P1) — the precision-sensitive step, per contract |
| `math_fidelity` | `HiFi4` | bf16 x bf16 mantissas fully retained on every FPU multiply; the op is DRAM-bound so the extra FPU passes are free |
| `math_approx_mode` | `False` | exact `rsqrt` / `sigmoid` |
| `dst_full_sync_en` | **`False`** | **load-bearing for performance, not a precision choice.** `can_use_fast_tilize()` tests `!get_dst_full_sync_enabled()` (`tilize_helpers.inl:66-79`), so enabling full sync silently drops P7a's 128-tile re-tile onto the slow tilize path on **every core, every block**. See risk 14. |

Note the interaction: `fp32_dest_acc_en=True` with half-sync gives
`DEST_AUTO_LIMIT = 4` (`dest_helpers.hpp:95-99`), which is exactly what
`untilize<V_TILES=4>` needs for the single-pass `pack_untilize` path
(`untilize_helpers.inl:78-87`).

**Config type contract (pinned — do not guess).** The public parameter is a
`ttnn.DeviceComputeKernelConfig`, per the required signature, so
`default_compute_kernel_config()` returns a **`ttnn.WormholeComputeKernelConfig`**
(the concrete `DeviceComputeKernelConfig` variant carrying the four fields above).
`ttnn.generic_op`'s kernel descriptor, however, takes a
`ttnn.ComputeConfigDescriptor`. The program descriptor therefore **translates
field-by-field** — `math_fidelity`, `fp32_dest_acc_en`, `math_approx_mode`,
`dst_full_sync_en` — into a fresh `ComputeConfigDescriptor` at exactly one place,
and the entry point resolves `None` through the factory before that translation.
The acceptance test exercises both ends of this path
(`test_onorm_compute_kernel_config` passes a caller-built
`WormholeComputeKernelConfig`; `test_onorm_default_compute_kernel_config_is_exported`
passes the factory's own output), so a design that accepts only one of the two
types fails the spec.

## 4. Tensors

### Input

| Property | `o` | `gate` | `weight` |
|----------|-----|--------|----------|
| Shape | `[B, T, HV, V]` | `[B, T, HV*V]` | `[1, 1, 1, V]` |
| Dtype | bfloat16 | bfloat16 | bfloat16 |
| Layout | TILE | TILE | TILE |
| Memory | interleaved DRAM | interleaved DRAM | interleaved DRAM |
| Tile grid | `(B*T)` images of `1 x V_TILES` | per batch `Tt x FLAT_TILES` | `1 x V_TILES` |
| Tile-id of block `(b, r)` | `((b*T) + r*TOKENS_PER_BLOCK) * V_TILES`, then **128 consecutive** | `(b*Tt + r) * FLAT_TILES`, then **128 consecutive** | `0 .. V_TILES-1` |
| Padding | none (`HV == 32`, `V == 4*32`) | none (`T % 32 == 0`, `FLAT == 128*32`) | rows 1..31 are tile padding, never read (only row 0 is broadcast) |

That all three per-block streams are **128 consecutive tile ids** is what makes
`DM_BLOCK_TILES`-sized read/write groups trivially expressible.

### Output

| Property | Value |
|----------|-------|
| Shape | `[B, T, HV*V]` (== `gate.shape`) |
| Dtype | bfloat16 |
| Layout | TILE |
| Memory | interleaved DRAM (`gate.memory_config()`) |
| Allocation | `ttnn.allocate_tensor_on_device(ttnn.Shape(out_shape), dtype, layout, device, memory_config)` — **positional args only**; passed **last** in `io_tensors` |

`io_tensors = [o, gate, weight, output]`.

## 5. Support contract (for the implementer's `SUPPORTED` block)

Axis names must match `eval/golden_tests/onorm/feature_spec.py`'s `TARGET`
exactly: **`dtype`** and **`layout`** — those are the only two. Phase-0
`SUPPORTED` already equals `TARGET`: `dtype: [ttnn.bfloat16]`,
`layout: [ttnn.TILE_LAYOUT]`. `INPUT_TAGGERS = {}` (no shape facets — fixed
`HV`/`V`, tile-aligned `T`). `EXCLUSIONS = []`. `validate()` is the first line of
`onorm()` and raises `UnsupportedAxisValue` / `ExcludedCell` from
`ttnn.operations._op_contract`; it must check **all three** input tensors' dtype
and layout against the axes. It must **not** declare or check `INVALID`.

`PROPERTIES`: `multi_core = True` (verified — the program descriptor's core-range
set comes from `split_work_to_cores`), `bounded_cb = True` (declared — §6.2 shows
the footprint is independent of `B` and `T`).

**Structural impossibilities**: none beyond what `feature_spec.py` already
records (`INVALID = []`) — TARGET is a single `(bf16, TILE)` cell.

## 6. Dataflow Strategy

```
DRAM  o[b, 32r..32r+31, :, :]  ──NoC0(reader)──▶ cb_o_tiles          (tiles, head-major)
DRAM  weight[1,1,1,V]          ──NoC0(reader)──▶ cb_weight           (tiles, row-0 valid; once per core)
                                   reader fills  cb_scaler           (bf16 reduce scaler = 1/V; once per core)

compute (per NORM_CHUNK_TOKENS tokens):
   cb_o_tiles ──DEST-accumulate o² over V_TILES──▶ cb_sumsq          (1 tile / token)
   cb_sumsq   ──reduce<SUM,REDUCE_ROW>(scaler=1/V)
              ──── + eps + rsqrt fused in the SAME DEST window ────▶ cb_rstd   (col-0 valid)
   cb_o_tiles x cb_rstd  (bcast Col) ──────────▶ cb_normed
   cb_normed  x cb_weight (bcast Row) ─────────▶ cb_onorm            ← head-major, normalized, scaled
   cb_onorm   ──pack_untilize<V_TILES>─────────▶ cb_rm_flat_rows     ← ROW-MAJOR: token t's 4096
                                                                       features, contiguous, at
                                                                       byte offset t*FLAT*2

compute (once per token-block):
   cb_rm_flat_rows ──tilize<FLAT_TILES>────────▶ cb_flat_tiles       ← flat token-major tiles

compute (per GATE_CHUNK_TILES output tiles):
   cb_gate_tiles ──sigmoid (SFPU, DEST)────────▶ cb_gate_sig         ← packed to L1 on purpose
   cb_flat_tiles x cb_gate_sig ──FPU mul_tiles─▶ cb_out_tiles        ← FPU, NOT SFPU (§9)

DRAM  gate[b, 32r..32r+31, :]  ──NoC0(reader)──▶ cb_gate_tiles
cb_out_tiles ──NoC1(writer)──▶ DRAM out[b, 32r..32r+31, :]
```

**All DRAM reads are on the reader (NCRISC/NoC0) and all writes on the writer
(BRISC/NoC1).** This is not stylistic: reads issued on NoC1 measured 4.8x slower
and writes on NoC0 4.3x slower than the default pairing
(`examples/noc_placement/README.md:60-70`). The writer therefore never reads
`gate`, even though that would balance per-core byte counts.

### 6.1 The in-kernel head-major → flat token-major re-tile (the whole point)

`o` arrives head-major; the output needs heads *interleaved into the flat column
axis*. This is a genuine re-tiling: flat output tile `j` of a block equals the
`[32 tokens, 32 channels]` block for head `h = j / V_TILES`, channel-tile
`k = j % V_TILES` — i.e. it gathers **one row from each of 32 different input
tiles**. It is done in-kernel via **untilize → row-major → tilize**, never with a
tile transpose (a transpose swaps head↔V *within* a token and does not produce
the flat layout), and never with `ttnn.reshape` / `to_layout` / `tilize` /
`untilize` in the Python entry point.

Why it works, exactly:

1. `pack_untilize<V_TILES>` on token `t`'s `[HV=32, V=128]` tile-row emits a
   contiguous row-major block of 32 rows x 128 elements, linear index
   `h*V + c` = **the flat feature index `f`**. So token `t`'s 4096 features land
   contiguously in `V_TILES` tile-sized pages = `FLAT*2 = 8192` bytes.
2. Stacking 32 such untilize outputs back-to-back in `cb_rm_flat_rows` *is* a
   row-major `[32 tokens, FLAT]` stripe with row stride `FLAT*2` bytes.
3. `tilize<FLAT_TILES>` over that stripe (`num_blocks = 1`) emits exactly the
   flat token-major tile-row, column tile `j` = columns `32j..32j+31`.

**`cb_rm_flat_rows` must be sized to EXACTLY `FLAT_TILES` pages** (one block's
worth). It is filled from and fully drained back to the buffer base every block,
so the 128 pages are contiguous with no ring wrap inside a block — which is
precisely what the tilize address generator assumes.

**Deviation from the stated strategy, with reason.** The task rules ask for
`tilize<..., StreamMode::PerTile>` with a 2-tile `cb_flat`. That is **not
implementable here and would deadlock**: `tilize` and its consumer (the
sigmoid·multiply chain) both run in the *same* compute kernel, so all three
TRISCs execute them in sequence. `tilize`'s PACK thread would block in
`cb_reserve_back(cb_flat, 1)` at the 3rd tile; the consumer's `cb_pop_front`
is only reached by UNPACK *after* UNPACK finishes all 128 tilize unpacks; and
UNPACK is throttled by MATH which is throttled by PACK → hang. This is the
general rule "**full block** for intermediate CBs between sequential helpers
(both own all TRISCs — can't pipeline)". `StreamMode::PerTile` only pays when the
consumer is a *different* RISC. Phase 1 therefore uses the default
`StreamMode::Atomic` (bit-identical output bytes) with `cb_flat_tiles` sized to
`FLAT_TILES`. Narrowing it is a **scheme-change**, not a knob-turn: the tilize
row stride *is* `block_width_tiles`, so a column-chunked tilize would need a
strided-input tilize *and* a row-subset-capable untilize — neither exists
(`tilize_helpers.hpp:96-104`, `untilize_helpers.hpp:109-110`). Recorded as a
lamp. The rule that *is* honoured, and that matters for boundedness: `o`, `gate`
and the output all stream in small `DM_BLOCK_TILES`-granular double buffers, and
`cb_rm_flat_rows` / `cb_flat_tiles` are `O(TOKENS_PER_BLOCK * FLAT)` — **constant
in `T` and `B`**.

### 6.2 L1 budget (the block-factor knobs' constraint)

Every page is a `Float16_b` tile = 2048 B. `NC = NORM_CHUNK_TOKENS`.

| CB | pages (formula) | pages @ phase-1 | bytes |
|----|-----------------|-----------------|-------|
| `cb_o_tiles` | `V_TILES * NC * O_DEPTH` | 64 | 131072 |
| `cb_gate_tiles` | `DM_BLOCK_TILES * DM_DEPTH` | 8 | 16384 |
| `cb_weight` | `V_TILES` | 4 | 8192 |
| `cb_scaler` | 1 | 1 | 2048 |
| `cb_sumsq` | `NC` | 8 | 16384 |
| `cb_rstd` | `NC` | 8 | 16384 |
| `cb_normed` | `V_TILES * NC` | 32 | 65536 |
| `cb_onorm` | `V_TILES * NC` | 32 | 65536 |
| `cb_rm_flat_rows` | `FLAT_TILES` | 128 | 262144 |
| `cb_flat_tiles` | `FLAT_TILES` | 128 | 262144 |
| `cb_gate_sig` | `GATE_CHUNK_TILES` | 64 | 131072 |
| `cb_out_tiles` | `DM_BLOCK_TILES * DM_DEPTH` | 8 | 16384 |
| **total** | | **485** | **993 280 B ≈ 970 KB** |

Against ~1.34 MB of CB-available L1 that is ~72 %, leaving ~370 KB headroom.
The host **must** assert this total against `device.l1_size_per_core()` minus the
unreserved base, and the error message must name the knobs to lower and their
order: `GATE_CHUNK_TILES` first (it is a pure perf/L1 trade), then
`NORM_CHUNK_TOKENS`. `cb_rm_flat_rows` and `cb_flat_tiles` are **not** reducible
(§6.1).
This is the sizing argument for `NORM_CHUNK_TOKENS = 8` being the *coarse*
default rather than a minimal one:

* `NC = 1` would cost 5 helper invocations per token instead of per 8 tokens.
  Measured fixed cost is ~320 ns per extra pass/phase
  (`compute_block_size/README.md:92-103`); at `NC = 4` vs `8` that is ~20 extra
  calls x 320 ns ≈ 6.4 µs per block against a ~43 µs DRAM-bound block time
  (~15 %). So 8 is chosen *up*, not down.
* `NC = 16` would add ~304 KB (total ~1.16 MB, 87 % of L1) for a further ~7 %,
  and would put the two fixed 256 KB re-tile buffers at OOM risk. **8 is the
  coarsest value that fits the budget**, which is exactly the rule.

## 7. Work Distribution  (the Blocking Model's core-assignment, made concrete)

| Field | Value |
|-------|-------|
| Work unit | one **token-block** = `TOKENS_PER_BLOCK` tokens of one batch = 128 `o` tiles → 128 `out` tiles |
| Total units | `num_token_blocks = B * ceil(T / TOKENS_PER_BLOCK)` — **`ceil`, per batch**; never `floor(B*T/32)` (each batch's token axis is tile-padded independently) |
| Grid | `device.compute_with_storage_grid_size()` |
| Split | `ttnn.split_work_to_cores(grid, num_token_blocks, row_wise=True)` → `(num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2)` |
| **`row_wise=True` is mandatory** | the default `row_wise=False` lays cores out as a *column* line, measured **2.91x slower** on a DRAM↔DRAM stream (959.2 µs → 330.1 µs, 8 cores) because a column shares its NoC links (`examples/noc_placement/README.md:8,37-46`) |
| Per-core work | iterate `ttnn.corerange_to_cores(group, None, True)` in order, accumulate a running `start_block`; each core gets RT args `(start_block, num_blocks)` with `num_blocks` = `units_per_core_group_1` or `..._2` |
| Remainder | handled entirely by the two core groups (`split_work_to_cores` guarantees `group_2` count differs by at most 1). No core ever receives 0 blocks; `num_cores <= grid.x*grid.y`. If `num_token_blocks < num_cores`, only `num_cores = num_token_blocks` cores are in `all_cores`, and CBs/kernels are created on `all_cores` only. |
| Per-block index math (kernel side) | `b = bi / Tt`, `r = bi % Tt`; `o` first tile = `(b*T + r*TOKENS_PER_BLOCK) * V_TILES`; `gate`/`out` first tile = `(b*Tt + r) * FLAT_TILES`; each stream then reads/writes 128 consecutive tiles in `DM_BLOCK_TILES` groups with one `noc_async_*_barrier` per group |
| Compute regimes | **exactly one.** There is no shape- or grid-dependent branch in the compute kernel; the two core groups differ only in the `num_blocks` runtime arg. No regime-pinned tests are required. |

## 8. Circular Buffers

Every CB is `Float16_b` (`ttnn.bfloat16`), page size `ttnn.tile_size(ttnn.bfloat16)` = 2048 B, `total_size = pages * page_size`. A uniform format is deliberate — it makes `RECONFIG_MODE` a legal knob-turn (§1.5).

| Semantic Name | Index | Page Size | Num Pages | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|--------|----------|----------|----------|
| `cb_o_tiles` | 0 | 2048 | `V_TILES * NORM_CHUNK_TOKENS * O_DEPTH` | Float16_b | reader | compute | per chunk (held across the sum-of-squares and normalize passes, popped by the normalize pass) |
| `cb_gate_tiles` | 1 | 2048 | `DM_BLOCK_TILES * DM_DEPTH` | Float16_b | reader | compute | streaming, phase C only |
| `cb_weight` | 2 | 2048 | `V_TILES` | Float16_b | reader | compute | **persistent for the whole kernel** — waited each chunk, never popped |
| `cb_scaler` | 8 | 2048 | 1 | Float16_b | reader | compute | persistent (filled once at kernel start with `1/V`, never popped) |
| `cb_out_tiles` | 16 | 2048 | `DM_BLOCK_TILES * DM_DEPTH` | Float16_b | compute | writer | streaming, phase C only |
| `cb_sumsq` | 24 | 2048 | `NORM_CHUNK_TOKENS` | Float16_b | compute | compute | per chunk |
| `cb_rstd` | 25 | 2048 | `NORM_CHUNK_TOKENS` | Float16_b | compute | compute | per chunk — holds `rsqrt(mean+eps)`, **col-0 valid** |
| `cb_normed` | 27 | 2048 | `V_TILES * NORM_CHUNK_TOKENS` | Float16_b | compute | compute | per chunk |
| `cb_onorm` | 28 | 2048 | `V_TILES * NORM_CHUNK_TOKENS` | Float16_b | compute | compute | per chunk (untilize input) |
| `cb_rm_flat_rows` | 29 | 2048 | `FLAT_TILES` | Float16_b (**row-major payload**) | compute | compute | per token-block — the re-tile working set; must be **exactly** `FLAT_TILES` pages (§6.1) |
| `cb_flat_tiles` | 30 | 2048 | `FLAT_TILES` | Float16_b | compute | compute | per token-block (sequential-helper intermediate ⇒ full block, §6.1) |
| `cb_gate_sig` | 31 | 2048 | `GATE_CHUNK_TILES` | Float16_b | compute | compute | per gate chunk — `sigmoid(gate)` deliberately materialized in L1 so the **FPU** unpacker feeds the multiply (§9, P7c) |

Rationale per class: streaming input/output CBs get `DM_BLOCK_TILES * DM_DEPTH`
(the double-buffer knob); intermediates between two *sequential* helpers get the
full block they must hold; `cb_scaler` is a single constant page; `cb_weight` is
the `V_TILES`-wide reuse-shared operand.

### 8.1 CB sync ledger (push count == wait count), per token-block

`NB = NORM_CHUNK_TOKENS`, `NCH = TOKENS_PER_BLOCK / NB` chunks.

| CB | pushes / block | waits & pops / block | Where |
|----|----------------|----------------------|-------|
| `cb_o_tiles` | reader: `NCH * V_TILES*NB` = 128 | P1 `HeldBulk` waits `V_TILES*NB`, pops 0; P4 `Bulk` waits `V_TILES*NB`, pops `V_TILES*NB` ⇒ 128 popped | P1/P4 |
| `cb_gate_tiles` | reader: `FLAT_TILES` = 128 | P7b `Streaming`: 128 waits, 128 pops | P7b |
| `cb_weight` | reader: `V_TILES` **once per core** | P5 `HeldBulk`: waits `V_TILES`, pops 0 (idempotent re-wait each chunk) | P5 |
| `cb_scaler` | reader: 1 **once per core** | reduce waits 1, pops 0 | P2 |
| `cb_sumsq` | P1: `NB` per chunk ⇒ `NCH*NB` = 32 | P2 `BulkWaitBulkPop` (Wt=1): 1 wait + 1 pop per row ⇒ 32 | P1/P2 |
| `cb_rstd` | P2: `NB` per chunk ⇒ 32 | P4 `Bulk`/`Col` (window `Ht=NB`): `NB` waits + `NB` pops per chunk ⇒ 32 | P2/P4 |
| `cb_normed` | P4: `V_TILES*NB` per chunk ⇒ 128 | P5 `Bulk`/`Block`: `V_TILES*NB` waits + pops per chunk ⇒ 128 | P4/P5 |
| `cb_onorm` | P5: 128 | P6 `untilize`, `WaitBlock`: `V_TILES` waits + pops per block x `NB` blocks x `NCH` ⇒ 128 | P5/P6 |
| `cb_rm_flat_rows` | P6: `V_TILES` per untilize block ⇒ 128 pages | P7a `tilize`, `WaitBlock`, `num_blocks=1`: waits `FLAT_TILES`=128, pops 128 | P6/P7a |
| `cb_flat_tiles` | P7a: 128 (Atomic: one reserve/push of `FLAT_TILES`) | P7c `Streaming`: `GATE_CHUNK_TILES` waits + pops per chunk x `FLAT_TILES/GATE_CHUNK_TILES` chunks ⇒ 128 | P7a/P7c |
| `cb_gate_sig` | P7b: `GATE_CHUNK_TILES` per chunk ⇒ 128 | P7c `Streaming`: 128 waits, 128 pops | P7b/P7c |
| `cb_out_tiles` | P7c: 128 | writer: 128 waits, 128 pops | P7c/writer |

Every row balances. Every CB has **exactly one** producer kernel and **one**
consumer kernel; `cb_flat_tiles` is *not* written in place by phase C precisely
because that would make the writer a second consumer.

## 9. API Mapping

Every mechanism is a `kernel_lib` helper. There are **no raw-API fallbacks** in
the compute kernel; the only raw APIs are the reader/writer `TensorAccessor` +
`noc_async_*` calls, for which no helper exists.

| Phase | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Requirements |
|-------|------|----------|-----------|------------------------|----------|-----------|--------------|
| boot | helper | `compute_kernel_hw_startup(icb0, icb1, ocb)` | `tilize_helpers.hpp:119-123`, `reduce_helpers_compute.hpp:30-35` | `(cb_o_tiles, cb_weight, cb_onorm)` | — | — | **exactly once**, first statement of `MAIN()`, never in a loop. One boot is enough for all phases because every CB shares `Float16_b`. |
| reader: scaler | helper | `dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>(inv_V)` | `reduce_helpers_dataflow.hpp:58-60` | pool-type-aware overload (**not** the legacy 1-arg form). `inv_V` = fp32 bits of `1/V`, a **CT arg** computed once on the host — the divisor stays explicit and caller-supplied, never derived from tile geometry | — | `cb_scaler` | bf16 CB; once per core, before any reduce. `V = 128` is tile-aligned ⇒ single scaler tile, **no** partial-scaler pair, `ReducePartialScaler::none()` |
| **P1** sum-of-squares | helper | `eltwise_chain(EltwiseShape::grid(NB, V_TILES), BinaryFpu<cb_o_tiles, cb_o_tiles, BinaryFpuOp::Mul, BroadcastDim::None, InputLifecycle::HeldBulk, InputLifecycle::HeldBulk, BinaryDataFormatReconfig::Input, Dst::D0, OperandKind::Block, OperandKind::Block, TileOffset::Unset, TileOffset::Unset, DestAccumulation::Enabled>{}, PackTile<cb_sumsq, OutputLifecycle::DestAccumulation, PackTileReconfig::Output, Dst::D0>{}))` | `eltwise_chain.hpp:710-711`, `eltwise_chain.inl:2657-2708` (DEST-accum walk), `:809-818` (accum requires `Dst::D0`) | **`NB` = `NORM_CHUNK_TOKENS` is the block knob**; `V_TILES` = reduction width | `cb_o_tiles` | `cb_sumsq` | one outer row per token: `D0` sticky across that row's `V_TILES` inputs, packed once, reset by the next row's acquire. `HeldBulk` keeps `o` for P4. `fp32_dest_acc_en=True` ⇒ the accumulate is fp32. |
| **P2** mean over V **+ `eps` + `rsqrt`, fused** | helper | `reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, cb_sumsq, cb_scaler, cb_rstd, ReduceInputPolicy::BulkWaitBulkPop, ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT, ReduceAlgorithm::Auto>(ReduceInputBlockShape::of(1, 1, NB), ReduceInputMemoryLayout::contiguous(), NoAccumulation{}, post_op)` | `reduce_helpers_compute.hpp:522-538`; `ReduceInputBlockShape::of` `:215-217`; `post_reduce_op` contract `:422-427`, worked example `:498-508` | **`NB` is the batch knob** (one call per chunk, not per token). `post_op` = `[eps_bits](uint32_t dst){ binop_with_scalar_tile_init(); add_unary_tile(dst, eps_bits); rsqrt_tile_init(); rsqrt_tile(dst); }` | `cb_sumsq`, `cb_scaler` | `cb_rstd` | The `1/V` scaler tile turns the SUM into the mean, so the divisor is explicit and host-supplied. `Wt = 1` ⇒ `ReduceAlgorithm::Auto` resolves to `ReduceTile`, the fastest datapath at 1–2 tiles (`master.md:133-137`). **The `+eps`/`rsqrt` epilogue runs inside the reduce's own DEST window** — this is the catalog's one *winning* fusion shape (reduce + SFPU post-op, **1.01–1.07x**, `master.md:88-89`) and it removes both a whole phase per chunk (~320 ns each) and an entire intermediate CB versus a separate `eltwise_chain` pass. `post_reduce_op` is called once per output with `dst_idx=0` (`:424`). Output valid region = **column 0**. |
| **P4** normalize | helper | `mul<cb_o_tiles, cb_rstd, cb_normed, BroadcastDim::Col, InputLifecycle::Bulk, InputLifecycle::Bulk, OutputLifecycle::Streaming, BinaryDataFormatReconfig::Input, PackTileReconfig::Output, OperandKind::Block, OperandKind::Col>(EltwiseShape::grid(NB, V_TILES))` | `eltwise_convenience.hpp:81-98`; `BroadcastDim` semantics `eltwise_chain.hpp:523-534`; `OperandKind` `:336-351` | `NB` x `V_TILES` grid; `rstd` indexed by **row** (`Col` kind = index by `ht`) | `cb_o_tiles`, `cb_rstd` | `cb_normed` | `cb_rstd` is a REDUCE_ROW result (col-0 valid) ⇒ **`BroadcastDim::Col`**. `Bulk` on `cb_o_tiles` performs the deferred pop of P1's held window. |
| **P5** weight scale | helper | `mul<cb_normed, cb_weight, cb_onorm, BroadcastDim::Row, InputLifecycle::Bulk, InputLifecycle::HeldBulk, OutputLifecycle::Streaming, BinaryDataFormatReconfig::Input, PackTileReconfig::Output, OperandKind::Block, OperandKind::Row>(EltwiseShape::grid(NB, V_TILES))` | `eltwise_convenience.hpp:81-98`; `is_legal_kind_lifecycle(Row, HeldBulk)` `eltwise_chain.hpp:359-382` | `weight` indexed by **column tile** (`Row` kind = index by `wt`) | `cb_normed`, `cb_weight` | `cb_onorm` | `weight` is `[1, V]`-shaped (row-0 valid) ⇒ **`BroadcastDim::Row`**; no pre-broadcast pass is needed. `Row` kind **requires** a non-draining lifecycle ⇒ `HeldBulk`. |
| **P6** head-major → row-major | helper | `untilize<V_TILES, cb_onorm, cb_rm_flat_rows, InitUninitMode::InitAndUninit, WaitMode::WaitBlock, ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(NB)` | `untilize_helpers.hpp:145-154` (decl), `:97-104` (block geometry) | **`V_TILES` is the block-width knob; `NB` the block count** | `cb_onorm` | `cb_rm_flat_rows` | each of the `NB` blocks emits one token's 32x128 contiguous row-major region = that token's flat feature row. Symmetric (tile-sized) pages both sides. |
| **P7a** row-major → flat tiles | helper | `tilize<FLAT_TILES, cb_rm_flat_rows, cb_flat_tiles, InitUninitMode::InitAndUninit, WaitMode::WaitBlock, ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure, Fp32Mode::Fast, RemapMode::Configure, StreamMode::Atomic>(1)` | `tilize_helpers.hpp:243-254` (decl), `:80-108` (`StreamMode`), `:164-170` (block geometry) | **`FLAT_TILES` is the block-width knob** (and *is* the row stride) | `cb_rm_flat_rows` | `cb_flat_tiles` | `num_blocks = TOKENS_PER_BLOCK/32 = 1`. `Atomic` (not `PerTile`) — see §6.1 for the deadlock proof. Symmetric pages ⇒ `total_input_pages` omitted. |
| **P7b** `sigmoid(gate)` | helper | `unary<Sigmoid<>, cb_gate_tiles, cb_gate_sig, InputLifecycle::Streaming, OutputLifecycle::Streaming, CopyTileReconfig::Input, PackTileReconfig::Output>(EltwiseShape::tiles(GATE_CHUNK_TILES))` | `eltwise_convenience.hpp:126-139`; `Sigmoid` `eltwise_activations.hpp:24` / `eltwise_activations.inl:31-34` | **`GATE_CHUNK_TILES` is the block knob**; called `FLAT_TILES/GATE_CHUNK_TILES` times per block | `cb_gate_tiles` | `cb_gate_sig` | **the op owns the sigmoid** (`gate` is pre-sigmoid), and normalization (P1–P5) happens strictly first. The result is deliberately **packed to L1** rather than kept in DEST — see P7c. |
| **P7c** gate multiply | helper | `mul<cb_flat_tiles, cb_gate_sig, cb_out_tiles, BroadcastDim::None, InputLifecycle::Streaming, InputLifecycle::Streaming, OutputLifecycle::Streaming, BinaryDataFormatReconfig::Input, PackTileReconfig::Output>(EltwiseShape::tiles(GATE_CHUNK_TILES))` | `eltwise_convenience.hpp:81-98` | same `GATE_CHUNK_TILES` block knob, paired 1:1 with P7b | `cb_flat_tiles`, `cb_gate_sig` | `cb_out_tiles` | **FPU `mul_tiles`, fed from L1 by the unpacker.** This is the highest-volume multiply in the op (every one of the `FLAT_TILES` output tiles of every block), so the engine choice dominates: an SFPU multiply is a measured **0.58x** loss (+~22 000 ns of math), and DEST-reuse into an FPU consumer is **0.82x** while the L1 round-trip is **1.22x faster** (`compute_fusion/README.md:79,90-97,125,134`; `master.md:88-96`). The catalog's verbatim prescription for exactly this SFPU-then-FPU shape: "pack the intermediate to L1 and let the unpacker feed it back… Never use the SFPU for what the FPU does." Uses `Dst::D0` only ⇒ safe under `fp32_dest_acc_en` (`DEST_AUTO_LIMIT` = 4). |
| reader / writer | raw_api | `TensorAccessor` + `noc_async_read_tile` / `noc_async_write_tile` + one `noc_async_{read,write}_barrier` per `DM_BLOCK_TILES` group | `tech_reports/tensor_accessor/tensor_accessor.md` | `TensorAccessorArgs` go **last** in the CT-arg list, one per tensor slot | — | — | **Helpers considered and rejected:** `mcast_pipe.hpp` (`ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp:1`) — it implements NoC *multicast + semaphore handshake*; phase 1 has no cross-core operand sharing (every stream is disjoint per core), so there is no multicast to perform. `tilize_helpers_dataflow.hpp` — dataflow-side tilize; both re-tile steps here must run on the compute engine because they sit between compute phases. No `kernel_lib` helper wraps plain interleaved-DRAM tile streaming. |

**Not used, and why (mandatory justifications):**

* `reduce_helpers_compute.hpp` `reduce<...>` directly over `cb_o_tiles`: cannot —
  the reduce datapath reduces its *input tiles*, and RMSNorm needs
  `sum(o²)`; no `reduce` overload squares its input
  (`reduce_helpers_compute.hpp:522-538` — the only pre/post hook is
  `post_reduce_op`, applied *after* the reduction, `:421-427`). P1 therefore
  materializes the squares, and does so with the *cheaper* DEST-accumulate shape
  rather than a `V_TILES`-wide `cb_sq` + wide reduce.
* `reduce_mean<...>` (`reduce_helpers_compute.hpp:576-590`) for P2: **rejected in
  favour of `reduce<PoolType::SUM>` + a `1/V` scaler tile.** `reduce_mean`'s
  parameter list has **no `post_reduce_op`** slot (compare `:585-590` with
  `reduce<>`'s `:533-538`), so it cannot host the `+eps`/`rsqrt` epilogue; using it
  forces a separate `eltwise_chain` pass (one extra helper invocation per chunk,
  ~320 ns each, plus a whole extra `NORM_CHUNK_TOKENS`-page intermediate CB) and
  gives up the reduce+SFPU-post-op fusion that the catalog measures as the one
  shape where fusing *wins* (1.01–1.07x, `master.md:88-89`). Nothing is lost on the
  explicit-divisor front: the `1/V` scaler is a host-computed CT arg, exactly as
  caller-supplied as `n_reduced` would be.
* `streaming_reduce_helpers.hpp` `transform_in_place` (`:95-111`) for the
  `+eps`/`rsqrt` step: it is documented as a "1-tile in-place transform on a
  **1-page** CB" (`streaming_reduce_helpers.hpp:26`), but `cb_rstd` holds
  `NORM_CHUNK_TOKENS` = 8 pages, so it would need 8 invocations per chunk and still
  could not fuse into the reduce's DEST window. Superseded by P2's `post_reduce_op`.
  (The rest of `streaming_reduce_helpers.hpp` is also unneeded — the whole
  reduction, `V_TILES` = 4 tiles, is resident, so there is no
  accumulate-across-calls case. Note its `accumulate_reduce*` wrappers are stale
  against the current `reduce<>` template signature; do not call them.)
* `binary_sfpu<MulBinary<...>>` / `MulBinary` in DEST
  (`eltwise_binary_sfpu.hpp`) for the gate multiply (P7c): **rejected — measured
  0.58x.** The multiply is a plain elementwise product of two L1 tensors, which is
  FPU work; running it on the SFPU costs ~22 000 ns of extra math per measured
  cell (`compute_fusion/README.md:79,97,134`) on *every* output tile of *every*
  block — the largest single compute volume in the op. Fusing
  `sigmoid`→multiply through DEST (`DestReuseBinary`) is also a loss for the same
  reason P4+P5 stay split (0.82x vs a 1.22x-faster L1 round-trip). Hence the
  deliberate `cb_gate_sig` L1 hop between P7b and P7c.
* `DestReuseBinary` to fuse P4+P5 into one chain: measured **loss**. Fusing
  through DEST into an **FPU** consumer is 0.82–1.02x, and the L1 round-trip is
  1.22x *faster*, with a per-tile premium of 464 ns @ n=8 → 2301 ns @ n=32
  (`examples/master.md:89-96`, `examples/compute_fusion/README.md:152-162`). Two
  chain calls with `cb_normed` is the catalog-backed choice.
* `sfpu_activation_helpers.hpp` / `apply_activation_from_pack()`: only fills the
  `Activation` slot of `matmul_block` / `add_bias_bcast_rows`
  (`sfpu_activation_helpers.hpp` header scope); this op has no matmul, and the
  gate is a *binary* multiply by another tensor, not a unary activation.
* `matmul_block_helpers.hpp`, `bias_add_helpers.hpp`,
  `reblock_untilize_helpers.hpp`: `reblock_and_untilize` gathers **matmul
  SubblockMajor** output (`reblock_untilize_helpers.hpp:1`); there is no matmul
  here and `cb_onorm` is already plain row-major tile order.

## 10. Compute Phases

Loop structure: `for block in [start_block, start_block + num_blocks)` →
`for chunk in [0, TOKENS_PER_BLOCK / NORM_CHUNK_TOKENS)` → P1..P6 → then P7a once
per block → `for gchunk in [0, FLAT_TILES / GATE_CHUNK_TILES)` → P7b/P7c.
`NB = NORM_CHUNK_TOKENS`, `GC = GATE_CHUNK_TILES`.

Helper invocations per token-block:
`NCH*5 + 1 + 2*(FLAT_TILES/GC)` = `4*5 + 1 + 2*2` = **25**. Keeping this count low
is why `NORM_CHUNK_TOKENS` and `GATE_CHUNK_TILES` default **coarse** (§6.2) — the
measured fixed cost is ~320 ns per extra invocation
(`compute_block_size/README.md:92-103`).

| # | Operation | Helper? | Input CB (tiles, state) | Output CB (tiles) | CB State After |
|---|-----------|---------|--------------------------|-------------------|----------------|
| P1 | `o²` DEST-accumulated over `V_TILES` per token | yes (`eltwise_chain` + `DestAccumulation`) | `cb_o_tiles` (`V_TILES*NB`, waited upfront, **not popped**) | `cb_sumsq` (`NB`) | `cb_o_tiles` still holds the chunk (needed by P4) |
| P2 | `rsqrt(mean(o²) + eps)` — reduce **and** epilogue in one DEST window | yes (`reduce` + `post_reduce_op`) | `cb_sumsq` (`NB`, 1/row), `cb_scaler` (1, held) | `cb_rstd` (`NB`, **col-0 valid**) | `cb_sumsq` drained; `cb_scaler` intact |
| P4 | `o * rstd` (bcast Col) | yes (`mul`) | `cb_o_tiles` (`V_TILES*NB`), `cb_rstd` (`NB`, Col-indexed) | `cb_normed` (`V_TILES*NB`) | **`cb_o_tiles` and `cb_rstd` both drained** — the chunk's `o` is released here, which is what lets `O_DEPTH=2` prefetch the next chunk |
| — | *(P1..P6 above repeat per chunk; P7a..P7c below run once per token-block)* | | | | |
| P5 | `* weight` (bcast Row) | yes (`mul`) | `cb_normed` (`V_TILES*NB`), `cb_weight` (`V_TILES`, Row-indexed, held) | `cb_onorm` (`V_TILES*NB`) | `cb_normed` drained; `cb_weight` intact for every later chunk/block |
| P6 | untilize head-major → row-major | yes (`untilize<V_TILES>`) | `cb_onorm` (`V_TILES` per block x `NB` blocks) | `cb_rm_flat_rows` (+`V_TILES*NB` pages) | `cb_onorm` drained; `cb_rm_flat_rows` **accumulates across all chunks** and is only complete after the last chunk |
| — | *(chunk loop repeats)* | | | | after `TOKENS_PER_BLOCK/NB` chunks `cb_rm_flat_rows` holds exactly `FLAT_TILES` pages |
| P7a | tilize row-major → flat token-major | yes (`tilize<FLAT_TILES>`, `num_blocks=1`) | `cb_rm_flat_rows` (`FLAT_TILES`) | `cb_flat_tiles` (`FLAT_TILES`) | `cb_rm_flat_rows` fully drained back to the buffer base (required — §6.1) |
| P7b | `sigmoid(gate)` (SFPU) | yes (`unary<Sigmoid<>>`) | `cb_gate_tiles` (`GC`, streaming) | `cb_gate_sig` (`GC`) | `cb_gate_tiles` slot freed for the reader |
| P7c | `flat * sigmoid(gate)` (**FPU**) | yes (`mul`) | `cb_flat_tiles` (`GC`, streaming), `cb_gate_sig` (`GC`, streaming) | `cb_out_tiles` (streaming) | both drained; after `FLAT_TILES/GC` gchunks `cb_flat_tiles` is fully drained ⇒ ready for the next block |

## 11. Broadcast Verification

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast Dim |
|-------|-----|-------------------|-------------------|---------------|
| P1 | `mul_tiles(cb_o_tiles, cb_o_tiles)` (square, DEST-accumulating) | All `[H,W]` | All `[H,W]` (same buffer) | `None` |
| P4 | `mul_tiles_bcast(cb_o_tiles, cb_rstd)` | All `[H,W]` | **Col0** (REDUCE_ROW output) | **`Col`** |
| P5 | `mul_tiles_bcast(cb_normed, cb_weight)` | All `[H,W]` | **Row0** (1-D `[V]` operand) | **`Row`** |
| P7b | `sigmoid_tile(D0)` (SFPU unary) | All `[H,W]` | n/a (unary) | n/a |
| P7c | `mul_tiles(cb_flat_tiles, cb_gate_sig)` (**FPU**) | All `[H,W]` | All `[H,W]` | `None` |

`BroadcastDim` names the axis that is **broadcast**, not the one reduced
(`eltwise_chain.hpp:523-528`): a REDUCE_ROW result is column-shaped and
broadcasts back across columns with `Col`; a `[1,V]` weight row broadcasts down
the rows with `Row`.

## 12. Key Risks and Gotchas

1. **`cb_rm_flat_rows` must be EXACTLY `FLAT_TILES` pages.** Sizing it larger
   lets the ring wrap mid-block and the tilize address generator — which assumes
   one contiguous `[32, FLAT]` stripe with stride `FLAT*2` bytes — will read
   garbage. Sizing it smaller deadlocks the untilize.
2. **`cb_flat_tiles` must hold the full `FLAT_TILES` block.** A small
   `cb_flat_tiles` + `StreamMode::PerTile` **hangs** (`cb_reserve_back` on PACK
   vs `cb_pop_front` on UNPACK, both inside one compute kernel — §6.1). If a hang
   shows up here the triage signature is PACK blocked in `cb_reserve_back` on
   CB 30 with UNPACK still inside the tilize loop.
3. **Reduce over `V`, never over tokens or heads.** `o`'s tiled row axis is
   `HV`, so the reduction is `REDUCE_ROW` (over `W` = `V`) with `Ht = 1` and
   `batches = NB`. A `REDUCE_COL` here would silently reduce across heads.
4. **`cb_weight` and `cb_scaler` are never popped.** They are filled once per
   core before the block loop; every chunk re-`wait_front`s them (idempotent).
   Popping either desynchronizes the CB permanently.
5. **`cb_o_tiles` is held across P1..P4.** P1 uses `HeldBulk` (wait, no pop) and
   P4 does the pop. If P1 is changed to a popping lifecycle, P4 reads freed pages.
6. **`OperandKind::Row`/`Col` require a non-draining lifecycle** (`Bulk`,
   `HeldBulk`, `CallerManaged`, `DeferredPop` — `eltwise_chain.hpp:376-382`) and
   a 2-D `EltwiseShape::grid(...)`. `Streaming`/`HeldStream` on the weight or
   rstd operand is a compile error, and a 1-D `tiles(n)` shape makes `Row`/`Col`
   meaningless.
7. **`fp32_dest_acc_en=True` halves `DEST_AUTO_LIMIT` to 4**
   (`dest_helpers.hpp:95-99`). Every chain here uses `Dst::D0`/`D1` only; do not
   introduce `D4+`. This also makes `untilize<V_TILES=4>` fit the single-pass
   `pack_untilize` path *exactly* (`untilize_helpers.inl:78-87`) — widening
   `V_TILES` silently moves P6 onto the slower block-based path.
8. **`epsilon` reaches the kernel as a raw fp32 bit pattern**, consumed by
   `add_unary_tile(dst, eps_bits)` inside P2's `post_reduce_op` lambda
   (`binop_with_scalar_tile_init()` must precede it). Pack it on the host with
   `struct.unpack("I", struct.pack("f", eps))[0]` and pass it as an RT arg.
   `epsilon` is applied to the **mean of squares** before `rsqrt`, matching
   `torch: x * rsqrt(mean(x²) + eps)`.
9. **`row_wise=True`.** The `split_work_to_cores` default is column-major and
   measured 2.91x slower here (§7).
10. **`weight`'s tile rows 1..31 are padding.** Only row 0 is read (via
    `BroadcastDim::Row`), so the padding value is irrelevant — but do *not*
    switch P5 to a non-broadcast multiply.
11. **Uniform `Float16_b` across all 12 CBs is load-bearing**, not incidental:
    it is what makes the `RECONFIG_MODE` knob (§1.5) a legal one-line
    knob-turn, and it lets one `compute_kernel_hw_startup` serve every phase.
    The precision cost is bounded and deliberate: the sum-of-squares accumulates
    in fp32 DEST (contract-required), and P2's fused epilogue means the statistic
    makes **one** bf16 round-trip, not two — `cb_rstd` is the only stats CB left.
    That single pack quantizes `rstd` at bf16 resolution (~2⁻⁹ ≈ 0.2 % worst
    case), applied *coherently* to all `V` channels of one head, which is the same
    order as the bf16 quantization of the output elements themselves and well
    inside the bf16 golden tolerance (PCC 0.995 / RMS 0.04,
    `eval/golden_tests/onorm/helpers.py:82-86`). **If measured PCC lands
    marginal, promoting `cb_rstd` alone to `Float32` (+16 KB, `NC` pages) is the
    first precision lever** — it costs only that CB's reconfig, not the uniformity
    of the rest.
12. **No `ttnn.reshape` / `to_layout` / `tilize` / `untilize` in the Python entry
    point.** The head-major → flat conversion is P6+P7a, in-kernel, by contract.
13. **`num_token_blocks` uses `ceil` per batch.** `B * ceil(T/32)`, never
    `floor(B*T/32)` — even though the contract makes `T % 32 == 0`, a future
    non-tile-aligned `T` is exactly where that formula breaks. Related asymmetry
    to keep in mind for that future: in `o` the tiled dims are `(HV, V)`, so `o`'s
    token axis is **un-padded** (`B*T*V_TILES` pages), while in `gate`/`out` the
    tiled dims are `(T, FLAT)`, so their token axis **is** tile-padded
    (`B*Tt*FLAT_TILES` pages). `T % 32 == 0` is what makes the two views coincide.
14. **`dst_full_sync_en` must be `False`** (§3). It is not a neutral
    precision/DEST-capacity knob here: `can_use_fast_tilize()` gates on
    `!get_dst_full_sync_enabled()` (`tilize_helpers.inl:66-79`), so flipping it
    drops P7a's `FLAT_TILES`-wide re-tile off the fast-tilize datapath on every
    core, every block — a silent, uniform slowdown with no correctness signal.
    Also: do **not** tag `cb_rm_flat_rows` with
    `UnpackToDestMode::UnpackToDestFp32` — combined with fast tilize it silently
    corrupts output (`tilize_helpers.inl:259-266`).
15. **The gate multiply must stay on the FPU.** P7b/P7c are split across an L1
    CB *on purpose* (§9). Collapsing them into one chain with a DEST-resident
    SFPU `MulBinary` is the single most expensive mistake available in this kernel:
    it is a measured **0.58x** on the op's largest compute volume (every output
    tile of every block) — enough to turn a DRAM-bound op compute-bound.

## 13. Structural impossibilities and one feature-spec defect to fold in

`INVALID` lives in `eval/golden_tests/onorm/feature_spec.py` and is correctly
empty: TARGET is the single `(bfloat16, TILE_LAYOUT)` cell, so there is nothing
structurally impossible for the harness to skip. **The op file must not declare or
check `INVALID`**, and no new `INVALID` candidate is proposed here.

One **defect in the golden suite** that the implementer will hit and that is *not*
mine to edit — fold it in with another `/golden-tests` invocation:

> `eval/golden_tests/onorm/feature_spec.py:75` — `INPUTS[4]` is
> `((2, 64, HV, V), (2, 64, FLAT), (2, 1, 1, V))`. The weight shape `(2, 1, 1, V)`
> contradicts both the documented contract `weight : [1, 1, 1, V]` (same file,
> `:63`) and the reference implementation: `pytorch_onorm` does
> `weight.to(torch.float32).reshape(-1)` (`helpers.py:50`), which yields **2·V =
> 256** elements for that shape, so `normed[2,64,32,128] * weight[256]` raises a
> torch broadcast error. Because `run_onorm` computes `expected` *before* building
> any device tensor (`helpers.py:108`), this case fails inside the harness and
> never reaches the op at all. It should read `(1, 1, 1, V)`.

The op-side contract is unaffected: `weight` must have `shape[-1] == V` with every
leading dim `== 1`, enforced by a host assert (not an axis refusal — weight rank is
not a TARGET axis). The acceptance test at
`tests/ttnn/unit_tests/operations/onorm/test_onorm.py` uses `(1, 1, 1, V)` for
every case, `B = 2` included, so it is unaffected by the defect.

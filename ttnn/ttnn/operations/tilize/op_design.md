# Operation Design: tilize

Single source of truth for the implementer. Every statement is a decision.

---

## 1. Blocking Model (decided first — everything below is downstream)

### 1.1 The tensor's axes, and what this op's math makes of them

`tilize` is a **pure permutation of byte positions**. Output tile `(r, c)` is a function of exactly
one disjoint `tile_h x 32` region of the row-major input: sticks `[r*tile_h, (r+1)*tile_h)`,
byte range `[c*32*elem, (c+1)*32*elem)` of each. Nothing crosses that boundary, in either
direction, ever.

Fold the leading dims first (per-image, alignment-aware — see §4):

| Symbol | Meaning | Formula (single source of truth, host) |
|---|---|---|
| `TILE_H` | output tile height | `tile.height` if `tile=` given else 32 |
| `TILE_W` | output tile width | always 32 |
| `NT_H` | total tile-rows | `prod(target_padded_shape[:-2]) * ceil(target_padded_shape[-2] / TILE_H)` |
| `WT` | total tile-columns | `ceil(target_padded_shape[-1] / TILE_W)` |
| `STICKS` | row-major sticks in the input | `prod(input_shape[:-2]) * input_shape[-2]` |
| `STICK_BYTES` | bytes per input stick | `input_shape[-1] * elem_bytes(in_dtype)` |

### 1.2 Axis character table

| Axis | Character | Block-factor knob | Phase 0 value | Core-assignment | Later unlock |
|------|-----------|-------------------|---------------|-----------------|--------------|
| `NT_H` — tile-row axis (folded leading dims x H) | **independent** — each `TILE_H`-stick band produces its own row of tiles; no value crosses a band boundary | `NT_BLK` = tile-rows per reader barrier-block | `1` (the tilize LLK block is exactly one tile-row tall; `read_sticks_for_tilize` barriers once per tile-row, `tilize_helpers_dataflow.inl:126`) | **primary split.** Blocks `b -> (wc = b / NT_H, row = b % NT_H)`; `ttnn.split_work_to_cores(grid, NUM_BLOCKS, row_wise=True)` spreads them across the whole compute grid | **knob-turn** (raise `NT_BLK` -> deeper bytes-in-flight per barrier; needs a custom reader — see Lamp L3) |
| `WT` — tile-column axis | **independent** — output tile `(r,c)` reads a disjoint 32-column byte slice of the same sticks | `WT_CHUNK` = tile-columns per block | `WT` when `NT_H >= NUM_CORES` **and** it fits the L1 budget; otherwise the coarsest exact divisor of `WT` that fills the grid and fits L1 (§1.4) | **secondary split**, folded into the same 1-D block index. `WT_CHUNK == WT` makes it byte-identical to a pure height split | **knob-turn** |
| intra-tile (32x32 element permutation) | **atomic** — the LLK's unit | — | — | never split | — |
| dtype / format | not a work axis | — | — | — | — |

**There is no dependent axis.** No reduction, no scan, no cross-block combine — so **no cross-core
combine will ever be needed** and every parallelism decision in this op is a knob-turn. This is the
single most important structural fact about `tilize` and it is why the Phase 0 scheme can be the
final scheme.

**Operand-reuse check.** One input operand. Every input byte belongs to exactly one block on
exactly one core, under either split. Nothing is re-read; nothing is broadcast.
**No `mcast_pipe` / broadcast lamp exists for this op** — this is a positive finding, not an omission.

### 1.3 Bandwidth ranking of the candidate splits (qualitative, structural — no ns)

Total DRAM traffic is `1x tensor read + 1x tensor write` under **every** candidate below except C4;
what differs is **transaction size**, **transaction count**, and **how many cores are lit**.

| Rank | Candidate split | Bytes moved | Read transaction | Cores lit | Verdict |
|---|---|---|---|---|---|
| **1 (chosen)** | **2-D block: `1 tile-row x WT_CHUNK tile-cols`**, `WT_CHUNK` defaulted to `WT` | 1x + 1x | `WT_CHUNK * 32 * elem` B/stick (== whole stick when `WT_CHUNK == WT`) | **full grid on every shape regime** | **WINNER.** Degenerates *byte-identically* to candidate 2 when `WT_CHUNK == WT`, so it can never be worse than the pure height split; it is the only candidate that lights the grid on a short-wide shape and the only one that bounds CB L1 in `W`. |
| 2 | Pure height split (`WT_CHUNK == WT` forced) | 1x + 1x | whole stick (largest possible) | `min(NT_H, grid)` | **rejected.** Loses on `NT_H < NUM_CORES`: on the mandatory bench regime `[1,1,32,16384]` (`NT_H = 1`) it runs on **one core**. Also OOMs: its input CB is `CB_DEPTH * WT * tile_bytes`, unbounded in `W` (`WT=512` bf16 depth-2 = 2 MB > 1.5 MB L1). Lost on **grid occupancy** and **L1 boundedness**. |
| 3 | Pure per-tile gather (`WT_CHUNK == 1` forced) | 1x + 1x | `32*elem` B = **64 B** (bf16) | full grid | **rejected.** 32x more read transactions than candidate 1 for identical bytes; sits squarely in `master.md` **A0**'s transaction-rate-bound regime (`<=128 B/page`). Lost on **transaction count**. Retained only as the degenerate value candidate 1 falls back to when `WT` has no coarser divisor. |
| 4 | Two-pass: stage the tensor DRAM->L1, then tilize out of L1 | **2x** + 1x | — | full grid | **rejected.** +100% read bytes; an extra full-tensor DRAM pass shows up directly in tt-npe. Also explicitly forbidden by the op's Rules. Lost on **bytes moved**. |

> These are **predictions**, not measurements. A0 must check the winner against the device number.
> If the winner misses its predicted target, candidates 2 and 3 above are the recorded fall-backs.

**Why the tile-row is the primary split and the tile-column the secondary:** splitting `NT_H`
leaves the read as whole contiguous sticks (the largest coalesced transaction the tensor offers,
`master.md` **B5**) and the write as whole contiguous tile pages. Splitting `WT` cuts each stick
read into a byte range, shrinking the transaction. So `WT` is split only as far as the grid-fill
and L1 constraints demand, never further.

### 1.4 The three knobs — derivation (DRY: one source, everything derived)

All of this lives in **one** host function, `derive_blocking()`, in
`tilize_program_descriptor.py`. No literal below is restated anywhere else; kernels receive the
derived values as CT/RT args.

| Knob | Symbol | Phase 0 value | Rationale |
|---|---|---|---|
| Buffer depth | `CB_DEPTH` | `2 if use_double_buffer else 1` | `master.md` **C16**; the caller-facing `use_double_buffer` kwarg *is* this knob. |
| Block factor, H | `NT_BLK` | `1` | The LLK block is one tile-row (`tilize_helpers.inl:242-273` loops one `tilize_block` per block). |
| Block factor, W | `WT_CHUNK` | see below — **coarsest**, i.e. `WT` unless a constraint forces smaller | never the minimal unit |
| Core-assignment | `NUM_CORES` | `grid.x*grid.y` when `use_multicore`, else `1` | a **parameter**, never inlined |

```python
CB_L1_BUDGET = 1_048_576          # bytes of L1 reserved for the two streaming CBs (of 1_499_136)
FAST_TILIZE_MAX_W = 255           # tilize_helpers.inl:95 -> block_width_tiles < 256

def derive_blocking(NT_H, WT, in_tile_bytes, out_tile_bytes, num_cores, cb_depth):
    # --- L1 ceiling on the W block factor -------------------------------------
    per_chunk_tile = cb_depth * (in_tile_bytes + out_tile_bytes)
    wt_cap = max(1, min(FAST_TILIZE_MAX_W, CB_L1_BUDGET // per_chunk_tile))
    # --- grid-fill floor on the number of W chunks -----------------------------
    n_want = max(1, -(-num_cores // NT_H))      # ceil; 1 when the H axis already fills the grid
    n_want = max(n_want, -(-WT // wt_cap))      # ceil; L1 may demand more chunks than the grid does
    # --- snap to an exact divisor of WT so every block has the SAME width ------
    n_chunks = next(c for c in range(n_want, WT + 1) if WT % c == 0)   # c = WT always terminates
    WT_CHUNK = WT // n_chunks
    NUM_BLOCKS = NT_H * n_chunks
    return WT_CHUNK, n_chunks, NUM_BLOCKS
```

Three properties this buys, stated as commitments:

1. **`NT_H >= NUM_CORES` => `n_chunks == 1` => `WT_CHUNK == WT`.** The wide-shape machinery is
   *inert* on tall shapes: byte-identical to the pure height split, no regression
   (`master.md` Part 1 `distribution_gate` demands exactly this no-regression property).
2. **`WT_CHUNK` divides `WT` exactly.** Every block is the same width, so `block_width_tiles`
   is one compile-time constant and there is **one** compute kernel, no cliff-width variant,
   no straddling core.
3. **The default is the coarsest chunk that fits**, not the minimal unit. `WT_CHUNK` shrinks only
   in response to the L1 ceiling or the grid-fill floor — both explicit, both derived.

### 1.5 Buffer-depth knobs (per streaming CB)

| CB | Depth knob | Phase 0 | Note |
|---|---|---|---|
| `cb_input_sticks` | `CB_DEPTH` | 2 | reader -> compute; depth 2 lets the reader fill block *n+1* while compute drains *n* |
| `cb_output_tiles` | `CB_DEPTH` | 2 | compute -> writer; same |

Both fall to 1 when `use_double_buffer=False` (A6), and the host **auto-falls-back to 1** rather
than exceeding L1 (`master.md` C16, precedent `concat_program_factory.cpp:111`). There is no
compute->compute CB in this op, so no "full block" intermediate is needed.

### 1.6 Lamp — the scheme-changes Phase 0 leaves room for

| # | Lamp | Class | Why Phase 0 does not foreclose it |
|---|---|---|---|
| **L1** | **Zero-copy sharded I/O** (A3): back `cb_input_sticks` / `cb_output_tiles` on the resident L1 shard via `ttnn.cb_descriptor_from_sharded_tensor` (`program_descriptors.cpp:517-556`) — **no NoC on the sharded side**. | placement scheme-change | The blocking is already read *off* a per-core extent; a shard just pins `NUM_CORES`, the per-core row range and `WT_CHUNK = WT_shard`. The loop nest is unchanged; only the CB backing and the reader/writer kernel selection change. |
| **L2** | **Cross-spec reshard** (A3c): input shard on core A, output shard on core B -> a genuine cross-core L1 gather through a `TensorAccessor` over the L1-sharded source. | data-placement scheme-change (still **no combine** — nothing is reduced) | The reader is already accessor-driven and already takes an arbitrary `start_page`; the gather is a different accessor, not a different loop nest. |
| **L3** | **Multi-tile-row barrier block + trid double-issue** (`master.md` **B8**): `NT_BLK > 1`, reserve `NT_BLK*WT_CHUNK` pages, issue `NT_BLK*TILE_H` reads under one barrier, barrier on the *previous* trid. | knob-turn, but needs a custom reader (the library helper barriers per tile-row, `tilize_helpers_dataflow.inl:126`) | `NT_BLK` is already a named knob; the CB size formula is already written against it. Only measurable on a **>=2 blocks/core** shape — see §9 bench (c). |
| **L4** | **`split_reader`** (`master.md` Part 1, ~1.7x): both DM RISCs issue reads. | knob-turn | **Structurally inapplicable on the DRAM->DRAM path** (BRISC already runs the writer). Becomes applicable exactly on the **sharded-output** path (L1), where the writer does no NoC work and BRISC is free. Recorded so the completeness audit (A7) resolves it with a reason, not a shrug. |
| **L5** | **Pad-aware reader** (Track P): the fill is materialized in L1 as the block is assembled. | separate reader kernel, same blocking | The regime selector (§5) picks the reader; the block geometry, CB layout, work split and compute kernel are **identical**. Padding costs no extra pass. |
| **L6** | **Retile reader** (T2, Blackhole-only): a TILE input walked face-wise instead of stick-wise. | separate reader kernel, same blocking | Same: only the reader changes. |

---

## 2. Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (layout conversion with a compute stage) |
| Goal | Reorder a ROW_MAJOR tensor into TILE layout on device, optionally value-preserving-cast the dtype, optionally pad H/W up to a tile-multiple target with a caller-supplied fill, optionally emit a sub-32-row "tiny" tile. |
| Math | `out[..., i, j] = in[..., i, j]` for `i < H, j < W`; `out[..., i, j] = pad_value` otherwise. No arithmetic. |
| Mode | Derivative (mirrors the `untilize` factory family) |
| References | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp`, `ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp`, `ttnn/ttnn/operations/toy_tilize_untilize/`, `ttnn/ttnn/operations/examples/master.md`, `.claude/references/generic_op_template/`, `eval/golden_tests/tilize/feature_spec.py` |

### Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | ROW_MAJOR (TILE only on the retile path) | — | — |
| `memory_config` | `ttnn.MemoryConfig \| None` | no | interleaved DRAM/L1, or L1-sharded | input's | host |
| `dtype` | `ttnn.DataType \| None` | no | see SUPPORTED / TARGET `output_dtype` | input's | CT (CB format) |
| `use_multicore` | `bool` | no | `{False, True}` | `True` | host (`NUM_CORES`) |
| `use_double_buffer` | `bool` | no | `{False, True}` | `True` | host (`CB_DEPTH`) |
| `output_padded_shape` | `list[int] \| ttnn.Shape \| None` | no | `>= input shape` in every dim; last two dims tile multiples | `None` | host |
| `pad_value` | `float \| int \| None` | no | any value representable in the **input** dtype | `None` | RT (packed word) |
| `tile` | `ttnn.Tile \| None` | no | `[h, 32]`, `h in {32,16,8,4,2,1}` | `Tile([32,32])` | CT (CB `TileDescriptor`) |

**Canonicalization performed by the entry point before the support check** (so `validate()` sees
one convention):

| Input form | Canonical form |
|---|---|
| `output_padded_shape` absent, `pad_value` absent | `pad_mode="none"`, `target = input.padded_shape` |
| `pad_value` present, `output_padded_shape` absent | `pad_mode="auto"`, `target = shape with last two dims rounded up to (TILE_H, 32)` |
| `output_padded_shape` present | `pad_mode="explicit"`, `target = list(output_padded_shape)`; rank < 2 inputs get the tile dims synthesized |
| `output_padded_shape` present, `pad_value` absent | **raise** (a target with no fill is undefined) |
| `pad_mode="none"` and H or W is not a tile multiple | **raise** — padding is never implicit. The message **MUST contain the substring `pad`** (case-insensitive); the acceptance test's `expect_error` matches on it |
| `tile=None` | `tile = ttnn.Tile([32, 32])` |
| `ttnn.Shape` for `output_padded_shape` | `list[int]` |

### Tensors

**Input**

| Property | Requirement |
|----------|-------------|
| Shape | rank >= 2 (rank 0/1 only when padding is requested — the tile dims are synthesized) |
| Dtype | TARGET: `bfloat16` (primary), `float32`, `uint32`/`uint16`/`int32`, `uint8`. Phase 0: `bfloat16` |
| Layout | `ROW_MAJOR_LAYOUT`; `TILE_LAYOUT` only on the retile path (T2) |
| Memory | INTERLEAVED (DRAM or L1), or L1-SHARDED (row-major-sharded) |

**Output**

| Property | Value |
|----------|-------|
| Logical shape | **identical to the input's logical shape** — padding never widens it |
| Padded shape | `input.padded_shape` when `pad_mode="none"`, else the pad target |
| Dtype | `dtype` if given else input's; a real value-preserving cast at pack time, never a byte copy |
| Layout | `TILE_LAYOUT` with `tile = Tile([TILE_H, 32])` |
| Memory | `memory_config` if given else the input's |

---

## 3. Dataflow Strategy

```
                per block = 1 tile-row x WT_CHUNK tile-cols
DRAM/L1 (RM sticks) --NoC0--> [cb_input_sticks] --TRISC(tilize LLK)--> [cb_output_tiles] --NoC1--> DRAM/L1 (TILE pages)
   reader (NCRISC)              tile-sized pages,        compute            tile-sized pages,        writer (BRISC)
                                CB_DEPTH*WT_CHUNK                           CB_DEPTH*WT_CHUNK
```

| Stage | Format | Transaction | Barrier policy |
|---|---|---|---|
| DRAM -> `cb_input_sticks` | `TILE_H` sticks of `WT_CHUNK*32*elem_in` bytes each, written at L1 stride `WT_CHUNK*32*elem_in` | one `noc_async_read` per stick (`TILE_H` per block) | **one barrier per block** (`master.md` B7) — `tilize_helpers_dataflow.inl:126` |
| `cb_input_sticks` -> `cb_output_tiles` | `WT_CHUNK` tiles per block | `tilize_block` / `fast_tilize_block` | CB reserve/push owned by the helper |
| `cb_output_tiles` -> DRAM | `WT_CHUNK` whole tile pages | one `noc_async_write` per **whole tile page** (`master.md` B5) | **one barrier per block** |

**NoC assignment (`master.md` **B9**, measured 2.5-4.8x in `noc_placement`):** reader on
`ReaderConfigDescriptor()` (NCRISC / NOC0), writer on `WriterConfigDescriptor()` (BRISC / NOC1).
Never reversed.

**Core placement (`master.md` **A1**, measured ~2.9x in `noc_placement`):**
`ttnn.split_work_to_cores(grid, NUM_BLOCKS, row_wise=True)`. The binding default `row_wise=False`
is column-wise and is the documented trap (`noc_placement.py:12,33`) — **`row_wise=True` is a
day-1 decision, not a later tuning.**

**Active core count (`master.md` **A0**):** use the **full compute grid**; do **not** apply a
`dram_saturation`-style bandwidth-knee cap. A0 records this measured verbatim about this very op:
*"Measured on tilize: applying a 16-core knee cap was ~2.4x slower — the knee clause was
implemented, measured, and refuted precisely because the op is transaction-rate bound at 64 B/page."*
A0 of this run re-confirms it with the ablation; it is not re-litigated per refinement.

### Tensix-to-Tensix contract (for the unlocked schemes — Phase 0 does not use it)

| Scheme | Who talks to whom | What moves | Sync | Ordering |
|---|---|---|---|---|
| L1 zero-copy sharded (A3) | **nobody** — each core's shard is already in its own L1 | nothing on the NoC | CB handshake only | n/a |
| Interleaved -> sharded (A3b) | reader on core *k* pulls from DRAM into its own output shard | RM sticks in, nothing out | CB handshake | n/a |
| Cross-spec reshard (A3c) | core *k* reads L1 pages owned by cores `S(k)` through a `TensorAccessor` over the L1-sharded source | RM stick fragments, core->core | `noc_async_read_barrier` per block; **no semaphores** — the source shards are read-only and already resident before the program launches | per-block; blocks are independent, so any order is legal |

There is **no multicast and no semaphore handshake anywhere in this op**, in any scheme, because
there is no reuse-shared and no dependent axis (§1.2). `mcast_pipe.hpp` is deliberately unused.

---

## 4. Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one **block** = 1 tile-row x `WT_CHUNK` tile-columns = `WT_CHUNK` output tiles |
| Total work | `NUM_BLOCKS = NT_H * n_chunks` |
| Grid | `NUM_CORES = grid.x*grid.y` (`use_multicore=True`) or `1` (`use_multicore=False`) — a **parameter**, so widening the SUPPORTED rectangle from A0 to A1 is a knob-turn with **identical kernels, identical CB layout, identical runtime-arg shape** |
| Split | `ttnn.split_work_to_cores(core_grid, NUM_BLOCKS, row_wise=True)` (`ttnn/cpp/ttnn-nanobind/operations/core.cpp:466-534`) |
| Per-core work | contiguous block range `[start_block, start_block + num_blocks)` |
| Block -> geometry | `wc = b // NT_H`, `row = b % NT_H` (**W-chunk-major**) |
| Remainder | `split_work_to_cores` returns `core_group_1` (more blocks) and `core_group_2` (fewer, possibly empty). Both groups run the **same** kernels; only `num_blocks` in the runtime args differs. No cliff kernel, because `WT_CHUNK` divides `WT` exactly (§1.4 property 2). |

**Why W-chunk-major block ordering:** a core's consecutive blocks then share one `wc` (one
`byte_offset_within_page`) and walk consecutive tile-rows, so its DRAM stick reads march linearly
through page ids. When `n_chunks == 1` this is exactly "core *i* owns tile-rows
`[start, start+num_blocks)`" — the plain height split, unchanged.

### Alignment-aware tile geometry (used everywhere, from the start)

| Quantity | Formula | **Not** |
|---|---|---|
| tile-rows | `prod(target[:-2]) * ceil(target[-2] / TILE_H)` | `floor(prod * H / TILE_H)` — each image is tile-padded **independently** |
| tile-cols | `ceil(target[-1] / 32)` | `W // 32` |
| sticks per image | `input_shape[-2]` | — |
| H-alignment predicate | `input_shape[-2] % TILE_H == 0` | `% 32` — **a tiny tile redefines H-alignment**; `tag_alignment` measures against `TILE_H` |
| W-alignment predicate | `input_shape[-1] % 32 == 0` | — (tile width is always 32) |

`ceil` is used unconditionally even though Phase 0 only accepts `tile_aligned` inputs — the
alignment refinement (P1) is precisely what hits the boundary.

---

## 5. Regime selection (pinned — >1 compute regime)

Two **orthogonal** selectors. The host evaluates both; their product picks the kernel triple.

### 5.1 Fill regime (selects the READER)

```python
def fill_regime(input_tensor, target_padded_shape, TILE_H):
    if input_tensor.layout == ttnn.TILE_LAYOUT:
        return "R_RETILE"                                    # T2, Blackhole-only, arch-gated
    if list(target_padded_shape) != list(input_tensor.padded_shape):
        return "R_PAD"                                       # some output position has no input element
    return "R_ALIGNED"                                        # the hot path
```

The predicate is on the **pad region actually being non-empty**, *not* on the `pad_mode` string.
Consequence, and it is a required behaviour: `pad_mode="auto"` on an already-tile-aligned input
selects `R_ALIGNED` and is **byte-identical and equal-cost** to the bare call. The golden suite
has a cell for exactly this (`feature_spec.py:470-474`, "Catches an op that unconditionally takes
the pad reader and corrupts (or slows) the aligned path").

### 5.2 Placement regime (selects the CB backing and the reader/writer variant, per side)

```python
def side_regime(tensor, work_grid, shard_matches_blocking):
    if tensor.is_sharded() and tensor.buffer_type == ttnn.BufferType.L1 \
       and tensor.shard_grid == work_grid and shard_matches_blocking:
        return "P_LOCAL_SHARD"     # CB aliased on the resident shard, ZERO NoC on this side
    return "P_ACCESSOR"            # TensorAccessor over interleaved DRAM/L1, or a cross-core L1 gather
```

`shard_matches_blocking` means: the shard's width in elements is a multiple of 32 (so the shard's
L1 stick stride equals the tilize row stride) **and** the input and output shard specs agree
(same-spec). Anything else -> `P_ACCESSOR` (the A3c cross-spec gather).

Phase 0 realizes exactly `R_ALIGNED x (P_ACCESSOR, P_ACCESSOR)`.

### 5.3 Regime-pinned tests are REQUIRED

A regime that only triggers on some grids passes on one device and fails on another. The
implementer must add, alongside the golden suite, a regime-pinned test that asserts the *selected*
regime for a fixed input set — at minimum one case per row:

| Case | Expected regime | Why it must be pinned |
|---|---|---|
| `[1,1,64,128]` interleaved, no pad | `R_ALIGNED / P_ACCESSOR / P_ACCESSOR` | the hot path |
| `[1,1,64,64]`, `pad_mode="auto"`, `pad_value=0` | `R_ALIGNED` (degenerate no-op pad) | must NOT take the pad reader |
| `[1,1,50,50]` -> `[1,1,64,64]` | `R_PAD` | W tail + H tail |
| `[1,1,50,50]` -> `[1,1,128,128]` | `R_PAD` with whole pad tiles | the third pad region |
| L1 HEIGHT-sharded in/out, same spec | `P_LOCAL_SHARD` both sides | proves zero-copy is actually taken, not silently re-read via accessor |
| HEIGHT-sharded in -> DRAM out | `P_LOCAL_SHARD / P_ACCESSOR` | split path |
| `[1,1,32,16384]` interleaved | `n_chunks == 64`, `WT_CHUNK == 8`, **`NUM_CORES == 64`** | the grid-fill gate; a scheme that collapses here is a failed A1 |

---

## 6. Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|--------|----------|----------|----------|
| `cb_input_sticks` | 0 | `in_tile_bytes` = `TILE_H * 32 * elem_bytes(in_dtype)`; `tile = TileDescriptor(TILE_H, 32)` | `CB_DEPTH * WT_CHUNK` | input dtype | **reader** | **compute** | whole program |
| `cb_output_tiles` | 16 | `out_tile_bytes` = `TILE_H * 32 * elem_bytes(out_dtype)` (block-float: `ttnn.tile_size(out_dtype)`); `tile = TileDescriptor(TILE_H, 32)` | `CB_DEPTH * WT_CHUNK` | output dtype | **compute** | **writer** | whole program |

Two CBs. No intermediates — there is no compute->compute hand-off in this op.

### Sizing rationale (every number is a function of a knob, none of a whole-op dimension)

| Constraint | Source |
|---|---|
| `num_pages >= WT_CHUNK` on **both** CBs | `tilize_helpers.inl:225,227` — `ASSERT(get_dfb_num_pages(dfb) >= block_width_tiles)` |
| `num_pages >= WT_CHUNK` on the input CB (deadlock guard) | `tilize_helpers_dataflow.inl:105-108` |
| `CB_DEPTH * WT_CHUNK * (in_tile_bytes + out_tile_bytes) <= CB_L1_BUDGET` | §1.4; drives `wt_cap`, which drives `WT_CHUNK` |
| `WT_CHUNK < 256` | `tilize_helpers.inl:95` — beyond that the **fast tilize** path is disabled |

**Neither CB's size is a function of `WT`, `NT_H`, or any tensor dimension.** Both scale strictly
with `CB_DEPTH * WT_CHUNK`, and `WT_CHUNK` is itself clamped by `CB_L1_BUDGET`. A wide `W` produces
more *chunks*, never a bigger CB. This is the A3d property, held from Phase 0 rather than
retrofitted.

### Placement-regime overrides

| Regime | `cb_input_sticks` | `cb_output_tiles` |
|---|---|---|
| `P_ACCESSOR` (Phase 0) | streaming, `CB_DEPTH * WT_CHUNK` pages | streaming, `CB_DEPTH * WT_CHUNK` pages |
| `P_LOCAL_SHARD` input | **aliased on the resident RM shard** via `ttnn.cb_descriptor_from_sharded_tensor(0, input_tensor)` (`program_descriptors.cpp:517-556`); `num_pages = shard_tiles`, **depth 1** (nothing to overlap — the data is already there); the reader issues **no NoC read**, it only pushes | unchanged |
| `P_LOCAL_SHARD` output | unchanged | **aliased on the TILE-sharded output**; `num_pages = shard_tiles`, depth 1; compute packs straight into the shard; the writer issues **no NoC write**, it only drains (`wait_front`/`pop_front`) so the CB keeps exactly one consumer |

> Designing a `P_LOCAL_SHARD` side to be re-read through a `TensorAccessor` is **not** implementing
> sharding — it re-fetches over the NoC data already resident in L1. `P_ACCESSOR` is for interleaved
> I/O and for the genuinely non-local A3c gather, and for nothing else.

### CB invariants

| Invariant | Held by |
|---|---|
| push count == wait count, `cb_input_sticks` | reader pushes `WT_CHUNK`/block (`.inl:127`); `tilize` waits `WT_CHUNK`/block (`tilize_helpers.inl:250`) and pops the same (`:268`) |
| push count == wait count, `cb_output_tiles` | `tilize` reserves+pushes `WT_CHUNK`/block (`:253,267`); the writer waits and pops `WT_CHUNK`/block |
| exactly one producer, one consumer | reader->compute, compute->writer. Even in `P_LOCAL_SHARD` the drain-only writer is retained **precisely to keep the consumer count at one** |
| TILE page granularity, not ROW | see §7 rejection 3 — ROW granularity barriers per stick |

---

## 7. API Mapping

| Phase | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Requirements |
|-------|------|----------|-----------|------------------------|----------|-----------|--------------|
| Read RM sticks | **helper** | `dataflow_kernel_lib::read_sticks_for_tilize` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp:87-93` (decl); impl `tilize_helpers_dataflow.inl:67-159`, TILE branch `:96-128` | `<cb_input_sticks, TilizeGranularity::TILE>`; args `(accessor, total_num_rows = runs*TILE_H, row_bytes = WT_CHUNK*32*elem_in, start_page = row*TILE_H, byte_offset_within_page = wc*WT_CHUNK*32*elem_in)`. **`row_bytes` + `byte_offset_within_page` ARE the `WT_CHUNK` knob** (`hpp:72-85`) | — | `cb_input_sticks` | CB page size must equal `in_tile_bytes` and carry `TileDescriptor(TILE_H,32)` — the helper derives `tile_h/tile_w/elem_size` from `unpack_tile_r_dim[cb]` (`inl:75-80`). Not valid for block-float inputs (`inl:85`). |
| Tilize | **helper** | `compute_kernel_lib::tilize` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp:187-197` (decl); impl `tilize_helpers.inl:104-291` | `<WT_CHUNK, cb_input_sticks, cb_output_tiles, InitUninitMode::InitAndUninit, WaitMode::WaitBlock, RECONFIG, Fp32Mode::Fast>` called **once** per core with `num_blocks = blocks_this_core` (no `total_input_pages` — symmetric). **`WT_CHUNK` (`block_width_tiles`) IS the block-size knob** (`hpp:97`). | `cb_input_sticks` | `cb_output_tiles` | `compute_kernel_hw_startup(cb_input_sticks, cb_output_tiles)` first (`hpp:88-93`). One call amortizes init/uninit over all blocks (`master.md` Part 1 `compute_block_size`, 1.65x). |
| Reconfig policy | **helper knob** | `ReconfigureRegisterDatatypeMode` | `tilize_helpers.hpp:22-27`; effect `tilize_helpers.inl:158-194` | `NoReconfigure` when `out_dtype == in_dtype`; `UnpackAndPackReconfigure` **only on a real cast** | — | — | Compile-time `needs_cast` flag. ~150 ns fixed saving; 8-19% on the ~1 us small sharded cases. A measured lever -> ledger. |
| fp32 precision | **helper knob** | `Fp32Mode` | `tilize_helpers.hpp:63-71`; gate `tilize_helpers.inl:126-155` | **`Fast`** (the default) — never `Lossless` | — | — | `master.md` **F26**: lossless buys nothing downstream of an FPU phase, and tilize *is* one. Structurally closed, not swept. |
| Write TILE pages | **raw_api** | `TensorAccessor::get_noc_addr(page_id)` + `noc_async_write` + `noc_async_write_barrier` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:566` (write, cited by `master.md` **B5**); accessor contract `tech_reports/tensor_accessor/tensor_accessor.md` | one write per **whole tile page** (`out_tile_bytes`), `WT_CHUNK` per block, **one barrier per block** (`master.md` **B7**) | `cb_output_tiles` | — | see rejection 1 below |
| Pad fill (Track P) | **raw_api** | `fill_with_val<T>(begin_addr, n, val)` | `ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp:156-162` | packed fill word in the **input** element format, replicated across the 32-bit store word | — | `cb_input_sticks` | see rejection 2 below |
| Work split | **binding** | `ttnn.split_work_to_cores` | `ttnn/cpp/ttnn-nanobind/operations/core.cpp:466-534`; export `ttnn/ttnn/core.py:19` | `(core_grid, NUM_BLOCKS, row_wise=True)` | — | — | `row_wise=True` is mandatory (`master.md` **A1**) |
| Sharded CB alias | **binding** | `ttnn.cb_descriptor_from_sharded_tensor` | `ttnn/cpp/ttnn-nanobind/program_descriptors.cpp:517-556`; export `ttnn/ttnn/types.py:115` | `(cb_index, tensor, core_ranges=grid)` | — | — | `P_LOCAL_SHARD` only |
| Tile geometry | **binding** | `ttnn.TileDescriptor(height, width)`, `ttnn.FaceGeometry(face_r_dim, num_faces)` | `ttnn/ttnn/types.py:101-127` | `TileDescriptor(TILE_H, 32)` on both CB formats | — | — | T1 (tiny tiles) falls out of this **with no reader/compute change** — the reader derives `tile_h` from `unpack_tile_r_dim[cb]` (`tilize_helpers_dataflow.inl:75`). A5b (uint8) sets `FaceGeometry(face_r_dim=16, num_faces=4)` — the **per-face** row dim, not the full-tile one. |
| Accessor args | **binding** | `ttnn.TensorAccessorArgs(t).get_compile_time_args()` | `.claude/references/generic_op_template/template_op_program_descriptor.py` | appended **last**, after all scalar CT args | — | — | `master.md` **D18** |

### Helpers considered and rejected (mandatory justification)

**1. Writer — CB tiles -> interleaved TILE pages.**
Considered: `dataflow_kernel_lib::write_sticks_after_untilize`
(`ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp:129-135`).
Rejected because it is the **inverse direction**: its contract at `hpp:98-102` is *"Reads
total_num_rows worth of untilized data from the CB ... and writes the valid **sticks** to DRAM"*,
and `hpp:102` *"Handles non-tile-aligned widths by skipping L1 padding between rows"* — it
de-interleaves tiles into row-major sticks. Our destination pages are whole TILE pages that need
no de-interleave; using it would write stick fragments into a tiled buffer. No helper in
`ttnn/cpp/ttnn/kernel_lib/` covers CB-tiles -> tiled-tensor pages (the directory listing contains
no tile-page writer; `dfb_helpers_dataflow.hpp:14-19` exposes only tile-dimension queries).
Raw `TensorAccessor` + `noc_async_write` is the correct mechanism.

**2. Pad-path reader.**
Considered: `read_sticks_for_tilize` (same helper as the aligned path).
Rejected at `ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp:50-52`: *"Handles non-tile-aligned
heights by pushing full tile pages for the last partial block (**untouched rows contain stale
data**)"*, and `inl:120-123` reads only `row_bytes` while advancing L1 by `padded_row_bytes`,
leaving the W tail untouched. The golden pad oracle compares the pad region **exactly**
(`eval/golden_tests/tilize/feature_spec.py:17-18`), so stale L1 fails. The helper has no fill
parameter. Hence a separate `R_PAD` reader — which is *why* Phase 0 keeps the aligned reader
byte-identical.

**3. `TilizeGranularity::ROW`.**
Considered and rejected: `tilize_helpers_dataflow.inl:148-157` issues `noc_async_read_barrier()`
**inside the per-stick loop** — one barrier per transaction, the exact anti-pattern `master.md`
**B7** names. Its only stated benefit (`hpp:60-65`) is L1 savings when `total_num_rows < TILE_H`,
which our `WT_CHUNK` clamp already covers. TILE granularity, always.

**4. `mcast_pipe.hpp` / `host/mcast_host.hpp`.**
Not applicable: §1.2 establishes there is no reuse-shared and no dependent axis, so no operand is
ever read by more than one core. Recorded here so the A7 completeness audit resolves **B12** as
*structurally-impossible*, with this as the assertion.

**5. `dest_helpers.hpp` / `DEST_AUTO_LIMIT`.**
Not needed: the tilize LLK chunks its own DEST usage internally; `tilize_helpers.inl` performs no
DEST budgeting and imposes no `block_width_tiles <= 8` bound — the only width bound is
`< 256` for the fast path (`inl:95`). `WT_CHUNK` is therefore an **L1** knob, not a DEST knob.

---

## 8. Compute Phases

| # | Operation | Helper? | Input CB (tiles, state) | Output CB (tiles) | CB State After |
|---|-----------|---------|-------------------------|-------------------|----------------|
| 0 | `compute_kernel_hw_startup(cb_input_sticks, cb_output_tiles)` | required prerequisite (`tilize_helpers.hpp:88-93`) | — | — | — |
| 1 | `tilize<WT_CHUNK, cb_input_sticks, cb_output_tiles, InitAndUninit, WaitBlock, RECONFIG, Fast>(blocks_this_core)` | **yes** | `cb_input_sticks`, `WT_CHUNK` pages per block, `blocks_this_core` blocks | `cb_output_tiles`, `WT_CHUNK` pages per block | both drained: `tilize` pops its input and pushes its output per block (`tilize_helpers.inl:267-268`); the writer pops the output |

One helper call for the whole core. `RECONFIG` = `NoReconfigure` when `needs_cast == false`, else
`UnpackAndPackReconfigure`. There is no phase 2 — the op has exactly one compute phase.

---

## 9. Performance Methodology

**Classification: movement-dominated (predicted) — to be CONFIRMED by ablation at A0.**
Both sides are DRAM; the compute stage is a fixed-cost LLK permutation. But tilize *has* a compute
stage, so the NoC ceiling is only a **partial** bound.

### 9.1 A0's mandatory classification ablation

Stub the `tilize_block`/`fast_tilize_block` math inside the compute kernel, **keep** the CB
reserve/push/wait/pop and both barriers, re-measure with `/perf-measure`:

| Outcome | Conclusion | Consequence |
|---|---|---|
| duration barely moves | **DM-bound** | the `/perf-ceiling-dm` target applies; chase the DM levers |
| duration drops a lot | **compute-bound** | the DM ceiling does **not** apply; the target is the LLK throughput and DM levers are noise |

This single result is the baseline claim every later Track A refinement rests on. Record it.

### 9.2 Target computation (per case, per the skills — do not eyeball)

1. `/perf-ceiling-dm` Mode A. **Both sides are DRAM.** Characterize the read (ROW_MAJOR sticks,
   transaction = `WT_CHUNK*32*elem` B) and the write (TILE pages, transaction = `out_tile_bytes`)
   **separately**. Bracket each with `ONE_FROM_ALL`/`ONE_TO_ALL` (few-core, no contention) ...
   `ALL_FROM_ALL`/`ALL_TO_ALL` (full grid, full contention); the true value lands inside, near the
   full-contention end for round-robin interleaved. Step 4b caps every bound at `dram_peak`
   (WH 288 GB/s) and computes route-overlap congestion.
   `op_target = MAX(read_bound, write_bound, compute_bound)`, read and write overlapping per
   `CB_DEPTH` (they do **not** overlap at `CB_DEPTH=1` — that is the A6 counterfactual).
2. tt-npe: `tt_npe.sh <trace> --noc-trace` -> **PIN** estimated cycles, DRAM BW utilisation,
   congestion %, binding resource.
3. `/perf-measure`: device Tracy `DEVICE KERNEL DURATION [ns]`, **median of the trial loop**, never
   a single untrialed number. `achieved = measured / target`.

### 9.3 Candidate ranking (Mode A, run at design time — see §1.3)

The winner is candidate 1 (2-D block, `WT_CHUNK` defaulted to `WT`). Candidates 2, 3 and 4 are
recorded in §1.3 with the property that lost each of them. **They are the fall-back if the winner
misses its predicted target at A0** — do not re-brainstorm, re-measure from that table.

### 9.4 The perf bench (`tests/ttnn/unit_tests/operations/tilize/_bench_tilize.py`)

Separate from the golden suite (whose INPUTS are deliberately tiny and must not grow). No PCC
assert. Runs under `--collect-noc-traces` + Tracy. Lever counterfactuals expressed as
`levers=dict(...)` arms so `eval/verify_levers.py` can find the knobs.

| # | Shape | Regime it proves | Derived blocking (bf16, 64 cores) | Which levers it is the proving ground for |
|---|---|---|---|---|
| **(a)** | `[1,1,2048,2048]` | **grid-filling square** — per-core DRAM efficiency with the grid already full | `NT_H=64, WT=64, n_chunks=1, WT_CHUNK=64`, **1 block/core** | A0/A1 baseline, A1 scaling, B9, A1-placement |
| **(b)** | `[1,1,32,16384]` | **wide/short — MANDATORY.** `NT_H=1 < NUM_CORES` | `NT_H=1, WT=512, n_chunks=64, WT_CHUNK=8`, **1 block/core, 64 cores** | the grid-fill gate. **If this runs on `< NUM_CORES` cores the A1 perf gate FAILS**, however good (a) looks. |
| **(c)** | `[1,1,8192,1024]` | **multi-block-per-core** | `NT_H=256, WT=32, n_chunks=1, WT_CHUNK=32`, **4 blocks/core** | **B8 (trid double-issue), L3 (`NT_BLK>1`), C16 (depth-2), `split_reader`.** Added deliberately: shapes (a) and (b) are both **one block per core**, and `master.md` **B8** states verbatim that neither trid double-issue nor split_reader *"can show a win on a one-block-per-core shape — there is no next block to overlap against. If every benched shape is one-block, that is a **bench** gap, not evidence the lever doesn't apply."* Without (c) those levers would be silently reasoned away. |
| **(d)** | `[1,1,32,64]` | **smallest regime** (matches the smallest aligned golden INPUTS cell, `feature_spec.py:252-255`) | `NT_H=1, WT=2, n_chunks=2, WT_CHUNK=1`, 2 cores, 1 tile/core | **`master.md` B0**: every per-core-overhead lever (**B5, B7, B8, B10, B13**) must be counterfactualed **here**, not only on (a). `eval/verify_levers.py` resolves the smallest-regime check against this. Also the regime where the `NoReconfigure` lever is worth 8-19%. |
| (e) | `[1,1,512,64]` L1 HEIGHT-sharded in/out, same spec | zero-copy sharded (A3) | 4 cores, `WT_CHUNK=2`, shard = the block | A3 re-targeting: the write is **local L1, loopback — not DRAM**; tt-npe must show **zero** output-side DRAM. Also where **`split_reader` (L4) becomes applicable**. |

Each bench arm sweeps `dtype in {bfloat16, float32}` and `use_double_buffer in {True, False}`.

### 9.5 DM lever checklist for THIS data path

Read `ttnn/ttnn/operations/examples/master.md` **Part 2** for the levers, **Part 1** for the
runnable on-device demo of each. The ones this data path implicates, and their Phase 0 disposition:

| Lever | Disposition at Phase 0 | Where it is proved |
|---|---|---|
| **A0** active-core count | **full grid, no knee cap** — A0 records this measured on tilize (~2.4x slower when capped) | (a), (b) |
| **A1** spread across the DRAM-facing axis | `row_wise=True`, day 1 | (a) |
| **A4** cliff-core specialization | **not needed** — `WT_CHUNK \| WT` removes the width cliff; the block-count remainder is handled by `split_work_to_cores`'s two groups running the same kernel | — |
| **B5** whole-page transactions | writes are whole tile pages; reads are whole sticks when `n_chunks==1` | (a), (c) |
| **B6** one-packet fast path (<=512 B WH) | `WT_CHUNK` derivation keeps the read chunk as large as the grid allows; on (b) it lands at exactly 512 B | (b) |
| **B7** one barrier per block | held by the helper (`inl:126`) and by the writer | (c), (d) |
| **B8** trid double-issue | **deferred** — needs a custom reader (L3) | **(c)** |
| **B9** reader NOC0 / writer NOC1 | day 1 | (a) |
| **B10** per-reader VC | open | (a) |
| **B13** `set_state`/`with_state` | open — the writer issues `WT_CHUNK` same-shape writes per block, a natural fit | (c), (d) |
| **C14** zero-copy CB alias | A3 (L1) | (e) |
| **C16** depth-2 CBs | day 1, with the A6 opt-out and the never-OOM fallback | (c), (e) |
| **D18/D19/D21** CT accessor args, address-only RT args | day 1 | program-cache test |
| **F24** `bfp8_pack_precise` | gate on **input** dtype (`out==bfloat8_b and in==float32`), not on output alone; ~1.4x measured on a `bf16->bfp8_b` tilize | A4 |
| **F26** `Fp32Mode::Lossless` | **structurally closed** — tilize is an FPU phase | — |
| **B12** multicast | **structurally impossible** (§7 rejection 4) | — |
| `split_reader` | **structurally inapplicable on DRAM->DRAM** (BRISC runs the writer); applicable on sharded output | (e) |

Every lever the kernel actually **uses** gets the `/perf-ceiling-dm` **Mode C** used-optimization
audit (predicted counterfactual -> measured delta -> keep/drop) recorded in the ledger. A7 runs
**Mode D** over the full list.

---

## 10. Key Risks and Gotchas

| Risk | Detail | Mitigation |
|---|---|---|
| CB page count vs granularity mismatch | Reader pushes tiles while compute waits rows (or vice versa) -> **hang** | TILE granularity everywhere; the two are paired in §7 and asserted at `tilize_helpers_dataflow.inl:105-108` and `tilize_helpers.inl:225-227` |
| `WT_CHUNK` not dividing `WT` | would need a cliff-width compute kernel and could straddle cores | §1.4 snaps `n_chunks` to an exact divisor of `WT` — a hard invariant, not a hope |
| Wide `W` OOM | a CB sized by `WT` is a latent OOM (`WT=512` bf16 depth-2 = 2 MB) | CB size is `CB_DEPTH * WT_CHUNK * tile_bytes`, and `WT_CHUNK` is clamped by `CB_L1_BUDGET`. **No CB is a function of a whole-op dimension.** |
| Short-wide shapes under-filling the grid | `NT_H=1` -> one core | the `n_want = ceil(NUM_CORES / NT_H)` term; gated by bench (b) |
| `pad_mode="auto"` on an aligned input | taking the pad reader would corrupt/slow the hot path | §5.1 keys on the **pad region being non-empty**, not on the mode string |
| Stale L1 in the pad region | the library reader leaves it (`tilize_helpers_dataflow.hpp:50-52`) | separate `R_PAD` reader that fills all **three** regions: W tail, H tail, whole pad tiles |
| Sub-word fill written once per 32-bit word | invisible with `pad_value=0`, garbage on a nonzero fill | replicate the value **twice** in the word for 2-byte dtypes, **four times** for 1-byte; pack in the **input** element format (never `output_dtype`) |
| Fill packed in `output_dtype` | garbage whenever a cast is also requested | the fill is materialized into `cb_input_sticks`, **before** the compute stage -> input element width, always |
| Logical shape widened to the pad target | golden checks both readback views | output logical shape := input logical shape; padded shape := target |
| Hardcoding 32 as the H tile dim | a tiny tile redefines H-alignment; `tag_alignment` measures against `TILE_H` | every H formula uses `TILE_H` (§4); the CB carries `TileDescriptor(TILE_H, 32)` and the reader derives it (`inl:75`) |
| uint8 with the full-tile row dim | **strided tile — every other row zero**, shape-correct and value-wrong, survives a loose numeric check | `FaceGeometry(face_r_dim=16, num_faces=4)` (per-face), plus the alignment-aware reader for `<64 B` sticks; golden compares uint8 **exactly** |
| fp32 output silently loses precision | `can_use_fast_tilize` **excludes fp32 output** (`tilize_helpers.inl:90-96`) — the fast path's `Read_32b=0` pack truncates to bf16 | fp32 output automatically takes the regular path; do not force `use_fast`. Note this changes the compute-side cost -> re-run the ceiling per dtype (A4) |
| Tiny tiles disable fast tilize | `can_use_fast_tilize` requires `dfb_has_32x32_tiles<output_dfb>()` (`inl:95`) | expected; T1 is correctness-gated, not perf-gated. Do not read the slowdown as a defect |
| Block-float input | `tilize_helpers_dataflow.inl:85` asserts `tile_size % tile_hw == 0`; `tilize_helpers.inl:174` asserts the input is not block-float | bfloat8_b is an **output-only** dtype (feature_spec.py:137-143) — never an input |
| Reconfigure with nothing to cast | ~150 ns fixed, **8-19%** on the ~1 us small sharded cases | compile-time `needs_cast` -> `NoReconfigure`; ledger lever |
| `split_work_to_cores` default | `row_wise=False` is column-wise, the documented ~2.9x trap | always pass `row_wise=True` |
| Retile on Wormhole | LLK support does not exist | **arch-gate and SKIP**, never fail. Do **not** arch-gate plain tiny tiles |
| Retile + pad | mutually exclusive by construction | `validate()` refuses the combination outright |

---

## 11. SUPPORTED / TARGET wiring

`feature_spec.py`'s TARGET is the ambition. `SUPPORTED` (implementer-owned) starts at the Phase 0
rectangle in the run requirements and grows per refinement. `op_design.md` refers to every axis by
the name TARGET already uses: `dtype`, `output_dtype`, `use_multicore`, `double_buffer`,
`shard_api`, `out_scheme`, `buffer`, `rank`, `orientation`, `tile_height`, `in_layout`,
`in_tile_height`, `pad_mode`, `pad_value`, `alignment`.

Two axis notes the design pins:

- **`double_buffer`** is a tagged axis (`tag_double_buffer`, defaulting to `True`) and therefore
  must appear in `SUPPORTED` from Phase 0 as `[True]`, so the `use_double_buffer=False` INPUTS
  cells xfail cleanly until A6 widens it to `[False, True]`. Omitting the key entirely would make
  `validate()` KeyError.
- **Index/sentinel convention**: `in_tile_height` uses the string `"none"` sentinel for a
  ROW_MAJOR input. `SUPPORTED["in_tile_height"]` must list `"none"` at every phase — it is always
  legal.

The `memory_layout` ambition (`shard_api` / `out_scheme` / `buffer` here) is the **physical
realization of the logical shards the work-split already defines**, and every one of them is a
**knob-turn** for this op, because §1.2 proves there is no dependent axis: a HEIGHT shard cuts
`NT_H` (independent), a WIDTH shard cuts `WT` (independent), a BLOCK/nd shard cuts both — none
needs a combine. What each costs is a **placement** change (§5.2 / §6), not a new algorithm. That
is why L1/L2 in the lamp are placement scheme-changes rather than algorithmic ones.

### Structural impossibilities (candidates for the user to fold into `feature_spec.py` via `/golden-tests` — do NOT edit it here)

Only `dtype` and `output_dtype` are free cartesian axes (everything else is scenario-projected, so
incoherent combinations cannot be generated). One candidate INVALID cell crossing those two with a
projected axis appears to be missing:

- `{"output_dtype": ttnn.bfloat8_b, "tile_height": 16 / 8 / 4 / 2 / 1}` — a block-float format
  defines its shared exponent over the tile's fixed 16x16 face structure, so a sub-32-row tile has
  no legal `bfloat8_b` encoding. This is a property of the format, not of kernel effort. Confirmed
  in the same direction by `tilize_helpers.inl:95` (`dfb_has_32x32_tiles<output_dfb>()` is a
  precondition of the whole fast path) and `inl:174` (block-float is structurally excluded from the
  tilize reinterpretation).

---

## 12. Hardware Constraints checklist

- [x] CB sync: push count == wait count for every CB (§6)
- [x] CB ownership: exactly one producer, one consumer per CB — including the drain-only writer in `P_LOCAL_SHARD` (§6)
- [x] Reduce scaler: **n/a** — this op performs no reduction
- [x] DEST: **n/a** — the tilize LLK budgets its own DEST; `WT_CHUNK` is an L1 knob (§7 rejection 5)
- [x] Sequential helper intermediates: **n/a** — no compute->compute CB
- [x] Page sizes aligned to tile size; RM reads are `WT_CHUNK*32*elem` B, always a multiple of 32 B (>= DRAM read alignment, `master.md` **B11**)
- [x] Tile CBs count pages in **tiles** (TILE granularity, §7 rejection 3)
- [x] All `wait_front` calls on a CB use the same page count (`WT_CHUNK`)
- [x] `compute_kernel_hw_startup()` before any helper usage (§8 phase 0)
- [x] Tile geometry alignment-aware: `ceil`, per-image, against `TILE_H` not 32 (§4)
- [x] No knob is a constant: `CB_DEPTH`, `NT_BLK`, `WT_CHUNK`, `NUM_CORES` are all parameters with a single source in `derive_blocking()` (§1.4)
- [x] No CB is a function of a whole-op dimension (§6)
- [x] >1 regime -> selection function pinned + regime-pinned tests required (§5)

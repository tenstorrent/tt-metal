# Operation Design: tilize

`tilize` re-lays a ROW_MAJOR tensor into TILE layout. No arithmetic. The design below is a
**specification**: every statement is a decision the implementer executes, not an option to weigh.

---

## 1. Blocking Model

This section is decided **first**; Work Distribution (§5) is its realization and everything downstream
references it.

### 1.1 The op's axes and their characters

The output is a grid of tiles. Name the axes off *this* op's math:

| # | Axis (op-specific name) | Extent | Character | Why |
|---|---|---|---|---|
| 1 | **tile-row** `r` — index of a `tile_h`-stick group of the folded height | `nt_h = prod(shape[:-2]) * ceil(H / tile_h)` | **independent** | An output tile-row is produced from exactly `tile_h` source sticks and no others; no result spans two tile-rows. |
| 2 | **tile-column** `c` — index of a 32-element column group of each stick | `Wt = ceil(W / 32)` | **independent** | Output tile `(r,c)` reads only bytes `[c*32, c*32+32)` of its `tile_h` sticks; no result spans two tile-columns. |
| 3 | **stick-within-tile-row** `s` | `tile_h` (32, or the requested tiny-tile height) | **dependent** | One output tile *is* the interleaving of all `tile_h` sticks — a result spans the whole axis. The combine is the `tilize_block` LLK itself. |
| 4 | **element-within-stick / face** | 32 | **dependent** (folded into #3) | The four faces of a tile are a permutation of the same `tile_h × 32` element block. |

**Reuse-shared axes: none.** Every input byte contributes to exactly one output byte — the map is a
bijection. This is the single most consequential structural fact about the op:

- The **operand-reuse check** over `(operand, chosen split)` pairs is empty. `tilize` has one input
  operand and it varies along *both* split axes, so no core in any split re-reads bytes another core
  reads. There is **no broadcast / mcast opportunity anywhere in this op** — `mcast_pipe.hpp` is
  correctly unused, and `master.md` lever **B12** (`master.md:428`) is structurally not-applicable, not
  merely deferred.
- Consequently the op has **no arithmetic intensity**: total bytes moved is fixed at
  `2 × tensor_bytes` (one read + one write) for the interleaved path and `0` for the same-spec sharded
  path. The *only* remaining performance levers are **transaction shape, placement, overlap, and core
  count** — which is exactly why §8 is a data-movement plan and not a compute plan.

### 1.2 Ranking all candidate splits (qualitative, bytes/fan-out — no ns)

| Rank | Candidate split | Bytes moved | Transaction shape | Verdict |
|---|---|---|---|---|
| 1 | **tile-row** (axis 1) across cores | `2 × tensor_bytes` (each byte once) | read = one whole stick (`W*elem` B, up to 4096 B); write = whole 2048 B tile pages | **Cheapest.** Maximally coalesced on both sides. But per-core CB scales with `Wt` → unbounded L1, so it must be *combined* with a bounded column block, not used alone. |
| 2 | **tile-column** (axis 2) across cores | `2 × tensor_bytes` (each byte once — cores take **disjoint byte ranges of the same page**, not duplicate bytes) | read = `WT_BLOCK*32*elem` B at a byte offset inside each stick page; write = whole tile pages | **Same bytes, smaller reads.** Costs transaction size, not bandwidth. Required — it is the *only* parallelism when `nt_h < grid_cores` (`[1,1,32,16384]` has `nt_h = 1`). |
| 3 | **stick-within-tile-row** (axis 3, the dependent axis) across cores, with a NoC combine | `2 × tensor_bytes` **plus a full extra L1→L1 pass** to assemble partial tiles across cores | read = `tile_h/k` sticks per core; combine = cross-core face gather | **Rejected, and never needed.** The product of the two independent axes (`nt_h × Wt` ≥ `total_tiles`) exceeds the grid for any tensor with ≥ 64 tiles, and for smaller tensors the op is dispatch-bound anyway. Lamp recorded below as *anticipated-and-refuted*: Phase 0's structure does not need it and does not foreclose it (a per-block `(s_start, s_count)` runtime pair would be the only addition). |

**Primary split: the 2-D product of axes 1 and 2.** Both are independent, so this is a knob-turn with
**no cross-core communication**, and it is realized in **Phase 0**.

**A work-split is a logical shard.** Cutting axis 1 is a logical **height** shard; cutting axis 2 a
logical **width** shard; cutting both a logical **block** shard. Because both axes are *independent for
this op*, every one of those flavors is a **knob-turn**, not a scheme-change — the familiar "width shard
needs a combine" pairing does **not** apply to `tilize`, because nothing reduces along `W`. This is why
every `memory_layout` value in `TARGET` (`feature_spec.py:169`) is reachable by *placement* alone.

### 1.3 The three knobs

| Axis | Character | Block-factor knob | Phase 0 value | Core-assignment | Later unlock |
|---|---|---|---|---|---|
| tile-row `r` | independent | `NT_ROWS_PER_BLOCK` — **pinned at 1 by the LLK**: `compute_kernel_lib::tilize` calls the LLK once per `1 × block_width_tiles` tile-row (`tilize_helpers.hpp:121-125`) | `1` | contiguous runs of the linear block index, spread over the grid by `split_work_to_cores(..., row_wise=True)` | knob-turn (raise only if a multi-tile-row LLK block ever lands) |
| tile-column `c` | independent | **`WT_BLOCK`** — tiles per compute block; the `block_width_tiles` template param of the tilize helper (`tilize_helpers.hpp:188`) | `min(Wt, WT_BLOCK_MAX)`, `WT_BLOCK_MAX = max(2, TARGET_READ_BYTES // (32*elem_size))` → **8** for bf16 | the same linear block index — column-blocks are the *outer* dimension of the linearization, so a core owning one column-block owns a run of tile-rows within it | knob-turn (sweep `TARGET_READ_BYTES` ∈ {256, 512, 1024, 2048}) |
| stick-within-tile-row `s` | dependent | `tile_h` — **pinned by the tile format** (`unpack_tile_r_dim[cb_input_sticks]`, read by the reader at `tilize_helpers_dataflow.inl:75`) | `32` (or the `tile=` height) | single-core by construction | scheme-change (cross-core face combine) — **refuted**, see §1.2 rank 3 |

**Block-size default is coarse, not minimal — justified.** `WT_BLOCK = 8` for bf16 is **not** the
minimal unit (`1`); it is set in **bytes** so every dtype lands on the same measured sweet spot:

- `TARGET_READ_BYTES = 512` puts every read on the **one-packet NoC fast path** (`NOC_MAX_BURST_SIZE`
  = 512 B on WH; `master.md:409-411` lever **B6**).
- 8 tiles per block is the measured batching sweet spot for reads-per-barrier
  (`double_buffer/double_buffer.py:205` default `block = 8`; `double_buffer/report.md:29-42` shows
  `block=4`/`8` are the two best cells and `block=32` regresses; `master.md:411-412` lever **B7**).
- `master.md:358-361` records the *measured refutation* of the minimal choice on this very op: at
  **64 B per page** (`WT_BLOCK = 1`) tilize is **transaction-rate bound** and cannot reach DRAM
  bandwidth at any core count.
- The knob is a **byte target**, so fp32 gets `WT_BLOCK_MAX = 4` and uint8 gets `16` automatically —
  one source of truth, no per-dtype literal.

**Buffer-depth knobs** (a *distinct* knob from block factor — it buys read/compute/write overlap, not
reuse):

| CB | Depth knob | Phase 0 value | Rationale |
|---|---|---|---|
| `cb_input_sticks` | `CB_DEPTH` | **2** | `double_buffer/report.md:29-42`: batching alone plateaus at ~13 GB/s/core; depth-2 lifts it to 17.9 GB/s (**2.78×** combined with `block=4`). The two levers **compound**. |
| `cb_output_tiles` | `CB_DEPTH` | **2** | Same knob, same source of truth — lets the writer drain block *n* while compute fills *n+1*. |
| both, same-spec sharded zero-copy | `CB_DEPTH` | **1 (forced)** | The CB *is* the resident shard buffer; there is no second buffer to alternate into. |

`CB_DEPTH` is a **single host expression**:
`CB_DEPTH = 2 if (use_double_buffer and depth2_fits_l1 and not zero_copy) else 1`. Requirement A6's
auto-fallback ("use only if possible") is that `and depth2_fits_l1` clause, matching the cited
precedent `concat/device/concat_program_factory.cpp:111` and `master.md:443-445` lever **C16**.

### 1.4 The scheme Phase 0 commits to

> **One block = one output tile-row × `WT_BLOCK` output tile-columns.** Blocks are linearized
> `b = wchunk * nt_h + r` and that linear space is spread across the grid by
> `split_work_to_cores(grid, nt_h * n_wchunks, row_wise=True)`.

This linearization is the whole design, and it is chosen because it **subsumes both distribution
regimes with no gate expression at all**:

| Regime | `n_wchunks` | What the linearization degenerates to | Cores used |
|---|---|---|---|
| tall-narrow / square (`Wt ≤ WT_BLOCK_MAX`) | `1` | `b == r` — **exactly** the pure tile-row (height) split | `min(nt_h, grid)` |
| wide-short (`nt_h = 1`) | `ceil(Wt/WT_BLOCK)` | `b == wchunk` — **exactly** the pure tile-column (width) split | `min(n_wchunks, grid)` |
| general | `ceil(Wt/WT_BLOCK)` | a 2-D `(tile-row range × tile-column range)` rectangle per core | `min(nt_h*n_wchunks, grid)` |

`distribution_gate`'s no-regression property (`distribution_gate/report.md:33-35`: gated is
byte-identical to height_split on shapes height already saturated) is obtained **structurally** here
rather than by a runtime predicate: when `Wt ≤ WT_BLOCK_MAX` the block index *is* the tile-row index, so
the height-split path is byte-identical by construction. The prompt's requirement ("MUST also split
along the width axis when `nt_h < grid_cores`") is satisfied unconditionally, not behind a gate.

**Every knob is a parameter, never an inlined constant** — including the core count. `use_multicore=False`
is **not a second kernel**: it sets `grid_cores = 1`, the trivial value of the core-assignment
parameter, and the identical reader/compute/writer run on one core. There is exactly one program
factory. Phase 0's SUPPORTED rectangle only *accepts* `use_multicore=False`; refinement A1 flips the
SUPPORTED entry and changes **no kernel code**.

### 1.5 Single source of truth for every knob (DRY)

Defined once in `tilize_program_descriptor.py`; every dependent quantity is computed *from* it.

| Knob | Symbol | Defined as | Everything derived from it |
|---|---|---|---|
| read transaction byte target | `TARGET_READ_BYTES` | module constant `512` | `WT_BLOCK_MAX` |
| max block width | `WT_BLOCK_MAX` | `max(2, TARGET_READ_BYTES // (32*elem_size))` | `WT_BLOCK` |
| block width (tiles) | `WT_BLOCK` | `Wt_shard` (zero-copy) else `min(Wt, WT_BLOCK_MAX)` | compute CT arg, `n_wchunks`, `row_bytes`, both CBs' `num_pages`, reader/writer loop bounds |
| tail block width | `WT_TAIL` | `Wt - (n_wchunks-1)*WT_BLOCK` if `Wt % WT_BLOCK` else `WT_BLOCK` | 2nd compute CT instantiation |
| column-blocks per tile-row | `n_wchunks` | `ceil(Wt / WT_BLOCK)` | `total_blocks`, `tail_block_start` |
| tile-rows | `nt_h` | `prod(shape[:-2]) * ceil(H / tile_h)` | `total_blocks`, block→`(r, wchunk)` decode |
| tile-columns | `Wt` | `ceil(W / 32)` | `n_wchunks`, output page index |
| total blocks | `total_blocks` | `nt_h * n_wchunks` | `split_work_to_cores` |
| tail region start | `tail_block_start` | `(n_wchunks-1) * nt_h` | per-core `(n_full, n_tail)` |
| grid cores | `grid_cores` | `1 if not use_multicore else grid.x*grid.y` | core assignment |
| CB depth | `CB_DEPTH` | `2 if use_double_buffer and depth2_fits_l1 and not zero_copy else 1` | both CBs' `num_pages` |
| cast flag | `NEEDS_CAST` | `output_dtype != input_dtype` | compute's `ReconfigureRegisterDatatypeMode` |
| pad flag | `PAD_ENABLED` | `pad_mode != "none"` | reader code-path select |
| pad fill word | `PAD_WORD` | fill packed in the **input** element format, replicated to 32 bits | reader fill store |

**No CB page count and no loop bound may restate a literal that appears above.** Turning
`TARGET_READ_BYTES` must be a one-line change.

### 1.6 Lamp — scheme-changes Phase 0 leaves room for

| Lamp | Class | Kept reachable by | Refinement |
|---|---|---|---|
| **Physical shard consumed in place** (zero-copy CB aliased onto the resident L1 shard, zero DRAM traffic) | knob-turn + placement | The block-width knob already accepts a shard-given value (`WT_BLOCK = Wt_shard`); the CB descriptor construction is the only thing that changes (`ttnn.cb_descriptor_from_sharded_tensor`, `program_descriptors.cpp:517-556`). Reader/compute/writer *sources* are unchanged. | A3 |
| **Interleaved ↔ sharded crossover** (split reader / split writer) | knob-turn | Core assignment is a parameter; the crossover pins it to the shard's own cores (`master.md:383-386` lever **A2**) instead of `split_work_to_cores`. One side keeps its `TensorAccessor`, the other becomes a CB alias. | A3b |
| **Cross-spec reshard** (in-spec ≠ out-spec, uneven shards) | **scheme-change** — the only genuine one in the op | This is the one place a cross-core L1→L1 gather appears. Phase 0 does not foreclose it: the block index → `(r, wchunk)` decode is already a host-computed mapping, so a general host-computed page→(core, offset) table drops into the same slot. See §4.3 for the Tensix-to-Tensix contract of the *unlocked* scheme. | A3c |
| **Wide-W CB bound on the general/sharded path** | knob-turn | `WT_BLOCK` is already `min(Wt, WT_BLOCK_MAX)` on every non-zero-copy path, so per-core CB L1 is **already** constant in `W`. A3d is a *no-op on the interleaved path* and only needs the same clamp applied to a wide HEIGHT-shard crossover. | A3d |
| **Cross-core combine on the dependent axis** | scheme-change | **Anticipated and refuted** (§1.2 rank 3). Not needed at any shape; recorded so a later reader does not re-litigate it. | — |
| **Padded path** (three fill regions) | knob-turn behind a CT flag | `PAD_ENABLED` selects a second reader code path inside one reader source; the aligned path is byte-identical when the flag is 0. Track P cannot regress Track A **structurally**. | P1–P5 |
| **Tiny tile / retile geometry** | knob-turn (tiny) / new reader (retile) | `tile_h` is read from the CB's tile descriptor by both the reader (`tilize_helpers_dataflow.inl:75`) and the LLK, so a tiny tile is a **CB-descriptor change only**. Retile swaps the reader; compute and writer are unchanged because the retile reader emits the same `cb_input_sticks` contract. | T1 / T2 |

---

## 2. Overview

| Field | Value |
|-------|-------|
| Classification | `data_movement` (layout conversion with a compute stage) |
| Goal | Re-lay a ROW_MAJOR tensor into TILE layout (32-wide tiles of `tile_h` rows, four 16×16 faces), optionally padding H/W up to a tile multiple with a caller-supplied fill, optionally casting the output dtype. |
| Math | `output[tile_index(i)] = input[i]` — a bijection on byte positions; values unchanged (value-preserving cast when `dtype=` narrows). Pad positions = `pad_value`. |
| Mode | Hybrid (helper-driven compute; helper-driven aligned reader; raw dataflow for the pad / tile-page-write / retile paths, each justified in §6) |
| References | `tilize_helpers.hpp`, `tilize_helpers_dataflow.hpp`, `eval/golden_tests/tilize/feature_spec.py`, `ttnn/ttnn/operations/examples/master.md`, `METALIUM_GUIDE.md` |

### Parameters

| Name | Type | Required | Valid range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | ROW_MAJOR (TILE on the retile path), rank ≥ 2 (rank 0 allowed when padding) | — | — |
| `memory_config` | `ttnn.MemoryConfig \| None` | no | interleaved DRAM/L1, or L1-sharded (legacy 2D or nd) | input's | host |
| `dtype` | `ttnn.DataType \| None` | no (kw-only) | bf16, fp32, bf8b (out only), uint32/uint16/int32, uint8; same cast family as the input | input's | CT (`NEEDS_CAST`) + CB format |
| `use_multicore` | `bool` | no (kw-only) | — | `True` | host (`grid_cores`) |
| `use_double_buffer` | `bool` | no (kw-only) | — | `True` | host (`CB_DEPTH`) |
| `output_padded_shape` | `list[int] \| ttnn.Shape \| None` | no (kw-only) | ≥ input in every dim; last two dims tile multiples | `None` | host |
| `pad_value` | `float \| int \| None` | no (kw-only) | representable in the **input** element format | `None` | CT/RT (`PAD_WORD`) |
| `tile` | `ttnn.Tile \| None` | no (kw-only) | width **always 32**; height ∈ {32,16,8,4,2,1} | `Tile([32,32])` | CB tile descriptor |

Derived pad mode (the axis name `feature_spec.py:206` uses):

| `output_padded_shape` | `pad_value` | `pad_mode` | Pad target |
|---|---|---|---|
| absent | absent | `"none"` | none — an input whose last two dims are not tile multiples **raises** |
| absent | present | `"auto"` | input shape with the last two dims rounded up to the next multiple of (`tile_h`, 32) |
| present | present | `"explicit"` | `output_padded_shape` (validated) |

`output_padded_shape` present with `pad_value` absent → raise. `tile` with a TILE-layout input **and**
any pad argument → raise (mutually exclusive; `feature_spec.py:708-713`).

### Tensors

#### Input

| Property | Requirement |
|----------|-------------|
| Shape | rank ≥ 2 (rank 0 accepted only when a pad is requested); last two dims tile multiples unless padding is requested |
| Dtype | bfloat16 (primary), float32, uint32/uint16/int32, uint8 |
| Layout | `ROW_MAJOR_LAYOUT` (`TILE_LAYOUT` on the retile path only) |
| Memory | INTERLEAVED (DRAM or L1) or SHARDED (L1, ROW_MAJOR-sharded) |

#### Output

| Property | Value |
|----------|-------|
| Logical shape | **identical to the input's logical shape** — padding never changes it |
| Padded shape | the input's padded shape when `pad_mode == "none"`; the pad target otherwise |
| Dtype | input's, or `dtype=` (value-preserving cast) |
| Layout | `TILE_LAYOUT` with tile `(tile_h, 32)` |
| Memory | input's, or `memory_config=` |

---

## 3. Support contract (registry model)

Axis names are exactly those `feature_spec.py`'s TARGET uses, so `validate()` gates on the same names
`INPUT_TAGGERS` produces.

`INPUT_TAGGERS`, in declaration order: `tag_use_multicore, tag_shard_api, tag_out_scheme, tag_buffer,
tag_rank, tag_double_buffer, tag_pad_mode, tag_pad_value, tag_alignment, tag_orientation,
tag_tile_height, tag_in_layout, tag_in_tile_height`. `dtype` and `output_dtype` are free cartesian axes.

Implementation notes the implementer must honor (each is a spec, not a hint):

- `tag_alignment` measures H against the **scenario's tile height**, not a hardcoded 32
  (`feature_spec.py:96-99`). The op's own alignment check must use `tile_h` for the same reason, or the
  op and the registry will disagree about which cells the op supports.
- `tag_in_tile_height` must return the `"none"` sentinel **exactly** when the input is ROW_MAJOR, and
  `SUPPORTED["in_tile_height"]` must list `"none"` as an accepted value — it is always legal.
- `tag_orientation` returns `"none"` when both sides are interleaved.
- **Index/sign convention:** the op has no `dim` axis, so no canonicalization is needed. `pad_value` is
  bucketed by *sign* (`"none"/"zero"/"positive"/"negative"`), so `validate()` gates on the bucket.
- **`use_double_buffer` is tagged but not passed.** `eval/golden_tests/tilize/helpers.py::run_tilize`
  passes `memory_config`, `dtype`, `use_multicore`, the pad kwargs and `tile` — it does **not** pass
  `use_double_buffer`. So the `double_buffer=False` INPUTS cells are tagged `False` while the op runs at
  its `True` default. A6's golden gate is therefore "declare the axis and *accept* the `False` cells";
  the *measured* half of A6 (L1 bytes/core at depth 1 vs 2, and the perf delta) comes from the perf
  bench, where the kwarg is passed explicitly. Do not hunt for a kwarg the harness never sends.
- The op does **not** declare `INVALID` and `validate()` does not check it.

Phase 0 `SUPPORTED` is exactly the rectangle in the prompt. `EXCLUSIONS` at Phase 0 must at minimum
carry `{"use_multicore": False, "shard_api": "legacy_2d"}` and `{"use_multicore": False,
"shard_api": "nd"}` — a sharded input is inherently multi-core (its cores are fixed by the shard spec),
so those cells are refused *for now* rather than unsupported forever.

### Refusal message contract

The two **structural** refusals are asserted by the acceptance test through the repo's `expect_error`
fixture (`conftest.py:880-899`), which matches the message as a regex. Both messages **must** contain
the substring `pad` (matched case-insensitively as `(?i)pad`):

| Refusal | Raised when | Message must mention |
|---|---|---|
| padding is never implicit | `pad_mode == "none"` and the last two dims are not multiples of (`tile_h`, 32) | `pad` — and name the two arguments that would enable it (`pad_value`, `output_padded_shape`) |
| retile and padding are mutually exclusive | `in_layout == TILE` and any pad argument is present | `pad` |

Support refusals (a cell outside `SUPPORTED` / matching `EXCLUSIONS`) are a *different* mechanism and go
through `ttnn.operations._op_contract` (`UnsupportedAxisValue` / `ExcludedCell`), not these two.

### Structural impossibilities

**None found beyond what `feature_spec.py:682-713` already declares.** Two watch items for the
implementer — neither is an INVALID candidate, both are op-side notes:

1. `{"dtype": uint8, "tile_height": 1}` — a 1×32 uint8 tile is 32 bytes, i.e. exactly one DRAM read
   alignment unit (`ttnn.get_dram_alignment() == 32`) and 2× the L1 alignment unit. Legal, but the
   narrowest page in the whole universe; if it fails at T1 it is an alignment bug, not an impossibility.
2. `{"double_buffer": True, "out_scheme": <same-spec sharded>}` — depth-2 is *meaningless* on the
   zero-copy path (the CB is the shard). This is a forced-to-1 no-op, not an impossible cell;
   `CB_DEPTH`'s definition (§1.5) already encodes it.

---

## 4. Dataflow Strategy

### 4.1 The data path, per scheme — and where the block physically lives

The distinction below **is** the implementation, not a tuning of it. A physical shard's block is already
in the core's own L1; re-reading it through a `TensorAccessor` would re-fetch over the NoC data the core
already holds, and is not designing sharding.

| Scheme | Input block lives… | Reader | Compute | Writer | DRAM traffic |
|---|---|---|---|---|---|
| `interleaved_2d` | in DRAM/L1 interleaved — the block genuinely lives there | `TensorAccessor` NoC read of `tile_h` sticks → `cb_input_sticks` | `tilize` → `cb_output_tiles` | `TensorAccessor` NoC write of whole tile pages | read + write |
| `sharded_zero_copy` | **in this core's own L1** (pre-placed by the caller) | **no NoC read** — `cb_input_sticks` is *aliased onto the input shard buffer*; reader only arms the CB (`cb_push_back` of the whole shard) | `tilize` packs straight into the aliased output shard | **no NoC write** — reader/writer only run the CB handshake | **zero, both sides** |
| `sharded_crossover` | one side local L1, the other interleaved | sharded side = CB alias; interleaved side = `TensorAccessor` (split across both DM RISCs) | unchanged | mirror of the reader | one side only |
| `sharded_reshard` | remote L1 (another core's shard) | general `TensorAccessor` cross-core L1→L1 read, host-computed page map | unchanged | CB alias or accessor | **zero** (never stage through DRAM) |
| `retile` | in DRAM/L1 as TILE pages | face-walking reader assembles RM sticks into `cb_input_sticks` | unchanged | unchanged | read + write |

Format at each stage, aligned interleaved path:

```
DRAM (RM sticks, page = W*elem bytes)
  --NoC0 read, tile_h reads of WT_BLOCK*32*elem bytes, one barrier per block-->
L1 cb_input_sticks (RM block: tile_h sticks x WT_BLOCK*32 elems, contiguous, page = tile_size)
  --tilize_block / fast_tilize_block on UNPACK+MATH+PACK-->
L1 cb_output_tiles (WT_BLOCK tile pages)
  --NoC1 write, WT_BLOCK writes of tile_size bytes, one barrier per block-->
DRAM (TILE pages)
```

The reader lays `tile_h` sticks at stride `padded_row_bytes = WT_BLOCK*32*elem` starting at
`get_write_ptr(cb_input_sticks)` (`tilize_helpers_dataflow.inl:117-124`); `tile_h * WT_BLOCK*32*elem`
is exactly `WT_BLOCK` tile-sized pages, so the CB accounting is symmetric.

### 4.2 NoC and placement decisions (Phase 0, not deferred)

| Decision | Value | Source |
|---|---|---|
| Reader RISC-V / NoC | NCRISC, **NoC0** (`ReaderConfigDescriptor()` default) | `master.md:421-423` lever **B9**; `noc_placement/README.md:9-11` measures **2.5–4.8×** vs the reverse |
| Writer RISC-V / NoC | BRISC, **NoC1** (`WriterConfigDescriptor()` default) | same |
| Core line orientation | `split_work_to_cores(..., row_wise=True)` | `master.md:379-382` lever **A1**; `noc_placement/README.md:9-11` measures **~2.9×** for row over the *default* column line |
| Barriers | **one per block** (`tile_h` reads / `WT_BLOCK` writes batched) | `master.md:411-412` lever **B7**; `double_buffer/report.md:29-42` |
| Active-core count | **full grid** — do **not** apply a bandwidth-knee cap | `master.md:358-361`: the knee cap was implemented, measured **~2.4× slower** on *this op*, and refuted; tilize's reader is transaction-rate bound |
| Kernel structure | three separate kernels; do **not** fold reader/writer into compute | `master.md:434-440` lever **C14** + `zero_copy_fold/report.md`: folding is **0.74×** at 2 tiles/core |
| `TensorAccessorArgs` | **compile-time**, appended **last** after all scalar CT args | `master.md:450-452` lever **D18** |
| Runtime args | only buffer base addresses + per-core block range vary → program-cache friendly | `master.md:452-453` lever **D19** |

### 4.3 Tensix-to-Tensix contract for the *unlocked* cross-spec reshard scheme (A3c)

Phase 0 does not use this; it is specified now so A3c is a placement change rather than a redesign.

| Property | Contract |
|---|---|
| Topology | **Pull, not push.** Each *output* core owns its output shard and reads the input pages it needs from whichever input core holds them. No sender-side coordination, no semaphores, no mcast — every input page is read by exactly one output core (§1.1: the map is a bijection, so there is no fan-out to multicast). |
| Addressing | A host-computed table: for each output core, a list of `(src_core_x, src_core_y, src_l1_offset, byte_len)` runs. Passed as runtime args; the input shard buffer's base address is a single common runtime arg. |
| Synchronization | **None required beyond program-level.** Input shards are written by the *previous* op and are read-only for the duration of this program; TT-Metal's program barrier between ops is the only ordering guarantee needed. No semaphore, no `mcast_pipe`. |
| Ordering | Reads within an output block are issued back-to-back with **one barrier per block**, identical to the interleaved reader. Ordering across blocks is irrelevant (disjoint outputs). |
| Why no DRAM staging | The whole point of A3c's gate. Every run is an L1→L1 NoC read; the design never allocates a DRAM intermediate. |

---

## 5. Work Distribution

The Blocking Model's core-assignment made concrete.

| Field | Value |
|-------|-------|
| Work unit | **one block** = one output tile-row × `WT_BLOCK` output tile-columns |
| Grid | `grid_cores = 1` if `use_multicore == False` else `device.compute_with_storage_grid_size()` (a parameter, never inlined); `sharded_*` schemes pin the grid to the shard's own cores (`master.md:383-386`) |
| Total work | `total_blocks = nt_h * n_wchunks` |
| Split call | `ttnn.split_work_to_cores(grid, total_blocks, row_wise=True)` → `(num_cores, all_cores, group_1, group_2, per_core_1, per_core_2)` (`ttnn/cpp/ttnn-nanobind/operations/core.cpp:467-471, 483-490`) |
| Per-core work | a contiguous linear range `[b0, b0 + nb)`; `b0` accumulated over `grid_to_cores(num_cores, grid.x, grid.y, row_wise=True)` in the same order the split produced |
| Remainder | handled by `split_work_to_cores`' two groups (group 1 does more; group 2 is empty on an even division — `tt_metal/api/tt-metalium/work_split.hpp:39-47`). Both groups run the **same** kernel sources; only `nb` differs, and `nb` is a runtime arg. |

### 5.1 Alignment-aware geometry (mandatory `ceil`, per-image)

Written with `ceil` from the start even though Phase 0 only accepts tile-aligned shapes — the alignment
refinement is exactly what hits the boundary.

```
tile_h  = tile.height          # 32 by default; 16/8/4/2/1 for a tiny tile
elem    = element_size(input_dtype)

H, W    = shape[-2], shape[-1]           # rank 0/1 -> synthesized as 1 (pad path only)
Hp      = ceil(H / tile_h) * tile_h      # padded rows PER IMAGE  -- ceil, not floor
Wp      = ceil(W / 32)     * 32
nimg    = prod(shape[:-2])               # rank-agnostic fold; rank 2 -> 1

nt_h    = nimg * (Hp // tile_h)          # tile-rows.  NOT floor(nimg*H/tile_h)
Wt      = Wp // 32                       # tile-columns
```

`nt_h = nimg * ceil(H/tile_h)` is **not** `floor(nimg*H/tile_h)`: in TILE layout each image is
tile-padded independently. Rank-5 needs no new code — `nimg = prod(shape[:-2])` is rank-agnostic, which
is the only thing refinement A5 tests.

### 5.2 Block decode (kernel side, from two runtime args `b0`, `nb`)

```
for i in 0 .. nb-1:
    b      = b0 + i
    wchunk = b / nt_h                    # column-block index
    r      = b % nt_h                    # global tile-row index
    w      = (wchunk == n_wchunks-1) ? WT_TAIL : WT_BLOCK     # tiles this block
    c0     = wchunk * WT_BLOCK                                # first tile-column
    # source sticks (aligned path): global output row g = r*tile_h + s
    #   img = g / Hp ; row_in_img = g % Hp ; src_stick = img*H + row_in_img
    #   when H % tile_h == 0, Hp == H and src_stick == g  (the aligned fast path)
    # output tile page index for tile t of this block: r*Wt + c0 + t
```

Because the tail column-block is `wchunk == n_wchunks-1`, its blocks occupy the **contiguous suffix**
`[tail_block_start, total_blocks)` of the linear space. A core's contiguous `[b0, b0+nb)` therefore
crosses the full/tail boundary **at most once**, so the host passes
`n_full = clamp(tail_block_start - b0, 0, nb)` and `n_tail = nb - n_full`, and compute runs at most two
helper calls. No per-core kernel duplication, no cliff-core kernel variant.

### 5.3 Regime-selection function (pinned — regime-pinned tests required)

```python
def select_regime(in_t, out_mem, pad_mode, tile) -> str:
    if in_t.layout == ttnn.TILE_LAYOUT:            return "retile"
    in_sh, out_sh = in_t.memory_config.is_sharded(), out_mem.is_sharded()
    if in_sh and out_sh:
        return ("sharded_zero_copy"
                if (pad_mode == "none" and _same_shard_placement(in_t.memory_config, out_mem))
                else "sharded_reshard")
    if in_sh or out_sh:                            return "sharded_crossover"
    return "interleaved_2d"

def _same_shard_placement(a, b) -> bool:
    # zero-copy requires: same buffer type (L1), same core grid, same orientation,
    # and equal shard shapes (the RM shard's (h, w) == the TILE shard's (h, w)).
    # nd vs legacy is irrelevant once the *placement* matches.
    return (a.buffer_type == b.buffer_type == ttnn.BufferType.L1
            and _grid_of(a) == _grid_of(b)
            and _orientation_of(a) == _orientation_of(b)
            and _shard_hw(a) == _shard_hw(b))
```

Orthogonal compile-time flags, independent of the regime: `PAD_ENABLED`, `NEEDS_CAST`, `CB_DEPTH`,
`tile_h`, `WT_BLOCK`, `WT_TAIL`.

**Regime-pinned tests are required.** A regime that only triggers on some grids passes on one device
and fails on another: `sharded_zero_copy` needs a cell whose shard grid fits the *smallest* target
part, and `sharded_reshard` a cell whose in/out grids genuinely differ (`feature_spec.py:357-361`).
Assert the selected regime directly, not just the output values.

### 5.4 Grid-fill verification per regime (the A1 gate)

| Regime | Shape | `nt_h` | `Wt` | `n_wchunks` | `total_blocks` | Cores (8×8 grid) |
|---|---|---|---|---|---|---|
| grid-filling square (perf) | `[1,1,2048,2048]` | 64 | 64 | 8 | 512 | **64**, 8 blocks/core |
| wide-short (perf, **mandatory**) | `[1,1,32,16384]` | 1 | 512 | 64 | 64 | **64**, 1 block/core |
| wide-short (golden) | `[1,1,32,4096]` | 1 | 128 | 16 | 16 | 16 |
| tall-narrow (golden) | `[1,1,2048,64]` | 64 | 2 | 1 | 64 | **64** — degenerates to the pure height split |
| tiny (golden, smallest) | `[1,1,32,64]` | 1 | 2 | 1 | 1 | 1 |
| row vector (golden) | `[1,1,1,4096]` (pad→32) | 1 | 128 | 16 | 16 | 16 |

If `[1,1,32,16384]` runs on fewer than ~`grid_cores` cores, **the A1 perf gate fails regardless of how
good the square looks.** Verify the core count in the profiler / tt-npe, not by inspection.

---

## 6. Circular Buffers

Two CBs on every scheme. Semantic names are the primary identifier and flow through to all kernel code.

| Semantic name | Index | Page size | Num pages | Format | Producer | Consumer | Lifetime |
|---|---|---|---|---|---|---|---|
| `cb_input_sticks` | 0 | `tile_size(input_dtype, (tile_h,32))` | `CB_DEPTH * WT_BLOCK` | input dtype, tile `(tile_h, 32)` | `reader` | `compute` | whole program |
| `cb_output_tiles` | 1 | `tile_size(output_dtype, (tile_h,32))` | `CB_DEPTH * WT_BLOCK` | output dtype, tile `(tile_h, 32)` | `compute` | `writer` | whole program |

Exactly one producer thread and one consumer thread each. Nothing is in-place (the tilize helper
`static_assert`s `input_dfb != output_dfb`, `tilize_helpers.inl:116`), so no CB acquires a second
consumer.

### 6.1 Sizing rationale — a function of the knobs, never of an op dimension

`num_pages = CB_DEPTH * WT_BLOCK`, and `WT_BLOCK = min(Wt, WT_BLOCK_MAX)`. Therefore per-core CB L1 is
**bounded by a constant**:

```
per_core_CB_bytes = CB_DEPTH * WT_BLOCK * (tile_size(in) + tile_size(out))
                  = 2 * 8 * (2048 + 2048) = 64 KiB          # bf16, WT_BLOCK_MAX=8, depth 2
```

Independent of `H`, `W`, `Wt`, rank and batch. This satisfies the bounded-CB rule and `master.md`'s
"CBs bounded by a constant not by `Wt`" lever directly, and it is why refinement **A3d is a no-op on
the interleaved path**.

Two hard constraints the sizing must respect:

- `num_pages >= WT_BLOCK`, or the reader deadlocks: the reader reserves `width_in_tiles` pages before
  compute has popped anything (`tilize_helpers_dataflow.inl:105-108` asserts this), and compute waits
  `block_width_tiles` pages per block (`tilize_helpers.inl:225-227`). `CB_DEPTH >= 1` guarantees it.
- **Both** CBs use the same `WT_BLOCK`, and `max(WT_BLOCK, WT_TAIL) == WT_BLOCK` because
  `WT_TAIL <= WT_BLOCK` by construction — so the tail block never needs more pages than the CB has.

### 6.2 Zero-copy sharded variant (A3)

| Semantic name | Backing | Page size | Num pages | `CB_DEPTH` |
|---|---|---|---|---|
| `cb_input_sticks` | `ttnn.cb_descriptor_from_sharded_tensor(0, input_tensor)` | shard-derived | whole shard | 1 (forced) |
| `cb_output_tiles` | `ttnn.cb_descriptor_from_sharded_tensor(1, output_tensor)` | shard-derived | whole shard | 1 (forced) |

`WT_BLOCK = Wt_shard` here — **the shard hands you the block width, so honor it.** The RM shard in L1 is
`shard_h` sticks of `shard_w*elem` contiguous bytes, which is *exactly* the layout
`tilize_block` expects for a `1 × Wt_shard` block; any narrower `WT_BLOCK` would need a strided CB page
that a shard alias cannot express. So the block count is `nt_h_shard = shard_h / tile_h` and the block
width is the full resident shard width. Re-chunking the shard is only ever a response to an L1 limit,
and here there is none (the shard is already allocated). Reader/writer degenerate to a single
`cb_push_back` / `cb_pop_front` of the whole shard; **keep them as separate kernels**
(`zero_copy_fold/report.md`: folding them into compute measured **0.74×** at 2 tiles/core).

---

## 7. API Mapping

Every mechanism, helper or raw, with a verified file:line reference.

| Phase | Type | Function | File:Line | Template params / args | Input CB | Output CB | Requirements |
|---|---|---|---|---|---|---|---|
| boot (compute) | raw_api | `compute_kernel_hw_startup(cb_input_sticks, cb_output_tiles)` | `tilize_helpers.hpp:89-93` | — | — | — | **Must** precede any helper use; two-arg form sets srcA=srcB=input |
| read (aligned) | **helper** | `dataflow_kernel_lib::read_sticks_for_tilize` | `tilize_helpers_dataflow.hpp:87-93`, impl `.inl:96-128` | `<cb_input_sticks, TilizeGranularity::TILE>`; args `(accessor, tile_h, row_bytes = w*32*elem, start_page = src_stick, byte_offset_within_page = c0*32*elem)` | — | `cb_input_sticks` | CB page size **must** be `tile_size`; pushes `width_in_tiles` pages per call; **one barrier per block** (`.inl:126`) |
| tilize | **helper** | `compute_kernel_lib::tilize` | `tilize_helpers.hpp:187-197`, impl `.inl:104-291` | `<WT_BLOCK, cb_input_sticks, cb_output_tiles, InitUninitMode::InitAndUninit, WaitMode::WaitBlock, RECONFIG_MODE, Fp32Mode::Fast>(n_full)`; second instantiation `<WT_TAIL, ...>(n_tail)` | `cb_input_sticks` | `cb_output_tiles` | **`WT_BLOCK` is the block-factor knob** (`hpp:188`). Owns its own wait/pop/reserve/push (`.inl:250-268`). `ASSERT(num_blocks > 0)` (`.inl:121`) → guard both calls with `if (n)` |
| write | raw_api | `noc_async_write` + `TensorAccessor::get_noc_addr` + one `noc_async_write_barrier` per block | `tt_metal/hw/inc/api/dataflow/dataflow_api.h` (write), `tech_reports/tensor_accessor/tensor_accessor.md` | `w` writes of `tile_size` bytes to page `r*Wt + c0 + t` | `cb_output_tiles` | — | Whole-page coalesced writes (`master.md:406-407` lever **B5**); one barrier per block (**B7**) |
| read (padded) | raw_api | pad-aware block reader (CT-selected by `PAD_ENABLED`) | see §8.3 | replicated-`PAD_WORD` L1 fill + the same stick reads | — | `cb_input_sticks` | Fill packed in the **input** element format, replicated across the 32-bit store word |
| read (retile) | raw_api | face-walking block reader (CT-selected by regime) | see §8.4 | 2 sub-face reads of `16*elem` bytes per (stick, tile-column) | — | `cb_input_sticks` | Emits the **same** `cb_input_sticks` contract, so compute + writer are unchanged |
| host: work split | binding | `ttnn.split_work_to_cores(grid, total_blocks, row_wise=True)` | `ttnn/cpp/ttnn-nanobind/operations/core.cpp:467-471, 483-490` | — | — | — | Two groups; group 2 empty on even division (`work_split.hpp:39-47`) |
| host: core order | binding | `ttnn.grid_to_cores(num_cores, grid.x, grid.y, row_wise=True)` | `ttnn-python-utility-bindings.md:159-176` | — | — | — | Must use the **same** `row_wise` as the split |
| host: zero-copy CB | binding | `ttnn.cb_descriptor_from_sharded_tensor(idx, tensor)` | `ttnn/cpp/ttnn-nanobind/program_descriptors.cpp:517-556` | — | — | — | Raises if the tensor is not sharded (`:554-555`) |
| host: page/tile sizes | binding | `tensor.buffer_page_size()`, `tensor.buffer_aligned_page_size()`, `ttnn.tile_size(dtype)`, `ttnn.round_up`, `ttnn.div_up`, `ttnn.get_dram_alignment()`, `ttnn.get_l1_alignment()` | `ttnn-python-utility-bindings.md:9-37, 78-140, 106-124` | — | — | — | **Never hardcode** 2048 / 32 / 16 |

### 7.1 Compute knob settings — each is a decision

| Template param | Value | Reason |
|---|---|---|
| `block_width_tiles` | `WT_BLOCK` (and `WT_TAIL` for the second instantiation) | **The block-factor knob** (§1.3). Must stay < 256 or fast tilize is disabled (`tilize_helpers.inl:95`) |
| `init_uninit_mode` | `InitAndUninit` on **both** calls | The two calls use *different* `block_width_tiles`, and `tilize_init` takes the width (`tilize_helpers.inl:216`), so each needs its own init. `InitOnly`/`Neither`/`UninitOnly` (`hpp:33-38`) would be wrong here. |
| `wait_mode` | `WaitBlock` | Per-block wait is what enables reader/compute overlap; `WaitUpfront` (`.inl:234-237`) would serialize the whole core's work behind the reader. |
| `reconfig_mode` | `NoReconfigure` when `NEEDS_CAST == 0`, else `UnpackAndPackReconfigure` | **Prompt-mandated lever.** The reconfigure (`hpp:22-27`, `.inl:177-194`) exists *only* to drive a `dtype=` cast; with nothing to cast it is a ~150 ns fixed waste — negligible on large kernels, **8–19%** on the ~1 µs small sharded cases. Thread `NEEDS_CAST` as a compile-time arg. |
| `fp32_mode` | `Fast` | `Lossless` is wrong here: it requires `fp32_dest_acc_en` **and** `UnpackToDestFp32` on the input CB, and any FPU consumer downstream re-truncates to tf32 anyway (`hpp:52-71`). tilize's consumers are FPU ops by definition. |
| `remap_mode` | `Configure` | No caller pre-configures BH DEST remap (`hpp:75-78`). |

**Note on fp32 output** (refinement A4): `can_use_fast_tilize` returns false when the *output* format is
Float32 (`tilize_helpers.inl:90-96`), so an fp32→fp32 tilize silently takes the slower regular
`tilize_block` path. Expect a different achieved bandwidth for that dtype and **re-run the ceiling per
dtype** (A4's gate says exactly this — page size changes the bound).

**Note on tiny tiles** (T1): `can_use_fast_tilize` also requires `dfb_has_32x32_tiles<output_dfb>()`
(`tilize_helpers.inl:95`), so every tiny tile takes the regular path. This is automatic — no kernel
change, only the CB tile descriptor. Correct by construction.

### 7.2 Helpers considered and rejected (mandatory justification for every raw-API fallback)

| Raw-API site | Helper considered | File:line of the mismatch | Concrete reason |
|---|---|---|---|
| tile-page writer | `dataflow_kernel_lib::write_sticks_after_untilize` | `tilize_helpers_dataflow.hpp:129-135`; impl `.inl:232-236` | It writes **sticks**: `noc_async_write(l1_addr, accessor.get_noc_addr(start_page + block_row + row), row_bytes)` with `l1_addr += padded_row_bytes` — i.e. it indexes the destination by *row* and copies `row_bytes` per row. Our destination is a **TILE** tensor whose pages are tiles, not sticks; using it would scatter each tile's bytes across `tile_h` destination stick-pages. There is no tile-page writer in `kernel_lib` (`ls ttnn/cpp/ttnn/kernel_lib/` — no `write_tiles*`), so `noc_async_write` + `TensorAccessor::get_noc_addr(tile_index)` is the only mechanism. |
| padded reader fill | `dataflow_kernel_lib::read_sticks_for_tilize` | `tilize_helpers_dataflow.inl:117-127` and the header's own statement at `tilize_helpers_dataflow.hpp:52-54` | The helper writes **nothing** into the padded region: it reads `row_bytes` per stick at L1 stride `padded_row_bytes` (leaving the W-tail bytes untouched) and, for a short last block, "pushes full tile pages for the last partial block (**untouched rows contain stale data**)". A pad fill is therefore structurally absent, and stale L1 is exactly the bug the golden pad oracle catches. The pad path calls the helper's *read* shape but must own the reserve/fill/push itself. |
| padded reader fill | `dataflow_kernel_lib::zero_tile` / `prepare_zero_tile` | `l1_helpers.hpp:47-52`, `:58-65` | Both write **zeros** only (`noc.async_write_zeros`). `pad_value` is a signed / positive / negative bucket (`feature_spec.py:212`), so zeros cover exactly one of three buckets. No arbitrary-value L1 fill helper exists in `kernel_lib`. |
| retile reader | `compute_kernel_lib::untilize` chained into `compute_kernel_lib::tilize` | `tilize_helpers.hpp:137-138` ("Asymmetric CB page support exists only in the tilize helper. The untilize helper always uses symmetric (tile-sized) pages"), `tilize_helpers_dataflow.hpp:104-107`, `tilize_helpers.inl:250` + `:268` | A CB has exactly one page size. The intermediate would need `in_tile_h`-row tile pages for untilize's `push_back(block_width_tiles)` and `out_tile_h`-row tile pages for tilize's `wait_front(input_pages)`. T2 is *defined* by `in_tile_h != out_tile_h` (`feature_spec.py:53-54`), so no single page size satisfies both and the chain cannot be wired. |
| retile reader | `dataflow_kernel_lib::read_sticks_for_tilize` | `tilize_helpers_dataflow.inl:121` | It indexes the source by **stick**: `accessor.get_noc_addr(start_page + block_row + row, byte_offset)`. A TILE-layout source has tile pages, not stick pages, so `start_page + row` addresses the wrong bytes. |
| cross-spec reshard reader | `dataflow_kernel_lib::read_sticks_for_tilize` | `tilize_helpers_dataflow.inl:121` | Same reason at a different level: the source pages live on *other cores'* L1 at host-computed offsets, not at accessor-indexable stick pages of one buffer. |

No other compute or dataflow phase exists, and every one that does uses a helper.

---

## 8. Compute Phases

| # | Operation | Helper? | Input CB (name, pages, state) | Output CB (name, pages) | CB state after |
|---|---|---|---|---|---|
| 0 | `compute_kernel_hw_startup(cb_input_sticks, cb_output_tiles)` | raw | — | — | HW configured; no CB touched |
| 1 | `tilize<WT_BLOCK, ...>(n_full)` — full-width blocks | **yes** | `cb_input_sticks`, `WT_BLOCK` pages per block, RM sticks | `cb_output_tiles`, `WT_BLOCK` tile pages per block | both drained per block: helper waits/pops `WT_BLOCK` in, reserves/pushes `WT_BLOCK` out (`tilize_helpers.inl:250-268`) |
| 2 | `tilize<WT_TAIL, ...>(n_tail)` — tail-width blocks (skipped when `n_tail == 0`) | **yes** | `cb_input_sticks`, `WT_TAIL` pages per block | `cb_output_tiles`, `WT_TAIL` tile pages per block | both empty at kernel exit |

Only one compute stage exists; there is no intermediate CB between two sequential helpers, so the
"intermediates sized to full block" rule has no subject here.

### 8.1 CB sync ledger (push count == wait count, per CB, per block)

| CB | Producer pushes | Consumer waits/pops | Balanced |
|---|---|---|---|
| `cb_input_sticks` | reader: `cb_push_back(w)` per block (`tilize_helpers_dataflow.inl:127`) | compute: `in_dfb.wait_front(w)` / `pop_front(w)` per block (`tilize_helpers.inl:250, 268`) | ✅ `w` per block, both sides, where `w ∈ {WT_BLOCK, WT_TAIL}` and the reader and compute iterate blocks in the **same order** (increasing `b`) |
| `cb_output_tiles` | compute: `out_dfb.reserve_back(w)` / `push_back(w)` per block (`tilize_helpers.inl:253, 267`) | writer: `cb_wait_front(w)` / `cb_pop_front(w)` per block | ✅ `w` per block, both sides |

The reader, compute and writer must all derive `w` from the **same** decode (§5.2) — `w = (wchunk ==
n_wchunks-1) ? WT_TAIL : WT_BLOCK`. Because compute processes all `n_full` blocks before all `n_tail`
blocks and `b` increases monotonically, the reader's per-block `w` sequence matches compute's
concatenated `(WT_BLOCK × n_full, WT_TAIL × n_tail)` sequence exactly.

### 8.2 Aligned reader (Track A hot path, `PAD_ENABLED == 0`)

```
per block b in [b0, b0+nb):
    decode (r, wchunk, w, c0)             # §5.2
    read_sticks_for_tilize<cb_input_sticks, TILE>(
        src_accessor,
        /*total_num_rows*/ tile_h,
        /*row_bytes*/      w * 32 * elem,
        /*start_page*/     r * tile_h,             # == src_stick, since Hp == H when aligned
        /*byte_offset*/    c0 * 32 * elem)
```

One helper call per block: reserve `w` pages, issue `tile_h` reads of `w*32*elem` bytes at L1 stride
`w*32*elem`, **one** barrier, push `w`. That is `tile_h` outstanding reads per barrier (B7) at the
512 B one-packet size (B6), with `w == WT_BLOCK` for every interior block.

### 8.3 Padded reader (Track P, `PAD_ENABLED == 1`) — one code path, three regions

The pad path is a **CT-selected second body inside the same reader source**. When `PAD_ENABLED == 0`
the compiler emits only §8.2, so **Track P cannot regress Track A structurally** — not by convention.

```
per block b:
    decode (r, wchunk, w, c0)
    g0        = r * tile_h                              # first output row of this block
    real_rows = number of s in [0,tile_h) with ((g0+s) % Hp) < H and ((g0+s)/Hp) < nimg
    real_bytes= clamp(W*elem - c0*32*elem, 0, w*32*elem)   # 0 for a whole pad column-block

    cb_reserve_back(cb_input_sticks, w); l1 = get_write_ptr(cb_input_sticks)

    if real_rows == tile_h and real_bytes == w*32*elem:
        # fully-interior block: identical to the aligned path, zero fill cost
        issue tile_h reads; one barrier
    else:
        # 1 store-loop fills ALL THREE pad regions at once
        fill_l1_words(l1, tile_h * w*32*elem, PAD_WORD)   # W tail + H tail + whole pad tiles
        for s in 0 .. real_rows-1:                        # then read the real data over the top
            src = ((g0+s)/Hp)*H + ((g0+s)%Hp)
            noc_async_read(src_accessor.get_noc_addr(src, c0*32*elem),
                           l1 + s*(w*32*elem), real_bytes)
        one barrier
    cb_push_back(cb_input_sticks, w)
```

| Rule (from the prompt) | How this satisfies it |
|---|---|
| all three pad regions filled | The single fill covers the whole block region; the reads then overwrite exactly the real sub-rectangle. **W tail** = bytes `real_bytes .. w*32*elem` of a real row; **H tail** = rows `real_rows .. tile_h`; **whole pad tiles** = a block with `real_rows == 0` or `real_bytes == 0`, which fills and issues no reads at all. |
| fill packed in the **input** element format | `PAD_WORD` is computed on the host from `pad_value` and `input_dtype`, never `output_dtype`. The cast happens later, at pack time, on data that already carries the fill. |
| sub-word fill replicated across the 32-bit store word | `PAD_WORD` = the fill repeated `4/elem` times: **2×** for bf16/uint16, **4×** for uint8, **1×** for fp32/uint32/int32. Writing it once would leave every other pad element stale — invisible at `pad_value=0.0` and visible only on a nonzero fill, which is precisely why `pad_value` is a sign-bucketed axis (`feature_spec.py:208-212`). Negative integer fills go through a signed→unsigned `bit_cast` on the host before replication. |
| no extra pass, no host round-trip | The fill is L1 stores into a CB the reader already reserved. Zero extra DRAM bytes → invisible in the tt-npe DRAM number. |
| logical shape unchanged | Purely a host-side decision: allocate the output with `logical_shape = input.logical_shape` and `padded_shape = pad target`. The kernels never see a logical shape. |
| aligned no-op pad is bit-identical and not slower | `pad_mode="auto"` on an already-aligned input yields `real_rows == tile_h` and `real_bytes == w*32*elem` for **every** block, so the fill branch is never taken (`feature_spec.py:484-488`'s degenerate cell). |

`fill_l1_words` is a plain 32-bit store loop over the reserved L1 region (no NoC traffic, so the DM
ceiling does not describe it — the prompt's Track P rule). Bound: `tile_h * WT_BLOCK*32*elem` =
`WT_BLOCK * tile_size` bytes = 16 KiB per boundary block at bf16 / `WT_BLOCK=8`.

### 8.4 Retile reader (T2, Blackhole-only)

Emits the **same** `cb_input_sticks` contract, so compute and writer are byte-identical to the aligned
path. Per output stick `g` and tile-column `c`, the source is input tile
`(g / in_tile_h) * Wt + c`, and within that tile the row `g % in_tile_h` splits across two 16-wide
faces: elements `[0,16)` from the left face pair, `[16,32)` from the right. So **two reads of
`16*elem` bytes per (stick, tile-column)**, at face offsets derived from the input tile's geometry.

This is deliberately transaction-inefficient (32 B reads at bf16). Track T is **correctness-gated
only** — do not spend a DM lever on it, and do not report its duration against a NoC ceiling.
Arch-gate on Blackhole and **skip, not fail**, elsewhere; plain tiny tiles (T1) are *not* arch-gated and
must work everywhere.

### 8.5 uint8 (A5b) — the per-face row dim, and its failure signature

An 8-bit datum needs the standard **per-face** row dim (16), not the full-tile row dim (32) that the
16/32-bit formats use; the wrong one yields a **strided** tile — every other row zero — which is
shape-correct and value-wrong, so it survives a loose numeric check. The golden cells compare uint8
**exactly** for this reason.

The coupling the implementer must resolve at A5b, stated so it is not discovered by accident: the
aligned reader derives its sticks-per-block from `tile_h = unpack_tile_r_dim[cb_input_sticks]`
(`tilize_helpers_dataflow.inl:75`). Setting the input CB's tile row dim to 16 for uint8 therefore also
changes how many sticks the reader groups per push. Resolve it as an explicit decision, not a
side effect: either (a) keep the reader's stick grouping at the true `tile_h` and drive the per-face row
dim through the LLK/CB descriptor independently, or (b) push two half-blocks per output tile-row. Pin
the choice with a `tt-probe.sh` run that dumps one output tile and checks for the every-other-row-zero
signature *before* running the golden suite. Also: a 32-element uint8 chunk is 32 B, at the DRAM read
alignment floor — so `WT_BLOCK_MAX` for uint8 must keep `row_bytes >= 64`, i.e. `WT_BLOCK >= 2`, which
the `max(2, ...)` in §1.5 already guarantees.

---

## 9. Key Risks and Gotchas

| # | Risk | Mitigation |
|---|---|---|
| 1 | **Height-only split strands wide-short tensors.** `split_work_to_cores(nt_h)` puts `[1,1,32,16384]` on **one** core. `distribution_gate/report.md:27-40` measures **7.25×** slower on `32×4096`. | The linearization `b = wchunk*nt_h + r` (§1.4) makes the width split unconditional. Verify the core count on the wide-short bench, do not infer it. |
| 2 | **CB sized by `Wt`.** `Wt=512` × 2048 B × depth 2 = 2 MiB → L1 OOM on the wide-short bench. | `WT_BLOCK = min(Wt, WT_BLOCK_MAX)`; per-core CB is a **constant** 64 KiB (§6.1). Never let a CB page count reference `Wt` directly. |
| 3 | **`floor` instead of `ceil` in the tile-row count.** `floor(nimg*H/tile_h)` silently drops a tile-row per image once H is not a tile multiple. | `nt_h = nimg * ceil(H/tile_h)` from the start (§5.1), even though Phase 0 only accepts aligned shapes. |
| 4 | **The reconfigure with nothing to cast.** ~150 ns fixed; 8–19% on the ~1 µs small sharded cells. | `NEEDS_CAST` compile-time flag → `ReconfigureRegisterDatatypeMode::NoReconfigure` (§7.1). Record in the ledger. |
| 5 | **Sub-word fill written once.** Invisible at `pad_value=0.0`, garbage on any nonzero fill. | `PAD_WORD` replicates the value to 32 bits (§8.3). The `positive`/`negative` `pad_value` buckets exist to catch it. |
| 6 | **Fill packed in the output dtype.** Garbage whenever a cast is also requested. | `PAD_WORD` is built from `input_dtype` only (§8.3). |
| 7 | **Logical shape promoted to the pad target.** Bytes look right; `run_tilize` fails the second oracle (`helpers.py`, `logical_shape_promoted`). | Output allocated with `logical_shape = input.logical_shape`, `padded_shape = pad target`. |
| 8 | **Padding applied implicitly.** A bare `tilize(t)` on an unaligned input must **raise**, never silently round up. | `pad_mode == "none"` + unaligned → raise in `validate()`. |
| 9 | **A physical shard re-read through a `TensorAccessor`.** "Works" only because the core still holds its rows; it re-fetches over the NoC data already in L1 and is not sharding. | `cb_descriptor_from_sharded_tensor` aliases the CB onto the shard; the reader issues **no NoC read** (§4.1, §6.2). tt-npe must show **zero** DRAM on the sharded side. |
| 10 | **A sharded shard re-chunked below its resident width.** Ignores the block width the shard already handed you. | `WT_BLOCK = Wt_shard` on the zero-copy path (§6.2); sub-chunking only under a real L1 limit. |
| 11 | **Hardcoded 32 in the alignment check or the tile-row count.** A tiny tile redefines H-alignment; `tag_alignment` measures against the *requested* tile height, so the op would mis-gate its own cells. | Every use is `tile_h`, sourced from `tile.height` (§5.1, §3). |
| 12 | **uint8 full-tile row dim → strided tile.** Every other row zero; shape-correct, value-wrong. | §8.5, plus an eyeball probe before the golden run. |
| 13 | **Retile + pad accepted.** Silently ignoring one of them. | Explicit refusal in `validate()` (`feature_spec.py:708-713`). |
| 14 | **Retile hard-failing on Wormhole.** Reads as missing support when the real answer is "not on this silicon". | Arch-gate → **skip**. Tiny tiles are **not** arch-gated; a failure there is a real defect. |
| 15 | **`tilize` called with `num_blocks == 0`.** `ASSERT(num_blocks > 0)` (`tilize_helpers.inl:121`) trips under `--dev`. | Guard both instantiations with `if (n_full)` / `if (n_tail)`. `WT_TAIL` defaults to `WT_BLOCK` (never 0) so the second instantiation always compiles (`static_assert(block_width_tiles > 0)`, `.inl:115`). |
| 16 | **CB capacity < `WT_BLOCK` → deadlock**, not a wrong answer: the reader blocks in `cb_reserve_back` while compute waits for a block it will never get. | `num_pages = CB_DEPTH * WT_BLOCK` with `CB_DEPTH >= 1`; the helper asserts it (`tilize_helpers_dataflow.inl:105-108`, `tilize_helpers.inl:225-227`). |
| 17 | **Block-float input.** `tilize` asserts the input is not block-float (`tilize_helpers.inl:174`) and the reader's `elem_size = tile_size/tile_hw` derivation breaks (`.inl:82-85`). | bf8b is an **output** dtype only (the input is always ROW_MAJOR) — enforce in `validate()`. |
| 18 | **Column-line core placement.** `row_wise=False` is the `split_work_to_cores` **default** and is the ~2.9× trap. | Pass `row_wise=True` to **both** `split_work_to_cores` and `grid_to_cores`, or the runtime-arg assignment order silently mismatches the split's group order. |

---

## 10. Performance Methodology

**Classification: movement-dominated** (predicted). §1.1 proves the op has no reuse and therefore no
arithmetic intensity: exactly `2 × tensor_bytes` of DRAM traffic on the interleaved path, `0` on the
same-spec sharded path. But `tilize` **has** a compute stage, so the NoC ceiling is only a *partial*
bound and the classification is a **prediction that must be confirmed by ablation before any DM lever
is chased**.

### 10.1 Required per-phase gate sequence (run at Phase 0 and at every Track A refinement)

1. **`/perf-ceiling-dm`** — characterize both transfers (read = RM sticks, write = TILE pages, **both
   DRAM** on the interleaved path). Bracket each with `ONE_FROM_ALL`/`ONE_TO_ALL` (few-core,
   no-contention) … `ALL_FROM_ALL`/`ALL_TO_ALL` (full-grid, full-contention); the true value lands
   inside, near the full-contention end for round-robin interleaved. Cap every bound at `dram_peak`
   (WH 288 GB/s), compute the route-overlap congestion, and take
   `op_target = MAX(read_bound, write_bound, compute_bound)` with read/write overlapping per `CB_DEPTH`.
2. **tt-npe** (`tt_npe.sh <trace> --noc-trace`) — **pin** estimated cycles, DRAM BW utilization,
   congestion %, and the binding resource.
3. **`/perf-measure`** — device Tracy kernel duration, **median of the trial loop**, never one untrialed
   number. `achieved = measured / target`.
4. **Ablation (A0's deliverable, the claim everything else rests on):** stub the `tilize`/`tilize_block`
   math while **keeping** the CB reserve/push/wait/pop and the NoC barriers, then diff the duration.
   Duration barely moves → **DM-bound**, the ceiling is the target. Duration drops a lot →
   **compute-bound**, and the DM ceiling does not apply.

### 10.2 Bench shapes and what each one is the proving ground for

Perf shapes are **not** in `feature_spec.py` INPUTS. They live in a separate underscore-prefixed in-tree
bench that runs under `--collect-noc-traces` + Tracy and asserts **no** PCC (measurement and ablation
need no correctness; correctness stays on the small golden cells).

| Bench shape | Regime | Blocks/core (8×8) | This shape is the only place these levers can be measured |
|---|---|---|---|
| `[1,1,2048,2048]` bf16 | grid-filling square, **DRAM-bound** | 8 | **The multi-block proving ground.** B8 (trid double-issue), `split_reader`, and `CB_DEPTH=2` have nothing to overlap against on a one-block-per-core shape (`master.md:413-420`). Also the per-core DRAM-efficiency number. |
| `[1,1,32,16384]` bf16 | **wide-short, mandatory** (`nt_h=1`, `Wt=512`) | 1 | Whether the distribution actually **fills the grid**. A bench that measures only the square reports healthy while a height-only split strands this on one core — the exact defect this run must not ship. If it runs on `< grid_cores` cores the gate fails regardless of the square. |
| `[1,1,32,64]` bf16 | **smallest regime** (matches the smallest golden INPUTS shape, `feature_spec.py:252-255`) | 1 core, 1 block | `master.md:396-404` **B0**: every per-core-overhead lever (B5, B7, B8, B10, B13, and `CB_DEPTH`) must be counterfactualed **here**, because a lever that pays on the square can regress a tiny shape where fixed setup dominates ~1–8 tiles of real work. `eval/verify_levers.py` resolves the measurement's shape against the smallest INPUTS shape, so this is the entry it checks. |
| `[1,1,512,64]` L1-sharded, 4 cores | small sharded, ~1 µs | 1 | Where the `NoReconfigure` lever's 8–19% lives (§7.1), and where `zero_copy_fold`'s program-structure effect is largest. |
| `[1,1,2048,2048]` fp32 and uint8 | per-dtype re-target (A4/A5b) | 8 | Page size changes the bound, and fp32 output **disables fast tilize** (`tilize_helpers.inl:90-96`). Re-run the ceiling per dtype. |

### 10.3 Candidate algorithms — the Mode A ranking (a **PREDICTION**, to be checked at Phase 0)

Anchor for the interleaved DRAM→DRAM target: `double_buffer/report.md:78-88` measures a 64-core
`2048×2048` bf16 DRAM→DRAM stream at **87 916 ns / 190.8 GB/s**, where 190.8 GB/s is *total*
(read + write) traffic — 16.8 MiB / 87.9 µs. `dram_saturation/README.md` independently plateaus a
DRAM→DRAM copy of the same size at **191–195 GB/s from ~16 cores**. `tilize` on `[1,1,2048,2048]` moves
**exactly the same 16.8 MiB** of DRAM traffic (§1.1: the map is a bijection), so:

- **hard floor** (WH `dram_peak` 288 GB/s): 16.8 MiB / 288 GB/s ≈ **58 µs**
- **realistic op_target** (measured interleaved round-robin achievable, ~191 GB/s): ≈ **88 µs**
- predicted compute bound: 64 tiles/core over an 88 µs window = a 1.37 µs/tile budget, ~10× above what
  `fast_tilize` needs → **predicted DM-bound with a wide margin**. *This is the prediction step 10.1.4
  must confirm.*

| Rank | Candidate | Predicted target, `[1,1,2048,2048]` | Predicted, `[1,1,32,16384]` | The one property that decided it |
|---|---|---|---|---|
| **1 — the design** | **TILE-granularity row-block reader; 2-D linear block split `b = wchunk*nt_h + r`; `WT_BLOCK = 8` (512 B reads); one barrier per block; whole-tile writes; `row_wise=True`; NoC0 read / NoC1 write; `CB_DEPTH=2`** | **≈ 88 µs** (≈191 GB/s total traffic) | ≈ 11 µs bandwidth-wise, likely ~5–15 µs dispatch-bound; **64 cores** | Only candidate that is simultaneously (a) one-packet-sized on the read (B6), (b) whole-page on the write (B5), (c) one barrier per 32-read block (B7), (d) split across NoC0/NoC1 (B9), (e) **constant-L1**, and (f) grid-filling in *both* aspect-ratio regimes. |
| 2 | Same, but `WT_BLOCK = Wt` (whole-row 4096 B reads, no column chunking) | ≈ 88 µs — **no better**, because the square is already DRAM-saturated at 512 B | **L1 OOM** (`Wt=512` → 2 MiB of CBs); and only 64 blocks → 1 block/core, killing the multi-block overlap | Bigger transactions cannot beat an already-saturated DRAM interface, and the price is an unbounded CB. Rejected on the bounded-CB rule. Its *knob* survives as `TARGET_READ_BYTES` — sweep 512→2048 if the measurement disagrees. |
| 3 | `TilizeGranularity::ROW` reader (one CB page per stick, asymmetric tilize) | ≈ **150–220 µs** | worse | `tilize_helpers_dataflow.inl:148-157` puts `noc_async_read_barrier()` **inside** the per-stick loop — one barrier per transaction, the `double_buffer` `block=1` trap (**6.5 vs 17.9 GB/s/core**, `report.md:29-42`). Its only virtue (less L1 when rows < 32) is moot: candidate 1's CB is already a constant 64 KiB. |
| 4 | Height-only split, `split_work_to_cores(nt_h)` | ≈ 88 µs — **indistinguishable on the square** | ≈ **7.3× slower**, **1 core** (`distribution_gate/report.md:27-40`: 7.25× on `32×4096`) | Strands every wide-short shape. It looks identical on the square, which is exactly why the wide-short bench is mandatory. |
| 5 | Cap at the ~16-core DRAM bandwidth knee (`dram_saturation`'s exploit: full BW on ¼ of the grid) | predicted ≈ 88 µs on 16 cores, 48 cores freed | — | **Refuted on recorded evidence for this op.** `master.md:358-361`: the knee clause was implemented, measured **~2.4× slower** on tilize, and removed — tilize's reader is *transaction-rate* bound at small pages, so there is no reachable knee and shedding cores only adds per-core sync cost. Use the **full grid**. |

If candidate 1 misses its predicted 88 µs at Phase 0, the ordered fallbacks are: sweep
`TARGET_READ_BYTES` toward candidate 2's transaction size (bounded by L1), then B8 trid double-issue on
the multi-block shape, then `split_reader`. Do **not** reach for candidate 3, 4 or 5.

### 10.4 Used-optimization ledger (Mode C) — every landed lever must pay

Run the `/perf-ceiling-dm` **used-optimization audit** for each row: estimate the counterfactual (target
with that one lever flipped off), then confirm the delta on device with `/perf-measure`. Keep only
levers with a real measured payoff; **per `master.md:396-404` (B0), counterfactual every
per-core-overhead lever on the *smallest* regime it runs in, not only on the square.**

| Lever | master.md ref | Predicted delta | Counterfactual to run | Measure on |
|---|---|---|---|---|
| 2-D linear block split (width parallelism) | A0, `width_split`, `distribution_gate` | **6–7.8×** on wide-short; 1.00× on the square | height-only split | `[1,1,32,16384]` **and** `[1,1,2048,2048]` |
| `row_wise=True` core line | A1, `noc_placement` | ~2.2–2.9× at 8 cores; smaller at full grid | `row_wise=False` (the default column line) | square + smallest |
| reads NoC0 / writes NoC1 | B9, `noc_placement` | 2.5–4.8× vs reversed | swap the two configs | square |
| one barrier per block (32 reads) | B7, `double_buffer` | ~2× vs per-read barrier | barrier inside the stick loop | square + smallest |
| 512 B one-packet read chunk | B6, `double_buffer` | sets the transaction-rate ceiling | sweep `TARGET_READ_BYTES` ∈ {256,512,1024,2048} | square + wide-short |
| whole-tile coalesced writes | B5, `tile_reorder` | ≥ 1.0× vs 4×512 B face writes | face-granular writes | square |
| `CB_DEPTH = 2` | C16, `double_buffer` | ~1.4× at low core count; **predicted ~1.0× once DRAM-saturated** (`report.md:78-88` shows no gain at 64 cores) | `CB_DEPTH = 1` | **smallest + small-sharded** (where it can still pay) *and* square (where it may be a no-op — report honestly) |
| `NoReconfigure` when `NEEDS_CAST == 0` | prompt-mandated | ~150 ns fixed → **8–19%** on ~1 µs sharded cells | `UnpackAndPackReconfigure` | small-sharded `[1,1,512,64]` |
| zero-copy CB alias on shards | C14 | removes **100%** of DRAM traffic | read the shard through a `TensorAccessor` | A3 sharded cells (tt-npe DRAM = 0) |
| separate reader/compute/writer (not folded) | C14, `zero_copy_fold` | 1.05–1.35× vs folded | fold dataflow into compute | small-sharded |
| CT `TensorAccessorArgs`, address-only RT args | D18, D19 | program-cache hit on call 2 | runtime address-gen | any; assert with `ttnn.compute_program_descriptor_hash` |

### 10.5 Deferred levers (the A7 Mode D completeness ledger's starting list)

Recorded now so the run-closing audit has somewhere to start rather than rediscovering the space.

| Lever | master.md ref | Status at Phase 0 | Predicted delta if applied |
|---|---|---|---|
| B8 trid double-issue | `master.md:413-420` | **deferred** | ~5–15% on multi-block cores; **structurally 0 on one-block cores** — only measurable on `[1,1,2048,2048]` |
| `split_reader` (both DM RISCs) | `master.md:121-133` | **deferred** | up to ~1.7×, but **only if a DM RISC is proven issue-bound first**. Also A3b's mechanism. |
| B10 per-reader VC assignment | `master.md:424-425` | deferred | breaks FCFS serialization on shared routes; unquantified |
| B13 `set_state`/`with_state` | `master.md:430-431` | deferred | amortizes command-buffer setup across same-shape transfers; per B0 it can **regress** the smallest regime |
| A4 cliff-core specialization | `master.md:391-392` | **not-applicable** — the two-group split already carries the remainder in a runtime arg, and the tail column-block is handled by a second CT instantiation, not a second kernel | 0 |
| B12 multicast | `master.md:428-429` | **structurally not-applicable** — §1.1: no operand is shared across any split | 0 |
| C15 prefer sharded over interleaved | `master.md:441-443` | not the op's choice — the caller's `memory_config` decides | — |
| C17 in-place / no-copy | `master.md:446-447` | **not-applicable** — layout conversion cannot be in-place (`tilize_helpers.inl:116` `static_assert`s `input_dfb != output_dfb`) | 0 |
| D21 `InterleavedAddrGenFast` for pow2 pages | `master.md:457-459` | deferred | shifts instead of multiplies in address-gen; small, and `TensorAccessor` may already specialize |
| E22 Metal Trace + multi-CQ | `master.md:462-464` | out of single-op scope | — |
| `compute_block_size` (raise `NT_ROWS_PER_BLOCK`) | `master.md:192-215` | **not-applicable** — the tilize LLK block is 1 tile-row tall by construction (`tilize_helpers.hpp:121-125`); there is exactly one compute phase, so there are no phase-boundary reconfigs to amortize | 0 (the second lever of that example — dropping the wasted reconfig — **is** applied, as `NoReconfigure`) |

---

## 11. Hardware Constraints checklist

- [x] CB sync: push count == wait count for every CB (§8.1)
- [x] CB ownership: every CB has exactly one producer thread and one consumer thread (§6)
- [x] No unconditional op-parameter-sized CB — per-core CB L1 is a constant 64 KiB (§6.1)
- [x] Every block factor and buffer depth is a parameter with one source of truth (§1.5)
- [x] Block-size default is coarse (`WT_BLOCK = 8` at 512 B), not minimal, with measured justification (§1.3)
- [x] Sharded path honors the shard as the per-core block (`WT_BLOCK = Wt_shard`, §6.2)
- [x] Tile geometry alignment-aware: `ceil` + per-image (§5.1)
- [x] Regime-selection function pinned + regime-pinned tests required (§5.3)
- [x] `compute_kernel_hw_startup()` before any helper usage (§7, phase 0)
- [x] `block_width_tiles < 256` so fast tilize stays available (`tilize_helpers.inl:95`)
- [x] Every compute phase uses a helper; every raw-API fallback has a concrete file:line justification (§7.2)
- [x] Page sizes aligned to tile size; RM CBs count pages in sticks, tile CBs in tiles — both CBs here are tile-paged (§6)
- [x] All `wait_front` calls on a CB use the same page count as the matching `push_back` (§8.1)
- n/a Reduce scaler CB (no reduction in this op)
- n/a DEST tile budget (managed inside the tilize LLK; the only exposed bound is `block_width_tiles < 256`)
- n/a Sequential helper intermediates (exactly one compute phase, §8)
- n/a Broadcast verification (no binary ops)
- n/a Reduce direction verification (no reduce)

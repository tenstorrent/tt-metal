# Operation Design: tilize

> **Specification, not source.** This document contains **no kernel code**. Every compute phase is
> named by the helper that implements it plus its exact template arguments. The implementer writes
> the `.cpp` files and derives all kernel arguments from the CB table and the helper signatures cited
> here.

---

## Overview

| Field | Value |
|---|---|
| Classification | `data_movement` (pure layout conversion, zero arithmetic) |
| Goal | Re-lay a ROW_MAJOR tensor into TILE layout (32×32 tiles of four 16×16 faces) on device, with an optional value-preserving output-dtype cast. Maximally performant: 2D work split, bounded per-core L1, zero-copy on sharded I/O. |
| Math | `output[n,c,h,w] = input[n,c,h,w]` — identity on values; only byte positions change. Oracle is bit-identity (PCC when `dtype=` narrows). |
| Mode | Derivative (native `ttnn.tilize` existed; nuked in `a547a3ef00` for this eval clone) |
| Entry point | `from ttnn.operations.tilize import tilize` |
| Dispatch | Python `ttnn.generic_op` + `ttnn.ProgramDescriptor` (see `.claude/references/generic_op_template/`) |
| References | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp`, `tilize_helpers_dataflow.hpp`, `ttnn/ttnn/operations/toy_tilize_untilize/`, `ttnn/ttnn/operations/examples/master.md`, `eval/golden_tests/tilize/feature_spec.py` |

### Signature

```
tilize(input_tensor, memory_config=None, *, dtype=None,
       use_multicore=True, use_double_buffer=True) -> ttnn.Tensor
```

---

## Parameters

| Name | Type | Required | Valid Range | Default | Where it lands |
|---|---|---|---|---|---|
| `input_tensor` | `ttnn.Tensor` | yes | ROW_MAJOR_LAYOUT, on device, rank ≥ 2, `H%32==0 and W%32==0` | — | reader `TensorAccessorArgs` + RT base address |
| `memory_config` | `ttnn.MemoryConfig \| None` | no | interleaved (DRAM/L1) or sharded (L1) | input's memory config | output tensor allocation; selects the writer path (interleaved vs aliased CB) |
| `dtype` | `ttnn.DataType \| None` | no (kw-only) | `bfloat16`, `float32`, `bfloat8_b`, `uint32`, `uint16`, `int32`; int↔float crosses out of contract | input's dtype | `cb_tiled_output` `CBFormatDescriptor.data_format`; compute CT `needs_cast` |
| `use_multicore` | `bool` | no (kw-only) | `{False, True}` | `True` | host work split: `G = 1` vs `G = grid_cores` |
| `use_double_buffer` | `bool` | no (kw-only) | `{False, True}` | `True` | CB depth (`2` vs `1`); auto-falls back to `1` when depth-2 exceeds the L1 budget |

### Validation (Python entry point, before any device work)

| Condition | Exception | **Required message substring** (the acceptance test matches on it — this is a contract, not a suggestion) |
|---|---|---|
| `input_tensor.layout != ttnn.ROW_MAJOR_LAYOUT` | `RuntimeError` | `ROW_MAJOR` — e.g. "tilize requires ROW_MAJOR_LAYOUT input, got TILE_LAYOUT" |
| input not on device (`storage_type()` is host / `device()` raises) | `RuntimeError` | `device` — e.g. "tilize requires a tensor on device" |
| `len(shape) < 2` | `RuntimeError` | `rank` — e.g. "tilize requires rank >= 2" |
| `shape[-1] % 32 != 0` or `shape[-2] % 32 != 0` | `ValueError` | `divisible by 32` — e.g. "tilize: last two dims must be divisible by 32 (this op does not pad — use tilize_with_val_padding)" |
| registry-contract refusal (axis outside `SUPPORTED`, or `EXCLUSIONS` hit) | `UnsupportedAxisValue` / `ExcludedCell` from `ttnn/ttnn/operations/_op_contract.py:26,30` | free-form; the harness recognizes it by `isinstance`, not by wording (both subclass `NotImplementedError`) |

`tests/ttnn/unit_tests/operations/tilize/test_tilize.py` uses the repo-wide `expect_error` fixture
(`conftest.py:881-900`), which is `pytest.raises(error, match=message)` plus CI-triage logging — so
the substrings above are matched as regexes against the raised message. Changing them breaks the
acceptance test, which the implementer may not edit.

Ordering: shape/layout/device checks first (these are `ValueError`/`RuntimeError` per the spec), then
`validate()` (registry refusal). A malformed input must never reach `validate()` and be misreported
as a support refusal.

---

## Tensors

### Input

| Property | Requirement |
|---|---|
| Shape | rank ≥ 2, `[..., H, W]`, `H % 32 == 0`, `W % 32 == 0` |
| Dtype | `bfloat16` (primary), `float32`, `uint32`, `uint16`, `int32` |
| Layout | ROW_MAJOR — always |
| Memory | INTERLEAVED (DRAM or L1) **or** SHARDED L1 (legacy HEIGHT/WIDTH/BLOCK, or `NdShardSpec`), ROW_MAJOR-sharded |
| Page semantics | one page = one stick. Interleaved: `page_bytes == W * elem_size`. Sharded: `page_bytes == shard_W * elem_size` → a logical row spans `pages_per_row = W // shard_W` pages |

### Output

| Property | Value |
|---|---|
| Shape | identical to input's logical shape |
| Dtype | `dtype` if given, else input's |
| Layout | TILE_LAYOUT — always |
| Memory | `memory_config` if given, else the input's |
| Page semantics | one page = one 32×32 tile. Flat page index of the tile at (tile-row `r`, tile-col `c`) is `r * Wt + c`, with `r ∈ [0, nt_h)` running over the **folded** leading dims (see Derived Geometry) |

### Derived geometry (host, from the padded shape)

| Symbol | Formula | Meaning |
|---|---|---|
| `folded_H` | `prod(padded_shape[:-1])` | all leading dims × H collapsed into one row axis |
| `W` | `padded_shape[-1]` | |
| `nt_h` | `folded_H // 32` | tile-rows in the folded 2D view |
| `Wt` | `W // 32` | tile-columns |
| `total_tiles` | `nt_h * Wt` | |
| `elem_in` / `elem_out` | `input_tensor.element_size()` / output's | |
| `tile_in` / `tile_out` | `ttnn.tile_size(in_dtype)` / `ttnn.tile_size(out_dtype)` | 2048 bf16/uint16, 4096 fp32/uint32/int32, 1088 bf8b (probed on this build) |
| `tile_row_bytes` | `32 * elem_in` | bytes one stick contributes to one tile-column |

The fold is exact because both the RM stick order (`n, c, h`) and the TILE page order (`n, c, ht, wt`)
are row-major over the leading dims, and `H % 32 == 0` guarantees no leading dim straddles a tile
boundary. **Rank 2/3/4/5 need no special-casing** — only the fold.

---

## Dataflow Strategy

### Path A — interleaved in → interleaved out (the primary, DM-bound path)

```
DRAM/L1 (RM sticks)
  --NoC0 read, 32 strided stick-reads per barrier-->  cb_rm_input   (tile-sized pages, depth D)
  --TRISC unpack/math/pack: tilize LLK           -->  cb_tiled_output (tile-sized pages, depth D)
  --NoC1 write, chunk_wt whole-tile page-writes  -->  DRAM/L1 (TILE pages)
```

Stage-by-stage format:

| Stage | RISC | Format in L1 | Transfer granularity |
|---|---|---|---|
| Read | NCRISC (reader, NoC0) | RM sticks written back-to-back at `padded_row_bytes` stride into `chunk_wt` tile-page slots | `chunk_row_bytes = chunk_wt * 32 * elem_in` per stick; 32 sticks per barrier |
| Compute | TRISC0/1/2 | `chunk_wt` RM tile-page slots → `chunk_wt` real tiles | 1 LLK call per `1 × chunk_wt` block |
| Write | BRISC (writer, NoC1) | `chunk_wt` tile pages | `tile_out` bytes per `noc_async_write`; `chunk_wt` writes per barrier |

The **only** reason this works with tile-sized input CB pages is the RM↔tile byte equivalence:
`32 sticks × chunk_row_bytes == chunk_wt × tile_in`. The reader never "knows" about tiles; it just
fills `chunk_wt * tile_in` contiguous bytes with 32 rows. This is the documented contract of
`read_sticks_for_tilize` in TILE granularity (`ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp:54-58`).

### Path B — same-spec L1-sharded in → same-spec L1-sharded out (zero-copy, R3)

```
input L1 shard (RM)  ==aliased==>  cb_rm_input      (no NoC read at all)
                                        |  tilize LLK
                                        v
output L1 shard(TILE) <==aliased==  cb_tiled_output  (no NoC write at all)
```

Both CBs are built with `ttnn.cb_descriptor_from_sharded_tensor` (binding at
`ttnn/cpp/ttnn-nanobind/program_descriptors.cpp:517-556`), so the CB's L1 base address **is** the
shard's base address. The reader degenerates to a single push of the whole shard; the writer to a
single wait+pop. **Zero DRAM traffic on either side.** Depth is structurally 1 (the CB *is* the
shard) — `use_double_buffer` is inert on this path and must not attempt to double the allocation.

### Path C — crossover / cross-spec (R3b, R3c)

| Direction | Reader | Writer |
|---|---|---|
| DRAM-interleaved RM → sharded TILE | Path-A reader (`read_sticks_for_tilize`), plus **split reader**: NCRISC takes even tile-row blocks, BRISC takes odd ones (BRISC is otherwise idle because the write side is an aliased CB) | aliased CB, wait+pop only |
| sharded RM → DRAM-interleaved TILE | aliased CB push **when `pages_per_row == 1`**; raw strided reader when `pages_per_row > 1` | Path-A writer |
| cross-spec reshard (in spec ≠ out spec) | raw strided reader over the input `TensorAccessor` (cross-core L1→L1 via NoC; **never** staged through DRAM) | Path-A writer or aliased CB depending on the output spec |

No inter-Tensix multicast, no semaphores, no ring topology anywhere in this op. Every core's work is
disjoint in both the input and the output address space, so there is **no Tensix-to-Tensix contract
to specify**. (Cross-spec reshard in R3c reads remote L1 through the `TensorAccessor` — still a plain
unicast read with no handshake, because the source shard is fully written before the program starts.)

---

## Work Distribution

**Work unit:** one *chunk-block* = 32 rows × `chunk_wt` tile-columns = `chunk_wt` output tiles.
Each core owns a **2D rectangle**: a contiguous tile-row range × a contiguous column-chunk range.

### Rule this design exists to satisfy

A height-only split (`split_work_to_cores(nt_h)`) strands `[1,1,32,16384]` (`nt_h = 1`) on **one**
core. The `distribution_gate` example measures exactly this collapse at **7.25×** on the analogous
`[32,4096]` shape (`ttnn/ttnn/operations/examples/distribution_gate/report.md`). The split below is
**height-first, width-fills-the-remainder**, which degenerates to a pure height split whenever height
already fills the grid (so the square and tall-narrow regimes are byte-identical to the conventional
split — the `distribution_gate` no-regression property) and to a pure width split when `nt_h = 1`.

### Host planner (formula table — evaluated once per program build)

| # | Symbol | Formula |
|---|---|---|
| 1 | `G` | `1` if `not use_multicore` else `min(grid.x * grid.y, total_tiles)` — `grid = device.compute_with_storage_grid_size()` (8×8 = 64 on this box) |
| 2 | `depth` | `2 if use_double_buffer else 1`, then clamped down to `1` if step 5 would exceed `L1_CB_BUDGET_BYTES` |
| 3 | `bytes_per_chunk_tile` | `tile_in + tile_out` |
| 4 | `max_chunk_l1` | `max(1, L1_CB_BUDGET_BYTES // (depth * bytes_per_chunk_tile))` |
| 5 | `n_h` | `min(nt_h, G)` — height partitions |
| 6 | `want_chunks` | `ceil(G / n_h)` — how many column chunks are needed to fill the remaining cores |
| 7 | `max_chunk_par` | `max(1, Wt // want_chunks)` |
| 8 | `max_chunk` | `min(WT_CHUNK_MAX, max_chunk_l1, max_chunk_par)` |
| 9 | `chunk_wt` | **largest divisor of `Wt` that is `<= max_chunk`** (always ≥ 1) |
| 10 | `n_chunks` | `Wt // chunk_wt` |
| 11 | `n_w` | `min(n_chunks, max(1, G // n_h))` — width partitions |
| 12 | `ncores` | `n_h * n_w` |
| 13 | row ranges | `split_contiguous(nt_h, n_h)`: `base = nt_h // n_h`, `rem = nt_h % n_h`; the first `rem` partitions get `base+1` tile-rows, the rest `base` |
| 14 | chunk ranges | `split_contiguous(n_chunks, n_w)` |
| 15 | core `(i, j)` | physical core `= cores[i * n_w + j]`, `cores = ttnn.grid_to_cores(ncores, grid.x, grid.y, row_wise=True)` |
| 16 | `all_cores` | `ttnn.num_cores_to_corerangeset(ncores, grid, row_wise=True)` |

Constants (both are swept in Refinement 2 and their chosen values recorded in the ledger):

| Constant | Initial value | Rationale |
|---|---|---|
| `WT_CHUNK_MAX` | `16` | Caps read transaction size at 1024 B (bf16) / 2048 B (fp32). Sweep `{4, 8, 16, 32, 64}` in R2. |
| `L1_CB_BUDGET_BYTES` | `131072` (128 KiB, both CBs combined) | Conservative literal — Python has **no** `l1_size_per_core()` binding (verified on this build; `device` exposes only `compute_with_storage_grid_size`, `core_grid`, `dram_grid_size`, `get_optimal_dram_bank_to_logical_worker_assignment`). Bounding CB L1 by a constant is a hard requirement, not a heuristic. |

`chunk_wt` **divides `Wt` exactly** by construction (step 9). This is what makes every core's width a
whole number of chunks, which in turn makes `chunk_wt` a **single compile-time value shared by every
core** — so the program has exactly **one reader, one compute and one writer kernel descriptor**, and
all per-core variation (`row_start`, `row_count`, `chunk_start`, `chunk_count`) lives in runtime args.
No cliff kernels, no 4-way core-class fan-out.

`ttnn.find_max_divisor` is **not bound in this build** (verified) — step 9 is plain Python
(`max(d for d in range(max_chunk, 0, -1) if Wt % d == 0)`), and unlike the C++ helper it must **not**
skip 5 and 7 (skipping them would silently drop `chunk_wt` to 1 for `Wt = 35`, halving parallelism).
`ttnn.div_up` / `ttnn.round_up` are likewise unbound — use `-(-a // b)`.

### Per-core runtime work (derived)

| Quantity | Formula |
|---|---|
| `row_start_tile`, `row_count` | from the core's row range |
| `chunk_start`, `chunk_count` | from the core's chunk range |
| reader `start_page` | `row_start_tile * 32` (stick index) |
| reader `num_rows` | `row_count * 32` |
| reader per-chunk `byte_offset_within_page` | `(chunk_start + c) * chunk_row_bytes`, `c ∈ [0, chunk_count)` |
| compute `num_blocks` | `row_count * chunk_count` |
| writer first page of block `(c, r)` | `(row_start_tile + r) * Wt + (chunk_start + c) * chunk_wt` |

### Worked regime table (A0 conformance — 8×8 grid, bf16, `WT_CHUNK_MAX=16`, depth 2)

| Regime | Shape | `nt_h` | `Wt` | `chunk_wt` | `n_h` × `n_w` | **ncores** | A0 check `min(G, total_tiles)` | ✓ |
|---|---|---|---|---|---|---|---|---|
| square (perf bench a) | `[1,1,2048,2048]` | 64 | 64 | 16 | 64 × 1 | **64** | 64 | ✓ |
| wide-short (perf bench b) | `[1,1,32,16384]` | 1 | 512 | 8 | 1 × 64 | **64** | 64 | ✓ |
| wide-short (golden) | `[1,1,32,4096]` | 1 | 128 | 2 | 1 × 64 | **64** | 64 | ✓ |
| tall-narrow (golden) | `[1,1,2048,64]` | 64 | 2 | 2 | 64 × 1 | **64** | 64 | ✓ |
| medium | `[1,1,64,2048]` | 2 | 64 | 2 | 2 × 32 | **64** | 64 | ✓ |
| tiny | `[1,1,32,64]` | 1 | 2 | 1 | 1 × 2 | **2** | 2 | ✓ |
| small | `[1,1,64,128]` | 2 | 4 | 1 | 2 × 4 | **8** | 8 | ✓ |
| rank-3 | `[2,32,64]` | 2 | 2 | 1 | 2 × 2 | **4** | 4 | ✓ |
| fp32 wide (DeepSeek) | `[1,7168,2304]` fp32 | 224 | 72 | 8 | 64 × 1 | **64** | 64 | ✓ |
| single-core | any, `use_multicore=False` | — | — | `≤ max_chunk_l1` | 1 × 1 | **1** | 1 (forced) | ✓ |
| sharded (Path B) | shard grid | — | — | `shard_W // 32` | shard grid | **= shard cores** | shard's own cores | ✓ |

Every regime satisfies lever **A0** (`ttnn/ttnn/operations/examples/master.md`, Part 2 §A): interleaved
→ `active == min(grid, total_tiles)`; sharded → `active == the shard's own cores`.

### Remainder handling

Remainders are absorbed by `split_contiguous` (first `rem` partitions get one extra unit) on **both**
axes. Because the extra unit is a whole tile-row (height) or a whole chunk (width), the compile-time
`chunk_wt` is unaffected — the imbalance shows up only as a larger runtime `num_blocks`. There are no
cliff cores and no cliff kernels.

### Placement

`row_wise=True` throughout. `noc_placement` measures row/diagonal placement at **~2.9× over column**
on WH B0 with 8 cores, and column is exactly what `row_wise=False` (the library default) produces
(`ttnn/ttnn/operations/examples/noc_placement/report.md`). Reads ride NoC0
(`ttnn.ReaderConfigDescriptor()`), writes NoC1 (`ttnn.WriterConfigDescriptor()`) — the measured
winning pairing (reads NoC0 4.8× on row placement; writes NoC1 4.3×).

---

## Circular Buffers

Only **two** CBs. Tilize is a single-phase compute; there is no intermediate.

| Semantic Name | Index | Page Size | Num Pages | Format | Producer | Consumer | Lifetime |
|---|---|---|---|---|---|---|---|
| `cb_rm_input` | 0 | `tile_in` (`ttnn.tile_size(in_dtype)`) | `depth * chunk_wt` | input dtype | reader (`read_sticks_for_tilize`, Path A) / aliased shard (Path B) | compute (`tilize` helper) | whole kernel |
| `cb_tiled_output` | 16 | `tile_out` (`ttnn.tile_size(out_dtype)`) | `depth * chunk_wt` | **output** dtype | compute (`tilize` helper) | writer (Path A) / aliased shard (Path B) | whole kernel |

`total_size = num_pages * page_size` for both.

### Sizing rationale

| Requirement | Source | Satisfied by |
|---|---|---|
| input CB ≥ `block_width_tiles` pages | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.inl:225` | `depth * chunk_wt ≥ chunk_wt` ✓ |
| output CB ≥ `block_width_tiles` pages | `tilize_helpers.inl:227` | ✓ |
| reader deadlock guard `width_in_tiles ≤ fifo_num_pages` | `tilize_helpers_dataflow.inl:105-108` | ✓ |
| streaming (reader→compute→writer are independent RISCs) ⇒ small buffer suffices | `.claude/references/ttnn-cb-memory-fundamentals.md` | depth 2 is the double-buffer, not a full-block requirement |
| **no** sequential-compute-helper intermediate | only one compute helper in the kernel | no full-block CB needed anywhere |
| per-core CB L1 bounded by a constant in `W` | run requirement + `master.md` A0/B0 | `chunk_wt ≤ WT_CHUNK_MAX` (constant), never `Wt` |

Per-core CB L1 = `depth * chunk_wt * (tile_in + tile_out)`. With `WT_CHUNK_MAX = 16`, depth 2, bf16:
**128 KiB, independent of `W`**. At depth 1: **64 KiB** — this is the `use_double_buffer=False`
saving to record in the ledger.

### CB synchronisation ledger (push count == wait count)

| CB | Producer pushes | Consumer waits | Balanced? |
|---|---|---|---|
| `cb_rm_input` | reader: `chunk_count` helper calls × `row_count` blocks × `chunk_wt` pages | compute: `num_blocks = row_count * chunk_count` waits × `chunk_wt` pages | ✓ identical totals **and** identical per-wait page count |
| `cb_tiled_output` | compute: `num_blocks` pushes × `chunk_wt` pages | writer: `num_blocks` waits × `chunk_wt` pages | ✓ |

**Ordering invariant (the one that will hang the kernel if broken):** the reader iterates
**chunk-outer, tile-row-inner** — because `read_sticks_for_tilize` loops over tile-row blocks
*internally* (`tilize_helpers_dataflow.inl:110`) and the chunk loop is the caller's outer loop. The
writer's page arithmetic MUST use the same nesting order. Reversing it produces correct CB counts and
silently wrong output pages.

### Path B (aliased) CB overrides

| Field | Value |
|---|---|
| construction | `ttnn.cb_descriptor_from_sharded_tensor(index, tensor, core_ranges=shard_grid)` |
| page_size override | must be re-set to `tile_in` / `tile_out` after construction, via the read-modify-write-back idiom (`fds = cb.format_descriptors; fds[0].page_size = ...; cb.format_descriptors = fds`) — nanobind's bound vector copies on `__getitem__`, so in-place mutation is dropped. Live precedent: `ttnn/ttnn/operations/examples/zero_copy_fold/program_descriptor_with_inline_kernels.py:192-194` |
| num_pages | `shard_H // 32 * shard_W // 32` (the full shard) |
| depth | structurally 1 |
| `block_width_tiles` | `shard_W // 32` |

---

## API Mapping

| Phase | Type | Function | File:Line | Template / Args | Input CB | Output CB | Manages own CB ops? |
|---|---|---|---|---|---|---|---|
| Read RM sticks (Path A) | **helper** | `dataflow_kernel_lib::read_sticks_for_tilize` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp:87-93` | `<cb_rm_input, TilizeGranularity::TILE>`; args `(accessor, num_rows = row_count*32, row_bytes = chunk_row_bytes, start_page = row_start_tile*32, byte_offset_within_page = (chunk_start+c)*chunk_row_bytes)`; called once per chunk `c` | — (DRAM/L1) | `cb_rm_input` | **Yes** — `cb_reserve_back(chunk_wt)` / `cb_push_back(chunk_wt)` per block, one `noc_async_read_barrier()` per 32 sticks (`tilize_helpers_dataflow.inl:117-127`). Do **not** wrap it in CB calls. |
| Tilize (all paths) | **helper** | `compute_kernel_lib::tilize` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp:187-197` | `<chunk_wt, cb_rm_input, cb_tiled_output, InitUninitMode::InitAndUninit, WaitMode::WaitBlock, RECONFIG, FP32MODE>`; runtime `(num_blocks)`, **symmetric mode — no `total_input_pages`** | `cb_rm_input` | `cb_tiled_output` | **Yes** — `wait_front` / `reserve_back` / LLK / `push_back` / `pop_front` per block (`tilize_helpers.inl:249-268`). |
| HW init | **raw API** (mandatory prerequisite) | `compute_kernel_hw_startup(cb_rm_input, cb_tiled_output)` | prerequisite documented at `tilize_helpers.hpp:89-93` | 2-arg form (srcA = srcB = input) | — | — | n/a |
| Write TILE pages (Path A) | **raw API** | `noc_async_write` + `noc_async_write_barrier` over a `TensorAccessor` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h` (write), accessor pattern per `tech_reports/tensor_accessor/tensor_accessor.md` | one whole-tile write per page, `chunk_wt` writes per barrier | `cb_tiled_output` | — (DRAM/L1) | **No** — the kernel owns `cb_wait_front(chunk_wt)` / `cb_pop_front(chunk_wt)` per block |
| Read (Path B, aliased) | **raw API** | `cb_push_back(cb_rm_input, shard_tiles)` once | — | — | — | `cb_rm_input` | **No** — one push, no NoC traffic |
| Write (Path B, aliased) | **raw API** | `cb_wait_front(cb_tiled_output, shard_tiles)` then `cb_pop_front(...)` | — | — | `cb_tiled_output` | — | **No** |
| Read (Path C, `pages_per_row > 1`) | **raw API** | strided `noc_async_read` over `TensorAccessor`, page index `row * pages_per_row + page_col` | — | mirrors the helper's block structure: reserve `chunk_wt`, 32 strided reads, one barrier, push `chunk_wt` | — | `cb_rm_input` | **No** |

### `RECONFIG` and `FP32MODE` selection (compute kernel, compile time)

| CT input | Value | Rule |
|---|---|---|
| `needs_cast` (host CT arg, `0/1`) | `int(out_dtype != in_dtype)` | |
| `RECONFIG` | `needs_cast ? ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure : ReconfigureRegisterDatatypeMode::NoReconfigure` | Selected with `if constexpr` on the CT arg. **`NoReconfigure` on every no-cast call.** `compute_kernel_hw_startup(cb_rm_input, cb_tiled_output)` has already programmed srcA/srcB from the input CB and the packer from the output CB (`tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:81-88`), so the reconfigure is pure redundant CFG traffic (~150 ns/op) — dominant on the ~1 µs small sharded cases. Enum at `tilize_helpers.hpp:22-27`. |
| `FP32MODE` | `is_fp32_input_format<cb_rm_input>() ? Fp32Mode::Lossless : Fp32Mode::Fast` | Decided **inside the kernel** from the CB format — no CT arg needed. Enum + rationale at `tilize_helpers.hpp:63-71`. `Lossless` is mandatory for fp32 input: the fast path truncates fp32 → tf32, which fails the bit-identity oracle. |

`Fp32Mode::Lossless` carries two hard `static_assert`s (`tilize_helpers.inl:135-142`) that the host
**must** satisfy or the kernel will not compile:

| Requirement | Host action |
|---|---|
| `DST_ACCUM_MODE` | `ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=True)` |
| `UnpackToDestMode::UnpackToDestFp32` on `cb_rm_input` | `cfg.unpack_to_dest_mode = [ttnn.UnpackToDestMode.UnpackToDestFp32 if i == 0 else ttnn.UnpackToDestMode.Default for i in range(32)]` — verified on this build: the property accepts a plain 32-element Python list; the default value is an **empty** list, so it must be assigned wholesale. |

The inverse `static_assert` (`tilize_helpers.inl:149-155`) forbids fast-tilize + `UnpackToDestFp32` on
an fp32 input — satisfied automatically because `Lossless` forces the slow path for fp32 input.

### ComputeConfigDescriptor (host)

| Field | Value |
|---|---|
| `fp32_dest_acc_en` | `in_dtype in {float32, uint32, int32} or out_dtype in {float32, bfloat8_b, uint32, int32}` |
| `unpack_to_dest_mode[0]` | `UnpackToDestFp32` iff `fp32_dest_acc_en`, else `Default`; all other slots `Default`; list length 32 |
| `math_fidelity` | default (`HiFi4`) — irrelevant, no math |
| `dst_full_sync_en` | `False` — `can_use_fast_tilize` requires `!get_dst_full_sync_enabled()` (`tilize_helpers.inl:95`); enabling it silently disables fast tilize for bf16 |

The 32-bit-integer terms in `fp32_dest_acc_en` extend native's predicate. Rationale: `uint32`/`int32`
tiles carry a 32-bit payload that a 16-bit DEST would truncate. This is a **conservative** choice
(it can only cost DEST capacity, which tilize does not use) and is an explicit R4 verification item.

### Helpers considered and rejected

Every non-helper mechanism above, with the file:line in the helper's **own** source that proves the
mismatch.

| Raw mechanism | Helper considered | File:Line of the mismatch | Concrete reason |
|---|---|---|---|
| Path-A writer (`noc_async_write` of tile pages) | `dataflow_kernel_lib::write_sticks_after_untilize` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp:129-135`, body at `tilize_helpers_dataflow.inl:232-236` | The only write helper in `kernel_lib` writes **row-major sticks**: its inner loop issues one `noc_async_write` of `row_bytes` per row and advances the L1 pointer by `padded_row_bytes`, i.e. it de-interleaves a tile back into 32 sticks. tilize's output is TILE pages; using it would write the tile bytes to stick addresses and destroy the layout. It is the untilize partner (doc line `:109-110`), the wrong direction. |
| Path-A writer | any other `kernel_lib` dataflow helper | `dfb_helpers_dataflow.hpp:15-19` (only `get_tile_r_dim` / `get_tile_c_dim`), `l1_helpers.hpp:26-70` (only local-L1 pointer cast, `local_noc_addr`, `zero_tile`, `prepare_zero_tile`), `mcast_pipe.hpp` (multicast only) | No helper in `ttnn/cpp/ttnn/kernel_lib/` moves tile pages from a CB to a `TensorAccessor`-addressed buffer. The three dataflow-namespace headers are `tilize_helpers_dataflow`, `reduce_helpers_dataflow`, `dfb_helpers_dataflow`; none exposes such a function. |
| Path-B reader (`cb_push_back` only) | `read_sticks_for_tilize` | `tilize_helpers_dataflow.inl:117-127` | The helper's TILE-mode body is unconditionally `cb_reserve_back` → `noc_async_read` × 32 → `noc_async_read_barrier` → `cb_push_back`. There is **no** "data already resident" mode and no way to suppress the NoC reads (the only template knob is `TilizeGranularity`, `hpp:41-44`). On the zero-copy path the data is already at the CB's L1 address, so calling it would issue a full redundant DRAM/L1 round-trip of the entire shard — exactly the traffic the aliasing exists to eliminate. |
| Path-C reader (`pages_per_row > 1`) | `read_sticks_for_tilize` | `tilize_helpers_dataflow.inl:121` (`accessor.get_noc_addr(start_page + block_row + row, byte_offset_within_page)`); signature `tilize_helpers_dataflow.hpp:87-93` | The page index advances by **exactly 1 per row**, hard-coding "one page == one full logical row". A ROW_MAJOR-*sharded* input has `pages_per_row = W // shard_W > 1`, so consecutive rows are `pages_per_row` pages apart. The signature has no row-stride parameter, so the required addressing is inexpressible. The helper **is** used whenever `pages_per_row == 1` (all interleaved inputs and all HEIGHT-sharded inputs whose shard spans the full width). |
| `compute_kernel_hw_startup` | — | `tilize_helpers.hpp:89-93` | Not a rejection: the helper's own documentation makes this the caller's mandatory prerequisite. |

### Helpers considered and **rejected as an algorithm** (not as an implementation detail)

| Helper / mode | File:Line | Why not used |
|---|---|---|
| `read_sticks_for_tilize<..., TilizeGranularity::ROW>` + `tilize(num_blocks, total_input_pages)` (asymmetric mode) | `tilize_helpers_dataflow.inl:148-157`; asymmetric contract at `tilize_helpers.hpp:127-136` | ROW mode issues `cb_reserve_back(1)` → one read → **`noc_async_read_barrier()` per single row** (`tilize_helpers_dataflow.inl:155`) — a barrier per 512-byte transaction, violating lever **B7** (one barrier per block). Its documented advantage (`hpp:64-65`) is a smaller CB "when `total_num_rows < 32`", which **can never occur in this op**: `H % 32 == 0` guarantees every core's row count is a positive multiple of 32. So ROW mode is all cost and no benefit here. TILE mode is used everywhere. |
| `compute_kernel_lib::untilize` | `untilize_helpers.hpp:145-154` | Opposite direction. |

---

## Compute Phases

Single compute phase. The table below is the sequential contract for **one core**, one iteration of
the reader's chunk-outer loop; the compute kernel makes **one** `tilize` call covering all
`num_blocks = row_count * chunk_count` blocks.

| # | Operation | Helper? | Input CB (name, pages, state) | Output CB (name, pages) | CB state after |
|---|---|---|---|---|---|
| 0 | HW init | raw (`compute_kernel_hw_startup(cb_rm_input, cb_tiled_output)`) | — | — | srcA/srcB programmed from `cb_rm_input`; packer from `cb_tiled_output` |
| 1 | RM → TILE, `num_blocks` blocks of `1 × chunk_wt` | **yes** — `compute_kernel_lib::tilize<chunk_wt, cb_rm_input, cb_tiled_output, InitAndUninit, WaitBlock, RECONFIG, FP32MODE>(num_blocks)` | `cb_rm_input`, `chunk_wt` pages per block, produced by the reader | `cb_tiled_output`, `chunk_wt` pages per block | both CBs drained back to empty after the writer pops; helper issues `tilize_init` once and `tilize_uninit` once (`InitAndUninit`) around the whole `num_blocks` loop |

Phase-boundary contract with the dataflow kernels:

| Boundary | Producer contract | Consumer contract |
|---|---|---|
| reader → compute | pushes `chunk_wt` pages per block, in chunk-outer/row-inner order, `num_blocks` times total | waits `chunk_wt` pages per block, `num_blocks` times |
| compute → writer | pushes `chunk_wt` pages per block, `num_blocks` times, in the same order | waits `chunk_wt`, writes them to pages `base_page .. base_page+chunk_wt-1`, pops `chunk_wt` |

There is **no** persistent CB, no data that must survive across phases, and no second compute helper —
so none of the "size the intermediate to a full block" rules from
`.claude/references/ttnn-cb-memory-fundamentals.md` apply. Depth-2 is a *pipelining* choice, not a
correctness requirement, which is precisely why `use_double_buffer=False` is safe.

---

## Registry Contract (guidance for the implementer)

`INPUT_TAGGERS`, `SUPPORTED`, `EXCLUSIONS`, `validate()` are declared inline in
`ttnn/ttnn/operations/tilize/tilize.py`, per `eval/op_template.py`. `INVALID` is **not** declared by
the op — it lives in `eval/golden_tests/tilize/feature_spec.py` and the harness skips those cells.

### Taggers (names and sources are fixed by `feature_spec.py`'s docstring)

| Tagger | Reads | Returns |
|---|---|---|
| `tag_use_multicore` | `scenario["use_multicore"]` | `bool` |
| `tag_shard_api` | `scenario["shard_api"]` | `"none" \| "legacy_2d" \| "nd"` |
| `tag_out_scheme` | `scenario["out"]` | `"interleaved"` if the out spec is interleaved, else its `scheme`, or `"nd"` when `scheme is None` |
| `tag_buffer` | `scenario["in"]["buffer"]`, `scenario["out"]["buffer"]` | `f"{in}_to_{out}"` over `{dram, l1}` |
| `tag_rank` | `len(scenario["input_shape"])` | `int` |
| `tag_double_buffer` | `scenario.get("use_double_buffer", True)` | `bool` |

When `validate()` builds its own axes dict from the real call arguments, it must derive the same axis
names from `input_tensor.memory_config()` / the requested output memory config / `use_multicore` /
`use_double_buffer` / `len(shape)`.

### Phase-0 `SUPPORTED`

```
dtype: [bfloat16]          output_dtype: [bfloat16]     use_multicore: [False]
shard_api: ["none"]        out_scheme: ["interleaved"]  buffer: ["dram_to_dram"]
rank: [4]
```

**`double_buffer` must NOT appear in `SUPPORTED` before Refinement 6.** `eval/feature_matrix.py:150-153`
(`is_supported`) ignores axes the op does not declare, so omitting it is what keeps the two
`use_double_buffer=False` golden cells judged purely on their other axes. Declaring it as `[True]`
earlier would demand an xfail the op cannot produce (see the coverage caveat below) and turn those
cells red on XPASS-strict.

### `EXCLUSIONS` (kernel-scope refusals, promotable by a later refinement)

| Cell | Why |
|---|---|
| `{"use_multicore": False, "shard_api": "legacy_2d"}` and `{"use_multicore": False, "shard_api": "nd"}` | Sharded I/O is inherently multi-core (one shard per core); a single-core sharded program is not a thing the op will build. |
| any `out_scheme` not yet wired (`HEIGHT` / `WIDTH` / `BLOCK` / `"nd"` before R3) | not-yet-implemented, promoted per refinement |

### Coverage caveat the implementer must know

`eval/golden_tests/tilize/helpers.py:run_tilize` calls
`tilize(tt_input, memory_config=..., dtype=..., use_multicore=...)` and **never passes
`use_double_buffer`** — its `**_` swallows the tagged `double_buffer` axis. So the depth-1 CB path is
**not exercised by the golden suite at all**, even after R6: the harness tags `double_buffer=False`
while the op runs its default `True`. Consequences:

1. At R6, `SUPPORTED["double_buffer"] = [True, False]` — both values must be accepted, or those two
   cells fail. There is no way for `validate()` to see `False` from the harness.
2. **The depth-1 path's only correctness coverage is the acceptance test**
   (`tests/ttnn/unit_tests/operations/tilize/test_tilize.py::test_tilize_double_buffer_off`), which
   passes `use_double_buffer=False` explicitly. Do not regress it.
3. The L1 saving and read/write-overlap cost are measured on the perf bench, not the golden suite.

### Structural impossibilities (INVALID candidates)

`feature_spec.py` already declares the five `dtype`↔`output_dtype` int/float crosses, and they pass
all three sanity rules from `eval/REGISTRY_MODEL.md` (single-tensor coupling — both axes describe the
same tensor's storage format; universe-must-change — an int↔float reinterpretation is a different
operation, not a kernel improvement). **No additional structural impossibilities identified.** Two
non-INVALID observations for the record:

- `{use_multicore: False, shard_api: legacy_2d/nd}` is an **EXCLUSIONS** case, not INVALID — a future
  refinement could legitimately build a one-shard single-core program.
- `bfloat8_b` correctly does not appear in `TARGET["dtype"]`: block-float has no row-major form, and
  the input is always ROW_MAJOR. It appears only in `TARGET["output_dtype"]`, which is legal.

---

## Key Risks and Gotchas

| # | Risk | Detail / mitigation |
|---|---|---|
| **1** | **`tilize_helpers.inl` currently does not compile.** | `has_unpack_to_dest_fp32` is defined **twice, byte-identically**, at `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.inl:47-63` and `:65-81` — same template parameter list, same signature, same TU ⇒ `error: redefinition`. Verified at HEAD (`grep -n has_unpack_to_dest_fp32` reports definitions at lines 48 and 66). **Every kernel that includes `tilize_helpers.hpp` will fail to JIT-compile.** Fix: delete the second copy (lines 65-81). This is the first thing to do, before any Phase-0 build. |
| **2** | Reader/writer nesting order | The reader's chunk loop is **outer**, `read_sticks_for_tilize`'s tile-row loop is **inner** (`tilize_helpers_dataflow.inl:110`). The writer must compute `base_page = (row_start + r) * Wt + (chunk_start + c) * chunk_wt` with `c` outer, `r` inner. Getting this backwards keeps every CB count balanced and produces silently transposed output blocks — no hang, no assert, just wrong data. |
| **3** | `chunk_wt` must divide `Wt` | Step 9 of the planner. If it does not, cores get fractional chunks and the single-compile-time-`chunk_wt` invariant breaks, forcing cliff kernels. Assert it host-side. |
| **4** | `tilize` cannot be in-place | `static_assert(input_dfb != output_dfb)` at `tilize_helpers.inl:116`. `cb_rm_input` (0) and `cb_tiled_output` (16) must stay distinct even when `in_dtype == out_dtype`. |
| **5** | Block-float input is forbidden | `UNPACK(ASSERT(!is_block_float_format(unpack_src_format[input_dfb])))` at `tilize_helpers.inl:174`. `bfloat8_b` may only be an **output**. The op must reject a `bfloat8_b` input before dispatch. |
| **6** | fp32 must use `Fp32Mode::Lossless` | Fast tilize truncates fp32 → tf32 (`tilize_helpers.hpp:49-53`), which fails the bit-identity oracle and the `test_tilize_fp32_truncation` / `test_tilize_fp32_lossless` grader cells. Lossless requires **both** `fp32_dest_acc_en=True` **and** `unpack_to_dest_mode[0] = UnpackToDestFp32`, or the kernel refuses to compile (`tilize_helpers.inl:135-142`). |
| **7** | `unpack_to_dest_mode` must be assigned wholesale | The default value is an **empty** list (probed on this build). nanobind's bound vector copies on `__getitem__`, so `cfg.unpack_to_dest_mode[0] = ...` is silently dropped. Build a 32-element Python list and assign it in one statement. |
| **8** | Missing Python utility bindings | `ttnn.find_max_divisor`, `ttnn.div_up`, `ttnn.round_up`, `device.l1_size_per_core()` are **not** available on this build (all probed). `.claude/references/ttnn-python-utility-bindings.md` documents some of them — do not trust it here. Use plain Python arithmetic and the `L1_CB_BUDGET_BYTES` constant. |
| **9** | Aliased-CB page-size override | `cb_descriptor_from_sharded_tensor` derives `page_size` from the tensor's dtype/tile, which for a ROW_MAJOR-sharded **input** is not what the tilize helper wants. Override via read-modify-write-back (precedent: `zero_copy_fold/program_descriptor_with_inline_kernels.py:192-194`). |
| **10** | Sharded-path program-cache re-binding | With both buffer addresses riding on CBs and no `Buffer*` runtime arg, nothing forces address re-patching on a cache hit. Emit at least one runtime arg on the first core so `apply_resolved_bindings` re-patches the CB base addresses (`tt_metal/impl/program/program_descriptors.cpp:208`). |
| **11** | `pages_per_row > 1` on sharded RM input | A WIDTH- or BLOCK-sharded RM input has `page_bytes == shard_W * elem_size < W * elem_size`, so one logical row spans several pages. The helper cannot express this (see the rejection table). Detect `pages_per_row` host-side and select the raw strided reader. R3c also has to strip shard-width padding on the last page. |
| **12** | `dst_full_sync_en` silently disables fast tilize | `can_use_fast_tilize` requires `!get_dst_full_sync_enabled()` (`tilize_helpers.inl:95`). Leave it `False`. |
| **13** | `WaitMode::NoWait` still pops | `NoWait` suppresses only `wait_front`; `pop_front`, `reserve_back` and `push_back` still run unconditionally (`tilize_helpers.inl:249-268`). This design uses `WaitBlock` everywhere; do not "optimize" to `NoWait` and also pop by hand. |
| **14** | DRAM read alignment | Every read source address is `page_base + chunk_index * chunk_wt * 32 * elem_in`, always a multiple of `32 * elem_in ≥ 32 B` — satisfying the 32-B DRAM read alignment (lever B11) for every supported dtype. The DRAM-alignment staging CB that native's block factory carries is therefore **not needed**; do not add one. |
| **15** | `total_tiles < grid_cores` | `G = min(grid, total_tiles)` (planner step 1) is what keeps A0 satisfied on the tiny regime. Launching 64 cores for a 2-tile tensor is an A0 violation and pure dispatch overhead. |

---

## Performance Methodology

**This is a per-phase gate, not an end-of-run pass.** Every refinement below runs Steps 1–4 before it
is allowed to check off.

### Bound classification (Refinement 0, and re-checked whenever the algorithm changes)

tilize is **mixed**: it has a real compute stage (the tilize LLK), so the NoC ceiling is only a
*partial* bound. Run the `/perf-measure` ablation before chasing any DM lever:

| Variant | How | Reads as |
|---|---|---|
| Full | as shipped | baseline |
| No-compute | `constexpr SKIP_COMPUTE` CT flag: keep every CB reserve/wait/push/pop and the loop trip counts, skip the LLK call | duration ≈ Full ⇒ **DM-bound**, the NoC ceiling is the target |
| No-DM | `constexpr SKIP_DM` CT flag: keep every CB op and barrier, skip `noc_async_read`/`noc_async_write` | duration ≈ Full ⇒ **compute-bound**, the DM ceiling does not apply |
| Sync-only | both flags | residual dispatch/sync floor |

R0 must record the classification verdict in `changelog.md`. Every later DM claim rests on it.

### Perf bench (separate from the golden suite)

`tests/ttnn/unit_tests/operations/tilize/_bench_tilize.py` (underscore-prefixed, **not** in
`feature_spec.INPUTS`, asserts **no** PCC — measurement and ablation need no correctness). Run under
`tt_metal/tools/profiler/profile_this.py --collect-noc-traces`.

| Regime | Shape | Why it is mandatory |
|---|---|---|
| **(a) grid-filling square** | `[1, 1, 2048, 2048]` bf16 | per-core DRAM efficiency once the grid is already full |
| **(b) wide, short** | `[1, 1, 32, 16384]` bf16 (`nt_h = 1`, `Wt = 512` — the DeepSeek MLA `wo` shape) | **whether the split actually fills the grid.** A bench that measures only (a) reports healthy while a height-only split strands (b) on one core. **If (b) runs on `< grid_cores` cores the perf gate fails**, regardless of (a). |
| (c) single-core reference | `[1, 1, 512, 512]` bf16 | R0's single-core baseline, sized so the run is ~100 µs not ~1 ms |
| (d) tall-narrow guard | `[1, 1, 2048, 32]` bf16 | no-regression witness for the height regime (the `distribution_gate` discipline) |
| (e) dtype sweep | (a) at fp32 and bf8b-out | page size changes the bound — R4 re-runs the ceiling per dtype |
| (f) sharded | same-spec HEIGHT/WIDTH/BLOCK L1 shards, small (~1 µs) and large | R3's zero-DRAM claim and the `NoReconfigure` A/B |

Every bench cell must report **device core count** alongside the duration, so the A0 assertion is
machine-checkable rather than eyeballed.

### Ceiling computation (`/perf-ceiling-dm`)

| Quantity | Formula |
|---|---|
| `bytes_read` | `folded_H * W * elem_in` |
| `bytes_written` | `total_tiles * tile_out` |
| `dram_traffic` | `bytes_read + bytes_written` (Path A); **0 on the output side** for Path B, **0 on both sides** for same-spec sharded |
| read transfer group | `pattern=ALL_FROM_ALL`, `memory=DRAM_INTERLEAVED`, `transaction-size-bytes = chunk_wt * 32 * elem_in`, `num-transactions-per-barrier = 32` |
| write transfer group | `pattern=ALL_TO_ALL`, `memory=DRAM_INTERLEAVED`, `transaction-size-bytes = tile_out`, `num-transactions-per-barrier = chunk_wt` |
| composition | depth-2 ⇒ read and write groups overlap ⇒ `max`; depth-1 ⇒ `sum` (Step 4). Then `op_target = max(per_core_noc_bound, dram_traffic / dram_peak)` (Step 4b). |

Reference data points to bracket against (all WH B0, from in-tree measured reports — **not** CI bounds):

| Reference | Number | Source |
|---|---|---|
| 64-core spread DRAM→DRAM copy, `[2048,2048]` bf16, 16.78 MB traffic | **86.6 µs @ 193.8 GB/s** | `examples/dram_saturation/report.md` |
| bandwidth knee | **~16 cores @ 190.9 GB/s** (98% of peak); 16→64 cores buys +1.5% | same |
| stacked (column) placement penalty | 71.8 GB/s @ 8 cores vs 146.3 spread (**2×**) | same |
| 1-core bf16 depth-2 ceiling | **17.9–18.3 GB/s** (transaction-rate-limited at ~8.9 M transactions/s) | `examples/double_buffer/report.md` |
| 1-core fp32 / bf8b ceilings | 31.7–32.9 / 9.8–10.0 GB/s | same |
| row vs column placement | **2.91×** | `examples/noc_placement/report.md` |
| reads NoC0 / writes NoC1 | **4.8× / 4.3×** on row placement | same |

So: **R0 square-equivalent target** ≈ `dram_traffic / 18 GB/s`; **R1/R2 square target** ≈ `86.6 µs`
adjusted upward for the smaller read transaction (1024 B vs the reference's 2048 B pages — that ratio
is itself the R2 `WT_CHUNK_MAX` sweep's hypothesis). **Wide-short (b)** moves only 2.10 MB, so at
193.8 GB/s the DM floor is ~10.8 µs and the op is likely launch/latency-bound; **its gate is the core
count, not the absolute number.**

Pin each bracket to one congestion-modeled number with tt-npe (`/perf-ceiling-dm` Step 6):
`tt_npe.sh out_dir/.logs/<trace>.json --noc-trace`, plus `--cong none` to isolate the congestion cost.
Record cycles + DRAM util + congestion % and the binding resource.

### Mode A — candidate algorithms, ranked before writing a kernel

Enumerated by walking `master.md` Part 2, not brainstormed.

| # | Candidate | Predicted target | Verdict |
|---|---|---|---|
| **C1** | **FPU tilize, 2D height-first rectangular split, width-chunked reader (this design)** | read `chunk_wt*32*elem` transactions batched 32/barrier, write whole 2048 B tiles batched `chunk_wt`/barrier, both NoCs, depth-2 | **ADOPT** |
| C2 | NoC-only face scatter — no compute; reader/writer reorder RM→tile by moving 16-element face rows | 32 B per transaction (bf16) vs 2048 B ⇒ 64× the transaction count at a ~fixed ~110–125 ns per completed transaction (`double_buffer/report.md`) ⇒ **~64× worse**, and far below the 512 B one-packet threshold (B6) | REJECT |
| C3 | Height-only split (native default factory) | `1` core on `[1,1,32,16384]`; `distribution_gate` measures **7.25×** worse on the analogous shape | REJECT — violates A0 |
| C4 | Square-block 2D split (`split_blocks_for_tilize_wh` / `closest_square_larger_than_b`, `ttnn/cpp/ttnn/operations/core/work_split/work_split_tilize.hpp:61,171`) | comparable on the square; wastes cores when `nt_h` and `Wt` are not near-square; forces **4 core classes ⇒ 4 kernel instances**; A0 explicitly names "square-block" as a latent-bug fixed axis | REJECT in favour of C1's rectangular split (1 kernel instance, all variation in RT args) |
| C5 | `TilizeGranularity::ROW` asymmetric CB | one `noc_async_read_barrier()` **per row** (`tilize_helpers_dataflow.inl:155`) — violates B7; its only stated benefit (`hpp:64-65`) requires `total_num_rows < 32`, impossible here since `H % 32 == 0` | REJECT |
| **C6** | Zero-copy aliased CBs for same-spec L1 shards | **zero** DRAM traffic on both sides ⇒ target collapses to the compute+CB-sync floor | **ADOPT** for the sharded regime (R3) |
| **C7** | Split reader (both DM RISCs read) on DRAM→sharded | BRISC is idle on that path (write side is an aliased CB); `split_reader` measures **up to 1.7×** | **ADOPT** for R3b |

### Mode C — used-optimization ledger (run per refinement)

For each lever the kernel **actually uses**: estimate the counterfactual (re-run Steps 1–4b with that
one lever flipped off), then confirm on device with `/perf-measure` (median of the trial loop, N≥3
adaptively to 5/10, warm-up discarded; deltas within ~2–3% are noise). Record
`lever → predicted delta → measured delta → keep/drop/neutral`.

| Lever | master.md ID | How to flip it off | Counterfactual regime (B0) |
|---|---|---|---|
| 2D height-first split | A0 | force height-only | measure on **wide-short (b)** |
| `row_wise=True` placement | A1 | `row_wise=False` (column) | square (a) — expect ~2.9× |
| only cores with data (`G = min(grid, total_tiles)`) | A0/A2 | launch full grid always | **tiny** regime |
| 32 reads per barrier | B7 | barrier per read | (a) and (c) |
| whole-tile writes | B5 | 4× face writes | (a) |
| reads NoC0 / writes NoC1 | B9 | swap the configs | (a) — expect 2.5–4.8× |
| depth-2 CBs | C16 | `use_double_buffer=False` | (a) **and** (f)-small — record the L1 bytes/core saved (128 KiB → 64 KiB at `WT_CHUNK_MAX=16`, bf16) |
| `chunk_wt` (transaction size) | B5/B6 | sweep `WT_CHUNK_MAX ∈ {4,8,16,32,64}` | (a) |
| aliased CBs (zero-copy) | C14 | route the shard through DRAM | (f) — expect the DRAM traffic to appear in tt-npe |
| `ReconfigureRegisterDatatypeMode::NoReconfigure` | (compute-side) | force `UnpackAndPackReconfigure` | **(f)-small** — the ~150 ns fixed cost is 8–19% of a ~1 µs kernel; negligible on (a) |
| `TensorAccessorArgs` as CT args | D18 | runtime address-gen | (a) |
| base-addresses-only RT args (program cache) | D19 | rebuild per call | 2nd-call latency |
| split reader | B7/`split_reader` | single reader | R3b crossover |

**Per-core-overhead levers (B0) must be counterfactualed on the smallest regime they run in.** A lever
that pays on (a) and regresses (f)-small must be gated on a work-per-core threshold, not applied
globally.

### Mode D — completeness ledger (Refinement 7, run-closing)

Walk **all 24** `master.md` Part 2 levers plus the Part 1 examples; classify every one the kernel does
**not** use as `not-applicable` / `deferred` / `measured-no-payoff` / `missed`, with a predicted delta
for anything not clearly not-applicable. **A0–A2 are graded per regime** (tall-narrow, wide-short,
square, tiny, sharded-in), with the *actual* measured core count asserted against A0 — never
holistically. Pre-classification, to be confirmed or overturned by measurement:

| Lever | ID | Pre-classification | Note |
|---|---|---|---|
| active-core count per regime | A0 | **applied** | per-regime table above; assert measured core count |
| spread across the DRAM-facing axis | A1 | **applied** | `row_wise=True` |
| launch only on cores with data | A2 | **applied** | `G = min(grid, total_tiles)`; sharded ⇒ shard grid |
| reader adjacent to its bank | A3 | deferred | needs `get_optimal_dram_bank_to_logical_worker_assignment`; predicted small on top of A1 — quantify |
| cliff-core specialization | A4 | **not-applicable** | the design has no cliff kernels by construction (single `chunk_wt`); remainder rides in RT args |
| per-core-overhead gating | B0 | **applied** | the `NoReconfigure` and depth-2 decisions are both gated |
| coalesce whole pages | B5 | **applied** | whole-tile writes; whole-chunk stick reads |
| one-packet ≤512 B | B6 | measured-tradeoff | ≤512 B is a *latency* path; `double_buffer` shows bigger transactions win *bandwidth*. Resolved by the `WT_CHUNK_MAX` sweep. |
| one barrier per block | B7 | **applied** | 32 reads / `chunk_wt` writes per barrier |
| trid double-issue | B8 | **deferred** | predicted delta: keeps ≥1 request in flight across the barrier; estimate on (a) |
| reader NoC0 / writer NoC1 | B9 | **applied** | Reader/Writer config descriptors |
| per-reader VC assignment | B10 | deferred | estimate on (a) at full grid |
| alignment | B11 | **applied** (automatic) | all offsets are multiples of `32*elem_in` |
| multicast | B12 | **not-applicable** | no shared input; every core's read set is disjoint |
| `set_state` / `with_state` | B13 | deferred | 32 same-shape reads per block is exactly its use case; estimate on (a) |
| zero-copy CB aliasing | C14 | **applied** (R3) | |
| prefer sharded over interleaved | C15 | **not-applicable** | the memory layout is the caller's, not the op's |
| depth-2 CBs when they pay | C16 | **applied** (R6) | with the L1 auto-fallback |
| in-place / no-copy | C17 | **not-applicable** | RM→TILE always changes the byte layout; there is no overlap case |
| CT `TensorAccessorArgs` | D18 | **applied** | |
| base-address-only RT args | D19 | **applied** | program-cache hit on 2nd call is an acceptance criterion |
| special-case factory selection | D20 | **applied** | Path A / B / C selection by memory-config match |
| host-precomputed indexing, pow2 fast addr-gen | D21 | **applied** (partly) | all per-core indexing is host-computed; `InterleavedAddrGenFast` is subsumed by `TensorAccessor` |
| Metal Trace + multi-CQ | E22 | **not-applicable** | whole-model scope, outside a single op |

### Per-refinement perf gates

| Refinement | Gate |
|---|---|
| **R0** baseline | golden pass; ceiling target recorded; device duration (median of the trial loop) recorded; **ablation classification recorded** (DM-bound vs compute-bound) — the claim every later DM item rests on |
| **R1** multi-core + width blocking | golden pass with `use_multicore`; cycles ↓ vs R0 **and scale with core count on both (a) and (b)**; (b) must run on **~64 cores** (verify in the profiler / tt-npe) — a height-only split that collapses (b) fails the gate even if (a) looks perfect; used-optimization audit for every lever landed |
| **R2** close the gap | measured latency approaches the tt-npe pin; achieved BW ↑; congestion characterized; `WT_CHUNK_MAX` sweep recorded; every landed lever in the ledger |
| **R3 / 3b / 3c / 3d** sharded | golden pass per scheme; **tt-npe shows zero output-side DRAM** (R3) and zero DRAM on the sharded side (R3b); no hangs on R3c; per-core CB L1 **constant in `W`** on R3d |
| **R4** dtype + cast | golden pass; ceiling **re-run per dtype** (page size changes the bound) |
| **R5** rank 2/3 + L1 buffers | golden pass |
| **R6** `use_double_buffer` | identity for **both** values (the depth-1 coverage is the acceptance test — see the coverage caveat); per-core CB L1 in bytes recorded at both depths; perf delta on a wide bench shape; L1 saving + overlap cost in the ledger |
| **R7** retrospective | completeness ledger covering **every** master.md lever + ranked remaining opportunities; no regression on any prior bench |

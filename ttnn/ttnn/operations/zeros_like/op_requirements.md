# Operation Requirements: zeros_like

## Definition
- **Formula**: `output[i] = +0.0` for every element `i`; output shape, dtype, layout = the input's; output placed in the input's buffer (DRAM / L1), interleaved. The input's values are never read (spec R1).
- **PyTorch Reference**: `torch.zeros_like(tensor)` (spec `reference`: `torch.zeros_like(tensor.double())`)
- **Import Path**: `from ttnn.operations.zeros_like import zeros_like`
- **Function Signature**: `zeros_like(tensor: ttnn.Tensor) -> ttnn.Tensor` (exactly the specification's stub: no `dtype`, `layout`, `device`, `memory_config`, `optional_tensor`, and no `compute_kernel_config`. The op computes nothing and has no compute config.)
- **Specification**: `eval/specs/zeros_like/spec.py`

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.
> **Backend (this run)**: Quasar on ttsim, which has no clock. A perf refinement is gated on a **structural count** read off the op by a test (backend page, *Performance measurement*), and its gain in time is unconfirmed until a timed backend (`ttsim-perf`) or hardware measures it. Never report a simulator timing as a speedup.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED layout**: [TILE_LAYOUT]
- **SUPPORTED placement**: memory_config ∈ {DRAM}, sharding ∈ {interleaved}
- **SUPPORTED shape-derived axes**: alignment ∈ {tile_aligned, w_non_aligned, h_non_aligned}, rank ∈ {2, 3, 4}
- **SUPPORTED op-specific axes**: none (the op takes one tensor and no argument)
- **Cores**: full grid (32 clusters), output pages split in contiguous ranges (`split_work_to_cores(..., row_wise=True)`)
- **Thread plan**: (R, C, W) = (0, 0, 1): writer-only, no reader (R1), no compute kernel
- **Compute config**: none (no compute kernel; the spec gives the op no compute config)
- **Golden baseline**: 21 / 21 SUPPORTED cells passing; 147 `xfail_expected`, 0 loud (per `verifier_report.json`)

### [ ] Refinement 1 — float32, ROW_MAJOR layout and L1 placement on the native writer path

**Goal**: add `ttnn.float32` to `SUPPORTED["dtype"]`, `ttnn.ROW_MAJOR_LAYOUT` to `SUPPORTED["layout"]` and `"L1"` to `SUPPORTED["memory_config"]` (interleaved), all on the built regime `fill_from_local_zero_page`, with the kernel handling each at its data-access boundary: the page geometry comes from the output buffer (`buffer_page_size()` / `buffer_num_pages()`: one 4096 B tile for float32, one row of `W · elem_size` bytes for ROW_MAJOR), and the same `TensorAccessor` addresses L1 banks for an L1 output allocated by `ttnn.allocate_tensor_on_device(tensor.spec, device)` (R3). This closes all of `TARGET − SUPPORTED` (96 + 51 = 147 golden cells).

**Implementation skill**: /memory-layouts, /numeric-formats-metal

**Verifier notes**:
- With SUPPORTED temporarily widened to TARGET, the golden suite passed **168 / 168** with no kernel change (verifier experiment, reverted). The golden cells do **not** prove coverage, though: a fresh buffer on the simulator may already be zero. The work of this refinement is the proof that every page is written, plus whatever the proof turns up. Extend `test_every_page_written` (pre-filled non-zero output through `create_program_artifacts(out)`) over dtype × layout × placement, at W = 1 and W = 2, on shapes that engage all 32 clusters with a ragged split. Include ROW_MAJOR rows whose byte count is not a multiple of the L1 (16 B) or DRAM (64 B) alignment, e.g. W = 17 / 47 / 50 bf16 = 34 / 94 / 100 B. Also run one same-session test per new value: an implicit-sync program before, then `ttnn.to_layout` (or for a ROW_MAJOR result, a `ttnn.to_layout(..., TILE_LAYOUT)`) after.
- **Do not** take the `/numeric-formats-metal` steps that expose `compute_kernel_config` or set intermediate-CB formats. The signature is fixed by the specification, and the op has no compute kernel and no ring. From that skill, take only the dtype gate and the dtype matrix of the tests. A zero page is the same bits in bf16 and fp32.
- Native path only: no `ttnn.to_layout` / `ttnn.tilize` / `ttnn.to_memory_config` wrapper anywhere (one dispatch per call). ROW_MAJOR pages wider than `ZERO_SRC_MAX_BYTES` (64 KiB) take the kernel's existing `chunks_per_page` path. No INPUTS row reaches it (widest: 8192 fp32 = 32 KiB), so add one direct test with a row > 64 KiB fp32 (W ≥ 16385) to exercise it.
- `memory_config` L1 has no skill (sharding/placement is not `/memory-layouts`). It is interleaved L1, so it shares the DRAM accessor path; nothing about sharding changes (`sharding` stays `["interleaved"]`, and a sharded input stays refused by name: `test_refuses_sharded_input_by_name`).
- Ship it performant: keep the full-grid page split and one trid barrier per block on the new paths. ROW_MAJOR multiplies the page count (`(1,1,2048,256)`: 512 tiles → 2048 rows), so check that `pages_per_barrier` still bounds the per-core block (64 rows/core) and that no per-page barrier creeps in.
- Lands before Refinement 2, which tunes the L1 path this refinement opens.

**Done when**: `SUPPORTED` equals `TARGET`; the golden suite passes 168 / 168 with 0 in every loud category; the pre-filled every-page tests and the one-session tests pass for every new value; `test_refuses_sharded_input_by_name` still passes.

### [ ] Refinement 2 — Stagger each core's write order across banks (L1 and DRAM outputs)

**Type**: perf

**Goal**: no `LOOSE_CASES` entry is perf-flagged, so the region is free-selected: the large outputs, `(1024,1024)` / `(2,512,1024)` (1024 pages) and `(1,1,2048,256)`, in both placements, in TILE and ROW_MAJOR. Each core owns a contiguous page range and walks it from its first page, while interleaved pages go round-robin over the banks (32 L1 banks, 2 DRAM banks on this part). So at every issue step every core writes the same bank: for an L1 output of 1024 pages, all 32 clusters' k-th writes land on L1 bank `k % 32`, one core's L1 hammered by the whole grid. Rotate each core's walk to start at a different offset within its range (catalog `bank_stagger`, ⭐⭐ T2, in `ttnn/ttnn/operations/examples/master.md`; the rotation offset is one RT arg or derived from the core's index). Keep contiguous ranges, keep one barrier per block (the ragged-block rotation must stay inside the core's own range). No SUPPORTED change.

**Structural count (the gate on this backend)**: *peak same-bank fan-in per issue step* = max over issue step k and bank b of the number of cores whose k-th write targets b. It is read off the op by a test that takes the program's run args (`start_page`, `num_pages_this_core`, the stagger offset) and the device's bank counts (`ttnn._ttnn.device.get_memory_view(device, BufferType).num_banks`), reproducing the kernel's issue order. Verifier figures for the current op, and for a rotation by core index:

| Shape (bf16) | DRAM, before → after | L1, before → after |
|---|---|---|
| `(1024,1024)` TILE (1024 pages) | 32 → 16 | 32 → 1 |
| `(1,1,2048,256)` TILE (512 pages) | 32 → 16 | 16 → 2 |
| `(1,1,2048,256)` ROW_MAJOR (2048 rows) | 32 → 16 | 32 → 1 |
| `(70,512)` TILE (48 pages, traced call) | 24 → 16 | 2 → 2 |

The gain in time is **unconfirmed** on ttsim. Catalog figures for the lever are 1.04–1.32× on Blackhole, with less gain for interleaved sources and more blocks per core. Say so in the changelog, and measure on `ttsim-perf` or hardware when a run has one.

**Done when**: the fan-in count reaches the "after" column, or lower, on every shape of the table for both placements; a test reads it off the op; the pre-filled every-page tests (all dtype × layout × placement, W = 1 and 2) and the golden suite stay green; and there is no structural regression across the config-spanning guard set (one representative per layout × placement: same bytes written, same writes per page, same barriers per core, 0 bytes read).

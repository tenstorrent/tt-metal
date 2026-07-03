# Operation Requirements: all_gather

## Definition
- **Formula**: for a line of `N` devices where device `j` holds shard `S_j`
  (sharded along `gather_dim`),
  `output[d] = concat(S_0, S_1, …, S_{N-1}, dim=gather_dim)` for **every** device
  `d`. Identity gather — element values unchanged (PCC = 1.0, bit-exact).
- **PyTorch Reference** (single-device oracle for the sharded input):
  ```python
  def all_gather_ref(shards, gather_dim):
      # shards: list of N per-device tensors, each the device's slice along gather_dim
      return torch.cat(shards, dim=gather_dim)  # every device ends up holding this
  ```
- **Import Path**: `from ttnn.operations.all_gather import all_gather`
- **Function Signature**:
  ```python
  def all_gather(
      input_tensor: ttnn.Tensor,
      gather_dim: int,
      *,
      topology: ttnn.Topology = ttnn.Topology.Linear,
      output_tensor: ttnn.Tensor | None = None,
  ) -> ttnn.Tensor
  ```
- **Verification runner (MANDATORY)**: `scripts/run_multidevice_sim_pytest.py --op all_gather`
  (topology `wh_t3k_allmmio_all_gather`, mesh **(1, 8)**, `FABRIC_1D`). `run_safe_pytest.sh`
  is wrong for this op (forces slow dispatch; no multichip/hang awareness). Invoke the
  runner with **this clone's** `python_env/bin/python3` so `sys.executable` (and thus
  the spawned pytest's `ttnn`) resolves to this checkout, not a global one. The golden
  suite must be run in `-k` chunks (per dtype/layout) — a full single-process run
  exceeds the wall-clock backstop because the CCL golden `mesh_device` fixture re-inits
  the 8-device fabric per cell.

## Phases

> **Non-regression rule**: Every refinement must keep all prior-phase cells passing.
> **Drift signal**: XPASS-strict failures mean support was added without updating
> SUPPORTED — fix by updating `SUPPORTED` (and `validate()` if structure changes).
> **Checkbox protocol**: `[x]` complete + all tests pass; `[~]` real work landed but at
> least one named axis value deferred (surfaced as partial); `[ ]` nothing usable.
> **Refinement IDs**: primary `Refinement N`; partial follow-ups append a letter
> (`Refinement 1b`, `1c`, …) immediately after the parent. The runner parses exactly
> `Refinement \d+[a-z]?`.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16, float32]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED topology**: [Linear]
- **SUPPORTED gather_dim**: [-4] (negative convention; -4 ≡ dim 0 for rank-4 shards —
  the proven contiguous-slice case)
- **SUPPORTED alignment** (INPUT_TAGGERS): [tile_aligned, non_tile_aligned]
- **Cores**: two per device (`core_fwd = (0,0)`, `core_bwd = (0,1)`), bidirectional
  store-and-forward ring; three single-owner GlobalSemaphores (barrier + fwd/bwd
  counting).
- **Golden baseline**: **16 / 384 cells passing** (`supported_pass`); 304 xfail,
  64 invalid_skipped; all loud verifier categories 0 (per `verifier_report.json`).
- **Precision**: bit-exact (PCC 1.0, max/mean/RMS err = 0) across 4 shapes × {bf16,f32}.

---

### [x] Refinement 1 — Format axes: bfloat8_b dtype + ROW_MAJOR layout

**Goal**: add `ttnn.bfloat8_b` to `SUPPORTED["dtype"]` and `ttnn.ROW_MAJOR_LAYOUT` to
`SUPPORTED["layout"]`, natively in the reader/writer page handling. all_gather is pure
byte movement (it never tilizes/untilizes), so both are format-flexibility changes at
the same reader/writer + CB-format touch points:
- **ROW_MAJOR**: pure-RM reader/writer — the relay CB page is a row (or the RM page
  size), sized via `align(buffer_page_size(), 16)` which the descriptor already
  computes; at `gather_dim = -4` the shard is still a contiguous page range, so the
  existing contiguous-slice walk applies unchanged. This is native in-kernel RM data
  access — **not** an `ttnn.to_layout`/`tilize` wrapper around the existing kernel.
- **bfloat8_b**: set the relay CB `data_format` to the input dtype (already
  `data_format = input_tensor.dtype`) and confirm whole-tile block-float page movement
  (bf8b tile page = 1088 B, 16-B aligned). `bf8b × ROW_MAJOR` is INVALID (skipped, not
  a cell to chase).

**Moves these golden cells to passing**: `layout=ROW_MAJOR` (128 xfail) at the
already-supported dtypes, and `dtype=bfloat8_b` (64 xfail) at TILE — for the currently
supported `gather_dim=-4`, `topology=Linear`. (RM/bf8b × gather_dim∈{-3,-2,-1} and
× Ring remain xfail until Refinements 2 and 3.)

**Implementation skill**: /memory-layouts, /numeric-formats-metal

**Verifier notes**: land this first — lowest risk (reuses the contiguous-slice walk at
gather_dim=-4) and it establishes the dtype-driven / RM CB-format derivation that
Refinement 2's non-contiguous walk must then honour for every format. Expect one
possible `EXCLUSIONS` entry: if `bfloat8_b × non_tile_aligned` fails (block-float
sub-tile edge), exclude that cell rather than fight it (the `/numeric-formats-metal`
methodology) — do **not** shrink `SUPPORTED["alignment"]`, which bf16/f32 rely on.
`ROW_MAJOR × non_tile_aligned` should work as raw-row byte movement; verify it.

**Done when**: the `layout=ROW_MAJOR` and `dtype=bfloat8_b` cells at
`gather_dim=-4, topology=Linear` pass under the multidevice sim runner, SUPPORTED lists
both new values, and the verifier CLI is clean (loud categories 0).

---

### [ ] Refinement 2 — Non-contiguous concat addressing (gather_dim -3, -2, -1)

**Goal**: add `-3, -2, -1` to `SUPPORTED["gather_dim"]`. At `gather_dim ≠ 0` a slice is
no longer a contiguous output page range — the reader/writer must walk the interleaved
row-stride pattern from the design's stride table (`op_design.md` §"Dataflow Strategy"
concat-by-gather_dim table): slice from origin `j` starts at `tile_id = j·stride(gd)`,
with per-row strides `output_Wt` reading `input_Wt` per row for the H/W dims. This is
the one algorithmically substantial refinement (it changes the slice-walk in both the
own-slice read and the store-and-forward relay re-read, and the fabric write offsets).

**Moves these golden cells to passing**: `gather_dim ∈ {-3, -2, -1}` (240 xfail cells)
across all supported dtypes/layouts at `topology=Linear`.

**Verifier notes**: depends on Refinement 1 — the stride-walk must be correct for
**every** format already in SUPPORTED (bf16/f32/bf8b × TILE/ROW_MAJOR), so Refinement 1
must land first or this refinement will silently regress the format axes at
`gather_dim ≠ 0`. No skill in the current inventory covers CCL concat-stride
addressing; work from the `op_design.md` stride table (rows for `gd = 1/2/3`, tile
counts, `output_Wt`/`input_Wt` row strides) and the reference walk in
`minimal_default_writer.cpp:247-256, 449-457` cited there. Watch page-size vs
tile-count bookkeeping for the non-innermost dims. Keep the single-owner-semaphore /
data-before-inc coordination unchanged — only the addressing changes.

**Done when**: `gather_dim ∈ {-3,-2,-1}` cells pass across the SUPPORTED dtype/layout
set at `topology=Linear`, SUPPORTED lists all four gather_dims, verifier CLI clean, and
the precision baseline stays bit-exact for a non-zero gather_dim.

---

### [ ] Refinement 3 — Ring topology

**Goal**: add `ttnn.Topology.Ring` to `SUPPORTED["topology"]`. The design states the
current bidirectional store-and-forward kernel already runs on a Ring (a Ring wrap lets
a single direction suffice); the host route build (`ccl_dm_route`) already owns the
ring short-way sign, and the fabric-connection gating must allow the wraparound
neighbour (device 0 ↔ device N-1) instead of treating the ends as neighbour-less.

**Moves these golden cells to passing**: `topology=Ring` (160 xfail cells).

**Verifier notes**: **verification is infra-blocked on the current sim matrix.** The
only all_gather topology in `scripts/multidevice_sim_topologies.yaml` is `FABRIC_1D`
(a line); the golden `mesh_device` fixture opens a `FABRIC_1D` (1,8) line regardless of
the `topology` axis value, so a `topology=Ring` cell would run Ring routing against a
line fabric that has **no** ethernet wraparound link between device 0 and N-1 — it
cannot exercise (or prove) ring behaviour and will fail/hang fabric routing, not pass.
Before this refinement can be ticked green, a `FABRIC_RING` (or ring mesh-graph)
topology with `applies_to_ops: [all_gather]` must be added to the matrix and the golden
`mesh_device` fixture pointed at it for the Ring cells. Do this refinement **last** —
after the SUPPORTED rectangle (dtype/layout/gather_dim) is otherwise stable — and do
not silence the Ring cells via EXCLUSIONS or INVALID (Ring is a legitimate TARGET
value; the block is infra, not the op). No skill in the inventory covers CCL topology
changes (the inventory skills are single-device); this is verifier-authored.

**Done when**: a ring sim topology exists in the matrix, `topology=Ring` cells pass
under it, SUPPORTED lists `Ring`, and the verifier CLI is clean. If the ring topology
cannot be provisioned this cycle, leave `[ ]` and record the infra blocker in
`changelog.md` — do not fake-tick it.

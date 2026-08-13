# Operation Requirements: rms_norm

## Definition

- **Formula**:
  `out[..., r, c] = x[..., r, c] * rsqrt( (1/W) * Σ_{c'=0}^{W-1} x[..., r, c']² + epsilon ) * gamma[c]`
  (the sum runs over the **valid** W elements only — never over tile padding)

- **PyTorch Reference**:

```python
def rms_norm_reference(x: torch.Tensor, gamma: torch.Tensor | None = None, epsilon: float = 1e-6):
    xf = x.to(torch.float32)
    out = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + epsilon)
    if gamma is not None:
        out = out * gamma.to(torch.float32).reshape(-1)
    return out.to(x.dtype)
```

- **Import Path**: `from ttnn.operations.rms_norm import rms_norm`

- **Function Signature**:

```python
def rms_norm(
    input_tensor: ttnn.Tensor,
    *,
    gamma: Optional[ttnn.Tensor] = None,
    epsilon: float = 1e-6,
    compute_kernel_config: ttnn.ComputeConfigDescriptor = None,
) -> ttnn.Tensor
```

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16, float32]
- **SUPPORTED fp32_dest_acc_en**: [True]
- **SUPPORTED layout**: [TILE, ROW_MAJOR] — both native, no host-side transform
- **SUPPORTED shape-derived axes**: alignment ∈ {tile_aligned, w_non_aligned, h_non_aligned}; rank ∈ {2, 3, 4}
- **SUPPORTED op-specific axes**: gamma_mode ∈ {gamma, no_gamma}; gamma_dtype ∈ {bfloat16, float32, "none"}; gamma_layout ∈ {TILE, ROW_MAJOR, "none"}
- **SUPPORTED memory_layout**: [INTERLEAVED]
- **Cores**: multi-core, 2D partition — `num_row_groups` rectangles × `num_hidden_slices` cores per rectangle, with a real cross-core combine (gather-to-root + `Mcast2D` broadcast) already built for the dependent (hidden) axis
- **Compute config**: caller-supplied `ttnn.ComputeConfigDescriptor`; `math_fidelity` / `math_approx_mode` ungated; Phase 0 pins `fp32_dest_acc_en=True`
- **Golden baseline**: 737 / 737 supported cells passing; 0 supported_fail, 0 xpass_drift, 0 xfail_wrong_mode (per `verifier_report.json`)

---

### [ ] Refinement 1 — Numerical configurability expansion (unlocks the perf target's config)

**Goal**: grow the precision surface to the whole TARGET rectangle:

- add `False` to `SUPPORTED["fp32_dest_acc_en"]`,
- add `ttnn.bfloat8_b` to `SUPPORTED["dtype"]` **and** to `SUPPORTED["gamma_dtype"]`,
- add `{"dtype": ttnn.float32, "fp32_dest_acc_en": False}` to `EXCLUSIONS` — it is a permanent refusal (silently accumulating fp32 activations at reduced width is a lie to the caller) and it only becomes expressible as an EXCLUSION once `False` is inside SUPPORTED,
- keep every stat CB (`cb_sq_partials`, `cb_gathered_partials`, `cb_rms_bcast`, `cb_rms_recip`) at `float32` regardless of the input dtype and regardless of `fp32_dest_acc_en` — that is a measured accuracy requirement, not a default,
- any cell that fails out of the box goes to `EXCLUSIONS`, not to its own refinement.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**: this is the **largest single unlock in the queue and it gates the perf phases**, so it goes first. `fp32_dest_acc_en=False` alone accounts for 3609 of the 6174 xfail cells, and *every* hand-authored loose case — the whole `perf`, `resilience` and `pad_poison` corpus — pins `fp32_dest_acc_en=False`, so none of them runs today. In particular the mandatory perf target (Refinement 3) is specified at bf16 / **HiFi2** / **`fp32_dest_acc_en=False`** / TILE / INTERLEAVED; a perf pass may not stand a `fp32_dest_acc_en=True` proxy in for it, so this refinement must land that exact config as a supported cell. Two op-specific hazards to carry into the work: (1) `DEST_AUTO_LIMIT` doubles from 4 to 8 tiles when `fp32_dest_acc_en=False` — the helpers clamp themselves, but do not hardcode either number; (2) `cb_input_tiles` is rewritten **in place** twice, so a `bfloat8_b` input means the intermediate `x·r` is re-quantized to block-float before the gamma multiply — measure that cell against the golden bf8b tolerance (0.99 PCC / 0.10 rel-RMS) and, if it misses, exclude the cell rather than un-fusing the in-place path. `bfloat8_b` on a non-tile-aligned shape and `bfloat8_b + ROW_MAJOR` are already `INVALID` in `feature_spec.py`, so they will not appear.

**Done when**: `verify_supported` shows `supported_fail = 0`, `xpass_drift = 0`, `xfail_wrong_mode = 0` with the three axis values above inside SUPPORTED, and the `perf` / `resilience` / `pad_poison` loose-case groups run as supported cells instead of xfail (the `pad_poison` group is the padding-in-the-denominator guard and must pass).

---

### [ ] Refinement 2 — Sharded placement: HEIGHT / WIDTH / BLOCK

**Goal**: add `HEIGHT_SHARDED`, `WIDTH_SHARDED` and `BLOCK_SHARDED` to `SUPPORTED["memory_layout"]`, consumed **natively** — the caller's shard is already resident in each core's L1, so it must be read through a CB backed on the sharded buffer (`ttnn.cb_descriptor_from_sharded_tensor`, zero-copy), never re-read through a `TensorAccessor`. Also add the `memory_config: Optional[ttnn.MemoryConfig] = None` kwarg to the entry point (the golden runner passes the input's shard spec for every sharded cell and expects a matching sharded output) and allocate the output accordingly.

**Verifier notes**: no skill covers placement yet, so the mechanism is named here. Ordered second because it is the hardest generality work left (the difficulty ranking puts block-sharding at the top) and because every later phase is perf that would otherwise be re-tuned on top of it. What makes this a *placement* refinement rather than a new scheme is that **all three flavours are already the Phase 0 logical scheme**: `HEIGHT_SHARDED` cuts the independent row axis (the reduce stays core-local — the `num_hidden_slices == 1` regime), `WIDTH_SHARDED` cuts the dependent hidden axis (exactly the gather-to-root + `Mcast2D` combine that is already built), and `BLOCK_SHARDED` cuts both (the Phase 0 2D partition with the geometry pinned by the caller). So the work is: read `num_row_groups` / `num_hidden_slices` / `slice_hidden_tiles` / `core_row_tiles` **off the shard spec** instead of computing them in `_plan`, and swap `load_block` / `store_block` for CB placement on the sharded buffers. Three constraints to respect: (a) `Mcast2D` takes the *bounding box* of the core set as the rect, so a shard grid whose cores are not a rectangle must be refused via `EXCLUSIONS` rather than silently multicast into non-members; (b) the shard grid replaces the rect search, so `HIDDEN_TILES_PER_CORE_FLOOR` no longer applies on this path; (c) the ROW_MAJOR + sharded + TILE-gamma corners are already `INVALID` in `feature_spec.py` and will not appear. The five sharded perf loose cases (`_perf_case(..., _ML.WIDTH_SHARDED, ...)` and the block-sharded prefill) become measurable only after this lands — they are Refinement 5's targets.

**Done when**: the three sharded values are in SUPPORTED, the sharded `resilience` (44 shapes × 3 placements) and `pad_poison` (6 shapes × 3 placements) loose cases pass or are covered by a named `EXCLUSIONS` entry, no sharded cell reaches the kernel through a `TensorAccessor` read of its **own** local shard, and `verify_supported` stays clean on all three loud categories.

---

### [ ] Refinement 3 — Speed up the perf-flagged decode profile

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES` carries a `perf` group of model-derived profiles; the decisive one is
`_perf_case(32, 7168, 104259, minimum_expected_speedup=7.0)` — input `(1, 1, 32, 7168)`, bf16, **HiFi2**,
**`fp32_dest_acc_en=False`**, TILE, bf16 TILE gamma, **INTERLEAVED**, whose comment states plainly that a
marginal win is not sufficient and that the shape "is expected to expose a decisively better architecture":
at 1350 MHz its ≤104259 ns reference and ≥7× requirement imply a **≤14894 ns** goal (clock-scale before
comparing). Optimize **that exact config** — not a `fp32_dest_acc_en=True` proxy — together with its three
sibling interleaved decode cases (`W = 1024 / 2304 / 5120`, `achievable_ns` 9149 / 17003 / 75825). Pick the
levers from `ttnn/ttnn/operations/examples/master.md`; the entries about grid occupancy, keeping bytes in
flight, and per-core transfer size are the ones whose situation matches this regime. Soft precision gate
`pcc_threshold = 0.9995` (from the same `extras`) must still hold. No SUPPORTED change.

**Verifier notes**: depends on Refinement 1 (the config is unsupported until then). Start from what has
already been measured on this part (Blackhole p150b, 11×10 grid, bf16 + gamma, TILE, recorded in
`tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_perf.py`): `decode_w7168` = 13026 ns of which
**~3514 ns is a fixed dispatch + kernel-boot floor**, payload ≈ 97 GB/s, 56 of 110 cores engaged, `S = 4`
hidden tiles per core. Two knob sweeps are already done and came back **flat within 1–2 %** — the
`HIDDEN_TILES_PER_CORE_FLOOR ∈ {2,4,8,16}` sweep and the row-group rectangle search (11 vs 56 cores) — so
do not re-spend the phase there; the binding costs are the fixed floor and the short (S=4 tiles ≈ 8 KB)
per-core DRAM transfer. Named, unmeasured levers in scope: raise `DM_CHUNK_TILES` / restructure the reader
so a core's whole slice is in flight before the first barrier; widen `l1_working_budget` (today a
conservative 928 KB because `device.l1_size_per_core()` is unbound — see `l1_ledger.md`'s symbol table) so
fewer, fatter blocks are possible; and shorten the combine's critical path (one gather + one mcast +
handshake per core). The `rsqrt` over-compute lamp in `op_design.md` is a decode-regime item but scales
with `block_rows = 1` here, so it is small.

**Done when**: measured device-ns improves on `(1,1,32,7168)` at its exact `extras` config and moves it
toward the ≤14894 ns clock-scaled goal, the other three interleaved decode cases do not regress, the
0.9995 soft PCC gate holds, the golden suite is green, and there is no regression across a
config-spanning guard set (one representative per distinct kernel path × layout × placement:
TILE/ROW_MAJOR × `s == 1`/`s > 1` × gamma/no-gamma, interleaved).

---

### [ ] Refinement 4 — Speed up the perf-flagged prefill profile

**Type**: perf

**Goal**: the four interleaved **prefill** cases in the same `perf` group —
`(1,1,8192,W)` for `W ∈ {1024, 2304, 5120, 7168}` with `achievable_ns` 96744 / 211345 / 738307 / 1032281,
same fixed config (bf16, HiFi2, `fp32_dest_acc_en=False`, TILE, bf16 TILE gamma, INTERLEAVED). This is the
bandwidth-bound regime and the measured gap is the largest in the queue: the existing harness row
`prefill_2048x1024` = 45638 ns for 8.39 MB is ≈184 GB/s, which extrapolates to ≈182 µs for
`(1,1,8192,1024)` against a 96744 ns reference (≈347 GB/s) — roughly **1.9× off**. Optimize with the
relevant `ttnn/ttnn/operations/examples/master.md` patterns (block-size / buffer-depth co-tune, keeping
bytes in flight, NoC placement). No SUPPORTED change.

**Verifier notes**: this is the phase in which the **block-size × buffer-depth co-tune** actually has room,
because `core_row_tiles` is large here and Phase 0 takes the coarsest block that fits. Two concrete,
already-identified levers: (1) `l1_working_budget` is a conservative 928 KB (1 MB assumed − 96 KB reserve)
while the part reports 1.46 MB unreserved — a larger, *measured* budget directly coarsens `block_rows`;
(2) `IN_CB_DEPTH` is load-bearing at 1 (the in-place rewrite of x needs `get_write_ptr == get_read_ptr`),
so the overlap lamp must be measured as "**smaller `block_rows` + a second input buffer**", never as "same
block, deeper `cb_input_tiles`" — the descriptor asserts this. Also in scope and quantified: the
**GammaBroadcast** scheme lamp (`op_design.md`) removes the residual `(g−1)·Wt·T` DRAM term, which is
+11 % of DRAM bytes at the `s = 1` corner and ~1.4 % at `g = 8` — file it as a lever inside this phase
only if the budget shows gamma re-reads are actually binding; it is a new mcast family (one injector +
broadcast) and must be validated by building it and measuring device-ns, not by an ablation.

**Done when**: measured device-ns improves on at least the two most-impacted prefill shapes toward their
clock-scaled `achievable_ns`, the decode results from Refinement 3 do not regress, the golden suite is
green, and the config-spanning guard set shows no regression.

---

### [ ] Refinement 5 — Speed up the sharded perf geometries

**Type**: perf

**Goal**: the five sharded `perf` loose cases, each pinned to its measured-fastest geometry via
`extras.shard_shape` + `extras.core_grid`: `(1,1,32,1024)` WIDTH `[32,128]`/(8,1) @ 4110 ns,
`(1,1,32,2304)` WIDTH `[32,256]`/(9,1) @ 4617 ns, `(1,1,32,5120)` WIDTH `[32,160]`/(8,4) @ 5267 ns,
`(1,1,32,7168)` WIDTH `[32,256]`/(7,4) @ 5481 ns, and `(1,1,8192,1024)` BLOCK `[1024,128]`/(8,8) @ 25640 ns.
These references are 2–20× tighter than their interleaved siblings because the input is already resident,
so the whole budget is the combine plus the write-out. Optimize the sharded path against them using the
relevant `master.md` patterns. No SUPPORTED change.

**Verifier notes**: depends on Refinement 2 — these cells cannot even run until sharded placement is
supported, and they are the reason Refinement 2 must be *built* performantly (zero-copy CB placement on the
resident shard) rather than correct-only. The 4110–5481 ns references sit *below* the 3514 ns fixed
dispatch + boot floor measured for the interleaved path plus any per-core transfer, so expect the fixed
cost — not the payload — to be the binding term; treat "shrink the boot/dispatch critical path and the
combine handshake" as the primary lever family here.

**Done when**: measured device-ns improves on the sharded perf shapes at their pinned geometries toward the
clock-scaled `achievable_ns`, no regression on the interleaved decode/prefill results from Refinements 3–4,
the golden suite is green, and the config-spanning guard set (now including one sharded representative per
scheme) shows no regression.

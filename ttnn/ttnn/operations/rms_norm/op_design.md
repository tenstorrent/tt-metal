# Operation Design: rms_norm

Specification for `ttnn-implementer`. Every statement is a decision. The implementer writes the
kernels, the `SUPPORTED` block, and the kernel-argument tables; everything structural is fixed here.

Inputs consumed as authoritative (not edited by this document):
`eval/golden_tests/rms_norm/feature_spec.py` (TARGET / INPUTS / INVALID / LOOSE_CASES),
`eval/golden_tests/rms_norm/{axes,helpers,test_golden}.py`,
`references/precision_convention.md`.

---

## Overview

| Field | Value |
|-------|-------|
| Classification | compute (movement-dominated in every regime; see Performance Methodology) |
| Goal | Row-wise RMS normalization along the last dimension, with an optional per-column scale `gamma`, native for both ROW_MAJOR and TILE layouts and for non-tile-aligned H and W. |
| Math | `output[..., r, c] = input[..., r, c] * rsqrt( (1/W) * Σ_{c'=0}^{W-1} input[..., r, c']² + epsilon ) * gamma[c]` |
| Mode | Derivative (composed entirely from `kernel_lib` helpers) |
| References | `ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp`, `ttnn/cpp/ttnn/kernel_lib/eltwise/core/chain.hpp`, `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_{compute,dataflow}.hpp`, `ttnn/cpp/ttnn/kernel_lib/{tilize,untilize}_helpers.hpp`, `ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp`, `ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp`, `ttnn/cpp/ttnn/kernel_lib/host/mcast_host.hpp`, `ttnn/ttnn/operations/toy_variance/`, `ttnn/ttnn/operations/toy_reduce_partial/`, `ttnn/ttnn/operations/toy_tilize_untilize/`, `ttnn/ttnn/operations/examples/master.md` |

`W` in the formula is always the **true, unpadded** last-dimension extent. The denominator must
never include tile padding — this is load-bearing and is enforced structurally (see
[Regime selection](#regime-selection-pinned) and Key Risks R1).

---

## Blocking Model

The op's data has three axes. `R` is the flattened leading extent (everything except the last dim),
`W` is the last dim. In tiled form:

```
Rt = prod(shape[:-2]) * ceil(shape[-2] / 32)      # tile-rows, per-image tile padding -> ceil, NOT floor
Wt = ceil(shape[-1] / 32)                          # tile-columns
```

| Axis | Character | Block-factor knob | Phase 0 value | Core-assignment | Later unlock |
|------|-----------|-------------------|---------------|-----------------|--------------|
| **Rt** (tile-rows: batch × ceil(H/32)) | **independent** — each logical row's RMS is a function of that row alone; no cross-row term exists in the math | `BLOCK_HT` (tile-rows per compute block) | `budget_solver()` result, capped at `DEST_AUTO_LIMIT` and at `ceil(Rt/num_cores)`; = coarsest that fits L1, never 1 by default | **Spread across the whole grid in Phase 0.** `ttnn.split_work_to_cores(CORE_GRID, ceil(Rt/BLOCK_HT), row_wise=True)`; each core owns a contiguous run of row-blocks | **knob-turn** (raise `BLOCK_HT`, change `CORE_GRID` / `ACTIVE_CORE_CAP`) |
| **Wt** (tile-columns: the reduced axis) | **dependent** — one output element's denominator spans every `c` in `[0, W)`; a result spans the axis | `WT_REDUCE_BLOCK` (pass-A chunk), `WT_SCALE_BLOCK` (pass-B chunk) | `Wt_core` (the whole per-core width — one block) in Regime A; `budget_solver()` chunk in Regime B | **Single core owns the whole reduced extent in Phase 0** (sequential accumulate inside the core — cheap). One core = one full row. | **scheme-change** (cross-core partial-sum combine — Lamp L1) |
| **gamma[W]** (the scale operand) | **reuse-shared** — `gamma` is indexed only by `c`, so it is identical for every block along `Rt`; splitting `Rt` across cores makes every core re-read the same bytes | `WT_SCALE_BLOCK` (shares the pass-B chunk knob) | `Wt_core` (whole per-core width, resident, read once per core) | Replicated: every core in the `Rt` split reads the full `gamma` from DRAM once | **scheme-change** (read once on an injector + mcast — Lamp L2) |

**Buffer-depth knobs** (one per streaming CB, all parameters, never literals):

| CB | Knob | Phase 0 value |
|----|------|---------------|
| `cb_input_tiles` | `IN_BUF_DEPTH` | `2` if `budget_solver()` affords it, else `1` |
| `cb_output_tiles` | `OUT_BUF_DEPTH` | `2` |
| `cb_rm_in` / `cb_rm_out` (ROW_MAJOR only) | `RM_BUF_DEPTH` | `2` |
| `cb_gamma_tiles` | — (resident, depth 1, never popped) | `1` |
| `cb_squared`, `cb_normed` (compute→compute) | — (sequential-helper intermediates: must hold a **full** block, depth 1) | `1` |

**Single source of truth (DRY).** Every quantity above is emitted by exactly one host function,
`blocking_plan(input_tensor, gamma, device)` (see [Work Distribution](#work-distribution)), which
returns a frozen dataclass. Nothing downstream — CB page counts, kernel CT/RT args, loop trip counts,
grid sizing — recomputes a block factor; each reads the field. Turning a knob is a one-line change
in `blocking_plan`.

### Bandwidth ranking of the candidate splits (qualitative, structural)

Ranked by bytes moved and by fan-out, over the splits actually available:

| Rank | Split | Bytes moved | Fan-out / combine | Verdict |
|---|---|---|---|---|
| 1 | **Rt across cores** (independent) | `1× read + 1× write` of the tensor, plus `num_cores × gamma_bytes` | none | **Primary split, Phase 0.** Every core owns disjoint output rows; each row's reduction stays inside one core, so the reduce is a sequential in-DEST/in-CB accumulate with zero NoC cost. The only extra traffic is the replicated `gamma` (Lamp L2). |
| 2 | **Wt across cores** (dependent) | `1× read + 1× write`, and `gamma` is *not* replicated (each core reads only its own slice) | fan-in of `num_cores` one-tile partials + one mcast per row-block | **The available parallelism when `Rt` under-fills the grid** — and it moves *fewer* bytes than rank 1 because it kills the gamma replication. Costs a cross-core combine. Lamp L1. |
| 3 | **Rt across cores with a streaming (2-pass) read of Wt** | `2× read + 1× write` — **1.5× the DRAM traffic of rank 1** | none | Correctness fallback only (Regime B), used when the per-core working set will not fit L1. |
| 4 | Sub-tile split of `Rt` (32 logical rows inside one tile-row) | same as rank 1 | none | Rejected: sub-tile work units do not map onto any helper's block shape and would require hand-rolled per-row compute. |

`Rt` wins rank 1 on both bytes and simplicity, so it is the Phase 0 primary split. But `Rt` is
**structurally small for a whole class of this op's shapes**: every decode shape in
`feature_spec.LOOSE_CASES` is `(1, 1, 32, W)` → `Rt = 1` → the rank-1 split reaches **one core**.
That is why rank 2 is a first-class lamp rather than an afterthought, and why the design below fixes
its dataflow contract in full.

### Operand-reuse check (run per (operand, chosen-split) pair)

| Operand | Varies along `Rt` (the chosen split)? | Consequence |
|---|---|---|
| `input_tensor` | yes | correctly partitioned; no re-read |
| `gamma` | **no** | every core in the split pulls the identical `Wt` tiles from DRAM → **reuse-shared by construction of the split** → Lamp L2 (broadcast). Quantified below: `gamma` replication is `1/(2·rows_per_core)` of the op's DRAM traffic — **≈21 % at `Rt/num_cores ≈ 2.3`**, which is exactly the prefill regime on a full grid. |
| `epsilon`, `1/W` | n/a (scalars) | carried as fp32 bit-pattern runtime args into SFPU `AddUnary`/`MulUnary`; **no CB, no DRAM traffic, no bf16 quantization** |

Phase 0 mitigates L2 *without* mcast via the `ACTIVE_CORE_CAP` knob: capping active cores raises
`rows_per_core`, which shrinks the replication fraction quadratically-cheaply, and simultaneously
respects the DRAM-bandwidth knee (lever A0/A3). Measuring that cap is a Phase 0 perf task.

### Lamp — the scheme-changes Phase 0 leaves room for

| ID | Scheme-change | Why Phase 0 does not foreclose it | Phase |
|----|---------------|-----------------------------------|-------|
| **L1** | **Cross-core combine of the dependent `Wt` axis** (`WIDTH_PARALLEL` regime): each core reduces a `Wt`-slice, partials are gathered to a sender, summed, and multicast back. Unlocks (a) the `(1,1,32,{5120,7168})` decode perf gates, (b) `WIDTH_SHARDED` and (c) `BLOCK_SHARDED` `memory_layout` TARGET values, and (d) the `W ≥ 16384` single-tile-row LOOSE_CASES without a 2-pass read. | The per-core width is already the parameter `Wt_core` (Phase 0 value `Wt`), and the regime is already a compile-time arg selected by a pinned host predicate. Enabling L1 sets `Wt_core = Wt / num_w_cores`, adds two CBs and two semaphores, and inserts one combine step between compute phase 2 and phase 3. **No loop nest changes, no helper call changes.** The full contract is specified in [Dataflow Strategy → L1](#unlocked-scheme-l1-cross-core-combine-over-the-dependent-axis). |
| **L2** | **`gamma` broadcast**: read `gamma` once on an injector core and `mcast_pipe` it to the `Rt`-split cores instead of `num_cores` DRAM re-reads. | `cb_gamma_tiles` is already a resident, never-popped CB filled exactly once per core at kernel boot by the reader. Switching its fill from a `TensorAccessor` read to a `ReceiverPipe::receive()` is a reader-local change; the compute kernel is untouched. |
| **L3** | **`HEIGHT_SHARDED` input** (`memory_layout` TARGET value): the caller pins the `Rt` core-assignment and pre-places rows in L1. Because `HEIGHT_SHARDED` cuts the **independent** axis, the reduction stays entirely local — this is a **knob-turn**, not a scheme-change. | Phase 0's core-assignment along `Rt` is already exactly this geometry; L3 replaces `cb_input_tiles`'s descriptor with `ttnn.cb_descriptor_from_sharded_tensor(CB_INPUT_TILES, input_tensor)` (zero-copy, **no NoC read**) and sets `CORE_GRID` to the shard grid. |
| **L4** | **Fused gamma-scale** (`cb_normed` elimination): pre-broadcast `gamma` to full 32-row tiles once at boot, then fuse the two pass-B multiplies into one `eltwise_chain` using `DestReuseBinary` (`chain.hpp:518-520`). Removes an entire `BLOCK_HT × Wt_core` L1 write+read round-trip per row-block *and* widens Regime A's L1 reach to `Wt = 224` (W = 7168). | `cb_normed` exists only as the intermediate between two `mul` calls; the design already names the fused form as the ranked-#2 compute candidate, so removing it is a compute-kernel-local edit. |
| **L5** | **Tree/reduce-scatter combine** replacing L1's flat gather (catalog `tensix_all_reduce`: 4.64–6.48× over a ring push). | L1's gather leg is already a separate, named step with its own semaphore family; swapping its topology touches only the reader's combine block. |

---

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | rank ≥ 2 | — | — |
| `gamma` | `Optional[ttnn.Tensor]` | no (keyword-only) | shape `(…, W)` with `gamma.shape[-1] == input.shape[-1]` | `None` | — |
| `epsilon` | `float` | no (keyword-only) | > 0 | `1e-6` | RT (fp32 bit pattern, `uint32`) |
| `compute_kernel_config` | `ttnn.ComputeConfigDescriptor` | no (keyword-only) | see Precision | `None` → `default_compute_kernel_config()` | passed as `config=` on the compute `KernelDescriptor` |
| `HAS_GAMMA` | `bool` | derived | — | `gamma is not None` | CT |
| `IS_ROW_MAJOR` | `bool` | derived | — | `input.layout == ROW_MAJOR_LAYOUT` | CT |
| `GAMMA_IS_ROW_MAJOR` | `bool` | derived | — | `gamma.layout == ROW_MAJOR_LAYOUT` | CT |
| `REGIME` | `{A, B}` | derived | pinned predicate below | — | CT |
| `Wt`, `Rt`, `W_true`, `W_PARTIAL` | `uint32` | derived | `W_PARTIAL = W_true % 32` | — | CT |
| `BLOCK_HT`, `WT_REDUCE_BLOCK`, `WT_SCALE_BLOCK`, `DEST_BLOCK`, `IN_BUF_DEPTH`, `OUT_BUF_DEPTH`, `RM_BUF_DEPTH` | `uint32` | derived | from `blocking_plan()` | see Blocking Model | CT |
| `CORE_GRID`, `ACTIVE_CORE_CAP` | `CoreCoord` / `Optional[int]` | knob | — | `device.compute_with_storage_grid_size()` / `None` | host-only (shapes the grid) |
| `INV_W_BITS`, `EPS_BITS` | `uint32` | derived | `struct.unpack("I", struct.pack("f", x))[0]` of `1.0/W_true` and `epsilon` | — | RT (compute) |

### Precision

Per `references/precision_convention.md`. Exactly one exported factory is the source of truth; the
golden axis tagger (`eval/golden_tests/rms_norm/axes.py:38-41`) reads the same function.

```python
def default_compute_kernel_config():
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )
```

| Rule | Decision |
|---|---|
| Config resolution | `cfg = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()` — resolved in **one** place in the entry point. |
| Gated axes | `dtype` and `fp32_dest_acc_en` only. |
| Phase 0 corner | `fp32_dest_acc_en=True` for **both** `bfloat16` and `float32` input. `bfloat16 + fp32_dest_acc_en=False` is a later refinement (`TARGET − SUPPORTED`). |
| Natively rejected | `{dtype: float32, fp32_dest_acc_en: False}` → op-side `EXCLUSIONS`, **not** `feature_spec.INVALID`. |
| Not gated | `math_fidelity`, `math_approx_mode` — any caller value is accepted and passed through untouched. |
| Pass-through | after `validate()`, the caller's descriptor is handed to the compute kernel verbatim: `config=cfg`. Never rebuilt, never forced. |
| DEST budget | `fp32_dest_acc_en=True` halves DEST. `DEST_AUTO_LIMIT` (`dest_helpers.hpp:89-103`) = **4** tiles (half-sync, fp32 accum). Every `IterationShape::block_size(...)` and `BLOCK_HT` is capped at `DEST_AUTO_LIMIT` — **read the constant, never write `4`**. |

### Validation (entry point, before any device work)

| Check | Raise | Message contract |
|---|---|---|
| `len(input_tensor.shape) < 2` | `ValueError` | must contain `rank` (case-insensitive) |
| `gamma is not None and gamma.shape[-1] != input_tensor.shape[-1]` | `ValueError` | must contain `gamma` (case-insensitive) |
| axis value not in `SUPPORTED` | `UnsupportedAxisValue` (from `ttnn.operations._op_contract`) | `f"rms_norm: {axis}={value!r} not in SUPPORTED {allowed}"` |
| axes dict matches an `EXCLUSIONS` cell | `ExcludedCell` | names the excluded cell dict |

The two message contracts are load-bearing: the acceptance test asserts them through the repo's
`expect_error` fixture (`conftest.py:910-929`), which matches the message as a regex so CI log
triaging can classify the error as expected. `pytest.raises` is banned in `tests/` by a pre-commit
hook (`.pre-commit-config.yaml:51-56`).

`validate()` never inspects `feature_spec.INVALID` (the harness skips those cells before the op is
called). Both `gamma_dtype` and `gamma_layout` **must** accept the string sentinel `"none"` — it means
"no weight tensor" and is always legal; omitting it makes the canonical `no_gamma` cell fall outside
`SUPPORTED` and xpass-fail.

### Registry surface (axis names exactly as `feature_spec.TARGET` uses them)

`INPUT_TAGGERS`, exported from `ttnn/ttnn/operations/rms_norm/rms_norm.py` (declaration order matters):

| Tagger | Returns |
|---|---|
| `tag_alignment(inputs, axes)` | `"tile_aligned"` if `H % 32 == 0 and W % 32 == 0`; else `"w_non_aligned"` if `W % 32 != 0`; else `"h_non_aligned"` |
| `tag_rank(inputs, axes)` | `len(inputs[0])` |

`gamma_dtype` / `gamma_layout` / `gamma_mode` / `fp32_dest_acc_en` / `memory_layout` are tagged by the
golden suite's `axes.py`, not by `INPUT_TAGGERS`; `validate()` must accept the same values.

Phase 0 `SUPPORTED` target (the implementer authors the block; this is the coverage the design
delivers): `dtype ∈ {float32, bfloat16}`; `fp32_dest_acc_en ∈ {True}`; `layout ∈ {TILE, ROW_MAJOR}`;
`alignment ∈` all three; `rank ∈ {2,3,4}`; `gamma_mode ∈` both; `gamma_dtype ∈ {float32, bfloat16,
"none"}`; `gamma_layout ∈ {TILE, ROW_MAJOR, "none"}`; `memory_layout ∈ {INTERLEAVED}`.
`EXCLUSIONS = [{"dtype": ttnn.float32, "fp32_dest_acc_en": False}]`.
Refinement candidates (`TARGET − SUPPORTED`): `bfloat8_b`, `fp32_dest_acc_en=False` for bf16, and the
three `*_SHARDED` values (Lamps L1/L3).

### Structural impossibilities

`feature_spec.INVALID` is complete for this op; no additional structural impossibility was found. One
borderline cell is deliberately **not** proposed as INVALID: `{layout: ROW_MAJOR, dtype: float32,
memory_layout: WIDTH_SHARDED}` is legal but degenerate for very small `W` — `eval.sharding.auto_shard_config`
already produces a legal (padded) spec, so it belongs on the SUPPORTED ladder, not in INVALID.

---

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | rank ≥ 2, `(…, H, W)`. `H`, `W` need not be multiples of 32. |
| Dtype | `bfloat16`, `float32` (Phase 0); `bfloat8_b` is a refinement |
| Layout | `TILE_LAYOUT` or `ROW_MAJOR_LAYOUT` — **both native**. No `to_layout` / `tilize` / `untilize` / `pad` / `slice` on the host path. |
| Memory | interleaved DRAM (Phase 0). `HEIGHT_SHARDED` → Lamp L3; `WIDTH_SHARDED` / `BLOCK_SHARDED` → Lamp L1. |

### Gamma

| Property | Requirement |
|----------|-------------|
| Shape | `(…, W)` with `gamma.shape[-1] == input.shape[-1]`; logically `(1,1,1,W)` |
| Dtype | independent of the input dtype (mixed bf16-activation / fp32-weight is a first-class TARGET cell) |
| Layout | `TILE_LAYOUT` or `ROW_MAJOR_LAYOUT` — both native. (The requirements text says "always ROW_MAJOR"; `feature_spec.TARGET["gamma_layout"]` and every perf LOOSE_CASE say otherwise. **The feature spec wins**: both are supported.) |
| Memory | interleaved DRAM |
| Valid region after ingest | tile row 0 only (`Row0`) |

### Output

| Property | Value |
|----------|-------|
| Shape | identical to input |
| Dtype | `input_tensor.dtype` |
| Layout | `input_tensor.layout` (RM in → RM out; TILE in → TILE out) |
| Memory | `input_tensor.memory_config()` (sharded in → sharded out) |

Allocated with `ttnn.allocate_tensor_on_device` (**positional args only**), and passed **last** in
`ttnn.generic_op([input_tensor, gamma_or_nothing, output_tensor], program_descriptor)`.

---

## Regime selection (pinned)

Phase 0 selects between two compute regimes. The predicate is a single host function; its result is a
compile-time arg. **Regime-pinned tests are mandatory** (a regime that only triggers on some
grids/L1 budgets can pass on one device and fail on another).

```python
L1_RESERVED_BYTES = 256 * 1024          # single source of truth for the non-CB L1 reservation

def select_regime(plan) -> str:
    """Pinned Phase-0 regime selector. Returns "A" (resident, single-read) or "B" (streaming, 2-pass)."""
    # (1) Can the reduce see the padded columns without a mask?
    #     RM input: the reader zero-fills every stick's pad tail, so pad contributes exactly 0.
    #     TILE input: the pad lives in DRAM and may be poisoned -> a mask is mandatory.
    maskless_w = plan.is_row_major or (plan.w_true % 32 == 0)
    # (2) Does the minimal resident working set fit the CB budget?
    fits = resident_working_set_bytes(plan, block_ht=1, in_buf_depth=1) <= plan.l1_cb_budget
    return "A" if (maskless_w and fits) else "B"
```

| Regime | Predicate | Reduce mechanism | Padding handling | DRAM reads of `input` | `Wt` chunking |
|---|---|---|---|---|---|
| **A — RESIDENT-FUSED** | `maskless_w and fits` | `sum_of_squares` (fused `x*x` + per-row DEST accumulate, **no intermediate CB**) | none needed: pad is provably 0 (RM zero-fill) or absent (`W%32==0`) | **1** | none — `WT_REDUCE_BLOCK = WT_SCALE_BLOCK = Wt_core` |
| **B — STREAMING-MASKED** | otherwise | per-chunk `square` → `reduce<SUM, REDUCE_ROW>` with `Accumulate` and `ReducePartialScaler::last_tile_at(1)` | the reduce scaler's partial tile zeroes the padded columns of the **last** `W`-tile | **2** (pass A + pass B) | chunked at `budget_solver()`'s `WT_*_BLOCK` |

Both regimes divide by `W_true` (via the fp32 `MulUnary` scalar), so the RMS denominator reflects
only valid elements in both. **`H` padding needs no masking in either regime**: a pad row's sum is
computed independently of the valid rows (per-row reduction), and the writer never emits pad rows —
so a poisoned pad row can only corrupt an output row that is discarded. The one hazard is
`inf`/`NaN` in pad *columns* (`inf * 0 = NaN` would leak through a masked scaler), which is why
Regime B requires the pad to be **finite** — guaranteed by `ttnn.fill_implicit_tile_padding` on the
harness side and by the RM reader's zero-fill on the RM path.

### Mandatory regime-pinned tests

| Test | Shape / config | Lands in |
|---|---|---|
| aligned TILE, fits | `(1, 1, 64, 128)`, TILE | A |
| RM non-aligned W (zero-filled pad ⇒ still maskless) | `(1, 1, 32, 50)`, ROW_MAJOR | A |
| TILE non-aligned W (mask mandatory) | `(1, 1, 32, 72)`, TILE, poisoned pad | B |
| aligned TILE, working set too large | `(1, 1, 32, 32768)`, TILE | B |

---

## Work Distribution

This is the Blocking Model's core-assignment made concrete. **Phase 0 is multi-core.**

| Field | Value |
|-------|-------|
| Work unit | one **row-block** = `BLOCK_HT` tile-rows × `Wt_core` tile-columns |
| Total units | `num_row_blocks = ceil(Rt / BLOCK_HT)` |
| Grid | `CORE_GRID = device.compute_with_storage_grid_size()`, optionally truncated by `ACTIVE_CORE_CAP` (knob; Phase 0 `None` = full grid, per lever A0 for a transaction-rate-bound RM/tilize path) |
| Split | `ttnn.split_work_to_cores(CORE_GRID, num_row_blocks, row_wise=True)` → `(num_cores, all_cores, core_group_1, core_group_2, per_core_g1, per_core_g2)`. `row_wise=True` spreads across the DRAM-facing axis (lever A1). |
| Per-core work | `(start_row_block, num_row_blocks_here)` as **runtime** args on reader and writer; `num_row_blocks_here` only on compute |
| Remainder | handled by the helper's two core groups; iterate `((core_group_1, per_core_g1), (core_group_2, per_core_g2))`, skip a group when `per_core == 0`, advance `start_row_block` by `per_core` per core (lever A4) |

### Alignment-aware tile geometry (`ceil`, per-image — never `floor` / `//`)

```python
def tile_geometry(shape):
    W_true = shape[-1]
    H      = shape[-2]
    batch  = 1
    for d in shape[:-2]:
        batch *= d
    Rt = batch * ttnn.div_up(H, 32)     # per-image tile padding: ceil per image, NOT floor(batch*H/32)
    Wt = ttnn.div_up(W_true, 32)
    return Rt, Wt, W_true, W_true % 32
```

In TILE layout each `(…, H, W)` image is tile-padded independently, so `Rt = batch * ceil(H/32)`.
Writing `(batch*H)//32` is wrong for every `h_non_aligned` multi-batch shape in `INPUTS`
(e.g. `(4, 8, 47, 256)` → correct `Rt = 32·2 = 64`, wrong `Rt = 47·32/32 = 47`).

### The L1 budget solver — the single knob source

```python
def blocking_plan(input_tensor, gamma, device):
    """The ONLY place block factors and buffer depths are decided. Everything downstream reads
    these fields; nothing recomputes them."""
    Rt, Wt, W_true, W_partial = tile_geometry(list(input_tensor.shape))
    Wt_core = Wt                                  # Phase 0: no W split. Lamp L1 sets Wt/num_w_cores.
    l1_cb_budget = device.l1_size_per_core() - L1_RESERVED_BYTES

    # Allocation priority (movement-dominated op -> overlap beats amortization):
    #   1. resident gamma (Wt_core tiles)  -- required for Regime A's single gamma read
    #   2. cb_input_tiles at IN_BUF_DEPTH = 2, BLOCK_HT = 1   (double_buffer lever, 2.78x measured)
    #   3. grow BLOCK_HT up to min(DEST_AUTO_LIMIT, ceil(Rt/num_cores))
    #   4. grow IN_BUF_DEPTH further
    # Step 2 is preferred over step 3 because reader<->compute overlap is a larger measured win
    # than per-block-overhead amortization for this op's profile.
    ...
    return BlockingPlan(Rt=..., Wt=..., Wt_core=Wt_core, W_true=W_true, W_partial=W_partial,
                        BLOCK_HT=..., WT_REDUCE_BLOCK=..., WT_SCALE_BLOCK=...,
                        DEST_BLOCK=DEST_AUTO_LIMIT, IN_BUF_DEPTH=..., OUT_BUF_DEPTH=2,
                        RM_BUF_DEPTH=2, regime=..., ...)
```

**Coarse-by-default, minimal-only-under-pressure.** `BLOCK_HT`'s and `WT_*_BLOCK`'s Phase 0 values are
the coarsest the budget affords — `WT_*_BLOCK = Wt_core` (the *whole* per-core width, one block) unless
L1 forces a chunk; `BLOCK_HT` grown past 1 whenever budget remains after double-buffering. A block
factor pinned to its minimum is only ever the *output* of the budget solver, never its input.

**Sharded input (Lamps L1/L3) reads its blocking off the shard.** When `memory_layout` is
`*_SHARDED`, `CORE_GRID` and each core's total extent arrive already fixed **and the data is already
in that core's L1**. `blocking_plan` then sets `Wt_core` / rows-per-core from the shard spec and
`BLOCK_HT` defaults to the **whole resident shard height** — sub-chunking only if the shard exceeds the
working-set budget. `cb_input_tiles` is backed on the shard via
`ttnn.cb_descriptor_from_sharded_tensor(CB_INPUT_TILES, input_tensor)` (zero-copy, **no NoC read** —
re-fetching L1-resident rows through a `TensorAccessor` is not sharding). `TensorAccessor` still owns
interleaved I/O, the `gamma` read, and L1's genuinely non-local cross-core gather.

---

## Dataflow Strategy

### Phase 0 data path — TILE input

```
DRAM (interleaved, TensorAccessor)
  --NoC0--> reader (NCRISC) --> cb_input_tiles      [tiles]      (resident in A, streamed in B)
  --NoC0--> reader (NCRISC) --> cb_gamma_tiles      [tiles]      (once per core, never popped)
            reader (NCRISC) --> cb_reduce_scaler    [bf16 tiles] (Regime B only, once per core)

compute (TRISC x3):
  cb_input_tiles -> [A: sum_of_squares]                                  -> cb_sumsq
  cb_input_tiles -> [B: square -> cb_squared -> reduce+Accumulate]       -> cb_sumsq
  cb_sumsq       -> [MulUnary(1/W) -> AddUnary(eps) -> Rsqrt]            -> cb_rms_recip
  cb_input_tiles x cb_rms_recip(Col)  -> cb_normed        (or straight to cb_output_tiles if no gamma)
  cb_normed      x cb_gamma_tiles(Row) -> cb_output_tiles

cb_output_tiles --> writer (BRISC) --NoC1--> DRAM
```

Reader on NoC0, writer on NoC1 (lever B9 — `ReaderDataMovementConfig` / `WriterDataMovementConfig`
defaults) so the read and write streams overlap instead of contending.

### Phase 0 data path — ROW_MAJOR input

Two extra CBs and two extra compute phases; the tile-domain middle is byte-identical.

```
DRAM --NoC0--> reader --> cb_rm_in  [sticks]  --> compute: tilize<WT_BLOCK>   --> cb_input_tiles [tiles]
                                                  ... identical tile-domain math ...
cb_output_tiles [tiles] --> compute: untilize<WT_BLOCK> --> cb_rm_out [sticks] --> writer --NoC1--> DRAM
```

**Reader obligations on the RM path** (these make Regime A valid for non-aligned `W`):

1. Each `cb_rm_in` page is `Wt*32*element_size` bytes wide; the stick read fills only
   `W_true*element_size`. The reader **zero-fills the `(Wt*32 - W_true)*element_size`-byte tail after
   every stick read.** Without this, `tilize` promotes uninitialized L1 into the reduction.
   (P2 optimization: zero each CB slot's tail once at boot instead of per stick — the pad region is at
   a fixed offset the stick read never touches.)
2. For the last tile-row when `H % 32 != 0`, the reader supplies `32 - (H % 32)` **all-zero** sticks so
   `tilize` has a full tile-row. Zero (not garbage) is required: a zero row yields
   `rsqrt(0 + eps) = 1/sqrt(eps)`, finite, and the writer discards it — whereas `inf`/`NaN` garbage
   would be finite-safe only by luck.
3. Writer emits `W_true*element_size` bytes per stick and only `H % 32` sticks from the final
   tile-row — pad is never written back.

### Gamma ingest

| `gamma_layout` | Path |
|---|---|
| `TILE_LAYOUT` | `gamma` is tile-padded to `(1,1,32,Wt*32)`; the reader reads its `Wt_core` tiles straight into `cb_gamma_tiles` via `TensorAccessor`. Row 0 valid. |
| `ROW_MAJOR_LAYOUT` | The reader reads the single `W_true`-element stick into `cb_gamma_rm` (zero-filling the tail **and** the 31 following pad sticks), and compute runs `tilize<WT_BLOCK, cb_gamma_rm, cb_gamma_tiles>` **once at boot**. Row 0 valid, rows 1–31 zero. |

Either way `cb_gamma_tiles` holds `Wt_core` tiles with the valid data in tile row 0, is filled exactly
once per core, and is never popped — so the downstream `mul` with `BroadcastDim::Row` is layout-agnostic.

### Unlocked scheme L1 — cross-core combine over the dependent axis

Phase 0 does not execute this; the contract is fixed here so enabling it is additive.
It serves the interleaved `WIDTH_PARALLEL` regime **and** the `WIDTH_SHARDED` / `BLOCK_SHARDED`
`memory_layout` values identically — the only difference is where the slice comes from (DRAM via
`TensorAccessor` for a logical width split; the core's own L1 via
`ttnn.cb_descriptor_from_sharded_tensor` for a physical width/block shard).

| Item | Contract |
|---|---|
| Group | One **combine group** = the set of cores that jointly own one row-band's full `W`. 1-D `WIDTH_PARALLEL` / `WIDTH_SHARDED`: the whole split. `BLOCK_SHARDED`: the cores of one grid **row** (lever `mcast_topology` — a 2-D block split needs two 1-D mcast families on disjoint `base_sem_id`s, not one 2-D mcast; only the `PerRow` family is needed here because the combine runs along `W`). |
| Per-core slice | `Wt_core = ceil(Wt / group_size)`; the last core's slice is short and is masked by the same `ReducePartialScaler` mechanism Regime B already uses. |
| Step 1 — local partial | Each core computes `Σ_local x²` for its slice into `cb_partial_sumsq` (`BLOCK_HT` tiles). Identical helper call to Phase 0's compute phase 1/2, just over `Wt_core` instead of `Wt`. |
| Step 2 — gather (N→1) | Every non-sender core `noc_async_write`s its `BLOCK_HT` partial tiles into slot `i` of the sender's `cb_partial_gather` (`group_size * BLOCK_HT` tiles), then `noc_semaphore_inc`s the sender's `gather_sem`. The sender `noc_semaphore_wait`s for `group_size - 1`. |
| Step 3 — sum | Sender sums the `group_size` partial tiles element-wise into `cb_sumsq` with `add` over `IterationShape::tiles(BLOCK_HT)` per contributor, accumulating in DEST (`tensix_all_reduce_compute` / `acc_to_dest`; catalog reports 2.70×@2 → 6.75×@16 blocks over the naive form). |
| Step 4 — mcast (1→N) | Sender `SenderPipe::send()` (`mcast_pipe.hpp:171-197`) broadcasts the summed tile block to every group core's `cb_sumsq`; receivers `ReceiverPipe::receive()` (`mcast_pipe.hpp:250-279`). `PRE_HANDSHAKE = true` so the sender cannot overrun receiver buffers across consecutive row-blocks. |
| Step 5 — finish | Every core independently runs the unchanged rms chain on `cb_sumsq`, then scales **its own** slice with **its own** `gamma` slice and writes **its own** output slice. |
| Host wiring | `Mcast1D(device, group_grid, Mcast1DShape::PerRow, starting_sender_index=0, McastConfig(...))` (`host/mcast_host.hpp:156-215`) emits the mcast semaphores + CT/RT args; the gather semaphore is one additional `ttnn.SemaphoreDescriptor(id=..., core_ranges=group_grid, initial_value=0)`. |
| Ordering guarantee | Fully semaphore-ordered: no core reads `cb_sumsq` before the mcast data-ready semaphore fires, and the sender does not mcast before `gather_sem == group_size - 1`. Row-blocks are processed in lockstep within a group. |
| Byte accounting | Input read once, output written once, **`gamma` read once and never replicated** (each core reads only its own slice). Combine traffic is `group_size` one-tile writes + one `group_size`-fan-out mcast of one tile block per row-block. |
| Rejected combine variant | **All-mcast (every core broadcasts its partial to every other, each sums locally)**: `group_size²` tile transfers instead of `2·group_size`, and `group_size` local adds on every core instead of on one. Loses on fan-out at every group size ≥ 3. Recorded, not chosen. |

`ttnn.cb_descriptor_from_sharded_tensor` also makes the L3 (`HEIGHT_SHARDED`) unlock a pure
descriptor swap: the cut is along the **independent** axis, so no combine machinery is involved at all.

---

## Circular Buffers

Sizes are functions of the block/buffer knobs. No CB's *unconditional* size grows with an op
parameter; the two that scale with `Wt_core` (`cb_gamma_tiles`, `cb_normed`) exist **only** in Regime A,
whose selection predicate is precisely "this working set fits L1" — a predicate-guarded resident
fast-path with Regime B's bounded streaming as the fallback.

`T_in = ttnn.tile_size(input.dtype)`, `T_g = ttnn.tile_size(gamma.dtype)`,
`T_f32 = ttnn.tile_size(ttnn.float32)`, `T_bf16 = ttnn.tile_size(ttnn.bfloat16)`,
`S = Wt*32*input.element_size()` (padded stick bytes).

| Semantic Name | Index | Page Size | Num Pages | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|--------|----------|----------|----------|
| `cb_input_tiles` | 0 | `T_in` | A: `IN_BUF_DEPTH * BLOCK_HT * Wt_core`  ·  B: `IN_BUF_DEPTH * BLOCK_HT * WT_REDUCE_BLOCK` | input dtype | reader (TILE) / compute-`tilize` (RM) | compute | A: resident across both passes of one row-block. B: streamed, re-filled for pass B. |
| `cb_gamma_tiles` | 1 | `T_g` | `Wt_core` | gamma dtype | reader (TILE gamma) / compute-`tilize` (RM gamma) | compute | Whole kernel. Filled once, **never popped** — this is what makes the gamma read cost `1×` per core rather than `1×` per row-block. Regime A only; Regime B re-pushes per chunk. |
| `cb_gamma_rm` | 11 | `S` (gamma stick, padded) | `32` | gamma dtype | reader | compute | Boot only, RM gamma only. |
| `cb_reduce_scaler` | 2 | `T_bf16` | `1 + (W_PARTIAL > 0)` | **bfloat16** (mandatory) | reader | compute | Whole kernel; popped once at the very end. Regime B only. |
| `cb_squared` | 3 | `T_in` | `BLOCK_HT * WT_REDUCE_BLOCK` | input dtype | compute | compute | Regime B only. **Sequential-helper intermediate → must hold the FULL block `square` emits per call.** Undersizing it deadlocks. |
| `cb_sumsq_accum` | 4 | `T_f32` | `BLOCK_HT` | float32 | compute | compute | Regime B only; the `Accumulate` target across `W`-chunks. |
| `cb_sumsq` | 5 | `T_f32` | `BLOCK_HT` | float32 | compute | compute | One row-block. Valid region: **Col0**. |
| `cb_rms_recip` | 6 | `T_f32` | `BLOCK_HT` | float32 | compute | compute | One row-block; read `Once` / popped `Never` inside pass B, popped at row-block end. Valid region: **Col0**. |
| `cb_normed` | 7 | `T_in` | A: `BLOCK_HT * Wt_core`  ·  B: `BLOCK_HT * WT_SCALE_BLOCK` | input dtype | compute | compute | `HAS_GAMMA` only. Sequential-helper intermediate → full block. Eliminated by Lamp L4. |
| `cb_output_tiles` | 8 | `T_in` | TILE out: `OUT_BUF_DEPTH * BLOCK_HT * WT_SCALE_BLOCK`  ·  RM out: `BLOCK_HT * WT_SCALE_BLOCK` | input dtype | compute | writer (TILE out) / compute-`untilize` (RM out) | Streaming (real reader/writer pipelining) on the TILE path, so a small depth suffices. On the RM path it feeds `untilize`, a **sequential** helper, so it must hold the full block. |
| `cb_rm_in` | 9 | `S` | `RM_BUF_DEPTH * 32 * WT_BLOCK` sticks | input dtype | reader | compute | RM input only. Pages counted in **sticks**; the tile side counts **tiles**. |
| `cb_rm_out` | 10 | `S` | `RM_BUF_DEPTH * 32 * WT_BLOCK` sticks | input dtype | compute | writer | RM output only. |
| `cb_partial_sumsq` | 12 | `T_f32` | `BLOCK_HT` | float32 | compute | reader | **Lamp L1 only.** Compute produces the local partial; the reader (as gather sender) consumes it. Kept distinct from `cb_sumsq` precisely because a CB read by a dataflow kernel is *consumed* by it — merging them would give `cb_sumsq` two consumers. |
| `cb_partial_gather` | 13 | `T_f32` | `group_size * BLOCK_HT` | float32 | reader | compute | **Lamp L1 only**, sender core only. |

### CB invariants — verified

| Invariant | Verification |
|---|---|
| **Ownership**: exactly one producer kernel and one consumer kernel per CB | Every row above names exactly one of `reader` / `compute` / `writer` on each side. No cell contains "and". The compute→compute intermediates (`cb_squared`, `cb_sumsq_accum`, `cb_sumsq`, `cb_rms_recip`, `cb_normed`) are single-kernel and are the standard sequential-helper pattern (`toy_variance`). **No in-place eltwise anywhere** — every `mul`/`add`/`square` writes to a distinct CB, so compute never becomes a second producer of a reader-fed CB. The L1 partial-sum handoff is split into `cb_partial_sumsq` (compute→reader) and `cb_partial_gather` (reader→compute) for the same reason. |
| **Sync**: producer push count = consumer wait count | Regime A, per row-block: reader pushes `BLOCK_HT*Wt_core` to `cb_input_tiles`; `sum_of_squares` waits that many with `PopPolicy::Never`, pass B waits and **pops** the same count. Net pops == net pushes. Regime B, per row-block: reader pushes `BLOCK_HT*Wt_core` **twice** (pass A + pass B); compute waits/pops `BLOCK_HT*WT_REDUCE_BLOCK` per pass-A chunk over `ceil(Wt_core/WT_REDUCE_BLOCK)` chunks, and likewise in pass B → `2 * BLOCK_HT*Wt_core` total. Equal. `cb_reduce_scaler`: pushed `1+(W_PARTIAL>0)` once, never popped by `reduce()`, popped explicitly once at kernel end. `cb_gamma_tiles`: pushed `Wt_core` once, waited `Once`, popped never (Regime A). |
| **RM page counting** | `cb_rm_in`/`cb_rm_out`/`cb_gamma_rm` count **sticks**; the reader batches all `32*WT_BLOCK` sticks of a tile-row-block before a single `cb_push_back`. Pushing per stick while `tilize` waits for a block is the canonical deadlock. |
| **Scaler format** | `cb_reduce_scaler` is `bfloat16`, page `ttnn.tile_size(ttnn.bfloat16)`, filled by a **pool-type-aware** overload. |
| **No unconditional op-sized CB** | `cb_gamma_tiles`, `cb_normed`, and Regime-A `cb_input_tiles` are `Wt_core`-sized but exist only when `select_regime()` returned "A", i.e. only when the working set was *proved* to fit. Regime B is bounded by `WT_*_BLOCK`. |

---

## API Mapping

Every mechanism has a verified `file:line`. Paths are relative to `ttnn/cpp/ttnn/kernel_lib/` unless
absolute. `DEST_BLOCK` (= `DEST_AUTO_LIMIT`) and `BLOCK_HT` / `WT_*_BLOCK` are the **tunable block
knobs** — thread them from `blocking_plan`, never inline them.

| Phase | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Requirements |
|-------|------|----------|-----------|------------------------|----------|-----------|--------------|
| boot (all) | raw_api | `compute_kernel_hw_startup(cb_a, cb_b, cb_out)` | `eltwise/core/chain.hpp:24-32` (caller-init contract) | — | — | — | Called **once**, before any helper. The chain owns everything else (per-element init, dst-sync window, CB wait/pop/reserve/push, dtype reconfig). |
| boot | helper | `tilize()` | `tilize_helpers.hpp:187-197` | `<block_width_tiles = WT_BLOCK ← knob, cb_gamma_rm, cb_gamma_tiles>`; `num_blocks=1`, `total_input_pages=32` | `cb_gamma_rm` | `cb_gamma_tiles` | `GAMMA_IS_ROW_MAJOR` only. Pass `total_input_pages` because the input CB's pages are stick-sized, not tile-sized (`tilize_helpers.hpp:127-137`). |
| boot | helper | `mm_block_init()` | — | — | — | — | **Not used** — no matmul phase in this op. |
| RM in | helper | `tilize()` | `tilize_helpers.hpp:187-197` | `<WT_BLOCK ← knob, cb_rm_in, cb_input_tiles>`; `num_blocks = BLOCK_HT * ceil(Wt_core/WT_BLOCK)` | `cb_rm_in` | `cb_input_tiles` | `IS_ROW_MAJOR` only. Use `InitUninitMode` to amortize LLK init across back-to-back calls. |
| RM in | helper | `read_sticks_for_tilize()` | `reduce_helpers_dataflow.hpp` family / `toy_tilize_untilize/kernels/reader.cpp:18-24` | `<cb_rm_in, TilizeGranularity::ROW>`; `(accessor, total_num_rows, row_bytes)` | DRAM | `cb_rm_in` | Reader side. The zero-fill of the pad tail and of the `H`-pad sticks is the reader's own responsibility (see Dataflow Strategy). |
| B: scaler | helper | `prepare_partial_reduce_scalers()` | `reduce_helpers_dataflow.hpp:131-132` | `<cb_reduce_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW, W_PARTIAL>(1.0f)` | — | `cb_reduce_scaler` | **Pool-type-aware overload — mandatory.** `SUM`+`REDUCE_ROW` takes the matmul path and needs col-0 scaler layout; the pool-type-aware form auto-dispatches. Used when `W_PARTIAL > 0`. |
| B: scaler | helper | `prepare_reduce_scaler()` | `reduce_helpers_dataflow.hpp:58-60` | `<cb_reduce_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>(1.0f)` | — | `cb_reduce_scaler` | Used when `W_PARTIAL == 0`. Scaler is exactly `1.0` — `PoolType::SUM` (not `AVG`) is chosen deliberately so no `1/W` value is ever quantized into the mandatory-bf16 scaler tile; `1/W` is applied later in fp32. See Key Risks R2. |
| **A: reduce** | helper | `sum_of_squares()` | `eltwise/api/convenience.hpp:74-86` | `<input(cb_input_tiles, WaitPolicy::Once, PopPolicy::Never), row_output(cb_sumsq)>(IterationShape::grid(BLOCK_HT, Wt_core).block_size(DEST_BLOCK))` | `cb_input_tiles` | `cb_sumsq` | Fuses `x*x` and the per-row accumulate in DEST — **no intermediate CB at all**, which is exactly what lets `cb_input_tiles` stay resident for pass B. `PopPolicy::Never` preserves `x`. `block_size` and the `grid` extents are knobs (`chain.hpp:164-167`, `:133-152`). Does **not** mask padding — hence Regime A's `maskless_w` predicate. |
| **B: square** | helper | `square()` | `eltwise/api/convenience.hpp:63` | `<input(cb_input_tiles, PopPolicy::PerTile), output(cb_squared)>(IterationShape::grid(BLOCK_HT, WT_REDUCE_BLOCK).block_size(DEST_BLOCK))` | `cb_input_tiles` | `cb_squared` | `x*x` via FPU on the same CB. |
| **B: reduce** | helper | `reduce()` | `reduce_helpers_compute.hpp:593-611` | `<PoolType::SUM, ReduceDim::REDUCE_ROW, cb_squared, cb_reduce_scaler, cb_sumsq, ReduceInputPolicy::WaitAndPopPerTile, ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT, ReduceFp32Mode::Fast, ReduceAlgorithm::Auto>(ReduceInputBlockShape::of(BLOCK_HT, WT_REDUCE_BLOCK), ReduceInputMemoryLayout::contiguous(), accumulate, NoOp{}, partial_scaler)` | `cb_squared` | `cb_sumsq_accum` → `cb_sumsq` | `accumulate = Accumulate::at(cb_sumsq_accum, c)` for `c < last`, `Accumulate::at_last(cb_sumsq_accum, c)` on the last chunk (`reduce_helpers_compute.hpp:343-417`). `partial_scaler = (c == last && W_PARTIAL > 0) ? ReducePartialScaler::last_tile_at(1) : ReducePartialScaler::none()` (`:319-335`) — **only the last chunk's last tile** gets the partial scaler, per `toy_variance/kernels/compute.cpp:63-64,104-105`. `WT_REDUCE_BLOCK` is the block knob. |
| rms chain | helper | `eltwise_chain()` | `eltwise/core/chain.hpp:551-552` | `(IterationShape::tiles(BLOCK_HT), CopyTile<input(cb_sumsq)>{}, MulUnary<>{INV_W_BITS}, AddUnary<>{EPS_BITS}, Rsqrt<>{}, PackTile<output(cb_rms_recip)>{})` | `cb_sumsq` | `cb_rms_recip` | **One helper call, one dst-sync window, no constant CBs.** `MulUnary` / `AddUnary` (`eltwise/unary/scalar.hpp:28-31`, impl `scalar.inl:47-56`) take a `uint32_t` fp32 bit pattern, so `1/W` and `epsilon` are applied at **full fp32 precision**. `Rsqrt<Approx::Exact, Legacy::Off, Dst::D0>` (`eltwise/unary/math.hpp:37-38`) — `Approx::Exact` for the default; `Approx::Fast` is a P2 precision/perf knob (lever F27). |
| scale | helper | `mul()` | `eltwise/api/convenience.hpp:53` | `<input(cb_input_tiles, PopPolicy::PerTile), input(cb_rms_recip, BroadcastDim::Col, WaitPolicy::Once, PopPolicy::Never), output(cb_scale_out)>(IterationShape::grid(BLOCK_HT, WT_SCALE_BLOCK).block_size(DEST_BLOCK))` | `cb_input_tiles`, `cb_rms_recip` | `cb_scale_out` | `constexpr uint32_t cb_scale_out = HAS_GAMMA ? cb_normed : cb_output_tiles;` — the no-gamma path writes straight to the output CB, so it pays **zero** extra copies. `BroadcastDim::Col` broadcasts the reduce output's Col0 back across `W` (`chain.hpp:314-319`, semantics documented `:308-313`). |
| scale | helper | `mul()` | `eltwise/api/convenience.hpp:53` | `<input(cb_normed, PopPolicy::PerTile), input(cb_gamma_tiles, BroadcastDim::Row, WaitPolicy::Once, PopPolicy::Never), output(cb_output_tiles)>(IterationShape::grid(BLOCK_HT, WT_SCALE_BLOCK).block_size(DEST_BLOCK))` | `cb_normed`, `cb_gamma_tiles` | `cb_output_tiles` | `HAS_GAMMA` only. `BroadcastDim::Row` broadcasts gamma's tile-row 0 down all 32 rows. In Regime A `WT_SCALE_BLOCK == Wt_core` is a **correctness requirement**: `cb_gamma_tiles` is never popped, so a single call must span all gamma columns from the CB front. Regime B chunks and the reader re-pushes gamma per chunk. |
| RM out | helper | `untilize()` | `untilize_helpers.hpp:145-154` | `<WT_BLOCK ← knob, cb_output_tiles, cb_rm_out>`; `num_blocks = BLOCK_HT * ceil(Wt_core/WT_BLOCK)` | `cb_output_tiles` | `cb_rm_out` | `IS_ROW_MAJOR` only. Symmetric tile-sized pages on both sides (`untilize_helpers.hpp:107-110`). |
| DEST sizing | helper | `DEST_AUTO_LIMIT` | `dest_helpers.hpp:89-103` | — | — | — | Auto-detects `fp32_dest_acc_en` + full/half sync → **4** at the Phase 0 corner. Every `block_size` and `BLOCK_HT` reads this constant. Never hardcode `8` or `4`. |
| L1 (lamp) | helper | `SenderPipe::send()` / `ReceiverPipe::receive()` | `mcast_pipe.hpp:171-197`, `:250-279` | `McastArgs<CT_BASE, RT_BASE>::sender(noc)` / `::receiver(noc)` (`:327-395`) | `cb_partial_gather` | `cb_sumsq` | Replaces raw `noc_async_write_multicast` + hand-rolled semaphores for the 1→N leg. |
| L1 (lamp) | helper | `Mcast1D` | `host/mcast_host.hpp:156-215` | `(device, group_grid, Mcast1DShape::PerRow, 0, McastConfig{...})` | — | — | Host emitter of the mcast wire + semaphores + CT/RT args. |
| L1 (lamp) | raw_api | `noc_async_write` + `noc_semaphore_inc` / `noc_semaphore_wait` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h` | — | `cb_partial_sumsq` | `cb_partial_gather` | **Helpers considered and rejected:** `mcast_pipe.hpp`'s `SenderPipe`/`ReceiverPipe` (`mcast_pipe.hpp:171-279`) implement a **1→N multicast** handshake — `SenderPipe::send(src_l1, dst_l1, size)` writes one source to an `McastRect` of destinations. The gather leg is the opposite direction (**N→1** unicast fan-in into `group_size` distinct slots of one core's CB), which no constructor in that file expresses; `ReceiverPipe`'s `NUM_SENDERS` parameter (`:250`) governs multi-sender *signalling*, not a multi-source data fan-in. Concrete mismatch cited. The 1→N leg **does** use the helper. |

**Helper coverage audit — every compute phase is helper-covered.** There is no raw-API compute
fallback anywhere in this design; `compute_kernel_hw_startup` is the caller-init contract the chain
requires, not a bypass. The only raw-API use is the L1-lamp gather leg, justified above with the
file:line mismatch.

---

## Compute Phases

Per row-block, on every core. `Wt_core = Wt` in Phase 0.

### Regime A (RESIDENT-FUSED)

| # | Operation | Helper? | Input CB (name, tiles, state) | Output CB (name, tiles) | CB State After |
|---|-----------|---------|-------------------------------|-------------------------|----------------|
| A0 | boot: `compute_kernel_hw_startup`; RM-gamma `tilize` | yes | `cb_gamma_rm` (32 sticks) | `cb_gamma_tiles` (`Wt_core`) | `cb_gamma_tiles` **resident for the whole kernel**, never popped |
| A1 | RM only: `tilize` input | yes | `cb_rm_in` (`32*WT_BLOCK` sticks × blocks) | `cb_input_tiles` (`BLOCK_HT*Wt_core`) | `cb_input_tiles` full |
| A2 | fused sum-of-squares | `sum_of_squares` | `cb_input_tiles` (`BLOCK_HT*Wt_core`, `Wait Once` / `Pop Never`) | `cb_sumsq` (`BLOCK_HT`, Col0 valid) | **`cb_input_tiles` still holds `x`** — this is the whole point of Regime A |
| A3 | rms chain: `×1/W`, `+eps`, `rsqrt` | `eltwise_chain` | `cb_sumsq` (`BLOCK_HT`) | `cb_rms_recip` (`BLOCK_HT`, Col0 valid) | `cb_sumsq` freed |
| A4 | scale by `1/rms` | `mul` | `cb_input_tiles` (`BLOCK_HT*Wt_core`, popped), `cb_rms_recip` (`Wait Once`/`Pop Never`, `BroadcastDim::Col`) | `cb_normed` (`BLOCK_HT*Wt_core`) — or `cb_output_tiles` if `!HAS_GAMMA` | `cb_input_tiles` drained → reader may refill next row-block |
| A5 | scale by gamma | `mul` | `cb_normed` (popped), `cb_gamma_tiles` (`Wait Once`/`Pop Never`, `BroadcastDim::Row`) | `cb_output_tiles` (`BLOCK_HT*Wt_core`) | `HAS_GAMMA` only; `cb_gamma_tiles` unchanged |
| A6 | RM only: `untilize` | yes | `cb_output_tiles` | `cb_rm_out` | writer drains |
| — | end of row-block | — | — | — | `cb_rms_recip` popped |

### Regime B (STREAMING-MASKED)

| # | Operation | Helper? | Input CB (name, tiles, state) | Output CB (name, tiles) | CB State After |
|---|-----------|---------|-------------------------------|-------------------------|----------------|
| B0 | boot: `compute_kernel_hw_startup`; RM-gamma `tilize` | yes | `cb_gamma_rm` | `cb_gamma_tiles` | reader has filled `cb_reduce_scaler` (`1 + (W_PARTIAL>0)` bf16 tiles) |
| B1 | **pass A**, per chunk `c` of `WT_REDUCE_BLOCK`: `square` | `square` | `cb_input_tiles` (`BLOCK_HT*WT_REDUCE_BLOCK`, popped) | `cb_squared` (`BLOCK_HT*WT_REDUCE_BLOCK`) | `cb_squared` holds the **full** chunk (sequential-helper rule) |
| B2 | same chunk: masked accumulate-reduce | `reduce` + `Accumulate` | `cb_squared` (popped), `cb_reduce_scaler` (not popped) | `cb_sumsq_accum` (`BLOCK_HT`); last chunk → `cb_sumsq` | partial scaler applied **only** on the last chunk's last tile |
| B3 | rms chain | `eltwise_chain` | `cb_sumsq` | `cb_rms_recip` | identical to A3 |
| B4 | **pass B**, per chunk of `WT_SCALE_BLOCK`: scale by `1/rms` | `mul` | `cb_input_tiles` (**re-read from DRAM**), `cb_rms_recip` (`Pop Never`, Col) | `cb_normed` or `cb_output_tiles` | reader re-pushes both `x` and the gamma chunk |
| B5 | same chunk: scale by gamma | `mul` | `cb_normed`, `cb_gamma_tiles` chunk (Row) | `cb_output_tiles` | `HAS_GAMMA` only |
| B6 | RM only: `untilize` | yes | `cb_output_tiles` | `cb_rm_out` | writer drains |
| — | end of kernel | — | — | — | `cb_reduce_scaler` popped `1 + (W_PARTIAL>0)` pages **once** |

---

## Broadcast Verification

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast Dim |
|-------|-----|-------------------|-------------------|---------------|
| A4 / B4 | `mul(x, 1/rms)` | `cb_input_tiles`: 2D `[32, 32]` → **All** | `cb_rms_recip`: REDUCE_ROW output → **Col0** | `BroadcastDim::Col` |
| A5 / B5 | `mul(normed, gamma)` | `cb_normed`: 2D `[32, 32]` → **All** | `cb_gamma_tiles`: 1D `[W]` → **Row0** | `BroadcastDim::Row` |
| L1 step 3 | `add(partial_i, acc)` | `cb_partial_gather` slot: **Col0** | DEST accumulator: **Col0** | `BroadcastDim::None` |

Rules applied: 2D `[H,W]` → All; 1D `[W]` → Row0; REDUCE_ROW out → Col0.
`BroadcastDim` values per `chain.hpp:314-319`; the "REDUCE_ROW output broadcasts back with `Col`"
convention is documented at `chain.hpp:308-313`. **The implementer must confirm the A4 direction
empirically on a single-tile non-square shape before scaling up** — a swapped Col/Row here produces a
plausible-looking tensor with a badly wrong PCC, and it is the single most likely first-run bug.

*No Reduce Direction Verification section:* this op reduces along the last dimension only; `dim` is
not a parameter, so there is no multi-direction surface.

---

## Key Risks and Gotchas

| ID | Risk | Mitigation (structural, in this design) |
|----|------|----------------------------------------|
| **R1** | **Padding folded into the RMS denominator.** The `_PAD_POISON_SHAPES` LOOSE_CASES fill tile padding with `1000.0` on shapes where the pad is 11–38 % of the row; folding it in is a 6–27 % error that also survives PCC as a near-uniform scale. Two distinct bugs: leaking pad *values*, and dividing by the *padded* width. | Values: Regime B's `ReducePartialScaler::last_tile_at(1)` zeroes pad columns of the last `W`-tile; Regime A only runs where the pad is provably zero (RM zero-fill) or absent. Width: the divisor is always `1/W_true` via `MulUnary<>{INV_W_BITS}`, computed from `input.shape[-1]`, never from `Wt*32`. |
| **R2** | **`1/W` quantized into a bf16 scaler.** The reduce scaler CB is mandatorily bfloat16; putting `1/W` there (i.e. `PoolType::AVG`) gives ~0.4 % relative error at `W = 7168`, a systematic output scale error. | `PoolType::SUM` with scaler exactly `1.0`; `1/W` applied afterwards as an **fp32 bit pattern** through `MulUnary`. `epsilon` likewise. No scalar ever passes through bf16. |
| **R3** | **`inf * 0 = NaN` in masked padding.** A masked scaler multiplies the pad by 0, which is `NaN` if the pad is `inf`. | Pad must be finite. RM path: the reader zero-fills. TILE path: guaranteed by the harness (`ttnn.fill_implicit_tile_padding`) — and documented as an op precondition, mirroring `toy_variance.py:45-52`. |
| **R4** | **Sequential-helper intermediate undersized → hang.** `cb_squared`, `cb_normed`, and the RM path's `cb_output_tiles` sit between two compute helpers that each own all three TRISCs, so they cannot pipeline. | Each is sized to the **full** block its producer emits per call, stated explicitly in the CB table. This is the documented deadlock (`ttnn-cb-memory-fundamentals.md:148-154`). |
| **R5** | **`cb_gamma_tiles` never popped ⇒ `WT_SCALE_BLOCK` must equal `Wt_core` in Regime A.** Chunking pass B while gamma's CB front never advances would silently re-apply gamma columns `0..WT_SCALE_BLOCK-1` to every chunk. | Regime A pins `WT_SCALE_BLOCK = Wt_core`; Regime B chunks and the reader re-pushes gamma per chunk. Encoded as an assertion in `blocking_plan`. |
| **R6** | **`Rt` computed with `floor`.** `(batch*H)//32` is wrong for every multi-batch `h_non_aligned` shape. | One `tile_geometry()` function, `ceil` per image, used everywhere. |
| **R7** | **`Rt = 1` under-fills the grid.** Every decode shape is `(1,1,32,W)`. Phase 0 runs those on one core. | Acknowledged and quantified in Performance Methodology; Lamp L1 is the fix and its contract is fully specified so it is additive. |
| **R8** | **`BLOCK_HT` or `block_size` exceeding DEST.** `fp32_dest_acc_en=True` halves DEST to 4 tiles. | Both read `DEST_AUTO_LIMIT`; `blocking_plan` caps them. |
| **R9** | **Scaler tiles popped incorrectly.** Popping 1 when 2 exist leaves a stale tile; the `reduce()` helper never pops the scaler CB. | Pop `1 + (W_PARTIAL > 0)` pages exactly once, at kernel end. |
| **R10** | **Mixed gamma dtype forces a reconfig on the hot path.** `bf16` activations with `fp32` gamma reconfigures the unpacker at the A5 boundary on every block. | Accepted (it is a TARGET cell); `cb_gamma_tiles` carries gamma's own dtype and the chain elides reconfig at compile time when the dtype is unchanged (`compute_block_size`'s reconfig lever, up to 1.19×). The same-dtype case pays nothing. |
| **R11** | **Sharded input re-read over the NoC.** Designing a physical shard to be fetched through a `TensorAccessor` "works" only because the core still holds its rows — it re-fetches L1-resident data. | Lamps L1/L3 mandate `ttnn.cb_descriptor_from_sharded_tensor` (zero-copy) for the resident slice; `TensorAccessor` is reserved for interleaved I/O, `gamma`, and the genuinely non-local gather. |

---

## Performance Methodology

Run as a **per-phase gate** — at Phase 0 and at every refinement — not as an end-of-run pass.

**Classification: movement-dominated in every regime.** The compute is `1` FPU multiply-accumulate
plus `1–2` FPU multiplies per element, against `2–3` tensor-sized DRAM transfers. The op's time is
DRAM/NoC bytes and per-core transaction issue, not FPU throughput. Consequently
`/perf-ceiling-dm` **does** produce a target to chase, and the ranking below is quantitative.

### Bench shapes (named, so each lever has a proving ground)

| Role | Shape | Why |
|---|---|---|
| **Multi-block-per-core** (levers that keep requests in flight: B8 trid double-issue, `split_reader`, C16 depth) | `(1, 1, 8192, 1024)` → `Rt = 256` (≥ 2 row-blocks/core at full grid; 8/core at `ACTIVE_CORE_CAP=32`), and `(99991, 64)` → `Rt = 3125` (28 blocks/core) for the deepest pipeline | A one-block-per-core shape gives these levers **nothing to overlap against**; they would be reasoned away as inapplicable. This is a bench requirement, not a nice-to-have. |
| **Smallest regime** (B0 counterfactual for the per-core-overhead levers B5, B7, B8, B10, B13) | `(32, 17)` — the smallest `INPUTS` entry by element count (544 elements). `eval/verify_levers.py::smallest_input_shape` resolves against exactly this. | A lever that pays on a big shape can regress a shard with ~1 tile of real work. |
| **Grid-filling / DRAM-bound** | `(1, 1, 8192, 1024)`, `(1, 1, 8192, 7168)` | The actual perf targets. Never measure on the tiny golden cells. |
| **Grid-starved** | `(1, 1, 32, 7168)`, `(1, 1, 32, 32768)` | Where `Rt` under-fills and Lamp L1 is the only lever. |

### Levers to walk (`ttnn/ttnn/operations/examples/master.md` Part 2 = the checklist; Part 1 = the runnable measured demo of each)

Selected by where **this** op's time goes, not by habit:

| Lever | Why it applies here | Demo |
|---|---|---|
| **B7** one barrier per *block*, not per transaction | Both passes issue `Wt_core` tile reads per row-block; batching then one barrier is the primary mechanism | `double_buffer` (2.78× with C16) |
| **C16** double-buffer CBs (depth 2) | `IN_BUF_DEPTH` / `OUT_BUF_DEPTH` knobs; the budget solver prefers depth-2 over `BLOCK_HT>1` for exactly this reason | `double_buffer` |
| **B9** reader NoC0 / writer NoC1 | this op genuinely reads *and* writes a full tensor — overlapping the streams is the largest placement lever | `noc_placement` (2.5–4.8×) |
| **A1/A3** spread cores along the DRAM-facing axis (`row_wise=True`) | already in the split call | `noc_placement` (2.9×), `dram_saturation` |
| **A0** classify the bound, then pick the core count | prefill = bandwidth-bound (a knee exists, cap cores); RM/tilize = transaction-rate-bound (no knee, use the full grid). `ACTIVE_CORE_CAP` also shrinks the gamma replication. **Do not cap the tilize path** (a 16-core cap on tilize measured ~2.4× *slower*). | `dram_saturation`, `distribution_gate` |
| **B12 / `shared_input_reuse`** mcast the shared operand | this is Lamp L2 — `gamma` replication is ~21 % of DRAM traffic at `rows_per_core ≈ 2.3` | `shared_input_reuse` (1.71×) |
| **`distribution_gate`** gate the specialized split behind a utilization predicate | this is exactly how Lamp L1 must land: gate `WIDTH_PARALLEL` on `Rt` under-filling, so the prefill regime is provably unregressed (the demo measured byte-identical) | `distribution_gate` |
| **`row_reduce_accumulate` / `reduce_accumulate`** | from `W ≥ 4` tiles the accumulate+finalize path beats a single wide reduce (2.87–2.91× @32 tiles). Regime A's `sum_of_squares` **is** that path; Regime B's `Accumulate` is its chunked form. | `row_reduce_accumulate` |
| **`compute_block_size`** amortize init/reconfig over more tiles per call | `DEST_BLOCK`, `BLOCK_HT`, `WT_*_BLOCK`; plus skipping redundant format reconfig (up to 1.19×, compounding to 1.72×) when input and gamma dtypes match | `compute_block_size` |
| **`sfpu_tile_scope`** scope SFPU to the meaningful axis | the rms chain's `Rsqrt` runs on a whole 32×32 tile of which only **Col0** is meaningful — the ladder measured up to 7.26× for a row-0-scoped op. A cheap, contained P2 win. | `sfpu_tile_scope` |
| **F25/F27** precision cost | Phase 0 is pinned to the maxed-out corner, so these are **not** available at Phase 0. When `bfloat16 + fp32_dest_acc_en=False` is added, DEST doubles to 8 → `DEST_BLOCK` doubles → every B5/B7/C16 tuning must be re-swept. Never downgrade a caller-supplied config. | — |
| **B8 / `split_reader`** keep ≥1 request in flight | Only measurable on the multi-block bench shape above. Regime B's second read pass is a prime candidate if the reader RISC-V is issue-bound. | `split_reader` (1.7×) |
| **`tensix_all_reduce`** combine topology | Lamp L5 — reduce-scatter / tree beat a flat ring 4.6–6.5× | `tensix_all_reduce` |

### Candidate algorithms — ranked, with the losers preserved

**These are predictions, not measurements.** They exist to be reconciled at Phase 0 against the real
device number (`/perf-measure` Mode B: `achieved = target / measured`).

Anchoring: `feature_spec`'s own `achievable_ns` references are measured on `blackhole_p150b` at
1350 MHz. Dividing bytes by those references gives an **evidence-based** effective-bandwidth anchor
rather than an invented constant: `(1,1,8192,1024)` → 33.55 MB / 96 744 ns = **347 GB/s**;
`(1,1,8192,2304)` → 357 GB/s; `(1,1,8192,5120)` and `(1,1,8192,7168)` → **227 GB/s** each. The wide-W
prefill references sitting at exactly ⅔ of the narrow-W references is the signature of a **2-pass
read** (3 tensor-bytes instead of 2: `227 × 1.5 = 341 GB/s`). So the reference implementation streams
wide rows — which is precisely what Regime A avoids. Ranking constant used below:
`DRAM_ACHIEVABLE ≈ 350 GB/s`; per-core single-core NoC read taken as **30 GB/s** (Wormhole's
`double_buffer`/`dram_saturation` measurements of 17.9–21.7 GB/s/core, scaled for Blackhole's 64 B/cyc
link). Both are to be replaced by `noc_estimate.sh` output — the CLI is a test target and is **not
built in this tree** (`./build_metal.sh --build-tests` then re-run the wrapper); firming the bracket
is a Phase 0 reconciliation task.

| Rank | Candidate | Predicted target | Verdict |
|---|---|---|---|
| **1 — WINNER (Phase 0)** | **`Rt`-parallel over the full grid, Regime A resident single-read, Regime B streaming fallback.** Bytes = `1× read + 1× write + num_cores × gamma`. | `(1,1,8192,1024)`: 40.6 MB at 110 cores → **116 µs**; at `ACTIVE_CORE_CAP=32` → **102 µs**; with Lamp L2 → **96 µs** (ref 96.7 µs). `(1,1,32,1024)`: 1 core, 65.5 KB each way pipelined → **3–5 µs** (ref 9.1 µs, **~2× better**). `(1,1,32,2304)`: **~5 µs** (ref 17.0 µs). `(1,1,32,5120)`: Regime A fits marginally (960 KB) → **~11 µs** (ref 75.8 µs). `(1,1,32,7168)`: Regime A does **not** fit (1.34 MB) → Regime B, 1 core → **~46 µs**; even a fitted Regime A gives **11–20 µs** on one core. | Clears **7 of 8** interleaved perf references, several by 2–7×. **Misses the one hard goal**: `(1,1,32,7168)` requires ≤ 14 894 ns (104 259 / 7.0) and a single core straddles or misses it. |
| **2 — Lamp L1 (pre-scheduled P1)** | **`Rt`-parallel gated to `WIDTH_PARALLEL` when `Rt` under-fills**: cut the dependent `Wt` axis across cores, gather one-tile partials to a sender, sum in DEST, mcast back. Bytes = `1× read + 1× write`, `gamma` **not** replicated, plus `2 × group_size` one-tile combine transfers per row-block. | `(1,1,32,7168)` on 28 cores (`Wt_core = 8`): 16.4 KB read + 16.4 KB write per core ≈ 0.55 µs, plus a 56 KB flat gather ≈ 1.9 µs, plus mcast ≈ 0.2 µs → **≈ 3 µs**; with Lamp L5's tree gather → **≈ 1.2 µs**. Against the 14 894 ns goal: **2.5–5× margin.** (The spec's own `WIDTH_SHARDED` measurement for this shape is 5 481 ns, corroborating the bracket.) | **The only candidate that meets the `(1,1,32,7168)` gate.** Also removes the 21 % gamma replication and unlocks 3 `memory_layout` TARGET values. Loses at Phase 0 only because it requires a cross-core combine (semaphores + mcast) that the independent-axis split does not. |
| **3 — REJECTED** | **`Rt`-parallel with an unconditional streaming 2-pass read** (no Regime A at all — the simplest bounded design, and what the reference implementation appears to do for wide `W`). | `(1,1,8192,1024)`: 50.3 MB + gamma → **150 µs** vs a 96.7 µs reference. `(1,1,8192,7168)`: 352 MB → **1 006 µs** vs 1 032 µs — only barely inside. | **Losing property: it moves 1.5× the DRAM bytes.** On a purely movement-bound op that is a 1.5× floor it can never recover. Retained as Regime B, the *fallback* used only where residency provably does not fit — never as the default. |
| **4 — REJECTED** | **Lamp L4 variant promoted to Phase 0**: pre-broadcast gamma to full 32-row tiles at boot and fuse both pass-B multiplies with `DestReuseBinary`, eliminating `cb_normed`. | Removes a `BLOCK_HT × Wt_core` L1 write+read round-trip per row-block and widens Regime A to `Wt = 224`, which would put `(1,1,8192,7168)` on the single-read path: 235 MB → **672 µs** vs the 1 032 µs reference, **1.5×**. | **Losing property: it rests on `DestReuseBinary`'s reuse semantics (`chain.hpp:518-520`), which take a plain `InputSpec` with no `BroadcastDim` field** — so it needs a boot-time gamma broadcast-materialize pass and a `cb_ones` CB that the Phase 0 design does not otherwise require. Deferred to P2 with its predicted 1.5× on wide-W prefill recorded, **not discarded**. |
| **5 — REJECTED** | **All-mcast combine** for Lamp L1 (every core broadcasts its partial to every other; each sums locally). | `group_size²` one-tile transfers vs `2 × group_size`; at 28 cores that is 784 vs 56. | **Losing property: quadratic fan-out.** Recorded so the combine topology decision stays recoverable. |

### Reconciliation obligations

1. At Phase 0, run `/perf-measure` on all four named bench shapes and compare against the row-1
   predictions. `achieved = target / measured`. A miss on the `(1,1,8192,1024)` prediction of
   96–116 µs points at the gamma replication (turn `ACTIVE_CORE_CAP`, then Lamp L2) before anything else.
2. Run `/perf-ceiling-dm` **Mode A** with the estimator actually built, to replace the 350 GB/s and
   30 GB/s ranking constants with an NPE bracket, and re-check the row-1 vs row-2 crossover.
3. Record every Part-2 lever in the ledger `eval/verify_levers.py` reads, with the closed status
   vocabulary and the required evidence fields. The per-core-overhead levers (B5, B7, B8, B10, B13)
   **must** carry a counterfactual measured on `(32, 17)`.
4. Lamp L1 is **not optional**: the `(1,1,32,7168)` case carries
   `minimum_expected_speedup = 7.0`, and row 1's own prediction says the independent-axis split
   cannot reach it. Schedule it as P1.

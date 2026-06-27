# Operation Requirements: matmul

## Definition
- **Formula**: `C[..., m, n] = sum_k A[..., m, k] * B[k, n]` (matches `torch.matmul`).
- **PyTorch Reference**:
  ```python
  def matmul_ref(A, B):  # A: (..., M, K), B: (K, N) or (..., K, N)
      return torch.matmul(A.to(torch.float32), B.to(torch.float32)).to(A.dtype)
  ```
- **Import Path**: `from ttnn.operations.matmul import matmul`
- **Function Signature**:
  ```python
  matmul(
      input_tensor: ttnn.Tensor,            # activation A, (..., M, K), rank 2/3/4
      weight: ttnn.Tensor,                  # weight B, (K, N) or batched (..., K, N)
      *,
      compute_kernel_config: ttnn.ComputeConfigDescriptor = None,  # keyword-only
  ) -> ttnn.Tensor                          # (..., M, N), activation dtype, TILE_LAYOUT
  ```

> ⚠ **READ BEFORE EDITING KERNELS — cache staleness.** `matmul` dispatches via
> `ttnn.generic_op` with kernel source PATHS; the on-disk JIT binary cache keys on
> path + compile args + defines, **not** the `.cpp` content. Editing a kernel and
> re-running tests will **silently reuse the stale binary** and tests pass on the
> OLD code. Before testing any kernel edit:
> ```bash
> find "$TT_METAL_CACHE" -type d \( -name matmul_compute -o -name matmul_reader \
>   -o -name matmul_writer \) -exec rm -rf {} +
> ```
> and run with `--no-precompile` (golden runner also `--no-jit-server`). Confirmed
> by the verifier via a deliberate-compile-error probe.

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: `[x]` complete + all tests pass; `[~]` real work landed but ≥1 named axis value deferred (file the sharper follow-up as `Refinement Nb`); `[ ]` nothing usable produced.
> **`test_translated.py` is not registry-driven.** Its 3 loud signals at Phase 0 are artifacts (LoFi-fidelity precision; a rejection-expecting test; a batched-weight refusal on a non-xfail test) — do NOT treat them as drift. The batched one flips to a real pass under Refinement 3. Verify against `test_golden.py`.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [float32] (activation)
- **SUPPORTED weight_dtype**: [float32]
- **SUPPORTED layout**: [TILE] (== TARGET; matmul is tiled-FPU only, no gap)
- **SUPPORTED fp32_dest_acc_en**: [True]
- **SUPPORTED alignment**: [tile_aligned]
- **SUPPORTED weight_batch**: [single] (shared 2D weight; batched *activation* against it is free and works)
- **EXCLUSIONS**: `{dtype=float32, fp32_dest_acc_en=False}` (maxed input demands maxed accumulator)
- **Cores**: multi-core 2D grid (`GR × GC`) with dual orthogonal multicasts (already done — not a refinement)
- **L1**: bounded for arbitrary M/N/K (K-block streaming + per-core block loop) — all large/wide-K golden shapes pass, no OOM
- **Compute config**: caller-supplied `ComputeConfigDescriptor`; default HiFi4 + `fp32_dest_acc_en=True` + `math_approx_mode=False`
- **Golden baseline**: **20 / 20** registry-supported cells passing; 646 xfail_expected; 0 loud signals (per `verifier_report.json`)
- **Precision**: PCC ≥ 0.99999, relative RMS ≤ 0.0073 on the (32–1024, K up to 4096) ladder

---

### TARGET − SUPPORTED gap (every missing `(axis, value)` from the xfail_expected analysis)

| Axis | Missing value(s) | Refinement |
|---|---|---|
| `dtype` | bfloat16, bfloat8_b | Refinement 1 |
| `weight_dtype` | bfloat16, bfloat8_b | Refinement 1 |
| `fp32_dest_acc_en` | False | Refinement 1 |
| `alignment` | k_non_aligned, n_non_aligned, m_non_aligned | Refinement 2 |
| `weight_batch` | batched | Refinement 3 |
| `layout` | — (TILE == TARGET, no gap) | — |

No gap is covered by INVALID (`INVALID = []`, correctly). All gaps are bundled below.

---

### [ ] Refinement 1 — Numerical configurability (dtypes + fp32_dest_acc_en)

**Goal**: add `ttnn.bfloat16` and `ttnn.bfloat8_b` to `SUPPORTED["dtype"]` **and**
`SUPPORTED["weight_dtype"]` (the two are independent axes — also unlocks the mixed
bf16-activation × fp32-weight path), and add `False` to
`SUPPORTED["fp32_dest_acc_en"]`. Expose / honor `compute_kernel_config` end to end,
make `cb_interm` format and the L1 footprint estimate dtype-aware (interm stays
fp32 under `fp32_dest_acc_en`). Keep the `{float32, fp32_dest_acc_en=False}`
EXCLUSION. Cells that fail out of the box (canonically `bfloat8_b` +
`fp32_dest_acc_en=False`, or any bf16/bf8b cell that can't meet its tolerance) go
to `EXCLUSIONS`, **not** a separate refinement. Pass condition: the formerly-xfail
dtype/acc cells in `test_golden.py` pass at their per-dtype tolerance bands.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**:
- **Do this first** — Refinements 2 and 3 reuse the dtype-aware CB-format/footprint
  derivation introduced here (today `cb_interm`=float32 and the footprint uses
  `tileA_bytes` for every CB — fine only for Phase-0 fp32).
- **bf16 + HiFi4 corruption trap.** The golden harness
  (`eval/golden_tests/matmul/helpers.py:run_matmul`) hardcodes `MathFidelity.HiFi4`
  for **all** dtypes. The `matmul_block_helpers.hpp` header warns: *"AVOID HiFi4 +
  fp32_dest_acc_en with bf16 inputs — silent K-accumulator corruption on Wormhole
  B0 (issue #38306); use HiFi2/HiFi3."* bf16 cells will therefore be graded at the
  corruption-prone HiFi4+acc=True corner. Account for this in-kernel if possible;
  if the only fix is the harness fidelity, the harness is `/golden-tests` territory
  (do not silently edit `helpers.py`) — flag it and, if a bf16 cell can't pass at
  HiFi4, EXCLUSION it with that reason in the changelog rather than fighting #38306.
- Per the matmul_block header, bf16 inputs with Kt>1 need `fp32_dest_acc_en=True`
  to keep K-accumulation from rounding each step (error ~O(√K)); the bf16 +
  `fp32_dest_acc_en=False` cells are the likeliest EXCLUSION candidates.

### [ ] Refinement 2 — Non-tile-aligned M / K / N (in-kernel edge masking)

**Goal**: add `k_non_aligned`, `n_non_aligned`, `m_non_aligned` to
`SUPPORTED["alignment"]` via **in-kernel** masking at the data-access boundary — no
host-side pad/slice. The load-bearing case is **K**: the last K-block's partial
tile must contribute only the valid K-elements to the dot product (mask the invalid
rows/cols of in0/in1, or rely on guaranteed-zero tile padding). M/N edges are the
grid block-edge tiles — the writer already skips out-of-range *whole* tiles; the
remaining work is the partial last tile along each dim. Pass condition: the
formerly-xfail `*_non_aligned` cells in `test_golden.py` pass.

**Implementation skill**: /memory-layouts

**Verifier notes**:
- This is **masking only — matmul is TILE-only**, so there is NO new `layout`
  value to add (the `layout` axis already equals TARGET). `/memory-layouts` is
  pointed at purely for its non-tile-alignment methodology (the "last-tile
  zero-pad / mask in the reader or compute" rule); ignore its ROW_MAJOR-reader
  patterns — they do not apply.
- Depends on Refinement 1 for the dtype-aware CB derivation if combined with bf16
  (the cartesian has `alignment × dtype` cells). Sequence after R1.
- Empirically check whether ttnn already zero-fills tile padding for TILE_LAYOUT
  non-aligned tensors — if so, K-masking may reduce to "trust the zero padding"
  and the real work is just confirming M/N edge writes. Probe before writing
  masking code (`scripts/tt-probe.sh`); don't assume.
- All ragged-edge plumbing already exists for tile-aligned blocks (reader clamps
  `gm>=Mt`/`gn>=Nt`, writer skips `m_tile>=Mt`/`n_tile>=Nt`); this refinement adds
  *sub-tile* masking on top.

### [ ] Refinement 3 — Batched weight (true batched matmul)

**Goal**: add `batched` to `SUPPORTED["weight_batch"]` — a batched weight
`(..., K, N)` whose leading dims match the activation's (one matrix per batch). The
in1 (weight) DRAM read + multicast gains a per-batch tile-id offset (`b*Kt*Nt`) and
the weight is re-read per batch instead of shared. The structural shape-contract
check in `validate()` already permits matching batched leading dims; this lifts the
`weight_batch` SUPPORTED gate and adds the in1 batch-offset in the reader. Pass
condition: the formerly-xfail `weight_batch=batched` cells in `test_golden.py` pass
(and `test_translated.py::test_matmul_with_transpose_and_configs[1-2-4096-32-256]`
flips from `xfail_wrong_mode` to a real pass).

**Verifier notes**:
- No skill in the inventory covers this — it's an op-specific reader data-path
  change (the dtype/layout/memory/parallel skills don't apply). Work from the
  `op_design.md` "Batched weight (refinement)" note: in1 tile-id gains the
  `b*Kt*Nt` offset; activation read is unchanged.
- Sequence **after** Refinement 1 so the batched in1 read already handles the bf16
  / bf8b weight_dtype set (the cartesian has `weight_batch × weight_dtype` cells).
  Independent of Refinement 2; can run before or after it.
- Phase 0 already handles batched *activation* against a shared weight (folds into
  M, free); this is specifically the batched *weight* path.

**Done when**: every Phase 0 cell still passes, and the `weight_batch=batched`
cells in `test_golden.py` pass at their tolerance bands.

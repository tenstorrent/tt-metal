# Operation Requirements: point_to_point

## Definition

- **Formula**: identity / byte copy over the mesh —
  `out[receiver_coord] = in[sender_coord]`; `out[d] = in[d]` for every
  `d != receiver_coord`. No arithmetic (PCC = 1.0, bit-exact).
- **PyTorch Reference** (identity oracle on the sharded tensors):
  ```python
  def point_to_point_ref(shards_in, sender_idx, receiver_idx):
      shards_out = [s.clone() for s in shards_in]
      shards_out[receiver_idx] = shards_in[sender_idx]   # receiver gets sender's shard
      return shards_out                                  # all others unchanged
  ```
- **Import Path**: `from ttnn.operations.point_to_point import point_to_point`
- **Function Signature**:
  ```python
  def point_to_point(
      input_tensor: ttnn.Tensor,
      sender_coord: ttnn.MeshCoordinate,
      receiver_coord: ttnn.MeshCoordinate,
      *,
      topology: ttnn.Topology = ttnn.Topology.Linear,
      output_tensor: ttnn.Tensor | None = None,          # None -> in-place alias of input
      intermediate_tensor: ttnn.Tensor | None = None,    # None -> freshly allocated
  ) -> ttnn.Tensor
  ```

## Scope note — why the queue is a single refinement

`point_to_point` shipped Phase 0 at **near-TARGET coverage**. Per-axis
`TARGET − SUPPORTED`:

| Axis | TARGET | SUPPORTED (now) | Gap |
|---|---|---|---|
| dtype | bf16, fp32, bf8b, **uint16, int32, uint32** | bf16, fp32, bf8b | **uint16, int32, uint32** |
| layout | TILE, ROW_MAJOR | TILE, ROW_MAJOR | — (bf8b+RM is INVALID) |
| topology | Linear, Ring | Linear, Ring | — |
| alignment | tile_aligned, non_tile_aligned | tile_aligned, non_tile_aligned | — |

The only unlockable golden cells are the three integer dtypes → one refinement.
Multi-link/multi-core throughput and sharded-input support are **not** in the queue:
neither corresponds to a TARGET axis / golden cell (see `verification_report.md`
§Recommendations). Do not pad the queue to hit a refinement count — there is exactly
one support-surface gap.

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. Fix by updating SUPPORTED.
> **Checkbox protocol**: `[x]` complete + all tests pass; `[~]` real work landed but ≥1 named axis value deferred (queue treats as completed, surfaced as partial); `[ ]` nothing usable produced.
> **Refinement ID naming**: primary refinements are `Refinement N`; a partial's sharper follow-up appends a lowercase letter (`Refinement 1b`), ordered immediately after its parent. The runner parser matches exactly `Refinement \d+[a-z]?`.
> **Verification runner**: this is a multi-device CCL op — verify with
> `scripts/run_multidevice_sim_pytest.py --op point_to_point -- <target>` (mesh `(2,4)`,
> FABRIC_1D, topology `bh_8xP150_p2p`), **not** `run_safe_pytest.sh`. Activate the
> clone's `python_env` first so `sys.executable` is this build's python.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16, float32, bfloat8_b]
- **SUPPORTED layout**: [TILE, ROW_MAJOR]
- **SUPPORTED topology**: [Linear, Ring]
- **SUPPORTED shape-derived axes**: alignment ∈ {tile_aligned, non_tile_aligned}
- **Cores**: single worker core (0,0) per participating device; single fabric link (link_idx=0)
- **Coordination**: one op-internal `GlobalSemaphore` (module-cached), two-phase ready/done handshake
- **Golden baseline (representative 72-cell subset)**: 30 supported_pass / 36 xfail_expected / 6 invalid_skipped; all loud categories 0 (per `verifier_results/verifier_report.json`)
- **Precision**: PCC = 1.0, bit-exact, all shapes/dtypes

### [ ] Refinement 1 — Integer dtype passthrough (uint16, int32, uint32)

**Goal**: add `ttnn.uint16`, `ttnn.int32`, `ttnn.uint32` to `SUPPORTED["dtype"]`. This
flips the 36 currently-`xfail_expected` integer golden cells
(`dtype ∈ {UINT16, INT32, UINT32}` × {TILE, ROW_MAJOR} × {Linear, Ring} × shapes) to
passing. There is **no** INVALID for the integer dtypes (unlike bf8b, they have a
row-major representation), so **both** layouts must pass for each.

**Verifier notes**:
- No skill pointer. `/numeric-formats-metal` does **not** apply: it targets *float*
  precision surfaces (bf16/fp32/bf8b already shipped) and bundles
  `ttnn.ComputeKernelConfig` / `math_fidelity` / `fp32_dest_acc_en` /
  `UnpackToDestFp32` — all compute-kernel concepts. `point_to_point` has **no compute
  kernel**; integer passthrough is pure byte movement with no precision config.
- Expected to be **near-zero kernel work** — the four dataflow kernels are
  dtype-agnostic byte copies (`tt_memmove` + `noc_async_read/write` sized in bytes).
  The likely levers are host-side only:
  1. Add the three dtypes to `SUPPORTED["dtype"]` in `point_to_point.py`.
  2. Confirm `ttnn.CBFormatDescriptor(data_format=<int dtype>)` accepts uint16/int32/
     uint32 (the CB is a raw byte staging buffer; format only sets element width).
  3. Confirm `ttnn._ttnn.fabric.ccl_packet_dims(dtype, ...)` frames integer pages
     correctly — the bf16 `bit_floor` packet clamp is bf16-specific and should not
     apply, so integers use the full packet like fp32.
  4. `_datum_size` already returns `ttnn.element_size` for these (uint16→2,
     int32/uint32→4); the block-float fallback path is not taken.
- If any integer dtype proves harder than a focused pass (e.g. a CB-format or
  packet-framing surprise for one width), land the ones that work as `[~]`, add the
  hold-out to `EXCLUSIONS`, and file `Refinement 1b` naming the specific dtype +
  blocker. Do **not** add an integer dtype to INVALID — integers are structurally
  valid byte payloads (INVALID is reserved for the bf8b+RM data-format impossibility).

**Done when**: the golden integer cells pass on the `bh_8xP150_p2p` sim; a fresh
`eval.verify_supported` run shows `xfail_expected` for `dtype` dropped to 0 (only the
6 bf8b+ROW_MAJOR `invalid_skipped` remain), and no new `supported_fail` / `xpass_drift`.

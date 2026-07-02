# Operation Requirements: point_to_point

## Definition
- **Formula**: identity fabric byte-copy across two mesh devices.
  `output_shard[receiver_coord] = input_shard[sender_coord]`; `output_shard[c] = input_shard[c]`
  for every `c != receiver_coord`. No arithmetic.
- **PyTorch Reference** (per-device-shard oracle — identity):
  ```python
  def point_to_point_ref(input_shards, sender_idx, receiver_idx):
      out = [s.clone() for s in input_shards]
      out[receiver_idx] = input_shards[sender_idx].clone()
      return out
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
      output_tensor: ttnn.Tensor | None = None,
      intermediate_tensor: ttnn.Tensor | None = None,
  ) -> ttnn.Tensor
  ```

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean support was added but SUPPORTED was not updated — fix by updating SUPPORTED.
> **Checkbox protocol**: `[x]` complete + all tests pass; `[~]` real work landed but ≥1 named axis value deferred (surfaced as partial); `[ ]` nothing usable produced.
> **Refinement ID naming (runner parses `Refinement \d+[a-z]?`)**: primary refinements are `Refinement N`; partial follow-ups append a letter (`Refinement 1b`), ordered immediately after their parent.
>
> **Multi-device runner reminder** (this is a CCL op): verify with
> `scripts/run_multidevice_sim_pytest.py --topology bh_8xP150_p2p -- <test>` after
> `source python_env/bin/activate` (a bare `python3` may pick up a different clone's stale
> `ttnn._ttnn`). Do NOT use `run_safe_pytest.sh` (forces slow dispatch in sim). Target the
> `bh_8xP150_p2p` topology (not `--op`) so the run does not fan onto the optional
> `(4,2)`/`(8,4)` topologies the tests don't match. `mesh_device` is function-scoped, so the
> 8-chip fabric re-inits per test (~17 s/case) — select curated `-k` slices.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16, float32, bfloat8_b]
- **SUPPORTED layout**: [TILE, ROW_MAJOR]
- **SUPPORTED shape-derived axes**: alignment ∈ {tile_aligned, non_tile_aligned}
- **SUPPORTED op-specific axes**: topology ∈ {Linear, Ring}
- **Cores / links**: single Tensix core per participating device, single fabric link (1-D unicast)
- **Compute config**: N/A (pure data movement — no compute kernel, no ComputeKernelConfig)
- **Golden baseline**: curated golden run = 15 supported_pass / 16 xfail_expected /
  3 invalid_skipped, all loud categories 0 (`supported_fail=0`, `xpass_drift=0`,
  `xfail_wrong_mode=0`). Precision baseline 8/8 bit-exact (PCC=1.0). Multi-hop extended
  4/4. (Full matrix is 432 cells; the integer-dtype xfail band is the only non-passing
  region and is addressed below.)

### [ ] Refinement 1 — Integer passthrough dtypes (uint16 / int32 / uint32)

**Goal**: add `ttnn.uint16`, `ttnn.int32`, `ttnn.uint32` to `SUPPORTED["dtype"]`, moving
the integer-dtype `xfail_expected` cells to passing. In the curated golden run these are the
16 cells `Counter({INT32: 6, UINT32: 6, UINT16: 4})` across
`{TILE, ROW_MAJOR} × {Linear, Ring} × {tile_aligned, non_tile_aligned}`; across the full
432-cell matrix it is the entire `dtype ∈ {uint16, int32, uint32}` band (~144 cells, minus
none — integers have no INVALID entry). After this, `SUPPORTED["dtype"]` equals
`TARGET["dtype"]` (all six), and `TARGET − SUPPORTED` is empty on every axis.

**Verifier notes**:
- **No skill applies.** `/numeric-formats-metal` is for the *compute*-precision surface
  (math_fidelity / fp32_dest_acc_en / intermediate-CB formats / `UnpackToDestFp32`) of an
  op with a compute kernel. point_to_point has **no compute kernel** — it copies raw bytes
  and never interprets the dtype — so that skill's methodology does not transfer. This is a
  data-plane passthrough expansion; the levers are host-side, not kernel-precision.
- **Expected to be near-zero kernel work.** The four kernels are dtype-agnostic byte movers
  (`noc_async_read` / `tt_memmove` / fabric `write_page` operate on page/packet bytes). The
  concrete edits/checks are all host-side:
  1. `SUPPORTED["dtype"]` += the three integer dtypes (and re-confirm `validate()` — it reads
     `input_tensor.dtype` directly, so nothing structural changes).
  2. `_allocate_intermediate()` already handles them via the non-bf8b branch
     (`element_size()` = 2 B for uint16, 4 B for int32/uint32; layout mirrors input). Confirm
     `ttnn._ttnn.fabric.ccl_packet_dims` is happy with integer dtypes — it only special-cases
     `BFLOAT16` (`bit_floor`); integers take the full fabric packet size, which is the correct
     path. No new branch expected.
  3. The 16-byte page-size `validate()` gate: every INPUTS shard has last dim a multiple of 8,
     so uint16 (×2 B) and int32/uint32 (×4 B) RM pages stay 16-B-aligned — no gate trips.
- **The golden harness already supports these dtypes**: `helpers._TORCH_DTYPE` maps
  uint16/int32/uint32 → `torch.int32` and `_make_full_tensor` generates them via
  `torch.randint`; `check_output`'s PCC oracle casts to float. So once SUPPORTED admits them,
  the formerly-xfail cells should pass as-is.
- **The only real risk** is whether `from_torch` / `to_torch` round-trips uint16/uint32 cleanly
  through mesh sharding on this build (torch has no native uint16/uint32 in older versions).
  If a specific integer dtype does not round-trip (a framework limitation, not a kernel gap),
  drop *that dtype only* from SUPPORTED and note it — do **not** add an EXCLUSIONS cell
  (integers have no partner axis to exclude against). bfloat8_b×ROW_MAJOR remains INVALID and
  is unaffected.

**Done when**: `SUPPORTED["dtype"] == [bfloat16, float32, bfloat8_b, uint16, int32, uint32]`;
a golden run over the integer-dtype cells shows them `passed` (0 `xfail_expected` remaining on
`dtype`, 0 `xpass_drift`, 0 `supported_fail`); the receiver shard equals the sender shard
bit-for-bit for each integer dtype.

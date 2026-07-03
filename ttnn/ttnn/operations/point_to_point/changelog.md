# point_to_point — changelog

## 2026-07-03 — Phase 1: fresh implementation (self-contained Python CCL op)

Implemented `point_to_point` end-to-end from `op_design.md` as a self-contained
Python CCL op built on `ttnn.generic_op` + `ttnn.MeshProgramDescriptor`. Pure data
movement (identity byte copy, PCC ~1.0): the receiver device's shard becomes the
sender's input shard; every other device's shard is unchanged. Does NOT wrap the
bound C++ `ttnn.point_to_point`.

### What shipped

- **Entry point** (`point_to_point.py`): registry declarations (`INPUT_TAGGERS`
  with `tag_alignment`, `SUPPORTED`, `EXCLUSIONS=[]`), `validate()` (structural
  ValueError gates + registry axis gate), module-cached `GlobalSemaphore`,
  intermediate allocation, in-place output aliasing.
- **Descriptor** (`point_to_point_program_descriptor.py`): two `ProgramDescriptor`s
  on one `MeshProgramDescriptor` — send @ sender_coord, receive @ receiver_coord;
  route + packet framing via `ccl_dm_route` / `ccl_packet_dims`; fabric conn args
  appended in the kernel-matched `[has_forward][fwd][has_backward][bwd]` layout.
- **Four dataflow kernels** (no compute kernel):
  - `sender_reader` (NCRISC): input shard pages → cb_shard_pages.
  - `sender_writer` (BRISC): coalesce pages → packets (`tt_memmove`), `FabricStream`
    unicast into the receiver's intermediate, done-inc; op-owned ready-wait +
    cache-reuse sem reset (reset BEFORE the outgoing inc).
  - `receiver_reader` (NCRISC): ready-signal (`FabricStreamSender::signal`), wait
    done, local `noc_async_read` of the intermediate + de-coalesce → cb_shard_out;
    cache-reuse sem reset (reset AFTER the wait).
  - `receiver_writer` (BRISC): cb_shard_out → output shard.
- **Package re-export** (`__init__.py`): `point_to_point` + registry contract
  (`SUPPORTED`/`EXCLUSIONS`/`INPUT_TAGGERS`) at package level for the golden harness.

### SUPPORTED (Phase-0 proven subset of TARGET)

- `dtype`: `[bfloat16, float32, bfloat8_b]` (acceptance-test dtypes; integer
  passthrough uint16/int32/uint32 remain refinement candidates)
- `layout`: `[TILE, ROW_MAJOR]`
- `topology`: `[Linear, Ring]`
- `alignment`: `[tile_aligned, non_tile_aligned]`

### Accuracy

Identity byte copy — PCC exactly 1.0 (test tolerance 0.995–0.999). All shapes,
dtypes, layouts and topologies transfer bit-for-bit.

### Test results (required sim topology `bh_8xP150_p2p`, mesh (2,4), FABRIC_1D)

Full `test_point_to_point.py` acceptance matrix passes (60/60 parametrizations,
run in batches via `scripts/run_multidevice_sim_pytest.py`):
- `test_point_to_point`: 50/50 (5 shapes × {bf16,fp32}×{TILE,RM} + bf8b×TILE × {Linear,Ring})
- `test_point_to_point_nonparticipating_unchanged`: 4/4
- `test_point_to_point_output_tensor`: 4/4
- `test_point_to_point_program_cache`: 2/2 (semaphore re-arm + RT-arg re-apply across cache hit)

### Issues encountered

- **bf8b intermediate datum**: `ttnn.element_size(bfloat8_b)` throws
  (`tt::datum_size(Bfp8_b)` is undefined for block-float). Fixed with a
  block-float-safe `_datum_size` returning 1 byte — the intermediate is a pure
  byte landing zone whose addressing is driven by the packet_size TensorAccessor
  override, so datum=1 safely (over-)sizes the tile-padded intermediate. (The C++
  reference's `compute_output_specs` calls `datum_size` unconditionally and would
  throw here too; the Python op is the bf8b-capable path.)

### Advisory deviations from op_design.md

- Design's illustrative SUPPORTED (`dtype:[bfloat16,float32]`, `topology:[Linear]`)
  was written "e.g."; the immutable acceptance test exercises bf8b and Ring, so
  SUPPORTED was widened to the proven acceptance set. No binding change.

### Tests added

None beyond the immutable acceptance test (no deterministic debug test needed —
the only failure was a clear host-side ValueError, fixed directly).

## 2026-07-03 — Phase 0 verification (incremental-verifier)

- **What was done**: Code review + registry-conformance audit + golden suite
  (representative subset) + `eval.verify_supported` + precision baseline, all on the
  deterministic multi-device craq-sim runner (`bh_8xP150_p2p`, mesh (2,4), FABRIC_1D).
- **SUPPORTED at Phase 0** (unchanged — honest, no drift found):
  dtype=[bfloat16, float32, bfloat8_b], layout=[TILE, ROW_MAJOR],
  topology=[Linear, Ring], alignment=[tile_aligned, non_tile_aligned].
- **Accuracy achieved**: PCC = 1.000000, max_abs = 0.0, mean_abs = 0.0, rms = 0.0
  (bit-exact identity copy; measured on 8 cells / 4 shapes ×
  {bfloat16, float32}-TILE via `test_point_to_point_precision_baseline.py`).
- **Golden suite at Phase 0**: representative 72-cell subset (of 432; full run is
  ~105 min of kHz-sim device time). 30 supported_pass / 36 xfail_expected (integer
  dtypes) / 6 invalid_skipped (bf8b+ROW_MAJOR). **All loud categories 0** —
  supported_fail=0, xpass_drift=0, xfail_wrong_mode=0. Subset covers every axis value
  (see `verifier_results/verifier_report.json`).
- **Fixes applied**:
  - `point_to_point.py`: `validate()` is now the entry point's first line;
    `packet_dims` made lazy inside `validate()` (only for the optional
    intermediate-spec check). Behavior-preserving.
  - `test_point_to_point_precision_baseline.py`: mesh shape `(1,2)` → `(2,4)` (the
    `(1,2)` matched no sim topology and would hang fabric init); largest sweep shape
    `512×512` → `256×128` to fit the sim wall-clock budget.
- **Issues encountered**: no drift, no kernel bugs. Program-cache test (the semaphore
  cache-reuse footgun) passes on both topologies. The `(1,2)` precision-baseline mesh
  was the one real defect (a fabric-init hang), fixed. Reproducibility gotcha: the
  multi-device runner must be launched with the clone's `python_env` activated (else
  `sys.executable` can resolve a different checkout's venv that lacks
  `ttnn.operations._op_contract`).
- **Refinements queued**: 1 — integer dtype passthrough (uint16/int32/uint32); the
  only `TARGET − SUPPORTED` gap. See `op_requirements.md`.
- **Artifacts**: `verification_report.md`, `op_requirements.md`,
  `verifier_results/{verifier_report.json,test_results.json,test_axes.json}`.

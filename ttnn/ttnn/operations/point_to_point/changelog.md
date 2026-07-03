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

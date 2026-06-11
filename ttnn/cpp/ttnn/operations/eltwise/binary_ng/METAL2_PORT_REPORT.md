# Metal 2.0 Port Report — binary_ng (`BinaryNgDeviceOperation`)

**Status: IN PROGRESS — host fully ported & C++-GREEN; the no-broadcast · tile path AND the scalar
(b-absent) path are converted and DEVICE-VALIDATED on Wormhole (n300). The remaining runtime-selected
kernel paths (broadcast, row-major) are enumerated below as mechanical follow-on.**

Ported the single `ProgramFactory` from `ProgramDescriptorFactoryConcept` to `ProgramSpecFactoryConcept`.
The host `create_program_spec` is atomic and covers **all** paths; kernels are JIT-compiled per selected
path, so the C++ host builds/validates independently and kernel conversion proceeds path-by-path.

## On-device validation (Wormhole B0 n300)

**No-broadcast tile path:** `test_binary_ng_program_cache.py -k "not scalar and not broadcast"` →
**10/10 pass** (8 as-written + 2 with updated cache-count assertions, see Friction). FPU (bf16 add) +
SFPU (fp32 add/mul/sub), tensor-tensor, interleaved DRAM + L1, sub-core-grids, differing op-types /
input-dtypes / output-dtypes, correctness across differing logical shapes. No PCC mismatches, no hangs.

**Scalar (b-absent) path:** `test_binary_ng_program_cache.py -k "scalar"` → **3/3 pass**;
`test_binary_scalar.py` → **18/18 pass** (SFPU comparison ops gt/lt/ne/ge/le/eq, bf16). Validates the
DFB_B producer-moves-to-the-writer binding and the unpack-fp32-gating fix (see Friction).

Converted kernels:
| File | Notes |
|---|---|
| `kernels_ng/dataflow/reader_interleaved_no_bcast.cpp` | no-bcast tile reader. `ta::a`/`ta::b`, `dfb::cb_a`/`cb_b`, named RTAs, `has_sharding` CTA. Validated. |
| `kernels_ng/dataflow/writer_interleaved_no_bcast.cpp` | no-bcast tile writer. `ta::c`, `dfb::cb_c`. Validated. |
| `kernels/compute/eltwise_binary_no_bcast.cpp` | FPU no-bcast; `#if HAS_ACTIVATIONS`-gated interims. Validated (bf16). |
| `kernels/compute/eltwise_binary_sfpu_no_bcast.cpp` | SFPU no-bcast; isclose `rtol`/`atol` named. Validated (fp32 add/mul/sub). |
| `kernels/compute/eltwise_where_no_bcast.cpp` | where no-bcast; scalar → `args::scalar`. **Converted, not yet device-validated.** |
| `kernels/dataflow/reader_interleaved_no_bcast.cpp` (legacy `kernels/`) | scalar-path reader (a only); member `get_tile_size()`. Validated. |
| `kernels/dataflow/writer_interleaved_scalar.cpp` | scalar-path writer; **also producer of DFB_B** (fills c_1 with the packed scalar via `FILL_*`). `ta::c`. Validated. |
| `kernels/compute/eltwise_binary_scalar.cpp` | FPU scalar compute; interims `#if`-gated. Validated. |
| `kernels/compute/eltwise_binary_sfpu_scalar.cpp` | SFPU scalar compute; isclose named. Validated (comparison ops). |

## TTNN ProgramFactory
### Concept realized
`ProgramSpecFactoryConcept` (single program, no op-owned resources). Default `MaximizeCacheReuse`
(no `caching_strategy` declaration).

### Device-op-class edits
- **Custom `compute_program_hash` deleted**: `binary_ng_device_operation.cpp:487` (decl
  `binary_ng_device_operation.hpp:79`). Reverts to the default reflection hash.
- **Pybind entry points removed**: none — binary_ng has no pybind/nanobind exposure of `create_descriptor`.

### Open items
- **`dynamic_tensor_shape = true` adopted on a/b/c**, sanity-checked: all three accessors use
  `ArgConfig::RuntimeTensorShape` (shape read at runtime), the faithful mirror per the migration guide.
  **Confirmed forward-looking**: today's framework auto-hash still keys per shape, so the relaxation does
  *not* yet collapse cache entries across shapes (validated — see Friction "cache behavior"). It is
  correctness-accurate for the future generalized factory concept that auto-computes a relaxation-capturing
  hash (per invoker).
- **Relaxation candidate / custom-hash reintroduction** (for the op owner): the deleted custom hash keyed
  on dtype + memory_config + shard_volumes and omitted shape. Post-port the op keys per shape (correct,
  more cache entries). If a corrected shape-independent hash is ever wanted, that's an op-owner decision,
  not a port-time call.

## Handoff points
None. No cross-op kernels, no kernel-lib coupling, no `sem::`/`ta::` boundary violations, no removed
pybind surface. All sources are binary_ng-owned.

## Successes
- **Recipe + layernorm reference were sufficient to author the entire ~900-line host in one pass**; the
  C++ build went green after only **two** trivial API-name corrections (see Friction "doc gaps"). The
  patterns catalog's Conditional-binding pattern was exactly the lens for the `c_3`/`c_4` activation
  interims.
- **The atomic-unit / JIT split held** (as the layernorm report noted): the host compiled and validated
  independently of the kernels, so the large host rewrite was checkpointed C++-green before any kernel was
  touched, and kernels were converted/validated path-by-path. An unconverted path fails only at JIT when
  selected — never at host build.
- **`#if HAS_ACTIVATIONS(LHS)`-gated `dfb::cb_a_interim` worked**: the no-activation tests passed, i.e. the
  conditional DFB token was correctly excluded from name-lookup when `c_3`/`c_4` were unbound. No new
  define was needed — `HAS_ACTIVATIONS` (driven by the host `PROCESS_*_ACTIVATIONS` defines) is the same
  condition that gates the host-side binding, so the two stay in lockstep by construction.
- **The recipe's "a DFB's producer can move between kernels across paths" warning fired exactly as
  described** and was the key to the scalar path. DFB_B (c_1) is produced by the **reader** on tensor-b
  paths but by the **writer** on the scalar path (`writer_interleaved_scalar.cpp` fills c_1 with the
  packed scalar; compute consumes it). The 1:1 "reader produces b" binding from the no-bcast path would
  have left DFB_B with a consumer and no producer on the scalar path (validator: 0 producers). Resolved
  by mapping the role per path: reader-PRODUCER when `b` is a tensor, writer-PRODUCER when `b` is absent,
  compute-CONSUMER always. Validated (scalar 3/3 + 18/18).

## Friction
### Gaps (doc didn't match reality)
- **`TensorParameter` has no `advanced_options` member — it's `relaxations`.** Both the migration guide
  (TensorParameter §) and the brief say `TensorParameter::advanced_options.dynamic_tensor_shape`. The
  actual field is `TensorParameter::relaxations.dynamic_tensor_shape` (type `TensorParameterRelaxations`,
  `advanced_options.hpp:179/207`). Cost a build iteration. **Suggested:** fix the guide/brief to say
  `relaxations`. (Contrast `KernelSpec`/`DataflowBufferSpec`, which *do* use `advanced_options` — the
  asymmetry is the trap.)
- **`Table<DFBSpecName, UnpackToDestMode>` has no `.contains()`.** The migration guide says `Table` has
  "the syntax of `std::unordered_map`", which implies C++20 `.contains()`. It doesn't compile. Worked
  around by reordering (set the generic `Default` entries first, let the SFPU-specific entries overwrite
  via `operator[]`), avoiding the membership check entirely. **Suggested:** note in the recipe's `Table`
  paragraph which lookup ops are actually available (`operator[]`, `insert`; not `contains`).
- **The spec validator forbids `UnpackToDestFp32` on a non-Float32 DFB; the legacy CB-id-indexed
  `unpack_to_dest_mode` array set it unconditionally for SFPU.** Legacy set
  `unpack_to_dest_mode[c_0/c_1/...] = UnpackToDestFp32` for every SFPU op regardless of CB format —
  harmless on a bf16 CB in the legacy array. Metal 2.0's validator (`program_spec.cpp:825`) hard-rejects
  it: *"UnpackToDestFp32 ... but the DFB data format is not Float32 ... Use Default or omit."* Caught by
  `test_binary_scalar.py` (SFPU bf16 comparison ops): all 18 failed at `MakeProgramFromSpec` until the
  port gated the SFPU `UnpackToDestFp32` on `dfb_is_fp32(id)`. Numerics confirmed unchanged after the
  gate (18/18 pass), so the legacy Fp32-on-bf16 setting was a no-op in practice. **Suggested:** the
  migration guide's `unpack_to_dest_mode` section should state the FP32-only constraint explicitly — it
  currently says only "every Float32 DFB ... must appear", not "non-Float32 DFBs must NOT carry
  UnpackToDestFp32". A porter mechanically translating the legacy array hits this.

### Confusion / near-misses
- **`is_sfpu_op` routing surprised the "no-bcast FPU" framing.** fp32 `add` routes to the **SFPU**
  no-bcast kernel, not the FPU one — so validating the no-broadcast path with the (fp32-heavy)
  program-cache tests required converting *all three* no-bcast compute kernels (FPU / SFPU / where), not
  just `eltwise_binary_no_bcast.cpp`. The "no-broadcast path" is really {2 dataflow kernels} × {3 compute
  kernels selected by `is_sfpu`/`is_where`}. Worth a recipe note: when sizing a "path", the compute
  flavor axis (`is_sfpu`/`is_where`) multiplies it even within one broadcast type.
- **Cache behavior change is the expected consequence of the mandated hash deletion** (not a bug). Two
  tests asserted cross-shape program reuse (`cache_entries_counter.total == 1`); post-port the default
  hash keys per shape, so they observe one entry per distinct shape (2 and 3). Their docstrings cited
  `override_runtime_arguments` and "logical_shape excluded from compute_program_hash" — both mechanisms
  the port removes. Updated the two **cache-count** assertions to the new correct values (preserving the
  PCC checks) with comments referencing this report. This matches the TTNN-integration doc's guidance that
  the hash-deletion cache-miss trade-off is correct and should be recorded.
- **Unconverted-path JIT failure segfaults pytest.** When a not-yet-converted kernel is selected, its
  positional `get_compile_time_arg_val(0)` static-asserts (host now emits only named args); the resulting
  `TT_THROW` then **segfaults the Python process during traceback rendering** (exit 139) rather than a
  clean test failure. Cosmetic but it kills the whole pytest session, so paths must be excluded via `-k`
  until converted. Not card-related (no `0xdeadc0de`). **Suggested:** flag to the framework/test team.

## Open items for downstream
- **Cross-op kernel touches:** none — all sources binary_ng-owned.
- **`eltwise_where_sfpu_scalar` missing-`.cpp`** (`binary_ng_utils.cpp:125`): op-owner's latent bug, left
  untouched (audit Misc anomalies).
- **`ComputeHardwareConfig` defaults:** the op derives `fp32_dest_acc_en` from data formats (not from a
  `DeviceComputeKernelConfig`), so the port did **not** use `to_compute_hardware_config`; it sets only
  `fp32_dest_acc_en` + `unpack_to_dest_mode` and leaves `math_fidelity` / `math_approx_mode` /
  `dst_full_sync_en` at `ComputeHardwareConfig` defaults — mirroring the legacy `ComputeConfigDescriptor`,
  which set none. fp32 numerics passed, so the defaults are equivalent in the tested cases; worth a
  confirming glance if a fidelity-sensitive path later regresses.

### Remaining kernel conversions (mechanical follow-on — host already emits specs for all of these)
Each follows the validated pattern (named args, `dfb::`/`ta::`, `#if`-gated conditional DFBs). Map each
DFB's producer/consumer role **per source** before converting.

- **Broadcast tile computes + readers** (`kernels_ng/`): row / col / scalar / row-col-mixed × {fpu, sfpu,
  where where present}, plus `reader_interleaved_{row,col,scalar,row_col_mixed}_bcast.cpp`.
- **Row-major path** (`kernels_ng/...rm_*`): 6 readers + `writer_interleaved_rm_no_bcast.cpp` + the rm
  compute selection. (rm and sharding are mutually exclusive.)
- **where-scalar** (`eltwise_where_sfpu_scalar`): blocked by the op-owner's missing-`.cpp` bug
  (`binary_ng_utils.cpp:125`) — the path resolves to a nonexistent file pre-port, so it can't be
  exercised until the op owner fixes the extension. Not converted.

**Host-side item to resolve when the broadcast path is validated (best-effort / unvalidated today):**
- **Subtile-broadcast scratch `DFB_A_BCAST`/`DFB_B_BCAST` (c_5/c_6)** are bound reader-PRODUCER /
  compute-CONSUMER as a best-effort guess; verify against the broadcast readers/computes when converting
  them (a DFB's producer can move between kernels across paths — as the scalar path's DFB_B demonstrated).

**Resolved this session:** the scalar (b-absent) path is converted and validated — `DFB_B` is now
produced by the writer on that path (see Successes); the SFPU `unpack_to_dest_mode` is gated on the DFB
format (see Friction).

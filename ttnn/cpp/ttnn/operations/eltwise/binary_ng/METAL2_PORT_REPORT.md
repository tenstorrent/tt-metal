# Metal 2.0 Port Report — binary_ng (`BinaryNgDeviceOperation`)

**Status: IN PROGRESS — host fully ported & C++-GREEN; the no-broadcast · tile path is converted and
DEVICE-VALIDATED on Wormhole (n300). The remaining runtime-selected kernel paths (broadcast, row-major,
scalar) are enumerated below as mechanical follow-on.**

Ported the single `ProgramFactory` from `ProgramDescriptorFactoryConcept` to `ProgramSpecFactoryConcept`.
The host `create_program_spec` is atomic and covers **all** paths; kernels are JIT-compiled per selected
path, so the C++ host builds/validates independently and kernel conversion proceeds path-by-path.

## On-device validation (Wormhole B0 n300)

`test_binary_ng_program_cache.py -k "not scalar and not broadcast"` → **10/10 pass** (8 as-written + 2
with updated cache-count assertions, see Friction). Exercises the no-broadcast tile path end-to-end:
host spec build → framework validation → kernel JIT → device → PCC, across:
- **FPU** (bf16 add) and **SFPU** (fp32 add/mul/sub) — both no-broadcast compute kernels.
- tensor-tensor, interleaved **DRAM** and **L1**, sub-core-grids, differing op-types / input-dtypes /
  output-dtypes, and correctness across differing logical shapes (per-shape cache entries; see below).
- No PCC mismatches, no hangs.

Converted kernels (all no-broadcast tile):
| File | Notes |
|---|---|
| `kernels_ng/dataflow/reader_interleaved_no_bcast.cpp` | `ta::a`/`ta::b`, `dfb::cb_a`/`cb_b`, named RTAs, `has_sharding` CTA. Validated. |
| `kernels_ng/dataflow/writer_interleaved_no_bcast.cpp` | `ta::c`, `dfb::cb_c`, named RTAs. Validated. |
| `kernels/compute/eltwise_binary_no_bcast.cpp` | FPU; `#if HAS_ACTIVATIONS`-gated `dfb::cb_a_interim`/`cb_b_interim`. Validated (bf16). |
| `kernels/compute/eltwise_binary_sfpu_no_bcast.cpp` | SFPU; isclose `rtol`/`atol` → named RTAs. Validated (fp32 add/mul/sub). |
| `kernels/compute/eltwise_where_no_bcast.cpp` | where; scalar → `args::scalar`. **Converted, not yet device-validated** (no test in the run hit a no-bcast where). |

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
- **Scalar path (b absent)**: legacy `kernels/dataflow/reader_interleaved_no_bcast.cpp`,
  `writer_interleaved_scalar.cpp`, `eltwise_binary[_sfpu]_scalar.cpp`, `eltwise_where_sfpu_scalar.cpp`.

**Two host-side items to resolve when those paths are validated (best-effort / unvalidated today):**
1. **Scalar path `DFB_B` (c_1) is created always but bound only when `b` is present** — on the b-absent
   scalar path it would be an unbound DFB (validator requires ≥1 producer + ≥1 consumer). Resolve by
   binding it on the scalar compute kernel (if it reads `c_1`) or by not creating it when `b` is absent.
2. **Subtile-broadcast scratch `DFB_A_BCAST`/`DFB_B_BCAST` (c_5/c_6)** are bound reader-PRODUCER /
   compute-CONSUMER as a best-effort guess; verify against the broadcast readers/computes when converting
   them (a DFB's producer can move between kernels across paths).

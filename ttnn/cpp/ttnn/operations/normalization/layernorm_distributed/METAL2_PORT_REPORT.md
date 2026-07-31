# Metal 2.0 Port Report — `normalization/layernorm_distributed`

## Outcome

**`PORTED`** — all **five** factories converted to `ProgramSpecFactoryConcept`:
`LayerNormPreAllGatherProgramFactory`, `LayerNormPreAllGather2DProgramFactory`,
`LayerNormPreAllGatherWelfordProgramFactory`, `LayerNormPostAllGatherProgramFactory`,
`LayerNormPostAllGatherWelfordProgramFactory`. Nothing is left on the legacy concept; no
`create_descriptor` remains in the directory.

Verification status, including the paths this host cannot exercise, is under *Verification*.

## Provenance

- **Recipe docs (this port):** `4d6e9518bf5 2026-07-30 docs(metal_2.0): require an explicit opt_level when porting compute kernels`
- **Audit docs (inherited):** `4d6e9518bf5 2026-07-30 docs(metal_2.0): require an explicit opt_level when porting compute kernels`

## TTNN ProgramFactory

### Concept realized

`ProgramSpecFactoryConcept`, as the audit chose. All five factories return
`ttnn::device_operation::ProgramArtifacts` from `create_program_artifacts`; both
`program_factory_t` variants flip wholesale, so no per-factory concept mixing was needed.

### Device-op-class edits

- **Custom `compute_program_hash` deleted:** none — neither DeviceOperation declared one.
- **Pybind entry points removed:** none — `layernorm_distributed_nanobind.cpp` binds only the two
  user-facing functions, never a factory entry point, so no pybind surface changed.
- The only device-op-class change is the factory method signature in the two `*_device_operation.hpp`
  files (`create_descriptor` → `create_program_artifacts`) and the include swap that goes with it
  (`<tt-metalium/program_descriptors.hpp>` → `"ttnn/metal_v2_artifacts.hpp"`). `validate_on_program_cache_miss`,
  `compute_output_specs`, `create_output_tensors` and `select_program_factory` are untouched.

### Open items

- **Relaxation candidates:** none applied and none obviously available. Neither DeviceOperation has a
  custom hash, so no relaxation was ever in force to mirror.
- **Concept fit:** clean. No op-owned tensors, no op-owned `GlobalSemaphore`s, no per-coordinate
  program variation.

---

## Handoff points

### 1. `LayerNormPostAllGatherWelfordProgramFactory` is broken on `main`, and the port cannot reproduce the breakage — OPS TEAM

**Tagged: pre-existing defect surfaced by the port; a behavior change on one path is unavoidable.**

The shared post-allgather reader
(`device/kernels/dataflow/reader_unary_interleaved_ln_rm_gb_post_allgather.cpp`) reads **9** compile-time
args (`reduce_factor` at index 8, then `TensorAccessorArgs<9>()`) and **10** runtime args (`eps` at 5,
`gamma_addr` at 6, `beta_addr` at 7, `stats_addr` at 8, `y_offset` at 9). Pre-port, the Welford post
factory emitted **8** compile-time args (no `reduce_factor`) and **11** runtime args with
`packed_winv_value` inserted at slot 5, so:

- every runtime arg from `eps` onward landed one slot late (`eps` read `packed_winv_value`,
  `gamma_addr` read `eps`, `beta_addr` read the gamma address, `stats_addr` read the beta address);
- `TensorAccessorArgs<9>()` started one word inside the input accessor's argument block.

The default post factory has always matched the reader exactly, so the mismatch is Welford-only.

**Confirmed empirically on the pre-port tree.** A single-device
`layer_norm_pre_all_gather` → `layer_norm_post_all_gather` run with
`LayerNormDefaultProgramConfig(use_welford=True)` returned `max_abs_err = 10.48` against the torch
reference, with a NaN cosine similarity. The path survives on `main` because it is exercised only by
`mesh_device=(1, 8)` tests, which skip on a single-device host.

**Why the port cannot preserve it:** Metal 2.0 arguments are addressed by name, not by position, so
there is no positional shift left to reproduce — the ported factory binds `eps` to `eps` and the stats
address to the stats tensor. The ported Welford post path is therefore expected to be *correct* where
the legacy one produced garbage. This is the one place in the port where "no behavior change" does not
hold, and it is not a change the port could have declined to make.

**Ask:** confirm the ported behavior is the intended one, and decide whether the legacy path also needs
a fix on `main` ahead of this port landing, for anyone still on the old code.

### 2. RMSNorm + gamma + beta drove an unallocated buffer index — FUNCTIONAL FIX BUNDLED AT INVOKER'S DIRECTION

**Tagged: deliberate scope exception, authorized by the invoker mid-port.**

`rmsnorm_post_allgather.cpp` sets `cb_times_gamma_out_idx = tt::CBIndex::c_13` whenever both gamma and
beta are present and then drives it, but `layernorm_post_all_gather_program_factory.cpp` allocated
`c_13` only under `if (!is_rmsnorm)`. That config therefore drove an unconfigured buffer index (audit
*Misc anomalies* #1 / *Questions* #2). Metal 2.0 cannot bind a buffer no spec declares, so the port had
to resolve it one way or another.

Three options were put to the invoker: (a) faithful swap, letting that config fail at kernel-compile
time; (b) allocate the buffer for RMSNorm too; (c) reject the config in validation. **The invoker chose
(b).** The ported factory declares `TIMES_GAMMA_OUT` whenever gamma **and** beta are present, RMSNorm
included.

This is a **functional change inside a port diff**, which the recipe's scope discipline otherwise
forbids. It is called out here so a reviewer bisecting this commit knows the numerics of
RMSNorm + gamma + beta changed (from undefined to defined) for a reason unrelated to Metal 2.0.

### 3. Shared kernel forks created in the peer `rmsnorm_distributed` directory — RMSNORM_DISTRIBUTED OWNERS

Two compute kernels this op file-path-instantiates live in a peer op's directory. Neither had a
`_metal2` fork, so this port created them (rung 2), leaving the legacy originals untouched apart from
the pointer comment:

| original | fork created | remaining legacy consumers |
|---|---|---|
| `rmsnorm_distributed/device/kernels/compute/rmsnorm_pre_allgather.cpp` | `…/rmsnorm_pre_allgather_metal2.cpp` | **none** |
| `rmsnorm_distributed/device/kernels/compute/rmsnorm_post_allgather.cpp` | `…/rmsnorm_post_allgather_metal2.cpp` | **none** |

Both originals were bound *only* by this op's factories (`grep -rl` across `ttnn/` returns just the
three call sites in `layernorm_distributed`). With this port landed the legacy copies have **no
consumers left**, so they are immediately retirable — that deletion is the owners' call, not the
porter's. The pointer comment at the top of each original is the only other edit made there. No build
file was touched: the per-family `file(GLOB_RECURSE kernels …)` already covers that directory.

### 4. A shared kernel-pool helper still takes a `CircularBuffer` — KERNEL-LIB OWNERS

`generate_bcast_col_scalar(CircularBuffer cb, uint32_t scalar)` in
`ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp` takes the legacy wrapper **by value**. The
post-allgather reader calls it, so one `CircularBuffer` construction survives in this op's kernel code:

```cpp
generate_bcast_col_scalar(CircularBuffer(dfb::eps), eps);
```

The audit cleared this shape as ✓ (by value, not by reference), and it works because `dfb::eps`
converts to `uint32_t` at constexpr time. But it means the recipe's "no `CircularBuffer` reference
survives in the op directory" sweep cannot come back completely clean for this op. The helper is
outside the porter's scope, so it was not modified. A `DataflowBuffer` overload (or a templated one)
would let the last reference go.

---

## Successes

- **[Conditional / optional DFB bindings]** fired exactly as documented, and the warning that
  `if constexpr` does not gate name lookup was the single most load-bearing thing in the catalog for
  this op. Both post compute kernels gated their gamma/beta chains on `do_gamma` / `do_beta` **CTAs**
  via `if constexpr`; the pattern's *"promote a CTA gate to a define"* paragraph is exactly what those
  needed. Without it, the natural port would have kept the CTAs and hit a name-lookup failure on every
  gamma-absent build.

- **The same pattern's note about always-emitted defines** caught a subtler variant. The legacy
  `FUSE_PRE_ADD` define was always emitted (as `"0"` or `"1"`) and tested with `#if`, which *looks*
  like it already gates correctly. It does not: `#if FUSE_PRE_ADD` still leaves `dfb::res` in the token
  stream on the unfused path. The port emits the define only when true and tests with `#ifdef`.

- **[Two-toucher DFB → assign 1P+1C]**, specifically its *"re-derive, don't transcribe"* instruction,
  is the reason the Pre-Welford `c_1` disposition was caught rather than copied. See *Friction* #1: the
  brief's disposition is not expressible, and only re-running the census surfaced that.

- **[Compiler options]** is a section that would have been skipped without its own warning. Four of the
  five factories set no `opt_level` at all, and the recipe's insistence that an absent
  `KernelDescriptor::opt_level` still resolves to `O3` on a `ComputeConfigDescriptor` is what put an
  explicit `KernelBuildOptLevel::O3` on all five compute `KernelSpec`s. Nothing would have failed
  without it — it would just have been quietly slower.

- **[Hardware configuration]** steered the port to `to_compute_hardware_config` and the reader/writer
  helpers rather than hand-built Gen1 configs, removing the whole class of "did I invert
  `dst_full_sync_en`?" mistakes. The `double_buffer_dest = !dst_full_sync_en` inversion in particular is
  a trap the helper closes.

---

## Friction

### Gaps

**1. The audit's census-to-disposition table produces an unimplementable disposition when an extra
producer coexists with a compute self-loop.**

For `(c_1, LayerNormPreAllGatherWelfordProgramFactory)` the brief prescribes
`advanced_options.allow_instance_multi_binding`, on a correct census: the shared reader is a locked
producer (`reader_unary_interleaved_ln_rm_gb_pre_allgather.cpp:31-32` pre-port) and the compute kernel
is locked to **both** roles (`layernorm_pre_allgather_welford.cpp:215-220,277-283,290-293` pre-port).
Binding that literally — reader PRODUCER, compute PRODUCER + CONSUMER, flag set — is rejected by
`ValidateProgramSpec`: the self-loop rule requires the producer KernelSpec set to equal the consumer
set, and that check is **not** relaxed by the multi-binding flag.

The workable assignment is reader = PRODUCER, compute = CONSUMER, no flag. The per-node census is then
exactly 1P + 1C, no self-loop is declared, and on Gen1 the buffer lowers to a plain circular buffer
whose FIFO pointers live in SRAM and are driven by whichever RISC executes the call — so the compute
kernel's packer-side `push_back` behaves as before. Endpoint role also does not affect the per-buffer
data format or the `unpack_modes` slot, both of which are keyed on the buffer itself.

*Suggested doc change:* the endpoint-assignment procedure's third bullet ("≥2 kernels locked to the
same FIFO role → multi-binding") needs a carve-out: **if one of those kernels is locked to both roles,
multi-binding is not available — drop the self-loop and assign the two kernels opposite roles.** Worth
stating in the patterns catalog *and* in the audit's census table, since the auditor reaches the same
wrong disposition first.

**2. The per-node DFB invariant forces `KernelSpec` multiplicity that is not a work split, and neither
the recipe nor the catalog describes that shape.**

The 2D pre factory produces its final output buffer from a compute kernel declared over `all_cores` and
consumes it from a writer declared over `merge_cores`. That is legal with legacy CBs and is exactly
what the brief's *"DFB core range narrower than its binding kernel's core range"* watch-for points at —
but the brief frames it as "confirm the spec validator accepts it," and the validator does **not**: the
per-node census fails on every worker node with "producer but no consumer."

The resolution is to instantiate the compute source **twice** over disjoint node sets, with only the
merge instance binding the output buffer, and to move the `is_merge_core` selector from a runtime arg
to a `compiler_options.defines` entry so the conditional binding can be `#ifdef`-gated. That combines
*Preserved Multiplicity* with *Conditional / optional DFB bindings*, but the recipe discusses
multiplicity only for per-group CTAs from `split_work_to_cores`, and the plan template expects
"none — no work-split multiplicity in legacy" when there is no per-group CTA. That is literally true
here, and multiplicity was still required.

*Suggested doc change:* add a pattern entry — *"asymmetric producer / consumer placement → split the
wider kernel into per-region KernelSpecs"* — and reword the audit's watch-for from "confirm the
validator accepts it" to "this will be rejected; plan for a KernelSpec split." A porter who takes the
watch-for at face value discovers this only at first run, after the whole factory is written.

**3. The recipe's "no `CircularBuffer` survives" sweep cannot be satisfied when an out-of-scope donor
takes one by value.** See *Handoff points* #4. The kernel-side whitelist states the transition is
"total" and that a grep should return zero hits, while the audit's donor-shape table blesses
`generate_bcast_col_scalar(CircularBuffer cb, …)` as ✓. Both are reasonable; together they are
contradictory, and the porter has to pick which to violate. *Suggested doc change:* the whitelist's
totality claim should carve out "except a wrapper constructed inline at a call site whose out-of-scope
donor still takes `CircularBuffer` by value," and say to report it.

**4. `unpack_modes` needs entries the legacy vector never had, and the rule is easy to under-apply.**
The recipe describes the newly-required explicit entry for a Float32 buffer consumed with
`enable_32_bit_dest = true`, but frames it as a per-op curiosity. In this op it applies broadly: with
`fp32_dest_acc_en`, `cb_data_format` becomes Float32, so *most* intermediates in both post factories
become Float32 buffers the compute kernel consumes. Hand-listing them per config would have been
error-prone, so the port added a helper (`device/layernorm_distributed_metal2_helpers.hpp`,
`fill_default_unpack_modes`) that walks the compute kernel's CONSUMER bindings and fills `UnpackToSrc`
for any Float32 buffer without an explicit mode — reproducing the legacy all-`Default` vector exactly,
and leaving the deliberately-set `UnpackToDest` entries alone. *Suggested doc change:* recommend this
sweep as the mechanical approach rather than implying a short hand-written list.

### Confusion

**5. "Preserved Multiplicity" reads as being about work splits, so the *absence* of per-group CTAs felt
like a green light.** All five factories put their per-core row count in a runtime arg, so that plan
section legitimately reads "none." Correct for the anti-pattern it guards, but it primed the
expectation that one `KernelSpec` per legacy `KernelDescriptor` would always be right — which Friction
#2 then contradicted. Splitting the section into "multiplicity from work splits" and "multiplicity from
placement" would remove the false reassurance.

**6. The `Table` vs `Group` distinction bites hardest at `compiler_options.defines`.** `defines` is
`Table<std::string, std::string>`, and the natural conditional construction (declare empty, add entries
under an `if`) needs `emplace`, not `push_back`. The recipe does call this out and the call-out was
useful; noting it only because `defines` is the one field where conditional construction is the
*common* case, so it would be worth naming explicitly in that paragraph.

---

## Open items for downstream

### Shared kernel touches

Full detail in *Handoff points* #3: two forks created (rung 2) in
`rmsnorm_distributed/device/kernels/compute/`, pointer comments added to both originals, and **no
unmigrated consumers remain** for either original, so both legacy copies are retirable now.

Four kernel sources **inside** this op's directory are shared between its own factories:
`writer_unary_interleaved_start_id_blocked.cpp` (all five factories),
`reader_unary_interleaved_ln_rm_gb_pre_allgather.cpp` (two),
`reader_unary_interleaved_ln_rm_gb_post_allgather.cpp` (two), and the in-directory
`compute/chain_llk.hpp` header (two). Because all five factories converted in this one change, these
were converted **in place** and needed no intra-op fork. Porting the factories one at a time would have
forced three forks plus a duplicated header inside this directory — the concrete argument for keeping
this op's factories together, and worth knowing for any op with the same shape.

### Dead buffer allocations dropped

Every drop below removes an allocation with no toucher, so none changes behavior. The audit found the
first four; the last two are additional, config-conditional cases its per-`(CB, config)` table did not
separate out.

| buffer | site (pre-port) | scope |
|---|---|---|
| `c_9` (var + epsilon) | `layernorm_post_all_gather_program_factory.cpp:490-496` | dead in all configs |
| `c_9` | `layernorm_post_all_gather_welford_program_factory.cpp:554-560` | dead in all configs |
| `c_7` (mean²) | `layernorm_post_all_gather_welford_program_factory.cpp:583-589` | dead — Welford factory only |
| `c_8` (var) | `layernorm_post_all_gather_welford_program_factory.cpp:545-551` | dead — Welford factory only |
| `c_6` (reduced stats) | `layernorm_post_all_gather_program_factory.cpp:472-478` | dead **when `is_rmsnorm`** — that kernel reduces the stats straight into `c_8` and never names `c_6` |
| `c_13` (×gamma intermediate) | `layernorm_post_all_gather_program_factory.cpp:536-545` | dead **when beta is present but gamma is not** — the ×gamma stage is the buffer's only toucher, and it does not run |

Two kernel-side declarations that were dead alongside `c_9` went with it:
`layernorm_post_allgather.cpp:115` and `rmsnorm_post_allgather.cpp:52` (the latter only in the new fork;
the legacy original keeps it).

### Findings left for the ops team

1. **The Pre-2D factory ignores `is_rmsnorm`** (audit *Misc anomalies* #6). It hardcodes
   `layernorm_pre_allgather_2d.cpp` with no RMSNorm branch and forces `out0_tiles = 1`, while
   `compute_output_specs` sizes a LAYERNORM output at two tile columns — so a LAYERNORM +
   `use_2d_core_grid` request appears to produce only `E(x²)` into a two-column output tensor. The port
   reproduces this exactly and does not act on it.

2. **The Post-Welford factory's `is_rmsnorm` compute-source branch is unreachable** — validation
   rejects RMSNorm + Welford before the factory runs. The branch is kept as written, and the buffer set
   and argument schema are built for the Welford kernel, the only reachable source. Worth deleting on
   the ops track, together with a decision on whether RMSNorm + Welford should ever be supported.

3. **`log_debug(tt::LogOp, "device_id: {}", gamma.value().device()…)`** in
   `layernorm_post_all_gather_welford_program_factory.cpp` dereferences `gamma` unconditionally, so a
   Welford post-allgather call without a weight faults there before any validation message. Left
   exactly as-is.

4. **`packer_l1_acc` is destructured and then ignored in all five factories** (audit *Misc anomalies*
   #7). Callers that set `packer_l1_acc=true` silently get no effect, while the value still feeds the
   default program hash and so causes cache misses that change nothing. The port preserves this exactly:
   `to_compute_hardware_config` also does not translate `packer_l1_acc`. The destructure is retained
   (marked `[[maybe_unused]]` in the two factories where nothing else reads it) so the validation it
   performs on the config still runs.

5. **The pre-allgather reader pushes a reduce-scaler tile into the Welford factory's scratch buffer**
   (audit *Misc anomalies* #2). Still true, still harmless, and now visible in the spec as the reason
   that buffer needs a reader-producer / compute-consumer assignment instead of a plain self-loop.
   Gating the reader's scaler generation behind a define, or giving the Welford factory a scratch index
   the reader does not touch, would let it become an ordinary compute self-loop.

6. **`layernorm_pre_allgather_welford.cpp` pushes into the output buffer with no preceding
   `reserve_back`** (audit *Misc anomalies* #8). Unchanged by the port.

### Test coverage notes

**The Post-Welford path has no single-device coverage.** Every test that reaches it is a
`mesh_device=(1, 8)` parametrization, which skips on a one-device host — which is exactly why the
defect in *Handoff points* #1 went unnoticed on `main`. The 2D core-grid paths are *not* in this gap:
`test_distributed_rmsnorm_allgather.py::test_rmsnorm_2d_core_grid_single_device[use_2d_core_grid=True-…]`
exercises both the Pre-2D factory and the post factory's 2D work split on a single device.

Adding a single-device Post-Welford case would be cheap and worth doing. The probe written to confirm
*Handoff points* #1 is the whole recipe: `layer_norm_pre_all_gather` → `layer_norm_post_all_gather`
with `LayerNormDefaultProgramConfig(use_welford=True)` and a `create_layer_norm_reciprocals` tensor
reaches the Welford post factory with no mesh and no all-gather at all, and compares directly against
`torch.nn.functional.layer_norm`.

---

## Verification

### Build

`./build_metal.sh -e --enable-fake-kernels-target` — **SUCCESS**, no warnings introduced (the tree
builds with `-Werror`).

### Tests

Baseline (pre-port) and post-port runs over the invoker-confirmed set:

- `tests/ttnn/unit_tests/operations/fused/test_distributed_layernorm_exhaustive.py`
- `tests/ttnn/unit_tests/operations/fused/test_distributed_layernorm_sharded.py`
- `tests/ttnn/nightly/unit_tests/operations/fused/test_distributed_layernorm_pre_allgather.py`
- `tests/ttnn/nightly/unit_tests/operations/fused/test_distributed_layernorm_post_allgather.py`
- `tests/ttnn/nightly/unit_tests/operations/fused/test_distributed_rmsnorm_allgather.py`

`test_distributed_layernorm.py` and `tests/ttnn/distributed/test_distributed_layernorm_TG.py` skip
unconditionally today (`LEGACY_CCL_SKIP`, tt-metal#26649) and were excluded by the invoker.

| | passed | failed | skipped | xfailed |
|---|---|---|---|---|
| **Baseline (pre-port)** | 421 | 0 | 557 | 10 |
| **Post-port** | 421 | 0 | 557 | 10 |

Identical. The 557 skips are the `mesh_device=(1, 8)` parametrizations; this host has one device.

A confirmation run of the three nightly files after the final cleanup pass (dead destructures removed,
ternaries parenthesized) gave 231 passed, 0 failed, 102 skipped.

**Paths this run does cover:** Pre 1D (LN and RMSNorm, with and without a residual, dtype mismatches),
Pre Welford (including the fp32-precision and residual cases), Post default (1D and 2D work splits,
gamma-only, beta-only, non-tile-aligned width), and the Pre-2D factory via the single-device
2D-core-grid RMSNorm test.

**Path this run does not cover:** Post Welford — see *Test coverage notes*. It was verified instead by
the direct probe below.

### Direct probe of the Post-Welford path

Single-device `layer_norm_pre_all_gather` → `layer_norm_post_all_gather` with
`LayerNormDefaultProgramConfig(use_welford=True)`, gamma and beta present, compared against
`torch.nn.functional.layer_norm`:

| tree | max abs error | cosine similarity |
|---|---|---|
| pre-port (`main`) | 10.475094 | **NaN** |
| post-port | 0.244583 | 0.999938 |

This is the behavior change described in *Handoff points* #1: the legacy path returned garbage, the
ported path is correct. The residual max-abs error is ordinary bfloat16 rounding on the i/o tensors.

### Anti-pattern self-audit

| check | result |
|---|---|
| No `tensor.buffer()->address()` survived | ✅ zero hits in code |
| No magic CB indices / `CBIndex` / `CBDescriptor` in the op directory | ✅ zero hits |
| No `TensorAccessorArgs<N>()` in any ported kernel | ✅ zero hits |
| Conditional DFB bindings follow the pattern | ✅ `FUSE_PRE_ADD`, `FUSE_GAMMA`, `FUSE_BETA`, `IS_MERGE_CORE` each bound conditionally on the host, emitted as a `compiler_options.defines` entry, and `#ifdef`-gated on both the alias and every use. No binding made unconditional as a workaround. |
| No `.id` extraction or temp DFB wrappers at LLK call sites | ✅ zero hits |
| No CTA→RTA demotion | ✅ no per-group CTA existed to demote |
| No unnecessary multi-binding flag, never stacked with a self-loop | ✅ `allow_instance_multi_binding` appears nowhere in the port |
| All CTAs named | ✅ every `compile_time_args` is a `{name, value}` table |
| No nameable argument smuggled into varargs | ✅ no vararg mechanism used anywhere |
| No ephemeral doc cited from code | ✅ zero `.md` references in any changed `.cpp` / `.hpp` |
| Every `hw_config` reproduces the legacy resolved values | ✅ readers and writers use `create_reader_datamovement_config` / `create_writer_datamovement_config`, which reproduce the legacy `ReaderConfigDescriptor{}` / `WriterConfigDescriptor{}` defaults exactly; compute uses `to_compute_hardware_config`, which reads the same four fields the legacy `ComputeConfigDescriptor` was given, with the `dst_full_sync_en` → `double_buffer_dest` inversion handled inside the helper. `bfp_pack_precision_mode` stays default, matching the legacy unset `bfp8_pack_precise`. `unpack_modes` reproduces each legacy `unpack_to_dest_mode` vector entry for entry. |
| Every `KernelSpec`'s `opt_level` matches its legacy kernel's | ✅ explicit `KernelBuildOptLevel::O3` on all five compute specs (legacy `ComputeConfigDescriptor` default); readers and writers left at the `O2` default (legacy DM default) |

The one deliberate exception is the surviving `CircularBuffer(dfb::eps)` shim at
`device/kernels/dataflow/reader_unary_interleaved_ln_rm_gb_post_allgather.cpp`, forced by an
out-of-scope donor signature — see *Handoff points* #4.

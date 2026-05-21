# Port report — `reduction/generic`

Records what happened during the Metal 2.0 port of the four reduction program factories. Captures handoff points, doc successes, friction surfaced during the port, and items for downstream pickup.

## Handoff points

### API: ComputeConfiguration::unpack_to_dest_mode is fiddly for FP32-input compute kernels (recurring port-time stumble)

The host validator for compute kernels requires an explicit `unpack_to_dest_mode` entry for every CONSUMER binding of a Float32-format DFB when `fp32_dest_acc_en=true`. The first port attempt didn't supply any entries (the legacy kernels work fine without thinking about it) — half of the test failures were of this shape. The Default choice (`UnpackToDestMode::Default`) is right for any kernel that uses FPU operations (transpose, mul_tiles, copy_tile, sqrt_tile) on the DFB; non-default is only meaningful when the kernel never touches the DFB via FPU.

Sites for reduction/generic:
- `IN_DFB` when input is Float32 (all reduce factories + welford).
- `SCALER_DFB` when scaler is Float32 (Reduce factories — derived from input dtype).
- `ACC_DFB`/`INEG_DFB` when output is Float32 and negate=true (Reduce factories).
- `VAR_DFB` when fp32_dest_acc_en=true (Welford W, scratch is Float32).
- `COMBINED_DFB` always (Welford HW — always Float32).

Tagged "API: ergonomics — recurring port-time stumble." Three knobs would help: (a) infer Default when not specified and the kernel never uses non-default; (b) at minimum, name the binding's accessor_name in the diagnostic instead of just the dfb_spec_name so the porter can locate the call site faster; (c) consider a per-DFB ergonomic default (Default for FPU-using kernels, surfaced as a CTA-like attribute on the binding itself). Cite: `tt_metal/impl/metal2_host_api/program_spec.cpp:830`.

### Op: TTNN compute_program_hash needs logical_shape to be Metal 2.0-safe

`ReduceDeviceOperation::compute_program_hash` (in `reduce_op_device_operation.cpp`) included `padded_shape()` and `tensor_spec().tile()` but **not** `logical_shape()`. With the legacy ProgramDescriptor path, the kernel reads the tensor address as an RTA — different logical shapes with the same padded shape work fine on cache hit because the RTA gets re-set. With Metal 2.0 TensorBinding, the binding-runtime validator (`program_run_params.cpp:56`) asserts TensorSpec equality — and TensorSpec's `operator==` includes logical_shape. So the cache key must include it.

Tagged "Op: Metal 2.0 brings stricter cache semantics." Worth a doc callout in the migration guide or recipe: "When porting a legacy ProgramDescriptor op to Metal 2.0, double-check that compute_program_hash includes every field that's part of TensorSpec equality (`logical_shape`, `padded_shape`, `memory_config`, `dtype`, `tile`). The legacy hash may have been undercomplete." Welford's hash already had this; the gap was just in the Reduce variant.

### Op: intra-tensix DFB invariant is not in the migration guide

The host validator at `tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:638` requires `DataflowBufferSpec::disable_implicit_sync = true` for any self-loop DFB (a DFB whose producer and consumer are the same kernel — "intra-tensix"). The current `dataflow_buffer_spec.hpp` comment on `disable_implicit_sync` says "Implicit sync is handled via ISR (available on Gen2 only). Disabling may be useful in niche cases for fine tuning performance or performance debug" — but doesn't mention that it's *required* for the intra-tensix case.

Three sites in this op: ACC_DFB/INEG_DFB on the reduce-negate paths (all three Reduce factories) and VAR_DFB/SCALED_DFB on the Welford W variant. The first port attempt got the test crash `Intra-tensix DFBs do not support implicit sync (ISR-based credits)` and had to circle back.

Tagged "Doc: spec_hpp comment on `disable_implicit_sync` is misleading + migration-guide entry for self-loop DFBs needs the host-side invariant." Cite: `dataflow_buffer_spec.hpp:106-108`.

### Cross-op kernel: writer_unary_interleaved_start_id.cpp ↔ writer_unary_sharded.cpp forks

Forked both shared writers to `_metal2`-suffixed copies in their respective sibling directories:

- `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id_metal2.cpp` (next to the legacy copy)
- `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded_metal2.cpp` (next to the legacy copy)

The forks carry the sanctioned Metal 2.0 swaps: `get_arg(args::name)` for arg retrieval, `DataflowBuffer cb(dfb::out_dfb)` for the CB id, `TensorAccessor(ta::output)` for the writer's tensor handle. The legacy copies remain in place for unmigrated consumers.

**Sunset checklist** (delete the `_metal2` copies when the last unmigrated consumer ports):

- `writer_unary_interleaved_start_id.cpp` consumers (legacy): every eltwise unary op factory, eltwise binary, slice writer, prod, accumulation, etc. Long list — counts as bulk-port-wave coordination, not a single follow-up PR.
- `writer_unary_sharded.cpp` consumers (legacy): a smaller set under `data_movement/sharded/` and adjacent. Sunset earlier than the interleaved case.

Each sunset is a coordinated PR; this port report flags the touches as the load-bearing record.

### Cross-op kernel: kernel_lib `prepare_reduce_scaler` accepts `dfb::name` directly via implicit conversion

The reader kernels call `dataflow_kernel_lib::prepare_reduce_scaler<cb_id, REDUCE_OP, REDUCE_DIM>(scaler_f)` where `cb_id` is a non-type template parameter expecting `uint32_t`. The Metal 2.0 form `dataflow_kernel_lib::prepare_reduce_scaler<dfb::scaler_dfb, REDUCE_OP, REDUCE_DIM>(scaler_f)` compiles and works because `DFBAccessor::operator uint32_t()` is `constexpr` — confirmed via [Pattern: Pass DFB handles directly to LLKs and kernel-lib helpers, Template-parameter position](../../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers).

No handoff needed; recording for completeness because this is a non-obvious working case (a porter might worry that template-arg position needs special handling, but it doesn't).

## Successes

### `Pattern: Conditional / optional DFB bindings` — caught a near-miss on Welford W scaled_dfb

The initial port had `scaled_dfb` bound only when `do_scale == true` (following the legacy code's conditional CB creation). The kernel's `if constexpr (do_scale) { ... cb_scaled_obj.wait_front(...) ... }` block compiled when `do_scale=true` but had no chance to verify against `do_scale=false`. Mid-port re-read of [Pattern: Conditional / optional DFB bindings](../../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md#pattern-conditional--optional-dfb-bindings) showed the correct shape: **bind unconditionally on the host; gate uses with `if constexpr`**. Flipping the host to unconditional-bind made the kernel's existing top-level `DataflowBuffer cb_scaled_obj(dfb::scaled_dfb)` declaration compile cleanly in both modes. The L1 cost (~1 input-tile per core when `do_scale=false`) is acknowledged in the catalog as the documented temporary cost.

The "Why this shape" callout in the catalog entry was the load-bearing piece — explained exactly why `if constexpr (false) { dfb::scaled_dfb }` doesn't work even though the discarded branch shouldn't matter. Without that, the kernel-side `if constexpr` looked like it was already sufficient (it isn't — the wrapper-construction line at top level still needs the dfb symbol to exist).

### `Pattern: Self-loop DFB binding` — shared accessor name kept self-loops readable

ACC_DFB / INEG_DFB / VAR_DFB / SCALED_DFB are all self-loops (compute kernel binds the same DFB as both PRODUCER and CONSUMER). The pattern catalog's [Self-loop DFB binding](../../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md#pattern-self-loop-dfb-binding) recommends shared `local_accessor_name` so the kernel uses a single `DataflowBuffer` wrapper. Earlier reference code (borys-fixup branch) used distinct accessor names (`acc_w`/`acc_r`); the catalog correctly says these are equivalent but shared name is more natural. The shared-name form let the kernels keep one wrapper per buffer instead of two — kept the diffs minimal.

### `Anti-pattern: Demoting per-group CTA to RTA` was the right framing for Reduce H/W

Both Reduce H and W have a per-core-group compute multiplicity (the legacy code creates `compute_desc_g1` and `compute_desc_g2` differing only on `Wt` / `Ht` CTA). The anti-pattern entry explicitly forbids collapsing these into one KernelSpec with the dimension demoted to an RTA. The port preserved the multiplicity: two `KernelSpec`s of the same source, two `WorkUnitSpec`s, sharing `IN_DFB`/`SCALER_DFB`/`OUT_DFB` via multi-bindings.

### Recipe scope-boundary preamble caught me from over-modifying

After the audit's question about `REDUCE_POST_MUL`, I considered converting the `#ifdef REDUCE_POST_MUL` blocks in the compute kernels to `if constexpr (use_post_mul)`. Re-reading the [scope boundary](../../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md) preamble: "If the legacy kernel has a particular control flow, the port preserves it (modulo the sanctioned swaps)." The `#ifdef` is a control-flow gate (not a DFB-reference gate), so it stays — `KernelSpec::compiler_options.defines` carries `REDUCE_POST_MUL` when `use_post_mul=true`. Kept the kernel diff smaller than it could have been.

### Cross-op fork-with-`_metal2`-suffix path worked cleanly

The recipe restored fork-with-`_metal2`-suffix as the supported path for cross-op kernels (after the second test-drive surfaced the in-place-modification approach as too risky during the bulk-port window). For the two cross-op writers (`writer_unary_interleaved_start_id.cpp` and `writer_unary_sharded.cpp`), the fork path was straightforward — copy the kernel verbatim, swap the three sanctioned things (arg retrieval, DFB wrapper, TensorAccessor), preserve the `OUT_SHARDED` and `BACKWARDS` `#ifdef`s. No coordination with unmigrated sibling-op consumers needed.

### `tile_format_metadata` operational rule was followed mechanically

Every `DataflowBufferSpec` declaration in the port sets `.tile_format_metadata = tensor.tensor_spec().tile()`. For standard 32x32 tile inputs/outputs this is observably equivalent to leaving the field unset, but the [operational rule for porters](../../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_migration_guide.md#dataflowbufferspec) says "copy this field from the legacy CB's `format_descriptors[i].tile`" — followed mechanically, no thinking required.

## Friction

### Gap: testing/build environment setup not in the recipe

The recipe's verification step says:

```bash
cmake --build build_Release --target ttnncpp -j 8
pytest tests/ttnn/unit_tests/operations/<op_family>/ -x -v
```

But the build had to be bootstrapped from scratch in the worktree (no `build_Release` directory existed yet, `git submodule update --init --recursive` was needed to download dependencies, and the `python_env` from the parent tt-metal didn't pick up the worktree's freshly-built `_ttnn.so` and `_ttnncpp.so` without explicit `PYTHONPATH` + copying the `.so` files into the worktree's `ttnn/ttnn/` directory and setting `LD_LIBRARY_PATH` for the runtime to find `libtt_metal.so` etc.). Took about an hour of trial and error before I had a working test invocation.

Suggested doc improvement: a "First-time-in-this-worktree setup" callout in the verification section listing the env vars (`TT_METAL_HOME`, `ARCH_NAME`, `PYTHONPATH`, `LD_LIBRARY_PATH`), the `.so`-copy step (or symlink), and the build target ordering (`ttnncpp` for the shared library + `_ttnn.so` for the Python module). The recipe's verification section currently assumes the shell is already in a working state — it isn't, for a fresh worktree.

### Gap: ComputeConfiguration::unpack_to_dest_mode is unnamed in the migration guide

The migration guide's `KernelSpec` and `DataflowBufferSpec` sections don't mention `unpack_to_dest_mode` at all. A reader of the migration guide who's never seen the validator's complaint won't know they need to think about it. The patterns catalog also doesn't have an entry for this. First port attempts will hit the validator error and circle back; the doc improvement is to surface the rule up-front, ideally with a short rule-of-thumb: "If your compute kernel consumes a Float32-formatted DFB and `fp32_dest_acc_en=true`, declare `ComputeConfiguration::unpack_to_dest_mode = {{<dfb_id>, Default}}`. Default = unpack via SrcA/B (the ~19-bit precision floor that FPU operations use); UnpackToDestFp32 is only for SFPU-only direct-to-Dest reads." A worked example on Welford's `VAR_DFB` or `COMBINED_DFB` would make this concrete.

Tagged Gap because the rule is silent in the migration guide and forced a port-time backtrack.

### Gap: `disable_implicit_sync = true` invariant for self-loop DFBs

Same shape as the previous gap. The pattern catalog's [Self-loop DFB binding](../../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md#pattern-self-loop-dfb-binding) entry shows the correct binding shape (two `DFBBinding`s, PRODUCER + CONSUMER, shared accessor name) but doesn't mention the host-side spec invariant `.disable_implicit_sync = true`. The first port hit the runtime crash. Patches needed in both the catalog entry (recommend adding "Set `disable_implicit_sync = true` on the `DataflowBufferSpec` for self-loop DFBs") and ideally the migration guide's `DataflowBufferSpec` section ("Self-loop DFBs are intra-tensix and don't support implicit sync — set `disable_implicit_sync = true`. The dataflow_buffer_spec.hpp comment on the field is misleading; it's not 'niche', it's required for intra-tensix.").

### Gap: TensorSpec equality / cache hash interaction (Op-specific surprise)

The Metal 2.0 binding system enforces strict TensorSpec equality at runtime (`program_run_params.cpp:56`), but the legacy compute_program_hash may not include every field of TensorSpec. For ops where the legacy hash was undercomplete (Reduce here, omitting `logical_shape`), the cache-hit path can pass a mismatched-spec tensor and trip the runtime assertion.

This isn't a generic Metal-2.0-vs-legacy story — it's specific to ops whose legacy hash was less strict than TensorSpec equality (because the legacy path used RTA-supplied buffer addresses, which don't care about logical shape). But it's the kind of gotcha worth surfacing in the migration guide:

> **When porting, audit `compute_program_hash` against TensorSpec equality.** Metal 2.0 binding-runtime enforces `runtime_spec == binding_spec`, so every TensorSpec-relevant field must appear in the cache hash. Look at the legacy hash: if it omits any of `logical_shape`, `padded_shape`, `memory_config`, `dtype`, or `tile`, add them. The legacy `tensor.buffer()->address()` RTA pattern made the hash undercompleteness invisible; Metal 2.0 makes it surface.

### Confusion: bit-cast post_mul_scaler_bits as `uint32_t`-typed CTA but kernel-side treats it as opaque bit-cast

The Reduce compute kernels carry `post_mul_scaler_bits` as a CTA — packed-fp32 bits. The legacy code did `get_compile_time_arg_val(3)` and used the result as the second arg to `mul_unary_tile(dst_idx, post_mul_scaler_bits)`. After the port, `constexpr auto post_mul_scaler_bits = get_arg(args::post_mul_scaler_bits)` gives a `uint32_t`. The same opaque-bit pattern survives. This was an "is the bit-cast OK?" anxiety during construction; it turned out fine but worth noting as a small confusion gateway. The named-CTA mechanism passes the value type-erased (uint32_t in / uint32_t out), so any bit-cast-style use is preserved verbatim.

### Confusion: the `src_addr` RTA on the Welford H/HW + Reduce H readers

The legacy column-partitioned reader and writer kernels read `src_addr` / `dst_addr` as their first RTA. Metal 2.0's TensorBinding auto-injects the buffer base address into a hidden host-managed slot — the kernel doesn't need `src_addr` at all. I kept the `src_addr` RTA slot with a `[[maybe_unused]]` annotation, set to `0u` from the host. This works but is dead weight (a runtime arg that's written and never read). The cleaner Metal 2.0 shape is to drop the slot entirely, which would slightly streamline the RTA layout — but the cost is bigger: I'd be touching the ordering of all per-core RTA emission, which makes the diff harder to review for "did I get the column indexing right?". Kept it as-is; flag for cleanup later.

(Worth a doc note in the migration guide: "When porting, the buffer-address RTA on the legacy reader/writer can be omitted entirely — `TensorBinding` auto-injects the address through a host-managed slot. If you keep the slot for diff-minimization, mark it `[[maybe_unused]]` and set its host value to a placeholder.")

## Open items for downstream

### Cross-op kernel touches (sunset coordination)

Two `_metal2`-suffixed forks were created. Per the recipe and patterns catalog, these are the coordination signal for the next sibling-op port:

| Kernel (legacy path) | Fork path | Remaining unmigrated consumers |
|---|---|---|
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | `..._metal2.cpp` (sibling file) | All other ops that reference the legacy writer — bulk-port-wave list. |
| `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp` | `..._metal2.cpp` (sibling file) | Other sharded-output ops. |

Sunset trigger: when the last unmigrated consumer ports, delete the `_metal2` copy.

### Device 2.0 holdover sites in the legacy writer fork

The forked `writer_unary_interleaved_start_id_metal2.cpp` uses `cb.get_tile_size()` (DFB wrapper method) instead of the legacy `get_local_cb_interface(cb_id_out).fifo_page_size`. This is observably equivalent for tile-format DFBs (which is all reduction outputs), but **non-equivalent for ROW_MAJOR layouts** — `fifo_page_size` is the runtime-configured page size, `get_tile_size()` returns the format-descriptor-derived per-tile byte count. The reduction op validation requires TILE layout, so the substitution is safe for this op's consumers. If a future ROW_MAJOR consumer co-migrates onto this writer fork, the substitution will silently mis-compute the per-page byte count.

Suggested mitigation: rename the fork to something op-family-specific (e.g. `writer_unary_interleaved_start_id_reduce.cpp`) so a future ROW_MAJOR consumer of the legacy copy is forced to either (a) keep using the legacy or (b) port their own forked writer.

Tagged Open item for downstream coordination — not a port-time fix.

### Per-op carry-over: ops using `writer_unary_interleaved_start_id.cpp`

The Metal 2.0 ports of other ops that use the same writer (eltwise unary, ternary, binary_ng, sliced, etc.) can reuse the `_metal2` fork directly. The fork's `args::num_pages` / `args::start_id` / `dfb::out_dfb` / `ta::output` naming is conventional. Listing this as carry-over rather than a port-time concern.

### Test coverage: width-sharded H-reduce path

The W and HW factories are exercised heavily by `test_reduction.py` (636 test cases). The H factory's width-sharded path is exercised by `test_reduction_h_interleaved.py` (which I ran and passed), but the bulk of H-reduce tests use the interleaved path. Surfacing this so the next porter (or doc maintainer) knows the borrowed-memory-DFB code path is the lighter-tested of the two.

### Doc-evolution suggestions

Three doc improvements emerged that don't fit cleanly into a Gap entry:

- **Migration guide: spec-validator quick-reference table.** A one-pager listing the validator checks that bite porters: TensorSpec equality, intra-tensix `disable_implicit_sync`, FP32-consumer `unpack_to_dest_mode`, DFB producer/consumer ≥1 each, KernelSpec→WorkUnitSpec membership. Each row links to the relevant section of the migration guide or patterns catalog. A porter hitting any of these gets a quick visual map of which validator they tripped and why.

- **Patterns catalog: combined `intra-tensix + Float32` recipe.** Welford W's `VAR_DFB` is both self-loop (intra-tensix → disable implicit sync) AND Float32 (fp32_dest_acc_en=true → needs unpack_to_dest_mode). Currently each invariant has its own catalog entry but the combination isn't called out. A short "Self-loop FP32 DFB on a compute kernel" recipe entry would chain the two requirements together. Welford W is the worked example.

- **Recipe: kernel-side preserve `[[maybe_unused]]` for omittable RTAs.** When a legacy RTA is subsumed by a TensorBinding (the buffer-address case), the kernel-side RTA slot can either be (a) deleted entirely (cleaner long-term, but disturbs every other RTA's indexing in the legacy code) or (b) retained as `[[maybe_unused]]` with a host-side dummy value (preserves layout, easier to review diffs against). Worth recording as a tradeoff porters can choose; no need for a doctrinal answer.

---

Counts for the final summary:
- Handoff points: 5
- Successes: 6
- Friction: 5 (2 Gaps + 1 Op-specific Gap + 2 Confusions)
- Open items for downstream: 5

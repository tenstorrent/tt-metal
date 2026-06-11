# Metal 2.0 Port Report — layernorm (`LayerNormDeviceOperation`)

**Status: IN PROGRESS — host fully ported & C++-GREEN; non-welford/non-large/tile paths VALIDATED ON
DEVICE; remaining kernels (welford/large/rm) pending.**
Porting the multi-core (interleaved) `LayerNormMultiCoreProgramFactory` to `ProgramSpecFactoryConcept`.
The sharded factory stays on the legacy `ProgramDescriptor` concept (mixed-concept `program_factory_t`
variant is legal; disjoint kernels).

**On-device validation (Wormhole B0, after the card was reset):**
- `test_layer_norm[bf16, use_welford=False, w=64, h=32]` — **PASS** (base path: tile input, no
  gamma/beta/residual; exercises host spec build → framework validation → kernel JIT → device → PCC).
- `test_layer_norm_with_weight_bias_and_residual_input[bf16, use_welford=False, w=64, h=32]` — **PASS**
  (fused-pre-add + gamma + beta; validates the `#ifdef FUSE_PRE_ADD/FUSE_GAMMA/FUSE_BETA`
  conditional-binding gating and the tensor bindings end-to-end).

Two spec-validation fixes were needed to get there (both genuine Metal-2.0-vs-legacy semantics, see
Friction): gating the `cb_accumulate` unpack entry on its binding condition, and deriving
`unpack_to_dest_mode` for every consumed Float32 DFB (not just the special ones). These are the only
two iterations the base/fused paths required after the C++ build went green.

> Replaces the prior dogfood's grounded-stop report (preserved in git history). That pass stopped on
> one-pass *size* budget after producing the inventory + blueprint in `METAL2_PORT_PLAN.md`. This pass
> executed the port from that blueprint as an interactive primary session.

## Where this pass got to

**Done and verified (C++ host build GREEN — `ttnncpp` + `unit_tests_ttnn` link clean):**
- `device/layernorm_op_multi_core.cpp` — `create_descriptor` → `create_program_spec` returning
  `ProgramArtifacts`. The shared `cb_named_args` table dissolved into per-KernelSpec `dfb_bindings`;
  21 DataflowBufferSpecs built conditionally (incl. welford-fp32 aliased DFBs via `alias_with`,
  borrowed recip LUT via `borrowed_from`); TensorParameters + TensorBindings replace the
  buffer-address RTAs and `TensorAccessorArgs` plumbing; named CTAs/RTAs; per-node `KernelRunArgs`;
  `WorkUnitSpec`; `ProgramRunArgs`. `core_range_set` parameter dropped, production default inlined.
- `device/layernorm_device_operation.hpp` — multi-core factory now declares `create_program_spec`.
- `layernorm_nanobind.cpp` — multi-core `create_descriptor` pybind hook removed (see Handoff points).

**Kernels converted AND device-validated (the two passing tests above JIT-compile and run these):**
- `device/kernels/dataflow/layernorm_dataflow_utils.h` — `CircularBuffer&` → `DataflowBuffer&`;
  `use<CircularBuffer::AddrSelector::READ_PTR>` dropped (bare DFB source is read-ptr by construction).
- `device/kernels/dataflow/reader_unary_interleaved_ln.cpp` — full conversion (named args, `ta::`/`dfb::`,
  `#ifdef WELFORD_FP32_ALIAS`-gated alias). Exercised on the base + fused/gamma/beta paths.
- `device/kernels/dataflow/writer_unary_interleaved_start_id_blocked.cpp` — full conversion.
- `device/kernels/compute/layernorm.cpp` — full conversion incl. the `#ifdef FUSE_*`/`RMSNORM`/
  `TILIZE_IN`/`UNTILIZE_OUT` gating of every conditionally-bound DFB reference.

**Converted but NOT yet device-validated (no test path hit them yet):**
- `device/kernels/dataflow/writer_unary_interleaved_start_id_blocked_rm_output.cpp` (RM writer; the
  row-major *input* paths also need the compute-side `TILIZE_IN`/`UNTILIZE_OUT` which `layernorm.cpp`
  has, but a row-major test wasn't run this session).

**Remaining kernel conversions (mechanical, follow the validated pattern — see PLAN checklist):**
- compute: `layernorm_welford.cpp`, `layernorm_large_tensor.cpp`, `layernorm_large_tensor_welford.cpp`.
- readers: `reader_unary_interleaved_ln_large_tensor.cpp`, `..._large_tensor_welford.cpp`, `..._rm_gb.cpp`.
- `layernorm_compute_utils.h` needs **no change** (its helpers take `uint32_t` CB ids; `dfb::name`
  converts implicitly).

## Environment / workspace-bootstrap notes (cost real time; not the port)

The fresh worktree was not a ready-to-build/test environment. In order: (1) **git submodules were
uninitialized** — cmake configure failed until `git submodule update --init --recursive`; (2) the
**first clean build** (`build_metal.sh --build-tests`) is required and slow; (3) running pytest needs
the **Python bindings** (`_ttnn.so`) which a `cmake --build --target ttnncpp` does NOT produce — a full
`cmake --build` is required, and it lands `_ttnn.so` in `build_Release/ttnn/` but does NOT copy it into
the package dir, so `ttnn/ttnn/_ttnn.so` had to be symlinked; (4) `import ttnn` then needed `tracy`
(`tools/`) and `graphviz` etc. — only present in the **main checkout's `python_env` venv**, so tests
run as `PYTHONPATH=<wt>/ttnn:<wt>:<wt>/tools <main>/python_env/bin/python3 -m pytest …` (the recipe's
"PYTHONPATH=$(pwd)" assumes an editable install that a bare worktree doesn't have). The device was also
**hung (`deadc0de`)** at first and needed a reset (done by the invoker). **Suggested:** the workspace-setup
doc should cover the worktree case explicitly (submodules, full build for `_ttnn.so` + the package-dir
symlink, the venv to borrow, and the composite PYTHONPATH).

## Friction (captured live)

### Gaps
- **No in-tree reference port.** The only landed `create_program_spec` example (`accumulation`/`ema`) is
  on a *sibling* branch (`akertesz/porting-experiment-accumulation-jun10`), reachable via `git show`.
  Anchoring on it was high-leverage; the recipe's "optional reference port" note should mention it may
  not be on the porter's branch and give the `git show <branch>:<path>` form.
- **`CircularBuffer`-specific NoC helpers have no `DataflowBuffer` equivalent — "object swap, methods
  unchanged" undersells this.** `CircularBuffer` carries `AddrSelector` / `CircularBufferView` /
  `use<AddrSelector::READ_PTR>(cb)` (`circular_buffer.h:25,156-168`) that do **not** exist for
  `DataflowBuffer`. `layernorm_dataflow_utils.h::write_row_major_block_from_cb` used
  `noc.async_write(use<…READ_PTR>(cb_out_rm), …)`. The correct conversion *drops* the wrapper —
  `noc_traits_t<DataflowBuffer>::src_addr` is `get_read_ptr()` (`dataflow_buffer.h:188-194`), so a bare
  DFB as the `async_write` source is read-ptr-sourced. Clean once known, but not a literal method swap.
- **Conditional DFB binding forces a preprocessor define where legacy used only a CTA.** The compute
  kernels gate gamma/beta/fusion via the `do_gamma`/`do_beta` *CTAs* (`if constexpr`). But `cb_gamma`/
  `cb_beta`/`cb_fusion` are conditionally-bound DFBs, so their `dfb::<name>` references must be removed
  by the **preprocessor** (an `if constexpr(false)` branch is still name-looked-up). The port therefore
  emits `FUSE_GAMMA`/`FUSE_BETA` *defines* to the compute kernel (legacy emitted them only to the
  reader) and converts the relevant `if constexpr(do_gamma)` blocks to `#ifdef FUSE_GAMMA`. Same logic
  applies to the welford-fp32 aliases (new `WELFORD_FP32_ALIAS` / `WELFORD_STATE_FP32_ALIAS` defines).
  This is the [Conditional / optional DFB bindings] pattern, but the recipe/catalog frame it around DFBs
  that were *already* `#ifdef`-gated in legacy; the gamma/beta case is a CTA→define *promotion* the docs
  don't call out. **Suggested:** catalog note — "a conditionally-bound DFB whose legacy gate was a CTA
  (`if constexpr`) must be promoted to a `#define` gate."
- **Path-dependent DFB producer (cb_in in the ROW_MAJOR path).** `cb_in`'s producer is the *reader* on
  the TILE path but *compute* (via `tilize`) on the ROW_MAJOR path (the reader fills `cb_in_rm` instead).
  A 1:1 "reader produces cb_in" binding is wrong for RM. The port branches the binding on
  `input_is_row_major`. The role-map inventory must be taken *per kernel-source-selection path*, not once
  for the op — the recipe's runtime-source-selection note implies this but doesn't stress the
  producer-can-move consequence.

- **`unpack_to_dest_mode` entries must name a DFB the compute kernel actually binds.** Legacy set
  `unpack_to_dest_mode[c_26] = Fp32` whenever `float32_reduction`, regardless of whether CB 26
  (`cb_accumulate`) was configured — harmless for an array indexed by CB id. Metal 2.0 validates
  (`program_spec.cpp:799`, `bound_dfbs.contains(dfb_name)`) that every `unpack_to_dest_mode` key is a
  DFB bound to that kernel, so the entry must be gated on the same condition as the binding
  (`large_tensor_needed && !use_welford`). Caught on the first base-path device test. **Suggested:**
  catalog/recipe note — "unpack_to_dest_mode (and any per-DFB compute config) must be gated on the
  DFB's binding condition, not just the numerical condition the legacy array used."

### Confusion
- The dogfood role-map (delegated) mislabeled a couple of compute CTA slots (e.g. welford `TILE_SIZE` vs
  `tile_width`); the *legacy host* `compute_args` order is authoritative. Cross-checking the delegated
  inventory against the host source caught it. Worth a recipe note: trust the host emission order over a
  reconstructed kernel-side reading for CTA positions.

## Handoff points
1. **Removed pybind surface (API surface: removed entry point).** `LayerNormMultiCoreProgramFactory`'s
   `create_descriptor(... core_range_set)` hook was deleted from
   `ttnn/cpp/ttnn/operations/normalization/layernorm/layernorm_nanobind.cpp` (was ~:322). It returned a
   `ProgramDescriptor`, which the Metal 2.0 factory no longer produces. Downstream Python callers of
   `LayerNormMultiCoreProgramFactory.create_descriptor` must migrate. The **sharded** factory's
   `create_descriptor` hook (~:363) is intentionally **kept** (sharded stays legacy). Owner: layernorm op
   owner / TTNN. (Per invoker: TTNN will address the underlying problem on top of this work.)
2. **Device 2.0 `get_tile_size(cb_id)` holdovers — NOT absorbed.** The ported kernels keep the free-function
   `get_tile_size(dfb::name)` form (crosses the boundary via the `DFBAccessor → uint32_t` implicit
   conversion). The member-form cleanup is the Device 2.0 track's, per the audit. Owner: Device 2.0 track.
3. **Dead reader RTA dropped.** The legacy `packed_one_value` reader RTA (documented unused) is gone with
   the RTA→named conversion. (The legacy kernel comment already flagged it as unused.) Owner: op owner —
   informational only.

4. **CRITICAL — the reciprocal LUT should be a plain `TensorBinding` (Case 2), NOT a `borrowed_from`
   DFB.** The audit classified the welford reciprocal LUT (legacy CB 25) as "clean — borrowed-memory
   DFB" and "No Case 2 bindings," and the port followed that onto `DataflowBufferSpec::borrowed_from`.
   Both classifications are wrong, and tracing the provenance gives a clean fix:
   - **Provenance: it is an op INPUT tensor.** `tensor_args.recip_tensor`
     (`layernorm_device_operation_types.hpp:34`), created caller-side by
     `ttnn.create_layer_norm_reciprocals(device, core_range_set, w)` and passed into `ttnn.layer_norm`.
     It is **not op-owned** (the factory allocates nothing) and **not device-generated** — a host-computed,
     L1-resident, per-core read-only LUT. The port *already* declares it as a `TensorParameter`
     (`PARAM_RECIP`, host line ~859) with a `TensorArgument` (~940), because `borrowed_from` references
     it. The borrowed DFB built on top of that parameter is the redundant, wrong step.
   - **It is not a dataflow buffer in any sense** — verified across all three welford computes
     (`layernorm_welford.cpp:106`, `layernorm_large_tensor_welford.cpp:398`,
     `layernorm_sharded_welford.cpp:229`): every access is `get_pointer_to_cb_data(cb_reciprocals, 0)`
     then raw dereference. **No `wait_front`/`pop_front`/`reserve_back`/`push_back` anywhere, and no
     kernel produces it.** Reading an input tensor by raw base pointer is precisely a **Case 2** access —
     so the audit's "No Case 2 bindings" is wrong for this binding.
   - **Right model (no Buffer binding needed):** drop the borrowed DFB; bind `PARAM_RECIP` to the compute
     kernel as a `TensorBinding` (`ta::recip`); kernel-side replace `get_pointer_to_cb_data(cb_reciprocals)`
     with `TensorAccessor(ta::recip)` + `get_bank_base_address(...)` (the sanctioned Case-2 bridge,
     kernel-side whitelist rule 5), then index the LUT. This eliminates the borrowed DFB, the
     producer-invariant stress, AND the need for a Buffer-binding type — it uses only TensorBinding +
     the existing Case-2 bridge. (A **Buffer** binding would be right only for op-*owned* scratch; this
     is an input tensor, so TensorBinding is the correct home. Borys's "no buffers in ops" preference is
     satisfied by construction.)
   - **What this port does instead, and why it may not even validate:** it binds the borrowed DFB
     **consumer-only** (no producer — there is none to invent), stressing the DFB endpoint invariant
     ([`dataflow_buffer_spec.hpp:40-48`](../../../../../../tt_metal/api/tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp)).
     **Unverified this session** (the welford path that binds it wasn't run): if the validator enforces
     ≥1 producer for borrowed DFBs, the current port's welford path fails here — and the fix is the
     TensorBinding rewrite above, not a producer hack.
   - **One detail to confirm in the rewrite:** that `get_bank_base_address` on this L1, per-core LUT
     yields the local shard's base pointer the kernel expects (the legacy CB-borrow got it from the
     per-core borrowed buffer). Verifiable; expected to hold for an L1 tensor over the compute grid.
   - **Audit/recipe action:** a read-only **input** LUT read by raw pointer is a **Case 2 TensorBinding**,
     not a "clean borrowed DFB." The audit's borrowed-memory-DFB classification should be reserved for
     genuine FIFO-on-borrowed-memory cases (e.g. sharded input/output the kernel `wait_front`s), not
     pointer-addressed input LUTs. Owner: audit/recipe maintainers + Metal 2.0 framework.

## Successes
- **The accumulation reference port + the API headers were sufficient to author the entire host in one
  pass; it compiled with only 5 trivial `-Werror` unused-variable fixes** (the legacy buffer-address
  locals, now dead because addresses flow through TensorBindings). The patterns catalog's Aliased-DFBs vs
  Same-FIFO distinction and the Conditional-bindings pattern were exactly the lenses needed for the
  welford-fp32 aliasing and the gamma/beta gating.
- **The atomic-unit framing held but the C++/kernel split made it tractable:** because kernel sources are
  referenced by path and JIT-compiled at runtime, the *host* compiles and validates independently of the
  kernels — so the large host rewrite could be checkpointed green before the kernel slog, and kernels can
  be converted/validated path-by-path rather than all-or-nothing. Worth adding to the recipe: the C++
  build validates the host; kernel correctness is gated at test (device) time.

## Open items for downstream
- **Cross-op kernel touches:** none — all sources + headers are layernorm-owned.
- **Remaining kernel conversions** (above) follow the validated pattern; the welford compute kernels carry
  the subtle aliasing (`cb_x_welford` producer is reader on non-fused-TILE, compute on fused or RM;
  `cb_ex_welford`/`cb_ex2_welford` self-loops under `WELFORD_STATE_FP32_ALIAS`).
- **Reciprocal LUT should be a Buffer binding, not a DFB** — see **Handoff point 4** (elevated to a
  critical finding). It binds consumer-only today (a deliberate DFB abuse); the welford path is the
  first to exercise it and will answer whether the validator even accepts a producer-less borrowed DFB.
- **RM-input + welford + fp32-alias** is a rare combo whose welford-compute `TILIZE_IN` support is unverified;
  the host binds `cb_x_welford` as a compute self-loop there. Confirm against the welford compute kernel.
- **Sharded factory** remains a separate, larger port.

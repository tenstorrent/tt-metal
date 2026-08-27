# Port Report — rotary_embedding_llama_fused_qk

## Outcome

COMPLETE — built clean and tests pass, node-for-node identical to the pre-port baseline (16/16).

## Test evidence

Both runs: the full file `tests/tt_eager/python_api_testing/unit_testing/misc/test_rotary_embedding_llama_fused_qk.py` with `-q`, no `-k` filter, no marker selection — 16 nodes (12× `test_rotary_embedding_llama_fused_qk` over shape triples `8-1-128`, `71-32-64`, `8-1-256` × `decode_1/8/16/32`; 4× `test_rotary_embedding_llama_fused_qk_with_program_cache` over `8-1-128` × `decode_1/8/16/32`).

- **Post-port:** 16 passed, 0 failed, 0 skipped in 39.80s. Run from this worktree; `ttnn._ttnn.__file__` verified to resolve to this worktree's freshly installed `.so` before the run; `~/.cache/tt-metal-cache` purged first (JIT stats confirm 0/98 cache hits — every kernel built fresh). Metal 2.0 validation scaffold active throughout: 96 `METAL2_CHECKS_FORCED` markers (`program_spec.cpp`, `program_run_args.cpp`).
- **Pre-port baseline:** 16 passed, 0 failed, 0 skipped in 39.21s. Run by the coordinator from the pristine main checkout `/localdev/vsuresh/tt-metal` (this op's files there are untouched legacy; that tree's only branch commit touches `nlp_concat_heads` only), with its own JIT-cache purge and `ttnn._ttnn.__file__` verification.
- **Sequencing note:** the baseline was executed *after* the port, from the separate pristine tree — this worktree carried kernel edits before any baseline could run, and kernels are JIT-built from the working tree at test time, so a baseline from this tree was impossible post-edit. The JIT-cache purge between the two runs keeps the eras from cross-contaminating.

## Provenance

- **Recipe docs (this port):** version cannot be pinned — the metal_2.0 docs directory is untracked in this checkout (`git log -1 --format='%h %cs %s' -- docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/` prints nothing).
- **Audit docs (inherited):** version cannot be pinned — the metal_2.0 docs directory is untracked in this checkout (`git log` prints nothing for its path). *(copied from METAL2_PORT_BRIEF.md)*

## TTNN ProgramFactory

### Concept realized
`ProgramSpecFactoryConcept`, as the audit chose. Single factory; `create_descriptor` → `create_program_artifacts` swap inside the existing `program_factory_t` struct (`RotaryEmbeddingLlamaFusedQKProgramFactory`). No device-op-class restructuring needed.

### Device-op-class edits
- Pybind entry points removed: none (the nanobind file binds only the public composite op via `ttnn::bind_function`; untouched).
- Custom `compute_program_hash`: none.

### Open items
- See "Open items for downstream".

## Handoff points

none.

## Successes

- **Brief's q/k runtime-mux heads-up resolved without any workaround.** The brief flagged constructing a `DataflowBuffer` object from a runtime-selected index as the port's one non-mechanical kernel spot, with "build two objects and branch" as fallback. The device header already provides the public low-level `DataflowBuffer(uint16_t logical_dfb_id)` constructor (`tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:113`), so the legacy shape (runtime `uint32_t` mux feeding one object construction) ports 1:1 — `dfb::` tokens assigned into `constexpr uint32_t` locals, the `is_q` branch selecting the runtime id, and the in/out objects built from it (`device/kernels/compute/rotary_embedding_llama_sharded.cpp:62-63`). No per-object duplication, which mattered here because both kernels sit within ~4 B of the TRISC2 code-size limit. ("Go to the headers first" from the recipe steered this — the header answered it faster than any precedent hunt would have.)
- **Compiler-options trap fired exactly as documented.** The landed sibling Metal 2.0 factory (`rotary_embedding_llama_sharded_program_factory.cpp`) sets no `opt_level` on its compute `KernelSpec`, so a shape-copy of that reference would have silently dropped the compute kernel from its legacy-resolved O3 to Metal 2.0's O2 default. The recipe's [Compiler options] rule 2 caught it; this port sets `.compiler_options = {.opt_level = KernelBuildOptLevel::O3}` explicitly (factory:255). See Open items for the sibling itself.
- **Style-A "dropped field" check (recipe's Hardware configuration section) applied cleanly.** The legacy factory resolves the full TTNN compute config but copies only `math_fidelity` + `fp32_dest_acc_en` onto the descriptor; per the recipe the port mirrors the subset on a `ComputeGen1Config` and relies on the verified default coincidences (`sfpu_precision_mode = Precise` ⟷ `math_approx_mode = false`; `double_buffer_dest = true` ⟷ `dst_full_sync_en = false`) rather than routing the resolved config through `to_compute_hardware_config` (which would have handed the *caller's* `math_approx_mode=true` / `dst_full_sync_en` back — a silent behavior change; production callers pass `math_approx_mode=true`, which legacy ignored).

## Friction

- The worktree this port ran in lacked initialized submodules; `git submodule update --init --recursive` was needed before any build (workflow note, not a doc gap).
- First build died in third-party Tracy compilation with `ccache: ... No space left on device`: the default ccache dir sits on a 9.4 GB NFS home quota at 96%. Fixed by `CCACHE_DIR=/localdev/vsuresh/.ccache` (bench-specific workflow note).

## Open items for downstream

- **Sibling factories missing explicit compute `opt_level` (silent O3→O2 drop).** All three landed Metal 2.0 factories of the sibling op `rotary_embedding_llama` build compute `KernelSpec`s with no `compiler_options.opt_level` (e.g. `rotary_embedding_llama/device/rotary_embedding_llama_sharded_program_factory.cpp:161-166`), so their compute kernels now build at Metal 2.0's O2 default where legacy resolved O3 — the exact silent perf regression the recipe's Compiler-options section warns about. Not this port's file to fix; flagging for the sibling's owner.
- **Legacy bounding-box CB placement evaporates (deliberate, zero-functional).** Legacy allocated all 10 CBs over `all_cores_bb` (the cos/sin grid's bounding box), which could include "hole" cores belonging to neither q nor k, and configured q-backed CBs on k cores (and vice versa) at addresses where the backing shard has no per-core allocation — never dereferenced, so harmless. Metal 2.0 derives DFB placement from kernel bindings, so post-port every DFB lives exactly on `work_cores = q_cores ∪ k_cores`: hole cores get no DFB configs and no interm-buffer L1. No functional or observable change (holes run nothing and nothing ever read those configs); L1 on hole cores is freed as a side effect of the representation, not an optimization decision by the port.
- **Latent total/page mismatch in the legacy interm CBs is unexpressible (and unrealizable) in Metal 2.0.** Legacy `cos_interm`/`sin_interm` CBs set `.total_size = num_interm_tiles * input_single_tile_size` but `.page_size = cos/sin_single_tile_size` (factory:175-194 pre-port) — consistent only because validate() forces every tensor to bfloat16, making all tile sizes equal. `DataflowBufferSpec` expresses the total as `num_entries * entry_size`, which reproduces today's bytes exactly and makes the latent divergence impossible to restate. If dtypes ever diverge on this op, the DFB sizing derives from the cos/sin tile size; the legacy expression mixed bases and would have been wrong anyway (audit's "Misc anomalies" has the detail).
- **Row-major kernel variant has no in-tree test coverage.** `run_test_row_major_rotary_embedding_llama` (with a `fuse_qk` parameter) exists in `tests/ttnn/nightly/unit_tests/operations/experimental/test_rotary_embedding_llama.py:296` but no test function invokes it, and no other test drives the op with ROW_MAJOR q/k inputs. The row-major kernel (`..._sharded_row_major.cpp`) converted together with the factory (atomic unit) and is verified by compile + review only. Test-coverage note for the op owner.
- **Dead locals kept.** The tiled kernel's `cos_dfb_obj`/`sin_dfb_obj`/`trans_mat_dfb_obj` (constructed, never used — cos/sin/trans_mat are consumed via LLK index calls only) are carried across as `DataflowBuffer` objects for minimal diff, matching legacy. The audit noted they could be dropped (the row-major variant already omits them); left for the op owner as a separate cleanup, given the TRISC2 code-size sensitivity cuts the other way too.
- **Commented-out `has_work` early-return updated in place.** The legacy kernels carry a commented-out early-return (`get_arg_val<uint32_t>(0)`) disabled for TRISC2 code-size reasons; the port re-spelled the dead line in the named-args form (`get_arg(args::has_work)`) so it stays re-enableable — note that re-enabling would also need the host to declare the `has_work` named RTA.

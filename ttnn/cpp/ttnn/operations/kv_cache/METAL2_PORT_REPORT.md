# Metal 2.0 Port Report — kv_cache (UpdateKVCacheOperation)

## Outcome
**PORTED** — both factories (`UpdateCacheMultiCoreProgramFactory`, `FillCacheMultiCoreProgramFactory`)
converted to `CustomProgramSpecFactoryConcept`. Full `test_update_cache.py` suite green on Wormhole
(**666 passed, 576 skipped, 0 failed**), with the Metal 2.0 host-side legality checks force-enabled
throughout (`METAL2_CHECKS_FORCED` present in the run log).

## Provenance
- **Recipe docs (this port):** `63ca139b420 2026-08-24 docs(metal_2.0): a run in flight freezes the kernel sources`
- **Audit docs (inherited):** `63ca139b420 2026-08-24 docs(metal_2.0): a run in flight freezes the kernel sources`

## TTNN ProgramFactory
### Concept realized
`CustomProgramSpecFactoryConcept` (both factories), as the audit chose. Each factory's translated
`override_runtime_arguments` returns a `ProgramRunArgs` and refreshes **exactly** what the ported-from
void override refreshed:
- **update:** `tensor_args` = {cache (from `tensor_return_value`), input}; per-core reader `cache_start_id`;
  per-core writer `cache_start_id`, `Wbytes`, `offset` (tile_update_offset), `batch_read_offset`.
- **fill:** `tensor_args` = {input, dst=cache (from `tensor_return_value`)}; per-core writer `start_id`
  (= cache_start_id). Reader `num_tiles`/`start_id` are shape-derived (hash-covered) → not refreshed,
  matching the legacy override which only re-applied the reader's `src_addr` (now a tensor binding).

Both overrides name a `TensorArgument` for every io-tensor `TensorParameter` (cache + input / input + dst).
The legacy sharded-input `UpdateDynamicCircularBufferAddress` re-apply is subsumed: the borrowed-memory
input DFB re-resolves its backing L1 from the input `TensorArgument` automatically on cache hit.

### Device-op-class edits
- Pybind entry points removed: **none** (no pybound `create_descriptor` — `kv_cache_nanobind.cpp` binds
  only the four user-facing functions).
- Custom `compute_program_hash`: **left intact** at `update_cache_device_operation.cpp:160`.
- No direct-descriptor conversion needed (`program_factory_t` already present); no pybind-hook-only param.
- The device-op class (`update_cache_device_operation.{hpp,cpp}`) was **not touched**.

### Open items
- **First `CustomProgramSpecFactoryConcept` port in the tree.** No prior in-tree exemplar of a
  `ProgramRunArgs`-returning `override_runtime_arguments`; the concept and cache-hit path
  (`UpdateProgramRunArgs`) both worked as documented. Worth noting for the next custom-concept porter
  that kv_cache is now a working reference for the override shape.
- Two dead-ish includes in `update_cache_device_operation.hpp:13-14`
  (`global_circular_buffer.hpp`, `program_descriptor_patching.hpp`) were left in place (audit Misc;
  out of the factory-body scope, did not block the build). Candidate ops-team cleanup.
- No tensor-arg relaxation candidates noticed; the custom hash's exclusions are the designed cache key,
  not a relaxation.

## Handoff points
None. The port stayed entirely inside the op directory; no out-of-op kernel edits, no framework gaps hit,
no capitulation.

## Successes
- **Aliased DFBs pattern** (`port_patterns.md` — Pattern: Aliased DFBs) fit the interm0/interm1 (c_24/c_25)
  case exactly. The re-derived kernel-touch census (two distinct `buffer_index`, independent FIFO cursors,
  one shared L1 region → 1P+1C each) landed on the same disposition the (mid-port-updated) audit/brief
  prescribe. The `advanced_options.alias_with` strict-clique validated on device without a multi-binding
  flag — construction at `update_cache_multi_core_program_factory.cpp` interm0/interm1 specs.
- **`unpack_modes` required-entry rule** (recipe "Compute kernels" / Float32 trigger) fired as documented:
  the intermediate DFBs derive their format as `fp32_dest_acc_en ? Float32 : Float16_b`, so with no
  Float32 *tensor* anywhere the interm1 DFB still needs an explicit `UnpackToSrc` entry when
  `enable_32_bit_dest` is set. The fp32 test variants (`test_*_fp32`, 128 cases) exercised this and passed
  — had the entry been omitted the validator would have rejected the spec.
- **Legality-check forcing + proof markers** caught nothing wrong, but the `METAL2_CHECKS_FORCED` grep gave
  a definitive "checks are live" signal (2 markers/program-construction × cache misses) so the green result
  is trustworthy rather than a bypassed false-green.
- **Shared-kernel census disambiguation** (`port_patterns.md` — Caution: Porting a shared kernel): the
  filename grep flagged `paged_cache` / `deepseek_prefill`, but checking the *bound path* showed those are
  same-named private copies / their own forks — so the kv_cache kernels convert in place, and only the
  fill writer is a true borrowed donor (existing fork reused).

## Friction
### Gaps
- **First `CustomProgramSpecFactoryConcept` port** — no reference for the `ProgramRunArgs`-returning
  override; had to derive the signature from `operation_concepts.hpp:129-131` and the cache-hit contract
  from `ttnn_factory.md`. Both were sufficient; no doc gap, just no worked example. (Now there is one.)
- **`skip_for_blackhole` vs. actual hardware** — most `test_update_cache.py` cases carry
  `@skip_for_blackhole` (issue #12349). A stale note said this host was blackhole; live `tt-smi` showed a
  Wormhole n150, where the cases run. Not a doc gap, but a reminder that the porter must read the live arch
  rather than trust a prior note — a blackhole run would have skipped most coverage and produced a thin
  false-green.

### Confusion
- **c_24/c_25 aliased CB — resolved to Aliased DFBs.** The re-derived kernel-touch census is a clean
  aliased-DFB pair (two distinct `buffer_index`, independent FIFO cursors, one shared L1 region, 1P+1C
  each) → `advanced_options.alias_with`, not the multi-binding flag. The audit + brief were updated
  mid-port to prescribe exactly this; the port matches. (An earlier audit revision had hedged this as
  "very likely multi-binding"; the census-first re-derivation landed on the same answer the updated docs
  now state.) The audit's own "recipe notes" flag that the audit CB-endpoints *counting model*
  over-calls a multi-`buffer_index` `CBDescriptor` as multi-binding — worth folding the "split
  per-`buffer_index` before counting" rule into the audit subject so future auditors don't have to reach
  into the port recipe to classify this shape.

## Open items for downstream
- **Shared kernel (fill writer):** reused the existing donor Metal 2.0 fork
  `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id_metal2.cpp`
  (rung 1 — **reuse**; no new fork created, no pointer comment added by this port). Its interface
  (`dfb::out` CONSUMER, `tensor::dst`, named RTAs `num_pages` + `start_id`) is now a constraint the fill
  factory is built against. Remaining legacy-donor consumers are tracked in issue #52228 (that is a
  sunset/coordination list, not authorization to convert the legacy file in place).
- The kv_cache update/fill kernels are **not** shared: `experimental/paged_cache` and
  `experimental/deepseek_prefill` carry their own private copies/forks (distinct paths), so this port
  converted the kv_cache kernels in place. No coordination needed with those ops for this change.
- **Test skips are pre-existing.** 576 skips are param-validity (`batch_offset` only valid for
  num_users < 32) and arch conditions in the untouched test file — the port cannot change skip decisions,
  and 666 passed = the exact sum of every test function's passing param count, so no coverage was lost.

# Metal 2.0 Port Report — nlp_concat_heads

## Outcome

**Green — post-port results match the pre-port baseline exactly** (same box, serial runs, JIT
cache purged between eras, `ttnn` import sanity-checked against this tree).

| Suite | Baseline | Post-port |
|---|---|---|
| `tests/tt_eager/python_api_testing/unit_testing/misc/test_nlp_concat_heads.py` | 217 passed | **217 passed** (131.3s) |
| `tests/tt_eager/python_api_testing/unit_testing/misc/test_sharded.py::test_sharded_concat_heads` | 2 passed, 2 skipped | **2 passed, 2 skipped** (4.4s) |

The 2 skips are the 12x8-grid variants on this 8x8-grid box (identical in both eras).

Metal 2.0 legality checks were **forced on** for the post-port run via an uncommitted
working-tree scaffold in `tt_metal/impl/metal2_host_api/{program_run_args,program_spec}.cpp`
(every `skip_validation` site forced to validate, with a `METAL2_CHECKS_FORCED` log marker).
Marker counts in the test logs: 436 (interleaved log), 6 (sharded log) — non-zero in both, so
every green above ran with live legality checks. The scaffold is excluded from the commit.

## Provenance

- **Recipe docs (this port):** version cannot be pinned — the `metal_2.0` docs tree is untracked in
  this checkout (`git log -1 -- docs/.../metal_2.0/` prints nothing).
- **Audit docs (inherited):** version cannot be pinned — the `metal_2.0` docs tree is untracked in
  this checkout (`git log` prints nothing for it). *(Copied from `METAL2_PORT_BRIEF.md`.)*

## TTNN ProgramFactory

- **Concept realized:** `ProgramSpecFactoryConcept` — `NLPConcatHeadsProgramFactory` gained
  `create_program_artifacts(...)` returning a `ProgramSpec`; verified post-build that the
  instantiated adapter is `ProgramSpecMeshWorkloadFactoryAdapter<NLPConcatHeadsProgramFactory>`
  (i.e., the spec path was actually selected, not the legacy descriptor path).
- **Device-op-class edits:** none (the op already sat in a proper `program_factory_t` variant;
  no pybound `create_descriptor`; no custom hash).
- **Structure:** one factory, two host branches (interleaved / sharded), mirrored faithfully:
  - *Interleaved:* reader = own kernel (`reader_tm_tile_layout_nlp_concat_heads.cpp`, converted
    in place), writer = reused shared fork
    `writer_unary_interleaved_start_id_metal2.cpp` (untouched). One circular `q_out` DFB
    (2 tiles), `tensor::src0` / `tensor::dst` accessor bindings, per-core RTAs for
    tile offsets / counts.
  - *Sharded:* two instances of the converted sharded kernel
    (`reader_tm_tile_layout_nlp_concat_heads_sharded.cpp`) over one grid with Reader/Writer
    configs; borrowed (tensor-backed) `in0` / `out0` DFBs bound 1P+1C; geometry as CTAs; no RTAs.
- **Dispatch kinds:** all-constant geometry → CTAs; buffer/tensor identity → accessor bindings
  (`TensorAccessorArgs` eliminated); per-core-varying offsets → RTAs. No CRTAs, no varargs, no
  `allow_instance_multi_binding`.
- **Kernel `opt_level`:** not set — both owned kernels are DM kernels; the O2-vs-O3 compute
  trap does not apply (no compute kernel in this op).

## Handoff points

- **Latent broken config gets loud (ops team).** Legacy validation permits sharded-input +
  interleaved-output, but `create_descriptor` then took the sharded branch while creating no
  CB 16 — both kernel instances touched an unconfigured CB and nothing ever wrote the output
  (pre-existing latent bug; audit "Misc anomalies"). The port mirrors the host conditional
  faithfully (the `out0` DFB is declared only under `out_sharded`), so in that config the
  sharded kernel's `dfb::out0` token is not generated and the config now fails loudly at kernel
  JIT instead of silently producing garbage. No *intended* config changes behavior. The ops team
  should either forbid the config in `validate_on_program_cache_miss` or implement it.

## Successes

- **Two-toucher 1P+1C pattern fired correctly** (port_patterns.md "Two-toucher DFB → assign
  1P+1C"): the sharded config's dual-instance work-split (one source, Reader- + Writer-config
  over one grid, both raw-peeking the borrowed DFBs) initially reads as "two producers →
  multi-binding flag"; the endpoint-assignment procedure routed it to a clean 1P+1C instead
  (factory sharded branch, `dfb_bindings`). No `allow_instance_multi_binding` anywhere in the
  port.
- **Shared-kernel rung 1 (reuse) worked exactly as documented**: the existing
  `writer_unary_interleaved_start_id_metal2.cpp` fork fit this op's interleaved writer 1:1
  (`num_pages` / `start_id` RTAs, `dfb::out`, `tensor::dst`, no defines); the factory conformed
  to the fork's vocabulary with zero kernel edits on that file.
- **Audit open-question mechanism worked**: the dead `reserve_back` pair was surfaced by the
  audit, decided by the invoker (strip approved), and the port stayed on-whitelist throughout.

## Friction

- **Session orchestration:** the coordinator stated the branch
  `vsureshTT/Metal2_port_nlp_concat_heads_v2` was checked out in `/localdev/vsuresh/tt-metal`;
  verification found the checkout on `main` (the branch exists and points at the same commit,
  `4a5bfad59c6`). Branch switching is permission-blocked for this agent; flagged to the coordinator
  at the first checkpoint and resolved externally — the port is committed on
  `vsureshTT/Metal2_port_nlp_concat_heads_v2` as intended. Both refs point at the same base
  commit, so no working-tree content was affected.

## Findings (bugs / oddities preserved, not fixed)

- **Dead FIFO sync stripped with explicit approval** (the one sanctioned functional cleanup in
  this diff): `reader_tm_tile_layout_nlp_concat_heads_sharded.cpp` lines 35–36
  (`cb_in0.reserve_back(block_size)` self-annotated `// Redundant`, and
  `cb_out0.reserve_back(block_size)`), plus the already-commented-out
  `// cb_out0.push_back(block_size);` at line 62, were removed per the audit's open question #1,
  approved by the invoker on 2026-08-27. The calls were functionally dead (full-capacity reserve
  on an empty borrowed CB, never paired with a live push/pop); removing them frees both sharded
  DFBs to bind clean 1P+1C. This is the audit-recommended resolution, recorded here because it
  is the only edit in the port that is not a pure Metal 2.0 syntax swap.
- **`block_size` CTA is now read by no code** in the sharded kernel (its only uses were the
  stripped dead reserves). Kept on both host and kernel sides — it documents the shard geometry
  and mirrors the pre-existing precedent of the unused `single_tile_size_bytes` local (sharded
  kernel line 30, unused in legacy too, preserved). The ops team may drop both in a cleanup.
- **Stale comments preserved** in the factory: shape literals from a past model (`// 142`,
  `// Output shape is: [B, 1, s, 4544]`) and the `Grayskull Device Setup` banner; both reader
  kernels label their runtime args `// WRITER RUNTIME ARGS` (copy-paste). Cosmetic; left as-is
  per the comment-preservation rule.
- **`TT_ASSERT(output.buffer() != nullptr, ...)`** kept verbatim in the factory (guard census
  preserved) even though the `Buffer*` local it used to guard is gone.

## Open items for downstream

- **Shared kernel touch (rung 1 — reuse):**
  `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id_metal2.cpp`
  reused as the interleaved writer (`KernelSpec::source` points at it); **no new file created;
  fork treated as read-only** (it has prior consumers). Remaining unmigrated legacy-copy binders
  are tracked in issue #52228 (sunset list, per the audit).
- **`get_tile_size(cb_id)` → `get_entry_size()` mapping on DM kernels:** both own kernels read
  the tile size via the DM-side free helper into a `const` (not `constexpr`) local. The DFB
  member `get_tile_size()` exists only under `DFB_DESCRIPTORS_DEFINED`
  (`chlkc_descriptors.h` present), which is not guaranteed for a pure-DM program; the §B mapping
  `get_entry_size()` was used instead — byte-identical here (`entry_size == single_tile_size`,
  tile-paged buffers) and the same spelling the shared writer fork already uses. If a future
  op's DFB entry size diverges from its tile size, this substitution is not valid there.
- **Relaxation candidates:** none observed (`TensorParameter`s stay strict).

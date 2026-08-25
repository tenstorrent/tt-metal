# Metal 2.0 Port Report — rotary_embedding_hf

## Outcome

**PORTED** — both factories (`RotaryEmbeddingHfMultiCoreSharded` decode, `RotaryEmbeddingHfMultiCore`
prefill) converted to `ProgramSpecFactoryConcept`, all four internal configs. Verification: the
no-regression suite `tests/tt_eager/python_api_testing/unit_testing/misc/test_rotary_embedding_hf.py`
reproduced the pre-port baseline exactly (749 passed, 1 skipped) on a fresh build, with the Metal 2.0
legality checks forced on (both `METAL2_CHECKS_FORCED` markers live throughout the run). No unpack_modes
validator issue fired.

## Provenance

- **Recipe docs (this port):** `2cd0286fa17 2026-08-25 docs(metal_2.0): a run in flight freezes the kernel sources`
- **Audit docs (inherited):** `2cd0286fa17 2026-08-25 docs(metal_2.0): a run in flight freezes the kernel sources`

## TTNN ProgramFactory

### Concept realized
`ProgramSpecFactoryConcept`, both factories, as the audit chose. Each factory's `create_descriptor` became
`create_program_artifacts` returning `ProgramArtifacts{.spec, .run_params}`; the internal
single-tile/multi-tile host-side selection is preserved as two builder functions per factory. The sharded
factory emits **no** `KernelRunArgs` at all (legacy set zero runtime args — both kernels are RTA-free) and
carries all four io tensors as `TensorArgument`s backing borrowed DFBs.

### Device-op-class edits
- Pybind entry points removed: **none** (nanobind file binds only the public op function; untouched).
- Custom `compute_program_hash`: **none** exists; nothing touched.
- No device-op-class file was edited (`rotary_embedding_hf_device_operation.{hpp,cpp}` byte-identical; the
  `program_factory_t` variant already existed, so no exception-3 restructure).

## Handoff points

None. No boundary-rule violations, no kernel-lib gaps, no framework gaps hit, no pybind surface removed.

## Successes

- The audit brief's CB-endpoint census (self-loop vs 1P+1C per `(CB, config)`) matched an independent
  re-derivation from the kernel sources exactly — zero corrections. The brief's "wait_front-only is still a
  locked CONSUMER" note (trans_mat/scalar CBs) preempted a likely misread.
- Recipe Hardware configuration → "Check for a *dropped* field before using the helper" fired for real:
  both legacy factories resolve five compute knobs via `get_compute_kernel_config_args` but copy only
  `math_fidelity` + `fp32_dest_acc_en` onto `ComputeConfigDescriptor`. A naive `to_compute_hardware_config`
  would have silently started honoring the caller's `math_approx_mode` / `dst_full_sync_en`. Handled by
  `make_decode_compute_hw_config` / `make_prefill_compute_hw_config` (helper + explicit
  `sfpu_precision_mode = Precise`, `double_buffer_dest = true` — the legacy-descriptor defaults).
- Recipe Compiler options rule 2 applied mechanically: no legacy `opt_level` anywhere (grep: 0 hits) →
  explicit `KernelBuildOptLevel::O3` on all six compute KernelSpec construction sites (2 sharded configs +
  2 prefill configs × the g1/g2 lambda); verified by the opt_level pairing check below.
- The shared-kernel coordination (sibling creates the fork, this port reuses) worked exactly as scripted:
  the fork appeared between this port's inventory and its prefill construction, and its `dfb::`/`args::`
  vocabulary (`in/cos/sin/trans_mat/rotated_in_interm/cos_interm/sin_interm/out`, `num_rows` CTA) was
  adopted verbatim by the prefill factory (and happened to match the names this port had already used in
  the sharded factory).
- The borrowed-DFB ≥2-work-unit hazard (known framework bug from prior ports) is structurally excluded
  here: borrowing only occurs on shard-spec paths, where `core_group_2` is empty, so every config with a
  borrowed DFB emits exactly one WorkUnitSpec. Verified per config during construction.

## Friction

- **Gap (minor):** the recipe's unpack_modes required-entry rule says entries are required for Float32 DFBs
  a compute kernel "consumes", but doesn't say whether a *self-looped* (produce+consume) borrowed output DFB
  counts as consumed, or whether an entry on a produced-only DFB is rejected. This port added
  `UnpackToSrc` entries for every Float32 compute-CONSUMER-bound DFB (self-loops included, since they carry
  a CONSUMER binding) and omitted produced-only DFBs (prefill `out`). `UnpackToSrc` is semantically identical
  to omission, so either reading is behavior-preserving; a sentence in the recipe would remove the guess.
- **Confusion (minor):** the anti-pattern cb-name sweep says "expect zero hits", but the reviewed, landed
  reference port (`bcast_sharded_h_program_factory.cpp`) itself carries `// legacy CB c_0 ...` mapping
  comments that the grep flags. This port followed the landed convention (see self-audit below); the
  checklist text and the convention could be reconciled.
- Otherwise mechanical: the legacy factories were already on the descriptor API with `Buffer*`-annotated
  RTAs, which made Dropped Plumbing enumeration and the address→binding conversion routine.

## Anti-pattern self-audit results

Denominator: 20 `.cpp/.hpp` files in the op directory; 13 changed/new code files in the diff scanned for
citations. All sweeps run over a non-zero denominator:

- Buffer addresses in run-args (`buffer()->address()` / `emplace_runtime_args` / `Buffer*`): **0 hits**.
- `TensorAccessorArgs` anywhere: **0 hits** (all 9 third-arg page-size sites dropped with the collapse).
- `CircularBuffer` / `CBDescriptor` / `CBIndex` / `CBFormatDescriptor`: **0 hits**.
- Positional arg reads (`get_compile_time_arg_val` / `get_arg_val` / `get_common_arg_val`): **0 hits** in code
  (3 hits are in these .md artifacts).
- cb-name sweep: **33 hits, all adjudicated intentional** — every one is a `// legacy CB c_N` mapping
  comment on a `DFBSpecName` declaration, the same documentation convention the landed bcast port uses. No
  identifier, spec-name string, accessor name, or kernel local carries a `cb` name.
- Conditional DFB bindings: none exist (the only define, `OUT_SHARDED`, gates NoC use, not a binding — the
  legacy kernel already constructed its accessor unconditionally and the port mirrors that).
- `.id` extraction / temp-DFB-for-id: **0 hits**.
- CTA→RTA demotion: none — per-group `num_rows` preserved as per-KernelSpec CTAs (compute_g1/compute_g2).
- `allow_instance_multi_binding`: **0 hits**.
- Varargs: none used.
- Forced-legality scaffolding in diff: none — `git diff --name-only $BASE | grep ^tt_metal/` empty
  (the working tree's `skip_validation` force in `tt_metal/impl/metal2_host_api/` is unstaged invoker
  scaffolding and is never committed).
- Ephemeral-doc citations from code: **0 hits** over the 13 changed code files.
- TT_FATAL census: **no output** (the op's factories carried zero guards; the device-op's guards untouched).
- opt_level pairing: 4 compute construction sites ↔ 4 explicit O3 lines (the two prefill lambdas each build
  both g1/g2 specs, covering all 6 possible compute KernelSpecs). DM kernels: none set, legacy O2 == Metal
  2.0 O2 default.
- `constexpr` metadata token form (`get_tile_size(dfb::x)`, Gen1-only): **not needed anywhere** — every
  legacy `get_tile_size(cb)` was a non-constexpr local, so all became `dfb.get_tile_size()` member getters.
  No Quasar-uplift debt from token-form metadata.

## Open items for downstream

- **Shared kernel (borrowed) — rung 1, reused fork.**
  (a) Kernel: `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding/device/kernels/compute/rotary_embedding_single_tile.cpp`;
  (b) rung: **reused the existing `_metal2` fork**
  `.../rotary_embedding/device/kernels/compute/rotary_embedding_single_tile_metal2.cpp`, created by the
  sibling `rotary_embedding` port running in parallel in this tree — no new file created and nothing under
  `rotary_embedding/` touched by this port; this port's prefill single-tile compute KernelSpec conforms to
  the fork's binding vocabulary (`dfb::in/cos/sin/trans_mat/rotated_in_interm/cos_interm/sin_interm/out`,
  CTA `args::num_rows`; the `DECODE_MODE` define-path is never enabled by this op);
  (c) remaining unmigrated consumers of the *legacy* file after both ports land: none in this op family —
  the sunset set was {`rotary_embedding`, `rotary_embedding_hf`} and both now bind the fork, so the legacy
  copy's deletion is gated only on both ports merging.
- **Dead CTA** `rotary_embedding_hf_sharded.cpp` (kernel) — CTA `Ht` (fed `n_heads_t` by the sharded
  multi-tile factory) is read and immediately `(void)Ht;`-discarded; preserved faithfully as a named CTA
  (`{"Ht", n_heads_t}`). Ops-team cleanup candidate: drop the arg on both sides.
- **Inert writer plumbing under `OUT_SHARDED`** — `writer_rotary_embedding_hf_interleaved.cpp`: `start_id`
  RTA, `output_tile_bytes`, and the `TensorAccessor` are unused when `OUT_SHARDED` is defined (only
  `wait_front(num_tiles)` runs). Preserved; ops-team cleanup candidate.
- **Self-aliasing NoC read** in in-sharded multi-tile prefill: `reader_rotary_embedding_hf_interleaved.cpp`
  reads src tiles through the `tensor::src` accessor into the very DFB borrowed from `input.buffer()`.
  Functionally correct but wasteful; deliberately preserved byte-for-byte. Ops-team optimization candidate.
- **unpack_modes doc clarification** (see Friction): whether self-looped consumed DFBs need the
  Float32/32-bit-dest entry, and whether produced-only DFB entries are rejected — worth one sentence in the
  recipe's Hardware configuration section.

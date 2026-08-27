# Metal 2.0 Port Report — nlp_create_qkv_heads_decode

## Outcome

**PORTED** — all three factories (Interleaved, Sharded, ShardedSubcoregrid) converted to
`ProgramSpecFactoryConcept` in one pass; post-port tests match the pre-port baseline exactly
(**205 passed, 39 skipped, 0 failed** on WH n150), with the Metal 2.0 legality checks force-enabled and proven live
(both `METAL2_CHECKS_FORCED` markers, 410 hits across the run).

Notable: this is the **v2** of this port. The v1 attempt (2026-07-22) capitulated the Sharded + Subcoregrid
factories on a framework bug — a borrowed-memory DFB's device base address was corrupted when a node's
present-DFB-id set had an interior hole (the non-overlap multi-work-unit configs hit it). That bug was fixed on
`main` by `3f173de1a13` (*"[Bug fix]: #51409 on splitting dfb id to be program unique and adding a device facing id
that is unique within core group"*). This port re-exercised the failing shape — sharded non-overlap
(2 WorkUnitSpecs, borrowed output DFBs on disjoint q/k grids, 9 test variants) — and it now passes at PCC 1.0.

## Provenance

- **Recipe docs (this port):** the provenance command
  (`git log -1 --format='%h %cs %s' -- docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/`) prints nothing
  in this checkout — the recipe docs are untracked here (copied from the doc branch). The audit brief pins the doc
  revision below.
- **Audit docs (inherited):** `50e992a8ec2 2026-08-21 docs(metal_2.0): a run in flight freezes the kernel sources`
  *(hash is the HEAD of `origin/akertesz/op-porting-recipe` — the recipe docs are not on the working branch)*

## Verification summary

- Baseline captured **before** any kernel edit (kernels JIT from the working tree):
  `pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_nlp_create_qkv_heads_decode.py -q`
  on WH n150 @ `4a5bfad59c6`: 205 passed / 39 skipped / 0 failed (56s). Test set confirmed with the invoker.
- JIT kernel cache entries touched during the baseline were purged before the post-port run (stale-era protection);
  post-port kernels compiled fresh.
- Post-port: 205 passed / 39 skipped / 0 failed (59s) — identical outcome per test id, PCC 1.0 everywhere the tests
  assert it. Legality checks forced at all 9 `skip_validation` sites; both file markers observed in the test log;
  scaffolding reverted before commit (final diff touches only the op directory).
- Config coverage actually exercised on this n150 (identical skips to baseline):
  - Interleaved: L1 + DRAM, many shapes (84 DRAM variants) — `use_aligned_path` remains **off** on WH for
    bf16/fp32 (sub_tile_line_bytes ≥ WH dram alignment), so the `USE_ALIGNED_PATH` scratch-DFB path is
    compile-verified here but runtime-exercised only on Blackhole. Same was true pre-port.
  - Sharded: overlap and **non-overlap** (2 WUs, borrowed DFBs), max- and min-width shard.
  - Sharded + batch_offset (`test_create_heads_with_slice`, overlap **and** non-overlap, 3-loop program-cache
    checks) — this is the multi-binding `batch_offset` DFB config; the forced validator accepted it and cache-hit
    tensor refresh worked (`num_program_cache_entries == 1` asserts pass).
  - Subcoregrid: one variant (overlap, batch=1) — the others skip on this hardware ("Sub core grid is out of
    bounds"), exactly as in the baseline. See Open items.
- Anti-pattern self-audit: all sweeps ran over 16 files (non-zero denominator), all clean — no buffer-address
  run-args, no magic CB indices, no `TensorAccessorArgs` remnants, no positional arg reads, zero `cb`-substring
  leftovers, no `.id` extraction, TT_FATAL census unchanged, zero `.md` citations from code, no `tt_metal/` files in
  the final diff, `opt_level` n/a (pure-DM op, legacy unset = O2 = Metal 2.0 default on every KernelSpec).

## TTNN ProgramFactory

### Concept realized
`ProgramSpecFactoryConcept`, all three factories, exactly as the audit chose. Each factory's `create_descriptor`
was replaced by `create_program_artifacts` inside the existing `program_factory_t` variant; the device-operation
class is byte-identical.

### Device-op-class edits
- Pybind entry points removed: **none** (the nanobind file binds only the composite user-facing function).
- Custom compute_program_hash: **none** (default reflection hash; untouched).

### Open items
- No relaxation candidates observed; the kernels index by explicit shapes/offsets baked at CTA time.

## Handoff points

none — no capitulations, no boundary-rule violations, no kernel-lib or framework gaps hit, no pybind surface
removed. (The one framework interaction worth knowing predates this port and is fixed: the borrowed-DFB id-hole
bug, `3f173de1a13`, without which the Sharded/Subcoregrid factories would still be blocked.)

## Successes

- **The audit's CB-endpoint census matched re-derivation exactly** on all 10 `(CB, config)` rows; the
  endpoint-assignment procedure (patterns catalog, two-toucher entry) made every disposition mechanical:
  1P+1C on all output DFBs, self-loops on the interleaved scratch and subcoregrid batch-offset DFBs, the flag on
  sharded `batch_offset`, and the dead-CB drop of sharded `c_14`.
- **The brief's Watch-for on sharded `c_15` fired correctly** — the "don't hunt for a hidden consumer / don't give
  the writer its own DFB" note prevented exactly the wrong 'fix' (the subcoregrid factory's shape) from being
  applied to the sharded factory.
- **The conditional-binding pattern (defines + `#ifdef`) carried four distinct gates cleanly** in one op:
  `USE_ALIGNED_PATH` (interleaved scratch), `USE_BATCH_OFFSET` (optional tensor + DFB), and `PROCESS_QV`/`PROCESS_K`
  (per-instance output DFBs whose backing CB does not exist on the other grid in non-overlap mode).
- **The forced legality checks were worth it**: the validator ran on every program construction in the suite
  (410 marker hits), so the multi-binding, self-loop, borrowed-memory, and 2-work-unit shapes are all
  validator-approved, not merely test-green.

## Friction

- **Gap — multi-binding × per-node census × self-loop set-equality interaction.** The brief says "set the flag" for
  sharded `batch_offset` (two locked producers, no consumer anywhere). Neither the brief nor the recipe spells out
  that under `allow_instance_multi_binding` the per-node census still requires **≥1 CONSUMER per node**
  (`tt_metal/impl/metal2_host_api/program_spec.cpp:1426`), and that once any kernel binds both roles, the self-loop
  set-equality rule (`program_spec.cpp:1503`) forces the producer kernel set to equal the consumer kernel set. The
  only legal expression of "N locked producers, zero consumers" is therefore: **every** instance binds PRODUCER and
  CONSUMER, plus the flag. Suggest a sentence in the recipe's multi-binding paragraph (and the self-audit's
  "never stacked with a self-loop" item deserves a carve-out: with the flag set and all touchers locked to one role,
  the P+C-per-instance shape is *forced*, not a mis-slot).
- **Confusion — minor:** the recipe's "no `cb` survives" sweep catches kernel locals like `index_cb_wr_ptr` that are
  neither spec names nor accessor names; renaming them (`index_wr_ptr`) is trivially safe but is a rename beyond the
  documented `cb_*`→`dfb_*` variable rule. Worth one clarifying word ("including incidental locals").

## Open items for downstream

- **Shared kernel touches:** none — all three kernel sources are op-owned, each bound by exactly one factory, and
  all three factories converted in this port. No `_metal2` forks created anywhere.
- **Sharded factory's dead `c_14` dropped; the suspected missing writer-CB override is the ops team's call.**
  Legacy `nlp_create_qkv_heads_decode_sharded_program_factory.cpp:79-88` allocated a writer batch-offset CB
  (`c_14`) that no kernel CTA ever referenced — both RISCs `reserve_back`/`push_back` the *same* `c_15` instance
  (benign: identical data, no consumer). The subcoregrid factory *does* split reader/writer staging buffers. The
  port dropped the dead allocation (zero behavior change) and expressed the shared staging buffer faithfully via
  `allow_instance_multi_binding`. If the ops team decides the sharded writer was *meant* to have its own staging
  buffer (as in the subcoregrid factory), the Metal 2.0 change is small: add a second DFB spec, drop the flag, and
  bind each instance's own buffer (the subcoregrid factory in this port shows the exact shape).
- **One-past-the-end NoC-coordinate reads preserved.** All three shard-reading loops advance the core cursor after
  the final tile and immediately re-read the coordinate table one past its end (now via `get_vararg`); the fetched
  garbage value is never used. Behavior identical to legacy; a bounds guard would be an op-owner cleanup.
- **Subcoregrid V-output DFB sized from Q's shard spec** (legacy anomaly preserved;
  `nlp_create_qkv_heads_decode_sharded_subcoregrid_program_factory.cpp`, see the NOTE at the V spec). Masked while
  q/kv head counts both pad to 32; latent mis-size if the padding rule diverges. Ops-team cleanup candidate.
- **Coverage gaps on the local bench (pre-existing, identical in baseline):** subcoregrid non-overlap and
  subcoregrid-with-batch_offset variants skip on n150 ("Sub core grid is out of bounds" / batch≥32 non-overlap);
  their primary coverage is `tests/ttnn/distributed/test_multidevice_TG.py` (TG hardware). Recommend a TG CI run
  of that test before merge. The `USE_ALIGNED_PATH` interleaved path is Blackhole-only at runtime (WH alignment
  makes it unreachable for bf16/fp32); a BH run of the DRAM-interleaved tests would exercise it.
- **Composite entry point accepts and ignores `optional_output_tensors`** (`nlp_create_qkv_heads_decode.cpp:17`,
  surfaced as `output_tensors` by the nanobind file) — a user passing preallocated outputs is silently ignored.
  Pre-existing; out of port scope; op-owner review suggested.

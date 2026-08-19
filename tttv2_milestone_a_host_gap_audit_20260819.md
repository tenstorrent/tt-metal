# Milestone A Host-Only Gap Audit

## Goal

Independently inspect the Milestone A exit gates and the current changed files/tests for remaining
host-only correctness or regression gaps. Do not run TT hardware and do not edit shared production
or test files.

## Constraints

- Hardware commands and hardware tests are excluded.
- Shared production and test files are read-only for this audit.
- This uniquely named log is the audit's only file modification.

## Checkpoint 1: Scope and baseline inventory

- Read the authoritative plan and extracted the Milestone A sequence and exit gate.
- Captured the current changed-file inventory with `git status --short`.
- Confirmed the change set includes common-runtime prefill policy plumbing, all planned reusable 2D
  modules, Galaxy resource/CCL infrastructure, focused host tests, and WH Galaxy hardware tests.
- Host-verifiable gate areas selected for detailed audit: config fail-closed behavior, ownership and
  cleanup, repeat invocation, pre-hot-path strategy resolution, removal of `from_model_args`, 1D
  implementation immutability, default-runtime preservation, and topology-neutral runtime changes.
- Hardware-only claims (real WH execution, numerical PCC, and KV-cache PCC) will be assessed only
  for test structure and collection coverage, not executed or treated as passing evidence.

## Checkpoint 2: Static contract and ownership review

- The runtime policy change is topology-neutral and delegated through frozen config, but
  `BatchedPrefillPolicy.__post_init__` does not reject `minimum_active_rows` greater than
  `maximum_physical_batch`; it also lacks direct boundary tests for all constructor invariants.
- During the audit, concurrent work corrected the prefetch topology mismatch: validation now
  permits a positive `address_repeat_count` no greater than the global-CB mapping count, the WH
  helper constructs 12 active plus 8 dummy mappings while retaining 12 active address readers, and
  a focused host regression test covers the distinction and overflow. This is no longer an open
  audit finding.
- `MLP2D` has no `release`/`close` method despite owning three materialized lazy weights. Its decode
  and prefill paths are straight-line deallocation sequences without exception-safe cleanup, so a
  failure at W3, either reduce-scatter, multiply, all-gather, W2, or final all-reduce can leak owned
  inputs and intermediates. Existing host tests cover successful-path releases but no stage-failure
  matrix or module weight release.
- `RMSNorm2D` likewise has no `release`/`close` method for its materialized weight. Distributed
  decode/prefill release stats and converted inputs only after later operations succeed; collective
  or post-norm failures can leak intermediates. Lazy inputs/residuals also need explicit ownership
  assertions across head-local, fused, and distributed paths.
- Embedding, LM head, Rotary, Sampling, Attention, Prefetcher, Galaxy CCL, and GalaxyResources have
  explicit host ownership tests. MLP and RMSNorm are the conspicuous coverage exceptions.

## Checkpoint 3: Host execution and regression boundaries

- Ran the runtime policy, all focused 2D module, Prefetcher, Galaxy CCL, and GalaxyResources host
  suites without hardware: `353 passed in 54.71s`.
- Confirmed the current diff contains zero changed `models/common/modules/**/*_1d.py` files.
- Confirmed reusable 2D/Galaxy production files contain no `from_model_args` symbol and no imports
  from the prohibited legacy Llama/Qwen model packages.
- Existing evidence still records an incomplete 1D regression run and two unresolved host fixture
  failures in the broader runtime suite. The status page correctly leaves that exit-gate proof open;
  these failures should be fixed or conclusively rebased before Milestone A is declared complete.
- `MLP2D` and `RMSNorm2D` use Python `assert` for required fail-closed checks including mesh shape,
  architecture, device count, divisibility, weight shape/mesh ownership, prefetch/CCL mesh and mode
  compatibility, resolved state, input type, and mode. These checks disappear under `python -O`.
  Replace externally reachable contract assertions with explicit `TypeError`/`ValueError` checks
  and add optimized-interpreter or direct validation regression coverage.

## Prioritized actionable gaps

### P0: Make MLP2D and RMSNorm2D validation genuinely fail closed

- Replace externally reachable `assert` statements in config resolution and input/mode validation
  with explicit exceptions. Preserve assertions only for impossible internal invariants, if any.
- Add parameterized tests for bad mesh shape, architecture, device count, divisibility, weight shape,
  foreign weight/context/CCL mesh, context mode, input type, and unresolved config.
- Include a small subprocess test under `python -O` or an AST guard preventing contract validation
  from being implemented with `assert` again.

### P0: Complete MLP2D and RMSNorm2D ownership contracts

- Add idempotent `release`/`close` behavior that deallocates each materialized owned `LazyWeight`
  value once, clears handles/load state, preserves borrowed CCL/prefetch resources, and remains
  retryable after a deallocation failure.
- Refactor MLP decode/prefill stages into exception-safe ownership tracking. Add a stage-failure
  matrix covering W1/W3 projection, both reduce-scatters, activation, all-gather, W2, final
  all-reduce, and output placement; assert that caller inputs and persistent CCL buffers remain
  borrowed while every transient acquired before failure is released once.
- Add equivalent RMSNorm head-local, fused, distributed decode, and distributed prefill tests for
  lazy input/residual ownership and failures in pre-norm, gather, post-norm, and output placement.

### P1: Tighten BatchedPrefillPolicy invariants

- Reject `minimum_active_rows > maximum_physical_batch`; the current policy silently constructs a
  configuration that can never batch.
- Add direct constructor tests for tuple type, empty/duplicate/unsorted/unsupported batch sizes,
  maximum mismatch, non-positive integer limits, boolean types, and the active-row upper bound.
- Add a focused runtime-config test documenting whether an explicitly supplied policy is allowed to
  override the separate `max_prefill_batch_size` argument or must match it at the builder boundary.

### P1: Turn baseline regression exceptions into green evidence

- Update the stale executor-config exact-field expectation to include the pre-existing diagnostics
  field, without changing production behavior.
- Update the stale trace capture-plan test double to expose the existing `prime` contract.
- Re-run the complete common-runtime suite and the complete existing 1D module suite. A matching
  pre-change failure baseline demonstrates no new regression, but it does not satisfy Milestone A's
  explicit all-green exit gate.

### P2: Consolidate structural boundary checks

- Add one inventory-driven host test over every reusable `*2d.py` and Galaxy infrastructure file
  that rejects `from_model_args`, prohibited legacy model-package imports, and common `TT_CCL`
  imports. Current coverage is split across per-module tests and manual grep, leaving some modules
  without a durable source-level guard.
- Add one host composition test that constructs sealed Prefetcher2D contexts, GalaxyResources/CCL,
  and representative MLP/RMSNorm/Attention configs together, verifies mode compatibility before
  compute, activates both modes repeatedly, and checks reverse cleanup ownership. This targets
  interface drift between independently tested components without requiring hardware.

## Checkpoint 4: Final verification

- Re-ran the concurrently updated Prefetcher2D host suite: `16 passed in 2.42s`, including the new
  active-reader versus dummy-mapping regression.
- `git diff --check` passed.
- No TT hardware command or hardware test was run.
- No shared production or test file was edited by this audit. The only audit-owned modification is
  this work log.

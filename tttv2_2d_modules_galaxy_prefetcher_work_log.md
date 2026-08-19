# Milestone A Galaxy CCL and Prefetcher2D Work Log

## Checkpoint 1 - Contract and ownership audit

- Date: 2026-08-19
- Dedicated goal: implement the shared structural Galaxy CCL collaboration and `Prefetcher2D` lifecycle/config slice.
- Write scope: `models/common/models/galaxy/`, `models/common/modules/prefetcher/`, and focused host-only tests for those packages.
- Required invariants: frozen resolved configuration, Wormhole mesh `(8, 4)` with 32 devices, explicit weight registration followed by sealing, immutable borrowed mode contexts, deterministic idempotent cleanup, and no graph scanning or eager lazy-weight materialization.
- Exclusions: no TT hardware use, no edits to 1D modules or `models/common/modules/tt_ccl.py`, and no imports from legacy or model-named packages.
- Collaboration status: this lane is one of six parent-dispatched agents with a dedicated `/goal`; its write set is disjoint from the other Milestone A lanes.

## Checkpoint 2 - Initial implementation

- Added a frozen Galaxy CCL resource model for reduce-scatter, all-gather, all-reduce, and all-gather-concat, plus a structural collaborator protocol and deterministic per-operation semaphore cycling.
- Added `Prefetcher2DConfig`, immutable prefill/decode contexts, explicit subdevice-manager and global-CB creation, ordered device-weight registration, sealing with address metadata, operation-boundary activation, and idempotent cleanup.
- CCL modules continue to invoke TTNN collectives directly; the shared model-layer object only supplies resolved topology, axes, semaphore pools, buffers, and subdevice identity.
- Prefetcher ownership excludes registered weights: device weights are borrowed, while manager IDs and address metadata are lifecycle-owned.
- Added focused host-only tests using fake mesh/tensor resources. No TT hardware command has been run.

## Checkpoint 3 - Host verification and cross-lane synthesis

- Focused Galaxy CCL and `Prefetcher2D` suite: 18 passed.
- Cross-lane host integration suite covering Galaxy CCL, `Prefetcher2D`, Attention2D, Embedding2D, LMHead2D, RotarySetup2D, and Sampling2D: 94 passed.
- Python byte-compilation and `git diff --check` passed; the configured environment has no standalone `ruff` executable, while `python -m black --check` reported the implementation files unchanged.
- Cross-lane integration added immutable `sub_device_id` and named `weight_addresses` views expected by MLP2D/Attention2D, while retaining packed TTNN address metadata for the prefetch operation.
- Galaxy CCL now requires explicit mode activation at operation boundaries and delegates existing semaphore-access method names to the active immutable resource context.
- No TT hardware was used or reset. Hardware execution and numeric PCC remain outside this host-only lane.

### Lane modularity scorecard

- New shared model files: `models/common/models/galaxy/__init__.py`, `models/common/models/galaxy/ccl.py`.
- New 2D module files: `models/common/modules/prefetcher/__init__.py`, `models/common/modules/prefetcher/prefetcher_2d.py`.
- New focused tests: Galaxy CCL resource/config tests and Prefetcher2D lifecycle/config tests.
- Existing shared files changed by this lane: zero.
- 1D module implementation files changed by this lane: zero.
- Default runtime behaviors changed by this lane: zero.
- Legacy model-package imports: zero.
- Boundary status: resources remain in shared Galaxy model infrastructure and reusable module config/lifecycle ownership; no orchestration or 1D leakage.

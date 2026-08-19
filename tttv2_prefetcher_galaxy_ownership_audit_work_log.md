# Prefetcher2D / GalaxyResources Ownership Audit

## Checkpoint 1: Scope and plan contract

- Goal: audit and resolve Milestone A ownership between `Prefetcher2D` and `GalaxyResources` without running TT hardware.
- The plan assigns one `Prefetcher2D` owner per mesh. It owns subdevice-manager creation/activation, global circular buffers, weight registration/sealing, and cleanup.
- Modules and Galaxy CCL borrow resolved mode contexts; the executor activates prefill/decode contexts at operation boundaries.
- Initial code search shows both `Prefetcher2D` and `GalaxyResources` currently create, load, clear, and remove subdevice managers. This is competing ownership and must be replaced by an explicit borrowing contract if a bounded host-verifiable change is available.
- Constraints for this audit: production edits only in prefetcher/Galaxy resource files, dedicated host tests only, and no RMSNorm test edits or TT hardware execution.

## Checkpoint 2: Concrete integration design

- `Prefetcher2D` remains the sole owner of prefill/decode subdevice managers and mode activation.
- Add `Prefetcher2D.mesh_device` and `Prefetcher2D.borrow_context(...)`. The latter returns the immutable sealed context only when the caller's subdevices, worker ID, stall group, and local-L1 size exactly match the prefetcher configuration.
- Add a structural `Prefetcher2DResourceOwner` protocol for Galaxy wiring and host fakes.
- Require `prefetcher` in `GalaxyResources` and `create_galaxy_resources(...)`. Validate both mode plans through `borrow_context(...)` before allocating CCL resources.
- `GalaxyResources.activate(mode)` synchronizes the previous CCL mode, delegates manager/prefetch activation to `Prefetcher2D.activate(mode)`, then publishes the CCL mode. It never loads a manager or sets a stall group itself.
- `GalaxyResources.cleanup()` synchronizes and releases only Galaxy-owned CCL resources. It neither cleans up nor mutates the borrowed prefetcher owner; the future model/executor root must clean up Galaxy resources before the prefetcher.
- Host tests will assert exact-plan rejection, activation delegation, no duplicate manager lifecycle, no borrowed-owner cleanup, and allocation rollback limited to CCL resources.

## Checkpoint 3: Implementation and focused host verification

- Added `Prefetcher2DResourceOwner`, `Prefetcher2D.mesh_device`, and exact-policy `Prefetcher2D.borrow_context(...)`.
- Changed `GalaxyResources` and `create_galaxy_resources(...)` to require a borrowed prefetcher owner.
- Removed Galaxy manager creation, manager loading, stall-group mutation, manager rollback, and manager removal.
- Galaxy now validates both borrowed mode contexts before CCL allocation, delegates activation, and owns only CCL semaphores/buffers.
- Added dedicated host coverage for exact borrowing, delegated repeated/mode-switch activation, activation failure publication, incompatible-plan rejection before allocation, and cleanup ownership.
- Host result: `21 passed in 3.28s` for `test_prefetcher_2d.py` plus `test_resources.py`.
- Static result: `compileall` and `git diff --check` passed for the scoped production files, tests, and work log.
- No TT hardware command or test was run. `models/common/tests/modules/rmsnorm/test_rmsnorm_2d_wh_galaxy.py` was not modified.

## Checkpoint 4: Final audit

- Final host command: `pytest -q models/common/tests/models/galaxy/test_ccl.py models/common/tests/models/galaxy/test_resources.py models/common/tests/modules/prefetcher/test_prefetcher_2d.py`.
- Final result: `35 passed in 5.36s`.
- Final static checks: scoped `compileall` and `git diff --check` passed; Black formatted `models/common/models/galaxy/resources.py`. The standalone `ruff` executable is not installed.
- Production ownership is now unambiguous: `Prefetcher2D` owns managers, stall groups, global CB, address metadata, prefetch sessions, and their cleanup; `GalaxyResources` borrows the prefetcher and owns only Galaxy CCL allocations/cycles.
- Required construction order: initialize `Prefetcher2D`, explicitly register all expected weights, seal it, then call `create_galaxy_resources(..., prefetcher=prefetcher)`. Required cleanup order at the future model/executor root: `GalaxyResources.cleanup()` followed by `Prefetcher2D.cleanup()`.
- Changed production files: `models/common/modules/prefetcher/prefetcher_2d.py`, `models/common/modules/prefetcher/__init__.py`, and `models/common/models/galaxy/resources.py`.
- Changed dedicated host tests: `models/common/tests/modules/prefetcher/test_prefetcher_2d.py` and `models/common/tests/models/galaxy/test_resources.py`.
- No TT hardware was used and no RMSNorm test file was touched during this goal.

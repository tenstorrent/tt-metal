# Prefetcher + GalaxyResources Composition Work Log

## Checkpoint 1: Scope and existing coverage

- Created a dedicated goal for Milestone A host-only ownership and composition contracts.
- Inspected `Prefetcher2D`, `GalaxyResources`, and their existing isolated host tests.
- Confirmed the gap: Galaxy owner tests use a protocol fake and do not compose the concrete `Prefetcher2D` lifecycle with concrete `GalaxyResources`.
- Planned focused contracts for exact shared subdevice policy, repeated decode and mode transitions, worker-scoped synchronization, activation failure state publication, and independent cleanup ownership.
- TT hardware tests are explicitly out of scope. RMSNorm and attention hardware test files will not be modified.

## Checkpoint 2: Concrete composition contracts implemented

- Added a dedicated host-only suite that instantiates the concrete `Prefetcher2D` and `GalaxyResources` classes with injected fake resource bindings.
- Covered shared sealed contexts, repeated decode restart behavior, decode-to-prefill serialization, and worker-only synchronization.
- Covered split ownership: `GalaxyResources.cleanup()` releases CCL allocations without stopping or destroying the borrowed prefetcher; model-owned prefetch cleanup subsequently stops the sender, releases metadata/managers, and never deallocates borrowed weights.
- Covered exact subdevice-policy rejection before any CCL tensor allocation.
- Covered concrete prefetch-start failure without publishing an active Galaxy or CCL mode.
- Focused result: `4 passed in 0.65s`.

## Checkpoint 3: Related host regression gate complete

- Ran the new composition contracts with the existing Galaxy resource and Prefetcher2D host suites.
- Result: `29 passed in 4.31s`.
- Confirmed the added test file has no lines over 120 characters.
- `ruff` is not installed in the active environment, so no standalone Ruff check was available.
- No TT hardware tests or device-management commands were run.
- Final task-owned files are the new composition contract suite and this dedicated work log; RMSNorm and attention hardware tests were not edited.

# Verification Report: point_to_point

> **Headline / environment constraint.** `point_to_point` is an inherently **multi-device**
> CCL op (one device sends a shard to another over the fabric — needs **≥ 2 mesh devices**).
> This verification ran on a machine with **1 Blackhole device** (`ttnn.get_num_devices() == 1`),
> and **no simulator is configured** (`TT_METAL_SIMULATOR` unset). Per the global safety rules I
> did **not** change the IRD reservation. Consequently the **cross-device transfer itself was
> never executed** in this pass. Everything below that depends on the transfer is verified by
> **code review + the proven-reference derivation only**, not by on-hardware measurement.
> The single-device Python paths (packet framing, intermediate-spec math incl. bfloat8_b,
> allocation, global-semaphore, fabric-node-id) **are** exercised and pass.

## Code Review

### Fixed in this pass

1. **Package did not re-export the registry declarations (pipeline-blocking) — FIXED.**
   `ttnn/ttnn/operations/point_to_point/__init__.py` only exported the `point_to_point`
   function. `SUPPORTED` / `EXCLUSIONS` / `INPUT_TAGGERS` / `validate` lived on the inner
   `point_to_point.py` submodule and were **not visible on the package**. Both the golden
   harness (`from ttnn.operations.point_to_point import EXCLUSIONS, INPUT_TAGGERS, SUPPORTED`)
   and `eval.verify_supported` introspect the **package** — so the verifier CLI crashed with
   `AttributeError: op module 'ttnn.operations.point_to_point' missing required 'SUPPORTED'`.
   Fix: `__init__.py` now eagerly re-exports `EXCLUSIONS` / `INPUT_TAGGERS` / `validate`, and
   forwards the lazily-built `SUPPORTED` through a package-level `__getattr__` (PEP 562) so the
   ttnn-enum references are still evaluated late (circular-import safe). After the fix the
   verifier introspects cleanly and `SUPPORTED`/`EXCLUSIONS`/`INPUT_TAGGERS`/`validate` are all
   reachable from the package. This is the single most important fix in this pass — without it
   the op is invisible to the entire registry tooling.

2. **`reader_receive.cpp`: `packet_l1_addr` typed `uint64_t` — FIXED to `uint32_t`.**
   `get_write_ptr()` returns a `uint32_t` L1-local address; it was stored in a `uint64_t` and
   then narrowed back to `uint32_t` at every use (`noc_async_read` dst, page-offset math). No
   behavioral bug (the address fits in 32 bits), but the type was wrong and obscured intent.

### Reviewed and found correct (no change needed)

- **Kernel hygiene.** All four kernels use `void kernel_main()`, the `api/dataflow/dataflow_api.h`
  include path (not bare `dataflow_api.h`), and `TensorAccessor` (not the deprecated
  `InterleavedAddrGen`). The `packet_size_bytes` page-size override on the intermediate
  `TensorAccessor` is present on both the sender write and receiver read, matching the design's
  "identical NoC address + stale-AlignedPageSize" note.
- **Helper usage.** `FabricStreamSender<>` is used exactly per its documented armed-channel
  contract — `open → set_route_unicast → arm_unicast_write → arm_inc → write_page* → inc → close`
  on the sender, and `open → set_route_unicast → arm_inc → inc → close` for the receiver's
  "ready". The op correctly owns only what the helper banner says it must (page↔packet
  coalescing via `tt_memmove`, the local NoC ingress read, and the `noc_semaphore_wait_min` /
  `noc_semaphore_set` handshake). No CB ops wrap the helper.
- **CB sync ledger.** `cb_sender_pages`: `reader_send` pushes `num_pages`, `writer_send`
  waits/pops `num_pages` → balanced. `cb_receiver_pages`: `reader_receive` pushes `num_pages`,
  `writer_receive` waits/pops `num_pages` → balanced. The two packet-scratch CBs
  (`cb_packet_scratch`, `cb_packet_landing`) are single-owner staging buffers reserved once for
  a stable L1 base — no cross-kernel pairing, no sync obligation.
- **Page↔packet framing.** The two-regime coalescing (many small pages per packet; one large
  page split across `page_segments`) is internally consistent between `writer_send` (pack) and
  `reader_receive` (unpack): the `curr_pages_per_packet` recurrence differs by the expected
  off-by-one (sender uses `page_idx_end - page_idx - 1` after a completed page; receiver uses
  `page_idx_end - page_idx` at the start of a page), which is correct.
- **Cache-reuse semaphore reset (the central footgun).** Placement matches the design and the
  helper banner exactly: **sender** resets `noc_semaphore_set(sem, 0)` **before** its outgoing
  "done" inc; **receiver** resets **after** its wait. Both sides wait on their **local** copy
  with `noc_semaphore_wait_min(sem, 1)`.
- **Design conformance.** Algorithm (identity copy over fabric), pipeline topology (two-program
  `MeshProgramDescriptor`: send @ sender_coord, receive @ receiver_coord), work distribution
  (single worker core `{0,0}`, single link `link_idx=0`), and communication (one fresh
  `GlobalSemaphore`, 2-party handshake, route-direction reversal per endpoint) all match
  `op_design.md`.

### Advisory (not fixed — see Recommendations)

- The entry point issues **two** `ttnn.synchronize_device` host barriers per call (one right
  after `create_global_semaphore`, one after `generic_op`). The post-dispatch barrier is the
  design-mandated semaphore-lifetime guard. The first one is defensive (ensures the semaphore
  allocation/zero-init is complete before its address is used); it is *possibly* redundant given
  command-queue ordering, but removing it is a correctness risk I cannot test on 1 device.
  Left as-is; flagged as a perf micro-optimization to validate later.

## Registry Conformance

- **Confirmed present and correctly wired** (after the `__init__.py` fix): `INPUT_TAGGERS`
  (`{"alignment": _tag_alignment}`, tagger signature `(inputs, axes=None)` — has the second
  arg), `SUPPORTED` (lazy, all four axes: dtype / layout / topology / alignment), `EXCLUSIONS`
  (`[]`), and `validate()`. `validate()` checks structural fabric/mesh preconditions, then the
  per-axis SUPPORTED membership, then EXCLUSIONS; the public entry point calls `validate()`
  before any device work (only a `topology=None → Linear` default resolution precedes it).
- **Confirmed the op file does NOT declare an `INVALID` symbol** — the only occurrence of the
  word is in a docstring. INVALID is sourced from `feature_spec.py`, as required.
- **Refusal exception type.** `validate()` raises `NotImplementedError` for axis/EXCLUSIONS
  refusals and `ValueError` for structural-argument rejections. Note: the registry's typed
  contract module `ttnn.operations._op_contract` (`SupportRefusal` / `UnsupportedAxisValue` /
  `ExcludedCell`) **does not exist in this branch** — importing it raises `ModuleNotFoundError`.
  So `NotImplementedError` is the only viable refusal signal here, and it matches the
  `op_template.py` baseline ("Both raise NotImplementedError"). When `_op_contract` lands, the
  refusal raises should migrate to the typed classes; until then the current code is correct.
- **No auto-fix to SUPPORTED was applied** — there was no XPASS evidence to act on (no golden
  cells executed; see below). SUPPORTED already equals TARGET on every axis.

### INVALID audit (`eval/golden_tests/point_to_point/feature_spec.py`)

`INVALID = [{"dtype": ttnn.bfloat8_b, "layout": ttnn.ROW_MAJOR_LAYOUT}]` — **well-formed**:
- Single-tensor coupling ✓ — both axes describe the one input tensor.
- Universe-must-change ✓ — bfloat8_b is block-quantized over 32×32 tiles; ROW_MAJOR has no
  tiles, so the data-format definition itself would have to change. Structural impossibility,
  not a "not yet" (which would be EXCLUSIONS).
- Canonical **bf8b + ROW_MAJOR** entry for the (single) tile-or-RM activation is present ✓.
- No cross-tensor-axis coupling (only one tensor) ✓; norm-style no-weight canonicalization N/A
  (no weights) ✓.

## Precision Baseline

`point_to_point` performs **no arithmetic** — it is a bit-exact byte copy. The precision
baseline compares the receiver shard against the **sent** shard (read back through ttnn so both
sides share identical quantization). The analytical expectation for **every** dtype is therefore
exact:

| Shape | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err |
|-------|-----|-------------|--------------|------------------|
| (1,1,32,32)    | — *(needs ≥2 devices)* | — | — | — |
| (1,1,64,128)   | — *(needs ≥2 devices)* | — | — | — |
| (2,3,32,64)    | — *(needs ≥2 devices)* | — | — | — |
| (1,1,128,1024) | — *(needs ≥2 devices)* | — | — | — |

**Expected (analytical, to be confirmed on ≥2-device hardware):** PCC = 1.0, max_abs_err = 0,
mean_abs_err = 0, relative_rms_err = 0 for bfloat16 / float32 / bfloat8_b / uint16 / int32 /
uint32 alike — a correct copy round-trips the already-quantized payload unchanged, so any
non-zero error flags a transfer bug (dropped / duplicated / mis-framed pages), not quantization.

**Assessment:** the test (`test_point_to_point_precision_baseline.py`) is written, collects
cleanly, and **skips on this 1-device machine** (the `mesh_device` fixture auto-skips). It will
populate the table verbatim when run on ≥2 devices.

**Recommended tolerances** (for the copy-fidelity assertion): PCC ≥ 0.9999 for float32/integers,
≥ 0.999 for bfloat16, ≥ 0.99 for bfloat8_b; `assert_equal` (bitwise) for the integer dtypes.

## Verifier CLI Summary

Source: `verifier_report.json` (this directory; also `<results_dir>/verifier_report.json`).

- supported_pass: 0
- xfail_expected: 0
- invalid_skipped: 0
- supported_fail: 0   ✓ (none)
- xpass_drift: 0      ✓ (none)
- xfail_wrong_mode: 0 ✓ (none)
- **total cells: 0**

⚠️ **This is a *no-signal* run, not a clean run.** All categories are 0 because **zero golden
cells executed**, for three independent reasons:
1. `eval/golden_tests/conftest.py` fails to import (`from ttnn.operations._op_contract import
   SupportRefusal` → `ModuleNotFoundError`), which aborts golden collection for *every* op
   (pytest exit 4, 0 collected).
2. The `point_to_point` golden suite is **missing its harness wiring** — the directory contains
   only `feature_spec.py`; there is no `test_golden.py` / `helpers.py` / `conftest.py` /
   `axes.py` (compare `softmax/`, `rms_norm/`). The standard single-`device`-fixture golden
   harness also doesn't fit a multi-device op as-is.
3. Even with (1) and (2) resolved, the transfer needs **≥ 2 devices**.

So the loud categories being empty does **not** confirm SUPPORTED is honest — it confirms the
suite never ran. `TARGET - SUPPORTED = ∅` on every axis, so there are no `xfail_expected` cells
to iterate and no axis-expansion refinements to file (see `op_requirements.md`).

### Existing-test evidence that *did* run (single device, FABRIC_1D)

`scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/point_to_point/` →
**22 passed, 59 skipped**. The 22 passes are the single-device Python-path debug tests
(`ccl_packet_dims`, `_intermediate_spec` incl. the bfloat8_b tile-derived path, landing-buffer
allocation capacity, output allocation, global-semaphore create + address, grid helpers,
`get_fabric_node_id`, and `_supported()` builds without circular-import/enum errors). The 59
skips are the multi-device acceptance + MoE cells (need 2 / 4 / 32 devices). The two new
verifier-authored test files (`test_point_to_point_extended.py`, 18 cells;
`test_point_to_point_precision_baseline.py`, 12 cells) collect cleanly and skip on 1 device.

## Recommendations

1. **Generate the golden harness wiring for `point_to_point` and run it on ≥ 2 devices.** This
   is the single highest-value follow-up. The op claims the *entire* TARGET rectangle at Phase 0
   (`SUPPORTED == TARGET`), and that claim is currently backed only by code review. Producing
   `test_golden.py` / `helpers.py` / `conftest.py` (via `/golden-tests`, adapted for a multi-
   device `mesh_device` fixture + sender/receiver coords) and running it on a ≥2-device mesh is
   what would mechanically confirm the claim. **Blocker:** also requires fixing the missing
   `ttnn.operations._op_contract` module so `eval/golden_tests/conftest.py` imports. Both are
   test-infrastructure / environment items, not op refinements.
2. **When ≥2-device hardware is available, run the acceptance + extended + precision suites.** If
   any SUPPORTED cell fails: a *structural* capability gap (e.g. a genuinely unhandled framing
   regime) → move to `EXCLUSIONS`; an `OOM` / `numerical-precision` failure → file a refinement.
   Until then, do not pre-emptively narrow SUPPORTED — the pure-byte-copy reasoning (every dtype
   is bytes, every layout is pages, both topologies are just routes) makes the full-rectangle
   claim defensible for this op class, and bf8b+non_tile_aligned copies full padded tile pages
   bit-exactly (no compute, so it does **not** need the usual `/numeric-formats-metal`
   bf8b+non-aligned exclusion).
3. **Multi-link / multi-core fan-out** (`op_design.md` fixes `link_idx = 0`, single worker core)
   is a **throughput** optimization — it unlocks no golden cell (single-link already handles
   every shape), so it is **not** a refinement-queue entry. Note it here as future perf work; it
   would need a real inter-core/multi-link distribution strategy (no current skill covers the
   fabric multi-link split).
4. **Sharded memory configs** are explicitly rejected by `validate()` (rule 5) and there is **no
   `memory_config` axis in TARGET**, so sharding is outside the golden universe — not a queue
   entry. If sharded p2p is ever desired, it starts with adding a `memory_config` axis to
   `feature_spec.py`.
5. **Drop the redundant first `synchronize_device`** only after confirming on real hardware that
   command-queue ordering already guarantees the semaphore is initialized before `generic_op`
   runs. Minor host-side latency win; correctness-sensitive, so do not do it blind.

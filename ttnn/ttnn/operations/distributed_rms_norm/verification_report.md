# Verification Report: distributed_rms_norm

Box: Blackhole QuietBox 2 (4 chips). Golden and acceptance ran on the system 2×2 mesh, plus the 1×4 / 4×1 snake-ring
mocks under `FABRIC_2D_TORUS_XY` with the production 14 KiB router payload. Perf and precision ran on the perf case's
1×4 ring (G = 4, Ring, 2 links).

## Code Review

Every item below was **fixed in place**. Kernels were re-validated with `run_safe_pytest.sh --dev` (watcher, LLK
asserts).

1. **Kernel correctness bug: ragged block × multiple W chunks × γ** (`kernels/distributed_rms_norm_compute.cpp`).
   - Cause: `cb_x_gamma` is an `R·C`-page ring. On a core with fewer rows than `R` (a ragged split, e.g. 3 vs 2
     tile-rows per core), each W chunk pushed and popped only `R_b·C` pages. The next chunk's x·γ window then started
     mid-ring, and the scaling chain reads it by tile index from the read pointer with no wrap.
   - Effect: under `--dev`, an LLK assert ("Indexed tile read exceeds CB boundary", `llk_unpack_AB`) and a hang.
     Without `--dev`, **silently wrong output** (PCC ≈ 0). Found on the translated DeepSeek `prefill-8192`
     (8192×3584 per device, R1 with 2–3 rows per core, Wt = 112, 4 chunks).
   - Fix: the design's own rule, a nominal push count. After each chunk of a ragged block, compute pushes and pops
     the missing `(R − R_b)·C` filler pages, so every chunk's window starts at the ring base.
   - Regression test: `test_distributed_rms_norm_debug.py::test_ragged_block_chunked_gamma` (`(1,1,7040,1152)`;
     TILE and RM γ × both DEST modes). **Verified failing (assert + hang) with the fix disabled.**
2. **Persistent exchange resources pinned L1 for the rest of the session** (`distributed_rms_norm_program_descriptor.py`).
   - Cause: Phase 0 cached one port landing (L1 tensor) and one set of global semaphores **per port placement**.
     Sharded inputs move the ports off the shard grid, so every new shard geometry created another set. Each set was
     created mid-call, below the caller's L1 tensors, and never freed.
   - Effect: largest-contiguous-free L1 dropped from 1.46 MB to 0.92 MB after the first sharded call while only
     16.5 KB was allocated, and more sets lowered it further. In the long golden session **~340 sharded cells were
     refused** ("does not fit one resident tile-row") although they pass in a fresh process.
   - Fix:
     - the landing is now **allocated per call**, like an op intermediate, over the whole grid, and freed right after
       the dispatch;
     - the global semaphores are **one grow-only set per mesh over the whole grid**.

     Why per call is safe:
     - mesh allocation is lockstep, so every chip uses the same address;
     - the cross-call ready fence keeps every peer write after the receiver's program for the call has started;
     - the roots consume every arrival before the program ends.

     After the fix, L1 returns to 1.46 MB free after every call. Device time is unchanged (`ccl_perf compare`: null on
     all three perf cases).
3. **Long sequences crashed.** `prefill-32768` hit `AssertionError: CBs collide with the landing scratch`;
   `prefill-131072` hit an allocator `Out of Memory`.
   - Fix: the plan is re-solved against the landing's real bytes.
   - Fix: the design's mechanism cap "refuse above the port L1 budget" was never implemented. It is now
     `PORT_LANDING_MAX_BYTES` (512 KiB), a clean `NotImplementedError` before allocation.
   - Result: 32768 passes; 131072 is refused (Refinement 4).
4. **EXCLUSION `{bfloat8_b, w_non_aligned}` was carried without measurement.** The op already has a compute-side
   W-pad mask (a partial reduce scaler on the last tile, built for R3-s, where the caller's shard cannot be zeroed).
   It never needs the pad zeroed, which is exactly what Bfp8_b cannot do. The mask now also runs for interleaved
   Bfp8_b inputs with a non-aligned W (host-only change), and the EXCLUSION is removed. All 720 golden cells pass.
5. **DRY (blocking-model fidelity).** `footprint()` (the L1 solve) and the CB descriptors restated every CB page
   count independently. Both now read one list, `cb_inventory()`, so a knob change lands in one place.
6. **SUPPORTED drift.** `num_links=None` (every usable link) was always accepted: `validate()` resolves it before the
   gate. But the eval's call classifier records the literal `None`, so every default-links call read as unsupported:
   35 `xpass_drift` and 226 `xfail_wrong_mode` in the first run. `None` is now declared in `SUPPORTED["num_links"]`.
7. **Ledger / design currency.**
   - `l1_ledger.md` had stale `cb_dev_partial` (`2·Rw` vs `nwpb·Rw`), `Rw`, `num_waves`, depth rows and a total that
     still sized `cb_sumsq_partial` by `R`. All are fixed, and the landing and semaphore rows are rewritten.
   - `op_design.md` → Implementation notes now records each verifier change.

Not changed (reviewed, conformant):
- Helper usage: every compute phase is a kernel-lib helper (`eltwise_chain`, `reduce`, `calculate_and_prepare_*`).
  The raw dataflow is the column-0 record extract/scatter, the fabric port and the non-blocking event loop. The
  design lists why no helper expresses these, and they stay within the scope boundary (mechanism is not binding).
- `TensorAccessor` everywhere, `void kernel_main()`, `api/dataflow/...` includes.
- CB push/wait counts balance. The one imbalance was the ragged `cb_x_gamma` above.
- Broadcast dims: Row for γ, Col for inv_rms (column-0-valid tiles).
- Sharded paths consume the local shard zero-copy (`cb_descriptor_from_sharded_tensor`), never through an accessor.
- Work distribution fills the fused grid: 100 / 106 cores at the focus shape (R3-i 10 × 10), and 64 cores at
  2048×1280 (R1). Decode uses 14 cores by the base op's measured decode optimum.
- Expression check: reader batches V tiles per barrier per tile-row (row = wave unit), writer flushes per C chunk, and
  compute runs one chain per wave/chunk. No per-unit handshake inside a block operation.

## Registry Conformance

- `INPUT_TAGGERS` (`alignment`, `rank`, both `(inputs, axes)`), `SUPPORTED`, `EXCLUSIONS`, `validate()` present and
  wired. `validate()` runs first in the entry point: ValueError hard checks, then `UnsupportedAxisValue` per axis, then
  `ExcludedCell`, then regime refusals (`NotImplementedError`). The op file does **not** declare `INVALID`.
- SUPPORTED == TARGET on every axis, plus `num_links: None` (auto-fix above). EXCLUSIONS = `[{float32,
  fp32_dest_acc_en=False}]`. That combination is natively rejected per `precision_convention.md`: an fp32 input through a
  16-bit DEST. It is a **documented omission**, not a refinement candidate.
- **INVALID audit** (`feature_spec.py`): only the 6 gamma "none" canonicalization entries, all single-tensor
  (γ's own dtype/layout coupled to `gamma_mode`). No cross-tensor coupling; no "not yet" entries. The canonical
  bf8b × ROW_MAJOR activation entry is not needed: the activation layout TARGET is TILE only, and `gamma_dtype` has
  no bfloat8_b, as the spec's comment says. Well-formed; no change requested.
- **Prompt rules** (`eval/prompts/distributed_rms_norm.txt` → Rules): MUST be one fused native implementation
  (`generic_op`, own kernels, fabric transport), never another norm/CCL op or a composite. Conformant: one
  `ttnn.generic_op` per call, and the per-call landing is allocated on the host (not a dispatch).

## L1 Ledger Audit

- **Currency**: every declared CB has a row, and every size expression now matches `cb_inventory()`, which is also
  what the descriptors allocate. Stale rows were fixed (see Code Review 7).
- **Capacity vs live set**:
  - `cb_partial_col`, `cb_rms_root`: capacity `R`, live `Rw`. Justified: compute may finalize/collapse wave w+1 while
    the reader is still publishing or distributing wave w (one compute thread, the reader is the event loop).
  - `cb_dev_partial`: `nwpb·Rw` (`partial_depth`), justified the same way.
  - `cb_x_gamma`: `R·C`; it spans R by design (chunk 0 is hoisted for the whole block as the exchange's stall
    shadow).
  - `cb_gamma_tiles`: resident `Wp`. This is the term that fails the wide HEIGHT_SHARDED cells; it is **folded into
    Refinement 1** (the base's streaming fallback), not filed separately.
  - No unjustified over-capacity found.
- **Page format vs DEST**: intermediates are Float32 iff `fp32_dest_acc_en` (both directions correct).
  `cb_stats_record` is a dataflow-only uint32 payload. `cb_stats_gather` carries `UnpackToDestFp32` under fp32 DEST.
- **Disjoint lifetimes**: every pair carries a reason, e.g. `cb_dev_partial` vs `cb_partial_col` (different consumer
  protocols, both live on the root) and `cb_stats_gather` vs `cb_gather` (block-wide remote slot offsets).
- **Bounds / closed form**: every capacity is a block extent (`R`, `Rw`, `S`, `C`) × a bounded constant (`G ≤ 8`,
  `G_w ≤ 24`, depths), or the caller's shard. The total is closed-form (`l1_ledger.md`) and equals `footprint()`.
- **Data-movement budget**: present and consistent with the implemented split. x and out each cross DRAM once; γ is
  read once per active core (G-MC deferred with a measured positive reason from the base op); fabric carries
  `(G−1)·T·128 B` received per chip. The cheapest-traffic split (R1) is implemented where the base predicate selects
  it, and R3-i is a measured occupancy trade.
- **Block-size defaults**:
  - Interleaved: the coarsest R that fits after spreading over the fused grid. `MIN_BLOCKS_PER_CORE = 1` departs from
    the base's 2, justified by measurement (640×1792 30.9 → 23.9 µs, implementer).
  - Sharded: the whole shard as one block.
- **Per-core footprint**: `F(R, d) = d·R·WS·T_in + R·T_i·(1 + C·γ) + R·T_i·(2+G_w)·R3 + Wp·T_g·γ + 2·C·T_out + scaler +
  Rw·T_i·(1+2·masked) + R·T_i + G·Rw·T_i + (G+1)·Rw·128`.
  - Scales with R: x, inv_rms, x·γ, the R3 combine and dev_partial.
  - Scales with S/Wp: x and γ.
  - Scales with Rw·G: the exchange buffers.
  - Plus a per-call landing `num_waves·G·seg_stride` in every bank.

## Precision Baseline

`tests/ttnn/unit_tests/operations/distributed_rms_norm/test_distributed_rms_norm_precision_baseline.py`.
- Setup: G = 4 Ring (1×4), bf16 TILE γ, HiFi4; worst device of 4.
- Metrics: PCC via `assert_with_pcc`, plus `comp_allclose`; ULP in the output dtype; got/true ratio over elements
  with |ref| > 1 % of max.

| Shape (per device) | dtype | fp32 DEST | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | ULP max / mean | got/true ratio median [p5, p95] |
|---|---|---|---|---|---|---|---|---|
| (1, 1, 32, 1792) | bf16 | True | 0.999998 | 0.0433 | 0.00098 | 0.00186 | 1.01 / 0.271 | 0.9994 [0.9966, 1.0021] |
| (1, 1, 32, 1792) | bf16 | False | 0.999993 | 0.0864 | 0.00328 | 0.00614 | 3.89 / 0.941 | 1.0048 [0.9989, 1.0112] |
| (1, 1, 32, 1792) | fp32 | True | 1.000000 | 0.0156 | 0.00060 | 0.00114 | 5.7e+04 / 1.11e+04 | 0.9991 [0.9980, 1.0001] |
| (1, 1, 32, 1792) | bf8b | True | 0.999954 | 0.0646 | 0.00749 | 0.00966 | 256 / 15.7 | 1.0025 [0.9543, 1.0551] |
| (1, 1, 32, 1792) | bf8b | False | 0.999950 | 0.1544 | 0.01036 | 0.01378 | 259 / 16.3 | 1.0108 [0.9623, 1.0640] |
| (1, 1, 640, 1792) | bf16 | True | 0.999998 | 0.0419 | 0.00095 | 0.00186 | 1 / 0.272 | 0.9994 [0.9966, 1.0021] |
| (1, 1, 640, 1792) | bf16 | False | 0.999993 | 0.1565 | 0.00340 | 0.00622 | 4.5 / 0.958 | 1.0049 [0.9993, 1.0113] |
| (1, 1, 640, 1792) | fp32 | True | 1.000000 | 0.0344 | 0.00060 | 0.00111 | 5.5e+04 / 1.09e+04 | 0.9991 [0.9980, 1.0002] |
| (1, 1, 640, 1792) | bf8b | True | 0.999953 | 0.0840 | 0.00742 | 0.00977 | 257 / 16.2 | 1.0025 [0.9676, 1.0423] |
| (1, 1, 640, 1792) | bf8b | False | 0.999949 | 0.2192 | 0.01009 | 0.01369 | 259 / 16.7 | 1.0104 [0.9745, 1.0506] |
| (1, 1, 2048, 1280) | bf16 | True | 0.999998 | 0.0453 | 0.00097 | 0.00191 | 1.05 / 0.279 | 0.9992 [0.9964, 1.0019] |
| (1, 1, 2048, 1280) | bf16 | False | 0.999993 | 0.1826 | 0.00460 | 0.00808 | 5.29 / 1.32 | 1.0070 [1.0010, 1.0134] |
| (1, 1, 2048, 1280) | fp32 | True | 1.000000 | 0.0268 | 0.00068 | 0.00124 | 5.92e+04 / 1.26e+04 | 0.9990 [0.9978, 1.0000] |
| (1, 1, 2048, 1280) | bf8b | True | 0.999954 | 0.0898 | 0.00724 | 0.00969 | 257 / 14.9 | 1.0026 [0.9649, 1.0456] |
| (1, 1, 2048, 1280) | bf8b | False | 0.999949 | 0.1961 | 0.00966 | 0.01330 | 259 / 15.3 | 1.0100 [0.9720, 1.0533] |
| (1, 1, 128, 8192) | bf16 | True | 0.999998 | 0.0423 | 0.00098 | 0.00183 | 0.987 / 0.269 | 0.9994 [0.9966, 1.0021] |
| (1, 1, 128, 8192) | bf16 | False | 0.999994 | 0.1247 | 0.00238 | 0.00468 | 3.75 / 0.679 | 1.0030 [0.9978, 1.0092] |
| (1, 1, 128, 8192) | fp32 | True | 1.000000 | 0.0254 | 0.00059 | 0.00110 | 5.1e+04 / 1.08e+04 | 0.9991 [0.9980, 1.0002] |
| (1, 1, 128, 8192) | bf8b | True | 0.999954 | 0.0859 | 0.00749 | 0.00969 | 257 / 15.1 | 1.0027 [0.9624, 1.0464] |
| (1, 1, 128, 8192) | bf8b | False | 0.999950 | 0.1632 | 0.00926 | 0.01228 | 259 / 15.5 | 1.0084 [0.9678, 1.0522] |

(fp32 rows: ULP is counted in fp32 ulps, so 5e4 ulps ≈ 3e-3 relative; read `rel_rms`. bf8b: ULP counted at the bf16
mantissa; the shared exponent makes the true bf8b ulp larger.)

**Assessment**:
- **bf16 with fp32 DEST is within ~1 output ULP everywhere.** The cross-device fp32 record path adds nothing
  measurable.
- **Scale-vs-precision triage.** No row has the high-PCC + high-RMS signature, so there is no structural scale bug.
  Two small, explained biases:
  1. **16-bit DEST**: the got/true ratio sits at +0.3 to +0.7 % with a tight p5–p95. Σx² is accumulated and packed
     from 16-bit DEST before the (exact) fp32 widening onto the wire, so it comes out slightly low and inv_rms
     slightly high. The fp32-DEST rows of the same shapes center on 1.000, which rules out a divisor or structural
     error. This is design lamp L9 (keep fp32 DEST for collapse / finalize only).
  2. **fp32 input**: a −0.09 % ratio with rel_rms ≈ 1.1e-3, consistent with the FPU's tf32 operand truncation over
     the three multiplies.

  Neither fails a golden cell, so neither is a refinement (no failing cell to move). Both are recommendations below.

**Recommended tolerances**:
- bf16: PCC ≥ 0.9999, rtol 0.02, atol 0.05 (fp32 DEST) / 0.2 (16-bit DEST).
- fp32: PCC ≥ 0.99999, rtol 5e-3, atol 0.04.
- bf8b: PCC ≥ 0.9995, rtol 0.1, atol 0.25.

## Verifier CLI Summary

Final run on the verified code: `eval/eval_test_runner.sh --no-precompile eval/golden_tests/distributed_rms_norm/` then
`python3 -m eval.verify_supported … ttnn.operations.distributed_rms_norm`. Trimmed report:
`verifier_report.json` (this directory); full report under `generated/eval/drms_verify_final3/`.

| Category | First run (Phase 0 as handed over) | Final (after verifier fixes) |
|---|---|---|
| supported_pass | 6,197 | **10,908** |
| xfail_expected | 2,760 | 2,160 |
| invalid_skipped | 33,696 | 33,696 |
| infeasible_skipped | 320 | 40 |
| **supported_fail** | 3,704 | **133** |
| **xpass_drift** | 35 | **0** |
| **xfail_wrong_mode** | 226 | **1** |

Disposition of every remaining loud cell:

- **supported_fail 133**, none silenced with EXCLUSIONS:
  - **120 → Refinement 1.** L1 fit: the resident γ window on `HEIGHT_SHARDED` W_local ∈ {4096, 8192} shards. The
    no-γ variants pass. These stay failing as tracked capacity gaps, not EXCLUSIONs.
  - **12 → Refinement 2.** `test_simulated_distributed_layernorm`: a sharded input with an output shard spec on a
    different grid. This is a placement regime refusal; the base op refuses it too.
  - **1 → Refinement 4.** `test_ds_distributed_norm[prefill-131072]`: the per-call landing (1 MiB per port core)
    exceeds `PORT_LANDING_MAX_BYTES`.
- **xfail_wrong_mode 1**: `test_translated.py::test_pre_all_gather_non_welford_fp32_requires_fp32_dest_acc`. This
  translated test *asserts* that the op refuses `{float32, fp32_dest_acc_en=False}`. The op does refuse it
  (`ExcludedCell`), so the test body passes. Then the eval's L1-profiling plugin fails the test because a refused call
  allocates nothing (`peak_total=0`). This is a harness interaction; the op behaves exactly as the registry declares.
- **xfail_expected 2,160**: all of them are the `{float32, fp32_dest_acc_en=False}` EXCLUSION (natively rejected;
  documented omission above). No `TARGET − SUPPORTED` gap remains, so no xfail bucket lacks a refinement.
- **infeasible_skipped 40**: `HEIGHT_SHARDED` fp32 `(1,1,32,8192)` / `(128,8192)`. The harness cannot place the
  input + output shards in one core's L1 (setup OOM, uncharged).

**Harness findings for the user** (not op defects):
1. The first run's 3,364 + 212 failures were **all** `L1ProfilingError: peak_total=0`. The graph tracker records no
   `circular_buffer_allocate` node for a `generic_op` `MeshWorkload` (only `circular_buffer_deallocate_all`). So an
   op whose only per-call L1 is CBs reports 0, and every interleaved cell failed after passing numerically (confirmed
   with `EVAL_CAPTURE_L1=0`: those cells pass). With the per-call landing the plugin now sees a positive peak. But the
   recorded `device_l1_peak_bytes` is **the landing only, CBs excluded**, an under-count that comes from the
   framework's CB tracking (`tt_metal/impl/program/program.cpp` `track_allocate_cb` is not reached for this mesh
   workload).
2. The acceptance directory's modules each import the session-scoped `_fabric_mesh_cache` fixture. pytest then
   registers one fixture per module, so running two modules in one session opens a second mesh
   (`!devices_still_open`). Run the modules one session each.
3. Under `run_safe_pytest.sh --dev` (watcher on), the Ethernet cores are left stuck after fabric traffic, so a
   **mesh-shape change inside one session fails** (1×4 → 4×1) and every run ends with a ~80 s reset. Run one mesh
   shape per `--dev` session. All acceptance runs here did so: 121 / 121 pass (87 main + 8 + 8 ring + 12 knobs + 6
   debug).

## Recommendations

- **Perf breakdown** (fastest chip / slowest chip, µs, 5 experiments; exposed transport = op − `DRMS_ABLATE_EXCHANGE`):

  | Case | op | exchange ablated | exposed transport | roofline / target |
  |---|---|---|---|---|
  | 640×1792 (PERF FOCUS) | 23.93 / 25.77 | 20.99 / 21.52 | 2.94 | 9.0 / 11.9 |
  | 32×1792 | 12.03 / 13.61 | 10.44 / 10.56 | 1.60 | 0.45 / 0.6 |
  | 2048×1280 | 44.10 / 47.41 | 34.10 / 34.81 | 10.00 | 20.5 / 27.3 |

  Not dispatch-bound (max gap 0.56–0.59 µs). At the focus case the gap to the roofline is dominated by the read →
  exchange chain → write **serialization** (one block per core). That is why the queue leads with the OVERLAP
  refinement (3), then the TRANSPORT fence/links refinement (5). A local-half refinement (6) is conditional on
  headroom left after both.
- **Refinement ordering rationale**:
  - 1 = the from-base port (first, per the from-base rule; the only base mechanism Phase 0 did not land is the γ
    streaming fallback);
  - 2 = complicated sharding (placement reshards);
  - 3 = perf, OVERLAP;
  - 4 = the landing flow-control scheme change (TRANSPORT);
  - 5 = perf, TRANSPORT (reworks the same landing, so after 4);
  - 6 = perf, local, conditional.

  The perf-1 contract (bf16, fp32 DEST, TILE γ, interleaved, axis 1, Ring, 2 links) is already fully supported, so
  no unlock is needed before slot 3.
- **Precision (no failing cell, no refinement)**:
  - 16-bit DEST biases inv_rms by +0.3–0.7 %. If models run `fp32_dest_acc_en=False` on wide hidden dims, lamp L9
    (an fp32 DEST for `collapse` / `finalize` only) is the lever.
  - fp32 inputs carry ~1e-3 relative error from tf32 FPU operands. An SFPU multiply path for fp32 would be the lever.
- **Fence-free TRANSPORT (Refinement 5) vs the per-call landing**: a fence-free write needs a landing that exists
  before the receiver's program starts, i.e. a persistent one again. It must then be allocated where it cannot
  fragment L1 (see Code Review 2). The refinement entry carries this trade-off.
- **Persistent semaphores**: the remaining persistent state is G × 16 B per bank, created on the first call. If that
  first call runs while large L1 tensors are resident, these 16 B land mid-L1 and split the free space for the
  mesh's lifetime. That's harmless in the golden order, but a model integration should make the first call early, or
  expose a `create_resources(mesh)` hook to run at model init.

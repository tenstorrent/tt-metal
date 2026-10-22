# Self-Reflection: rms_norm

## Summary
- **Blind pass is clean.** 5,548 passed / 0 failed / 0 hangs out of 40,836 test cells: golden 5,429, translated 104 (+2 xfail), regression 15. `verifier_report.json` shows 0 `supported_fail` and 0 `xpass_drift`. SUPPORTED now covers every TARGET value. The only refused combinations are the two EXCLUSIONS.
- **The queue completed.** Phase 0, Refinements 1–6 and 2 perf rounds all ran, and every refinement is `[x]`. Golden stayed green from Refinement 4 on. The one transient `supported_fail` in Refinements 1–3, `(1,1,32,32768)`, was a planned regime refusal that Refinement 4 removed.
- **Most important finding.** It is framework-level, from the perf record. One of the three helper-bypass rows is mis-classified: `Semaphore<>::value()` already provides the non-blocking read the row calls missing. A raw-SFPU bypass from Refinement 6 has no bypass row at all, because only perf-tournament rounds are required to keep the table.
- **Everything else is framework-level and low-volume:**
  - 2 golden cells inherited from the reference suite are marked INVALID but still run;
  - 3 golden cells can never run (dead coverage);
  - 1 verifier routing-table gap;
  - 1 contradiction in the op prompt;
  - 2 rules that live only in `.inl` files or the README, not in the public header.
- No op-level kernel bug is evident.

## 1. Golden coverage → `eval/golden_tests/rms_norm/feature_spec.py`

No blind failures, so there are no failure clusters to project. The findings below are about coverage that is wrong or inert.

**G1 — "For now" INVALID entries contradict the translated suite (`invalid_unexpected` = 2).**
- **What:** `feature_spec.py:100-101` parks `{bfloat8_b, w_non_aligned}` and `{bfloat8_b, h_non_aligned}` in INVALID with the note "author-scoped, COULD eventually be supported". But the reference op's own translated tests exercise bfloat8_b on non-aligned W with `TensorMemoryLayout::WIDTH_SHARDED`. So the golden model calls these cells impossible while the reference treats them as a real use case.
- **Side effect:** `{bfloat8_b, h_non_aligned}` is *accepted* by the op (there is no matching EXCLUSION), yet golden never runs it.
- **Evidence:**
  - `test_translated.py::test_rms_norm_sharded_uneven_multicore_logical_width[w72_c2_nonaligned-bfloat8_b]` and `[w200_c3_nonaligned-bfloat8_b]` both show "op refused … {'dtype': BFLOAT8_B, 'alignment': 'w_non_aligned'}".
  - `rms_norm.py:110-118` has EXCLUSIONS for `w_non_aligned` only.
- **Recommendation:** Remove both bfloat8_b alignment entries from INVALID:
  - `w_non_aligned` becomes an honest `xfail_expected` refinement candidate (the op's EXCLUSION already refuses it cleanly);
  - `h_non_aligned` enters the cartesian product and gets real coverage across all four `TensorMemoryLayout` values.
  - If the cartesian cost is a concern, add one loose case instead: `{"inputs": ((1,1,47,1024),), dtype=bfloat8_b, fp32_dest_acc_en=False, gamma_dtype=bfloat16, layout/gamma_layout=TILE, memory_layout=HEIGHT_SHARDED}`. This only works after the `h_non_aligned` INVALID entry is removed; loose cases are routed through `invalid_reason` too.
- **Confidence:** med.

**G2 — 3 resilience loose cells have been infeasible in every phase since Refinement 4.**
- **What:** `_resilience_cases` (`feature_spec.py:437`) sweeps every placement blindly. With `TensorMemoryLayout::WIDTH_SHARDED`, the harness geometry for these shapes holds all rows on each Tensix core, so the input shard itself overflows L1:
  - `(5,3,928,544)` TILE: 13,920 rows × 32 elements × 2 bytes per core, plus the output shard, is about 1.78 MB;
  - `(99991,64)`: about 6.4 MB per core.
  These cells cost a collect/compile slot and can never produce a signal.
- **Evidence:** `golden_refinement_4`, `golden_refinement_5` and `golden_blind_final` all show the same three `test_op_loose[5x3x928x544-…WIDTH_SHARDED]` / `[99991x64-…WIDTH_SHARDED]` (TILE + ROW_MAJOR) cells as "INFEASIBLE_L1: input shard: TT_FATAL @ bank_manager.cpp:495".
- **Recommendation:** In `_resilience_cases`, skip `WIDTH_SHARDED` when `prod(shape[:-1]) × 32 × elem_bytes` exceeds about half of L1. Alternatively, pin a 2-D `BLOCK_SHARDED` geometry for these shapes, which is the adversarial intent anyway.
- **Confidence:** high (the evidence is deterministic); impact is low.

Not a gap: the 18 cartesian `FLOAT32 × HEIGHT_SHARDED × W=8192` infeasible cells fail on the *output*-shard allocation. Input and output are each 1 MiB per Tensix core, and placement must match. `eval/oom.py:69` deliberately counts that as `INFEASIBLE_L1`.

## 2. SUPPORTED honesty → `rms_norm.py` `SUPPORTED` / `EXCLUSIONS`

- **`supported_fail` = 0, `xpass_drift` = 0** (confirmed in `golden_blind_final/verifier_report.json`). Nothing to fix, demote or promote.
- **`xfail_expected` = 1,365.** Every one is `{dtype=FLOAT32, fp32_dest_acc_en=False}`: 39 combinations of the remaining axes × 35 shapes. That matches EXCLUSION `rms_norm.py:112`, the native rejection from `precision_convention.md`, so it is honest.

**S1 — One claimed combination is barely pinned (`{bfloat8_b, h_non_aligned}`).**
- **What:** The op claims bfloat8_b with non-aligned H on every placement. The only test is one unit-test shape, `(1,1,47,1024)`, on `TensorMemoryLayout::INTERLEAVED` (`test_rms_norm_numeric_config.py:67-72`). It was written in Refinement 2. Later work changed the code paths this combination runs through, and none of it re-tested the combination:
  - the sharded paths (Refinements 4/5);
  - the compute-side W-pad mask;
  - the srcB-hold block-float address path (Perf 2).
  The block-float hold path already had a bfloat8_b-specific bug that a bench caught, not golden: perf-part-optimizer breadcrumb "W-counter advance wrong for bfp8_b srcA".
- **Recommendation:** Do G1. Until then, consider one sharded bfloat8_b `h_non_aligned` unit pin.
- **Confidence:** med.

## 3. Helper / reference docs

#### Helper gaps (perf)

| helper | claimed | verdict | evidence | proposed fix |
|---|---|---|---|---|
| `Semaphore<>` (`api/semaphore.h`) | capability: "only blocking waits exist; needs non-blocking `try_wait` / value read" (Perf 1) | **undocumented** | `api/semaphore.h:179` `std::uint32_t value() const` is "The settled current value". On DM it is a plain load, non-blocking, and it predates the run (ancestor of `6a97f55e290`). The site still does a raw `*(volatile uint32_t*)get_semaphore(id)` at `rms_norm_reader.cpp:451`, and the comment at `:447-449` states the false premise. The other half, `CircularBuffer::pages_available_at_front`, *is* a helper. So this row is no bypass at all. | Replace `sem_value` with `Semaphore<>(id).value()`; no ns change expected. In the class summary block (`api/semaphore.h:17-40`), name `value()` as *the* non-blocking probe for poll loops. |
| `ckl::BinaryFpu` + `BroadcastDim::Row/Col` (`eltwise/core/chain.inl`) | capability: srcB re-unpacked every tile (Perf 2) | **missing** | `chain.inl:1203` runs `llk_unpack_AB_init<bt>`, then `chain.inl:1249-1253` runs `mul_tiles_bcast<bt>` per tile. No `InputSpec` / `BroadcastInputSpec` field (`chain.hpp:317-363`) controls srcB lifetime. Real win: 15,685 → 14,801 ns, bit-exact. "Stock per-tile LLK without the chain" measured flat (15,732 ns), so the gain is the hold, not chain overhead. | Add a srcB-hold option to `BroadcastInputSpec`, with the walk order chosen by the helper. See sketch A. |
| (none): face-subset tile transport | capability: no helper writes faces 0/2 only (Perf 1) | **missing** | `SenderPipe::send(src, dst, size)` is one region per call (`mcast_pipe.hpp:92`), and no unicast tile helper exists. The raw sequence (`rms_norm_combine.hpp:28-38`) is plain Device-2.0 dataflow API, not LLK, so the cost of going raw was small. Win: 22,723 → 18,615 ns. | Low priority. A faces-mask variant of a tile write (sketch B). The more reusable part is the *contract*; see H2. |
| **unrecorded:** stock SFPU unary ops (`rsqrt_tile`, `mul_unary_tile`, `add_unary_tile`) | none (no bypass row: introduced in Refinement 6, a `Type: perf` refinement, not a perf round) | **missing** | `rms_norm_compute.cpp:330` calls raw `_llk_math_eltwise_unary_sfpu_params_(finalize_col0_body, dst, VectorMode::C, …)` with a custom SFPI body. The stock API hard-codes `VectorMode::RC` (`rsqrt.h:52`, `binop_with_scalar.h:70`), and `kernel_lib/eltwise` has no `VectorMode` at all (grep: 0 hits). The claim holds, but it is recorded only inline (`rms_norm_compute.cpp:73-82`, `changelog.md:311`), not as a bypass row. | Record it in a bypass table (see P2). Offer a column-scoped affine + rsqrt finalize (sketch C). |
| `McastArgs::sender()` / `receiver()`: absent role (Perf 1 friction, no bypass) | ergonomics: static_assert in untaken branch, so the kernel needs a generic lambda and a `run_blocks` template | **undocumented** | `mcast_args.inl:64-74` `optional_sender()` returns `nullopt` with no assert when `!sender_available`, and `optional_receiver()` likewise. Both are documented in `mcast/README.md:20` but missing from the header's example call sites (`mcast_args.hpp:38-52`). The rms_norm kernels never call them (grep: 0 hits). | Add the `if (auto s = mc.optional_sender(noc)) s->send(…)` pattern to the `mcast_args.hpp` example block. |
| `McastArgs::sender()`: NoC mismatch (Perf 2 friction, no bypass) | ergonomics: one channel on two kernel descriptors with different NoCs cannot compile | **too-hard** | Even `optional_sender()` calls `sender()`, which `static_assert(sender_noc == noc_index)` (`mcast_args.inl:46`). The op had to add `kernels/rms_norm_mcast_roles.hpp` (49 lines) with inert stand-ins. The two ns values are equal: zero speed was gained, and the whole cost was working around the API. | Make `optional_sender()` return `nullopt` when `sender_noc != noc_index` (sketch D). |

```cpp
// A — srcB hold (derived from hold_mul_row/hold_mul_col(rows, cw, a_base, a_rs, b_base, o_base, o_rs))
ckl::input(cb_gamma_tiles, ckl::BroadcastDim::Row, ckl::WaitPolicy::None, ckl::PopPolicy::None,
           ckl::InputTileMapping::Row, ckl::DataFormatReconfig::Enabled, ckl::TileAddressing::Direct,
           ckl::SrcBLifetime::HoldAcrossWalk);   // new; Row -> helper walks column-major, Col -> row-major
// call site unchanged otherwise:
ckl::eltwise_chain(ckl::grid(rows, cw), ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, A, GammaHeld, ckl::Dst::D0>{}, out);
```
```cpp
// B — face-subset write (confidence: low, single call site)
noc_async_write_tile_faces<TILE_BYTES, Faces::F0 | Faces::F2>(src_l1, get_noc_addr(x, y, dst_l1), n_tiles);
```
```cpp
// C — column-0 fused finalize, inv = rsqrt(x * scale + bias) (confidence: low, single call site)
rsqrt_affine_tile_init();
rsqrt_affine_tile(dst, /*scale_bits=*/inv_w_bits, /*bias_bits=*/eps_bits, VectorMode::C /* or a Col0 scope */);
```
```cpp
// D — role pipes that tolerate an off-NoC kernel (replaces rms_norm_mcast_roles.hpp)
if (auto s = mc.optional_sender(noc)) { s->send(src_l1, dst_l1, bytes); }   // nullopt also when sender_noc != noc_index
```

**H1 — `InputTileMapping` × wait × pop legality exists only in the `.inl`.**
- **What:** `op_design.md:300` specified `input(cb_x_gamma, PerTile, PerTile, InputTileMapping::Block)`. `chain.inl:53-75` (`is_legal_input_policy_for_mapping`) rejects that combination; `Block` takes only PerBlockSize / Cumulative / Upfront / None. The public doc of `InputTileMapping` (`chain.hpp:219-231`) never states the legal combinations.
- **Evidence:** implementer breadcrumb (09:18:16): "Design specified … which chain.inl is_legal_input_policy_for_mapping rejects".
- **Recommendation:** Add a 4-line legality table under `chain.hpp:231`: "Block: PerBlockSize/PerBlockSize, Cumulative/{None,AtEnd}, {Upfront,None}/{None,AtEnd}; Scalar: wait ∈ {None, PerTile, Upfront}; …".
- **Confidence:** high.

**H2 — The REDUCE_ROW "faces 0/2 are the only valid faces" contract is implicit.**
- **What:** Perf 1 relied on it: halving combine bytes, proven with NaN-poisoned faces 1/3 (`rms_norm_combine.hpp:9-17`). `reduce_helpers_compute.hpp:176-201` says only that the result is "column-0-valid". It never says which *faces* are valid, or that the `BroadcastDim::Col` srcB unpack reads faces 0/2 only.
- **Recommendation:** Add one line next to `reduce_helpers_compute.hpp:176`: "A REDUCE_ROW result is valid in column 0, i.e. faces 0 and 2; faces 1/3 may hold anything, and every lane-wise or Col-broadcast consumer ignores them."
- **Confidence:** med.

**H3 — The atomic barrier is required at kernel exit, not just before a dependent operation.**
- **What:** Perf 1 found a latent race: "the reader exited with its last semaphore increments still in flight, which hangs under `--dev`" (`changelog.md` Perf 1, `combine_bytes`). The fix was `noc_async_atomic_barrier()` at kernel end (`rms_norm_reader.cpp:940`). The references state the rule only as "before the next operation that depends on the signal" (`static-analysis-checklist.md:298`, `semaphore-debugging-strategy.md:357`). That reads as *not needed* when nothing local depends on the increment.
- **Recommendation:** Add to `semaphore-debugging-strategy.md` Pitfall 1: "Also required before kernel exit: an in-flight `noc_semaphore_inc` with no local consumer still needs `noc_async_atomic_barrier()`."
- **Confidence:** med.

**H4 — BRISC NoC atomics trip a `--dev` assert. Absent from all references.**
- **What:** A writer-side (BRISC) `noc_semaphore_inc` trips `noc_assert_targ_addr_mid_clear`, because firmware leaves a PCIe MID value in that command buffer. It cost a hang-triage cycle and forced the token hand-off to NCRISC.
- **Evidence:** perf-part-optimizer breadcrumb (`hang_fix`, 18:39:10); grep of `.claude/references` for "mid_clear" gives 0 hits.
- **Recommendation:** Add a line to `ttnn-riscv-debugger-reference.md`: "`noc_assert_targ_addr_mid_clear` on BRISC atomics = firmware MID residue; issue atomics from NCRISC."
- **Confidence:** med (it may be a firmware bug to fix instead).

Not a doc gap: the Refinement 6 ragged-owner ring-wrap bug (commit `8d730c7e57a`) violated a rule the reference already states (`ttnn-cb-memory-fundamentals.md:85`, "CB Ring-Wrap Invariant"). Golden loose tall shards (`3232x96`, `4064x160`, `99991x64`) caught it, as intended.

## 4. Agent prompts → `.claude/agents/*.md`, `eval/prompts/`

**P1 — The verifier's `supported_fail` routing table has no regime-refusal row.**
- **What:** `incremental-verifier.md:215` routes OOM / precision / bug / hang / compilation. It does not cover a predicate-guarded `NotImplementedError` (a residency fast-path refusing out-of-domain shapes) on a cell that a refinement has just brought into SUPPORTED. The verifier had to infer "OOM-class, keep failing".
- **Evidence:** verifier breadcrumb (09:46:31). `test_op_loose[1x1x32x32768-…]` was `supported_fail` (validation) in `golden_refinement_1..3`, and `op_requirements.md` Refinement 1 notes "Expected residual failing cell … Do not add it to EXCLUSIONS".
- **Recommendation:** Add a row: "regime-refusal `NotImplementedError` on an in-SUPPORTED cell = resource boundary; route like OOM (stays failing, owned by the refinement that builds the regime; never EXCLUSIONS or a shape tagger)."
- **Confidence:** high.

**P2 — Bypass tables are required only in perf-tournament rounds, so `Type: perf` refinements bypass helpers unrecorded.**
- **What:** Only `perf-coordinator.md:226-264` mandates `### Helper bypasses`. Refinement 6 (`Type: perf`) added the raw-SFPU finalize (`rms_norm_compute.cpp:330`), which appears in no table. See §3, "unrecorded" row.
- **Recommendation:** In `incremental-verifier.md` (perf-refinement "Done when") and `ttnn-implementer.md`, require the same `### Helper bypasses` table, or "— none", in the changelog entry of every `Type: perf` refinement.
- **Confidence:** high.

**P3 — The op prompt tells the implementer to register tensor-property axes in `INPUT_TAGGERS`.**
- **What:** `eval/prompts/rms_norm.txt:92` says "INPUT_TAGGERS: `tag_gamma_dtype` / `tag_gamma_layout` MUST return 'none'…". But `eval/feature_matrix.py:112-113` drops every `INPUT_TAGGERS` key from the cartesian product, and taggers receive only shapes. Following the prompt would silently collapse every real `gamma_dtype` / `gamma_layout` cell to "none". The implementer caught this (`rms_norm.py:73-75` comment; breadcrumb 09:18:16). A less careful run would have lost the whole gamma grid.
- **Recommendation:** Reword to "classify `gamma_dtype` / `gamma_layout` in `validate()` (the 'none' sentinel when absent); do NOT register them in `INPUT_TAGGERS`; only shape facets belong there". Add the same one-line rule to the prompt-authoring guidance for every op with optional tensors.
- **Confidence:** high.

**P4 — The planner emitted an illegal chain policy combination.**
- **What:** Same evidence as H1 (`op_design.md:300`).
- **Recommendation:** In `incremental-planner.md`, add: "check every `(InputTileMapping, WaitPolicy, PopPolicy)` against `chain.inl` `is_legal_input_policy_for_mapping`; streaming per-tile inputs use `Scalar`." H1 makes this check cheaper.
- **Confidence:** med.

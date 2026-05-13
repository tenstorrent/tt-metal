# Eltwise Helper — Gate-2 Test Plan

Companion to `eltwise_helper_proposal.md` (Gate-1 approved, commit `678abc7fd5c`). Per `llk_helpers_hq.md` Gate 2: **BLOCKING — no test added, removed, skipped, retitled, tolerance-changed, parameterization-changed, dtype-matrix-changed, or XFAIL/PASS-flipped until this plan is explicitly approved.**

Existing eltwise validation kernels live at `ttnn/cpp/ttnn/kernel_lib/tests/eltwise/kernels/`. Existing pytests live at `tests/ttnn/unit_tests/kernel_lib/`. All new kernels land in the same kernel dir; the pytest entries land in the existing `test_eltwise.py` (or split per category if growth justifies it — TBD during implementation, not a Gate-2 question).

---

## Pass/fail gates (HQ-derived)

- `num_tiles ∈ {1, 8, 64}` for every new kernel (single tile, fits in default DEST, spans multiple DEST windows).
- PCC ≥ 0.9999 for bf16-only paths.
- PCC ≥ 0.999 when fp32 mixed dtypes participate.
- Blackhole skipped unless the feature is explicitly Blackhole-tested.
- Runner: `scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/kernel_lib/*.py`.

---

## Existing kernels — rebuilt against new APIs

These are the kernels in `tests/eltwise/kernels/` today. They keep their pytest entries; only their kernel sources change to track API renames from items 1, 2, 3, 5.

| Kernel | Items it now covers | Changes from API-rename | Tolerance |
|---|---|---|---|
| `copy_exp_pack.cpp` | items 1 (dispatch), 6 (wait-late no-op on per-tile path) | `Exp::exec_impl()` shape (item 1) | bf16 PCC ≥ 0.9999 |
| `copy_sfpu_pack.cpp` | items 1, 8 (dead-trait drop) | dispatch rename | bf16 PCC ≥ 0.9999 |
| `copy_upfront.cpp` | items 5 (private state), 6 (wait-late in block_path) | `cb_tile_idx_` private; ctor-only writes | bf16 PCC ≥ 0.9999 |
| `binary_fpu.cpp` | items 1, 3 (`Fp32DestAcc::On/Off`) | `Fp32DestAcc::On` replaces `true` | bf16 + fp32_dest_acc PCC ≥ 0.999 |
| `binary_block.cpp` | item 2 (Block→streaming auto-block) | drop `BlockCopyTile`/`BlockBinaryFpu`/`BlockPackTile`; use plain `CopyTile`/`BinaryFpu`/`PackTile` with `AutoBlock::On` default | bf16 + fp32 PCC ≥ 0.999 |
| `pack_block.cpp` | item 2 | drop `BlockPackTile`; use `PackTile` under `AutoBlock::On` | bf16 PCC ≥ 0.9999 |
| `pack_lifecycle.cpp` | items 5, 6 | private state; wait-late equivalence | bf16 PCC ≥ 0.9999 |
| `dest_reuse.cpp` | items 1, 3 | dispatch + Fp32DestAcc | bf16 + fp32 PCC ≥ 0.999 |
| `fanout.cpp` | items 1, 7 (shared-CB CopyTile trait generalisation) | dispatch rename; fanout uses CopyTile pair on shared CB → trait still classifies as hoist-safe | bf16 PCC ≥ 0.9999 |
| `multi_chain.cpp` | items 1, 7 | dispatch rename; same-CB CopyTiles → hoist-safe path | bf16 PCC ≥ 0.9999 |
| `optional_element.cpp` | items 1, 4 (stub-default drop on OptionalChainElement<false>) | dispatch rename; tag fallback rewires | bf16 PCC ≥ 0.9999 |
| `inplace_accumulate.cpp` | items 1, 5 | dispatch + private state | bf16 PCC ≥ 0.9999 |
| `fill_scalar.cpp` | item 1 | `FillScalar::exec_impl()` | bf16 PCC ≥ 0.9999 |
| `conv_unary.cpp` | items 1, 3 | dispatch + Fp32DestAcc | bf16 + fp32 PCC ≥ 0.999 |

Existing parameterization stays: `num_tiles ∈ {1, 8, 64}` plus per-test special cases already in the pytest files. No tolerance change, no skip change. Re-running the existing suite after each API rename commit gates that commit.

---

## New kernels

### Test K1 — `dispatch_contract.cpp` (item 1)

**Purpose:** confirm unified `exec(uint32_t i)` dispatch handles static-SFPU + runtime-param-SFPU mixed in one chain.

**Chain:**
```cpp
eltwise_chain(num_tiles,
    CopyTile<cb_in, Dst::D0, CopyTilePolicy::WaitAndPop>{},
    Exp<>{},                                      // static exec_impl
    MulUnary<Dst::D0>{ pack_fp32_scalar(2.0f) },  // runtime exec_impl with param
    PackTile<cb_out, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{}
);
```

**Golden:** `torch.exp(x) * 2.0`.

**Shape:** `(1, 1, 32, 32 * num_tiles)`.
**`num_tiles`:** {1, 8, 64}.
**Dtype matrix:** `bf16` → PCC ≥ 0.999 (Exp on bf16 + scalar mul compounds rounding). `fp32_dest_acc_en=True` → PCC ≥ 0.999.
**Skip:** Blackhole.

---

### Test K2 — `auto_block_dest_packed.cpp` (item 2)

**Purpose:** confirm `AutoBlock::On` picks `BlockSize = DEST_AUTO_LIMIT / lane_width` and the chain runs correctly with multi-lane DEST utilisation.

**Chain:**
```cpp
eltwise_chain(num_tiles,
    CopyTile<cb_in, Dst::D0, CopyTilePolicy::WaitAndPop>{},
    Exp<>{},
    PackTile<cb_out, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{}
);
```
Lane width = 1; expected `BlockSize` = 16 (bf16) / 8 (fp32_dest_acc).

**Validation:** assert chain processed `n_tiles / BlockSize` outer iterations (visible via DPRINT-instrumented counter) AND output PCC vs `torch.exp(x)`.

**Shape:** `(1, 1, 32, 32 * num_tiles)`.
**`num_tiles`:** {1, 8, 64}.
**Dtype matrix:** `bf16` PCC ≥ 0.9999. `fp32_dest_acc_en=True` PCC ≥ 0.999.
**Skip:** Blackhole.

---

### Test K3 — `auto_block_off.cpp` (item 2 toggle)

**Purpose:** verify `AutoBlock::Off` reproduces today's per-tile dispatch (BlockSize = 1) for kernels that depend on outer-loop-tile-count matching.

**Chain:** same as K2 but invoked as `eltwise_chain<AutoBlock::Off>(num_tiles, …)`.

**Validation:** outer iteration count == `num_tiles`. PCC vs torch golden.

**Shape:** `(1, 1, 32, 32 * num_tiles)`.
**`num_tiles`:** {1, 8, 64}.
**Dtype matrix:** `bf16` PCC ≥ 0.9999.
**Skip:** Blackhole.

---

### Test K4 — `tail_handling.cpp` (item 2 tail)

**Purpose:** verify `n_tiles % BlockSize != 0` is handled correctly. Default `AutoBlock::On`; pick `num_tiles` values that force a runtime-clamped tail inner iter.

**Chain:** same as K2.

**`num_tiles`:** {1, 3, 5, 7, 15, 17, 33, 65} — deliberately non-multiples of 16 and 8 to exercise tail on both bf16 (BlockSize=16) and fp32 (BlockSize=8) DEST configs.

**Dtype matrix:** `bf16` PCC ≥ 0.9999. `fp32_dest_acc_en=True` PCC ≥ 0.999.
**Skip:** Blackhole.

---

### Test K5 — `lane_width_high_slot.cpp` (item 2 OC-2b)

**Purpose:** confirm correctness when user picks a high `Dst` slot, collapsing BlockSize. Not a perf test — correctness only.

**Chain:**
```cpp
eltwise_chain(num_tiles,
    CopyTile<cb_in, Dst::D7, CopyTilePolicy::WaitAndPop>{},
    Exp<Dst::D7>{},
    PackTile<cb_out, Dst::D7, PackTilePolicy::PerTileReserveAndPush>{}
);
```
Lane width = 8 → BlockSize = 2 (bf16) / 1 (fp32_dest_acc).

**Golden:** `torch.exp(x)`.

**Shape:** `(1, 1, 32, 32 * num_tiles)`.
**`num_tiles`:** {1, 8, 64}.
**Dtype matrix:** `bf16` PCC ≥ 0.9999. `fp32_dest_acc_en=True` PCC ≥ 0.999.
**Skip:** Blackhole.

---

### Test K6 — `multi_cb_copy_no_clash.cpp` (item 7)

**Purpose:** verify the generalised `chain_loads_share_cb_v` correctly forces per-iter init when CopyTile elements use different CBs. Before item 7's fix this chain would silently produce wrong output under the auto-hoist gate.

**Chain:**
```cpp
eltwise_chain(num_tiles,
    CopyTile<cb_a, Dst::D0, CopyTilePolicy::WaitAndPop>{},
    CopyTile<cb_b, Dst::D1, CopyTilePolicy::WaitAndPop>{},
    AddBinary<Dst::D0, Dst::D1, Dst::D0>{},        // SFPU-binary, DestOnly
    PackTile<cb_out, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{}
);
```

`chain_has_non_copy_tile_fpu_clash_v` = false (AddBinary is DestOnly). `chain_copy_tiles_share_cb_v` = false (cbA ≠ cbB). Hoist gate fails → per-iter init.

**Golden:** `a + b`.

**Shape:** `(1, 1, 32, 32 * num_tiles)` each input.
**`num_tiles`:** {1, 8, 64}.
**Dtype matrix:** `bf16` PCC ≥ 0.9999. `fp32_dest_acc_en=True` PCC ≥ 0.999.
**Skip:** Blackhole.

**Compile-time control test:** add a `static_assert(!chain_is_hoist_safe_v<EltwiseChain<…>>)` in the kernel that fails to compile if the trait misclassifies this chain as hoist-safe. (Verifies the trait at compile time alongside the device PCC check.)

---

### Test K7 — `mask_op.cpp` (item 10)

**Purpose:** verify `Mask<DF, DataSlot>` honours the `mask_tile` LLK contract (mask at `DataSlot + 1`) and integrates into the chain.

**Chain:**
```cpp
eltwise_chain(num_tiles,
    CopyTile<cb_data, Dst::D0, CopyTilePolicy::WaitAndPop>{},
    CopyTile<cb_mask, Dst::D1, CopyTilePolicy::WaitAndPop>{},
    Mask<DataFormat::Float16_b, Dst::D0>{},
    PackTile<cb_out, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{}
);
```

Lane width = 2 (Mask touches D0, D1) → BlockSize = 8 (bf16) / 4 (fp32_dest_acc).

**Golden:** `torch.where(mask != 0, data, torch.tensor(-inf))` (or whatever `mask_tile`'s semantics are — confirm by reading `mask_tile.h` during implementation).

**Shape:** `(1, 1, 32, 32 * num_tiles)`.
**`num_tiles`:** {1, 8, 64}.
**Dtype matrix:** `bf16` PCC ≥ 0.9999.
**Skip:** Blackhole.

---

### Test K8 — `cumulative_wait_block_iter.cpp` (item 12)

**Purpose:** verify `CumulativeWaitPopAtEnd` policy + `BlockIter` index correctly stages a growing window; consumer must start computing on tile 0 before all N tiles have been pushed.

**Chain:**
```cpp
eltwise_chain(num_tiles,
    CopyTile<cb_in, Dst::D0, CopyTilePolicy::CumulativeWaitPopAtEnd, CbIndexMode::BlockIter>{},
    Exp<>{},
    PackTile<cb_out, Dst::D0, PackTilePolicy::UpfrontReservePushAtEnd, PackTileIndexMode::BlockIter>{}
);
```

**Golden:** `torch.exp(x)`.

**Reader-side discipline:** reader pushes tiles in `cb_in` one-at-a-time with a deliberate inter-push stall (configurable; e.g. spin N cycles) to confirm the consumer is NOT blocked waiting for the full N tiles upfront. Validation = correctness + (optional, non-blocking) measurement that wall time is closer to `(num_tiles - 1) * stall_cycles + compute` than to `num_tiles * stall_cycles + compute`.

**Shape:** `(1, 1, 32, 32 * num_tiles)`.
**`num_tiles`:** {1, 8, 64}.
**Dtype matrix:** `bf16` PCC ≥ 0.9999. `fp32_dest_acc_en=True` PCC ≥ 0.999.
**Skip:** Blackhole.

---

## Compile-time-only tests (no device kernel)

These run as `static_assert` probes in the kernel_lib test driver source — no device dispatch, no pytest entry.

### CT1 — Item 4 missing-CB-id

**Purpose:** confirm a tag-stub drop forces a `static_assert` at chain entry when a CbReader op forgets to override `cb_a_id()`.

**Probe:** in a `.cpp` next to the device tests, define a malformed op
```cpp
struct BadCopyTile : compute_kernel_lib::CbReaderTag { /* no cb_a_id */ };
```
and confirm `eltwise_chain(0u, BadCopyTile{}, …)` triggers the named-override `static_assert`. Gate-2 marker: compile-fail test (CI must catch the failure on a sentinel branch; locally verified via `./build_metal.sh`'s `-fsyntax-only` driver if available, otherwise via a `#ifdef` macro guard so the failure is opt-in).

### CT2 — Item 1 missing exec override

**Purpose:** confirm dropping `call()` from the CRTP base forces compile error when an op author forgets to define `exec_impl`.

**Probe:** malformed op
```cpp
template <Dst Slot> struct BadOp : UnaryOp<BadOp<Slot>, Slot> {
    static ALWI void init() {}
    // no exec_impl
};
```
Compile-fail test via the same `#ifdef` guard as CT1.

---

## What is NOT tested

- Item 8 (dead-trait deletion) — no behavioural surface; covered by existing-suite regression.
- Item 11 (Reduce↔Broadcast doc table) — documentation only.
- Cross-helper regression set (`tests/ttnn/unit_tests/kernel_lib/test_helpers_chain_and_binary.py`) — runs unchanged after each commit per HQ infrastructure-regression discipline. Not a Gate-2 line item.
- Multi-chip / Blackhole-only paths — out of scope per `untestable_locally` annotation; CI handoff if any tests need them.

---

## Test ordering (run after each implementation commit)

Maps to the implementation order in `eltwise_helper_proposal.md`:

1. Commit 1 (item 11, doc) — no test run.
2. Commit 2 (items 4 + 8) — run existing suite + CT1.
3. Commit 3 (item 5) — run existing suite.
4. Commit 4 (item 3, `Fp32DestAcc` rename) — run existing suite.
5. Commit 5 (item 1, dispatch contract) — run existing suite + K1 + CT2.
6. Commit 6 (items 7 + 6, traits + wait-late) — run existing suite + K6.
7. Commit 7 (item 2, auto-block + `eltwise_block.hpp` deletion) — run existing suite (with `binary_block`/`pack_block` rewrites) + K2 + K3 + K4 + K5.
8. Commit 8 (items 10 + 12) — run existing suite + K7 + K8.

Per-commit: `scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/kernel_lib/*.py`. No outer `timeout` (HQ rule).

Each commit is bisectable. If any commit's test set regresses, that commit is the bisect target.

---

## Awaiting sign-off

Test plan at `ttnn/cpp/ttnn/kernel_lib/agents/eltwise_helper_test_plan.md`. Awaiting sign-off.

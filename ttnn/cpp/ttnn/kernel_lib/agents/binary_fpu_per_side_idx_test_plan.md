# Test Plan — `BinaryFpu` per-side local-vs-absolute index toggle

Status: **AWAITING SIGN-OFF** (Gate 2).
Companion to `binary_fpu_per_side_idx_proposal.md` (Gate 1: approved).
Pipeline phase: Phase 4 sub-stage 2c / 2d (validation).

---

## 1. Coverage matrix

Three new test kernels + pytests, one regression sweep on the existing baseline.

### 1.1 `binary_fpu_per_side_idx_a_local.cpp` — A streams in chunks, B held upfront

Models the canonical post-allgather gamma-multiply shape:
- A operand (data) — `WaitAndPopPerBlock` + `BlockIter` (chunk-local `j`).
- B operand (weight) — `WaitUpfrontPopAtEnd` + `BlockIter` (absolute `base_tile + j`).
- Pack — `PerBlockReserveAndPush` + `BlockIter`.

Compute: `cb_out[i] = cb_a[i] * cb_b[i]` for `i ∈ [0, num_tiles)`.

### 1.2 `binary_fpu_per_side_idx_b_local.cpp` — symmetric (B chunked, A upfront)

Verifies the mixed combo also works when the per-block side is B.
- A — `WaitUpfrontPopAtEnd` + `BlockIter`.
- B — `WaitAndPopPerBlock` + `BlockIter`.
- Pack — `PerBlockReserveAndPush` + `BlockIter`.

Compute: same `cb_out[i] = cb_a[i] * cb_b[i]`. Torch golden identical to 1.1.

### 1.3 `binary_fpu_per_side_idx_bcast.cpp` — chunked A + row-broadcast B

Models the post-allgather beta-add shape where B is a Wt-wide vector
broadcast across rows.
- A — `WaitAndPopPerBlock` + `BlockIter`, 2D chain (`Ht, Wt`).
- B — `WaitUpfrontPopAtEnd` + `RowBcast`, full `Wt`-tile window.
- Op — `BinaryFpuOp::Add` with `BroadcastDim::Row`.
- Pack — `PerBlockReserveAndPush` + `BlockIter`.

Compute: `cb_out[ht, wt] = cb_a[ht, wt] + cb_b[wt]`.

Note: `RowBcast` index is independent of `i` (returns `wt` directly), so it
sidesteps the local/absolute distinction. Included to confirm the relaxed
asserts don't break the existing bcast paths.

### 1.4 Regression — existing `test_block_binary_fpu` sweep

No change to `binary_block.cpp` or `test_block_binary_fpu`. Confirms zero
perf/behaviour regression on the same-regime path (both sides agree, hits
the 2-arg `exec` forwarder).

## 2. Parameterization

For each of 1.1 / 1.2 (1D chain over `num_tiles`):

| Axis | Values | Rationale |
|---|---|---|
| `num_tiles` | `4, 8, 16, 64` | divisible by all block sizes; covers single-DEST-window and multi-window |
| `block_size` | `2, 4` | both sub-DEST_AUTO_LIMIT; 2 exercises 2-deep DEST lanes, 4 exercises max common |
| `op_name` | `Add, Sub, Mul` | all three FPU binary ops — proves regime toggle independent of op |
| `fp32_dest_acc` | `False, True` | per HQ; verifies `DEST_AUTO_LIMIT` shrink-by-half path |
| `dtype` | `bfloat16` | input dtype; matches existing test_eltwise.py default |

Total: `4 × 2 × 3 × 2 = 48` cases per direction × 2 directions = **96 cases**.

For 1.3 (2D chain over `(Ht, Wt)`):

| Axis | Values | Rationale |
|---|---|---|
| `(Ht, Wt)` | `(1, 4), (2, 8), (4, 16)` | covers tail-iter and multi-row walks |
| `block_size` | `2, 4` | as above |
| `fp32_dest_acc` | `False, True` | as above |
| `dtype` | `bfloat16` | as above |

Total: `3 × 2 × 2 = 12` cases.

**Grand total: 108 new test cases.**

## 3. Acceptance criteria

- All 108 new cases pass with `comp_pcc(...) >= 0.9999` (bf16-only path).
- Existing `tests/ttnn/unit_tests/kernel_lib/test_eltwise.py` baseline: **483 passed, 7 skipped** — must remain identical (no test added, removed, retitled, or XFAIL-flipped on the existing surface).
- Existing `tests/ttnn/unit_tests/operations/eltwise/test_add.py` — **110 passed, 1 skipped** — unchanged.
- Existing `tests/ttnn/unit_tests/operations/eltwise/test_binary_ng_bcast_fp32_dest_acc.py + test_binary_ng_program_cache.py` — **22 passed** — unchanged.
- Existing `tests/ttnn/unit_tests/operations/fused/test_group_norm.py` — pass — unchanged.

## 4. Skip rationale

No tests are skipped on purpose. If the test infrastructure (reader kernel
not handling the per-side scenario) blocks a case, that's a Gate-2 redline,
not a silent skip.

Blackhole arch — not targeted; existing test_eltwise.py auto-skips Blackhole.

## 5. Implementation notes (test kernels)

Each test kernel:
- Same CB layout convention as existing eltwise tests: `c_0` (A), `c_1` (B),
  `c_16` (out).
- Drives the chain via `eltwise_chain<BLOCK_SIZE>(num_tiles, BinElt{}, PackElt{})`.
- Uses the existing test infra (`run_kernel_with_reader`) — no new reader
  kernel needed; reader pushes `num_tiles` tiles to each input CB, the
  chunked policy still works because `cb_wait_front(N)` is a no-op once N
  tiles are already present.

The pytest harness (`test_eltwise.py`) will gain three new test functions
following the existing pattern at `test_block_binary_fpu`:

```python
@pytest.mark.parametrize("num_tiles", [4, 8, 16, 64])
@pytest.mark.parametrize("block_size", [2, 4])
@pytest.mark.parametrize("op_name,torch_op,pcc", [
    ("Add", torch.Tensor.add, 0.9999),
    ("Sub", torch.Tensor.sub, 0.9999),
    ("Mul", torch.Tensor.mul, 0.9999),
])
@pytest.mark.parametrize("fp32_dest_acc", [False, True])
def test_binary_fpu_per_side_idx_a_local(device, num_tiles, block_size,
                                          op_name, torch_op, pcc, fp32_dest_acc):
    if num_tiles % block_size != 0:
        pytest.skip("num_tiles must be divisible by block_size")
    run_kernel_with_reader(
        device,
        kernel_path="ttnn/cpp/ttnn/kernel_lib/tests/eltwise/kernels/"
                    "binary_fpu_per_side_idx_a_local.cpp",
        defines={"BLOCK_SIZE": str(block_size),
                 "BINARY_OP_NAME": op_name},
        num_tiles=num_tiles,
        torch_op=torch_op,
        pcc=pcc,
        fp32_dest_acc=fp32_dest_acc,
        label=f"per_side_idx_a_local {op_name} N={block_size}",
        cb_pages=block_size * 2,  # producer keeps 2 chunks staged
    )

# test_binary_fpu_per_side_idx_b_local — analogous, swap A/B policies in the kernel.
# test_binary_fpu_per_side_idx_bcast — analogous, 2D shape parameterization.
```

## 6. Run sequence (post-Gate-2 sign-off)

1. Write the three test kernel `.cpp` files.
2. Add three test functions to `test_eltwise.py`.
3. Implement the helper changes per the proposal.
4. `./build_metal.sh` clean.
5. `scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/kernel_lib/test_eltwise.py` — confirm 483 baseline + 108 new = **591 passed, 7 skipped**.
6. Spot-check downstream: `test_add.py`, `test_binary_ng_*.py`, `test_group_norm.py`.
7. Report Phase-4 complete; the rmsnorm/layernorm migration commits (Step 4 of HQ migration loop) follow as separate per-kernel commits.

---

Test plan at `ttnn/cpp/ttnn/kernel_lib/agents/binary_fpu_per_side_idx_test_plan.md`. Awaiting sign-off.

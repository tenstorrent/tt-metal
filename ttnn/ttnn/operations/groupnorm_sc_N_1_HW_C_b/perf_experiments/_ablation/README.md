# Perf 1 ablation (cumulative peel) — groupnorm_sc_N_1_HW_C_b

`kernels/` are copies of the op's kernels at the start of Perf 1, with payload stubs behind defines.

> **Stale after Perf 1.** These are the *pre-Perf-1* kernels (reduce/eltwise-chain moments, the x·(x−k) CB, the
> chain finalize and expand, the seed and the scaler). The graduated op has a different CB map and raw-LLK stages
> that `CKL_ELTWISE_CHAIN_SKIP_COMPUTE` / `SKIP_COMPUTE` do not reach (`moments_rows`, `FinalizeStats`,
> `expand_matmul`, `normalize_chunk`). Round 2 must rebuild the kit from the current kernels (hand stubs for those
> four sites) before trusting a peel; use this one only to reproduce the Perf 1 baseline breakdown.
Every stub keeps the CB / semaphore protocol, so trip and transaction counts stay the same.

| define | stubs |
|---|---|
| `GN_ABL_COMPUTE` | all compute payload: `CKL_ELTWISE_CHAIN_SKIP_COMPUTE` + matmul `SKIP_COMPUTE` + hand stubs for reduce / tilize / untilize |
| `GN_ABL_DM` | DRAM chunk reads (reader) and output writes (writer) |
| `GN_ABL_PROLOGUE` | reader step-0: pilot read, map / pilot / seed zero-fill, one-hot stores, shift-row loops |
| `GN_ABL_PEER` | gather writes + semaphore, root semaphore wait, stats mcast send / receive |

To reproduce, the descriptor took two temporary hooks (not committed):

```python
KERNEL_DIR = Path(os.environ.get("GN_KERNEL_DIR", str(Path(__file__).parent / "kernels")))
_ABL = [("GN_ABL_" + f, "1") for f in os.environ.get("GN_ABLATE", "").split(",") if f]
_ABL_C = _ABL + ([("CKL_ELTWISE_CHAIN_SKIP_COMPUTE", "1"), ("SKIP_COMPUTE", "1")] if ("GN_ABL_COMPUTE", "1") in _ABL else [])
# reader / writer KernelDescriptor(defines=_ABL), compute KernelDescriptor(defines=_ABL_C)
```

Results (DEVICE KERNEL DURATION ns, Blackhole, block zones compiled in) are in `changelog.md` under `## Perf 1`.

# Verification Report: glu_fused

## Code Review

### Design conformance (binding dimensions)

Compared the implementation against `op_design.md` along the four binding
dimensions:

| Dimension | Design says | Code does | Match? |
|-----------|-------------|-----------|--------|
| Algorithm | Single fused chain: `Load A → Load B → Sigmoid<Exact>(B) → SfpuMul(A, B)` packed to output, no intermediate L1 round-trip | `glu_fused_compute.cpp:38-42` — exactly this chain via `sfpu_chain` + `sfpu_pipeline` | ✓ |
| Data pipeline topology | Reader NCRISC: 2 reads / output tile (A then B). Compute single Tensix: chain in DEST. Writer BRISC: 1 write / output tile | Kernels match exactly; CB layout (`cb_input_a=0`, `cb_input_b=1`, `cb_output_tiles=16`) matches design CB table | ✓ |
| Parallelization / work distribution | `split_work_to_cores(grid_size, total_output_tiles)`, per-core walk g1→g2, work unit = 1 output tile | `glu_fused_program_descriptor.py:60-67`, `:113-134` — matches | ✓ |
| Inter-core comms | None | None | ✓ |

No bugs found on the binding dimensions.

### Fixes applied

**1. Reader: fused two reads into a single NoC barrier**
(`kernels/glu_fused_reader.cpp`)

The original reader issued `noc_async_read_tile → barrier → push` twice per
output tile — two barriers per iteration. Both A and B reads share NoC0,
so they can be in-flight concurrently and synchronized with a single
barrier. Changed to:

```cpp
cb_reserve_back(cb_input_a, 1);
cb_reserve_back(cb_input_b, 1);
noc_async_read_tile(a_tile_idx, src_accessor, l1_write_addr_a);
noc_async_read_tile(b_tile_idx, src_accessor, l1_write_addr_b);
noc_async_read_barrier();         // single barrier for both
cb_push_back(cb_input_a, 1);
cb_push_back(cb_input_b, 1);
```

This is the canonical two-input reader pattern (mirrors
`backward_softmax_reader.cpp:86-97`). Halves the per-iter barrier count
and lets the two NoC transactions overlap. All 20 acceptance tests still
pass after the change.

**2. Compute: removed redundant `sfpu_pipeline` template parameters**
(`kernels/glu_fused_compute.cpp`)

The call site explicitly listed all four template params:
```cpp
sfpu_pipeline<
    SfpuBatching::Auto,
    SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig::INPUT_AND_OUTPUT>(chain, cb_output_tiles, num_output_tiles);
```

All four match the helper's defaults at `sfpu_helpers.hpp:1406-1412`.
Simplified to `sfpu_pipeline(chain, cb_output_tiles, num_output_tiles)`
with a comment recording which defaults are being relied on. Behavior is
identical. All 20 acceptance tests still pass.

### Helper usage check (after fixes)

| Compute phase | Helper used | Could be better? |
|---------------|-------------|------------------|
| Init | `init_sfpu` | No — canonical SFPU init |
| Load A / B / Sigmoid(B) / Mul(A,B) / pack | `sfpu_pipeline + sfpu_chain` | No — this is exactly the helper's intended use case; single call covers the entire compute body |

| Reader phase | Helper used | Could be better? |
|--------------|-------------|------------------|
| Tile-by-tile DRAM read | raw `noc_async_read_tile + barrier` | No — no helper expresses "read two distinct tile-ids of the same input tensor into two distinct CBs". The design doc explicitly considered and rejected `cb_helpers_dataflow.hpp` and `tilize_helpers_dataflow.hpp` for this case. |

| Writer phase | Helper used | Could be better? |
|--------------|-------------|------------------|
| Tile-by-tile DRAM write | raw `noc_async_write_tile + barrier` | No — no helper for tile-id-ordered drain to DRAM. |

### Other correctness checks

| Check | Result |
|-------|--------|
| CB sync: push count = wait count | ✓ Reader pushes 1 each per iter to A, B; compute waits/pops 1 each per iter; compute pushes 1 per iter to output; writer waits/pops 1 per iter. |
| TensorAccessor (not deprecated InterleavedAddrGen) | ✓ Both reader and writer use `TensorAccessor` with `TensorAccessorArgs`. |
| Kernel entry: `void kernel_main()` not namespace pattern | ✓ All three kernels. |
| Include paths: `api/dataflow/dataflow_api.h` (not bare) | ✓ Reader and writer. |
| `compute_kernel_lib` include path | ✓ Compute uses `ttnn/cpp/ttnn/kernel_lib/sfpu_helpers.hpp` (matches working pattern). |
| Broadcast efficiency | N/A — operation is pointwise, no broadcast. |
| Manual L1↔L1 copies | None — all moves go through CBs as designed. |
| Compile-time arg redundancy | `Wt = 2 * Wt_half` is derived in the reader (constexpr) rather than passed as a separate CT arg — already optimal. |

### Deferred (architectural, not in verification scope)

None. The two fixable findings were fixed in place.

---

## Precision Baseline

Test: `tests/ttnn/unit_tests/operations/glu_fused/test_glu_fused_precision_baseline.py`

Compared against `torch.nn.functional.glu(x, dim=-1)` on `torch.randn` inputs
(`seed=42`):

| Shape | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err |
|-------|-----|-------------|--------------|------------------|
| `(1,1,32,64)`    | 0.9999999999999992 | 1.192e-07 | 4.00e-09 | 2.92e-08 |
| `(1,1,32,128)`   | 0.9999999999999997 | 2.384e-07 | 4.69e-09 | 3.29e-08 |
| `(1,1,256,128)`  | 0.9999999999999997 | 2.384e-07 | 4.71e-09 | 3.36e-08 |
| `(2,2,128,256)`  | 0.9999999999999991 | 3.576e-07 | 4.44e-09 | 3.25e-08 |

**Assessment**: Achieved precision is at the **fp32 floor** — max abs error
is on the order of `1–3 × fp32_eps (≈ 1.19e-7)`, mean abs error is ~3 fp32
ULPs at the typical output magnitudes, and PCC is indistinguishable from
1.0 in fp64. The errors come from (a) the Wormhole sigmoid LUT (6-piece
PLU, not `1/(1+exp(-x))` — the dominant source) and (b) a single fp32
multiply, with no accumulation depth to compound either. The `fp32_dest_acc_en`
+ `UnpackToDestFp32` configuration is correctly preserving fp32 precision
end-to-end.

**Recommended tolerances** (for refinement agents tightening the
acceptance test):
- PCC ≥ 0.99999
- max_abs ≤ 5e-6 (well above the LUT noise floor)
- relative RMS ≤ 1e-6
- The current acceptance tolerances (`PCC ≥ 0.999`, `max_abs ≤ 0.05`,
  `rel_RMS ≤ 1e-3`) are extremely loose — they were sized for the
  general bf16 case. Refinements that introduce lower-precision dtypes
  should keep the loose bounds; the fp32 path could tighten by ~5 orders
  of magnitude without becoming a flake risk.

---

## Test Results

| Suite | File | Result |
|-------|------|--------|
| Acceptance | `test_glu_fused.py` | **20/20 passing** (after fixes) |
| Precision baseline | `test_glu_fused_precision_baseline.py` (new) | **4/4 passing** |
| Extended | `test_glu_fused_extended.py` (new) | **5/5 passing** |
| **Total** | | **29/29 passing** |

Extended test additions (kept small per protocol):
- L1 memory_config preservation (not covered by acceptance — only DRAM tested there).
- Wide-W ascending-value structural check (`W=512`, 8 output tile-cols per row — exercises the per-iter split offset on a wider stride than acceptance does).
- Sigmoid saturation parametrized at `b=±20` (validates the accurate-sigmoid path under saturation; the fast-approx Schraudolph variant would mis-saturate here).
- Determinism (running the op twice on the same input must produce bit-identical output).

---

## Recommendations

Synthesizing findings from the numerical stability analysis and data
transfer analysis (both in this directory):

### From numerical stability

- **`math_fidelity=HiFi4` is dead config** on this kernel. The only multiply
  is `SfpuMul` (SFPU), which is not fidelity-gated. This is documented in
  `numerical_stability.md` and noted in the capabilities table. Refinement
  that exposes a compute config should still default to HiFi4 (for forward
  compatibility if the op is ever extended with an FPU op), but should not
  spend complexity on tuning it.
- **Sigmoid LUT is the dominant error source.** Bounded ≲ 1% absolute error
  is the Wormhole LUT's documented behavior. There is no further mitigation
  short of replacing the LUT with `1/(1+exp(-x))`, which would degrade
  perf significantly. The precision baseline confirms this is well within
  fp32 range for randn inputs.
- **`UnpackToDestFp32` correctly mitigates the TF32 truncation hazard** on
  the SrcA/SrcB unpack path. This is critical for the fp32 baseline; any
  refinement that adds a bf16/bfloat8_b path must NOT enable
  `UnpackToDestFp32` for those low-precision inputs (or the perf would
  suffer with no precision benefit).

### From data transfer

- **Read amplification is already 1.0×** — no cross-core duplication, no
  multicast opportunity. The 2:1 NoC0:NoC1 imbalance is **inherent** to GLU
  (input is twice the output by definition); no refinement can fix this.
- **The reader barrier-fusion fix applied here** removes the only data-flow
  inefficiency.

### Priorities for refinement (mapped to `op_requirements.md`)

| Refinement | Driver | Source of finding |
|-----------|--------|-------------------|
| Compute config exposure | Users want fp32_dest_acc_en / math_approx_mode knobs; also unlocks fast-approx sigmoid as a perf trade-off | Numerical stability analysis (mode currently hard-coded) |
| bfloat16 / bfloat8_b support | Standard model dtype expansion | Capabilities matrix (currently float32-only) |
| Arbitrary rank and `dim` parameter | PyTorch parity (PyTorch supports any rank + any `dim`) | Capabilities matrix (currently rank==4, dim=-1) |
| Non-tile-aligned shape support | PyTorch allows any `W` such that `W % 2 == 0` | Capabilities matrix (currently `W % 64 == 0`) |
| Sharded memory support | Perf-sensitive deployments use sharded L1 | Capabilities matrix (currently interleaved-only) |
| L1 footprint for very large shapes | CB pages are 4096 B × (2+2+2) = 24 KB per core — negligible — but very large per-core tile counts may stress the writer's barrier-per-tile cadence | Data transfer (memory pressure not a Phase-0 issue but listed last per protocol) |

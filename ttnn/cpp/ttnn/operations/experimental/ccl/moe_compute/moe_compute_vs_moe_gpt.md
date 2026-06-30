# `moe_compute` vs `moe_gpt`: Semantics Report

> Scope: expected semantics of `ttnn.experimental.moe_compute` and how it compares
> to `ttnn.experimental.moe_gpt`. Derived from the host API, device op, kernels, and
> golden test references in `ttnn/cpp/ttnn/operations/experimental/ccl/{moe_compute,moe_gpt}`.

## 1. Executive summary

Both ops are **fused Mixture-of-Experts FFN kernels** for the tt-metal CCL stack. They
share the same architectural skeleton (selective-tilize → ring-distributed expert
matmuls → on-chip/cross-device combine), but sit at different points in the codebase's
evolution:

- **`moe_gpt`** (`ttnn.experimental.moe_gpt`, added Mar 2026, PR #40565) is the original,
  **GPT-OSS-specific** implementation. Its geometry is **hard-compiled**:
  `hidden_size = intermediate_size = 2880` (90 tiles), 4 experts/device, a fixed 12-core
  ring, hand-tuned 7/8-tile sharding lookup tables, and a **hardcoded SwiGLU** activation.
  It is Wormhole-oriented and used only by the GPT-OSS throughput demo.

- **`moe_compute`** (`ttnn.experimental.moe_compute`, the generalized successor — roots in
  PR #38049 Feb 2026, generalized via #42982/#45368 and onward) is the **production,
  model-agnostic** op. It supports **arbitrary `(hidden_size, intermediate_size)`** via
  compile-time Euclidean-rhythm (Bresenham) sharding, **three activations**
  (SILU/SWIGLU/GELU), **optional bias**, **shared experts** (tensor-parallel split), a
  **compute-only mode**, **both Wormhole (12-core ring) and Blackhole (8/12/16)**, and a
  real **fused cross-device A2A combine** (`selective_reduce_combine`). It is the op wired
  into the common MoE decode module and DeepSeek-V3.

In short: **`moe_compute` is the generalization of `moe_gpt`.** `moe_gpt` is best
understood as the special case `moe_compute(activation=SWIGLU, hidden=2880,
intermediate=2880, experts/device=4, WH, no shared experts, combine = local gather only)`
frozen into compile-time constants.

## 2. The computation each op performs

### 2.1 Common FFN structure

Both compute a per-token, per-selected-expert two-layer gated FFN over three weight
matrices:

- **W0** — gate projection `[hidden → intermediate]`
- **W1** — up projection `[hidden → intermediate]`
- **W2** — down projection `[intermediate → hidden]`

with `gate = x·W0`, `up = x·W1`, `out = activation(gate, up)·W2`.

### 2.2 Activation — the key numerical difference

**`moe_gpt`** uses a **single hardcoded GPT-OSS SwiGLU** (`swiglu_sfpu.h:40-43`, golden in
`test_moe_gpt_e2e.py:458-461`):

```
gate_c = clamp(gate, max=7.0)
up_c   = clamp(up,  min=-7.0, max=7.0)
out    = (up_c + 1.0) · gate_c · sigmoid(1.702 · gate_c)        # alpha=1.702, clamp_limit=7.0
```

**`moe_compute`** selects the activation at runtime via
`MoEActivationFunction {SILU, SWIGLU, GELU}` (`config.hpp:12`), each verified against a
distinct golden (`test_moe_compute_6U.py:1364-1373`):

| Activation | Formula | Typical model |
|---|---|---|
| `SILU` (default) | `silu(x·W0) · (x·W1)` | DeepSeek-V3 |
| `SWIGLU` | identical clamped GPT-OSS formula above (α=1.702, clamp=7) | GPT-OSS |
| `GELU` | `gelu(x·W0, tanh-approx) · (x·W1)` | Gemma |

So `moe_compute`'s `SWIGLU` path is numerically equivalent to `moe_gpt`'s only activation.

### 2.3 Routing scores and the combine — a critical semantic gap

This is the most important behavioral difference and easy to get wrong:

- **`moe_gpt` does NOT apply routing scores and does NOT reduce across experts.** The
  scores tensor is read only to be copied into the activation-metadata output buffer
  (`tilize_reader.cpp:459`, `tilize_writer.cpp:326`); it is never multiplied into the FFN
  result. `combine_dm1.cpp` is purely a **semaphore barrier** (wait for the 4 ring cores
  in a width column to finish — `combine_dm1.cpp:27-29`). moe_gpt's "combine" therefore
  just **gathers each expert's dense output into a token-parallel × data-parallel sharded
  layout**; the score-weighting and the top-k sum are left to a downstream step. The e2e
  golden confirms this — it checks per-expert `[E·M, K]` outputs, not a score-weighted
  token sum.

- **`moe_compute` (Full mode) performs the real combine.** It emits the unweighted
  per-expert matmul output **and** a separate combine output produced by the fused
  `selective_reduce_combine` device op, which performs the **score-weighted, cross-device
  (along `cluster_axis`) reduction** of the top-k expert contributions per token. In
  `compute_only` mode this combine is skipped entirely.

## 3. I/O contract

### 3.1 Inputs (first four tensors come from the A2A dispatch op in both)

| Tensor | `moe_compute` | `moe_gpt` |
|---|---|---|
| dispatched tokens (sparse) | `tilize_input_tensor`, RM, **bf16**, rank ≥3, `hidden` from last dim | `input_tensor`, rank ≥2 |
| expert indices | `tilize_expert_indices_tensor`, **uint16**, `k` from last dim | `expert_indices`, HEIGHT_SHARDED L1 |
| expert scores | `tilize_expert_scores_tensor` | `expert_scores`, HEIGHT_SHARDED L1 |
| expert→device mapping | `tilize_expert_mapping_tensor`, rank 2 `[devices, experts]` | `expert_mapping` |
| packed gate+up weights | `matmul_w0_w1_tensor`, **rank-6** `[cores, L, E, groups, K, 4·TILE]` | `w0_w1_tensor`, **rank-6**, dim0 == #DRAM banks(=12), dim5 == 128 |
| packed down weights | `matmul_w2_tensor`, **rank-6** `[cores, L, E, groups, N, 4·TILE]` | `w2_tensor`, rank-6, **dim3 fixed == 2** |

Weights are **bfloat4_b**, DRAM-sharded one shard per bank, W0/W1 interleaved, W2
ring-rotated. Reference packers live in `ttnn._experimental.moe_compute_utils` /
`ttnn.experimental.prepare_*` (executable spec of the byte layout). `moe_gpt` has its own
private packers inside its test, valid only for the 90/90/12 geometry.

### 3.2 Key parameters

**`moe_compute`** (kw-only): `layer_id` (req), `output_height_shard_dim` (token-parallel
cores), `intermediate_size` (req), `has_bias`, `num_shared_experts_per_device`,
`activation_type`, `compute_only`, plus combine config (`cluster_axis` req when
`!compute_only`, `topology` {Linear|Ring}, `num_links`, `mux_core_range_set`,
`output_memory_config`, `optional_output_tensor`, `optional_cross_device_semaphore`).
Matmul ring size is **auto-detected** (8 BH / 12 WH); `num_data_parallel_cores` is
auto-derived (largest `d≤4` dividing both `hidden_tiles` and ring size).

**`moe_gpt`**: just `output_height_shard_dim=4`, `output_width_shard_dim=3`,
`hidden_size=2880`, `cluster_axis`. Geometry otherwise fixed.

### 3.3 Outputs

| Slot | `moe_compute` | `moe_gpt` |
|---|---|---|
| 0 | per-expert token counts | per-expert token counts |
| 1 | expert activation metadata (`token_id, k_idx[], scores[]`) | same (+1 sentinel row) |
| 2 | expert→token map (`e_t`, sentinel-terminated) | same |
| 3 | tilize/MM output buffer, TILE layout (bf16, double-buffered) | same |
| 4 | **matmul output** (RM alias of slot 3) — *final output in `compute_only`* | output (RM alias) — gathered per-expert result |
| 5 | **combine output** — score-weighted cross-device reduction (*Full mode only*) | — (no 6th tensor) |

`moe_compute` returns **6** tensors in Full mode, **5** in compute-only. `moe_gpt` always
returns **5**.

## 4. Shared pipeline & how the "ring" works

Both ops run the same 6-kernel pipeline across a tilize-core set + a matmul ring + combine
cores:

1. **`tilize_reader`** — pulls top-k indices/scores/mapping, builds the per-expert token
   lists (`e_t`) and sparse routing metadata, fans token rows in from DRAM.
2. **`tilize_compute`** — `fast_tilize` row-major tokens → 32×32 tiles, chunked at 32
   tokens/chunk.
3. **`tilize_writer`** — multicasts tilized chunks across tilize cores and hands them to
   matmul cores via semaphores.
4. **`dm0`** — streams **W0/W1** weights from DRAM banks (triple-buffered, transaction-ID
   pipelined).
5. **`compute`** — per expert, per chunk: **W0/W1 matmul → activation (SFPU) → W2 matmul**,
   coordinating with dm1 via `cb_c2w_rdy`/`cb_w2c_rdy`; final `pack_untilize` back to
   row-major.
6. **`dm1`** — streams **W2** weights **and** drives the **all-to-all ring**: because each
   ring core owns only a width-shard of the intermediate dim, intermediate activations are
   rotated core→neighbor over `num_a2a_iters` passes so every core's W2 shard eventually
   contracts over the full intermediate dimension. dm1 then writes results to the combine
   cores (Full) or directly to output (compute-only).

Bias, where present, is applied **inside the matmul** by contracting a ones-tile against an
appended bias row (`gate += ones·b0`, etc.) rather than a separate add.

## 5. Tile-distribution / sharding — the core generalization

This is where `moe_compute` structurally supersedes `moe_gpt`:

- **`moe_gpt` (`moe_gpt_ring_common.h`)**: hand-written `constexpr` **lookup tables** for
  the 90-tile-over-12-core split — `W0_W1_TILES_PER_CORE_PER_STEP_A[12][12]`,
  `W2_TILES_PER_CORE_A[12] = {8,8,7,7,…}`, `IN2_TILES_PER_STEP=8`, `NUM_A2A_ITERS=2`. Valid
  **only** for `K=N=2880, cores=12`. Comments literally enumerate "cores {0,1,4,5,8,9} get
  8 tiles."

- **`moe_compute` (`moe_ring_common.h`)**: **closed-form Euclidean-rhythm formulas**
  computed at compile time for *any* `(Ht, Nt, n_cores)`:
  - `shard_tiles(n_tiles, core, n_cores)` distributes the remainder via a Bresenham
    predicate (`is_big_w0w1`) for W0/W1.
  - `w2_shard_tiles(...)` uses a complementary pattern (when `Nt%n + Ht%n == n`) so W0/W1
    and W2 load-balance against each other.
  - `MoeRingConfig<Ht,Nt,num_cores,has_bias,SharedExpertTp>` derives `in2_tiles_per_step`,
    `w*_blocks_per_expert`, `num_a2a_iters`, plus shared-expert TP variants
    (`TpNt = ceil(Nt/tp)`), all parametrically.
  - These C++ `constexpr` formulas are kept byte-for-byte in sync with the Python packers
    in `moe_compute_utils.py`.

Shared blocks `W0_W1_TILES_PER_TXN=14`, `BLOCK_TILES_H=7`, `4·TILE` packing, bfloat4_b
weights, Float16_b activations are identical between the two.

## 6. Hardware, modes, and validation

| Aspect | `moe_compute` | `moe_gpt` |
|---|---|---|
| Arch | WH (ring=12) **and** BH (ring ∈ {8,12,16}, auto-detected) | WH-oriented; validation requires dim0 == #DRAM banks (12) |
| Compute-only mode | Yes (5-tensor return, no fabric/combine) | No |
| Cross-device combine | Yes, fused `selective_reduce_combine` along `cluster_axis`, topology Linear/Ring | No (local gather only) |
| Shared experts | Yes — TP-split, occupy tail expert slots, zero-padded W2 walk | No |
| Bias | Yes — strict K/N tile-multiple validation when `has_bias` | Reference treats bias as zeros |
| Validation | Extensive: `hidden%32`, `intermediate%32`, `intermediate_tiles ≥ ring`, `tiles_per_step` even, `ring∈{8,12,16}`, `num_shared ≤ experts/device`, compute_only⊕cluster_axis, topology∈{Linear,Ring} | Lighter: rank-6 weights, dim checks, HEIGHT_SHARDED L1 indices/scores |
| Hardware caveats | Unharvested WH grid (drain core hardcoded `(6,9)`), `DispatchCoreAxis.COL` required (see issue #41132) | Similar grid assumptions, GPT-OSS layout |

### Test coverage / golden tolerances

- **`moe_compute`** is tested across DeepSeek-V3 (`N=2048, hidden=7168, k=8, SILU`),
  GPT-OSS (`2880/2880, k=4, e/dev=4, SWIGLU, bias`), Gemma (`GELU`), Qwen, GLM,
  DeepSeek-OCR — single-card and 6U/Galaxy. PCC thresholds: SWIGLU 0.984, SILU/GELU 0.986,
  with-bias 0.988.
- **`moe_gpt`** is tested only for GPT-OSS (`2880/2880, k=4, e/dev=4`, 4×8 mesh, ring along
  `cluster_axis=0`), SWIGLU only, PCC 0.984, checking per-expert `[E·M, K]` outputs.

## 7. Practical guidance

- **Use `moe_compute`** for any new MoE work. It is the maintained, general path: arbitrary
  shapes, three activations, bias, shared experts, BH+WH, and a real fused score-weighted
  combine. It's already wired into `models/common/modules/moe` and DeepSeek-V3.
- **`moe_gpt`** is effectively legacy/special-purpose — a hard-compiled GPT-OSS-only kernel
  whose only live consumer is `models/demos/gpt_oss/tt/experts_throughput/`. Its
  functionality is a strict subset of `moe_compute`'s `SWIGLU` path, minus
  score-weighting/cross-device reduction.
- **Mind the combine semantics** when porting between them: `moe_gpt`'s output is
  *unweighted per-expert results in a token-sharded layout*; `moe_compute`'s slot-5 combine
  output is the *score-weighted, expert-reduced, cross-device* result. They are not
  interchangeable without accounting for where the gate-scaling + top-k sum happens.

## Appendix: source map

| Concern | `moe_compute` | `moe_gpt` |
|---|---|---|
| Host API | `moe_compute.{hpp,cpp}`, `moe_compute_nanobind.cpp` | `moe_gpt.{hpp,cpp}`, `moe_gpt_nanobind.cpp` |
| Device op / validation | `device/moe_compute_device_operation.cpp`, `..._types.hpp` | `device/moe_gpt_device_operation.cpp`, `..._types.hpp` |
| Ring constants / sharding | `device/kernels/moe_ring_common.h`, `device/hostdevcommon/config.hpp` | `device/kernels/moe_gpt_ring_common.h` |
| Activation | selectable in `device/kernels/compute.cpp` | `device/kernels/swiglu_sfpu.h` |
| Kernels | `device/kernels/{tilize_*,dm0,dm1,compute}.cpp` | `device/kernels/{tilize_*,dm0,dm1,combine_dm1,compute}.cpp` |
| Weight packers | `moe_compute_utils.{hpp,cpp}`, `ttnn/ttnn/_experimental/moe_compute_utils.py` | private helpers in the e2e test |
| Tests / goldens | `test_moe_compute_single_card.py`, `tests/nightly/tg/ccl/moe/test_moe_compute_6U.py` | `test_moe_gpt_e2e.py` |

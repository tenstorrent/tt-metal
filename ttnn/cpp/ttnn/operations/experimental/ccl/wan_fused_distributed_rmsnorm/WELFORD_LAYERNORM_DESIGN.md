# Adding Welford LayerNorm to `wan_fused_distributed_rmsnorm`

**Status:** design proposal · **Scope:** add a numerically-stable, Welford-based distributed
LayerNorm variant to the existing fused distributed RMSNorm op, reusing the fabric all-gather,
CB plumbing, weight/bias, RoPE, block-major/streaming machinery, and output drain. **Target:**
Wormhole 4×8 galaxy first; Blackhole port to follow.

---

## 0. TL;DR / Recommendation

The op already implements ~90% of a LayerNorm: `epsilon`, **weight**, **bias**, per-head, RoPE,
and the cross-shard fabric all-gather of partial stats are all present. RMSNorm and LayerNorm
differ in exactly two places:

| Phase | RMSNorm (today) | LayerNorm (to add) |
|---|---|---|
| **PRE** (per shard) | one partial: `sum(x²)` | two partials: per-shard Welford `(mean_i, M2_i)` |
| **POST** (after AG) | additive sum of `ring_size` partials → `x·rsqrt(E[x²]+ε)` | Welford-**merge** the `ring_size` partials → `(x − mean)·rsqrt(var+ε)` |

**Recommendation:** add a compile-time `norm_type ∈ {RMS, LAYERNORM}` attribute. Compute
per-shard Welford partials in PRE (numerically stable single pass over the shard width), gather
`(mean_i, M2_i)` over the existing fabric ring, and **merge them in POST with the parallel-Welford
formula** (`combine_welford_partials`, the exact primitive `dit_layernorm_post_allgather` already
uses). Keep one compute kernel; gate the PRE-stat and POST-reduce sections on `norm_type` with
`if constexpr`, and add a single `(x − mean)` subtraction before the existing normalize-multiply.
Everything downstream (weight, bias, RoPE, block-major, head-major, streaming, output) is shared
verbatim.

This is preferred over the simpler "gather additive `sum(x)` + `sum(x²)`, then `var = E[x²]−E[x]²`"
because that formula suffers catastrophic cancellation for the large/skewed activations these DiT
models produce — which is the whole reason the codebase grew a Welford path. See §6.2.

---

## 1. How distributed norm is used in `tt_dit` (the consumer contract)

All norm modules live in `models/tt_dit/layers/normalization.py`. The DiT denoisers use
**`DistributedLayerNorm`** for the *body* (block) and *output* norms and **`DistributedRMSNorm`**
for QK-norm and a few models' block norms. The reduction (embedding) dim is **sharded across the
TP mesh axis**, so every norm needs a cross-device combine of partial stats — this is the
distributed mean+variance case.

### 1.1 Which models use LayerNorm vs RMSNorm

- **DistributedLayerNorm** (block + output norms): SD3.5, Flux, Qwen-Image, Wan2.2, and the shared
  `blocks/transformer_block.py`; Mochi uses it for the **final** norm only.
  - `transformer_block.py:75,95,123,147`; `transformer_sd35.py:65,84,108,131,404`;
    `transformer_flux1.py:72,337`; `transformer_qwenimage.py:110`;
    `wan2_2/transformer_wan.py:67,104,128,344`; `transformer_mochi.py:460`;
    `ltx/transformer_ltx.py:531,570`.
  - Almost all are `norm_elementwise_affine=False, bias=False, eps=1e-6`. The exception is
    **Wan2.2 `norm2`** (`transformer_wan.py:104`, `norm_elementwise_affine=True`) — the only static
    affine LayerNorm.
- **DistributedRMSNorm** (block norms): Mochi (all), LTX (`norm1/2/3`). QK-norm (`head_dim`-wide,
  per-head): SD3.5, Mochi, Wan, LTX, shared `blocks/attention.py`.

### 1.2 The adaLN fusion contract (what callers expect)

The dominant fusion is **adaptive LayerNorm**: the affine γ/β are *per-call* tensors (timestep
modulation), not static params. `DistributedLayerNorm.forward(x, dynamic_weight=…, dynamic_bias=…)`
(`normalization.py:414-429`) implements `norm(x)·(1 + scale) + shift` by passing
`dynamic_weight = (1 + scale)` and `dynamic_bias = shift`. Examples:
`transformer_block.py:233,258,289,314`; `transformer_flux1.py:154-157`;
`wan2_2/transformer_wan.py:198,226,590`. **Implication for us:** the LayerNorm variant must accept a
**per-token (`[N,H]`) weight and bias**, not just broadcast `[1,H]` — the op already validates and
handles both (`device_operation.cpp:51-86`, compute P_WEIGHT/P_BIAS), so this is free. Some final
norms apply the modulation *outside* the op via `ttnn.addcmul` (LTX `transformer_ltx.py:894`), which
also keeps working.

### 1.3 Per-model dims (hidden = heads × head_dim)

| Model | heads | head_dim | hidden | eps |
|---|---|---|---|---|
| SD3.5 | 18 | 64 | 1152 | 1e-6 |
| Wan2.2 | 40 | 128 | 5120 | eps |
| LTX video / audio | 32 | 128 / 64 | 4096 / 2048 | norm_eps |
| Mochi | 24 | 128 | 3072 | 1e-6 |
| Flux / Qwen | cfg | cfg | heads×hd | 1e-6 |

Sharding constraint: `embedding_dim % (32 · mesh_width) == 0` (`normalization.py:363`). With TP up
to 8, each shard holds `hidden/TP` columns (e.g. Wan2.2 TP4 → 1280 cols = 40 tiles). The local
per-shard reduction is therefore over hundreds–thousands of elements: **exactly where Welford's
stability matters**; the cross-shard merge is only over `ring_size ≤ 8` partials.

### 1.4 The existing distributed-LayerNorm op (our reference implementation)

`DistributedLayerNorm.forward` (`normalization.py:414-452`) is a 3-stage chain that **already uses
Welford**:

```python
stats = ttnn.experimental.dit_layernorm_pre_allgather(x, self.recip_tensor, ...)   # Welford partials
stats = self.ccl_manager.all_gather_persistent_buffer(stats, dim=-1, mesh_axis=…)  # gather partials
x     = ttnn.experimental.dit_layernorm_post_allgather(x, stats, weight=…, bias=…, epsilon=…)
```

- Pre op: `experimental/transformer/dit_layernorm_pre_all_gather/` — Welford program factory,
  emits per-device `(mean, M2/var)`; needs a precomputed `recip_tensor` (1/width reciprocals).
- Post op: `experimental/transformer/dit_layernorm_post_all_gather/` — **merges** the per-device
  partials with `combine_welford_partials(...)`, then `(x−mean)·rsqrt(var+ε)·γ + β`.

Our fused op is the *single-op* collapse of this chain (CCL inside the op), already proven for
RMSNorm. Adding LayerNorm = porting `dit_layernorm`'s Welford pre/merge logic into our PRE/POST,
on top of our fabric transport. This de-risks correctness substantially — we can diff against
`dit_layernorm` and against the torch reference in `tests/unit/test_normalization.py:50-52`.

---

## 2. The Welford LLK & compute API

Welford computes running `mean` and `M2` (sum of squared deviations) in one numerically-stable
pass. The public compute API is `tt_metal/hw/inc/api/compute/welford.h` (namespace `ckernel`); it
runs entirely on the **MATH/SFPU thread** (everything wrapped in `MATH(( ))`). Lower layers:
`tt-llk/.../sfpu/ckernel_sfpu_welfords.h` (SFPU recurrence) → `llk_math_welfords_sfpu*.h` →
`ckernels/.../llk_math_welfords_sfpu_entry.h`.

### 2.1 API the kernel author calls

- `welford_init<WelfordInitMode mode = ClearStats>()` — programs the SFPU replay buffer + addrmod;
  `ClearStats` zeroes running mean (LREG4) / M2 (LREG5), `PreserveStats` keeps them.
- `welford_update<reciprocal_size>(input_dst_idx, start_idx, recip_lut)` — folds **all 1024
  elements** of the tile at `input_dst_idx` into LREG4/5; `start_idx` indexes the cumulative-count
  reciprocal LUT. `reciprocal_size==0` + `{}` → runtime `1/(idx+1)` division.
- `welford_update_rows<…>(input_dst_idx, start_idx, start_row, num_rows, lut)` — partial-width
  update (last tile with padding, or per-group).
- `welford_finalize_to_row<reciprocal_size>(mean_dst_idx, scale_idx, lut)` — `var = M2·(1/(scale_idx+1))`;
  writes **mean → row 0 of tile `mean_dst_idx`**, **var → row 0 of tile `mean_dst_idx+1`**.
  `scale_idx = W−1` → population var (1/W); `W−2` → Bessel (1/(W−1)). (`_to_face` + `group_id`
  flavors exist for groupnorm.)
- `welford_save_state` / `welford_restore_state` — spill/reload LREG4/5 to a tile, for large-tensor
  width-blocking.

Read-out is via **DST register tiles**, row-major. Welford's "row" is the original tile's *column*,
so callers `transpose_wh_tile` the input before `welford_update` and transpose the result back.

### 2.2 The parallel-Welford merge (for cross-shard combine)

`ttnn/cpp/ttnn/operations/normalization/kernel_util/compute/combine_welford.h` —
`combine_welford_partials(cb_partials, cb_combined, num_sets, next_set_size_fn, RSqrtPolicy)`
folds `num_sets` interleaved `[mean, var]` tiles on the MATH thread:

```
delta     = mean_b − mean_acc
mean_acc  = mean_acc·(n_a/(n_a+n_b)) + mean_b·(n_b/(n_a+n_b))
M2_acc   += var_b·n_b  +  delta²·(n_a·n_b/(n_a+n_b))
… finally  var = M2_acc / N ; optional RSqrtPolicy → add ε + rsqrt
```

This is exactly what `dit_layernorm_post_allgather` calls with `next_set_size_fn = W`
(per-device width) to merge `num_devices` partials. **For us `num_sets = ring_size` and each
`n_i = width_per_device` is a compile-time constant and equal across shards** (§1.3 divisibility),
so the merge simplifies (mean = mean-of-means). A scalar/dataflow variant
(`groupnorm/.../dataflow/welford_combine.h`, `combine()`/`combine_welford_stats`) also exists if we
ever want the merge off the MATH thread.

### 2.3 Gotchas the kernel must respect

1. **Transpose first, transpose back.** `transpose_wh_tile(cb, …, input_dst)` before
   `welford_update`; transpose mean/var back to column orientation after finalize.
   (`transpose_wh_dest` on the result is buggy — tt-llk #549.)
2. **fp32 DST accumulation.** For fp32 input, set `fp32_dest_acc_en=true` (the op already forces
   this, `device_operation.cpp:288`) **and** route the input CB through an `UnpackToDestFp32`
   *alias* CB, else the unpacker truncates to TF32.
3. **Replay-buffer collision (the big one).** `welford_init` records all 32 SFPU replay slots. On
   the fp32 path, `transpose_wh_tile` re-records slots [16,32), clobbering Welford's LREG2/3 ops.
   **Fix: call `welford_init<PreserveStats>()` after each `transpose_wh_tile`** (keeps LREG4/5).
   On bf16 input the transpose goes through SrcA and doesn't touch the replay buffer (gate the
   recovery out). See `layernorm_welford.cpp:179-242` for the canonical dance.
4. **Reciprocal LUT.** Built host-side by `ttnn.create_layer_norm_reciprocals(device, crs, width)`
   → `[1/(i+1) … ]`, **Float32**, L1 height-sharded, bound to a CB. `dit_layernorm` passes it as
   `recip_tensor`; we'd allocate it the same way (and can cache per `(mesh, width_per_device)` like
   `normalization.py:328` does). `reciprocal_size=0` fallback works but is slower.
5. **Plain-function static_assert trap (our op).** The compute kernel is a *plain function* with
   constexpr CT args, so static_asserts in *discarded* `if constexpr` branches still fire. Any new
   LayerNorm branch must guard array sizes / asserts exactly like the existing `stats_tiles_cols ==
   1 ||` exemptions (compute `wan_rmsnorm_fused_compute.cpp:386-394`).
6. **Welford is gated off for RMSNorm upstream** (`layernorm_device_operation.cpp:234` `TT_FATAL`)
   — confirming the intended split: RMS keeps its `sum(x²)` path, LayerNorm gets Welford. We mirror
   that here.
7. **WH vs BH divergence** (8 vs 6 SFPU instr/row, addrmod) is handled inside the LLK — not our
   concern at the kernel-author level, but it means the BH re-sweep (see the BH re-sweep memo)
   should re-validate timing.

---

## 3. The op today, and the extension points

Op dir: `ttnn/cpp/ttnn/operations/experimental/ccl/wan_fused_distributed_rmsnorm/`.

### 3.1 Pipeline

`PRE (per-shard partial stat) → fabric ring all-gather of partials → POST (reduce → ε+rsqrt →
normalize-mul → weight → bias → RoPE → output)`. Two control paths:
`is_tp_1` (`ring_size==1 || per_head_norm`, reduce locally, no fabric) and the **all-gather path**
(`use_mux`: `num_workers` workers + `num_forwarders` forwarders; per row a worker computes its
partial, transposes col0→row0, NoC-writes a 128 B stick into its forwarder's packet; the forwarder
ring-mcasts to a DRAM scratch page on every chip; workers read it back and finish POST).

### 3.2 Where "what statistic" is decided

**In the kernels, not the host.** The host passes `stats_tiles_cols = ring_size` (compute CT
arg 17) = the *count* of partial-stat tiles to reduce, not their content. `sum(x²)` is hardcoded in
PRE. The host's stat-shaping knobs: `per_row_stats_count`, `stats_local_tiles`, `reduce_factor =
per_head_norm ? head_dim : H_full` (the AVG/recip divisor), and `stick_bytes = 128` (one fp32 stat
per token-tile). All buffer geometry derives from `compute_sizing` (the single source of truth;
`types.hpp:116-126`), consumed by both `create_stats_buffer` and `compute_output_specs`.

### 3.3 PRE / POST in the compute kernel (`device/kernels/compute/wan_rmsnorm_fused_compute.cpp`)

- **PRE** (lines 165-291): square via `mul_tiles` with L1-acc into `pre_intermediate_cb`, then one
  `reduce<SUM, REDUCE_ROW>` → one fp32 tile with `sum(x²)` in col 0. Then transpose col0→row0
  (298-313) for the packed stick.
- **POST** reduce+ε+rsqrt (361-477): TP>1 path **FPU-adds** the `ring_size` gathered partial-sum
  tiles (`add_tiles`, 410-416) — note this deliberately replaces a matmul reduce to dodge a packer
  hang — then `mul_unary(1/H)`, `add_unary(eps_bits)`, `rsqrt_tile`. The `eps_rsqrt` lambda
  (355-360) is the surgical insertion point for variance arithmetic.
- **Normalize-mul** (P_NMUL, 482-541): `mul_tiles_bcast_cols(input, 1/rms)`. **LayerNorm inserts a
  `(x − mean)` here.**
- **weight** (P_WEIGHT, 893-935), **bias** (P_BIAS, 937-972, *already present*), **RoPE** (P_MM +
  P_ROPE), plus `block_major_post` (710-891), head-major per-head (543-700), `fuse_mm_rope`
  (974-1044) variants — **all reused unchanged**.

### 3.4 Stats transport capacity

A stick = 128 B = 32 fp32 = **one statistic for the 32 tokens in a tile-row** (worker_writer
`stick_bytes=128`). The packet holds `window_size = sticks_per_packet` (≈8) sticks, and
`total_pages = ring_size · num_forwarders · max_rounds`. So a token-tile currently uses 1 of
~8 slots — **there is headroom for a second stat** without changing the packet size.

---

## 4. Proposed design

### 4.1 Op surface

Add an enum attribute (default preserves today's behavior):

```cpp
enum class WanFusedNormType : uint8_t { RMS = 0, LAYERNORM = 1 };   // types.hpp
// WanFusedDistributedRmsnormParams gains:  WanFusedNormType norm_type = WanFusedNormType::RMS;
```

Plumb it through the nanobind op (`norm_type="rms"|"layernorm"` or a bool `layer_norm=False`),
the device op attributes, and into a **compute CT arg** (e.g. arg 39). For LayerNorm, also accept
the **reciprocal LUT** tensor (allocate via `create_layer_norm_reciprocals` for `width_per_device`,
cache per `(mesh, width)` like the Python module does) and require `bias`/`dynamic_bias` to be
accepted (already validated). `epsilon`, `weight`, RoPE, `per_head_norm`, topology unchanged.

### 4.2 Program factory changes (small, localized)

1. **Reciprocal LUT CB** — add a CB (fp32, L1) bound to the recip tensor, mirroring `dit_layernorm`
   (`cb_reciprocals`). Reader reads it once.
2. **Two stat lanes** — LayerNorm gathers `(mean_i, M2_i)`. Two implementation options (pick in
   review; (a) is less invasive):
   - **(a) Two sticks per token-tile:** emit a mean-stick and an M2-stick; double the
     arrival/slot accounting in worker_writer + forwarder; halves `workers_per_forwarder` capacity
     (already clamped, factory 92-94/673-677). No page-format change.
   - **(b) Widen the stick to 256 B:** pack mean+M2 into adjacent face-rows; doubles `stick_bytes`
     and `page_size_bytes` in `compute_sizing`. Cleaner data layout, touches buffer geometry.
   Either way, doubling lives in `compute_sizing` and propagates to `create_stats_buffer` /
   `compute_output_specs` automatically.
3. **`norm_type` CT args** to compute + (where stat width matters) reader/writer.
4. CB sizing for `stats_*` and the POST reduce-result region grows by one tile (carry `mean`
   alongside `1/std`). Everything else (input_cb, weight/bias, RoPE, intermediate/output,
   block-major) unchanged.

### 4.3 Compute kernel changes (one new PRE-stat path, one new POST-merge path, one `−mean`)

Keep a single kernel; gate with `if constexpr (norm_type == LAYERNORM)`:

- **PRE:** replace the square+`reduce<SUM>` with the Welford dance over the shard's `num_tile_cols`:
  `welford_init()`; per col-tile `transpose_wh_tile → (fp32: welford_init<PreserveStats>) →
  welford_update<W>(input_dst, start_N, recip_lut)`; last tile `welford_update_rows`;
  `welford_finalize_to_row<W>(mean_dst, W−1, recip_lut)` (population var, `scale_idx=width−1`).
  Emit `(mean, M2)` (or `(mean, var)`) as the two-lane partial. *Reuse* the existing
  transpose-to-stick step — but note Welford's finalize already writes row 0, so reconcile the
  PRE-transpose (compute 298-313) accordingly (§5 risk).
- **POST:** replace the additive `add_tiles` reduce with `combine_welford_partials(stats_gathered,
  stats_reduced, ring_size, []{return width_per_device;}, RSqrtPolicy{ε})` → yields merged `var`
  and `mean`; apply `add ε + rsqrt` (RSqrtPolicy or the existing `eps_rsqrt` lambda) → `1/std`;
  **carry `mean` out** into a second broadcast CB.
- **Normalize-mul:** before the existing `mul_tiles_bcast_cols(input, 1/std)`, insert
  `sub_tiles_bcast_cols(input, mean)`. This touches all four sub-phase-1 sites (resident, streaming,
  block-major, head-major) — the only downstream change.
- **weight / bias / RoPE / output:** unchanged (`bias` already wired).

RMSNorm path is the `else` branch — byte-identical to today.

### 4.4 What is shared vs new

| Component | Shared | New for LayerNorm |
|---|---|---|
| Fabric forwarder, DRAM page addressing, ping-pong sems | ✅ | — |
| CB plumbing, reader (input/weight/bias/cos/sin) | ✅ | + recip LUT read |
| weight mul, **bias add**, RoPE, block-major, head-major, streaming, output drain | ✅ | — |
| Reduce scalars (SUM/AVG) | ✅ (Welford uses recip LUT instead of AVG scalar) | — |
| PRE statistic | — | Welford `(mean, M2)` |
| POST reduce | additive add → kept for RMS | Welford-merge |
| Sub-phase-1 normalize | mul by `1/rms` | `(x−mean)` then mul by `1/std` |
| Stat transport width | 1 lane | 2 lanes |

---

## 5. Risks & open questions

1. **Stick-layout reconciliation.** RMS transposes col0→row0 to build the 128 B stick; Welford's
   `finalize_to_row` already lands stats in row 0 (mean in tile T, var in tile T+1). The PRE→stick
   packing must be reconciled (likely *simpler* for Welford — drop the extra transpose — but verify
   the worker_writer's face-row span assumptions).
2. **Packer-hang interaction.** The TP>1 POST deliberately uses FPU `add_tiles` instead of a matmul
   reduce to avoid a packer hang. `combine_welford_partials` runs on MATH/SFPU; confirm it doesn't
   reintroduce the hang on wide shards. Validate early on Wan2.2 TP4/TP8.
3. **Replay-buffer / `welford_init<PreserveStats>` cost.** Re-arming after every transpose is per
   col-tile overhead; profile against the current RMS PRE (~29% of the floor per the kernel's own
   note). If too costly on the widest shards, consider bf16-input fast path (no replay clobber).
4. **fp32 alias CB.** Need an `UnpackToDestFp32` alias for `input_cb` on the LayerNorm fp32 path
   (extra CB index sharing SRAM); confirm it coexists with `streaming_low_l1` / `block_major_post`
   L1 budgets (the `decide_*` heuristics may need the LayerNorm CBs added — cf. open task #10
   "audit all CBs").
5. **`per_head_norm` + LayerNorm.** Per-head LayerNorm (mean/var per head over `head_dim`) is
   plausible but no `tt_dit` caller needs it today (QK-norm is RMS). Propose **scoping LayerNorm to
   the whole-row (`per_head_norm=false`) case first**; keep per-head a follow-up.
6. **Block-major + 2-lane stats.** Block-major POST and the 2-lane transport are orthogonal but both
   touch `compute_sizing`; co-validate on the wide low-TP LayerNorm shapes (Wan2.2 TP1/TP2).
7. **Numerical target.** Match `tests/unit/test_normalization.py` torch reference (`mean`,
   `var(unbiased=False)`, `(x−mean)/sqrt(var+ε)`), PCC ≥ 0.999. Population variance (`scale_idx =
   W−1`) matches `unbiased=False`.
8. **BH port.** Re-validate timing on Blackhole (6 vs 8 Welford instr/row); fold into the existing
   BH re-sweep effort.

---

## 6. Decisions

### 6.1 One kernel vs a new kernel

**Recommend one kernel** with `if constexpr (norm_type)` gating the PRE-stat and POST-reduce
sections, because the entire downstream (mul/weight/bias/RoPE/block-major/head-major/streaming/
output) is identical and duplicating it would be a large maintenance hazard. The Welford PRE/POST
sections are self-contained enough to read clearly under `if constexpr`. **Mind the plain-function
static_assert trap** (§2.3.5). If the Welford replay-buffer setup proves to fight the RMS path's
init sequencing, fall back to factoring PRE-stat and POST-reduce into separate `.hpp` helper
functions selected by `norm_type` (still one compiled kernel, cleaner separation).

### 6.2 Welford-merge vs additive `sum`+`sumsq`

**Recommend Welford-merge.** The alternative — gather additive `sum(x)` and `sum(x²)` (both additive
across shards, so it reuses the existing `add_tiles` ring-reduce unchanged) and compute
`var = E[x²] − E[x]²` — is *tempting* because it's a smaller diff. But `E[x²] − E[x]²` is the
textbook catastrophic-cancellation formula; for DiT activations with large means it loses most of
the variance's significant bits, and it's precisely why the codebase added Welford
(`accuracy_tips.md:141-217` shows the Frobenius-error gap on skewed inputs). Since the user
explicitly wants *Welford* LayerNorm and the stable `dit_layernorm` reference already exists, do the
Welford partial + parallel-merge. The merge is cheap (`ring_size ≤ 8`, equal compile-time counts).
Keep the additive path documented only as a fallback if profiling later shows Welford PRE is a
bottleneck.

---

## 7. Phased implementation plan

1. **Plumbing (no math):** add `WanFusedNormType` to types/op/nanobind/device-op + compute CT arg;
   thread the recip LUT tensor + CB; double the stat lanes in `compute_sizing` and transport
   (option 4.2a). RMS path unchanged; full RMS regression must stay green (`test_corr_det`,
   model-shape perf).
2. **PRE Welford:** implement the Welford per-shard partial behind `if constexpr`; unit-validate the
   partial `(mean_i, M2_i)` against torch per shard (TP1 first — no fabric).
3. **POST merge + `−mean`:** `combine_welford_partials` + variance + `(x−mean)`; validate TP1 → TP2
   → TP4 → TP8 against torch LayerNorm and against `dit_layernorm_post_allgather`, PCC ≥ 0.999, plus
   the 10× determinism check (ping-pong discipline already in the test harness).
4. **Wide / fused paths:** validate block-major, streaming, and RoPE-fused LayerNorm on the wide
   low-TP shapes; add LayerNorm configs to `test_corr_det` (`_CORR_PARAMS`) and a `norm_type`
   parametrization.
5. **Wire `tt_dit`:** add a fused LayerNorm path to `DistributedLayerNorm.forward` (mirroring the
   `WAN_USE_FUSED_RMSNORM` switch) calling the op with `dynamic_weight=(1+scale)`,
   `dynamic_bias=shift`; A/B against the `dit_layernorm` chain end-to-end.
6. **BH re-sweep:** re-validate correctness + perf on Blackhole.

## 8. Key files

- Op: `ttnn/cpp/ttnn/operations/experimental/ccl/wan_fused_distributed_rmsnorm/` — `*.hpp/.cpp`,
  `*_nanobind.cpp`, `device/*device_operation.cpp`, `device/types.hpp`,
  `device/wan_fused_distributed_rmsnorm_program_factory.cpp`,
  `device/kernels/compute/wan_rmsnorm_fused_compute.cpp`,
  `device/kernels/dataflow/{*reader*,*worker_writer*,*forwarder*,*writer*}.cpp`,
  `device/kernels/dataflow/wan_rmsnorm_scalar_setup.hpp`.
- Welford API: `tt_metal/hw/inc/api/compute/welford.h`; SFPU
  `tt_metal/tt-llk/tt_llk_{wormhole_b0,blackhole}/common/inc/sfpu/ckernel_sfpu_welfords.h`.
- Merge: `ttnn/cpp/ttnn/operations/normalization/kernel_util/compute/combine_welford.h`.
- Reference op: `ttnn/cpp/ttnn/operations/experimental/transformer/dit_layernorm_{pre,post}_all_gather/`
  (Welford pre/post + `layernorm_{pre,post}_allgather_welford.cpp`).
- Reference kernels: `ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm_welford.cpp`.
- Consumers/tests: `models/tt_dit/layers/normalization.py`,
  `models/tt_dit/tests/unit/test_normalization.py`,
  `models/tt_dit/tests/test_distributed_rmsnorm_fused.py`.

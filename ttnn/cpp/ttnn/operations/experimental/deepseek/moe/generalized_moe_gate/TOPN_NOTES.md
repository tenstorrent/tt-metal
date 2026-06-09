# Generalized MoE Gate — configurable top-n (k ≤ 8): design, journey, pitfalls & solutions

Status: **✅ top-4 / top-6 / top-8 supported**, on both the 256 single-op path and the 512 combine path, on
**Wormhole B0**. `test_generalized_moe_gate` and `test_generalized_moe_gate_512_global` are parametrized over
`topk ∈ {8, 6, 4}`. The generic per-lane mask also covers k=5/7 (untested). k<4 and k>8 are NOT yet supported.
(Chinese version: `TOPN_NOTES.zh.md`.)

---

## 1. Goal & constraints

- Make the routed expert count `k` a runtime/compile-time parameter instead of the baked-in 8. DeepSeek uses
  k=8; we also want k = 4 / 6 (and later 10).
- "Going **down**" (k < 8) is much easier than "going up" (k > 8): the front of the pipeline already produces a
  sorted **global top-8**, so for k ≤ 8 we keep the entire 32→top-8 machinery and only change the **final
  merge/normalize tail**. Going up (k=10) needs more candidates than the merge currently keeps and is deferred.
- **Single op** (softmax will be fused later). Target arch **Wormhole B0**, `fp32_dest_acc_en = false` (16-bit
  DEST), `dst_full_sync_en = true`.

## 2. Approach — zero ranks ≥ k before normalize

After `merge16_core` + `store8`, the global **top-8 are sorted descending** and committed to DEST. Normalization
then divides each kept score by the sum of the kept scores and multiplies by `scaling_factor`. So top-n is just:

> **Zero the score (and the idx) of every rank ≥ k, _before_ the normalize.** The denominator then auto-becomes
> the sum of the top-n, and the dropped slots emit `0`. k=8 → no-op.

Nothing upstream of the finalize changes. The output keeps rank `r` in column `r`; ranks `k..7` come out `(0, 0)`.

## 3. The working recipe (`_generalized_moe_gate_finalize_ungrouped<…, topk>`)

The sorted-8 live in the **scores/indices regions at SFPU/TTI offsets `{0, 4}`**: offset 0 = ranks 0-3, offset 4
= ranks 4-7 (see §4 for the lane layout inside a row). Three cases:

```
topk == 8 :  no-op (full top-8).

topk <= 4 :  drop the ENTIRE offset-4 row (ranks 4-7):
               TTI_SFPSTORE(LCONST_0, scores_offset + 4)
               TTI_SFPSTORE(LCONST_0, indices_offset + 4)

4 < topk < 8 (k=5/6/7) :  per-lane mask of the offset-4 row. rank (4+j) sits at tileid 16*j
                          (ranks 4,5,6,7 -> tileid 0,16,32,48). Drop ranks k..7 = lanes with
                          tileid >= 16*(topk-4); keep ranks 4..k-1.
   constexpr int drop_thr = 16 * (topk - 4);                 // k=5->16, k=6->32, k=7->48
   constexpr int sc4 = (scores_offset  + 4) / 2;             // dst_reg index (see §4): scores+4  -> 2
   constexpr int ix4 = (indices_offset + 4) / 2;             //                         indices+4 -> 34
   TTI_SETRWC(..., SET_D);                                   // reset Dst RWC so dst_reg base == TTI base
   vFloat sc = dst_reg[sc4]; v_if(vConstTileId >= drop_thr){ sc = 0.0f; } v_endif; dst_reg[sc4] = sc;
   vFloat ix = dst_reg[ix4]; v_if(vConstTileId >= drop_thr){ ix = 0.0f; } v_endif; dst_reg[ix4] = ix;
```

The 512 **combine** path needed the same fix: `combine_finalize` calls `finalize_ungrouped` and previously
hard-coded the default k=8. Thread `topk` through `combine_finalize<is_32bit, topk>` so 512 is configurable too.

## 4. DEST / SFPU layout facts (WH B0, verified)

- After `store8`, the descending top-8 occupy **scores/indices offsets `{0, 4}`** (TTI addr): **offset 0 = ranks
  0-3, offset 4 = ranks 4-7**; the output packs rank `r` → column `r`.
- **Inside an offset row the 4 ranks are NOT adjacent — they are spread every 8 lanes** ("even_cols" store →
  lanes 0, 8, 16, 24). Since `vConstTileId = 2 * lane`, rank `(4+j)` lands at **tileid `16*j`**: ranks 4,5,6,7 →
  tileid 0, 16, 32, 48. This stride of 16 is the single most important (and most counter-intuitive) fact for the
  mask threshold.
- **sfpi `dst_reg[ix]` addresses TTI addr `ix * SFP_DESTREG_STRIDE` (= `ix*2`)**, with default mod-0 (`SrcB`)
  load/store. So a field at TTI addr `A` is `dst_reg[A/2]`: `scores+4` (addr 4) = `dst_reg[2]`, `indices+4`
  (addr 68) = `dst_reg[34]`. The mod-0 read matches the normalize's mod-0 `TTI_SFPLOAD` of the scores, and is a
  **raw-bit passthrough that round-trips 16-bit ids losslessly** (e.g. 337 = 0x0151 survives).
- `vConstTileId` = `[0, 2, 4, …, 62]` per lane (the tile-id const-reg, sfpi reg 15). sfpi materializes it via
  `sfpreadlreg`; a raw `TTI_SFPMOV`/`TTI_SFPIADD` reading reg 15 directly does **not** give per-lane values.

## 5. Pitfalls & solutions (in the order they were hit)

| # | Symptom | Root cause → fix |
|---|---------|------------------|
| 1 | raw-TTI per-lane mask zeros the **whole** offset-4 row (top-n collapses to top-4) | `TTI_SFPMOV(LTILEID→LREG)` then `TTI_SFPIADD(…CC_GTE0)`, and even `SFPIADD` with `lreg_c = LTILEID` directly, do **not** read per-lane tileid — the CC came out all-enabled, so `SFPMOV LCONST_0` zeroed every lane. (`SFPMOV` *does* honor the CC — confirmed by `ckernel_sfpu_dropout.h`; the bug was the tileid read.) → **use `sfpi v_if(vConstTileId >= thr)`**, which materializes the const-reg correctly via `sfpreadlreg`. |
| 2 | sfpi mask has **no effect at all** — full top-8 still comes out | `dst_reg[ix]` addresses TTI addr **`ix*2`** (`SFP_DESTREG_STRIDE = 2`). `dst_reg[scores_offset+4] = dst_reg[4]` wrote to **addr 8**, not `scores+4` (addr 4), so the masked value landed in dead space and `scores+4` was untouched. → **`dst_reg[(scores_offset+4)/2] = dst_reg[2]`**. |
| 3 | mask drops **too many** ranks (k=6 came out as top-5, then thresholds 4/8/16 all gave top-5) | Wrong lane-layout model. I assumed the 4 ranks sat in adjacent lanes (tileid step 2), then 4, then 8. They are actually **every 8 lanes → tileid step 16** (ranks 4-7 at tileid 0,16,32,48). With stride 16, any `drop_thr ∈ {4,8,16}` drops ranks 5,6,7 alike (all ≥ their threshold), which *looks* like a stuck result. → **`drop_thr = 16*(topk-4)`** (k=6 → 32). |
| 4 | (false lead) "identical results across thresholds ⇒ stale kernel cache" | Clearing `~/.cache/tt-metal-cache` changed **nothing** — the cache was fine and *does* recompile on header edits. The identical results were genuine (pitfall 3): stride-16 makes thresholds ≤16 behave the same. Lesson: don't blame the cache; pin the layout with a dump. |
| 5 | how the stride was finally pinned | **Diagnostic dump:** in the mask branch, overwrite `scores+4` with `int32_to_float(vConstTileId)` (no mask) and read output cols 4-7. They came back in the exact ratio **0 : 1 : 2 : 3** → uniform stride; combined with "thresholds ≤16 drop rank5" → absolute stride = 16. One dump replaced ~4 blind threshold guesses. |
| 6 | dropped **idx** not zeroed (scores were) | A **predicated** sfpi store `v_if(...){ dst_reg[ix4] = vUInt(0); }` simply **did not land**, and a **`vUInt`** read-modify-write was a **no-op** too. → read/write the idx row as **`vFloat`** (the default mod-0 `SrcB` raw-bit passthrough, same as the scores): zeros the dropped lanes and preserves the kept ids bit-for-bit. |

## 6. Key files & functions

- **`.../tt_llk/.../ckernel_sfpu_generalized_moe_gate_topk_single_face.h`** — `_generalized_moe_gate_finalize_ungrouped`
  gained `template <…, uint32_t topk = 8>` and the three-case zero-ranks-≥-k block (§3) after `store8`, before the
  normalize tail.
- **`.../compute_kernel_api/generalized_moe_gate.h`** — `generalized_moe_gate<…, topk>` (gate template) and
  `generalized_moe_gate_combine_finalize<is_32bit, topk>` both thread `topk` to `finalize_ungrouped`.
- **`device/unified_kernels/generalized_moe_gate.hpp`** — 256 path passes `CTArgs::topk`; combine path passes
  `combine_finalize<false, CTArgs::topk>`.
- **Plumbing of `topk` (a named compile-time arg) end-to-end:** `op.py` / nanobind (`topk=8`) → `generalized_moe_gate`
  op entry → `device_operation::invoke` → `operation_attributes_t.topk` → `program_descriptor_builder`
  (`{"moe_gate_topk", attrs.topk}`) → `ComputeCTArgs::topk` → `generalized_moe_gate_kernel.cpp`
  (`get_named_compile_time_arg_val("moe_gate_topk")`). `hash_moe_gate_program_structure` hashes
  `named_compile_time_args`, so each `k` gets its **own compiled program** (no cache collision).

## 7. Tests

`models/demos/deepseek_v3/tests/test_generalized_moe_gate.py`:
- `test_generalized_moe_gate` — 256/128 path, parametrized `topk ∈ {8, 6, 4}` × sigmoid × seed × batch.
- `test_generalized_moe_gate_512_global` — 512 combine, parametrized `topk ∈ {8, 6, 4}`.
- `golden(…, topk)` uses `torch.topk(bias_flat, topk)` + normalize over the top-k; both tests slice the output to
  `[:, 0, :topk]` (ranks 0..k-1; dropped ranks are now `0` in both score and idx).

## 8. Remaining work

1. **Top-10 (k > 8):** the hard direction. The pipeline currently keeps only a top-**8** (`merge16_core` →
   `store8`), so there is nothing to "un-mask" past rank 7. Needs the merge to retain ≥10 candidates (a wider
   store + a top-16 sort, or a second merge pass), then normalize over 10.
2. **k < 4** (k = 1/2/3): the offset-0 row (ranks 0-3) also needs lane masking — same `drop_thr` idea, applied to
   `scores/indices_offset + 0` with the rank→tileid map there.
3. **Softmax / sqrt-softplus** normalization variants (see `SOFTMAX_NOTES.md`).

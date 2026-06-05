# Revisit audit — moreh reductions (mask + reduce kernels)

Audit of 5 moreh reduction compute kernels for raw LLK eltwise stages left unmigrated to
`compute_kernel_lib::eltwise_chain`. Read-only. Reduce stages use the separate
`compute_kernel_lib::reduce` family and are intentionally left alone.

Helper capabilities verified from:
- `ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp`
- `ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp`
- `ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp`

Key cross-cutting finding: `mask_tile_to_cb` (defined in
`ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp:536-575`) is a raw-LLK helper whose body is
*exactly* an eltwise chain:

```
cb_reserve_back(ocb); cb_wait_front(icb, itile+1); cb_wait_front(maskcb, mtile+1);
tile_regs_acquire();
copy_tile_init_with_dt(icb);    copy_tile(icb, itile, dst0);     // CopyTile<icb, D0, ..., Block/Offset>
copy_tile_init_with_dt(maskcb); copy_tile(maskcb, mtile, dst1);  // CopyTile<maskcb, D1>
mask_tile_init(); mask_tile(dst0, dst1);                          // Mask<DataFormat, D0>
tile_regs_commit(); tile_regs_wait();
pack_tile_with_dt(dst0, ocb);                                     // PackTile<ocb>
tile_regs_release();
if (pop) cb_pop_front(icb, pop); if (popm) cb_pop_front(maskcb, popm);
cb_push_back(ocb);
```

This is the identical shape already inlined as a chain elsewhere in these same kernels
(e.g. moreh_softmax_w.cpp:83-91, moreh_softmax_backward_h_large.cpp:42-50). The remaining
raw call sites are MIGRATABLE; the lifecycle simply mirrors the `pop`/`popm`/`itile`/`mtile`
args.

---

## moreh_mean_w.cpp

Path: `ttnn/cpp/ttnn/operations/moreh/moreh_mean/device/kernels/moreh_mean_w.cpp`

| Stage | Lines | Current LLK | Verdict |
|-------|-------|-------------|---------|
| Phase-1 W reduce (matmul accumulate over Wt-1) | 56-74 | `mm_init_short` + `matmul_tiles` + `pack_tile_with_dt` | OUT-OF-SCOPE:matmul (reduce via matmul-with-scaler, not eltwise) |
| do_mask_w mask step | 83-91 | already `eltwise_chain` (CopyTile+CopyTile+Mask+PackTile) | ALREADY MIGRATED |
| Final-tile reduce: copy accum + matmul | 95-115 | `copy_tile_init_with_dt`+`copy_tile`(cb_accum_dst) then `mm_init_short`+`matmul_tiles`+`pack_tile_with_dt` | OUT-OF-SCOPE:matmul — the `copy_tile(cb_accum_dst,0,reduce_dst_idx)` seeds DEST for the in-DEST matmul accumulation; it is part of the matmul reduce, not a standalone eltwise copy |

Migratable stages: NONE. Masking is already migrated; the rest is matmul-based W-reduce.

---

## moreh_softmax_w.cpp

Path: `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/moreh_softmax_w.cpp`

| Stage | Lines | Current LLK | Verdict |
|-------|-------|-------------|---------|
| Wt==1 mask | 53 | `mask_tile_to_cb(cb_in0, cb_mask, cb_tmp, 0, 0, pop0=0, popm=0)` | **MIGRATABLE** |
| Phase-1 bulk MAX reduce | 59-77 | `reduce_init`/`reduce_tile`(MAX) + `pack_tile_with_dt` | OUT-OF-SCOPE:reduce |
| Phase-2 mask (last tile) | 80 | `mask_tile_to_cb(cb_in0, cb_mask, cb_tmp, Wt-1, 0, pop0=0, popm=0)` | **MIGRATABLE** |
| Phase-2 copy cb_max + reduce-merge MAX | 86-99 | `copy_tile_init_with_dt`+`copy_tile`(cb_max) seeding DEST for reduce-merge | OUT-OF-SCOPE:reduce (copy seeds DEST for the in-DEST MAX reduce-merge) |
| x - max (sub bcast Col) | 111-122 | already `sub<>` | ALREADY MIGRATED |
| exp(x-max) bulk + last+mask | 129-166 | already `eltwise_chain` ×2 | ALREADY MIGRATED |
| SUM reduce / recip / log | 168-196 | `reduce` family | OUT-OF-SCOPE:reduce |
| final mul/sub bcast Col | 207-235 | already `mul<>` / `sub<>` | ALREADY MIGRATED |

Migratable stages (2):
- **Wt==1 mask** (line 53): chain `CopyTile<cb_in0> + CopyTile<cb_mask, D1, HeldStream> + Mask<Float16_b, D0> + PackTile<cb_tmp>` over `onetile`. pop0=0/popm=0 → both inputs non-popping/held (`HeldStream` or `CallerManaged`; outer code holds cb_mask via the loop-level wait at line 46 and pops once at end — use `CallerManaged`/`HeldStream` to match pop=0).
- **Phase-2 mask** (line 80): same chain but cb_in0 read at offset `Wt-1` → `CopyTile<cb_in0, D0, CallerManaged, Input, OperandKind::Block, TileOffset::Set>{Wt-1}` (TileOffset::Set + Block/CallerManaged is legal per `is_legal_input_lifecycle_with_base`). cb_mask `CallerManaged`. PackTile<cb_tmp>. pop0=0/popm=0 → no pops.

---

## moreh_softmax_w_large.cpp

Path: `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/moreh_softmax_w_large.cpp`

| Stage | Lines | Current LLK | Verdict |
|-------|-------|-------------|---------|
| Wt==1 mask | 42 | `mask_tile_to_cb(cb_in0, cb_mask, cb_tmp, 0, 0, pop0=1, popm=0)` | **MIGRATABLE** |
| Phase-1 pop-as-we-go MAX reduce | 51-70 | `reduce_init`/`reduce_tile`(MAX) + `pack_tile_with_dt` | OUT-OF-SCOPE:reduce |
| Phase-2 mask (last tile) | 73 | `mask_tile_to_cb(cb_in0, cb_mask, cb_tmp, 0, 0, pop0=1, popm=0)` | **MIGRATABLE** |
| Phase-2 copy cb_max + reduce-merge MAX | 79-91 | `copy_tile`(cb_max) seeding DEST for MAX reduce-merge | OUT-OF-SCOPE:reduce |
| step1 exp(x-max)[+mask] | 105-173 | already `eltwise_chain` (all variants) | ALREADY MIGRATED |
| step1 accumulate (copy/add) | 177-181 | already `copy<>`/`add<>` | ALREADY MIGRATED |
| SUM reduce / log / recip | 184-210 | `reduce` family | OUT-OF-SCOPE:reduce |
| step3 final sub/mul/exp | 213-284 | already `sub<>`/`mul<>`/`eltwise_chain` | ALREADY MIGRATED |

Migratable stages (2):
- **Wt==1 mask** (line 42): chain `CopyTile<cb_in0, D0, NoWaitPop or Streaming> + CopyTile<cb_mask, D1, HeldStream> + Mask<Float16_b, D0> + PackTile<cb_tmp>` over `onetile`. pop0=1 → cb_in0 pops 1 (Streaming-style wait+pop, itile=0). popm=0 → cb_mask held (`HeldStream`).
- **Phase-2 mask** (line 73): identical (itile=0, pop0=1, popm=0). Same chain as Wt==1 case — `CopyTile<cb_in0>` (wait+pop 1) + `CopyTile<cb_mask, D1, HeldStream>` + `Mask` + `PackTile<cb_tmp>`.

---

## moreh_softmax_backward_h_large.cpp

Path: `ttnn/cpp/ttnn/operations/moreh/moreh_softmax_backward/device/kernels/moreh_softmax_backward_h_large.cpp`

Fully migrated. Every eltwise stage already uses the chain/convenience helpers:

| Stage | Lines | Status |
|-------|-------|--------|
| LOG: mask last tile (single / accumulate) | 42-62 | already `eltwise_chain` + `add<>` |
| LOG: copy/add accumulate | 63-70 | already `copy<>`/`add<>` |
| LOG: SUM reduce (COL) | 74-75 | OUT-OF-SCOPE:reduce |
| LOG: exp / mul-bcast-Row / sub | 79-96 | already `unary<Exp>`/`mul<>`/`sub<>` |
| non-LOG: y*dy[+mask] | 105-116 | already `eltwise_chain` + `mul<>` |
| non-LOG: copy/add accumulate | 119-123 | already `copy<>`/`add<>` |
| non-LOG: SUM reduce (COL) | 126-127 | OUT-OF-SCOPE:reduce |
| non-LOG: sub-bcast-Row / mul[+Neg] | 130-148 | already `sub<>`/`mul<>`/`eltwise_chain` |

Migratable stages: NONE (all eltwise already migrated; only the two reduces remain, out of scope).

---

## moreh_sum_w.cpp

Path: `ttnn/cpp/ttnn/operations/moreh/moreh_sum/device/moreh_sum_w_impl_kernels/moreh_sum_w.cpp`

Structurally identical to moreh_mean_w.cpp (matmul-with-scaler W-reduce + mask).

| Stage | Lines | Current LLK | Verdict |
|-------|-------|-------------|---------|
| Phase-1 W reduce (matmul accumulate over Wt-1) | 51-70 | `mm_init_short` + `matmul_tiles` + `pack_tile`/`pack_reconfig_data_format` | OUT-OF-SCOPE:matmul |
| do_mask_w mask step | 78-86 | already `eltwise_chain` (CopyTile+CopyTile+Mask+PackTile) | ALREADY MIGRATED |
| Final-tile reduce: copy accum + matmul | 90-114 | `copy_tile_to_dst_init_short`+`copy_tile`(cb_accum_dst) then `mm_init_short`+`matmul_tiles`+`pack_tile` | OUT-OF-SCOPE:matmul — copy seeds DEST for in-DEST matmul accumulation |

Migratable stages: NONE. Masking already migrated; the rest is matmul-based W-reduce.

---

## Summary

| Kernel | Migratable | Already migrated | Out-of-scope |
|--------|-----------|------------------|--------------|
| moreh_mean_w.cpp | 0 | 1 (mask) | 2 (matmul-reduce) |
| moreh_softmax_w.cpp | **2** (Wt==1 mask, phase-2 mask) | 4 | 3 (max/sum reduce) |
| moreh_softmax_w_large.cpp | **2** (Wt==1 mask, phase-2 mask) | 4 | 3 (max/sum reduce) |
| moreh_softmax_backward_h_large.cpp | 0 | 8 | 2 (sum reduce) |
| moreh_sum_w.cpp | 0 | 1 (mask) | 2 (matmul-reduce) |

Total migratable: **4 stages**, all the same pattern — raw `mask_tile_to_cb(...)` calls in
the two `moreh_softmax_*_w` kernels' `Wt==1` and `phase-2` (masked-last-tile) branches.
Each maps to `CopyTile + CopyTile<mask,D1> + Mask + PackTile`, with lifecycle/offset taken
from the call's pop0/popm/itile args. The inline `eltwise_chain` mask sites already present
in these very files are the migration template.

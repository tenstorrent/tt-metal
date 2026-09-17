# Metal 2.0 Post-Port Passes — `moreh_mean`

Procedure: `docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/post_port/pass_procedure.md`.
Each section below is one pass, in the Step 5 report shape.

**Op:** `ttnn/cpp/ttnn/operations/moreh/moreh_mean/`
**Sentinels:** `tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_mean.py`
**Passes run:** 3 — one style, two semantic.

## Run provenance

| Pass | Recipe | Branch | Outcome |
|---|---|---|---|
| 1 | `style/sync_free_dfbs` | `anasuya/post_port_moreh_mean` | NO SITES FOUND |
| 2 | `semantic/dm_self_loop_dfbs` | `anasuya/post_port_moreh_mean` | NO SITES FOUND |
| 3 | `semantic/gen2_hardware_configs` | `anasuya/post_port_moreh_mean` | APPLIED (2 sites) |

Pass 3 shifted the H and W factories by one line each, so the numbers in the original write-up no
longer all resolve. The two semantic passes ran back-to-back on this branch, each with its own
baseline → apply → verify cycle.

---

## Pass 1 — `style/sync_free_dfbs`

### Outcome

`NO SITES FOUND`. All 17 `DataflowBufferSpec`s across the op's three program factories carry real
FIFO synchronization. Nothing converted; op directory untouched.

### Sites

None. Every DFB surveyed, with what synchronizes it:

| Factory | DFB | Synchronized by |
|---|---|---|
| `_w` (`device/moreh_mean_w_program_factory.cpp:80-115`) | `input`, `masked_input` | reader `reserve_back`/`push_back`; compute `wait_front`/`pop_front` via the runtime-selected `cb_input` handle (`device/kernels/moreh_mean_w.cpp:56-119`) |
| | `scaler`, `mask_w`, `accum_dst`, `out` | direct `wait_front`/`pop_front`/`reserve_back`/`push_back` in compute; producer side via `generate_mm_scaler` / `generate_mask_w` |
| `_h` (`device/moreh_mean_h_program_factory.cpp:80-115`) | `input`, `scaler`, `mask_h`, `masked_input` | direct calls in `device/kernels/moreh_mean_h.cpp`; producers via `calculate_and_prepare_reduce_scaler` / `generate_mask_h` |
| | `accum_dst`, `out` | no hit anywhere in the op directory — see below |
| `_nc` (`device/moreh_mean_nc_program_factory.cpp:85-114`) | `input`, `in1`, `scalar`, `intermed0`, `out` | direct calls in `device/kernels/moreh_mean_nc.cpp`; `in1`/`scalar` filled by `fill_cb_with_value` |

**The two that grep alone got wrong.** `_h`'s `accum_dst` and `out` produce zero hits on a recursive
grep of the whole op directory — the recipe's way 4, and the same shape as its cited `moreh_dot`
case. Both ids are passed as template arguments into
`compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM, dfb::input, dfb::scaler, dfb::accum_dst>` and
through `compute_kernel_lib::Accumulate::at(dfb::accum_dst, …)` at
`device/kernels/moreh_mean_h.cpp:50-88`. The credit calls live **outside** the op, in
`ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.inl:227,251` (`accum_dfb.wait_front` /
`accum_dfb.pop_front`) and `:465-470` (`output_dfb.reserve_back` / `output_dfb.push_back`).
Converting either to a `Scratchpad` would have deleted live synchronization inside a compute kernel.

### Verification

Baseline: `./build_metal.sh --build-tests` exit 0, then the sentinel file → **76 passed, 72 skipped**
in 52.92s.

No re-verify run: the survey produced no change, so the baseline is the end state. The diff is
empty — this is "nothing was edited", not "verified green after an edit".

### Noticed, not done

- **Per-configuration sync-free DFBs, correctly not sites.** `mask_w`/`mask_h` and `masked_input`
  are genuine FIFOs when masking is on and entirely untouched when it is off (the compute-side
  `wait_front`/`pop_front` are compile-time eliminated). That is the recipe's way-3 "sync-free on
  only some paths → not a site". Worth flagging because the host code deliberately adds a
  compensating self-loop `PRODUCER` binding on `mask_*` in the no-mask configuration purely so the
  DFB still presents one endpoint of each kind
  (`device/moreh_mean_w_program_factory.cpp:245-251`, `device/moreh_mean_h_program_factory.cpp:248-254`)
  — a DFB declared, allocated, and endpoint-balanced to hold a buffer nobody touches. If the recipes
  ever grow a story for configuration-conditional buffers, this op is a clean specimen.
- **`if` vs `if constexpr` divergence on the same construct.** `device/kernels/moreh_mean_w.cpp:41`
  uses `if (do_mask_w)` on a `constexpr bool`, where `device/kernels/moreh_mean_h.cpp:37` uses
  `if constexpr (do_mask_h)` for the identical pattern. Same net effect at O3, but the H spelling is
  the one that guarantees elimination.
- **Dead compile-time arg.** `_nc` passes `units_per_core` to both compute `KernelSpec`s and the
  kernel never reads it — `device/kernels/moreh_mean_nc.cpp` takes both loop bounds from RTAs. The
  host comment at `device/moreh_mean_nc_program_factory.cpp:215-218` says this is deliberate, kept
  as a 1:1 image of the legacy descriptors. Fine, but it means the two per-group specs differ only
  in a value nothing consumes.
- **Unused include.** `device/kernels/reader_moreh_mean_nc.cpp:5` includes `api/debug/dprint.h` with
  no `DPRINT` use anywhere in the file. It is the only file in the op that includes it.
- **`in1` as a zero-tile identity.** `_nc` allocates a full DFB tile filled with `0.0f`
  (`fill_cb_with_value`) solely to serve as the add identity on the first accumulation iteration. It
  is properly synchronized, so out of scope here — but it spends a DFB id on a constant, which
  matters given the id-budget pressure this recipe cites.
- **No `borrowed_from` is set on any spec in this op**, so the `LocalTensorAccessor` fork of the
  recipe was never reachable — any site would have been a `Scratchpad`.
- **No `semantic/dm_self_loop_dfbs` follow-up implied.** The self-loops here (`accum_dst`,
  `masked_input`, `intermed0`) all run full FIFO machinery against themselves, which is that
  recipe's way-1 "not a site" case rather than a fake-FIFO. Pass 2 below confirmed this
  independently, on the separate ground that they are all compute-bound.

---

## Pass 2 — `semantic/dm_self_loop_dfbs`

### Outcome

`NO SITES FOUND`. The op has seven self-loop DFB bindings, but every one is bound by a **compute**
`KernelSpec`. The recipe is data-movement-only and excludes these by name: *"Do not generalize this
pass to compute kernels. Self-loop DFBs are supported there — on Gen2 as well as Gen1 — and are not
a problem to be solved."*

### Sites

None. The self-loops found and correctly left alone:

| File:line | DFB | Binder |
|---|---|---|
| `device/moreh_mean_h_program_factory.cpp:222,227` / `:233,238` | `ACCUM_DST_DFB`, `MASKED_INPUT_DFB` | compute |
| `device/moreh_mean_h_program_factory.cpp:248-254` | `MASK_H_DFB` (when `!do_mask_h`) | compute |
| `device/moreh_mean_w_program_factory.cpp:218,223` / `:230,235` | `ACCUM_DST_DFB`, `MASKED_INPUT_DFB` | compute |
| `device/moreh_mean_w_program_factory.cpp:245-251` | `MASK_W_DFB` (when `!do_mask_w`) | compute |
| `device/moreh_mean_nc_program_factory.cpp:200,205` | `INTERMED0_DFB` | compute |

Applying the recipe's spec-level test to the DM kernels: the readers bind `input`/`scaler`/`mask_*`
as `PRODUCER` only, and the writers bind `out` as `CONSUMER` only. No DM `KernelSpec` takes both
roles on any DFB in any of the three factories, so there is no fake-FIFO DM self-loop.

Survey steps 4-6 were not reached — with no candidate DFB there was nothing to account for against
the covered-use list, and no `borrowed_from` or `dfb_run_overrides` question to ask. (Pass 1
separately established that no spec in this op sets `borrowed_from`.)

### Verification

No change made. The pass's baseline — `./build_metal.sh --build-tests` exit 0, sentinel file
**76 passed, 72 skipped, 0 failed** (148 collected) — is also its end state.

### Noticed, not done

- **The mask-DFB self-loop is a structural formality, and an owner should know it.** In the H and W
  factories, when masking is off the compute kernel binds `MASK_*_DFB` as *both* endpoints purely so
  the DFB presents one endpoint of each kind — nothing ever produces real data into it, and the
  kernel's FIFO calls are compile-time eliminated. This is legal, is a compute self-loop, and
  therefore survives Quasar, so neither recipe touches it. But it is a buffer whose declared
  structure does not describe what happens, which is the same *category* of thing the DM recipe
  exists to prevent. This is the same construct pass 1 flagged from the other direction; two
  independent passes arriving at it is the reason it is repeated here.
- **The recipe's "way-1 vs fake-FIFO" distinction held cleanly on this op.** Every self-loop here
  runs real FIFO machinery *and* is compute-bound, so both of the recipe's exclusion grounds apply
  independently. An op where only one applied would be the interesting case; this is not it.

---

## Pass 3 — `semantic/gen2_hardware_configs`

### Outcome

`APPLIED` (2 sites).

### Sites

- `device/moreh_mean_h_program_factory.cpp:189` — shape 3, `std::get_if` form
- `device/moreh_mean_w_program_factory.cpp:186` — shape 3, `std::get_if` form

Both were the **quieter `get_if` variant** the recipe calls out. On Quasar
`to_compute_hardware_config` returns the Gen2 alternative, `std::get_if<ComputeGen1Config>` yields
null, the block silently does not execute, and `unpack_modes` is simply never set — no
`bad_variant_access`, no crash, nothing to make anyone notice. Per the recipe this is recorded
explicitly: the absence of a Quasar exception is why these survived the port.

Each was converted to the arch-agnostic `unpack_modes()` accessor
(`tt_metal/api/tt-metalium/experimental/metal2_host_api/compute_hardware_config.hpp:229`) with the
`#52269` marker beside it. Entries copied verbatim — `UnpackToDest` in H, `UnpackToSrc` in W. No
arch branch introduced, which is correct for shape 3: the helper already used `arch` to choose the
alternative, and the accessor reaches whichever one it chose.

The guard at each site carried **two** conditions — `compute_gen1 && fp32_dest_acc_en`. Only the
generation-naming half was removed; `fp32_dest_acc_en` is a genuine runtime condition and survives
as the `if`.

**Non-sites, confirmed:** `_nc` is shape 2 (`device/moreh_mean_nc_program_factory.cpp:171`, helper
with nothing set afterwards) and all six DM configs are shape 1
(`create_reader_datamovement_config(device->arch())` / `create_writer_datamovement_config(…)`). No
shape 4 anywhere — the op contains no hand-written Gen1 config.

**Gen1 path unchanged.** Reviewed against the recipe's "The Gen1 path must come out unchanged"
requirement: on Gen1 the accessor `std::visit`s to the same `ComputeGen1Config::unpack_modes` field
the `get_if` branch wrote, with the same value, under the same `fp32_dest_acc_en` condition. No
field, value or ordering changed. No custom config was rerouted through the arch-agnostic TTNN
helper.

**No marker in `_nc`, and that absence is meaningful.** The recipe puts the `#52269` marker wherever
*this pass* causes Quasar's `unpack_modes` to take a Gen1-derived value. `_nc` sets nothing after
the helper, so on Quasar it takes the helper's own default — nothing Gen1-derived to mark. Recorded
so a later reader does not read the gap as an oversight.

### Verification

Sentinel set run whole-file, no filter, before and after:

- Before: **76 passed, 72 skipped, 0 failed** (148 collected)
- After: **76 passed, 72 skipped, 0 failed** (148 collected)

Builds: `./build_metal.sh --build-tests` exit 0 both times.

Two limits on what that verification is entitled to conclude, stated rather than implied:

1. `pytest -q` gave counts, not per-test names, so this is a count-level and collection-level match
   (148 collected both runs, no filter, so no empty-selection false green), not a name-by-name
   comparison.
2. Per the recipe's Step 4 override, **nothing here checks the Gen2 values** — that branch is never
   taken on the Gen1 bench this ran on. The build checks the Gen2 config's *structure*; the
   sentinels check the Gen1 path. Correctness of the Gen2 side rests on the transcription being
   mechanical, which it was: entries copied unchanged, no field decided.

### Noticed, not done

- **`std::get<m2::ComputeHardwareConfig>` on the outer variant was left alone**, per the recipe's
  "what not to tidy" list. It distinguishes a compute config from a DM one, which is a programming
  error rather than something the architecture decides, and it does not throw on Quasar.
- **`bfp_pack_precision_mode` does not appear in this op**, so the Gen1-only exception the recipe
  carves out never arose and no arch-guarded statement remains after the conversion.
- **No `TT_FATAL`-on-`holds_alternative` "Gen1-only" wrapper exists in this op**, so there was no
  guard to remove after converting the sites, and no half-converted-with-stale-guard risk.
- **`experimental/quasar/` was not consulted** at any point, in code or in reasoning, for any of the
  three passes.

---

## Feature requests to raise

None. None of the three recipes reached a condition that asks for one.

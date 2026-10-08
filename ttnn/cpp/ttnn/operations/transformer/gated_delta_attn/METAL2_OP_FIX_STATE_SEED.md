# Op fix (separate pre-port PR): seed `gated_delta_attn`'s state through its own buffer

**For:** the `transformer/gated_delta_attn` op owner. This is a functional kernel change, so it lands as its own PR **before** the Metal 2.0 port, followed by a re-audit.
**Why:** the Metal 2.0 port is blocked on the state CB `c_8` (`cb_S`), which has two producers. The alternative is the host-side validator relaxation in `METAL2_HOST_FIX_SELF_LOOP_MULTIBINDING.md`. **Either fix alone unblocks the port; both are not needed.** This one also removes the multi-binding flag, which is Gen2-illegal and Quasar-uplift debt, so it is the cleaner long-term shape.
**Base:** written against the legacy (pre-port) code at `9e90e641f58`. Line numbers refer to that revision.

## The problem

`c_8` / `cb_S` is filled by two kernels on the same core:

- **Reader** seeds it once: `reserve_back` / `push_back`, `device/kernels/dataflow/reader_gated_delta_attn.cpp:103-115`, with either the initial state or zeros.
- **Compute** consumes it every chunk (`wait_front` `:231`, `pop_front` `:415`). On every non-last chunk it also pushes the updated state back into it: step 7b, `dst_cb = is_last_chunk ? cb_final_state : cb_S`, `compute/gated_delta_attn.cpp:421-436`.

Metal 2.0 can express a buffer that one kernel both produces and consumes (a compute self-loop). It can also express a buffer one kernel produces and another consumes. It does not accept a self-looped buffer that a second kernel also produces into. The fix gives each role its own buffer:

- **`S0` (new, `c_7`):** reader → compute, holding the initial state. It is consumed once, on chunk 0.
- **`S` (`c_8`):** compute-only. Compute writes the recycled state into it and reads it on chunks ≥ 1.

`c_7` is free today (the factory comment at `device/gated_delta_attn_program_factory.cpp:98` says "CB7: unused (was identity_32)").

## Changes

### 1. Program factory: `device/gated_delta_attn_program_factory.cpp`

Allocate `c_7` with the same shape as `c_8`:

```cpp
-    // CB7: unused (was identity_32)
+    make_cb(7, df_f32, state_tiles, 1);  // S0 — initial state, seeded by reader, consumed on chunk 0
     make_cb(8, df_f32, state_tiles, 1);  // S — persistent state
```

Nothing else on the host changes: no new args, no new tensors.

**L1 cost: +64 KiB per core** (16 fp32 tiles). The per-core CB total goes from 305 tiles (1,249,280 B) to 321 tiles (1,314,816 B). Confirm this still fits the usable L1 on both WH and BH (the factory comment at `:90` cites a 1.5 MB budget). If it doesn't, use the zero-L1 variant at the end of this document.

### 2. Reader: `device/kernels/dataflow/reader_gated_delta_attn.cpp`

Seed `c_7` instead of `c_8`. Only the buffer changes; the read and zero-fill logic stays the same.

```cpp
-    constexpr uint32_t cb_S = tt::CBIndex::c_8;
+    constexpr uint32_t cb_S0 = tt::CBIndex::c_7;  // initial state for chunk 0
 ...
-    constexpr uint32_t f32_tile = get_tile_size(cb_S);
+    constexpr uint32_t f32_tile = get_tile_size(cb_S0);
 ...
-    CircularBuffer cb_S_o(cb_S);
+    CircularBuffer cb_S0_o(cb_S0);
```

Then rename `cb_S_o` → `cb_S0_o` in the seed block (`:103-115`: `reserve_back`, the `async_read` / `async_write_zeros` targets, `push_back`). Update the comment `// === Load initial state S into CB8 ===` to say CB7 / S0.

### 3. Compute: `device/kernels/compute/gated_delta_attn.cpp`

Read the state from `S0` on chunk 0 and from `S` afterwards. The write side (step 7b) is unchanged.

```cpp
 constexpr uint32_t cb_dl_exp = tt::CBIndex::c_6;
-// c_7: unused (was identity_32)
+constexpr uint32_t cb_S0 = tt::CBIndex::c_7;  // initial state (seeded by reader, read on chunk 0)
 constexpr uint32_t cb_S = tt::CBIndex::c_8;  // persistent state
```

```cpp
-    // Initial state pre-loaded by reader into cb_S.
-    CircularBuffer(cb_S).wait_front(state_tiles);
+    // Initial state pre-loaded by reader into cb_S0; later chunks read the state compute recycled into cb_S.
+    CircularBuffer(cb_S0).wait_front(state_tiles);

     for (uint32_t c = 0; c < num_chunks; c++) {
+        const uint32_t cb_state = (c == 0) ? cb_S0 : cb_S;
```

Inside the loop, replace `cb_S` with `cb_state` at the **read** sites only:

| line | site |
|---|---|
| `:284`, `:289` | step 1 `matmul_reconfig_and_init(cb_k_cum, cb_S, …)` / `matmul_tiles(cb_k_cum, cb_S, …)` |
| `:323`, `:328` | step 3 `matmul_reconfig_and_init(cb_q_decay, cb_S, …)` / `matmul_tiles(cb_q_decay, cb_S, …)` |
| `:405`, `:408` | step 7a `mul_bcast_scalar_init(cb_S, …)` / `mul_tiles_bcast_scalar(cb_S, …)` |
| `:415` | step 7a `CircularBuffer(cb_S).pop_front(state_tiles)` → `CircularBuffer(cb_state).pop_front(state_tiles)` |

**Leave these alone:**

- `:423` `dst_cb = is_last_chunk ? cb_final_state : cb_S`. The recycled state still goes into `cb_S`.
- `:228` `compute_kernel_hw_startup<SrcOrder::Reverse>(cb_v_beta_sc, cb_S, cb_v_cor)`. `S0` and `S` have identical formats, so the startup configuration is the same either way. Keeping `cb_S` avoids touching the "must run before any copy_init" comment.

**Ordering note.** Chunks ≥ 1 read `cb_S` without a `wait_front`. That is the same as today: the only `wait_front(cb_S)` is the one before the loop, and the recycled state is consumed without another wait. Today's code relies on the pack thread being in-order: step 1's `wait_front(cb_k_cum)` cannot pass until pack has pushed `k_cum`, which comes after the previous chunk's 7b push into `cb_S`. The fix keeps exactly that behavior. Adding an explicit `CircularBuffer(cb_S).wait_front(state_tiles)` at the top of chunks ≥ 1 would be more robust, but it is a separate change; don't fold it in unless the owner wants it.

## Expected effect

- **Numerics: bit-identical.** Every tile is computed from the same inputs by the same LLK sequence; only which L1 buffer holds the chunk-0 state changes. Verify bit-exactness against the pre-change op (see below), not just PCC.
- **Performance:** unchanged apart from the extra CB allocation. No extra copies.
- **Edge cases unchanged:** `NC == 1` (chunk 0 is also the last chunk, so the state goes straight to `cb_final_state` and `cb_S` is never written), and the existing `NC == 0` hang.

## Verification for the op-fix PR

1. Before the change, capture outputs (`output`, `final_state`) from the current op for a config sweep. After the change, require **bit-exact equality** (`torch.equal`). Include:
   - NC = 1, 2, 3, 4
   - with and without `initial_state`
   - BH from small up to the full grid (64 on an 8×8 WH)
   - a second run per config with fresh tensors (program-cache hit)
2. Sanity-check against `ttnn.operations.transformer_golden.gated_delta_attn_seq_golden` (PCC ≥ 0.999).
3. Run `models/experimental/gated_attention_gated_deltanet/tests/test_ttnn_validation.py::test_gated_deltanet_chunked_ttnn`. That test alone is not enough: it is NC = 1 with no `initial_state`, so it never exercises `c_7` with data or the recycle path.

The Metal 2.0 port session already has a harness that does steps 1–2: `/home/edwinlee/.claude/jobs/daa75c58/tmp/gda_harness.py` (`save` before the change, `compare` after). It lives in a job scratch directory and will be removed with the job; copy it out if you want to keep it.

## What the Metal 2.0 port then looks like

After this lands and a re-audit, the port's binding census becomes:

- `S0`: reader PRODUCER → compute CONSUMER. Plain 1P+1C.
- `S`: compute PRODUCER + CONSUMER. Plain self-loop, **no `allow_instance_multi_binding`**.
- Compute's runtime choice `cb_state = c == 0 ? S0 : S` becomes `c == 0 ? dfb::S0 : dfb::S`, the same pattern the port already uses for `dst_cb`.

Everything else in the existing port (branch `worktree-metal2-gated-delta-attn`, WIP commit `e9c863c3085`) carries over unchanged.

## Zero-L1 variant (only if +64 KiB doesn't fit)

`attn_tiles == state_tiles == 16`, because validation pins C = Dk = Dv = 128. So the reader can seed the initial state through `c_0` (`L_unit`):

1. The reader pushes the seed into `c_0` as its first push.
2. Before the chunk loop, compute `wait_front(c_0)`, copies the 16 tiles into `cb_S` (`copy_init` / `copy_tile` / `pack_tile`, `push_back`), then `pop_front(c_0)`.

`S` again becomes compute-only, with no extra L1. The cost is a 16-tile copy per launch, a buffer with a double role, and a hidden dependency on `attn_tiles == state_tiles`. Prefer the main fix.

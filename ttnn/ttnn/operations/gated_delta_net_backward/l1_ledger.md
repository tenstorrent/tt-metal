# L1 Ledger — gated_delta_net_backward

Companion to `op_design.md`. Schema and audits: `.claude/references/l1-footprint-discipline.md`.

**This ledger describes the IMPLEMENTATION.** Where the buffer inventory departs from the one
`op_design.md` sketched, the departure and its reason are recorded in
[Departures from the design's inventory](#departures-from-the-designs-inventory) and in the affected
row's `Shares with / why not` cell. Single source of truth for every number here:
`gated_delta_net_backward_program_descriptor.py::_cb_blocks()` / `_cb_pages()` (host). The device
does **not** recompute any of it: the seven uniform block sizes and the three named-slot counts
travel to `kernels/gdn_common.hpp` as compile-time args (`_BLOCK_CT_ORDER`), so there is one
definition and both sides read it. (Before the verification pass the formulas were written out twice,
once per language, with a "mirrors `_cb_pages()`" comment holding them together; a divergence there
is a fifo wrap, i.e. a hang, so it was routed through CT args instead.)

**Named block axes** (from `op_design.md` → Blocking Model → Axes):

| Abbrev | Axis | Extent knob | Implementation symbol |
|---|---|---|---|
| `b` | `batch` | `block_batch = 1` | flattened into `bh` |
| `h` | `head` | `block_head = 1` | flattened into `bh` |
| `n` | `chunk` | `block_chunks = 1` | `BLOCK_CHUNKS` |
| `C` | `token-in-chunk` | `block_chunk_tiles = Ct` | `Ct = chunk_size / 32` |
| `K` | `key_dim` | `block_key_tiles = Kt` | `Kt = ceil(K/32)` |
| `V` | `value_dim` | `block_val_tiles = Vb` | solved on host |
| `w` | `gather token window` | `gather_stage_tokens` | `GATHER_STAGE_TOKENS = 16` |

`spans` = the buffer's live set is simultaneously resident along that axis.
`streams` = the buffer is re-entered / re-filled as that axis advances, but never holds more than
one position of it at a time.

`tf32 = tile_size(float32) = 4096`, `tin = tile_size(in_dtype)`, `e = element_size(in_dtype)`,
`Dg = GATHER_DEPTH = 2`, `De = EGRESS_DEPTH = 2`, `Da = ACCUM_DEPTH = 2`.

Derived block sizes (`_cb_blocks()`; passed to the kernels as CT args 40..46):

```
MAXV     = max(Ct, Kt) * Vb                                 cb_vin block
LVB      = max(2*Ct*Kt + Ct*Vb + Kt*Vb + 1, 2*Ct*Vb + 2*Kt*Vb)   cb_load_vb block
LITEM    = Ct*Ct + 4*Ct                                     cb_load_item block
MAXBLK   = max(Ct*Ct, Ct*Kt, Ct*Vb, Kt*Vb, 4*Ct)            cb_egr block
MAXBLK_G = max(Ct*Kt, Ct*Vb, Ct)                            cb_gegr block
```

---

## The two capacity patterns (why so many CBs are `Da × block`)

A compute→compute CB is the *only* synchronization between the TRISC pack thread (which produces it)
and the unpack thread (which consumes it), so a value cannot be transformed in place without a
push/pop round trip. Every working buffer in this op therefore uses one of exactly two shapes, and
no other:

* **`capacity == block`** — produce once, read by index, pop. The read pointer is always at the
  fifo base, so a block can never straddle the wrap point.
* **`capacity == Da * block`** — the in-place update: wait the old block at the front, reserve the
  other half, pack the new value, push, pop the old. The pointers alternate between two aligned
  halves, so again no block ever wraps.

This is not a depth-for-pipelining knob; it is what makes `X <- f(X)` expressible at all. A
multi-block pop or a multi-block group at a non-group-aligned read pointer *does* wrap, and that was
a real hang in this op (`cb pop_front: fifo_rd_ptr would exceed fifo_limit`) — which is why the three
column-list CBs below have capacity equal to their exact per-item push count.

---

## Buffer table

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_const` | `6*Ct*Ct + 2 + Ct` | all of it | `{b,h,n,K,V,w: streams; C: spans→Ct×Ct}` | `Float32` | reader | compute | whole program | **Already the merge.** Six `[C,C]` masks (`LT`, `NSTRICT`, `UT`, `EYE`, `BIAS`, `SUT`), the scale column, the zero tile and `Ct` row-of-ones tiles are ONE CB, not nine. Pushed once at boot and never popped, so its lifetime overlaps everything; merging was forced by the 32-index CB budget and costs nothing (a constant block is addressed by offset either way). `NSTRICT` carries `-1` so it is simultaneously the strict-lower mask and the negation both its consumers want; `BIAS` (`-1e4` above the diagonal) replaces a separate tril multiply on `L`. |
| `cb_colones` | `max(Ct, Kt, Vb)` | same | `{b,h,n,C,K,V,w: streams}` | `Float32` | reader | compute | whole program | **No share.** Column-0 ones; `X @ colones` is every row-sum in the op. Replaces the design's `cb_reduce_scaler` (see Departures). Cannot fold into `cb_const`: it is the `in1` operand while `cb_const` is the `in0` operand of the *same* matmuls, and one CB cannot configure both unpackers. |
| `cb_gather` | `Dg` pages of `GATHER_STAGE_TOKENS * row_span_stride + 64` B | one staging window | `{b,h,n,C,K,V: streams; w: spans→GATHER_STAGE_TOKENS}` | raw bytes | reader | reader | stage P (writer re-uses it as scalar-scatter staging in stage G) | **No share.** Reader-local scratch; the generic-op model has no L1 allocator other than a CB. The only non-tile-paged buffer and the only one spanning `w`. Capacity is `Dg ×` the live set: that is the software pipeline (window `w`'s reads in flight while window `w-1` is re-packed), and `GATHER_STAGE_TOKENS` was halved 32→16 when `Dg` went 1→2 so the product stays L1-neutral. |
| `cb_qin` | `Ct*Kt` | same | `{b,h,n,V,w: streams; C: spans→Ct; K: spans→Kt}` | `in_dtype` | reader | compute | stages P, G | **No share.** Live with `cb_kin` (both feed `q̃@kᵀ`). Capacity == block: produced once per item, popped once. |
| `cb_kin` | `Ct*Kt` | same | as `cb_qin` | `in_dtype` | reader | compute | stages P, G | **No share.** Concurrent with `cb_qin`. |
| `cb_vin` | `MAXV = max(Ct,Kt)*Vb` | `Ct*Vb` (P/G) or `Kt*Vb` (S) | `{b,h,n,w: streams; C: spans→Ct; V: spans→Vb}` | `in_dtype` | reader | compute | stages P, S, G | **Shares by role reuse:** the gathered/compact `v` slice **and** the `initial_state` / `dht` slices, which are `in_dtype` and so cannot go through the float32 `cb_load_vb`. Capacity is the max of the two blocks because both are transferred as one uniform push size — a CB with two push sizes wraps. |
| `cb_doin` | `Ct*Vb` | same | as `cb_vin` | `in_dtype` | reader | compute | stages P, G | **No share.** Concurrent with `cb_vin` in every phase that reads both. |
| `cb_gatein` | `2*Ct` | same | `{b,h,n,K,V,w: streams; C: spans→Ct (2 column-tiles)}` | `in_dtype` | reader | compute | stage P | **No share.** `g` and `β` are two columns of one block and already share this CB. Cannot merge with `cb_veca`: the reader produces this one and compute produces that one, and a CB may have only one producer. |
| `cb_load_item` | `LITEM = Ct*Ct + 4*Ct` | same | `{b,h,n,K,V,w: streams; C: spans→Ct×Ct and →Ct}` | `Float32` | reader | compute | stage G | **Already the merge.** `Tinv` plus the four decay column-tiles arrive in ONE transfer; giving each its own CB would cost four more of the 32 indices for four blocks with identical lifetimes. |
| `cb_load_vb` | `LVB` | same | `{b,h,n,w: streams; C: spans→Ct; K: spans→Kt; V: spans→Vb}` | `Float32` | reader | compute | stages S, G | **Already the merge.** Every per-V-block float32 scratch load (stage S: `kcd,P,v_corr,c,Γ`; stage G: `v_new,dv_new,S,dS`) is one uniform `LVB` push. Uniform because two different push sizes on one CB wrap the fifo; the tail of the smaller group is simply unused. |
| `cb_ka` | `Da*Ct*Kt` | `Ct*Kt` | `{b,h,n,V,w: streams; C: spans→Ct; K: spans→Kt}` | `Float32` | compute | compute | stages P, G | **Shares by role reuse:** `q̃ → Q` (stage P), `U → (ndU·U) → (dQ·q̃) → kβ` (stage G) — six disjoint lifetimes, one allocation. `Da` is the in-place mechanism, not pipelining. |
| `cb_kb` | `Da*Ct*Kt` | `Ct*Kt` | as `cb_ka` | `Float32` | compute | compute | stages P, G | **Shares by role reuse:** `k → P` (stage P), `k` (stage G, live to the last phase: `M@k`, `W@k`, `dP·k`, `kβ`). Cannot merge with `cb_ka`: `q̃` and `k` are both operands of `q̃@kᵀ`. |
| `cb_kc` | `Da*Ct*Kt` | `Ct*Kt` | `{…; V: streams — the V loop ACCUMULATES into it, so it must not scale with Vb, and it does not}` | `Float32` | compute | compute | stages P, G | **Shares by role reuse:** `kβ → U` (stage P), `dQ` accumulator → `dq` → `W@k` → `dkβ` (stage G). |
| `cb_kd` | `Da*Ct*Kt` | `Ct*Kt` | as `cb_kc` | `Float32` | compute | compute | stages P, G | **Shares by role reuse:** `kcd` (stage P), `dP` accumulator → `dP·w` → `W^T@kβ` → `dkβ·β` (stage G). |
| `cb_ke` | `Da*Ct*Kt` | `Ct*Kt` | as `cb_ka` | `Float32` | compute | compute | stage G | **No share.** `q̃` is live from the head of the stage-G item to `dk`'s first term, concurrently with all five other `[C,K]` buffers. |
| `cb_kf` | `Da*Ct*Kt` | `Ct*Kt` | as `cb_kc` | `Float32` | compute | compute | stage G | **Shares by role reuse:** `ndkcd` accumulator → six successive scratch products → the `dk` accumulator. This is the sixth `[C,K]` buffer; the design budgeted four, and the stage-G live set genuinely needs six (see Departures). |
| `cb_ktr` | `Da*Ct*Kt` | `Kt*Ct` or `Ct*Kt` | `{b,h,n,V,w: streams; C: spans→Ct; K: spans→Kt}` | `Float32` | compute | compute | stages P, S, G | **Shares by role reuse:** the materialized transposes `Qᵀ` (P), `Pᵀ` and `kcdᵀ` (S), and then `ndU` (G) — a `[K,C]` block and a `[C,K]` block have the same tile count, so one allocation serves both. Exists because `matmul` transposes **in1 only**; an `Aᵀ@B` must materialize `Aᵀ`. |
| `cb_va` | `Da*Ct*Vb` | `Ct*Vb` | `{b,h,n,K,w: streams; C: spans→Ct; V: spans→Vb}` | `Float32` | compute | compute | stages P, S, G | **Shares by role reuse:** `v_corr` (P), `v_new`/`P@dS` (S), `do` (G). |
| `cb_vb` | `Da*Ct*Vb` | `Ct*Vb` | as `cb_va` | `Float32` | compute | compute | stages P, S, G | **Shares by role reuse:** `do` (P), `kcd@S`/`dv_new` (S), `v → v_beta` in place (G). |
| `cb_vc` | `Da*Ct*Vb` | `Ct*Vb` | as `cb_va` | `Float32` | compute | compute | stages P, G | **Shares by role reuse:** `v → v_beta → u` (P), `dv_beta → (dv_beta·v) → v_new` (G). Cannot merge with `cb_va`/`cb_vb`: all three are live inside the stage-G V loop. |
| `cb_ca` | `Da*Ct*Ct` | `Ct*Ct` | `{b,h,n,K,V,w: streams; C: spans→Ct×Ct}` | `Float32` | compute | compute | stages P, G | **Shares by role reuse:** `L`'s column-broadcast operand and `Tinv` accumulator (P); then seven successive `[C,C]` scratches in stage G (`ndkcd@Uᵀ`, `d_attn@Tinvᵀ`, `q̃kᵀ`, `kβkᵀ`, `Rᵀ`, `Mᵀ`, `Wᵀ`). |
| `cb_cbL` | `Ct*Ct` | same | as `cb_ca` | `Float32` | compute | compute | stages P, G | **No share.** `L` is live from `decay_mask_block` to the last phase of its stage, concurrently with every other `[C,C]` buffer. Capacity == block: `L` is never updated in place. Recomputed in stage G from the stored decay columns rather than stored as a `[BH,NC,C,C]` DRAM tensor — an inventory decision, not a footprint one. |
| `cb_cc` | `Da*Ct*Ct` | `Ct*Ct` | `{…; V: streams — `Mraw` ACCUMULATES over the V loop}` | `Float32` | compute | compute | stages P, G | **Shares by role reuse:** `A → intra` (P), `Mraw` accumulator `→ M → Mᵀ` (G). |
| `cb_cd` | `Da*Ct*Ct` | `Ct*Ct` | as `cb_cc` | `Float32` | compute | compute | stages P, G | **Shares by role reuse:** `L`'s row-broadcast operand and the Neumann running power `A^{2^j}` (P), `d_attn` accumulator `→ dAn → W` (G). Cannot merge with `cb_cc`: the Neumann loop needs power and product live at once, and stage G needs `M` and `W` live at once. |
| `cb_ce` | `Da*Ct*Ct` | `Ct*Ct` | as `cb_ca` | `Float32` | compute | compute | stages P, G | **Shares by role reuse:** Neumann `Tinv@Pw` scratch and `intraᵀ` (P), `Tinvᵀ → R` (G). |
| `cb_sa` | `Da*Kt*Vb` | `Kt*Vb` | `{b,h,C,w: streams; n: streams (the state is carried ACROSS chunks — re-entered, never stacked); K: spans→Kt; V: spans→Vb}` | `Float32` | compute | compute | stages P, S, G | **Shares by role reuse:** `c_i` (P), the running `S` across **all** `tensor_chunks` steps (S forward) and the `kcdᵀ@dv_new` term (S reverse), `S_i`/`dS·S`/the `dGamma` transpose (G). The `n: streams` tag is load-bearing: the live set is ONE `[K,V]` state, not `NC` of them. |
| `cb_sb` | `Da*Kt*Vb` | `Kt*Vb` or `Kt` | as `cb_sa` | `Float32` | compute | compute | stages S, G | **Shares by role reuse:** `Pᵀ@v_new` and the running `dS` (S), the `[Kt,1]` `dGamma` column accumulator (G). The `Kt`-tile group is legal in a `Da*Kt*Vb` fifo because that capacity is a multiple of `Kt`. |
| `cb_veca` | `NUM_VECA_SLOTS*Ct = 5*Ct` | same | `{b,h,n,K,V,w: streams; C: spans→Ct (5 column-tiles)}` | `Float32` | compute | compute | stages P, G | **Already the merge.** `decay, γ, w, β, dc1` — all five live at once (`key_grad_block` and the decay-gradient phases read several in the same phase). Capacity is EXACTLY the five per-item pushes, so the read pointer returns to the fifo base every item and a five-block group can never wrap. `NUM_VECA_SLOTS` is a host constant and arrives as a CT arg; the kernel `static_assert`s it against its slot layout. |
| `cb_vecb` | `NUM_VECB_SLOTS*Ct = 9*Ct` | same | `{…; C: spans→Ct (9 column-tiles); V: streams}` | `Float32` | compute | compute | stage G | **Already the merge.** The nine stage-G column values (`dβ_v, s1, dγγ, rowsum(R-Rᵀ), dww, dβ, Γ, LT@dΓΓ, UT@e`). Capacity is EXACTLY nine — same wrap argument as `cb_veca`. The count is the host's `NUM_VECB_SLOTS`, arriving as a CT arg that the kernel `static_assert`s against its slot layout. |
| `cb_vecc` | `Da*Ct` | `Ct` | `{…; C: spans→Ct (1 column-tile)}` | `Float32` | compute | compute | stages P, S, G | **Shares by role reuse:** every single-block column transient (`g → rmg` in P, the `Γ` tile in S, `w`-temp / `s2 → s2·γ` / `dw` / `dβ`-partial / `dΓ → dΓΓ` / `e` / `dg` in G), plus the `dβ_v` V-loop accumulator. All of them are one-block groups, which is why `Da*Ct` suffices at any read-pointer alignment. |
| `cb_egr` | `De*MAXBLK` | `MAXBLK` | `{b,h,n,w: streams; C: spans→Ct; K: spans→Kt; V: spans→Vb}` | `Float32` | compute | writer | stages P, S | **No share — structural.** Every value it carries is also a live compute working buffer, and a CB may have only one consumer: the writer cannot read a compute→compute buffer. Sized to the largest block that crosses the boundary. `De` is the genuine overlap knob: the writer drains block `n` while compute fills `n+1`, load-bearing in stage S, the low-parallelism path. |
| `cb_gegr` | `De*MAXBLK_G` | `MAXBLK_G` | as `cb_egr`, plus `dh0` in stage S | `in_dtype` | compute | writer | stages S, G | **No share with `cb_egr`** — different page format, and both are live when the last scratch drain overlaps the first gradient write. All six gradients (and `dh0`) are sequenced through this one CB in a compile-time-fixed order rather than getting six CBs. |

**Audit 1 (capacity vs live set).** Three CBs have capacity > live set for *pipelining*:
`cb_gather` (`Dg`), `cb_egr` and `cb_gegr` (`De`), each with its overlap mechanism named. Fifteen have
capacity `Da ×` live set for the *in-place update*, which is a correctness mechanism, not a depth
knob — see [The two capacity patterns](#the-two-capacity-patterns-why-so-many-cbs-are-da--block).
The rest have capacity == live set. No CB `spans` an axis its capacity does not scale with: every
`spans→Ct` row carries a `Ct` factor, every `spans→Kt` a `Kt`, every `spans→Vb` a `Vb`; no capacity
expression mentions `B`, `T`, `H` or `NC`.

**Audit 2 (page format vs DEST width).** `fp32_dest_acc_en = True` by default, so DEST is 32-bit and
every `Float32` page is a value accumulated in fp32 DEST (states, `Tinv`, decay exponentials,
gradient accumulators). The `in_dtype` pages (`cb_qin/kin/vin/doin/gatein`, `cb_gegr`) are the op's
boundary buffers: they hold values that were already `in_dtype` in DRAM or are about to be written
back as `in_dtype`, so a wider page carries no extra precision. Measured caveat, recorded because it
bounds `dg`: a value that round-trips a `Float32` CB and is then read back **as an FPU operand**
lands in SrcA/SrcB at ~tf32 width (10 explicit mantissa bits), so the effective resolution of a
`Float32` page is set by its *consumer*, not by the page. That is where the `decay` column loses
`dg`'s last two digits at the saturated-gate setting, and it is why the values that must survive at
full width (`g`) are moved with `copy_tile` — the datacopy path unpacks straight to DEST at 32-bit
and never touches a source register (`transpose.h:112` / the `eltwise_unary` init selects
`UnpackToDestEn` from the operand's DEST format). Verified during the verification pass by
modelling the two `L` constructions at 10-bit operand width: the current
`decay ⊗ 1 − (decay ⊗ 1)ᵀ` form gives `L` a relative RMS error of 3.4e-2 at `g_scale = 8`, while the
algebraically identical `LT @ diag(g) @ strict_lower` form — whose operands are `g`, not the
cumulative sum — gives 6.0e-3. That 5.8× is the lever `op_requirements.md` Refinement 1 pulls. When
a caller passes `fp32_dest_acc_en = False` the `Float32` pages stay; they are the op's declared
precision floor, not a DEST artefact, and that is the one deliberate exception.

**Audit 3 (disjoint lifetime).** Every `Shares with / why not` cell is filled. Nineteen of the 32 CBs
carry two or more roles across disjoint lifetimes; every "No share" cell names the concurrent phase
that forbids the merge.

**Audit 4 (bounds and closed form).** Symbol table and closed form below; no unbounded op dimension
appears in any capacity expression.

---

## Symbol table

| Symbol | Meaning | Bound | Predicate establishing the bound |
|--------|---------|-------|----------------------------------|
| `Ct` | `block_chunk_tiles` | `Ct ∈ {1, 2}` | `validate()` enforces `chunk_size % 32 == 0`; `SUPPORTED["chunk_size"] = [32, 64]` |
| `Kt` | `block_key_tiles` = `ceil(K/32)` | `Kt ≤ 4` over INPUTS (`K ≤ 128`) | not a hard bound — the footprint solve consumes `Kt` and shrinks `Vb` as `Kt` grows. The solve bounds the *product*. Guarded: the host raises a clear `RuntimeError` naming the minimum footprint if even `Vb = 1` does not fit. |
| `Vt` | `ceil(V/32)` | `Vt ≤ 8` over INPUTS (`V ≤ 256`) | same — bounded through the solve |
| `Vb` | `block_val_tiles` | divides `Vt`; `footprint_bytes(Vb) ≤ l1_budget` | the host extent solve below |
| `H` | heads | `H ≤ 32` | mechanism cap, enforced in `validate()`: the source page index assumes `ceil(H/32) == 1`. Does **not** appear in the footprint. |
| `Dg`, `De`, `Da` | gather / egress / accum depth | 2 each | host constants |
| `GATHER_STAGE_TOKENS` | tokens per staging window | 16, must divide 32 | a destination tile row is 32 tokens; `WPT = 32 / GATHER_STAGE_TOKENS` windows per tile row |
| `row_span_stride` | staging line stride | `round_up(272*e + 64, 64)` | a tile row is a 272-element span; the `+64` lets a line start at `(dram_offset % 64)`, which is what the NoC requires of a DRAM read's L1 destination |
| `e` | `element_size(in_dtype)` | 4 (fp32) or 2 (bf16) | `SUPPORTED["dtype"]` |
| `l1_budget` | usable worker L1 | `get_max_worker_l1_unreserved_size() - L1_KERNEL_CONFIG_RESERVE` | `L1_KERNEL_CONFIG_RESERVE = 80 KB`: the query measures from the kernel-config base and still includes the ring buffer this program's five kernel binaries live in (~70 KB here). Budgeting the raw query overflows L1 at program build — measured. |

`B`, `T`, `NC` appear **nowhere** in the L1 footprint: they are absorbed entirely by the work-item
count and the DRAM scratch.

---

## Total per-core footprint (closed form)

Mirrors `_cb_pages()` exactly.

```
f32_tiles(Ct,Kt,Vb) =
      (6*Ct*Ct + 2 + Ct)                      # cb_const
    + max(Ct,Kt,Vb)                           # cb_colones
    + (Ct*Ct + 4*Ct)                          # cb_load_item
    + LVB                                     # cb_load_vb
    + Da*6*Ct*Kt + Da*Ct*Kt                   # cb_ka..kf, cb_ktr
    + Da*3*Ct*Vb                              # cb_va, cb_vb, cb_vc
    + (4*Da + 1)*Ct*Ct                        # cb_ca, cb_cc, cb_cd, cb_ce (+ cb_cbL)
    + Da*2*Kt*Vb                              # cb_sa, cb_sb
    + 5*Ct + 9*Ct + Da*Ct                     # cb_veca, cb_vecb, cb_vecc
    + De*MAXBLK                               # cb_egr

in_tiles(Ct,Kt,Vb) =
      2*Ct*Kt + MAXV + Ct*Vb + 2*Ct           # cb_qin, cb_kin, cb_vin, cb_doin, cb_gatein
    + De*MAXBLK_G                             # cb_gegr

footprint_bytes = tf32 * f32_tiles + tin * in_tiles
                + Dg * (GATHER_STAGE_TOKENS * row_span_stride + 64)     # cb_gather
```

Which term scales with which knob:

| Term | Scales with |
|------|-------------|
| `6*Ct*Ct`, `(4*Da+1)*Ct*Ct` | `block_chunk_tiles` **quadratically** — the `[C,C]` UT / attention matrices |
| `5*Ct`, `9*Ct`, `Da*Ct`, `2*Ct` | `block_chunk_tiles` — the column vectors |
| `Da*7*Ct*Kt`, `2*Ct*Kt` | `block_chunk_tiles × block_key_tiles` |
| `Da*3*Ct*Vb`, `Ct*Vb` | `block_chunk_tiles × block_val_tiles` |
| `Da*2*Kt*Vb`, `MAXV` | `block_key_tiles × block_val_tiles` — the state; the only term independent of `block_chunk_tiles` |
| `LVB` | the larger of the stage-S and stage-G per-V-block load groups |
| egress terms | `egress_depth ×` the largest block crossing the compute/writer boundary |
| gather term | `gather_depth × gather_stage_tokens` (their product is what costs; halving one pays for doubling the other) |
| nothing | `block_batch`, `block_head`, `block_chunks` — they select *which* block, not how big it is |

### Extent solve (the only host solve in this op)

```python
block_val_tiles = max(vb for vb in divisors(Vt)
                      if footprint_bytes(Ct, Kt, vb, e) <= l1_budget)
num_v_blocks    = Vt // block_val_tiles
```

Rule 1 (inventory before solve) was applied first: 19 of the 32 CBs carry two or more roles across
disjoint lifetimes; nine CB indices were saved by merging the constants into one block and the
per-item / per-V-block scratch loads into one transfer each; and three values that could have been
DRAM scratch are recomputed instead (`L` from the stored decay columns, `U` and `kβ` from `k`, `β`,
`γ`). Two phase boundaries that could have been storage are in-place transforms
(`v_corr → v_new`, `u → dv_new`).

### Measured footprint (host solve against the real device budget, `l1_budget = 1 449 984 B`)

Recomputed from `_cb_bytes()` during the verification pass. The `cb_veca` row above said `6*Ct`
where the code allocates `5*Ct`, which inflated every number in this table by exactly `Ct` f32
tiles; the corrected column now agrees with the **device-measured** peak to 0.1 KB (see below).

| Shape (B,T,H,K,V), chunk | `Ct` | `Kt` | `Vt` | `Vb` | `NVB` | fp32 | bf16 | `Vb = Vt` would be (fp32) |
|---|---|---|---|---|---|---|---|---|
| `(1,32,1,32,32)`, 32 | 1 | 1 | 1 | 1 | 1 | **376 KB** | 344 KB | 376 KB |
| `(1,512,2,64,64)`, 32 | 1 | 2 | 2 | 2 | 1 | **568 KB** | 520 KB | 568 KB |
| `(1,128,2,64,128)`, 64 | 2 | 2 | 4 | 4 | 1 | **1396 KB** | 1292 KB | 1396 KB |
| `(1,128,2,128,128)`, 64 | 2 | 4 | 4 | 1 | 4 | **1368 KB** | 1268 KB | 1940 KB — over budget |
| `(1,256,4,128,256)`, 64 | 2 | 4 | 8 | 1 | 8 | **1368 KB** | 1268 KB | 2884 KB — over budget |

**Corroborated on device.** The golden run captures per-test peak L1
(`metric.device_l1_peak_bytes`): 1396.1 KB at `(1,128,2,64,128)` c64 fp32 against the 1396 KB
predicted here, and 344.1 KB at `(1,32,1,32,32)` against 344 KB (bf16). The closed form is the
footprint, not an estimate of it.

The last two rows are why the solve exists and why `num_v_blocks > 1` is a real, tested code path:
the acceptance suite pins `(1,256,4,128,256)` precisely to exercise it. Note the footprint is not
monotone in the *shape* — it is monotone in the *solved block*, which is the point.

**`Vb = 1` at `Kt = 4` is the extent at its FLOOR, and that is the op's live perf question.** The
solve lands on 1368 KB of a 1416 KB budget, so raising `Vb` to 2 would need ~190 KB more than
exists; `num_v_blocks` is then 4 or 8, and the data-movement budget below shows exactly one term
that scales with it (stage S re-reading the V-independent `sc[kcd]` / `sc[p]`, once per V block).
`Da × block` on the seven `[C,K]` buffers is 448 KB and on the five `[C,C]` buffers 144 KB of that
1368 KB, so the bytes to buy `Vb = 2` exist in principle but only by shortening a stage-G lifetime
the Audit-3 column currently argues is concurrent. That trade is filed as a measured perf
refinement in `op_requirements.md`, not resolved here.

---

## Departures from the design's inventory

Recorded here because `op_design.md`'s Circular Buffers table is the planner's sketch and this is
what was built.

| Departure | Why |
|---|---|
| **No `cb_reduce_scaler`; `cb_colones` instead.** | The reduce helper is not used. Every reduction in this op is a contraction over an axis whose operand is already a resident block inside a matmul-shaped phase, and the helper's `Accumulate` path is single-tile while these outputs are `Ct` tiles. `X @ colones` is exact for SUM, keeps the pipeline in matmul state at all seven reduce sites, and needs no scaler. Full argument at the head of the compute kernel. |
| **Four constant CBs → one `cb_const`** (and six mask blocks, not four). | The 32-index CB budget. `SUT` (strict-upper ones) was added so the exclusive reverse cumsum is one matmul; `BIAS` was added so `L`'s upper triangle is masked by `exp(-1e4)` instead of a separate multiply — which also removes an `exp()` of a large positive argument, i.e. an `inf * 0 = NaN` at the saturated-gate setting. `ROWONES` was added so `L`'s column-broadcast matrix is an outer product, removing the unary- and ROW-broadcast LLK instantiations entirely (the program is kernel-config-ring-buffer bound; see the code-size note in the kernels). |
| **Six `[C,K]` CBs (`cb_ka..kf`) + `cb_ktr`, not four `cb_kmat_*`.** | The stage-G live set after the V loop is genuinely six: `U`, `k`, `dQ`, `dP`, `q̃`, `ndkcd`. `cb_ktr` is the materialized-transpose home the design's "matmul transposes in1 only" note requires. |
| **`cb_load_item` / `cb_load_vb` added.** | Stages S and G read derived scratch back from DRAM; it has to arrive in a CB, and the design's table has no reader→compute float32 buffer. |
| **`cb_decay_block` / `cb_grad_vec` → `cb_veca` (6) / `cb_vecb` (9) / `cb_vecc` (`Da`).** | Sized to the exact per-item push count so the fifo cycles back to base — the wrap rule above. `cb_vecc` absorbs every one-block transient. |
| **Most working CBs are `Da × block`, not depth 1.** | The in-place update is the only way a compute→compute CB can be transformed in place while keeping the pack and unpack threads synchronized. It is a correctness mechanism, not the design's "overlap" depth knob, and it roughly doubles the working-set term — the single largest reason the measured footprints above exceed the design's estimates. |
| **Two flat DRAM scratch tensors (`sc` float32, `scin` in_dtype) instead of 15 named ones.** | One `TensorAccessor` each instead of fifteen. Accessor compile-time args and kernel code size are both binding here, and a flat `[N*32, 32]` tiled tensor makes every block a tile-index range computed on host. |
| **`sc_p` added; `sc_vec` is `4*Ct`, not `3*Ct`.** | Stage S needs `P` and the design's `store_prep_block` list omits it. The fourth vec slot carries `w`: recomputing it in stage G as `exp(dc1 - decay)` subtracts two nearly-equal values of magnitude `|sum(g)|` at the FPU's ~tf32 source width, and `exp()` magnifies what survives. |
| **`GATHER_STAGE_TOKENS` 32 → 16.** | Paid for `GATHER_DEPTH` 1 → 2 at constant L1. Measured 1.01–1.02× (small because the reader is RISC-issue bound, not NoC bound). |
| **Block sizes and slot counts moved host → kernel as CT args** (verification pass). | They were previously written out in both languages with a "mirrors `_cb_pages()`" comment. Same value, two definitions, and a divergence is a hang — so the host is now the only definition and `gdn_common.hpp` reads CT args 40..49. `NUM_CONST_MASKS` / `NUM_VECA_SLOTS` / `NUM_VECB_SLOTS` travel the same way and are `static_assert`ed against the kernel's slot layout. No footprint change. |
| **`l1_budget` is the query minus `L1_KERNEL_CONFIG_RESERVE`.** | `get_max_worker_l1_unreserved_size()` measures from the kernel-config base and still includes the ring buffer holding this program's five kernel binaries. Budgeting the raw query overflows L1 at program build — measured, not theorized. |

---

## Data-movement budget

For the implemented split (regime R1), per public invocation. `e` = element size,
`L_qk = B·T·H·K·e`, `L_v = B·T·H·V·e`, `L_g = B·T·H·e`, `Skv = B·H·K·V·4`,
`NC = ceil(T/chunk_size)`, `NV = num_v_blocks`.

| Tensor | DRAM crossings | Why that many |
|--------|----------------|---------------|
| `q` | **3 × `L_qk`** — 1 face-row read (P), 1 compact write (P), 1 compact read (G) | The face-row read is the only one at 272-element-span granularity, and it is paid **once** because the reader writes a head-major compact copy on the way past. The two compact crossings are full-page. |
| `k` | 3 × `L_qk` | same |
| `v` | 3 × `L_v` | same |
| `do` | 3 × `L_v` | same |
| `g`, `beta` | 1 whole-page read per (b, token-tile) + 1 compact write (inside `sc_vec`) + 1 read (G) | The gate tensors are tiled over `(T,H)`, so one head is a *column*; the reader takes the whole page once per token tile and extracts 32 scalars, rather than 32 sub-16-byte DRAM reads. |
| `initial_state`, `dht` | 1 × `Skv` each | read once by the scan core, one V-column slice per V block (the slices partition `V`) |
| `sc[attn]` `[NI,C,C]` | 1 write (P) + 1 read (G) | `Tinv` costs `2·neumann_steps` block matmuls to rebuild; storing is cheaper |
| `sc[kcd]`, `sc[p]` `[NI,C,K]` | 1 write (P) + **`NV`** reads (S) + 1 read (G, `kcd` only) | both are V-independent, so stage S re-reads them once per V block. **This is the one traffic term that grows with `num_v_blocks`**, and it is why the extent solve takes the *coarsest* `Vb` that fits rather than a small one. |
| `sc[vcorr]`, `sc[u]` `[NI,C,V]` | 1 write (P) + 1 read (S) each | scan inputs |
| `sc[c]` `[NI,K,V]` | 1 write (P) + 1 read (S) | `c_i = Q_iᵀ@do_i` is precomputed in the *high-parallelism* stage so the sequential scan does 4 matmuls per chunk instead of 5 |
| `sc[vec]` `[NI,4,C]` | 1 write (P) + `NV` reads (S, the `dc1` tile only) + 1 read (G) | 4 column-tiles per item; the stage-S read is a single tile |
| `sc[S]`, `sc[dS]` `[NI,K,V]` | 1 write (S) + 1 read (G) each | `S_i` cannot be recovered backwards: `S_i = (S_{i+1} − Pᵀv_new_i)/Γ_i` divides by `Γ_i`, which is `exp(−53)` at the default gate distribution and numerically dead. Storing is the correctness-preserving choice. |
| `sc[vnew]`, `sc[dvnew]` `[NI,C,V]` | 1 write (S) + 1 read (G) each | stage-G inputs |
| `scin[q,k,v,do]` | (counted in the `q/k/v/do` rows) | the head-major compact copies |
| `dq`, `dk` | 1 write each (`L_qk`) | face-row scatter, rows `t < T` only |
| `dv` | 1 write (`L_v`) | face-row scatter |
| `dg`, `dbeta` | 1 write each (`L_g`) | scalar scatter into column `h`; each value is staged at a matching 16-byte modulo first, because the NoC requires `(l1_src & 15) == (dram_dst & 15)` for a DRAM write and the element offset of column `h` is not 16-byte aligned |
| `dh0` | 1 write (`Skv`) | whole pages from the scan core |
| **barriers** | — | `NC` semaphore increments per `bh` group (fan-in) + one per stage-G core in the group (fan-out). **No payload** — semaphore words only, twice per program. |

**Totals per tier.** *DRAM at face-row granularity (the governing term):* `2·L_qk + 2·L_v` read as
272-element spans — `B·T·(2·Kt + 2·Vt)` transactions spread over `min(grid, BH·NC)` cores — plus
`2·L_qk + L_v` written as 16-element runs. *DRAM at full-page granularity:* the compact copies and
the whole derived scratch. *Cross-core:* semaphore words only.

**Measured** (device kernel ns, fresh cache, `--profile`): the face-row gather is **16–39%** of the
whole op and the compact copy-out is ~1%. Stage P dominates except where `NC ≫ BH`, where the
sequential scan (regime R5, out of Phase 0) takes over. Dividing the gather term by the number of
gathered rows gives ~280 ns per row against a ~1 KB read: the reader is **RISC-issue bound, not NoC
bound**, which is why unrolling the re-pack won 1.04–1.09× while doubling the gather depth won only
1.01–1.02×.

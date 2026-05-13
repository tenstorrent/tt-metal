# Eltwise Helper — Gap Audit (Phase 1)

Compares the current `eltwise_*.{hpp,inl}` implementation against `eltwise_helper_lessons.md`. Phase 1 deliverable — gap map only, no code changes. Feeds the Gate-1 proposal in the next phase.

Files reviewed:
- `eltwise_chain.hpp` / `.inl`
- `eltwise_block.hpp`
- `eltwise_binary_sfpu.hpp`
- `eltwise_convenience.hpp`
- `eltwise_optional.hpp`
- `eltwise_{activations, math, misc, special, scalar, predicates, fill, rand, rounding, trig}.hpp`

Severity:
- **BLOCKER** — silent miscompile or correctness risk.
- **SHOULD-FIX** — design anti-pattern named explicitly by the lessons; fix before next migration cycle.
- **NICE** — cleanup / clarity / minor doc; safe to batch.

---

## BLOCKER — auto init-hoist on every non-clash chain

**Lesson:** §3.4 "Init hoisting is opt-in, narrowly scoped, never the default." §9 "Auto fast paths. … Make optimizations explicit, never silent." §4.2 "Don't ship fast paths that don't carry their weight."

**§3.4 precondition set:**
1. Chain shape is exactly `CopyTile + 1 SFPU op` (one CB load + one SFPU compute). Longer chains, multiple CopyTiles, or any FPU-clobbering element disqualify.
2. No element between iterations reprograms hardware state the hoisted `init()` set up.

**Current state** (`eltwise_chain.inl` `eltwise_chain()`):
```cpp
constexpr bool has_clash = chain_has_non_copy_tile_fpu_clash_v<Chain>;
constexpr bool emit_init_per_tile = has_clash;
…
if constexpr (!emit_init_per_tile) {
    detail::hoisted_init_for_each(IdxSeq{}, elts...);
}
```

Hoist fires on ANY chain whose elements are all either `CopyTile`, `DestOnly` SFPU, or `PackTile`. The defined-but-unused `chain_is_hoist_safe_v` trait (specialised only for size-2 and size-3 chains with the exact §3.4 shape) is the right gate — but the pipeline does not consult it.

**Concrete miscompile risk:** `CopyTile + Exp + Sqrt + PackTile` would hoist all inits per §3.4's worked example ("chain length > 2, not hoist-safe"). Per-element SFPU `init()`s in a longer chain that don't touch each other's MOP are *probably* safe in practice — but the lesson's hard rule is "single CB load + single SFPU compute". The pipeline's actual gate is laxer than the lesson's.

**Severity:** BLOCKER for new ops that wire multi-SFPU chains; latent for the currently-shipped single-SFPU chains.

**Proposed fix:** replace `emit_init_per_tile = has_clash` with `emit_init_per_tile = !chain_is_hoist_safe_v<Chain>`. Extend `chain_is_hoist_safe` specialisations to cover `CopyTile + DestOnly` (with or without trailing `PackTile`), but keep the strict precondition — multi-CopyTile, multi-FPU, or any opted-in `clashes_with_fpu` element disqualifies.

---

## BLOCKER — dual dispatch (static `call()` + member `exec(i)`)

**Lesson:** §1.5 "Classification is not dispatch. … Pick one dispatch shape … the tag classifies, the call signature is uniform." §4.5 "Pick one dispatch shape (a single member method that takes the loop index) for every element. … SFINAE detectors that pick between dispatch shapes are a band-aid for letting two contracts coexist; one contract is the fix." §9 anti-pattern "Two dispatch contracts."

**Current state** (`eltwise_chain.inl` `elem_compute_exec`):
```cpp
template <class T> struct has_member_exec : std::false_type {};
template <class T> struct has_member_exec<T, std::void_t<decltype(std::declval<const T>().exec(uint32_t{}))>>
    : std::true_type {};
…
if constexpr (has_member_exec_v<E>) {
    if constexpr (EmitInit) E::init();
    e.exec(i);                       // member dispatch (Hardtanh, Elu, Power, Threshold, UnaryEq, …)
} else if constexpr (is_dest_only_op_v<E>) {
    if constexpr (EmitInit) E::init();
    E::exec();                       // CRTP static dispatch (Exp, Sqrt, Relu, Sin, …)
}
```

Runtime-param SFPU op structs (`Hardtanh`, `Elu`, `Selu`, `Softplus`, `Prelu`, `LeakyRelu`, `Power`, `Rpow`, `Threshold`, `Clamp`, `Round`, `AddUnary`/`SubUnary`/`MulUnary`/`DivUnary`/`RsubUnary`/`RdivUnary`, `UnaryEq`/`Ne`/`Gt`/`Ge`/`Lt`/`Le`, `FillScalar`/`FillInt`/`FillBitcast`, `RandTile`) define an empty `static void call(uint32_t)` stub AND a member `exec(uint32_t)` carrying the runtime param. If a future op author forgets the member `exec`, the SFINAE probe silently routes to the empty static `call()` via the CRTP base — exact §1.7 stub-default silent-footgun pattern.

**Proposed fix:** every chain element (op struct or chain element) exposes one member `exec(uint32_t)`. The CRTP bases' `exec()` becomes a forwarder to `call(idst)` exposed as `exec(uint32_t)`; runtime-param ops override `exec` directly. The pipeline collapses to a single `e.exec(i)` dispatch. Removes `has_member_exec_v` SFINAE and the static-vs-member fork.

---

## SHOULD-FIX — Block elements are parallel types, not template parameters

**Lesson:** §3.9 "Block-mode is an axis on existing elements, not a parallel sibling kind. The test that distinguishes 'missing parameter' from 'missing kind': if the proposed sibling shares the lifecycle, dispatch hooks, and init contract with its non-block counterpart and only differs in iteration count, it's a parameter. If structurally different lifecycle … it's a kind." §9 anti-pattern "Parallel sibling kinds for an axis on an existing element."

**Current state:** `BlockCopyTile`, `BlockBinaryFpu`, `BlockPackTile` live in `eltwise_block.hpp` as separate types parallel to `CopyTile`, `BinaryFpu`, `PackTile`.

Comparing `BlockCopyTile` vs `CopyTile`:
- Lifecycle: same (per-tile wait/pop with policy enum; upfront variants).
- Reconfig timing: same (fold-driven prev-CB via `reconfig_srca_cb`).
- Init contract: same (`copy_tile_init(Cb)`).
- Difference: `wait_per_tile` waits `BlockSize` instead of `1`; `exec` runs `BlockSize` inner ops; `pop_per_tile` pops `BlockSize`.

That is purely an iteration-count axis. §3.9's test points at "missing parameter."

Same observation for `BlockBinaryFpu` vs `BinaryFpu`, `BlockPackTile` vs `PackTile`.

**Proposed fix:** collapse the Block variants into a `BlockSize` template parameter on the streaming elements:
```cpp
template <uint32_t Cb, Dst DstSlot = Dst::D0, CopyTilePolicy Policy = …,
          CbIndexMode IndexMode = CbIndexMode::FirstTile,
          CopyTileReconfig Reconfig = …, uint32_t BlockSize = 1>
struct CopyTile : CopyTileTag { … };
```
The block traits machinery (`a_wait_count`, `BaseDst + j`) generalises trivially. Block-only headers become a deprecation-stub forwarder during the transition.

Trade-off: every kernel that constructs `BlockCopyTile<…>` needs a name change. Mechanical.

---

## SHOULD-FIX — `EnableFp32DestAcc` is a `bool`, not an enum

**Lesson:** §2.5 "Policy enums are NEVER booleans. Don't ship `Reconfig=true/false` template params. … the bool form requires the reader to remember which way is on."

**Current state:** `BinaryFpu`, `BlockBinaryFpu`, `DestReuseBinary`, `UnaryBcast`, `PackTile`, `PackTileBlock`, `BlockPackTile` all carry `bool EnableFp32DestAccV = false`.

At a call site `BinaryFpu<…, true>{}` the `true` is unannotated. The reader has to count template arguments to know it's the fp32 flag — exact case §2.5 calls out.

**Proposed fix:** introduce
```cpp
enum class Fp32DestAcc : bool { Off = false, On = true };
```
in `eltwise_chain.hpp`'s named-enum section (alongside `Approx`, `Legacy`) and replace every `bool EnableFp32DestAccV` template parameter with `Fp32DestAcc EnableFp32DestAccV = Fp32DestAcc::Off`. The fold's SFINAE probe (`fp32_or_default`, `has_fp32_dest_acc`) reads `static constexpr Fp32DestAcc EnableFp32DestAcc` instead of `bool`; trivially adapted.

---

## SHOULD-FIX — Stub-default trait providers on the tag types

**Lesson:** §1.7 "Stub-default member functions are a silent footgun. Whenever a downstream check short-circuits on a stub return value (an id stub returning 0, a flag stub defaulting false), an element that forgot to override it passes the check by accident — no diagnostic. Force the contract at chain entry via a static_assert that names the missing override."

**Current state** (`eltwise_chain.hpp`):
```cpp
struct CbReaderTag { static constexpr uint32_t pack_cb_id() { return 0; } };
struct CbWriterTag { static constexpr uint32_t cb_a_id() { return 0; }
                     static constexpr uint32_t cb_b_id() { return 0; } };
struct DestOnlyTag { static constexpr bool is_upfront = false;
                     static constexpr bool clashes_with_fpu = false;
                     static constexpr uint32_t cb_a_id() { return 0; }
                     static constexpr uint32_t cb_b_id() { return 0; }
                     static constexpr uint32_t pack_cb_id() { return 0; } };
```

`first_cb_a()` / `first_pack_cb()` / `reader_pair_collide()` reason about `cb_a_id() != 0` — but CB id `0` (`tt::CBIndex::c_0`) IS a legitimate CB. The deducers already work around this with `has_any_*` predicates, but the tag-side stub still allows an opted-in CbReader op that forgot to override `cb_a_id()` to silently report `0` and be classified by trait sweeps as "first CB-reader of CB 0".

**Proposed fix:** drop the stub defaults. Make `cb_a_id()` / `cb_b_id()` / `pack_cb_id()` undefined at the tag level; require every CbReader/CbWriter to declare them. The pipeline then routes via SFINAE (`has_cb_a` / `has_cb_b` / `has_reconfig_pack`) — which it already does for the reconfig fold; same pattern.

This converts "forgot to declare cb id" from a silent classification bug into a compile-time error at the element site.

---

## SHOULD-FIX — Pipeline state is public, not private/mutable

**Lesson:** §4.1 "Pipeline state is private and reset by the pipeline. `CopyTile::cb_tile_idx_` is `mutable uint32_t` and `private`. Only `WaitUpfrontPopAtEnd` reads/writes it, and the pipeline zeroes it at end-of-block. Callers cannot override it."

**Current state:**
- `CopyTile::cb_tile_idx` — `uint32_t` public (default `0`).
- `BinaryFpu::a_tile_idx`, `BinaryFpu::b_tile_idx` — public.
- `DestReuseBinary::cb_tile_idx` — public.
- `PackTile::output_tile_idx` — public.

These exist for `Pinned` / `Absolute` index modes — the caller passes the value via the element ctor. Lesson treats this surface as a runtime tile index dedicated to the pipeline, not as a user-set value. The current shape makes it look like a caller-tunable knob.

**Proposed fix:** rename to `cb_tile_idx_` (trailing underscore convention), make `private`, expose `ALWI uint32_t tile_idx() const`. The ctor stays the public surface for setting it — `CopyTile<Cb, …>{idx}` — so the caller-facing API doesn't change. Lesson allows runtime payload via ctor (§1.2, e.g. `Power::exponent`); the rule is "don't expose pipeline-driven state as a field readers might write."

Trade-off: ctor → private field assignment is a one-line change per element. Mechanical.

---

## SHOULD-FIX — `block_path` emits wait/reserve before the per-tile loop

**Lesson:** §11 "Wait as late as possible. The latest valid wait position is the instruction just before the unpacker reads the tile. … Reserve as late as possible. The latest valid reserve position is right before `pack_tile`." §11 ("Per wait-shape positioning"): "All three wait shapes are emitted inside the per-tile loop. Upfront and cumulative shapes are NOT moved to a pre-loop block — `cb_wait_front` is cumulative-count idempotent."

**Current state** (`eltwise_chain.inl`):
```cpp
if constexpr (block_path) {
    (detail::elem_wait_upfront(elts, n_tiles), ...);     // ← before the loop
    (detail::elem_reserve_upfront(elts, n_tiles), ...);  // ← before the loop
    for (uint32_t i = 0; i < n_tiles; ++i) { … }
    (detail::elem_pop_upfront_end(elts, n_tiles), ...);
    (detail::elem_push_at_end(elts, n_tiles), ...);
}
```

Functionally correct (N-tile wait blocks once, no-op thereafter). But §11 calls this out by name as suboptimal: "Moving the wait inside the loop (with full count N on every iter) recovers that overlap for free — the wait short-circuits once the producer has caught up, and the producer wasn't blocked from pushing in the meantime."

The lesson amendment also calls it "the structural overlap fix that supersedes a separate `Cumulative` policy for most real cases."

**Proposed fix:** move `elem_wait_upfront` and `elem_reserve_upfront` inside the per-tile loop, at the latest-still-valid position (top of loop, before `tile_regs_acquire`). They are policy-guarded internally (no-op for non-upfront policies), so emitting them every iter is correct and idempotent. Pop / push stay at end-of-loop.

This also lets `block_path` and `per-tile path` share more emission code — the only structural difference becomes "use `wait_upfront(n)` or `wait_per_tile(i)` at the same loop position."

---

## NICE — `chain_is_hoist_safe_v` defined but unused

`chain_is_hoist_safe<EltwiseChain<A, B>>` (CopyTile + DestOnly) and `<A, B, C>` (CopyTile + DestOnly + PackTile) are specialised in `eltwise_chain.inl`, exposed as `chain_is_hoist_safe_v`. The actual pipeline gate is `has_clash` (§3.4 BLOCKER). After fixing the BLOCKER the trait becomes the gate, so this is closed in the same change.

---

## NICE — `chain_loads_share_cb_v` only specialised for 2-element chains

`chain_loads_share_cb<EltwiseChain<A, B>>` checks if both are `CopyTile` with same `cb`. Larger chains fall through to the `false_type` default. Per the trait's name and lesson §3.3, the intent is "any two CopyTile elements share a CB" across the chain — useful for `chain_is_hoist_safe` ("loads_share_cb" being one of its preconditions). Should generalise to a pack-fold any-pair check (the `reader_pair_collide` machinery in `chain_has_duplicate_upfront_cbs` is the existing template — extend the pair predicate to drop the `is_upfront` gate).

---

## NICE — Runtime ASSERT for `Pinned` / `Absolute` index bounds is missing

**Lesson:** §2.7 "For runtime-supplied indices (`Pinned k`, `Absolute idx`), the `WaitUpfrontPopAtEnd(N)` path runtime-`ASSERT`s `idx < N`; the single-tile-window policies runtime-`ASSERT` the index is `0`." §4.3 "Runtime asserts are the fallback for things the type system genuinely can't see."

**Current state:** Compile-time `static_assert` rejects `Policy=WaitAndPop + IndexMode=BlockIter|Absolute`, etc. But for `Pinned`/`Absolute` under a legal policy, no runtime `ASSERT` checks the supplied index against the waited window.

**Proposed fix:** add `ASSERT(cb_tile_idx < n_tiles)` or `ASSERT(cb_tile_idx == 0)` in the appropriate `exec()` paths. Gated by `--dev` build via the existing `ASSERT` macro.

---

## NICE — Missing `Mask` op struct (§1.4 worked example)

**Lesson:** §1.4 — `Mask` is the worked example for "mirror hardcoded LLK contracts in the type": mask is always read from `DataSlot + 1`, encode at compile time so the user can't pass a separate slot index.

**Current state:** No `Mask` op struct in any `eltwise_*.hpp`. Mask exists as a raw LLK call; no helper coverage.

**Proposed fix:** add `Mask<DataFormat DF, Dst DataSlot>` to `eltwise_misc.hpp` (or a new `eltwise_mask.hpp`) following the §1.4 worked example. Low-priority unless a real kernel needs it — this is a "missing op struct" (§10.2 says fix-and-continue, not a blocker).

---

## NICE — `BroadcastDim` doc: missing Reduce↔Broadcast table

**Lesson:** §5.3 "`BroadcastDim::{NONE,ROW,COL,SCALAR}` + companion table mapping `REDUCE_ROW → COL`, `REDUCE_COL → ROW`, etc. The 'reduce-row produces column-shaped output' surprise lives where it is needed."

**Current state:** `eltwise_chain.hpp` declares the enum but no companion mapping table. Existing doc-comment is one line.

**Proposed fix:** add a markdown table in the `BroadcastDim` doc-comment showing the reduce-input ↔ broadcast-output mapping. Pure documentation.

---

## DEFERRED — Cumulative wait policy

**Lesson:** §2.1, §11 — `CumulativeWaitPopAtEnd` is "the missing useful cell" but ships only when a real consumer demands it. §11 also notes: "The mechanical effect of moving `cb_wait_front(cb, N)` from boot-time into the per-iter loop … is functionally equivalent when N is constant" — i.e. the SHOULD-FIX "block_path wait-late amendment" supersedes the cumulative policy for most real cases.

Action: do not add until a real kernel needs growing-window indexing. Leave as known-missing.

---

## CONFIRMED OK (no gap)

- §1.1 CRTP bases (`UnaryOp` / `BinaryOp` / `TernaryOp` / `QuaternaryOp` with `DEST_AUTO_LIMIT` static_asserts).
- §1.2 Op struct templates carry compile-time state; runtime fields only for genuine runtime values.
- §1.3 Self-documenting enums (`Approx`, `Legacy`, `Dst`).
- §1.6 `if constexpr` for opt-in capabilities (`is_upfront`, `clashes_with_fpu`).
- §1.8 Fill / Rand tagged distinctly (`FillTileTag`, `RandTileTag` separate from `CopyTileTag`).
- §2.1 `CopyTilePolicy` enum covers exactly the five real wait/pop shapes.
- §2.2 No auto-merge of same-CB elements at the chain level.
- §2.3 Independent A/B policies (`APolicy`, `BPolicy`) on `BinaryFpu` and `BlockBinaryFpu`.
- §2.4 Reconfig naming parity (`*::Input` across all reconfig enums).
- §2.6 Defaults pick the safe option (`InputAndOutput`, `PerTileReserveAndPush`).
- §2.7 Per-side `AIndex` / `BIndex` preserved; same-CB asymmetric-index `static_assert`.
- §2.8 `UnaryBcast` is a chain element, not a separate helper.
- §3.1 One dispatch path (the chain); convenience wrappers are pure forwarders.
- §3.2 `clashes_with_fpu` named after cause, not consequence.
- §3.5 Fan-out as N explicit `CopyTile`s (no auto-expansion); element named `CopyTile`, not `Load`.
- §3.6 Same-CB dedup at the helper (`same_cb` in `BinaryFpu`/`BlockBinaryFpu`).
- §3.7 Held-DEST patterns out of scope; documented.
- §3.8 One eltwise helper; convenience wrappers are inline forwarders.
- §3.10 Reconfig attaches to the element owning the CB.
- §4.4 Pipeline emits BOTH per-tile and upfront lifecycle ops in `block_path` (Reg-C fix for mixed-policy chains).
- §5.1 One file per op category; aggregator header retained.
- §5.2 Worked examples in `eltwise_chain.hpp` doc-comment.
- §6 Names already evolved (`CopyTileReconfig::Input` not `::Srca`; `clashes_with_fpu` not `needs_parent_reinit`; `EltwiseBatching` deleted).
- §7.1 srcB reconfig path covered via `DestReuseType` selector.

---

## Suggested order of operations for Phase 2 proposal (Gate 1)

1. **BLOCKER fixes first**, in one proposal — closing both means the chain pipeline matches the lesson contract:
   - Hoist gate uses `chain_is_hoist_safe_v` (also generalises the trait).
   - Single member-dispatch contract (drops `has_member_exec_v`).
2. **SHOULD-FIX batch**, one proposal:
   - `BlockSize` template parameter on streaming elements (deprecates `eltwise_block.hpp`).
   - `Fp32DestAcc` enum replacing `bool`.
   - Drop stub-default tag fields; require overrides via SFINAE.
   - Pipeline state private + ctor-only.
   - `block_path` wait-late amendment.
3. **NICE batch**, one proposal:
   - `chain_loads_share_cb_v` generalisation.
   - Runtime index `ASSERT`s.
   - `Mask` op struct + Reduce↔Broadcast doc.

Each proposal is its own Gate-1 round-trip; commit-on-approval per HQ.

## Open questions for the user before drafting the Gate-1 proposal

1. **Backwards-compat surface.** Are existing kernels (binary_ng, eltwise_binary_*, softmax phase 2, …) free to be edited at the call site as part of the Block→`BlockSize` collapse, or do they need a deprecation-stub period? Mechanical rename either way; the question is whether one PR or two.
2. **`bool EnableFp32DestAcc` → `Fp32DestAcc` enum.** Same question — drop-in for every existing call site, or stub the old name for one cycle?
3. **Hoist gate tightening.** After fixing the BLOCKER, chains like `CopyTile + Exp + Sqrt + PackTile` would re-init per tile. Is that the right call, or do we want a relaxed `chain_is_hoist_safe` that allows multi-SFPU as long as each SFPU op's `init()` is provably MOP-disjoint? The lesson's strict rule is "single CB load + single SFPU compute"; lifting it means defining per-SFPU `clobbers_unpack_mop` / `reconfigs_srca` traits (§3.4 mentions this as the principled extension).
4. **Pipeline state private.** Acceptable to add a `ctor → private field` boilerplate to every element that uses `Pinned` / `Absolute`, or prefer leaving the field public and instead documenting "do not write after construction"?

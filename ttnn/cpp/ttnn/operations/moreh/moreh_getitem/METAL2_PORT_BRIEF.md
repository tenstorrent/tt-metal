# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/moreh/moreh_getitem`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `b72c35b810e 2026-08-04 docs(metal_2.0): require an explicit opt_level when porting compute kernels` *(carry this line into the port report's Provenance section)*

## Read this before you plan: it is three programs, not two

`MorehGetItemOperation` has two factories, but `MorehGetItemTilizedFactory` branches internally on `is_w_index_exist` (`moreh_getitem_tilized_factory.cpp:87`, `else` path begins `:354`) and emits **two structurally different programs** — different kernels, different CB sets, different runtime-arg lists. Plan for three:

| Shape | Condition | Reader | Writer | CBs |
|---|---|---|---|---|
| **RM** | `MorehGetItemRmFactory` | `moreh_getitem_kernels/reader_moreh_getitem.cpp` | `…/writer_moreh_getitem.cpp` | `c_0`, `c_1`–`c_5`*, ~~`c_16`~~ |
| **Tilized-W** | Tilized, `is_w_index_exist` | `moreh_getitem_tilized_kernels/reader_moreh_getitem_tilize_w.cpp` | `…/writer_moreh_getitem_tilize_w.cpp` | `c_0`, `c_1`–`c_5`*, ~~`c_16`~~, `c_17` |
| **Tilized-noW** | Tilized, else | `moreh_getitem_tilized_kernels/reader_moreh_getitem_tilize.cpp` | `…/writer_moreh_getitem_tilize.cpp` | `c_0`, `c_1`–`c_4`*, ~~`c_16`~~ |

\* only for index dimensions the caller actually supplied. Each shape further sub-configures on `ROW_MAJOR_INDEX` / `TILIZE_INDEX` (`moreh_getitem_tilized_factory.cpp:183-187, 433-437`), which selects between two `#ifdef` blocks inside the tilized readers.

There are **no compute kernels** — the op is pure data movement, reader → `c_0` → writer.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `ProgramSpecFactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` — `moreh_getitem_device_operation.hpp:34,41`
- **Op-owned tensors:** none
- **Target concept:** `ProgramSpecFactoryConcept` (plain), same for both factories
- **Gate-cleared, confirmed absent** (each would have blocked this brief): custom hash · `get_dynamic_runtime_args` · `override_runtime_arguments` · pybind `create_descriptor` · other migration-risky pybind. All `no`, confirmed both on the readiness sheet and independently in code.

## Do this first — one decision is not yours to make

**Question 1 in the audit blocks a design choice you will hit immediately.** The RM factory passes an explicit page size to all 7 of its `TensorAccessor`s, and the audit classified those sites **Class 1 (dynamic page size)**. The mechanical action would be "drop the override, set `dynamic_tensor_shape`" — but the recipe's safety check for that relaxation compares it against the op's custom hash, and **this op has no custom hash**, so the check cannot be run. The two live readings differ:

- **override is load-bearing** → you must set `dynamic_tensor_shape`, or you reintroduce the staleness the kernel comment describes;
- **override is redundant** → drop the arg and add **no** relaxation, because setting it would broaden cache reuse beyond legacy behavior — a semantic change the port is not allowed to make.

**Get the answer from the ops / relaxation-design owner before you write the `TensorParameter` declaration.** Do not infer it from the kernel comment alone.

## Construct — to do

### Tensor bindings

Up to **7 per program**, all **Case 1** — every access goes through a `TensorAccessor`, in every kernel:

- **`input`** — Case 1 → `TensorAccessor(tensor::input)`.
- **`output`** — Case 1 → `TensorAccessor(tensor::output)`.
- **`index0`–`index4`** — Case 1, **optional** (see the heads-up below).

Bases arrive today as **`Buffer*` entries** in the RTA lists, never as `->address()` — the framework's `BufferBinding` form, superseded by the typed binding. RTA sites: `moreh_getitem_rm_factory.cpp:187-250` · `moreh_getitem_tilized_factory.cpp:259-344` (W) · `:507-589` (noW). Each site is followed by the host-side `TensorAccessorArgs(...).append_to(...)` CTA plumbing (`rm_factory.cpp:144-148, 159-160`; `tilized_factory.cpp:189-193, 205-206, 439-443, 455-456`); both disappear together per binding.

**No Case 2 anywhere — do not reach for the `get_bank_base_address` bridge.** The readers and the Tilized-W writer do plenty of raw L1 pointer arithmetic (`reader_moreh_getitem.cpp:186-190`, `writer_moreh_getitem_tilize_w.cpp:86-99`), but always on **CB memory** obtained via `dfb.get_write_ptr()` / `get_read_ptr()` — never on tensor memory. Leave that arithmetic untouched.

**TensorAccessor 3rd arg:** drop at all **7 RM sites** — `reader_moreh_getitem.cpp:75, 79, 80, 81, 82, 83` and `writer_moreh_getitem.cpp:27` — paired with the Question 1 decision. **The tilized kernels pass no 3rd argument** (all 13 sites already 2-arg); nothing to do there.

### CB endpoints

**Dead-CB drop — `c_16`, all three shapes, confirmed:**

> `CBIndex::c_16` is allocated as the output CB in every shape — `moreh_getitem_rm_factory.cpp:129-138`, `moreh_getitem_tilized_factory.cpp:156-166`, `moreh_getitem_tilized_factory.cpp:418-427` — and **no kernel in this op references index 16 anywhere**. Every writer drains `c_0` (the input CB) instead: `writer_moreh_getitem.cpp:22`, `writer_moreh_getitem_tilize.cpp:33`, `writer_moreh_getitem_tilize_w.cpp:37`.
>
> Drop all three allocations. A dead CB has no behavior, so removing it changes none; and a bindingless DFB cannot be expressed in Metal 2.0, so this is not optional. Record each drop with `file:line` in the port report.
>
> Confirmation is requested as audit Question 2 — check it has been answered before you delete, but the evidence is unambiguous.

**Self-loop** (single toucher — bind that one kernel PRODUCER *and* CONSUMER):

| Shape | CB | Toucher |
|---|---|---|
| RM | `c_1`–`c_4` (defined dims) | reader — full FIFO cycle in one kernel (`reserve_back:161` … `push_back:184` … `wait_front:189` … `pop_front:190`) |
| Tilized-W | `c_1`–`c_5` (defined dims) | reader — locked producer: `reserve_back` + `get_write_ptr`, **no** `push_back` (`:176-201, 247-288`) |
| Tilized-W | `c_17` | writer — role-free: `get_read_ptr` (`:50`), raw stores (`:91,98`), NoC source (`:104`); no FIFO ops |
| Tilized-noW | `c_1`–`c_4` (defined dims) | reader — locked producer, same shape as Tilized-W |

**Legal 1P+1C, no action:** `c_0` in all three shapes (reader produces, writer consumes).

**Multi-binding flag:** none — no CB on any node has ≥3 touchers or two kernels locked to the same FIFO role. The op has no semaphores, so no hidden semaphore-gated co-fill is possible.

**Open — RM `c_5`:** allocated by `moreh_getitem_rm_factory.cpp:111-127` when a normalized index dim of 4 is defined, but `reader_moreh_getitem.cpp` has no `DataflowBuffer` for `c_5` and its loop runs `dim = 3 … 0` (`:146`) — zero endpoints. This is audit **Question 3**, and it is entangled with a possible pre-existing correctness bug in the RM guard. **Do not resolve it yourself and do not drop `c_5` on your own initiative.**

## Watch for

- **Optional / absent tensor bindings — the biggest design item in this port.** Both factories pass all five `index_info[N].buffer` slots, and undefined slots are `nullptr` (`rm_factory.cpp:192-196`; `tilized_factory.cpp:264-268, 512-516`). The framework handles that deliberately: `emplace_runtime_args_impl` (`tt_metal/impl/program/program_descriptors.cpp:239-247`) emits a literal `0u` **with no binding** for a nullptr `Buffer*`, specifically so optional inputs do not invalidate the fast cache-hit path. The kernels then construct all five `TensorAccessor`s unconditionally and use only the ones `index_is_defined[dim]` selects.

  Metal 2.0's typed binding channel needs an equivalent for "declared but absent in this instantiation." This is **not** an Appendix A feature, so it does not gate — but do not invent a shape for it under time pressure. Raise it with the framework side early; it is the most likely thing in this port to need a conversation.
- **Runtime-selected DFB handles.** All three readers choose which DFB to act on from a runtime dimension index, via `if (dim == N)` chains over five distinct objects: `reader_moreh_getitem.cpp:159-182`, `reader_moreh_getitem_tilize.cpp:162-185` and `:202-241`, `reader_moreh_getitem_tilize_w.cpp:176-201` and `:247-288`. `dfb::name` tokens are static, so expect to keep the `if`-chain and bind all five rather than index into anything.

  The RM reader is the hardest: it holds a **pointer** to the selected buffer — `DataflowBuffer* index_dfb_obj = nullptr;` (`:158`), assigned inside the `if`-chain and dereferenced at `:184, 189, 190`. There is no binding-token analogue for a pointer-to-DFB; the `push_back`/`wait_front`/`pop_front` at those three lines will need to move inside the per-`dim` branches.
- **`index_cbs[5]` is dead in all three readers — do not port it.** Declared at `reader_moreh_getitem.cpp:93-99`, `reader_moreh_getitem_tilize.cpp:100-106`, `reader_moreh_getitem_tilize_w.cpp:101-107`. In the two tilized readers it is never read at all; in the RM reader its only use is `tt::CBIndex idx_cb = index_cbs[dim];` (`:151`), and `idx_cb` is itself never used. Translating it would manufacture five bindings the kernel does not need. (Team-side cleanup is tracked in the audit's Misc anomalies; for you it is simply not a toucher.)
- **Cross-op / shared kernels:** **none** — and this is worth knowing, because it is unusual for this family. The op owns all 6 kernels plus `moreh_getitem_tilized_kernels/common.hpp`; no other op instantiates them; and they include nothing outside `api/*` (LLK/HAL) and their own directory. In particular this op does **not** depend on `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp`, unlike its siblings. No `_metal2` fork question, no sunset list, no donor coordination.
- **RTA varargs:** none — but there are a *lot* of named args (up to 38 in the Tilized-W reader). Every kernel reads a fixed run through a running `i++` counter at the top (`reader_moreh_getitem.cpp:11-57` and equivalents), which is the recipe's explicit non-signal, so **name every one of them**; do not let the volume tempt you into a vararg block. The host-side arg lists are commented by group (`// buffers`, `// input`, `// index`, `// output`, `// etc`) and line up positionally with the kernel reads — use them as the naming source.
- **`experimental/quasar/` holds no copy of this op** — checked. If you find a `*_metal2.cpp` that looks like a solved version of a problem here, it is not from this op; do not use it as a source.

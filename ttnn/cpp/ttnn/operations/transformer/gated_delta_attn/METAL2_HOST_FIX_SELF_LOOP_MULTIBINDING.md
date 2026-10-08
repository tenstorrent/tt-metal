# Metal 2.0 host fix: let a self-looped DFB take extra endpoints under `allow_instance_multi_binding`

**For:** Metal 2.0 host API owners (`tt_metal/impl/metal2_host_api/`).
**Raised by:** the `transformer/gated_delta_attn` Metal 2.0 port (2026-10-08). The port is blocked on this rule, or on the op-side alternative in `METAL2_OP_FIX_STATE_SEED.md`. **Either fix alone unblocks the port; both are not needed.**
**Status:** proposal only. The port does not touch `program_spec.cpp`.

## Is a host fix needed?

**Only if the op keeps its current shape.** `gated_delta_attn` has a state buffer `S` with two genuine producers on one node:

- The **reader** (DM) seeds `S` once (`reserve_back` / `push_back`): the initial state, or zeros.
- The **compute** kernel consumes `S` every chunk (`wait_front` / `pop_front`). On every non-last chunk it also pushes the updated state back into `S` (`reserve_back` / `push_back`).

The faithful Metal 2.0 binding is: reader PRODUCER; compute PRODUCER + CONSUMER; `allow_instance_multi_binding = true` on `S` (two kernels locked to PRODUCER). `ValidateProgramSpec` rejects it:

```
TT_FATAL: DFB 'S' is self-looped (some kernel appears as both producer and consumer), but the set of
producer KernelSpecs differs from the set of consumer KernelSpecs. When a DFB is self-looped, every
same-side binding must come from a self-loop participant (i.e. a kernel that appears on both sides).
  @ tt_metal/impl/metal2_host_api/program_spec.cpp:2040 (HEAD 9e90e641f58)
```

No legal binding satisfies the rule without misdescribing the kernels:

- Dropping compute's PRODUCER binding mislabels a real `push_back`er.
- Adding a CONSUMER binding on the reader (so the producer set equals the consumer set) declares an endpoint the reader never uses. That is the self-loop + multi-binding stacking the port recipe forbids.

If the op is rewritten per `METAL2_OP_FIX_STATE_SEED.md`, `S` becomes a plain compute self-loop and this host fix is not needed for this op. It may still matter for other ops with the same "seed, then recycle" state-loop shape.

## Why the rule looks inconsistent with the rest of the validator

All line numbers are `tt_metal/impl/metal2_host_api/program_spec.cpp` at `9e90e641f58`.

1. **The flag already waives every per-role constraint, for this exact reason** (`:1851-1857`):
   > allow_instance_multi_binding (Gen1-only; rejected on Gen2 earlier in this function) turns the per-node DFB into a plain shared circular buffer, which has no per-role hardware config to share — no processor mask, DFB scheduler, or credit machinery. Every per-role uniformity requirement below exists solely to guarantee such a shared config, so none apply under the flag.

   Under the flag, the access-pattern, `num_threads`, and kind (compute vs DM) uniformity checks are skipped, and the per-node census relaxes to "at least one". `CPU_DFBMixedKindProducersOnSameNodeSucceedsWithFlag` (`tests/tt_metal/tt_metal/api/metal2_host_api/unit_tests/invariant_tests/program_spec/dfb_endpoints.cpp:203`) locks in a compute producer plus a DM producer on one node.

2. **The lowering agrees** (`:3704-3713`):
   > On Gen1 the DFB lowers to a plain circular buffer where the mask is inert (it never reaches the device blob), so the single representative mask MakeDataflowBufferConfig takes from the first binding is harmless.

3. **The self-loop set-equality rule is the one check that does not consult the flag** (`:2026-2045`). Its stated rationale:
   > rejecting the case where a self-looping kernel shares the DFB with an unrelated kernel (which would make the producer/consumer mask and lowering semantics ambiguous)

   That is the same per-role mask concern the flag already waives in (1) and (2).

   The DM-self-loop-on-Gen2 check just above it (`:2017-2024`) is unaffected. `S`'s self-loop participant is a compute kernel, and the flag is Gen2-illegal anyway (`:1818-1830`).

## Proposed change

The rule is kept as-is without the flag and skipped with it. The `allow_multi` local is in scope from `:1857`.

```cpp
// program_spec.cpp, ValidateProgramSpec, inside `if (self_loop_kernel != nullptr) { ... }`
            // Self-loop interplay with multi-binding: the producer set must equal the consumer set
            // as sets of KernelSpec*. ...
-           std::unordered_set<const KernelSpec*> producer_kernels;
-           ...
-           TT_FATAL(
-               producer_kernels == consumer_kernels,
-               "DFB '{}' is self-looped ...",
-               dfb.unique_id);
+           // Under allow_instance_multi_binding (Gen1-only) the DFB lowers to a plain shared circular
+           // buffer with no per-role mask, so a self-looping kernel may share it with other producer
+           // or consumer kernels (e.g. a DM kernel seeds a state buffer that compute then recycles).
+           if (!allow_multi) {
+               std::unordered_set<const KernelSpec*> producer_kernels;
+               ...
+               TT_FATAL(producer_kernels == consumer_kernels, "DFB '{}' is self-looped ...", dfb.unique_id);
+           }
```

Also update the summary comment at `:1846-1847` ("Self-loop … is currently restricted to the simple single-producer-single-consumer case") to name the multi-binding exception. The `allow_instance_multi_binding` doc comment in `tt_metal/api/tt-metalium/experimental/metal2_host_api/advanced_options.hpp:215-244` could also mention that the flag permits a self-loop alongside other endpoints.

**Owner decision needed:** the `:1846` comment says the single-producer-single-consumer self-loop is the *current* scope. That may be a deliberate scope limit rather than an oversight. If it is deliberate, say so, and the op takes the op-side fix instead.

## Tests to add or keep

In `tests/tt_metal/tt_metal/api/metal2_host_api/unit_tests/invariant_tests/program_spec/dfb_endpoints.cpp`:

- **New, Gen1:** `CPU_DFBSelfLoopPlusExtraProducerSucceedsWithFlag`. A compute kernel binds `dfb` PRODUCER + CONSUMER, a DM kernel (RISCV_1) binds it PRODUCER, all on one node, with `allow_instance_multi_binding = true` → `EXPECT_NO_THROW(MakeProgramFromSpec(...))`. This is the `gated_delta_attn` shape.
- **New, Gen1:** the same spec without the flag → still throws the "is self-looped … differs" message.
- **Existing:** `CPU_DFBSelfLoopWithExtraProducerSideKernelFails` (`:302`, Quasar, no flag) must keep failing unchanged. `CPU_DFBMultiBindingSelfLoopWithMatchingSidesSucceeds` (`:269`) is unaffected.

## Evidence the relaxed shape runs correctly on Gen1

On 2026-10-08, on the Metal_Ports 8×8 Wormhole b0, with the Metal 2.0 legality checks force-enabled (both `METAL2_CHECKS_FORCED` markers logged, 5× each), I ran the full `gated_delta_attn` port. It was satisfied by the reader-CONSUMER binding described above; that local experiment was never committed. On Gen1 the extra label is inert, so this is the binding set the proposed rule would accept, minus one cosmetic label.

The port was **bit-exact against the pre-port legacy op on 10/10 runs**:

- 5 configs: (BH, NC, initial_state) = (12,1,no), (12,3,yes), (5,2,no), (64,2,yes; full grid), (7,4,yes).
- Each config ran twice; the second run was a program-cache hit with fresh tensors at shifted addresses.
- Golden PCC was ≥ 0.9999 throughout.

Watcher could not be used: the legacy op already exceeds the kernel-config buffer under Watcher (`Program size (72320) too large … (70656)`), and that failure predates the port.

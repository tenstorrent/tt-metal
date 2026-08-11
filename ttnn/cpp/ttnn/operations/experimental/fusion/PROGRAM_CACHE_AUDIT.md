# Program Cache Audit — `experimental/fusion`

Audit of `FusionDispatchOpDeviceOperation::compute_program_hash` against the framework default
("hash everything") key. The Python front end that produces the descriptors audited here lives in
`models/experimental/ops/descriptors/fusion/`.

| | |
|---|---|
| Device operation | `ttnn::operations::experimental::fusion::FusionDispatchOpDeviceOperation` (`device/fusion_dispatch_op_device_operation.hpp`) |
| Custom hash | `device/fusion_dispatch_op_device_operation.cpp:35` |
| `operation_attributes_t` | `tt::tt_metal::experimental::MeshProgramDescriptor` — `mesh_programs`, a vector of (`MeshCoordinateRange`, `ProgramDescriptor`) |
| `tensor_args_t` | `fusion_dispatch_tensor_args_t` — `io_tensors` (all inputs + outputs), `output_tensor` |
| Program factory | none; `create_descriptor` returns the caller's descriptor near-verbatim |
| `override_runtime_arguments` | **No** |
| `get_dynamic_runtime_args` | **No** |
| Cache-hit patch mechanism | Framework **slow-path rebuild**, plus a Python-owned pre-patch |

## Cache-hit patch mechanism

This op deliberately steers itself into the framework's slow path. `create_descriptor` strips the
inherited buffer bindings that the fused sub-op descriptors carried:

```67:71:ttnn/cpp/ttnn/operations/experimental/fusion/device/fusion_dispatch_op_device_operation.cpp
    for (auto& kd : desc.kernels) {
        kd.buffer_bindings.clear();
        kd.common_buffer_bindings.clear();
    }
    return desc;
}
```

With no declared runtime-arg bindings and no `get_dynamic_runtime_args`, the adapter's fast-path
predicate fails and it falls through to a full re-derivation:

```748:753:ttnn/api/ttnn/mesh_device_operation_adapter.hpp
                    } else {
                        const ttnn::MeshCoordinate mesh_coord = coordinate_range.start_coord();
                        const std::optional<ttnn::MeshCoordinate> mesh_dispatch_coordinate(mesh_coord);
                        auto desc = invoke_per_coord(attrs, tensor_args, tensor_return_value, mesh_dispatch_coordinate);
                        tt::tt_metal::apply_descriptor_runtime_args(program, desc);
                    }
```

`apply_descriptor_runtime_args` copies **every** per-core runtime arg, every common runtime arg, the
Blaze named args, and every CB backing address out of the descriptor into the cached `Program`
(`tt_metal/impl/program/program_descriptors.cpp:184-229`). Since `create_descriptor` hands back the
descriptor the caller just patched, the cached program is fully re-armed on every hit.

**Consequence for this audit:** runtime-arg *values* legitimately do not need to be hashed — they
are re-applied wholesale. What must still be hashed is everything baked into the cached `Program`
at construction and never revisited: kernel sources, compile-time args, defines, kernel configs,
core ranges, CB counts/sizes/formats, and semaphores. That is precisely the split
`compute_program_descriptor_hash` implements.

## Which validator runs on a cache hit

The dispatcher runs exactly *one* validator on a hit, and which one it picks is the reverse of the
intuitive reading:

```262:266:ttnn/api/ttnn/device_operation.hpp
    if constexpr (HasValidateOnProgramCacheHit<mesh_device_operation_t>) {
        mesh_device_operation_t::validate_on_program_cache_hit(operation_attributes, tensor_args);
    } else {
        mesh_device_operation_t::validate_on_program_cache_miss(operation_attributes, tensor_args);
    }
```

Fusion is in the less common branch: it **defines** `validate_on_program_cache_hit`
(`device/fusion_dispatch_op_device_operation.hpp:25`), so the hit validator *replaces* the miss
validator and the miss validator does not run on hits at all. Normally that is a hazard — a narrow hit
validator silently disables every check in the miss validator on the fast path. Here it costs nothing,
because both are empty:

```19:23:ttnn/cpp/ttnn/operations/experimental/fusion/device/fusion_dispatch_op_device_operation.cpp
void FusionDispatchOpDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t&, const tensor_args_t&) {}

void FusionDispatchOpDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t&, const tensor_args_t&) {}
```

Diffed against each other, the hit path pins exactly what the miss path pins: nothing. The practical
consequences run through the rest of this document.

- **No verdict here can be "VALID — pinned by validation".** There is no `TT_FATAL` at the device-operation
  level to cite. The only hard checks anywhere on the path are structural preconditions in the entry
  points — `io_tensors.size() >= 2` (`device/fusion_dispatch_op_device_operation.cpp:81-84`),
  non-empty `mesh_programs` (`:58`), non-empty tensors and a non-null device
  (`fusion_dispatch_op_nanobind.cpp:29-31, 48-50`), and matching `TensorSpec`s for shared outputs
  (`:90-95`). None of them constrains the aliasing, addressing or descriptor-provenance properties the
  omissions below turn on.
- **Nothing about `tensor_args` is enforced on either path.** When applying the reachability test to the
  omissions below, the answer to "is there an enforced constraint blocking this configuration?" is
  simply no, on both the miss and the hit path. That is what drives omission #5 to a BUG rather than a
  caveat.
- **The empty hit validator is a trap for future work.** Any check added to
  `validate_on_program_cache_miss` alone would not execute on hits, so it could not support a
  "pinned by validation" verdict for the very call that inherits a cached program. Checks intended to
  hold on hits must go into `validate_on_program_cache_hit`, or into a shared helper both call.

## Baseline: what the default hash would cover

The default would be `hash_objects_with_default_seed(type_hash<FusionDispatchOpDeviceOperation>,
attrs, tensor_args)`, i.e.:

| Source | Fields |
|---|---|
| `attrs.mesh_programs` | each `MeshCoordinateRange` plus each `ProgramDescriptor` via `std::hash<ProgramDescriptor>` |
| `tensor_args.io_tensors` | for every input and output tensor: storage kind, `logical_shape`, `dtype`, `page_config`, `memory_config`, `alignment` |
| `tensor_args.output_tensor` | the same six properties again (it is `io_tensors.back()`) |

## What the custom hash covers

```35:45:ttnn/cpp/ttnn/operations/experimental/fusion/device/fusion_dispatch_op_device_operation.cpp
ttsl::hash::hash_t FusionDispatchOpDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t&) {
    // Must differ from GenericOpDeviceOperation::compute_program_hash — same descriptor would
    // otherwise hit the wrong cached_mesh_workload_t layout (segfault in override).
    size_t hash = ttsl::hash::type_hash<FusionDispatchOpDeviceOperation>;
    for (const auto& [mesh_coord_range, program_descriptor] : operation_attributes.mesh_programs) {
        ttsl::hash::hash_combine(hash, mesh_coord_range);
        ttsl::hash::hash_combine(hash, ttnn::operations::generic::compute_program_descriptor_hash(program_descriptor));
    }
    return hash;
}
```

The `tensor_args_t` parameter is unnamed — **no tensor contributes to the key at all**. The whole
key is the op type plus the structural content of the descriptors.

## Omitted parameters

### 1. All of `tensor_args` — every input and output tensor spec

**Verdict: VALID — invariant.** This op has no program factory in the usual sense: it does not
*derive* anything from tensor properties. The descriptor arrives fully formed from Python, so
tensor shape/dtype/layout can only influence the program through values already baked into that
descriptor — compile-time args, CB page sizes, core ranges — all of which
`compute_program_descriptor_hash` covers directly. Hashing the specs as well would be strictly
redundant: two calls with different input shapes necessarily produce different descriptors (the
fused sub-ops' compile-time args and CB sizes differ), so they already miss.

This is also the relaxation the op is built around: an inference loop re-invokes the same fused
graph with fresh output tensors allocated at fresh addresses every step. Under the default hash
those would still hit, because addresses are not in a tensor's hash — but the *descriptor*
identity check is much cheaper than walking every tensor spec, which matters for an op whose whole
purpose is a low-overhead steady-state dispatch.

The obligation this creates is that the tensor addresses must be re-armed, which is exactly what
the slow path plus the Python pre-patch do (see omission 2).

### 2. Buffer addresses in runtime args and CB backing buffers

**Verdict: VALID — patched**, by two independent mechanisms in series.

First, `hash_kernel` hashes only the *count* of runtime args, never their values, and reduces a CB's
backing buffer to a nullness bit:

```61:68:ttnn/cpp/ttnn/operations/generic/device/generic_op_device_operation.cpp
            kernel.common_runtime_args.size(),
            // Blaze-only experimental named args (issue #50953): hash the FULL named-RT-arg schema
            // (names/lengths/dispatch across all 4 variants), NOT values. Replaces the previous
            // partial hashing that used .size() of only 3 of 4 variants and never the names.
            tt::tt_metal::experimental::blaze::hash_named_args_schema(kernel.blaze_named_args),
            kernel.runtime_args.size(),
            kernel.config.index(),
            kernel.config);
```

```92:93:ttnn/cpp/ttnn/operations/generic/device/generic_op_device_operation.cpp
        ttsl::hash::hash_combine(hash, cb.buffer != nullptr);
        ttsl::hash::hash_combine(hash, cb.global_circular_buffer != nullptr);
```

Second, before the primitive is ever invoked, Python refreshes every address slot in the
descriptor:

```178:181:ttnn/cpp/ttnn/operations/experimental/fusion/device/fusion_dispatch_op_helpers.hpp
    for (const auto& slot : slots.cb_slots) {
        desc.cbs[slot.cb_idx].buffer = io_tensors[slot.io_tensor_index].buffer();
    }
```

and the framework's slow path then copies those refreshed values into the cached program. Barrier
semaphore L1 addresses ride the same mechanism via `patch_semaphore_addresses`, which is what lets
the fusion cache hold semaphores as allocation *specs* rather than live objects.

This verdict covers *whether* each address slot is refreshed, which it is. *Which* tensor's address
each slot receives is a separate question, decided by the frozen slot map, and that is where omission
5 finds a bug.

### 3. Runtime-arg values that are **not** addresses

**Verdict: VALID — patched.** Same argument as omission 2 for the hash side, and the slow path
re-applies *all* runtime args (not just address slots), so a fused program whose scalar args shift
between calls is still correct. Note this is a stronger position than a fast-path op enjoys: an op
in the buffer-binding fast path freezes its non-address scalars, and this op does not.

### 4. `ProgramDescriptor::custom_program_hash` — a full caller escape hatch

**Verdict: CAVEAT.** The shared descriptor hasher short-circuits entirely if the caller pre-set a
hash:

```48:51:ttnn/cpp/ttnn/operations/generic/device/generic_op_device_operation.cpp
ttsl::hash::hash_t compute_program_descriptor_hash(const tt::tt_metal::ProgramDescriptor& program_descriptor) {
    if (program_descriptor.custom_program_hash) {
        return *program_descriptor.custom_program_hash;
    }
```

and the fusion front end always takes that branch, snapshotting the structural hash once at build
time:

```284:285:models/experimental/ops/descriptors/fusion/fusion.py
    if desc.custom_program_hash is None:
        desc.custom_program_hash = ttnn.compute_program_descriptor_hash(desc)
```

That is sound *as used*: the snapshot is taken from the fully-built descriptor, and everything
mutated afterwards (`patch_stale_descriptor`, `patch_semaphore_addresses`) touches only addresses,
which the structural hash excludes anyway (`generic_op_device_operation.cpp:61-68` hashes runtime-arg
*counts*, and `:92-93` reduces a CB's backing buffer to a nullness bit).

Nothing enforces the invariant — there is no `TT_FATAL` relating `custom_program_hash` to the
descriptor it labels, and both of this op's validators are empty, so the reachability test has no
enforced constraint to point at. What keeps this a caveat rather than a bug is the other limb of the
test: on this op's own path the value cannot take a second value for a given structure. It is not
caller-chosen but computed by `ttnn.compute_program_descriptor_hash(desc)` from the fully-built
descriptor, so it *is* the structural hash, and the only mutations applied after the assignment are
address writes that the structural hash excludes by construction. There is no reachable configuration
today in which the frozen value and the descriptor it stands for disagree.

The residual exposure is real but sits outside that argument, and it is worth naming precisely:

- A **future in-place edit** that changes a compile-time arg, a CB size, or a core range after
  `custom_program_hash` is assigned would be invisible to the cache and would produce a wrong hit. A
  code change is not a reachable configuration of today's API, so it does not move the verdict, but it
  is exactly the kind of change nothing would catch.
- An **out-of-tree caller** that bypasses `fusion.py` and builds its own descriptor can set the field
  by hand: it is exposed `def_rw` on `ttnn.ProgramDescriptor`
  (`ttnn-nanobind/program_descriptors.cpp:1111-1114`), and `fusion_dispatch_op` takes a caller-supplied
  `ProgramDescriptor` (`fusion_dispatch_op_nanobind.cpp:271-280`). Supplying a hash that does not
  identify the descriptor is not a configuration this op mishandles, though — it is the caller taking
  over the cache key. `generic`, which exposes the same escape hatch, grades it the same way.

The framework does protect the one case it can: `merge_program_descriptors` invalidates the field
(`tt_metal/impl/program/program_descriptors.cpp:126-127`). Worth an assertion or a debug-mode
recompute-and-compare for the rest.

### 5. Aliasing pattern among `io_tensors`

**Verdict: BUG.** `AddressSlots` is built by *value matching*: an arg is classified as an
address slot because its value equals some live IO tensor's address, and the first matching tensor
wins.

```127:137:ttnn/cpp/ttnn/operations/experimental/fusion/device/fusion_dispatch_op_helpers.hpp
        for (const auto& [coord, args] : kd.runtime_args) {
            for (size_t ai = 0; ai < args.size(); ++ai) {
                if (auto ti = find_io_tensor_index(args[ai], tensor_addrs)) {
                    slots.per_core_rt_arg_slots.push_back(
                        {static_cast<uint32_t>(ki), coord, static_cast<uint32_t>(ai), *ti});
                } else if (auto si = find_semaphore_index(args[ai], sem_addrs)) {
                    slots.sem_rt_arg_slots.push_back(
                        {static_cast<uint32_t>(ki), coord, static_cast<uint32_t>(ai), *si});
                }
            }
        }
```

The map is computed once, at build time, and then reused for the lifetime of the cache entry — the
fusion build cache stores it (`fusion.py:341`) and `FusionDispatchState` holds it as
`address_slots_` across every dispatch (`fusion_dispatch_op_nanobind.cpp:113, 146`). After build it is
index-based: `patch_stale_descriptor` writes `io_tensors[slot.io_tensor_index]`'s current address into
the recorded position, so a slot bound to the wrong index at build time stays bound to the wrong index
forever.

Two consequences. Both are ordinary two-call cache defects: the offending value is not in the device
hash (no tensor reaches it at all), it is not in the Python build cache key either — that key is
`(container kind, tree shape, per-branch program cache key / descriptor hash, mesh identity)`
(`fusion.py:11-15, 539-547`) — and it is not correctly re-derived on a hit, because the map is frozen.

- **Aliasing sensitivity.** When two IO tensors share an address at build time, `find_io_tensor_index`
  returns the *first* match, so every slot for that address binds to the lower index regardless of
  which tensor the arg semantically belongs to. Reproduction: build a fused container in which a
  sub-op writes in place, so the same buffer appears in `io_tensors` once as an input (index *i*) and
  once as an output (index *j > i*); every output-address slot binds to *i*. On the next dispatch the
  hot path allocates a *fresh* output tensor at a different address (`allocate_outputs`,
  `fusion_dispatch_op_nanobind.cpp:67-101`, called from `FusionDispatchState::dispatch` at
  `fusion_dispatch_op_nanobind.cpp:135`), but the frozen map still writes the *input's* address into
  those runtime-arg slots. The writer kernel then writes over the input buffer, and the output tensor
  handed back to the caller is never written. Silent wrong data on call 2 with no cache miss to hint at
  the cause. The same shape of failure occurs in reverse if two tensors are distinct at build time and
  alias later.
- **False positives.** A genuine scalar runtime arg whose value happens to equal an IO tensor's address
  is misclassified as an address slot. On call 1 the overwrite is a no-op, since the value already
  equals the address; on call 2 the same arg is overwritten with the *new* address, so a tile count, a
  loop bound or a stride silently becomes a large L1/DRAM address. The probability is low but not
  negligible for small L1 addresses versus small tile counts, and severity is high because there is no
  bounds check downstream.

Nothing enforces either precondition. `compute_address_slots` takes the first match without checking
for a second (`fusion_dispatch_op_helpers.hpp:34-41, 129`), `patch_stale_descriptor` silently skips a
slot whose tensor has no device buffer (`if (buf)`, `:161-162`) rather than failing loudly, and both
validators are empty no-ops, so no `TT_FATAL` on either path constrains the aliasing pattern or the
scalar/address disjointness. Aliased IO is not even an exotic configuration for this op — the shared
output path deliberately produces two `io_tensors` entries backed by one tensor
(`fusion_dispatch_op_nanobind.cpp:86-96`).

The reachability is broader than the Python front end, too: `fusion_dispatch_op` is a public binding
that takes the `io_tensors` and the `AddressSlots` as *independent* arguments
(`fusion_dispatch_op_nanobind.cpp:271-280`), with nothing checking that the map was computed from a set
of tensors aliased the same way as the set being dispatched.

Being an experimental op, and the in-tree fusion front end not exercising the in-place case today, are
severity context — they keep the defect latent — not grounds for downgrading it.

### 6. `type_hash` divergence from `GenericOpDeviceOperation`

Not an omission — the opposite — but it belongs in this audit. `GenericOpDeviceOperation::compute_program_hash`
starts from `size_t hash = 0` and never mixes in its own type hash, whereas this op explicitly does,
with a comment attributing a segfault to the difference. Given that `ProgramCacheKey::canonical`
already carries the op type name as a prefix, the two ops should not be able to alias on the cache
key today; the explicit `type_hash` here is defence in depth. Keep it.

## Framework side effect of having a custom hash

Because this op defines `compute_program_hash`, its canonical key degrades to the op-identity
prefix only:

```1012:1014:ttnn/api/ttnn/mesh_device_operation_adapter.hpp
        if constexpr (requires { DeviceOperation::compute_program_hash(attrs, tensor_args); }) {
            return key;  // custom hash -> opt out beyond the op-identity prefix
        } else {
```

For this op the exposure is unusually broad: the entire key is one 64-bit reduction of a whole
fused program's structure, so a collision between two genuinely different fused graphs resolves to
a wrong hit rather than a rebuild. The Python-side `_BUILD_CACHE` uses a collision-free tuple key
and so does not have this problem, but it sits in front of the device cache rather than protecting
it.

## Summary

| Omitted vs. default | Used by program? | Patched on hit? | Verdict |
|---|---|---|---|
| All input/output tensor specs | No (descriptor is pre-built) | Addresses yes, specs n/a | VALID — invariant |
| Buffer addresses (rt args, CB buffers) | Yes | Yes (Python `AddressSlots` + slow path) | VALID — patched |
| Non-address runtime-arg values | Yes | Yes (slow-path `apply_descriptor_runtime_args`) | VALID — patched |
| Descriptor structure behind `custom_program_hash` | Yes | No | CAVEAT — unenforced, but derived from the structure |
| `io_tensors` aliasing pattern | Yes (slot map) | No — map frozen at build | **BUG** — value-matched slots |

**One program-cache bug found**: the value-matched address slots (omission 5). A slot bound at build
time to the first IO tensor whose address matched keeps that binding for the life of the cache entry,
so a later dispatch whose tensors alias differently — the natural case being an in-place sub-op whose
build-time output shares a buffer with an input, against a hot path that allocates a fresh output —
writes the wrong tensor's address into the cached program. Nothing hashes the aliasing pattern and no
`TT_FATAL` on either validator path rejects it.

The hash's *field selection* is not where the problem is, and that is worth stating plainly so the fix
is aimed correctly. The op's core design is coherent: it hashes exactly the structural content of the
program and routes itself to the slow path so that every value it declined to hash is unconditionally
re-applied. Both remaining findings are about the descriptor's *provenance* rather than the key —
the frozen `custom_program_hash`, which stays a caveat because it is computed from the very structure
it labels, and the value-matched address slots, which do not.

## Recommendations

1. **Fix the bug.** Make `compute_address_slots` reject ambiguity instead of silently taking the first
   match: if a scanned value matches more than one distinct IO tensor address, or matches both a tensor
   address and a semaphore address, fail loudly. This is the minimal fix for omission 5 and it is free
   on the hot path — `compute_address_slots` runs once at build time, not per dispatch, so the check
   costs nothing in steady state. Note that it converts the defect into a loud build-time failure
   rather than making the aliased case work; that is the right first step, since today the same
   configuration corrupts data silently.
2. **The structural fix for the same bug**, and the one to aim at: narrow address-slot discovery from
   value matching to a declared mapping emitted by the codegen that stitched the fused descriptor. The
   codegen knows which arg is which tensor's address; recovering that by scanning for numeric equality
   discards information it already had. A declared mapping removes both the aliasing mis-binding and
   the scalar false-positive class at once, and it makes recommendation 1 unnecessary.
3. Do **not** try to close omission 5 by adding a check to `validate_on_program_cache_miss`. This op
   defines `validate_on_program_cache_hit`, so the miss validator does not run on hits, and the
   offending call is precisely the one that hits. If a dispatch-time guard is wanted anyway — for
   example, asserting that no two `io_tensors` entries share an address — it must go in
   `validate_on_program_cache_hit`, and it is then paid on every dispatch: an O(n²) address comparison
   over `io_tensors` on the fast path of an op whose entire purpose is low-overhead steady-state
   dispatch. Prefer the build-time checks above.
4. Have `patch_stale_descriptor` `TT_FATAL` on a slot whose tensor has no device buffer rather than
   skipping it, so a host tensor in `io_tensors` cannot leave a stale address behind. This one does run
   per dispatch, but it is a null check on a pointer already being loaded.
5. Under a debug flag, recompute `compute_program_descriptor_hash` at dispatch and assert it equals
   the frozen `custom_program_hash`, closing caveat #4 cheaply. `-DTT_DESCRIPTOR_PATCHING_PARITY_CHECK`
   is the natural home, since it is already the CI-only oracle for cache-hit re-application.

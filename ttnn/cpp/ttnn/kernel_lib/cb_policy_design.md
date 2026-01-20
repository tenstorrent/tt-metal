# CB Policy Design for Kernel Helpers

> **Status:** Design Proposal
> **Scope:** `binary_op_helpers.hpp`, `reduce_helpers.hpp`, future dataflow helpers
> **Goal:** Replace implicit CB behavior with explicit, composable policies

---

## 1. Problem Statement

### 1.1 Current State

The current `BinaryInputMode` enum bundles multiple orthogonal concerns:

```cpp
enum class BinaryInputMode {
    STREAMING,         // Wait/pop 1 at a time
    STREAMING_BATCHED, // Wait/pop chunks
    PRELOADED,         // Caller manages wait/pop
    PERSISTENT         // Wait upfront, never pop
};
```

This creates several problems:

1. **Implicit behavior:** The pop behavior for B-input varies by broadcast dimension (ROW/SCALAR persist, COL/NONE don't), but this is hidden in the implementation.

2. **16-combination matrix:** 4 input modes × 4 broadcast dimensions = 16 distinct behaviors that must be memorized.

3. **No shared vocabulary:** Reader and writer kernels must "know" what the compute kernel expects. There's no type-safe contract.

4. **Hard to extend:** Adding new behaviors requires modifying the enum and all switch statements.

### 1.2 Current Implicit Rules (from implementation)

| Broadcast | B Wait | B Pop | Hidden Rule |
|-----------|--------|-------|-------------|
| NONE | Same as A | Same as A | Follows A pattern |
| ROW | Wt upfront | **Never** | Persists for row reuse |
| COL | 1 per row | 1 per row | Consumed per row |
| SCALAR | 1 upfront | **Never** | Persists for all tiles |

These rules are **not documented in the API** — they're buried in the implementation.

### 1.3 Note on Current Inconsistency

The inconsistent B-input behavior across broadcast dimensions (ROW/SCALAR persist, COL/NONE pop) was **not an intentional design decision** — it simply evolved that way during implementation. There's no fundamental reason why:
- ROW broadcast B should persist but COL broadcast B should not
- The behavior should be implicitly tied to broadcast dimension at all

The policy-based design gives us an opportunity to:
1. Make all behavior explicit (no hidden rules)
2. Let the user decide persistence behavior independent of broadcast dimension
3. Provide sensible defaults that can be overridden

**The current behavior should be documented as "what exists today" — not as a specification to preserve.**

---

## 2. Proposed Solution: Explicit Policies

### 2.1 Core Insight

> **Policies create a shared vocabulary between producer and consumer kernels.**

When a reader kernel and compute kernel both use the same policy type, they're speaking the same language about CB behavior. No implicit contracts, no hidden behaviors.

### 2.2 Inspiration

The `ttnn/cpp/ttnn/operations/normalization/kernel_util/compute/policies.h` file demonstrates this pattern:

```cpp
struct PartialBlockWithPopPolicy {
    static constexpr bool pop = true;
    static constexpr bool sync_full_block = false;
};

struct FullBlockWithoutPopPolicy {
    static constexpr bool pop = false;
    static constexpr bool sync_full_block = true;
};
```

Benefits of this approach:
- Zero runtime overhead (all `constexpr`)
- Self-documenting through naming
- Can be used in `if constexpr` for compile-time dispatch
- Extensible (add more fields as needed)

---

## 3. Design Proposal

### 3.1 Orthogonal Policy Components

Split the current bundled behaviors into orthogonal, composable pieces:

#### Wait Policies (when/how to call cb_wait_front)

```cpp
namespace compute_kernel_lib::policies {

/// Wait for 1 tile at a time, inside the processing loop
struct WaitPerTile {
    static constexpr bool per_tile = true;
    static constexpr bool per_chunk = false;
    static constexpr bool upfront = false;
    static constexpr bool caller_managed = false;
};

/// Wait for DEST_LIMIT tiles at a time (chunked processing)
struct WaitPerChunk {
    static constexpr bool per_tile = false;
    static constexpr bool per_chunk = true;
    static constexpr bool upfront = false;
    static constexpr bool caller_managed = false;
};

/// Wait for all tiles once at start of operation
struct WaitUpfront {
    static constexpr bool per_tile = false;
    static constexpr bool per_chunk = false;
    static constexpr bool upfront = true;
    static constexpr bool caller_managed = false;
};

/// Wait for 1 tile per row (for COL broadcast B-input)
struct WaitPerRow {
    static constexpr bool per_tile = false;
    static constexpr bool per_chunk = false;
    static constexpr bool per_row = true;
    static constexpr bool upfront = false;
    static constexpr bool caller_managed = false;
};

/// Caller is responsible for cb_wait_front before calling
struct WaitCallerManaged {
    static constexpr bool per_tile = false;
    static constexpr bool per_chunk = false;
    static constexpr bool upfront = false;
    static constexpr bool caller_managed = true;
};

} // namespace
```

#### Pop Policies (when/how to call cb_pop_front)

```cpp
namespace compute_kernel_lib::policies {

/// Pop 1 tile immediately after processing
struct PopPerTile {
    static constexpr bool per_tile = true;
    static constexpr bool per_chunk = false;
    static constexpr bool per_row = false;
    static constexpr bool at_end = false;
    static constexpr bool never = false;
    static constexpr bool caller_managed = false;
};

/// Pop chunk of tiles after processing chunk
struct PopPerChunk {
    static constexpr bool per_tile = false;
    static constexpr bool per_chunk = true;
    static constexpr bool per_row = false;
    static constexpr bool at_end = false;
    static constexpr bool never = false;
    static constexpr bool caller_managed = false;
};

/// Pop 1 tile after each row (for COL broadcast B-input)
struct PopPerRow {
    static constexpr bool per_tile = false;
    static constexpr bool per_chunk = false;
    static constexpr bool per_row = true;
    static constexpr bool at_end = false;
    static constexpr bool never = false;
    static constexpr bool caller_managed = false;
};

/// Pop all tiles at end of operation
struct PopAtEnd {
    static constexpr bool per_tile = false;
    static constexpr bool per_chunk = false;
    static constexpr bool per_row = false;
    static constexpr bool at_end = true;
    static constexpr bool never = false;
    static constexpr bool caller_managed = false;
};

/// Never pop - tiles persist for subsequent operations
struct PopNever {
    static constexpr bool per_tile = false;
    static constexpr bool per_chunk = false;
    static constexpr bool per_row = false;
    static constexpr bool at_end = false;
    static constexpr bool never = true;
    static constexpr bool caller_managed = false;
};

/// Caller is responsible for cb_pop_front after operation
struct PopCallerManaged {
    static constexpr bool per_tile = false;
    static constexpr bool per_chunk = false;
    static constexpr bool per_row = false;
    static constexpr bool at_end = false;
    static constexpr bool never = false;
    static constexpr bool caller_managed = true;
};

} // namespace
```

#### Output Policies (reserve/push behavior)

```cpp
namespace compute_kernel_lib::policies {

/// Reserve/push 1 tile at a time (streaming)
struct OutputPerTile {
    static constexpr bool per_tile = true;
    static constexpr bool per_chunk = false;
    static constexpr bool bulk = false;
};

/// Reserve/push chunk at a time
struct OutputPerChunk {
    static constexpr bool per_tile = false;
    static constexpr bool per_chunk = true;
    static constexpr bool bulk = false;
};

/// Reserve all upfront, push all at end
struct OutputBulk {
    static constexpr bool per_tile = false;
    static constexpr bool per_chunk = false;
    static constexpr bool bulk = true;
};

} // namespace
```

### 3.2 Combined Input Policy

Compose wait and pop policies into a single input policy:

```cpp
namespace compute_kernel_lib::policies {

/// Combined input policy: how to handle an input CB
template <typename WaitPolicy, typename PopPolicy>
struct InputPolicy {
    using wait = WaitPolicy;
    using pop = PopPolicy;

    // Convenience accessors
    static constexpr bool waits_upfront = WaitPolicy::upfront;
    static constexpr bool pops_never = PopPolicy::never;
    static constexpr bool caller_manages_wait = WaitPolicy::caller_managed;
    static constexpr bool caller_manages_pop = PopPolicy::caller_managed;
};

} // namespace
```

### 3.3 Predefined Policy Combinations

For backward compatibility and common use cases:

```cpp
namespace compute_kernel_lib::policies {

// === Equivalent to current BinaryInputMode enum ===

/// STREAMING: Wait/pop 1 tile at a time
using Streaming = InputPolicy<WaitPerTile, PopPerTile>;

/// STREAMING_BATCHED: Wait/pop chunks of DEST_LIMIT tiles
using StreamingBatched = InputPolicy<WaitPerChunk, PopPerChunk>;

/// PRELOADED: Caller manages all wait/pop
using Preloaded = InputPolicy<WaitCallerManaged, PopCallerManaged>;

/// PERSISTENT: Wait all upfront, never pop
using Persistent = InputPolicy<WaitUpfront, PopNever>;


// === Broadcast-specific B-input policies (sensible defaults, not requirements) ===
//
// These are provided as convenient defaults that match common usage patterns.
// Users can override with any policy - the broadcast dimension does NOT
// enforce a specific policy. For example, you CAN use PopAtEnd with ROW
// broadcast if that fits your use case.

/// For ROW broadcast: B has Wt tiles, waited upfront, never popped
/// (Common pattern: bias addition where bias is reused across rows)
using BroadcastRowB = InputPolicy<WaitUpfront, PopNever>;

/// For COL broadcast: B has Ht tiles, 1 waited/popped per row
/// (Common pattern: per-row scaling where each row needs different scale)
using BroadcastColB = InputPolicy<WaitPerRow, PopPerRow>;

/// For SCALAR broadcast: B has 1 tile, waited upfront, never popped
/// (Common pattern: global scaling factor applied to all tiles)
using BroadcastScalarB = InputPolicy<WaitUpfront, PopNever>;

/// For NONE broadcast: B follows same pattern as A
/// (Common pattern: element-wise operations on same-shaped tensors)
template <typename InputAPolicy>
using BroadcastNoneB = InputAPolicy;

} // namespace
```

### 3.4 New API Signature

```cpp
namespace compute_kernel_lib {

/// Binary operation with explicit policies
template <
    BinaryOpType op_type,
    BroadcastDim bcast_dim,
    typename InputAPolicy = policies::Streaming,
    typename InputBPolicy = /* auto-select based on bcast_dim */,
    typename OutputPolicy = policies::OutputPerTile,
    typename AccumT = NoAccumulation,
    typename PostOp = NoOp>
ALWI void binary_op(
    uint32_t icb_a,
    uint32_t icb_b,
    uint32_t ocb,
    BinaryTileShape shape,
    BinaryTileLayout layout = {},
    AccumT accum = {},
    PostOp post_op = {});

} // namespace
```

### 3.5 Auto-Selection of B Policy

For convenience, the default B policy can be auto-selected based on broadcast dimension:

```cpp
namespace compute_kernel_lib::detail {

template <BroadcastDim bcast_dim, typename InputAPolicy>
struct DefaultInputBPolicy {
    using type = typename std::conditional_t<
        bcast_dim == BroadcastDim::ROW, policies::BroadcastRowB,
        std::conditional_t<
            bcast_dim == BroadcastDim::COL, policies::BroadcastColB,
            std::conditional_t<
                bcast_dim == BroadcastDim::SCALAR, policies::BroadcastScalarB,
                InputAPolicy  // NONE: follows A
            >
        >
    >;
};

} // namespace
```

---

## 4. Usage Examples

### 4.1 Simple Streaming (Current Default Behavior)

```cpp
// Explicit policies (new API)
compute_kernel_lib::binary_op<
    BinaryOpType::ADD,
    BroadcastDim::NONE,
    policies::Streaming,    // A: wait/pop per tile
    policies::Streaming,    // B: wait/pop per tile
    policies::OutputPerTile // Out: reserve/push per tile
>(cb_a, cb_b, cb_out, shape);

// Or with defaults (equivalent)
compute_kernel_lib::add(cb_a, cb_b, cb_out, shape);
```

### 4.2 Row Broadcast with Explicit B Persistence

```cpp
// Clear that B persists - no hidden behavior
compute_kernel_lib::binary_op<
    BinaryOpType::MUL,
    BroadcastDim::ROW,
    policies::Streaming,        // A: stream normally
    policies::BroadcastRowB,    // B: Wt tiles, persist (explicit!)
    policies::OutputPerTile
>(cb_data, cb_bias, cb_out, shape);
```

### 4.3 Custom Policy for Special Pattern

```cpp
// Stream A, but keep B for multiple operations
using StreamA = policies::Streaming;
using PersistB = policies::InputPolicy<policies::WaitUpfront, policies::PopNever>;

compute_kernel_lib::mul<BroadcastDim::SCALAR, StreamA, PersistB>(
    cb_data, cb_scale, cb_temp, shape);

// B tiles still available!
compute_kernel_lib::add<BroadcastDim::SCALAR, StreamA, PersistB>(
    cb_temp, cb_bias, cb_out, shape);
```

### 4.4 Preloaded with Caller-Managed CBs

```cpp
// Caller waits for all tiles
cb_wait_front(cb_a, Ht * Wt);
cb_wait_front(cb_b, Ht * Wt);

// Compute with caller-managed policy
compute_kernel_lib::add<
    BroadcastDim::NONE,
    policies::Preloaded,  // Caller manages A
    policies::Preloaded,  // Caller manages B
    policies::OutputBulk  // Bulk output
>(cb_a, cb_b, cb_out, shape);

// Caller pops
cb_pop_front(cb_a, Ht * Wt);
cb_pop_front(cb_b, Ht * Wt);
```

---

## 5. Shared Vocabulary with Dataflow Helpers

### 5.1 The Key Benefit

When future dataflow helpers use the same policies, reader-compute-writer coordination becomes type-safe:

```cpp
// === Reader kernel (producer) ===
template <typename InputBPolicy>
void read_broadcast_tiles(uint32_t cb_b, uint32_t num_tiles, ...) {
    // Reader knows the contract from the policy
    for (uint32_t i = 0; i < num_tiles; ++i) {
        // ... read tile ...
        cb_push_back(cb_b, 1);
    }

    // If policy says tiles persist, reader knows not to expect pop signals
    if constexpr (InputBPolicy::pop::never) {
        // Tiles will remain until explicitly managed
    }
}

// === Compute kernel (consumer) ===
template <typename InputBPolicy>
void binary_op_with_broadcast(...) {
    // Compute honors the same contract
    if constexpr (InputBPolicy::wait::upfront) {
        cb_wait_front(cb_b, num_tiles);  // Wait all at once
    }

    // ... process ...

    if constexpr (!InputBPolicy::pop::never) {
        cb_pop_front(cb_b, num_tiles);   // Pop if policy says to
    }
    // else: tiles persist for caller
}
```

### 5.2 Example: Complete Reader-Compute Coordination

```cpp
// Shared policy definition
using BiasInputPolicy = policies::InputPolicy<
    policies::WaitUpfront,   // Compute will wait for all Wt tiles
    policies::PopNever       // Tiles persist for potential reuse
>;

// --- In reader kernel ---
void reader_main() {
    // Reader uses same policy type
    dataflow_lib::read_tiles<BiasInputPolicy>(cb_bias, Wt, dram_addr);
    // Reader knows: push Wt tiles, they'll persist
}

// --- In compute kernel ---
void compute_main() {
    // Compute uses same policy type
    compute_kernel_lib::add<
        BroadcastDim::ROW,
        policies::Streaming,
        BiasInputPolicy  // Same policy!
    >(cb_data, cb_bias, cb_out, shape);
    // Compute knows: wait Wt upfront, don't pop
}
```

---

## 6. Implementation Strategy

Since the binary helpers are not yet widely used in the codebase, we can do a **clean replacement** rather than a gradual migration. This avoids maintaining two parallel APIs.

### 6.1 Phase 1: Policy Infrastructure

1. Create `cb_policies.hpp` with all policy types
2. Define orthogonal Wait, Pop, and Output policies
3. Define combined `InputPolicy` template
4. Define predefined combinations (`Streaming`, `Persistent`, etc.)

### 6.2 Phase 2: Replace Binary Helpers API

1. Rewrite `binary_op_helpers.hpp` to use policies as template parameters
2. Remove the `BinaryInputMode` enum entirely
3. Implementation uses `if constexpr` to dispatch based on policy traits
4. Update the few existing usages in the codebase

### 6.3 Phase 3: Apply to Reduce Helpers

1. Apply same policy system to `reduce_helpers.hpp`
2. Remove `ReduceInputMode` enum
3. Consistent API across all helpers

### 6.4 Phase 4: Dataflow Helpers

1. Create dataflow helper library using same policies
2. Reader helpers: `read_tiles<Policy>`, `read_broadcast<Policy>`
3. Writer helpers: `write_tiles<Policy>`
4. Policies become the shared contract language between reader-compute-writer

### 6.5 Phase 5: Documentation

1. Update `binary_op_cb_contract.md` to use policy terminology
2. Create examples showing reader-compute-writer coordination
3. Document recommended policies for common patterns

---

## 7. Design Decisions & Trade-offs

### 7.1 Verbose vs Implicit

**Trade-off:** Policy-based API is more verbose than enum-based.

```cpp
// Enum (concise)
binary_op<ADD, ROW, STREAMING>(...);

// Policy (verbose but explicit)
binary_op<ADD, ROW, policies::Streaming, policies::BroadcastRowB>(...);
```

**Mitigation:**
- Provide predefined policy combinations
- Default template arguments for common cases
- Convenience aliases (`add`, `mul`, etc.)

### 7.2 Compile-Time vs Runtime

**Decision:** Policies are compile-time only (template parameters).

**Rationale:**
- Zero runtime overhead
- Enables `if constexpr` optimization
- Type-safe contracts
- Matches existing pattern (current enums are also template params)

### 7.3 Granularity

**Decision:** Separate Wait and Pop policies (not bundled).

**Rationale:**
- More flexible composition
- Clearer semantics
- Can express patterns like "wait upfront, pop at end"
- Avoids combinatorial explosion of named combinations

---

## 8. Open Questions

### 8.1 Naming

- `InputPolicy` vs `CBInputPolicy` vs `CircularBufferPolicy`?
- `PopNever` vs `Persist` vs `NoPop`?
- Namespace: `policies::` vs `cb_policies::` vs inline in `compute_kernel_lib::`?

### 8.2 Default B Policy Selection

Should the default B policy be auto-selected based on broadcast dimension?

```cpp
// Option A: Always explicit
binary_op<ADD, ROW, StreamingA, BroadcastRowB>(...);

// Option B: Auto-select B based on broadcast (current behavior as default)
binary_op<ADD, ROW, StreamingA>(...);  // B policy auto-selected
```

### 8.3 Validation

Should we add compile-time validation for incompatible policy combinations?

```cpp
// Should this fail at compile time?
binary_op<ADD, ROW,
    policies::Streaming,
    policies::Streaming  // Wrong! ROW broadcast B should persist
>(...);
```

### 8.4 Reduce Helpers

Should `reduce_helpers.hpp` use the same policy system?

Current reduce has similar modes: STREAMING, STREAMING_BATCHED, PRELOADED, PERSISTENT.

### 8.5 Output Policies

Are output policies needed, or is the current behavior (follows input mode) sufficient?

---

## 9. Next Steps

1. **Review this design** - Gather feedback on policy granularity and naming
2. **Prototype policies.hpp** - Implement basic policy types
3. **Add policy-based overload** - Single function with policies, verify it works
4. **Document CB contracts** - Update `binary_op_cb_contract.md` to use policy terminology
5. **Design dataflow policies** - Plan how reader/writer helpers will use same policies

---

## Appendix A: Policy Quick Reference

| Policy | Wait Behavior | Pop Behavior |
|--------|---------------|--------------|
| `Streaming` | Per tile | Per tile |
| `StreamingBatched` | Per chunk | Per chunk |
| `Preloaded` | Caller managed | Caller managed |
| `Persistent` | Upfront | Never |
| `BroadcastRowB` | Upfront (Wt) | Never |
| `BroadcastColB` | Per row (1) | Per row |
| `BroadcastScalarB` | Upfront (1) | Never |

## Appendix B: Before/After Comparison

### Old API (Enum-based, to be replaced)

```cpp
compute_kernel_lib::mul<
    BroadcastDim::SCALAR,
    BinaryInputMode::STREAMING>(cb_data, cb_scale, cb_out, shape);
// Hidden: B tile persists (not popped) - must know implementation details
```

### New API (Policy-based)

```cpp
// Explicit control over both inputs
compute_kernel_lib::mul<
    BroadcastDim::SCALAR,
    policies::Streaming,       // A: stream normally
    policies::BroadcastScalarB // B: 1 tile, persists (explicit!)
>(cb_data, cb_scale, cb_out, shape);

// Or with custom B behavior (user decides persistence)
compute_kernel_lib::mul<
    BroadcastDim::SCALAR,
    policies::Streaming,
    policies::InputPolicy<policies::WaitUpfront, policies::PopAtEnd>  // B: pop when done
>(cb_data, cb_scale, cb_out, shape);
```

The key difference: **the user explicitly chooses whether B persists or not**, rather than having it implicitly determined by broadcast dimension.

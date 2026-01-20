# Unary SFPU Operation Helper Design Research

## Executive Summary

This document analyzes how SFPU unary operations (rsqrt, exp, recip, log, etc.) are used across the tt-metal codebase and proposes a simple, minimal-overhead helper design that can handle:
1. **L1→L1 operations**: Full pipeline with CB management
2. **DEST→L1 operations**: For post-op use when tile is already in DST register
3. **Post-op lambdas**: For use within other helpers (reduce, binary_op, etc.)

## 1. Current Usage Patterns

### 1.1 The Basic Pattern: init + tile

Every SFPU unary operation follows this two-step pattern:
```cpp
rsqrt_tile_init();    // Initialize the operation (once before loop or per tile)
rsqrt_tile(dst_idx);  // Execute on tile in DST[dst_idx]
```

The initialization configures the SFPU hardware. The tile call executes on data already in a DST register.

### 1.2 Single Operation L1→L1

Most common pattern - applies one unary op to tiles from a CB:

```cpp
// From batch_norm_sfpu_kernel.cpp:53-57
rsqrt_tile_init();
for (uint32_t i = 0; i < onetile; ++i) {
    rsqrt_tile(i * 2);
    pack_tile(i * 2, cb_den);
}
```

**Frequency**: ~50+ occurrences across the codebase

### 1.3 Chained Operations (Multiple Unary Ops in Sequence)

Less common but important - multiple operations applied to the same tile:

```cpp
// From mish_kernel.cpp:36-41 - 3 ops chained
exp_tile_init<1u>();
exp_tile<1u>(0);           // Op 1: exp(x)

log1p_tile_init<true>();
log1p_tile<true>(0);       // Op 2: log1p(exp(x))

tanh_tile_init();
tanh_tile(0);              // Op 3: tanh(log1p(exp(x)))
```

**Key insight**: Each init must be called before its corresponding tile call, but they can be interleaved. The result stays in the same DST register, feeding into the next operation.

### 1.4 Conditional Chaining

```cpp
// From moreh_softmax_h.cpp:78-97
copy_tile(cb_x_m_max, h, dst0);

#ifndef SOFTMAX
negative_tile_init();
negative_tile(dst0);       // Conditional op 1
#endif

exp_tile_init();
exp_tile(dst0);            // Always executed op 2

if (h == Ht - 1) {
    mask_tile_init();
    mask_tile(dst0, dst1); // Conditional op 3 on last iteration
}
```

### 1.5 Post-Op Pattern in Reduce Helper

The reduce helper already supports post-op lambdas:

```cpp
// From moreh_softmax_h.cpp:116-119
compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_COL, ...>(
    cb_exps, cb_bcast_scaler, cb_recipsumexps,
    compute_kernel_lib::TileShape::col(Ht),
    {}, {},  // layout, accum
    [](uint32_t dst_idx) {   // Post-op lambda
        log_tile_init();
        log_tile(dst_idx);
    });
```

## 2. Analysis: When Are Multiple Ops Chained?

Searched the codebase for chaining patterns:

| Pattern | Count | Examples |
|---------|-------|----------|
| Single unary op | ~80% | rsqrt, exp, log alone |
| 2 ops chained | ~15% | negative+exp, sub+exp, exp+recip |
| 3+ ops chained | ~5% | mish (exp→log1p→tanh), softplus+negative |

**Conclusion**: Most operations are single unary ops. Chaining is relatively rare but important for operations like:
- **Softmax**: exp → sum → recip/log
- **Mish**: exp → log1p → tanh
- **Sigmoid**: exp → add(1) → recip
- **LayerNorm**: add → rsqrt

## 3. DST Register Management

### 3.1 The DST Register API

The DST (destination) register is a shared resource between MATH and PACK threads. It's an array of 16 tiles of 32x32 elements each.

```cpp
// From tt_metal/include/compute_kernel_api/reg_api.h

// MATH thread operations
tile_regs_acquire();    // Acquire exclusive lock on DST for MATH thread (blocking)
tile_regs_commit();     // Release lock, signal results ready for PACK thread

// PACK thread operations
tile_regs_wait();       // Acquire lock on DST for PACK thread (blocking, waits for MATH commit)
tile_regs_release();    // Release lock, signal DST available for next MATH operation
```

**Thread synchronization flow:**
```
MATH: acquire → compute → commit ────────────────────→ (can acquire again)
                              ↓
PACK: ───────────────────── wait → pack → release
```

### 3.2 Standard Pattern
```cpp
tile_regs_acquire();        // MATH acquires DST

// Load tile from CB to DST
copy_tile_to_dst_init_short(cb_in);
copy_tile(cb_in, tile_idx, dst_idx);

// Apply operations (can chain multiple here!)
rsqrt_tile_init();
rsqrt_tile(dst_idx);

tile_regs_commit();         // MATH releases, signals PACK

tile_regs_wait();           // PACK acquires (waits for commit)
pack_tile(dst_idx, cb_out);
tile_regs_release();        // PACK releases
```

### 3.3 Key Observations

1. **Acquire/release bracket multiple operations** - don't need to release between chained ops
2. **Commit/wait is the handoff point** - MATH→PACK synchronization happens here
3. **Init calls are cheap** - they configure SFPU, not load data
4. **DST index is preserved** - result overwrites input in same register
5. **16 tiles max** - DST has 16 tile slots, but actual usable count depends on data format/sync mode (4-16 tiles, see `dest_helpers.hpp`)

### 3.4 Common Mistakes

1. **Missing acquire/release** - leads to undefined behavior or deadlock
2. **Missing commit before pack** - PACK thread never gets the data
3. **Wrong thread calling wrong API** - e.g., calling `tile_regs_wait()` from MATH thread

## 4. SFPU Init Constraints (Critical!)

### 4.1 What Init Does

Each `*_tile_init()` function configures the SFPU hardware state:

```cpp
// From tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h:80-85
inline void _llk_math_eltwise_unary_sfpu_init_()
{
    sfpu::_init_sfpu_config_reg();                    // Configure SFPU config register
    eltwise_unary_sfpu_configure_addrmod<sfpu_op>();  // Configure address mode for this op
    math::reset_counters(p_setrwc::SET_ABD_F);        // Reset counters
}
```

### 4.2 The "One Init at a Time" Rule

**CRITICAL**: Calling a new init **OVERWRITES** the previous SFPU configuration.

```cpp
// WRONG - rsqrt config is overwritten!
rsqrt_tile_init();
exp_tile_init();      // ← SFPU is now configured for exp, NOT rsqrt!
rsqrt_tile(0);        // ← Will NOT work correctly!
exp_tile(0);

// CORRECT - init immediately before use
rsqrt_tile_init();
rsqrt_tile(0);        // ← Works: SFPU configured for rsqrt
exp_tile_init();      // ← Reconfigure for exp
exp_tile(0);          // ← Works: SFPU configured for exp
```

### 4.3 Why This Matters for Chaining

When chaining operations, you MUST re-init before each different operation:

```cpp
// From mish_kernel.cpp - correct chaining pattern
exp_tile_init<1u>();
exp_tile<1u>(0);           // Op 1: SFPU configured for exp

log1p_tile_init<true>();   // ← Must re-init for log1p!
log1p_tile<true>(0);       // Op 2: SFPU configured for log1p

tanh_tile_init();          // ← Must re-init for tanh!
tanh_tile(0);              // Op 3: SFPU configured for tanh
```

### 4.4 Init Cost is Low

Init calls are cheap - they just configure registers, not move data. Don't try to "optimize" by caching init state - it causes subtle bugs.

```cpp
// This is fine performance-wise:
for (uint32_t i = 0; i < num_tiles; ++i) {
    rsqrt_tile_init();  // ← Called every iteration, but cheap
    rsqrt_tile(i);
}

// This is also fine (init once, reuse for same op):
rsqrt_tile_init();
for (uint32_t i = 0; i < num_tiles; ++i) {
    rsqrt_tile(i);      // ← Same op, init still valid
}
```

### 4.5 Forgetting Init - Silent Failure

If you forget to call init, the operation may:
- Produce garbage results (using previous op's config)
- Appear to work in some cases (if previous config happens to be compatible)
- Fail silently with no error message

This is why a helper that auto-inits is valuable for safety.

## 5. Existing Op Chain Infrastructure

The codebase already has infrastructure for chaining unary ops - used by the `ttnn::unary` operations:

### 5.1 How SFPU_OP_CHAIN Works

```cpp
// From unary_op_utils.cpp:891-910
// Generates macros like SFPU_OP_CHAIN_0_INIT_0, SFPU_OP_CHAIN_0_FUNC_0, etc.
std::map<std::string, std::string> get_block_defines(
    const std::vector<EltwiseUnaryWithParam>& op_chain,
    ...) {
    for (uint32_t i = 0; i < op_chain.size(); i++) {
        std::string init_def = fmt::format("SFPU_OP_CHAIN_{}_INIT_{}", block_id, i);
        std::string func_def = fmt::format("SFPU_OP_CHAIN_{}_FUNC_{}", block_id, i);
        // ...
    }
}
```

### 5.2 The Generated Kernel Pattern

```cpp
// From eltwise_sfpu.cpp - the template kernel
for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
    tile_regs_acquire();
    cb_wait_front(tt::CBIndex::c_0, 1);
    copy_tile(tt::CBIndex::c_0, 0, 0);

#ifdef SFPU_OP_CHAIN_0
    SFPU_OP_CHAIN_0   // ← Expands to: init1(); op1(); init2(); op2(); ...
#endif

    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, tt::CBIndex::c_2);
    cb_pop_front(tt::CBIndex::c_0, 1);
    tile_regs_release();
}
```

### 5.3 Key Insight

The existing infrastructure handles the init/apply pattern correctly by generating interleaved init+apply sequences. This validates our design principle: **each op must have its init immediately before use**.

## 6. Design Options

### Option A: Simple Functor-Based Design (Recommended)

```cpp
// Operation functors - zero overhead due to inlining
struct Rsqrt {
    ALWI void init() const { rsqrt_tile_init(); }
    ALWI void apply(uint32_t dst_idx) const { rsqrt_tile(dst_idx); }
};

struct Exp {
    ALWI void init() const { exp_tile_init(); }
    ALWI void apply(uint32_t dst_idx) const { exp_tile(dst_idx); }
};

// For parametrized ops
template<bool approx = false>
struct Recip {
    ALWI void init() const { recip_tile_init<approx>(); }
    ALWI void apply(uint32_t dst_idx) const { recip_tile<approx>(dst_idx); }
};
```

### Option B: Lambda-Compatible Post-Op Pattern

The reduce helper already uses this pattern effectively:

```cpp
// Works today:
reduce<SUM, REDUCE_ROW>(..., [](uint32_t dst_idx) {
    rsqrt_tile_init();
    rsqrt_tile(dst_idx);
});
```

This is simple and requires no new infrastructure.

### Option C: Variadic Template Chain (More Complex)

```cpp
template<typename... Ops>
ALWI void apply_chain(uint32_t dst_idx, Ops... ops) {
    (ops.init(), ...);        // Initialize all
    (ops.apply(dst_idx), ...); // Apply all in sequence
}

// Usage:
apply_chain(0, Rsqrt{}, Mul{});
```

**Problem**: Init order may matter for some ops, and this approach complicates error messages.

## 7. Recommended Design: Minimal Unary Helper

### 7.1 Core Principle

Keep it simple. Provide two modes:

1. **L1→L1**: Full standalone operation with CB management
2. **DEST→DEST**: For use as post-op (tile already in DST, result stays in DST)

### 7.2 Proposed API

```cpp
// File: unary_helpers.hpp

namespace compute_kernel_lib {

// Mode for unary operations
enum class UnaryInputMode {
    STREAMING,   // L1→L1: Wait/pop input, reserve/push output
    IN_DEST      // DEST→DEST: Tile already in DST, apply in-place
};

/**
 * Apply rsqrt to tiles
 *
 * STREAMING mode: Full L1→L1 pipeline
 *   - Waits for input tiles from icb
 *   - Applies rsqrt
 *   - Writes results to ocb
 *
 * IN_DEST mode: For post-op use
 *   - Tile already loaded in DST[dst_start]
 *   - Applies rsqrt in-place
 *   - Result remains in DST for caller to pack
 */
template<UnaryInputMode mode = UnaryInputMode::STREAMING,
         bool init = true,
         bool reconfig = true>
ALWI void rsqrt(
    uint32_t icb,              // Input CB (ignored if IN_DEST)
    uint32_t ocb,              // Output CB (ignored if IN_DEST)
    uint32_t num_tiles = 1,
    uint32_t dst_start = 0)    // Starting DST index for IN_DEST mode
{
    if constexpr (mode == UnaryInputMode::STREAMING) {
        // Full L1→L1 operation
        if constexpr (reconfig) {
            reconfig_data_format(icb, icb);
            pack_reconfig_data_format(ocb);
        }

        copy_tile_to_dst_init_short(icb);
        if constexpr (init) rsqrt_tile_init();

        for (uint32_t i = 0; i < num_tiles; ++i) {
            cb_wait_front(icb, 1);
            tile_regs_acquire();

            copy_tile(icb, 0, 0);
            rsqrt_tile(0);

            tile_regs_commit();
            cb_reserve_back(ocb, 1);
            tile_regs_wait();
            pack_tile(0, ocb);
            cb_push_back(ocb, 1);
            tile_regs_release();
            cb_pop_front(icb, 1);
        }
    } else {
        // IN_DEST mode - tile already loaded
        if constexpr (init) rsqrt_tile_init();
        for (uint32_t i = 0; i < num_tiles; ++i) {
            rsqrt_tile(dst_start + i);
        }
    }
}

// Convenience: post-op lambda generator
// Returns a lambda compatible with reduce/binary_op post_op parameter
ALWI auto rsqrt_post_op() {
    return [](uint32_t dst_idx) {
        rsqrt_tile_init();
        rsqrt_tile(dst_idx);
    };
}

}  // namespace compute_kernel_lib
```

### 7.3 Usage Examples

**Standalone L1→L1:**
```cpp
// Replace 15 lines of boilerplate with 1 line:
compute_kernel_lib::rsqrt(cb_var, cb_invstd, num_tiles);
```

**Post-op with reduce:**
```cpp
// Apply rsqrt after variance reduction
compute_kernel_lib::reduce<SUM, REDUCE_ROW>(
    cb_centered_sq, cb_scaler, cb_invstd,
    TileShape::row(Wt),
    {}, {},
    compute_kernel_lib::rsqrt_post_op());
```

**In manual DEST management:**
```cpp
tile_regs_acquire();
copy_tile(cb_in, 0, 0);

// Chain operations manually
compute_kernel_lib::add_scalar<UnaryInputMode::IN_DEST>(eps_cb, 0, 1, 0);  // var + eps
compute_kernel_lib::rsqrt<UnaryInputMode::IN_DEST>(0, 0, 1, 0);            // rsqrt(var + eps)

tile_regs_commit();
// ... pack ...
```

### 7.4 Chaining Support

For the rare cases where chaining is needed, keep it simple:

```cpp
// Option 1: Sequential calls (most explicit)
compute_kernel_lib::exp<IN_DEST>(0, 0, 1, 0);
compute_kernel_lib::log1p<IN_DEST>(0, 0, 1, 0);
compute_kernel_lib::tanh<IN_DEST>(0, 0, 1, 0);

// Option 2: Direct raw calls (already simple)
exp_tile_init(); exp_tile(0);
log1p_tile_init(); log1p_tile(0);
tanh_tile_init(); tanh_tile(0);
```

The raw API is already quite clean for chaining. A variadic template helper would add complexity without significant benefit.

## 8. Operations to Support

Based on usage frequency:

| Priority | Operation | Usage Count | Notes |
|----------|-----------|-------------|-------|
| High | rsqrt | 30+ | LayerNorm, BatchNorm |
| High | exp | 40+ | Softmax, activations |
| High | recip | 25+ | Softmax |
| High | log | 15+ | LogSoftmax |
| Medium | sqrt | 10+ | Various |
| Medium | negative | 10+ | Softmax variants |
| Medium | tanh | 8+ | Activations |
| Medium | sigmoid | 5+ | Activations |
| Low | gelu | 5+ | Activations |
| Low | relu | 5+ | Activations |

## 9. Implementation Recommendations

### 9.1 Start Simple

1. Implement `rsqrt` helper first (most needed for layernorm work)
2. Add `recip`, `exp`, `log` next (softmax family)
3. Extend to others as needed

### 9.2 Template Structure

```cpp
// Generic template that can be specialized for each op
template<typename Op, UnaryInputMode mode = UnaryInputMode::STREAMING, ...>
ALWI void unary_op(uint32_t icb, uint32_t ocb, uint32_t num_tiles, uint32_t dst_start);

// Convenient aliases
template<UnaryInputMode mode = UnaryInputMode::STREAMING, ...>
ALWI void rsqrt(...) { unary_op<RsqrtOp, mode>(...);}
```

### 9.3 Post-Op Factory

Provide a factory for each operation to generate post-op lambdas:

```cpp
// Simple lambda generators
ALWI auto rsqrt_post_op() { return [](uint32_t d) { rsqrt_tile_init(); rsqrt_tile(d); }; }
ALWI auto recip_post_op() { return [](uint32_t d) { recip_tile_init(); recip_tile(d); }; }
ALWI auto exp_post_op()   { return [](uint32_t d) { exp_tile_init(); exp_tile(d); }; }
```

These can be used directly with the existing reduce helper's post_reduce_op parameter.

## 10. Avoiding Over-Engineering

**Don't do:**
- Complex variadic template chains
- Runtime operation dispatch
- Heavy abstraction layers
- Automatic reconfig detection

**Do:**
- Simple, explicit APIs
- Compile-time template parameters
- Manual chaining when needed (raw API is already clean)
- Let user control init/reconfig flags

## 11. Conclusion

The recommended design is minimal:

1. **One function per operation** with mode template parameter (STREAMING vs IN_DEST)
2. **Post-op lambda factories** for integration with reduce/binary_op helpers
3. **Manual chaining** using either helper IN_DEST mode or raw API calls

This approach:
- Handles 95% of use cases (single ops)
- Supports complex cases (chaining) with minimal overhead
- Integrates cleanly with existing helpers
- Avoids over-engineering
- Has zero runtime overhead due to templates and inlining

The raw API (`rsqrt_tile_init()` + `rsqrt_tile()`) is already quite clean for the rare chaining cases, so providing a complex chaining abstraction would add more complexity than value.

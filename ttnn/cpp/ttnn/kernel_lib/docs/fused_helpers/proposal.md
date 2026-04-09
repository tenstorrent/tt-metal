# Fused Matmul Helpers — Design Proposal

**Commit**: a62a03c2181e083484fb6ba0496610b2d66c0ba7
**Branch**: wransom/fused2

Three design options for fused matmul helpers. All options share the same scope: handle one
"output block" iteration (K-blocking loop + optional bias + optional activation + optional
untilize + PACK_RELU lifecycle). Outer loops (batch/bh/bw) remain the caller's responsibility.

All options also share a common building block:

## Common: reblock_and_untilize helper (all options include this)

Extracted from identical code in C1:108-142 and D1:150-178.

```cpp
// File: reblock_and_untilize_helpers.hpp
namespace compute_kernel_lib {

template <uint32_t out_subblock_w, uint32_t out_block_w>
ALWI void reblock_and_untilize(
    uint32_t num_out_subblocks_in_col,  // = in1_num_subblocks
    uint32_t out_subblock_num_tiles,    // = out_subblock_h * out_subblock_w
    uint32_t out_subblock_h,
    uint32_t interm_cb,
    uint32_t out_cb);

}  // namespace compute_kernel_lib
```

Algorithm: wait for row of subblocks in interm_cb, for each tile row gather subblocks
via copy_tile, pack with pack_untilize_dest, push to out_cb. Identical to current
duplicated code.

---

## Option A: Enhanced matmul_block (incremental)

**Approach**: Extend the existing `matmul_block` helper with additional template parameters
for bias and untilize. No new top-level helpers beyond reblock_and_untilize.

### API Changes to matmul_block

```cpp
template <
    uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t interm_cb,
    uint32_t in0_num_subblocks, uint32_t in1_num_subblocks,
    uint32_t out_subblock_h, uint32_t out_subblock_w,
    bool transpose = false,
    bool packer_l1_acc = false,
    bool pack_last_to_interm = false,   // existing
    typename PostComputeFn = matmul_block_config::NoPostCompute,
    typename PreKBlockFn = matmul_block_config::NoPreKBlock,
    // NEW parameters:
    bool hw_relu = false,               // replaces HwRelu tag type
    uint32_t bias_cb = CB_NONE,         // CB_NONE = no bias
    uint32_t untilize_out_cb = CB_NONE, // CB_NONE = no untilize
    typename PostBiasFn = matmul_block_config::NoPostCompute>
ALWI void matmul_block(
    const uint32_t block_w,
    const uint32_t num_k_blocks,
    const uint32_t batch = 1,
    PostComputeFn post_compute = {},
    PreKBlockFn pre_k_block = {},
    PostBiasFn post_bias = {},
    const uint32_t bias_width_tiles = 0);  // only used when bias_cb != CB_NONE
```

When `bias_cb != CB_NONE`: the helper sets `pack_last_to_interm=true` internally,
runs the bias add loop after K-blocks, applies PostBiasFn after bias, packs to
`out_cb`. When `untilize_out_cb != CB_NONE`: packs to interm_cb, then runs
reblock_and_untilize.

### Migration Example (C1 kernel)

**Before** (~200 lines of K-loop + bias + untilize + #ifdef paths):
```cpp
// ... complex K-loop with 10 #ifdef paths ...
#ifdef FUSE_BIAS
    // ... 60 lines of bias add ...
#endif
if constexpr (untilize_out) {
    // ... 20 lines of untilize ...
}
```

**After**:
```cpp
mm_block_init(in0_cb_id, in1_cb_id, mm_partials_cb_id, ...);
#ifdef SFPU_OP_INIT_ACTIVATION
    SFPU_OP_INIT_ACTIVATION
#endif

for (uint32_t b = 0; b < batch; b++) {
    for (uint32_t bh = 0; bh < num_blocks_h_dim; ++bh) {
        for (uint32_t bw = 0; bw < num_blocks_w_dim; ++bw) {
            compute_kernel_lib::matmul_block<
                in0_cb_id, in1_cb_id, out_cb_id, mm_partials_cb_id,
                in0_num_subblocks, in1_num_subblocks,
                out_subblock_h, out_subblock_w,
                in1_transpose_tile,
                PACKER_L1_ACC_ENABLED,
                false,  // pack_last_to_interm (managed internally)
                ActivationFn, PreKBlockFn,
                PACK_RELU_ENABLED,
                BIAS_CB,         // CB_NONE when no bias
                UNTILIZE_OUT_CB, // CB_NONE when no untilize
                PostBiasFn
            >(block_w, num_k_blocks, 1, activation_fn, pre_k_block_fn,
              post_bias_fn, bias_width_tiles);
        }
    }
}
```

### Trade-offs

| Pro | Con |
|-----|-----|
| Minimal new code — extends existing proven helper | Template parameter list grows to 17+ params |
| Single function call per output block | Harder for agents to understand which params interact |
| No new concepts to learn | `pack_last_to_interm` and `bias_cb` create implicit coupling |
| Backward compatible (new params have defaults) | Testing combinatorial explosion (hw_relu × bias × untilize × l1_acc × ...) |

### Migration Tiers

- **Tier 1**: C1 (fused bias activation) — direct replacement, all features map to params
- **Tier 2**: D1 (conv) — matmul core maps, but tilize + CB pointer manipulation stay outside
- **Tier 3**: C2/C3 (gathered), SDPA, MoE — cannot use (too specialized)

---

## Option B: Composable Building Blocks (modular)

**Approach**: Keep existing matmul_block and bias_add helpers unchanged. Add new standalone
helpers for each phase, plus a thin orchestrator that chains them with correct reconfiguration.

### New Helpers

#### 1. reblock_and_untilize (same as common, above)

#### 2. matmul_output_block — orchestrator

Chains: matmul_block → [bias_add] → [untilize] with correct PACK_RELU lifecycle,
data format reconfiguration, and L1_ACC management between phases.

```cpp
namespace compute_kernel_lib {

namespace fused_matmul_config {

// Tag: no bias phase
struct NoBias {};

// Config: bias phase enabled
struct BiasConfig {
    uint32_t bias_cb;
    uint32_t bias_width_tiles;
};

// Tag: no untilize phase
struct NoUntilize {};

// Config: untilize phase enabled (uses reblock_and_untilize)
template <uint32_t out_subblock_w, uint32_t out_block_w>
struct UntilizeConfig {};

}  // namespace fused_matmul_config

template <
    // Matmul config (same as existing matmul_block)
    uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t interm_cb,
    uint32_t in0_num_subblocks, uint32_t in1_num_subblocks,
    uint32_t out_subblock_h, uint32_t out_subblock_w,
    bool transpose = false,
    bool packer_l1_acc = false,
    // Post-matmul config (new)
    typename BiasOpt = fused_matmul_config::NoBias,        // NoBias or BiasConfig
    typename UntilizeOpt = fused_matmul_config::NoUntilize, // NoUntilize or UntilizeConfig<w,bw>
    typename PostComputeFn = matmul_block_config::NoPostCompute,  // SFPU after matmul (no-bias path)
    typename PostBiasFn = bias_add_config::NoPostBias,            // SFPU after bias
    typename PreKBlockFn = matmul_block_config::NoPreKBlock,
    bool hw_relu = false>
ALWI void matmul_output_block(
    const uint32_t block_w,
    const uint32_t num_k_blocks,
    BiasOpt bias_opt = {},
    PostComputeFn post_compute = {},
    PostBiasFn post_bias = {},
    PreKBlockFn pre_k_block = {});

}  // namespace compute_kernel_lib
```

### Internal Implementation

```
matmul_output_block:
  1. if hw_relu && !has_bias: configure PACK_RELU lifecycle in matmul_block
  2. call matmul_block<..., pack_last_to_interm=has_bias||has_untilize>
  3. if has_bias:
     a. if hw_relu: PACK((llk_pack_relu_config(ReluType::ZERO_RELU)))
     b. pack_reconfig + L1_ACC disable
     c. call add_bias_bcast_rows<interm_cb, bias_cb, untilize_target_cb>(...)
  4. if has_untilize:
     a. if !has_bias: reconfig_data_format_srca + pack_reconfig + L1_ACC disable
     b. pack_untilize_dest_init
     c. call reblock_and_untilize<subblock_w, block_w>(...)
     d. pack_untilize_uninit
  5. if hw_relu: PACK((llk_pack_relu_config(ReluType::NO_RELU)))
```

### Migration Example (C1 kernel)

**After**:
```cpp
using namespace compute_kernel_lib;
using namespace fused_matmul_config;

mm_block_init(in0_cb_id, in1_cb_id, mm_partials_cb_id, ...);

// Conditionally define bias and untilize configs
#ifdef FUSE_BIAS
constexpr auto bias_opt = BiasConfig{bias_cb_id, in1_block_w};
#else
constexpr auto bias_opt = NoBias{};
#endif

for (uint32_t b = 0; b < batch; b++) {
    for (uint32_t bh = 0; bh < num_blocks_h_dim; ++bh) {
        for (uint32_t bw = 0; bw < num_blocks_w_dim; ++bw) {

            if constexpr (untilize_out) {
                matmul_output_block<
                    in0_cb_id, in1_cb_id, out_cb_id, mm_partials_cb_id,
                    in0_num_subblocks, in1_num_subblocks,
                    out_subblock_h, out_subblock_w,
                    in1_transpose_tile, PACKER_L1_ACC_ENABLED,
                    decltype(bias_opt),
                    UntilizeConfig<out_subblock_w, out_block_w>,
                    ActivationFn, PostBiasFn, PreKBlockFn,
                    PACK_RELU_ENABLED
                >(block_w, num_k_blocks, bias_opt, activation_fn, post_bias_fn, pre_k_block_fn);
            } else {
                matmul_output_block<
                    in0_cb_id, in1_cb_id, out_cb_id, mm_partials_cb_id,
                    in0_num_subblocks, in1_num_subblocks,
                    out_subblock_h, out_subblock_w,
                    in1_transpose_tile, PACKER_L1_ACC_ENABLED,
                    decltype(bias_opt), NoUntilize,
                    ActivationFn, PostBiasFn, PreKBlockFn,
                    PACK_RELU_ENABLED
                >(block_w, num_k_blocks, bias_opt, activation_fn, post_bias_fn, pre_k_block_fn);
            }

            // Caller handles: mm_block_init_short reconfigure for next bh/bw
        }
    }
}
```

### Trade-offs

| Pro | Con |
|-----|-----|
| Existing helpers unchanged — no regression risk | New `matmul_output_block` is still moderately complex |
| Each building block independently testable | Caller needs to handle bh/bw reconfiguration |
| Tag types (NoBias, NoUntilize) make intent clear | Two `if constexpr` branches for untilize (template arg) |
| Reuses proven matmul_block + bias_add internally | Config structs add a new pattern to learn |

### Migration Tiers

Same as Option A.

---

## Option C: Config-Struct Pipeline (highest abstraction)

**Approach**: A single pipeline function configured by a compile-time config struct that
describes which stages are active. The struct captures ALL the parameters, reducing the
template argument list. Uses `if constexpr` extensively to eliminate inactive stages.

### API

```cpp
namespace compute_kernel_lib {

template <
    uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t interm_cb,
    uint32_t in0_num_subblocks, uint32_t in1_num_subblocks,
    uint32_t out_subblock_h, uint32_t out_subblock_w>
struct MatmulPipelineConfig {
    // Matmul phase
    static constexpr bool transpose = false;
    static constexpr bool packer_l1_acc = false;

    // Bias phase (set bias_cb to enable)
    static constexpr uint32_t bias_cb = CB_NONE;  // CB_NONE = no bias

    // Untilize phase (set to true to enable)
    static constexpr bool untilize_out = false;
    static constexpr uint32_t untilize_block_w = 0;  // = out_subblock_w * in1_num_subblocks

    // PACK_RELU
    static constexpr bool hw_relu = false;
};

// User specializes for their kernel:
struct MyConfig : MatmulPipelineConfig<cb_in0, cb_in1, cb_out, cb_interm, 2, 4, 2, 2> {
    static constexpr bool packer_l1_acc = true;
    static constexpr uint32_t bias_cb = cb_bias;
    static constexpr bool untilize_out = true;
    static constexpr uint32_t untilize_block_w = 8;
    static constexpr bool hw_relu = true;
};

template <
    typename Config,
    typename PostComputeFn = matmul_block_config::NoPostCompute,
    typename PostBiasFn = bias_add_config::NoPostBias,
    typename PreKBlockFn = matmul_block_config::NoPreKBlock>
ALWI void matmul_pipeline(
    const uint32_t block_w,
    const uint32_t num_k_blocks,
    const uint32_t bias_width_tiles = 0,
    PostComputeFn post_compute = {},
    PostBiasFn post_bias = {},
    PreKBlockFn pre_k_block = {});

}  // namespace compute_kernel_lib
```

### Migration Example (C1 kernel)

**After**:
```cpp
using namespace compute_kernel_lib;

struct FusedConfig : MatmulPipelineConfig<
    in0_cb_id, in1_cb_id, out_cb_id, mm_partials_cb_id,
    in0_num_subblocks, in1_num_subblocks, out_subblock_h, out_subblock_w> {
    static constexpr bool transpose = in1_transpose_tile;
    static constexpr bool packer_l1_acc = PACKER_L1_ACC_ENABLED;
    static constexpr uint32_t bias_cb = BIAS_CB;  // CB_NONE when no bias
    static constexpr bool untilize_out = UNTILIZE_OUT;
    static constexpr uint32_t untilize_block_w = out_block_w;
    static constexpr bool hw_relu = PACK_RELU_ENABLED;
};

mm_block_init(in0_cb_id, in1_cb_id, mm_partials_cb_id, ...);

for (uint32_t b = 0; b < batch; b++) {
    for (uint32_t bh = 0; bh < num_blocks_h_dim; ++bh) {
        for (uint32_t bw = 0; bw < num_blocks_w_dim; ++bw) {
            matmul_pipeline<FusedConfig, ActivationFn, PostBiasFn, PreKBlockFn>(
                block_w, num_k_blocks, bias_width_tiles,
                activation_fn, post_bias_fn, pre_k_block_fn);
            // Caller handles bh/bw reconfigure
        }
    }
}
```

### Trade-offs

| Pro | Con |
|-----|-----|
| Cleanest call site — single function + config struct | New pattern (config struct inheritance) unfamiliar to RISC-V kernel devs |
| Config is self-documenting — all options visible in one struct | Template error messages may be harder to debug |
| Easy to add new stages without changing the function signature | Agent must learn config struct pattern |
| Compile-time validation in config struct | Requires C++17 aggregate inheritance support in RISC-V toolchain |

### Migration Tiers

Same as Options A/B.

---

## Comparison Matrix

| Criterion | Option A (Enhanced) | Option B (Composable) | Option C (Config Struct) |
|-----------|--------------------|-----------------------|--------------------------|
| **New files** | 1 (reblock_and_untilize) | 2 (reblock_and_untilize + matmul_output_block) | 2 (reblock_and_untilize + matmul_pipeline) |
| **Changes to existing helpers** | Major (matmul_block signature) | None | None |
| **Template params at call site** | 17+ | 16 | 4 (Config + 3 functors) |
| **Agent comprehension** | Medium (many params to reason about) | Good (composable, each piece simple) | Best (config is self-documenting) but unfamiliar pattern |
| **Testing complexity** | High (one big function) | Medium (each piece testable) | Medium (pipeline testable) |
| **Backward compatibility** | Full (defaults) | Full (existing helpers unchanged) | Full (existing helpers unchanged) |
| **Risk** | Medium (modifying proven code) | Low (additive) | Low (additive) but toolchain risk |
| **Existing kernel migration effort** | Low | Medium | Medium |

## Recommendation

**Option B (Composable Building Blocks)** balances correctness risk, testability, and agent
comprehension. It doesn't modify existing proven helpers, each piece can be tested in
isolation, and the tag-type pattern (NoBias, NoUntilize) is already established in the
codebase (NoPostCompute, HwRelu, NoPreKBlock). The orchestrator `matmul_output_block`
is new but thin — it mostly chains existing helpers with the correct reconfiguration glue.

Option C is cleaner at the call site but introduces a config struct inheritance pattern
that may not compile correctly on the RISC-V toolchain. Option A risks regressions in
the existing proven matmul_block helper.

---

## HUMAN CHECKPOINT

Please review and select an option (or request revision). Key decisions needed:
1. Which option (A, B, or C)?
2. Should the existing `bias_add` helper gain template params for subblock dims (like matmul_block did), or keep them as runtime?
3. Should we handle the in0_transpose via PreKBlockFn (existing mechanism) or a dedicated helper?
4. Acceptable performance overhead threshold for the helper vs raw LLK?

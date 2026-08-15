#pragma once

#include <cstdint>

namespace ttnn::experimental::moe::global_count_contract {

enum class Stage : uint8_t { FusedLocal, MetadataOnly, ReduceGlobal, ComputeWithGlobal };

// Compile-only ABI description for the future in-program mesh count exchange.
// The existing per_expert_total_tokens CB remains local; global_count_cb_id is
// populated only after a fabric reduction and must never alias the local page.
struct CountExchangeSpec {
    uint32_t local_count_cb_id;
    uint32_t global_count_cb_id;
    uint32_t expert_count;
    uint32_t tokens_per_chunk;
};

// Host graph contract for the eventual two-stage path. MetadataOnly must
// produce both local counts and the tilize outputs needed by ComputeWithGlobal;
// ReduceGlobal applies ttnn::all_reduce(sum) to the count tensor between the
// two device programs. FusedLocal remains the production default until the
// metadata-only operation is implemented.
struct TwoStagePlan {
    Stage stage;
    CountExchangeSpec counts;
    bool preserve_fused_default;
};

// In-program alternative: allocate a global-count tensor with
// [num_devices, expert_count] uint32 slots. Each tilize drain core writes its
// local row to its device slot using the same fabric output-address/page-index
// contract as dispatch metadata, waits on the existing init/completion barrier,
// sums all rows locally, then multicasts the sum page to dm0/dm1. This avoids
// arbitrary L1 fabric writes and gives empty owners the global round count.

constexpr bool valid(const CountExchangeSpec& spec) {
    return spec.local_count_cb_id != spec.global_count_cb_id && spec.expert_count > 0 && spec.tokens_per_chunk > 0;
}

constexpr uint32_t chunk_rounds(uint32_t global_tokens, uint32_t tokens_per_chunk) {
    return tokens_per_chunk == 0 ? 0 : (global_tokens + tokens_per_chunk - 1) / tokens_per_chunk;
}

constexpr bool valid(const TwoStagePlan& plan) { return valid(plan.counts) && plan.preserve_fused_default; }

}  // namespace ttnn::experimental::moe::global_count_contract

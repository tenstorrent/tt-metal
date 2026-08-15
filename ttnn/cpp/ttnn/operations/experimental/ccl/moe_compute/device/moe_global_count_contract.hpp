#pragma once

#include <cstdint>

namespace ttnn::experimental::moe::global_count_contract {

// Compile-only ABI description for the future in-program mesh count exchange.
// The existing per_expert_total_tokens CB remains local; global_count_cb_id is
// populated only after a fabric reduction and must never alias the local page.
struct CountExchangeSpec {
    uint32_t local_count_cb_id;
    uint32_t global_count_cb_id;
    uint32_t expert_count;
    uint32_t tokens_per_chunk;
};

constexpr bool valid(const CountExchangeSpec& spec) {
    return spec.local_count_cb_id != spec.global_count_cb_id && spec.expert_count > 0 && spec.tokens_per_chunk > 0;
}

constexpr uint32_t chunk_rounds(uint32_t global_tokens, uint32_t tokens_per_chunk) {
    return tokens_per_chunk == 0 ? 0 : (global_tokens + tokens_per_chunk - 1) / tokens_per_chunk;
}

}  // namespace ttnn::experimental::moe::global_count_contract

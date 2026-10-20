// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Ring joint SDPA writer for the named precision recipes B/C/D/E (compute: ring_joint_sdpa_recipe.cpp).
// Compute keeps one recurrent state across the ring. Multi-Q workers checkpoint its raw tile bytes to the
// internal state tensor (common runtime arg 0; accessor args follow the CB block) on CB17 request / CB18 ack.

#pragma GCC optimize("Os")  // leaves kernel config buffer for the recipe compute

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"
#include "recipe_state_transfer.hpp"

struct RingJointWriterPolicy {
    static constexpr bool kResidentRingState = true;
    // Common runtime arg 0: the internal state tensor (state_backing below).
    static constexpr uint32_t kCommonArgCount = 1;
    // The recipe compute's named compile-time arguments (the ring factory gives every ring kernel the set).
    static constexpr bool kFp32State = get_named_compile_time_arg_val("recipe_fp32") != 0;
    // Streamed checkpoints: BF16 recipes with fused chunks (recipe_state_transfer.hpp, stream_recipe_state).
    static constexpr bool kStreamState = get_named_compile_time_arg_val("recipe_ring_stream_state") != 0;

    template <uint32_t args_offset>
    static FORCE_INLINE auto state_backing() {
        constexpr auto state_args = TensorAccessorArgs<args_offset>();
        return TensorAccessor(state_args, get_common_arg_val<uint32_t>(0));
    }

    template <uint32_t Sq_chunk_t, uint32_t vDHt, typename Accessor>
    static FORCE_INLINE void transfer_state(Noc& noc, const Accessor& backing) {
        if constexpr (kStreamState) {
            stream_recipe_state<Sq_chunk_t, 17, 18, vDHt>(noc, backing);
        } else {
            transfer_recipe_state<kFp32State, Sq_chunk_t, 17, 18, vDHt>(noc, backing);
        }
    }
};

#include "ring_joint_writer_impl.hpp"

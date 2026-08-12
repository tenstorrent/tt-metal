// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "roll_device_operation_types.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::prim {

// Native single-dim roll for ROW_MAJOR sharded tensors (HEIGHT / WIDTH / BLOCK).
// Implemented as a per-core gather of segment copies over the sharded buffers.
struct RollShardedProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const RollParams& operation_attributes, const RollInputs& tensor_args, Tensor& tensor_return_value);
};

// The DRAM row-major reader stages sources into 2 CBs (`src_base[2]`). Higher-dim rolls whose
// shard band straddles an outer-dim period can need 3+ sources on one dst core — caller must
// route those away (e.g. interleaved round-trip). Pure function of shape / shard / shift / dim.
bool dram_rm_roll_needs_extra_source_shards(const Tensor& input, uint32_t shift, int32_t dim);

}  // namespace ttnn::prim

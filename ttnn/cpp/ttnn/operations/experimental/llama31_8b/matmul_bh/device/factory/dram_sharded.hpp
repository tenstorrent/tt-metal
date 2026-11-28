// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"
#include "ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.hpp"

namespace ttnn::prim::llama31_8b_matmul_bh::dram_sharded {

struct DramShardedReadOptions {
    bool uneven_readers = false;
    bool streamed_write_back = false;
};

struct DsPlanReader {
    uint16_t column_offset_tiles, row_read_tiles;
};

struct DramShardedPlan {
    uint32_t Nt, Kt, in0_block_w, workers_per_bank, num_readers, in1_shard_width_tiles, in0_storage_cores;
    uint32_t per_core_N_in1_sender, per_core_N_compute, out_subblock_h, out_subblock_w, last_subblock_w_valid;
    uint32_t in1_num_subblocks, in0_num_subblocks, num_blocks, per_core_N_storage, num_cores_written_back;
    const DsPlanReader* readers;
    const uint8_t* in0_sender_slots;
};

ttnn::device_operation::ProgramArtifacts create_program_artifacts(
    const ttnn::prim::MatmulParams& operation_attributes,
    const ttnn::prim::MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value,
    const DramShardedReadOptions& read_options,
    const DramShardedPlan& plan);

}  // namespace ttnn::prim::llama31_8b_matmul_bh::dram_sharded

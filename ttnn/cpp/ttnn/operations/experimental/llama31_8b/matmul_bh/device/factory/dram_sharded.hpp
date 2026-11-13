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

ttnn::device_operation::ProgramArtifacts create_program_artifacts(
    const ttnn::prim::MatmulParams& operation_attributes,
    const ttnn::prim::MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value,
    const DramShardedReadOptions& read_options);

}  // namespace ttnn::prim::llama31_8b_matmul_bh::dram_sharded

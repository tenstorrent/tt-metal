// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "shared_expert_ffn.hpp"

#include <tt-metalium/core_coord.hpp>

#include "ttnn/operations/ccl/reduce_scatter/reduce_scatter.hpp"
#include "ttnn/operations/core/to_memory_config/to_memory_config_op.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/matmul/matmul.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::shared_expert_ffn {

ttnn::Tensor shared_expert_ffn(
    const ttnn::Tensor& x,
    const ttnn::Tensor& gate_proj,
    const ttnn::Tensor& up_proj,
    const ttnn::Tensor& down_proj,
    uint32_t cluster_axis,
    uint32_t tp_axis_size,
    uint32_t num_links,
    tt::tt_fabric::Topology topology,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id) {
    auto gate_out = ttnn::matmul(
        /*input_tensor_a=*/x,
        /*input_tensor_b=*/gate_proj,
        /*transpose_a=*/false,
        /*transpose_b=*/false,
        /*memory_config=*/std::nullopt,
        /*dtype=*/std::nullopt,
        /*program_config=*/std::nullopt,
        /*activation=*/std::string("silu"),
        /*compute_kernel_config=*/compute_kernel_config,
        /*core_grid=*/std::nullopt,
        /*output_tile=*/std::nullopt,
        /*optional_output_tensor=*/std::nullopt,
        /*global_cb=*/std::nullopt,
        /*sub_device_id=*/subdevice_id);

    auto up_out = ttnn::matmul(
        /*input_tensor_a=*/x,
        /*input_tensor_b=*/up_proj,
        /*transpose_a=*/false,
        /*transpose_b=*/false,
        /*memory_config=*/std::nullopt,
        /*dtype=*/std::nullopt,
        /*program_config=*/std::nullopt,
        /*activation=*/std::nullopt,
        /*compute_kernel_config=*/compute_kernel_config,
        /*core_grid=*/std::nullopt,
        /*output_tile=*/std::nullopt,
        /*optional_output_tensor=*/std::nullopt,
        /*global_cb=*/std::nullopt,
        /*sub_device_id=*/subdevice_id);

    // When a sub-device is specified, shard gate_out/up_out onto cores inside
    // that sub-device before the elementwise multiply. Without this the binary
    // op's worker grid (binary_device_operation.cpp:459-465) is the union of
    // every loaded sub-device's cores, which puts the multiply on the dispatch
    // sub-device's cores too and serializes the whole pipeline. With sharded
    // inputs the binary op's sharded branch picks only the sub-devices whose
    // worker cores intersect the shard grid.
    if (subdevice_id.has_value()) {
        constexpr uint32_t TILE = 32;
        auto* device = x.device();
        auto sub_device_cores = device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, *subdevice_id);

        const auto& padded = gate_out.padded_shape();
        uint32_t total_h = 1;
        for (size_t i = 0; i + 1 < padded.size(); ++i) {
            total_h *= padded[i];
        }
        uint32_t total_w = padded[padded.size() - 1];
        uint32_t total_h_tiles = (total_h + TILE - 1) / TILE;
        uint32_t avail = sub_device_cores.num_cores();
        uint32_t num_cores = std::min(total_h_tiles, avail);
        while (num_cores > 1 && total_h_tiles % num_cores != 0) {
            --num_cores;
        }
        uint32_t shard_h = (total_h_tiles / num_cores) * TILE;

        auto shard_grid = tt::tt_metal::select_from_corerangeset(sub_device_cores, 0, num_cores - 1, /*row_wise=*/true);
        auto shard_spec =
            tt::tt_metal::ShardSpec(shard_grid, {shard_h, total_w}, tt::tt_metal::ShardOrientation::ROW_MAJOR);
        auto sharded_cfg = tt::tt_metal::MemoryConfig{
            tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED, tt::tt_metal::BufferType::L1, shard_spec};

        gate_out = ttnn::to_memory_config(gate_out, sharded_cfg);
        up_out = ttnn::to_memory_config(up_out, sharded_cfg);
    }

    ttnn::multiply_(/*lhs=*/gate_out, /*rhs=*/up_out);
    up_out.deallocate();

    if (subdevice_id.has_value()) {
        gate_out = ttnn::to_memory_config(
            gate_out,
            tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM});
    }

    auto full_out = ttnn::matmul(
        /*input_tensor_a=*/gate_out,
        /*input_tensor_b=*/down_proj,
        /*transpose_a=*/false,
        /*transpose_b=*/false,
        /*memory_config=*/std::nullopt,
        /*dtype=*/std::nullopt,
        /*program_config=*/std::nullopt,
        /*activation=*/std::nullopt,
        /*compute_kernel_config=*/compute_kernel_config,
        /*core_grid=*/std::nullopt,
        /*output_tile=*/std::nullopt,
        /*optional_output_tensor=*/std::nullopt,
        /*global_cb=*/std::nullopt,
        /*sub_device_id=*/subdevice_id);

    if (tp_axis_size <= 1) {
        return full_out;
    }

    return ttnn::reduce_scatter(
        /*input_tensor=*/full_out,
        /*dim=*/-1,
        /*cluster_axis=*/cluster_axis,
        /*subdevice_id=*/subdevice_id,
        /*memory_config=*/std::nullopt,
        /*intermediate_memory_config=*/std::nullopt,
        /*optional_output_tensor=*/std::nullopt,
        /*num_links=*/num_links,
        /*topology=*/topology);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::shared_expert_ffn

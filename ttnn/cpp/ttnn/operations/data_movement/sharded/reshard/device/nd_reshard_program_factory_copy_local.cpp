// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/sharded/reshard/device/nd_reshard_program_factory_copy_local.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {

template <bool local_is_input>
ttnn::device_operation::ProgramArtifacts NdReshardCopyLocalShardFactory<local_is_input>::create_program_spec(
    const ReshardParams& /*operation_attributes*/, const ReshardInputs& tensor_args, Tensor& output_tensor) {
    const auto& input = tensor_args.input;
    auto& output = output_tensor;

    auto* input_buffer = input.buffer();
    auto* output_buffer = output.buffer();

    // Choose buffer and aligned page size based on local_is_input flag
    auto* local_buffer = local_is_input ? input_buffer : output_buffer;
    auto aligned_page_size = local_buffer->aligned_page_size();
    auto other_aligned_page_size =
        local_is_input ? output_buffer->aligned_page_size() : input_buffer->aligned_page_size();

    // This implementation assumes that input and output grids are the same.
    auto cores_vec = local_buffer->buffer_distribution_spec()->cores_with_data();
    auto grid = CoreRangeSet(cores_vec);

    uint32_t num_shards = static_cast<uint32_t>(local_buffer->buffer_distribution_spec()->num_shards());

    // num cores with data * 2 because we have two kernels
    uint32_t shard_id_stride =
        static_cast<uint32_t>(local_buffer->buffer_distribution_spec()->num_cores_with_data()) * 2u;

    // Prepare compile time arguments
    auto logical_size = input.logical_shape();
    uint32_t logical_width = static_cast<uint32_t>(logical_size[-1] * input.element_size());
    uint32_t source_width = logical_width;
    uint32_t destination_width = logical_width;
    uint32_t base_page_size = aligned_page_size;

    if (input.memory_config().shard_spec().has_value() && output.memory_config().shard_spec().has_value()) {
        auto input_buffer_type = input.memory_config().memory_layout();
        auto output_buffer_type = output.memory_config().memory_layout();

        // for block sharded
        CoreCoord input_shard_grid = input_buffer->shard_spec().grid().ranges()[0].grid_size();
        uint32_t input_num_shard_cores = input_shard_grid.x;
        if (input_buffer->shard_spec().orientation() == ShardOrientation::COL_MAJOR) {
            input_num_shard_cores = input_shard_grid.y;
        }

        CoreCoord output_shard_grid = output_buffer->shard_spec().grid().ranges()[0].grid_size();
        uint32_t output_num_shard_cores = output_shard_grid.x;
        if (output_buffer->shard_spec().orientation() == ShardOrientation::COL_MAJOR) {
            output_num_shard_cores = output_shard_grid.y;
        }
        // for width sharded
        if (input_buffer_type == TensorMemoryLayout::WIDTH_SHARDED &&
            output_buffer_type == TensorMemoryLayout::WIDTH_SHARDED) {
            input_num_shard_cores = input_shard_grid.x == 1 ? input_shard_grid.y : input_shard_grid.x;
            output_num_shard_cores = output_shard_grid.x == 1 ? output_shard_grid.y : output_shard_grid.x;
        }

        source_width =
            static_cast<uint32_t>(input_buffer->shard_spec().shape()[1] * input.element_size() * input_num_shard_cores);
        destination_width = static_cast<uint32_t>(
            output_buffer->shard_spec().shape()[1] * output.element_size() * output_num_shard_cores);
        uint32_t input_page_size = input_buffer->page_size();
        uint32_t output_page_size = output_buffer->page_size();
        base_page_size = std::gcd(input_page_size, output_page_size);
    }
    // Named scalar CTAs (shared by both kernels). The legacy positional
    // TensorAccessorArgs CTAs for src/dst are now carried by the TensorBindings.
    const uint32_t src_page_size = static_cast<uint32_t>(aligned_page_size);
    const uint32_t dst_page_size = static_cast<uint32_t>(other_aligned_page_size);

    const TensorParamName kInput{"input"};
    const TensorParamName kOutput{"output"};
    const KernelSpecName kBrisc{"brisc"};
    const KernelSpecName kNcrisc{"ncrisc"};

    const std::filesystem::path kKernelSource(
        "ttnn/cpp/ttnn/operations/data_movement/sharded/reshard/device/kernels/nd_reshard_copy_local_shards.cpp");

    ProgramSpec spec;
    spec.name = "nd_reshard_copy_local_shards";

    // Tensor parameters: both kernels access input (src) and output (dst).
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = kInput, .spec = input.tensor_spec()});
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = kOutput, .spec = output.tensor_spec()});

    // Both kernels share identical CTAs and CRTA schema; they differ only in the
    // DataMovement processor/role and their per-core first_shard_id RTA value.
    // No DFB: data moves local-L1 <-> remote-bank directly via the TensorAccessors
    // (and CoreLocalMem derived from the accessor's own page noc_addr).
    auto make_kernel = [&](const KernelSpecName& unique_id, DataMovementRoleHint role) {
        KernelSpec k;
        k.unique_id = unique_id;
        k.source = kKernelSource;
        k.compile_time_args = {
            {"src_page_size", src_page_size},
            {"dst_page_size", dst_page_size},
            {"is_reader", static_cast<uint32_t>(local_is_input)},
            {"logical_width", logical_width},
            {"src_width", source_width},
            {"dst_width", destination_width},
            {"transfer_size", base_page_size},
        };
        k.runtime_arg_schema.runtime_arg_names = {"first_shard_id"};
        k.runtime_arg_schema.common_runtime_arg_names = {"num_shards", "shard_id_stride"};
        k.tensor_bindings = {
            TensorBinding{.tensor_parameter_name = kInput, .accessor_name = "input"},
            TensorBinding{.tensor_parameter_name = kOutput, .accessor_name = "output"},
        };
        k.hw_config = DataMovementHardwareConfig{.role = role};
        return k;
    };

    spec.kernels.push_back(make_kernel(kBrisc, DataMovementRoleHint::READER));
    spec.kernels.push_back(make_kernel(kNcrisc, DataMovementRoleHint::WRITER));

    // Single WorkUnitSpec hosting both kernels on the shard-data grid.
    spec.work_units.push_back(WorkUnitSpec{
        .name = "wu",
        .kernels = {kBrisc, kNcrisc},
        .target_nodes = grid,
    });

    // Run args.
    ProgramRunArgs run_args;

    KernelRunArgs brisc_run;
    brisc_run.kernel = kBrisc;
    brisc_run.common_runtime_arg_values = {{"num_shards", num_shards}, {"shard_id_stride", shard_id_stride}};
    KernelRunArgs ncrisc_run;
    ncrisc_run.kernel = kNcrisc;
    ncrisc_run.common_runtime_arg_values = {{"num_shards", num_shards}, {"shard_id_stride", shard_id_stride}};

    // Per-core unique runtime args: [first_shard_id]
    // brisc copies shards [0, num_data_cores*2, num_data_cores*4, num_data_cores*6, ...]
    // ncrisc copies shards [num_data_cores, num_data_cores*3, num_data_cores*5, num_data_cores*7, ...]
    uint32_t start_shard_id = 0;
    for (const auto& core : cores_vec) {
        brisc_run.runtime_arg_values.push_back(
            KernelRunArgs::NodeRuntimeArgs{.node = core, .args = {{"first_shard_id", start_shard_id}}});
        ncrisc_run.runtime_arg_values.push_back(KernelRunArgs::NodeRuntimeArgs{
            .node = core, .args = {{"first_shard_id", start_shard_id + shard_id_stride / 2}}});
        ++start_shard_id;
    }

    run_args.kernel_run_args.push_back(std::move(brisc_run));
    run_args.kernel_run_args.push_back(std::move(ncrisc_run));

    run_args.tensor_args = {
        {kInput, input.mesh_tensor()},
        {kOutput, output.mesh_tensor()},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

// Explicit template instantiations
template struct NdReshardCopyLocalShardFactory<true>;
template struct NdReshardCopyLocalShardFactory<false>;

}  // namespace ttnn::prim

// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/quasar/reshard/device/nd_reshard_program_factory_copy_local.hpp"

#include <filesystem>
#include <numeric>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim::qsr {

namespace {

// Names for the copy-local-shard factory (role-prefixed for unity-build hygiene).
const TensorParamName COPY_LOCAL_INPUT{"copy_local_input"};
const TensorParamName COPY_LOCAL_OUTPUT{"copy_local_output"};
const KernelSpecName COPY_LOCAL_BRISC{"copy_local_brisc"};
const KernelSpecName COPY_LOCAL_NCRISC{"copy_local_ncrisc"};

}  // namespace

template <bool local_is_input>
ttnn::device_operation::ProgramArtifacts NdReshardCopyLocalShardFactory<local_is_input>::create_program_artifacts(
    const ReshardParams& /*operation_attributes*/, const ReshardInputs& tensor_args, Tensor& output_tensor) {
    const auto& input = tensor_args.input;
    auto& output = output_tensor;

    const auto& input_mt = input.mesh_tensor();
    const auto& output_mt = output.mesh_tensor();

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

    // Tensor parameters replace the legacy TensorAccessorArgs CTA plumbing and the
    // input/output Buffer-base CRTA bindings. Both kernels (brisc + ncrisc) bind both.
    Group<TensorParameter> tensor_parameters = {
        TensorParameter{.unique_id = COPY_LOCAL_INPUT, .spec = input_mt.tensor_spec()},
        TensorParameter{.unique_id = COPY_LOCAL_OUTPUT, .spec = output_mt.tensor_spec()},
    };

    // Shared CTAs for both kernels.
    const KernelSpec::CompileTimeArgs compile_time_args = {
        {"src_page_size", aligned_page_size},
        {"dst_page_size", other_aligned_page_size},
        {"is_reader", static_cast<uint32_t>(local_is_input)},
        {"logical_width", logical_width},
        {"src_width", source_width},
        {"dst_width", destination_width},
        {"transfer_size", base_page_size},
    };

    const std::filesystem::path kernel_source(
        "ttnn/cpp/ttnn/operations/experimental/quasar/reshard/device/kernels/nd_reshard_copy_local_shards.cpp");

    auto make_tensor_bindings = []() {
        return Group<TensorBinding>{
            TensorBinding{.tensor_parameter_name = COPY_LOCAL_INPUT, .accessor_name = "src"},
            TensorBinding{.tensor_parameter_name = COPY_LOCAL_OUTPUT, .accessor_name = "dst"},
        };
    };

    // Preserve the legacy explicit RISCV_0 / NOC RISCV_0_default placement.
    DataMovementHardwareConfig brisc_hw;
    if (input.device()->arch() == tt::ARCH::QUASAR) {
        brisc_hw = DataMovementGen2Config{};
    } else {
        brisc_hw = DataMovementGen1Config{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
        };
    }
    KernelSpec brisc{
        .unique_id = COPY_LOCAL_BRISC,
        .source = kernel_source,
        .tensor_bindings = make_tensor_bindings(),
        .compile_time_args = compile_time_args,
        .runtime_arg_schema =
            {.runtime_arg_names = {"first_shard_id"}, .common_runtime_arg_names = {"num_shards", "shard_id_stride"}},
        .hw_config = std::move(brisc_hw),
    };

    // Preserve the legacy explicit RISCV_1 / NOC RISCV_1_default placement.
    DataMovementHardwareConfig ncrisc_hw;
    if (input.device()->arch() == tt::ARCH::QUASAR) {
        ncrisc_hw = DataMovementGen2Config{};
    } else {
        ncrisc_hw = DataMovementGen1Config{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
        };
    }
    KernelSpec ncrisc{
        .unique_id = COPY_LOCAL_NCRISC,
        .source = kernel_source,
        .tensor_bindings = make_tensor_bindings(),
        .compile_time_args = compile_time_args,
        .runtime_arg_schema =
            {.runtime_arg_names = {"first_shard_id"}, .common_runtime_arg_names = {"num_shards", "shard_id_stride"}},
        .hw_config = std::move(ncrisc_hw),
    };

    // Common runtime args (broadcast to all nodes).
    ProgramRunArgs::KernelRunArgs brisc_run_args{.kernel = COPY_LOCAL_BRISC};
    ProgramRunArgs::KernelRunArgs ncrisc_run_args{.kernel = COPY_LOCAL_NCRISC};
    brisc_run_args.common_runtime_arg_values = {{"num_shards", num_shards}, {"shard_id_stride", shard_id_stride}};
    ncrisc_run_args.common_runtime_arg_values = {{"num_shards", num_shards}, {"shard_id_stride", shard_id_stride}};

    // Per-core unique runtime args: [first_shard_id]
    // brisc copies shards [0, num_data_cores*2, num_data_cores*4, ...]
    // ncrisc copies shards [num_data_cores, num_data_cores*3, num_data_cores*5, ...]
    uint32_t start_shard_id = 0;
    for (const auto& core : cores_vec) {
        brisc_run_args.runtime_arg_values["first_shard_id"][core] = start_shard_id;
        ncrisc_run_args.runtime_arg_values["first_shard_id"][core] = start_shard_id + shard_id_stride / 2;
        ++start_shard_id;
    }

    ProgramSpec spec{
        .name = "nd_reshard_copy_local_shards",
        .kernels = {std::move(brisc), std::move(ncrisc)},
        .tensor_parameters = std::move(tensor_parameters),
        .work_units =
            {
                WorkUnitSpec{.name = "wu", .kernels = {COPY_LOCAL_BRISC, COPY_LOCAL_NCRISC}, .target_nodes = grid},
            },
    };

    ProgramRunArgs run_args{
        .kernel_run_args = {std::move(brisc_run_args), std::move(ncrisc_run_args)},
        .tensor_args =
            {
                {COPY_LOCAL_INPUT, input_mt},
                {COPY_LOCAL_OUTPUT, output_mt},
            },
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

// Explicit template instantiations
template struct NdReshardCopyLocalShardFactory<true>;
template struct NdReshardCopyLocalShardFactory<false>;

}  // namespace ttnn::prim::qsr

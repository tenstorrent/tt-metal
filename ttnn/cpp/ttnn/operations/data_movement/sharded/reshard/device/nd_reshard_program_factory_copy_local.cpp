// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/sharded/reshard/device/nd_reshard_program_factory_copy_local.hpp"

#include <numeric>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {

namespace {

constexpr const char* kKernelPath =
    "ttnn/cpp/ttnn/operations/data_movement/sharded/reshard/device/kernels/nd_reshard_copy_local_shards.cpp";

// Resource / parameter names referenced by the kernel source (ta::/args:: accessors).
constexpr const char* kSrcTensorParam = "src";
constexpr const char* kDstTensorParam = "dst";

}  // namespace

template <bool local_is_input>
ttnn::device_operation::ProgramArtifacts NdReshardCopyLocalShardFactory<local_is_input>::create_program_artifacts(
    const ReshardParams& /*operation_attributes*/, const ReshardInputs& tensor_args, Tensor& output_tensor) {
    const auto& input = tensor_args.input;
    auto& output = output_tensor;

    auto* input_buffer = input.buffer();
    auto* output_buffer = output.buffer();

    // Choose buffer and aligned page size based on local_is_input flag.
    auto* local_buffer = local_is_input ? input_buffer : output_buffer;
    const uint32_t aligned_page_size = local_buffer->aligned_page_size();
    const uint32_t other_aligned_page_size =
        local_is_input ? output_buffer->aligned_page_size() : input_buffer->aligned_page_size();

    // This implementation assumes that input and output grids are the same.
    const auto cores_vec = local_buffer->buffer_distribution_spec()->cores_with_data();
    const CoreRangeSet grid(cores_vec);

    const uint32_t num_shards = static_cast<uint32_t>(local_buffer->buffer_distribution_spec()->num_shards());

    // num cores with data * 2 because we have two kernels (brisc + ncrisc) splitting the shards.
    const uint32_t shard_id_stride =
        static_cast<uint32_t>(local_buffer->buffer_distribution_spec()->num_cores_with_data()) * 2u;

    // Compile-time argument values (computation mirrors the legacy ProgramDescriptor factory).
    auto logical_size = input.logical_shape();
    uint32_t logical_width = static_cast<uint32_t>(logical_size[-1] * input.element_size());
    uint32_t source_width = logical_width;
    uint32_t destination_width = logical_width;
    uint32_t base_page_size = aligned_page_size;

    if (input.memory_config().shard_spec().has_value() && output.memory_config().shard_spec().has_value()) {
        auto input_memory_layout = input.memory_config().memory_layout();
        auto output_memory_layout = output.memory_config().memory_layout();

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
        if (input_memory_layout == TensorMemoryLayout::WIDTH_SHARDED &&
            output_memory_layout == TensorMemoryLayout::WIDTH_SHARDED) {
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

    // Compile-time args are identical for both kernels. Names match the kernel's get_arg(args::...) tokens.
    // (Order of positional CTAs in the legacy factory is preserved as named values here.)
    const KernelSpec::CompileTimeArgs compile_time_args = {
        {"src_page_size", aligned_page_size},
        {"dst_page_size", other_aligned_page_size},
        {"is_reader", static_cast<uint32_t>(local_is_input)},
        {"logical_width", logical_width},
        {"src_width", source_width},
        {"dst_width", destination_width},
        {"transfer_size", base_page_size},
    };

    // ------------------------------------------------------------------
    // ProgramSpec (immutable)
    // ------------------------------------------------------------------
    ProgramSpec spec;
    spec.name = "nd_reshard_copy_local_shards";

    // Two data-movement workers running the same kernel source on every core with data; they
    // split the shard set across the two RISCs. The role hints select RISCV_0/NOC0 ("brisc") and
    // RISCV_1/NOC1 ("ncrisc") respectively (matching the legacy factory's processor/NOC assignment);
    // they carry no read/write-direction semantics here (the direction is the is_reader CTA).
    const auto make_worker = [&](const char* name, DataMovementRoleHint role) {
        KernelSpec k{
            .unique_id = KernelSpecName{name},
            .source = std::filesystem::path(kKernelPath),
            .hw_config = DataMovementHardwareConfig{.role = role},
        };
        k.compile_time_args = compile_time_args;
        k.tensor_bindings.push_back(
            TensorBinding{.tensor_parameter_name = TensorParamName{kSrcTensorParam}, .accessor_name = kSrcTensorParam});
        k.tensor_bindings.push_back(
            TensorBinding{.tensor_parameter_name = TensorParamName{kDstTensorParam}, .accessor_name = kDstTensorParam});
        k.runtime_arg_schema.runtime_arg_names = {"first_shard_id"};
        k.runtime_arg_schema.common_runtime_arg_names = {"num_shards", "shard_id_stride"};
        return k;
    };

    KernelSpec brisc = make_worker("brisc", DataMovementRoleHint::READER);
    KernelSpec ncrisc = make_worker("ncrisc", DataMovementRoleHint::WRITER);

    spec.kernels = {brisc, ncrisc};
    spec.tensor_parameters = {
        TensorParameter{.unique_id = TensorParamName{kSrcTensorParam}, .spec = input.tensor_spec()},
        TensorParameter{.unique_id = TensorParamName{kDstTensorParam}, .spec = output.tensor_spec()},
    };
    spec.work_units = {WorkUnitSpec{
        .name = "reshard_local_shards_work_unit",
        .kernels = {KernelSpecName{"brisc"}, KernelSpecName{"ncrisc"}},
        .target_nodes = grid,
    }};

    // ------------------------------------------------------------------
    // ProgramRunArgs (mutable)
    // ------------------------------------------------------------------
    KernelRunArgs brisc_run_args{.kernel = KernelSpecName{"brisc"}};
    KernelRunArgs ncrisc_run_args{.kernel = KernelSpecName{"ncrisc"}};
    brisc_run_args.common_runtime_arg_values = {{"num_shards", num_shards}, {"shard_id_stride", shard_id_stride}};
    ncrisc_run_args.common_runtime_arg_values = {{"num_shards", num_shards}, {"shard_id_stride", shard_id_stride}};

    // Per-core first shard id:
    //   brisc  copies shards [i, i + stride, i + 2*stride, ...]
    //   ncrisc copies shards [i + stride/2, i + stride/2 + stride, ...]
    // where i is the core's index in cores_vec, stride == shard_id_stride.
    uint32_t start_shard_id = 0;
    for (const auto& core : cores_vec) {
        const NodeCoord node{core.x, core.y};
        brisc_run_args.runtime_arg_values.push_back({node, {{"first_shard_id", start_shard_id}}});
        ncrisc_run_args.runtime_arg_values.push_back(
            {node, {{"first_shard_id", start_shard_id + shard_id_stride / 2}}});
        ++start_shard_id;
    }

    ProgramRunArgs run_params;
    run_params.kernel_run_args = {std::move(brisc_run_args), std::move(ncrisc_run_args)};
    run_params.tensor_args = {
        {TensorParamName{kSrcTensorParam}, TensorArgument{input.mesh_tensor()}},
        {TensorParamName{kDstTensorParam}, TensorArgument{output.mesh_tensor()}},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_params),
    };
}

// Explicit template instantiations
template struct NdReshardCopyLocalShardFactory<true>;
template struct NdReshardCopyLocalShardFactory<false>;

}  // namespace ttnn::prim

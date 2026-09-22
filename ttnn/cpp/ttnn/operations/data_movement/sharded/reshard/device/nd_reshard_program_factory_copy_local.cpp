// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/sharded/reshard/device/nd_reshard_program_factory_copy_local.hpp"

#include <filesystem>
#include <numeric>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "tt-metalium/host_api.hpp"

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {

namespace {

// (Names are prefixed to avoid unity-build collisions with the sibling reshard factories.)
constexpr const char* kCLKernelPath =
    "ttnn/cpp/ttnn/operations/data_movement/sharded/reshard/device/kernels/nd_reshard_copy_local_shards.cpp";

// Resource / parameter names referenced by the kernel source (tensor:: accessors).
constexpr const char* kCLInputTensorParam = "input";
constexpr const char* kCLOutputTensorParam = "output";

constexpr const char* kCLBriscKernel = "brisc";
constexpr const char* kCLNcriscKernel = "ncrisc";

}  // namespace

template <bool local_is_input>
ttnn::device_operation::ProgramArtifacts NdReshardCopyLocalShardFactory<local_is_input>::create_program_artifacts(
    const ReshardParams& /*operation_attributes*/, const ReshardInputs& tensor_args, Tensor& output_tensor) {
    const auto& input = tensor_args.input;
    auto& output = output_tensor;

    auto* input_buffer = input.buffer();
    auto* output_buffer = output.buffer();

    // Choose buffer and aligned page size based on local_is_input flag
    auto* local_buffer = local_is_input ? input_buffer : output_buffer;
    const uint32_t aligned_page_size = static_cast<uint32_t>(local_buffer->aligned_page_size());
    const uint32_t other_aligned_page_size =
        static_cast<uint32_t>(local_is_input ? output_buffer->aligned_page_size() : input_buffer->aligned_page_size());

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

    // ------------------------------------------------------------------
    // ProgramSpec (immutable)
    // ------------------------------------------------------------------
    ProgramSpec spec;
    spec.name = "nd_reshard_copy_local_shards";

    const KernelSpec::CompileTimeArgs compile_time_args = {
        {"src_page_size", aligned_page_size},
        {"dst_page_size", other_aligned_page_size},
        {"is_reader", static_cast<uint32_t>(local_is_input)},
        {"logical_width", logical_width},
        {"src_width", source_width},
        {"dst_width", destination_width},
        {"transfer_size", base_page_size},
        // TILE reshards are a pure page permutation (equal src/dst page size); the kernel copies
        // page N -> page N and skips the row-major row/col re-strider.
        {"page_to_page", static_cast<uint32_t>(input.layout() == Layout::TILE)},
    };

    const Group<TensorBinding> tensor_bindings = {
        TensorBinding{
            .tensor_parameter_name = TensorParamName{kCLInputTensorParam}, .accessor_name = kCLInputTensorParam},
        TensorBinding{
            .tensor_parameter_name = TensorParamName{kCLOutputTensorParam}, .accessor_name = kCLOutputTensorParam},
    };

    // Both kernel instances run the same source with the same specialization; only the
    // (processor, noc) placement and the per-node start shard id differ. The legacy config was a
    // custom DataMovementConfigDescriptor triple (RISCV_0/NOC_0 and RISCV_1/NOC_1), matching
    // neither the reader nor the writer default, so both are replicated field-for-field here.
    const auto make_worker = [&](const char* name, DataMovementProcessor processor, NOC noc) {
        // Gen2 has no (processor, noc, noc_mode) placement concept, so on Quasar the custom Gen1
        // placement is replaced by a default-constructed Gen2 config (matching what the
        // arch-agnostic reader/writer helpers do).
        DataMovementHardwareConfig hw_config =
            DataMovementGen1Config{.processor = processor, .noc = noc, .noc_mode = NOC_MODE::DM_DEDICATED_NOC};
        if (input.device()->arch() == tt::ARCH::QUASAR) {
            hw_config = DataMovementGen2Config{};
        }
        return KernelSpec{
            .unique_id = KernelSpecName{name},
            .source = std::filesystem::path(kCLKernelPath),
            .tensor_bindings = tensor_bindings,
            .compile_time_args = compile_time_args,
            .runtime_arg_schema =
                {.runtime_arg_names = {"first_shard_id"},
                 .common_runtime_arg_names = {"num_shards", "shard_id_stride"}},
            .hw_config = std::move(hw_config),
        };
    };

    spec.kernels = {
        make_worker(kCLBriscKernel, DataMovementProcessor::RISCV_0, NOC::RISCV_0_default),
        make_worker(kCLNcriscKernel, DataMovementProcessor::RISCV_1, NOC::RISCV_1_default),
    };

    spec.tensor_parameters = {
        TensorParameter{.unique_id = TensorParamName{kCLInputTensorParam}, .spec = input.tensor_spec()},
        TensorParameter{.unique_id = TensorParamName{kCLOutputTensorParam}, .spec = output.tensor_spec()},
    };

    spec.work_units = {WorkUnitSpec{
        .name = "nd_reshard_copy_local_shards_work_unit",
        .kernels = {KernelSpecName{kCLBriscKernel}, KernelSpecName{kCLNcriscKernel}},
        .target_nodes = grid,
    }};

    // ------------------------------------------------------------------
    // ProgramRunArgs (mutable)
    // ------------------------------------------------------------------
    KernelRunArgs brisc_run_args{
        .kernel = KernelSpecName{kCLBriscKernel},
        .common_runtime_arg_values = {{"num_shards", num_shards}, {"shard_id_stride", shard_id_stride}}};
    KernelRunArgs ncrisc_run_args{
        .kernel = KernelSpecName{kCLNcriscKernel},
        .common_runtime_arg_values = {{"num_shards", num_shards}, {"shard_id_stride", shard_id_stride}}};

    // Per-core unique runtime args: [first_shard_id]
    // brisc copies shards [0, num_data_cores*2, num_data_cores*4, num_data_cores*6, ...]
    // ncrisc copies shards [num_data_cores, num_data_cores*3, num_data_cores*5, num_data_cores*7, ...]
    uint32_t start_shard_id = 0;
    for (const auto& core : cores_vec) {
        AddRuntimeArgsForNode(brisc_run_args.runtime_arg_values, core, {{"first_shard_id", start_shard_id}});
        AddRuntimeArgsForNode(
            ncrisc_run_args.runtime_arg_values, core, {{"first_shard_id", start_shard_id + shard_id_stride / 2}});
        ++start_shard_id;
    }

    ProgramRunArgs run_params;
    run_params.kernel_run_args = {std::move(brisc_run_args), std::move(ncrisc_run_args)};
    run_params.tensor_args = {
        {TensorParamName{kCLInputTensorParam}, TensorArgument{input.mesh_tensor()}},
        {TensorParamName{kCLOutputTensorParam}, TensorArgument{output.mesh_tensor()}},
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

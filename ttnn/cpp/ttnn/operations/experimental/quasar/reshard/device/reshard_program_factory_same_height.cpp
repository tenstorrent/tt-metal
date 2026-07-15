// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/quasar/reshard/device/reshard_program_factory_same_height.hpp"
#include "ttnn/operations/data_movement/sharded/sharded_common.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

#include <algorithm>
#include <filesystem>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim::qsr {

namespace {

// Reader source reads remote -> local; writer source writes local -> remote.
// (Names are prefixed to avoid Unity-build collisions with the sibling reshard factories.)
constexpr const char* kSHReaderKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/quasar/reshard/device/kernels/dataflow/reshard_same_height_reader.cpp";
constexpr const char* kSHWriterKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/quasar/reshard/device/kernels/dataflow/reshard_same_height_writer.cpp";

// Resource / parameter names referenced by the kernel sources (tensor::/dfb:: accessors).
constexpr const char* kSHRemoteTensorParam = "remote";
constexpr const char* kSHLocalTensorParam = "local_shard";
constexpr const char* kSHDfbName = "shard_cb";

}  // namespace

template <bool local_is_output>
ttnn::device_operation::ProgramArtifacts ReshardSameHeightFactory<local_is_output>::create_program_artifacts(
    const ReshardParams& /*operation_attributes*/, const ReshardInputs& tensor_args, Tensor& output_tensor) {
    const auto& input = tensor_args.input;
    auto& output = output_tensor;
    const auto& local_tensor = local_is_output ? output : input;
    const auto& remote_tensor = local_is_output ? input : output;
    const auto local_shard_spec = local_tensor.shard_spec().value();
    const auto remote_shard_spec = remote_tensor.shard_spec().value();

    auto* device = input.device();

    const auto remote_core_type = remote_tensor.buffer()->core_type();
    const bool interface_with_dram = (remote_core_type == tt::CoreType::DRAM);
    auto local_cores = get_optimal_worker_cores_for_sharded_tensor(local_tensor);
    auto all_cores = CoreRangeSet(ttsl::Span<const CoreCoord>(local_cores));
    auto remote_cores = remote_tensor.buffer()->buffer_distribution_spec().value().cores_with_data();

    const auto data_format = tt::tt_metal::datatype_to_dataformat_converter(local_tensor.dtype());
    const uint32_t element_size = tt::datum_size(data_format);

    TT_FATAL(local_tensor.layout() == Layout::ROW_MAJOR, "Expected row major tensor");
    const uint32_t unit_size =
        static_cast<uint32_t>(local_shard_spec.shape[1] * local_tensor.element_size());  // width * element size
    const uint32_t remote_units_per_shard = remote_shard_spec.shape[0];                  // height

    auto* remote_buffer = remote_tensor.buffer();
    auto remote_buffer_type = remote_buffer->buffer_type();

    // Generate all read/write offsets for each core
    auto [runtime_args_for_each_core, total_num_sticks, local_stride_bytes, remote_stride_bytes] =
        ttnn::operations::data_movement::detail::compute_width_sharding_reshard_segments(
            local_shard_spec.shape,
            remote_shard_spec.shape,
            local_cores,
            remote_cores,
            remote_buffer_type,
            remote_core_type,
            device,
            element_size);  // local_core_idx -> runtime args[]

    // Split work across each kernel along tensor height since this is the best way to split work evenly
    const uint32_t total_num_sticks_kernel_0 = total_num_sticks / 2;
    const uint32_t total_num_sticks_kernel_1 = total_num_sticks - total_num_sticks_kernel_0;

    // Varargs carry the per-segment tail (4 values/segment). The count varies per core, so we set a
    // uniform schema count of the max-over-cores and pad shorter cores; the kernel only reads the
    // real prefix (it loops over the num_segments named RTA).
    uint32_t max_num_segments = 0;
    for (const auto& args_for_all_segments : runtime_args_for_each_core) {
        max_num_segments = std::max(max_num_segments, static_cast<uint32_t>(args_for_all_segments.size()));
    }
    const uint32_t num_varargs = max_num_segments * 4;

    // ------------------------------------------------------------------
    // ProgramSpec (immutable)
    // ------------------------------------------------------------------
    ProgramSpec spec;
    spec.name = "reshard_same_height";

    const KernelSpec::CompileTimeArgs compile_time_args = {
        {"interface_with_dram", static_cast<uint32_t>(interface_with_dram)},
    };

    const char* kernel_path = local_is_output ? kSHReaderKernelPath : kSHWriterKernelPath;

    // Two data-movement workers run the same kernel source on every local core; they split the
    // shard height across the two RISCs. The local sharded DFB is bound producer on k0 / consumer
    // on k1 to satisfy the DFB endpoint invariant (the CB is used only as an address source).
    const auto make_worker = [&](const char* name, DataMovementHardwareConfig hw_config, DFBEndpointType endpoint) {
        return KernelSpec{
            .unique_id = KernelSpecName{name},
            .source = std::filesystem::path(kernel_path),
            .dfb_bindings = {DFBBinding{
                .dfb_spec_name = DFBSpecName{kSHDfbName},
                .accessor_name = kSHDfbName,
                .endpoint_type = endpoint,
            }},
            .tensor_bindings =
                {TensorBinding{
                     .tensor_parameter_name = TensorParamName{kSHRemoteTensorParam},
                     .accessor_name = kSHRemoteTensorParam},
                 TensorBinding{
                     .tensor_parameter_name = TensorParamName{kSHLocalTensorParam},
                     .accessor_name = kSHLocalTensorParam}},
            .compile_time_args = compile_time_args,
            .runtime_arg_schema =
                {.runtime_arg_names =
                     {"total_num_sticks", "local_stride_bytes", "remote_stride_bytes", "num_segments"}},
            .hw_config = std::move(hw_config),
            .advanced_options = {.num_runtime_varargs = num_varargs},
        };
    };

    KernelSpec k0 =
        make_worker("reader", ttnn::create_reader_datamovement_config(device->arch()), DFBEndpointType::PRODUCER);
    KernelSpec k1 =
        make_worker("writer", ttnn::create_writer_datamovement_config(device->arch()), DFBEndpointType::CONSUMER);

    DataflowBufferSpec shard_dfb{
        .unique_id = DFBSpecName{kSHDfbName},
        .entry_size = unit_size,
        .num_entries = remote_units_per_shard,
        .data_format_metadata = data_format,
        .borrowed_from = TensorParamName{kSHLocalTensorParam},
    };

    spec.kernels = {k0, k1};
    spec.dataflow_buffers = {shard_dfb};
    spec.tensor_parameters = {
        TensorParameter{.unique_id = TensorParamName{kSHRemoteTensorParam}, .spec = remote_tensor.tensor_spec()},
        TensorParameter{.unique_id = TensorParamName{kSHLocalTensorParam}, .spec = local_tensor.tensor_spec()},
    };
    spec.work_units = {WorkUnitSpec{
        .name = "reshard_same_height_work_unit",
        .kernels = {KernelSpecName{"reader"}, KernelSpecName{"writer"}},
        .target_nodes = all_cores,
    }};

    // ------------------------------------------------------------------
    // ProgramRunArgs (mutable)
    // ------------------------------------------------------------------
    KernelRunArgs k0_run_args{.kernel = KernelSpecName{"reader"}};
    KernelRunArgs k1_run_args{.kernel = KernelSpecName{"writer"}};

    for (uint32_t core_idx = 0; core_idx < local_cores.size(); core_idx++) {
        const auto& args_for_all_segments = runtime_args_for_each_core[core_idx];
        const uint32_t num_segments = static_cast<uint32_t>(args_for_all_segments.size());
        const NodeCoord node{local_cores[core_idx].x, local_cores[core_idx].y};

        KernelRunArgs::RuntimeArgValues& k0_rtas = k0_run_args.runtime_arg_values;
        KernelRunArgs::RuntimeArgValues& k1_rtas = k1_run_args.runtime_arg_values;
        AddRuntimeArgsForNode(
            k0_rtas,
            node,
            {
                {"total_num_sticks", total_num_sticks_kernel_0},
                {"local_stride_bytes", local_stride_bytes},
                {"remote_stride_bytes", remote_stride_bytes},
                {"num_segments", num_segments},
            });
        AddRuntimeArgsForNode(
            k1_rtas,
            node,
            {
                {"total_num_sticks", total_num_sticks_kernel_1},
                {"local_stride_bytes", local_stride_bytes},
                {"remote_stride_bytes", remote_stride_bytes},
                {"num_segments", num_segments},
            });

        AdvancedKernelRunArgs::Varargs varargs_0(num_varargs, 0u);
        AdvancedKernelRunArgs::Varargs varargs_1(num_varargs, 0u);
        uint32_t idx = 0;
        for (const auto& args : args_for_all_segments) {
            // Adjust read/write offsets to the correct stick address (work is split across 2 kernels).
            const uint32_t adjusted_read_offset = args.read_offset + (total_num_sticks_kernel_0 * local_stride_bytes);
            const uint32_t adjusted_write_offset =
                args.write_offset + (total_num_sticks_kernel_0 * remote_stride_bytes);

            varargs_0[idx + 0] = args.write_size;
            varargs_0[idx + 1] = args.read_offset;
            varargs_0[idx + 2] = args.bank_id;
            varargs_0[idx + 3] = args.write_offset;

            varargs_1[idx + 0] = args.write_size;
            varargs_1[idx + 1] = adjusted_read_offset;
            varargs_1[idx + 2] = args.bank_id;
            varargs_1[idx + 3] = adjusted_write_offset;
            idx += 4;
        }
        k0_run_args.advanced_options.runtime_varargs.emplace(node, std::move(varargs_0));
        k1_run_args.advanced_options.runtime_varargs.emplace(node, std::move(varargs_1));
    }

    ProgramRunArgs run_params;
    run_params.kernel_run_args = {std::move(k0_run_args), std::move(k1_run_args)};
    run_params.tensor_args = {
        {TensorParamName{kSHRemoteTensorParam}, TensorArgument{remote_tensor.mesh_tensor()}},
        {TensorParamName{kSHLocalTensorParam}, TensorArgument{local_tensor.mesh_tensor()}},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_params),
    };
}

// Explicit template instantiations
template struct ReshardSameHeightFactory<true>;
template struct ReshardSameHeightFactory<false>;

}  // namespace ttnn::prim::qsr

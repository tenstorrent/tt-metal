// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/sharded/reshard/device/reshard_program_factory_same_height.hpp"
#include "ttnn/operations/data_movement/sharded/sharded_common.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

#include <algorithm>
#include <filesystem>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {

namespace {

// (Names are prefixed to avoid unity-build collisions with the sibling reshard factories.)
constexpr const char* kSHReaderKernelPath =
    "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/reshard_same_height_reader.cpp";
constexpr const char* kSHWriterKernelPath =
    "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/reshard_same_height_writer.cpp";

// Resource / parameter names referenced by the kernel sources (tensor:: / dfb:: accessors).
constexpr const char* kSHRemoteTensorParam = "remote";
constexpr const char* kSHLocalTensorParam = "local";
constexpr const char* kSHShardDfbName = "shard";

constexpr const char* kSHReaderKernel = "reader";
constexpr const char* kSHWriterKernel = "writer";

}  // namespace

template <bool local_is_output>
ttnn::device_operation::ProgramArtifacts ReshardSameHeightFactory<local_is_output>::create_program_artifacts(
    const ReshardParams& /*operation_attributes*/, const ReshardInputs& tensor_args, Tensor& output_tensor) {
    const auto& input = tensor_args.input;
    const auto& output = output_tensor;
    const auto& local_tensor = local_is_output ? output : input;
    const auto& remote_tensor = local_is_output ? input : output;
    const auto local_shard_spec = local_tensor.shard_spec().value();
    const auto remote_shard_spec = remote_tensor.shard_spec().value();

    auto* device = input.device();

    const auto remote_core_type = remote_tensor.buffer()->core_type();
    bool interface_with_dram = (remote_core_type == tt::CoreType::DRAM);
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

    // Plain copies of the structured-binding results, so the run-args lambda below can capture them.
    const uint32_t local_stride = local_stride_bytes;
    const uint32_t remote_stride = remote_stride_bytes;

    // Split work across each kernel along tensor height since this is the best way to split work evenly
    const uint32_t total_num_sticks_kernel_0 = total_num_sticks / 2;
    const uint32_t total_num_sticks_kernel_1 = total_num_sticks - total_num_sticks_kernel_0;

    // Per-(kernel, node) varargs: 4 words per segment. The remote base address (legacy RTA slot 3)
    // is gone (now tensor::remote); the four leading scalars become named RTAs.
    std::vector<std::vector<uint32_t>> reader_varargs(local_cores.size());
    std::vector<std::vector<uint32_t>> writer_varargs(local_cores.size());
    std::vector<uint32_t> num_segments_per_core(local_cores.size());
    uint32_t max_varargs = 0;

    for (uint32_t core_idx = 0; core_idx < local_cores.size(); core_idx++) {
        const auto& args_for_all_segments = runtime_args_for_each_core[core_idx];
        num_segments_per_core[core_idx] = static_cast<uint32_t>(args_for_all_segments.size());

        auto& varargs_0 = reader_varargs[core_idx];
        auto& varargs_1 = writer_varargs[core_idx];
        varargs_0.reserve(args_for_all_segments.size() * 4);
        varargs_1.reserve(args_for_all_segments.size() * 4);

        for (const auto& args : args_for_all_segments) {
            varargs_0.push_back(args.write_size);
            varargs_0.push_back(args.read_offset);
            varargs_0.push_back(args.bank_id);
            varargs_0.push_back(args.write_offset);

            // Adjust read and write offsets to the correct stick address because we are splitting work across 2 kernels
            const uint32_t adjusted_read_offset = args.read_offset + (total_num_sticks_kernel_0 * local_stride_bytes);
            const uint32_t adjusted_write_offset =
                args.write_offset + (total_num_sticks_kernel_0 * remote_stride_bytes);

            varargs_1.push_back(args.write_size);
            varargs_1.push_back(adjusted_read_offset);
            varargs_1.push_back(args.bank_id);
            varargs_1.push_back(adjusted_write_offset);
        }
        max_varargs = std::max(max_varargs, static_cast<uint32_t>(varargs_0.size()));
    }
    // A single per-kernel vararg count covers every node, so each node's list is zero-padded up to
    // the longest one. The kernel loop is bounded by num_segments, so padding words are never read.
    const uint32_t num_varargs = max_varargs;

    // ------------------------------------------------------------------
    // ProgramSpec (immutable)
    // ------------------------------------------------------------------
    ProgramSpec spec;
    spec.name = "reshard_same_height";

    const char* kernel_path = local_is_output ? kSHReaderKernelPath : kSHWriterKernelPath;

    const KernelSpec::CompileTimeArgs compile_time_args = {
        {"interface_with_dram", static_cast<uint32_t>(interface_with_dram)},
    };

    const auto make_worker = [&](const char* name, DataMovementHardwareConfig hw_config, DFBEndpointType endpoint) {
        return KernelSpec{
            .unique_id = KernelSpecName{name},
            .source = std::filesystem::path(kernel_path),
            .dfb_bindings = {DFBBinding{
                .dfb_spec_name = DFBSpecName{kSHShardDfbName},
                .accessor_name = kSHShardDfbName,
                .endpoint_type = endpoint,
            }},
            .tensor_bindings = {TensorBinding{
                .tensor_parameter_name = TensorParamName{kSHRemoteTensorParam}, .accessor_name = kSHRemoteTensorParam}},
            .compile_time_args = compile_time_args,
            .runtime_arg_schema =
                {.runtime_arg_names =
                     {"total_num_sticks", "local_stride_bytes", "remote_stride_bytes", "num_segments"}},
            .hw_config = std::move(hw_config),
            .advanced_options = {.num_runtime_varargs = num_varargs},
        };
    };

    // The two same-source instances split the sticks of each core, so both raw-touch the shard DFB.
    // Two role-free touchers over one grid -> assign 1P + 1C.
    spec.kernels = {
        make_worker(
            kSHReaderKernel, ttnn::create_reader_datamovement_config(device->arch()), DFBEndpointType::PRODUCER),
        make_worker(
            kSHWriterKernel, ttnn::create_writer_datamovement_config(device->arch()), DFBEndpointType::CONSUMER),
    };

    // Local sharded DFB, built on the local buffer's borrowed memory so its backing L1 address is
    // refreshed from the tensor argument on every enqueue.
    //
    // The DFB is only an address source (the kernel reaches the shard through get_write_ptr() /
    // get_read_ptr() + offset and never touches the FIFO), so the entry count does not bound its
    // accesses. Clamp it to the backing tensor's packed size: a padded shard shape can push the
    // shard-derived size past what Metal 2.0's spec-time borrowed-DFB check allows (it compares
    // against the TensorSpec's packed size, having no Buffer at spec time to see the larger real
    // per-bank allocation the legacy dynamic CB was checked against). Behaviour is unchanged
    // because nothing reads the size.
    const uint32_t shard_dfb_bytes = remote_units_per_shard * unit_size;
    const uint32_t local_packed_bytes =
        static_cast<uint32_t>(local_tensor.tensor_spec().compute_packed_buffer_size_bytes());
    spec.dataflow_buffers = {DataflowBufferSpec{
        .unique_id = DFBSpecName{kSHShardDfbName},
        .entry_size = unit_size,
        .num_entries = std::min(shard_dfb_bytes, local_packed_bytes) / unit_size,
        .data_format_metadata = data_format,
        .borrowed_from = TensorParamName{kSHLocalTensorParam},
    }};

    // `local` is a borrowed-only TensorParameter: it backs the shard DFB and is never bound by a
    // kernel. The spec validator counts a borrowed_from reference as a use.
    spec.tensor_parameters = {
        TensorParameter{.unique_id = TensorParamName{kSHRemoteTensorParam}, .spec = remote_tensor.tensor_spec()},
        TensorParameter{.unique_id = TensorParamName{kSHLocalTensorParam}, .spec = local_tensor.tensor_spec()},
    };

    spec.work_units = {WorkUnitSpec{
        .name = "reshard_same_height_work_unit",
        .kernels = {KernelSpecName{kSHReaderKernel}, KernelSpecName{kSHWriterKernel}},
        .target_nodes = all_cores,
    }};

    // ------------------------------------------------------------------
    // ProgramRunArgs (mutable)
    // ------------------------------------------------------------------
    const auto build_kernel_run_args = [&](const char* name,
                                           uint32_t total_num_sticks_for_kernel,
                                           const std::vector<std::vector<uint32_t>>& per_core_varargs) {
        KernelRunArgs run_args{.kernel = KernelSpecName{name}};
        for (uint32_t core_idx = 0; core_idx < local_cores.size(); core_idx++) {
            const auto& node = local_cores[core_idx];
            AddRuntimeArgsForNode(
                run_args.runtime_arg_values,
                node,
                {{"total_num_sticks", total_num_sticks_for_kernel},
                 {"local_stride_bytes", local_stride},
                 {"remote_stride_bytes", remote_stride},
                 {"num_segments", num_segments_per_core[core_idx]}});
            AdvancedKernelRunArgs::Varargs varargs(num_varargs, 0u);
            std::copy(per_core_varargs[core_idx].begin(), per_core_varargs[core_idx].end(), varargs.begin());
            run_args.advanced_options.runtime_varargs.emplace(node, std::move(varargs));
        }
        return run_args;
    };

    ProgramRunArgs run_params;
    run_params.kernel_run_args = {
        build_kernel_run_args(kSHReaderKernel, total_num_sticks_kernel_0, reader_varargs),
        build_kernel_run_args(kSHWriterKernel, total_num_sticks_kernel_1, writer_varargs),
    };
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

}  // namespace ttnn::prim

// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/sharded/reshard/device/reshard_program_factory_same_height.hpp"
#include "ttnn/operations/data_movement/sharded/sharded_common.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {

template <bool local_is_output>
ttnn::device_operation::ProgramArtifacts ReshardSameHeightFactory<local_is_output>::create_program_spec(
    const ReshardParams& /*operation_attributes*/, const ReshardInputs& tensor_args, Tensor& output_tensor) {
    const auto& input = tensor_args.input;
    auto& output = output_tensor;
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

    // Names. The local sharded buffer backs a borrowed-memory DFB; the remote tensor flows through the
    // typed TensorAccessor channel (Case-2 bridge: base address pulled kernel-side via get_bank_base_address).
    const TensorParamName kLocal{"local"};
    const TensorParamName kRemote{"remote"};
    const DFBSpecName kShardCb{"shard_cb"};
    const KernelSpecName kReader{"reader"};
    const KernelSpecName kWriter{"writer"};

    // Both KernelDescriptors use the SAME source (selected by local_is_output); the CT define for the
    // bank type differs by name across the two source files. The reader/writer roles split work along
    // tensor height (each kernel handles half the sticks) and each reads/writes the shared local shard CB
    // purely by base pointer, so the borrowed DFB is bound as a self-loop on both kernels.
    const std::filesystem::path kKernelSource =
        local_is_output ? std::filesystem::path(
                              "ttnn/cpp/ttnn/operations/data_movement/sharded/reshard/device/kernels/"
                              "reshard_same_height_reader_m2.cpp")
                        : std::filesystem::path(
                              "ttnn/cpp/ttnn/operations/data_movement/sharded/reshard/device/kernels/"
                              "reshard_same_height_writer_m2.cpp");
    // The legacy CTA slot 1 (interface_with_dram) becomes a named CTA whose name matches the source file's
    // read_from_dram / write_to_dram constexpr.
    const std::string kDramCtaName = local_is_output ? "read_from_dram" : "write_to_dram";

    ProgramSpec spec;
    spec.name = "reshard_same_height";

    // Tensor parameters: the local (borrowed-memory) shard and the remote tensor (Case-2 binding).
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = kLocal, .spec = local_tensor.tensor_spec()});
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = kRemote, .spec = remote_tensor.tensor_spec()});

    // Local sharded CB bound to the local buffer: borrowed-memory DFB. The backing L1 address resolves at
    // runtime from the local TensorArgument (legacy set CBDescriptor.buffer + UpdateDynamicCircularBufferAddress).
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = kShardCb,
        .entry_size = unit_size,
        .num_entries = remote_units_per_shard,
        .data_format_metadata = data_format,
        .borrowed_from = kLocal,
    });

    // Generate all read/write offsets for each core
    auto remote_buffer_type = remote_buffer->buffer_type();
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

    // Self-loop the borrowed DFB on both kernels (base-pointer access only; no real FIFO).
    auto bind_self_loop = [&](KernelSpec& k) {
        k.dfb_bindings = {
            DFBBinding{
                .dfb_spec_name = kShardCb, .accessor_name = "shard_cb", .endpoint_type = DFBEndpointType::PRODUCER},
            DFBBinding{
                .dfb_spec_name = kShardCb, .accessor_name = "shard_cb", .endpoint_type = DFBEndpointType::CONSUMER},
        };
    };

    auto make_kernel = [&](const KernelSpecName& unique_id, DataMovementRoleHint role) {
        KernelSpec k;
        k.unique_id = unique_id;
        k.source = kKernelSource;
        k.compile_time_args = {{kDramCtaName, static_cast<uint32_t>(interface_with_dram)}};
        k.runtime_arg_schema.runtime_arg_names = {
            "total_num_sticks", "local_stride_bytes", "remote_stride_bytes", "num_segments"};
        bind_self_loop(k);
        k.tensor_bindings = {TensorBinding{.tensor_parameter_name = kRemote, .accessor_name = "remote"}};
        k.hw_config = DataMovementHardwareConfig{.role = role};
        return k;
    };

    KernelSpec reader = make_kernel(kReader, DataMovementRoleHint::READER);
    KernelSpec writer = make_kernel(kWriter, DataMovementRoleHint::WRITER);

    // Determine the max per-core vararg tail length (4 values per segment).
    uint32_t max_tail = 0;
    for (uint32_t core_idx = 0; core_idx < local_cores.size(); core_idx++) {
        max_tail = std::max<uint32_t>(max_tail, 4u * runtime_args_for_each_core[core_idx].size());
    }
    reader.advanced_options.num_runtime_varargs = max_tail;
    writer.advanced_options.num_runtime_varargs = max_tail;

    spec.kernels.push_back(reader);
    spec.kernels.push_back(writer);

    // Both kernels run on all_cores and self-loop the borrowed DFB: one WorkUnitSpec (Local-DFB rule).
    spec.work_units.push_back(WorkUnitSpec{
        .name = "wu",
        .kernels = {kReader, kWriter},
        .target_nodes = all_cores,
    });

    // Run args.
    ProgramRunArgs run_args;
    KernelRunArgs reader_run;
    reader_run.kernel = kReader;
    KernelRunArgs writer_run;
    writer_run.kernel = kWriter;
    reader_run.runtime_arg_values.reserve(local_cores.size());
    writer_run.runtime_arg_values.reserve(local_cores.size());

    // Here all we do is convert pre-computed offsets into vectors so they can be passed as runtime arguments
    for (uint32_t core_idx = 0; core_idx < local_cores.size(); core_idx++) {
        const auto& args_for_all_segments = runtime_args_for_each_core[core_idx];
        const NodeCoord node{local_cores[core_idx]};

        // Named scalars (the legacy buffer-address RTA slot 3 is dropped — Case-2 binding supplies it).
        reader_run.runtime_arg_values.push_back(KernelRunArgs::NodeRuntimeArgs{
            .node = node,
            .args = {
                {"total_num_sticks", total_num_sticks_kernel_0},
                {"local_stride_bytes", local_stride_bytes},
                {"remote_stride_bytes", remote_stride_bytes},
                {"num_segments", static_cast<uint32_t>(args_for_all_segments.size())}}});
        writer_run.runtime_arg_values.push_back(KernelRunArgs::NodeRuntimeArgs{
            .node = node,
            .args = {
                {"total_num_sticks", total_num_sticks_kernel_1},
                {"local_stride_bytes", local_stride_bytes},
                {"remote_stride_bytes", remote_stride_bytes},
                {"num_segments", static_cast<uint32_t>(args_for_all_segments.size())}}});

        // Per-segment packed vararg tail: [read/write_size, offset, bank_id, remote_offset].
        AdvancedKernelRunArgs::Varargs reader_tail;
        AdvancedKernelRunArgs::Varargs writer_tail;
        reader_tail.reserve(max_tail);
        writer_tail.reserve(max_tail);
        for (const auto& args : args_for_all_segments) {
            reader_tail.push_back(args.write_size);
            reader_tail.push_back(args.read_offset);
            reader_tail.push_back(args.bank_id);
            reader_tail.push_back(args.write_offset);

            // Adjust read and write offsets to the correct stick address because we are splitting work across 2 kernels
            const uint32_t adjusted_read_offset = args.read_offset + (total_num_sticks_kernel_0 * local_stride_bytes);
            const uint32_t adjusted_write_offset =
                args.write_offset + (total_num_sticks_kernel_0 * remote_stride_bytes);

            writer_tail.push_back(args.write_size);
            writer_tail.push_back(adjusted_read_offset);
            writer_tail.push_back(args.bank_id);
            writer_tail.push_back(adjusted_write_offset);
        }
        reader_tail.resize(max_tail, 0u);
        writer_tail.resize(max_tail, 0u);
        reader_run.advanced_options.runtime_varargs[node] = std::move(reader_tail);
        writer_run.advanced_options.runtime_varargs[node] = std::move(writer_tail);
    }

    run_args.kernel_run_args.push_back(std::move(reader_run));
    run_args.kernel_run_args.push_back(std::move(writer_run));

    run_args.tensor_args = {
        {kLocal, local_tensor.mesh_tensor()},
        {kRemote, remote_tensor.mesh_tensor()},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

// Explicit template instantiations
template struct ReshardSameHeightFactory<true>;
template struct ReshardSameHeightFactory<false>;

}  // namespace ttnn::prim

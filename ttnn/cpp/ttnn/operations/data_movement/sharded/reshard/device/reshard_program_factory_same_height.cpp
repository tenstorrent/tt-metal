// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/sharded/reshard/device/reshard_program_factory_same_height.hpp"
#include "ttnn/operations/data_movement/sharded/sharded_common.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

namespace {

// Anonymous-namespace helper unique to reshard same-height to avoid unity-build collisions.
void push_reshard_same_height_cb_pair(
    ProgramDescriptor& desc,
    uint32_t cb_index,
    tt::DataFormat data_format,
    uint32_t total_size,
    uint32_t page_size,
    const CoreRangeSet& core_ranges,
    Buffer* bound_buffer) {
    CBDescriptor cb;
    cb.total_size = total_size;
    cb.core_ranges = core_ranges;
    cb.format_descriptors.push_back(CBFormatDescriptor{
        .buffer_index = static_cast<uint8_t>(cb_index),
        .data_format = data_format,
        .page_size = page_size,
    });
    cb.buffer = bound_buffer;
    desc.cbs.push_back(std::move(cb));
}

}  // namespace

template <bool local_is_output>
ProgramDescriptor ReshardSameHeightFactory<local_is_output>::create_descriptor(
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
    const uint32_t total_size = remote_units_per_shard * unit_size;

    constexpr uint32_t cb_index = tt::CBIndex::c_0;

    auto* local_buffer = local_tensor.buffer();
    auto* remote_buffer = remote_tensor.buffer();

    ProgramDescriptor desc;

    // Local sharded CB. Bind to local buffer for dynamic-CB rebinding on cache hits via cb.buffer.
    push_reshard_same_height_cb_pair(
        desc, cb_index, data_format, total_size, unit_size, all_cores, /*bound_buffer=*/local_buffer);

    const std::string kernel_name =
        local_is_output
            ? "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/reshard_same_height_reader.cpp"
            : "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/reshard_same_height_writer.cpp";

    KernelDescriptor reader_desc;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.kernel_source = kernel_name;
    reader_desc.core_ranges = all_cores;
    reader_desc.config = ReaderConfigDescriptor{};
    reader_desc.compile_time_args = {cb_index, static_cast<uint32_t>(interface_with_dram)};

    KernelDescriptor writer_desc;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.kernel_source = kernel_name;
    writer_desc.core_ranges = all_cores;
    writer_desc.config = WriterConfigDescriptor{};
    writer_desc.compile_time_args = {cb_index, static_cast<uint32_t>(interface_with_dram)};

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

    // Here all we do is convert pre-computed offsets into vectors so they can be passed as runtime arguments
    for (uint32_t core_idx = 0; core_idx < local_cores.size(); core_idx++) {
        const auto& args_for_all_segments = runtime_args_for_each_core[core_idx];

        // arg 3 is remote-buffer base address (binding via Buffer*).
        KernelDescriptor::RTArgList runtime_args_0;
        runtime_args_0.push_back(total_num_sticks_kernel_0);
        runtime_args_0.push_back(local_stride_bytes);
        runtime_args_0.push_back(remote_stride_bytes);
        runtime_args_0.push_back(remote_buffer);
        runtime_args_0.push_back(static_cast<uint32_t>(args_for_all_segments.size()));

        KernelDescriptor::RTArgList runtime_args_1;
        runtime_args_1.push_back(total_num_sticks_kernel_1);
        runtime_args_1.push_back(local_stride_bytes);
        runtime_args_1.push_back(remote_stride_bytes);
        runtime_args_1.push_back(remote_buffer);
        runtime_args_1.push_back(static_cast<uint32_t>(args_for_all_segments.size()));

        for (const auto& args : args_for_all_segments) {
            runtime_args_0.push_back(args.write_size);
            runtime_args_0.push_back(args.read_offset);
            runtime_args_0.push_back(args.bank_id);
            runtime_args_0.push_back(args.write_offset);

            // Adjust read and write offsets to the correct stick address because we are splitting work across 2 kernels
            const uint32_t adjusted_read_offset = args.read_offset + (total_num_sticks_kernel_0 * local_stride_bytes);
            const uint32_t adjusted_write_offset =
                args.write_offset + (total_num_sticks_kernel_0 * remote_stride_bytes);

            runtime_args_1.push_back(args.write_size);
            runtime_args_1.push_back(adjusted_read_offset);
            runtime_args_1.push_back(args.bank_id);
            runtime_args_1.push_back(adjusted_write_offset);
        }
        reader_desc.emplace_runtime_args(local_cores[core_idx], runtime_args_0);
        writer_desc.emplace_runtime_args(local_cores[core_idx], runtime_args_1);
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));

    return desc;
}

// Explicit template instantiations
template struct ReshardSameHeightFactory<true>;
template struct ReshardSameHeightFactory<false>;

}  // namespace ttnn::prim

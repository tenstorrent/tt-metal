// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "high_bw_sparse_push_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"
#include "sparse_push_program.hpp"

namespace ttnn::operations::experimental::high_bw_all_gather {

namespace {

sparse_push::ReaderConfig reader_config(
    const HighBwAllGatherParams& params, const HighBwAllGatherInputs& tensor_args, uint32_t num_devices) {
    return sparse_push::ReaderConfig{
        .input = tensor_args.input_tensor,
        .table = *tensor_args.push_table,
        .batch_index_tensor = tensor_args.input_batch_index_tensor,
        .num_devices = num_devices,
        .stripe_chunk = params.push_stripe_chunk,
        .max_rows = params.push_max_rows,
        .push_part = params.push_part};
}

}  // namespace

HighBwSparsePushFactory::cached_mesh_workload_t HighBwSparsePushFactory::create_mesh_workload(
    const HighBwAllGatherParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const HighBwAllGatherInputs& tensor_args,
    Tensor& output_tensor) {
    const auto& input_tensor = tensor_args.input_tensor;
    auto* mesh_device = input_tensor.device();
    const auto& mesh_view = mesh_device->get_view();
    const uint32_t num_devices = mesh_view.shape().mesh_size();
    TT_FATAL(
        operation_attributes.ready_semaphore.has_value() && operation_attributes.data_valid_semaphore.has_value(),
        "the sparse KV push needs caller-owned ready_semaphore / data_valid_semaphore");

    const uint32_t page_bytes = input_tensor.buffer()->aligned_page_size();
    TT_FATAL(
        output_tensor.buffer()->aligned_page_size() == page_bytes,
        "the sparse KV push needs matching input/output row pages ({} vs {})",
        page_bytes,
        output_tensor.buffer()->aligned_page_size());
    TT_FATAL(
        output_tensor.buffer()->num_pages() == output_tensor.padded_shape()[2],
        "the sparse KV push needs one output page per compact-buffer row");

    const auto layout = sparse_push::core_layout(
        operation_attributes.resolved_worker_core_grid, operation_attributes.num_links.value_or(2));
    const auto mux = sparse_push::mux_config(mesh_device);
    const auto defines = sparse_push::kernel_defines(mesh_device);
    const auto cfg = reader_config(operation_attributes, tensor_args, num_devices);
    const uint32_t banks = mesh_device->allocator()->get_num_banks(tt::tt_metal::BufferType::DRAM);
    const auto reader_ct = sparse_push::reader_compile_args(cfg, banks);
    const auto reader_common = sparse_push::reader_common_args(
        cfg,
        operation_attributes.batch_slot_num_layers,
        operation_attributes.batch_slot_layer_idx,
        operation_attributes.input_batch_index.value_or(0));
    const uint32_t packet_bytes = sparse_push::pages_per_packet(page_bytes) * page_bytes;

    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    for (const auto& coord : tensor_coords.coords()) {
        tt::tt_metal::Program program;
        const uint32_t my_rank = ttnn::operations::ccl::common::get_linearized_index(coord, mesh_view);
        const auto neighbors = sparse_push::torus_neighbors(mesh_view, coord);

        for (const auto& cb : sparse_push::circular_buffers(packet_bytes)) {
            tt::tt_metal::CreateCircularBuffer(
                program,
                layout.sender_range,
                tt::tt_metal::CircularBufferConfig(cb.page_bytes * cb.pages, {{cb.id, cb.format}})
                    .set_page_size(cb.id, cb.page_bytes));
        }
        const std::string kdir = sparse_push::kKernelDir;
        auto reader = tt::tt_metal::CreateKernel(
            program,
            kdir + "sparse_push_table_reader.cpp",
            layout.sender_range,
            tt::tt_metal::ReaderDataMovementConfig(reader_ct, defines));
        auto writer = tt::tt_metal::CreateKernel(
            program,
            kdir + "sparse_push_writer.cpp",
            layout.sender_range,
            tt::tt_metal::WriterDataMovementConfig(
                sparse_push::writer_compile_args(output_tensor, num_devices, my_rank, layout, mux), defines));
        auto mux_kernel = tt::tt_metal::CreateKernel(
            program,
            "tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp",
            layout.mux_range,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::RISCV_0_default,
                .compile_args = mux.get_fabric_mux_compile_time_args(),
                .opt_level = tt::tt_metal::KernelBuildOptLevel::O3});

        tt::tt_metal::SetCommonRuntimeArgs(program, reader, reader_common);
        tt::tt_metal::SetCommonRuntimeArgs(
            program,
            writer,
            {output_tensor.buffer()->address(),
             operation_attributes.ready_semaphore->address(),
             operation_attributes.data_valid_semaphore->address()});
        for (uint32_t c = 0; c < layout.channels(); ++c) {
            tt::tt_metal::SetRuntimeArgs(
                program,
                mux_kernel,
                layout.muxes[c],
                sparse_push::mux_runtime_args(mesh_device, coord, neighbors, layout, c, mux, program));
        }
        for (uint32_t w = 0; w < layout.num_senders(); ++w) {
            tt::tt_metal::SetRuntimeArgs(
                program, reader, layout.senders[w], sparse_push::reader_runtime_args(layout, w, my_rank));
            tt::tt_metal::SetRuntimeArgs(
                program,
                writer,
                layout.senders[w],
                sparse_push::writer_runtime_args(mesh_device, layout, w, mux, program));
        }

        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(program));
        shared_variables.emplace(
            ttnn::MeshCoordinateRange(coord),
            shared_variables_t{.reader_kernel_id = reader, .writer_kernel_id = writer});
    }
    return cached_mesh_workload_t{std::move(workload), std::move(shared_variables)};
}

void HighBwSparsePushFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const HighBwAllGatherParams& operation_attributes,
    const HighBwAllGatherInputs& tensor_args,
    Tensor& output_tensor) {
    const uint32_t num_devices = tensor_args.input_tensor.device()->get_view().shape().mesh_size();
    const auto reader_common = sparse_push::reader_common_args(
        reader_config(operation_attributes, tensor_args, num_devices),
        operation_attributes.batch_slot_num_layers,
        operation_attributes.batch_slot_layer_idx,
        operation_attributes.input_batch_index.value_or(0));
    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        auto& vars = cached_workload.shared_variables.at(range);
        auto& reader = tt::tt_metal::GetCommonRuntimeArgs(program, vars.reader_kernel_id);
        for (size_t i = 0; i < reader_common.size(); ++i) {
            reader[i] = reader_common[i];
        }
        auto& writer = tt::tt_metal::GetCommonRuntimeArgs(program, vars.writer_kernel_id);
        writer[0] = output_tensor.buffer()->address();
        writer[1] = operation_attributes.ready_semaphore->address();
        writer[2] = operation_attributes.data_valid_semaphore->address();
    }
}

}  // namespace ttnn::operations::experimental::high_bw_all_gather

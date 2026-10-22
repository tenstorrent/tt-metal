// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// sparse_sdpa with a fused next-layer sparse KV push. The attention program is the plain sparse_sdpa one on
// worker_core_grid; the push of the next layer's prefix rows (high_bw_all_gather's sparse push, part 1: rows written
// by earlier chunks) is appended on push_core_grid, so the two overlap in one program. The next layer pushes the
// rest (part 2: its own chunk's rows) before its attention.

#include "ttnn/operations/transformer/sdpa/device/sparse_sdpa_device_operation.hpp"
#include "ttnn/operations/experimental/high_bw_all_gather/device/sparse_push_program.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include <string>
#include <vector>

namespace ttnn::prim {

namespace {

namespace sparse_push = ttnn::operations::experimental::high_bw_all_gather::sparse_push;

// Kernel order in the fused program: sparse_sdpa's reader, writer, compute, then these.
constexpr uint32_t kPushReaderKernel = 3, kPushWriterKernel = 4;
constexpr uint32_t kPrefixPart = 1;

sparse_push::ReaderConfig reader_config(const SparseSDPAFusedPush& push, const SparseSDPAInputs& t, uint32_t devices) {
    return sparse_push::ReaderConfig{
        .input = *t.push_input,
        .table = *t.push_table,
        .batch_index_tensor = t.push_batch_index_tensor,
        .num_devices = devices,
        .stripe_chunk = push.stripe_chunk,
        .max_rows = push.max_rows,
        .push_part = kPrefixPart};
}

tt::tt_metal::KernelDescriptor::Defines to_defines(const std::map<std::string, std::string>& defines) {
    tt::tt_metal::KernelDescriptor::Defines out;
    for (const auto& [k, v] : defines) {
        out.emplace_back(k, v);
    }
    return out;
}

}  // namespace

tt::tt_metal::ProgramDescriptor SparseSDPAOperation::SparseSDPAPushFusedFactory::create_descriptor(
    const operation_attributes_t& attrs,
    const tensor_args_t& t,
    tensor_return_value_t& output,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    TT_FATAL(mesh_dispatch_coordinate.has_value(), "fused push programs are per device");
    auto desc = SparseSDPAProgramFactory::create_descriptor(attrs, t, output);
    TT_FATAL(desc.kernels.size() == kPushReaderKernel, "sparse_sdpa kernel layout changed");

    const auto& push = *attrs.fused_push;
    const auto coord = *mesh_dispatch_coordinate;
    auto* mesh_device = t.q.device();
    const auto& mesh_view = mesh_device->get_view();
    const uint32_t num_devices = mesh_view.shape().mesh_size();
    const uint32_t my_rank = ttnn::operations::ccl::common::get_linearized_index(coord, mesh_view);
    const auto neighbors = sparse_push::torus_neighbors(mesh_view, coord);
    const auto layout = sparse_push::core_layout(push.push_core_grid, push.num_links);
    const auto mux = sparse_push::mux_config(mesh_device);
    const auto defines = to_defines(sparse_push::kernel_defines(mesh_device));
    const auto cfg = reader_config(push, t, num_devices);
    const uint32_t page_bytes = t.push_input->buffer()->aligned_page_size();
    const uint32_t packet_bytes = sparse_push::pages_per_packet(page_bytes) * page_bytes;

    for (const auto& cb : sparse_push::circular_buffers(packet_bytes)) {
        desc.cbs.push_back(tt::tt_metal::CBDescriptor{
            .total_size = cb.page_bytes * cb.pages,
            .core_ranges = layout.sender_range,
            .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(cb.id), .data_format = cb.format, .page_size = cb.page_bytes}}},
        });
    }
    const std::string kdir = sparse_push::kKernelDir;

    tt::tt_metal::KernelDescriptor reader;
    reader.kernel_source = kdir + "sparse_push_table_reader.cpp";
    reader.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = layout.sender_range;
    reader.compile_time_args =
        sparse_push::reader_compile_args(cfg, mesh_device->allocator()->get_num_banks(tt::tt_metal::BufferType::DRAM));
    reader.defines = defines;
    reader.config = tt::tt_metal::ReaderConfigDescriptor{};
    // Same order as sparse_push::reader_common_args; buffers are bound so a cache hit re-patches them.
    reader.emplace_common_runtime_args(
        {t.push_input->buffer(),
         t.push_table->buffer(),
         t.push_batch_index_tensor.has_value() ? t.push_batch_index_tensor->buffer()->address() : 0u,
         push.batch_slot_num_layers,
         push.batch_slot_layer_idx,
         push.input_batch_index});

    tt::tt_metal::KernelDescriptor writer;
    writer.kernel_source = kdir + "sparse_push_writer.cpp";
    writer.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = layout.sender_range;
    writer.compile_time_args = sparse_push::writer_compile_args(*t.push_output, num_devices, my_rank, layout, mux);
    writer.defines = defines;
    writer.config = tt::tt_metal::WriterConfigDescriptor{};
    writer.emplace_common_runtime_args(
        {t.push_output->buffer(),
         static_cast<uint32_t>(push.ready_semaphore->address()),  // smuggled-rta-ok: semaphore, re-patched on hits
         static_cast<uint32_t>(push.data_valid_semaphore->address())});  // smuggled-rta-ok: semaphore, re-patched

    tt::tt_metal::KernelDescriptor mux_kernel;
    mux_kernel.kernel_source = "tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp";
    mux_kernel.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    mux_kernel.core_ranges = layout.mux_range;
    mux_kernel.compile_time_args = mux.get_fabric_mux_compile_time_args();
    mux_kernel.config = tt::tt_metal::DataMovementConfigDescriptor{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = tt::tt_metal::NOC::RISCV_0_default};

    for (uint32_t c = 0; c < layout.channels(); ++c) {
        tt::tt_metal::KernelDescriptor::RTArgList rt;
        rt.append(sparse_push::mux_runtime_args(mesh_device, coord, neighbors, layout, c, mux, desc));
        mux_kernel.emplace_runtime_args(layout.muxes[c], rt);
    }
    for (uint32_t w = 0; w < layout.num_senders(); ++w) {
        tt::tt_metal::KernelDescriptor::RTArgList reader_rt;
        reader_rt.append(sparse_push::reader_runtime_args(layout, w, my_rank));
        reader.emplace_runtime_args(layout.senders[w], reader_rt);
        tt::tt_metal::KernelDescriptor::RTArgList writer_rt;
        writer_rt.append(sparse_push::writer_runtime_args(mesh_device, layout, w, mux, desc));
        writer.emplace_runtime_args(layout.senders[w], writer_rt);
    }
    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(writer));
    desc.kernels.push_back(std::move(mux_kernel));
    return desc;
}

void SparseSDPAOperation::SparseSDPAPushFusedFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    SparseSDPAProgramFactory::override_runtime_arguments(
        program, operation_attributes, tensor_args, tensor_return_value, mesh_dispatch_coordinate);
    const auto& push = *operation_attributes.fused_push;
    const uint32_t num_devices = tensor_args.q.device()->get_view().shape().mesh_size();
    const auto common = sparse_push::reader_common_args(
        reader_config(push, tensor_args, num_devices),
        push.batch_slot_num_layers,
        push.batch_slot_layer_idx,
        push.input_batch_index);
    auto& reader = tt::tt_metal::GetCommonRuntimeArgs(program, kPushReaderKernel);
    for (size_t i = 0; i < common.size(); ++i) {
        reader[i] = common[i];
    }
    auto& writer = tt::tt_metal::GetCommonRuntimeArgs(program, kPushWriterKernel);
    writer[0] = tensor_args.push_output->buffer()->address();
    writer[1] = push.ready_semaphore->address();
    writer[2] = push.data_valid_semaphore->address();
}

}  // namespace ttnn::prim

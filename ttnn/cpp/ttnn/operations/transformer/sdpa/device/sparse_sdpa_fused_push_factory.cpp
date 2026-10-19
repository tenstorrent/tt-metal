// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PROTOTYPE: sparse_sdpa with a fused next-layer sparse KV push. The attention program is the plain sparse_sdpa one on
// worker_core_grid; the push (high_bw_all_gather's table reader + sender writer + fabric mux, push_part 1) is appended
// on push_core_grid. Each attention writer, when done, increments the quit semaphore of every push sender; a sender
// stops at its next packet boundary and records how many packets it sent, so the next layer resumes (push_part 3).

#include "ttnn/operations/transformer/sdpa/device/sparse_sdpa_device_operation.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/sparse_sdpa_common.hpp"
#include "ttnn/operations/experimental/high_bw_all_gather/device/sparse_push_plan_device_operation.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include <algorithm>
#include <cstdlib>
#include <set>
#include <string>
#include <vector>

namespace ttnn::prim {

namespace fused_push_detail {

constexpr uint32_t kDirections = 4;  // E, W, N, S
constexpr uint32_t kSendersPerChannel = 2;
constexpr uint32_t kPacketSlots = 8;
constexpr uint32_t kMetaEntryBytes = 16;
constexpr uint32_t kScalarCbBytes = 64;
enum PushCb : uint32_t { kData = 0, kMeta = 1, kRuns = 2, kScalar = 3, kHeaders = 4 };
// Kernel order in the fused program: sparse_sdpa's reader, writer, compute, then these.
constexpr uint32_t kPushReaderKernel = 3, kPushWriterKernel = 4;

}  // namespace fused_push_detail

using namespace fused_push_detail;

std::vector<CoreCoord> sparse_sdpa_push_worker_cores(const SparseSDPAFusedPush& push) {
    const auto cores = corerange_to_cores(push.push_core_grid, std::nullopt, true);
    const uint32_t workers = push.num_links * kDirections * kSendersPerChannel;
    TT_FATAL(
        cores.size() >= workers + push.num_links * kDirections,
        "fused push needs {} sender + {} mux cores, push_core_grid has {}",
        workers,
        push.num_links * kDirections,
        cores.size());
    return {cores.begin(), cores.begin() + workers};
}

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
    const auto& input = *t.push_input;
    const auto& table = *t.push_table;
    const auto& out = *t.push_output;
    const auto& progress = *t.push_progress;
    auto* mesh_device = t.q.device();
    const auto& mesh_view = mesh_device->get_view();
    const auto mesh_shape = mesh_view.shape();
    const uint32_t num_devices = mesh_shape[0] * mesh_shape[1];
    const uint32_t links = push.num_links;
    const uint32_t channels = links * kDirections;
    const uint32_t workers = channels * kSendersPerChannel;

    const auto all_cores = corerange_to_cores(push.push_core_grid, std::nullopt, true);
    const std::vector<CoreCoord> worker_cores = sparse_sdpa_push_worker_cores(push);
    const std::vector<CoreCoord> mux_cores(all_cores.begin() + workers, all_cores.begin() + workers + channels);
    auto to_range = [](const std::vector<CoreCoord>& cores) {
        std::set<CoreRange> ranges;
        for (const auto& c : cores) {
            ranges.insert(CoreRange(c, c));
        }
        return CoreRangeSet(ranges);
    };
    const CoreRangeSet worker_range = to_range(worker_cores);
    const CoreRangeSet mux_range = to_range(mux_cores);

    const uint32_t page_bytes = input.buffer()->aligned_page_size();
    const uint32_t pages_per_slot = input.buffer()->num_pages() / input.padded_shape()[0];
    const uint32_t banks = mesh_device->allocator()->get_num_banks(tt::tt_metal::BufferType::DRAM);
    const uint32_t pages_per_packet = std::max<uint32_t>(
        1u, static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_max_payload_size_bytes()) / page_bytes);
    const uint32_t packet_bytes = pages_per_packet * page_bytes;
    const auto layout = ttnn::operations::experimental::high_bw_all_gather::SparsePushPlanLayout::make(
        push.max_rows, push.stripe_chunk);
    using Layout = ttnn::operations::experimental::high_bw_all_gather::SparsePushPlanLayout;

    tt::tt_fabric::FabricMuxConfig mux_config(
        kSendersPerChannel,
        0,
        2,
        0,
        tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes(),
        mesh_device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1));

    std::vector<uint32_t> dest_chip_ids, dest_mesh_ids;
    for (const auto& c : tt::tt_metal::distributed::MeshCoordinateRange(mesh_shape)) {
        const auto node = mesh_device->get_fabric_node_id(c);
        dest_chip_ids.push_back(static_cast<uint32_t>(node.chip_id));
        dest_mesh_ids.push_back(static_cast<uint32_t>(*node.mesh_id));
    }
    const uint32_t my_rank = ttnn::operations::ccl::common::get_linearized_index(coord, mesh_view);
    const auto [neighbors, directions] =
        ttnn::operations::ccl::common::get_neighbors(mesh_view, coord, tt::tt_fabric::Topology::Torus, std::nullopt);
    TT_FATAL(
        std::all_of(directions.begin(), directions.end(), [](bool d) { return d; }),
        "fused push needs all four torus neighbours");

    auto cb = [&](uint32_t id, uint32_t page, uint32_t pages, tt::DataFormat df) {
        desc.cbs.push_back(tt::tt_metal::CBDescriptor{
            .total_size = page * pages,
            .core_ranges = worker_range,
            .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(id), .data_format = df, .page_size = page}}},
        });
    };
    cb(kData, packet_bytes, kPacketSlots, tt::DataFormat::RawUInt8);
    cb(kMeta, kMetaEntryBytes, kPacketSlots, tt::DataFormat::RawUInt32);
    cb(kRuns, 4096, 1, tt::DataFormat::RawUInt32);
    cb(kScalar, kScalarCbBytes, 1, tt::DataFormat::RawUInt32);
    cb(kHeaders, tt::tt_fabric::get_tt_fabric_packet_header_size_bytes(), 1 + 4, tt::DataFormat::RawUInt32);

    // Quit semaphore on the push senders; sparse_sdpa's writers increment it (its id is the first one this program
    // allocates, matching the base factory's assumption).
    const uint32_t quit_sem_id = static_cast<uint32_t>(desc.semaphores.size());
    TT_FATAL(quit_sem_id == 0, "the attention writers signal semaphore 0");
    desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
        .id = quit_sem_id, .core_type = tt::CoreType::WORKER, .core_ranges = worker_range, .initial_value = 0});
    // Prototype knob: SPARSE_PUSH_FUSED_QUIT=1 stops the push when attention is done (the next layer's resume pushes
    // the rest); by default the push runs to completion, so the resume only pushes the current chunk's rows.
    const char* quit_env = std::getenv("SPARSE_PUSH_FUSED_QUIT");
    const bool quit_early = quit_env != nullptr && std::string(quit_env) != "0";
    const uint32_t quit_target = quit_early ? attrs.worker_core_grid.num_cores() : 0u;

    std::vector<uint32_t> reader_ct = {
        page_bytes,
        pages_per_packet,
        banks,
        pages_per_slot,
        0u,  // slot from a host scalar
        num_devices,
        push.stripe_chunk,
        layout.num_blocks,
        layout.pages,
        Layout::kOwnerOff,
        Layout::kBitmapOff,
        Layout::kOwnerPrefixOff,
        1u,  // push_part 1: the prefix rows
        quit_sem_id,
        quit_target,
        1u,  // records progress
    };
    tt::tt_metal::TensorAccessorArgs(input.buffer()).append_to(reader_ct);
    tt::tt_metal::TensorAccessorArgs(table.buffer()).append_to(reader_ct);
    tt::tt_metal::TensorAccessorArgs(progress.buffer()).append_to(reader_ct);
    std::vector<uint32_t> writer_ct = {page_bytes, num_devices, my_rank, workers, packet_bytes, links};
    tt::tt_metal::TensorAccessorArgs(out.buffer()).append_to(writer_ct);
    writer_ct.push_back(1u);  // mux
    ttnn::ccl::fabric_mux_connection_ct_args(
        kSendersPerChannel, tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL, mux_config, writer_ct);
    const tt::tt_metal::KernelDescriptor::Defines defines = {
        {"DEST_CHIP_ID", ttnn::operations::ccl::common::stringify(dest_chip_ids)},
        {"DEST_MESH_ID", ttnn::operations::ccl::common::stringify(dest_mesh_ids)},
        {"FABRIC_2D", "1"},
    };
    const std::string kdir = "ttnn/cpp/ttnn/operations/experimental/high_bw_all_gather/device/kernels/";

    tt::tt_metal::KernelDescriptor reader;
    reader.kernel_source = kdir + "sparse_push_table_reader.cpp";
    reader.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = worker_range;
    reader.compile_time_args = reader_ct;
    reader.defines = defines;
    reader.config = tt::tt_metal::ReaderConfigDescriptor{};
    reader.emplace_common_runtime_args(
        {input.buffer(), table.buffer(), 0u, 1u, 0u, push.input_batch_index, progress.buffer()});

    tt::tt_metal::KernelDescriptor writer;
    writer.kernel_source = kdir + "sparse_push_writer.cpp";
    writer.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = worker_range;
    writer.compile_time_args = writer_ct;
    writer.defines = defines;
    writer.config = tt::tt_metal::WriterConfigDescriptor{};
    writer.emplace_common_runtime_args(
        {out.buffer(),
         static_cast<uint32_t>(push.ready_semaphore->address()),  // smuggled-rta-ok: semaphore, re-patched on hits
         static_cast<uint32_t>(push.data_valid_semaphore->address())});  // smuggled-rta-ok: semaphore, re-patched

    tt::tt_metal::KernelDescriptor mux;
    mux.kernel_source = "tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp";
    mux.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    mux.core_ranges = mux_range;
    mux.compile_time_args = mux_config.get_fabric_mux_compile_time_args();
    mux.config = tt::tt_metal::DataMovementConfigDescriptor{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = tt::tt_metal::NOC::RISCV_0_default};

    const auto src_node = mesh_device->get_fabric_node_id(coord);
    const auto sync_core = mesh_device->worker_core_from_logical_core(worker_cores[0]);
    for (uint32_t c = 0; c < channels; ++c) {
        const uint32_t link = c / kDirections, dir = c % kDirections;
        tt::tt_metal::KernelDescriptor::RTArgList mux_rt;
        mux_rt.append(mux_config.get_fabric_mux_run_time_args(
            src_node, mesh_device->get_fabric_node_id(neighbors[dir]), link, desc, mux_cores[c]));
        mux.emplace_runtime_args(mux_cores[c], mux_rt);
    }
    for (uint32_t w = 0; w < workers; ++w) {
        const uint32_t c = w / kSendersPerChannel, k = w % kSendersPerChannel;
        const uint32_t link = c / kDirections, dir = c % kDirections;
        reader.emplace_runtime_args(worker_cores[w], {link, dir, my_rank, k, kSendersPerChannel, links, w});
        std::vector<uint32_t> writer_rt = {
            w, dir, link, k, static_cast<uint32_t>(sync_core.x), static_cast<uint32_t>(sync_core.y)};
        for (uint32_t other = 0; other < workers; ++other) {
            const auto phys = mesh_device->worker_core_from_logical_core(worker_cores[other]);
            writer_rt.push_back(static_cast<uint32_t>(phys.x));
            writer_rt.push_back(static_cast<uint32_t>(phys.y));
        }
        ttnn::ccl::fabric_mux_connection_rt_args(
            /*mux_connection_valid=*/true,
            /*is_termination_master=*/k == 0,
            tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
            mesh_device->worker_core_from_logical_core(mux_cores[c]),
            /*worker_id=*/k,
            worker_cores[w],
            mux_config,
            desc,
            mesh_device->worker_core_from_logical_core(worker_cores[c * kSendersPerChannel]),
            writer_rt);
        tt::tt_metal::KernelDescriptor::RTArgList rt;
        rt.append(writer_rt);
        writer.emplace_runtime_args(worker_cores[w], rt);
    }
    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(writer));
    desc.kernels.push_back(std::move(mux));
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
    auto& reader = tt::tt_metal::GetCommonRuntimeArgs(program, kPushReaderKernel);
    reader[0] = tensor_args.push_input->buffer()->address();
    reader[1] = tensor_args.push_table->buffer()->address();
    reader[5] = push.input_batch_index;
    reader[6] = tensor_args.push_progress->buffer()->address();
    auto& writer = tt::tt_metal::GetCommonRuntimeArgs(program, kPushWriterKernel);
    writer[0] = tensor_args.push_output->buffer()->address();
    writer[1] = push.ready_semaphore->address();
    writer[2] = push.data_valid_semaphore->address();
}

}  // namespace ttnn::prim

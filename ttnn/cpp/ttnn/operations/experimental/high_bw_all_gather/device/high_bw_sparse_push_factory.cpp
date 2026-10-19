// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "high_bw_sparse_push_factory.hpp"

#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "sparse_push_plan_device_operation.hpp"

namespace ttnn::operations::experimental::high_bw_all_gather {

namespace {

constexpr uint32_t kDirections = 4;  // E, W, N, S: one fabric channel per (link, direction)
constexpr uint32_t kMaxChannels = 8;
// Senders per channel. 0 = one sender owns the router connection directly; K >= 1 = K senders feed a fabric mux
// core that owns it. Prototype knob: SPARSE_PUSH_MUX_K.
constexpr uint32_t kDefaultSendersPerChannel = 2;
constexpr uint32_t kPacketSlots = 8;
constexpr uint32_t kMetaEntryBytes = 16;
constexpr uint32_t kBlockWords = 256;  // stream words per refill
constexpr uint32_t kScalarCbBytes = 64;

enum Cb : uint32_t { kData = 0, kMeta = 1, kRuns = 2, kScalar = 3, kHeaders = 4 };

struct WorkerKernels {
    tt::tt_metal::KernelHandle reader;
    tt::tt_metal::KernelHandle writer;
};

uint32_t num_links(const HighBwAllGatherParams& params, const HighBwAllGatherInputs& tensor_args) {
    if (params.push_stripe_chunk > 0) {
        return params.num_links.value_or(2);
    }
    // push_runs_tensor per device: [1, 1, links, words]; one stream page per link, shared by its 4 senders.
    return tensor_args.push_runs_tensor->padded_shape()[-2];
}

}  // namespace

HighBwSparsePushFactory::cached_mesh_workload_t HighBwSparsePushFactory::create_mesh_workload(
    const HighBwAllGatherParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const HighBwAllGatherInputs& tensor_args,
    Tensor& output_tensor) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& runs_tensor = *tensor_args.push_runs_tensor;
    auto* mesh_device = input_tensor.device();
    const auto& mesh_view = mesh_device->get_view();
    const auto mesh_shape = mesh_view.shape();
    const uint32_t mesh_rows = mesh_shape[0];
    const uint32_t mesh_cols = mesh_shape[1];
    const uint32_t num_devices = mesh_rows * mesh_cols;

    const uint32_t links = num_links(operation_attributes, tensor_args);
    const bool from_table = operation_attributes.push_stripe_chunk > 0;
    const uint32_t channels = links * kDirections;
    TT_FATAL(channels >= 1 && channels <= kMaxChannels, "sparse push supports 1..2 links, got {}", links);
    TT_FATAL(
        operation_attributes.ready_semaphore.has_value(),
        "sparse push (prototype) requires caller-owned ready_semaphore / data_valid_semaphore");

    const uint32_t page_bytes = input_tensor.buffer()->aligned_page_size();
    TT_FATAL(
        output_tensor.buffer()->aligned_page_size() == page_bytes,
        "sparse push needs matching input/output row pages ({} vs {})",
        page_bytes,
        output_tensor.buffer()->aligned_page_size());
    const uint32_t out_rows = output_tensor.padded_shape()[2];
    TT_FATAL(output_tensor.buffer()->num_pages() == out_rows, "sparse push needs one output page per gathered row");
    TT_FATAL(out_rows % num_devices == 0, "output rows {} must split evenly over {} ranks", out_rows, num_devices);
    const uint32_t out_rows_per_rank = out_rows / num_devices;
    const uint32_t pages_per_slot = input_tensor.buffer()->num_pages() / input_tensor.padded_shape()[0];
    const uint32_t num_dram_banks = mesh_device->allocator()->get_num_banks(tt::tt_metal::BufferType::DRAM);
    const uint32_t max_payload = tt::tt_fabric::get_tt_fabric_max_payload_size_bytes();
    const uint32_t pages_per_packet = std::max(1u, max_payload / page_bytes);
    const uint32_t packet_bytes = pages_per_packet * page_bytes;
    const uint32_t runs_page_bytes = runs_tensor.buffer()->aligned_page_size();
    const bool batch_from_meta = tensor_args.has_batch_index_metadata();

    const auto cores_all = corerange_to_cores(operation_attributes.resolved_worker_core_grid, std::nullopt, true);
    uint32_t requested_k = kDefaultSendersPerChannel;
    if (const char* k_env = std::getenv("SPARSE_PUSH_MUX_K")) {
        requested_k = static_cast<uint32_t>(std::stoul(k_env));
    }
    // Each channel needs K senders plus its mux core; fall back to fewer senders when the grid is short.
    uint32_t per_channel_k = requested_k;
    while (per_channel_k > 0 && channels * (per_channel_k + 1) > cores_all.size()) {
        --per_channel_k;
    }
    if (per_channel_k != requested_k) {
        log_warning(
            tt::LogOp,
            "sparse push: {} senders per channel need {} cores, only {} available; using {}",
            requested_k,
            channels * (requested_k + 1),
            cores_all.size(),
            per_channel_k);
    }
    const bool use_mux = per_channel_k > 0;
    const uint32_t senders_per_channel = use_mux ? per_channel_k : 1u;
    const uint32_t workers = channels * senders_per_channel;
    TT_FATAL(cores_all.size() >= workers, "sparse push needs {} worker cores", workers);
    // Layout: channel c = link * 4 + dir; worker w = c * K + k. Mux cores follow the workers.
    std::vector<CoreCoord> worker_cores(cores_all.begin(), cores_all.begin() + workers);
    std::vector<CoreCoord> mux_cores;
    if (use_mux) {
        mux_cores.assign(cores_all.begin() + workers, cores_all.begin() + workers + channels);
    }
    auto to_range = [](const std::vector<CoreCoord>& cores) {
        std::set<CoreRange> ranges;
        for (const auto& c : cores) {
            ranges.insert(CoreRange(c, c));
        }
        return CoreRangeSet(ranges);
    };
    const CoreRangeSet worker_range = to_range(worker_cores);
    const CoreRangeSet mux_range = to_range(mux_cores);
    tt::tt_fabric::FabricMuxConfig mux_config(
        /*num_full_size_channels=*/senders_per_channel,
        /*num_header_only_channels=*/0,
        /*num_buffers_full_size_channel=*/2,
        /*num_buffers_header_only_channel=*/0,
        /*buffer_size_bytes_full_size_channel=*/tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes(),
        /*base_l1_address=*/mesh_device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1));

    std::vector<uint32_t> dest_chip_ids, dest_mesh_ids;
    for (const auto& coord : tt::tt_metal::distributed::MeshCoordinateRange(mesh_shape)) {
        const auto node = mesh_device->get_fabric_node_id(coord);
        dest_chip_ids.push_back(static_cast<uint32_t>(node.chip_id));
        dest_mesh_ids.push_back(static_cast<uint32_t>(*node.mesh_id));
    }

    const auto ready_sem = *operation_attributes.ready_semaphore;
    const auto done_sem = *operation_attributes.data_valid_semaphore;

    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    for (const auto& coord : tensor_coords.coords()) {
        tt::tt_metal::Program program;
        const uint32_t my_rank = ttnn::operations::ccl::common::get_linearized_index(coord, mesh_view);
        const auto [neighbors, directions] = ttnn::operations::ccl::common::get_neighbors(
            mesh_view, coord, tt::tt_fabric::Topology::Torus, std::nullopt);
        TT_FATAL(
            std::all_of(directions.begin(), directions.end(), [](bool d) { return d; }),
            "sparse push (prototype) needs all four torus neighbours");

        auto cb = [&](uint32_t id, uint32_t page, uint32_t pages, tt::DataFormat df) {
            tt::tt_metal::CircularBufferConfig cfg =
                tt::tt_metal::CircularBufferConfig(page * pages, {{id, df}}).set_page_size(id, page);
            tt::tt_metal::CreateCircularBuffer(program, worker_range, cfg);
        };
        cb(kData, packet_bytes, kPacketSlots, tt::DataFormat::RawUInt8);
        cb(kMeta, kMetaEntryBytes, kPacketSlots, tt::DataFormat::RawUInt32);
        // Table mode: requester's per-owner counts (256 B) plus a window of 16 bitmap blocks (128 B each).
        cb(kRuns, from_table ? 4096 : kBlockWords * sizeof(uint32_t), 1, tt::DataFormat::RawUInt32);
        cb(kScalar, kScalarCbBytes, 1, tt::DataFormat::RawUInt32);
        cb(kHeaders, tt::tt_fabric::get_tt_fabric_packet_header_size_bytes(), 1 + 4, tt::DataFormat::RawUInt32);

        std::vector<uint32_t> reader_ct;
        if (from_table) {
            const auto layout =
                SparsePushPlanLayout::make(operation_attributes.push_max_rows, operation_attributes.push_stripe_chunk);
            TT_FATAL(
                runs_tensor.padded_shape()[2] == layout.pages * num_devices,
                "push table has {} pages, expected {} per device",
                runs_tensor.padded_shape()[2],
                layout.pages);
            reader_ct = {
                page_bytes,
                pages_per_packet,
                num_dram_banks,
                pages_per_slot,
                batch_from_meta ? 1u : 0u,
                num_devices,
                operation_attributes.push_stripe_chunk,
                layout.num_blocks,
                layout.pages,
                SparsePushPlanLayout::kOwnerOff,
                SparsePushPlanLayout::kBitmapOff,
                SparsePushPlanLayout::kOwnerPrefixOff,
                operation_attributes.push_part};
        } else {
            reader_ct = {
                page_bytes,
                pages_per_packet,
                num_dram_banks,
                out_rows_per_rank,
                pages_per_slot,
                batch_from_meta ? 1u : 0u,
                kBlockWords,
                runs_page_bytes,
            };
        }
        tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(reader_ct);
        tt::tt_metal::TensorAccessorArgs(runs_tensor.buffer()).append_to(reader_ct);
        if (batch_from_meta) {
            tt::tt_metal::TensorAccessorArgs(tensor_args.input_batch_index_tensor->buffer()).append_to(reader_ct);
        }
        std::vector<uint32_t> writer_ct = {page_bytes, num_devices, my_rank, workers, packet_bytes, links};
        tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(writer_ct);
        writer_ct.push_back(use_mux ? 1u : 0u);
        if (use_mux) {
            ttnn::ccl::fabric_mux_connection_ct_args(
                senders_per_channel, tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL, mux_config, writer_ct);
        } else {
            writer_ct.insert(writer_ct.end(), {1, 0, 0, 0, 1});
        }
        // Both kernels look up each destination's first hop in the fabric routing table.
        std::map<std::string, std::string> writer_defines = {
            {"DEST_CHIP_ID", ttnn::operations::ccl::common::stringify(dest_chip_ids)},
            {"DEST_MESH_ID", ttnn::operations::ccl::common::stringify(dest_mesh_ids)},
            {"FABRIC_2D", "1"},
        };
        // Profiling switch for the prototype: SPARSE_PUSH_SKIP=reads|sends times one side without the other.
        std::map<std::string, std::string> reader_defines = writer_defines;
        if (const char* skip = std::getenv("SPARSE_PUSH_SKIP")) {
            (std::string(skip) == "reads" ? reader_defines : writer_defines)["SPARSE_PUSH_SKIP"] = "1";
        }
        const std::string kdir = "ttnn/cpp/ttnn/operations/experimental/high_bw_all_gather/device/kernels/";
        auto reader = tt::tt_metal::CreateKernel(
            program,
            kdir + (from_table ? "sparse_push_table_reader.cpp" : "sparse_push_reader.cpp"),
            worker_range,
            tt::tt_metal::ReaderDataMovementConfig(reader_ct, reader_defines));
        auto writer = tt::tt_metal::CreateKernel(
            program,
            kdir + "sparse_push_writer.cpp",
            worker_range,
            tt::tt_metal::WriterDataMovementConfig(writer_ct, writer_defines));

        tt::tt_metal::SetCommonRuntimeArgs(
            program,
            reader,
            {input_tensor.buffer()->address(),
             runs_tensor.buffer()->address(),
             batch_from_meta ? tensor_args.input_batch_index_tensor->buffer()->address() : 0u,
             operation_attributes.batch_slot_num_layers,
             operation_attributes.batch_slot_layer_idx,
             operation_attributes.input_batch_index.value_or(0)});
        tt::tt_metal::SetCommonRuntimeArgs(
            program, writer, {output_tensor.buffer()->address(), ready_sem.address(), done_sem.address()});

        const auto sync_core = mesh_device->worker_core_from_logical_core(worker_cores[0]);
        const auto src_node = mesh_device->get_fabric_node_id(coord);
        // get_neighbors lists E, W, N, S, the eth_chan_directions order, so neighbors[dir] is a channel's hop.
        if (use_mux) {
            auto mux = tt::tt_metal::CreateKernel(
                program,
                "tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp",
                mux_range,
                tt::tt_metal::DataMovementConfig{
                    .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                    .noc = tt::tt_metal::NOC::RISCV_0_default,
                    .compile_args = mux_config.get_fabric_mux_compile_time_args(),
                    .opt_level = tt::tt_metal::KernelBuildOptLevel::O3});
            for (uint32_t c = 0; c < channels; ++c) {
                const uint32_t link = c / kDirections, dir = c % kDirections;
                tt::tt_metal::SetRuntimeArgs(
                    program,
                    mux,
                    mux_cores[c],
                    mux_config.get_fabric_mux_run_time_args(
                        src_node, mesh_device->get_fabric_node_id(neighbors[dir]), link, program, mux_cores[c]));
            }
        }
        for (uint32_t w = 0; w < workers; ++w) {
            const uint32_t c = w / senders_per_channel, k = w % senders_per_channel;
            const uint32_t link = c / kDirections, dir = c % kDirections;
            tt::tt_metal::SetRuntimeArgs(
                program, reader, worker_cores[w], {link, dir, my_rank, k, senders_per_channel, links});
            std::vector<uint32_t> writer_rt = {
                w, dir, link, k, static_cast<uint32_t>(sync_core.x), static_cast<uint32_t>(sync_core.y)};
            for (uint32_t other = 0; other < workers; ++other) {
                const auto phys = mesh_device->worker_core_from_logical_core(worker_cores[other]);
                writer_rt.push_back(static_cast<uint32_t>(phys.x));
                writer_rt.push_back(static_cast<uint32_t>(phys.y));
            }
            if (use_mux) {
                ttnn::ccl::fabric_mux_connection_rt_args(
                    /*mux_connection_valid=*/true,
                    /*is_termination_master=*/k == 0,
                    tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
                    mesh_device->worker_core_from_logical_core(mux_cores[c]),
                    /*worker_id=*/k,
                    worker_cores[w],
                    mux_config,
                    program,
                    mesh_device->worker_core_from_logical_core(worker_cores[c * senders_per_channel]),
                    writer_rt);
            } else {
                tt::tt_fabric::append_fabric_connection_rt_args(
                    src_node,
                    mesh_device->get_fabric_node_id(neighbors[dir]),
                    link,
                    program,
                    worker_cores[w],
                    writer_rt);
            }
            tt::tt_metal::SetRuntimeArgs(program, writer, worker_cores[w], writer_rt);
        }

        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(program));
        shared_variables.emplace(
            ttnn::MeshCoordinateRange(coord),
            shared_variables_t{
                .worker_cores = worker_cores,
                .reader_kernel_id = reader,
                .writer_kernel_id = writer,
                .ready_sem = ready_sem,
                .done_sem = done_sem});
    }
    return cached_mesh_workload_t{std::move(workload), std::move(shared_variables)};
}

void HighBwSparsePushFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const HighBwAllGatherParams& operation_attributes,
    const HighBwAllGatherInputs& tensor_args,
    Tensor& output_tensor) {
    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        auto& vars = cached_workload.shared_variables.at(range);
        auto& reader = tt::tt_metal::GetCommonRuntimeArgs(program, vars.reader_kernel_id);
        reader[0] = tensor_args.input_tensor.buffer()->address();
        reader[1] = tensor_args.push_runs_tensor->buffer()->address();
        reader[2] =
            tensor_args.has_batch_index_metadata() ? tensor_args.input_batch_index_tensor->buffer()->address() : 0u;
        reader[3] = operation_attributes.batch_slot_num_layers;
        reader[4] = operation_attributes.batch_slot_layer_idx;
        reader[5] = operation_attributes.input_batch_index.value_or(0);
        auto& writer = tt::tt_metal::GetCommonRuntimeArgs(program, vars.writer_kernel_id);
        writer[0] = output_tensor.buffer()->address();
        writer[1] = operation_attributes.ready_semaphore->address();
        writer[2] = operation_attributes.data_valid_semaphore->address();
    }
}

}  // namespace ttnn::operations::experimental::high_bw_all_gather

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

namespace ttnn::operations::experimental::high_bw_all_gather {

namespace {

constexpr uint32_t kDirections = 4;  // E, W, N, S: one sender per (link, direction)
constexpr uint32_t kMaxWorkers = 8;
constexpr uint32_t kPacketSlots = 8;
constexpr uint32_t kMetaEntryBytes = 16;
constexpr uint32_t kBlockWords = 256;  // stream words per refill
constexpr uint32_t kScalarCbBytes = 64;

enum Cb : uint32_t { kData = 0, kMeta = 1, kRuns = 2, kScalar = 3, kHeaders = 4 };

struct WorkerKernels {
    tt::tt_metal::KernelHandle reader;
    tt::tt_metal::KernelHandle writer;
};

// PROTOTYPE switch: SPARSE_RELAY=1 selects the one-hop relay kernels (relay_push_*.cpp), which take one stream page
// per sender instead of one per link.
bool relay_mode() {
    const char* v = std::getenv("SPARSE_RELAY");
    return v != nullptr && std::string(v) != "0";
}

uint32_t num_links(const HighBwAllGatherInputs& tensor_args) {
    // push_runs_tensor per device: [1, 1, links, words]; one stream page per link, shared by its 4 senders.
    // Relay: [1, 1, links * 4, words], one page per (link, direction) sender.
    const uint32_t pages = tensor_args.push_runs_tensor->padded_shape()[-2];
    return relay_mode() ? pages / kDirections : pages;
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

    const bool relay = relay_mode();
    const uint32_t links = num_links(tensor_args);
    const uint32_t workers = links * kDirections;
    TT_FATAL(workers >= 1 && workers <= kMaxWorkers, "sparse push supports 1..2 links, got {}", links);
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

    const auto worker_cores_all =
        corerange_to_cores(operation_attributes.resolved_worker_core_grid, std::nullopt, true);
    TT_FATAL(worker_cores_all.size() >= workers, "sparse push needs {} worker cores", workers);
    std::vector<CoreCoord> worker_cores(worker_cores_all.begin(), worker_cores_all.begin() + workers);
    const CoreRangeSet worker_range(std::set<CoreRange>([&] {
        std::set<CoreRange> ranges;
        for (const auto& c : worker_cores) {
            ranges.insert(CoreRange(c, c));
        }
        return ranges;
    }()));

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
        cb(kRuns, kBlockWords * sizeof(uint32_t), 1, tt::DataFormat::RawUInt32);
        cb(kScalar, kScalarCbBytes, 1, tt::DataFormat::RawUInt32);
        cb(kHeaders, tt::tt_fabric::get_tt_fabric_packet_header_size_bytes(), 1 + 4, tt::DataFormat::RawUInt32);

        if (relay) {
            const uint32_t per_round = links * kDirections;
            std::vector<uint32_t> reader_ct = {
                page_bytes,
                pages_per_packet,
                pages_per_slot,
                batch_from_meta ? 1u : 0u,
                kBlockWords,
                runs_page_bytes,
                per_round,
            };
            tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(reader_ct);
            tt::tt_metal::TensorAccessorArgs(runs_tensor.buffer()).append_to(reader_ct);
            tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(reader_ct);
            if (batch_from_meta) {
                tt::tt_metal::TensorAccessorArgs(tensor_args.input_batch_index_tensor->buffer()).append_to(reader_ct);
            }
            std::vector<uint32_t> writer_ct = {page_bytes, workers, packet_bytes, per_round};
            tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(writer_ct);
            const std::map<std::string, std::string> defines = {{"FABRIC_2D", "1"}};
            const std::string kdir = "ttnn/cpp/ttnn/operations/experimental/high_bw_all_gather/device/kernels/";
            auto reader = tt::tt_metal::CreateKernel(
                program,
                kdir + "relay_push_reader.cpp",
                worker_range,
                tt::tt_metal::ReaderDataMovementConfig(reader_ct, defines));
            auto writer = tt::tt_metal::CreateKernel(
                program,
                kdir + "relay_push_writer.cpp",
                worker_range,
                tt::tt_metal::WriterDataMovementConfig(writer_ct, defines));
            tt::tt_metal::SetCommonRuntimeArgs(
                program,
                reader,
                {input_tensor.buffer()->address(),
                 runs_tensor.buffer()->address(),
                 batch_from_meta ? tensor_args.input_batch_index_tensor->buffer()->address() : 0u,
                 operation_attributes.batch_slot_num_layers,
                 operation_attributes.batch_slot_layer_idx,
                 operation_attributes.input_batch_index.value_or(0),
                 output_tensor.buffer()->address(),
                 done_sem.address()});
            tt::tt_metal::SetCommonRuntimeArgs(
                program, writer, {output_tensor.buffer()->address(), ready_sem.address(), done_sem.address()});
            const auto src_node = mesh_device->get_fabric_node_id(coord);
            for (uint32_t w = 0; w < workers; ++w) {
                const uint32_t link = w / kDirections, dir = w % kDirections;
                tt::tt_metal::SetRuntimeArgs(program, reader, worker_cores[w], {w});
                const auto nbr_node = mesh_device->get_fabric_node_id(neighbors[dir]);
                std::vector<uint32_t> writer_rt = {
                    w, link, static_cast<uint32_t>(nbr_node.chip_id), static_cast<uint32_t>(*nbr_node.mesh_id)};
                for (uint32_t other = 0; other < kMaxWorkers; ++other) {
                    const auto phys =
                        mesh_device->worker_core_from_logical_core(worker_cores[std::min(other, workers - 1)]);
                    writer_rt.push_back(static_cast<uint32_t>(phys.x));
                    writer_rt.push_back(static_cast<uint32_t>(phys.y));
                }
                tt::tt_fabric::append_fabric_connection_rt_args(
                    src_node, nbr_node, link, program, worker_cores[w], writer_rt);
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
                    .done_sem = done_sem,
                    .relay = true});
            continue;
        }

        std::vector<uint32_t> reader_ct = {
            page_bytes,
            pages_per_packet,
            num_dram_banks,
            out_rows_per_rank,
            pages_per_slot,
            batch_from_meta ? 1u : 0u,
            kBlockWords,
            runs_page_bytes,
        };
        tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(reader_ct);
        tt::tt_metal::TensorAccessorArgs(runs_tensor.buffer()).append_to(reader_ct);
        if (batch_from_meta) {
            tt::tt_metal::TensorAccessorArgs(tensor_args.input_batch_index_tensor->buffer()).append_to(reader_ct);
        }
        std::vector<uint32_t> writer_ct = {page_bytes, num_devices, my_rank, workers, packet_bytes, links};
        tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(writer_ct);
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
            kdir + "sparse_push_reader.cpp",
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
        for (uint32_t w = 0; w < workers; ++w) {
            const uint32_t link = w / kDirections, dir = w % kDirections;
            tt::tt_metal::SetRuntimeArgs(program, reader, worker_cores[w], {link, dir, my_rank});
            std::vector<uint32_t> writer_rt = {
                w, dir, link, static_cast<uint32_t>(sync_core.x), static_cast<uint32_t>(sync_core.y)};
            for (uint32_t other = 0; other < kMaxWorkers; ++other) {
                const auto phys =
                    mesh_device->worker_core_from_logical_core(worker_cores[std::min(other, workers - 1)]);
                writer_rt.push_back(static_cast<uint32_t>(phys.x));
                writer_rt.push_back(static_cast<uint32_t>(phys.y));
            }
            // get_neighbors lists E, W, N, S, the eth_chan_directions order, so neighbors[dir] is this sender's hop.
            tt::tt_fabric::append_fabric_connection_rt_args(
                src_node, mesh_device->get_fabric_node_id(neighbors[dir]), link, program, worker_cores[w], writer_rt);
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
        if (vars.relay) {
            reader[6] = output_tensor.buffer()->address();
            reader[7] = operation_attributes.data_valid_semaphore->address();
        }
        auto& writer = tt::tt_metal::GetCommonRuntimeArgs(program, vars.writer_kernel_id);
        writer[0] = output_tensor.buffer()->address();
        writer[1] = operation_attributes.ready_semaphore->address();
        writer[2] = operation_attributes.data_valid_semaphore->address();
    }
}

}  // namespace ttnn::operations::experimental::high_bw_all_gather

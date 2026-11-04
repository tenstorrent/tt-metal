// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_reduce_async_program_factory.hpp"
#include "ttnn/operations/ccl/shared_with_host/ccl_runtime_args.hpp"

#include <algorithm>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>

#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/experimental/ccl/llama_common.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/operations/ccl/common/types/ccl_types_args_emitters.hpp"
#include "ttnn/operations/ccl/common/host/ccl_command_stream_builders.hpp"

#include "ttnn/operations/ccl/common/uops/command_lowering.hpp"

#include "ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
#include "ttnn/operations/ccl/common/host/command_backend_runtime_args_overrider.hpp"
#include <sstream>
#include <type_traits>
#include <ranges>
#include <optional>

using namespace tt::constants;

namespace ttnn {

CoreRangeSet cores_to_corerangeset(const std::vector<CoreCoord>& cores) {
    std::vector<CoreRange> core_ranges;
    core_ranges.reserve(cores.size());
    for (const auto& core : cores) {
        core_ranges.push_back(CoreRange(core));
    }
    return CoreRangeSet(core_ranges);
}

std::tuple<CoreRangeSet, std::vector<CoreCoord>> ar_choose_worker_cores(
    size_t num_links, size_t num_workers_per_link, const CoreRangeSet& available_cores) {
    std::tuple<CoreRangeSet, std::vector<CoreCoord>> result;
    CoreRangeSet sender_worker_core_range;
    const size_t num_workers_preferred = num_workers_per_link * num_links;
    if (available_cores.num_cores() < num_workers_preferred) {
        log_warning(
            tt::LogOp,
            "AllGather is being launched on a subdevice with fewer worker cores available than ideal. Ideally {} "
            "cores ({} per link and {} links) are made available but only {} are available. This may lead to "
            "performance loss.",
            num_workers_preferred,
            num_workers_per_link,
            num_links,
            available_cores.num_cores());
    }
    for (const auto& cr : available_cores.ranges()) {
        auto start = cr.start_coord;
        auto end = cr.end_coord;
        for (size_t y = start.y; y <= end.y; y++) {
            for (size_t x = start.x; x <= end.x; x++) {
                sender_worker_core_range =
                    sender_worker_core_range.merge(CoreRangeSet(CoreRange(CoreCoord(x, y), CoreCoord(x, y))));
                if (sender_worker_core_range.num_cores() == num_workers_preferred) {
                    break;
                }
            }
            if (sender_worker_core_range.num_cores() == num_workers_preferred) {
                break;
            }
        }
        if (sender_worker_core_range.num_cores() == num_workers_preferred) {
            break;
        }
    }
    return {sender_worker_core_range, corerange_to_cores(sender_worker_core_range, std::nullopt, true)};
}

}  // namespace ttnn

namespace ttnn::experimental::prim {

namespace {

// Descriptor kernel indices, fixed by push order in create_descriptor.
constexpr uint32_t kReductionReader = 0;
constexpr uint32_t kReductionCompute = 1;
constexpr uint32_t kWorkerReader = 2;
constexpr uint32_t kWorkerWriter = 3;

static_assert(ttnn::ccl::AllReduceReaderCommonArgs::input == 0);
static_assert(ttnn::ccl::AllReduceSemaphoreCommonArgs::semaphore == 0);

tt::tt_metal::KernelDescriptor::RTArgList uint32_rt_args(const std::vector<uint32_t>& args) {
    tt::tt_metal::KernelDescriptor::RTArgList list;
    list.reserve(args.size());
    for (uint32_t value : args) {
        list.push_back(value);
    }
    return list;
}

void emplace_uint32_runtime_args(
    tt::tt_metal::KernelDescriptor& kernel, const CoreCoord& core, const std::vector<uint32_t>& args) {
    kernel.emplace_runtime_args(core, uint32_rt_args(args));
}

void emplace_uint32_runtime_args(
    tt::tt_metal::KernelDescriptor& kernel, const CoreRangeSet& cores, const std::vector<uint32_t>& args) {
    const auto list = uint32_rt_args(args);
    for (const auto& core : tt::tt_metal::corerange_to_cores(cores, std::nullopt, true)) {
        kernel.emplace_runtime_args(core, list);
    }
}

// One-shot mode (AllReduceAsyncParams::one_shot): extra common runtime args after index 0 (which keeps the
// default path's meaning: input address for the sender reader, semaphore address for the writer / receiver).
constexpr uint32_t kOneShotReaderOutputArg = 1;    // sender reader: output buffer (all-gather source)
constexpr uint32_t kOneShotWriterOutputArg = 1;    // sender writer: output buffer (all-gather destination)
constexpr uint32_t kOneShotReceiverInputArg = 1;   // receiver: input buffer (owner's local contribution)
constexpr uint32_t kOneShotRsBatchChunks = 4;      // sender reader: chunks per read barrier
constexpr uint32_t kOneShotRoleNone = 0;           // receiver roles
constexpr uint32_t kOneShotRoleGather = 1;
constexpr uint32_t kOneShotRoleOwner = 2;

uint32_t pack_noc_xy(const CoreCoord& virtual_core) {
    return (static_cast<uint32_t>(virtual_core.x) << 16) | static_cast<uint32_t>(virtual_core.y);
}

// One-shot all-reduce: reduce-scatter by direct unicast into the slice owners' reduction buffers, owner-only
// reduction with the default kernel, then a line-multicast all-gather of the reduced slices into every peer's
// output shards. Output core c belongs to slice c / (cores / ring_size); device d owns slice d. Each output core's
// shard (one 32-row tile row of chunk_tiles tiles) is one "chunk" and travels on link (rank in slice + owner) %
// num_links, so a slice's chunks alternate links and every directed link carries about the same packet count.
//
// Per sender (one per link and direction): the reader stages the input tiles of every remote-owned chunk of its link
// and direction (farthest destination first) in the RS CB; the writer unicasts each chunk to its owner's buffer slot
// my_chip with one fused write + semaphore increment (the default path's slot layout, so the owner's reduction sees
// exactly the default path's operands in the same order). Owner cores copy their own chunk into slot my_chip, wait for
// ring_size - 1 increments, reduce (same compute kernel and slot order as the default: bit-identical), then increment
// the ag_ready semaphore of each of their link's senders. Each sender's reader then copies the link's reduced chunks
// into the AG CB and its writer multicasts each one in its direction (the default path's static-alternate
// line-multicast ranges) into the same core's output shard on every peer with a fused semaphore increment. Non-owner
// output cores wait for that one increment. RS increments land only on owner cores and AG increments only on
// non-owner cores, so the caller's single global semaphore serves both; every receiving core resets it to 0 as in the
// default path.
//
// The persistent buffer cannot be overwritten early: a peer's next all-reduce starts only after it received every
// owner's reduced slice of this one, i.e. after every owner finished reading its buffer slots.
tt::tt_metal::ProgramDescriptor create_one_shot_descriptor(
    const AllReduceAsyncParams& operation_attributes,
    const AllReduceAsyncInputs& tensor_args,
    Tensor& output_tensor,
    const ttnn::MeshCoordinate& coord) {
    using tt::tt_metal::corerange_to_cores;
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& buffer_tensor = tensor_args.buffer_tensor;
    auto* mesh_device = input_tensor.device();
    const uint32_t ring_size = operation_attributes.ring_size;
    const uint32_t num_links = operation_attributes.num_links;
    const auto topology = operation_attributes.topology;
    const auto& sub_device_id = operation_attributes.sub_device_id;

    const uint32_t device_index =
        ttnn::ccl::get_linearized_index_from_physical_coord(input_tensor, coord, operation_attributes.cluster_axis);
    const std::optional<MeshCoordinate> forward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
        input_tensor, coord, 1, topology, operation_attributes.cluster_axis);
    const std::optional<MeshCoordinate> backward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
        input_tensor, coord, -1, topology, operation_attributes.cluster_axis);
    auto [num_targets_forward, num_targets_backward] =
        ttnn::ccl::get_forward_backward_line_mcast_distance(ring_size, device_index, topology, true);
    if (!forward_coord.has_value()) {
        num_targets_forward = 0;
    }
    if (!backward_coord.has_value()) {
        num_targets_backward = 0;
    }
    TT_FATAL(
        num_targets_forward + num_targets_backward == ring_size - 1,
        "all_reduce_async one_shot: device {} reaches {} + {} peers, expected {}",
        device_index,
        num_targets_forward,
        num_targets_backward,
        ring_size - 1);

    // Pages, chunks, slices.
    const tt::DataFormat in_df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    const tt::DataFormat out_df = tt::tt_metal::datatype_to_dataformat_converter(operation_attributes.dtype);
    const uint32_t in_page_size = input_tensor.tensor_spec().tile().get_tile_size(in_df);
    const uint32_t out_page_size = output_tensor.tensor_spec().tile().get_tile_size(out_df);
    const auto& in_shard_spec = input_tensor.memory_config().shard_spec().value();
    const auto& out_shard_spec = output_tensor.memory_config().shard_spec().value();
    const uint32_t in_shard_tiles = in_shard_spec.shape[1] / TILE_WIDTH;
    const uint32_t chunk_tiles = out_shard_spec.shape[1] / TILE_WIDTH;
    const auto in_cores = corerange_to_cores(in_shard_spec.grid, std::nullopt, true);
    const auto out_cores = corerange_to_cores(out_shard_spec.grid, std::nullopt, true);
    const uint32_t num_out_cores = out_cores.size();
    TT_FATAL(
        num_out_cores % ring_size == 0 && in_shard_tiles > 0 && chunk_tiles > 0 &&
            in_cores.size() * in_shard_tiles >= num_out_cores * chunk_tiles,
        "all_reduce_async one_shot: {} output cores x {} tiles cannot be split into {} slices / read from {} input "
        "cores x {} tiles",
        num_out_cores,
        chunk_tiles,
        ring_size,
        in_cores.size(),
        in_shard_tiles);
    const uint32_t cores_per_slice = num_out_cores / ring_size;
    auto owner_of = [&](uint32_t c) { return c / cores_per_slice; };
    auto link_of = [&](uint32_t c) { return ((c % cores_per_slice) + owner_of(c)) % num_links; };

    const size_t packet_size_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    const uint32_t in_pages_per_packet = packet_size_bytes / in_page_size;
    const uint32_t out_pages_per_packet = packet_size_bytes / out_page_size;
    TT_FATAL(
        in_pages_per_packet > 0 && out_pages_per_packet > 0,
        "all_reduce_async one_shot: fabric packet of {} B is smaller than a tile ({} / {} B)",
        packet_size_bytes,
        in_page_size,
        out_page_size);

    // Sender workers: the default path's placement (one per link, outside the output cores' bounding boxes).
    auto sub_device_cores = mesh_device->worker_cores(
        tt::tt_metal::HalProgrammableCoreType::TENSIX,
        sub_device_id.value_or(mesh_device->get_sub_device_ids().at(0)));
    const CoreRangeSet output_tensor_cores = out_shard_spec.grid;
    std::vector<CoreRange> output_bboxes;
    for (const auto& cr : sub_device_cores.ranges()) {
        const auto intersection = output_tensor_cores.intersection(cr);
        if (!intersection.empty()) {
            output_bboxes.push_back(intersection.bounding_box());
        }
    }
    const CoreRangeSet output_cores_all(output_bboxes);
    const auto available_cores = sub_device_cores.subtract(output_cores_all);
    // One sender per (link, direction): each sender serializes only its own direction's packets and holds only that
    // direction's fabric connection (one worker connection per router, as on the default path). Sender of (link, d) is
    // sender_worker_cores[link * num_dirs + d] over the directions this device reaches (both on a ring).
    std::vector<uint32_t> dirs;  // 0 = forward, 1 = backward
    if (num_targets_forward > 0) {
        dirs.push_back(0);
    }
    if (num_targets_backward > 0) {
        dirs.push_back(1);
    }
    const uint32_t num_dirs = dirs.size();
    auto [sender_worker_core_range, sender_worker_cores] =
        operation_attributes.use_optimal_ccl_for_llama
            ? llama_specific::get_custom_worker_core_placement(num_links * num_dirs)
            : ar_choose_worker_cores(num_links, /*num_workers_per_link=*/num_dirs, available_cores);
    TT_FATAL(
        sender_worker_cores.size() >= num_links * num_dirs,
        "all_reduce_async one_shot: found {} sender cores for {} links x {} directions",
        sender_worker_cores.size(),
        num_links,
        num_dirs);
    const auto output_cores_unused = output_cores_all.subtract(output_tensor_cores);
    const auto all_cores = output_cores_all.merge(sender_worker_core_range);

    // Input tiles of output core c as (input core, first tile in its shard, tile count) runs.
    auto input_segments = [&](uint32_t c) {
        std::vector<uint32_t> segs;  // num_segs, then (xy, offset, count) per run
        segs.push_back(0);
        uint32_t t = c * chunk_tiles;
        const uint32_t end = t + chunk_tiles;
        while (t < end) {
            const uint32_t ic = t / in_shard_tiles;
            const uint32_t off = t % in_shard_tiles;
            const uint32_t n = std::min(in_shard_tiles - off, end - t);
            segs.push_back(pack_noc_xy(mesh_device->worker_core_from_logical_core(in_cores.at(ic))));
            segs.push_back(off);
            segs.push_back(n);
            segs[0]++;
            t += n;
        }
        return segs;
    };
    // Route from this device to slice owner s: (0 = forward / 1 = backward, hops). The static-alternate ranges are
    // the default path's multicast ranges, so a ring's opposite device is reached the same way both phases.
    auto route_to = [&](uint32_t s) -> std::pair<uint32_t, uint32_t> {
        const uint32_t fwd_hops = (s + ring_size - device_index) % ring_size;
        if (fwd_hops <= num_targets_forward) {
            return {0u, fwd_hops};
        }
        const uint32_t bwd_hops = ring_size - fwd_hops;
        TT_FATAL(
            bwd_hops <= num_targets_backward, "all_reduce_async one_shot: device {} cannot reach {}", device_index, s);
        return {1u, bwd_hops};
    };

    // Per-link chunk lists.
    std::vector<std::vector<uint32_t>> rs_chunks(num_links);  // remote-owned output cores, in send order
    std::vector<std::vector<uint32_t>> ag_chunks(num_links);  // this device's owned output cores
    for (uint32_t link = 0; link < num_links; link++) {
        std::vector<uint32_t> per_dir[2];
        for (uint32_t c = 0; c < num_out_cores; c++) {
            if (link_of(c) != link) {
                continue;
            }
            if (owner_of(c) == device_index) {
                ag_chunks[link].push_back(c);
            } else {
                per_dir[route_to(owner_of(c)).first].push_back(c);
            }
        }
        // Farthest owner first in each direction, then alternate directions chunk by chunk.
        for (auto& v : per_dir) {
            std::stable_sort(v.begin(), v.end(), [&](uint32_t a, uint32_t b) {
                return route_to(owner_of(a)).second > route_to(owner_of(b)).second;
            });
        }
        for (size_t i = 0; i < std::max(per_dir[0].size(), per_dir[1].size()); i++) {
            for (const auto& v : per_dir) {
                if (i < v.size()) {
                    rs_chunks[link].push_back(v[i]);
                }
            }
        }
    }
    size_t max_ag_chunks = 1;
    for (uint32_t link = 0; link < num_links; link++) {
        max_ag_chunks = std::max(max_ag_chunks, ag_chunks[link].size());
    }

    tt::tt_metal::ProgramDescriptor desc;

    // CBs. RS staging: double-buffered batches of chunks (a batch never straddles the ring's end; only the last batch
    // can be partial). AG staging: every reduced chunk of a link at once.
    const uint32_t rs_cb_index = tt::CBIndex::c_0;
    const uint32_t reduction_cb_index = tt::CBIndex::c_1;
    const uint32_t out_cb_index = tt::CBIndex::c_2;
    const uint32_t reserved_packet_header_cb_index = tt::CBIndex::c_3;
    const uint32_t residual_cb_index = tt::CBIndex::c_4;
    const uint32_t ag_cb_index = tt::CBIndex::c_5;
    const uint32_t rs_cb_pages = 2 * kOneShotRsBatchChunks * chunk_tiles;
    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = rs_cb_pages * in_page_size,
        .core_ranges = sender_worker_core_range,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(rs_cb_index),
            .data_format = in_df,
            .page_size = in_page_size,
        }}},
    });
    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = static_cast<uint32_t>(max_ag_chunks) * chunk_tiles * out_page_size,
        .core_ranges = sender_worker_core_range,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(ag_cb_index),
            .data_format = out_df,
            .page_size = out_page_size,
        }}},
    });
    constexpr uint32_t num_packet_headers_storable = 4;
    const auto packet_header_size_bytes =
        static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_packet_header_size_bytes());
    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = num_packet_headers_storable * packet_header_size_bytes,
        .core_ranges = sender_worker_core_range,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(reserved_packet_header_cb_index),
            .data_format = tt::DataFormat::RawUInt32,
            .page_size = packet_header_size_bytes,
        }}},
    });
    // Reduction CB over the persistent buffer: the owners' operands; on the senders only its base address is used
    // (the remote slot address, identical on every core and device).
    const uint32_t reduction_cb_page_size = output_tensor.tensor_spec().tile().get_tile_size(in_df);
    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = chunk_tiles * ring_size * reduction_cb_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(reduction_cb_index),
            .data_format = in_df,
            .page_size = reduction_cb_page_size,
        }}},
        .buffer = buffer_tensor.buffer(),
    });
    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = chunk_tiles * out_page_size,
        .core_ranges = output_tensor_cores,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(out_cb_index),
            .data_format = out_df,
            .page_size = out_page_size,
        }}},
        .buffer = output_tensor.buffer(),
    });
    const bool fuse_residual = tensor_args.residual_tensor.has_value();
    if (fuse_residual) {
        const auto& residual_tensor = tensor_args.residual_tensor.value();
        const uint32_t residual_page_size = residual_tensor.tensor_spec().tile().get_tile_size(in_df);
        desc.cbs.push_back(tt::tt_metal::CBDescriptor{
            .total_size = chunk_tiles * residual_page_size,
            .core_ranges = output_tensor_cores,
            .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(residual_cb_index),
                .data_format = in_df,
                .page_size = residual_page_size,
            }}},
            .buffer = residual_tensor.buffer(),
        });
    }

    // Owners -> their link sender: "my reduced chunk is in my output shard".
    const uint32_t ag_ready_sem_id = static_cast<uint32_t>(desc.semaphores.size());
    desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
        .id = ag_ready_sem_id,
        .core_type = tt::CoreType::WORKER,
        .core_ranges = all_cores,
        .initial_value = 0,
    });

    // Kernels, in the default path's index order (kReductionReader, kReductionCompute, kWorkerReader, kWorkerWriter).
    tt::tt_metal::KernelDescriptor receiver_desc;
    receiver_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/kernels/dataflow/"
        "one_shot_receiver.cpp";
    receiver_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    receiver_desc.core_ranges = output_cores_all;
    receiver_desc.compile_time_args = {
        reduction_cb_index,
        out_cb_index,
        static_cast<uint32_t>(fuse_residual),
        residual_cb_index,
        fuse_residual ? chunk_tiles : 0u,
        chunk_tiles,
        in_page_size,
        ring_size,
        ag_ready_sem_id,
    };
    receiver_desc.config = tt::tt_metal::DataMovementConfigDescriptor{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
        .noc = tt::tt_metal::NOC::NOC_1,
        .noc_mode = tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC,
    };
    desc.kernels.push_back(std::move(receiver_desc));

    tt::tt_metal::KernelDescriptor compute_desc;
    compute_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/kernels/compute/"
        "reduction.cpp";
    compute_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = output_cores_all;
    compute_desc.compile_time_args = {
        reduction_cb_index,
        out_cb_index,
        static_cast<uint32_t>(fuse_residual),
        residual_cb_index,
    };
    tt::tt_metal::ComputeConfigDescriptor compute_config{
        .math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
        .math_approx_mode = false,
    };
    if (operation_attributes.fp32_dest_acc) {
        compute_config.fp32_dest_acc_en = true;
        compute_config.dst_full_sync_en = true;
    }
    compute_desc.config = compute_config;
    desc.kernels.push_back(std::move(compute_desc));

    tt::tt_metal::KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/kernels/dataflow/"
        "one_shot_reader.cpp";
    reader_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = sender_worker_core_range;
    reader_desc.compile_time_args = {
        rs_cb_index,
        ag_cb_index,
        in_page_size,
        out_page_size,
        chunk_tiles,
        kOneShotRsBatchChunks,
        ag_ready_sem_id,
    };
    reader_desc.config = tt::tt_metal::DataMovementConfigDescriptor{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
        .noc = tt::tt_metal::NOC::NOC_1,
        .noc_mode = tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC,
    };
    desc.kernels.push_back(std::move(reader_desc));

    tt::tt_metal::KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/kernels/dataflow/"
        "one_shot_writer.cpp";
    writer_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = sender_worker_core_range;
    writer_desc.compile_time_args = {
        device_index,
        reserved_packet_header_cb_index,
        rs_cb_index,
        ag_cb_index,
        in_page_size,
        out_page_size,
        chunk_tiles,
        in_pages_per_packet,
        out_pages_per_packet,
        reduction_cb_index,
    };
    writer_desc.config = tt::tt_metal::DataMovementConfigDescriptor{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
        .noc = tt::tt_metal::NOC::NOC_0,
        .noc_mode = tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC,
    };
    desc.kernels.push_back(std::move(writer_desc));

    // Receiver / compute runtime args per output core.
    const std::vector<uint32_t> no_work_rt_args = {kOneShotRoleNone, 0u, 0u};
    for (uint32_t c = 0; c < num_out_cores; c++) {
        const CoreCoord& core = out_cores[c];
        if (owner_of(c) == device_index) {
            std::vector<uint32_t> args = {
                kOneShotRoleOwner,
                ring_size - 1,  // remote reduce-scatter increments
                device_index,   // slot of the local contribution
                num_dirs,       // senders of this core's link, each told when the reduced chunk is ready
            };
            for (uint32_t d = 0; d < num_dirs; d++) {
                args.push_back(pack_noc_xy(
                    mesh_device->worker_core_from_logical_core(sender_worker_cores.at(link_of(c) * num_dirs + d))));
            }
            const auto segs = input_segments(c);
            args.insert(args.end(), segs.begin(), segs.end());
            emplace_uint32_runtime_args(desc.kernels[kReductionReader], core, args);
            emplace_uint32_runtime_args(desc.kernels[kReductionCompute], core, {1u, ring_size, chunk_tiles});
        } else {
            emplace_uint32_runtime_args(desc.kernels[kReductionReader], core, {kOneShotRoleGather, 1u});
            emplace_uint32_runtime_args(desc.kernels[kReductionCompute], core, no_work_rt_args);
        }
    }
    if (!output_cores_unused.empty()) {
        emplace_uint32_runtime_args(desc.kernels[kReductionReader], output_cores_unused, no_work_rt_args);
        emplace_uint32_runtime_args(desc.kernels[kReductionCompute], output_cores_unused, no_work_rt_args);
    }

    // Sender runtime args per (link, direction): that direction's reduce-scatter chunks (in the link's send order) and
    // every reduced chunk of the link, multicast in that direction only.
    for (uint32_t link = 0; link < num_links; link++) {
        for (uint32_t d = 0; d < num_dirs; d++) {
            const uint32_t dir = dirs[d];
            const CoreCoord& core = sender_worker_cores[link * num_dirs + d];
            std::vector<uint32_t> my_rs;
            for (const uint32_t c : rs_chunks[link]) {
                if (route_to(owner_of(c)).first == dir) {
                    my_rs.push_back(c);
                }
            }
            std::vector<uint32_t> reader_args = {
                static_cast<uint32_t>(my_rs.size()), static_cast<uint32_t>(ag_chunks[link].size())};
            std::vector<uint32_t> writer_args = {
                static_cast<uint32_t>(my_rs.size()),
                static_cast<uint32_t>(ag_chunks[link].size()),
                dir == 0 ? num_targets_forward : 0u,
                dir == 1 ? num_targets_backward : 0u,
            };
            for (const uint32_t c : my_rs) {
                const auto segs = input_segments(c);
                reader_args.insert(reader_args.end(), segs.begin(), segs.end());
                writer_args.push_back(pack_noc_xy(mesh_device->worker_core_from_logical_core(out_cores[c])));
                writer_args.push_back((dir << 8) | route_to(owner_of(c)).second);
            }
            for (const uint32_t c : ag_chunks[link]) {
                const uint32_t xy = pack_noc_xy(mesh_device->worker_core_from_logical_core(out_cores[c]));
                reader_args.push_back(xy);
                writer_args.push_back(xy);
            }
            emplace_uint32_runtime_args(desc.kernels[kWorkerReader], core, reader_args);

            writer_args.push_back(dir == 0);
            if (dir == 0) {
                tt::tt_fabric::append_fabric_connection_rt_args<tt::tt_metal::ProgramDescriptor>(
                    mesh_device->get_fabric_node_id(coord),
                    mesh_device->get_fabric_node_id(forward_coord.value()),
                    link,
                    desc,
                    core,
                    writer_args);
            }
            writer_args.push_back(dir == 1);
            if (dir == 1) {
                tt::tt_fabric::append_fabric_connection_rt_args<tt::tt_metal::ProgramDescriptor>(
                    mesh_device->get_fabric_node_id(coord),
                    mesh_device->get_fabric_node_id(backward_coord.value()),
                    link,
                    desc,
                    core,
                    writer_args);
            }
            emplace_uint32_runtime_args(desc.kernels[kWorkerWriter], core, writer_args);
        }
    }

    const uint32_t semaphore_address = static_cast<uint32_t>(operation_attributes.semaphore.address());
    desc.kernels[kWorkerReader].emplace_common_runtime_args({input_tensor.buffer(), output_tensor.buffer()});
    // smuggled-rta-ok: caller GlobalSemaphore excluded from the program-cache key; re-applied by
    // override_runtime_arguments (as on the default path).
    desc.kernels[kWorkerWriter].emplace_common_runtime_args({semaphore_address, output_tensor.buffer()});
    desc.kernels[kReductionReader].emplace_common_runtime_args({semaphore_address, input_tensor.buffer()});
    return desc;
}

}  // namespace

tt::tt_metal::ProgramDescriptor AllReduceAsyncMeshWorkloadFactory::create_descriptor(
    const AllReduceAsyncParams& operation_attributes,
    const AllReduceAsyncInputs& tensor_args,
    Tensor& output_tensor,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    TT_FATAL(
        mesh_dispatch_coordinate.has_value(),
        "AllReduceAsyncMeshWorkloadFactory::create_descriptor requires a mesh dispatch coordinate");
    const auto& coord = mesh_dispatch_coordinate.value();
    if (operation_attributes.one_shot) {
        return create_one_shot_descriptor(operation_attributes, tensor_args, output_tensor, coord);
    }

    const auto& input_tensor = tensor_args.input_tensor;
    const auto& buffer_tensor = tensor_args.buffer_tensor;

    log_debug(tt::LogOp, "all_reduce_async create_program at physical coordinate {} is called", coord);

    uint32_t device_index =
        ttnn::ccl::get_linearized_index_from_physical_coord(input_tensor, coord, operation_attributes.cluster_axis);

    std::optional<MeshCoordinate> forward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
        input_tensor, coord, 1, operation_attributes.topology, operation_attributes.cluster_axis);

    std::optional<MeshCoordinate> backward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
        input_tensor, coord, -1, operation_attributes.topology, operation_attributes.cluster_axis);

    auto input_tensor_shape = input_tensor.padded_shape();

    auto input_tensor_memory_config = input_tensor.memory_config();
    auto output_tensor_memory_config = output_tensor.memory_config();
    [[maybe_unused]] uint32_t input_shard_num_cores = input_tensor_memory_config.shard_spec()->grid.num_cores();
    [[maybe_unused]] uint32_t output_shard_num_cores = output_tensor_memory_config.shard_spec()->grid.num_cores();

    log_debug(tt::LogOp, "input_tensor_shape: {}", input_tensor_shape);
    log_debug(tt::LogOp, "input_tensor_memory_config: {}", input_tensor_memory_config);
    log_debug(tt::LogOp, "output_tensor_memory_config: {}", output_tensor_memory_config);
    log_debug(tt::LogOp, "input_shard_num_cores: {}", input_shard_num_cores);
    log_debug(tt::LogOp, "output_shard_num_cores: {}", output_shard_num_cores);
    log_debug(
        tt::LogOp,
        "input_tensor_memory_config.shard_spec()->shape: {}",
        input_tensor_memory_config.shard_spec()->shape);
    log_debug(
        tt::LogOp,
        "output_tensor_memory_config.shard_spec()->shape: {}",
        output_tensor_memory_config.shard_spec()->shape);

    log_debug(tt::LogOp, "Running TG Llama specific all_reduce_async_minimal_multi_core_with_workers");
    // previously parameters from all_reduce_async_minimal_multi_core_with_workers
    const auto& output_dtype = operation_attributes.dtype;
    const auto& num_links = operation_attributes.num_links;
    const auto& ring_size = operation_attributes.ring_size;
    const auto& topology = operation_attributes.topology;
    const auto& semaphore = operation_attributes.semaphore;
    const auto& sub_device_id = operation_attributes.sub_device_id;
    const auto& use_noc1_only = operation_attributes.use_noc1_only;
    const auto& use_optimal_ccl_for_llama = operation_attributes.use_optimal_ccl_for_llama;

    // KERNEL CREATION
    tt::tt_metal::NOC reader_noc = tt::tt_metal::NOC::NOC_1;
    tt::tt_metal::NOC writer_noc = use_noc1_only ? tt::tt_metal::NOC::NOC_1 : tt::tt_metal::NOC::NOC_0;

    tt::tt_metal::ProgramDescriptor desc;
    auto* mesh_device = input_tensor.device();
    [[maybe_unused]] bool is_first_chip = device_index == 0;
    [[maybe_unused]] bool is_last_chip = device_index == ring_size - 1;
    log_trace(
        tt::LogOp, "DEBUG: device coord: {}, is_first_chip: {}, is_last_chip: {}", coord, is_first_chip, is_last_chip);

    // Get OP Config, topology config
    std::vector<Tensor> input_tensors = {input_tensor};
    std::vector<Tensor> output_tensors = {output_tensor};
    const auto& op_config = ttnn::ccl::CCLOpConfig(input_tensors, output_tensors, topology);
    auto [num_targets_forward, num_targets_backward] =
        ttnn::ccl::get_forward_backward_line_mcast_distance(ring_size, device_index, topology, true);
    auto [forward_args, backward_args] = ttnn::ccl::get_forward_backward_line_mcast_configuration(
        coord, forward_coord, backward_coord, num_targets_forward, num_targets_backward, mesh_device);

    // Tensor Info
    [[maybe_unused]] const auto input_tensor_num_pages = input_tensor.buffer()->num_pages();
    const auto input_tensor_cores = input_tensor.memory_config().shard_spec()->grid;
    const auto input_tensor_shard_shape = input_tensor.memory_config().shard_spec()->shape;
    const auto input_tensor_shard_num_pages = input_tensor_shard_shape[0] * input_tensor_shard_shape[1] / TILE_HW;
    const auto num_input_cores = input_tensor_cores.num_cores();
    const auto output_tensor_num_pages = output_tensor.buffer()->num_pages();
    // Get only cores that have actual data
    const auto& output_tensor_original_corerangeset = output_tensor.memory_config().shard_spec()->grid;
    const auto& cores_with_data = output_tensor.buffer()->buffer_distribution_spec()->cores_with_data();

    // filter output_tensor_cores to only include cores that have data and preserve original order
    CoreRangeSet output_tensor_cores;
    if (cores_with_data.size() == output_tensor_original_corerangeset.num_cores()) {
        output_tensor_cores = output_tensor_original_corerangeset;
    } else {
        std::vector<CoreRange> output_core_ranges;
        output_core_ranges.reserve(cores_with_data.size());
        for (const auto& coord : cores_with_data) {
            output_core_ranges.emplace_back(coord, coord);
        }
        output_tensor_cores = CoreRangeSet(output_core_ranges);
    }
    const auto output_tensor_shard_shape = output_tensor.memory_config().shard_spec()->shape;
    const auto output_tensor_shard_num_pages = output_tensor_shard_shape[0] * output_tensor_shard_shape[1] / TILE_HW;
    const auto num_output_cores = output_tensor_cores.num_cores();

    auto sub_device_cores = mesh_device->worker_cores(
        tt::tt_metal::HalProgrammableCoreType::TENSIX, sub_device_id.value_or(mesh_device->get_sub_device_ids().at(0)));

    std::vector<CoreRange> output_cores;
    output_cores.reserve(sub_device_cores.ranges().size());
    for (const auto& cr : sub_device_cores.ranges()) {
        const auto intersection = output_tensor_cores.intersection(cr);
        if (!intersection.empty()) {
            output_cores.push_back(intersection.bounding_box());
        }
    }
    // output_cores_all is the bounding box of the output_tensor_cores but respecting boundaries of subdevice grids
    CoreRangeSet output_cores_all(output_cores);

    const auto& reserved_cores = output_cores_all;
    auto available_cores = sub_device_cores.subtract(reserved_cores);
    // Get worker cores, assuming 1 worker per link
    uint32_t num_workers_per_link = 1;
    CoreRangeSet sender_worker_core_range;
    std::vector<CoreCoord> sender_worker_cores;
    std::tie(sender_worker_core_range, sender_worker_cores) =
        use_optimal_ccl_for_llama ? llama_specific::get_custom_worker_core_placement(num_links)
                                  : ar_choose_worker_cores(num_links, num_workers_per_link, available_cores);

    constexpr bool has_work = true;

    // output_cores_unused is the cores that should do no work
    auto output_cores_unused = output_cores_all.subtract(output_tensor_cores);
    // all_cores is both sender and worker cores
    auto all_cores = output_cores_all.merge(sender_worker_core_range);

    log_debug(tt::LogOp, "input_tensor_num_pages: {}", input_tensor_num_pages);
    log_debug(tt::LogOp, "input_tensor_cores: {}", input_tensor_cores);
    log_debug(tt::LogOp, "input_tensor_shard_shape: {}", input_tensor_shard_shape);
    log_debug(tt::LogOp, "input_tensor_shard_num_pages: {}", input_tensor_shard_num_pages);
    log_debug(tt::LogOp, "output_tensor_cores: {}", output_tensor_cores);
    log_debug(tt::LogOp, "output_tensor_shard_shape: {}", output_tensor_shard_shape);
    log_debug(tt::LogOp, "output_tensor_shard_num_pages: {}", output_tensor_shard_num_pages);

    // L1 Scratch CB Creation
    const size_t packet_size_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    uint32_t l1_scratch_cb_page_size_bytes = op_config.get_page_size();
    uint32_t num_pages_per_packet = packet_size_bytes / l1_scratch_cb_page_size_bytes;
    uint32_t cb_num_pages = tt::div_up(output_tensor_cores.num_cores(), num_links) * output_tensor_shard_num_pages;
    uint32_t src0_cb_index = tt::CBIndex::c_0;
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    tt::DataFormat output_df = tt::tt_metal::datatype_to_dataformat_converter(output_dtype);
    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = cb_num_pages * l1_scratch_cb_page_size_bytes,
        .core_ranges = sender_worker_core_range,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
            .data_format = df,
            .page_size = l1_scratch_cb_page_size_bytes,
        }}},
    });
    // Set aside a buffer we can use for storing packet headers in (particularly for atomic incs)
    const auto reserved_packet_header_CB_index = tt::CBIndex::c_3;
    static constexpr auto num_packet_headers_storable = 8;
    auto packet_header_size_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = num_packet_headers_storable * packet_header_size_bytes * 2,
        .core_ranges = sender_worker_core_range,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(reserved_packet_header_CB_index),
            .data_format = tt::DataFormat::RawUInt32,
            .page_size = packet_header_size_bytes,
        }}},
    });

    // Reduction kernel setup
    auto input_cores_vec = corerange_to_cores(input_tensor_cores, std::nullopt, true);
    auto output_cores_vec = corerange_to_cores(output_tensor_cores, std::nullopt, true);

    // Create output tensor splits
    // TODO: Currently does not support output shards being split across multiple links
    std::vector<CoreRangeSet> output_corerangeset_per_link;
    output_corerangeset_per_link.reserve(num_links);
    std::vector<uint32_t> num_output_cores_in_link(num_links, 0);
    uint32_t output_cores_per_link = tt::div_up(output_tensor_cores.num_cores(), num_links);
    uint32_t num_assigned_cores = 0;
    for (uint32_t link = 0; link < num_links; link++) {
        uint32_t num_cores_this_link = std::min(output_cores_per_link, num_output_cores - num_assigned_cores);
        output_corerangeset_per_link.emplace_back(
            cores_to_corerangeset(std::vector<CoreCoord>(
                                      output_cores_vec.begin() + num_assigned_cores,
                                      output_cores_vec.begin() + num_assigned_cores + num_cores_this_link))
                .merge_ranges());
        num_output_cores_in_link[link] = num_cores_this_link;
        num_assigned_cores += num_cores_this_link;
    }

    // Create output tensor page splits
    std::vector<uint32_t> output_tensor_pages_in_link;
    output_tensor_pages_in_link.reserve(num_links);
    uint32_t num_assigned_pages = 0;
    for (uint32_t link = 0; link < num_links; link++) {
        uint32_t num_output_pages_per_link = output_tensor_shard_num_pages * num_output_cores_in_link[link];
        uint32_t num_pages_this_link =
            std::min(num_output_pages_per_link, output_tensor_num_pages - num_assigned_pages);
        output_tensor_pages_in_link.push_back(num_pages_this_link);
        num_assigned_pages += num_pages_this_link;
    }

    // Create input tensor splits
    /*
        Overview of algorithm:

        - Output: each link gets assigned a start and end core index, since multiple links
            may have to read different offesets within a shard on the same core
        - First, assign all the necessary cores needed for a link. This may result in the link
            containing extra pages. This will result in an overflow, which is used to detect
            the tile offset (within a shard) for the next link
        - Once you have the start_core_idx, the end_core_idx is calculated by
            getting the upper bound on the number of cores needed to read the pages assigned
            to the link, accounting for the tile offset. This calculation is done by dividing
            the upper bound on the number of pages assigned to this link
            (num_pages_this_link + input_tensor_tile_offset) by the number of pages in a shard.
            This gives the number of cores needed to read the pages assigned to this link.
        - If an overflow is detected, then the start_core_idx for the next link is set
            to the end_core_idx of the current link. Ie, 2 links read from the same core
    */
    std::vector<std::pair<uint32_t, uint32_t>> input_cores_idx_per_link(num_links, {0, 0});
    std::vector<uint32_t> input_tensor_tile_offset_per_link;
    input_tensor_tile_offset_per_link.reserve(num_links);
    uint32_t start_core_idx = 0;
    uint32_t num_pages_overflow = 0;
    for (uint32_t link = 0; link < num_links; link++) {
        uint32_t num_pages_this_link = output_tensor_pages_in_link[link];

        // Get offset based on previous overflow
        uint32_t input_tensor_tile_offset =
            (input_tensor_shard_num_pages - num_pages_overflow) % input_tensor_shard_num_pages;
        input_tensor_tile_offset_per_link.push_back(input_tensor_tile_offset);

        uint32_t end_core_idx = std::min(
            start_core_idx + tt::div_up(num_pages_this_link + input_tensor_tile_offset, input_tensor_shard_num_pages),
            num_input_cores);

        // Num pages allocated based on number of input cores selected for this link
        uint32_t num_pages_allocated =
            ((end_core_idx - start_core_idx) * input_tensor_shard_num_pages) - input_tensor_tile_offset;

        // Update overflow
        num_pages_overflow = num_pages_allocated - num_pages_this_link;

        // Store core indices
        input_cores_idx_per_link[link] = {start_core_idx, end_core_idx};

        // Set start index based on overflow
        if (num_pages_overflow > 0) {
            start_core_idx = end_core_idx - 1;
        } else {
            start_core_idx = end_core_idx;
        }
    }

    // Live-rows packing (operation_attributes.live_rows_packing): with <= 16 logical rows in a single 32-row tile row,
    // every live row sits in faces 0/1 of its tile, i.e. the first half of a bf16 32x32 tile page. The writer then
    // sends only those halves, four per fabric packet (one 4-chunk scatter write of 4 x page/2 bytes), halving the
    // fabric bytes of the decode all-reduce. The reader packs the halves back to back in cb0 (two cb pages per packet).
    // Every packet stays exactly 4 half-tiles: links whose tile count is not a multiple of 4, odd shard / offset tile
    // counts, or any other layout fall back to the default full-tile path (no partial packets are ever sent).
    constexpr uint32_t kHalvesPerPacket = 4;  // == NOC_SCATTER_WRITE_MAX_CHUNKS (static_assert in worker_writer.cpp)
    bool half_tile_packing = false;
    if (operation_attributes.live_rows_packing) {
        const auto& input_logical_shape = input_tensor.logical_shape();
        const auto& input_padded_shape = input_tensor.padded_shape();
        const auto& input_tile = input_tensor.tensor_spec().tile();
        const uint32_t half_page_size = op_config.get_page_size() / 2;
        const uint64_t padded_rows = input_padded_shape.volume() / input_padded_shape[-1];
        const uint64_t logical_rows = input_logical_shape.volume() / input_logical_shape[-1];
        half_tile_packing =
            input_tensor.dtype() == DataType::BFLOAT16 && output_dtype == DataType::BFLOAT16 &&
            input_tile.get_height() == TILE_HEIGHT && input_tile.get_width() == TILE_WIDTH &&
            input_tile.get_face_shape()[0] == FACE_HEIGHT && input_tile.get_face_shape()[1] == FACE_WIDTH &&
            !input_tile.get_transpose_within_face() && !input_tile.get_transpose_of_faces() &&
            input_tensor_shard_shape[0] == TILE_HEIGHT && padded_rows == TILE_HEIGHT && logical_rows <= FACE_HEIGHT &&
            op_config.get_page_size() == input_tile.get_tile_size(df) && op_config.get_page_size() % 2 == 0 &&
            kHalvesPerPacket * half_page_size <= packet_size_bytes && input_tensor_shard_num_pages % 2 == 0;
        for (uint32_t link = 0; link < num_links && half_tile_packing; link++) {
            half_tile_packing = output_tensor_pages_in_link[link] % kHalvesPerPacket == 0 &&
                                input_tensor_tile_offset_per_link[link] % 2 == 0;
        }
        log_debug(
            tt::LogOp,
            "all_reduce_async live_rows_packing: logical_rows={} padded_rows={} -> half_tile_packing={}",
            logical_rows,
            padded_rows,
            half_tile_packing);
    }

    // Create reduction semaphores for each link
    std::vector<uint32_t> reduction_semaphore_ids;
    reduction_semaphore_ids.reserve(num_links);
    for (uint32_t link = 0; link < num_links; link++) {
        uint32_t sem_id = static_cast<uint32_t>(desc.semaphores.size());
        desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
            .id = sem_id,
            .core_type = tt::CoreType::WORKER,
            .core_ranges = all_cores,
            .initial_value = 0,
        });
        reduction_semaphore_ids.push_back(sem_id);
    }

    /* reduction cb */
    uint32_t reduction_CB_single_tile_size = output_tensor.tensor_spec().tile().get_tile_size(df);
    uint32_t reduction_CB_tiles = output_tensor_shard_num_pages * ring_size;
    uint32_t reduction_CB_size = reduction_CB_tiles * reduction_CB_single_tile_size;

    uint32_t reduction_cb_index = tt::CBIndex::c_1;
    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = reduction_CB_size,
        .core_ranges = all_cores,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(reduction_cb_index),
            .data_format = df,
            .page_size = reduction_CB_single_tile_size,
        }}},
        .buffer = buffer_tensor.buffer(),
    });

    /* out cb */
    uint32_t out_CB_single_tile_size = output_tensor.tensor_spec().tile().get_tile_size(output_df);
    uint32_t out_CB_tiles = output_tensor_shard_num_pages;
    uint32_t out_CB_size = out_CB_tiles * out_CB_single_tile_size;

    uint32_t out_cb_index = tt::CBIndex::c_2;
    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = out_CB_size,
        .core_ranges = output_tensor_cores,  // TODO: This should be the output cores instead
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(out_cb_index),
            .data_format = output_df,
            .page_size = out_CB_single_tile_size,
        }}},
        .buffer = output_tensor.buffer(),  // TODO: Remove once new cb attached for output
    });

    /* fused residual cb: the residual tensor's own L1 shards (same shard spec as the output), read in place */
    const bool fuse_residual = tensor_args.residual_tensor.has_value();
    uint32_t residual_cb_index = tt::CBIndex::c_4;
    if (fuse_residual) {
        const auto& residual_tensor = tensor_args.residual_tensor.value();
        const uint32_t residual_CB_single_tile_size = residual_tensor.tensor_spec().tile().get_tile_size(df);
        desc.cbs.push_back(tt::tt_metal::CBDescriptor{
            .total_size = output_tensor_shard_num_pages * residual_CB_single_tile_size,
            .core_ranges = output_tensor_cores,
            .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(residual_cb_index),
                .data_format = df,
                .page_size = residual_CB_single_tile_size,
            }}},
            .buffer = residual_tensor.buffer(),
        });
    }

    const std::vector<uint32_t> no_work_rt_args = {static_cast<uint32_t>(!has_work), 0u, 0u};

    // Create reduction dataflow kernel
    tt::tt_metal::KernelDescriptor reduction_reader_kernel_desc;
    reduction_reader_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/kernels/dataflow/"
        "reduction_receiver.cpp";
    reduction_reader_kernel_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    reduction_reader_kernel_desc.core_ranges = output_cores_all;
    reduction_reader_kernel_desc.compile_time_args = {
        reduction_cb_index,                                  // reduction_cb_index
        reduction_CB_tiles,                                  // total_num_reduction_tiles
        static_cast<uint32_t>(fuse_residual),                // fuse_residual
        residual_cb_index,                                   // residual_cb_index
        fuse_residual ? output_tensor_shard_num_pages : 0u,  // num_residual_tiles
    };
    reduction_reader_kernel_desc.config = tt::tt_metal::DataMovementConfigDescriptor{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
        .noc = use_noc1_only ? tt::tt_metal::NOC::NOC_1 : reader_noc,
        .noc_mode = use_noc1_only ? tt::tt_metal::NOC_MODE::DM_DYNAMIC_NOC : tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC,
    };
    desc.kernels.push_back(std::move(reduction_reader_kernel_desc));
    if (!output_cores_unused.empty()) {
        emplace_uint32_runtime_args(desc.kernels[kReductionReader], output_cores_unused, no_work_rt_args);
    }

    // Create reduction dataflow kernel
    tt::tt_metal::KernelDescriptor reduction_compute_kernel_desc;
    reduction_compute_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/kernels/compute/"
        "reduction.cpp";
    reduction_compute_kernel_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    reduction_compute_kernel_desc.core_ranges = output_cores_all;
    reduction_compute_kernel_desc.compile_time_args = {
        reduction_cb_index,                    // reduction_cb_index
        out_cb_index,                          // out_cb_index
        static_cast<uint32_t>(fuse_residual),  // fuse_residual
        residual_cb_index,                     // residual_cb_index
    };
    tt::tt_metal::ComputeConfigDescriptor reduction_compute_config{
        .math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
        .math_approx_mode = false,
    };
    if (operation_attributes.fp32_dest_acc) {
        // fp32 dest accumulation -> ring sum independent of ETH arrival order.
        reduction_compute_config.fp32_dest_acc_en = true;
        reduction_compute_config.dst_full_sync_en = true;
    }
    reduction_compute_kernel_desc.config = reduction_compute_config;
    desc.kernels.push_back(std::move(reduction_compute_kernel_desc));
    const std::vector<uint32_t> reduction_compute_rt_args = {1u, ring_size, output_tensor_shard_num_pages};
    emplace_uint32_runtime_args(desc.kernels[kReductionCompute], output_tensor_cores, reduction_compute_rt_args);
    if (!output_cores_unused.empty()) {
        emplace_uint32_runtime_args(desc.kernels[kReductionCompute], output_cores_unused, no_work_rt_args);
    }

    // Reader
    std::vector<uint32_t> reader_compile_args = {
        device_index,               // my_chip_id
        src0_cb_index,              // cb0_id
        op_config.get_page_size(),  // tensor0_page_size
    };
    log_trace(tt::LogOp, "Reader Compile Args:");
    tt::tt_metal::KernelDescriptor worker_sender_reader_kernel_desc;
    worker_sender_reader_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/kernels/dataflow/"
        "worker_reader.cpp";
    worker_sender_reader_kernel_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    worker_sender_reader_kernel_desc.core_ranges = sender_worker_core_range;
    worker_sender_reader_kernel_desc.compile_time_args = std::move(reader_compile_args);
    if (half_tile_packing) {
        worker_sender_reader_kernel_desc.defines.emplace_back("LIVE_ROWS_HALF_TILE", "1");
    }
    worker_sender_reader_kernel_desc.config = tt::tt_metal::DataMovementConfigDescriptor{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
        .noc = reader_noc,
        .noc_mode = use_noc1_only ? tt::tt_metal::NOC_MODE::DM_DYNAMIC_NOC : tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC,
    };
    desc.kernels.push_back(std::move(worker_sender_reader_kernel_desc));

    // Writer
    std::vector<uint32_t> writer_compile_args = {
        device_index,                     // my_chip_id
        reserved_packet_header_CB_index,  // reserved_packet_header_cb_id
        num_packet_headers_storable,      // num_packet_headers_storable
        src0_cb_index,                    // cb0_id
        num_pages_per_packet,             // packet_size_in_pages
        op_config.get_page_size(),        // tensor0_page_size
        num_targets_forward,              // num_targets_forward_direction
        num_targets_backward,             // num_targets_backward_direction
    };
    writer_compile_args.insert(writer_compile_args.end(), forward_args.begin(), forward_args.end());
    writer_compile_args.insert(writer_compile_args.end(), backward_args.begin(), backward_args.end());
    log_trace(tt::LogOp, "Writer Compile Args:");
    tt::tt_metal::KernelDescriptor worker_sender_writer_kernel_desc;
    worker_sender_writer_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/kernels/dataflow/"
        "worker_writer.cpp";
    worker_sender_writer_kernel_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    worker_sender_writer_kernel_desc.core_ranges = sender_worker_core_range;
    worker_sender_writer_kernel_desc.compile_time_args = std::move(writer_compile_args);
    if (half_tile_packing) {
        worker_sender_writer_kernel_desc.defines.emplace_back("LIVE_ROWS_HALF_TILE", "1");
    }
    worker_sender_writer_kernel_desc.config = tt::tt_metal::DataMovementConfigDescriptor{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
        .noc = writer_noc,
        .noc_mode = use_noc1_only ? tt::tt_metal::NOC_MODE::DM_DYNAMIC_NOC : tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC,
    };
    desc.kernels.push_back(std::move(worker_sender_writer_kernel_desc));

    // Kernel Runtime Args
    for (uint32_t link = 0; link < num_links; link++) {
        CoreCoord core = sender_worker_cores[link];
        CoreCoord drain_sync_core = mesh_device->worker_core_from_logical_core(core);
        uint32_t worker_num_tiles_to_read = output_tensor_pages_in_link[link];

        uint32_t input_first_core_tile_start_offset = input_tensor_tile_offset_per_link[link];
        uint32_t output_first_core_tile_start_offset = 0;

        const uint32_t num_input_cores_in_link =
            input_cores_idx_per_link[link].second - input_cores_idx_per_link[link].first;
        std::vector<uint32_t> input_tensor_cores_x;
        input_tensor_cores_x.reserve(num_input_cores_in_link);
        std::vector<uint32_t> input_tensor_cores_y;
        input_tensor_cores_y.reserve(num_input_cores_in_link);
        std::vector<uint32_t> output_tensor_cores_x;
        output_tensor_cores_x.reserve(num_output_cores_in_link[link]);
        std::vector<uint32_t> output_tensor_cores_y;
        output_tensor_cores_y.reserve(num_output_cores_in_link[link]);
        for (uint32_t i = input_cores_idx_per_link[link].first; i < input_cores_idx_per_link[link].second; i++) {
            auto this_core = mesh_device->worker_core_from_logical_core(input_cores_vec[i]);
            input_tensor_cores_x.push_back(this_core.x);
            input_tensor_cores_y.push_back(this_core.y);
        }
        for (uint32_t i = output_cores_per_link * link;
             i < output_cores_per_link * link + num_output_cores_in_link[link];
             i++) {
            auto this_core = mesh_device->worker_core_from_logical_core(output_cores_vec[i]);
            output_tensor_cores_x.push_back(this_core.x);
            output_tensor_cores_y.push_back(this_core.y);
        }

        // Set reader runtime args
        std::vector<uint32_t> reader_rt_args = {
            input_tensor_shard_num_pages,        // num_tiles_per_core
            worker_num_tiles_to_read,            // num_tiles_to_read
            input_first_core_tile_start_offset,  // first_core_tile_start_offset
            input_tensor_cores_x.size(),         // num_cores
        };
        reader_rt_args.insert(reader_rt_args.end(), input_tensor_cores_x.begin(), input_tensor_cores_x.end());
        reader_rt_args.insert(reader_rt_args.end(), input_tensor_cores_y.begin(), input_tensor_cores_y.end());
        log_trace(tt::LogOp, "Reader Runtime Args:");
        for ([[maybe_unused]] const auto& arg : reader_rt_args) {
            log_trace(tt::LogOp, "\t{}", arg);
        }
        emplace_uint32_runtime_args(desc.kernels[kWorkerReader], core, reader_rt_args);

        // Set writer runtime args
        const size_t num_mcast_ranges = output_corerangeset_per_link[link].ranges().size();
        std::vector<uint32_t> mcast_start_x;
        mcast_start_x.reserve(num_mcast_ranges);
        std::vector<uint32_t> mcast_start_y;
        mcast_start_y.reserve(num_mcast_ranges);
        std::vector<uint32_t> mcast_end_x;
        mcast_end_x.reserve(num_mcast_ranges);
        std::vector<uint32_t> mcast_end_y;
        mcast_end_y.reserve(num_mcast_ranges);

        uint32_t num_mcast_cores = 0;
        for (const auto& range : output_corerangeset_per_link[link].ranges()) {
            auto start_core = mesh_device->worker_core_from_logical_core(range.start_coord);
            auto end_core = mesh_device->worker_core_from_logical_core(range.end_coord);
            num_mcast_cores += (end_core.x - start_core.x + 1) * (end_core.y - start_core.y + 1);
            bool mcast_range_contains_self =
                start_core.x <= core.x && core.x <= end_core.x && start_core.y <= core.y && core.y <= end_core.y;
            if (mcast_range_contains_self) {
                num_mcast_cores -= 1;
            }
            if (writer_noc == tt::tt_metal::NOC::NOC_1) {
                std::swap(start_core, end_core);
            }
            mcast_start_x.push_back(start_core.x);
            mcast_start_y.push_back(start_core.y);
            mcast_end_x.push_back(end_core.x);
            mcast_end_y.push_back(end_core.y);
        }

        uint32_t out_ready_sem_wait_value = ring_size;
        std::vector<uint32_t> writer_rt_args = {
            reduction_cb_index,                   // tensor_address0
            output_tensor_shard_num_pages,        // num_tiles_per_core
            worker_num_tiles_to_read,             // num_tiles_to_read
            output_first_core_tile_start_offset,  // first_core_tile_start_offset
            output_tensor_cores_x.size(),         // num_cores
            num_mcast_cores,                      // num_mcast_cores
            drain_sync_core.x,                    // out_ready_sem_noc0_x
            drain_sync_core.y,                    // out_ready_sem_noc0_y
            out_ready_sem_wait_value,             // out_ready_sem_wait_value
            reduction_semaphore_ids[link],        // reduction_semaphore_id
            mcast_start_x.size(),                 // num_mcast_ranges
            link,                                 // link
        };
        writer_rt_args.insert(writer_rt_args.end(), output_tensor_cores_x.begin(), output_tensor_cores_x.end());
        writer_rt_args.insert(writer_rt_args.end(), output_tensor_cores_y.begin(), output_tensor_cores_y.end());

        writer_rt_args.insert(writer_rt_args.end(), mcast_start_x.begin(), mcast_start_x.end());
        writer_rt_args.insert(writer_rt_args.end(), mcast_start_y.begin(), mcast_start_y.end());
        writer_rt_args.insert(writer_rt_args.end(), mcast_end_x.begin(), mcast_end_x.end());
        writer_rt_args.insert(writer_rt_args.end(), mcast_end_y.begin(), mcast_end_y.end());

        log_trace(tt::LogOp, "Writer Runtime Args:");
        for ([[maybe_unused]] const auto& arg : writer_rt_args) {
            log_trace(tt::LogOp, "\t{}", arg);
        }

        writer_rt_args.push_back(forward_coord.has_value());
        if (forward_coord.has_value()) {
            const auto target_fabric_node_id = mesh_device->get_fabric_node_id(coord);
            const auto forward_device_fabric_node_id = mesh_device->get_fabric_node_id(forward_coord.value());
            tt::tt_fabric::append_fabric_connection_rt_args<tt::tt_metal::ProgramDescriptor>(
                target_fabric_node_id, forward_device_fabric_node_id, link, desc, core, writer_rt_args);
        }

        writer_rt_args.push_back(backward_coord.has_value());
        if (backward_coord.has_value()) {
            const auto target_fabric_node_id = mesh_device->get_fabric_node_id(coord);
            const auto backward_device_fabric_node_id = mesh_device->get_fabric_node_id(backward_coord.value());
            tt::tt_fabric::append_fabric_connection_rt_args<tt::tt_metal::ProgramDescriptor>(
                target_fabric_node_id, backward_device_fabric_node_id, link, desc, core, writer_rt_args);
        }

        emplace_uint32_runtime_args(desc.kernels[kWorkerWriter], core, writer_rt_args);

        // Set reduction worker runtime args
        std::vector<uint32_t> reduction_reader_rt_args = {
            has_work,
            reduction_semaphore_ids[link],  // reduction_semaphore_id
            out_ready_sem_wait_value,       // out_ready_sem_wait_value
        };
        emplace_uint32_runtime_args(
            desc.kernels[kReductionReader], output_corerangeset_per_link[link], reduction_reader_rt_args);
    }

    desc.kernels[kWorkerReader].emplace_common_runtime_args({input_tensor.buffer()});
    desc.kernels[kWorkerWriter].emplace_common_runtime_args(
        {static_cast<uint32_t>(semaphore.address())});  // smuggled-rta-ok: caller GlobalSemaphore excluded from the
                                                        // program-cache key; re-applied by override_runtime_arguments
    desc.kernels[kReductionReader].emplace_common_runtime_args(
        {static_cast<uint32_t>(semaphore.address())});  // smuggled-rta-ok: caller GlobalSemaphore excluded from the
                                                        // program-cache key; re-applied by override_runtime_arguments
    return desc;
}

void AllReduceAsyncMeshWorkloadFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const AllReduceAsyncParams& operation_attributes,
    const AllReduceAsyncInputs& tensor_args,
    Tensor& output_tensor,
    const std::optional<ttnn::MeshCoordinate>& coord) {
    (void)coord;
    tt::tt_metal::GetCommonRuntimeArgs(program, kWorkerReader)[ttnn::ccl::AllReduceReaderCommonArgs::input] =
        tensor_args.input_tensor.buffer()->address();
    tt::tt_metal::GetCommonRuntimeArgs(program, kWorkerWriter)[ttnn::ccl::AllReduceSemaphoreCommonArgs::semaphore] =
        static_cast<uint32_t>(operation_attributes.semaphore.address());
    tt::tt_metal::GetCommonRuntimeArgs(program, kReductionReader)[ttnn::ccl::AllReduceSemaphoreCommonArgs::semaphore] =
        static_cast<uint32_t>(operation_attributes.semaphore.address());
    if (operation_attributes.one_shot) {
        tt::tt_metal::GetCommonRuntimeArgs(program, kWorkerReader)[kOneShotReaderOutputArg] =
            output_tensor.buffer()->address();
        tt::tt_metal::GetCommonRuntimeArgs(program, kWorkerWriter)[kOneShotWriterOutputArg] =
            output_tensor.buffer()->address();
        tt::tt_metal::GetCommonRuntimeArgs(program, kReductionReader)[kOneShotReceiverInputArg] =
            tensor_args.input_tensor.buffer()->address();
    }

    for (const auto& cb : program.circular_buffers()) {
        if (!cb->globally_allocated()) {
            continue;
        }
        const auto& indices = cb->buffer_indices();
        if (indices.contains(static_cast<uint8_t>(tt::CBIndex::c_1))) {
            tt::tt_metal::UpdateDynamicCircularBufferAddress(program, cb->id(), *tensor_args.buffer_tensor.buffer());
        } else if (indices.contains(static_cast<uint8_t>(tt::CBIndex::c_2))) {
            tt::tt_metal::UpdateDynamicCircularBufferAddress(program, cb->id(), *output_tensor.buffer());
        } else if (
            indices.contains(static_cast<uint8_t>(tt::CBIndex::c_4)) && tensor_args.residual_tensor.has_value()) {
            tt::tt_metal::UpdateDynamicCircularBufferAddress(
                program, cb->id(), *tensor_args.residual_tensor.value().buffer());
        }
    }
}

}  // namespace ttnn::experimental::prim

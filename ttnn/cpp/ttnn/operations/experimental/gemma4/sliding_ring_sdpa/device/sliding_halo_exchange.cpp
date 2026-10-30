// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/gemma4/sliding_ring_sdpa/device/sliding_halo_exchange.hpp"
#include "ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/ring_attention_all_gather_async_multi_core_with_workers_program_factory.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "cpp/ttnn/operations/ccl/common/types/ccl_types_args_emitters.hpp"
#include "cpp/ttnn/operations/ccl/common/host/ccl_command_stream_builders.hpp"
#include "cpp/ttnn/operations/ccl/common/uops/command_lowering.hpp"
#include "cpp/ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
#include "cpp/ttnn/operations/ccl/common/host/command_backend_runtime_args_overrider.hpp"
#include <optional>
#include <tuple>

namespace ttnn::experimental::gemma4 {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

constexpr uint32_t kBatchDimension = 0;
constexpr uint32_t kHeadDimension = 1;
constexpr uint32_t kSequenceDimension = 2;
constexpr uint32_t kWidthDimension = 3;

// Independent performance tunables. Keep these named rather than deriving one from another: the reader prefetch
// window and writer header pool protect different pipelines.
constexpr uint32_t kPrefetchPackets = 4;
// The halo writer scatters up to the fabric's scatter-write chunk limit (NOC_SCATTER_WRITE_MAX_CHUNKS).
constexpr uint32_t kMaxHaloPagesPerPacket = 4;
constexpr uint32_t kPacketHeaderSlots = 8;
constexpr uint32_t kDoubleBufferingFactor = 2;

constexpr uint32_t kSingleWorkerPerLink = 1;

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

void sliding_halo_exchange_helper(
    tt::tt_metal::ProgramDescriptor& desc,
    const std::vector<Tensor>& input_tensors,
    const MeshCoordinate& target_device_coord,
    const MeshCoordinate& transport_device_coord,
    const MeshCoordinate& unicast_destination_coord,
    std::vector<Tensor>& output_tensors,
    uint32_t num_links,
    uint32_t ring_size,
    uint32_t ring_index,
    ttnn::ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphores,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    const ttnn::experimental::ccl::AllGatherFusedOpSignaler& fused_op_signaler,
    CoreCoord core_grid_offset,
    ttnn::ccl::CoreAllocationStrategy core_allocation_strategy,
    std::optional<uint32_t> input_batch_slice_idx,
    std::optional<uint32_t> gather_valid_Ht,
    const SlidingHaloConfig& halo) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    using tt::tt_metal::CBDescriptor;
    using tt::tt_metal::CBFormatDescriptor;
    using tt::tt_metal::KernelDescriptor;
    using tt::tt_metal::ReaderConfigDescriptor;
    using tt::tt_metal::SemaphoreDescriptor;
    using tt::tt_metal::WriterConfigDescriptor;

    TT_FATAL(!input_tensors.empty() && input_tensors.size() == output_tensors.size(), "Invalid halo tensor list");
    TT_FATAL(
        semaphores.size() >= 3,
        "Neighbor halo requires three global semaphores: backward all-gather, forward all-gather, and dedicated halo. "
        "Sliding-window models must allocate 3 semaphores; non-halo models may allocate 2");
    TT_FATAL(
        halo.derives_cache_batch_on_device() == halo.derives_start_on_device(),
        "Neighbor halo slot_id and kv_actual_isl metadata must be supplied together");

    auto* mesh_device = input_tensors.front().device();
    const bool wrap_endpoint = ring_index == 0 || ring_index + 1 == ring_size;
    const auto transport_topology = wrap_endpoint ? topology : ttnn::ccl::Topology::Linear;
    auto mutable_input_tensors = input_tensors;
    const auto op_config = ttnn::ccl::CCLOpConfig(mutable_input_tensors, output_tensors, transport_topology);
    auto unicast_forward_args = std::get<0>(ttnn::ccl::get_forward_backward_line_unicast_configuration(
        target_device_coord, unicast_destination_coord, std::nullopt, mesh_device));
    if (tt::tt_fabric::is_1d_fabric_config(tt::tt_fabric::GetFabricConfig())) {
        unicast_forward_args[1] = halo.distance;
    }

    const auto [worker_core_range, worker_cores] = ttnn::ccl::choose_worker_cores(
        num_links,
        kSingleWorkerPerLink,
        mesh_device,
        sub_device_id,
        core_grid_offset,
        std::nullopt,
        core_allocation_strategy);
    const CoreRangeSet workers(worker_core_range);

    const uint32_t page_size = op_config.get_page_size();
    const uint32_t packet_buffer_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes(*mesh_device);
    const uint32_t pages_per_packet = std::min(packet_buffer_bytes / page_size, kMaxHaloPagesPerPacket);
    const uint32_t cb_pages = kDoubleBufferingFactor * kPrefetchPackets * pages_per_packet;
    const auto data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensors.front().dtype());
    constexpr uint32_t data_cb = tt::CB::c_in2;
    constexpr uint32_t packet_header_cb = tt::CB::c_in1;
    constexpr uint32_t reader_meta_cb = tt::CB::c_in4;
    constexpr uint32_t writer_meta_cb = tt::CB::c_in5;
    constexpr uint32_t meta_cb_page_size = 32;
    const uint32_t packet_header_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();

    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_pages * page_size,
        .core_ranges = workers,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(data_cb), .data_format = data_format, .page_size = page_size}}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = kPacketHeaderSlots * packet_header_bytes * kDoubleBufferingFactor,
        .core_ranges = workers,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(packet_header_cb),
            .data_format = tt::DataFormat::RawUInt32,
            .page_size = packet_header_bytes}}},
    });
    if (halo.derives_start_on_device()) {
        for (const uint32_t meta_cb : {reader_meta_cb, writer_meta_cb}) {
            desc.cbs.push_back(CBDescriptor{
                .total_size = meta_cb_page_size,
                .core_ranges = workers,
                .format_descriptors = {{CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(meta_cb),
                    .data_format = tt::DataFormat::RawUInt32,
                    .page_size = meta_cb_page_size,
                }}},
            });
        }
    }

    const uint32_t num_inputs = input_tensors.size();
    KernelDescriptor reader_kernel{};
    reader_kernel.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/gemma4/sliding_ring_sdpa/device/kernels/dataflow/"
        "sliding_halo_reader.cpp";
    reader_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_kernel.core_ranges = workers;
    reader_kernel.config = WriterConfigDescriptor{};
    reader_kernel.compile_time_args = {
        ring_index,
        ring_size,
        data_cb,
        pages_per_packet,
        page_size,
        num_inputs,
        kPrefetchPackets,
        reader_meta_cb,
        static_cast<uint32_t>(halo.collects_arrivals()),
        halo.arrivals_expected,
        static_cast<uint32_t>(halo.multicast())};
    for (uint32_t input = 0; input < num_inputs; ++input) {
        reader_kernel.compile_time_args.push_back(page_size);
    }
    for (const auto& input : input_tensors) {
        tt::tt_metal::TensorAccessorArgs(input.buffer()).append_to(reader_kernel.compile_time_args);
    }
    // Metadata accessors follow the input accessors.
    reader_kernel.compile_time_args.push_back(halo.derives_start_on_device() ? 1u : 0u);
    if (halo.derives_start_on_device()) {
        tt::tt_metal::TensorAccessorArgs(halo.slot_id->buffer()).append_to(reader_kernel.compile_time_args);
        tt::tt_metal::TensorAccessorArgs(halo.kv_actual_isl->buffer()).append_to(reader_kernel.compile_time_args);
    }

    KernelDescriptor writer_kernel{};
    writer_kernel.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/gemma4/sliding_ring_sdpa/device/kernels/dataflow/"
        "sliding_halo_writer.cpp";
    writer_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_kernel.core_ranges = workers;
    writer_kernel.config = ReaderConfigDescriptor{};
    writer_kernel.compile_time_args = {
        packet_header_cb,
        data_cb,
        pages_per_packet,
        page_size,
        num_inputs,
        unicast_forward_args[0],
        unicast_forward_args[1],
        static_cast<uint32_t>(halo.send_backward),
        writer_meta_cb,
        static_cast<uint32_t>(halo.multicast()),
    };
    for (uint32_t input = 0; input < num_inputs; ++input) {
        writer_kernel.compile_time_args.push_back(page_size);
    }
    for (const auto& output : output_tensors) {
        tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_kernel.compile_time_args);
    }
    writer_kernel.compile_time_args.push_back(halo.derives_start_on_device() ? 1u : 0u);
    if (halo.derives_start_on_device()) {
        tt::tt_metal::TensorAccessorArgs(halo.kv_actual_isl->buffer()).append_to(writer_kernel.compile_time_args);
    }

    auto halo_signaler = fused_op_signaler;
    const auto worker_list = corerange_to_cores(workers, std::nullopt, true);
    if (worker_list.size() > 1) {
        const uint32_t sem_id = desc.semaphores.size();
        desc.semaphores.push_back(SemaphoreDescriptor{
            .id = sem_id,
            .core_type = tt::CoreType::WORKER,
            .core_ranges = workers,
            .initial_value = 0,
        });
        halo_signaler.all_gather_worker_sync_semaphore = sem_id;
    }
    halo_signaler.all_gather_worker_cores_noc.clear();
    for (const auto& worker : worker_list) {
        halo_signaler.all_gather_worker_cores_noc.push_back(mesh_device->worker_core_from_logical_core(worker));
    }
    halo_signaler.initialized_all_gather = true;

    // semaphores[0]/[1] coordinate the backward/forward all-gather directions. Keep halo
    // arrivals on semaphores[2] so the protocols cannot consume each other's signals.
    const auto& halo_semaphore = semaphores.at(2);
    for (uint32_t link = 0; link < num_links; ++link) {
        KernelDescriptor::RTArgList reader_args;
        reader_args.push_back(
            static_cast<uint32_t>(halo_semaphore.address()));  // smuggled-rta-ok: persistent GlobalSemaphore address
        reader_args.push_back(link);
        KernelDescriptor::RTArgList writer_args;
        const CoreCoord worker_physical = mesh_device->worker_core_from_logical_core(worker_cores[link]);
        // Every exchange increments the SAME per-link core's semaphore on the receiver, so one reader
        // per link can wait for the whole halo. Sender and receiver lay out workers identically, so the
        // sender's rendezvous cores are also the receiver's.
        TT_FATAL(
            !halo.has_rendezvous() || halo.rendezvous_noc.size() == num_links,
            "Neighbor halo has {} rendezvous cores for {} links",
            halo.rendezvous_noc.size(),
            num_links);
        const CoreCoord arrival_core = halo.has_rendezvous() ? halo.rendezvous_noc[link] : worker_physical;
        writer_args.push_back(arrival_core.x);
        writer_args.push_back(arrival_core.y);
        writer_args.push_back(
            static_cast<uint32_t>(halo_semaphore.address()));  // smuggled-rta-ok: persistent GlobalSemaphore address

        writer_args.push_back(link);

        std::vector<uint32_t> halo_input_Wt;
        halo_input_Wt.reserve(num_inputs);
        for (uint32_t input = 0; input < num_inputs; ++input) {
            const auto input_shape = input_tensors[input].padded_shape();
            const auto output_shape = output_tensors[input].padded_shape();
            const uint32_t input_heads = input_shape[kHeadDimension];
            const uint32_t input_Wt = input_shape[kWidthDimension] / tt::constants::TILE_WIDTH;
            const uint32_t input_Ht = input_shape[kSequenceDimension] / tt::constants::TILE_HEIGHT;
            const uint32_t output_Wt = output_shape[kWidthDimension] / tt::constants::TILE_WIDTH;
            const uint32_t output_Ht = output_shape[kSequenceDimension] / tt::constants::TILE_HEIGHT;
            TT_FATAL(
                output_Wt == input_Wt,
                "Neighbor halo requires matching input/output tile widths, got input Wt={} and output Wt={}",
                input_Wt,
                output_Wt);
            const uint32_t tail_rows = halo.tail_rows();
            const uint32_t dest_end =
                halo.send_to_next_count_Ht == tail_rows ? halo.dest_row_base : halo.second_dest_row_base;
            TT_FATAL(
                output_Ht >= dest_end + tail_rows,
                "Neighbor halo output has {} tile rows but hop {} requires {}",
                output_Ht,
                halo.hop,
                dest_end + tail_rows);
            TT_FATAL(
                halo.send_to_next_start_Ht <= input_Ht && tail_rows <= input_Ht - halo.send_to_next_start_Ht,
                "Neighbor halo [{}, {}) exceeds input Ht={}",
                halo.send_to_next_start_Ht,
                halo.send_to_next_start_Ht + halo.send_to_next_count_Ht,
                input_Ht);

            const uint32_t range_start_page = 0;
            TT_FATAL(
                halo.send_to_next_count_Ht == tail_rows ||
                    (halo.send_second_start_Ht <= input_Ht && tail_rows <= input_Ht - halo.send_second_start_Ht),
                "Second neighbor halo exceeds the input cache");
            const uint32_t range_page_count = halo.send_to_next_count_Ht * input_Wt;
            const uint32_t valid_pages = std::min(gather_valid_Ht.value_or(input_Ht), input_Ht) * input_Wt;
            TT_FATAL(
                (halo.send_to_next_start_Ht + tail_rows) * input_Wt <= valid_pages &&
                    (halo.send_to_next_count_Ht == tail_rows ||
                     (halo.send_second_start_Ht + tail_rows) * input_Wt <= valid_pages),
                "Neighbor halo [{}, {}) exceeds the valid per-head page prefix {}",
                range_start_page,
                range_start_page + range_page_count,
                valid_pages);
            TT_FATAL(
                range_page_count >= num_links,
                "Neighbor halo has {} pages for {} links; every worker must send at least one page",
                range_page_count,
                num_links);
            const uint32_t pages_per_worker = range_page_count / num_links;
            const uint32_t remainder = range_page_count % num_links;
            const uint32_t input_tile_start = range_start_page + link * pages_per_worker + std::min(link, remainder);
            const uint32_t input_tile_end =
                range_start_page + (link + 1) * pages_per_worker + std::min(link + 1, remainder);
            TT_FATAL(
                !input_batch_slice_idx.has_value() || *input_batch_slice_idx < input_shape[kBatchDimension],
                "input_batch_slice_idx={} out of range for input batch={}",
                input_batch_slice_idx.value_or(0),
                input_shape[kBatchDimension]);
            const uint32_t batch_head_count = input_batch_slice_idx.has_value() || halo.derives_cache_batch_on_device()
                                                  ? input_heads
                                                  : input_shape[kBatchDimension] * input_heads;
            const uint32_t input_batch_base = ttnn::ring_attention_all_gather_async_detail::input_batch_base_pages(
                input_batch_slice_idx.value_or(0), input_heads, input_Ht, input_Wt);

            reader_args.push_back(input_Ht * input_Wt);
            reader_args.push_back(batch_head_count);
            reader_args.push_back(input_tile_start);
            reader_args.push_back(input_tile_end);
            reader_args.push_back(input_batch_base);
            reader_args.push_back(halo.send_to_next_start_Ht * input_Wt);
            reader_args.push_back(halo.send_second_start_Ht * input_Wt);
            reader_args.push_back(tail_rows * input_Wt);
            if (halo.derives_cache_batch_on_device()) {
                reader_args.push_back(input_shape[kBatchDimension]);
            }

            writer_args.push_back(output_Ht * output_Wt);
            writer_args.push_back(batch_head_count);
            writer_args.push_back(input_tile_start);
            writer_args.push_back(input_tile_end);
            // Where this hop's block starts in each of the receiver's halo slots, and the pages per tail.
            writer_args.push_back(halo.dest_row_base * output_Wt);
            writer_args.push_back(halo.second_dest_row_base * output_Wt);
            writer_args.push_back(tail_rows * output_Wt);
            halo_input_Wt.push_back(input_Wt);
        }

        if (halo.multicast()) {
            for (auto* args : {&reader_args, &writer_args}) {
                args->push_back(halo.hop_origin_rows.size());
                for (const uint32_t origin : halo.hop_origin_rows) {
                    args->push_back(origin);
                }
            }
            for (const uint32_t wt : halo_input_Wt) {
                reader_args.push_back(wt);
            }
        }

        // Runtime metadata follows the tensor descriptors and precedes accessor addresses.
        if (halo.derives_start_on_device()) {
            uint32_t cache_local_tile_rows = input_tensors.front().padded_shape()[2] / tt::constants::TILE_HEIGHT;
            for (const auto& input : input_tensors) {
                cache_local_tile_rows = std::min(
                    cache_local_tile_rows, static_cast<uint32_t>(input.padded_shape()[2] / tt::constants::TILE_HEIGHT));
            }
            uint32_t halo_slot_count =
                output_tensors.front().padded_shape()[2] / tt::constants::TILE_HEIGHT / halo.halo_tile_rows;
            for (const auto& output : output_tensors) {
                halo_slot_count = std::min(
                    halo_slot_count,
                    static_cast<uint32_t>(output.padded_shape()[2] / tt::constants::TILE_HEIGHT / halo.halo_tile_rows));
            }
            const auto append_halo_meta =
                [&](KernelDescriptor::RTArgList& args, bool with_cache_batch, bool with_ring_size) {
                    if (with_cache_batch) {
                        args.push_back(halo.slot_id->buffer());
                        args.push_back(halo.kv_cache_num_layers);
                        args.push_back(halo.kv_cache_layer_idx);
                    }
                    args.push_back(halo.kv_actual_isl->buffer());
                    args.push_back(halo.q_local_tile_rows);
                    args.push_back(halo.halo_tile_rows);
                    args.push_back(cache_local_tile_rows);
                    args.push_back(halo_slot_count);
                    args.push_back(halo.source_device);
                    if (with_ring_size) {
                        args.push_back(ring_size);
                    }
                    args.push_back(num_links);
                    args.push_back(halo.hop);
                    for (const uint32_t wt : halo_input_Wt) {
                        args.push_back(wt);
                    }
                };
            // The reader has ring_size as a compile-time arg; the writer does not.
            append_halo_meta(reader_args, /*with_cache_batch=*/true, /*with_ring_size=*/false);
            append_halo_meta(writer_args, /*with_cache_batch=*/false, /*with_ring_size=*/true);
        }
        for (const auto& input : input_tensors) {
            reader_args.push_back(input.buffer());
        }
        std::vector<uint32_t> signaler_args;
        halo_signaler.push_all_gather_fused_op_rt_args(signaler_args, num_links, link, 0);
        reader_args.append(signaler_args);
        reader_kernel.emplace_runtime_args(worker_cores[link], reader_args);

        for (const auto& output : output_tensors) {
            writer_args.push_back(output.buffer());
        }
        std::vector<uint32_t> fabric_args;
        // FabricConnectionManager consumes flags in forward-then-backward order. The linear
        // wrap sender uses the backward connection, so its sender descriptor must follow the
        // backward flag instead of the forward one.
        writer_args.push_back(halo.send_backward ? 0u : 1u);
        if (halo.send_backward) {
            writer_args.push_back(1u);
        }
        tt::tt_fabric::append_fabric_connection_rt_args(
            mesh_device->get_fabric_node_id(target_device_coord),
            mesh_device->get_fabric_node_id(transport_device_coord),
            halo.link_base + link,
            desc,
            worker_cores[link],
            fabric_args);
        writer_args.append(fabric_args);
        if (!halo.send_backward) {
            writer_args.push_back(0u);
        }
        // Link hand-off, read by the writer straight after the fabric args; both flags are 0 when
        // this hop owns its link.
        writer_args.push_back(static_cast<uint32_t>(halo.waits_for_predecessor));
        writer_args.push_back(static_cast<uint32_t>(halo.signals_successor));
        writer_args.push_back(halo.chain_semaphore_id);
        writer_args.push_back(halo.successor_noc_x);
        writer_args.push_back(halo.successor_noc_y);
        writer_kernel.emplace_runtime_args(worker_cores[link], writer_args);
    }

    desc.kernels.push_back(std::move(reader_kernel));
    desc.kernels.push_back(std::move(writer_kernel));
}

}  // namespace ttnn::experimental::gemma4

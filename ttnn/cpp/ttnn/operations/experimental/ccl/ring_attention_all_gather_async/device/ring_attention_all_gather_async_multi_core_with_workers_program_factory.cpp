// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_attention_all_gather_async_multi_core_with_workers_program_factory.hpp"
#include "ring_attention_all_gather_async_device_operation_types.hpp"
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
#include <sstream>
#include <type_traits>
#include <ranges>
#include <optional>

namespace ttnn::experimental::prim {

namespace {

// Per-coord ProgramDescriptor build. Pulled into an anonymous-namespace helper so
// create_workload_descriptor() can loop coords and reuse this body verbatim. The
// op-specific name suffix avoids Unity-build collisions across sibling factories.
tt::tt_metal::ProgramDescriptor build_ring_attention_all_gather_program_descriptor(
    const RingAttentionAllGatherAsyncMultiCoreWithWorkersProgramFactory::operation_attributes_t& operation_attributes,
    const RingAttentionAllGatherAsyncMultiCoreWithWorkersProgramFactory::tensor_args_t& tensor_args,
    RingAttentionAllGatherAsyncMultiCoreWithWorkersProgramFactory::tensor_return_value_t& tensor_return_value,
    const ttnn::MeshCoordinate& mesh_coordinate) {
    tt::tt_metal::ProgramDescriptor desc;
    std::optional<ttnn::experimental::ccl::AllGatherFusedOpSignaler> empty_fused_op_signaler;
    log_debug(tt::LogOp, "DEBUG: build_ring_attention_all_gather_program_descriptor is called");

    uint32_t device_index = ttnn::ccl::get_linearized_index_from_physical_coord(
        tensor_args.input_tensor[0], mesh_coordinate, operation_attributes.cluster_axis);

    std::optional<MeshCoordinate> forward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
        tensor_args.input_tensor[0],
        mesh_coordinate,
        1,
        operation_attributes.topology,
        operation_attributes.cluster_axis);

    std::optional<MeshCoordinate> backward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
        tensor_args.input_tensor[0],
        mesh_coordinate,
        -1,
        operation_attributes.topology,
        operation_attributes.cluster_axis);

    ring_attention_all_gather_async_multi_core_with_workers_helper(
        desc,
        tensor_args.input_tensor,
        mesh_coordinate,
        forward_coord,
        backward_coord,
        tensor_return_value,
        operation_attributes.dim,
        operation_attributes.num_links,
        operation_attributes.ring_size,
        device_index,
        operation_attributes.topology,
        operation_attributes.semaphore,
        operation_attributes.sub_device_id,
        empty_fused_op_signaler);

    return desc;
}

}  // namespace

// Returns a WorkloadDescriptor with one ProgramDescriptor per coord: device_index /
// forward_coord / backward_coord all depend on the mesh coordinate, so descriptors
// cannot be shared across coords.
tt::tt_metal::WorkloadDescriptor
RingAttentionAllGatherAsyncMultiCoreWithWorkersProgramFactory::create_workload_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    tt::tt_metal::WorkloadDescriptor wd;
    const auto coords = tensor_coords.coords();
    wd.programs.reserve(coords.size());
    for (const auto& coord : coords) {
        auto desc = build_ring_attention_all_gather_program_descriptor(
            operation_attributes, tensor_args, tensor_return_value, coord);
        wd.programs.push_back({ttnn::MeshCoordinateRange(coord), std::move(desc)});
    }
    return wd;
}

}  // namespace ttnn::experimental::prim

namespace ttnn {

void ring_attention_all_gather_async_multi_core_with_workers_helper(
    tt::tt_metal::ProgramDescriptor& desc,
    const std::vector<Tensor>& input_tensor,
    const MeshCoordinate& target_device_coord,
    std::optional<MeshCoordinate> forward_device_coord,
    std::optional<MeshCoordinate> backward_device_coord,
    std::vector<Tensor>& output_tensor,
    int32_t dim,
    uint32_t num_links,
    uint32_t ring_size,
    uint32_t ring_index,
    ttnn::ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphore,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    std::optional<ttnn::experimental::ccl::AllGatherFusedOpSignaler>& fused_op_signaler,
    const CoreCoord core_grid_offset,
    ttnn::ccl::CoreAllocationStrategy core_allocation_strategy,
    std::optional<uint32_t> input_batch_slice_idx,
    std::optional<uint32_t> gather_valid_Ht) {
    using tt::tt_metal::CBDescriptor;
    using tt::tt_metal::CBFormatDescriptor;
    using tt::tt_metal::KernelDescriptor;
    using tt::tt_metal::ReaderConfigDescriptor;
    using tt::tt_metal::SemaphoreDescriptor;
    using tt::tt_metal::WriterConfigDescriptor;

    auto* mesh_device = input_tensor[0].device();
    [[maybe_unused]] const bool is_first_chip = ring_index == 0;
    [[maybe_unused]] const bool is_last_chip = ring_index == ring_size - 1;
    log_trace(
        tt::LogOp,
        "DEBUG: device: {}, is_first_chip: {}, is_last_chip: {}",
        input_tensor.at(0).device()->id(),
        is_first_chip,
        is_last_chip);

    /* All gather fusion */
    const bool fuse_op = fused_op_signaler.has_value();

    std::optional<ttnn::experimental::ccl::AllGatherFusedOpSignaler> fused_op_signaler_sender_workers;
    std::optional<ttnn::experimental::ccl::AllGatherFusedOpSignaler> fused_op_signaler_forward;
    std::optional<ttnn::experimental::ccl::AllGatherFusedOpSignaler> fused_op_signaler_backward;

    if (fuse_op) {
        fused_op_signaler_sender_workers = fused_op_signaler.value();
        fused_op_signaler_forward = fused_op_signaler.value();
        fused_op_signaler_backward = fused_op_signaler.value();
    }

    // Get OP Config, topology config
    std::vector<Tensor> input_tensors = input_tensor;
    const std::vector<Tensor>& output_tensors = output_tensor;
    const auto& op_config = ttnn::ccl::CCLOpConfig(input_tensors, output_tensors, topology);
    auto [unicast_forward_args, unicast_backward_args] = ccl::get_forward_backward_line_unicast_configuration(
        target_device_coord, forward_device_coord, backward_device_coord, mesh_device);
    auto [num_targets_forward, num_targets_backward, dynamic_alternate] =
        ttnn::ccl::get_forward_backward_configuration(ring_size, ring_index, topology);
    if (topology == ttnn::ccl::Topology::Ring && ring_index % 2 == 0) {
        std::swap(num_targets_forward, num_targets_backward);
    }
    // Get worker cores
    // 2 sender (forward/backward, each with a reader/writer)
    uint32_t num_senders_per_link = 2;
    const auto [sender_worker_core_range, sender_worker_cores] = ttnn::ccl::choose_worker_cores(
        num_links,
        num_senders_per_link,
        mesh_device,
        sub_device_id,
        core_grid_offset,
        std::nullopt,
        core_allocation_strategy);

    std::set<CoreRange> sender_forward_core_ranges_set;
    std::set<CoreRange> sender_backward_core_ranges_set;

    for (int i = 0; i < static_cast<int>(sender_worker_cores.size()); i++) {
        const auto& core = sender_worker_cores[i];
        if (i % 2 == 1) {
            sender_forward_core_ranges_set.insert(CoreRange(core));
        } else {
            sender_backward_core_ranges_set.insert(CoreRange(core));
        }
    }

    // Convert std::set<CoreRange> -> CoreRangeSet for descriptor APIs.
    CoreRangeSet sender_forward_core_ranges(sender_forward_core_ranges_set);
    CoreRangeSet sender_backward_core_ranges(sender_backward_core_ranges_set);

    // L1 Scratch CB Creation
    const size_t packet_size_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    const uint32_t l1_scratch_cb_page_size_bytes = op_config.get_page_size();
    const uint32_t max_scatter_write_pages = 2;
    const uint32_t num_pages_per_packet =
        std::min(static_cast<uint32_t>(packet_size_bytes / l1_scratch_cb_page_size_bytes), max_scatter_write_pages);
    // Must be >= 2 * PREFETCH_PACKETS(=4) * num_pages_per_packet for deadlock-free double-buffering
    // (see PREFETCH_PACKETS in ring_attention_all_gather_reader.cpp).
    const uint32_t cb_num_pages = 8 * num_pages_per_packet;
    const tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor[0].dtype());

    // CBs for transferring data between sender_reader and sender_writer
    uint32_t sender_forward_cb_index = tt::CB::c_in0;
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_num_pages * l1_scratch_cb_page_size_bytes,
        .core_ranges = sender_forward_core_ranges,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(sender_forward_cb_index),
            .data_format = df,
            .page_size = l1_scratch_cb_page_size_bytes,
        }}},
    });

    uint32_t sender_backward_cb_index = tt::CB::c_in2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_num_pages * l1_scratch_cb_page_size_bytes,
        .core_ranges = sender_backward_core_ranges,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(sender_backward_cb_index),
            .data_format = df,
            .page_size = l1_scratch_cb_page_size_bytes,
        }}},
    });

    // Set aside a buffer we can use for storing packet headers in (particularly for atomic incs)
    const auto reserved_packet_header_forward_CB_index = tt::CB::c_in1;
    static constexpr auto num_packet_headers_storable = 8;
    const auto packet_header_size_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_packet_headers_storable * packet_header_size_bytes * 2,
        .core_ranges = sender_forward_core_ranges,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(reserved_packet_header_forward_CB_index),
            .data_format = tt::DataFormat::RawUInt32,
            .page_size = packet_header_size_bytes,
        }}},
    });

    const auto reserved_packet_header_backward_CB_index = tt::CB::c_in1;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_packet_headers_storable * packet_header_size_bytes * 2,
        .core_ranges = sender_backward_core_ranges,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(reserved_packet_header_backward_CB_index),
            .data_format = tt::DataFormat::RawUInt32,
            .page_size = packet_header_size_bytes,
        }}},
    });

    // Tensor Info
    const uint32_t num_inputs = input_tensor.size();

    uint32_t tiles_to_write_per_packet = 1;
    // KERNEL CREATION
    // Forward Direction
    // Reader
    KernelDescriptor sender_reader_forward_kernel{};
    sender_reader_forward_kernel.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/kernels/"
        "ring_attention_all_gather_reader.cpp";
    sender_reader_forward_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    sender_reader_forward_kernel.core_ranges = sender_forward_core_ranges;
    sender_reader_forward_kernel.config = WriterConfigDescriptor{};
    sender_reader_forward_kernel.compile_time_args = {
        ring_index,                       // my_chip_id
        sender_forward_cb_index,          // cb_forward_id
        num_pages_per_packet,             // packet_size_in_pages
        op_config.get_page_size(),        // tensor0_page_size
        num_targets_forward,              // num_slices_forward_direction
        num_targets_backward,             // num_slices_backward_direction
        static_cast<uint32_t>(topology),  // topology
        tiles_to_write_per_packet,        // contig_pages_advanced
        num_inputs,                       // num_inputs
        1,                                // direction
        fuse_op,                          // fused op
    };
    for (uint32_t i = 0; i < num_inputs; i++) {
        sender_reader_forward_kernel.compile_time_args.push_back(op_config.get_page_size());
    }
    for (uint32_t i = 0; i < num_inputs; i++) {
        tt::tt_metal::TensorAccessorArgs(input_tensor[i].buffer())
            .append_to(sender_reader_forward_kernel.compile_time_args);
    }
    for (uint32_t i = 0; i < num_inputs; i++) {
        tt::tt_metal::TensorAccessorArgs(output_tensor[i].buffer())
            .append_to(sender_reader_forward_kernel.compile_time_args);
    }

    // Writer
    KernelDescriptor sender_writer_forward_kernel{};
    sender_writer_forward_kernel.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/kernels/"
        "ring_attention_all_gather_writer.cpp";
    sender_writer_forward_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    sender_writer_forward_kernel.core_ranges = sender_forward_core_ranges;
    sender_writer_forward_kernel.config = ReaderConfigDescriptor{};
    sender_writer_forward_kernel.compile_time_args = {
        ring_index,                               // my_chip_id
        reserved_packet_header_forward_CB_index,  // reserved_packet_header_cb_id
        num_packet_headers_storable,              // num_packet_headers_storable
        sender_forward_cb_index,                  // cb_forward_id
        num_pages_per_packet,                     // packet_size_in_pages
        op_config.get_page_size(),                // tensor0_page_size
        num_targets_forward,                      // num_targets_forward_direction
        num_targets_backward,                     // num_targets_backward_direction
        dynamic_alternate,                        // alternate
        fuse_op,                                  // fused op
        static_cast<uint32_t>(topology),          // topology
        tiles_to_write_per_packet,                // contig_pages_advanced
        num_inputs,                               // num_inputs
        1,                                        // direction
        unicast_backward_args[0],                 // unicast route arg0 (dst_mesh_id or 0)
        unicast_backward_args[1],                 // unicast route arg1 (dst_chip_id or distance_in_hops)
    };
    for (uint32_t i = 0; i < num_inputs; i++) {
        sender_writer_forward_kernel.compile_time_args.push_back(op_config.get_page_size());
    }
    for (uint32_t i = 0; i < num_inputs; i++) {
        tt::tt_metal::TensorAccessorArgs(output_tensor[i].buffer())
            .append_to(sender_writer_forward_kernel.compile_time_args);
    }

    // Backward Direction
    // Reader
    KernelDescriptor sender_reader_backward_kernel{};
    sender_reader_backward_kernel.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/kernels/"
        "ring_attention_all_gather_reader.cpp";
    sender_reader_backward_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    sender_reader_backward_kernel.core_ranges = sender_backward_core_ranges;
    sender_reader_backward_kernel.config = WriterConfigDescriptor{};
    sender_reader_backward_kernel.compile_time_args = {
        ring_index,                       // my_chip_id
        sender_backward_cb_index,         // cb_backward_id
        num_pages_per_packet,             // packet_size_in_pages
        op_config.get_page_size(),        // tensor0_page_size
        num_targets_forward,              // num_slices_forward_direction
        num_targets_backward,             // num_slices_backward_direction
        static_cast<uint32_t>(topology),  // topology
        tiles_to_write_per_packet,        // contig_pages_advanced
        num_inputs,                       // num_inputs
        0,                                // direction
        fuse_op,                          // fused op
    };
    for (uint32_t i = 0; i < num_inputs; i++) {
        sender_reader_backward_kernel.compile_time_args.push_back(op_config.get_page_size());
    }
    for (uint32_t i = 0; i < num_inputs; i++) {
        tt::tt_metal::TensorAccessorArgs(input_tensor[i].buffer())
            .append_to(sender_reader_backward_kernel.compile_time_args);
    }
    for (uint32_t i = 0; i < num_inputs; i++) {
        tt::tt_metal::TensorAccessorArgs(output_tensor[i].buffer())
            .append_to(sender_reader_backward_kernel.compile_time_args);
    }

    // Writer
    KernelDescriptor sender_writer_backward_kernel{};
    sender_writer_backward_kernel.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/kernels/"
        "ring_attention_all_gather_writer.cpp";
    sender_writer_backward_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    sender_writer_backward_kernel.core_ranges = sender_backward_core_ranges;
    sender_writer_backward_kernel.config = ReaderConfigDescriptor{};
    sender_writer_backward_kernel.compile_time_args = {
        ring_index,                                // my_chip_id
        reserved_packet_header_backward_CB_index,  // reserved_packet_header_cb_id
        num_packet_headers_storable,               // num_packet_headers_storable
        sender_backward_cb_index,                  // cb_backward_id
        num_pages_per_packet,                      // packet_size_in_pages
        op_config.get_page_size(),                 // tensor0_page_size
        num_targets_forward,                       // num_targets_forward_direction
        num_targets_backward,                      // num_targets_backward_direction
        dynamic_alternate,                         // alternate
        fuse_op,                                   // fused op
        static_cast<uint32_t>(topology),           // topology
        tiles_to_write_per_packet,                 // contig_pages_advanced
        num_inputs,                                // num_inputs
        0,                                         // direction
        unicast_forward_args[0],                   // unicast route arg0 (dst_mesh_id or 0)
        unicast_forward_args[1],                   // unicast route arg1 (dst_chip_id or distance_in_hops)
    };
    for (uint32_t i = 0; i < num_inputs; i++) {
        sender_writer_backward_kernel.compile_time_args.push_back(op_config.get_page_size());
    }
    for (uint32_t i = 0; i < num_inputs; i++) {
        tt::tt_metal::TensorAccessorArgs(output_tensor[i].buffer())
            .append_to(sender_writer_backward_kernel.compile_time_args);
    }

    /* All gather fusion */
    // Inline equivalent of AllGatherFusedOpSignaler::init_all_gather for the descriptor
    // pattern. The original init_all_gather mutates a Program; here we instead append
    // to `desc.semaphores` and update the signaler's noc-coord and semaphore-id state
    // directly. Semaphore IDs are sequential and start at the current desc.semaphores.size().
    if (fuse_op) {
        auto sender_workers_forward = corerange_to_cores(sender_forward_core_ranges, std::nullopt, true);
        auto sender_workers_backward = corerange_to_cores(sender_backward_core_ranges, std::nullopt, true);

        auto init_all_gather_descriptor =
            [&](std::optional<ttnn::experimental::ccl::AllGatherFusedOpSignaler>& signaler,
                const CoreRangeSet& workers_range,
                const std::vector<CoreCoord>& worker_cores) {
                // Mirror AllGatherFusedOpSignaler::init_all_gather: only allocate the sync semaphore
                // when there is more than one worker core (otherwise no inter-worker sync is needed).
                if (worker_cores.size() > 1) {
                    const uint32_t sem_id = static_cast<uint32_t>(desc.semaphores.size());
                    desc.semaphores.push_back(SemaphoreDescriptor{
                        .id = sem_id,
                        .core_type = tt::CoreType::WORKER,
                        .core_ranges = workers_range,
                        .initial_value = 0,
                    });
                    signaler->all_gather_worker_sync_semaphore = sem_id;
                }
                signaler->all_gather_worker_cores_noc.clear();
                for (const auto& core : worker_cores) {
                    signaler->all_gather_worker_cores_noc.push_back(mesh_device->worker_core_from_logical_core(core));
                }
                signaler->initialized_all_gather = true;
            };

        init_all_gather_descriptor(fused_op_signaler_forward, sender_forward_core_ranges, sender_workers_forward);
        init_all_gather_descriptor(fused_op_signaler_backward, sender_backward_core_ranges, sender_workers_backward);
        init_all_gather_descriptor(
            fused_op_signaler_sender_workers, sender_forward_core_ranges, sender_workers_forward);
    }
    // Kernel Runtime Args
    for (uint32_t link = 0; link < num_links; link++) {
        // Set Sender Reader runtime args

        std::vector<uint32_t> tensor_descriptor_args;
        for (uint32_t i = 0; i < num_inputs; i++) {
            const auto input_tensor_num_pages = input_tensor[i].buffer()->num_pages();
            const auto input_tensor_shape = input_tensor[i].padded_shape();
            const auto output_tensor_shape = output_tensor[i].padded_shape();
            const uint32_t num_heads = input_tensor_shape[1];
            // single_batch_head_num_pages is always pages-per-(batch,head); independent of slicing.
            const uint32_t full_batch_head_size = input_tensor_shape[0] * num_heads;

            uint32_t single_batch_head_num_pages = input_tensor_num_pages / full_batch_head_size;
            const uint32_t base_pages_per_worker = single_batch_head_num_pages / num_links;
            const uint32_t remainder = single_batch_head_num_pages % num_links;
            const uint32_t input_tile_id_start = (link * base_pages_per_worker) + std::min(link, remainder);
            const uint32_t input_tile_id_end = ((link + 1) * base_pages_per_worker) + std::min(link + 1, remainder);

            const uint32_t input_tensor_Wt = input_tensor_shape[3] / tt::constants::TILE_WIDTH;
            const uint32_t input_tensor_Ht = input_tensor_shape[2] / tt::constants::TILE_HEIGHT;
            const uint32_t output_tensor_Wt = output_tensor_shape[3] / tt::constants::TILE_WIDTH;
            const uint32_t output_tensor_Ht = output_tensor_shape[2] / tt::constants::TILE_HEIGHT;
            TT_ASSERT(!(input_tensor_shape[3] % tt::constants::TILE_WIDTH));
            TT_ASSERT(!(output_tensor_shape[3] % tt::constants::TILE_WIDTH));

            // Single-slot gather: read only slot `input_batch_slice_idx`'s `num_heads` blocks (from
            // `input_batch_base`); the writer emits them to output slot 0. A batch-1 output suffices,
            // but a full-batch output also works (only slot 0 written). std::nullopt => full batch.
            uint32_t batch_head_size = full_batch_head_size;
            uint32_t input_batch_base = 0;
            if (input_batch_slice_idx.has_value()) {
                TT_FATAL(
                    *input_batch_slice_idx < input_tensor_shape[0],
                    "input_batch_slice_idx={} out of range for input batch={}",
                    *input_batch_slice_idx,
                    input_tensor_shape[0]);
                batch_head_size = num_heads;
                input_batch_base = ring_attention_all_gather_async_detail::input_batch_base_pages(
                    *input_batch_slice_idx, num_heads, input_tensor_Ht, input_tensor_Wt);
            }

            tensor_descriptor_args.push_back(input_tensor_Wt);      // 0 == input_tensor_Wt
            tensor_descriptor_args.push_back(input_tensor_Ht);      // 1 == input_tensor_Ht
            tensor_descriptor_args.push_back(output_tensor_Wt);     // 2 == output_tensor_Wt
            tensor_descriptor_args.push_back(output_tensor_Ht);     // 3 == output_tensor_Ht
            tensor_descriptor_args.push_back(batch_head_size);      // 4 == batch_head_size (bh-loop count)
            tensor_descriptor_args.push_back(input_tile_id_start);  // 5 == input_tile_id_start
            tensor_descriptor_args.push_back(input_tile_id_end);    // 6 == input_tile_id_end
            tensor_descriptor_args.push_back(input_batch_base);     // 7 == input_batch_base (phase-1 input page offset)
            // 8 == valid pages per (batch,head) to gather. Default: full input slab (no clamp). When
            // gather_valid_Ht is set (fused ring_joint_sdpa with an oversized cache), bound it to the
            // first gather_valid_Ht tile-rows so only kv_actual-sized data moves. The fused path also
            // re-patches this per dispatch on cache hits (apply_ring_joint_scalar_runtime_args); setting
            // it here makes the cache-miss (first) dispatch bounded too.
            const uint32_t valid_pages_per_batch_head =
                gather_valid_Ht.has_value() ? std::min(*gather_valid_Ht, input_tensor_Ht) * input_tensor_Wt
                                            : single_batch_head_num_pages;
            tensor_descriptor_args.push_back(valid_pages_per_batch_head);  // 8 == valid_pages_per_batch_head
        }

        KernelDescriptor::RTArgList reader_forward_rt_args;
        reader_forward_rt_args.push_back(static_cast<uint32_t>(dim));  // dim to gather on
        reader_forward_rt_args.push_back(ring_size);                   // ring_size
        reader_forward_rt_args.push_back(
            static_cast<uint32_t>(semaphore.at(1).address()));  // out_ready_semaphore_backward
        reader_forward_rt_args.append(tensor_descriptor_args);
        for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++) {
            reader_forward_rt_args.push_back(input_tensor[input_idx].buffer());
        }
        for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++) {
            reader_forward_rt_args.push_back(output_tensor[input_idx].buffer());
        }
        if (fuse_op) {
            std::vector<uint32_t> reader_forward_signaler_args;
            fused_op_signaler_forward->push_all_gather_fused_op_rt_args(
                reader_forward_signaler_args, num_links, link, 1);
            reader_forward_rt_args.append(reader_forward_signaler_args);
        }
        sender_reader_forward_kernel.emplace_runtime_args(sender_worker_cores[(link * 2) + 1], reader_forward_rt_args);

        KernelDescriptor::RTArgList reader_backward_rt_args;
        reader_backward_rt_args.push_back(static_cast<uint32_t>(dim));  // dim to gather on
        reader_backward_rt_args.push_back(ring_size);                   // ring_size
        reader_backward_rt_args.push_back(
            static_cast<uint32_t>(semaphore.at(0).address()));  // out_ready_semaphore_backward
        reader_backward_rt_args.append(tensor_descriptor_args);
        for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++) {
            reader_backward_rt_args.push_back(input_tensor[input_idx].buffer());
        }
        for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++) {
            reader_backward_rt_args.push_back(output_tensor[input_idx].buffer());
        }
        if (fuse_op) {
            std::vector<uint32_t> reader_backward_signaler_args;
            fused_op_signaler_backward->push_all_gather_fused_op_rt_args(
                reader_backward_signaler_args, num_links, link, 0);
            reader_backward_rt_args.append(reader_backward_signaler_args);
        }
        sender_reader_backward_kernel.emplace_runtime_args(sender_worker_cores[link * 2], reader_backward_rt_args);

        const CoreCoord sender_forward_worker_core =
            mesh_device->worker_core_from_logical_core(sender_worker_cores[(link * 2) + 1]);
        const CoreCoord sender_backward_worker_core =
            mesh_device->worker_core_from_logical_core(sender_worker_cores[link * 2]);

        // Writer
        KernelDescriptor::RTArgList writer_forward_rt_args;
        writer_forward_rt_args.push_back(static_cast<uint32_t>(dim));                           // dim to gather on
        writer_forward_rt_args.push_back(static_cast<uint32_t>(sender_forward_worker_core.x));  // out_ready_sem_noc0_x
        writer_forward_rt_args.push_back(static_cast<uint32_t>(sender_forward_worker_core.y));  // out_ready_sem_noc0_y
        writer_forward_rt_args.push_back(ring_size);                                            // ring_size
        writer_forward_rt_args.push_back(
            static_cast<uint32_t>(semaphore.at(1).address()));  // out_ready_semaphore_backward
        writer_forward_rt_args.append(tensor_descriptor_args);
        for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++) {
            writer_forward_rt_args.push_back(output_tensor[input_idx].buffer());
        }
        writer_forward_rt_args.push_back(0u);
        writer_forward_rt_args.push_back(static_cast<uint32_t>(backward_device_coord.has_value()));
        // Fabric/signaler helpers expect std::vector<uint32_t>&; collect their args separately,
        // then merge so any BufferBinding entries above are preserved.
        std::vector<uint32_t> writer_forward_extra_args;
        if (backward_device_coord.has_value()) {
            const auto target_fabric_node_id = mesh_device->get_fabric_node_id(target_device_coord);
            const auto backward_fabric_node_id = mesh_device->get_fabric_node_id(backward_device_coord.value());
            tt::tt_fabric::append_fabric_connection_rt_args(
                target_fabric_node_id,
                backward_fabric_node_id,
                link,
                desc,
                sender_worker_cores[(link * 2) + 1],
                writer_forward_extra_args);
        }
        if (fuse_op) {
            fused_op_signaler_sender_workers->push_all_gather_fused_op_rt_args(
                writer_forward_extra_args, num_links, link, 1);
        }
        writer_forward_rt_args.append(writer_forward_extra_args);
        sender_writer_forward_kernel.emplace_runtime_args(sender_worker_cores[(link * 2) + 1], writer_forward_rt_args);

        KernelDescriptor::RTArgList writer_backward_rt_args;
        writer_backward_rt_args.push_back(static_cast<uint32_t>(dim));  // dim to gather on
        writer_backward_rt_args.push_back(
            static_cast<uint32_t>(sender_backward_worker_core.x));  // out_ready_sem_noc0_x
        writer_backward_rt_args.push_back(
            static_cast<uint32_t>(sender_backward_worker_core.y));  // out_ready_sem_noc0_y
        writer_backward_rt_args.push_back(ring_size);               // ring_size
        writer_backward_rt_args.push_back(
            static_cast<uint32_t>(semaphore.at(0).address()));  // out_ready_semaphore_backward
        writer_backward_rt_args.append(tensor_descriptor_args);
        for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++) {
            writer_backward_rt_args.push_back(output_tensor[input_idx].buffer());
        }
        writer_backward_rt_args.push_back(static_cast<uint32_t>(forward_device_coord.has_value()));
        std::vector<uint32_t> writer_backward_extra_args;
        if (forward_device_coord.has_value()) {
            const auto target_fabric_node_id = mesh_device->get_fabric_node_id(target_device_coord);
            const auto forward_fabric_node_id = mesh_device->get_fabric_node_id(forward_device_coord.value());
            tt::tt_fabric::append_fabric_connection_rt_args(
                target_fabric_node_id,
                forward_fabric_node_id,
                link,
                desc,
                sender_worker_cores[link * 2],
                writer_backward_extra_args);
        }
        writer_backward_rt_args.append(writer_backward_extra_args);
        writer_backward_rt_args.push_back(0u);
        if (fuse_op) {
            std::vector<uint32_t> writer_backward_signaler_args;
            fused_op_signaler_sender_workers->push_all_gather_fused_op_rt_args(writer_backward_signaler_args, 1, 0, 0);
            writer_backward_rt_args.append(writer_backward_signaler_args);
        }
        sender_writer_backward_kernel.emplace_runtime_args(sender_worker_cores[link * 2], writer_backward_rt_args);
    }

    // Kernel descriptors are pushed last, with their runtime args fully populated.
    // The descriptor framework allocates KernelHandles when materializing the
    // descriptor into a Program; runtime-arg auto-patching on cache hits removes
    // the need to expose those handles back to the caller.
    desc.kernels.push_back(std::move(sender_reader_forward_kernel));
    desc.kernels.push_back(std::move(sender_writer_forward_kernel));
    desc.kernels.push_back(std::move(sender_reader_backward_kernel));
    desc.kernels.push_back(std::move(sender_writer_backward_kernel));
}

}  // namespace ttnn

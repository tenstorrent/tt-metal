// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_attention_all_gather_async_device_operation.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"

// TODO: Remove these extra headers once ring join sdpa is migrated to new infra
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::experimental::ccl::ring_attention_all_gather_async {

RingAttentionAllGatherAsyncDeviceOperation::program_factory_t
RingAttentionAllGatherAsyncDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Only one program factory available
    return RingAttentionAllGatherAsyncMultiCoreWithWorkersProgramFactory{};
}

void RingAttentionAllGatherAsyncDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(operation_attributes, tensor_args);
}

void RingAttentionAllGatherAsyncDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensors = tensor_args.input_tensor;
    TT_FATAL(
        !input_tensors.empty(), "Error, Input tensor size should be greater than 0 but has {}", input_tensors.size());

    const auto& first_input_tensor = input_tensors[0];
    const auto& dtype = first_input_tensor.dtype();
    const auto& memory_config = first_input_tensor.memory_config();
    const auto& input_shape = first_input_tensor.logical_shape();

    // Validate all input tensors
    for (size_t i = 0; i < input_tensors.size(); ++i) {
        const auto& input_tensor = input_tensors[i];

        TT_FATAL(input_tensor.layout() == Layout::TILE, "Input tensor {} must be tiled", i);
        TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Input tensor {} must be on device", i);
        TT_FATAL(input_tensor.buffer() != nullptr, "Input tensor {} must be allocated in buffers on device", i);

        TT_FATAL(
            input_tensor.dtype() == dtype,
            "All input tensors must have the same dtype. Input tensor {} has dtype {} but expected {}",
            i,
            input_tensor.dtype(),
            dtype);

        TT_FATAL(
            input_tensor.memory_config() == memory_config,
            "All input tensors must have the same memory config. Input tensor {} has different memory config",
            i);

        TT_FATAL(
            input_tensor.logical_shape() == input_shape,
            "All input tensors must have the same shape. Input tensor {} has different shape",
            i);
    }

    TT_FATAL(
        operation_attributes.num_links > 0,
        "Error, num_links should be more than 0 but has {}",
        operation_attributes.num_links);

    TT_FATAL(
        memory_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Unsupported memory layout {}.",
        memory_config.memory_layout());
}

RingAttentionAllGatherAsyncDeviceOperation::spec_return_value_t
RingAttentionAllGatherAsyncDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensors = tensor_args.input_tensor;
    const auto& input_tensor = input_tensors[0];
    auto shape = input_tensor.logical_shape();
    shape[operation_attributes.dim] *= operation_attributes.ring_size;

    // Need to determine output memory config - this should come from operation_attributes
    // For now, using input memory config as fallback
    MemoryConfig output_mem_config = input_tensor.memory_config();

    std::vector<ttnn::TensorSpec> output_specs;
    output_specs.reserve(input_tensors.size());
    for (uint32_t i = 0; i < input_tensors.size(); i++) {
        output_specs.push_back(TensorSpec(
            shape, TensorLayout(input_tensor.dtype(), input_tensor.tensor_spec().page_config(), output_mem_config)));
    }
    return output_specs;
}

RingAttentionAllGatherAsyncDeviceOperation::tensor_return_value_t
RingAttentionAllGatherAsyncDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensors = tensor_args.input_tensor;
    auto output_specs = compute_output_specs(operation_attributes, tensor_args);

    std::vector<Tensor> output_tensors;
    output_tensors.reserve(output_specs.size());
    for (const auto& output_spec : output_specs) {
        output_tensors.emplace_back(create_device_tensor(output_spec, input_tensors[0].device()));
    }
    return output_tensors;
}

tt::stl::hash::hash_t RingAttentionAllGatherAsyncDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensors = tensor_args.input_tensor;
    auto input_shape = input_tensors[0].padded_shape();
    auto input_memory_layout = input_tensors[0].layout();
    auto input_dtype = input_tensors[0].dtype();
    auto input_memory_config = input_tensors[0].memory_config();

    // Need to determine output_mem_config - this should be in operation_attributes
    // For now, using input memory config as fallback
    MemoryConfig output_mem_config = input_tensors[0].memory_config();

    return tt::tt_metal::operation::hash_operation<RingAttentionAllGatherAsyncDeviceOperation>(
        operation_attributes.dim,
        operation_attributes.num_links,
        operation_attributes.ring_size,
        output_mem_config,
        operation_attributes.topology,
        operation_attributes.sub_device_id.has_value(),
        operation_attributes.sub_device_id.has_value()
            ? input_tensors[0].device()->worker_cores(
                  tt::tt_metal::HalProgrammableCoreType::TENSIX, operation_attributes.sub_device_id.value())
            : CoreRangeSet(CoreRange({0, 0}, {0, 0})),
        input_shape,
        input_memory_layout,
        input_dtype,
        input_memory_config);
}

std::tuple<
    RingAttentionAllGatherAsyncDeviceOperation::operation_attributes_t,
    RingAttentionAllGatherAsyncDeviceOperation::tensor_args_t>
RingAttentionAllGatherAsyncDeviceOperation::invoke(
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>& persistent_output_buffer,
    int32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    ttnn::ccl::Topology topology,
    uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id) {
    const auto& mesh_view = mesh_device.get_view();
    TT_FATAL(
        mesh_view.is_mesh_2d(),
        "all-gather invoked with cluster_axis API withou 2D mesh, which is currently unsupported");
    uint32_t ring_size = (cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();
    int32_t rank = input_tensors[0].logical_shape().rank();
    int32_t gather_dim = (dim < 0) ? rank + dim : dim;

    TT_FATAL(
        gather_dim >= -rank && gather_dim <= rank - 1,
        "Dimension input should be in between -{} and {}, but has {}",
        rank,
        rank - 1,
        dim);

    std::vector<std::optional<Tensor>> optional_output_tensors;
    optional_output_tensors.reserve(persistent_output_buffer.size());
    for (size_t i = 0; i < persistent_output_buffer.size(); ++i) {
        optional_output_tensors.push_back(persistent_output_buffer[i]);
    }

    return {
        operation_attributes_t{
            {},
            gather_dim,
            num_links,
            ring_size,
            memory_config.value_or(input_tensors[0].memory_config()),
            topology,
            multi_device_global_semaphore,
            sub_device_id,
            cluster_axis,
        },
        tensor_args_t{.input_tensor = input_tensors, .persistent_output_buffer = optional_output_tensors}};
}

}  // namespace ttnn::operations::experimental::ccl::ring_attention_all_gather_async

namespace ttnn {
tt::tt_metal::operation::ProgramWithCallbacks ring_attention_all_gather_async_multi_core_with_workers_helper(
    tt::tt_metal::Program& program,
    const std::vector<Tensor>& input_tensor,
    IDevice* target_device,
    std::optional<IDevice*> forward_device,
    std::optional<IDevice*> backward_device,
    std::vector<Tensor>& output_tensor,
    const uint32_t dim,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    ttnn::ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphore,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    std::optional<experimental::ccl::AllGatherFusedOpSignaler>& fused_op_signaler,
    const CoreCoord core_grid_offset) {
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

    std::optional<experimental::ccl::AllGatherFusedOpSignaler> fused_op_signaler_sender_workers;
    std::optional<experimental::ccl::AllGatherFusedOpSignaler> fused_op_signaler_forward;
    std::optional<experimental::ccl::AllGatherFusedOpSignaler> fused_op_signaler_backward;

    if (fuse_op) {
        fused_op_signaler_sender_workers = fused_op_signaler.value();
        fused_op_signaler_forward = fused_op_signaler.value();
        fused_op_signaler_backward = fused_op_signaler.value();
    }

    // Get OP Config, topology config
    std::vector<Tensor> input_tensors = input_tensor;
    const std::vector<Tensor>& output_tensors = output_tensor;
    const auto& op_config = ttnn::ccl::CCLOpConfig(input_tensors, output_tensors, topology);
    auto [num_targets_forward, num_targets_backward, dynamic_alternate] =
        ccl::get_forward_backward_configuration(ring_size, ring_index, topology);
    if (topology == ttnn::ccl::Topology::Ring && ring_index % 2 == 0) {
        std::swap(num_targets_forward, num_targets_backward);
    }
    // Get worker cores
    // 2 sender (forward/backward, each with a reader/writer)
    uint32_t num_senders_per_link = 2;
    const auto [sender_worker_core_range, sender_worker_cores] =
        ttnn::ccl::choose_worker_cores(num_links, num_senders_per_link, mesh_device, sub_device_id, core_grid_offset);

    std::set<CoreRange> sender_forward_core_ranges;
    std::set<CoreRange> sender_backward_core_ranges;

    for (int i = 0; i < sender_worker_cores.size(); i++) {
        const auto& core = sender_worker_cores[i];
        if (i % 2 == 1) {
            sender_forward_core_ranges.insert(CoreRange(core));
        } else {
            sender_backward_core_ranges.insert(CoreRange(core));
        }
    }

    // L1 Scratch CB Creation
    const size_t packet_size_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    const uint32_t l1_scratch_cb_page_size_bytes = op_config.get_page_size();
    const uint32_t max_scatter_write_pages = 2;
    const uint32_t num_pages_per_packet =
        std::min((uint32_t)(packet_size_bytes / l1_scratch_cb_page_size_bytes), max_scatter_write_pages);
    const uint32_t cb_num_pages = 3 * num_pages_per_packet;  // triple buffering
    const tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor[0].dtype());

    // CBs for transferring data between sender_reader and sender_writer
    uint32_t sender_forward_cb_index = tt::CB::c_in0;
    tt::tt_metal::CircularBufferConfig cb_sender_forward_config =
        tt::tt_metal::CircularBufferConfig(
            cb_num_pages * l1_scratch_cb_page_size_bytes, {{sender_forward_cb_index, df}})
            .set_page_size(sender_forward_cb_index, l1_scratch_cb_page_size_bytes);
    CreateCircularBuffer(program, sender_forward_core_ranges, cb_sender_forward_config);
    uint32_t sender_backward_cb_index = tt::CB::c_in2;
    tt::tt_metal::CircularBufferConfig cb_sender_backward_config =
        tt::tt_metal::CircularBufferConfig(
            cb_num_pages * l1_scratch_cb_page_size_bytes, {{sender_backward_cb_index, df}})
            .set_page_size(sender_backward_cb_index, l1_scratch_cb_page_size_bytes);
    CreateCircularBuffer(program, sender_backward_core_ranges, cb_sender_backward_config);

    // Set aside a buffer we can use for storing packet headers in (particularly for atomic incs)
    const auto reserved_packet_header_forward_CB_index = tt::CB::c_in1;
    static constexpr auto num_packet_headers_storable = 8;
    const auto packet_header_size_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
    tt::tt_metal::CircularBufferConfig cb_reserved_packet_header_forward_config =
        tt::tt_metal::CircularBufferConfig(
            num_packet_headers_storable * packet_header_size_bytes * 2,
            {{reserved_packet_header_forward_CB_index, tt::DataFormat::RawUInt32}})
            .set_page_size(reserved_packet_header_forward_CB_index, packet_header_size_bytes);
    CreateCircularBuffer(program, sender_forward_core_ranges, cb_reserved_packet_header_forward_config);
    const auto reserved_packet_header_backward_CB_index = tt::CB::c_in1;
    tt::tt_metal::CircularBufferConfig cb_reserved_packet_header_backward_config =
        tt::tt_metal::CircularBufferConfig(
            num_packet_headers_storable * packet_header_size_bytes * 2,
            {{reserved_packet_header_backward_CB_index, tt::DataFormat::RawUInt32}})
            .set_page_size(reserved_packet_header_backward_CB_index, packet_header_size_bytes);
    CreateCircularBuffer(program, sender_backward_core_ranges, cb_reserved_packet_header_backward_config);

    // Tensor Info
    const auto input_tensor_num_pages = input_tensor[0].buffer()->num_pages();
    const auto input_tensor_shape = input_tensor[0].padded_shape();
    const auto output_tensor_shape = output_tensor[0].padded_shape();
    const uint32_t num_inputs = input_tensor.size();

    uint32_t tiles_to_write_per_packet = 1;
    // KERNEL CREATION
    // Forward Direction
    // Reader
    auto sender_reader_forward_kernel_config = tt::tt_metal::ReaderDataMovementConfig{};
    sender_reader_forward_kernel_config.compile_args = {
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
        sender_reader_forward_kernel_config.compile_args.push_back(op_config.get_page_size());
    }
    for (uint32_t i = 0; i < num_inputs; i++) {
        tt::tt_metal::TensorAccessorArgs(input_tensor[i].buffer())
            .append_to(sender_reader_forward_kernel_config.compile_args);
    }
    for (uint32_t i = 0; i < num_inputs; i++) {
        tt::tt_metal::TensorAccessorArgs(output_tensor[i].buffer())
            .append_to(sender_reader_forward_kernel_config.compile_args);
    }
    auto worker_sender_reader_forward_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/kernels/"
        "ring_attention_all_gather_reader.cpp",
        sender_forward_core_ranges,
        sender_reader_forward_kernel_config);

    // Writer
    auto sender_writer_forward_kernel_config = tt::tt_metal::WriterDataMovementConfig{};
    sender_writer_forward_kernel_config.compile_args = {
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
    };
    for (uint32_t i = 0; i < num_inputs; i++) {
        sender_writer_forward_kernel_config.compile_args.push_back(op_config.get_page_size());
    }
    for (uint32_t i = 0; i < num_inputs; i++) {
        tt::tt_metal::TensorAccessorArgs(output_tensor[i].buffer())
            .append_to(sender_writer_forward_kernel_config.compile_args);
    }
    auto worker_sender_writer_forward_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/kernels/"
        "ring_attention_all_gather_writer.cpp",
        sender_forward_core_ranges,
        sender_writer_forward_kernel_config);

    // Backward Direction
    // Reader
    auto sender_reader_backward_kernel_config = tt::tt_metal::ReaderDataMovementConfig{};
    sender_reader_backward_kernel_config.compile_args = {
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
        sender_reader_backward_kernel_config.compile_args.push_back(op_config.get_page_size());
    }
    for (uint32_t i = 0; i < num_inputs; i++) {
        tt::tt_metal::TensorAccessorArgs(input_tensor[i].buffer())
            .append_to(sender_reader_backward_kernel_config.compile_args);
    }
    for (uint32_t i = 0; i < num_inputs; i++) {
        tt::tt_metal::TensorAccessorArgs(output_tensor[i].buffer())
            .append_to(sender_reader_backward_kernel_config.compile_args);
    }
    auto worker_sender_reader_backward_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/kernels/"
        "ring_attention_all_gather_reader.cpp",
        sender_backward_core_ranges,
        sender_reader_backward_kernel_config);

    // Writer
    auto sender_writer_backward_kernel_config = tt::tt_metal::WriterDataMovementConfig{};
    sender_writer_backward_kernel_config.compile_args = {
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
    };
    for (uint32_t i = 0; i < num_inputs; i++) {
        sender_writer_backward_kernel_config.compile_args.push_back(op_config.get_page_size());
    }
    for (uint32_t i = 0; i < num_inputs; i++) {
        tt::tt_metal::TensorAccessorArgs(output_tensor[i].buffer())
            .append_to(sender_writer_backward_kernel_config.compile_args);
    }
    auto worker_sender_writer_backward_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/kernels/"
        "ring_attention_all_gather_writer.cpp",
        sender_backward_core_ranges,
        sender_writer_backward_kernel_config);

    /* All gather fusion */
    if (fuse_op) {
        auto sender_workers_forward = corerange_to_cores(sender_forward_core_ranges, std::nullopt, true);
        auto sender_workers_backward = corerange_to_cores(sender_backward_core_ranges, std::nullopt, true);
        fused_op_signaler_forward->init_all_gather(
            program, mesh_device, sender_forward_core_ranges, sender_workers_forward);
        fused_op_signaler_backward->init_all_gather(
            program, mesh_device, sender_backward_core_ranges, sender_workers_backward);
        fused_op_signaler_sender_workers->init_all_gather(
            program, mesh_device, sender_forward_core_ranges, sender_workers_forward);
    }
    // Kernel Runtime Args
    uint32_t reader_sender_rt_offset = 0;
    uint32_t writer_sender_rt_offset = 0;
    for (uint32_t link = 0; link < num_links; link++) {
        // Set Sender Reader runtime args
        const uint32_t batch_head_size = input_tensor_shape[0] * input_tensor_shape[1];

        uint32_t single_batch_head_num_pages = input_tensor_num_pages / batch_head_size;
        const uint32_t base_pages_per_worker = single_batch_head_num_pages / num_links;
        const uint32_t remainder = single_batch_head_num_pages % num_links;
        const uint32_t input_tile_id_start = (link * base_pages_per_worker) + std::min(link, remainder);
        const uint32_t input_tile_id_end = ((link + 1) * base_pages_per_worker) + std::min(link + 1, remainder);

        TT_ASSERT(!(input_tensor_shape[3] % tt::constants::TILE_WIDTH));
        TT_ASSERT(!(output_tensor_shape[3] % tt::constants::TILE_WIDTH));
        const uint32_t input_tensor_Wt = input_tensor_shape[3] / tt::constants::TILE_WIDTH;
        const uint32_t input_tensor_Ht = input_tensor_shape[2] / tt::constants::TILE_WIDTH;
        const uint32_t output_tensor_Wt = output_tensor_shape[3] / tt::constants::TILE_WIDTH;
        const uint32_t output_tensor_Ht = output_tensor_shape[2] / tt::constants::TILE_WIDTH;

        std::vector<uint32_t> reader_forward_rt_args = {
            input_tensor_Wt,            // width in tiles of the input shard
            input_tensor_Ht,            // height in tiles of the input shard
            output_tensor_Wt,           // width in tiles of the entire output
            output_tensor_Ht,           // height in tiles of the entire output
            dim,                        // dim to gather on
            batch_head_size,            // product of the first two dims
            input_tile_id_start,        //
            input_tile_id_end,          // slice_num_pages
            ring_size,                  // ring_size
            semaphore.at(1).address(),  // out_ready_semaphore_backward
        };
        reader_sender_rt_offset = reader_forward_rt_args.size();
        for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++) {
            reader_forward_rt_args.push_back(input_tensor[input_idx].buffer()->address());
        }
        for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++) {
            reader_forward_rt_args.push_back(output_tensor[input_idx].buffer()->address());
        }
        if (fuse_op) {
            fused_op_signaler_forward->push_all_gather_fused_op_rt_args(reader_forward_rt_args, num_links, link, 1);
        }
        tt::tt_metal::SetRuntimeArgs(
            program,
            worker_sender_reader_forward_kernel_id,
            {sender_worker_cores[(link * 2) + 1]},
            reader_forward_rt_args);

        std::vector<uint32_t> reader_backward_rt_args = {
            input_tensor_Wt,            // width in tiles of the input shard
            input_tensor_Ht,            // height in tiles of the input shard
            output_tensor_Wt,           // width in tiles of the entire output
            output_tensor_Ht,           // height in tiles of the entire output
            dim,                        // dim to gather on
            batch_head_size,            // product of the first two dims
            input_tile_id_start,        // slice_num_pages
            input_tile_id_end,          // slice_num_pages
            ring_size,                  // ring_size
            semaphore.at(0).address(),  // out_ready_semaphore_backward
        };
        for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++) {
            reader_backward_rt_args.push_back(input_tensor[input_idx].buffer()->address());
        }
        for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++) {
            reader_backward_rt_args.push_back(output_tensor[input_idx].buffer()->address());
        }
        if (fuse_op) {
            fused_op_signaler_backward->push_all_gather_fused_op_rt_args(reader_backward_rt_args, num_links, link, 0);
        }
        tt::tt_metal::SetRuntimeArgs(
            program, worker_sender_reader_backward_kernel_id, {sender_worker_cores[link * 2]}, reader_backward_rt_args);

        const CoreCoord sender_forward_worker_core =
            mesh_device->worker_core_from_logical_core(sender_worker_cores[(link * 2) + 1]);
        const CoreCoord sender_backward_worker_core =
            mesh_device->worker_core_from_logical_core(sender_worker_cores[link * 2]);

        // Writer
        std::vector<uint32_t> writer_forward_rt_args = {
            input_tensor_Wt,               // width in tiles of the input shard
            input_tensor_Ht,               // height in tiles of the input shard
            output_tensor_Wt,              // width in tiles of entire output
            output_tensor_Ht,              // height in tiles of entire output
            dim,                           // dim to gather on
            batch_head_size,               // product of the first two dims
            input_tile_id_start,           //
            input_tile_id_end,             //
            sender_forward_worker_core.x,  // out_ready_sem_noc0_x
            sender_forward_worker_core.y,  // out_ready_sem_noc0_y
            ring_size,                     // ring_size
            semaphore.at(1).address()      // out_ready_semaphore_backward
        };
        writer_sender_rt_offset = writer_forward_rt_args.size();
        for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++) {
            writer_forward_rt_args.push_back(output_tensor[input_idx].buffer()->address());
        }
        writer_forward_rt_args.push_back(false);
        writer_forward_rt_args.push_back(backward_device.has_value());
        if (backward_device.has_value()) {
            const auto target_fabric_node_id =
                tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(target_device->id());
            const auto backward_fabric_node_id =
                tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(backward_device.value()->id());
            tt::tt_fabric::append_fabric_connection_rt_args(
                target_fabric_node_id,
                backward_fabric_node_id,
                link,
                program,
                sender_worker_cores[(link * 2) + 1],
                writer_forward_rt_args);
        }
        if (fuse_op) {
            fused_op_signaler_sender_workers->push_all_gather_fused_op_rt_args(
                writer_forward_rt_args, num_links, link, 1);
        }
        tt::tt_metal::SetRuntimeArgs(
            program,
            worker_sender_writer_forward_kernel_id,
            sender_worker_cores[(link * 2) + 1],
            writer_forward_rt_args);

        std::vector<uint32_t> writer_backward_rt_args = {
            input_tensor_Wt,                // width in tiles of the input shard
            input_tensor_Ht,                // height in tiles of the input shard
            output_tensor_Wt,               // width in tiles of entire output
            output_tensor_Ht,               // height in tiles of entire output
            dim,                            // dim to gather on
            batch_head_size,                // product of the first two dims
            input_tile_id_start,            //
            input_tile_id_end,              //
            sender_backward_worker_core.x,  // out_ready_sem_noc0_x
            sender_backward_worker_core.y,  // out_ready_sem_noc0_y
            ring_size,                      // ring_size
            semaphore.at(0).address()       // out_ready_semaphore_backward
        };
        for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++) {
            writer_backward_rt_args.push_back(output_tensor[input_idx].buffer()->address());
        }
        writer_backward_rt_args.push_back(forward_device.has_value());
        if (forward_device.has_value()) {
            const auto target_fabric_node_id =
                tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(target_device->id());
            const auto forward_fabric_node_id =
                tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(forward_device.value()->id());
            tt::tt_fabric::append_fabric_connection_rt_args(
                target_fabric_node_id,
                forward_fabric_node_id,
                link,
                program,
                sender_worker_cores[link * 2],
                writer_backward_rt_args);
        }
        writer_backward_rt_args.push_back(false);
        if (fuse_op) {
            fused_op_signaler_sender_workers->push_all_gather_fused_op_rt_args(writer_backward_rt_args, 1, 0, 0);
        }
        tt::tt_metal::SetRuntimeArgs(
            program, worker_sender_writer_backward_kernel_id, sender_worker_cores[link * 2], writer_backward_rt_args);
    }

    auto override_runtime_arguments_callback =
        [worker_sender_reader_forward_kernel_id,
         worker_sender_writer_forward_kernel_id,
         worker_sender_reader_backward_kernel_id,
         worker_sender_writer_backward_kernel_id,
         sender_worker_cores,
         num_inputs,
         reader_sender_rt_offset,
         writer_sender_rt_offset,
         num_links](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            const auto& semaphore =
                static_cast<const ttnn::operations::experimental::ccl::ring_attention_all_gather_async::
                                RingAttentionAllGatherAsyncDeviceOperation::operation_attributes_t*>(operation)
                    ->semaphore;
            // const auto& semaphore = static_cast<const RingAttentionAllGatherAsync*>(operation)->semaphore;
            // update senders
            auto& worker_reader_sender_forward_runtime_args_by_core =
                GetRuntimeArgs(program, worker_sender_reader_forward_kernel_id);
            auto& worker_writer_sender_forward_runtime_args_by_core =
                GetRuntimeArgs(program, worker_sender_writer_forward_kernel_id);
            auto& worker_reader_sender_backward_runtime_args_by_core =
                GetRuntimeArgs(program, worker_sender_reader_backward_kernel_id);
            auto& worker_writer_sender_backward_runtime_args_by_core =
                GetRuntimeArgs(program, worker_sender_writer_backward_kernel_id);

            for (int link = 0; link < num_links; link++) {
                auto& worker_reader_sender_forward_runtime_args =
                    worker_reader_sender_forward_runtime_args_by_core[sender_worker_cores[1 + (link * 2)].x]
                                                                     [sender_worker_cores[1 + (link * 2)].y];
                auto& worker_reader_sender_backward_runtime_args =
                    worker_reader_sender_backward_runtime_args_by_core[sender_worker_cores[0 + (link * 2)].x]
                                                                      [sender_worker_cores[0 + (link * 2)].y];
                auto& worker_writer_sender_forward_runtime_args =
                    worker_writer_sender_forward_runtime_args_by_core[sender_worker_cores[1 + (link * 2)].x]
                                                                     [sender_worker_cores[1 + (link * 2)].y];
                auto& worker_writer_sender_backward_runtime_args =
                    worker_writer_sender_backward_runtime_args_by_core[sender_worker_cores[0 + (link * 2)].x]
                                                                      [sender_worker_cores[0 + (link * 2)].y];

                worker_reader_sender_forward_runtime_args[9] = semaphore.at(1).address();
                worker_reader_sender_backward_runtime_args[9] = semaphore.at(0).address();
                worker_writer_sender_forward_runtime_args[11] = semaphore.at(1).address();
                worker_writer_sender_backward_runtime_args[11] = semaphore.at(0).address();
                for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++) {
                    // sender reader
                    worker_reader_sender_forward_runtime_args[reader_sender_rt_offset + input_idx] =
                        input_tensors[input_idx].buffer()->address();
                    worker_reader_sender_forward_runtime_args[reader_sender_rt_offset + num_inputs + input_idx] =
                        output_tensors[input_idx].buffer()->address();
                    worker_reader_sender_backward_runtime_args[reader_sender_rt_offset + input_idx] =
                        input_tensors[input_idx].buffer()->address();
                    worker_reader_sender_backward_runtime_args[reader_sender_rt_offset + num_inputs + input_idx] =
                        output_tensors[input_idx].buffer()->address();
                    // sender writer
                    worker_writer_sender_forward_runtime_args[writer_sender_rt_offset + input_idx] =
                        output_tensors[input_idx].buffer()->address();
                    worker_writer_sender_backward_runtime_args[writer_sender_rt_offset + input_idx] =
                        output_tensors[input_idx].buffer()->address();
                }
            }
        };
    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}
}  // namespace ttnn

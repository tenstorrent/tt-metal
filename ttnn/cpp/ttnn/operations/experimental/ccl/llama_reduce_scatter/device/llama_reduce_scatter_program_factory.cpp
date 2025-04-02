// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "llama_reduce_scatter_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <vector>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/device_pool.hpp>
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_op.hpp"
#include "cpp/ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"
#include "cpp/ttnn/operations/ccl/sharding_addrgen_helper.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/erisc_datamover_builder.hpp>
#include "cpp/ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
#include <tt-metalium/sub_device.hpp>

namespace ttnn::operations::experimental::ccl {

namespace detail {

std::string device_order_array_string(uint32_t ring_size, uint32_t ring_index) {
    ttnn::SmallVector<uint32_t> device_order;
    device_order.reserve(ring_size - 1);
    // Add all indices except ring_index
    for (uint32_t i = 0; i < ring_size; i++) {
        if (i != ring_index) {
            device_order.push_back(i);
        }
    }

    // Sort based on absolute difference from ring_index in descending order
    std::sort(device_order.begin(), device_order.end(), [ring_index](uint32_t a, uint32_t b) {
        return std::abs(static_cast<int>(a) - static_cast<int>(ring_index)) >
               std::abs(static_cast<int>(b) - static_cast<int>(ring_index));
    });

    // Convert to string format
    std::string result = "{";
    for (size_t i = 0; i < device_order.size(); i++) {
        result += std::to_string(device_order[i]);
        if (i < device_order.size() - 1) {
            result += ", ";
        }
    }
    result += "}";
    return result;
}

std::string cores_to_string(const std::vector<CoreCoord>& cores) {
    std::string result = "{";
    for (const auto& core : cores) {
        result += "{" + std::to_string(core.x) + ", " + std::to_string(core.y) + "}, ";
    }
    result += "}";
    return result;
}

void append_fabric_connection_rt_args(
    const std::optional<tt::tt_fabric::SenderWorkerAdapterSpec>& connection,
    const CoreCoord& core,
    tt::tt_metal::Program& program,
    std::vector<uint32_t>& writer_rt_args) {
    writer_rt_args.push_back(connection.has_value());
    if (connection.has_value()) {
        auto sender_worker_flow_control_semaphore_id = CreateSemaphore(program, {core}, 0);
        auto sender_worker_teardown_semaphore_id = CreateSemaphore(program, {core}, 0);
        auto sender_worker_buffer_index_semaphore_id = CreateSemaphore(program, {core}, 0);
        append_worker_to_fabric_edm_sender_rt_args(
            connection.value(),
            sender_worker_flow_control_semaphore_id,
            sender_worker_teardown_semaphore_id,
            sender_worker_buffer_index_semaphore_id,
            writer_rt_args);
    }
}

struct ReadRequest {
    uint32_t bank_id;
    uint32_t read_offset;
    uint32_t read_size;  // in pages
};

std::vector<std::vector<ReadRequest>> distribute_work_evenly(
    uint32_t num_shards, uint32_t num_workers, uint32_t tiles_per_core_width, uint32_t packet_size) {
    // 1) Compute total number of pages and total packets
    uint32_t total_pages = num_shards * tiles_per_core_width;
    uint32_t total_packets = (total_pages + packet_size - 1) / packet_size;  // ceil division

    // 2) Figure out how many packets each worker should handle
    //    Distribute "total_packets" as evenly as possible among the "num_workers".
    uint32_t base_packets = total_packets / num_workers;  // integer division
    uint32_t remainder = total_packets % num_workers;     // leftover packets to distribute
    // So the first 'remainder' workers get (base_packets + 1), the rest get base_packets
    std::vector<uint32_t> packets_per_worker(num_workers, base_packets);
    for (uint32_t w = 0; w < remainder; ++w) {
        packets_per_worker[w] += 1;
    }

    // 3) Prepare the output structure: each worker gets a list of (bank_id, read_size).
    std::vector<std::vector<ReadRequest>> schedule(num_workers);

    // We'll iterate over all banks in order [0..num_shards-1],
    // reading them chunk by chunk until we've assigned all pages.
    uint32_t current_bank = 0;
    uint32_t pages_left_in_bank = (num_shards > 0) ? tiles_per_core_width : 0;

    // We'll move through workers [0..num_workers-1].
    uint32_t current_worker = 0;
    uint32_t formed_packets = 0;  // how many packets the current worker has formed

    // leftover_in_packet tells us how many pages are *already* in the partial packet buffer
    // for the current worker (mod packet_size).
    uint32_t leftover_in_packet = 0;

    while (current_bank < num_shards && current_worker < num_workers) {
        // If this worker has already formed all of its allocated packets, move to the next one.
        if (formed_packets == packets_per_worker[current_worker]) {
            current_worker++;
            // If we’re moving to a new worker, reset leftover/packets_formed
            if (current_worker < num_workers) {
                formed_packets = 0;
                leftover_in_packet = 0;
            }
            continue;
        }
        // If we've also run out of workers entirely, break (though normally that means we have no pages left).
        if (current_worker == num_workers) {
            break;
        }

        // We will read some chunk of the current bank. Decide how large that chunk should be.
        // Two constraints:
        //   1) We cannot read more than "pages_left_in_bank"
        //   2) We do not want to exceed the remaining packet quota for this worker

        // The maximum # of new full packets the worker can still form is:
        uint32_t packets_remaining_for_worker = packets_per_worker[current_worker] - formed_packets;

        // The total # of pages we can still accumulate for this worker
        // without exceeding its packet quota is:
        //   packets_remaining_for_worker * packet_size - leftover_in_packet
        //
        // Because leftover_in_packet pages are already in the partial packet buffer.
        uint32_t max_pages_for_worker = packets_remaining_for_worker * packet_size - leftover_in_packet;

        // So the chunk we read must be <= pages_left_in_bank
        // and <= max_pages_for_worker
        uint32_t chunk_size = (pages_left_in_bank < max_pages_for_worker) ? pages_left_in_bank : max_pages_for_worker;

        // Where in the bank does the chunk start
        uint32_t read_offset = std::max(0, int(tiles_per_core_width) - int(pages_left_in_bank));

        // Create a read request for that chunk
        schedule[current_worker].push_back({current_bank, read_offset, chunk_size});

        // Update leftover and count how many packets we formed by adding this chunk
        uint32_t new_total = leftover_in_packet + chunk_size;
        uint32_t new_packets = new_total / packet_size;  // how many full packets formed now
        leftover_in_packet = new_total % packet_size;

        formed_packets += new_packets;  // add the new fully formed packets

        // We consumed 'chunk_size' pages from this bank
        pages_left_in_bank -= chunk_size;

        // If we've exhausted this bank, move on to the next
        if (pages_left_in_bank == 0) {
            current_bank++;
            if (current_bank < num_shards) {
                pages_left_in_bank = tiles_per_core_width;
            }
        }
    }

    // Return the schedule
    return schedule;
}

uint32_t find_atomic_inc_core(std::vector<std::vector<ReadRequest>> schedule) {
    uint32_t atomic_inc_core = 0;
    uint32_t min_packets = schedule[0].size();
    for (uint32_t i = 1; i < schedule.size(); ++i) {
        if (schedule[i].size() < min_packets) {
            min_packets = schedule[i].size();
            atomic_inc_core = i;
        }
    }
    return atomic_inc_core;
}

std::vector<ReadRequest> flatten_schedule(const std::vector<std::vector<ReadRequest>>& schedule) {
    // create a flattened schedule
    std::vector<ReadRequest> schedule_flattened;
    for (uint32_t i = 0; i < schedule.size(); ++i) {
        schedule_flattened.insert(schedule_flattened.end(), schedule[i].begin(), schedule[i].end());
    }
    return schedule_flattened;
}

std::string schedule_to_string(const std::vector<std::vector<ReadRequest>>& schedule) {
    auto flattened_schedule = flatten_schedule(schedule);
    std::string result = "{";
    for (uint32_t i = 0; i < flattened_schedule.size(); ++i) {
        result += "{" + std::to_string(flattened_schedule[i].bank_id) + ", " +
                  std::to_string(flattened_schedule[i].read_offset) + ", " +
                  std::to_string(flattened_schedule[i].read_size) + "}, ";
    }
    result += "}";
    return result;
}

uint32_t max_packets_per_worker(const std::vector<std::vector<ReadRequest>>& schedule) {
    uint32_t max_packets_per_worker = 0;
    for (const auto& worker_schedule : schedule) {
        max_packets_per_worker = std::max(max_packets_per_worker, (uint32_t)worker_schedule.size());
    }
    return max_packets_per_worker;
}

CoreRangeSet get_worker_cores(const CoreRangeSet& available_cores, const uint32_t num_workers, bool row_wise) {
    CoreRangeSet worker_cores;
    for (const auto& cr : available_cores.ranges()) {
        auto cores = corerange_to_cores(cr, std::nullopt, row_wise);
        for (const auto& core : cores) {
            worker_cores = worker_cores.merge(CoreRangeSet(CoreRange(core, core)));
            if (worker_cores.num_cores() == num_workers) {
                break;
            }
        }
        if (worker_cores.num_cores() == num_workers) {
            break;
        }
    }
    return worker_cores;
}

}  // namespace detail

LlamaReduceScatterDeviceOperation::LlamaReduceScatterAdd::cached_program_t
LlamaReduceScatterDeviceOperation::LlamaReduceScatterAdd::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;
    using namespace tt::tt_fabric;
    using namespace ttnn::ccl;
    uint32_t ring_size = operation_attributes.ring_devices;
    uint32_t num_devices = ring_size;

    const auto& input_tensor = tensor_args.input_tensor;
    tt::tt_metal::IDevice* device = input_tensor.device();
    bool enable_persistent_fabric = true;
    uint32_t num_links = operation_attributes.num_links;

    uint32_t ring_index = operation_attributes.ring_index;
    std::string device_order = detail::device_order_array_string(ring_size, ring_index);

    std::map<std::string, std::string> reader_defines = {{"DEVICE_ORDER", device_order}};

    const auto& input_shape = input_tensor.get_logical_shape();
    const auto dim = operation_attributes.dim;
    uint32_t rank = input_shape.size();
    auto& output_tensor = tensor_return_value;
    auto& output_shape = output_tensor.get_logical_shape();
    auto& padded_output_shape = output_tensor.get_padded_shape();
    const auto& tile_shape = input_tensor.get_tensor_spec().tile().get_tile_shape();
    const auto& face_shape = input_tensor.get_tensor_spec().tile().get_face_shape();
    auto shard_spec = input_tensor.shard_spec().value();
    auto output_shard_spec = output_tensor.shard_spec().value();
    const auto& cross_device_semaphore = operation_attributes.cross_device_semaphore;

    auto input_tensor_buffer = input_tensor.buffer();
    auto output_tensor_buffer = output_tensor.buffer();
    auto packet_buffer = tensor_args.intermediate_packet_buffer.buffer();
    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());

    uint32_t input_page_size = tile_size(cb_data_format);
    uint32_t output_page_size = tile_size(cb_data_format);

    // Get OP Config, topology config
    std::vector<Tensor> input_tensors = {input_tensor};
    std::vector<Tensor> output_tensors = {output_tensor};

    const auto& op_config = ttnn::ccl::CCLOpConfig(input_tensors, output_tensors, ttnn::ccl::Topology::Linear);
    LineTopology line_topology(ring_size, ring_index);

    uint32_t shard_height = shard_spec.shape[0];
    uint32_t shard_width = shard_spec.shape[1];
    uint32_t tiles_per_core_width = shard_width / tile_shape[1];

    uint32_t shard_height_output = output_shard_spec.shape[0];
    uint32_t shard_width_output = output_shard_spec.shape[1];
    uint32_t tiles_per_core_width_output = shard_width_output / tile_shape[1];
    auto input_grid = shard_spec.grid;
    auto output_grid = output_shard_spec.grid;

    auto sub_device_cores = device->worker_cores(
        tt::tt_metal::HalProgrammableCoreType::TENSIX,
        operation_attributes.subdevice_id.value_or(device->get_sub_device_ids().at(0)));

    tt::tt_metal::Program program{};

    std::optional<ttnn::ccl::EdmLineFabricOpInterface> local_fabric_handle =
        ttnn::ccl::EdmLineFabricOpInterface::build_program_builder_worker_connection_fabric(
            device,
            operation_attributes.forward_device.value_or(nullptr),
            operation_attributes.backward_device.value_or(nullptr),
            &program,
            enable_persistent_fabric,
            num_links);

    const size_t packet_size_bytes = local_fabric_handle->get_edm_buffer_size_bytes();
    uint32_t num_pages_per_packet = packet_size_bytes / input_page_size;

    uint32_t ncores_input = shard_spec.num_cores();
    uint32_t input_shard_cores_per_device = ncores_input / num_devices;
    uint32_t output_cores_per_device = output_grid.num_cores();
    uint32_t num_workers_per_link = 1;

    auto intermediate_packet_buffer_grid = tensor_args.intermediate_packet_buffer.shard_spec().value().grid;
    // UNCOMMENT this once we can allocate persistent buffers across all device lifetimes
    uint32_t num_packets_total_per_device =
        (input_shard_cores_per_device * tiles_per_core_width + num_pages_per_packet - 1) / num_pages_per_packet;
    auto packet_worker_cores_grid = detail::get_worker_cores(
        intermediate_packet_buffer_grid,
        num_packets_total_per_device,
        shard_spec.orientation == ShardOrientation::ROW_MAJOR);

    auto available_cores = sub_device_cores.subtract(packet_worker_cores_grid);

    auto sender_core_grid = detail::get_worker_cores(
        available_cores, num_workers_per_link * num_links, shard_spec.orientation == ShardOrientation::ROW_MAJOR);
    auto all_cores_grid = packet_worker_cores_grid.merge(sender_core_grid);

    auto schedule = detail::distribute_work_evenly(
        input_shard_cores_per_device, num_workers_per_link * num_links, tiles_per_core_width, num_pages_per_packet);
    auto schedule_string = detail::schedule_to_string(schedule);

    // input sharded buffer
    uint32_t input_tensor_cb_id = tt::CBIndex::c_0;
    // output sharded buffer
    uint32_t output_tensor_cb_id = tt::CBIndex::c_1;
    // client interface
    uint32_t packet_header_cb_index = tt::CBIndex::c_2;
    // fabric sender
    uint32_t fabric_sender_cb_index = tt::CBIndex::c_3;
    // fabric receiver where we receive the data from the other device
    uint32_t fabric_receiver_cb_index = tt::CBIndex::c_4;
    // accumulator before we perform the reduction
    uint32_t accumulator_cb_index = tt::CBIndex::c_5;

    tt::tt_metal::CircularBufferConfig cb_input_tensor_config =
        tt::tt_metal::CircularBufferConfig(
            tiles_per_core_width * input_page_size, {{input_tensor_cb_id, cb_data_format}})
            .set_page_size(input_tensor_cb_id, input_page_size)
            .set_globally_allocated_address(*input_tensor_buffer);
    // CB to represent the output sharded buffer
    tt::tt_metal::CircularBufferConfig cb_output_tensor_config =
        tt::tt_metal::CircularBufferConfig(
            tiles_per_core_width_output * input_page_size, {{output_tensor_cb_id, cb_data_format}})
            .set_page_size(output_tensor_cb_id, input_page_size)
            .set_globally_allocated_address(*output_tensor_buffer);

    constexpr uint32_t buffering_factor = 2;
    // Allocate space for the client interface
    static constexpr auto num_packet_headers_storable = 8;
    static constexpr auto packet_header_size_bytes = sizeof(tt::tt_fabric::PacketHeader);
    tt::tt_metal::CircularBufferConfig packet_header_cb_config =
        tt::tt_metal::CircularBufferConfig(
            num_packet_headers_storable * packet_header_size_bytes * buffering_factor,
            {{packet_header_cb_index, DataFormat::RawUInt32}})
            .set_page_size(packet_header_cb_index, packet_header_size_bytes);

    uint32_t max_packets_per_worker = detail::max_packets_per_worker(schedule);
    uint32_t num_packets_total = max_packets_per_worker * (num_devices - 1);
    uint32_t num_packet_pages_total = num_packets_total * num_pages_per_packet;

    // There is one sender from link, and each sender splits the packet workload for each device
    // For llama there are 3 senders, and each device sends 30 pages to each other device
    // there is thus div_up(30, 4) = 8 packets
    // Sender 0 -> 3 packets per device, Sender 1 -> 3 packets per device, Sender 2 -> 2 packets per device
    // Sender 2 will have less work, so we use it to perform the atomic increment
    // The below CB will read in the packets to send to each other device

    /*
    Sender 0:
    Device 0 first 3 packets
    Device 1 first 3 packets
    Device 2 first 3 packets

    Sender 1:
    Device 0 next 3 packets
    Device 1 next 3 packets
    Device 2 next 3 packets

    Sender 2:
    Device 0 last 2 packets and the atomic increment packet
    Device 1 last 2 packets and the atomic increment packet
    Device 2 last 2 packets and the atomic increment packet
    */
    tt::tt_metal::CircularBufferConfig fabric_sender_cb_config =
        tt::tt_metal::CircularBufferConfig(
            num_packet_pages_total * input_page_size, {{fabric_sender_cb_index, cb_data_format}})
            .set_page_size(fabric_sender_cb_index, input_page_size);

    // buffer for receiving packets from the other devices.
    // each core handles one packet from each device
    // these packets are all received at this CB, which is allocated as a tensor to prevent deallocation before all
    // devices are done the packet worker core that has this buffer will then do a reduction across the equivalent pages
    // from each packet from each device As there are 4 devices, there will be 4 sections in the buffer One of the
    // sections will have the local data from the device Final buffer before reduction:
    /*
    --------------------------------------
    Device 0 section:
    -----
    Page 0
    Page 1
    Page 2
    Page 3
    --------------------------------------
    Device 1 section:
    -----
    Page 0
    Page 1
    Page 2
    Page 3
    --------------------------------------
    Device 2 section:
    -----
    Page 0
    Page 1
    Page 2
    Page 3
    --------------------------------------
    Device 3 section:
    -----
    Page 0
    Page 1
    Page 2
    Page 3
    --------------------------------------
    */
    tt::tt_metal::CircularBufferConfig fabric_receiver_cb_config =
        tt::tt_metal::CircularBufferConfig(
            num_pages_per_packet * num_devices * input_page_size, {{fabric_receiver_cb_index, cb_data_format}})
            .set_page_size(fabric_receiver_cb_index, input_page_size)
            .set_globally_allocated_address(*packet_buffer);

    // After reduction, the data will be packed into the accumulator cb
    /*
    --------------------------------------
    Page 0 reduced across all devices
    Page 1 reduced across all devices
    Page 2 reduced across all devices
    Page 3 reduced across all devices
    --------------------------------------
    */
    tt::tt_metal::CircularBufferConfig accumulator_cb_config =
        tt::tt_metal::CircularBufferConfig(
            buffering_factor * num_pages_per_packet * tiles_per_core_width_output * input_page_size * num_devices,
            {{accumulator_cb_index, cb_data_format}})
            .set_page_size(accumulator_cb_index, input_page_size);

    auto cb_input_tensor_handle =
        tt::tt_metal::CreateCircularBuffer(program, all_cores_grid, cb_input_tensor_config);  // input buffer
    auto cb_output_tensor_handle =
        tt::tt_metal::CreateCircularBuffer(program, all_cores_grid, cb_output_tensor_config);  // output buffer
    auto cb_client_interface_handle =
        tt::tt_metal::CreateCircularBuffer(program, all_cores_grid, packet_header_cb_config);  // client interface
    auto cb_fabric_receiver_handle =
        tt::tt_metal::CreateCircularBuffer(program, all_cores_grid, fabric_receiver_cb_config);
    auto cb_fabric_sender_handle = tt::tt_metal::CreateCircularBuffer(program, all_cores_grid, fabric_sender_cb_config);
    auto cb_accumulator_handle = tt::tt_metal::CreateCircularBuffer(program, all_cores_grid, accumulator_cb_config);

    auto input_cores =
        corerange_to_cores(input_grid, std::nullopt, shard_spec.orientation == ShardOrientation::ROW_MAJOR);
    auto output_cores =
        corerange_to_cores(output_grid, std::nullopt, shard_spec.orientation == ShardOrientation::ROW_MAJOR);
    auto sender_cores =
        corerange_to_cores(sender_core_grid, std::nullopt, shard_spec.orientation == ShardOrientation::ROW_MAJOR);
    auto all_cores =
        corerange_to_cores(all_cores_grid, std::nullopt, shard_spec.orientation == ShardOrientation::ROW_MAJOR);
    auto packet_worker_cores = corerange_to_cores(
        packet_worker_cores_grid, std::nullopt, shard_spec.orientation == ShardOrientation::ROW_MAJOR);
    auto packet_receiver_core = packet_worker_cores.at(0);

    const uint32_t chip_id = ring_index;

    auto to_worker_cores = [device](const std::vector<CoreCoord>& cores) -> std::vector<CoreCoord> {
        std::vector<CoreCoord> worker_cores;
        for (const auto& core : cores) {
            worker_cores.push_back(device->worker_core_from_logical_core(core));
        }
        return worker_cores;
    };

    auto packet_bounding_box = packet_worker_cores_grid.bounding_box();
    auto packet_start_worker_core = to_worker_cores({packet_bounding_box.start_coord});
    auto packet_end_worker_core = to_worker_cores({packet_bounding_box.end_coord});

    std::vector<uint32_t> reader_compile_time_args = {
        input_tensor_cb_id,
        fabric_sender_cb_index,
        packet_header_cb_index,
        fabric_receiver_cb_index,
        accumulator_cb_index,
        output_tensor_cb_id,
        (uint32_t)chip_id,
        tiles_per_core_width,
        tiles_per_core_width_output,
        num_pages_per_packet,
        input_shard_cores_per_device,
        num_devices,
        input_page_size,
        output_cores_per_device,
        packet_start_worker_core.at(0).x,
        packet_start_worker_core.at(0).y,
        packet_end_worker_core.at(0).x,
        packet_end_worker_core.at(0).y,
        sender_cores.size()};

    reader_defines["INPUT_CORE_XY"] = detail::cores_to_string(to_worker_cores(input_cores));
    reader_defines["OUTPUT_CORE_XY"] = detail::cores_to_string(to_worker_cores(output_cores));
    reader_defines["PACKET_WORKER_CORES"] = detail::cores_to_string(to_worker_cores(packet_worker_cores));
    reader_defines["SCHEDULE"] = schedule_string;

    // create local semaphore
    auto local_semaphore = tt::tt_metal::CreateSemaphore(program, all_cores_grid, INVALID);

    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter/device/kernels/dataflow/"
        "reader_llama_reduce_scatter.cpp",
        all_cores_grid,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_time_args,
            .defines = reader_defines});

    auto packet_receiver_worker_core = to_worker_cores({packet_receiver_core}).at(0);

    std::vector<uint32_t> writer_compile_time_args = {
        input_tensor_cb_id,
        fabric_sender_cb_index,
        packet_header_cb_index,
        fabric_receiver_cb_index,
        accumulator_cb_index,
        output_tensor_cb_id,
        (uint32_t)chip_id,
        tiles_per_core_width,
        tiles_per_core_width_output,
        num_pages_per_packet,
        input_shard_cores_per_device,
        num_devices,
        input_page_size,
        output_cores_per_device,
        packet_receiver_worker_core.x,
        packet_receiver_worker_core.y,
    };

    auto writer_defines = reader_defines;
    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter/device/kernels/dataflow/"
        "writer_llama_reduce_scatter.cpp",
        all_cores_grid,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_time_args,
            .defines = writer_defines});

    const std::vector<uint32_t> compute_compile_time_args = {
        fabric_receiver_cb_index, accumulator_cb_index, num_devices, tiles_per_core_width_output, num_pages_per_packet};

    bool fp32_dest_acc_en = cb_data_format == tt::DataFormat::Float32;
    const auto compute_kernel_file =
        "ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter/device/kernels/compute/reduction.cpp";
    const auto compute_kernel_id = tt_metal::CreateKernel(
        program,
        compute_kernel_file,
        packet_worker_cores_grid,
        tt_metal::ComputeConfig{.fp32_dest_acc_en = fp32_dest_acc_en, .compile_args = compute_compile_time_args});

    uint32_t offset_for_input = chip_id * input_shard_cores_per_device * tiles_per_core_width;
    uint32_t local_page = 0;

    std::vector<uint32_t> reader_runtime_args = {
        cross_device_semaphore->address(), local_semaphore, false, false, 0, false, 0, 0};
    uint32_t is_reader_sender_core_idx = 2;
    uint32_t is_reader_worker_core_idx = 3;
    uint32_t is_linear_input_packet_start_idx = 4;
    uint32_t is_reader_receiver_core_idx = 5;
    uint32_t reader_sender_packet_start_idx = 6;
    uint32_t reader_sender_packet_end_idx = 7;

    uint32_t is_writer_sender_core_idx = 2;
    uint32_t is_writer_worker_core_idx = 3;
    uint32_t is_linear_output_page_start_idx = 4;
    uint32_t writer_sender_packet_start_idx = 5;
    uint32_t writer_sender_packet_end_idx = 6;

    uint32_t sender_packet_start = 0;
    uint32_t sender_core_idx = 0;

    for (auto core : all_cores) {
        std::vector<uint32_t> writer_runtime_args = {
            cross_device_semaphore->address(), local_semaphore, false, false, 0, 0, 0};

        if (sender_core_grid.contains(core)) {
            reader_runtime_args[is_reader_sender_core_idx] = true;
            reader_runtime_args[is_reader_worker_core_idx] = false;
            reader_runtime_args[is_reader_receiver_core_idx] = false;
            reader_runtime_args[reader_sender_packet_start_idx] = sender_packet_start;
            reader_runtime_args[reader_sender_packet_end_idx] = sender_packet_start + schedule[sender_core_idx].size();

            writer_runtime_args[is_writer_sender_core_idx] = true;
            writer_runtime_args[is_writer_worker_core_idx] = false;
            writer_runtime_args[writer_sender_packet_start_idx] = sender_packet_start;
            writer_runtime_args[writer_sender_packet_end_idx] = sender_packet_start + schedule[sender_core_idx].size();

            sender_packet_start += schedule[sender_core_idx].size();
            sender_core_idx++;

            std::optional<tt::tt_fabric::SenderWorkerAdapterSpec> forward_fabric_connection =
                line_topology.is_first_device_in_line(ttnn::ccl::EdmLineFabricOpInterface::Direction::BACKWARD)
                    ? std::nullopt
                    : std::optional<tt::tt_fabric::SenderWorkerAdapterSpec>(
                          local_fabric_handle->uniquely_connect_worker(
                              device, ttnn::ccl::EdmLineFabricOpInterface::FORWARD));
            std::optional<tt::tt_fabric::SenderWorkerAdapterSpec> backward_fabric_connection =
                line_topology.is_last_device_in_line(ttnn::ccl::EdmLineFabricOpInterface::Direction::BACKWARD)
                    ? std::nullopt
                    : std::optional<tt::tt_fabric::SenderWorkerAdapterSpec>(
                          local_fabric_handle->uniquely_connect_worker(
                              device, ttnn::ccl::EdmLineFabricOpInterface::BACKWARD));

            detail::append_fabric_connection_rt_args(forward_fabric_connection, core, program, writer_runtime_args);
            detail::append_fabric_connection_rt_args(backward_fabric_connection, core, program, writer_runtime_args);
        } else if (packet_worker_cores_grid.contains(core)) {
            reader_runtime_args[is_reader_sender_core_idx] = false;
            reader_runtime_args[is_reader_worker_core_idx] = true;
            reader_runtime_args[is_linear_input_packet_start_idx] = local_page + offset_for_input;

            writer_runtime_args[is_writer_sender_core_idx] = false;
            writer_runtime_args[is_writer_worker_core_idx] = true;
            writer_runtime_args[is_linear_output_page_start_idx] = local_page;

            local_page += num_pages_per_packet;
            if (core == packet_receiver_core) {
                reader_runtime_args[is_reader_receiver_core_idx] = true;
            } else {
                reader_runtime_args[is_reader_receiver_core_idx] = false;
            }
        } else {
            reader_runtime_args[is_reader_sender_core_idx] = false;
            reader_runtime_args[is_reader_worker_core_idx] = false;
            reader_runtime_args[is_reader_receiver_core_idx] = false;
            writer_runtime_args[is_writer_sender_core_idx] = false;
            writer_runtime_args[is_writer_worker_core_idx] = false;
        }
        tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_runtime_args);
    }

    return {
        std::move(program),
        {.unary_reader_kernel_id = unary_reader_kernel_id,
         .unary_writer_kernel_id = unary_writer_kernel_id,
         .compute_kernel_id = compute_kernel_id,
         .cb_handles = {cb_input_tensor_handle, cb_output_tensor_handle, cb_fabric_receiver_handle},
         .core_range = all_cores_grid}};
}

void LlamaReduceScatterDeviceOperation::LlamaReduceScatterAdd::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    auto& unary_reader_kernel_id = cached_program.shared_variables.unary_reader_kernel_id;
    auto& unary_writer_kernel_id = cached_program.shared_variables.unary_writer_kernel_id;

    const auto& input_tensor = tensor_args.input_tensor;
    const auto& intermediate_packet_buffer = tensor_args.intermediate_packet_buffer;
    auto& output_tensor = tensor_return_value;

    auto input_tensor_buffer = input_tensor.buffer();
    auto output_tensor_buffer = output_tensor.buffer();
    auto packet_buffer = intermediate_packet_buffer.buffer();

    auto& all_cores_grid = cached_program.shared_variables.core_range;

    auto cores = corerange_to_cores(all_cores_grid, std::nullopt);

    UpdateDynamicCircularBufferAddress(program, cached_program.shared_variables.cb_handles[0], *input_tensor_buffer);
    UpdateDynamicCircularBufferAddress(program, cached_program.shared_variables.cb_handles[1], *output_tensor_buffer);
    UpdateDynamicCircularBufferAddress(program, cached_program.shared_variables.cb_handles[2], *packet_buffer);

    for (const auto& core : cores) {
        auto& writer_runtime_args = tt::tt_metal::GetRuntimeArgs(program, unary_writer_kernel_id, core);
        writer_runtime_args[0] = (uint32_t)operation_attributes.cross_device_semaphore->address();
        auto& reader_runtime_args = tt::tt_metal::GetRuntimeArgs(program, unary_reader_kernel_id, core);
        reader_runtime_args[0] = (uint32_t)operation_attributes.cross_device_semaphore->address();
    }
}

}  // namespace ttnn::operations::experimental::ccl

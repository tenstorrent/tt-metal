// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "llama_reduce_scatter_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <vector>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/device_pool.hpp>
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_op.hpp"
#include "cpp/ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"
#include "cpp/ttnn/operations/ccl/sharding_addrgen_helper.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/erisc_datamover_builder.hpp>
#include "cpp/ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/fabric.hpp>

namespace ttnn::operations::experimental::ccl {

namespace detail {

std::string device_order_array_string(uint32_t ring_size, uint32_t ring_index, tt::tt_fabric::Topology topology) {
    ttnn::SmallVector<uint32_t> device_order;
    device_order.reserve(ring_size - 1);
    // Add all indices except ring_index
    for (uint32_t i = 0; i < ring_size; i++) {
        if (i != ring_index) {
            device_order.push_back(i);
        }
    }

    if (topology == tt::tt_fabric::Topology::Linear) {
        // Sort based on absolute difference from ring_index in descending order
        std::sort(device_order.begin(), device_order.end(), [ring_index](uint32_t a, uint32_t b) {
            return std::abs(static_cast<int>(a) - static_cast<int>(ring_index)) >
                   std::abs(static_cast<int>(b) - static_cast<int>(ring_index));
        });
    } else if (topology == tt::tt_fabric::Topology::Ring) {
        // Sort based on ring distance
        // 0 -> 1 -> 2 -> ... -> ring_size - 1 -> 0
        std::sort(device_order.begin(), device_order.end(), [ring_index, ring_size](uint32_t a, uint32_t b) {
            // Calculate shortest distance for 'a' from 'ring_index' in a ring of 'ring_size'
            // Cast to int for std::abs to work as expected with unsigned differences.
            // This is safe as ring_index and device IDs (a, b) are expected to be much smaller than INT_MAX.
            uint32_t diff_a = std::abs(static_cast<int>(a) - static_cast<int>(ring_index));
            uint32_t dist_a = std::min(diff_a, ring_size - diff_a);

            // Calculate shortest distance for 'b' from 'ring_index'
            uint32_t diff_b = std::abs(static_cast<int>(b) - static_cast<int>(ring_index));
            uint32_t dist_b = std::min(diff_b, ring_size - diff_b);

            if (dist_a != dist_b) {
                return dist_a > dist_b;
            }
            // Tie-breaking: if distances are equal, sort by the device ID itself.
            // This ensures a stable and predictable order for devices at the same distance.
            return a < b;
        });
    }

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

uint32_t get_num_entries_in_schedule(const std::vector<std::vector<ReadRequest>>& schedule) {
    std::size_t total = 0;
    for (const auto& schedule_per_worker : schedule) {
        total += schedule_per_worker.size();
    }
    return total;
}

uint32_t get_total_num_pages_in_schedule(const std::vector<ReadRequest>& schedule_per_worker) {
    std::size_t total = 0;
    for (const auto& schedule : schedule_per_worker) {
        total += schedule.read_size;
    }
    return total;
}

uint32_t max_shards_per_worker(const std::vector<std::vector<ReadRequest>>& schedule) {
    uint32_t max_shards_per_worker = 0;
    for (const auto& worker_schedule : schedule) {
        max_shards_per_worker = std::max(max_shards_per_worker, (uint32_t)worker_schedule.size());
    }
    return max_shards_per_worker;
}
CoreRangeSet get_llama_ccl_cores() {
    // We externally reserve this core range for CCL cores
    return CoreRangeSet(CoreRange({1, 6}, {2, 7}));
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

ttnn::device_operation::CachedProgram<LlamaReduceScatterDeviceOperation::LlamaReduceScatterAdd::shared_variables_t>
LlamaReduceScatterDeviceOperation::LlamaReduceScatterAdd::create_at_helper(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    tt::tt_metal::Program program{};
    return LlamaReduceScatterDeviceOperation::LlamaReduceScatterAdd::create_at(
        operation_attributes, mesh_coordinate, tensor_args, tensor_return_value, program);
}

LlamaReduceScatterDeviceOperation::LlamaReduceScatterAdd::cached_mesh_workload_t
LlamaReduceScatterDeviceOperation::LlamaReduceScatterAdd::create_mesh_workload(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;
    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = create_at_helper(operation_attributes, coord, tensor_args, tensor_return_value);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(coord, std::move(cached_program.shared_variables));
    }
    return cached_mesh_workload_t(std::move(workload), std::move(shared_variables));
}

ttnn::device_operation::CachedProgram<LlamaReduceScatterDeviceOperation::LlamaReduceScatterAdd::shared_variables_t>
LlamaReduceScatterDeviceOperation::LlamaReduceScatterAdd::create_at(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    tt::tt_metal::Program& program) {
    return {
        std::move(program),
        create_at_program_processing(operation_attributes, mesh_coordinate, tensor_args, tensor_return_value, program)};
}

LlamaReduceScatterDeviceOperation::LlamaReduceScatterAdd::shared_variables_t
LlamaReduceScatterDeviceOperation::LlamaReduceScatterAdd::create_at_program_processing(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    tt::tt_metal::Program& program) {
    using namespace tt;
    using namespace tt::tt_metal;
    using namespace tt::tt_fabric;
    using namespace ttnn::ccl;

    const auto& input_tensor = tensor_args.input_tensor;
    auto mesh_device = input_tensor.mesh_device();
    const auto& mesh_view = mesh_device->get_view();
    const uint32_t ring_devices =
        (operation_attributes.cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();
    TT_FATAL(ring_devices > 1, "reduce_scatter async op will only work for ring_devices > 1, but has {}", ring_devices);

    auto target_device = mesh_device->get_device(mesh_coordinate);

    const uint32_t ring_size = operation_attributes.ring_devices;
    const uint32_t num_devices = ring_size;

    auto topology = operation_attributes.topology;

    uint32_t ring_index = 0;  // Initialize device index
    std::optional<IDevice*> forward_device = std::nullopt;
    std::optional<IDevice*> backward_device = std::nullopt;

    std::vector<IDevice*> devices = (operation_attributes.cluster_axis == 0)
                                        ? mesh_view.get_devices_on_column(mesh_coordinate[1])
                                        : mesh_view.get_devices_on_row(mesh_coordinate[0]);

    for (uint32_t i = 0; i < ring_size; ++i) {
        if (devices.at(i) == target_device) {
            ring_index = i;
            if (i != 0) {
                backward_device = devices.at(i - 1);
            } else if (topology == ttnn::ccl::Topology::Ring) {
                backward_device = devices.at(ring_size - 1);
            }
            if (i != ring_size - 1) {
                forward_device = devices.at(i + 1);
            } else if (topology == ttnn::ccl::Topology::Ring) {
                forward_device = devices.at(0);
            }
        }
    }
    uint32_t num_links = operation_attributes.num_links;

    std::string device_order = detail::device_order_array_string(ring_size, ring_index, topology);

    std::map<std::string, std::string> reader_defines = {{"DEVICE_ORDER", device_order}};

    const auto& input_shape = input_tensor.logical_shape();
    const auto dim = operation_attributes.dim;
    uint32_t rank = input_shape.size();
    auto& output_tensor = tensor_return_value;
    auto& output_shape = output_tensor.logical_shape();
    auto& padded_output_shape = output_tensor.padded_shape();
    const auto& input_tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();
    const auto& output_tile_shape = output_tensor.tensor_spec().tile().get_tile_shape();
    auto input_tensor_width = input_tensor.logical_shape()[-1];
    auto output_tensor_width = output_tensor.logical_shape()[-1];
    auto input_tensor_width_in_tiles = input_tensor.logical_shape()[-1] / input_tile_shape[1];
    auto output_tensor_width_in_tiles = output_tensor.logical_shape()[-1] / output_tile_shape[1];
    auto input_shard_spec = input_tensor.shard_spec().value();
    auto output_shard_spec = output_tensor.shard_spec().value();
    const auto& cross_device_semaphore = operation_attributes.cross_device_semaphore;

    uint32_t input_shard_height = input_shard_spec.shape[0];
    uint32_t input_shard_width = input_shard_spec.shape[1];
    uint32_t input_tiles_per_core_width = input_shard_width / input_tile_shape[1];

    uint32_t output_shard_height = output_shard_spec.shape[0];
    uint32_t output_shard_width = output_shard_spec.shape[1];
    uint32_t output_tiles_per_core_width = output_shard_width / input_tile_shape[1];

    uint32_t ncores_input = (input_tensor_width + input_shard_width - 1) / input_shard_width;
    if (ncores_input % num_devices != 0) {
        ncores_input = ((ncores_input + num_devices - 1) / num_devices) * num_devices;
    }
    uint32_t ncores_output = (output_tensor_width + output_shard_width - 1) / output_shard_width;
    uint32_t input_shard_cores_per_device = ncores_input / num_devices;
    uint32_t output_cores_per_device = ncores_output;

    auto input_tensor_buffer = input_tensor.buffer();
    auto output_tensor_buffer = output_tensor.buffer();
    auto packet_buffer = tensor_args.intermediate_packet_buffer.buffer();
    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());

    uint32_t input_page_size = tile_size(cb_data_format);
    uint32_t output_page_size = tile_size(cb_data_format);

    // Get OP Config, topology config
    std::vector<Tensor> input_tensors = {input_tensor};
    std::vector<Tensor> output_tensors = {output_tensor};

    // need to drop unused cores in shard spec
    auto input_grid = input_shard_spec.grid;
    auto output_grid = output_shard_spec.grid;

    auto sub_device_cores = mesh_device->worker_cores(
        tt::tt_metal::HalProgrammableCoreType::TENSIX,
        operation_attributes.subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0)));


    auto fabric_max_packet_size = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    size_t packet_size_bytes =
        input_tensor.dtype() == DataType::BFLOAT16 ? std::bit_floor(fabric_max_packet_size) : fabric_max_packet_size;
    uint32_t num_pages_per_packet = packet_size_bytes / input_page_size;
    auto per_worker_num_tiles = (output_tensor_width_in_tiles + num_links - 1) / num_links;
    if (per_worker_num_tiles < num_pages_per_packet) {  // if num_tiles per worker is smaller than packet size
        packet_size_bytes = per_worker_num_tiles * input_page_size;
        num_pages_per_packet = packet_size_bytes / input_page_size;
    }
    auto num_packets_to_send = (output_tensor_width_in_tiles + num_pages_per_packet - 1) / num_pages_per_packet;
    auto num_packets_to_send_per_worker = (num_packets_to_send + num_links - 1) / num_links;

    TT_FATAL(
        num_pages_per_packet % input_tiles_per_core_width == 0 || input_tiles_per_core_width > num_pages_per_packet,
        "must have num_pages per packet divisible by num_tiles per core, or num_tiles per core larger than num_pages "
        "per packet");

    uint32_t num_workers_per_link = 1;

    auto intermediate_packet_buffer_grid = tensor_args.intermediate_packet_buffer.shard_spec().value().grid;
    // UNCOMMENT this once we can allocate persistent buffers across all device lifetimes
    uint32_t num_packets_total_per_device =
        (input_shard_cores_per_device * input_tiles_per_core_width + num_pages_per_packet - 1) / num_pages_per_packet;
    auto packet_worker_cores_grid = detail::get_worker_cores(
        intermediate_packet_buffer_grid,
        num_packets_total_per_device,
        input_shard_spec.orientation == ShardOrientation::ROW_MAJOR);

    auto available_cores = detail::get_llama_ccl_cores();

    auto sender_core_grid = detail::get_worker_cores(
        available_cores, num_workers_per_link * num_links, input_shard_spec.orientation == ShardOrientation::ROW_MAJOR);
    auto all_cores_grid = packet_worker_cores_grid.merge(sender_core_grid);

    auto schedule = detail::distribute_work_evenly(
        input_shard_cores_per_device,
        num_workers_per_link * num_links,
        input_tiles_per_core_width,
        num_pages_per_packet);
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
            input_tiles_per_core_width * input_page_size, {{input_tensor_cb_id, cb_data_format}})
            .set_page_size(input_tensor_cb_id, input_page_size)
            .set_globally_allocated_address(*input_tensor_buffer);
    // CB to represent the output sharded buffer
    tt::tt_metal::CircularBufferConfig cb_output_tensor_config =
        tt::tt_metal::CircularBufferConfig(
            output_tiles_per_core_width * input_page_size, {{output_tensor_cb_id, cb_data_format}})
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

    uint32_t max_shards_per_worker = detail::max_shards_per_worker(schedule);
    uint32_t num_shards_total = max_shards_per_worker * (num_devices - 1);
    uint32_t num_pages_total = num_shards_total * input_tiles_per_core_width;

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
            num_pages_total * input_page_size, {{fabric_sender_cb_index, cb_data_format}})
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
            buffering_factor * num_pages_per_packet * output_tiles_per_core_width * input_page_size * num_devices,
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
        corerange_to_cores(input_grid, std::nullopt, input_shard_spec.orientation == ShardOrientation::ROW_MAJOR);
    auto output_cores =
        corerange_to_cores(output_grid, std::nullopt, input_shard_spec.orientation == ShardOrientation::ROW_MAJOR);
    auto sender_cores =
        corerange_to_cores(sender_core_grid, std::nullopt, input_shard_spec.orientation == ShardOrientation::ROW_MAJOR);
    auto all_cores =
        corerange_to_cores(all_cores_grid, std::nullopt, input_shard_spec.orientation == ShardOrientation::ROW_MAJOR);
    auto packet_worker_cores = corerange_to_cores(
        packet_worker_cores_grid, std::nullopt, input_shard_spec.orientation == ShardOrientation::ROW_MAJOR);
    auto packet_receiver_core = packet_worker_cores.at(0);

    const uint32_t chip_id = ring_index;

    auto to_worker_cores = [mesh_device](
                               const std::vector<CoreCoord>& cores,
                               std::optional<uint32_t> num_max_cores = std::nullopt) -> std::vector<CoreCoord> {
        std::vector<CoreCoord> worker_cores;
        auto num_cores = num_max_cores.has_value() ? num_max_cores.value() : cores.size();
        for (uint32_t i = 0; i < num_cores; ++i) {
            const auto& core = cores[i];
            worker_cores.push_back(mesh_device->worker_core_from_logical_core(core));
        }
        return worker_cores;
    };

    auto packet_bounding_box = packet_worker_cores_grid.bounding_box();
    auto packet_start_worker_core = to_worker_cores({packet_bounding_box.start_coord});
    auto packet_end_worker_core = to_worker_cores({packet_bounding_box.end_coord});
    auto total_num_read_txns = detail::get_num_entries_in_schedule(schedule);

    std::vector<uint32_t> reader_compile_time_args = {
        input_tensor_cb_id,
        fabric_sender_cb_index,
        packet_header_cb_index,
        fabric_receiver_cb_index,
        accumulator_cb_index,
        output_tensor_cb_id,
        (uint32_t)chip_id,
        input_tiles_per_core_width,
        output_tiles_per_core_width,
        num_pages_per_packet,
        input_shard_cores_per_device,
        num_devices,
        input_page_size,
        output_cores_per_device,
        packet_start_worker_core.at(0).x,
        packet_start_worker_core.at(0).y,
        packet_end_worker_core.at(0).x,
        packet_end_worker_core.at(0).y,
        sender_cores.size(),
        total_num_read_txns};

    if (packet_worker_cores_grid.num_cores() == 1) {
        reader_defines["SKIP_MCAST"] = "1";
    }
    reader_defines["INPUT_CORE_XY"] = detail::cores_to_string(to_worker_cores(input_cores, ncores_input));
    reader_defines["OUTPUT_CORE_XY"] = detail::cores_to_string(to_worker_cores(output_cores, ncores_output));
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
    auto num_packet_worker_cores = packet_worker_cores.size();

    std::vector<uint32_t> writer_compile_time_args = {
        input_tensor_cb_id,
        fabric_sender_cb_index,
        packet_header_cb_index,
        fabric_receiver_cb_index,
        accumulator_cb_index,
        output_tensor_cb_id,
        (uint32_t)chip_id,
        input_tiles_per_core_width,
        output_tiles_per_core_width,
        num_pages_per_packet,
        input_shard_cores_per_device,
        num_devices,
        input_page_size,
        output_cores_per_device,
        packet_receiver_worker_core.x,
        packet_receiver_worker_core.y,
        num_packet_worker_cores,
        topology == tt::tt_fabric::Topology::Ring ? 1u : 0u};

    auto writer_defines = reader_defines;
    bool skip_write_back = output_cores == packet_worker_cores and num_pages_per_packet == output_tiles_per_core_width;
    if (skip_write_back) {
        writer_defines["SKIP_WRITE_BACK"] = "1";
    }
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

    auto output_cb_index = skip_write_back ? output_tensor_cb_id : accumulator_cb_index;
    const std::vector<uint32_t> compute_compile_time_args = {
        fabric_receiver_cb_index, output_cb_index, num_devices, output_tiles_per_core_width, num_pages_per_packet};

    bool fp32_dest_acc_en = cb_data_format == tt::DataFormat::Float32;
    const auto compute_kernel_file =
        "ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter/device/kernels/compute/reduction.cpp";
    const auto compute_kernel_id = tt_metal::CreateKernel(
        program,
        compute_kernel_file,
        packet_worker_cores_grid,
        tt_metal::ComputeConfig{.fp32_dest_acc_en = fp32_dest_acc_en, .compile_args = compute_compile_time_args});

    uint32_t offset_for_input = chip_id * input_shard_cores_per_device * input_tiles_per_core_width;
    uint32_t local_page = 0;

    std::vector<uint32_t> reader_runtime_args = {
        cross_device_semaphore->address(), local_semaphore, false, false, 0, false, 0, 0, 0};
    uint32_t is_reader_sender_core_idx = 2;
    uint32_t is_reader_worker_core_idx = 3;
    uint32_t is_linear_input_packet_start_idx = 4;
    uint32_t is_reader_receiver_core_idx = 5;
    uint32_t reader_sender_packet_start_idx = 6;
    uint32_t reader_sender_packet_end_idx = 7;
    uint32_t reader_sender_total_num_pages_idx = 8;

    uint32_t is_writer_sender_core_idx = 2;
    uint32_t is_writer_worker_core_idx = 3;
    uint32_t is_linear_output_page_start_idx = 4;
    uint32_t writer_sender_packet_start_idx = 5;
    uint32_t writer_sender_packet_end_idx = 6;
    uint32_t writer_sender_total_num_pages_idx = 7;

    uint32_t reader_sender_packet_start = 0;
    uint32_t writer_sender_packet_start = 0;
    uint32_t sender_core_idx = 0;

    uint32_t link_idx = 0;

    bool forward_fabric_connection = forward_device.has_value();
    bool backward_fabric_connection = backward_device.has_value();

    for (auto core : all_cores) {
        std::vector<uint32_t> writer_runtime_args = {
            cross_device_semaphore->address(), local_semaphore, false, false, 0, 0, 0, 0};

        uint32_t num_shards_to_read_per_worker = schedule[sender_core_idx].size();

        if (sender_core_grid.contains(core)) {
            auto sender_total_num_pages = detail::get_total_num_pages_in_schedule(schedule[sender_core_idx]);

            reader_runtime_args[is_reader_sender_core_idx] = true;
            reader_runtime_args[is_reader_worker_core_idx] = false;
            reader_runtime_args[is_reader_receiver_core_idx] = false;
            reader_runtime_args[reader_sender_packet_start_idx] = reader_sender_packet_start;
            reader_runtime_args[reader_sender_packet_end_idx] =
                reader_sender_packet_start + num_shards_to_read_per_worker;
            reader_runtime_args[reader_sender_total_num_pages_idx] = sender_total_num_pages;

            writer_runtime_args[is_writer_sender_core_idx] = true;
            writer_runtime_args[is_writer_worker_core_idx] = false;
            writer_runtime_args[writer_sender_packet_start_idx] = writer_sender_packet_start;
            writer_runtime_args[writer_sender_packet_end_idx] =
                writer_sender_packet_start + num_packets_to_send_per_worker;
            writer_runtime_args[writer_sender_total_num_pages_idx] = sender_total_num_pages;

            reader_sender_packet_start += num_shards_to_read_per_worker;
            writer_sender_packet_start += num_packets_to_send_per_worker;
            sender_core_idx++;

            writer_runtime_args.push_back(forward_fabric_connection);
            if (forward_fabric_connection) {
                tt::tt_fabric::append_fabric_connection_rt_args(
                    target_device->id(), forward_device.value()->id(), link_idx, program, core, writer_runtime_args);
            }

            writer_runtime_args.push_back(backward_fabric_connection);
            if (backward_fabric_connection) {
                tt::tt_fabric::append_fabric_connection_rt_args(
                    target_device->id(), backward_device.value()->id(), link_idx, program, core, writer_runtime_args);
            }

            link_idx++;
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
        .unary_reader_kernel_id = unary_reader_kernel_id,
        .unary_writer_kernel_id = unary_writer_kernel_id,
        .compute_kernel_id = compute_kernel_id,
        .cb_handles = {cb_input_tensor_handle, cb_output_tensor_handle, cb_fabric_receiver_handle},
        .core_range = all_cores_grid};
}

void LlamaReduceScatterDeviceOperation::LlamaReduceScatterAdd::override_runtime_arguments_per_program(
    const LlamaReduceScatterDeviceOperation::LlamaReduceScatterAdd::shared_variables_t& shared_variables,
    tt::tt_metal::Program& program,
    const LlamaReduceScatterDeviceOperation::operation_attributes_t& operation_attributes,
    const LlamaReduceScatterDeviceOperation::tensor_args_t& tensor_args,
    LlamaReduceScatterDeviceOperation::tensor_return_value_t& tensor_return_value) {
    auto& unary_reader_kernel_id = shared_variables.unary_reader_kernel_id;
    auto& unary_writer_kernel_id = shared_variables.unary_writer_kernel_id;

    const auto& input_tensor = tensor_args.input_tensor;
    const auto& intermediate_packet_buffer = tensor_args.intermediate_packet_buffer;
    auto& output_tensor = tensor_return_value;

    auto input_tensor_buffer = input_tensor.buffer();
    auto output_tensor_buffer = output_tensor.buffer();
    auto packet_buffer = intermediate_packet_buffer.buffer();

    auto& all_cores_grid = shared_variables.core_range;

    auto cores = corerange_to_cores(all_cores_grid, std::nullopt);

    UpdateDynamicCircularBufferAddress(program, shared_variables.cb_handles[0], *input_tensor_buffer);
    UpdateDynamicCircularBufferAddress(program, shared_variables.cb_handles[1], *output_tensor_buffer);
    UpdateDynamicCircularBufferAddress(program, shared_variables.cb_handles[2], *packet_buffer);

    for (const auto& core : cores) {
        auto& writer_runtime_args = tt::tt_metal::GetRuntimeArgs(program, unary_writer_kernel_id, core);
        writer_runtime_args[0] = (uint32_t)operation_attributes.cross_device_semaphore->address();
        auto& reader_runtime_args = tt::tt_metal::GetRuntimeArgs(program, unary_reader_kernel_id, core);
        reader_runtime_args[0] = (uint32_t)operation_attributes.cross_device_semaphore->address();
    }
}

void LlamaReduceScatterDeviceOperation::LlamaReduceScatterAdd::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        const auto& shared_variables = cached_workload.shared_variables.at(range);
        override_runtime_arguments_per_program(
            shared_variables, program, operation_attributes, tensor_args, tensor_return_value);
    }
}

}  // namespace ttnn::operations::experimental::ccl

// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "llama_reduce_scatter_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <vector>
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/experimental/ccl/llama_common.hpp"
#include "ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"
#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"
#include <tt-metalium/core_coord.hpp>
#include "ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"

namespace ttnn::operations::experimental::ccl {

namespace detail {

std::string device_order_array_string(uint32_t ring_size, uint32_t ring_index, tt::tt_fabric::Topology topology) {
    ttsl::SmallVector<uint32_t> device_order;
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
        uint32_t max_pages_for_worker = (packets_remaining_for_worker * packet_size) - leftover_in_packet;

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
    size_t total_num_requests = 0;
    for (const auto& chunk : schedule) {
        total_num_requests += chunk.size();
    }
    schedule_flattened.reserve(total_num_requests);
    for (const auto& chunk : schedule) {
        schedule_flattened.insert(schedule_flattened.end(), chunk.begin(), chunk.end());
    }
    return schedule_flattened;
}

std::string schedule_to_string(const std::vector<std::vector<ReadRequest>>& schedule) {
    auto flattened_schedule = flatten_schedule(schedule);
    std::string result = "{";
    for (const auto& entry : flattened_schedule) {
        result += "{" + std::to_string(entry.bank_id) + ", " + std::to_string(entry.read_offset) + ", " +
                  std::to_string(entry.read_size) + "}, ";
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

// Shared by the Program builder and the ProgramDescriptor builder.
struct LlamaReduceScatterPlan {
    static constexpr uint32_t input_tensor_cb_id = tt::CBIndex::c_0;
    static constexpr uint32_t output_tensor_cb_id = tt::CBIndex::c_1;
    static constexpr uint32_t packet_header_cb_index = tt::CBIndex::c_2;
    static constexpr uint32_t fabric_sender_cb_index = tt::CBIndex::c_3;
    static constexpr uint32_t fabric_receiver_cb_index = tt::CBIndex::c_4;
    static constexpr uint32_t accumulator_cb_index = tt::CBIndex::c_5;
    static constexpr uint32_t buffering_factor = 2;
    static constexpr uint32_t num_packet_headers_storable = 8;

    uint32_t ring_size = 0;
    uint32_t ring_index = 0;
    tt::tt_fabric::Topology topology = tt::tt_fabric::Topology::Linear;
    std::optional<tt::tt_fabric::FabricNodeId> forward_fabric_node_id;
    std::optional<tt::tt_fabric::FabricNodeId> backward_fabric_node_id;
    uint32_t num_links = 0;
    uint32_t input_tiles_per_core_width = 0;
    uint32_t output_tiles_per_core_width = 0;
    uint32_t ncores_input = 0;
    uint32_t ncores_output = 0;
    uint32_t input_shard_cores_per_device = 0;
    uint32_t output_cores_per_device = 0;
    uint32_t input_page_size = 0;
    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    tt::tt_metal::Buffer* input_tensor_buffer = nullptr;
    tt::tt_metal::Buffer* output_tensor_buffer = nullptr;
    tt::tt_metal::Buffer* packet_buffer = nullptr;
    uint32_t num_pages_per_packet = 0;
    uint32_t num_packets_to_send_per_worker = 0;
    uint32_t num_pages_total = 0;
    uint32_t packet_header_size_bytes = 0;
    CoreRangeSet input_grid;
    CoreRangeSet output_grid;
    CoreRangeSet sender_core_grid;
    CoreRangeSet all_cores_grid;
    CoreRangeSet packet_worker_cores_grid;
    std::vector<std::vector<ReadRequest>> schedule;
    std::vector<CoreCoord> all_cores;
    CoreCoord packet_receiver_core;
    std::map<std::string, std::string> reader_defines;
    std::map<std::string, std::string> writer_defines;
    std::vector<uint32_t> reader_compile_time_args;
    std::vector<uint32_t> writer_compile_time_args;
    std::vector<uint32_t> compute_compile_time_args;
    bool skip_write_back = false;
    bool fp32_dest_acc_en = false;
    uint32_t output_cb_index = output_tensor_cb_id;
};

struct LlamaCoreArgCursor {
    uint32_t reader_sender_packet_start = 0;
    uint32_t writer_sender_packet_start = 0;
    uint32_t sender_core_idx = 0;
    uint32_t local_page = 0;
    uint32_t link_idx = 0;
};

struct LlamaCoreRuntimeArgs {
    std::vector<uint32_t> reader;
    std::vector<uint32_t> writer;
    bool is_sender = false;
    uint32_t link_idx = 0;
};

LlamaReduceScatterPlan make_llama_reduce_scatter_plan(
    const LlamaReduceScatterDeviceOperation::operation_attributes_t& operation_attributes,
    const LlamaReduceScatterDeviceOperation::tensor_args_t& tensor_args,
    LlamaReduceScatterDeviceOperation::tensor_return_value_t& tensor_return_value,
    const ttnn::MeshCoordinate& coord,
    bool fused_signaler) {
    using namespace tt;
    using namespace tt::tt_metal;

    LlamaReduceScatterPlan plan;
    const auto& input_tensor = tensor_args.input_tensor;
    auto* mesh_device = input_tensor.device();
    const auto& mesh_view = mesh_device->get_view();
    const uint32_t ring_devices =
        (operation_attributes.cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();
    TT_FATAL(ring_devices > 1, "reduce_scatter async op will only work for ring_devices > 1, but has {}", ring_devices);

    plan.ring_size = operation_attributes.ring_devices;
    plan.topology = operation_attributes.topology;
    plan.ring_index = coord[operation_attributes.cluster_axis];
    plan.num_links = operation_attributes.num_links;
    const auto fabric_node_ids = (operation_attributes.cluster_axis == 0)
                                     ? mesh_view.get_fabric_node_ids_on_column(coord[1])
                                     : mesh_view.get_fabric_node_ids_on_row(coord[0]);
    if (plan.ring_index != 0) {
        plan.backward_fabric_node_id = fabric_node_ids.at(plan.ring_index - 1);
    } else if (plan.topology == tt::tt_fabric::Topology::Ring) {
        plan.backward_fabric_node_id = fabric_node_ids.at(plan.ring_size - 1);
    }
    if (plan.ring_index != plan.ring_size - 1) {
        plan.forward_fabric_node_id = fabric_node_ids.at(plan.ring_index + 1);
    } else if (plan.topology == tt::tt_fabric::Topology::Ring) {
        plan.forward_fabric_node_id = fabric_node_ids.at(0);
    }

    auto& output_tensor = tensor_return_value;
    const auto& input_tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();
    const auto& output_tile_shape = output_tensor.tensor_spec().tile().get_tile_shape();
    const auto input_tensor_width = input_tensor.logical_shape()[-1];
    const auto output_tensor_width = output_tensor.logical_shape()[-1];
    const auto output_tensor_width_in_tiles = output_tensor.logical_shape()[-1] / output_tile_shape[1];
    const auto input_shard_spec = input_tensor.shard_spec().value();
    const auto output_shard_spec = output_tensor.shard_spec().value();
    const uint32_t input_shard_width = input_shard_spec.shape[1];
    plan.input_tiles_per_core_width = input_shard_width / input_tile_shape[1];
    const uint32_t output_shard_width = output_shard_spec.shape[1];
    plan.output_tiles_per_core_width = output_shard_width / input_tile_shape[1];
    plan.ncores_input = (input_tensor_width + input_shard_width - 1) / input_shard_width;
    if (plan.ncores_input % plan.ring_size != 0) {
        plan.ncores_input = ((plan.ncores_input + plan.ring_size - 1) / plan.ring_size) * plan.ring_size;
    }
    plan.ncores_output = (output_tensor_width + output_shard_width - 1) / output_shard_width;
    plan.input_shard_cores_per_device = plan.ncores_input / plan.ring_size;
    plan.output_cores_per_device = plan.ncores_output;
    plan.input_tensor_buffer = input_tensor.buffer();
    plan.output_tensor_buffer = output_tensor.buffer();
    plan.packet_buffer = tensor_args.intermediate_packet_buffer.buffer();
    plan.cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    plan.input_page_size = tile_size(plan.cb_data_format);
    plan.input_grid = input_shard_spec.grid;
    plan.output_grid = output_shard_spec.grid;

    const auto fabric_max_packet_size = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes(*input_tensor.device());
    size_t packet_size_bytes =
        input_tensor.dtype() == DataType::BFLOAT16 ? std::bit_floor(fabric_max_packet_size) : fabric_max_packet_size;
    plan.num_pages_per_packet = packet_size_bytes / plan.input_page_size;
    const auto per_worker_num_tiles = (output_tensor_width_in_tiles + plan.num_links - 1) / plan.num_links;
    if (per_worker_num_tiles < plan.num_pages_per_packet) {
        packet_size_bytes = per_worker_num_tiles * plan.input_page_size;
        plan.num_pages_per_packet = packet_size_bytes / plan.input_page_size;
    }
    const auto num_packets_to_send =
        (output_tensor_width_in_tiles + plan.num_pages_per_packet - 1) / plan.num_pages_per_packet;
    plan.num_packets_to_send_per_worker = (num_packets_to_send + plan.num_links - 1) / plan.num_links;
    TT_FATAL(
        plan.num_pages_per_packet % plan.input_tiles_per_core_width == 0 ||
            plan.input_tiles_per_core_width > plan.num_pages_per_packet,
        "must have num_pages per packet divisible by num_tiles per core, or num_tiles per core larger than num_pages "
        "per packet");

    constexpr uint32_t num_workers_per_link = 1;
    const auto intermediate_packet_buffer_grid = tensor_args.intermediate_packet_buffer.shard_spec().value().grid;
    const uint32_t num_packets_total_per_device =
        (plan.input_shard_cores_per_device * plan.input_tiles_per_core_width + plan.num_pages_per_packet - 1) /
        plan.num_pages_per_packet;
    const bool row_major = input_shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    plan.packet_worker_cores_grid =
        get_worker_cores(intermediate_packet_buffer_grid, num_packets_total_per_device, row_major);
    plan.schedule = distribute_work_evenly(
        plan.input_shard_cores_per_device,
        num_workers_per_link * plan.num_links,
        plan.input_tiles_per_core_width,
        plan.num_pages_per_packet);
    const auto schedule_string = schedule_to_string(plan.schedule);
    std::tie(plan.sender_core_grid, plan.all_cores_grid) =
        LlamaReduceScatterDeviceOperation::get_rs_core_grids(operation_attributes, tensor_args);
    plan.packet_header_size_bytes = static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_packet_header_size_bytes());
    const uint32_t max_shards = max_shards_per_worker(plan.schedule);
    plan.num_pages_total = max_shards * (plan.ring_size - 1) * plan.input_tiles_per_core_width;

    plan.all_cores = corerange_to_cores(plan.all_cores_grid, std::nullopt, row_major);
    const auto input_cores = corerange_to_cores(plan.input_grid, std::nullopt, row_major);
    const auto output_cores = corerange_to_cores(plan.output_grid, std::nullopt, row_major);
    const auto sender_cores = corerange_to_cores(plan.sender_core_grid, std::nullopt, row_major);
    const auto packet_worker_cores = corerange_to_cores(plan.packet_worker_cores_grid, std::nullopt, row_major);
    plan.packet_receiver_core = packet_worker_cores.at(0);

    auto to_worker_cores = [mesh_device](
                               const std::vector<CoreCoord>& cores,
                               std::optional<uint32_t> num_max_cores = std::nullopt) -> std::vector<CoreCoord> {
        std::vector<CoreCoord> worker_cores;
        const auto num_cores = num_max_cores.has_value() ? num_max_cores.value() : cores.size();
        worker_cores.reserve(num_cores);
        for (uint32_t i = 0; i < num_cores; ++i) {
            worker_cores.push_back(mesh_device->worker_core_from_logical_core(cores[i]));
        }
        return worker_cores;
    };
    const auto packet_bounding_box = plan.packet_worker_cores_grid.bounding_box();
    const auto packet_start_worker_core = to_worker_cores({packet_bounding_box.start_coord});
    const auto packet_end_worker_core = to_worker_cores({packet_bounding_box.end_coord});
    const auto packet_receiver_worker_core = to_worker_cores({plan.packet_receiver_core}).at(0);
    const auto total_num_read_txns = get_num_entries_in_schedule(plan.schedule);

    plan.reader_defines = {{"DEVICE_ORDER", device_order_array_string(plan.ring_size, plan.ring_index, plan.topology)}};
    if (plan.packet_worker_cores_grid.num_cores() == 1) {
        plan.reader_defines["SKIP_MCAST"] = "1";
    }
    plan.reader_defines["INPUT_CORE_XY"] = cores_to_string(to_worker_cores(input_cores, plan.ncores_input));
    plan.reader_defines["OUTPUT_CORE_XY"] = cores_to_string(to_worker_cores(output_cores, plan.ncores_output));
    plan.reader_defines["PACKET_WORKER_CORES"] = cores_to_string(to_worker_cores(packet_worker_cores));
    plan.reader_defines["SCHEDULE"] = schedule_string;
    plan.reader_compile_time_args = {
        LlamaReduceScatterPlan::input_tensor_cb_id,
        LlamaReduceScatterPlan::fabric_sender_cb_index,
        LlamaReduceScatterPlan::packet_header_cb_index,
        LlamaReduceScatterPlan::fabric_receiver_cb_index,
        LlamaReduceScatterPlan::accumulator_cb_index,
        LlamaReduceScatterPlan::output_tensor_cb_id,
        plan.ring_index,
        plan.input_tiles_per_core_width,
        plan.output_tiles_per_core_width,
        plan.num_pages_per_packet,
        plan.input_shard_cores_per_device,
        plan.ring_size,
        plan.input_page_size,
        plan.output_cores_per_device,
        packet_start_worker_core.at(0).x,
        packet_start_worker_core.at(0).y,
        packet_end_worker_core.at(0).x,
        packet_end_worker_core.at(0).y,
        static_cast<uint32_t>(sender_cores.size()),
        total_num_read_txns,
        fused_signaler ? 1u : 0u};
    plan.writer_compile_time_args = {
        LlamaReduceScatterPlan::fabric_sender_cb_index,
        LlamaReduceScatterPlan::packet_header_cb_index,
        LlamaReduceScatterPlan::fabric_receiver_cb_index,
        LlamaReduceScatterPlan::accumulator_cb_index,
        LlamaReduceScatterPlan::output_tensor_cb_id,
        plan.ring_index,
        plan.input_tiles_per_core_width,
        plan.output_tiles_per_core_width,
        plan.num_pages_per_packet,
        plan.input_shard_cores_per_device,
        plan.ring_size,
        plan.input_page_size,
        plan.output_cores_per_device,
        packet_receiver_worker_core.x,
        packet_receiver_worker_core.y,
        static_cast<uint32_t>(packet_worker_cores.size()),
        plan.topology == tt::tt_fabric::Topology::Ring ? 1u : 0u};
    plan.writer_defines = plan.reader_defines;
    plan.skip_write_back =
        output_cores == packet_worker_cores && plan.num_pages_per_packet == plan.output_tiles_per_core_width;
    if (plan.skip_write_back) {
        plan.writer_defines["SKIP_WRITE_BACK"] = "1";
    }
    plan.output_cb_index = plan.skip_write_back ? LlamaReduceScatterPlan::output_tensor_cb_id
                                                : LlamaReduceScatterPlan::accumulator_cb_index;
    plan.fp32_dest_acc_en = plan.cb_data_format == tt::DataFormat::Float32;
    plan.compute_compile_time_args = {
        LlamaReduceScatterPlan::fabric_receiver_cb_index,
        plan.output_cb_index,
        plan.ring_size,
        plan.output_tiles_per_core_width,
        plan.num_pages_per_packet};
    return plan;
}

LlamaCoreRuntimeArgs fill_llama_core_runtime_args(
    const LlamaReduceScatterPlan& plan, LlamaCoreArgCursor& cursor, uint32_t local_semaphore, const CoreCoord& core) {
    constexpr uint32_t is_reader_sender_core_idx = 2;
    constexpr uint32_t is_reader_worker_core_idx = 3;
    constexpr uint32_t is_linear_input_packet_start_idx = 4;
    constexpr uint32_t is_reader_receiver_core_idx = 5;
    constexpr uint32_t reader_sender_packet_start_idx = 6;
    constexpr uint32_t reader_sender_packet_end_idx = 7;
    constexpr uint32_t reader_sender_total_num_pages_idx = 8;
    constexpr uint32_t is_writer_sender_core_idx = 2;
    constexpr uint32_t is_writer_worker_core_idx = 3;
    constexpr uint32_t is_linear_output_page_start_idx = 4;
    constexpr uint32_t writer_sender_packet_start_idx = 5;
    constexpr uint32_t writer_sender_packet_end_idx = 6;
    constexpr uint32_t writer_sender_total_num_pages_idx = 7;

    LlamaCoreRuntimeArgs args;
    args.reader = {0u, local_semaphore, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
    args.writer = {0u, local_semaphore, 0u, 0u, 0u, 0u, 0u, 0u};
    const uint32_t num_shards_to_read_per_worker = plan.schedule[cursor.sender_core_idx].size();
    if (plan.sender_core_grid.contains(core)) {
        const auto sender_total_num_pages = get_total_num_pages_in_schedule(plan.schedule[cursor.sender_core_idx]);
        args.is_sender = true;
        args.link_idx = cursor.link_idx;
        args.reader[is_reader_sender_core_idx] = 1u;
        args.reader[is_reader_worker_core_idx] = 0u;
        args.reader[is_reader_receiver_core_idx] = 0u;
        args.reader[reader_sender_packet_start_idx] = cursor.reader_sender_packet_start;
        args.reader[reader_sender_packet_end_idx] = cursor.reader_sender_packet_start + num_shards_to_read_per_worker;
        args.reader[reader_sender_total_num_pages_idx] = sender_total_num_pages;
        args.writer[is_writer_sender_core_idx] = 1u;
        args.writer[is_writer_worker_core_idx] = 0u;
        args.writer[writer_sender_packet_start_idx] = cursor.writer_sender_packet_start;
        args.writer[writer_sender_packet_end_idx] =
            cursor.writer_sender_packet_start + plan.num_packets_to_send_per_worker;
        args.writer[writer_sender_total_num_pages_idx] = sender_total_num_pages;
        cursor.reader_sender_packet_start += num_shards_to_read_per_worker;
        cursor.writer_sender_packet_start += plan.num_packets_to_send_per_worker;
        cursor.sender_core_idx++;
        cursor.link_idx++;
    } else if (plan.packet_worker_cores_grid.contains(core)) {
        const uint32_t offset_for_input =
            plan.ring_index * plan.input_shard_cores_per_device * plan.input_tiles_per_core_width;
        args.reader[is_reader_sender_core_idx] = 0u;
        args.reader[is_reader_worker_core_idx] = 1u;
        args.reader[is_linear_input_packet_start_idx] = cursor.local_page + offset_for_input;
        args.reader[is_reader_receiver_core_idx] = core == plan.packet_receiver_core ? 1u : 0u;
        args.writer[is_writer_sender_core_idx] = 0u;
        args.writer[is_writer_worker_core_idx] = 1u;
        args.writer[is_linear_output_page_start_idx] = cursor.local_page;
        cursor.local_page += plan.num_pages_per_packet;
    }
    return args;
}

}  // namespace detail

std::tuple<CoreRangeSet, CoreRangeSet> LlamaReduceScatterDeviceOperation::get_rs_core_grids(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    const uint32_t ring_size = operation_attributes.ring_devices;
    const auto& input_tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();
    auto input_tensor_width = input_tensor.logical_shape()[-1];
    auto input_shard_spec = input_tensor.shard_spec().value();
    uint32_t input_shard_width = input_shard_spec.shape[1];
    uint32_t input_tiles_per_core_width = input_shard_width / input_tile_shape[1];
    uint32_t num_links = operation_attributes.num_links;
    const uint32_t num_devices = ring_size;
    auto intermediate_packet_buffer_grid = tensor_args.intermediate_packet_buffer.shard_spec().value().grid;
    uint32_t ncores_input = (input_tensor_width + input_shard_width - 1) / input_shard_width;
    if (ncores_input % num_devices != 0) {
        ncores_input = ((ncores_input + num_devices - 1) / num_devices) * num_devices;
    }
    uint32_t input_shard_cores_per_device = ncores_input / num_devices;
    auto fabric_max_packet_size = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes(*input_tensor.device());
    size_t packet_size_bytes =
        input_tensor.dtype() == DataType::BFLOAT16 ? std::bit_floor(fabric_max_packet_size) : fabric_max_packet_size;
    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t input_page_size = tile_size(cb_data_format);
    uint32_t num_pages_per_packet = packet_size_bytes / input_page_size;
    uint32_t num_packets_total_per_device =
        (input_shard_cores_per_device * input_tiles_per_core_width + num_pages_per_packet - 1) / num_pages_per_packet;
    auto packet_worker_cores_grid = detail::get_worker_cores(
        intermediate_packet_buffer_grid,
        num_packets_total_per_device,
        input_shard_spec.orientation == ShardOrientation::ROW_MAJOR);

    auto available_cores = llama_specific::get_custom_cores(num_links);

    auto sender_core_grid = detail::get_worker_cores(
        available_cores, num_links, input_shard_spec.orientation == ShardOrientation::ROW_MAJOR);
    auto all_cores_grid = packet_worker_cores_grid.merge(sender_core_grid);
    return {sender_core_grid, all_cores_grid};
}

LlamaReduceScatterDeviceOperation::LlamaReduceScatterAdd::descriptor_artifacts_t
LlamaReduceScatterDeviceOperation::LlamaReduceScatterAdd::create_at_program_descriptor_processing(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    tt::tt_metal::ProgramDescriptor& desc,
    const std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler>& signaler) {
    using namespace tt;
    using namespace tt::tt_metal;
    using namespace tt::tt_fabric;
    using namespace ttnn::ccl;

    TT_FATAL(
        operation_attributes.cross_device_semaphore.has_value(),
        "llama_reduce_scatter requires a cross-device semaphore");
    const auto plan = detail::make_llama_reduce_scatter_plan(
        operation_attributes, tensor_args, tensor_return_value, mesh_coordinate, signaler.has_value());
    auto* mesh_device = tensor_args.input_tensor.device();
    TT_FATAL(plan.input_tensor_buffer != nullptr, "llama_reduce_scatter input tensor must have a buffer");
    TT_FATAL(plan.output_tensor_buffer != nullptr, "llama_reduce_scatter output tensor must have a buffer");
    TT_FATAL(plan.packet_buffer != nullptr, "llama_reduce_scatter packet buffer must have a buffer");
    TT_FATAL(!plan.all_cores.empty(), "llama_reduce_scatter requires at least one worker core");

    auto push_cb = [&](uint32_t cb_id, uint32_t total_size, Buffer* buffer) {
        const bool is_header = cb_id == detail::LlamaReduceScatterPlan::packet_header_cb_index;
        desc.cbs.push_back(CBDescriptor{
            .total_size = total_size,
            .core_ranges = plan.all_cores_grid,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(cb_id),
                .data_format = is_header ? DataFormat::RawUInt32 : plan.cb_data_format,
                .page_size = is_header ? plan.packet_header_size_bytes : plan.input_page_size,
            }}},
            .buffer = buffer,
        });
    };
    push_cb(
        detail::LlamaReduceScatterPlan::input_tensor_cb_id,
        plan.input_tiles_per_core_width * plan.input_page_size,
        plan.input_tensor_buffer);
    push_cb(
        detail::LlamaReduceScatterPlan::output_tensor_cb_id,
        plan.output_tiles_per_core_width * plan.input_page_size,
        plan.output_tensor_buffer);
    push_cb(
        detail::LlamaReduceScatterPlan::packet_header_cb_index,
        detail::LlamaReduceScatterPlan::num_packet_headers_storable * plan.packet_header_size_bytes *
            detail::LlamaReduceScatterPlan::buffering_factor,
        nullptr);
    push_cb(
        detail::LlamaReduceScatterPlan::fabric_receiver_cb_index,
        plan.num_pages_per_packet * plan.ring_size * plan.input_page_size,
        plan.packet_buffer);
    push_cb(
        detail::LlamaReduceScatterPlan::fabric_sender_cb_index, plan.num_pages_total * plan.input_page_size, nullptr);
    push_cb(
        detail::LlamaReduceScatterPlan::accumulator_cb_index,
        detail::LlamaReduceScatterPlan::buffering_factor * plan.num_pages_per_packet *
            plan.output_tiles_per_core_width * plan.input_page_size * plan.ring_size,
        nullptr);

    // An id that is free on every core of the grid, also next to semaphores the caller already put there.
    const uint32_t local_semaphore_id =
        ttnn::experimental::ccl::add_semaphore_descriptor(desc, plan.all_cores_grid, INVALID);

    const auto reader_noc = operation_attributes.use_noc1_only ? NOC::NOC_1 : NOC::RISCV_1_default;
    const auto writer_noc = operation_attributes.use_noc1_only ? NOC::NOC_1 : NOC::RISCV_0_default;
    const auto noc_mode = operation_attributes.use_noc1_only ? NOC_MODE::DM_DYNAMIC_NOC : NOC_MODE::DM_DEDICATED_NOC;
    const auto reader_kernel_index = static_cast<uint32_t>(desc.kernels.size());
    desc.kernels.push_back(KernelDescriptor{
        .kernel_source = "ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter/device/kernels/dataflow/"
                         "reader_llama_reduce_scatter.cpp",
        .core_ranges = plan.all_cores_grid,
        .compile_time_args = plan.reader_compile_time_args,
        .defines = KernelDescriptor::Defines(plan.reader_defines.begin(), plan.reader_defines.end()),
        .config =
            DataMovementConfigDescriptor{
                .processor = DataMovementProcessor::RISCV_1, .noc = reader_noc, .noc_mode = noc_mode},
    });
    const auto writer_kernel_index = static_cast<uint32_t>(desc.kernels.size());
    desc.kernels.push_back(KernelDescriptor{
        .kernel_source = "ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter/device/kernels/dataflow/"
                         "writer_llama_reduce_scatter.cpp",
        .core_ranges = plan.all_cores_grid,
        .compile_time_args = plan.writer_compile_time_args,
        .defines = KernelDescriptor::Defines(plan.writer_defines.begin(), plan.writer_defines.end()),
        .config =
            DataMovementConfigDescriptor{
                .processor = DataMovementProcessor::RISCV_0, .noc = writer_noc, .noc_mode = noc_mode},
    });
    desc.kernels.push_back(KernelDescriptor{
        .kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter/device/kernels/compute/reduction.cpp",
        .core_ranges = plan.packet_worker_cores_grid,
        .compile_time_args = plan.compute_compile_time_args,
        .config = ComputeConfigDescriptor{.fp32_dest_acc_en = plan.fp32_dest_acc_en},
    });

    detail::LlamaCoreArgCursor cursor;
    const uint32_t semaphore_address = static_cast<uint32_t>(operation_attributes.cross_device_semaphore->address());
    for (const auto& core : plan.all_cores) {
        auto core_args = detail::fill_llama_core_runtime_args(plan, cursor, local_semaphore_id, core);
        if (core_args.is_sender) {
            core_args.writer.push_back(static_cast<uint32_t>(plan.forward_fabric_node_id.has_value()));
            if (plan.forward_fabric_node_id.has_value()) {
                const auto target_device_fabric_node_id = mesh_device->get_fabric_node_id(mesh_coordinate);
                tt::tt_fabric::append_fabric_connection_rt_args<ProgramDescriptor>(
                    target_device_fabric_node_id,
                    plan.forward_fabric_node_id.value(),
                    core_args.link_idx,
                    desc,
                    core,
                    core_args.writer);
            }
            core_args.writer.push_back(static_cast<uint32_t>(plan.backward_fabric_node_id.has_value()));
            if (plan.backward_fabric_node_id.has_value()) {
                const auto target_device_fabric_node_id = mesh_device->get_fabric_node_id(mesh_coordinate);
                tt::tt_fabric::append_fabric_connection_rt_args<ProgramDescriptor>(
                    target_device_fabric_node_id,
                    plan.backward_fabric_node_id.value(),
                    core_args.link_idx,
                    desc,
                    core,
                    core_args.writer);
            }
        }
        if (signaler.has_value()) {
            signaler.value().push_llama_rs_rt_args_for_rs(core_args.reader);
        }
        // smuggled-rta-ok: caller GlobalSemaphore, re-applied by apply_cross_device_semaphore
        core_args.reader[kSemaphoreArgIdx] = semaphore_address;
        core_args.writer[kSemaphoreArgIdx] = semaphore_address;
        KernelDescriptor::RTArgList reader_rt;
        reader_rt.append(core_args.reader);
        desc.kernels[reader_kernel_index].emplace_runtime_args(core, reader_rt);
        KernelDescriptor::RTArgList writer_rt;
        writer_rt.append(core_args.writer);
        desc.kernels[writer_kernel_index].emplace_runtime_args(core, writer_rt);
    }
    return descriptor_artifacts_t{
        .reader_kernel_index = reader_kernel_index, .writer_kernel_index = writer_kernel_index};
}

tt::tt_metal::ProgramDescriptor LlamaReduceScatterDeviceOperation::LlamaReduceScatterAdd::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& mesh_coordinate) {
    TT_FATAL(mesh_coordinate.has_value(), "llama_reduce_scatter emits a different program on each device");
    tt::tt_metal::ProgramDescriptor desc;
    const auto artifacts = create_at_program_descriptor_processing(
        operation_attributes, mesh_coordinate.value(), tensor_args, tensor_return_value, desc, std::nullopt);
    // Built into an empty descriptor, so the reader and writer are kernels[0] and kernels[1], where
    // override_runtime_arguments looks for the semaphore slot.
    TT_FATAL(
        artifacts.reader_kernel_index == kReaderKernelIdx && artifacts.writer_kernel_index == kWriterKernelIdx,
        "llama_reduce_scatter: reader/writer landed at kernels {}/{}, expected {}/{}",
        artifacts.reader_kernel_index,
        artifacts.writer_kernel_index,
        kReaderKernelIdx,
        kWriterKernelIdx);
    return desc;
}

void LlamaReduceScatterDeviceOperation::LlamaReduceScatterAdd::apply_cross_device_semaphore(
    tt::tt_metal::Program& program,
    const descriptor_artifacts_t& artifacts,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args) {
    // cross_device_semaphore's address is not part of GlobalSemaphore's reflected attributes, so two
    // calls can share this program and still carry different semaphore addresses.
    TT_FATAL(
        operation_attributes.cross_device_semaphore.has_value(),
        "llama_reduce_scatter requires a cross-device semaphore");
    const uint32_t semaphore_address = static_cast<uint32_t>(operation_attributes.cross_device_semaphore->address());
    const auto all_cores_grid = std::get<1>(get_rs_core_grids(operation_attributes, tensor_args));
    auto& reader_runtime_args = tt::tt_metal::GetRuntimeArgs(program, artifacts.reader_kernel_index);
    auto& writer_runtime_args = tt::tt_metal::GetRuntimeArgs(program, artifacts.writer_kernel_index);
    for (const auto& core : corerange_to_cores(all_cores_grid, std::nullopt)) {
        reader_runtime_args[core.x][core.y][kSemaphoreArgIdx] = semaphore_address;
        writer_runtime_args[core.x][core.y][kSemaphoreArgIdx] = semaphore_address;
    }
}

void LlamaReduceScatterDeviceOperation::LlamaReduceScatterAdd::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>&) {
    // This hook replaces binding patching, so the globally allocated circular buffers are refreshed here by
    // CBIndex, then the caller semaphore, which is not in the program hash.
    TT_FATAL(tensor_args.input_tensor.is_allocated(), "llama_reduce_scatter input tensor must have a buffer");
    TT_FATAL(tensor_return_value.is_allocated(), "llama_reduce_scatter output tensor must have a buffer");
    TT_FATAL(
        tensor_args.intermediate_packet_buffer.is_allocated(), "llama_reduce_scatter packet buffer must have a buffer");

    constexpr uint32_t kExpectedGlobalCbs = 3;
    uint32_t updated_global_cbs = 0;
    for (const auto& cb : program.circular_buffers()) {
        if (!cb->globally_allocated()) {
            continue;
        }
        const auto& indices = cb->buffer_indices();
        if (indices.contains(static_cast<uint8_t>(tt::CBIndex::c_0))) {
            tt::tt_metal::UpdateDynamicCircularBufferAddress(program, cb->id(), *tensor_args.input_tensor.buffer());
        } else if (indices.contains(static_cast<uint8_t>(tt::CBIndex::c_1))) {
            tt::tt_metal::UpdateDynamicCircularBufferAddress(program, cb->id(), *tensor_return_value.buffer());
        } else if (indices.contains(static_cast<uint8_t>(tt::CBIndex::c_4))) {
            tt::tt_metal::UpdateDynamicCircularBufferAddress(
                program, cb->id(), *tensor_args.intermediate_packet_buffer.buffer());
        } else {
            TT_FATAL(false, "llama_reduce_scatter: unexpected globally allocated circular buffer");
        }
        ++updated_global_cbs;
    }
    TT_FATAL(
        updated_global_cbs == kExpectedGlobalCbs,
        "llama_reduce_scatter: expected {} globally allocated circular buffers, updated {}",
        kExpectedGlobalCbs,
        updated_global_cbs);

    apply_cross_device_semaphore(
        program,
        descriptor_artifacts_t{.reader_kernel_index = kReaderKernelIdx, .writer_kernel_index = kWriterKernelIdx},
        operation_attributes,
        tensor_args);
}

}  // namespace ttnn::operations::experimental::ccl

// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_to_all_dispatch_device_operation.hpp"
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
#include <tt-metalium/mesh_graph.hpp>
#include <tt-metalium/hal.hpp>
namespace tt::tt_fabric {
class FabricContext {
public:
    size_t get_fabric_max_payload_size_bytes() const;
};
}  // namespace tt::tt_fabric
#include <limits>

namespace ttnn::operations::ccl {

namespace detail {

std::string stringify_vector(const std::vector<uint32_t>& vec) {
    std::string result = "{";
    for (const auto& elem : vec) {
        result += std::to_string(elem) + ", ";
    }
    result += "}";
    return result;
}

std::string stringify_array(const std::array<bool, 4>& arr) {
    std::string result = "{";
    for (const auto& elem : arr) {
        result += std::to_string(elem) + ", ";
    }
    result += "}";
    return result;
}

tt::tt_metal::Shape2D get_physical_size(const ttnn::Tensor& tensor) {
    auto memory_config = tensor.memory_config();
    auto tensor_spec = tensor.tensor_spec();
    auto page_config = tensor_spec.page_config();
    if (tensor.layout() == tt::tt_metal::Layout::TILE) {
        return page_config.get_tile().get_tile_shape();
    } else if (tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR) {
        if (memory_config.shard_spec().has_value()) {
            return {1, memory_config.shard_spec().value().shape[1]};
        } else {
            return {1, tensor.padded_shape()[-1]};
        }
    } else {
        TT_FATAL(false, "Invalid layout: neither tile nor row major");
        // Fallback return to satisfy compiler; execution never reaches here due to TT_FATAL abort.
        return {0, 0};
    }
}

uint32_t get_num_pages(const ttnn::Tensor& tensor) {
    const auto& shape = tensor.padded_shape();
    auto physical_size = get_physical_size(tensor);
    auto num_pages = shape.volume() / (physical_size.height() * physical_size.width());
    return num_pages;
}

uint32_t get_page_size(const ttnn::Tensor& tensor) {
    auto memory_config = tensor.memory_config();
    auto tensor_spec = tensor.tensor_spec();
    auto page_config = tensor_spec.page_config();
    auto physical_size = get_physical_size(tensor);
    bool sharded = memory_config.shard_spec().has_value();

    std::optional<tt::tt_metal::Shape2D> physical_shard_size =
        sharded ? std::optional<tt::tt_metal::Shape2D>(memory_config.shard_spec().value().shape) : std::nullopt;
    auto page_shape = page_config.get_page_shape(physical_size, tensor.dtype(), memory_config, physical_shard_size);
    auto page_size = page_config.get_page_size_bytes(page_shape, tensor.dtype());
    return (uint32_t)page_size;
}

uint32_t get_aligned_page_size(const ttnn::Tensor& tensor) {
    auto BUFFER_ALIGNMENT = tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                ? tt::tt_metal::hal::get_dram_alignment()
                                : tt::tt_metal::hal::get_l1_alignment();
    return tt::round_up(get_page_size(tensor), BUFFER_ALIGNMENT);
}

uint32_t get_num_rows(const ttnn::Tensor& tensor) {
    auto logical_volume = tensor.logical_shape().volume();
    auto hidden_size = tensor.logical_shape()[-1];
    TT_FATAL(logical_volume % hidden_size == 0, "Logical volume must be divisible by hidden size");
    return logical_volume / hidden_size;
}

uint32_t device_index(const std::vector<tt::tt_metal::IDevice*>& devices, const tt::tt_metal::IDevice* device) {
    for (uint32_t i = 0; i < devices.size(); i++) {
        if (devices[i] == device) {
            return i;
        }
    }
    TT_FATAL(false, "Device not found in device_index");
    return std::numeric_limits<uint32_t>::max();
}

std::pair<std::vector<tt::tt_metal::IDevice*>, std::array<bool, 4>> get_neighbors(
    const MeshDeviceView& mesh_view,
    const MeshCoordinate& mesh_coordinate,
    tt::tt_fabric::Topology& topology,
    const std::optional<uint32_t> axis) {
    std::vector<tt::tt_metal::IDevice*> neighbors;
    std::array<bool, 4> directions;  // east, west, north, south
    if (topology == tt::tt_fabric::Topology::Ring) {
        directions = {true, true, true, true};
    } else {
        directions = {false, false, false, false};
    }
    auto control_plane = tt::tt_fabric::get_control_plane();
    auto src_device = mesh_view.get_device(mesh_coordinate);
    if (axis.has_value()) {
        if (axis.value() == 1) {
            auto axis_neighbors = mesh_view.get_devices_on_row(mesh_coordinate[0]);
            auto index = device_index(axis_neighbors, src_device);
            if (index == 0) {
                // if Line, no neighbor, so don't push back, but for Ring, we need to push back the last device
                neighbors.push_back(axis_neighbors[index + 1]);
                directions[0] = true;  // east
                // if Line, no neighbor, so don't push back, but for Ring, we need to push back the last device
                if (topology == tt::tt_fabric::Topology::Ring) {
                    neighbors.push_back(axis_neighbors[axis_neighbors.size() - 1]);
                    directions[1] = true;
                }
            } else if (index == axis_neighbors.size() - 1) {
                // if Line, no neighbor, so don't push back, but for Ring, we need to push back the first device
                neighbors.push_back(axis_neighbors[0]);
                directions[1] = true;
                if (topology == tt::tt_fabric::Topology::Ring) {
                    neighbors.push_back(axis_neighbors[0]);
                    directions[1] = true;
                }
            } else {
                neighbors.push_back(axis_neighbors[index + 1]);
                directions[0] = true;
                neighbors.push_back(axis_neighbors[index - 1]);
                directions[1] = true;
            }
        } else if (axis.value() == 0) {
            auto axis_neighbors = mesh_view.get_devices_on_column(mesh_coordinate[1]);
            auto index = device_index(axis_neighbors, src_device);
            if (index == 0) {                                    // northmost device
                neighbors.push_back(axis_neighbors[index + 1]);  // south exists
                directions[3] = true;
                if (topology == tt::tt_fabric::Topology::Ring) {
                    neighbors.push_back(axis_neighbors[axis_neighbors.size() - 1]);
                    directions[2] = true;
                }
            } else if (index == axis_neighbors.size() - 1) {
                neighbors.push_back(axis_neighbors[index - 1]);
                directions[2] = true;
                if (topology == tt::tt_fabric::Topology::Ring) {
                    neighbors.push_back(axis_neighbors[0]);
                    directions[3] = true;
                }
            } else {
                neighbors.push_back(axis_neighbors[index + 1]);
                directions[2] = true;
                neighbors.push_back(axis_neighbors[index - 1]);
                directions[3] = true;
            }
        }
    } else {
        {
            auto axis_neighbors = mesh_view.get_devices_on_row(mesh_coordinate[0]);
            auto index = device_index(axis_neighbors, src_device);
            if (index == 0) {
                // if Line, no neighbor, so don't push back, but for Ring, we need to push back the last device
                neighbors.push_back(axis_neighbors[index + 1]);
                directions[0] = true;  // east
                // if Line, no neighbor, so don't push back, but for Ring, we need to push back the last device
                if (topology == tt::tt_fabric::Topology::Ring) {
                    neighbors.push_back(axis_neighbors[axis_neighbors.size() - 1]);
                    directions[1] = true;
                }
            } else if (index == axis_neighbors.size() - 1) {
                // if Line, no neighbor, so don't push back, but for Ring, we need to push back the first device
                neighbors.push_back(axis_neighbors[0]);
                directions[1] = true;
                if (topology == tt::tt_fabric::Topology::Ring) {
                    neighbors.push_back(axis_neighbors[0]);
                    directions[1] = true;
                }
            } else {
                neighbors.push_back(axis_neighbors[index + 1]);
                directions[0] = true;
                neighbors.push_back(axis_neighbors[index - 1]);
                directions[1] = true;
            }
        }
        {
            auto axis_neighbors = mesh_view.get_devices_on_column(mesh_coordinate[1]);
            auto index = device_index(axis_neighbors, src_device);
            if (index == 0) {                                    // northmost device
                neighbors.push_back(axis_neighbors[index + 1]);  // south exists
                directions[3] = true;
                if (topology == tt::tt_fabric::Topology::Ring) {
                    neighbors.push_back(axis_neighbors[axis_neighbors.size() - 1]);
                    directions[2] = true;
                }
            } else if (index == axis_neighbors.size() - 1) {
                neighbors.push_back(axis_neighbors[index - 1]);
                directions[2] = true;
                if (topology == tt::tt_fabric::Topology::Ring) {
                    neighbors.push_back(axis_neighbors[0]);
                    directions[3] = true;
                }
            } else {
                neighbors.push_back(axis_neighbors[index + 1]);
                directions[2] = true;
                neighbors.push_back(axis_neighbors[index - 1]);
                directions[3] = true;
            }
        }
    }
    TT_FATAL(neighbors.size() > 0, "No neighbors found");
    TT_FATAL(!(axis.has_value() && neighbors.size() > 2), "Along a single axis, there can only be 2 neighbors");
    TT_FATAL(
        !(topology == tt::tt_fabric::Topology::Ring && neighbors.size() != 4), "Ring topology must have 4 neighbors");
    return {neighbors, directions};
}

uint32_t select_link(
    const MeshDeviceView& mesh_view,
    const MeshCoordinate& src,
    const MeshCoordinate& dst,
    uint32_t num_links,
    tt::tt_fabric::Topology topology) {
    auto same_row = src[0] == dst[0];
    auto same_col = src[1] == dst[1];
    auto rows = mesh_view.num_rows();
    auto cols = mesh_view.num_cols();
    TT_FATAL(same_row ^ same_col, "src & dst must be neighbours");

    if (same_row) {  // ----- horizontal -----
        bool east = false;
        if (topology == tt::tt_fabric::Topology::Ring) {
            east = dst[1] == (src[1] + 1) % cols;  // wrap-around permitted
        } else {                                   /* Linear */
            east = dst[1] == src[1] + 1;           // no wrap-around
        }
        return (src[1] + (east ? 0 : 1)) % num_links;  // link id
    } else {                                           // ----- vertical -----
        bool south = false;
        if (topology == tt::tt_fabric::Topology::Ring) {
            south = dst[0] == (src[0] + 1) % rows;  // wrap-around permitted
        } else {                                    /* Linear */
            south = dst[0] == src[0] + 1;           // no wrap-around
        }
        return (src[0] + (south ? 0 : 1)) % num_links;  // link id
    }
}

std::vector<tt::tt_metal::IDevice*> get_token_parallel_devices(
    const MeshDeviceView& mesh_view, const MeshCoordinate& mesh_coordinate, const std::optional<uint32_t> axis) {
    std::vector<tt::tt_metal::IDevice*> devices;
    if (axis.has_value()) {
        if (axis.value() == 0) {
            devices = mesh_view.get_devices_on_row(mesh_coordinate[0]);
        } else {
            devices = mesh_view.get_devices_on_column(mesh_coordinate[1]);
        }
    } else {
        devices = mesh_view.get_devices();
    }
    return devices;
}

}  // namespace detail

AllToAllDispatchDeviceOperation::AllToAllDispatchSparse::cached_mesh_workload_t
AllToAllDispatchDeviceOperation::AllToAllDispatchSparse::create_mesh_workload(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;
    auto mesh_device = tensor_args.input_tensor.mesh_device();

    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = create_at(operation_attributes, coord, tensor_args, tensor_return_value, tensor_coords);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(coord, std::move(cached_program.shared_variables));
    }
    return cached_mesh_workload_t(std::move(workload), std::move(shared_variables));
}

ttnn::device_operation::CachedProgram<AllToAllDispatchDeviceOperation::AllToAllDispatchSparse::shared_variables_t>
AllToAllDispatchDeviceOperation::AllToAllDispatchSparse::create_at(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    using namespace tt;
    using namespace tt::tt_metal;
    using namespace tt::tt_fabric;
    using namespace ttnn::ccl;

    tt::tt_metal::Program program{};

    auto input_tensor = tensor_args.input_tensor;
    auto indices_tensor = tensor_args.expert_indices_tensor;
    auto mapping_tensor = tensor_args.expert_mapping_tensor;
    auto output_tensor = tensor_return_value.at(0);
    auto metadata_tensor = tensor_return_value.at(1);
    auto num_links = operation_attributes.num_links;
    auto topology = operation_attributes.topology;

    auto mesh_device = input_tensor.mesh_device();
    const auto& mesh_view = mesh_device->get_view();
    auto src_device = mesh_device->get_device(mesh_coordinate);
    auto src_physical_device_id = src_device->id();

    auto control_plane = tt::tt_fabric::get_control_plane();
    auto fabric_node_id = control_plane->get_fabric_node_id_from_physical_chip_id(src_device->id());
    uint32_t src_mesh_id = *fabric_node_id.mesh_id;
    uint32_t src_chip_id = (uint32_t)fabric_node_id.chip_id;

    tt::log_info(
        tt::LogAlways,
        "Creating all to all dispatch program for mesh coordinate: ({}, {}) with physical device id: {} mesh id: {} "
        "chip id: {}",
        mesh_coordinate[0],
        mesh_coordinate[1],
        src_device->id(),
        src_mesh_id,
        src_chip_id);

    const auto [neighbors, directions] =
        detail::get_neighbors(mesh_view, mesh_coordinate, topology, operation_attributes.axis);

    auto input_shape = input_tensor.get_tensor_spec().logical_shape();
    auto indices_shape = indices_tensor.get_tensor_spec().logical_shape();
    auto mapping_shape = mapping_tensor.get_tensor_spec().logical_shape();

    uint32_t num_devices = mesh_view.num_devices();
    uint32_t dispatch_devices =
        operation_attributes.axis.has_value()
            ? operation_attributes.axis.value() == 0 ? mesh_view.num_rows() : mesh_view.num_cols()
            : mesh_view.num_devices();

    uint32_t hidden_size = input_shape[-1];
    uint32_t batch_size = input_shape[0] * dispatch_devices;
    uint32_t seq_len = indices_shape[-2];

    uint32_t tokens_per_device = detail::get_num_rows(input_tensor);
    uint32_t selected_experts_k = indices_shape[-1];
    uint32_t experts = mapping_shape[-2];

    auto input_page_size = detail::get_page_size(input_tensor);
    auto indices_page_size = detail::get_page_size(indices_tensor);
    auto mapping_page_size = detail::get_page_size(mapping_tensor);
    auto output_page_size = detail::get_page_size(output_tensor);
    auto metadata_page_size = detail::get_page_size(metadata_tensor);

    auto input_pages = detail::get_num_pages(input_tensor);
    auto indices_pages = detail::get_num_pages(indices_tensor);
    auto mapping_pages = detail::get_num_pages(mapping_tensor);
    auto output_pages = detail::get_num_pages(output_tensor);
    auto metadata_pages = detail::get_num_pages(metadata_tensor);

    auto input_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());
    auto indices_data_format = tt::tt_metal::datatype_to_dataformat_converter(indices_tensor.get_dtype());
    auto mapping_data_format = tt::tt_metal::datatype_to_dataformat_converter(mapping_tensor.get_dtype());

    constexpr uint32_t buffering_factor = 2;

    // input sharded buffer
    uint32_t input_tensor_cb_id = tt::CBIndex::c_0;
    // full indices buffer
    uint32_t indices_tensor_cb_id = tt::CBIndex::c_1;
    // full mapping buffer
    uint32_t mapping_tensor_cb_id = tt::CBIndex::c_2;
    // client interface
    uint32_t packet_header_cb_id = tt::CBIndex::c_3;
    // metadata buffer
    uint32_t send_preparation_buffer_id = tt::CBIndex::c_4;

    uint32_t aligned_input_page_size = detail::get_aligned_page_size(input_tensor);
    tt::log_info(
        tt::LogAlways,
        "input shape: {}, input_pages: {}, input_page_size: {}, aligned_input_page_size: {}",
        input_tensor.logical_shape(),
        input_pages,
        input_page_size,
        aligned_input_page_size);

    uint32_t aligned_indices_page_size = detail::get_aligned_page_size(indices_tensor);
    tt::log_info(
        tt::LogAlways,
        "indices shape: {}, indices_pages: {}, indices_page_size: {}, aligned_indices_page_size: {}",
        indices_tensor.logical_shape(),
        indices_pages,
        indices_page_size,
        aligned_indices_page_size);

    uint32_t aligned_mapping_page_size = detail::get_aligned_page_size(mapping_tensor);
    tt::log_info(
        tt::LogAlways,
        "mapping shape: {}, mapping_pages: {}, mapping_page_size: {}, aligned_mapping_page_size: {}",
        mapping_tensor.logical_shape(),
        mapping_pages,
        mapping_page_size,
        aligned_mapping_page_size);

    uint32_t aligned_output_page_size = detail::get_aligned_page_size(output_tensor);
    tt::log_info(
        tt::LogAlways,
        "output shape: {}, output_pages: {}, output_page_size: {}, aligned_output_page_size: {}",
        output_tensor.logical_shape(),
        output_pages,
        output_page_size,
        aligned_output_page_size);

    uint32_t aligned_metadata_page_size = detail::get_aligned_page_size(metadata_tensor);
    tt::log_info(
        tt::LogAlways,
        "metadata shape: {}, metadata_pages: {}, metadata_page_size: {}, aligned_metadata_page_size: {}",
        metadata_tensor.logical_shape(),
        metadata_pages,
        metadata_page_size,
        aligned_metadata_page_size);

    tt::tt_metal::CircularBufferConfig cb_input_tensor_config =
        tt::tt_metal::CircularBufferConfig(
            input_pages * aligned_input_page_size, {{input_tensor_cb_id, input_data_format}})
            .set_page_size(input_tensor_cb_id, aligned_input_page_size);

    tt::tt_metal::CircularBufferConfig cb_indices_tensor_config =
        tt::tt_metal::CircularBufferConfig(
            indices_pages * aligned_indices_page_size, {{indices_tensor_cb_id, indices_data_format}})
            .set_page_size(indices_tensor_cb_id, aligned_indices_page_size);

    tt::tt_metal::CircularBufferConfig cb_mapping_tensor_config =
        tt::tt_metal::CircularBufferConfig(
            mapping_pages * aligned_mapping_page_size, {{mapping_tensor_cb_id, mapping_data_format}})
            .set_page_size(mapping_tensor_cb_id, aligned_mapping_page_size);

    static constexpr auto num_packet_headers_storable = 8;
    static constexpr auto packet_header_size_bytes = sizeof(tt::tt_fabric::PacketHeader);
    tt::tt_metal::CircularBufferConfig packet_header_cb_config =
        tt::tt_metal::CircularBufferConfig(
            num_packet_headers_storable * packet_header_size_bytes * buffering_factor,
            {{packet_header_cb_id, tt::DataFormat::RawUInt32}})
            .set_page_size(packet_header_cb_id, packet_header_size_bytes);

    uint32_t send_preparation_buffer_size = num_devices * tokens_per_device * aligned_metadata_page_size;

    tt::tt_metal::CircularBufferConfig send_preparation_buffer_config =
        tt::tt_metal::CircularBufferConfig(
            send_preparation_buffer_size, {{send_preparation_buffer_id, tt::DataFormat::UInt16}})
            .set_page_size(send_preparation_buffer_id, send_preparation_buffer_size);

    auto subdevice_core_range_set =
        mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, operation_attributes.subdevice_id);

    auto subdevice_cores = corerange_to_cores(subdevice_core_range_set);
    TT_FATAL(
        subdevice_cores.size() >= num_links,
        "Not enough cores {} to send all links {}",
        subdevice_cores.size(),
        num_links);

    std::vector<CoreCoord> sender_cores;
    // select
    for (uint32_t i = 0; i < num_links; i++) {
        sender_cores.push_back(subdevice_cores.at(i));
    }

    // select the first core as the sender core for now, in the future we will distribute the work evenly across links
    auto sender_core = sender_cores.at(0);

    // create circular buffers
    auto input_tensor_cb = tt::tt_metal::CreateCircularBuffer(program, sender_core, cb_input_tensor_config);
    auto indices_tensor_cb = tt::tt_metal::CreateCircularBuffer(program, sender_core, cb_indices_tensor_config);
    auto mapping_tensor_cb = tt::tt_metal::CreateCircularBuffer(program, sender_core, cb_mapping_tensor_config);
    auto packet_header_cb = tt::tt_metal::CreateCircularBuffer(program, sender_core, packet_header_cb_config);
    auto send_preparation_buffer =
        tt::tt_metal::CreateCircularBuffer(program, sender_core, send_preparation_buffer_config);

    std::vector<uint32_t> dest_mesh_id, dest_chip_id;
    for (const auto& coord : tensor_coords.coords()) {
        auto device = mesh_device->get_device(coord);
        auto fabric_node_id = control_plane->get_fabric_node_id_from_physical_chip_id(device->id());
        dest_mesh_id.push_back(*fabric_node_id.mesh_id);
        dest_chip_id.push_back((uint32_t)fabric_node_id.chip_id);
    }
    tt::log_info(tt::LogAlways, "dest_chip_id: {}", detail::stringify_vector(dest_chip_id));
    tt::log_info(tt::LogAlways, "dest_mesh_id: {}", detail::stringify_vector(dest_mesh_id));
    // tt::log_info(tt::LogAlways, "directions: {}", detail::stringify_array(directions));

    // TODO: add fabric node and mesh id to the compile time args
    // TODO: add an array mapping logical device id to physical device id

    const auto& fabric_context = control_plane->get_fabric_context();
    auto fabric_max_packet_size = fabric_context.get_fabric_max_payload_size_bytes();

    std::vector<uint32_t> reader_compile_time_args = {
        input_tensor.buffer()->is_dram(),
        indices_tensor.buffer()->is_dram(),
        mapping_tensor.buffer()->is_dram(),
        output_tensor.buffer()->is_dram(),
        metadata_tensor.buffer()->is_dram(),

        input_tensor_cb_id,
        indices_tensor_cb_id,
        mapping_tensor_cb_id,
        packet_header_cb_id,
        send_preparation_buffer_id,

        input_pages,
        indices_pages,
        mapping_pages,
        output_pages,
        metadata_pages,

        input_page_size,
        indices_page_size,
        mapping_page_size,
        output_page_size,
        metadata_page_size,

        num_devices,
        hidden_size,
        batch_size,
        selected_experts_k,
        experts,
        tokens_per_device,

        num_links,
        topology == tt::tt_fabric::Topology::Ring ? 1u : 0u,

        src_mesh_id,
        (uint32_t)src_chip_id,
        mesh_view.num_rows(),
        mesh_view.num_cols(),

        aligned_input_page_size,
        aligned_indices_page_size,
        aligned_mapping_page_size,
        aligned_output_page_size,
        aligned_metadata_page_size,

        fabric_max_packet_size,
    };

    auto writer_compile_time_args = reader_compile_time_args;

    auto input_buffer = input_tensor.buffer();
    auto indices_buffer = indices_tensor.buffer();
    auto mapping_buffer = mapping_tensor.buffer();
    auto output_buffer = output_tensor.buffer();
    auto metadata_buffer = metadata_tensor.buffer();

    std::map<std::string, std::string> reader_defines = {
        {"AXIS", std::to_string(operation_attributes.axis.has_value() ? operation_attributes.axis.value() : -1)},
    };

    tt::tt_metal::KernelHandle ternary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/ccl/all_to_all_dispatch/device/kernels/dataflow/reader_all_to_all_dispatch.cpp",
        sender_core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::NOC_1,
            .compile_args = reader_compile_time_args,
            .defines = reader_defines});

    std::map<std::string, std::string> writer_defines = {
        {"DEST_CHIP_ID", detail::stringify_vector(dest_chip_id)},
        {"DEST_MESH_ID", detail::stringify_vector(dest_mesh_id)},
        {"DIRECTIONS", detail::stringify_array(directions)}};

    if (operation_attributes.axis.has_value()) {
        writer_defines["AXIS"] = std::to_string(operation_attributes.axis.value());
    }

    tt::tt_metal::KernelHandle binary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/ccl/all_to_all_dispatch/device/kernels/dataflow/writer_all_to_all_dispatch.cpp",
        sender_core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::NOC_0,
            .compile_args = writer_compile_time_args,
            .defines = writer_defines});

    tt::log_info(tt::LogAlways, "metadata tensor address: {}", metadata_tensor.buffer()->address());
    std::vector<uint32_t> reader_runtime_args = {
        input_tensor.buffer()->address(),
        indices_tensor.buffer()->address(),
        mapping_tensor.buffer()->address(),
        output_tensor.buffer()->address(),
        metadata_tensor.buffer()->address(),
        (uint32_t)operation_attributes.cross_device_semaphore->address(),
    };

    std::vector<uint32_t> writer_runtime_args = {
        input_tensor.buffer()->address(),
        indices_tensor.buffer()->address(),
        mapping_tensor.buffer()->address(),
        output_tensor.buffer()->address(),
        metadata_tensor.buffer()->address(),
        (uint32_t)operation_attributes.cross_device_semaphore->address(),
    };

    for (auto& neighbor : neighbors) {
        auto neighbor_coordinate = mesh_view.find_device(neighbor->id());
        uint32_t link_id = detail::select_link(mesh_view, mesh_coordinate, neighbor_coordinate, num_links, topology);
        tt::log_info(
            tt::LogAlways,
            "Connection between ({}, {}) and ({}, {}) will choose link_id: {}",
            mesh_coordinate[0],
            mesh_coordinate[1],
            neighbor_coordinate[0],
            neighbor_coordinate[1],
            link_id);
        tt::tt_fabric::append_fabric_connection_rt_args(
            src_physical_device_id, neighbor->id(), link_id, program, sender_core, writer_runtime_args);
    }

    tt::tt_metal::SetRuntimeArgs(program, ternary_reader_kernel_id, sender_cores.at(0), reader_runtime_args);
    tt::tt_metal::SetRuntimeArgs(program, binary_writer_kernel_id, sender_cores.at(0), writer_runtime_args);
    return {
        std::move(program),
        {.ternary_reader_kernel_id = ternary_reader_kernel_id,
         .binary_writer_kernel_id = binary_writer_kernel_id,
         .core = sender_cores.at(0)}};
}

void AllToAllDispatchDeviceOperation::AllToAllDispatchSparse::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        const auto& shared_variables = cached_workload.shared_variables.at(range);
        auto& ternary_reader_kernel_id = shared_variables.ternary_reader_kernel_id;
        auto& binary_writer_kernel_id = shared_variables.binary_writer_kernel_id;
        auto& core = shared_variables.core;

        auto output_tensor = tensor_return_value.at(0);
        auto metadata_tensor = tensor_return_value.at(1);

        auto& reader_runtime_args = tt::tt_metal::GetRuntimeArgs(program, ternary_reader_kernel_id, core);
        auto& writer_runtime_args = tt::tt_metal::GetRuntimeArgs(program, binary_writer_kernel_id, core);
        reader_runtime_args.at(0) = tensor_args.input_tensor.buffer()->address();
        reader_runtime_args.at(1) = tensor_args.expert_indices_tensor.buffer()->address();
        reader_runtime_args.at(2) = tensor_args.expert_mapping_tensor.buffer()->address();
        reader_runtime_args.at(3) = output_tensor.buffer()->address();
        reader_runtime_args.at(4) = metadata_tensor.buffer()->address();
        reader_runtime_args.at(5) = (uint32_t)operation_attributes.cross_device_semaphore->address();

        writer_runtime_args.at(0) = tensor_args.input_tensor.buffer()->address();
        writer_runtime_args.at(1) = tensor_args.expert_indices_tensor.buffer()->address();
        writer_runtime_args.at(2) = tensor_args.expert_mapping_tensor.buffer()->address();
        writer_runtime_args.at(3) = output_tensor.buffer()->address();
        writer_runtime_args.at(4) = metadata_tensor.buffer()->address();
        writer_runtime_args.at(5) = (uint32_t)operation_attributes.cross_device_semaphore->address();
    }
}

}  // namespace ttnn::operations::ccl

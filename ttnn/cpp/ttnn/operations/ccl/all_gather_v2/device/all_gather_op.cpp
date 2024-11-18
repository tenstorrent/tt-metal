// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/ccl/all_gather_v2/device/all_gather_op.hpp"
#include "ttnn/operations/math.hpp"

#include "tt_metal/host_api.hpp"

#include "ttnn/tensor/tensor_utils.hpp"

#include "eth_l1_address_map.h"

namespace ttnn {
namespace ccl{
namespace all_gather_detail{

AllGatherV2 create_all_gather_struct(
    const std::vector<Program*>& programs,
    const ttnn::ccl::EdmLineFabricOpInterface& line_fabric,
    const Tensor& input_tensor,
    const uint32_t dim,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    const std::vector<Device*>& devices,
    const ttnn::ccl::Topology topology
) {
    uint32_t num_devices = devices.size();

    uint32_t device_index = 0; // Initialize device index
    Program* program = nullptr;
    for (uint32_t i = 0; i < num_devices; ++i) {
        if (devices.at(i) == input_tensor.device()) {
            device_index = i;
            program = programs.at(i);
        }
    }

    return ttnn::AllGatherV2{
        program, /*line_fabric, devices,*/ dim, num_links, num_devices, device_index, memory_config.value_or(input_tensor.memory_config()), topology};
}
} // namespace all_gather_v2_detail
} // namespace ccl

void AllGatherV2::validate(const std::vector<Tensor> &input_tensors) const {
    TT_FATAL(input_tensors.size() == 1, "Error, Input tensor size should be 1 but has {}", input_tensors.size());
    const auto& input_tensor = input_tensors[0];
    const auto& layout = input_tensors[0].get_layout();
    const auto& dtype = input_tensors[0].get_dtype();
    const auto& page_size = input_tensors[0].buffer()->page_size();
    TT_FATAL(page_size % input_tensors[0].buffer()->alignment() == 0, "All Gather currently requires aligned pages");

    // TODO: This can be removed by passing two page sizes, actual and aligned to be used for address offsets
    // Buffer sizes also need to take this aligned page size into consideration
    // TODO: Validate ring
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to all_gather need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr , "Operands to all_gather need to be allocated in buffers on device!");
    TT_FATAL(this->num_links > 0, "Error, num_links should be more than 0 but has {}", this->num_links);
    TT_FATAL(this->num_links <= input_tensor.device()->compute_with_storage_grid_size().y, "Worker cores used by links are parallelizaed over rows");

    TT_FATAL(input_tensor.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED ||
        input_tensor.memory_config().memory_layout == TensorMemoryLayout::WIDTH_SHARDED ||
        input_tensor.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED ||
        input_tensor.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED,
        "Unsupported memory layout {}.", input_tensor.memory_config().memory_layout);

    // Sharding Config checks
    bool input_sharded = input_tensor.is_sharded();
    if (input_sharded) {
        // TODO(snijjar)
    }
}

std::vector<ttnn::SimpleShape> AllGatherV2::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    auto shape = input_tensors[0].get_padded_shape(); // TODO: Replace with get_logical_shape()
    shape[this->dim] *= this->ring_size;
    return std::vector<ttnn::SimpleShape>(input_tensors.size(), shape);
}

std::vector<Tensor> AllGatherV2::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors[0];
    if(this->output_mem_config.is_sharded()) {
        return {create_device_tensor(
            this->compute_output_shapes(input_tensors).at(0),
            input_tensor.get_dtype(),
            input_tensor.get_layout(),
            input_tensor.device(),
            this->output_mem_config,
            input_tensor.get_tile()
            )};
    } else {
        return operation::generic_create_output_tensors(*this, input_tensors, input_tensor.get_dtype(), input_tensor.get_layout(), this->output_mem_config, input_tensor.get_tile());
    }
}

operation::ProgramWithCallbacks AllGatherV2::create_program(const std::vector<Tensor> & input_tensors, std::vector<Tensor> &output_tensors) const {
    tt::log_info(tt::LogOp, "DEBUG: create_program is called");
    return all_gather_multi_core_with_workers_new(*this->program, input_tensors[0], output_tensors[0], this->dim, this->num_links, this->ring_size, this->ring_index, this->topology);
}

const operation::Hash AllGatherV2::compute_program_hash(
    const std::vector<Tensor> &input_tensors) const {
    return operation::hash_operation<AllGatherV2>(
        this->dim,
        this->num_links,
        this->ring_size,
        this->ring_index,
        this->output_mem_config,
        this->topology);
}

namespace operations {
namespace ccl {

Tensor all_gather_v2(
    const Tensor& input_tensor,
    const uint32_t dim,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<size_t> user_defined_num_workers,
    const std::optional<size_t> user_defined_num_buffers_per_channel,
    const ttnn::ccl::Topology topology) {

    tt::log_info(tt::LogOp, "DEBUG: all_gather is called");

    TT_FATAL(std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr, "all_gather op is only supported for Fast Dispatch");
    auto devices = input_tensor.get_workers();
    uint32_t num_devices = devices.size();
    TT_FATAL(num_devices > 1, "all_gather op will only work for num_devices > 1, but has {}", num_devices);
    ttnn::ccl::Topology ccl_topology = topology;

    if (num_devices == 2){
        ccl_topology = ttnn::ccl::Topology::Linear;
    }
    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor}))};

    // Make programs vector persist by moving it to heap or making it static
    static auto programs = std::vector<Program>(devices.size());
    auto program_ptrs = std::vector<Program*>(devices.size());
    std::transform(programs.begin(), programs.end(), program_ptrs.begin(),
        [](auto& program) {
            program = tt::tt_metal::Program{}; // Initialize each Program
            return &program;
        });
    TT_FATAL(num_links == 1, "all_gather op is only supported for num_links == 1, but has {}", num_links);
    tt::log_info(tt::LogOp, "DEBUG: creating line_fabric with num devices: {}, num links: {}", devices.size(), num_links);
    auto line_fabric = ttnn::ccl::EdmLineFabricOpInterface(devices, program_ptrs, num_links);
    tt::log_info(tt::LogOp, "DEBUG: line_fabric is created");

    operation::launch_op(
        [dim, num_links, memory_config, devices, ccl_topology, line_fabric, program_ptrs](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {

            const auto& input_tensor = input_tensors.at(0);

            return operation::run(
                ttnn::ccl::all_gather_detail::create_all_gather_struct(program_ptrs, line_fabric, input_tensor, dim, num_links, memory_config, devices, ccl_topology),
                {input_tensor});
        },
        {input_tensor},
        output_tensors);
    return output_tensors.at(0);
}

Tensor all_gather_v2(
    const Tensor& input_tensor,
    const uint32_t dim,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<size_t> user_defined_num_workers,
    const std::optional<size_t> user_defined_num_buffers_per_channel,
    const ttnn::ccl::Topology topology) {

    tt::log_info(tt::LogOp, "DEBUG: all_gather with cluster_axis is called");

    TT_FATAL(topology == ttnn::ccl::Topology::Linear, "This all_gather API with cluster_axis is currently supported only for the Linear topology");
    const auto mesh_view = mesh_device.get_view();
    std::size_t num_devices = (cluster_axis == 0) ? mesh_view->num_rows() : mesh_view->num_cols();

    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor}))};

    // Untested!!!
    // create devices vector
    std::vector<Device*> devices;
    const auto coordinate = mesh_view->find_device(input_tensor.device()->id());
    const auto view_index = (cluster_axis == 0) ? coordinate.col : coordinate.row;
    for (std::size_t i = 0; i < num_devices; i++) {
        std::size_t row_idx = (cluster_axis == 0) ? i : view_index;
        std::size_t col_idx = (cluster_axis == 0) ? view_index : i;
        devices.push_back(mesh_device.get_device(row_idx, col_idx));
    }

    // Make programs vector persist by moving it to heap or making it static
    static auto programs = std::vector<Program>(devices.size());
    auto program_ptrs = std::vector<Program*>(devices.size());
    std::transform(programs.begin(), programs.end(), program_ptrs.begin(),
        [](auto& program) {
            program = tt::tt_metal::Program{}; // Initialize each Program
            return &program;
        });
    TT_FATAL(num_links == 1, "all_gather op is only supported for num_links == 1, but has {}", num_links);
    tt::log_info(tt::LogOp, "DEBUG: creating line_fabric with num devices: {}, num links: {}", num_devices, num_links);
    auto line_fabric = ttnn::ccl::EdmLineFabricOpInterface(devices, program_ptrs, num_links);
    tt::log_info(tt::LogOp, "DEBUG: line_fabric is created");

    operation::launch_op(
        [dim, num_links, memory_config, mesh_view, cluster_axis, num_devices, topology, devices, program_ptrs, line_fabric](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {

            const auto& input_device_tensor = input_tensors.at(0);

            const auto coordinate = mesh_view->find_device(input_device_tensor.device()->id());
            const auto view_index = (cluster_axis == 0) ? coordinate.col : coordinate.row;
            const auto device_index = (cluster_axis == 0) ? coordinate.row : coordinate.col;

            auto get_chip_id = [&](std::size_t line_index) -> std::optional<chip_id_t> {
                auto new_coord = coordinate;
                if (cluster_axis == 0) {
                    new_coord.row = line_index % num_devices;
                } else {
                    new_coord.col = line_index % num_devices;
                }
                return mesh_view->find_device_id(new_coord);
            };

            return operation::run(
                ttnn::AllGatherV2{
                    program_ptrs[device_index], /*line_fabric, devices,*/ dim, num_links, num_devices, device_index, memory_config.value_or(input_device_tensor.memory_config()), topology},
                {input_device_tensor});
        },
        {input_tensor},
        output_tensors);
    return output_tensors.at(0);

}


} // namespace ccl
} // namespace operations

}  // namespace ttnn

// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/ccl/line_all_gather/device/line_all_gather_op.hpp"
#include "ttnn/operations/ccl/all_gather/device/all_gather_op.hpp"
#include "tt_metal/impl/device/device_mesh_view.hpp"
#include "tt_dnn/op_library/math.hpp"

#include "tt_metal/host_api.hpp"

#include "ttnn/tensor/tensor_utils.hpp"

#include "eth_l1_address_map.h"

namespace ttnn {

void LineAllGather::validate(const std::vector<Tensor> &input_tensors) const {
    TT_FATAL(input_tensors.size() == 1);
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
    TT_FATAL(this->num_links > 0);
    TT_FATAL(this->num_links <= input_tensor.device()->compute_with_storage_grid_size().y, "Worker cores used by links are parallelizaed over rows");
    TT_FATAL(this->receiver_device_id.has_value() || this->sender_device_id.has_value());
    if (this->receiver_device_id == this->sender_device_id) {
        TT_FATAL(input_tensor.device()->get_ethernet_sockets(this->receiver_device_id.value()).size() >= 2 * this->num_links, "2 Device all gather requires at least 2 eth connections per link");
    } else {
        TT_FATAL(this->topology == all_gather_op::Topology::Linear || (this->receiver_device_id.has_value() && input_tensor.device()->get_ethernet_sockets(this->receiver_device_id.value()).size() >= this->num_links), "All gather requires at least 1 eth connection per link between sender device {} and receiver device {}", this->sender_device_id, this->receiver_device_id);
        TT_FATAL(this->topology == all_gather_op::Topology::Linear || (this->sender_device_id.has_value() &&input_tensor.device()->get_ethernet_sockets(this->sender_device_id.value()).size() >= this->num_links), "All gather requires at least 1 eth connection per link between sender device {} and receiver device {}", this->sender_device_id, this->receiver_device_id);
    }

    TT_FATAL(input_tensor.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED ||
        input_tensor.memory_config().memory_layout == TensorMemoryLayout::WIDTH_SHARDED ||
        input_tensor.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED);

    // Sharding Config checks
    bool input_sharded = input_tensor.is_sharded();
    if (input_sharded) {
        // TODO(snijjar)
    }
}

std::vector<tt::tt_metal::Shape> LineAllGather::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    auto shape = input_tensors[0].get_legacy_shape();
    shape[this->dim] *= this->ring_size;
    return std::vector<tt::tt_metal::Shape>(input_tensors.size(), shape);
}

std::vector<Tensor> LineAllGather::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors[0];
    if(this->output_mem_config.is_sharded()) {
        return {create_device_tensor(
            this->compute_output_shapes(input_tensors).at(0),
            input_tensor.get_dtype(),
            input_tensor.get_layout(),
            input_tensor.device(),
            this->output_mem_config
            )};
    } else {
        return operation::generic_create_output_tensors(*this, input_tensors, input_tensor.get_dtype(), input_tensor.get_layout(), this->output_mem_config);
    }
}

operation::ProgramWithCallbacks LineAllGather::create_program(const std::vector<Tensor> & input_tensors, std::vector<Tensor> &output_tensors) const {
    return all_gather_multi_core_with_workers(input_tensors[0], output_tensors[0], this->dim, this->num_links, this->ring_size, this->ring_index, this->receiver_device_id, this->sender_device_id, this->topology);
}

namespace operations {
namespace ccl {

Tensor line_all_gather(
    const Tensor& input_tensor, const uint32_t dim, const uint32_t num_links, const std::optional<MemoryConfig>& memory_config) {

    TT_FATAL(std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr, "This op is only supported for Fast Dispatch");

    auto devices = input_tensor.get_workers();
    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor}))};
    operation::launch_op(
        [dim, num_links, memory_config, devices](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {

            const auto& input_tensor = input_tensors.at(0);
            uint32_t num_devices = devices.size();

            uint32_t device_index = 0; // Initialize device index
            std::optional<uint32_t> receiver_device_id = std::nullopt; // Initialize receiver device ID
            std::optional<uint32_t> sender_device_id = std::nullopt; // Initialize sender device ID

            for (uint32_t i = 0; i < num_devices; ++i) {
                if (devices[i] == input_tensor.device()) {
                    device_index = i;
                    bool is_last_chip_in_clockwise_direction = i == (num_devices - 1);
                    bool is_last_chip_in_counter_clockwise_direction = i == 0;
                    receiver_device_id = is_last_chip_in_clockwise_direction ?
                        std::nullopt :
                        std::optional<chip_id_t>(devices.at(i+1)->id());
                    sender_device_id = is_last_chip_in_counter_clockwise_direction ?
                        std::nullopt :
                        std::optional<chip_id_t>(devices.at(i-1)->id());
                    break;
                }
            }

            return operation::run(
                ttnn::LineAllGather{
                    dim, num_links, num_devices, device_index, receiver_device_id, sender_device_id, memory_config.value_or(input_tensor.memory_config()), ttnn::all_gather_op::Topology::Linear},
                {input_tensor});
        },
        {input_tensor},
        output_tensors);
    return output_tensors.at(0);
}

Tensor line_all_gather(
    const Tensor& input_tensor,
    const uint32_t dim,
    const uint32_t cluster_axis,
    const DeviceMesh& device_mesh,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config) {

    const auto view = DeviceMeshView(device_mesh);
    const auto device_views = (cluster_axis == 0) ? view.get_column_views() : view.get_row_views();

    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor}))};

    operation::launch_op(
        [&dim, &num_links, &memory_config, &device_views](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {

            const auto& input_tensor = input_tensors[0];
            const DeviceMeshView::DeviceView* selected_view = nullptr;

            uint32_t device_index = 0;
            for (const auto& view : device_views) {
                auto it = std::find(view.begin(), view.end(), input_tensor.device());
                if (it != view.end()) {
                    selected_view = &view;
                    device_index = std::distance(view.begin(), it);
                    break;
                }
            }

            TT_ASSERT(selected_view != nullptr, "Device not found in any view");

            uint32_t num_devices = selected_view->size();

            bool is_last_chip_in_clockwise_direction = device_index == (num_devices - 1);
            bool is_last_chip_in_counter_clockwise_direction = device_index == 0;
            std::optional<chip_id_t> receiver_device_id = is_last_chip_in_clockwise_direction ?
                std::nullopt :
                std::optional<chip_id_t>((*selected_view)[(device_index + 1) % num_devices]->id());
            std::optional<chip_id_t> sender_device_id = is_last_chip_in_counter_clockwise_direction ?
                std::nullopt :
                std::optional<chip_id_t>((*selected_view)[(device_index + num_devices - 1) % num_devices]->id());

            return operation::run(
                ttnn::LineAllGather{
                    dim, num_links, num_devices, device_index, receiver_device_id, sender_device_id, memory_config.value_or(input_tensor.memory_config()), ttnn::all_gather_op::Topology::Linear},
                {input_tensor});
        },
        {input_tensor},
        output_tensors);
    return output_tensors.at(0);

}

} // namespace ccl
} // namespace operations

}  // namespace ttnn

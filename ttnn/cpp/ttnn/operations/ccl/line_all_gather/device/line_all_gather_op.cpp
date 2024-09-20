// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/ccl/line_all_gather/device/line_all_gather_op.hpp"
#include "ttnn/operations/ccl/all_gather/device/all_gather_op.hpp"
#include "tt_metal/impl/device/mesh_device_view.hpp"
#include "tt_dnn/op_library/math.hpp"

#include "tt_metal/host_api.hpp"

#include "ttnn/tensor/tensor_utils.hpp"

#include "eth_l1_address_map.h"

namespace ttnn {
namespace operations {
namespace ccl {

Tensor line_all_gather(
    const Tensor& input_tensor, const uint32_t dim, const uint32_t num_links, const std::optional<MemoryConfig>& memory_config, const std::optional<size_t> user_defined_num_workers, const std::optional<size_t> user_defined_num_buffers_per_channel) {

    TT_FATAL(std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr, "This op is only supported for Fast Dispatch");

    auto devices = input_tensor.get_workers();
    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor}))};
    operation::launch_op(
        [dim, num_links, memory_config, user_defined_num_workers, user_defined_num_buffers_per_channel, devices](
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
                ttnn::AllGather{
                    dim, num_links, num_devices, device_index, user_defined_num_workers, user_defined_num_buffers_per_channel, receiver_device_id, sender_device_id, memory_config.value_or(input_tensor.memory_config()), ttnn::all_gather_op::Topology::Linear},
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
    const MeshDevice& mesh_device,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<size_t> user_defined_num_workers,
    const std::optional<size_t> user_defined_num_buffers_per_channel) {

    const auto mesh_view = mesh_device.get_view();
    std::size_t num_devices = (cluster_axis == 0) ? mesh_view->num_rows() : mesh_view->num_cols();

    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor}))};

    operation::launch_op(
        [dim, num_links, memory_config, mesh_view, cluster_axis, user_defined_num_workers, user_defined_num_buffers_per_channel, num_devices](
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

            bool is_last_chip_in_clockwise_direction = device_index == (num_devices - 1);
            bool is_last_chip_in_counter_clockwise_direction = device_index == 0;
            auto receiver_device_id = is_last_chip_in_clockwise_direction ? std::nullopt : get_chip_id(device_index + 1);
            auto sender_device_id = is_last_chip_in_counter_clockwise_direction ? std::nullopt : get_chip_id(device_index + num_devices - 1);

            return operation::run(
                ttnn::AllGather{
                    dim, num_links, num_devices, device_index, user_defined_num_workers, user_defined_num_buffers_per_channel, receiver_device_id, sender_device_id, memory_config.value_or(input_device_tensor.memory_config()), ttnn::all_gather_op::Topology::Linear},
                {input_device_tensor});
        },
        {input_tensor},
        output_tensors);
    return output_tensors.at(0);

}

} // namespace ccl
} // namespace operations

}  // namespace ttnn

// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/experimental/tensor/types.hpp"
#include "ttnn/run_operation.hpp"
#include "types.hpp"

namespace ttnn {
    using DeviceBuffer = std::shared_ptr<Buffer>;
    using queue_id = uint8_t;

    DeviceBuffer allocate_buffer_on_device(uint32_t buffer_size_bytes, types::Device* device, const Shape& shape, DataType data_type, Layout layout, const MemoryConfig& memory_config, const std::optional<ShardSpecBuffer>& shard_spec = std::nullopt);

    void write_buffer(queue_id cq_id, Tensor& dst, std::vector<std::shared_ptr<void>> src, const std::optional<std::size_t> transfer_size = std::nullopt);

    void read_buffer(queue_id cq_id, Tensor& src, std::vector<std::shared_ptr<void>> dst, const std::optional<std::size_t> transfer_size = std::nullopt, size_t src_offset = 0, bool blocking = true);

    void queue_synchronize(CommandQueue& cq);

    void event_synchronize(Device* device, std::shared_ptr<Event> event);

    bool event_query(std::shared_ptr<Event> event);

    void wait_for_event(CommandQueue& cq, std::shared_ptr<Event> event);

    void record_event(CommandQueue& cq, std::shared_ptr<Event> event);

    // Generic Device Op dispatch function. Templated on Op structs.
    template<typename OpConfig>
    std::vector<Tensor> run_operation(
        queue_id cq_id,
        OpConfig devop,
        const tt::tt_metal::operation::Tensors& input_tensors,
        const tt::tt_metal::operation::OptionalConstTensors& optional_input_tensors = {},
        const tt::tt_metal::operation::OptionalTensors& optional_output_tensors = {}) {
        static_assert(tt::tt_metal::operation::detail::is_device_operation<OpConfig>(), "ttnn::run_operation can only dispatch Device Operations!");
        // Create output tensor vector by examining the number of output shapes created by the device operation
        std::vector<Tensor> outputs(tt::tt_metal::operation::DeviceOperation<tt::tt_metal::operation::Tensors>(devop).compute_output_shapes(input_tensors).size());
        // Populate the workers of the output tensors, based on the input tensors. This is needed for the async engine.
        for (int i = 0; i < outputs.size(); i++) {
            outputs[i] = Tensor(tt::tt_metal::operation::get_workers_for_op_output(std::move(input_tensors), std::move(optional_input_tensors)));
        }
        // Send the operation to the async engine, which will populate the output tensors.
        for (auto worker : outputs.at(0).workers) {
            tt::tt_metal::operation::launch_op(
                [devop, worker, cq_id] (const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors, const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
                    return operation::run(std::move(devop), input_tensors, optional_input_tensors, optional_output_tensors, cq_id);
                }, input_tensors, outputs, optional_input_tensors, optional_output_tensors);
        }
        return outputs;
    }
}

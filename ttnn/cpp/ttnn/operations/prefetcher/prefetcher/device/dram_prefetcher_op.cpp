// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dram_prefetcher_op.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/math.hpp"

#include "tt_metal/common/constants.hpp"

#include <optional>

using uint32_t = std::uint32_t;
using namespace tt::constants;

namespace ttnn::operations::dram_prefetcher {

void DramPrefetcher::validate(const std::vector<Tensor>& input_tensors) const {
    TT_FATAL(input_tensors.size() >= 1, "Must have at least one input tensor");
    // Check that all tensors are on the same device
    for (const auto& tensor : input_tensors) {
        TT_FATAL(tensor.device() == input_tensors[0].device(), "All tensors must be on the same device");
    }

    /*
        TODO List of validations to add:
        - input tensor is sharded


    */

    TT_FATAL(global_cb.has_value(), "Global circular buffer must be provided");
    // Check that all tensors' k is divisible by number of cores in global CB receiver
    uint32_t num_receiver_cores = global_cb->receiver_cores().num_cores();
    for (const auto& tensor : input_tensors) {
        TT_FATAL(
            tensor.get_legacy_shape()[0] % num_receiver_cores == 0,
            "All tensors' k must be divisible by the number of receiver cores = {}.",
            num_receiver_cores);
    }
    // Check that global_cb sender_receiver_core_mapping has same number of receivers for each sender core
    auto sender_receiver_core_mapping = global_cb->sender_receiver_core_mapping();
    for (const auto& [sender_core, receiver_core_range] : sender_receiver_core_mapping) {
        TT_FATAL(
            receiver_core_range.size() == sender_receiver_core_mapping.begin()->second.size(),
            "Global circular buffer must have same number of receivers for each sender core");
    }
}
std::vector<ttnn::SimpleShape> DramPrefetcher::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    return {input_tensors.at(0).get_logical_shape()};
}
std::vector<Tensor> DramPrefetcher::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    // Configure L1 interleaved memory layout
    // TODO: Update this to have an output tensor for all input tensors

    auto input_tensor = input_tensors.at(0);
    auto input_buffer = input_tensor.buffer();

    std::array<uint32_t, 2> shard_shape = input_tensor.shard_spec()->shape;
    ShardSpec shard_spec = {
        global_cb->sender_cores(),
        shard_shape,
        ShardOrientation::ROW_MAJOR,
        false,
    };

    auto output_mem_config = MemoryConfig(TensorMemoryLayout::WIDTH_SHARDED, BufferType::L1, shard_spec);

    auto tensor_layout =
        TensorLayout(input_tensor.get_dtype(), input_tensor.get_tensor_spec().page_config(), output_mem_config);

    auto tensor_spec = TensorSpec(input_tensor.get_logical_shape(), tensor_layout);

    auto& global_cb_buffer = global_cb->cb_buffer();
    ShardedBufferConfig output_buffer_config = {
        input_tensor.device(),
        input_buffer->size(),
        input_buffer->page_size(),
        BufferType::L1,
        input_buffer->buffer_layout(),
        input_buffer->shard_spec(),
    };
    std::shared_ptr<Buffer> output_buffer = CreateBuffer(output_buffer_config, global_cb->buffer_address());

    DeviceStorage device_storage = DeviceStorage(output_buffer);

    auto output_tensor = Tensor(device_storage, tensor_spec);

    return {output_tensor};
}
operation::ProgramWithCallbacks DramPrefetcher::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    auto& output_tensor = output_tensors.at(0);
    return dram_prefetcher_multi_core(input_tensors, this->tensor_addrs, this->global_cb, output_tensor);
}

}  // namespace ttnn::operations::dram_prefetcher

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

/*
TODO fixes for multiple output tensors
- ✅ refactor compute_output_shapes to return a vector of shapes (each shape is same as each input tensor)
- refactor create_output_tensors to create a vector of output tensors (for-loop over all input shapes
- ✅ Fix pybind to output list of output tensors

How to handle writing to output cb? (since now there are multiple output tensors)
- If create_device_tensor results in contiguous tensor allocation, then create a CB that is sizes for ALL output
tensors, and then align it to the base of the first tensor
- If not -- multiple CBs, one for each output tensor?
*/
std::vector<ttnn::SimpleShape> DramPrefetcher::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    // Output shape is the same as the input shape, but the height is multiplied by the number of input tensors
    auto input_shape = input_tensors.at(0).get_legacy_shape();
    return {
        ttnn::SimpleShape{input_shape[0] * input_tensors.size(), input_shape[1]},
        ttnn::SimpleShape{input_shape[0] * input_tensors.size(), input_shape[1]}};
}
std::vector<Tensor> DramPrefetcher::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    auto output_shape = this->compute_output_shapes(input_tensors).at(0);
    auto input_tensor = input_tensors.at(0);
    auto tensor_layout = TensorLayout(
        input_tensor.get_dtype(), input_tensor.get_tensor_spec().page_config(), this->reader_output_mem_config);
    auto tensor_spec = TensorSpec(output_shape, tensor_layout);
    ShardedBufferConfig output_buffer_config = {
        input_tensor.device(),
        tensor_spec.compute_packed_buffer_size_bytes(),
        tensor_spec.compute_page_size_bytes(),
        this->reader_output_mem_config.buffer_type,
        this->reader_output_mem_config.memory_layout,
        *(tensor_spec.compute_shard_spec_buffer()),
    };
    std::shared_ptr<Buffer> output_buffer = CreateBuffer(output_buffer_config, global_cb->buffer_address());
    DeviceStorage device_storage = DeviceStorage{output_buffer};
    auto output_tensor = Tensor(device_storage, tensor_spec);
    return {output_tensor};
}
operation::ProgramWithCallbacks DramPrefetcher::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    return dram_prefetcher_multi_core(
        input_tensors, this->tensor_addrs, this->num_layers, this->global_cb, output_tensors);
}

}  // namespace ttnn::operations::dram_prefetcher

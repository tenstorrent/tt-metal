// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "swap_tensor_async_op.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/math.hpp"
#include "cpp/ttnn/global_semaphore.hpp"

#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn {
namespace ccl {
namespace swap_tensor_detail {

SwapTensorAsync create_swap_tensor_async_struct(
    const Tensor& input_tensor,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    const std::vector<IDevice*>& devices,
    const ttnn::ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphores,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id) {
    uint32_t num_devices = devices.size();

    std::optional<IDevice*> forward_device = std::nullopt;
    std::optional<IDevice*> backward_device = std::nullopt;
    std::optional<GlobalSemaphore> semaphore = std::nullopt;
    uint32_t device_index = 0;  // Initialize device index
    for (uint32_t i = 0; i < num_devices; ++i) {
        if (devices.at(i) == input_tensor.device()) {
            device_index = i;
            semaphore = semaphores.at(i);  // Get raw pointer
            if (i != 0) {
                backward_device = devices.at(i - 1);
            } else if (topology == ttnn::ccl::Topology::Ring) {
                backward_device = devices.at(num_devices - 1);
            }
            if (i != num_devices - 1) {
                forward_device = devices.at(i + 1);
            } else if (topology == ttnn::ccl::Topology::Ring) {
                forward_device = devices.at(0);
            }
        }
    }

    return ttnn::SwapTensorAsync{
        forward_device,
        backward_device,
        num_links,
        num_devices,
        device_index,
        memory_config.value_or(input_tensor.memory_config()),
        topology,
        semaphore.value(),
        sub_device_id};
}

}  // namespace swap_tensor_detail
}  // namespace ccl

void SwapTensorAsync::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    TT_FATAL(
        input_tensors.size() == 1 || input_tensors.size() == 3,
        "Error, Input tensor size should be 1 or 3 but has {}",
        input_tensors.size());
    const auto& input_tensor = input_tensors[0];
    const auto& layout = input_tensors[0].get_layout();
    const auto& dtype = input_tensors[0].get_dtype();
    const auto& page_size = input_tensors[0].buffer()->page_size();
    TT_FATAL(page_size % input_tensors[0].buffer()->alignment() == 0, "Swap Tensor currently requires aligned pages");

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to swap_tensor need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to swap_tensor need to be allocated in buffers on device!");
    TT_FATAL(this->num_links == 1, "Error, num_links should be 1 but has {}", this->num_links);
    TT_FATAL(
        this->num_links <= input_tensor.device()->compute_with_storage_grid_size().y,
        "Worker cores used by links are parallelizaed over rows");

    TT_FATAL(
        input_tensor.memory_config().memory_layout == TensorMemoryLayout::WIDTH_SHARDED ||
            input_tensor.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED ||
            input_tensor.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED,
        "Unsupported memory layout {}.",
        input_tensor.memory_config().memory_layout);

    if (output_tensors.size() > 0 and output_tensors[0].has_value()) {
        TT_FATAL(
            output_tensors.size() == 1,
            "Error, Number of output tensors should be 1 but has {}",
            output_tensors.size());

        const auto& output_tensor = output_tensors[0];
        TT_FATAL(
            output_tensor.value().storage_type() == StorageType::DEVICE,
            "Operands to swap_tensor need to be on device!");
        TT_FATAL(
            output_tensor.value().get_layout() == layout,
            "Error, Output tensor layout should be same as input tensor layout but has {}",
            output_tensor.value().get_layout());
        TT_FATAL(
            output_tensor.value().get_dtype() == dtype,
            "Error, Output tensor dtype should be same as input tensor dtype but has {}",
            output_tensor.value().get_dtype());
        TT_FATAL(
            output_tensor.value().get_tensor_spec().page_config() == input_tensor.get_tensor_spec().page_config(),
            "Error, Output tensor page config should be same as input tensor page config but has {}",
            output_tensor.value().get_tensor_spec().page_config());
        TT_FATAL(
            output_tensor.value().memory_config() == this->output_mem_config,
            "Error, Output tensor memory config should be same as output_mem_config but has {}",
            output_tensor.value().memory_config());

        // check the output tensor size
        auto output_shape = output_tensor.value().get_padded_shape();
        auto input_shape = input_tensor.get_padded_shape();
        TT_FATAL(
            output_shape.size() == input_shape.size(),
            "Error, Output tensor shape should have same number of dimensions as input tensor but has {}",
            output_shape.size());

        TT_FATAL(
            input_shape.size() == 4,
            "Error, Input tensor shape should have 4 dimensions but has {}",
            input_shape.size());
        TT_FATAL(
            input_shape[0] == 1, "Error, Input tensor shape should have size 1 in dim 0 but has {}", input_shape[0]);
        TT_FATAL(
            input_shape[1] == 1, "Error, Input tensor shape should have size 1 in dim 1 but has {}", input_shape[1]);
        TT_FATAL(
            input_shape[1] == 32, "Error, Input tensor shape should have size 32 in dim 2 but has {}", input_shape[1]);
        TT_FATAL(
            input_shape[3] % 32,
            "Error, Input tensor shape should be a multiple of 32 in dim3 but is {}",
            input_shape[1]);

        // check memory layout
        TT_FATAL(
            output_tensor.value().memory_config().memory_layout == input_tensor.memory_config().memory_layout,
            "Error, Output tensor memory layout should be same as input tensor memory layout but has {}",
            output_tensor.value().memory_config().memory_layout);
    }

    // Validate the priority tensors if provided
    if (input_tensors.size() > 1) {
        const auto& priority_tensor_a = input_tensors[1];
        const auto& priority_tensor_b = input_tensors[2];

        TT_FATAL(
            priority_tensor_a.storage_type() == StorageType::DEVICE, "Operands to swap_tensor need to be on device!");
        TT_FATAL(
            priority_tensor_b.storage_type() == StorageType::DEVICE, "Operands to swap_tensor need to be on device!");

        TT_FATAL(
            priority_tensor_a.get_layout() == layout,
            "Error, Priority tensor A layout should be same as input tensor layout but has {}",
            priority_tensor_a.get_layout());
        TT_FATAL(
            priority_tensor_b.get_layout() == layout,
            "Error, Priority tensor B layout should be same as input tensor layout but has {}",
            priority_tensor_b.get_layout());

        TT_FATAL(
            priority_tensor_a.get_dtype() == DataType::INT32,
            "Error, Priority tensor A dtype should be INT32 but has {}",
            priority_tensor_a.get_dtype());
        TT_FATAL(
            priority_tensor_b.get_dtype() == DataType::INT32,
            "Error, Priority tensor B dtype should be INT32 but has {}",
            priority_tensor_b.get_dtype());

        TT_FATAL(
            priority_tensor_a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED &&
                priority_tensor_a.memory_config().is_dram(),
            "Error, Priority tensor A should be DRAM Interleaved but has {}",
            priority_tensor_a.memory_config());
        TT_FATAL(
            priority_tensor_b.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED &&
                priority_tensor_b.memory_config().is_dram(),
            "Error, Priority tensor B should be DRAM Interleaved but has {}",
            priority_tensor_b.memory_config());
    }
}

std::vector<ttnn::TensorSpec> SwapTensorAsync::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors[0];
    auto shape = input_tensor.get_padded_shape();  // TODO: Replace with get_logical_shape()

    return {TensorSpec(
        shape,
        TensorLayout(input_tensor.get_dtype(), input_tensor.get_tensor_spec().page_config(), output_mem_config))};
}

SwapTensorAsyncVersion SwapTensorAsync::select_version(const Tensor& input_tensor) const {
    // Only MINIMAL_SHARDED is supported for now, and is checked for in validation

    return SwapTensorAsyncVersion::MINIMAL_SHARDED;
}

tt::tt_metal::operation::ProgramWithCallbacks SwapTensorAsync::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    tt::log_debug(tt::LogOp, "DEBUG: create_program is called");
    SwapTensorAsyncVersion version = select_version(input_tensors[0]);

    log_trace(tt::LogOp, "version: {}", static_cast<uint32_t>(version));

    const std::optional<Tensor> priority_tensor_a =
        input_tensors.size() > 1 ? std::make_optional(input_tensors[1]) : std::nullopt;
    const std::optional<Tensor> priority_tensor_b =
        input_tensors.size() > 2 ? std::make_optional(input_tensors[2]) : std::nullopt;

    switch (version) {
        case SwapTensorAsyncVersion::MINIMAL_SHARDED:
        default:
            return swap_tensor_async_llama_sharded(
                input_tensors[0],
                priority_tensor_a,
                priority_tensor_b,
                this->forward_device,
                this->backward_device,
                output_tensors[0],
                this->num_links,
                this->ring_size,
                this->ring_index,
                this->topology,
                this->semaphore,
                this->sub_device_id);
    }
}

tt::tt_metal::operation::Hash SwapTensorAsync::compute_program_hash(const std::vector<Tensor>& input_tensors) const {
    log_trace(tt::LogOp, "compute_program_hash is called");
    SwapTensorAsyncVersion version = select_version(input_tensors[0]);
    log_trace(tt::LogOp, "version: {}", static_cast<uint32_t>(version));
    auto input_shape = input_tensors[0].get_padded_shape();
    auto input_memory_layout = input_tensors[0].get_layout();
    auto input_dtype = input_tensors[0].get_dtype();
    auto input_memory_config = input_tensors[0].memory_config();

    return tt::tt_metal::operation::hash_operation<SwapTensorAsync>(
        this->num_links,
        this->ring_size,
        this->ring_index,
        this->output_mem_config,
        this->topology,
        input_shape,
        input_memory_layout,
        input_dtype,
        input_memory_config);
}

namespace operations {
namespace experimental {
namespace ccl {

Tensor swap_tensor_async(
    const Tensor& input_tensor,
    const global_semaphore::MultiDeviceGlobalSemaphore& multi_device_global_semaphore,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id) {
    TT_FATAL(
        std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr,
        "swap_tensor_async op is only supported for Fast Dispatch");
    auto devices = input_tensor.get_workers();
    uint32_t num_devices = devices.size();
    TT_FATAL(num_devices > 1, "swap_tensor_async op will only work for num_devices > 1, but has {}", num_devices);
    ttnn::ccl::Topology ccl_topology = topology;

    if (num_devices == 2) {
        ccl_topology = ttnn::ccl::Topology::Linear;
    }
    std::vector<Tensor> output_tensors = {Tensor(tt::tt_metal::operation::get_workers_for_op_output({input_tensor}))};

    tt::log_debug(
        tt::LogOp, "DEBUG: creating line_fabric with num devices: {}, num links: {}", devices.size(), num_links);
    tt::log_debug(tt::LogOp, "DEBUG: line_fabric is created");

    std::vector<GlobalSemaphore> semaphores = multi_device_global_semaphore.global_semaphores;

    tt::tt_metal::operation::launch_op(
        [num_links, num_devices, memory_config, devices, ccl_topology, semaphores, sub_device_id](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            const auto& input_tensor = input_tensors.at(0);

            return tt::tt_metal::operation::run(
                ttnn::ccl::swap_tensor_detail::create_swap_tensor_async_struct(
                    input_tensor, num_links, memory_config, devices, ccl_topology, semaphores, sub_device_id),
                {input_tensor},
                optional_input_tensors,
                optional_output_tensors);
        },
        {input_tensor},
        output_tensors);
    return output_tensors.at(0);
}

Tensor swap_tensor_async(
    const Tensor& input_tensor,
    const Tensor& priority_tensor_a,
    const Tensor& priority_tensor_b,
    const global_semaphore::MultiDeviceGlobalSemaphore& multi_device_global_semaphore,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id) {
    TT_FATAL(
        std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr,
        "swap_tensor_async op is only supported for Fast Dispatch");
    auto devices = input_tensor.get_workers();
    uint32_t num_devices = devices.size();
    TT_FATAL(num_devices > 1, "swap_tensor_async op will only work for num_devices > 1, but has {}", num_devices);
    ttnn::ccl::Topology ccl_topology = topology;

    if (num_devices == 2) {
        ccl_topology = ttnn::ccl::Topology::Linear;
    }
    std::vector<Tensor> output_tensors = {Tensor(tt::tt_metal::operation::get_workers_for_op_output({input_tensor}))};

    tt::log_debug(
        tt::LogOp, "DEBUG: creating line_fabric with num devices: {}, num links: {}", devices.size(), num_links);
    tt::log_debug(tt::LogOp, "DEBUG: line_fabric is created");

    std::vector<GlobalSemaphore> semaphores = multi_device_global_semaphore.global_semaphores;

    tt::tt_metal::operation::launch_op(
        [num_links, num_devices, memory_config, devices, ccl_topology, semaphores, sub_device_id](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            const auto& input_tensor = input_tensors.at(0);
            const auto& priority_tensor_a = input_tensors.at(1);
            const auto& priority_tensor_b = input_tensors.at(2);

            return tt::tt_metal::operation::run(
                ttnn::ccl::swap_tensor_detail::create_swap_tensor_async_struct(
                    input_tensor, num_links, memory_config, devices, ccl_topology, semaphores, sub_device_id),
                {input_tensor, priority_tensor_a, priority_tensor_b},
                optional_input_tensors,
                optional_output_tensors);
        },
        {input_tensor, priority_tensor_a, priority_tensor_b},
        output_tensors);
    return output_tensors.at(0);
}

}  // namespace ccl
}  // namespace experimental
}  // namespace operations

}  // namespace ttnn

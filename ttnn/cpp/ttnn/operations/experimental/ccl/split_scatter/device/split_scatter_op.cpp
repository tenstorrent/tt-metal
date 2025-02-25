// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "split_scatter_op.hpp"
#include "ttnn/operations/math.hpp"
#include "cpp/ttnn/global_semaphore.hpp"

#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn {
namespace ccl {
namespace split_scatter_detail {

SplitScatter create_split_scatter_struct(
    const Tensor& input_tensor,
    const uint32_t dim,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    const std::vector<IDevice*>& devices,
    const ttnn::ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphores,
    std::optional<SubDeviceId> sub_device_id,
    bool enable_persistent_fabric_mode) {
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
                std::cout << "\n BWWWWWWW \n";
                backward_device = devices.at(i - 1);
            }
            if (i != num_devices - 1) {
                std::cout << "\n FWWWWWWW \n";
                forward_device = devices.at(i + 1);
            }
        }
    }

    return ttnn::SplitScatter{
        forward_device,
        backward_device,
        dim,
        num_links,
        num_devices,
        device_index,
        memory_config.value_or(input_tensor.memory_config()),
        topology,
        semaphore.value(),
        sub_device_id,
        enable_persistent_fabric_mode};
}

}  // namespace split_scatter_detail
}  // namespace ccl

void SplitScatter::validate(const std::vector<Tensor>& input_tensors) const {
    TT_FATAL(input_tensors.size() == 1, "Error, Input tensor size should be 1 but has {}", input_tensors.size());
    const auto& input_tensor = input_tensors[0];
    const auto& layout = input_tensors[0].get_layout();
    const auto& dtype = input_tensors[0].get_dtype();
    const auto& page_size = input_tensors[0].buffer()->page_size();
    TT_FATAL(page_size % input_tensors[0].buffer()->alignment() == 0, "Split Scatter currently requires aligned pages");

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to split_scatter need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to split_scatter need to be allocated in buffers on device!");
    TT_FATAL(this->num_links > 0, "Error, num_links should be more than 0 but has {}", this->num_links);
    TT_FATAL(
        this->num_links <= input_tensor.device()->compute_with_storage_grid_size().y,
        "Worker cores used by links are parallelizaed over rows");

    TT_FATAL(
        input_tensor.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED ||
            input_tensor.memory_config().memory_layout == TensorMemoryLayout::WIDTH_SHARDED ||
            input_tensor.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED ||
            input_tensor.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED,
        "Unsupported memory layout {}.",
        input_tensor.memory_config().memory_layout);
}

static void validate_output_tensor_allocation(const std::vector<Tensor>& output_tensors) {
    for (const auto& output_tensor : output_tensors) {
        const auto& buffers = output_tensor.buffers();
        const auto first_address = buffers.front()->address();
        TT_FATAL(
            std::all_of(
                buffers.begin(),
                buffers.end(),
                [&first_address](const auto& buffer) {
                    return buffer != nullptr && buffer->address() == first_address;
                }),
            "Output buffers for split_scatter async must be lock-step allocated but some of the tensors were allocated "
            "at "
            "different addresses across devices.");
    }
}

std::vector<ttnn::TensorSpec> SplitScatter::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors[0];
    auto shape = input_tensor.get_padded_shape();  // TODO: Replace with get_logical_shape()
    shape[this->dim] /= this->ring_size;
    return {TensorSpec(
        shape,
        TensorLayout(input_tensor.get_dtype(), input_tensor.get_tensor_spec().page_config(), output_mem_config))};
}

operation::ProgramWithCallbacks SplitScatter::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    tt::log_debug(tt::LogOp, "DEBUG: create_program is called");

    log_trace(tt::LogOp, "version: {}", static_cast<uint32_t>(version));

    log_trace(
        tt::LogOp,
        "Detected split scatter specialized shape. split_scatter_interleaved is "
        "called");
    return split_scatter_interleaved(
        input_tensors[0],
        this->forward_device,
        this->backward_device,
        output_tensors[0],
        this->dim,
        this->num_links,
        this->ring_size,
        this->ring_index,
        this->topology,
        this->semaphore,
        this->sub_device_id,
        this->enable_persistent_fabric_mode);
}

const operation::Hash SplitScatter::compute_program_hash(const std::vector<Tensor>& input_tensors) const {
    log_trace(tt::LogOp, "compute_program_hash is called");
    auto input_shape = input_tensors[0].get_padded_shape();
    auto input_memory_layout = input_tensors[0].get_layout();
    auto input_dtype = input_tensors[0].get_dtype();
    auto input_memory_config = input_tensors[0].memory_config();

    return operation::hash_operation<SplitScatter>(
        this->dim,
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

Tensor split_scatter(
    Tensor& input_tensor,
    const uint32_t dim,
    const global_semaphore::MultiDeviceGlobalSemaphore& multi_device_global_semaphore,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    const ttnn::ccl::Topology topology,
    std::optional<SubDeviceId> sub_device_id,
    bool enable_persistent_fabric_mode) {
    std::cout << "\n 111111111";
    input_tensor.print();
    std::cout << "\n 222222222";
    TT_FATAL(
        std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr, "split_scatter op is only supported for Fast Dispatch");

    std::cout << "\n fc1";
    auto devices = input_tensor.get_workers();
    std::vector<Tensor> input_tensors = ttnn::distributed::get_tensors_from_multi_device_storage(input_tensor);
    std::cout << "\n fc2";
    std::vector<Tensor> input_tensorss = {input_tensors[0], input_tensors[0].cpu().to_device(devices.at(1))};
    input_tensor = ttnn::distributed::aggregate_as_tensor(
        input_tensorss, ttnn::distributed::get_distributed_tensor_config_from_tensor(input_tensor));
    // input_tensor = ttnn::distributed::create_multi_device_tensor(
    //     input_tensorss, StorageType::MULTI_DEVICE,
    //     ttnn::distributed::get_distributed_tensor_config_from_tensor(input_tensor));
    std::cout << "\n fc3";
    // auto multi_out = ccl_input_tensors[0];
    std::cout << "\n multi out tensor";
    // for (const auto& id : multi_out) {
    //     id.print();
    // }
    std::cout << "\n 44444444";
    input_tensor.print();
    std::cout << "\n 55555555";

    uint32_t num_devices = devices.size();
    std::cout << "\n NUM_DEVICES AFTER " << num_devices;
    TT_FATAL(num_devices > 1, "split_scatter op will only work for num_devices > 1, but has {}", num_devices);
    ttnn::ccl::Topology ccl_topology = topology;

    if (num_devices == 2) {
        ccl_topology = ttnn::ccl::Topology::Linear;
    }
    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor}))};

    tt::log_debug(
        tt::LogOp, "DEBUG: creating line_fabric with num devices: {}, num links: {}", devices.size(), num_links);
    tt::log_debug(tt::LogOp, "DEBUG: line_fabric is created");

    // create this semaphore for all cores since we don't know which core will be used for teardown draining
    CoreCoord grid_size = devices[0]->compute_with_storage_grid_size();
    auto core_grid = CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1});

    std::vector<GlobalSemaphore> semaphores = multi_device_global_semaphore.global_semaphores;
    bool is_multi_device_tensor = ttnn::distributed::is_multi_device_tensor(input_tensor);
    std::cout << "\n is_multi_device_tensor BEFORE   " << is_multi_device_tensor;
    std::cout << "\n is_multi_device_tensor BEFORE    ";
    operation::launch_op(
        [dim,
         num_links,
         num_devices,
         memory_config,
         devices,
         ccl_topology,
         semaphores,
         sub_device_id,
         enable_persistent_fabric_mode](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            const auto& input_tensor = input_tensors.at(0);

            bool is_multi_device_tensor = ttnn::distributed::is_multi_device_tensor(input_tensor);
            std::cout << "\n is_multi_device_tensor " << is_multi_device_tensor;
            std::cout << "\n is_multi_device_tensor ";

            return operation::run(
                ttnn::ccl::split_scatter_detail::create_split_scatter_struct(
                    input_tensor,
                    dim,
                    num_links,
                    memory_config,
                    devices,
                    ccl_topology,
                    semaphores,
                    sub_device_id,
                    enable_persistent_fabric_mode),
                {input_tensor});
        },
        {input_tensor},
        output_tensors);
    return output_tensors.at(0);
}

}  // namespace ccl
}  // namespace experimental
}  // namespace operations

}  // namespace ttnn

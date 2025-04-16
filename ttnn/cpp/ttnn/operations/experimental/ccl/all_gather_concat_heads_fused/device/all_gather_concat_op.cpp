// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_concat_op.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/operations/core/core.hpp"
#include "cpp/ttnn/global_semaphore.hpp"
#include <tt-metalium/work_split.hpp>

#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn {
namespace ccl {
namespace all_gather_concat_detail {

AllGatherConcat create_all_gather_concat_struct(
    const Tensor& input_tensor,
    const uint32_t dim,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    const std::vector<IDevice*>& devices,
    const ttnn::ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphores,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    const uint32_t num_heads) {
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
            }
            if (i != num_devices - 1) {
                forward_device = devices.at(i + 1);
            }
        }
    }

    return ttnn::AllGatherConcat{
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
        num_heads};
}

}  // namespace all_gather_concat_detail
}  // namespace ccl

void AllGatherConcat::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors[0];
    const auto& layout = input_tensors[0].get_layout();
    const auto& dtype = input_tensors[0].get_dtype();
    const auto& page_size = input_tensors[0].buffer()->page_size();
    const auto input_core_ranges = input_tensor.buffer()->shard_spec().grid().ranges();
    const auto padded_input_shape = input_tensor.get_padded_shape();
    TT_FATAL(page_size % input_tensors[0].buffer()->alignment() == 0, "All Gather currently requires aligned pages");

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to all_gather need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to all_gather need to be allocated in buffers on device!");
    TT_FATAL(this->num_links > 0, "Error, num_links should be more than 0 but has {}", this->num_links);
    TT_FATAL(
        this->num_links <= input_tensor.device()->compute_with_storage_grid_size().y,
        "Worker cores used by links are parallelizaed over rows");

    TT_FATAL(
        input_tensor.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED,
        "Unsupported memory layout {}.",
        input_tensor.memory_config().memory_layout);

    TT_FATAL(
        input_core_ranges[0].start_coord.x == 1 && input_core_ranges[0].end_coord.x == 3 &&
            input_core_ranges[0].start_coord.y == 0 && input_core_ranges[0].end_coord.y == 1 &&
            input_core_ranges[1].start_coord.x == 1 && input_core_ranges[1].end_coord.x == 2 &&
            input_core_ranges[1].start_coord.y == 2 && input_core_ranges[1].end_coord.y == 2,
        "Unsupported input core ranges!");
    CoreCoord grid_size = input_tensors[0].device()->compute_with_storage_grid_size();
    TT_FATAL(grid_size.x >= 3 && grid_size.y >= 3, "Input core grid out of bound!");
    TT_FATAL(
        padded_input_shape[0] == 1 && padded_input_shape[1] == 8 && padded_input_shape[2] == 32 &&
            padded_input_shape[3] == 128,
        "Unsupported input shape, should be [1, 8, 32, 128]!");
}

static void validate_output_tensor_alloc(const std::vector<Tensor>& output_tensors) {
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
            "Output buffers for all_gather async must be lock-step allocated but some of the tensors were allocated at "
            "different addresses across devices.");
    }
}

std::vector<ttnn::TensorSpec> AllGatherConcat::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors[0];
    auto input_shape = input_tensor.get_padded_shape();  // TODO: Replace with get_logical_shape()
    auto num_heads = this->num_heads;
    auto sequence_length = input_shape[0];
    auto batch = input_shape[1];
    auto head_dim = input_shape[3];
    // pad batch to 32 if necessary
    uint32_t batch_size = 32;
    if (batch < batch_size) {
        batch = batch_size;
    }
    auto hidden_dim = num_heads * head_dim;

    Shape output_shape({sequence_length, 1, batch, hidden_dim});

    CoreRangeSet output_core_grid;
    auto core_range_1 = CoreRange(CoreCoord{1, 0}, CoreCoord{3, 1});
    auto core_range_2 = CoreRange(CoreCoord{1, 2}, CoreCoord{2, 2});
    output_core_grid = CoreRangeSet(std::vector{core_range_1, core_range_2});
    tt::tt_metal::ShardSpec shard_spec{output_core_grid, {batch, head_dim}};
    auto mem_config =
        tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED, tt::tt_metal::BufferType::L1};
    mem_config.shard_spec = shard_spec;

    return {TensorSpec(
        output_shape, tt::tt_metal::TensorLayout(input_tensor.get_dtype(), tt::tt_metal::Layout::TILE, mem_config))};
}

tt::tt_metal::operation::ProgramWithCallbacks AllGatherConcat::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    tt::log_debug(tt::LogOp, "DEBUG: create_program is called");

    log_trace(tt::LogOp, "Detected all gather specialized shape. all_gather_concat_llama_sharded is called");
    CoreCoord compute_with_storage_grid_size = input_tensors[0].device()->compute_with_storage_grid_size();
    return all_gather_concat_llama_sharded(
        input_tensors[0],
        input_tensors[1],
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
        this->num_heads);
}

tt::tt_metal::operation::Hash AllGatherConcat::compute_program_hash(const std::vector<Tensor>& input_tensors) const {
    log_trace(tt::LogOp, "compute_program_hash is called");
    auto input_shape = input_tensors[0].get_padded_shape();
    auto input_memory_layout = input_tensors[0].get_layout();
    auto input_dtype = input_tensors[0].get_dtype();
    auto input_memory_config = input_tensors[0].memory_config();

    return tt::tt_metal::operation::hash_operation<AllGatherConcat>(
        this->dim,
        this->num_links,
        this->ring_size,
        this->ring_index,
        this->output_mem_config,
        this->topology,
        input_shape,
        input_memory_layout,
        input_dtype,
        input_memory_config,
        this->num_heads);
}

namespace operations {
namespace experimental {
namespace ccl {

Tensor all_gather_concat(
    const Tensor& input_tensor,
    Tensor& buffer_tensor,
    const uint32_t dim,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const global_semaphore::MultiDeviceGlobalSemaphore& multi_device_global_semaphore,
    const uint32_t num_heads,
    const std::optional<uint32_t> num_links,
    const std::optional<MemoryConfig>& memory_config,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id) {
    TT_FATAL(
        topology == ttnn::ccl::Topology::Linear,
        "This all_gather API with cluster_axis is currently supported only for the Linear topology");
    const auto mesh_view = mesh_device.get_view();
    auto devices = input_tensor.get_workers();
    uint32_t num_devices = (cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();

    int32_t rank = input_tensor.get_logical_shape().rank();

    int32_t gather_dim = (dim < 0) ? rank + dim : dim;
    TT_FATAL(
        gather_dim >= -rank && gather_dim <= rank - 1,
        "Dimension input should be in between -{} and {}, but has {}",
        rank,
        rank - 1,
        dim);

    std::vector<Tensor> output_tensors = {Tensor(tt::tt_metal::operation::get_workers_for_op_output({input_tensor}))};
    // create this semaphore for all cores since we don't know which core will be used for teardown draining
    CoreCoord grid_size = devices[0]->compute_with_storage_grid_size();
    auto core_grid = CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    std::vector<GlobalSemaphore> semaphores = multi_device_global_semaphore.global_semaphores;

    tt::tt_metal::operation::launch_op(
        [gather_dim,
         mesh_view,
         cluster_axis,
         num_links,
         num_devices,
         memory_config,
         devices,
         topology,
         semaphores,
         sub_device_id,
         num_heads](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            const auto& input_tensor = input_tensors.at(0);
            const auto& buffer_tensor = input_tensors.at(1);

            TT_FATAL(
                mesh_view.is_mesh_2d(),
                "all-gather invoked with cluster_axis API on >2D mesh, which is currently unsupported");
            const auto coordinate = mesh_view.find_device(input_tensor.device()->id());
            std::vector<IDevice*> devices = (cluster_axis == 0) ? mesh_view.get_devices_on_column(coordinate[1])
                                                                : mesh_view.get_devices_on_row(coordinate[0]);
            return tt::tt_metal::operation::run(
                ttnn::ccl::all_gather_concat_detail::create_all_gather_concat_struct(
                    input_tensor,
                    gather_dim,
                    num_links.has_value() ? num_links.value() : 1,
                    memory_config,
                    devices,
                    topology,
                    semaphores,
                    sub_device_id,
                    num_heads),
                {input_tensor, buffer_tensor});
        },
        {input_tensor, buffer_tensor},
        output_tensors);
    return output_tensors.at(0);
}

}  // namespace ccl
}  // namespace experimental
}  // namespace operations

}  // namespace ttnn

// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rms_allgather_op.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/math.hpp"

#include <tt-metalium/constants.hpp>

#include <optional>

using uint32_t = std::uint32_t;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::fused::normalization {

RMSAllGather create_rms_struct(
    const Tensor& input_tensor,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    const std::vector<IDevice*>& devices,
    const ttnn::ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphores,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    float epsilon,
    const ttnn::operations::normalization::LayerNormProgramConfig program_config,
    const DeviceComputeKernelConfig compute_kernel_config,
    std::optional<DataType> dtype,
    const bool is_pre) {
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
    return RMSAllGather(
        epsilon,
        memory_config.value_or(input_tensor.memory_config()),
        program_config,
        compute_kernel_config,
        dtype,
        topology,
        is_pre,
        num_links,
        num_devices,
        device_index,
        semaphore.value(),
        sub_device_id,
        forward_device,
        backward_device);
}

const tt::tt_metal::operation::Hash RMSAllGather::compute_program_hash(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    log_trace(tt::LogOp, "compute_program_hash is called");
    auto input_shape = input_tensors[0].get_padded_shape();
    auto input_memory_layout = input_tensors[0].get_layout();
    auto input_dtype = input_tensors[0].get_dtype();
    auto input_memory_config = input_tensors[0].memory_config();
    return tt::tt_metal::operation::hash_operation<RMSAllGather>(
        this->eps,
        this->dtype,
        this->is_pre,
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

void RMSAllGather::validate(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    TT_FATAL(
        input_tensors.size() == 1 and optional_input_tensors.size() <= 4, "Must have between 1 to 4 input tensors");
    auto& a = input_tensors.at(0);
    uint32_t input_width = a.get_tensor_spec().tile().get_tile_shape()[1];
    uint32_t input_height = a.get_tensor_spec().tile().get_tile_shape()[0];
    const auto& b = optional_input_tensors.at(0);
    const auto& gamma = optional_input_tensors.at(1);
    const auto& stats = optional_input_tensors.at(2);
    TT_FATAL(
        this->output_mem_config.shard_spec.value().orientation == ShardOrientation::ROW_MAJOR,
        "Minimal version requires row major sharding orientation");
    TT_FATAL(
        a.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR,
        "Minimal version requires row major sharding orientation");
    TT_FATAL(a.get_layout() == Layout::TILE, "Error");
    TT_FATAL(
        a.get_dtype() == DataType::FLOAT32 or a.get_dtype() == DataType::BFLOAT16 or
            a.get_dtype() == DataType::BFLOAT8_B,
        "Error");
    TT_FATAL(
        a.storage_type() == StorageType::DEVICE || a.storage_type() == StorageType::MULTI_DEVICE,
        "Operands to frmsnorm need to be on device!");
    TT_FATAL(a.buffer() != nullptr, "Operands to frmsnorm need to be allocated in buffers on device!");

    if (b.has_value()) {
        TT_FATAL(b.value().get_layout() == Layout::TILE, "layout is not tile!");
        TT_FATAL(a.get_padded_shape() == b.value().get_padded_shape(), "shape is not same!");
        TT_FATAL(b.value().buffer() != nullptr, "Operands to frmsnorm need to be allocated in buffers on device!");
        TT_FATAL(a.device() == b.value().device(), "device is not same!");
    }
    if (!this->is_pre) {
        TT_FATAL(
            gamma.has_value() and gamma.value().get_layout() == Layout::ROW_MAJOR,
            "Post all gather requires a weight which is row major");
        TT_FATAL(stats.has_value(), "Post all gather layernorm requires stats");
        TT_FATAL(stats.value().is_sharded(), "Stats must be sharded");
        TT_FATAL(stats.value().get_layout() == Layout::TILE, "Only tile layout is supported for stats");
        TT_FATAL(stats.value().get_dtype() == DataType::BFLOAT16, "Only bfloat16 is supported for stats");
        TT_FATAL(
            stats.value().storage_type() == StorageType::DEVICE ||
                stats.value().storage_type() == StorageType::MULTI_DEVICE,
            "Operands to layernorm need to be on device!");
        TT_FATAL(stats.value().buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
        TT_FATAL(
            stats.value().get_padded_shape()[-1] % input_width == 0,
            "Stats is expected to have E(x) for each device stacked in the last dimension");
    }

    if (gamma.has_value()) {
        if (gamma.value().get_layout() == Layout::TILE) {
            TT_FATAL(
                a.get_padded_shape()[-1] == gamma.value().get_padded_shape()[-1],
                "{} != {}",
                a.get_padded_shape()[-1],
                gamma.value().get_padded_shape()[-1]);
            TT_FATAL(
                gamma.value().buffer() != nullptr, "Operands to frmsnorm need to be allocated in buffers on device!");
            TT_FATAL(a.device() == gamma.value().device(), "Error");
            TT_FATAL(gamma.value().get_padded_shape()[-2] == input_height, "Error");
        } else {
            TT_FATAL(gamma.value().get_layout() == Layout::ROW_MAJOR, "Error");
            TT_FATAL(
                (gamma.value().get_padded_shape()[-1] == input_width &&
                 gamma.value().volume() / input_width == a.get_padded_shape()[-1] / input_width),
                "Error");
            TT_FATAL(
                gamma.value().buffer() != nullptr, "Operands to frmsnorm need to be allocated in buffers on device!");
            TT_FATAL(a.device() == gamma.value().device(), "Error");
            TT_FATAL(
                gamma.value().get_dtype() == DataType::FLOAT32 or gamma.value().get_dtype() == DataType::BFLOAT16,
                "Error");
        }
    }

    if (a.is_sharded()) {
        // TODO: Add support for this (should be similar to interleaved)
        TT_FATAL(
            a.memory_config().memory_layout != TensorMemoryLayout::HEIGHT_SHARDED,
            "Height sharded inputs are not supported.");
        TT_FATAL(
            this->output_mem_config.is_sharded() &&
                this->output_mem_config.memory_layout != TensorMemoryLayout::HEIGHT_SHARDED,
            "Sharded inputs require sharded outputs.");
        if (b.has_value()) {
            TT_FATAL(b.value().is_sharded(), "residual tensor b should be sharded if input a is sharded");
            TT_FATAL(b.value().shard_spec() == a.shard_spec(), "Both a and b should have the same shard spec");
            TT_FATAL(b.value().memory_config() == a.memory_config(), "Both a and b should have the same memory config");
        }
    }

    TT_FATAL(a.get_padded_shape()[-2] == input_height, "Only activations with batch size = 32 are supported");
    if (b.has_value()) {
        TT_FATAL(
            b.value().get_padded_shape()[-2] == input_height,
            "Only residual tensors with batch size = 32 are supported");
    }
    std::visit(
        [&](const auto& program_config) {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            TT_FATAL(
                (std::is_same_v<
                    ProgramConfigType,
                    ttnn::operations::normalization::LayerNormShardedMultiCoreProgramConfig>),
                "Fused RMS Allgather only supports Sharded");
            if constexpr (std::is_same_v<
                              ProgramConfigType,
                              ttnn::operations::normalization::LayerNormShardedMultiCoreProgramConfig>) {
                if (program_config.inplace) {
                    TT_FATAL(this->output_mem_config.is_sharded(), "Error");
                }
                TT_FATAL(a.memory_config().buffer_type == this->output_mem_config.buffer_type, "Error");
                TT_FATAL(a.memory_config().memory_layout == this->output_mem_config.memory_layout, "Error");

                // tensor shape
                const auto shape = a.get_padded_shape();
                uint32_t M = a.volume() / shape[-1];
                uint32_t K = shape[-1];

                uint32_t Mt = M / input_height;
                uint32_t Kt = K / input_width;
                // block
                uint32_t block_w = program_config.block_w * input_width;
                const auto shard_spec = a.shard_spec().value();
                uint32_t num_subblocks_w = program_config.block_w / program_config.subblock_w;
                // check dims
                TT_FATAL(
                    program_config.block_w % program_config.subblock_w == 0,
                    "block_w must be divisible by subblock_w.");
                TT_FATAL(M % input_height == 0, "M must be divisible by tile height.");
                TT_FATAL(K % input_width == 0, "K must be divisible by tile width.");
                const auto bbox = shard_spec.grid.bounding_box();
                TT_FATAL(
                    bbox.end_coord.x - bbox.start_coord.x < program_config.compute_with_storage_grid_size.x &&
                        bbox.end_coord.y - bbox.start_coord.y < program_config.compute_with_storage_grid_size.y,
                    "Error");

                TT_FATAL(M == input_height, "Minimal version assumes (1,1,TILE_HEIGHT,N) shape");
                TT_FATAL(program_config.block_h == 1, "Minimal version assumes block_h is 1");
                bool row_wise = shard_spec.orientation == ShardOrientation::ROW_MAJOR;
                TT_FATAL(
                    tt::div_up(Kt, shard_spec.num_cores()) == program_config.block_w,
                    "block_w must equal to K / num_cores.");
                TT_FATAL(Mt == program_config.block_h, "block_h must equal to M.");
                TT_FATAL(a.memory_config().memory_layout != TensorMemoryLayout::HEIGHT_SHARDED, "Error");
                if (b.has_value()) {
                    TT_FATAL(b.value().is_sharded(), "Error");
                    TT_FATAL(b.value().shard_spec() == shard_spec, "Error");
                }
                TT_FATAL(program_config.block_w * input_width == shard_spec.shape[1], "Error");
                TT_FATAL(
                    program_config.block_w % program_config.subblock_w == 0,
                    "block_w must be divisible by subblock_w.");
            }
        },
        this->program_config);
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
            "Output buffers for all_gather async must be lock-step allocated but some of the tensors were allocated at "
            "different addresses across devices.");
    }
}

std::vector<TensorSpec> RMSAllGather::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto output_shape = input_tensor.get_logical_shape();
    auto output_padded_shape = input_tensor.get_padded_shape();

    // WARNING!!!!! This line is ONLY true when only doing pre-allgather only
    if (this->is_pre) {
        output_shape[3] = input_tensor.get_tensor_spec().tile().get_tile_shape()[1] * this->ring_size;
    }

    return std::visit(
        [&](const auto& program_config) -> std::vector<TensorSpec> {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            if constexpr (std::is_same_v<
                              ProgramConfigType,
                              ttnn::operations::normalization::LayerNormShardedMultiCoreProgramConfig>) {
                if (this->is_pre) {
                    auto shard_spec = input_tensor.shard_spec().value();
                    shard_spec.shape[1] = output_shape[3];
                    CoreCoord grid_start_core = shard_spec.grid.bounding_box().start_coord;
                    CoreRangeSet output_grid({CoreRange(grid_start_core, grid_start_core)});
                    shard_spec.grid = output_grid;
                    auto mem_config = this->output_mem_config;
                    mem_config.shard_spec = shard_spec;
                    return {TensorSpec(
                        output_shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), mem_config))};
                } else {
                    auto output_shard_spec = this->output_mem_config.shard_spec.value();
                    auto input_shard_spec = input_tensor.shard_spec().value();
                    if (output_shard_spec != input_shard_spec) {
                        output_padded_shape[3] = output_shard_spec.shape[1] * output_shard_spec.num_cores();
                    }
                }
                if (program_config.inplace) {
                    return {input_tensor.get_tensor_spec()};
                }

                auto mem_config = this->output_mem_config;
                if (!mem_config.shard_spec.has_value()) {
                    mem_config.shard_spec = input_tensor.shard_spec().value();
                }

                return {ttnn::TensorSpec(
                    output_shape,
                    TensorLayout::fromPaddedShape(
                        this->dtype.value_or(input_tensor.get_dtype()),
                        PageConfig(Layout::TILE),
                        mem_config,
                        output_shape,
                        output_padded_shape))};
            }
            TT_FATAL(false, "Tensor Spec does not match");
            return {TensorSpec(
                output_shape,
                TensorLayout(input_tensor.get_dtype(), PageConfig(Layout::TILE), this->output_mem_config))};
        },
        this->program_config);
}
std::vector<Tensor> RMSAllGather::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    return std::visit(
        [&](const auto& program_config) -> std::vector<Tensor> {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            if constexpr (std::is_same_v<
                              ProgramConfigType,
                              ttnn::operations::normalization::LayerNormShardedMultiCoreProgramConfig>) {
                if ((!this->is_pre) && program_config.inplace) {
                    return {input_tensors.at(0)};
                }
            }
            auto output_spec = compute_output_specs(input_tensors)[0];
            return {create_device_tensor(output_spec, input_tensors.at(0).device())};
        },
        this->program_config);
}
tt::tt_metal::operation::ProgramWithCallbacks RMSAllGather::create_program(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    const auto& a = input_tensors.at(0);
    const auto& b = optional_input_tensors.at(0);
    const auto& gamma = optional_input_tensors.at(1);
    const auto& stats = optional_input_tensors.at(2);
    auto& output_tensor = output_tensors.at(0);

    return std::visit(
        [&](const auto& program_config) -> tt::tt_metal::operation::ProgramWithCallbacks {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            if constexpr (std::is_same_v<
                              ProgramConfigType,
                              ttnn::operations::normalization::LayerNormShardedMultiCoreProgramConfig>) {
                uint32_t num_cores_x = program_config.compute_with_storage_grid_size.x;
                uint32_t num_cores_y = program_config.compute_with_storage_grid_size.y;
                CoreCoord grid_size = CoreCoord(num_cores_x, num_cores_y);
                if (this->is_pre) {
                    return frmsnorm_pre_multi_core_sharded(
                        a,
                        b,
                        output_tensor,
                        this->eps,
                        program_config.compute_with_storage_grid_size,
                        program_config.subblock_w,
                        program_config.block_w,
                        this->compute_kernel_config,
                        // New Parameters
                        this->forward_device,
                        this->backward_device,
                        this->num_links,
                        this->ring_size,
                        this->ring_index,
                        this->topology,
                        this->semaphore,
                        this->sub_device_id);
                } else {
                    return frmsnorm_post_multi_core_sharded(
                        a,
                        gamma,
                        stats,
                        output_tensor,
                        this->eps,
                        program_config.compute_with_storage_grid_size,
                        program_config.subblock_w,
                        program_config.block_w,
                        this->compute_kernel_config);
                }
            } else {
                TT_FATAL(false, "Program Config does not match");

                using ProgramConfigType = std::decay_t<decltype(program_config)>;
                uint32_t num_cores_x = 1;
                uint32_t num_cores_y = 1;
                CoreCoord grid_size = CoreCoord(num_cores_x, num_cores_y);

                return frmsnorm_pre_multi_core_sharded(
                    a,
                    b,
                    output_tensor,
                    this->eps,
                    grid_size,
                    1,
                    1,
                    this->compute_kernel_config,
                    this->forward_device,
                    this->backward_device,
                    this->num_links,
                    this->ring_size,
                    this->ring_index,
                    this->topology,
                    this->semaphore,
                    this->sub_device_id);
            }
        },
        this->program_config);
}

}  // namespace ttnn::operations::fused::normalization

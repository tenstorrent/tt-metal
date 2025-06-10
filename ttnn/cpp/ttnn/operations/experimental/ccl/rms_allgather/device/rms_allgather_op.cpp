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

tt::tt_metal::operation::Hash RMSAllGather::compute_program_hash(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    log_trace(tt::LogOp, "compute_program_hash is called");
    auto input_shape = input_tensors[0].padded_shape();
    auto input_memory_layout = input_tensors[0].layout();
    auto input_dtype = input_tensors[0].dtype();
    auto input_memory_config = input_tensors[0].memory_config();
    return tt::tt_metal::operation::hash_operation<RMSAllGather>(
        this->eps,
        this->dtype,
        this->num_links,
        this->ring_size,
        this->output_mem_config,
        this->topology,
        this->cluster_axis,
        optional_input_tensors.at(0).has_value(),
        optional_input_tensors.at(1).has_value(),
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
    TT_FATAL(a.padded_shape().rank() == 4, "Input shape must be rank 4");
    uint32_t input_width = a.tensor_spec().tile().get_tile_shape()[1];
    uint32_t input_height = a.tensor_spec().tile().get_tile_shape()[0];
    const auto& b = optional_input_tensors.at(0);
    const auto& gamma = optional_input_tensors.at(1);
    const auto& stats = optional_input_tensors.at(2);
    TT_FATAL(
        this->output_mem_config.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR,
        "Minimal version requires row major sharding orientation");
    TT_FATAL(
        a.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR,
        "Minimal version requires row major sharding orientation");
    TT_FATAL(a.layout() == Layout::TILE, "Error");
    TT_FATAL(
        a.dtype() == DataType::FLOAT32 or a.dtype() == DataType::BFLOAT16 or a.dtype() == DataType::BFLOAT8_B, "Error");
    TT_FATAL(a.storage_type() == StorageType::DEVICE, "Operands to frmsnorm need to be on device!");
    TT_FATAL(a.buffer() != nullptr, "Operands to frmsnorm need to be allocated in buffers on device!");

    if (b.has_value()) {
        TT_FATAL(b.value().layout() == Layout::TILE, "layout is not tile!");
        TT_FATAL(a.padded_shape() == b.value().padded_shape(), "shape is not same!");
        TT_FATAL(b.value().buffer() != nullptr, "Operands to frmsnorm need to be allocated in buffers on device!");
        TT_FATAL(a.device() == b.value().device(), "device is not same!");
    }
    TT_FATAL(
        gamma.has_value() and gamma.value().layout() == Layout::ROW_MAJOR,
        "RMS all gather requires a weight which is row major");

    if (gamma.has_value()) {
        if (gamma.value().layout() == Layout::TILE) {
            TT_FATAL(
                a.padded_shape()[-1] == gamma.value().padded_shape()[-1],
                "{} != {}",
                a.padded_shape()[-1],
                gamma.value().padded_shape()[-1]);
            TT_FATAL(
                gamma.value().buffer() != nullptr, "Operands to frmsnorm need to be allocated in buffers on device!");
            TT_FATAL(a.device() == gamma.value().device(), "Error");
            TT_FATAL(gamma.value().padded_shape()[-2] == input_height, "Error");
        } else {
            TT_FATAL(gamma.value().layout() == Layout::ROW_MAJOR, "Error");
            TT_FATAL(
                (gamma.value().padded_shape()[-1] == input_width &&
                 gamma.value().physical_volume() / input_width == a.padded_shape()[-1] / input_width),
                "Error");
            TT_FATAL(
                gamma.value().buffer() != nullptr, "Operands to frmsnorm need to be allocated in buffers on device!");
            TT_FATAL(a.device() == gamma.value().device(), "Error");
            TT_FATAL(
                gamma.value().dtype() == DataType::FLOAT32 or gamma.value().dtype() == DataType::BFLOAT16, "Error");
        }
    }

    if (a.is_sharded()) {
        // TODO: Add support for this (should be similar to interleaved)
        TT_FATAL(
            a.memory_config().memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED,
            "Height sharded inputs are not supported.");
        TT_FATAL(
            this->output_mem_config.is_sharded() &&
                this->output_mem_config.memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED,
            "Sharded inputs require sharded outputs.");
        if (b.has_value()) {
            TT_FATAL(b.value().is_sharded(), "residual tensor b should be sharded if input a is sharded");
            TT_FATAL(b.value().shard_spec() == a.shard_spec(), "Both a and b should have the same shard spec");
            TT_FATAL(b.value().memory_config() == a.memory_config(), "Both a and b should have the same memory config");
        }
    }

    TT_FATAL(a.padded_shape()[-2] == input_height, "Only activations with batch size = 32 are supported");
    if (b.has_value()) {
        TT_FATAL(
            b.value().padded_shape()[-2] == input_height, "Only residual tensors with batch size = 32 are supported");
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
                TT_FATAL(a.memory_config().buffer_type() == this->output_mem_config.buffer_type(), "Error");
                TT_FATAL(a.memory_config().memory_layout() == this->output_mem_config.memory_layout(), "Error");

                // tensor shape
                const auto shape = a.padded_shape();
                uint32_t M = a.physical_volume() / shape[-1];
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
                TT_FATAL(a.memory_config().memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED, "Error");
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

std::vector<TensorSpec> RMSAllGather::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto output_shape = input_tensor.logical_shape();
    auto output_padded_shape = input_tensor.padded_shape();

    return std::visit(
        [&](const auto& program_config) -> std::vector<TensorSpec> {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            if constexpr (std::is_same_v<
                              ProgramConfigType,
                              ttnn::operations::normalization::LayerNormShardedMultiCoreProgramConfig>) {
                auto output_shard_spec = this->output_mem_config.shard_spec().value();
                auto input_shard_spec = input_tensor.shard_spec().value();
                if (output_shard_spec != input_shard_spec) {
                    output_padded_shape[3] = output_shard_spec.shape[1] * output_shard_spec.num_cores();
                }
                if (program_config.inplace) {
                    return {input_tensor.tensor_spec()};
                }

                auto mem_config = this->output_mem_config;
                if (!mem_config.shard_spec().has_value()) {
                    mem_config = mem_config.with_shard_spec(input_tensor.shard_spec().value());
                }

                return {ttnn::TensorSpec(
                    output_shape,
                    TensorLayout::fromPaddedShape(
                        this->dtype.value_or(input_tensor.dtype()),
                        PageConfig(Layout::TILE),
                        mem_config,
                        output_shape,
                        output_padded_shape))};
            }
            TT_FATAL(false, "Tensor Spec does not match");
            return {TensorSpec(
                output_shape, TensorLayout(input_tensor.dtype(), PageConfig(Layout::TILE), this->output_mem_config))};
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
                if (program_config.inplace) {
                    return {input_tensors.at(0)};
                }
            }
            auto output_spec = compute_output_specs(input_tensors)[0];
            return {create_device_tensor(output_spec, input_tensors.at(0).device())};
        },
        this->program_config);
}

tt::tt_metal::operation::MeshWorkloadWithCallbacks RMSAllGather::create_mesh_workload(
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    return ccl::create_mesh_workload_from_programs(
        tensor_coords, input_tensors, output_tensors, [&, this](const ttnn::MeshCoordinate& coord) {
            return create_program_at(coord, input_tensors, optional_input_tensors, output_tensors);
        });
}

tt::tt_metal::operation::ProgramWithCallbacks RMSAllGather::create_program_at(
    const ttnn::MeshCoordinate& mesh_coord,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    ttnn::MeshDevice* mesh_device = input_tensors.at(0).mesh_device();
    const auto target_device = mesh_device->get_device(mesh_coord);
    const auto mesh_view = mesh_device->get_view();
    TT_FATAL(
        mesh_view.is_mesh_2d(), "all-gather invoked with cluster_axis API on >2D mesh, which is currently unsupported");
    std::vector<IDevice*> devices = (cluster_axis == 0) ? mesh_view.get_devices_on_column(mesh_coord[1])
                                                        : mesh_view.get_devices_on_row(mesh_coord[0]);

    std::optional<IDevice*> forward_device = std::nullopt;
    std::optional<IDevice*> backward_device = std::nullopt;
    uint32_t device_index = 0;  // Initialize device index
    for (uint32_t i = 0; i < this->ring_size; ++i) {
        if (devices.at(i) == target_device) {
            device_index = i;
            if (i != 0) {
                backward_device = devices.at(i - 1);
            } else if (topology == ttnn::ccl::Topology::Ring) {
                backward_device = devices.at(this->ring_size - 1);
            }
            if (i != this->ring_size - 1) {
                forward_device = devices.at(i + 1);
            } else if (topology == ttnn::ccl::Topology::Ring) {
                forward_device = devices.at(0);
            }
        }
    }

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
                return frmsnorm_multi_core_sharded(
                    a,
                    b,
                    gamma,
                    stats,
                    output_tensor,
                    this->eps,
                    program_config.compute_with_storage_grid_size,
                    program_config.subblock_w,
                    program_config.block_w,
                    this->compute_kernel_config,
                    // New Parameters
                    target_device,
                    forward_device,
                    backward_device,
                    this->num_links,
                    this->ring_size,
                    device_index,
                    this->topology,
                    this->semaphore,
                    this->sub_device_id);
            } else {
                TT_FATAL(false, "Program Config does not match");

                using ProgramConfigType = std::decay_t<decltype(program_config)>;
                uint32_t num_cores_x = 1;
                uint32_t num_cores_y = 1;
                CoreCoord grid_size = CoreCoord(num_cores_x, num_cores_y);

                return frmsnorm_multi_core_sharded(
                    a,
                    b,
                    gamma,
                    stats,
                    output_tensor,
                    this->eps,
                    grid_size,
                    1,
                    1,
                    this->compute_kernel_config,
                    target_device,
                    forward_device,
                    backward_device,
                    this->num_links,
                    this->ring_size,
                    device_index,
                    this->topology,
                    this->semaphore,
                    this->sub_device_id);
            }
        },
        this->program_config);
}

}  // namespace ttnn::operations::fused::normalization

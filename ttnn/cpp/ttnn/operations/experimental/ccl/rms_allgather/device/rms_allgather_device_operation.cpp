// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rms_allgather_device_operation.hpp"

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/device.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::experimental::prim {
void RMSAllGatherDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& a = tensor_args.input;
    const auto& b = tensor_args.residual_input_tensor;
    const auto& gamma = tensor_args.weight;

    // Populated by ttnn::prim::rms_allgather even when the caller passes stats=None (it allocates an
    // op-managed scratch), and it backs the globally-allocated cb_stats circular buffer.
    TT_FATAL(tensor_args.stats.has_value(), "fused_rms_minimal: the stats scratch tensor was not populated.");
    const auto& stats = tensor_args.stats.value();
    // The program factory binds cb_stats onto this buffer and reads its shard spec, so the layout is
    // part of the contract rather than a preference.
    TT_FATAL(
        stats.memory_config().buffer_type() == BufferType::L1,
        "fused_rms_minimal requires the stats buffer in L1 but got buffer type {}.",
        stats.memory_config().buffer_type());
    TT_FATAL(
        stats.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED,
        "fused_rms_minimal requires a width-sharded stats buffer but got memory layout {}.",
        stats.memory_config().memory_layout());
    // One E(x^2) partial tile is gathered per device on the cluster axis, at slot
    // device_index % tiles_per_core, and cb_stats is bound to a single core's bank, so the bound is on
    // the per-core shard width rather than the tensor width. A wider bank only leaves the tail unused.
    const uint32_t stats_tiles_per_core = stats.shard_spec().value().shape[1] / TILE_WIDTH;
    TT_FATAL(
        stats_tiles_per_core >= args.ring_size,
        "fused_rms_minimal requires a stats buffer of at least ring_size ({}) tiles per core but got {}; a "
        "caller-provided buffer must be replicated across the mesh with per-core shard shape (32, ring_size * 32).",
        args.ring_size,
        stats_tiles_per_core);

    {
        const bool fp32_dest_acc_en =
            std::get<2>(get_compute_kernel_config_args(a.device()->arch(), args.compute_kernel_config));
        const auto expected_stats_dtype = fp32_dest_acc_en ? DataType::FLOAT32 : DataType::BFLOAT16;
        TT_FATAL(
            tensor_args.stats->dtype() == expected_stats_dtype,
            "fused_rms_minimal: stats tensor dtype ({}) must match the compute accumulation format; with "
            "fp32_dest_acc_en={} the stats tensor must be {}. Allocate the stats tensor with that dtype (this is "
            "required for correct circular-buffer sizing and the internal all-gather handshake).",
            tensor_args.stats->dtype(),
            fp32_dest_acc_en,
            expected_stats_dtype);
    }

    TT_FATAL(a.padded_shape().rank() == 4, "Input shape must be rank 4");
    TT_FATAL(
        a.logical_shape()[0] == 1 && a.logical_shape()[1] == 1 && a.logical_shape()[2] <= 32 &&
            a.logical_shape()[3] % 32 == 0,
        "Input tensor shape does not meet the requirements set by this OP: input tensor shape must be (1,1,M,N) "
        "where "
        "M <= 32 and N is a multiple of 32");
    TT_FATAL(
        (tt::tt_metal::hal::get_arch_name() != "blackhole") || (a.memory_config().buffer_type() != BufferType::DRAM),
        "This kernel does not support blackhole dram as it does not use an accessor to get the noc address as needed "
        "by the fabric api");

    // Input layout contract (see the fused_rms_minimal docstring). The op is width-sharded in L1 and does not reshard
    // internally, so reject anything that would otherwise dereference an empty shard spec below with a bad optional.
    TT_FATAL(
        a.is_sharded(),
        "fused_rms_minimal requires a width-sharded L1 input; an interleaved input is not supported. Reshard to "
        "width-sharded L1 first.");
    TT_FATAL(
        a.memory_config().buffer_type() == BufferType::L1,
        "fused_rms_minimal requires the input in L1 but got buffer type {}.",
        a.memory_config().buffer_type());
    TT_FATAL(
        a.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED,
        "fused_rms_minimal requires a width-sharded input but got memory layout {}.",
        a.memory_config().memory_layout());
    // The all-gather addresses the input on the same cores the caller built the global_semaphore on, so the input
    // shard grid must be covered by the semaphore's core range.
    // Copy by value: attribute_values() returns a temporary tuple, so a reference into it would dangle.
    const auto semaphore_cores = std::get<0>(args.semaphore.attribute_values());
    TT_FATAL(
        semaphore_cores.contains(a.shard_spec().value().grid),
        "fused_rms_minimal requires the input to be sharded on the grid the global_semaphore was created on; the "
        "input shard grid {} is not contained within the semaphore grid {}.",
        a.shard_spec().value().grid,
        semaphore_cores);

    uint32_t input_width = a.tensor_spec().tile().get_tile_shape()[1];
    uint32_t input_height = a.tensor_spec().tile().get_tile_shape()[0];
    TT_FATAL(
        args.output_mem_config.shard_spec().has_value(),
        "fused_rms_minimal requires a sharded output memory config (width-sharded in L1); an interleaved output "
        "memory config is not supported.");
    TT_FATAL(
        args.output_mem_config.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR,
        "Minimal version requires row major sharding orientation");
    TT_FATAL(
        a.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR,
        "Minimal version requires row major sharding orientation");
    TT_FATAL(a.layout() == Layout::TILE, "Input tensor layout must be TILE but got {}", a.layout());
    TT_FATAL(
        a.dtype() == DataType::FLOAT32 or a.dtype() == DataType::BFLOAT16 or a.dtype() == DataType::BFLOAT8_B,
        "Input tensor dtype must be FLOAT32, BFLOAT16, or BFLOAT8_B but got {}",
        a.dtype());
    TT_FATAL(a.storage_type() == StorageType::DEVICE, "Operands to frmsnorm need to be on device!");
    TT_FATAL(a.buffer() != nullptr, "Operands to frmsnorm need to be allocated in buffers on device!");

    if (b.has_value()) {
        TT_FATAL(b.value().layout() == Layout::TILE, "layout is not tile!");
        TT_FATAL(a.padded_shape() == b.value().padded_shape(), "shape is not same!");
        TT_FATAL(b.value().buffer() != nullptr, "Operands to frmsnorm need to be allocated in buffers on device!");
        TT_FATAL(a.device() == b.value().device(), "device is not same!");
    }
    TT_FATAL(
        gamma.has_value(), "fused_rms_minimal requires a weight (gamma) tensor; passing weight=None is not supported.");
    TT_FATAL(gamma.value().layout() == Layout::ROW_MAJOR, "RMS all gather requires a weight which is row major");

    if (gamma.has_value()) {
        if (gamma.value().layout() == Layout::TILE) {
            TT_FATAL(
                a.padded_shape()[-1] == gamma.value().padded_shape()[-1],
                "{} != {}",
                a.padded_shape()[-1],
                gamma.value().padded_shape()[-1]);
            TT_FATAL(
                gamma.value().buffer() != nullptr, "Operands to frmsnorm need to be allocated in buffers on device!");
            TT_FATAL(a.device() == gamma.value().device(), "Input tensor device must match gamma tensor device");
            TT_FATAL(
                gamma.value().padded_shape()[-2] == input_height,
                "Gamma tensor height ({}) must equal input height ({})",
                gamma.value().padded_shape()[-2],
                input_height);
        } else {
            TT_FATAL(
                gamma.value().layout() == Layout::ROW_MAJOR,
                "Gamma tensor layout must be ROW_MAJOR but got {}",
                gamma.value().layout());
            TT_FATAL(
                (gamma.value().padded_shape()[-1] == input_width &&
                 gamma.value().physical_volume() / input_width == a.padded_shape()[-1] / input_width),
                "Gamma tensor width ({}) must equal input width ({}) and physical volume / width must match",
                gamma.value().padded_shape()[-1],
                input_width);
            TT_FATAL(
                gamma.value().buffer() != nullptr, "Operands to frmsnorm need to be allocated in buffers on device!");
            TT_FATAL(a.device() == gamma.value().device(), "Input tensor device must match gamma tensor device");
            TT_FATAL(
                gamma.value().dtype() == DataType::FLOAT32 or gamma.value().dtype() == DataType::BFLOAT16,
                "Gamma tensor dtype must be FLOAT32 or BFLOAT16 but got {}",
                gamma.value().dtype());
        }
    }

    if (a.is_sharded()) {
        // TODO: Add support for this (should be similar to interleaved)
        TT_FATAL(
            a.memory_config().memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED,
            "Height sharded inputs are not supported.");
        TT_FATAL(
            args.output_mem_config.is_sharded() &&
                args.output_mem_config.memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED,
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

    if (args.inplace) {
        TT_FATAL(args.output_mem_config.is_sharded(), "Output memory config must be sharded for inplace operation");
    }
    TT_FATAL(
        a.memory_config().buffer_type() == args.output_mem_config.buffer_type(),
        "Input tensor buffer type ({}) must match output memory config buffer type ({})",
        a.memory_config().buffer_type(),
        args.output_mem_config.buffer_type());
    TT_FATAL(
        a.memory_config().memory_layout() == args.output_mem_config.memory_layout(),
        "Input tensor memory layout ({}) must match output memory config layout ({})",
        a.memory_config().memory_layout(),
        args.output_mem_config.memory_layout());

    // tensor shape
    const auto& shape = a.padded_shape();
    uint32_t M = a.physical_volume() / shape[-1];
    uint32_t K = shape[-1];

    uint32_t Kt = K / input_width;
    // block
    const auto shard_spec = a.shard_spec().value();
    // check dims
    TT_FATAL(args.block_wt % args.subblock_wt == 0, "block_w must be divisible by subblock_w.");
    TT_FATAL(M % input_height == 0, "M must be divisible by tile height.");
    TT_FATAL(K % input_width == 0, "K must be divisible by tile width.");
    const auto bbox = shard_spec.grid.bounding_box();
    TT_FATAL(
        bbox.end_coord.x - bbox.start_coord.x < args.grid_size.x &&
            bbox.end_coord.y - bbox.start_coord.y < args.grid_size.y,
        "Shard grid bounding box must fit within compute grid size");

    TT_FATAL(M == input_height, "Minimal version assumes (1,1,TILE_HEIGHT,N) shape");
    TT_FATAL(tt::div_up(Kt, shard_spec.num_cores()) == args.block_wt, "block_w must equal to K / num_cores.");
    TT_FATAL(
        a.memory_config().memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED,
        "Input tensor memory layout must not be HEIGHT_SHARDED but got {}",
        a.memory_config().memory_layout());
    if (b.has_value()) {
        TT_FATAL(b.value().is_sharded(), "Tensor B must be sharded");
        TT_FATAL(b.value().shard_spec() == shard_spec, "Tensor B shard spec must match input tensor shard spec");
    }
    TT_FATAL(
        args.block_wt * input_width == shard_spec.shape[1],
        "block_w ({}) * input_width ({}) must equal shard_spec shape[1] ({})",
        args.block_wt,
        input_width,
        shard_spec.shape[1]);
}

tt::tt_metal::TensorSpec RMSAllGatherDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    if (args.inplace) {
        return input_tensor.tensor_spec();
    }

    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output->tensor_spec();
    }

    auto output_shape = input_tensor.logical_shape();
    auto output_padded_shape = input_tensor.padded_shape();

    // This runs before validate_on_program_cache_miss (create_output_tensors is called ahead of validation in
    // device_operation::launch), so guard the dereference here too rather than relying on the check in validate.
    TT_FATAL(
        args.output_mem_config.shard_spec().has_value(),
        "fused_rms_minimal requires a sharded output memory config (width-sharded in L1); an interleaved output "
        "memory config is not supported.");
    auto output_shard_spec = args.output_mem_config.shard_spec().value();
    auto input_shard_spec = input_tensor.shard_spec().value();
    if (output_shard_spec != input_shard_spec) {
        output_padded_shape[3] = output_shard_spec.shape[1] * output_shard_spec.num_cores();
    }

    auto mem_config = args.output_mem_config;

    return tt::tt_metal::TensorSpec(
        output_shape,
        TensorLayout::fromPaddedShape(
            args.dtype.value_or(input_tensor.dtype()),
            PageConfig(Layout::TILE),
            mem_config,
            output_shape,
            output_padded_shape));
}

Tensor RMSAllGatherDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output.value();
    }

    if (args.inplace) {
        return tensor_args.input;
    }
    auto output_spec = compute_output_specs(args, tensor_args);
    return create_device_tensor(output_spec, tensor_args.input.device());
}

ttsl::hash::hash_t RMSAllGatherDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    log_trace(tt::LogOp, "RMSAllGatherDeviceOperation::compute_program_hash is called");

    auto subdevice_id = args.sub_device_id;
    auto* mesh_device = tensor_args.input.device();
    auto sd_id = subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0));
    auto subdevice_core_range_set = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);
    return tt::tt_metal::operation::hash_operation<RMSAllGatherDeviceOperation>(
        args.eps,
        args.output_mem_config,
        args.subblock_wt,
        args.block_wt,
        args.inplace,
        args.grid_size,
        args.compute_kernel_config,
        args.dtype,
        args.topology,
        args.num_links,
        args.ring_size,
        args.cluster_axis,
        args.use_noc1_only,
        subdevice_core_range_set,
        tensor_args);
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

ttnn::experimental::prim::RMSAllGatherDeviceOperation::tensor_return_value_t rms_allgather(
    const Tensor& input_tensor,
    const ttnn::prim::LayerNormProgramConfig& program_config,
    uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const GlobalSemaphore& semaphore,
    const std::optional<ttnn::Tensor>& persistent_output_tensor,
    std::optional<size_t> num_preferred_links,
    ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    std::optional<const DataType> dtype,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<const ttnn::Tensor>& residual_input_tensor,
    float epsilon,
    const std::optional<const ttnn::Tensor>& weight,
    const std::optional<const ttnn::Tensor>& stats,
    bool use_noc1_only) {
    using OperationType = ttnn::experimental::prim::RMSAllGatherDeviceOperation;
    auto arch = is_device_tensor(input_tensor) ? input_tensor.device()->arch() : ttnn::GetDefaultDevice()->arch();
    auto kernel_config_val =
        init_device_compute_kernel_config(arch, compute_kernel_config, tt::tt_metal::MathFidelity::HiFi4, true, false, false);
    const auto& mesh_view = mesh_device.get_view();
    std::size_t num_devices = (cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();

    tt::tt_fabric::Topology topology_ = ::ttnn::ccl::get_usable_topology(input_tensor, topology, cluster_axis);

    auto [subblock_wt, block_wt, inplace, grid_size] = std::visit(
        [](const auto& config) -> std::tuple<uint32_t, uint32_t, bool, CoreCoord> {
            using T = std::decay_t<decltype(config)>;
            if constexpr (std::is_same_v<T, ttnn::prim::LayerNormShardedMultiCoreProgramConfig>) {
                return {
                    static_cast<uint32_t>(config.subblock_w),
                    static_cast<uint32_t>(config.block_w),
                    config.inplace,
                    config.compute_with_storage_grid_size};
            } else {
                TT_FATAL(false, "RMSAllGather only supports LayerNormShardedMultiCoreProgramConfig");
                return {0, 0, false, CoreCoord{0, 0}};
            }
        },
        program_config);

    auto operation_attributes = OperationType::operation_attributes_t(
        epsilon,
        memory_config.value_or(input_tensor.memory_config()),
        subblock_wt,
        block_wt,
        inplace,
        grid_size,
        kernel_config_val,
        dtype,
        topology_,
        num_preferred_links.value_or(1),
        num_devices,
        semaphore,
        subdevice_id,
        cluster_axis,
        use_noc1_only);

    // Allocate the stats scratch when the caller omits it. Each device fabric-writes its E(x^2) partial
    // into one tile of this buffer, so it must be width-sharded L1 on the first core of the input shard
    // grid, where the program factory places cb_stats. An op-managed transient is trace-native.
    std::optional<const ttnn::Tensor> stats_scratch = stats;
    if (!stats_scratch.has_value()) {
        auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
            get_compute_kernel_config_args(arch, kernel_config_val);
        // cb_stats is mapped onto this buffer and its page size follows cb_data_format, so the tile
        // size must match.
        const DataType stats_dtype = fp32_dest_acc_en ? DataType::FLOAT32 : DataType::BFLOAT16;
        const uint32_t stats_width = static_cast<uint32_t>(num_devices) * TILE_WIDTH;
        TT_FATAL(
            input_tensor.shard_spec().has_value(),
            "fused_rms_minimal requires a sharded input to place the internal stats scratch.");
        const CoreCoord stats_core = input_tensor.shard_spec().value().grid.bounding_box().start_coord;
        const ShardSpec stats_shard_spec(
            CoreRangeSet(CoreRange(stats_core, stats_core)), {TILE_HEIGHT, stats_width}, ShardOrientation::ROW_MAJOR);
        const MemoryConfig stats_mem_config(TensorMemoryLayout::WIDTH_SHARDED, BufferType::L1, stats_shard_spec);
        const TensorSpec stats_spec(
            ttnn::Shape({1, 1, TILE_HEIGHT, stats_width}),
            TensorLayout(stats_dtype, PageConfig(Layout::TILE), stats_mem_config));
        // emplace, not assign: std::optional<const Tensor> has no copy-assignment operator.
        stats_scratch.emplace(create_device_tensor(stats_spec, input_tensor.device()));
    }

    auto tensor_args = OperationType::tensor_args_t{
        .input = input_tensor,
        .residual_input_tensor = residual_input_tensor,
        .weight = weight,
        .stats = stats_scratch,
        .preallocated_output = persistent_output_tensor};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim

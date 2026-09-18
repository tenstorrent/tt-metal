// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/ccl/matmul_reduce_scatter_sp_async/device/matmul_reduce_scatter_sp_async_device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>

#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_op_device_operation.hpp"
#include "ttnn/operations/experimental/ccl/sp_matmul_fusion_common/sp_matmul_fusion_common.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation.hpp"
#include "ttnn/operations/matmul/device/utilities/matmul_utilities.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

namespace {

uint32_t weight_k(const Tensor& weight, bool transpose_b) {
    return transpose_b ? weight.logical_shape()[-1] : weight.logical_shape()[-2];
}

uint32_t weight_n(const Tensor& weight, bool transpose_b) {
    return transpose_b ? weight.logical_shape()[-2] : weight.logical_shape()[-1];
}

// [B,1,S,N] tiled, in the matmul output memory config: the RS input.
TensorSpec mm_partial_spec(
    const MatmulReduceScatterSpAsyncParams& args, const MatmulReduceScatterSpAsyncInputs& tensor_args) {
    const auto& input = tensor_args.input;
    const auto& in_shape = input.logical_shape();
    const Shape out_shape(
        {in_shape[0], in_shape[1], in_shape[2], weight_n(tensor_args.weight, args.matmul_params.transpose_b)});
    return TensorSpec(
        out_shape,
        TensorLayout(
            args.matmul_params.output_dtype.value_or(input.dtype()),
            input.tensor_spec().page_config(),
            args.matmul_params.output_mem_config));
}

ReduceScatterMinimalAsyncInputs rs_inputs(const Tensor& mm_partial) {
    return ReduceScatterMinimalAsyncInputs{mm_partial, std::nullopt, std::nullopt, std::nullopt};
}

}  // namespace

void MatmulReduceScatterSpAsyncDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    const auto& weight = tensor_args.weight;
    const auto& rs = args.reduce_scatter_params;
    const auto& mm = args.matmul_params;

    TT_FATAL(
        input.storage_type() == StorageType::DEVICE && weight.storage_type() == StorageType::DEVICE,
        "matmul_reduce_scatter_sp_async: input and weight must be on device");
    TT_FATAL(
        input.device() == weight.device(), "matmul_reduce_scatter_sp_async: input and weight must share a mesh device");
    TT_FATAL(
        input.layout() == Layout::TILE && weight.layout() == Layout::TILE,
        "matmul_reduce_scatter_sp_async: input and weight must be TILE layout");
    TT_FATAL(
        !input.is_sharded() && !weight.is_sharded() && input.memory_config().buffer_type() == BufferType::DRAM &&
            weight.memory_config().buffer_type() == BufferType::DRAM,
        "matmul_reduce_scatter_sp_async: input and weight must be interleaved DRAM tensors");
    TT_FATAL(
        input.dtype() == DataType::BFLOAT16 || input.dtype() == DataType::FLOAT32,
        "matmul_reduce_scatter_sp_async: input dtype must be bfloat16 or float32, got {}",
        input.dtype());
    TT_FATAL(
        weight.dtype() == DataType::BFLOAT16 || weight.dtype() == DataType::FLOAT32 ||
            weight.dtype() == DataType::BFLOAT8_B,
        "matmul_reduce_scatter_sp_async: weight dtype must be bfloat16, float32 or bfloat8_b, got {}",
        weight.dtype());

    const auto& in_shape = input.logical_shape();
    const auto& w_shape = weight.logical_shape();
    TT_FATAL(in_shape.rank() == 4, "matmul_reduce_scatter_sp_async: input must be rank 4 [B,1,S,K], got {}", in_shape);
    TT_FATAL(in_shape[1] == 1, "matmul_reduce_scatter_sp_async: input dim 1 must be 1 ([B,1,S,K]), got {}", in_shape);
    TT_FATAL(
        w_shape.rank() == 4 && w_shape[0] == 1 && w_shape[1] == 1,
        "matmul_reduce_scatter_sp_async: weight must be [1,1,K,N] (or [1,1,N,K] with transpose_b), got {}",
        w_shape);
    TT_FATAL(
        input.padded_shape() == in_shape,
        "matmul_reduce_scatter_sp_async: input must be tile aligned (S and K multiples of 32), got {} padded to {}",
        in_shape,
        input.padded_shape());
    TT_FATAL(
        weight.padded_shape() == w_shape,
        "matmul_reduce_scatter_sp_async: weight must be tile aligned, got {} padded to {}",
        w_shape,
        weight.padded_shape());

    TT_FATAL(rs.dim == 2, "matmul_reduce_scatter_sp_async scatters on dim 2 only, got dim {}", rs.dim);
    TT_FATAL(
        rs.topology == ttnn::ccl::Topology::Ring || rs.topology == ttnn::ccl::Topology::Linear,
        "matmul_reduce_scatter_sp_async: topology must be Ring or Linear, got {}",
        rs.topology);
    TT_FATAL(
        rs.ring_size >= 2,
        "matmul_reduce_scatter_sp_async: needs >= 2 devices along cluster_axis, got {}",
        rs.ring_size);
    TT_FATAL(
        rs.topology != ttnn::ccl::Topology::Ring || rs.ring_size % 2 == 0,
        "matmul_reduce_scatter_sp_async: the Ring reduce-scatter needs an even number of devices, got {}",
        rs.ring_size);
    TT_FATAL(
        in_shape[2] % (tt::constants::TILE_HEIGHT * rs.ring_size) == 0,
        "matmul_reduce_scatter_sp_async: S ({}) must be a multiple of 32 x num_devices ({})",
        in_shape[2],
        tt::constants::TILE_HEIGHT * rs.ring_size);
    TT_FATAL(
        weight_k(weight, mm.transpose_b) == in_shape[3],
        "matmul_reduce_scatter_sp_async: weight K ({}) does not match input K ({}) (transpose_b={}, weight {})",
        weight_k(weight, mm.transpose_b),
        in_shape[3],
        mm.transpose_b,
        w_shape);
    TT_FATAL(
        weight_n(weight, mm.transpose_b) % tt::constants::TILE_WIDTH == 0,
        "matmul_reduce_scatter_sp_async: N ({}) must be a multiple of 32",
        weight_n(weight, mm.transpose_b));
    TT_FATAL(
        rs.semaphore.size() == 3,
        "matmul_reduce_scatter_sp_async: expected 3 global semaphores (as reduce_scatter_minimal_async), got {}",
        rs.semaphore.size());
    TT_FATAL(!rs.using_persistent_buffers, "matmul_reduce_scatter_sp_async allocates all of its buffers itself");
    TT_FATAL(rs.cluster_axis.has_value(), "matmul_reduce_scatter_sp_async: cluster_axis is required");
    TT_FATAL(
        *rs.cluster_axis < input.device()->shape().dims(),
        "matmul_reduce_scatter_sp_async: cluster_axis {} out of range for mesh shape {}",
        *rs.cluster_axis,
        input.device()->shape());
    TT_FATAL(
        rs.num_workers_per_link.has_value(), "matmul_reduce_scatter_sp_async: num_workers_per_link must be resolved");
    TT_FATAL(
        mm.bcast_batch.value_or(false), "matmul_reduce_scatter_sp_async: the weight is broadcast over sub-batches");
    TT_FATAL(!mm.transpose_a, "matmul_reduce_scatter_sp_async: transpose_a is not supported");
    TT_FATAL(
        in_shape[0] * rs.ring_size < 256,
        "matmul_reduce_scatter_sp_async: B x num_devices ({}) must be < 256 (8-bit sub-batch indices in the matmul "
        "slice schedule)",
        in_shape[0] * rs.ring_size);

    // Core grid split: CCL workers in the bottom ccl_core_rows rows, matmul above.
    const auto grid = input.device()->compute_with_storage_grid_size();
    TT_FATAL(
        args.ccl_core_rows >= 1 && args.ccl_core_rows < grid.y,
        "matmul_reduce_scatter_sp_async: ccl_core_rows ({}) must be in [1, {})",
        args.ccl_core_rows,
        grid.y);
    const uint32_t rs_cores = sp_reduce_scatter_core_count(rs.topology, rs.num_links, *rs.num_workers_per_link);
    TT_FATAL(
        rs_cores <= args.ccl_core_rows * grid.x,
        "matmul_reduce_scatter_sp_async: the reduce-scatter needs {} cores ({} links x 2 directions x ({} workers + "
        "mux)) but only {} rows x {} columns = {} are reserved; raise ccl_core_rows or lower num_workers_per_link / "
        "num_links",
        rs_cores,
        rs.num_links,
        *rs.num_workers_per_link,
        args.ccl_core_rows,
        grid.x,
        args.ccl_core_rows * grid.x);

    // User program config override: must be the 2D mcast config, one sub-batch per batch iteration, within the
    // matmul rectangle.
    if (mm.program_config.has_value()) {
        const auto mm_grid = sp_matmul_core_grid(input, args.ccl_core_rows);
        std::visit(
            [&](const auto& config) {
                using ProgramConfigType = std::decay_t<decltype(config)>;
                if constexpr (std::is_same_v<
                                  ProgramConfigType,
                                  operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig>) {
                    TT_FATAL(
                        !config.fuse_batch,
                        "matmul_reduce_scatter_sp_async: program_config.fuse_batch must be false (one sub-batch per "
                        "iteration)");
                    TT_FATAL(
                        !config.transpose_mcast,
                        "matmul_reduce_scatter_sp_async: program_config.transpose_mcast is not supported");
                    TT_FATAL(
                        config.compute_with_storage_grid_size.x <= mm_grid.x &&
                            config.compute_with_storage_grid_size.y <= mm_grid.y,
                        "matmul_reduce_scatter_sp_async: program_config grid {}x{} exceeds the matmul rectangle {}x{} "
                        "(grid minus ccl_core_rows={})",
                        config.compute_with_storage_grid_size.x,
                        config.compute_with_storage_grid_size.y,
                        mm_grid.x,
                        mm_grid.y,
                        args.ccl_core_rows);
                } else {
                    TT_THROW(
                        "matmul_reduce_scatter_sp_async: program_config must be "
                        "MatmulMultiCoreReuseMultiCastProgramConfig");
                }
            },
            mm.program_config.value());
    }

    // Matmul validation on the sub-batched views with the (derived or user) program config.
    const auto mm_params = resolve_sp_matmul_params(args, tensor_args);
    const Tensor in0_view = ttnn::experimental::ccl::sub_batched_view(input, rs.ring_size);
    ttnn::prim::MatmulDeviceOperation::validate_on_program_cache_miss(
        mm_params,
        ttnn::prim::MatmulInputs{
            .input_tensors = {in0_view, weight},
            .optional_input_tensors = {std::nullopt},
            .optional_output_tensors = {}});
}

void MatmulReduceScatterSpAsyncDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    TT_FATAL(tensor_args.input.storage_type() == StorageType::DEVICE, "Input tensor must be on device");
    TT_FATAL(tensor_args.input.buffer() != nullptr, "Input tensor must have a buffer");
    TT_FATAL(tensor_args.weight.buffer() != nullptr, "Weight tensor must have a buffer");
}

MatmulReduceScatterSpAsyncDeviceOperation::spec_return_value_t
MatmulReduceScatterSpAsyncDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto partial_spec = mm_partial_spec(args, tensor_args);
    // The reduce-scatter specs are derived from its input TENSOR (the Ring staging layout needs the buffer's page
    // size / page count), so size them off a device tensor of the mm-partial spec. Not on the launch path:
    // ttnn::device_operation::launch calls create_output_tensors below, so this costs nothing in steady state.
    const Tensor probe = create_device_tensor(partial_spec, tensor_args.input.device());
    auto rs_specs =
        ReduceScatterMinimalAsyncDeviceOperation::compute_output_specs(args.reduce_scatter_params, rs_inputs(probe));
    spec_return_value_t specs{partial_spec};
    specs.insert(specs.end(), rs_specs.begin(), rs_specs.end());
    return specs;
}

MatmulReduceScatterSpAsyncDeviceOperation::tensor_return_value_t
MatmulReduceScatterSpAsyncDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    Tensor mm_partial = create_device_tensor(mm_partial_spec(args, tensor_args), tensor_args.input.device());
    // {intermediate, output[, penult intermediate]} exactly as the standalone op allocates them.
    auto rs_tensors = ReduceScatterMinimalAsyncDeviceOperation::create_output_tensors(
        args.reduce_scatter_params, rs_inputs(mm_partial));
    tensor_return_value_t out{std::move(mm_partial)};
    out.insert(out.end(), rs_tensors.begin(), rs_tensors.end());
    TT_FATAL(
        out.size() == kRsOutputIdx + 1 || out.size() == kRsPenultIdx + 1,
        "matmul_reduce_scatter_sp_async: unexpected number of reduce-scatter buffers ({})",
        rs_tensors.size());
    return out;
}

ttsl::hash::hash_t MatmulReduceScatterSpAsyncDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& rs = args.reduce_scatter_params;
    return tt::tt_metal::operation::hash_operation<MatmulReduceScatterSpAsyncDeviceOperation>(
        rs.dim,
        rs.num_links,
        rs.ring_size,
        rs.output_mem_config,
        rs.optional_intermediate_mem_config,
        rs.topology,
        rs.cluster_axis,
        rs.barrier_semaphore.has_value(),
        rs.sub_device_id,
        rs.num_workers_per_link,
        rs.compute_kernel_config,
        args.matmul_params,
        args.ccl_core_rows,
        args.debug_serialize_reduce_scatter,
        tensor_args.input,
        tensor_args.weight);
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::vector<Tensor> matmul_reduce_scatter_sp_async(
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    const uint32_t cluster_axis,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    const bool transpose_b,
    const uint32_t num_links,
    const uint32_t ring_size,
    const ttnn::ccl::Topology topology,
    const uint32_t ccl_core_rows,
    const uint32_t num_workers_per_link,
    const std::optional<MemoryConfig>& memory_config,
    const tt::tt_metal::DataType output_dtype,
    const ttnn::DeviceComputeKernelConfig& matmul_compute_kernel_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& reduce_scatter_compute_kernel_config,
    const std::optional<const operations::matmul::MatmulProgramConfig>& program_config,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    const bool debug_serialize_reduce_scatter) {
    using OperationType = ttnn::experimental::prim::MatmulReduceScatterSpAsyncDeviceOperation;

    const MemoryConfig out_mem_config = memory_config.value_or(input_tensor.memory_config());

    ttnn::experimental::prim::ReduceScatterMinimalAsyncParams reduce_scatter_params{
        .dim = 2,
        .num_links = num_links,
        .ring_size = ring_size,
        .output_mem_config = out_mem_config,
        .optional_intermediate_mem_config = std::nullopt,
        .topology = topology,
        .semaphore = multi_device_global_semaphore,
        .barrier_semaphore = barrier_semaphore,
        .using_persistent_buffers = false,
        .sub_device_id = sub_device_id,
        .cluster_axis = cluster_axis,
        .chunks_per_sync = std::nullopt,
        .num_workers_per_link = num_workers_per_link,
        .num_buffers_per_channel = std::nullopt,
        .compute_kernel_config = reduce_scatter_compute_kernel_config,
    };

    // output_tile is what create_matmul_attributes would derive; the matmul validation dereferences it.
    const auto in0_tile = operations::matmul::utilities::get_matmul_tile(input_tensor, /*transpose=*/false);
    const auto in1_tile = operations::matmul::utilities::get_matmul_tile(weight_tensor, transpose_b);
    const tt::tt_metal::Tile output_tile =
        operations::matmul::utilities::get_output_tile(out_mem_config, in0_tile, in1_tile, std::nullopt, std::nullopt);

    ttnn::prim::MatmulParams matmul_params{
        .program_config = program_config,
        .bcast_batch = true,
        .output_mem_config = out_mem_config,
        .output_dtype = output_dtype,
        .compute_kernel_config = matmul_compute_kernel_config,
        .untilize_out = false,
        .user_core_coord = std::nullopt,
        .user_fused_activation = std::nullopt,
        .user_run_batched = false,
        .transpose_a = false,
        .transpose_b = transpose_b,
        .output_tile = output_tile,
        .global_cb = std::nullopt,
    };

    auto operation_attributes = OperationType::operation_attributes_t{
        .reduce_scatter_params = std::move(reduce_scatter_params),
        .matmul_params = std::move(matmul_params),
        .ccl_core_rows = ccl_core_rows,
        .debug_serialize_reduce_scatter = debug_serialize_reduce_scatter,
    };
    auto tensor_args = OperationType::tensor_args_t{.input = input_tensor, .weight = weight_tensor};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim

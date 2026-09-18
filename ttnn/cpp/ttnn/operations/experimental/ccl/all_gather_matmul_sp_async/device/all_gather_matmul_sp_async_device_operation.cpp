// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/ccl/all_gather_matmul_sp_async/device/all_gather_matmul_sp_async_device_operation.hpp"

#include <tt-metalium/core_coord.hpp>

#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_device_operation.hpp"
#include "ttnn/operations/experimental/ccl/sp_matmul_fusion_common/sp_matmul_fusion_common.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config_types.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation.hpp"
#include "ttnn/operations/matmul/device/utilities/matmul_utilities.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::experimental::prim {

namespace {

uint32_t weight_n(const Tensor& weight, bool transpose_b) {
    const auto& shape = weight.logical_shape();
    return transpose_b ? shape[-2] : shape[-1];
}

uint32_t weight_k(const Tensor& weight, bool transpose_b) {
    const auto& shape = weight.logical_shape();
    return transpose_b ? shape[-1] : shape[-2];
}

}  // namespace

uint32_t sp_default_all_gather_workers(
    const Tensor& input,
    const uint32_t ring_size,
    const ttnn::ccl::Topology topology,
    const uint32_t num_links,
    const uint32_t ccl_core_rows) {
    const auto grid = input.device()->compute_with_storage_grid_size();
    const uint32_t budget = ccl_core_rows * grid.x;
    // Same candidate list as default_workers(): the gathered output is ring_size x the input; per link and (for a
    // ring) per direction; > 256 KiB -> try 4 workers, <= 4 KiB -> 1 worker, else 2.
    const double output_bytes = static_cast<double>(input.buffer()->size()) * ring_size;
    const double data_moved_per_link_bytes =
        output_bytes * (ring_size - 1) / ring_size / num_links / (topology == ttnn::ccl::Topology::Ring ? 2 : 1);
    constexpr double DATA_THRESHOLD = 256.0 * 1024;
    constexpr double SINGLE_PACKET_THRESHOLD = 4.0 * 1024;
    std::vector<uint32_t> candidates;
    if (data_moved_per_link_bytes > DATA_THRESHOLD) {
        candidates = {4, 2, 1};
    } else if (data_moved_per_link_bytes <= SINGLE_PACKET_THRESHOLD) {
        candidates = {1};
    } else {
        candidates = {2, 1};
    }
    for (uint32_t workers : candidates) {
        const uint32_t mux = workers > 1 ? 1 : 0;
        if (num_links * 2 * (workers + mux) <= budget) {
            return workers;
        }
    }
    // 1 worker per direction (no mux core) is the minimum the builder supports.
    TT_FATAL(
        num_links * 2 <= budget,
        "all_gather_matmul_sp_async: {} link(s) need at least {} cores but the {} bottom row(s) only have {}",
        num_links,
        num_links * 2,
        ccl_core_rows,
        budget);
    return 1;
}

void AllGatherMatmulSpAsyncDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void AllGatherMatmulSpAsyncDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const Tensor& input = tensor_args.input;
    const Tensor& weight = tensor_args.weight;
    const auto& ag = args.all_gather;
    const bool transpose_b = args.matmul.transpose_b;

    AllGatherAsyncDeviceOperation::validate_on_program_cache_miss(ag, args.all_gather_tensor_args);
    TT_FATAL(ag.dim == 2, "all_gather_matmul_sp_async gathers dim 2, got dim {}", ag.dim);
    TT_FATAL(ag.cluster_axis.has_value(), "all_gather_matmul_sp_async needs an explicit cluster_axis");
    TT_FATAL(ag.ring_size >= 2, "all_gather_matmul_sp_async needs >= 2 devices along cluster_axis");
    TT_FATAL(
        ag.topology == ttnn::ccl::Topology::Ring || ag.topology == ttnn::ccl::Topology::Linear,
        "all_gather_matmul_sp_async supports Ring and Linear topologies");
    TT_FATAL(ag.num_workers_per_link.has_value(), "num_workers_per_link must be resolved");

    TT_FATAL(input.logical_shape().rank() == 4, "input must be [B,1,S/T,K], got {}", input.logical_shape());
    TT_FATAL(input.logical_shape()[1] == 1, "input must be [B,1,S/T,K], got {}", input.logical_shape());
    TT_FATAL(input.layout() == Layout::TILE, "input must be TILE layout");
    TT_FATAL(
        !input.is_sharded() && input.memory_config().buffer_type() == BufferType::DRAM,
        "input must be DRAM interleaved (the matmul reads it through the gathered tensor's accessor config)");
    TT_FATAL(
        !ag.output_mem_config.is_sharded() && ag.output_mem_config.buffer_type() == BufferType::DRAM,
        "outputs must be DRAM interleaved");
    TT_FATAL(!args.matmul.output_mem_config.is_sharded(), "the matmul output must be interleaved (SP slice schedule)");
    const uint32_t tile_h = input.tensor_spec().tile().get_height();
    TT_FATAL(
        input.logical_shape()[2] % tile_h == 0 && input.padded_shape()[2] == input.logical_shape()[2],
        "input S/T ({}) must be a multiple of the tile height ({})",
        input.logical_shape()[2],
        tile_h);
    TT_FATAL(
        weight.logical_shape().rank() == 4 && weight.logical_shape()[0] == 1 && weight.logical_shape()[1] == 1,
        "weight must be [1,1,K,N] (or [1,1,N,K] with transpose_b), got {}",
        weight.logical_shape());
    TT_FATAL(weight.layout() == Layout::TILE, "weight must be TILE layout");
    TT_FATAL(
        weight_k(weight, transpose_b) == input.logical_shape()[3],
        "K mismatch: input K = {}, weight K = {} (transpose_b={})",
        input.logical_shape()[3],
        weight_k(weight, transpose_b),
        transpose_b);
    if (tensor_args.bias.has_value()) {
        const auto& bshape = tensor_args.bias->logical_shape();
        TT_FATAL(
            bshape[-1] == weight_n(weight, transpose_b) && (bshape.rank() < 2 || bshape[-2] == 1),
            "bias must be row-broadcastable [1,...,1,N], got {}",
            bshape);
    }

    TT_FATAL(args.matmul.program_config.has_value(), "matmul program config not populated");
    TT_FATAL(
        std::holds_alternative<operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig>(
            args.matmul.program_config.value()),
        "all_gather_matmul_sp_async needs a MatmulMultiCoreReuseMultiCastProgramConfig");
    const auto& cfg =
        std::get<operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig>(args.matmul.program_config.value());
    TT_FATAL(!cfg.fuse_batch, "the SP matmul program config must have fuse_batch=false");
    TT_FATAL(
        cfg.compute_with_storage_grid_size.x <= args.matmul_grid.x &&
            cfg.compute_with_storage_grid_size.y <= args.matmul_grid.y,
        "matmul grid {}x{} does not fit the matmul region {}x{} (device grid minus the {} bottom all-gather rows)",
        cfg.compute_with_storage_grid_size.x,
        cfg.compute_with_storage_grid_size.y,
        args.matmul_grid.x,
        args.matmul_grid.y,
        args.ccl_core_rows);
    TT_FATAL(!args.matmul.transpose_a, "transpose_a is not supported");
    TT_FATAL(args.matmul.bcast_batch.value_or(false), "the weight must be broadcast over the batch");
    TT_FATAL(
        args.matmul.compute_kernel_config.has_value() && args.matmul.output_dtype.has_value() &&
            args.matmul.output_tile.has_value(),
        "matmul attributes not fully populated");
}

AllGatherMatmulSpAsyncDeviceOperation::spec_return_value_t AllGatherMatmulSpAsyncDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto gathered_spec =
        AllGatherAsyncDeviceOperation::compute_output_specs(args.all_gather, args.all_gather_tensor_args);
    const auto& g = gathered_spec.logical_shape();
    const ttnn::Shape mm_shape({g[0], g[1], g[2], weight_n(tensor_args.weight, args.matmul.transpose_b)});
    const tt::tt_metal::TensorSpec mm_spec(
        mm_shape,
        tt::tt_metal::TensorLayout(
            args.matmul.output_dtype.value(),
            tt::tt_metal::PageConfig(Layout::TILE, args.matmul.output_tile.value()),
            args.matmul.output_mem_config));
    return {gathered_spec, mm_spec};
}

AllGatherMatmulSpAsyncDeviceOperation::tensor_return_value_t
AllGatherMatmulSpAsyncDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    Tensor gathered =
        AllGatherAsyncDeviceOperation::create_output_tensors(args.all_gather, args.all_gather_tensor_args);
    const auto specs = compute_output_specs(args, tensor_args);
    Tensor mm = ttnn::create_device_tensor(specs.at(1), tensor_args.input.device());
    return {gathered, mm};
}

ttsl::hash::hash_t AllGatherMatmulSpAsyncDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& ag = args.all_gather;
    auto* mesh_device = tensor_args.input.device();
    auto sd_id = ag.sub_device_id.value_or(mesh_device->get_sub_device_ids().at(0));
    auto subdevice_core_range_set = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);
    return tt::tt_metal::operation::hash_operation<AllGatherMatmulSpAsyncDeviceOperation>(
        ag.dim,
        ag.num_links,
        ag.ring_size,
        ag.output_mem_config,
        ag.topology,
        ag.cluster_axis,
        ag.barrier_semaphore.has_value(),
        ag.num_workers_per_link,
        ag.chunks_per_sync,
        ag.num_buffers_per_channel,
        args.matmul,
        args.ccl_core_rows,
        args.all_gather_core_grid_offset,
        args.matmul_grid,
        args.debug_serialize_ag,
        args.ag_signal_on_receive,
        args.in1_resident,
        subdevice_core_range_set,
        tensor_args.input,
        tensor_args.weight,
        tensor_args.bias);
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

ttnn::experimental::prim::AllGatherMatmulSpAsyncDeviceOperation::tensor_return_value_t all_gather_matmul_sp_async(
    const Tensor& input,
    const Tensor& weight,
    const uint32_t cluster_axis,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    const bool transpose_b,
    const std::optional<const Tensor>& bias,
    const uint32_t num_links,
    const ttnn::ccl::Topology topology,
    const uint32_t ccl_core_rows,
    const uint32_t num_workers_per_link,
    const std::optional<MemoryConfig>& memory_config,
    const DataType output_dtype,
    const DeviceComputeKernelConfig& compute_kernel_config,
    const std::optional<const operations::matmul::MatmulProgramConfig>& program_config,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    const bool debug_serialize_ag,
    const bool ag_signal_on_receive,
    const bool in1_resident) {
    using OperationType = ttnn::experimental::prim::AllGatherMatmulSpAsyncDeviceOperation;
    auto* mesh_device = input.device();
    TT_FATAL(mesh_device != nullptr, "all_gather_matmul_sp_async: input must be on a mesh device");

    // Core grid split: bottom ccl_core_rows rows for the all-gather workers, the rest for the matmul at (0,0).
    const auto grid = mesh_device->compute_with_storage_grid_size();
    TT_FATAL(
        ccl_core_rows >= 1 && ccl_core_rows < grid.y,
        "ccl_core_rows ({}) must be in [1, grid.y-1] (grid.y={})",
        ccl_core_rows,
        grid.y);
    const CoreCoord matmul_grid(grid.x, grid.y - ccl_core_rows);
    const CoreCoord all_gather_core_grid_offset(0, grid.y - ccl_core_rows);
    {
        const uint32_t mux = num_workers_per_link > 1 ? 1 : 0;
        const uint32_t ag_cores = num_links * 2 * (num_workers_per_link + mux);
        TT_FATAL(
            ag_cores <= ccl_core_rows * grid.x,
            "all_gather_matmul_sp_async: {} link(s) x {} worker(s) need {} cores, the {} bottom row(s) have {}",
            num_links,
            num_workers_per_link,
            ag_cores,
            ccl_core_rows,
            ccl_core_rows * grid.x);
    }

    const MemoryConfig out_mem_config = memory_config.value_or(input.memory_config());

    /* All gather (dim 2 along cluster_axis) */
    const auto [all_gather_attrs, all_gather_tensor_args] =
        ttnn::experimental::prim::all_gather_async_build_operation_args(
            input,
            /*persistent_output_buffer=*/std::nullopt,
            /*dim=*/2,
            multi_device_global_semaphore,
            num_links,
            out_mem_config,
            topology,
            sub_device_id,
            cluster_axis,
            /*use_optimal_ccl_for_llama=*/false,
            /*use_all_gather_async_llama_sharded=*/false,
            /*use_all_gather_async_via_broadcast=*/false,
            barrier_semaphore,
            /*chunks_per_sync=*/std::nullopt,
            num_workers_per_link,
            /*num_buffers_per_channel=*/std::nullopt,
            /*reverse_order=*/false,
            /*sub_core_grid=*/std::nullopt,
            /*optional_mesh_device=*/nullptr);

    /* Matmul on the [B*T,1,S/T,K] view. The input has the same per-sub-batch Mt and Kt as that view, so the
       derived program config can be computed from it directly. */
    operations::matmul::MatmulProgramConfig resolved_program_config =
        program_config.has_value()
            ? operations::matmul::MatmulProgramConfig(program_config.value())
            : operations::matmul::MatmulProgramConfig(ttnn::experimental::ccl::sp_matmul_program_config(
                  input, weight, matmul_grid, transpose_b, compute_kernel_config));
    operations::matmul::normalize_program_config(resolved_program_config, grid);

    ttnn::prim::MatmulParams matmul_params;
    matmul_params.program_config = resolved_program_config;
    matmul_params.output_mem_config = out_mem_config;
    matmul_params.output_dtype = output_dtype;
    matmul_params.compute_kernel_config = compute_kernel_config;
    matmul_params.transpose_b = transpose_b;
    // create_matmul_attributes only needs in0 for dtype/tile/broadcast decisions; the sequence-sharded input
    // stands in for the gathered tensor (same dtype, tile and batch dims).
    auto matmul_struct = ttnn::prim::create_matmul_attributes(input, weight, matmul_params, {});

    auto operation_attributes = OperationType::operation_attributes_t{
        all_gather_attrs,
        all_gather_tensor_args,
        matmul_struct,
        ccl_core_rows,
        all_gather_core_grid_offset,
        matmul_grid,
        debug_serialize_ag,
        ag_signal_on_receive,
        in1_resident,
    };
    auto tensor_args = OperationType::tensor_args_t{.input = input, .weight = weight, .bias = bias};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim

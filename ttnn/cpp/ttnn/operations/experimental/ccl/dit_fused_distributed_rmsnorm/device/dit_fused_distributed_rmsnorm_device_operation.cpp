// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dit_fused_distributed_rmsnorm_device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>

#include "ttnn/device.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::experimental::prim {

void DitFusedDistributedRmsnormDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    const auto& weight = tensor_args.weight;
    const auto& bias = tensor_args.bias;
    const auto& trans_mat = tensor_args.transformation_mat;
    const auto& rope_cos = tensor_args.rope_cos;
    const auto& rope_sin = tensor_args.rope_sin;

    // Welford LayerNorm (Phases 2-3): whole-row, bf16 input, no RoPE. TP=1 reduces
    // locally; TP>1 gathers per-shard Welford (mean, var) partials over the fabric
    // ring and merges them (parallel Welford). per_head_norm, fp32 input, and
    // RoPE-fused LayerNorm are later phases.
    if (args.norm_type == ttnn::experimental::DitFusedNormType::LAYERNORM) {
        TT_FATAL(!args.per_head_norm, "LayerNorm does not support per_head_norm yet");
        TT_FATAL(
            input.dtype() == DataType::BFLOAT16, "LayerNorm currently requires BFLOAT16 input; got {}", input.dtype());
        TT_FATAL(
            !trans_mat.has_value() && !rope_cos.has_value() && !rope_sin.has_value(),
            "LayerNorm does not support fused RoPE yet");
        // Optional reciprocal LUT: when provided, the writer reads it into a CB and the
        // Welford LLK does an array load instead of a soft-float 1/(N+1) per sample.
        // Absent -> the kernel uses the runtime-division fallback (still correct).
        if (tensor_args.reciprocals.has_value()) {
            const auto& recip = tensor_args.reciprocals.value();
            TT_FATAL(recip.dtype() == DataType::FLOAT32, "reciprocals must be FLOAT32, got {}", recip.dtype());
            TT_FATAL(recip.layout() == Layout::ROW_MAJOR, "reciprocals must be ROW_MAJOR, got {}", recip.layout());
            TT_FATAL(
                recip.buffer() != nullptr && recip.storage_type() == StorageType::DEVICE,
                "reciprocals must be an allocated device tensor");
            // [.., reduce_width] where reduce_width == per-device H (the Welford reduce
            // width). The LLK indexes it by sample count 0..reduce_width-1.
            TT_FATAL(
                recip.logical_shape()[-1] == input.logical_shape()[3],
                "reciprocals last dim ({}) must equal per-device H ({})",
                recip.logical_shape()[-1],
                input.logical_shape()[3]);
        }
    }

    TT_FATAL(input.storage_type() == StorageType::DEVICE, "Input must be on device");
    TT_FATAL(input.buffer() != nullptr, "Input must be allocated");
    TT_FATAL(input.layout() == Layout::TILE, "Input layout must be TILE, got {}", input.layout());
    TT_FATAL(
        input.dtype() == DataType::BFLOAT16 || input.dtype() == DataType::FLOAT32,
        "Input dtype must be BFLOAT16 or FLOAT32, got {}",
        input.dtype());

    const auto& shape = input.logical_shape();
    TT_FATAL(shape.rank() == 4, "Input rank must be 4, got {}", shape.rank());
    // dim0 must be 1; dim1 is the batch dim (folded into the row count via
    // physical_volume/W, so each batch contributes N_pad/TILE_HEIGHT tile-rows).
    // Two batched output layouts (see compute_output_specs):
    //   - num_heads_per_device==1: batch preserved at dim1 -> [1, batch, N, H] (Motif adaLN;
    //     the caller's squeeze(0) recovers [batch, N, H]).
    //   - num_heads_per_device>1 (head-split, e.g. batched QK-norm + RoPE): heads occupy dim1,
    //     so batch is folded into the seq dim -> [1, num_heads, batch*N, head_dim]. This matches
    //     the writer's flat (b*rows_per_batch + r)*head_dim_tiles layout; the caller reshapes the
    //     seq dim back to (batch, N).
    TT_FATAL(shape[0] == 1, "Input dim0 must be 1 (shape [1, batch, N, H]); got {}", shape[0]);
    const uint32_t batch = shape[1];
    if (batch > 1) {
        TT_FATAL(!args.per_head_norm, "Batched input (dim1={}) does not support per_head_norm", batch);
        // Per-token weight/bias ([.,N,H]) with batch>1 is untested: the per-token reader indexes
        // weight/bias by the global folded row assuming a single batch's N rows. Reject up front
        // (the per-batch [batch,1,H] adaLN path IS supported and has logical[-2]==1).
        const bool pt_w = weight.has_value() && weight->logical_shape()[-2] > 1;
        const bool pt_b = bias.has_value() && bias->logical_shape()[-2] > 1;
        TT_FATAL(
            !pt_w && !pt_b,
            "Batched input (dim1={}) does not support per-token ([.,N,H]) weight/bias yet "
            "(the per-token reader indexes by the global folded row)",
            batch);
    }
    TT_FATAL(args.num_heads_per_device >= 1, "num_heads_per_device must be >= 1");
    TT_FATAL(
        shape[3] % args.num_heads_per_device == 0,
        "Input H ({}) must be divisible by num_heads_per_device ({})",
        shape[3],
        args.num_heads_per_device);

    const auto& padded = input.padded_shape();
    TT_FATAL(padded[3] == shape[3], "Input last logical dim ({}) must equal padded last dim ({})", shape[3], padded[3]);

    if (weight.has_value()) {
        TT_FATAL(weight->layout() == Layout::TILE, "Weight layout must be TILE");
        TT_FATAL(
            weight->dtype() == DataType::BFLOAT16 || weight->dtype() == DataType::FLOAT32,
            "Weight dtype must be BFLOAT16 or FLOAT32, got {}",
            weight->dtype());
        TT_FATAL(
            weight->padded_shape()[-1] == padded[3],
            "Weight last dim ({}) must equal input H per device ({})",
            weight->padded_shape()[-1],
            padded[3]);
        // Per-token weight has shape [N, H] (or [..., N, H]); broadcast is
        // [1, H]. Distinguish via the logical (not padded) seqlen dim.
        const auto& w_logical = weight->logical_shape();
        const auto w_n = w_logical[-2];
        TT_FATAL(
            w_n == 1 || w_n == shape[2],
            "Weight second-to-last logical dim ({}) must be 1 (broadcast) or N ({}) for per-token",
            w_n,
            shape[2]);
    }

    if (bias.has_value()) {
        TT_FATAL(weight.has_value(), "bias requires weight to also be provided");
        TT_FATAL(bias->layout() == Layout::TILE, "Bias layout must be TILE");
        TT_FATAL(
            bias->dtype() == DataType::BFLOAT16 || bias->dtype() == DataType::FLOAT32,
            "Bias dtype must be BFLOAT16 or FLOAT32, got {}",
            bias->dtype());
        TT_FATAL(
            bias->padded_shape()[-1] == padded[3],
            "Bias last dim ({}) must equal input H per device ({})",
            bias->padded_shape()[-1],
            padded[3]);
        const auto& b_logical = bias->logical_shape();
        const auto b_n = b_logical[-2];
        TT_FATAL(
            b_n == 1 || b_n == shape[2],
            "Bias second-to-last logical dim ({}) must be 1 (broadcast) or N ({}) for per-token",
            b_n,
            shape[2]);
    }

    const bool rope_present = trans_mat.has_value() || rope_cos.has_value() || rope_sin.has_value();
    const bool rope_complete = trans_mat.has_value() && rope_cos.has_value() && rope_sin.has_value();
    TT_FATAL(
        !rope_present || rope_complete,
        "RoPE requires transformation_mat, rope_cos, and rope_sin all to be provided together");

    // RoPE cos/sin shape: [B, num_heads_dim, N, head_dim] where num_heads_dim is
    // either 1 (broadcast across heads — same cos/sin for every head) or
    // num_heads_per_device (per-head cos/sin). Both rope_cos and rope_sin must
    // match.
    if (rope_complete) {
        const auto& cos_shape = rope_cos->logical_shape();
        const auto& sin_shape = rope_sin->logical_shape();
        TT_FATAL(
            cos_shape == sin_shape, "rope_cos and rope_sin must have the same shape ({} vs {})", cos_shape, sin_shape);
        TT_FATAL(cos_shape.rank() == 4, "rope_cos must be 4D, got rank {}", cos_shape.rank());
        TT_FATAL(
            cos_shape[1] == 1 || cos_shape[1] == args.num_heads_per_device,
            "rope_cos dim 1 ({}) must be 1 (broadcast across heads) or num_heads_per_device ({})",
            cos_shape[1],
            args.num_heads_per_device);
        // Batched RoPE: cos/sin dim 0 is either 1 (the same RoPE broadcast to every input batch)
        // or `batch` (each input batch gets its own cos/sin slice). The reader indexes cos/sin by
        // the WITHIN-batch seq row (global_row % rope_seqlen_tiles) plus a per-batch offset, so the
        // cos/sin seq length must match the input's per-batch N and N must be tile-aligned (so the
        // input fold gives an integer rope_seqlen_tiles rows per batch).
        TT_FATAL(
            cos_shape[0] == 1 || cos_shape[0] == batch,
            "rope_cos dim 0 ({}) must be 1 (broadcast across the input batch) or the input batch ({})",
            cos_shape[0],
            batch);
        TT_FATAL(
            cos_shape[2] == shape[2],
            "rope_cos seq dim ({}) must equal input N ({}) — the reader indexes cos/sin by the "
            "within-batch seq row",
            cos_shape[2],
            shape[2]);
        if (batch > 1) {
            TT_FATAL(
                shape[2] % TILE_HEIGHT == 0,
                "Batched fused RoPE requires input N ({}) to be tile-aligned (multiple of {})",
                shape[2],
                TILE_HEIGHT);
        }
    }

    TT_FATAL(args.ring_size >= 1, "ring_size must be >= 1");
    TT_FATAL(args.num_links >= 1, "num_links must be >= 1");
    TT_FATAL(args.cluster_axis < 2, "cluster_axis must be 0 or 1");
    // per_head_norm skips the AG entirely; the kernel does NOT need a fabric
    // sem when this path is active, even for ring_size > 1.
    if (args.ring_size > 1 && !args.per_head_norm) {
        TT_FATAL(
            !args.multi_device_global_semaphore.empty(),
            "multi_device_global_semaphore must be non-empty when ring_size > 1");
    }
    if (args.per_head_norm) {
        TT_FATAL(
            args.num_heads_per_device > 1,
            "per_head_norm requires num_heads_per_device > 1 (got {})",
            args.num_heads_per_device);
    }

    // The all-gather path (TP>1, whole-row norm) requires a caller-supplied
    // persistent buffer for the gathered-stats DRAM scratch. The buffer must
    // be allocated as a mesh-coherent MeshBuffer (same DRAM address on every
    // chip in the cluster) — required for the fabric mcast. Allocating it
    // outside the op lets the caller pre-create one MeshBuffer per cluster
    // and reuse it across launches; we reject mismatched shape/dtype/layout
    // up front so a wrong tensor doesn't silently corrupt the AG.
    const auto sizing = compute_sizing(args, input);
    if (sizing.use_mux) {
        // Validate the caller's buffer against the SAME spec compute_output_specs would allocate,
        // so the check can never drift from the allocation.
        const auto expected = make_stats_tensor_spec(sizing);
        const auto& e_shape = expected.logical_shape();
        TT_FATAL(
            tensor_args.persistent_output_buffer.has_value(),
            "persistent_output_buffer is required for TP>1 with multiple workers (use_mux). "
            "Allocate it as a regular device tensor with shape "
            "[1, 1, {}, {}], dtype=FLOAT32, layout=ROW_MAJOR, DRAM INTERLEAVED.",
            e_shape[2],
            e_shape[3]);
        const auto& buf = tensor_args.persistent_output_buffer.value();
        TT_FATAL(buf.storage_type() == StorageType::DEVICE, "persistent_output_buffer must be on device");
        TT_FATAL(buf.buffer() != nullptr, "persistent_output_buffer must be allocated");
        TT_FATAL(
            buf.layout() == expected.layout(),
            "persistent_output_buffer layout must be {} (got {})",
            expected.layout(),
            buf.layout());
        TT_FATAL(
            buf.dtype() == expected.data_type(),
            "persistent_output_buffer dtype must be {} (got {})",
            expected.data_type(),
            buf.dtype());
        TT_FATAL(
            buf.memory_config().buffer_type() == expected.memory_config().buffer_type(),
            "persistent_output_buffer must be in DRAM");
        TT_FATAL(
            buf.memory_config().memory_layout() == expected.memory_config().memory_layout(),
            "persistent_output_buffer must be INTERLEAVED");
        const auto& b_shape = buf.logical_shape();
        TT_FATAL(b_shape.rank() == 4, "persistent_output_buffer rank must be 4 (got {})", b_shape.rank());
        TT_FATAL(
            b_shape[0] == e_shape[0] && b_shape[1] == e_shape[1] && b_shape[2] == e_shape[2] &&
                b_shape[3] == e_shape[3],
            "persistent_output_buffer shape must be [1, 1, {}, {}], got [{}, {}, {}, {}]",
            e_shape[2],
            e_shape[3],
            b_shape[0],
            b_shape[1],
            b_shape[2],
            b_shape[3]);
    }
}

DitFusedDistributedRmsnormDeviceOperation::spec_return_value_t
DitFusedDistributedRmsnormDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    const auto& logical = input.logical_shape();

    std::vector<tt::tt_metal::TensorSpec> specs;
    specs.reserve(2);

    // Post-allgather output reshapes to [1, num_heads_per_device, N, H/num_heads_per_device].
    // The writer lays out all folded tile-rows flat as
    //   h * (batch*N/32) * head_dim_tiles + (b*rows_per_batch + r) * head_dim_tiles + c,
    // so two batched layouts fall out of the same flat index:
    //   - num_heads_per_device==1: batch stays at dim1 -> [1, batch, N, H]. Identical tile
    //     ordering to [1,1,batch*N,H]; the caller's squeeze(0) recovers [batch, N, H] (Motif).
    //   - num_heads_per_device>1 (head-split): heads occupy dim1, so batch folds into the seq
    //     dim -> [1, num_heads, batch*N, head_dim]. batch==1 leaves this as [1, num_heads, N, hd]
    //     (unchanged). The caller reshapes the seq dim back to (batch, N).
    const uint32_t batch = logical[1];
    const uint32_t out_dim1 = (args.num_heads_per_device == 1) ? batch : args.num_heads_per_device;
    const uint32_t out_dim2 = (args.num_heads_per_device == 1) ? logical[2] : batch * logical[2];
    ttnn::Shape output_shape({1u, out_dim1, out_dim2, logical[3] / args.num_heads_per_device});
    const auto out_dtype = args.dtype.value_or(input.dtype());
    specs.emplace_back(output_shape, TensorLayout(out_dtype, PageConfig(Layout::TILE), args.output_mem_config));

    // Persistent stats DRAM scratch for the all-gather path. Each page holds one
    // chunk's worth of post-reduce stats in row-major form: TILE_HEIGHT *
    // window_size fp32 values =
    // TILE_HEIGHT * window_size * 4 bytes per page. There are total_pages =
    // ring_size * num_chunks_per_device pages — independent of num_workers
    // by design, so the caller need not know the worker count.
    //
    // We expose the buffer as a ROW_MAJOR fp32 tensor of shape
    // [1, 1, total_pages, TILE_HEIGHT * window_size]; ttnn defaults each
    // row to one accessor page, so TensorAccessor page_idx maps 1:1 to the
    // packed-page index the writer/reader address.
    //
    // Allocated as a regular device tensor so the framework's
    // create_device_tensor places it as a mesh-coherent MeshBuffer — every
    // chip gets the same DRAM address, which the fabric mcast relies on.
    const auto sizing = compute_sizing(args, input);
    if (sizing.use_mux) {
        specs.emplace_back(make_stats_tensor_spec(sizing));
    }

    return specs;
}

DitFusedDistributedRmsnormDeviceOperation::tensor_return_value_t
DitFusedDistributedRmsnormDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto specs = compute_output_specs(args, tensor_args);
    std::vector<Tensor> tensors;
    tensors.reserve(specs.size());
    auto* mesh_device = tensor_args.input.device();
    // tensors[0] = user-visible output (always allocated fresh).
    tensors.push_back(create_device_tensor(specs[0], mesh_device));
    // tensors[1] = stats DRAM scratch (only present when use_mux). Caller
    // must provide a pre-allocated mesh-coherent buffer via
    // tensor_args.persistent_output_buffer; we validated it in
    // validate_on_program_cache_miss.
    if (specs.size() > 1) {
        // Re-check here so cache hits also get a clean error if the caller
        // forgot the buffer (validate_on_program_cache_miss runs only once).
        TT_FATAL(
            tensor_args.persistent_output_buffer.has_value(),
            "persistent_output_buffer is required for TP>1 with multiple workers");
        tensors.push_back(tensor_args.persistent_output_buffer.value());
    }
    return tensors;
}

ttsl::hash::hash_t DitFusedDistributedRmsnormDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    log_trace(tt::LogOp, "DitFusedDistributedRmsnormDeviceOperation::compute_program_hash");
    auto* mesh_device = tensor_args.input.device();
    auto sd_id = args.sub_device_id.value_or(mesh_device->get_sub_device_ids().at(0));
    auto subdevice_core_range_set = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);
    return tt::tt_metal::operation::hash_operation<DitFusedDistributedRmsnormDeviceOperation>(
        args.epsilon,
        args.num_heads_per_device,
        args.per_head_norm,
        args.dtype,
        args.output_mem_config,
        args.cluster_axis,
        args.num_links,
        args.ring_size,
        args.topology,
        args.compute_kernel_config,
        static_cast<uint8_t>(args.norm_type),
        subdevice_core_range_set,
        tensor_args);
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor dit_fused_distributed_rmsnorm(
    const Tensor& input_tensor,
    uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    ttnn::ccl::Topology topology,
    float epsilon,
    uint32_t num_heads_per_device,
    bool per_head_norm,
    const std::optional<const Tensor>& weight,
    const std::optional<const Tensor>& bias,
    const std::optional<const Tensor>& transformation_mat,
    const std::optional<const Tensor>& rope_cos,
    const std::optional<const Tensor>& rope_sin,
    const std::optional<const DataType>& dtype,
    const std::optional<Tensor>& persistent_output_buffer,
    std::optional<size_t> num_preferred_links,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<const DeviceComputeKernelConfig>& compute_kernel_config,
    ttnn::experimental::DitFusedNormType norm_type,
    const std::optional<const Tensor>& reciprocals) {
    using OperationType = ttnn::experimental::prim::DitFusedDistributedRmsnormDeviceOperation;

    auto arch = is_device_tensor(input_tensor) ? input_tensor.device()->arch() : ttnn::GetDefaultDevice()->arch();
    // Match composite ops' fused_rmsnorm_{pre,post}_allgather defaults so
    // both paths produce numerically equivalent output when the caller
    // omits compute_kernel_config: math_approx=false, fp32_dest_acc_en=true.
    auto kernel_config_val = init_device_compute_kernel_config(
        arch, compute_kernel_config, tt::tt_metal::MathFidelity::HiFi4, false, true, false);

    const auto& mesh_view = mesh_device.get_view();
    const std::size_t num_devices = (cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();

    // get_usable_topology reaches into the fabric context, which is null when the op runs on a
    // single device with fabric uninitialized (TP=1, ring_size==1). At num_devices==1 there is no
    // ring / all-gather and the topology is never used (every ring path is guarded on
    // ring_size>1), so skip the fabric query and use a harmless default.
    tt::tt_fabric::Topology topology_ = (num_devices > 1)
                                            ? ::ttnn::ccl::get_usable_topology(input_tensor, topology, cluster_axis)
                                            : tt::tt_fabric::Topology::Linear;

    auto operation_attributes = OperationType::operation_attributes_t(
        epsilon,
        num_heads_per_device,
        per_head_norm,
        dtype,
        memory_config.value_or(input_tensor.memory_config()),
        cluster_axis,
        static_cast<uint32_t>(num_preferred_links.value_or(1)),
        static_cast<uint32_t>(num_devices),
        topology_,
        multi_device_global_semaphore,
        subdevice_id,
        kernel_config_val,
        norm_type);

    auto tensor_args = OperationType::tensor_args_t{
        .input = input_tensor,
        .weight = weight,
        .bias = bias,
        .transformation_mat = transformation_mat,
        .rope_cos = rope_cos,
        .rope_sin = rope_sin,
        .persistent_output_buffer = persistent_output_buffer,
        .reciprocals = reciprocals};

    auto outputs = ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
    // outputs[0] = rmsnorm output, outputs[1] (if present) = stats DRAM scratch.
    return outputs[0];
}

}  // namespace ttnn::prim

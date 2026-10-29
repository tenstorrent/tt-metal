// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/device/sparse_ring_sdpa_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"

#include <algorithm>
#include <variant>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/device.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/ring_attention_all_gather_async_device_operation.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/ring_joint_chain_layout.hpp"
#include "ttnn/operations/transformer/sdpa/device/sparse_ring_sdpa_device_operation_types.hpp"
#include "ttnn/operations/transformer/sdpa/device/sparse_ring_sdpa_program_factory.hpp"
#include "ttnn/operations/transformer/sdpa/device/sdpa_perf_model.hpp"
#include "ttnn/tensor/types.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

using namespace experimental::ccl;

namespace {

namespace ring_joint = ttnn::operations::transformer::sdpa::ring_joint;

// Block sizes the kernels support today (SparseRingSDPAParams::block_size). The kernels take the block size as a
// compile-time parameter; other sizes need the work listed in sparse_ring_sdpa_refactor_plan.md.
constexpr uint32_t kSupportedBlockSize = 64;

void validate_all_gather_on_program_cache_miss(
    const ttnn::experimental::prim::RingAttentionAllGatherAsyncParams& operation_attributes,
    const ttnn::experimental::prim::RingAttentionAllGatherAsyncInputs& tensor_args) {
    const auto& input_tensors = tensor_args.input_tensor;
    TT_FATAL(
        !input_tensors.empty(), "Error, Input tensor size should be greater than 0 but has {}", input_tensors.size());
    TT_FATAL(input_tensors[0].buffer() != nullptr, "Input tensor 0 must be allocated in buffers on device");

    const auto dtype = input_tensors[0].dtype();
    const auto page_size = input_tensors[0].buffer()->page_size();
    for (size_t i = 0; i < input_tensors.size(); ++i) {
        const auto& input_tensor = input_tensors[i];

        TT_FATAL(input_tensor.layout() == Layout::TILE, "Input tensor {} must be tiled", i);
        TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Input tensor {} must be on device", i);
        TT_FATAL(input_tensor.buffer() != nullptr, "Input tensor {} must be allocated in buffers on device", i);
        TT_FATAL(
            input_tensor.dtype() == dtype,
            "All input tensors must have the same dtype. Input tensor {} has dtype {} but expected {}",
            i,
            input_tensor.dtype(),
            dtype);
        TT_FATAL(
            input_tensor.buffer()->page_size() == page_size,
            "All input tensors must have the same page size. Input tensor {} has page size {} but expected {}",
            i,
            input_tensor.buffer()->page_size(),
            page_size);
    }

    TT_FATAL(
        operation_attributes.num_links > 0,
        "Error, num_links should be more than 0 but has {}",
        operation_attributes.num_links);

    const auto& output_tensors = tensor_args.persistent_output_buffer;
    TT_FATAL(
        output_tensors.size() == input_tensors.size(),
        "Number of output tensors ({}) must match number of input tensors ({})",
        output_tensors.size(),
        input_tensors.size());
    for (size_t i = 0; i < output_tensors.size(); ++i) {
        TT_FATAL(output_tensors[i].has_value(), "SparseRingSDPA requires persistent all-gather output tensor {}", i);
        const auto& output_tensor = output_tensors[i].value();

        TT_FATAL(output_tensor.layout() == Layout::TILE, "Output tensor {} must be tiled", i);
        TT_FATAL(output_tensor.storage_type() == StorageType::DEVICE, "Output tensor {} must be on device", i);
        TT_FATAL(output_tensor.buffer() != nullptr, "Output tensor {} must be allocated in buffers on device", i);
        TT_FATAL(
            output_tensor.dtype() == dtype,
            "Output tensor {} dtype should match input tensors but has {}",
            i,
            output_tensor.dtype());
        TT_FATAL(
            output_tensor.buffer()->page_size() == page_size,
            "Output tensor {} page size should match input tensors but has {}",
            i,
            output_tensor.buffer()->page_size());
        TT_FATAL(
            output_tensor.memory_config() == operation_attributes.output_mem_config,
            "Output tensor {} memory config should match output_mem_config",
            i);

        auto output_shape = output_tensor.logical_shape();
        auto expected_output_shape = input_tensors[i].logical_shape();
        expected_output_shape[operation_attributes.dim] *= operation_attributes.ring_size;
        for (int d = 0; d < static_cast<int>(output_shape.rank()); ++d) {
            if (d == operation_attributes.dim) {
                TT_FATAL(
                    output_shape[d] >= expected_output_shape[d],
                    "Output tensor {} gather dim {} too small: got {}, expected >= {}",
                    i,
                    d,
                    output_shape[d],
                    expected_output_shape[d]);
            } else {
                TT_FATAL(
                    output_shape[d] == expected_output_shape[d],
                    "Output tensor {} non-gather dim {} mismatch: got {}, expected {}",
                    i,
                    d,
                    output_shape[d],
                    expected_output_shape[d]);
            }
        }
    }
}

}  // namespace

void SparseRingSDPADeviceOperation::validate_on_program_cache_miss(
    const SparseRingSDPAParams& args, const SparseRingSDPAInputs& tensor_args) {
    const auto& input_tensor_q = tensor_args.input_q;
    const auto& gathered_k = tensor_args.gathered_k;
    const auto& gathered_v = tensor_args.gathered_v;

    TT_FATAL(args.program_config.has_value(), "Program config must be provided");
    const uint32_t block_size = args.block_size;
    TT_FATAL(
        block_size == kSupportedBlockSize,
        "SparseRingSDPA: block_size must be {} (only {} is supported for now), got {}",
        kSupportedBlockSize,
        kSupportedBlockSize,
        block_size);
    TT_FATAL(
        args.get_q_chunk_size() == block_size,
        "SparseRingSDPA computes one {}-row query block per mask row: q_chunk_size must equal block_size, got {}",
        block_size,
        args.get_q_chunk_size());
    TT_FATAL(
        args.get_k_chunk_size() > 0 && args.get_k_chunk_size() % block_size == 0,
        "k_chunk_size must be a positive multiple of the {}-token block, got {}",
        block_size,
        args.get_k_chunk_size());
    TT_FATAL(
        !get_fp32_dest_acc_en(args.compute_kernel_config),
        "SparseRingSDPA requires streaming compute; set fp32_dest_acc_en=false");
    TT_FATAL(
        !args.q_blocks_per_pass.has_value() || (*args.q_blocks_per_pass >= 1 && *args.q_blocks_per_pass <= 32),
        "SparseRingSDPA: q_blocks_per_pass must be in [1, 32], got {}",
        args.q_blocks_per_pass.value_or(0));

    validate_all_gather_on_program_cache_miss(args.all_gather_operation_attributes, args.all_gather_tensor_args);
    TT_FATAL(!args.all_gather_operation_attributes.full_mesh, "SparseRingSDPA gathers along one mesh axis");

    // Check that SDPA coregrid does not overlap with AllGather coregrid
    const auto strategy = args.all_gather_operation_attributes.core_allocation_strategy;
    if (strategy == ttnn::ccl::CoreAllocationStrategy::COL_MAJOR) {
        TT_FATAL(
            args.ccl_core_grid_offset.x >= args.program_config.value().compute_with_storage_grid_size.x,
            "SDPA coregrid overlaps with AllGather coregrid (column-major)");
    } else {
        TT_FATAL(
            args.ccl_core_grid_offset.y >= args.program_config.value().compute_with_storage_grid_size.y,
            "SDPA coregrid overlaps with AllGather coregrid (row-major)");
    }

    const std::vector<Tensor> sdpa_input_tensors = {input_tensor_q, gathered_k, gathered_v};
    const auto dtype = input_tensor_q.dtype();
    for (const auto& tensor : sdpa_input_tensors) {
        TT_FATAL(
            tensor.dtype() == dtype,
            "All tensors must have the same dtype. Expected {}, got {}",
            dtype,
            tensor.dtype());
        TT_FATAL(tensor.storage_type() == StorageType::DEVICE, "Operands to SparseRingSDPA need to be on device");
        TT_FATAL(tensor.buffer() != nullptr, "Operands to SparseRingSDPA need to be allocated in buffers on device");
        TT_FATAL(tensor.layout() == Layout::TILE, "Inputs to SparseRingSDPA must be tilized");
        TT_FATAL(
            tensor.dtype() == DataType::BFLOAT16 || tensor.dtype() == DataType::BFLOAT8_B ||
                tensor.dtype() == DataType::BFLOAT4_B,
            "Inputs to SparseRingSDPA must be BF16 or BF8 or BF4");
        TT_FATAL(
            tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM,
            "Operands to SparseRingSDPA need to be in DRAM");
        // Only the sequence dimension may be padded.
        const auto& logical_shape = tensor.logical_shape();
        const auto& padded_shape = tensor.padded_shape();
        TT_FATAL(logical_shape[0] == padded_shape[0], "Padding is not supported on the batch dimension");
        TT_FATAL(logical_shape[1] == padded_shape[1], "Padding is not supported on the num_heads dimension");
        TT_FATAL(logical_shape[3] == padded_shape[3], "Padding is not supported on the head_dim dimension");
    }

    const auto& q_shape = input_tensor_q.logical_shape();
    const auto& k_shape = gathered_k.logical_shape();
    const auto& v_shape = gathered_v.logical_shape();
    const auto B = q_shape[0];
    const auto NQH = q_shape[1];
    const auto NKH = k_shape[1];
    const auto NVH = tensor_args.input_v.logical_shape()[1];
    const auto DH = q_shape[3];
    const auto N_local_q = q_shape[2];
    const auto N_local_kv = tensor_args.local_kv_seq_len();
    const auto N_global = k_shape[2];

    TT_FATAL(NQH > 0 && NKH > 0, "Q and K num_heads must be greater than 0. Got Q: {}, K: {}", NQH, NKH);
    TT_FATAL(
        N_local_q == N_local_kv,
        "Per-device Q seq length must equal the per-device K/V seq length. Got N_local_q={}, N_local_kv={}",
        N_local_q,
        N_local_kv);
    TT_FATAL(
        tensor_args.input_v.logical_shape()[2] == N_local_kv,
        "V local seq length must match K local seq length. Got V: {}, K: {}",
        tensor_args.input_v.logical_shape()[2],
        N_local_kv);
    TT_FATAL(
        N_local_q % tt::constants::TILE_HEIGHT == 0,
        "Per-device Q seq length must be divisible by TILE_HEIGHT. Got N_local_q: {}, TILE_HEIGHT: {}",
        N_local_q,
        tt::constants::TILE_HEIGHT);
    TT_FATAL(k_shape[0] == B, "K batch size must match Q. Got Q: {}, K: {}", B, k_shape[0]);
    TT_FATAL(v_shape[0] == B, "V batch size must match Q. Got Q: {}, V: {}", B, v_shape[0]);
    TT_FATAL(
        k_shape[3] == DH && v_shape[3] == DH,
        "Head dimensions must match. Got Q: {}, K: {}, V: {}",
        DH,
        k_shape[3],
        v_shape[3]);
    TT_FATAL(
        k_shape[2] == v_shape[2],
        "K sequence length must be equal to V sequence length. Got K: {}, V: {}",
        k_shape[2],
        v_shape[2]);
    TT_FATAL(
        N_global >= N_local_kv * args.ring_size,
        "Gathered K seq length must be >= per-device K shard times ring size. Got N_global: {}, N_local_kv: {}, "
        "ring_size: {}",
        N_global,
        N_local_kv,
        args.ring_size);
    TT_FATAL(
        args.logical_n > 0,
        "Logical sequence length must be > 0; kernels derive last-valid-tile = logical_nt - 1 and would underflow.");
    TT_FATAL(
        args.logical_n <= N_global,
        "Logical sequence length must be less than or equal to global sequence length. Got logical sequence length: "
        "{}, global sequence length: {}",
        args.logical_n,
        N_global);

    const bool tensor_mha = (NQH == NKH) && (NKH == NVH);
    const bool tensor_separate_v_shared_k = (NKH == 1) && (NVH == NQH);
    const bool tensor_gqa_grouped_kv =
        ring_joint::is_gqa_grouped_kv_head_mode(/*v_shares_k_buffer=*/false, NQH, NKH, NVH);
    TT_FATAL(
        tensor_mha || tensor_separate_v_shared_k || tensor_gqa_grouped_kv,
        "Unsupported K/V head relationship. Expected MHA (NQH == NKH == NVH), separate-V shared-K "
        "(NKH == 1 && NVH == NQH), or GQA (NKH == NVH < NQH && NQH % NKH == 0). Got NQH: {}, NKH: {}, NVH: {}",
        NQH,
        NKH,
        NVH);

    // Block mask and (optional) pooled K / V.
    const auto& mask = tensor_args.block_mask;
    TT_FATAL(
        mask.layout() == Layout::ROW_MAJOR && mask.dtype() == DataType::UINT32, "block_mask must be uint32 row-major");
    TT_FATAL(
        mask.buffer() != nullptr && mask.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM,
        "block_mask must be in DRAM");
    TT_FATAL(
        tensor_args.pooled_k.has_value() == tensor_args.pooled_v.has_value(),
        "pooled_k and pooled_v must be passed together (Sol-Attn) or both omitted (kept blocks only)");
    const uint32_t n_blocks_pad = mask.logical_shape()[3];
    if (tensor_args.pooled_route()) {
        const auto& pk = *tensor_args.pooled_k;
        const auto& pv = *tensor_args.pooled_v;
        for (const auto* t : {&pk, &pv}) {
            TT_FATAL(
                t->layout() == Layout::TILE && t->dtype() == DataType::BFLOAT16,
                "pooled_k / pooled_v must be bf16 tile");
            TT_FATAL(
                t->buffer() != nullptr && t->buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM,
                "pooled_k / pooled_v must be in DRAM");
        }
        TT_FATAL(pk.logical_shape() == pv.logical_shape(), "pooled_k and pooled_v shapes differ");
        TT_FATAL(
            pk.logical_shape()[1] == NQH && pk.logical_shape()[2] == n_blocks_pad && pk.logical_shape()[3] == DH,
            "pooled_k must be [1, NH, n_blocks_pad = block_mask width {}, DH], got {}",
            n_blocks_pad,
            pk.logical_shape());
    }
    const uint32_t nb_pad = n_blocks_pad / args.ring_size;
    TT_FATAL(
        n_blocks_pad % (32 * args.ring_size) == 0,
        "block_mask width {} must be ring_size x a multiple of 32 blocks",
        n_blocks_pad);
    TT_FATAL(nb_pad >= tt::div_up(N_local_kv, block_size), "mask blocks per shard {} < blocks in the K shard", nb_pad);
    const uint32_t nqb = tt::div_up(N_local_q, block_size);
    TT_FATAL(
        mask.logical_shape()[1] == NQH && mask.logical_shape()[2] >= nqb && mask.logical_shape()[3] == n_blocks_pad,
        "block_mask must be [1, NH, >= {} q blocks, {} blocks], got {}",
        nqb,
        n_blocks_pad,
        mask.logical_shape());
    const uint32_t nkc_local = tt::div_up(N_local_kv, args.get_k_chunk_size());
    TT_FATAL(nkc_local <= 64, "SparseRingSDPA: at most 64 K pieces per shard ({} given)", nkc_local);
    // Every shard inside logical_n must hold at least one full block. Only full blocks are ever computed exactly; a
    // shard with a live tail shorter than one block has none, so its ring iteration streams nothing, compute runs no
    // step for it, and the writer waits forever. Pad logical_n up or re-shard so no shard has a sub-block tail.
    for (uint32_t s = 0; s < args.ring_size; ++s) {
        const int64_t valid = std::min<int64_t>(
            static_cast<int64_t>(N_local_kv),
            static_cast<int64_t>(args.logical_n) - static_cast<int64_t>(s) * static_cast<int64_t>(N_local_kv));
        TT_FATAL(
            valid <= 0 || valid >= block_size,
            "SparseRingSDPA: shard {} holds {} valid tokens; every shard inside logical_n needs a full {}-token "
            "block",
            s,
            valid,
            block_size);
    }
}

void SparseRingSDPADeviceOperation::validate_on_program_cache_hit(
    const SparseRingSDPAParams& /*args*/, const SparseRingSDPAInputs& /*tensor_args*/) {
    // Every attribute the miss path checks is part of the program hash.
}

SparseRingSDPAResultSpec SparseRingSDPADeviceOperation::compute_output_specs(
    const SparseRingSDPAParams& args, const SparseRingSDPAInputs& tensor_args) {
    const auto& input = tensor_args.input_q;
    const auto v_head_dim = tensor_args.input_v.logical_shape()[3];
    auto out_shape = input.logical_shape();
    out_shape[3] = v_head_dim;

    return {tt::tt_metal::TensorSpec(
        out_shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), args.output_memory_config))};
}

SparseRingSDPAResult SparseRingSDPADeviceOperation::create_output_tensors(
    const SparseRingSDPAParams& args, const SparseRingSDPAInputs& tensor_args) {
    auto output_specs = compute_output_specs(args, tensor_args);
    return {create_device_tensor(output_specs[SPARSE_RING_SDPA_OUTPUT_IDX], tensor_args.input_q.device())};
}

ttsl::hash::hash_t SparseRingSDPADeviceOperation::compute_program_hash(
    const SparseRingSDPAParams& args, const SparseRingSDPAInputs& tensor_args) {
    std::vector<Tensor> input_tensors = {
        tensor_args.input_q,
        tensor_args.input_k,
        tensor_args.input_v,
        tensor_args.gathered_k,
        tensor_args.gathered_v,
        tensor_args.block_mask};
    if (tensor_args.pooled_route()) {
        input_tensors.push_back(*tensor_args.pooled_k);
        input_tensors.push_back(*tensor_args.pooled_v);
    }
    // logical_n is baked into the kernels (compile-time args and the bias generator), so it is keyed.
    // The slots per pass and the K/V CB depth are derived from these inputs plus the live L1 top
    // (lowest_occupied_compute_l1_address), which is not keyed: a cached program keeps the choice of the call that
    // built it.
    return tt::tt_metal::operation::hash_operation<SparseRingSDPADeviceOperation>(
        input_tensors,
        args.scale,
        args.logical_n,
        args.ring_size,
        args.compute_kernel_config,
        args.program_config,
        args.ccl_core_grid_offset,
        args.all_gather_operation_attributes,
        args.all_gather_tensor_args,
        args.block_size,
        args.q_blocks_per_pass,
        tensor_args.pooled_route());
}

tt::tt_metal::operation::OpPerformanceModelGeneral<Tensors> SparseRingSDPADeviceOperation::create_op_performance_model(
    const SparseRingSDPAParams& args, const SparseRingSDPAInputs& tensor_args, SparseRingSDPAResult& output_tensors) {
    Tensors input_tensors = {
        tensor_args.input_q, tensor_args.input_k, tensor_args.input_v, tensor_args.gathered_k, tensor_args.gathered_v};

    auto& output_tensor = output_tensors[SPARSE_RING_SDPA_OUTPUT_IDX];
    auto arch = output_tensor.storage_type() == StorageType::DEVICE ? output_tensor.device()->arch()
                                                                    : ttnn::GetDefaultDevice()->arch();

    if (arch != tt::ARCH::WORMHOLE_B0 && arch != tt::ARCH::BLACKHOLE) {
        log_warning(tt::LogOp, "SparseRingSDPA perf model does not support arch '{}'", enchantum::to_string(arch));
        return operation::OpPerformanceModelGeneral<Tensors>(input_tensors, output_tensors, 0);
    }

    const auto& q_shape = tensor_args.input_q.logical_shape();
    const auto& gathered_k_shape = tensor_args.gathered_k.logical_shape();

    CoreCoord grid = args.program_config.has_value() ? args.program_config->compute_with_storage_grid_size
                                                     : output_tensor.device()->compute_with_storage_grid_size();
    // QK^T and softmax @ V dominate the modeled cycles, so use their fidelity when it is overridden.
    const auto matmul_fidelity =
        args.program_config.has_value() ? args.program_config->matmul_math_fidelity : std::nullopt;
    tt::tt_metal::MathFidelity fidelity = matmul_fidelity.value_or(ttnn::get_math_fidelity(args.compute_kernel_config));

    // Dense upper bound: the kept-block density is only known on device.
    int ideal_cycles = operations::transformer::sdpa::compute_sdpa_ideal_cycles(
        q_shape[0],
        q_shape[1],
        q_shape[2],
        gathered_k_shape[2],
        q_shape[3],
        tensor_args.input_v.logical_shape()[3],
        /*is_causal=*/false,
        fidelity,
        grid.x * grid.y);

    return operation::OpPerformanceModelGeneral<Tensors>(input_tensors, output_tensors, ideal_cycles);
}

SparseRingSDPAResult sparse_ring_sdpa(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const ttnn::Tensor& block_mask,
    const std::optional<ttnn::Tensor>& pooled_k,
    const std::optional<ttnn::Tensor>& pooled_v,
    ttnn::Tensor& persistent_output_buffer_k,
    ttnn::Tensor& persistent_output_buffer_v,
    const std::size_t logical_n,
    ttnn::operations::transformer::SDPAProgramConfig program_config,
    const int32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const uint32_t num_links,
    const uint32_t cluster_axis,
    const ttnn::MeshDevice& mesh_device,
    const ttnn::ccl::Topology topology,
    const CoreCoord ccl_core_grid_offset,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    const std::optional<float> scale,
    const std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    const ttnn::ccl::CoreAllocationStrategy core_allocation_strategy,
    const uint32_t block_size,
    const std::optional<uint32_t> q_blocks_per_pass) {
    using OperationType = ttnn::prim::SparseRingSDPADeviceOperation;

    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor_q.device()->arch(), compute_kernel_config, tt::tt_metal::MathFidelity::HiFi2, true, false, false);

    log_debug(
        tt::LogOp,
        "Launching SparseRingSDPA with core_allocation_strategy {}",
        enchantum::to_string(core_allocation_strategy));

    // The fused ring all-gather of K and V; a member of the op's attributes.
    const auto& mesh_view = mesh_device.get_view();
    TT_FATAL(
        mesh_view.is_mesh_2d(),
        "all-gather invoked with cluster_axis API without 2D mesh, which is currently unsupported");
    const int32_t rank = input_tensor_k.logical_shape().rank();
    const int32_t gather_dim = (dim < 0) ? rank + dim : dim;
    TT_FATAL(
        gather_dim >= -rank && gather_dim <= rank - 1,
        "Dimension input should be in between -{} and {}, but has {}",
        rank,
        rank - 1,
        dim);
    const auto& mesh_shape = mesh_device.shape();
    TT_FATAL(
        cluster_axis < mesh_shape.dims(),
        "SparseRingSDPA cluster_axis must be an in-range mesh axis; got {} for mesh {}",
        cluster_axis,
        mesh_shape);
    const std::size_t num_devices = cluster_axis == 0 ? mesh_view.num_rows() : mesh_view.num_cols();

    auto all_gather_operation_attributes = ttnn::experimental::prim::RingAttentionAllGatherAsyncParams{
        gather_dim,
        num_links,
        static_cast<uint32_t>(num_devices),
        persistent_output_buffer_k.memory_config(),
        topology,
        multi_device_global_semaphore,
        subdevice_id,
        cluster_axis,
        core_allocation_strategy};
    std::vector<Tensor> all_gather_input_tensors = {input_tensor_k, input_tensor_v};
    std::vector<std::optional<Tensor>> all_gather_output_tensors = {
        persistent_output_buffer_k, persistent_output_buffer_v};
    auto all_gather_tensor_args = ttnn::experimental::prim::RingAttentionAllGatherAsyncInputs{
        std::move(all_gather_input_tensors), std::move(all_gather_output_tensors)};

    auto operation_attributes = OperationType::operation_attributes_t(
        scale,
        logical_n,
        num_devices,
        tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        std::move(program_config),
        kernel_config_val,
        std::move(all_gather_operation_attributes),
        std::move(all_gather_tensor_args),
        ccl_core_grid_offset,
        block_size,
        q_blocks_per_pass);

    auto tensor_args = OperationType::tensor_args_t{
        .input_q = input_tensor_q,
        .input_k = input_tensor_k,
        .input_v = input_tensor_v,
        .gathered_k = persistent_output_buffer_k,
        .gathered_v = persistent_output_buffer_v,
        .block_mask = block_mask,
        .pooled_k = pooled_k,
        .pooled_v = pooled_v};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim

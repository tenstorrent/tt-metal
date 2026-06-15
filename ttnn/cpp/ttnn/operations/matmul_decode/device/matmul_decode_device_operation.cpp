// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_decode_device_operation.hpp"

#include "tt-metalium/math.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "tt-metalium/work_split.hpp"

namespace ttnn::operations::matmul_decode {

namespace {

// Dimensions recovered from a "partial width-sharded" B tensor.
//
// In this layout the caller has already reshaped + permuted B so that a [K, N]
// weight becomes a width-sharded tensor whose shard shape is [Kc, Nc] across
// K_blocks * N_blocks cores, where Kc = K / K_blocks and Nc = N / N_blocks.
// Because the reshape folds the K-block dimension into the (width-shardable) last
// dimension, the partial-sharded B's *logical* shape no longer matches [K, N];
// instead its height is Kc and its width is K_blocks * N_blocks * Nc. We therefore
// recover the real matmul dims from the shard spec plus A's K dimension.
struct PartialDims {
    int K_blocks;
    int N_blocks;
    int Kc;  // shard height (== K / K_blocks)
    int Nc;  // shard width  (== N / N_blocks)
    int N;   // recovered output width (== N_blocks * Nc)
};

// std::optional<PartialDims> detect_partial_width_sharded(const Tensor& input_tensor_a, const Tensor& input_tensor_b) {
//     const auto& b_mem = input_tensor_b.memory_config();
//     if (b_mem.memory_layout() != TensorMemoryLayout::WIDTH_SHARDED) {
//         return std::nullopt;
//     }
//     const auto& shard_spec = b_mem.shard_spec();
//     if (!shard_spec.has_value()) {
//         return std::nullopt;
//     }
//     const int K = input_tensor_a.logical_shape()[-1];
//     const int Kc = static_cast<int>(shard_spec->shape[0]);
//     const int Nc = static_cast<int>(shard_spec->shape[1]);
//     // Full width-sharded keeps the entire K dimension per shard (Kc == K); only treat
//     // B as partial when its shard height is a strict, even divisor of K.
//     if (Kc <= 0 || Kc >= K || K % Kc != 0) {
//         return std::nullopt;
//     }
//     const int K_blocks = K / Kc;
//     const int num_cores = static_cast<int>(shard_spec->grid.num_cores());
//     if (K_blocks <= 0 || num_cores % K_blocks != 0) {
//         return std::nullopt;
//     }
//     const int N_blocks = num_cores / K_blocks;
//     return PartialDims{K_blocks, N_blocks, Kc, Nc, N_blocks * Nc};
// }

}  // namespace

MatmulDecodeDeviceOperation::program_factory_t MatmulDecodeDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& /*tensor_args*/) {
    // The flag explicitly requests the partial factory; otherwise fall back to detecting
    // the partial layout from the inputs (B sharded along both K and N).
    if (operation_attributes.partial_width_sharded) {
        return PartialWidthSharded{};
    }
    return FullWidthSharded{};
}

void MatmulDecodeDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;

    TT_FATAL(input_tensor_a.layout() == Layout::TILE, "Input tensor A must be in tile layout");
    TT_FATAL(input_tensor_b.layout() == Layout::TILE, "Input tensor B must be in tile layout");
    TT_FATAL(
        input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED,
        "Input tensor A must be in width sharded memory layout, but got {}",
        input_tensor_a.memory_config().memory_layout());
    TT_FATAL(
        input_tensor_b.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED,
        "Input tensor B must be in width sharded memory layout, but got {}",
        input_tensor_b.memory_config().memory_layout());
    TT_FATAL(
        input_tensor_a.logical_shape()[-1] == operation_attributes.K,
        "Input tensor A must have the same K dimension as the operation attributes");
    TT_FATAL(
        input_tensor_a.logical_shape()[-2] == operation_attributes.M,
        "Input tensor A must have the same M dimension as the operation attributes");

    if (operation_attributes.partial_width_sharded) {
        // deep-plan_13 Phase 3/4: AUTHOR the HEIGHT/BLOCK validator (was a no-op stub).
        // Recover the 2D (K x N) block geometry from the reshaped/permuted B shard spec and
        // validate tile-alignment + N recovery. The phase-2 reduce now handles ARBITRARY
        // K_blocks (sec 6.4 fix), so even-K is NO LONGER a hard op-side FATAL -- the wrapper
        // still PREFERS even K_blocks (kb%2==0) as a perf/selector choice. HEIGHT = N_blocks==1.
        const auto& b_shard = input_tensor_b.memory_config().shard_spec();
        TT_FATAL(b_shard.has_value(), "partial_width_sharded matmul_decode requires a B shard spec");
        const uint32_t Kc = b_shard->shape[0];
        const uint32_t Nc = b_shard->shape[1];
        TT_FATAL(
            Kc % tt::constants::TILE_HEIGHT == 0 && Nc % tt::constants::TILE_WIDTH == 0,
            "partial_width_sharded B shard dims [{}, {}] must be tile-aligned",
            Kc,
            Nc);
        const uint32_t K_tiles = tt::div_up(operation_attributes.K, tt::constants::TILE_HEIGHT);
        const uint32_t Kc_tiles = Kc / tt::constants::TILE_HEIGHT;
        TT_FATAL(
            Kc_tiles > 0 && K_tiles % Kc_tiles == 0,
            "partial_width_sharded: Kc_tiles {} must evenly divide K_tiles {}",
            Kc_tiles,
            K_tiles);
        const uint32_t K_blocks = K_tiles / Kc_tiles;
        const uint32_t num_B_cores = b_shard->grid.num_cores();
        TT_FATAL(
            num_B_cores % K_blocks == 0,
            "partial_width_sharded: num_B_cores {} must be divisible by K_blocks {}",
            num_B_cores,
            K_blocks);
        const uint32_t N_blocks = num_B_cores / K_blocks;
        const uint32_t recovered_N = N_blocks * Nc;
        TT_FATAL(
            recovered_N == static_cast<uint32_t>(operation_attributes.N),
            "partial_width_sharded recovered N={} (N_blocks {} * Nc {}) does not match attribute N={}",
            recovered_N,
            N_blocks,
            Nc,
            operation_attributes.N);
        return;
    }

    // Full width-sharded B: each shard holds the full K dimension for its N-slice.
    if (input_tensor_a.logical_shape().rank() > 2) {
        for (int i = 0; i < input_tensor_a.logical_shape().rank() - 2; i++) {
            TT_FATAL(
                input_tensor_a.logical_shape()[i] == input_tensor_b.logical_shape()[i],
                "Input tensor A and B must have the same shape for all dimensions except the last two, but got {} and "
                "{}",
                input_tensor_a.logical_shape(),
                input_tensor_b.logical_shape());
        }
    }
    TT_FATAL(
        input_tensor_b.logical_shape()[-2] == operation_attributes.K,
        "Input tensor B must have the same K dimension as the operation attributes");
    TT_FATAL(
        input_tensor_b.logical_shape()[-1] == operation_attributes.N,
        "Input tensor B must have the same N dimension as the operation attributes");
}

MatmulDecodeDeviceOperation::spec_return_value_t MatmulDecodeDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor_a = tensor_args.input_tensor_a;

    // Output shape is the LHS shape with the last dim replaced by N. We use the
    // operation attribute N (rather than B's last logical dim) so that the partial
    // width-sharded layout -- whose reshaped/permuted B has a different logical
    // shape -- still produces a correct [..., M, N] output.
    ttnn::Shape output_shape(input_tensor_a.logical_shape());
    output_shape[-1] = operation_attributes.N;

    const auto dtype = operation_attributes.output_dtype.value_or(input_tensor_a.dtype());
    int n_tiles = tt::div_up(operation_attributes.N, tt::constants::TILE_WIDTH);
    int output_num_cores;
    int per_core_output_width;
    if (operation_attributes.partial_width_sharded) {
        // deep-plan_13 sec 7: HEIGHT/BLOCK output is sharded over EXACTLY N_blocks cores
        // (each holding Nc = N / N_blocks width). The partial factory recovers N_blocks from
        // the B shard geometry and FATALs unless output_core_range_set.num_cores() == N_blocks.
        // Recover it identically here: K_blocks = K_tiles / Kc_tiles, N_blocks = num_B_cores /
        // K_blocks. (HEIGHT => N_blocks==1 => the whole [M,N] output lands on one base core.)
        const auto& input_tensor_b = tensor_args.input_tensor_b;
        const auto& b_shard = input_tensor_b.memory_config().shard_spec().value();
        const uint32_t Kc_tiles = b_shard.shape[0] / tt::constants::TILE_WIDTH;
        const uint32_t Nc_tiles = b_shard.shape[1] / tt::constants::TILE_WIDTH;
        const uint32_t K_tiles = tt::div_up(operation_attributes.K, tt::constants::TILE_HEIGHT);
        const uint32_t K_blocks = K_tiles / Kc_tiles;
        const uint32_t num_B_cores = b_shard.grid.num_cores();
        const uint32_t N_blocks = num_B_cores / K_blocks;
        output_num_cores = static_cast<int>(N_blocks);
        per_core_output_width = static_cast<int>(Nc_tiles) * tt::constants::TILE_WIDTH;
    } else {
        // deep-plan_13 sec 4.6: wide-N output-core cap (recovered reverted feature). The bare
        // div_up(N,32) yields 512 cores for N=16384 -> FATALs past the ~110-core live grid AND
        // makes wide-N fat-fill (npc>1) impossible. Cap to the LARGEST DIVISOR of N_tiles that
        // is <= CORE_CAP (104) so the B shard grid == output grid (the full-WS inputB==output
        // FATAL), giving npc = N_tiles/out_cores N-tiles per core (which also feeds the fat
        // out_w fill). For N_tiles <= cap this is N_tiles cores @ npc=1 (byte-identical to the
        // pre-cap spec). Identical rule to the wrapper's _wide_n_out_cores.
        constexpr int CORE_CAP = 104;
        output_num_cores = n_tiles;
        if (n_tiles > CORE_CAP) {
            int chosen = 1;
            for (int c = 1; c <= CORE_CAP && c <= n_tiles; ++c) {
                if (n_tiles % c == 0) {
                    chosen = c;
                }
            }
            output_num_cores = chosen;
        }
        per_core_output_width = (n_tiles / output_num_cores) * tt::constants::TILE_WIDTH;
    }
    CoreRangeSet output_core_range_set = tt::tt_metal::num_cores_to_corerangeset(
        output_num_cores, input_tensor_a.device()->compute_with_storage_grid_size(), true);
    std::array<uint32_t, 2> shard_shape = {operation_attributes.M, per_core_output_width};
    auto shard_spec =
        tt::tt_metal::ShardSpec(output_core_range_set, shard_shape, tt::tt_metal::ShardOrientation::ROW_MAJOR);
    auto memory_config = MemoryConfig(TensorMemoryLayout::WIDTH_SHARDED, BufferType::L1, shard_spec);
    log_info(tt::LogOp, "matmul_decode with output memory_config: {}", memory_config);
    return TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout(dtype, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), memory_config));
}

MatmulDecodeDeviceOperation::tensor_return_value_t MatmulDecodeDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_spec, tensor_args.input_tensor_a.device());
}

}  // namespace ttnn::operations::matmul_decode

namespace ttnn::prim {
ttnn::operations::matmul_decode::MatmulDecodeDeviceOperation::tensor_return_value_t matmul_decode(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    bool partial_width_sharded,
    std::optional<const DataType> dtype,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    std::optional<uint32_t> out_subblock_h,
    std::optional<uint32_t> out_subblock_w,
    uint32_t in0_block_w,
    bool k_stream,
    uint32_t k_slice_tiles) {
    using OperationType = ttnn::operations::matmul_decode::MatmulDecodeDeviceOperation;

    // For the partial width-sharded layout the caller reshapes/permutes B, so its
    // last logical dim is no longer N; recover N from the shard spec in that case.
    int M, N, K;
    if (partial_width_sharded) {
        M = input_tensor_a.logical_shape()[-2];
        int K_a = input_tensor_a.logical_shape()[-1];
        int K_b = input_tensor_b.logical_shape()[-2];
        N = input_tensor_b.logical_shape()[-1];
        if (K_a >= K_b) {
            TT_FATAL(K_a % K_b == 0, "K_a must be divisible by K_b");
            int K_ratio = K_a / K_b;
            N = N / K_ratio;
        }
        K = K_a;
    } else {
        M = input_tensor_a.logical_shape()[-2];
        N = input_tensor_b.logical_shape()[-1];
        K = input_tensor_a.logical_shape()[-1];
    }
    log_info(tt::LogOp, "matmul_decode partial_width_sharded={} with M={}, N={}, K={}", partial_width_sharded, M, N, K);
    // Resolve the (optional) user compute kernel config into a concrete one, mirroring
    // ttnn::matmul. Defaults: math_fidelity=HiFi4 (the op's established fidelity floor),
    // math_approx_mode=false, and fp32_dest_acc_en=false (OPT-IN -- the precision boost is
    // off unless the caller passes a config with fp32_dest_acc_en=true). packer_l1_acc /
    // dst_full_sync default false; the ComputeConfigDescriptor used by the factories only
    // consumes math_fidelity / fp32_dest_acc_en / math_approx_mode / dst_full_sync_en.
    auto resolved_compute_kernel_config = ttnn::init_device_compute_kernel_config(
        input_tensor_a.device()->arch(),
        compute_kernel_config,
        /*default_fidelity=*/MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/false,
        /*default_l1_acc=*/false);
    auto operation_attributes = OperationType::operation_attributes_t{
        M,
        N,
        K,
        input_tensor_a.memory_config(),
        dtype.has_value() ? std::optional<DataType>(*dtype) : std::nullopt,
        partial_width_sharded,
        // deep-plan_14 Lever 0: thread the explicit knobs from the entry instead of
        // hardcoding nullopt. Defaults (nullopt/1/false/0) keep the auto-derived,
        // out_w-only, one-shot path byte-identical to deep-plan_13.
        out_subblock_h,
        out_subblock_w,
        in0_block_w,
        k_stream,
        k_slice_tiles,
        resolved_compute_kernel_config,
    };
    auto tensor_args = OperationType::tensor_args_t{input_tensor_a, input_tensor_b};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}
}  // namespace ttnn::prim

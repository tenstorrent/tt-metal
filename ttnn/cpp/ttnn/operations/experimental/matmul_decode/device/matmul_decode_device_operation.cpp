// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_decode_device_operation.hpp"

#include "tt-metalium/math.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "tt-metalium/work_split.hpp"

namespace ttnn::operations::experimental::matmul_decode {

MatmulDecodeDeviceOperation::program_factory_t MatmulDecodeDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.input_tensor_a.logical_shape().rank() == 4 && operation_attributes.batch > 1) {
        return BatchedWidthSharded{};
    }
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

    if (input_tensor_a.logical_shape().rank() == 4 && operation_attributes.batch > 1) {
        const int batch = operation_attributes.batch;
        const int b_blocks = operation_attributes.b_blocks;
        const int n_blocks = operation_attributes.n_blocks;

        TT_FATAL(
            input_tensor_a.logical_shape()[0] * input_tensor_a.logical_shape()[1] == batch,
            "Batched matmul_decode expects A leading dims {} x {} to multiply to the operation batch {}",
            input_tensor_a.logical_shape()[0],
            input_tensor_a.logical_shape()[1],
            batch);
        // A real batch (> 1) requires rank-4 weights carrying the same batch size.
        if (batch > 1) {
            TT_FATAL(
                input_tensor_b.logical_shape().rank() == 4,
                "batched matmul_decode with batch {} > 1 requires rank-4 weights, but got rank {}",
                batch,
                input_tensor_b.logical_shape().rank());
        }
        TT_FATAL(
            b_blocks > 0 && batch % b_blocks == 0,
            "Batched matmul_decode requires b_blocks {} to be a positive divisor of the batch {}",
            b_blocks,
            batch);
        TT_FATAL(
            n_blocks > 0 && operation_attributes.N % n_blocks == 0,
            "Batched matmul_decode requires n_blocks {} to be a positive divisor of N {}",
            n_blocks,
            operation_attributes.N);
        const int Bc = batch / b_blocks;
        const int Nc = operation_attributes.N / n_blocks;

        const auto& a_tile = input_tensor_a.tensor_spec().tile();
        const uint32_t a_tile_height = a_tile.get_height();
        const int M_tiles = tt::div_up(operation_attributes.M, static_cast<int>(a_tile_height));

        TT_FATAL(
            M_tiles <= 8,
            "batched matmul_decode requires M_tiles (= ceil(M / tile_height)) <= 8 so the output block fits in "
            "DST, but got M_tiles={} (M={}, tile_height={})",
            M_tiles,
            operation_attributes.M,
            a_tile_height);

        const auto& a_shard = input_tensor_a.memory_config().shard_spec().value();
        TT_FATAL(
            a_shard.shape[1] % tt::constants::TILE_WIDTH == 0,
            "Input tensor A shard width {} must be divisible by the tile width {}",
            a_shard.shape[1],
            tt::constants::TILE_WIDTH);

        const auto& b_shard = input_tensor_b.memory_config().shard_spec().value();
        const uint32_t b_shard_h = b_shard.shape[0];
        const uint32_t b_shard_w = b_shard.shape[1];
        TT_FATAL(
            b_shard_h % tt::constants::TILE_HEIGHT == 0 && b_shard_w % tt::constants::TILE_WIDTH == 0,
            "batched matmul_decode requires B shard dims [{}, {}] to be tile-aligned (tile {}x{})",
            b_shard_h,
            b_shard_w,
            tt::constants::TILE_HEIGHT,
            tt::constants::TILE_WIDTH);
        TT_FATAL(
            b_shard_h == static_cast<uint32_t>(Bc) * operation_attributes.K,
            "batched matmul_decode expects B shard height {} to equal Bc * K = {} * {} = {}",
            b_shard_h,
            Bc,
            operation_attributes.K,
            Bc * operation_attributes.K);
        TT_FATAL(
            b_shard_w == static_cast<uint32_t>(Nc),
            "batched matmul_decode expects B shard width {} to equal Nc = N / n_blocks = {}",
            b_shard_w,
            Nc);

        const int num_B_cores = static_cast<int>(b_shard.grid.num_cores());
        TT_FATAL(
            num_B_cores == b_blocks * n_blocks,
            "batched matmul_decode expects B sharded across b_blocks * n_blocks = {} * {} = {} cores, but got {}",
            b_blocks,
            n_blocks,
            b_blocks * n_blocks,
            num_B_cores);

        TT_FATAL(
            input_tensor_b.logical_shape()[-2] == Bc * operation_attributes.K,
            "batched matmul_decode expects B logical height {} to equal Bc * K = {} * {} = {}",
            input_tensor_b.logical_shape()[-2],
            Bc,
            operation_attributes.K,
            Bc * operation_attributes.K);
        TT_FATAL(
            input_tensor_b.logical_shape()[-1] == b_blocks * operation_attributes.N,
            "batched matmul_decode expects B logical width {} to equal b_blocks * N = {} * {} = {} (the weights are "
            "reshaped/permuted so the batch-blocks fold into the width)",
            input_tensor_b.logical_shape()[-1],
            b_blocks,
            operation_attributes.N,
            b_blocks * operation_attributes.N);
        return;
    }

    if (operation_attributes.partial_width_sharded) {
        const auto& a_tile = input_tensor_a.tensor_spec().tile();
        const uint32_t a_tile_height = a_tile.get_height();
        const int M_tiles = tt::div_up(operation_attributes.M, static_cast<int>(a_tile_height));
        const int K_tiles = tt::div_up(operation_attributes.K, static_cast<int>(tt::constants::TILE_HEIGHT));
        const int N_tiles = tt::div_up(operation_attributes.N, static_cast<int>(tt::constants::TILE_WIDTH));

        TT_FATAL(
            M_tiles <= 8,
            "partial_width_sharded matmul_decode requires M_tiles (= ceil(M / tile_height)) <= 8 so the output "
            "block fits in DST, but got M_tiles={} (M={}, tile_height={})",
            M_tiles,
            operation_attributes.M,
            a_tile_height);

        const auto& a_shard = input_tensor_a.memory_config().shard_spec().value();
        TT_FATAL(
            a_shard.shape[0] == static_cast<uint32_t>(M_tiles) * a_tile_height,
            "Input tensor A shard height {} must equal M_tiles {} * tile height {}",
            a_shard.shape[0],
            M_tiles,
            a_tile_height);
        TT_FATAL(
            a_shard.shape[1] % tt::constants::TILE_WIDTH == 0,
            "Input tensor A shard width {} must be divisible by the tile width {}",
            a_shard.shape[1],
            tt::constants::TILE_WIDTH);

        const auto& b_shard = input_tensor_b.memory_config().shard_spec().value();
        const uint32_t Kc = b_shard.shape[0];
        const uint32_t Nc = b_shard.shape[1];
        TT_FATAL(
            Kc % tt::constants::TILE_HEIGHT == 0 && Nc % tt::constants::TILE_WIDTH == 0,
            "partial_width_sharded matmul_decode requires B shard dims [{}, {}] to be tile-aligned (tile {}x{})",
            Kc,
            Nc,
            tt::constants::TILE_HEIGHT,
            tt::constants::TILE_WIDTH);
        const int Kc_tiles = static_cast<int>(Kc) / tt::constants::TILE_HEIGHT;
        const int Nc_tiles = static_cast<int>(Nc) / tt::constants::TILE_WIDTH;

        TT_FATAL(
            K_tiles % Kc_tiles == 0,
            "partial_width_sharded matmul_decode requires K_tiles {} to be divisible by the B shard height in "
            "tiles {} (Kc={})",
            K_tiles,
            Kc_tiles,
            Kc);
        const int K_blocks = K_tiles / Kc_tiles;
        // K_blocks must be even: base-core reduction sums partials pairwise.
        TT_FATAL(
            K_blocks % 2 == 0,
            "partial_width_sharded matmul_decode requires an even number of K-blocks (the cross-core reduction "
            "sums partials pairwise), but got K_blocks={}",
            K_blocks);

        TT_FATAL(
            N_tiles % Nc_tiles == 0,
            "partial_width_sharded matmul_decode requires N_tiles {} to be divisible by the B shard width in "
            "tiles {} (Nc={})",
            N_tiles,
            Nc_tiles,
            Nc);
        const int N_blocks = N_tiles / Nc_tiles;

        const int num_B_cores = static_cast<int>(b_shard.grid.num_cores());
        TT_FATAL(
            num_B_cores == K_blocks * N_blocks,
            "partial_width_sharded matmul_decode expects B sharded across K_blocks * N_blocks = {} * {} = {} "
            "cores, but got {}",
            K_blocks,
            N_blocks,
            K_blocks * N_blocks,
            num_B_cores);

        TT_FATAL(
            input_tensor_b.logical_shape()[-2] == static_cast<int>(Kc),
            "partial_width_sharded matmul_decode expects B logical height {} to equal the shard height Kc={}",
            input_tensor_b.logical_shape()[-2],
            Kc);
        TT_FATAL(
            input_tensor_b.logical_shape()[-1] == K_blocks * operation_attributes.N,
            "partial_width_sharded matmul_decode expects B logical width {} to equal K_blocks * N = {} * {} = {} "
            "(B is reshaped/permuted so the K-blocks fold into the width)",
            input_tensor_b.logical_shape()[-1],
            K_blocks,
            operation_attributes.N,
            K_blocks * operation_attributes.N);
        return;
    }

    if (input_tensor_a.logical_shape().rank() > 2 && input_tensor_b.logical_shape().rank() > 2) {
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
    const auto& input_tensor_b = tensor_args.input_tensor_b;
    // Use operation N (not B's logical last dim) so folded partial-B layouts still yield [..., M, N].
    ttnn::Shape output_shape(input_tensor_a.logical_shape());
    output_shape[-1] = operation_attributes.N;

    const auto dtype = operation_attributes.output_dtype.value_or(input_tensor_a.dtype());

    if (input_tensor_a.logical_shape().rank() == 4) {
        const auto memory_config = operation_attributes.output_mem_config.value_or(
            MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM));

        return TensorSpec(
            output_shape,
            tt::tt_metal::TensorLayout(
                dtype,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE, input_tensor_a.tensor_spec().tile()),
                memory_config));
    }

    CoreRangeSet output_core_range_set = input_tensor_b.memory_config().shard_spec().value().grid;
    int output_num_cores = output_core_range_set.num_cores();
    if (operation_attributes.partial_width_sharded) {
        const auto& b_shard_spec = input_tensor_b.memory_config().shard_spec().value();
        const int N_tiles = tt::div_up(operation_attributes.N, tt::constants::TILE_WIDTH);
        const int Nc_tiles = static_cast<int>(b_shard_spec.shape[1]) / tt::constants::TILE_WIDTH;
        const int N_blocks = N_tiles / Nc_tiles;
        output_num_cores = N_blocks;
        output_core_range_set = tt::tt_metal::num_cores_to_corerangeset(
            output_num_cores, input_tensor_a.device()->compute_with_storage_grid_size(), true);
    }
    int per_core_output_width = tt::div_up(operation_attributes.N, output_num_cores);
    const uint32_t shard_height =
        tt::round_up(operation_attributes.M, input_tensor_a.tensor_spec().tile().get_height());
    std::array<uint32_t, 2> shard_shape = {shard_height, per_core_output_width};
    auto shard_spec =
        tt::tt_metal::ShardSpec(output_core_range_set, shard_shape, tt::tt_metal::ShardOrientation::ROW_MAJOR);
    auto memory_config = operation_attributes.output_mem_config.value_or(
        MemoryConfig(TensorMemoryLayout::WIDTH_SHARDED, BufferType::L1, shard_spec));

    return TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout(
            dtype,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE, input_tensor_a.tensor_spec().tile()),
            memory_config));
}

MatmulDecodeDeviceOperation::tensor_return_value_t MatmulDecodeDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_spec, tensor_args.input_tensor_a.device());
}

}  // namespace ttnn::operations::experimental::matmul_decode

namespace ttnn::prim {
ttnn::operations::experimental::matmul_decode::MatmulDecodeDeviceOperation::tensor_return_value_t matmul_decode(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    bool partial_width_sharded,
    std::optional<const DataType> dtype,
    const std::optional<MemoryConfig>& output_mem_config) {
    using OperationType = ttnn::operations::experimental::matmul_decode::MatmulDecodeDeviceOperation;

    if (input_tensor_a.logical_shape().rank() == 4) {
        const int batch = input_tensor_a.logical_shape()[0] * input_tensor_a.logical_shape()[1];
        const int M = input_tensor_a.logical_shape()[-2];
        const int K = input_tensor_a.logical_shape()[-1];
        // A real batch (> 1) requires rank-4 weights carrying the same batch size.
        if (batch > 1) {
            TT_FATAL(
                input_tensor_b.logical_shape().rank() == 4,
                "batched matmul_decode with batch {} > 1 requires rank-4 weights, but got rank {}",
                batch,
                input_tensor_b.logical_shape().rank());
            const int weight_height = input_tensor_b.logical_shape()[-2];  // = Bc * K
            const int weight_width = input_tensor_b.logical_shape()[-1];   // = b_blocks * N
            TT_FATAL(
                K > 0 && weight_height % K == 0,
                "batched matmul_decode: weight height {} must be a multiple of K {} (weight height = Bc * K)",
                weight_height,
                K);
            const int Bc = weight_height / K;
            TT_FATAL(
                Bc > 0 && batch % Bc == 0,
                "batched matmul_decode: batch {} must be a multiple of Bc {} (Bc = weight_height / K)",
                batch,
                Bc);
            const int b_blocks = batch / Bc;
            TT_FATAL(
                weight_width % b_blocks == 0,
                "batched matmul_decode: weight width {} must be a multiple of b_blocks {} (weight width = b_blocks * "
                "N)",
                weight_width,
                b_blocks);
            const int N = weight_width / b_blocks;
            const int num_B_cores =
                static_cast<int>(input_tensor_b.memory_config().shard_spec().value().grid.num_cores());
            TT_FATAL(
                num_B_cores % b_blocks == 0,
                "batched matmul_decode: number of weight cores {} must be a multiple of b_blocks {}",
                num_B_cores,
                b_blocks);
            const int n_blocks = num_B_cores / b_blocks;
            log_debug(
                tt::LogOp,
                "matmul_decode (batched) batch={}, M={}, N={}, K={}, Bc={}, b_blocks={}, n_blocks={}",
                batch,
                M,
                N,
                K,
                Bc,
                b_blocks,
                n_blocks);
            auto operation_attributes = OperationType::operation_attributes_t{
                M,
                N,
                K,
                output_mem_config,
                dtype.has_value() ? std::optional<DataType>(*dtype) : std::nullopt,
                /*partial_width_sharded=*/false,
                batch,
                b_blocks,
                n_blocks,
            };
            auto tensor_args = OperationType::tensor_args_t{input_tensor_a, input_tensor_b};
            return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
        }
    }

    int M, N, K;
    if (partial_width_sharded) {
        // Folded B logical width is K_blocks * N; recover true N from K_a / K_b.
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
    log_debug(
        tt::LogOp, "matmul_decode partial_width_sharded={} with M={}, N={}, K={}", partial_width_sharded, M, N, K);
    auto operation_attributes = OperationType::operation_attributes_t{
        M,
        N,
        K,
        output_mem_config,
        dtype.has_value() ? std::optional<DataType>(*dtype) : std::nullopt,
        partial_width_sharded,
    };
    auto tensor_args = OperationType::tensor_args_t{input_tensor_a, input_tensor_b};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}
}  // namespace ttnn::prim

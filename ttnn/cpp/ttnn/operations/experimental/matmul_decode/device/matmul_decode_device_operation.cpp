// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_decode_device_operation.hpp"

#include "tt-metalium/math.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "tt-metalium/work_split.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"

namespace ttnn::operations::experimental::matmul_decode {

namespace {
uint32_t gcb_num_receivers(const tt::tt_metal::experimental::GlobalCircularBuffer& gcb) {
    return gcb.receiver_cores().num_cores();
}
}  // namespace

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

    // Mirrors select_program_factory so the geometry validated here is always the geometry the
    // chosen factory will consume.
    const bool batched = input_tensor_a.logical_shape().rank() == 4 && operation_attributes.batch > 1;
    const bool partial = !batched && operation_attributes.partial_width_sharded;

    if (operation_attributes.all_gather) {
        TT_FATAL(
            operation_attributes.ring_size > 1,
            "matmul_decode all_gather requires a multi-device mesh, but the input mesh has {} device(s)",
            operation_attributes.ring_size);
        TT_FATAL(
            !operation_attributes.global_cb.has_value(),
            "matmul_decode all_gather is not supported with global_cb (tensor prefetcher)");
        TT_FATAL(!batched, "matmul_decode all_gather is not supported with the batched width-sharded factory");
    }

    TT_FATAL(input_tensor_a.layout() == Layout::TILE, "Input tensor A must be in tile layout");
    TT_FATAL(input_tensor_b.layout() == Layout::TILE, "Input tensor B must be in tile layout");
    TT_FATAL(
        input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED,
        "Input tensor A must be in width sharded memory layout, but got {}",
        input_tensor_a.memory_config().memory_layout());
    if (operation_attributes.packed_weight.has_value()) {
        // Fused-weight path: B is one big height-sharded L1 tensor carrying many weights, so
        // nothing about this weight can be read off B's shape or shard spec. Everything the
        // legacy checks below would derive from B is instead validated against the spec here,
        // and the rest of this function is skipped -- its B checks would reject the fused
        // tensor. A's own checks are shared with the factories (they re-verify M_tiles/shard
        // geometry on the packed path too).
        const auto& pw = *operation_attributes.packed_weight;
        const auto& tile = input_tensor_b.tensor_spec().tile();
        const uint32_t tile_h = tile.get_height();
        const uint32_t tile_w = tile.get_width();

        TT_FATAL(
            input_tensor_b.buffer()->buffer_type() == tt::tt_metal::BufferType::L1,
            "matmul_decode with packed_weight requires the fused weight tensor to be L1-resident");
        TT_FATAL(
            input_tensor_b.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
            "matmul_decode with packed_weight requires the fused weight tensor to be HEIGHT_SHARDED "
            "(one equal shard per core), but got {}",
            input_tensor_b.memory_config().memory_layout());
        const auto& b_shard = input_tensor_b.memory_config().shard_spec().value();
        TT_FATAL(
            b_shard.shape[1] == tile_w && b_shard.shape[0] % tile_h == 0,
            "matmul_decode with packed_weight requires one-tile-wide, tile-aligned shards (a shard is a "
            "stack of tiles), but the fused tensor's shard is [{}, {}] with tile {}x{}",
            b_shard.shape[0],
            b_shard.shape[1],
            tile_h,
            tile_w);
        TT_FATAL(
            b_shard.grid.contains(pw.cores),
            "matmul_decode with packed_weight requires the weight's cores {} to be covered by the fused "
            "tensor's shard grid {}",
            pw.cores.str(),
            b_shard.grid.str());

        TT_FATAL(
            static_cast<int>(pw.K) == operation_attributes.K && static_cast<int>(pw.N) == operation_attributes.N,
            "packed_weight [K, N] = [{}, {}] does not match the operation's [{}, {}]",
            pw.K,
            pw.N,
            operation_attributes.K,
            operation_attributes.N);
        TT_FATAL(
            pw.K % tt::constants::TILE_HEIGHT == 0 && pw.N % tt::constants::TILE_WIDTH == 0,
            "packed_weight [K, N] = [{}, {}] must be tile-aligned",
            pw.K,
            pw.N);

        // The slab shape the factory will consume, by mode; also pins down the core count.
        const uint32_t n_blocks = pw.n_blocks();
        TT_FATAL(
            n_blocks > 0 && pw.N % n_blocks == 0 && (pw.N / n_blocks) % tt::constants::TILE_WIDTH == 0,
            "packed_weight: N ({}) does not cut into {} tile-aligned N-blocks on {} cores",
            pw.N,
            n_blocks,
            pw.num_cores());
        uint32_t slab_rows = 0;
        if (batched) {
            TT_FATAL(
                static_cast<int>(pw.batch) == operation_attributes.batch &&
                    static_cast<int>(pw.b_blocks) == operation_attributes.b_blocks,
                "packed_weight batch/b_blocks ({}/{}) do not match the operation's ({}/{})",
                pw.batch,
                pw.b_blocks,
                operation_attributes.batch,
                operation_attributes.b_blocks);
            TT_FATAL(
                pw.b_blocks > 0 && pw.batch % pw.b_blocks == 0 && pw.num_cores() == pw.b_blocks * n_blocks,
                "packed_weight: batch {} on {} cores does not cut into b_blocks {} x n_blocks {}",
                pw.batch,
                pw.num_cores(),
                pw.b_blocks,
                n_blocks);
            slab_rows = (pw.batch / pw.b_blocks) * pw.K;
        } else if (partial) {
            TT_FATAL(
                pw.k_blocks > 1 && pw.K % pw.k_blocks == 0 && (pw.K / pw.k_blocks) % tt::constants::TILE_HEIGHT == 0 &&
                    pw.num_cores() == pw.k_blocks * n_blocks,
                "packed_weight: K {} on {} cores does not cut into k_blocks {} x n_blocks {} tile-aligned blocks",
                pw.K,
                pw.num_cores(),
                pw.k_blocks,
                n_blocks);
            slab_rows = pw.K / pw.k_blocks;
        } else {
            slab_rows = pw.K;
        }
        const uint32_t slab_tiles =
            (slab_rows / tt::constants::TILE_HEIGHT) * (pw.N / n_blocks / tt::constants::TILE_WIDTH);
        const uint32_t shard_tiles = b_shard.shape[0] / tile_h;
        TT_FATAL(
            pw.tile_offset + slab_tiles <= shard_tiles,
            "packed_weight region [{}, {}) does not fit in the fused tensor's {}-tile shard",
            pw.tile_offset,
            pw.tile_offset + slab_tiles,
            shard_tiles);

        TT_FATAL(
            input_tensor_a.logical_shape()[-1] == operation_attributes.K,
            "Input tensor A must have the same K dimension as the packed weight");
        TT_FATAL(
            input_tensor_a.logical_shape()[-2] == operation_attributes.M,
            "Input tensor A must have the same M dimension as the operation attributes");
        return;
    }
    if (operation_attributes.global_cb.has_value()) {
        // Prefetcher-fed weights live in DRAM as an ND-sharded (receiver-contiguous) tensor:
        // one contiguous slab per receiver core, whose shape depends on the factory that will
        // consume it. The receiver grid comes from the GCB, not from a legacy shard spec.
        //
        // memory_layout() is deliberately not checked: TensorSpec back-fills an equivalent legacy
        // spec whenever the ND spec happens to be expressible as one (a full-height ROUND_ROBIN_1D
        // shard whose shard count fits the DRAM grid, i.e. num_receivers <= num_dram_banks), which
        // reports the layout as WIDTH_SHARDED even though the tensor was created ND-sharded. The
        // NdShardSpec below is the property this path actually needs.
        TT_FATAL(
            input_tensor_b.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM,
            "matmul_decode with global_cb requires input tensor B to live in DRAM (the prefetcher reads DRAM), "
            "but it is in L1");
        const int num_receivers = static_cast<int>(gcb_num_receivers(*operation_attributes.global_cb));
        TT_FATAL(num_receivers > 0, "matmul_decode with global_cb requires the GCB to have at least one receiver core");
        // Note: the NdShardSpec lives on the Tensor, not on the MemoryConfig, and the shard count
        // comes from the buffer's BufferDistributionSpec -- same accessors the recv-contig weight
        // validator in ttnn/core/global_circular_buffer.cpp uses.
        const auto& nd = input_tensor_b.nd_shard_spec();
        TT_FATAL(
            nd.has_value(),
            "matmul_decode with global_cb requires input tensor B to carry an NdShardSpec (receiver-contiguous "
            "layout)");
        const auto& bds = input_tensor_b.buffer()->buffer_distribution_spec();
        TT_FATAL(
            bds.has_value(), "matmul_decode with global_cb requires input tensor B to have a BufferDistributionSpec");
        const int num_shards = static_cast<int>(bds->num_shards());
        TT_FATAL(
            num_shards == num_receivers,
            "matmul_decode with global_cb requires one weight shard per GCB receiver, but the weight has {} shards "
            "and the GCB has {} receivers",
            num_shards,
            num_receivers);

        const int K = operation_attributes.K;
        const int N = operation_attributes.N;
        const int slab_h = static_cast<int>(nd->shard_shape[-2]);
        const int slab_w = static_cast<int>(nd->shard_shape[-1]);
        if (batched) {
            const int b_blocks = operation_attributes.b_blocks;
            const int n_blocks = operation_attributes.n_blocks;
            TT_FATAL(
                b_blocks > 0 && n_blocks > 0 && operation_attributes.batch % b_blocks == 0 && N % n_blocks == 0,
                "batched matmul_decode with global_cb requires b_blocks ({}) and n_blocks ({}) to be positive "
                "divisors of the batch ({}) and N ({})",
                b_blocks,
                n_blocks,
                operation_attributes.batch,
                N);
            const int Bc = operation_attributes.batch / b_blocks;
            const int Nc = N / n_blocks;
            TT_FATAL(
                slab_h == Bc * K && slab_w == Nc,
                "batched matmul_decode with global_cb requires each weight shard to be [Bc*K, Nc] = [{}, {}], but "
                "got [{}, {}]",
                Bc * K,
                Nc,
                slab_h,
                slab_w);
        } else if (partial) {
            TT_FATAL(
                slab_h > 0 && slab_w > 0 && K % slab_h == 0 && N % slab_w == 0,
                "partial_width_sharded matmul_decode with global_cb requires each weight shard to be [Kc, Nc] with "
                "Kc dividing K ({}) and Nc dividing N ({}), but got [{}, {}]",
                K,
                N,
                slab_h,
                slab_w);
        } else {
            TT_FATAL(
                N % num_receivers == 0,
                "matmul_decode with global_cb requires N ({}) to be divisible by the GCB receiver count ({})",
                N,
                num_receivers);
            TT_FATAL(
                slab_h == K && slab_w == N / num_receivers,
                "full width-sharded matmul_decode with global_cb requires each weight shard to be [K, "
                "N/num_receivers] = [{}, {}], but got [{}, {}]",
                K,
                N / num_receivers,
                slab_h,
                slab_w);
        }
    } else {
        TT_FATAL(
            input_tensor_b.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED,
            "Input tensor B must be in width sharded memory layout, but got {}",
            input_tensor_b.memory_config().memory_layout());
    }
    TT_FATAL(
        input_tensor_a.logical_shape()[-1] == operation_attributes.K,
        "Input tensor A must have the same K dimension as the operation attributes");
    TT_FATAL(
        input_tensor_a.logical_shape()[-2] == operation_attributes.M,
        "Input tensor A must have the same M dimension as the operation attributes");

    if (batched) {
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

        // A prefetcher weight carries no legacy shard spec; the GCB branch above has already
        // checked the equivalent slab shape and block count against the receiver grid.
        if (!operation_attributes.global_cb.has_value()) {
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
        }

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

    if (partial) {
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

        const bool b_from_gcb = operation_attributes.global_cb.has_value();
        // A prefetcher weight carries no legacy shard spec, so its [Kc, Nc] slab comes from the ND
        // shard shape the GCB branch above validated.
        const uint32_t Kc = b_from_gcb ? static_cast<uint32_t>(input_tensor_b.nd_shard_spec()->shard_shape[-2])
                                       : input_tensor_b.memory_config().shard_spec().value().shape[0];
        const uint32_t Nc = b_from_gcb ? static_cast<uint32_t>(input_tensor_b.nd_shard_spec()->shard_shape[-1])
                                       : input_tensor_b.memory_config().shard_spec().value().shape[1];
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

        // The prefetcher weight is a plain [K, N] tensor cut into slabs by its ND shard spec, not
        // the K-block-folded L1 layout, and its block count was checked against the GCB above.
        if (!b_from_gcb) {
            const int num_B_cores =
                static_cast<int>(input_tensor_b.memory_config().shard_spec().value().grid.num_cores());
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
        }
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

ttsl::hash::hash_t MatmulDecodeDeviceOperation::compute_program_hash(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    // GlobalCircularBuffer hashes on (sender_receiver_core_mapping, size, buffer_type) and not on its
    // address, so two identically shaped GCBs are indistinguishable to the default reflection hash.
    // The program bakes in this GCB's addresses, so without folding an address in, a second call with
    // a different same-shaped GCB reuses a program pointing at the first one and hangs waiting for
    // credits written into the other buffer. Hashing `attributes` and `tensor_args` wholesale keeps
    // this additive: both still cover every member (and any added later) exactly as the default hash
    // did. Both GCB addresses are folded in because the program depends on both, and the config and
    // data buffers are independent allocations -- a replacement GCB could reuse one address without
    // reusing the other.
    const auto gcb_identity =
        attributes.global_cb.has_value()
            ? std::make_pair(attributes.global_cb->config_address(), attributes.global_cb->buffer_address())
            : std::make_pair(tt::tt_metal::DeviceAddr{0}, tt::tt_metal::DeviceAddr{0});
    return tt::tt_metal::operation::hash_operation<MatmulDecodeDeviceOperation>(attributes, gcb_identity, tensor_args);
}

MatmulDecodeDeviceOperation::spec_return_value_t MatmulDecodeDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;
    // Use operation N (not B's logical last dim) so folded partial-B layouts still yield [..., M, N].
    // all_gather concatenates each device's N-shard, so the returned width is N * ring_size.
    const int output_N = operation_attributes.all_gather
                             ? operation_attributes.N * static_cast<int>(operation_attributes.ring_size)
                             : operation_attributes.N;
    ttnn::Shape output_shape(input_tensor_a.logical_shape());
    output_shape[-1] = output_N;

    const auto dtype = operation_attributes.output_dtype.value_or(input_tensor_a.dtype());

    if (input_tensor_a.logical_shape().rank() == 4) {
        const auto memory_config = operation_attributes.output_mem_config.value_or(
            MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM));

        return tt::tt_metal::TensorSpec(
            output_shape,
            tt::tt_metal::TensorLayout(
                dtype,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE, input_tensor_a.tensor_spec().tile()),
                memory_config));
    }

    // Neither a prefetcher-fed weight (ND-sharded in DRAM) nor a packed weight (a region of the
    // fused L1 tensor) carries a usable legacy shard spec, so the weight-holding (= output) grid
    // comes from the GCB's receivers or the packed spec's cores instead.
    const bool is_packed = operation_attributes.packed_weight.has_value();
    CoreRangeSet output_core_range_set = operation_attributes.global_cb.has_value()
                                             ? operation_attributes.global_cb->receiver_cores()
                                         : is_packed ? operation_attributes.packed_weight->cores
                                                     : input_tensor_b.memory_config().shard_spec().value().grid;
    int output_num_cores = output_core_range_set.num_cores();
    if (operation_attributes.partial_width_sharded) {
        const int Nc = operation_attributes.global_cb.has_value()
                           ? static_cast<int>(input_tensor_b.nd_shard_spec().value().shard_shape[-1])
                       : is_packed
                           ? static_cast<int>(operation_attributes.N / operation_attributes.packed_weight->n_blocks())
                           : static_cast<int>(input_tensor_b.memory_config().shard_spec().value().shape[1]);
        const int N_tiles = tt::div_up(operation_attributes.N, tt::constants::TILE_WIDTH);
        const int Nc_tiles = Nc / tt::constants::TILE_WIDTH;
        const int N_blocks = N_tiles / Nc_tiles;
        output_num_cores = N_blocks;
        if (operation_attributes.global_cb.has_value() || is_packed) {
            // The factory reduces the K-partials onto the k_idx == 0 row of the weight grid --
            // its first N_blocks cores in row-major order -- and requires every one of them to be
            // in the output grid. A grid anchored at (0, 0) only satisfies that when the weight
            // grid happens to be anchored there too.
            const auto base_cores =
                tt::tt_metal::corerange_to_cores(output_core_range_set, output_num_cores, /*row_wise=*/true);
            output_core_range_set = CoreRangeSet(base_cores);
        } else {
            output_core_range_set = tt::tt_metal::num_cores_to_corerangeset(
                output_num_cores, input_tensor_a.device()->compute_with_storage_grid_size(), true);
        }
    }
    int per_core_output_width = tt::div_up(output_N, output_num_cores);
    const uint32_t shard_height =
        tt::round_up(operation_attributes.M, input_tensor_a.tensor_spec().tile().get_height());
    std::array<uint32_t, 2> shard_shape = {shard_height, per_core_output_width};
    auto shard_spec =
        tt::tt_metal::ShardSpec(output_core_range_set, shard_shape, tt::tt_metal::ShardOrientation::ROW_MAJOR);
    auto memory_config = operation_attributes.output_mem_config.value_or(
        MemoryConfig(TensorMemoryLayout::WIDTH_SHARDED, BufferType::L1, shard_spec));

    return tt::tt_metal::TensorSpec(
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
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb,
    uint32_t global_cb_k_blocks,
    const std::optional<ttnn::operations::experimental::matmul_decode::PackedWeightSpec>& packed_weight,
    bool all_gather) {
    using OperationType = ttnn::operations::experimental::matmul_decode::MatmulDecodeDeviceOperation;
    using ttnn::operations::experimental::matmul_decode::gcb_num_receivers;

    auto with_all_gather = [&](OperationType::operation_attributes_t attrs) {
        attrs.all_gather = all_gather;
        if (all_gather) {
            attrs.ring_size = ::ttnn::ccl::get_topological_dimension(input_tensor_a, std::nullopt);
            TT_FATAL(
                attrs.ring_size > 1,
                "matmul_decode all_gather requires a multi-device mesh, but the input mesh has {} device(s)",
                attrs.ring_size);
        }
        return attrs;
    };

    // `compute_output_specs` runs before `validate_on_program_cache_miss` and already reads the
    // weight's ND shard shape on the GCB path, so these preconditions have to sit ahead of both --
    // otherwise a legacy-sharded weight dies there with a bare bad_optional_access.
    //
    // The per-mode "does this cut the slab evenly" checks need tile geometry and live in the
    // factories; these two only need the arguments.
    TT_FATAL(
        global_cb_k_blocks >= 1, "matmul_decode global_cb_k_blocks must be at least 1, but got {}", global_cb_k_blocks);
    TT_FATAL(
        global_cb.has_value() || global_cb_k_blocks == 1,
        "matmul_decode global_cb_k_blocks ({}) applies only to the global_cb path: without a GCB the weight is "
        "already L1-resident and there is nothing to stream",
        global_cb_k_blocks);
    if (global_cb.has_value()) {
        TT_FATAL(
            input_tensor_b.nd_shard_spec().has_value(),
            "matmul_decode with global_cb requires input tensor B to be ND_SHARDED (receiver-contiguous, one shard "
            "per receiver), but its memory layout is {} and it carries no NdShardSpec",
            input_tensor_b.memory_config().memory_layout());
    }

    if (packed_weight.has_value()) {
        // Fused-weight path: B is one big height-sharded tensor and its shape says nothing about
        // this weight, so every piece of geometry the code below would infer from the operand
        // shapes comes from the spec instead. The mode is also the spec's to pick: batch > 1 is
        // batched, k_blocks > 1 is partial, otherwise full width-sharded; the
        // `partial_width_sharded` argument is ignored.
        const auto& pw = *packed_weight;
        TT_FATAL(
            !global_cb.has_value(),
            "matmul_decode packed_weight and global_cb are mutually exclusive: a packed weight is already "
            "L1-resident inside the fused tensor, there is nothing for the prefetcher to stream");
        TT_FATAL(
            pw.K > 0 && pw.N > 0 && pw.num_cores() > 0,
            "matmul_decode packed_weight must carry the weight's [K, N] and its cores, but got [{}, {}] on {} cores",
            pw.K,
            pw.N,
            pw.num_cores());

        const int M = input_tensor_a.logical_shape()[-2];
        const bool batched = pw.batch > 1;
        if (batched) {
            TT_FATAL(
                input_tensor_a.logical_shape().rank() == 4 &&
                    input_tensor_a.logical_shape()[0] * input_tensor_a.logical_shape()[1] == static_cast<int>(pw.batch),
                "matmul_decode packed_weight with batch {} requires a rank-4 A whose leading dims multiply to it",
                pw.batch);
        }
        const bool partial = !batched && pw.k_blocks > 1;
        log_debug(
            tt::LogOp,
            "matmul_decode (packed) M={}, N={}, K={}, tile_offset={}, cores={}, k_blocks={}, batch={}, b_blocks={}",
            M,
            pw.N,
            pw.K,
            pw.tile_offset,
            pw.num_cores(),
            pw.k_blocks,
            pw.batch,
            pw.b_blocks);
        auto operation_attributes = with_all_gather(OperationType::operation_attributes_t{
            M,
            static_cast<int>(pw.N),
            static_cast<int>(pw.K),
            output_mem_config,
            dtype.has_value() ? std::optional<DataType>(*dtype) : std::nullopt,
            partial,
            static_cast<int>(pw.batch),
            static_cast<int>(pw.b_blocks),
            static_cast<int>(pw.n_blocks()),
            /*global_cb=*/std::nullopt,
            /*global_cb_k_blocks=*/1,
            packed_weight,
        });
        auto tensor_args = OperationType::tensor_args_t{input_tensor_a, input_tensor_b};
        return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
    }

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
            // A prefetcher weight is ND-sharded in DRAM and carries no legacy shard spec, so the
            // weight-holding core count is the GCB receiver count.
            const int num_B_cores =
                global_cb.has_value()
                    ? static_cast<int>(gcb_num_receivers(*global_cb))
                    : static_cast<int>(input_tensor_b.memory_config().shard_spec().value().grid.num_cores());
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
            auto operation_attributes = with_all_gather(OperationType::operation_attributes_t{
                M,
                N,
                K,
                output_mem_config,
                dtype.has_value() ? std::optional<DataType>(*dtype) : std::nullopt,
                /*partial_width_sharded=*/false,
                batch,
                b_blocks,
                n_blocks,
                global_cb,
                global_cb_k_blocks,
            });
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
    auto operation_attributes = with_all_gather(OperationType::operation_attributes_t{
        M,
        N,
        K,
        output_mem_config,
        dtype.has_value() ? std::optional<DataType>(*dtype) : std::nullopt,
        partial_width_sharded,
        /*batch=*/1,
        /*b_blocks=*/1,
        /*n_blocks=*/1,
        global_cb,
        global_cb_k_blocks,
    });
    auto tensor_args = OperationType::tensor_args_t{input_tensor_a, input_tensor_b};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}
}  // namespace ttnn::prim

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mix_streams.hpp"

#include <numeric>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/work_split.hpp>

#include "device/mix_streams_device_operation.hpp"
#include "ttnn/operations/core/to_memory_config/to_memory_config_op.hpp"
#include "ttnn/operations/data_movement/repeat/repeat.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/matmul/matmul.hpp"

namespace ttnn::experimental::deepseek::mix_streams {

namespace {

using ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig;

// HiFi4 / fp32-dest-acc / packer-l1-acc -- mirrors ``_HIFI4`` in
// models/experimental/deepseek_v4_flash/tt/common.py so the fused matmul
// matches the eager ``ttnn.matmul(..., compute_kernel_config=_HIFI4)`` path.
DeviceComputeKernelConfig default_hifi4_config() {
    DeviceComputeKernelConfig cfg{};
    cfg.math_fidelity = tt::tt_metal::MathFidelity::HiFi4;
    cfg.math_approx_mode = false;
    cfg.fp32_dest_acc_en = true;
    cfg.packer_l1_acc = true;
    return cfg;
}

void validate_inputs(const Tensor& post, const Tensor& comb, const Tensor& sublayer_out, const Tensor& streams) {
    TT_FATAL(post.storage_type() == StorageType::DEVICE, "mix_streams: post must be on device");
    TT_FATAL(comb.storage_type() == StorageType::DEVICE, "mix_streams: comb must be on device");
    TT_FATAL(sublayer_out.storage_type() == StorageType::DEVICE, "mix_streams: sublayer_out must be on device");
    TT_FATAL(streams.storage_type() == StorageType::DEVICE, "mix_streams: streams must be on device");

    const auto& streams_shape = streams.logical_shape();
    TT_FATAL(
        streams_shape.rank() == 4,
        "mix_streams: streams must be rank-4 [B, S, hc, D], got rank {}",
        streams_shape.rank());

    const uint32_t b = static_cast<uint32_t>(streams_shape[0]);
    const uint32_t s = static_cast<uint32_t>(streams_shape[1]);
    const uint32_t hc = static_cast<uint32_t>(streams_shape[2]);
    const uint32_t d = static_cast<uint32_t>(streams_shape[3]);

    const auto& post_shape = post.logical_shape();
    const auto& comb_shape = comb.logical_shape();
    const auto& out_shape = sublayer_out.logical_shape();

    TT_FATAL(
        post_shape.rank() == 4 && static_cast<uint32_t>(post_shape[0]) == b &&
            static_cast<uint32_t>(post_shape[1]) == s && static_cast<uint32_t>(post_shape[2]) == hc &&
            static_cast<uint32_t>(post_shape[3]) == 1,
        "mix_streams: post must be [B, S, hc, 1] = [{}, {}, {}, 1], got {}",
        b,
        s,
        hc,
        post_shape);
    TT_FATAL(
        comb_shape.rank() == 4 && static_cast<uint32_t>(comb_shape[0]) == b &&
            static_cast<uint32_t>(comb_shape[1]) == s && static_cast<uint32_t>(comb_shape[2]) == hc &&
            static_cast<uint32_t>(comb_shape[3]) == hc,
        "mix_streams: comb must be [B, S, hc, hc] = [{}, {}, {}, {}], got {}",
        b,
        s,
        hc,
        hc,
        comb_shape);
    TT_FATAL(
        out_shape.rank() == 4 && static_cast<uint32_t>(out_shape[0]) == b && static_cast<uint32_t>(out_shape[1]) == s &&
            static_cast<uint32_t>(out_shape[2]) == 1 && static_cast<uint32_t>(out_shape[3]) == d,
        "mix_streams: sublayer_out must be [B, S, 1, D] = [{}, {}, 1, {}], got {}",
        b,
        s,
        d,
        out_shape);
}

// Pick (out_subblock_h, out_subblock_w) dividing the per-core block with h*w <= 8.
std::pair<uint32_t, uint32_t> find_subblock(uint32_t per_core_m, uint32_t per_core_n) {
    for (uint32_t h = per_core_m; h > 0; --h) {
        if (per_core_m % h != 0) {
            continue;
        }
        for (uint32_t w = per_core_n; w > 0; --w) {
            if (per_core_n % w == 0 && h * w <= 8) {
                return {h, w};
            }
        }
    }
    return {1, 1};
}

// ``comb^T @ streams`` with streams WIDTH_SHARDED along D. gather_in0 gathers the
// comb activation around the ring while each core keeps its local N-slice of streams.
// The core count must divide both K and N in tiles (gather_in0 width-shards A along K),
// so we may reshard down to gcd(K_tiles, N_tiles) cores when streams is over-sharded
// relative to the tiny hc reduction dim.
Tensor matmul_gather_in0_width_sharded(
    Tensor comb_r, Tensor streams_r, const DeviceComputeKernelConfig& compute_kernel_config) {
    TT_FATAL(
        streams_r.is_sharded() && streams_r.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED &&
            streams_r.shard_spec().has_value(),
        "mix_streams: gather_in0 path requires WIDTH_SHARDED streams with a shard_spec");

    constexpr uint32_t tile_h = tt::constants::TILE_HEIGHT;
    constexpr uint32_t tile_w = tt::constants::TILE_WIDTH;

    // gather_in0 rejects transpose_a, so materialise comb^T up front.
    comb_r = ttnn::transpose(comb_r, /*dim1=*/-2, /*dim2=*/-1);

    const auto b_shard = streams_r.shard_spec().value();  // copy — streams_r may be reassigned below
    const uint32_t m_padded = tt::round_up(static_cast<uint32_t>(comb_r.logical_shape()[-2]), tile_h);
    const uint32_t k_padded = tt::round_up(static_cast<uint32_t>(comb_r.logical_shape()[-1]), tile_w);
    const uint32_t n = static_cast<uint32_t>(streams_r.logical_shape()[-1]);
    TT_FATAL(n % tile_w == 0, "mix_streams: streams width {} must be tile-aligned", n);
    TT_FATAL(k_padded % tile_w == 0, "mix_streams: padded K {} must be tile-aligned", k_padded);

    const uint32_t k_tiles = k_padded / tile_w;
    const uint32_t n_tiles = n / tile_w;
    // gather_in0 width-shards A along K and B along N on the same grid, so the
    // core count must divide both. Prefer the incoming streams grid when it
    // already qualifies; otherwise fall back to gcd.
    uint32_t num_cores = b_shard.grid.num_cores();
    CoreRangeSet grid = b_shard.grid;
    if (k_tiles % num_cores != 0 || n_tiles % num_cores != 0) {
        num_cores = std::gcd(k_tiles, n_tiles);
        TT_FATAL(num_cores > 0, "mix_streams: gather_in0 needs at least one core");
        TT_FATAL(
            k_tiles % num_cores == 0 && n_tiles % num_cores == 0,
            "mix_streams: cannot form gather_in0 grid: K_tiles={}, N_tiles={}, num_cores={}",
            k_tiles,
            n_tiles,
            num_cores);
        grid = tt::tt_metal::num_cores_to_corerangeset(
            num_cores, streams_r.device()->compute_with_storage_grid_size(), /*row_wise=*/true);
    }

    const uint32_t k_per_shard = k_padded / num_cores;
    const uint32_t n_per_shard = n / num_cores;

    const MemoryConfig a_mem_config(
        TensorMemoryLayout::WIDTH_SHARDED,
        BufferType::L1,
        tt::tt_metal::ShardSpec(grid, {m_padded, k_per_shard}, ShardOrientation::ROW_MAJOR));
    const MemoryConfig b_mem_config(
        TensorMemoryLayout::WIDTH_SHARDED,
        BufferType::L1,
        tt::tt_metal::ShardSpec(grid, {k_padded, n_per_shard}, ShardOrientation::ROW_MAJOR));
    const MemoryConfig out_mem_config(
        TensorMemoryLayout::WIDTH_SHARDED,
        BufferType::L1,
        tt::tt_metal::ShardSpec(grid, {m_padded, n_per_shard}, ShardOrientation::ROW_MAJOR));

    comb_r = ttnn::to_memory_config(comb_r, a_mem_config);
    if (streams_r.memory_config() != b_mem_config) {
        streams_r = ttnn::to_memory_config(streams_r, b_mem_config);
    }

    const uint32_t per_core_m = m_padded / tile_h;
    const uint32_t per_core_n = n_per_shard / tile_w;
    const uint32_t in0_block_w = k_per_shard / tile_w;
    const auto [out_subblock_h, out_subblock_w] = find_subblock(per_core_m, per_core_n);

    const auto bbox = grid.bounding_box();
    const CoreCoord compute_grid{bbox.end_coord.x - bbox.start_coord.x + 1, bbox.end_coord.y - bbox.start_coord.y + 1};

    MatmulMultiCoreReuseMultiCast1DProgramConfig program_config{
        .compute_with_storage_grid_size = compute_grid,
        .in0_block_w = in0_block_w,
        .out_subblock_h = out_subblock_h,
        .out_subblock_w = out_subblock_w,
        .out_block_h = per_core_m,
        .out_block_w = per_core_n,
        .per_core_M = per_core_m,
        .per_core_N = per_core_n,
        .fuse_batch = true,
        .fused_activation = std::nullopt,
        .mcast_in0 = false,
        .gather_in0 = true,
        .hop_cores = CoreRangeSet{},
        .num_global_cb_receivers = 1,
        .untilize_out = false,
    };

    return ttnn::matmul(
        /*input_tensor_a=*/comb_r,
        /*input_tensor_b=*/streams_r,
        /*transpose_a=*/false,
        /*transpose_b=*/false,
        /*memory_config=*/out_mem_config,
        /*dtype=*/std::nullopt,
        /*program_config=*/program_config,
        /*activation=*/std::nullopt,
        /*compute_kernel_config=*/compute_kernel_config);
}

}  // namespace

Tensor mix_streams(
    const Tensor& post,
    const Tensor& comb,
    const Tensor& sublayer_out,
    const Tensor& streams,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
    validate_inputs(post, comb, sublayer_out, streams);

    // Single fused kernel: one device op for the broadcast-multiply, the comb^T matmul
    // and the add. The composite fallback below still covers the shapes/dtypes the
    // kernel does not handle (hc > 32, non-tile-aligned D, non-bfloat16 inputs).
    namespace device = ttnn::operations::experimental::deepseek::mix_streams;
    if (device::is_fusable(post, comb, sublayer_out, streams)) {
        return ttnn::prim::mix_streams(post, comb, sublayer_out, streams, memory_config, compute_kernel_config);
    }

    const auto& streams_shape = streams.logical_shape();
    const uint32_t b = static_cast<uint32_t>(streams_shape[0]);
    const uint32_t s = static_cast<uint32_t>(streams_shape[1]);
    const uint32_t hc = static_cast<uint32_t>(streams_shape[2]);
    const uint32_t d = static_cast<uint32_t>(streams_shape[3]);
    const uint32_t t = b * s;
    const auto ck_config = compute_kernel_config.value_or(default_hifi4_config());

    // placement = post[..,None] * sublayer_out[..,None,:] -> [1, T, hc, D].
    auto out = ttnn::reshape(sublayer_out, ttnn::Shape({1, t, 1, d}));
    out = ttnn::repeat(out, ttnn::Shape({1, 1, hc, 1}));  // broadcast over the stream axis
    auto placement = ttnn::multiply(out, ttnn::reshape(post, ttnn::Shape({1, t, hc, 1})));

    // mixed = matmul(comb^T, streams): sum over the FIRST hc axis.
    auto comb_r = ttnn::reshape(comb, ttnn::Shape({1, t, hc, hc}));
    auto streams_r = ttnn::reshape(streams, ttnn::Shape({1, t, hc, d}));
    // Preserve the caller's residual layout when the gather_in0 path has to
    // temporarily reshard onto a K/N-compatible core grid.
    const auto streams_mem_config = streams.memory_config();

    Tensor mixed;
    if (streams_r.is_sharded() && streams_r.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED) {
        mixed = matmul_gather_in0_width_sharded(std::move(comb_r), std::move(streams_r), ck_config);
        // Keep the add in the same WIDTH_SHARDED layout as the matmul output / residual.
        placement = ttnn::to_memory_config(placement, mixed.memory_config());
    } else {
        // Interleaved path: fold the comb transpose into the matmul (transpose_a=True)
        // to drop a separate transpose device op -- at 4x4 the transpose is ~30us of
        // dispatch overhead for ~2us of compute, so the op is pure launch cost.
        mixed = ttnn::matmul(
            /*input_tensor_a=*/comb_r,
            /*input_tensor_b=*/streams_r,
            /*transpose_a=*/true,
            /*transpose_b=*/false,
            /*memory_config=*/std::nullopt,
            /*dtype=*/std::nullopt,
            /*program_config=*/std::nullopt,
            /*activation=*/std::nullopt,
            /*compute_kernel_config=*/ck_config);
    }

    auto result = ttnn::reshape(ttnn::add(placement, mixed), ttnn::Shape({b, s, hc, d}));
    const MemoryConfig& dst_mem_config = memory_config.has_value() ? *memory_config : streams_mem_config;
    if (result.memory_config() != dst_mem_config) {
        result = ttnn::to_memory_config(result, dst_mem_config);
    }
    return result;
}

}  // namespace ttnn::experimental::deepseek::mix_streams

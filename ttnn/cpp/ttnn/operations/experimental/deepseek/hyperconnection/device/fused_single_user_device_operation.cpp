// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_single_user_device_operation.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek::hyperconnection {

namespace {

constexpr uint32_t kCollapseCores = 8;
constexpr uint32_t kTotalCores = 10;
// tt::tt_metal::NUM_SEMAPHORES, which no public header exposes to a ttnn op.
constexpr uint32_t kSemaphoresPerCore = 16;

void validate_width_shard(const Tensor& tensor, uint32_t num_cores, uint32_t expected_width, const char* name) {
    TT_FATAL(
        tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED, "{} must be WIDTH_SHARDED", name);
    const auto& shard_spec = tensor.shard_spec();
    TT_FATAL(shard_spec.has_value(), "{} must have a shard specification", name);
    TT_FATAL(shard_spec->grid.num_cores() == num_cores, "{} must use {} cores", name, num_cores);
    const auto bbox = shard_spec->grid.bounding_box();
    const CoreCoord expected_start(0, 0);
    const CoreCoord expected_end(num_cores - 1, 0);
    TT_FATAL(
        bbox.start_coord == expected_start && bbox.end_coord == expected_end,
        "{} must use row-major cores [0,{}]",
        name,
        num_cores - 1);
    TT_FATAL(
        shard_spec->shape[1] == expected_width / num_cores,
        "{} shard width must be {}, got {}",
        name,
        expected_width / num_cores,
        shard_spec->shape[1]);
}

// The fn dot of the fn_weight program: fn_weight's layout and the dot grid it is cut over.
void validate_fn_dot(const FusedSingleUserParams& attributes, const FusedSingleUserInputs& tensor_args) {
    const auto& w = *tensor_args.fn_weight;
    const auto& hidden_streams = tensor_args.hidden_streams;
    const uint32_t hc = attributes.num_streams;
    const uint32_t k = hc * static_cast<uint32_t>(hidden_streams.logical_shape()[-1]);

    TT_FATAL(w.storage_type() == StorageType::DEVICE, "fn_weight must be on device");
    TT_FATAL(
        hidden_streams.device()->arch() == tt::ARCH::BLACKHOLE,
        "the fused fn dot uses Blackhole-only LLKs (custom_mm and the fused RMSNorm)");
    TT_FATAL(
        hidden_streams.layout() == Layout::ROW_MAJOR,
        "the fused fn dot reads each collapse core's [H, D/8] ROW_MAJOR shard as its K-slices, got {}",
        hidden_streams.layout());
    TT_FATAL(
        hidden_streams.dtype() == DataType::BFLOAT16,
        "the fused fn dot takes BFLOAT16 streams (its partials and fused_w are BFLOAT16), got {}",
        hidden_streams.dtype());
    TT_FATAL(
        hidden_streams.buffer()->aligned_page_size() == hidden_streams.buffer()->page_size(),
        "the fused fn dot needs each shard's rows back to back, but a {}-byte row is padded to {}",
        hidden_streams.buffer()->page_size(),
        hidden_streams.buffer()->aligned_page_size());
    TT_FATAL(
        w.layout() == Layout::TILE && w.tensor_spec().tile().get_height() == tt::constants::TILE_HEIGHT &&
            w.tensor_spec().tile().get_width() == tt::constants::TILE_WIDTH,
        "fn_weight must use 32x32 TILE layout");
    TT_FATAL(
        w.dtype() == DataType::BFLOAT16 || w.dtype() == DataType::BFLOAT8_B || w.dtype() == DataType::BFLOAT4_B,
        "fn_weight must be BFLOAT16, BFLOAT8_B or BFLOAT4_B, got {}",
        w.dtype());
    const auto& w_shape = w.logical_shape();
    TT_FATAL(
        w_shape.rank() == 4 && w_shape[0] == 1 && w_shape[1] == 1 && w_shape[2] == k &&
            w_shape[3] == tt::constants::TILE_WIDTH,
        "fn_weight must be [1,1,H*D,32] = [1,1,{},32], got {}",
        k,
        w_shape);

    CoreRangeSet grid;
    if (w.buffer()->buffer_type() == tt::tt_metal::BufferType::L1) {
        TT_FATAL(
            w.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED && w.shard_spec().has_value() &&
                w.shard_spec()->orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR,
            "an L1 fn_weight must be HEIGHT_SHARDED in ROW_MAJOR orientation (shard i is K-slice i), got {}",
            w.memory_config());
        grid = w.shard_spec()->grid;
        TT_FATAL(
            !attributes.dot_core_grid.has_value() || *attributes.dot_core_grid == grid,
            "dot_core_grid {} must be an L1 fn_weight's shard grid {}",
            attributes.dot_core_grid->str(),
            grid.str());
    } else {
        TT_FATAL(
            w.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "a DRAM fn_weight must be interleaved, got {}",
            w.memory_config());
        TT_FATAL(attributes.dot_core_grid.has_value(), "a DRAM fn_weight needs dot_core_grid");
        grid = *attributes.dot_core_grid;
    }

    const uint32_t num_dot_cores = grid.num_cores();
    TT_FATAL(
        num_dot_cores >= 1 && k % (num_dot_cores * tt::constants::TILE_WIDTH) == 0,
        "K = {} must split into {} tile-aligned slices",
        k,
        num_dot_cores);
    const uint32_t kc = k / num_dot_cores;
    TT_FATAL(
        (k / kCollapseCores) % kc == 0,
        "each dot core's K-slice ({}) must lie within one collapse core's {}-wide shard",
        kc,
        k / kCollapseCores);
    TT_FATAL(kc / tt::constants::TILE_WIDTH <= 256, "a dot core's K-slice holds at most 256 tiles, got {}", kc / 32);
    if (w.buffer()->buffer_type() == tt::tt_metal::BufferType::L1) {
        TT_FATAL(
            w.shard_spec()->shape[0] == kc && w.shard_spec()->shape[1] == tt::constants::TILE_WIDTH,
            "an L1 fn_weight's shard must be [K / cores, 32] = [{}, 32], got [{}, {}]",
            kc,
            w.shard_spec()->shape[0],
            w.shard_spec()->shape[1]);
    }
    const CoreRange hc_row(CoreCoord(0, 0), CoreCoord(kTotalCores - 1, 0));
    TT_FATAL(
        !grid.intersects(hc_row), "dot_core_grid {} must leave cores (0..9, 0) to the hyperconnection", grid.str());
    const auto device_grid = hidden_streams.device()->compute_with_storage_grid_size();
    const auto bbox = grid.bounding_box();
    TT_FATAL(
        bbox.end_coord.x < device_grid.x && bbox.end_coord.y < device_grid.y,
        "dot_core_grid {} is past the {}x{} worker grid",
        grid.str(),
        device_grid.x,
        device_grid.y);

    const uint32_t fan_in = attributes.reduce_fan_in;
    TT_FATAL(fan_in >= 2, "reduce_fan_in must be >= 2, got {}", fan_in);
    uint32_t levels = 0;
    for (uint64_t stride = 1; stride < num_dot_cores; stride *= fan_in) {
        ++levels;
    }
    // One semaphore per tree level, plus the one the root raises on the cores it pushes fused_w to, and with the
    // folded mix step the one a collapse core raises once its mixed shard is ready.
    const uint32_t extra_semaphores = tensor_args.mix_post.has_value() ? 2 : 1;
    TT_FATAL(
        levels + extra_semaphores <= kSemaphoresPerCore,
        "{} dot cores at fan-in {} make {} tree levels; at most {} fit beside the other semaphores",
        num_dot_cores,
        fan_in,
        levels,
        kSemaphoresPerCore - extra_semaphores);
    if (attributes.rms_norm_epsilon.has_value()) {
        TT_FATAL(
            *attributes.rms_norm_epsilon >= 0.0F,
            "rms_norm_epsilon must be non-negative, got {}",
            *attributes.rms_norm_epsilon);
    }
}

// The previous sublayer's mix step, folded ahead of the collapse and the fn dot (see FusedSingleUserInputs).
void validate_mix(const FusedSingleUserParams& attributes, const FusedSingleUserInputs& tensor_args) {
    const bool any = tensor_args.mix_post.has_value() || tensor_args.mix_comb.has_value() ||
                     tensor_args.mix_sublayer_out.has_value();
    if (!any) {
        return;
    }
    TT_FATAL(
        tensor_args.mix_post.has_value() && tensor_args.mix_comb.has_value() &&
            tensor_args.mix_sublayer_out.has_value(),
        "the folded mix step takes all of mix_post, mix_comb and mix_sublayer_out");
    // The fn dot has to see the mixed streams, so it must run in this program; and the collapse runs before fused_w
    // is known, so it collapses with pre_mix.
    TT_FATAL(tensor_args.fn_weight.has_value(), "the folded mix step needs fn_weight (the fn dot in this program)");
    TT_FATAL(tensor_args.pre_mix.has_value(), "the folded mix step needs pre_mix");
    const uint32_t hc = attributes.num_streams;
    TT_FATAL(hc <= 4, "the folded mix step holds at most four streams in Dst, got {}", hc);
    const auto& hidden_streams = tensor_args.hidden_streams;
    const uint32_t d = static_cast<uint32_t>(hidden_streams.logical_shape()[-1]);
    const auto& post = *tensor_args.mix_post;
    const auto& comb = *tensor_args.mix_comb;
    const auto& sub = *tensor_args.mix_sublayer_out;
    for (const auto* tensor : {&post, &comb, &sub}) {
        TT_FATAL(tensor->storage_type() == StorageType::DEVICE, "the mix step's inputs must be on device");
        TT_FATAL(
            tensor->dtype() == DataType::BFLOAT16, "the mix step's inputs must be BFLOAT16, got {}", tensor->dtype());
    }
    for (const auto* tensor : {&post, &comb}) {
        TT_FATAL(
            tensor->layout() == Layout::TILE &&
                tensor->tensor_spec().tile().get_height() == tt::constants::TILE_HEIGHT &&
                tensor->tensor_spec().tile().get_width() == tt::constants::TILE_WIDTH &&
                tensor->memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "mix_post / mix_comb must be 32x32 TILE interleaved, got {} {}",
            tensor->layout(),
            tensor->memory_config());
    }
    TT_FATAL(
        post.logical_shape() == ttnn::Shape({1, 1, hc, 1}), "mix_post must be [1,1,H,1], got {}", post.logical_shape());
    TT_FATAL(
        comb.logical_shape() == ttnn::Shape({1, 1, hc, hc}),
        "mix_comb must be [1,1,H,H], got {}",
        comb.logical_shape());
    TT_FATAL(
        sub.logical_shape() == ttnn::Shape({1, 1, 1, d}),
        "mix_sublayer_out must be [1,1,1,D] = [1,1,1,{}], got {}",
        d,
        sub.logical_shape());
    TT_FATAL(sub.layout() == Layout::ROW_MAJOR, "mix_sublayer_out must be ROW_MAJOR, got {}", sub.layout());
    // Each collapse core reads its D/8 columns a page piece at a time, 64 B aligned (DRAM's alignment).
    const uint32_t page_elems = sub.shard_spec().has_value() ? static_cast<uint32_t>(sub.shard_spec()->shape[1]) : d;
    TT_FATAL(
        page_elems % tt::constants::TILE_WIDTH == 0 && d % page_elems == 0 &&
            sub.buffer()->aligned_page_size() == sub.buffer()->page_size(),
        "mix_sublayer_out's {}-wide pages must be a multiple of 32 wide, divide D = {} and carry no padding",
        page_elems,
        d);
}

void validate_single_user_tensors(const FusedSingleUserParams& attributes, const FusedSingleUserInputs& tensor_args) {
    const auto& pre_bias = tensor_args.pre_bias;
    const auto& post_bias = tensor_args.post_bias;
    const auto& comb_bias = tensor_args.comb_bias;
    const auto& hidden_streams = tensor_args.hidden_streams;
    TT_FATAL(
        tensor_args.fused_w.has_value() != tensor_args.fn_weight.has_value(),
        "fused hyperconnection takes exactly one of fused_w and fn_weight");

    for (const auto* tensor : {&pre_bias, &post_bias, &comb_bias, &hidden_streams}) {
        TT_FATAL(tensor->storage_type() == StorageType::DEVICE, "all fused hyperconnection inputs must be on device");
        TT_FATAL(tensor->dtype() == DataType::BFLOAT16, "all fused hyperconnection inputs must be BFLOAT16");
    }
    if (tensor_args.fused_w.has_value()) {
        const auto& fused_w = *tensor_args.fused_w;
        TT_FATAL(fused_w.storage_type() == StorageType::DEVICE, "all fused hyperconnection inputs must be on device");
        TT_FATAL(fused_w.dtype() == DataType::BFLOAT16, "all fused hyperconnection inputs must be BFLOAT16");
        // fused_w is only read element-wise out of its first row, so either layout works for it.
        TT_FATAL(
            fused_w.layout() == Layout::TILE || fused_w.layout() == Layout::ROW_MAJOR,
            "fused_w must use TILE or ROW_MAJOR layout, got {}",
            fused_w.layout());
    }
    for (const auto* tensor : {&pre_bias, &post_bias, &comb_bias}) {
        TT_FATAL(tensor->layout() == Layout::TILE, "fused hyperconnection biases must use TILE");
    }
    // ROW_MAJOR hidden_streams: the collapse cores gather their [H, D/8] shard into tiles locally.
    TT_FATAL(
        hidden_streams.layout() == Layout::TILE || hidden_streams.layout() == Layout::ROW_MAJOR,
        "hidden_streams must use TILE or ROW_MAJOR, got {}",
        hidden_streams.layout());

    const uint32_t hc = attributes.num_streams;
    const auto& hidden_shape = hidden_streams.logical_shape();
    const uint32_t packed_width = (2 + hc) * hc;
    TT_FATAL(hc >= 1 && hc <= 32, "num_streams must be in [1,32], got {}", hc);
    TT_FATAL(attributes.sinkhorn_iters >= 1, "sinkhorn_iters must be >= 1");
    TT_FATAL(
        packed_width <= tt::constants::TILE_WIDTH,
        "single-user fused_w currently packs into one tile; packed width {} exceeds {}",
        packed_width,
        tt::constants::TILE_WIDTH);
    if (tensor_args.fused_w.has_value()) {
        const auto& fused_w = *tensor_args.fused_w;
        const auto& fused_shape = fused_w.logical_shape();
        TT_FATAL(
            fused_shape.rank() == 4 && fused_shape[0] == 1 && fused_shape[1] == 1 && fused_shape[2] == 1 &&
                fused_shape[3] >= packed_width,
            "fused_w must be [1,1,1,>=((2+H)*H)] with packed width {}, got {}",
            packed_width,
            fused_shape);
        TT_FATAL(
            fused_w.padded_shape()[-1] % 32 == 0,
            "fused_w padded width must be tile-aligned, got {}",
            fused_w.padded_shape()[-1]);
        validate_width_shard(fused_w, 1, static_cast<uint32_t>(fused_w.padded_shape()[-1]), "fused_w");
    }
    TT_FATAL(hidden_shape.rank() == 4, "hidden_streams must be rank-4, got {}", hidden_shape.rank());
    TT_FATAL(
        hidden_shape[0] == 1 && hidden_shape[1] == 1 && hidden_shape[2] == hc,
        "hidden_streams must be [1,1,H,D] with H={}, got {}",
        hc,
        hidden_shape);
    const uint32_t d = static_cast<uint32_t>(hidden_shape[3]);
    TT_FATAL(d % (kCollapseCores * 32) == 0, "hidden size D must be divisible by {}", kCollapseCores * 32);
    TT_FATAL(
        pre_bias.logical_shape() == ttnn::Shape({1, 1, 1, hc}),
        "pre_bias must be [1,1,1,H], got {}",
        pre_bias.logical_shape());
    TT_FATAL(
        post_bias.logical_shape() == ttnn::Shape({1, 1, 1, hc}),
        "post_bias must be [1,1,1,H], got {}",
        post_bias.logical_shape());
    TT_FATAL(
        comb_bias.logical_shape() == ttnn::Shape({1, 1, hc, hc}) ||
            comb_bias.logical_shape() == ttnn::Shape({1, 1, 1, hc * hc}),
        "comb_bias must be [1,1,H,H] or [1,1,1,H*H], got {}",
        comb_bias.logical_shape());

    validate_width_shard(hidden_streams, kCollapseCores, d, "hidden_streams");
    if (hidden_streams.layout() == Layout::ROW_MAJOR) {
        TT_FATAL(
            hidden_streams.buffer()->buffer_type() == tt::tt_metal::BufferType::L1 &&
                hidden_streams.shard_spec()->shape[0] == hc,
            "ROW_MAJOR hidden_streams must be L1 with shard height H = {}, got {}",
            hc,
            hidden_streams.memory_config());
    }

    TT_FATAL(
        attributes.collapsed_layout == Layout::TILE || attributes.collapsed_layout == Layout::ROW_MAJOR,
        "collapsed_layout must be TILE or ROW_MAJOR, got {}",
        attributes.collapsed_layout);
    const bool fuse_norm = tensor_args.collapsed_norm_weight.has_value();
    if (fuse_norm) {
        const auto& gamma = *tensor_args.collapsed_norm_weight;
        TT_FATAL(
            attributes.collapsed_layout == Layout::ROW_MAJOR && hidden_streams.layout() == Layout::ROW_MAJOR,
            "collapsed_norm_weight needs ROW_MAJOR hidden_streams and a ROW_MAJOR collapsed_layout");
        TT_FATAL(
            hidden_streams.device()->arch() == tt::ARCH::BLACKHOLE,
            "the fused collapsed RMSNorm uses Blackhole-only LLKs (mul_reduce_scalar accumulating in one tile)");
        TT_FATAL(gamma.storage_type() == StorageType::DEVICE, "collapsed_norm_weight must be on device");
        TT_FATAL(
            gamma.dtype() == DataType::BFLOAT16 && gamma.layout() == Layout::ROW_MAJOR &&
                gamma.logical_shape() == ttnn::Shape({1, 1, 1, d}) &&
                gamma.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "collapsed_norm_weight must be [1,1,1,{}] BFLOAT16 ROW_MAJOR interleaved, got {} {} {} {}",
            d,
            gamma.logical_shape(),
            gamma.dtype(),
            gamma.layout(),
            gamma.memory_config());
        TT_FATAL(
            attributes.collapsed_norm_epsilon >= 0.0F,
            "collapsed_norm_epsilon must be non-negative, got {}",
            attributes.collapsed_norm_epsilon);
        const auto& out_mem = attributes.collapsed_output_mem_config;
        const bool dram = out_mem.memory_layout() == TensorMemoryLayout::INTERLEAVED &&
                          out_mem.buffer_type() == tt::tt_metal::BufferType::DRAM;
        const bool own_shard =
            out_mem.memory_layout() == TensorMemoryLayout::WIDTH_SHARDED &&
            out_mem.buffer_type() == tt::tt_metal::BufferType::L1 && out_mem.shard_spec().has_value() &&
            out_mem.shard_spec()->grid == hidden_streams.shard_spec()->grid && out_mem.shard_spec()->shape[0] == 1 &&
            out_mem.shard_spec()->shape[1] == d / kCollapseCores;
        TT_FATAL(
            dram || own_shard,
            "with collapsed_norm_weight, collapsed must be DRAM interleaved or L1 WIDTH_SHARDED [1, {}] on the "
            "hidden_streams cores, got {}",
            d / kCollapseCores,
            out_mem);
    } else if (attributes.collapsed_layout == Layout::ROW_MAJOR) {
        // The collapse packs its row straight into the output shard, so it must be that core's [1, D/8].
        TT_FATAL(
            hidden_streams.layout() == Layout::ROW_MAJOR,
            "a ROW_MAJOR collapsed output needs ROW_MAJOR hidden_streams, got {}",
            hidden_streams.layout());
        const auto& out_mem = attributes.collapsed_output_mem_config;
        TT_FATAL(
            out_mem.memory_layout() == TensorMemoryLayout::WIDTH_SHARDED &&
                out_mem.buffer_type() == tt::tt_metal::BufferType::L1 && out_mem.shard_spec().has_value() &&
                out_mem.shard_spec()->grid == hidden_streams.shard_spec()->grid &&
                out_mem.shard_spec()->shape[0] == 1 && out_mem.shard_spec()->shape[1] == d / kCollapseCores,
            "a ROW_MAJOR collapsed output must be L1 WIDTH_SHARDED [1, {}] on the hidden_streams cores, got {}",
            d / kCollapseCores,
            out_mem);
    }

    if (attributes.collapsed_replica_grid.has_value()) {
        TT_FATAL(
            attributes.collapsed_layout == Layout::ROW_MAJOR && hidden_streams.layout() == Layout::ROW_MAJOR,
            "collapsed_replica_grid needs ROW_MAJOR hidden_streams and a ROW_MAJOR collapsed");
        TT_FATAL(
            attributes.collapsed_replica_grid->ranges().size() == 1,
            "collapsed_replica_grid must be one rectangle of cores, got {}",
            *attributes.collapsed_replica_grid);
    }

    if (tensor_args.pre_mix.has_value()) {
        const auto& pre_mix = *tensor_args.pre_mix;
        TT_FATAL(pre_mix.storage_type() == StorageType::DEVICE, "pre_mix must be on device");
        TT_FATAL(pre_mix.dtype() == DataType::BFLOAT16, "pre_mix must be BFLOAT16");
        // The collapse cores copy its single tile straight into the matmul's in0 buffer.
        TT_FATAL(
            pre_mix.layout() == Layout::TILE &&
                pre_mix.tensor_spec().tile().get_height() == tt::constants::TILE_HEIGHT &&
                pre_mix.tensor_spec().tile().get_width() == tt::constants::TILE_WIDTH,
            "pre_mix must use 32x32 TILE layout");
        TT_FATAL(
            pre_mix.logical_shape() == ttnn::Shape({1, 1, 1, hc}),
            "pre_mix must be [1,1,1,H], got {}",
            pre_mix.logical_shape());
    }

    const auto device_grid = hidden_streams.device()->compute_with_storage_grid_size();
    TT_FATAL(
        device_grid.x >= kTotalCores, "single-user hyperconnection requires {} cores in the first row", kTotalCores);

    if (tensor_args.fn_weight.has_value()) {
        validate_fn_dot(attributes, tensor_args);
    }
    validate_mix(attributes, tensor_args);
}

}  // namespace

FusedSingleUserDeviceOperation::program_factory_t FusedSingleUserDeviceOperation::select_program_factory(
    const operation_attributes_t& /*attributes*/, const tensor_args_t& tensor_args) {
    if (tensor_args.fn_weight.has_value()) {
        return FusedSingleUserDotProgramFactory{};
    }
    return FusedSingleUserProgramFactory{};
}

void FusedSingleUserDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    validate_single_user_tensors(attributes, tensor_args);
}

void FusedSingleUserDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    validate_single_user_tensors(attributes, tensor_args);
}

FusedSingleUserDeviceOperation::spec_return_value_t FusedSingleUserDeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto make_layout = [&](const MemoryConfig& memory_config) {
        return tt::tt_metal::TensorLayout(
            tensor_args.hidden_streams.dtype(), tt::tt_metal::PageConfig(Layout::TILE), memory_config);
    };
    const auto& hidden_shape = tensor_args.hidden_streams.logical_shape();
    const uint32_t d = static_cast<uint32_t>(hidden_shape[3]);
    spec_return_value_t specs = {
        tt::tt_metal::TensorSpec(
            ttnn::Shape({1, 1, attributes.num_streams, 1}), make_layout(attributes.post_comb_output_mem_config)),
        tt::tt_metal::TensorSpec(
            ttnn::Shape({1, 1, attributes.num_streams, attributes.num_streams}),
            make_layout(attributes.post_comb_output_mem_config)),
        tt::tt_metal::TensorSpec(
            ttnn::Shape({1, 1, 1, d}),
            tt::tt_metal::TensorLayout(
                tensor_args.hidden_streams.dtype(),
                tt::tt_metal::PageConfig(attributes.collapsed_layout),
                attributes.collapsed_output_mem_config))};
    if (tensor_args.pre_mix.has_value()) {
        specs.emplace_back(
            ttnn::Shape({1, 1, 1, attributes.num_streams}), make_layout(attributes.post_comb_output_mem_config));
    }
    if (tensor_args.mix_post.has_value()) {
        // The mixed streams sit where the streams did: same shape, layout and shards.
        specs.push_back(tensor_args.hidden_streams.tensor_spec());
    }
    if (attributes.collapsed_replica_grid.has_value()) {
        // all_gather_for_matmul's replica: the row once per core, stacked along dim -2.
        const auto& grid = *attributes.collapsed_replica_grid;
        const tt::tt_metal::ShardSpec shard(grid, {1, d}, tt::tt_metal::ShardOrientation::ROW_MAJOR);
        specs.emplace_back(
            ttnn::Shape({1, 1, grid.num_cores(), d}),
            tt::tt_metal::TensorLayout(
                tensor_args.hidden_streams.dtype(),
                tt::tt_metal::PageConfig(Layout::ROW_MAJOR),
                MemoryConfig(TensorMemoryLayout::HEIGHT_SHARDED, tt::tt_metal::BufferType::L1, shard)));
    }
    return specs;
}

FusedSingleUserDeviceOperation::tensor_return_value_t FusedSingleUserDeviceOperation::create_output_tensors(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    auto* device = tensor_args.hidden_streams.device();
    tensor_return_value_t outputs;
    for (const auto& spec : compute_output_specs(attributes, tensor_args)) {
        outputs.push_back(create_device_tensor(spec, device));
    }
    return outputs;
}

}  // namespace ttnn::operations::experimental::deepseek::hyperconnection

namespace ttnn::prim {

std::vector<Tensor> fused_hyperconnection_single_user(
    const std::optional<Tensor>& fused_w,
    const Tensor& pre_bias,
    const Tensor& post_bias,
    const Tensor& comb_bias,
    const Tensor& hidden_streams,
    uint32_t num_streams,
    uint32_t sinkhorn_iters,
    float pre_scale,
    float post_scale,
    float comb_scale,
    float eps,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& pre_mix,
    std::optional<Layout> collapsed_layout,
    const std::optional<Tensor>& fn_weight,
    const std::optional<CoreRangeSet>& dot_core_grid,
    uint32_t reduce_fan_in,
    std::optional<float> rms_norm_epsilon,
    const std::optional<Tensor>& mix_post,
    const std::optional<Tensor>& mix_comb,
    const std::optional<Tensor>& mix_sublayer_out,
    const std::optional<Tensor>& collapsed_norm_weight,
    float collapsed_norm_epsilon,
    bool collapsed_norm_sharded,
    const std::optional<CoreRangeSet>& collapsed_replica_grid) {
    using OperationType = ttnn::operations::experimental::deepseek::hyperconnection::FusedSingleUserDeviceOperation;
    const MemoryConfig post_comb_output_mem_config = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG);
    const Layout resolved_collapsed_layout = collapsed_layout.value_or(Layout::TILE);
    // collapsed keeps the input's grid and shard width, with a tile-high shard (TILE) or a one-row one.
    MemoryConfig collapsed_output_mem_config = hidden_streams.memory_config();
    if (hidden_streams.layout() == Layout::ROW_MAJOR && hidden_streams.shard_spec().has_value()) {
        auto shard_spec = hidden_streams.shard_spec().value();
        shard_spec.shape[0] = resolved_collapsed_layout == Layout::ROW_MAJOR ? 1u : tt::constants::TILE_HEIGHT;
        collapsed_output_mem_config = MemoryConfig(
            hidden_streams.memory_config().memory_layout(), hidden_streams.memory_config().buffer_type(), shard_spec);
    }
    // The normed collapsed row goes to DRAM, where the next op (the MoE request) reads it, unless it is kept sharded.
    if (collapsed_norm_weight.has_value() && !collapsed_norm_sharded) {
        collapsed_output_mem_config = ttnn::DRAM_MEMORY_CONFIG;
    }
    collapsed_output_mem_config = memory_config.value_or(collapsed_output_mem_config);
    auto operation_attributes = OperationType::operation_attributes_t{
        .num_streams = num_streams,
        .sinkhorn_iters = sinkhorn_iters,
        .pre_scale = pre_scale,
        .post_scale = post_scale,
        .comb_scale = comb_scale,
        .eps = eps,
        .post_comb_output_mem_config = post_comb_output_mem_config,
        .collapsed_output_mem_config = collapsed_output_mem_config,
        .collapsed_layout = resolved_collapsed_layout,
        .dot_core_grid = dot_core_grid,
        .reduce_fan_in = reduce_fan_in,
        .rms_norm_epsilon = rms_norm_epsilon,
        .collapsed_norm_epsilon = collapsed_norm_epsilon,
        .collapsed_replica_grid = collapsed_replica_grid,
    };
    auto tensor_args = OperationType::tensor_args_t{
        .fused_w = fused_w,
        .pre_bias = pre_bias,
        .post_bias = post_bias,
        .comb_bias = comb_bias,
        .hidden_streams = hidden_streams,
        .pre_mix = pre_mix,
        .fn_weight = fn_weight,
        .mix_post = mix_post,
        .mix_comb = mix_comb,
        .mix_sublayer_out = mix_sublayer_out,
        .collapsed_norm_weight = collapsed_norm_weight,
    };
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim

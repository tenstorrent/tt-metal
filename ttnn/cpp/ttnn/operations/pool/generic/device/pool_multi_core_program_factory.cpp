// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "pool_op.hpp"
#include "tt-metalium/constants.hpp"
#include "tt-metalium/tt_backend_api_types.hpp"
#include "ttnn/operations/pool/pool_utils.hpp"
#include "tt-metalium/host_buffer.hpp"
#include "tt-metalium/buffer.hpp"
#include "ttnn/tensor/types.hpp"
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include "ttnn/tensor/storage.hpp"
#include <tt-metalium/hal.hpp>
#include <algorithm>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

#include "ttnn/metal2_artifacts.hpp"
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

namespace ttnn::operations::pool {

using namespace tt::tt_metal;
namespace m2 = tt::tt_metal::experimental;

/**
 * Generic pool implementation that uses the new sliding window infrastructure.
 */
struct ScalarInfo {
    // Scalar Info is used to store the information about the scalar used in avg pool op
    // start and end refer to indices of the output stick core is calculating.
    // These are directly mapped to the for loop that can be found in reader and compute kernel of the pool op
    // for (uint32_t i = 0; i < nsticks_per_core; ++i), start is first stick which should be reduced and multiplied by
    // scalar value, end is the first stick which should not be reduced and multiplied by scalar value. So the interval
    // [start, end) is the range of sticks that should be reduced and multiplied by scalar value.
    uint16_t start;
    uint16_t value;
    uint16_t end;
};

// This function generates a vector of elements of type ScalarInfo. It is called once per core and generates the
// adequate scalars for each output element that core should produce. It should be called only for avg pool operation
// and only if the divisor_override is NOT set and the idea behind it is to generate config tensor in cases where one
// scalar per core is not sufficient to create correct result. Those scenarios are ceil_mode == true and (ceil_pad_h > 0
// || ceil_pad_w > 0) or count_include_pad == false || (pad_h > 0 || pad_w > 0). Both of these scenarios can be
// irrelevant if the divisor_override is set, in which case we don't calculate the divisor since it is already passed as
// an argument. It only adds scalars that are different than the scalar preceding it not to have duplicates of data,
// this is why we use start and end indices to know how many sequential output elements should be multiplied by the same
// scalar value.
std::vector<ScalarInfo> get_bf16_avg_pool_config_scalars(
    const AvgPoolConfig& config, uint32_t output_stick_x, uint32_t output_stick_y) {
    TT_FATAL(
        !is_pool_op_one_scalar_per_core(
            Pool2DType::AVG_POOL2D,
            config.ceil_mode,
            config.ceil_h,
            config.ceil_w,
            config.count_include_pad,
            config.pad_t + config.pad_b,
            config.pad_l + config.pad_r,
            config.divisor_override),
        "Avg pool scalars config should be calculated only for ceil_mode == true and "
        "(ceil_pad_h > 0 || ceil_pad_w > 0) or count_include_pad == false and (pad_h > 0 || pad_w > 0)");

    std::vector<ScalarInfo> scalars;
    bool first_scalar = true;
    uint32_t last_pool_area = 0;

    for (uint32_t i = 0; i < config.max_out_nhw_per_core; i++) {
        // Compute starting and ending indices of the pooling window
        int h_start = (output_stick_x * config.stride_h) - config.pad_t;
        int w_start = (output_stick_y * config.stride_w) - config.pad_l;
        // omit any ceiling mode related padding from end point calculations as these are not used in the
        // calculation of the pool area even when count_include_pad is on
        int h_end = std::min(h_start + static_cast<int>(config.kernel_h), static_cast<int>(config.in_h + config.pad_b));
        int w_end = std::min(w_start + static_cast<int>(config.kernel_w), static_cast<int>(config.in_w + config.pad_r));

        int pool_area = 0;
        if (config.count_include_pad) {
            pool_area = (h_end - h_start) * (w_end - w_start);
        } else {
            int valid_h_start = (h_start > 0) ? h_start : 0;
            int valid_w_start = (w_start > 0) ? w_start : 0;
            int valid_h_end = std::min(h_end, static_cast<int>(config.in_h));
            int valid_w_end = std::min(w_end, static_cast<int>(config.in_w));

            int effective_h = valid_h_end - valid_h_start;
            int effective_w = valid_w_end - valid_w_start;
            pool_area = (effective_h > 0 && effective_w > 0) ? effective_h * effective_w : 0;
        }
        float value = pool_area > 0 ? 1.f / static_cast<float>(pool_area) : 0.f;

        // Add new scalar if padding config changes
        if (first_scalar || static_cast<uint32_t>(pool_area) != last_pool_area) {
            if (!scalars.empty()) {
                scalars.back().end = i;
            }
            // TODO: #27672: Truncation should be removed once we figure a root cause of regression without it
            scalars.push_back({i, std::bit_cast<uint16_t>(bfloat16::truncate(value)), i});
            first_scalar = false;
        }
        last_pool_area = static_cast<uint32_t>(pool_area);

        // Advance output element coordinates
        output_stick_y = (output_stick_y + 1) % config.out_w;
        if (output_stick_y == 0) {
            output_stick_x = (output_stick_x + 1) % config.out_h;
        }
    }

    scalars.back().end = config.max_out_nhw_per_core;
    return scalars;
}

static void push_back_scalar_info_or_zero(
    std::vector<uint16_t>& config_vector,
    const std::vector<ScalarInfo>& scalars,
    uint32_t max_scalars_cnt,
    uint32_t repeats) {
    for (uint32_t r = 0; r < repeats; ++r) {
        for (uint32_t j = 0; j < max_scalars_cnt; ++j) {
            if (j < scalars.size()) {
                config_vector.insert(config_vector.end(), {scalars[j].start, scalars[j].value, scalars[j].end});
            } else {
                config_vector.insert(config_vector.end(), {0, 0, 0});
            }
        }
    }
}

// This function creates a scalar config tensor for the AvgPool2D operation. It is entirely made of
// a vector of ScalarInfo structs, which are used to configure the pooling operation. The config tensor
// is filled with max_out_nhw_per_core number of ScalarInfos for each core and then sharded across the cores.
// Since we don't usually have that many different scalars, we fill the rest of the config tensor with 0s.
static Tensor create_scalar_config_tensor(
    const AvgPoolConfig& config,
    TensorMemoryLayout in_memory_layout,
    uint32_t n_dim,
    uint32_t num_shards_c,
    uint32_t num_cores,
    bool config_tensor_in_dram) {
    std::vector<uint16_t> config_vector;

    size_t max_scalars_cnt = 0;
    std::vector<std::vector<ScalarInfo>> scalars_per_core = {};
    uint32_t num_iterations = 0;

    switch (in_memory_layout) {
        case TensorMemoryLayout::HEIGHT_SHARDED: num_iterations = num_cores; break;
        case TensorMemoryLayout::BLOCK_SHARDED: num_iterations = num_cores / num_shards_c; break;
        case TensorMemoryLayout::WIDTH_SHARDED: num_iterations = 1; break;
        default: break;
    }

    {
        uint32_t nhw_linear = 0;
        uint32_t output_stick_n = 0;
        uint32_t output_stick_h = 0;
        uint32_t output_stick_w = 0;
        for (uint32_t i = 0; i < num_iterations; ++i) {
            scalars_per_core.emplace_back(get_bf16_avg_pool_config_scalars(config, output_stick_h, output_stick_w));
            max_scalars_cnt = std::max(max_scalars_cnt, scalars_per_core.back().size());

            // Width sharded layout requires only one iteration, so we can break here
            if (in_memory_layout == TensorMemoryLayout::HEIGHT_SHARDED ||
                in_memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
                nhw_linear += config.max_out_nhw_per_core;
                output_stick_w = nhw_linear % config.out_w;
                output_stick_h = (nhw_linear / config.out_w) % config.out_h;
                output_stick_n = nhw_linear / (config.out_w * config.out_h);

                if (output_stick_n == n_dim) {
                    nhw_linear -= output_stick_n * config.out_w * config.out_h;
                    output_stick_h = 0;
                    output_stick_w = 0;
                }
            }
        }
    }

    constexpr uint32_t entry_size = 3;
    const uint32_t entries_per_core = entry_size * max_scalars_cnt;

    TT_FATAL(
        entries_per_core != 0,
        "entries_per_core cannot be zero. max_scalars_cnt: {}, num_iterations: {}, in_memory_layout: {}",
        max_scalars_cnt,
        num_iterations,
        in_memory_layout);

    switch (in_memory_layout) {
        case TensorMemoryLayout::HEIGHT_SHARDED:
        // With height sharded layout each scalar is unique and needs to be calculated
        case TensorMemoryLayout::BLOCK_SHARDED: {
            // With block sharded layout scalars across sequential channels shard repeat, so we use repeats variable not
            // to recalculate scalars that we can reuse
            for (const std::vector<ScalarInfo>& scalars : scalars_per_core) {
                uint32_t repeats = config_tensor_in_dram ? 1 : num_shards_c;
                push_back_scalar_info_or_zero(config_vector, scalars, max_scalars_cnt, repeats);
            }
            break;
        }
        case TensorMemoryLayout::WIDTH_SHARDED: {
            // With width sharded layout scalars should be calculated only once, so we push them back num_shards_c times
            // but have only one array of scalars
            uint32_t repeats = config_tensor_in_dram ? 1 : num_shards_c;
            push_back_scalar_info_or_zero(config_vector, scalars_per_core[0], max_scalars_cnt, repeats);
            break;
        }

        default: break;
    }

    TT_FATAL(
        config_vector.size() % entries_per_core == 0,
        "Config vector size {} should be a multiple of {}",
        config_vector.size(),
        entries_per_core);

    ttnn::Shape config_shape = ttnn::Shape({tt::div_up(config_vector.size(), entries_per_core), entries_per_core});
    tt::tt_metal::HostBuffer buffer(std::move(config_vector));
    return Tensor(std::move(buffer), config_shape, DataType::UINT16, Layout::ROW_MAJOR);
}

std::vector<uint32_t> generate_core_starting_indices(
    const std::vector<uint32_t>& op_trace_metadata,
    const std::vector<sliding_window::ShardBoundary>& shard_boundaries,
    const tt::tt_metal::TensorMemoryLayout shard_scheme,
    const uint32_t num_cores_x,
    const uint32_t ncores) {
    std::vector<uint32_t> starting_indices;
    uint32_t repeat_factor = 0;
    switch (shard_scheme) {
        case tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED: repeat_factor = 1; break;
        case tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED: repeat_factor = ncores; break;
        case tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED: repeat_factor = num_cores_x; break;
        default: TT_FATAL(false, "Unsupported shard scheme");
    };
    for (const auto& item : shard_boundaries) {
        const auto& [output_shard_start, _] = item.output_range;
        if (output_shard_start >= op_trace_metadata.size()) {
            // this core has no output
            starting_indices.push_back(0);
            continue;
        }
        TT_ASSERT(item.input_range.start == op_trace_metadata[output_shard_start]);
        for (uint32_t r = 0; r < repeat_factor; r++) {
            starting_indices.push_back(op_trace_metadata[output_shard_start]);
        }
    }

    return starting_indices;
}

namespace {

// Metal 2.0 DFB / tensor-parameter / kernel names. Each name replaces a legacy magic CB index
// (or a tensor buffer-address RTA/CTA) with a typed, named binding.
constexpr const char* DFB_IN_SCALAR_0 = "in_scalar_0";
constexpr const char* DFB_IN_SCALAR_1 = "in_scalar_1";
constexpr const char* DFB_CLEAR_VALUE = "clear_value";
constexpr const char* DFB_RAW_IN = "raw_in";
constexpr const char* DFB_READER_INDICES = "reader_indices";
constexpr const char* DFB_IN_0 = "in_0";
constexpr const char* DFB_IN_1 = "in_1";
constexpr const char* DFB_PRE_TILIZE = "pre_tilize";
constexpr const char* DFB_FAST_TILIZE = "fast_tilize";
constexpr const char* DFB_OUT = "out";
constexpr const char* DFB_CONFIG = "config";

constexpr const char* TP_INPUT = "input";
constexpr const char* TP_OUTPUT = "output";
constexpr const char* TP_READER_INDICES = "reader_indices_tensor";
constexpr const char* TP_CONFIG = "config_tensor";

constexpr const char* KERNEL_READER0 = "reader0";
constexpr const char* KERNEL_READER1 = "reader1";
constexpr const char* KERNEL_COMPUTE = "compute";

// Forked compute kernel (compute_pool_2d.cpp is shared by rotate / grid_sample, so it is forked
// to a Metal 2.0 *_m2 copy and the factory points here). The reader is exclusive to this factory
// and ported in place.
constexpr const char* READER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/pool/generic/device/kernels/dataflow/reader_pool_2d.cpp";
constexpr const char* COMPUTE_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/pool/generic/device/kernels/compute/compute_pool_2d_m2.cpp";

// Common preamble shared between the resource-allocation phase and the spec build of
// create_program_spec(): pulls per-op fields out of the SlidingWindowConfig and computes the
// parallel config / output shape pieces we need in both phases. Lives in an anonymous namespace
// because it's purely a local helper for this factory.
struct PoolSetup {
    sliding_window::ParallelConfig parallel_config;
    bool is_block_sharded;
    uint32_t out_h, out_w, out_c;
    uint32_t in_n, in_c, in_h, in_w;
    uint32_t kernel_h, kernel_w;
    uint32_t stride_h, stride_w;
    uint32_t pad_t, pad_b, pad_l, pad_r;
    uint32_t ceil_pad_h, ceil_pad_w;
    bool ceil_mode;
    uint32_t dilation_h, dilation_w;
    uint32_t num_shards_c;
};

PoolSetup compute_pool_setup(const Pool2D::operation_attributes_t& op_attr, const Tensor& input) {
    const auto& sliding_window_config = op_attr.sliding_window_config_;
    PoolSetup setup;
    setup.parallel_config = sliding_window::ParallelConfig{
        .grid = input.shard_spec().value().grid,
        .shard_scheme = input.memory_config().memory_layout(),
        .shard_orientation = input.shard_spec().value().orientation,
    };

    auto output_shape = sliding_window_config.get_output_shape();
    setup.out_h = output_shape[1];
    setup.out_w = output_shape[2];
    setup.out_c = output_shape[3];

    setup.is_block_sharded = input.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED;
    setup.in_n = sliding_window_config.batch_size;
    setup.in_c = sliding_window_config.channels;
    setup.in_h = sliding_window_config.input_hw.first;
    setup.in_w = sliding_window_config.input_hw.second;
    setup.kernel_h = sliding_window_config.window_hw.first;
    setup.kernel_w = sliding_window_config.window_hw.second;
    setup.stride_h = sliding_window_config.stride_hw.first;
    setup.stride_w = sliding_window_config.stride_hw.second;
    setup.pad_t = sliding_window_config.get_pad_top();
    setup.pad_b = sliding_window_config.get_pad_bottom();
    setup.pad_l = sliding_window_config.get_pad_left();
    setup.pad_r = sliding_window_config.get_pad_right();
    setup.ceil_pad_h = sliding_window_config.get_ceil_pad_h();
    setup.ceil_pad_w = sliding_window_config.get_ceil_pad_w();
    setup.ceil_mode = sliding_window_config.ceil_mode;
    setup.dilation_h = sliding_window_config.dilation_hw.first;
    setup.dilation_w = sliding_window_config.dilation_hw.second;
    setup.num_shards_c = sliding_window_config.num_cores_c;
    return setup;
}

}  // namespace

ttnn::device_operation::ProgramArtifacts Pool2D::MultiCore::create_program_spec(
    const operation_attributes_t& op_attr, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensors) {
    const auto& input = tensor_args.input_tensor_;
    const auto& sliding_window_config = op_attr.sliding_window_config_;
    PoolSetup setup = compute_pool_setup(op_attr, input);

    const Pool2DType pool_type = op_attr.pool_type_;
    const Layout output_layout = op_attr.output_layout_;
    const bool count_include_pad = op_attr.count_include_pad_;
    const std::optional<int32_t> divisor_override = op_attr.divisor_override_;
    const bool return_indices = op_attr.return_indices_;

    // ----- SCOPE GUARDS (this port covers only the L1-config, non-return-indices path) -----
    // The DRAM-config path threads buffer->address() through CTAs into the kernel's
    // load_config_tensor_if_in_dram<> / TensorAccessorArgs<> plumbing — a raw-address-through-CTA
    // into a data-movement kernel. That is a Metal 2.0 stop signal and stays BLOCKED; gate it out.
    TT_FATAL(
        !op_attr.config_tensor_in_dram,
        "Pool2D Metal 2.0 port: the DRAM-config path is not ported (raw buffer->address() through "
        "CTAs is a blocker). Use the L1 config path (config_tensor_in_dram == false).");
    // The return_indices (MPWI) path uses reader_mpwi.cpp / compute_mpwi.cpp — out of scope here.
    TT_FATAL(
        !return_indices,
        "Pool2D Metal 2.0 port: the return_indices (MPWI) path is not ported yet. Only the "
        "reader_pool_2d / compute_pool_2d path is on Metal 2.0.");

    const bool is_block_sharded = input.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED;
    const bool is_width_sharded = input.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED;

    // distributing out_hw across the grid
    const auto all_cores = input.shard_spec().value().grid;
    const uint32_t ncores = all_cores.num_cores();
    const uint32_t num_cores_x = input.memory_config().shard_spec()->grid.bounding_box().grid_size().x;
    const uint32_t rectangular_x = is_block_sharded ? all_cores.ranges()[0].end_coord.x + 1 : num_cores_x;
    const uint32_t max_out_nhw_per_core = output_tensors[0].shard_spec()->shape[0];
    const uint32_t max_in_nhw_per_core = input.shard_spec()->shape[0];
    TT_FATAL(
        max_in_nhw_per_core <= std::numeric_limits<uint16_t>::max(),
        "Input nhw per core {} exceeds uint16_t limit, this will cause overflow because the reader indices are stored "
        "as uint16_t",
        max_in_nhw_per_core);

    const uint32_t in_n = setup.in_n;
    const uint32_t in_c = setup.in_c;
    const uint32_t in_h = setup.in_h;
    const uint32_t in_w = setup.in_w;
    const uint32_t out_h = setup.out_h;
    const uint32_t out_w = setup.out_w;
    const uint32_t kernel_h = setup.kernel_h;
    const uint32_t kernel_w = setup.kernel_w;
    const uint32_t stride_h = setup.stride_h;
    const uint32_t stride_w = setup.stride_w;
    const uint32_t pad_t = setup.pad_t;
    const uint32_t pad_l = setup.pad_l;
    const uint32_t ceil_pad_w = setup.ceil_pad_w;
    const uint32_t ceil_pad_h = setup.ceil_pad_h;
    const uint32_t dilation_h = setup.dilation_h;
    const uint32_t dilation_w = setup.dilation_w;
    const uint32_t num_shards_c = setup.num_shards_c;

    const uint32_t bf16_scalar = get_bf16_pool_scalar(pool_type, kernel_h, kernel_w, divisor_override);
    const uint32_t bf16_init_value = get_bf16_pool_init_value(pool_type);
    FactoryParameters params = get_factory_parameters(
        num_shards_c,
        input.dtype(),
        output_tensors[0].dtype(),
        kernel_h,
        kernel_w,
        in_c,
        pool_type,
        return_indices,
        in_h,
        in_w,
        output_layout);
    uint32_t pad_h = pad_t + setup.pad_b;
    uint32_t pad_w = pad_l + setup.pad_r;
    [[maybe_unused]] const uint32_t in_h_padded = in_h + pad_h + ceil_pad_h;
    [[maybe_unused]] const uint32_t in_w_padded = in_w + pad_w + ceil_pad_w;
    const bool one_scalar_per_core = is_pool_op_one_scalar_per_core(
        pool_type, setup.ceil_mode, ceil_pad_h, ceil_pad_w, count_include_pad, pad_h, pad_w, divisor_override);

    // ---------------------------------------------------------------------------------------------
    // Op-owned tensors: build & upload the per-core sliding-window halo lookup table, and (for
    // avg-pool variants that need it) the per-output-stick scalar config tensor. These are
    // host-POPULATED, sharded ttnn::Tensors that get parked in ProgramArtifacts::op_owned_tensors.
    // The framework holds them alive (stable address) for the cached Program's life and reuses them
    // on a cache hit; we do NOT re-allocate them empty.
    // ---------------------------------------------------------------------------------------------
    std::vector<uint32_t> op_trace_metadata =
        ttnn::operations::sliding_window::generate_op_trace_metadata(sliding_window_config);
    std::vector<sliding_window::ShardBoundary> shard_boundaries =
        ttnn::operations::sliding_window::generate_shard_boundaries(sliding_window_config);
    std::vector<std::vector<uint16_t>> top_left_indices =
        sliding_window::generate_sliding_window_op_config(op_trace_metadata, shard_boundaries, stride_w);
    const uint32_t reader_indices_size = top_left_indices[0].size();

    Tensor reader_indices_host = sliding_window::construct_on_host_config_tensor(
        top_left_indices, setup.parallel_config, op_attr.config_tensor_in_dram);
    Tensor reader_indices_on_device = sliding_window::move_config_tensor_to_device(
        reader_indices_host, setup.parallel_config, is_block_sharded, input.device(), op_attr.config_tensor_in_dram);

    // op_owned_tensors[0] = reader_indices, op_owned_tensors[1] = scalar config (if !one_scalar_per_core)
    std::vector<tt::tt_metal::Tensor> op_owned_tensors;
    op_owned_tensors.reserve(2);
    op_owned_tensors.push_back(std::move(reader_indices_on_device));

    if (!one_scalar_per_core) {
        AvgPoolConfig avg_pool_config = {
            .kernel_h = kernel_h,
            .kernel_w = kernel_w,
            .in_h = in_h,
            .in_w = in_w,
            .out_h = out_h,
            .out_w = out_w,
            .stride_h = stride_h,
            .stride_w = stride_w,
            .ceil_mode = setup.ceil_mode,
            .ceil_h = ceil_pad_h,
            .ceil_w = ceil_pad_w,
            .count_include_pad = count_include_pad,
            .pad_t = pad_t,
            .pad_b = setup.pad_b,
            .pad_l = pad_l,
            .pad_r = setup.pad_r,
            .max_out_nhw_per_core = max_out_nhw_per_core,
            .divisor_override = divisor_override};
        Tensor config_tensor = create_scalar_config_tensor(
            avg_pool_config, input.memory_config().memory_layout(), in_n, num_shards_c, ncores, /*in_dram=*/false);

        const std::array<uint32_t, 2> shard_shape =
            std::array<uint32_t, 2>({1, static_cast<uint32_t>(config_tensor.logical_shape()[-1])});
        const tt::tt_metal::ShardOrientation config_tensor_shard_orientation = input.shard_spec().value().orientation;
        const tt::tt_metal::ShardSpec config_shard_spec(
            input.shard_spec().value().grid, shard_shape, config_tensor_shard_orientation);
        const MemoryConfig l1_small_memory_config{
            TensorMemoryLayout::HEIGHT_SHARDED, BufferType::L1_SMALL, config_shard_spec};

        op_owned_tensors.push_back(config_tensor.to_device(input.device(), l1_small_memory_config));
    }

    // Bind the op-owned tensors AFTER the vector is fully populated (identity footgun: TensorArguments
    // must reference the vector's elements, never a pre-move local).
    const tt::tt_metal::Tensor& reader_indices_owned = op_owned_tensors[0];
    const tt::tt_metal::Tensor* config_owned = one_scalar_per_core ? nullptr : &op_owned_tensors[1];

    tt::tt_metal::Buffer* reader_indices_buffer = reader_indices_owned.buffer();
    tt::tt_metal::Buffer* config_buffer = config_owned != nullptr ? config_owned->buffer() : nullptr;

    // ---------------------------------------------------------------------------------------------
    // CB / DFB sizing (unchanged from the legacy factory; just feeds DataflowBufferSpecs now).
    // ---------------------------------------------------------------------------------------------
    auto output_shard_shape = output_tensors[0].shard_spec().value().shape;
    PoolCBSizes cb_sizes = calculate_pool_cb_sizes(
        params,
        one_scalar_per_core,
        return_indices,
        output_layout,
        output_tensors[0].dtype(),
        {output_shard_shape[0], output_shard_shape[1]},
        /*config_tensor_in_dram=*/false,
        /*reader_indices_actual_page_size=*/std::nullopt);

    const auto& input_shape = input.padded_shape();
    const uint32_t shard_width = input.shard_spec()->shape[1];
    const uint32_t in_c_per_shard_ceil = in_c % shard_width != 0 && num_shards_c > 1
                                             ? (in_c - (in_c % shard_width)) / (num_shards_c - 1)
                                             : in_c / num_shards_c;
    const uint32_t in_nbytes_c = in_c_per_shard_ceil * params.nbytes;  // row of input (channels)
    const uint32_t shard_width_bytes = input_shape[3] / num_shards_c * params.nbytes;

    TT_FATAL(
        input_shape[3] % num_shards_c == 0,
        "Input channels {} should be divisible by number of shards {}",
        input_shape[3],
        num_shards_c);
    const uint32_t in_nbytes_leftover =
        params.is_wide_reduction &&
                (input_shape[3] / num_shards_c) % (params.MAX_TILES_PER_REDUCTION * tt::constants::TILE_WIDTH) != 0
            ? tt::round_up(
                  (input_shape[3] / num_shards_c) % (params.MAX_TILES_PER_REDUCTION * tt::constants::TILE_WIDTH),
                  tt::constants::TILE_WIDTH) *
                  params.nbytes
            : tt::round_up(input_shape[3] / num_shards_c, tt::constants::TILE_WIDTH) * params.nbytes;

    // ---- ProgramSpec (immutable) ----
    m2::ProgramSpec spec;
    spec.name = "pool2d_multi_core";

    // DFBs are added to this vector; the helper lambdas mirror the legacy add_local_cb / add_sharded_cb.
    auto& dfbs = spec.dataflow_buffers;
    auto add_local_dfb = [&](const char* name,
                             uint32_t page_size,
                             uint32_t num_pages,
                             tt::DataFormat data_format,
                             std::optional<FaceGeometry> face_geometry = std::nullopt,
                             std::optional<tt::tt_metal::Tile> tile = std::nullopt) {
        dfbs.push_back(m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{name},
            .entry_size = page_size,
            .num_entries = num_pages,
            .data_format_metadata = data_format,
            .tile_format_metadata = tile,
            .unpack_face_geometry_metadata = face_geometry,
        });
    };
    // Borrowed (globally-allocated) DFB: backing memory comes from a TensorParameter's buffer.
    // The backing L1 address resolves at runtime from the corresponding TensorArgument.
    auto add_borrowed_dfb = [&](const char* name,
                                uint32_t page_size,
                                uint32_t num_pages,
                                tt::DataFormat data_format,
                                const char* tensor_param_name,
                                std::optional<FaceGeometry> face_geometry = std::nullopt,
                                std::optional<tt::tt_metal::Tile> tile = std::nullopt) {
        dfbs.push_back(m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{name},
            .entry_size = page_size,
            .num_entries = num_pages,
            .data_format_metadata = data_format,
            .tile_format_metadata = tile,
            .unpack_face_geometry_metadata = face_geometry,
            .borrowed_from = m2::TensorParamName{tensor_param_name},
        });
    };

    // ---- scalar CBs ----
    const auto scalar_face_geometry = FaceGeometry{.face_r_dim = 1, .num_faces = 4};
    add_local_dfb(
        DFB_IN_SCALAR_0,
        cb_sizes.scalar_cb_pagesize,
        cb_sizes.scalar_cb_npages,
        params.data_format,
        scalar_face_geometry);
    if (cb_sizes.has_second_scalar_cb) {
        add_local_dfb(
            DFB_IN_SCALAR_1,
            cb_sizes.scalar_cb_pagesize,
            cb_sizes.scalar_cb_npages,
            params.data_format,
            scalar_face_geometry);
    }

    // CB storing just "clear value" (-inf for maxpool, 0 for avgpool)
    add_local_dfb(DFB_CLEAR_VALUE, cb_sizes.clear_value_cb_size, 1, params.data_format);

    // incoming data is the input cb instead of raw l1/dram addr; this input shard has halo and
    // padding inserted. Borrowed from the input tensor (read only as an address source by the reader).
    const uint32_t raw_in_cb_npages = input.shard_spec().value().shape[0];
    const uint32_t raw_in_cb_pagesize = in_nbytes_c;
    add_borrowed_dfb(DFB_RAW_IN, raw_in_cb_pagesize, raw_in_cb_npages, params.data_format, TP_INPUT);

    // reader indices (L1 path: sharded, backed by the op-owned reader_indices tensor)
    const uint32_t in_reader_indices_cb_pagesize =
        tt::round_up(reader_indices_size, 4);  // pagesize needs to be multiple of 4
    constexpr uint32_t in_reader_indices_cb_npages = 1;
    const uint32_t max_reader_indices_size =
        (max_out_nhw_per_core * 3 * sizeof(uint16_t)) + 2;  // worst case of 3 indices per output element
    const uint32_t actual_reader_indices_buffer_page_size = reader_indices_buffer->page_size();
    TT_FATAL(
        actual_reader_indices_buffer_page_size <= max_reader_indices_size,
        "Reader indices buffer page size {} exceeds max expected size {}",
        actual_reader_indices_buffer_page_size,
        max_reader_indices_size);
    add_borrowed_dfb(
        DFB_READER_INDICES,
        in_reader_indices_cb_pagesize,
        in_reader_indices_cb_npages,
        tt::DataFormat::UInt16,
        TP_READER_INDICES);

    // in_nblocks_c is factory-specific (used for kernel args), derived from the shared in_cb_raw_size
    uint32_t in_nblocks_c = 1;
    if (return_indices || params.is_wide_reduction) {
        in_nblocks_c =
            static_cast<uint32_t>(std::ceil(static_cast<float>(params.in_ntiles_c) / params.MAX_TILES_PER_REDUCTION));
    }

    // reader output == input to tilize
    const uint32_t in_cb_pagesize = cb_sizes.in_cb_pagesize;
    const uint32_t in_cb_npages = cb_sizes.in_cb_npages;

    const uint32_t window_size_hw = kernel_h * kernel_w;
    const uint32_t raw_face_r = std::min(window_size_hw, 16u);
    const uint32_t num_faces_in_input_tile_for_cb =
        (params.max_rows_for_reduction < tt::constants::TILE_HEIGHT || window_size_hw <= tt::constants::FACE_HEIGHT)
            ? 2u
            : 4u;
    const std::optional<FaceGeometry> input_face_geometry =
        return_indices
            ? std::nullopt
            : std::optional{FaceGeometry{.face_r_dim = raw_face_r, .num_faces = num_faces_in_input_tile_for_cb}};

    add_local_dfb(DFB_IN_0, in_cb_pagesize, in_cb_npages, params.data_format, input_face_geometry);
    if (cb_sizes.has_split_reader) {
        add_local_dfb(DFB_IN_1, in_cb_pagesize, in_cb_npages, params.data_format, input_face_geometry);
    }

    const bool is_output_tiled = output_layout == Layout::TILE;
    const bool is_output_block_format = is_block_float(output_tensors[0].dtype());
    const bool zero_pages = is_output_tiled && is_output_block_format;

    constexpr uint32_t pack_untilize_face_r_dim = 1;
    const bool last_tile_is_partial = in_c % tt::constants::TILE_WIDTH != 0;
    const bool single_partial_fits_in_face = last_tile_is_partial && in_c <= tt::constants::FACE_WIDTH;
    const uint32_t pack_untilize_num_faces = single_partial_fits_in_face ? 1u : 2u;
    const auto pack_untilize_face_geometry =
        FaceGeometry{.face_r_dim = pack_untilize_face_r_dim, .num_faces = pack_untilize_num_faces};
    const std::optional<tt::tt_metal::Tile> pack_untilize_tile =
        single_partial_fits_in_face ? std::optional{tt::tt_metal::Tile{{1, tt::constants::FACE_WIDTH}, false}}
                                    : std::nullopt;

    // pre_tilize_cb / fast_tilize_cb are two ALIASED DFBs: one L1 allocation backs two DFBs that
    // present different face_geometry/page_size views of the same bytes (legacy multi-format CB).
    //   * pre_tilize -- producer view, used by pack_untilize. Stick-packed (face_r_dim=1,
    //                   num_faces<=2), pagesize=TILE_WIDTH*nbytes.
    //   * fast_tilize -- consumer view, used by fast_tilize. Full-tile (face_r_dim=16, num_faces=4),
    //                    pagesize=tile_size.
    // The kernel keeps both FIFOs in lock-step by pushing/popping the same byte amount per round,
    // so their rd/wr pointers always point at matching L1 addresses.
    if (cb_sizes.has_pre_tilize) {
        const uint32_t pre_tilize_total_size = cb_sizes.pre_tilize_cb_pagesize * cb_sizes.pre_tilize_cb_npages;
        const uint32_t fast_tilize_page_size = tt::tile_size(params.data_format);
        TT_FATAL(
            pre_tilize_total_size % fast_tilize_page_size == 0,
            "pre_tilize_cb total size {} must be a multiple of fast_tilize tile size {}",
            pre_tilize_total_size,
            fast_tilize_page_size);
        // pre_tilize (producer view)
        dfbs.push_back(m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{DFB_PRE_TILIZE},
            .entry_size = cb_sizes.pre_tilize_cb_pagesize,
            .num_entries = cb_sizes.pre_tilize_cb_npages,
            .data_format_metadata = params.data_format,
            .tile_format_metadata = pack_untilize_tile,
            .unpack_face_geometry_metadata = pack_untilize_face_geometry,
            .advanced_options = m2::DFBAdvancedOptions{.alias_with = {m2::DFBSpecName{DFB_FAST_TILIZE}}},
        });
        // fast_tilize (consumer view): full-tile face_geometry (default 32x32, 4 faces of 16x16).
        dfbs.push_back(m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{DFB_FAST_TILIZE},
            .entry_size = fast_tilize_page_size,
            .num_entries = pre_tilize_total_size / fast_tilize_page_size,
            .data_format_metadata = params.data_format,
            .advanced_options = m2::DFBAdvancedOptions{.alias_with = {m2::DFBSpecName{DFB_PRE_TILIZE}}},
        });
    }

    // out CB: borrowed from the output tensor (compute writes directly into the sharded output buffer).
    add_borrowed_dfb(
        DFB_OUT,
        cb_sizes.out_cb_pagesize,
        cb_sizes.out_cb_npages,
        params.output_data_format,
        TP_OUTPUT,
        is_output_tiled ? std::nullopt : std::optional{pack_untilize_face_geometry},
        is_output_tiled ? std::nullopt : pack_untilize_tile);

    for (const auto& out : output_tensors) {
        TT_FATAL(
            out.memory_config().is_sharded(),
            "Output memory config needs to be sharded, but got {}",
            out.memory_config());
    }

    // config (scalar config) CB: borrowed from the op-owned scalar config tensor (avg-pool only).
    if (!one_scalar_per_core) {
        TT_FATAL(config_buffer != nullptr, "scalar config buffer must be populated when !one_scalar_per_core");
        uint32_t max_config_tensor_size =
            max_out_nhw_per_core * 3 * sizeof(uint16_t);  // worst case of 3 entries per output element
        TT_FATAL(
            config_buffer->page_size() <= max_config_tensor_size,
            "Config tensor buffer page size {} exceeds max expected size {}",
            config_buffer->page_size(),
            max_config_tensor_size);
        add_borrowed_dfb(DFB_CONFIG, config_buffer->page_size(), 1, tt::DataFormat::RawUInt32, TP_CONFIG);
    }

    // ---------------------------------------------------------------------------------------------
    // Reader kernel CTAs. CB-index CTA slots become DFB bindings (named via dfb::), so they are NOT
    // in this Table. The four legacy CTA slots that carried buffer addresses/page-sizes for the DRAM
    // path (config_dram_addr/page_size, reader_dram_addr/page_size) are passed as named 0 constants
    // — the L1 path's kernel never reads them (they are under `if constexpr (config_in_dram)`), and
    // no raw address crosses the CTA channel.
    // ---------------------------------------------------------------------------------------------
    m2::KernelSpec::CompileTimeArgs reader_cta = {
        {"reader_nindices", max_out_nhw_per_core},
        {"kernel_h", kernel_h},
        {"kernel_w", kernel_w},
        {"pad_w", pad_w},
        {"in_nbytes_leftover", in_nbytes_leftover},
        {"in_w", in_w},
        {"in_c", in_c_per_shard_ceil},
        {"split_reader", params.split_reader},
        // reader_id is the only per-kernel-differing CTA; set per reader below.
        {"bf16_scalar", bf16_scalar},
        {"bf16_init_value", bf16_init_value},
        {"in_nblocks_c", in_nblocks_c},
        {"in_cb_sz", cb_sizes.in_cb_raw_size},
        {"max_sticks_for_reduction", params.max_rows_for_reduction},
        {"ceil_pad_w", ceil_pad_w},
        {"pool_type", static_cast<uint32_t>(pool_type)},
        {"one_scalar_per_core", one_scalar_per_core},
        {"in_nbytes_c", in_nbytes_c},
        {"shard_width_bytes", shard_width_bytes},
        {"multi_buffering_factor", params.multi_buffering_factor},
        {"stride_w", stride_w},
        {"dilation_h", dilation_h},
        {"dilation_w", dilation_w},
        {"zero_pages", static_cast<uint32_t>(zero_pages)},
        // DRAM-path CTAs: forced to 0 on the L1 path (see note above). config_in_dram drives the
        // dead `if constexpr` branches in the kernel, so the addresses are never consumed.
        {"config_in_dram", 0},
        {"config_dram_addr", 0},
        {"config_page_size", 0},
        {"reader_dram_addr", 0},
        {"reader_page_size", 0},
    };

    // Per-reader DFB bindings. reader0 produces in_0 + in_scalar_0; reader1 produces in_1 +
    // in_scalar_1 (each reader writes its OWN input/scalar CB; compute consumes both). raw_in /
    // reader_indices / config are borrowed-memory DFBs the reader reads only as an address source
    // (get_read_ptr) — bound as self-loop (PRODUCER + CONSUMER on the reader) to satisfy the DFB
    // producer/consumer invariant (no real FIFO peer). clear_value is filled by reader0 (PRODUCER)
    // and waited on by reader1 (CONSUMER) — bind both endpoints on each reader (self-loop) so the
    // per-node invariant holds regardless of which reader runs on the node.
    // Both reader KernelSpecs compile the SAME source, so each binds its own input/scalar DFB under
    // the SAME kernel-side accessor name ("in" / "in_scalar"). reader0 -> in_0/in_scalar_0,
    // reader1 -> in_1/in_scalar_1. The kernel just constructs dfb::in / dfb::in_scalar.
    auto reader_dfb_bindings = [&](uint32_t reader_id) {
        m2::Group<m2::DFBBinding> b;
        const char* in_name = (reader_id == 0) ? DFB_IN_0 : DFB_IN_1;
        b.push_back(m2::DFBBinding{
            .dfb_spec_name = m2::DFBSpecName{in_name},
            .accessor_name = "in",
            .endpoint_type = m2::DFBEndpointType::PRODUCER});
        // scalar CB binding (accessor "in_scalar"):
        //   - reader0 always produces in_scalar_0.
        //   - reader1 produces in_scalar_1 ONLY when a second scalar CB exists (avg && split &&
        //     !one_scalar). Otherwise reader1 does NOT bind a scalar CB (it never touches one), so a
        //     second producer/consumer of in_scalar_0 is not introduced on the node. The kernel gates
        //     its dfb::in_scalar handle on the READER_BINDS_SCALAR define accordingly.
        const bool reader_binds_scalar = (reader_id == 0) || cb_sizes.has_second_scalar_cb;
        if (reader_binds_scalar) {
            const char* scalar_name = (reader_id == 1) ? DFB_IN_SCALAR_1 : DFB_IN_SCALAR_0;
            b.push_back(m2::DFBBinding{
                .dfb_spec_name = m2::DFBSpecName{scalar_name},
                .accessor_name = "in_scalar",
                .endpoint_type = m2::DFBEndpointType::PRODUCER});
        }
        // clear_value: reader0 fills it (PRODUCER), reader1 waits on it (CONSUMER). With split_reader
        // the two readers form the producer/consumer pair on the node. Without split_reader, reader0
        // is the only kernel, so it self-loops (PRODUCER + CONSUMER) to satisfy the invariant.
        if (reader_id == 0) {
            b.push_back(m2::DFBBinding{
                .dfb_spec_name = m2::DFBSpecName{DFB_CLEAR_VALUE},
                .accessor_name = "clear_value",
                .endpoint_type = m2::DFBEndpointType::PRODUCER});
            if (!cb_sizes.has_split_reader) {
                b.push_back(m2::DFBBinding{
                    .dfb_spec_name = m2::DFBSpecName{DFB_CLEAR_VALUE},
                    .accessor_name = "clear_value",
                    .endpoint_type = m2::DFBEndpointType::CONSUMER});
            }
        } else {
            b.push_back(m2::DFBBinding{
                .dfb_spec_name = m2::DFBSpecName{DFB_CLEAR_VALUE},
                .accessor_name = "clear_value",
                .endpoint_type = m2::DFBEndpointType::CONSUMER});
        }
        // raw_in / reader_indices / config are borrowed-memory DFBs read by BOTH readers purely as an
        // address source (get_read_ptr). To satisfy "exactly one producer + one consumer per node"
        // without a real FIFO: with split_reader, reader0 = PRODUCER, reader1 = CONSUMER; without
        // split, reader0 self-loops (PRODUCER + CONSUMER). The endpoint role is a formality here — the
        // kernel only reads the backing tensor's base pointer.
        auto bind_address_source = [&](const char* dfb_name, const char* accessor) {
            if (reader_id == 0) {
                b.push_back(m2::DFBBinding{
                    .dfb_spec_name = m2::DFBSpecName{dfb_name},
                    .accessor_name = accessor,
                    .endpoint_type = m2::DFBEndpointType::PRODUCER});
                if (!cb_sizes.has_split_reader) {
                    b.push_back(m2::DFBBinding{
                        .dfb_spec_name = m2::DFBSpecName{dfb_name},
                        .accessor_name = accessor,
                        .endpoint_type = m2::DFBEndpointType::CONSUMER});
                }
            } else {
                b.push_back(m2::DFBBinding{
                    .dfb_spec_name = m2::DFBSpecName{dfb_name},
                    .accessor_name = accessor,
                    .endpoint_type = m2::DFBEndpointType::CONSUMER});
            }
        };
        bind_address_source(DFB_RAW_IN, "raw_in");
        bind_address_source(DFB_READER_INDICES, "reader_indices");
        if (!one_scalar_per_core) {
            bind_address_source(DFB_CONFIG, "config");
        }
        return b;
    };

    // Tensor bindings the reader needs (raw_in / reader_indices / config back borrowed DFBs).
    auto reader_tensor_bindings = [&]() {
        m2::Group<m2::TensorBinding> t;
        t.push_back(
            m2::TensorBinding{.tensor_parameter_name = m2::TensorParamName{TP_INPUT}, .accessor_name = "input"});
        t.push_back(m2::TensorBinding{
            .tensor_parameter_name = m2::TensorParamName{TP_READER_INDICES}, .accessor_name = "reader_indices_tensor"});
        if (!one_scalar_per_core) {
            t.push_back(m2::TensorBinding{
                .tensor_parameter_name = m2::TensorParamName{TP_CONFIG}, .accessor_name = "config_tensor"});
        }
        return t;
    };

    auto make_reader = [&](const char* id, uint32_t reader_id, DataMovementProcessor proc) {
        m2::KernelSpec::CompileTimeArgs cta = reader_cta;
        cta.insert({"reader_id", reader_id});
        // Per-reader conditional-DFB defines (gate the dfb:: tokens that aren't bound on this kernel).
        m2::KernelSpec::CompilerOptions::Defines defines;
        const bool reader_binds_scalar = (reader_id == 0) || cb_sizes.has_second_scalar_cb;
        if (reader_binds_scalar) {
            defines.insert({"READER_BINDS_SCALAR", "1"});
        }
        if (!one_scalar_per_core) {
            defines.insert({"HAS_CONFIG_CB", "1"});
        }
        return m2::KernelSpec{
            .unique_id = m2::KernelSpecName{id},
            .source = std::filesystem::path{READER_KERNEL_PATH},
            .compiler_options = {.defines = std::move(defines)},
            .dfb_bindings = reader_dfb_bindings(reader_id),
            .tensor_bindings = reader_tensor_bindings(),
            .compile_time_args = std::move(cta),
            .runtime_arg_schema = {.runtime_arg_names = {"out_nhw_this_core", "core_nhw_index"}},
            .hw_config =
                m2::DataMovementHardwareConfig{
                    .role = (proc == DataMovementProcessor::RISCV_0) ? m2::DataMovementRoleHint::READER
                                                                     : m2::DataMovementRoleHint::WRITER},
        };
    };

    m2::KernelSpec reader0 = make_reader(KERNEL_READER0, 0, DataMovementProcessor::RISCV_0);
    std::optional<m2::KernelSpec> reader1;
    if (params.split_reader) {
        reader1 = make_reader(KERNEL_READER1, 1, DataMovementProcessor::RISCV_1);
    }

    // ---- Compute kernel ----
    m2::KernelSpec::CompileTimeArgs compute_cta = {
        {"in_ntiles_c", params.in_ntiles_c},
        {"window_size_hw", kernel_h * kernel_w},
        {"split_reader", params.split_reader},
        {"max_out_sticks_per_core", 0},  // used for grid sample but not for pool
        {"in_c", in_c_per_shard_ceil},
        {"in_nblocks_c", in_nblocks_c},
        {"max_sticks_for_reduction", params.max_rows_for_reduction},
        {"one_scalar_per_core", one_scalar_per_core},
        {"is_output_tiled", is_output_tiled},
        {"is_output_block_format", is_output_block_format},
        {"force_max_tiles_per_reduction_4", 0},  // off for pool2d
    };

    // Get device arch for compute kernel config initialization
    auto device_arch = input.device()->arch();
    auto device_compute_kernel_config = init_device_compute_kernel_config(
        device_arch,
        op_attr.compute_kernel_config_,
        tt::tt_metal::MathFidelity::HiFi4,
        false,                                                             // math_approx_mode
        (params.is_avg_pool && params.is_large_kernel) || return_indices,  // fp32_dest_acc_en
        false,                                                             // packer_l1_acc
        (params.is_large_kernel && return_indices) || return_indices       // dst_full_sync_en
    );

    const auto pool_defines_map = get_defines(pool_type);
    m2::KernelSpec::CompilerOptions::Defines compute_defines;
    for (const auto& [k, v] : pool_defines_map) {
        compute_defines.insert({k, v});
    }
    // Conditional DFB bindings need a matching kernel-side preprocessor flag (Metal 2.0 conditional-
    // binding pattern): the dfb:: token for a DFB that isn't bound on this path must not enter name
    // lookup. SPLIT_READER gates dfb::in_1; HAS_SECOND_SCALAR_CB gates dfb::in_scalar_1
    // (is_avg_pool && split_reader && !one_scalar_per_core); IS_OUTPUT_TILED gates the aliased
    // dfb::pre_tilize / dfb::fast_tilize pair.
    if (cb_sizes.has_split_reader) {
        compute_defines.insert({"SPLIT_READER", "1"});
    }
    if (cb_sizes.has_second_scalar_cb) {
        compute_defines.insert({"HAS_SECOND_SCALAR_CB", "1"});
    }
    if (cb_sizes.has_pre_tilize) {
        compute_defines.insert({"IS_OUTPUT_TILED", "1"});
    }

    // Compute DFB bindings: consumes in_0/in_1/scalars, produces out, and self-loops the aliased
    // pre_tilize/fast_tilize pair (compute is their only user).
    m2::Group<m2::DFBBinding> compute_dfb_bindings;
    compute_dfb_bindings.push_back(m2::DFBBinding{
        .dfb_spec_name = m2::DFBSpecName{DFB_IN_0},
        .accessor_name = "in_0",
        .endpoint_type = m2::DFBEndpointType::CONSUMER});
    if (cb_sizes.has_split_reader) {
        compute_dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = m2::DFBSpecName{DFB_IN_1},
            .accessor_name = "in_1",
            .endpoint_type = m2::DFBEndpointType::CONSUMER});
    }
    compute_dfb_bindings.push_back(m2::DFBBinding{
        .dfb_spec_name = m2::DFBSpecName{DFB_IN_SCALAR_0},
        .accessor_name = "in_scalar_0",
        .endpoint_type = m2::DFBEndpointType::CONSUMER});
    if (cb_sizes.has_second_scalar_cb) {
        compute_dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = m2::DFBSpecName{DFB_IN_SCALAR_1},
            .accessor_name = "in_scalar_1",
            .endpoint_type = m2::DFBEndpointType::CONSUMER});
    }
    // out: compute produces it (and is its only user — self-loop to satisfy the invariant).
    compute_dfb_bindings.push_back(m2::DFBBinding{
        .dfb_spec_name = m2::DFBSpecName{DFB_OUT},
        .accessor_name = "out",
        .endpoint_type = m2::DFBEndpointType::PRODUCER});
    compute_dfb_bindings.push_back(m2::DFBBinding{
        .dfb_spec_name = m2::DFBSpecName{DFB_OUT},
        .accessor_name = "out",
        .endpoint_type = m2::DFBEndpointType::CONSUMER});
    if (cb_sizes.has_pre_tilize) {
        compute_dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = m2::DFBSpecName{DFB_PRE_TILIZE},
            .accessor_name = "pre_tilize",
            .endpoint_type = m2::DFBEndpointType::PRODUCER});
        compute_dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = m2::DFBSpecName{DFB_PRE_TILIZE},
            .accessor_name = "pre_tilize",
            .endpoint_type = m2::DFBEndpointType::CONSUMER});
        compute_dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = m2::DFBSpecName{DFB_FAST_TILIZE},
            .accessor_name = "fast_tilize",
            .endpoint_type = m2::DFBEndpointType::PRODUCER});
        compute_dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = m2::DFBSpecName{DFB_FAST_TILIZE},
            .accessor_name = "fast_tilize",
            .endpoint_type = m2::DFBEndpointType::CONSUMER});
    }

    m2::KernelSpec compute{
        .unique_id = m2::KernelSpecName{KERNEL_COMPUTE},
        .source = std::filesystem::path{COMPUTE_KERNEL_PATH},
        .compiler_options = {.defines = std::move(compute_defines)},
        .dfb_bindings = std::move(compute_dfb_bindings),
        .compile_time_args = std::move(compute_cta),
        // compute_pool_2d only reads out_nhw_this_core (RTA slot 0); core_nhw_index is reader-only.
        .runtime_arg_schema = {.runtime_arg_names = {"out_nhw_this_core"}},
        .hw_config =
            m2::ComputeHardwareConfig{
                .math_fidelity = get_math_fidelity(device_compute_kernel_config),
                .fp32_dest_acc_en = get_fp32_dest_acc_en(device_compute_kernel_config),
                .dst_full_sync_en = get_dst_full_sync_en(device_compute_kernel_config),
                .math_approx_mode = false,
            },
    };

    // ---- Tensor parameters ----
    spec.tensor_parameters.push_back(
        m2::TensorParameter{.unique_id = m2::TensorParamName{TP_INPUT}, .spec = input.tensor_spec()});
    spec.tensor_parameters.push_back(
        m2::TensorParameter{.unique_id = m2::TensorParamName{TP_OUTPUT}, .spec = output_tensors[0].tensor_spec()});
    spec.tensor_parameters.push_back(m2::TensorParameter{
        .unique_id = m2::TensorParamName{TP_READER_INDICES}, .spec = reader_indices_owned.tensor_spec()});
    if (!one_scalar_per_core) {
        spec.tensor_parameters.push_back(
            m2::TensorParameter{.unique_id = m2::TensorParamName{TP_CONFIG}, .spec = config_owned->tensor_spec()});
    }

    // ---- Kernels + WorkUnitSpec ----
    // All kernels share the single core grid `all_cores`. The local DFBs (in_0/in_1/scalars/
    // clear_value/pre_tilize/fast_tilize) require their producer and consumer to share the SAME
    // WorkUnitSpec — so reader0 (+reader1) and compute all live in one WorkUnitSpec on all_cores.
    m2::Group<m2::KernelSpecName> wu_kernels;
    spec.kernels.push_back(reader0);
    wu_kernels.push_back(m2::KernelSpecName{KERNEL_READER0});
    if (reader1.has_value()) {
        spec.kernels.push_back(*reader1);
        wu_kernels.push_back(m2::KernelSpecName{KERNEL_READER1});
    }
    spec.kernels.push_back(compute);
    wu_kernels.push_back(m2::KernelSpecName{KERNEL_COMPUTE});

    spec.work_units.push_back(m2::WorkUnitSpec{
        .name = "pool2d",
        .kernels = std::move(wu_kernels),
        .target_nodes = all_cores,
    });

    // ---- ProgramRunArgs (mutable; per-core runtime args + tensor args) ----
    m2::ProgramRunArgs run;
    m2::KernelRunArgs reader0_run{.kernel = m2::KernelSpecName{KERNEL_READER0}};
    m2::KernelRunArgs reader1_run{.kernel = m2::KernelSpecName{KERNEL_READER1}};
    m2::KernelRunArgs compute_run{.kernel = m2::KernelSpecName{KERNEL_COMPUTE}};

    // set the starting indices for each core as runtime args
    uint32_t total_out_nhw = in_n * out_h * out_w;
    for (uint32_t core_i = 0; core_i < ncores; core_i++) {
        const uint32_t core_x_i = core_i % rectangular_x;
        const uint32_t core_y_i = core_i / rectangular_x;
        const CoreCoord core(core_x_i, core_y_i);

        uint32_t total_out_nhw_processed;
        uint32_t core_nhw_index = 0;
        if (is_block_sharded) {
            total_out_nhw_processed = core_y_i * max_out_nhw_per_core;
            core_nhw_index = core_y_i;
        } else if (is_width_sharded) {
            total_out_nhw_processed = 0;
            core_nhw_index = 0;
        } else {
            total_out_nhw_processed = core_i * max_out_nhw_per_core;
            core_nhw_index = core_i;
        }
        uint32_t remaining_out_nhw =
            total_out_nhw_processed < total_out_nhw ? total_out_nhw - total_out_nhw_processed : 0;
        uint32_t out_nhw_this_core = std::min(max_out_nhw_per_core, remaining_out_nhw);

        m2::ProgramRunArgs::KernelRunArgs::RuntimeArgValues reader_args = {
            {"out_nhw_this_core", out_nhw_this_core}, {"core_nhw_index", core_nhw_index}};

        reader0_run.runtime_arg_values.push_back({core, reader_args});
        if (params.split_reader) {
            reader1_run.runtime_arg_values.push_back({core, reader_args});
        }
        // compute reads only out_nhw_this_core.
        compute_run.runtime_arg_values.push_back({core, {{"out_nhw_this_core", out_nhw_this_core}}});
    }

    run.kernel_run_args.push_back(std::move(reader0_run));
    if (params.split_reader) {
        run.kernel_run_args.push_back(std::move(reader1_run));
    }
    run.kernel_run_args.push_back(std::move(compute_run));

    // Tensor arguments: io tensors + op-owned tensors (referenced by mesh_tensor() identity).
    run.tensor_args = {
        {m2::TensorParamName{TP_INPUT}, input.mesh_tensor()},
        {m2::TensorParamName{TP_OUTPUT}, output_tensors[0].mesh_tensor()},
        {m2::TensorParamName{TP_READER_INDICES}, reader_indices_owned.mesh_tensor()},
    };
    if (!one_scalar_per_core) {
        run.tensor_args.insert({m2::TensorParamName{TP_CONFIG}, config_owned->mesh_tensor()});
    }

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec), .run_params = std::move(run), .op_owned_tensors = std::move(op_owned_tensors)};
}

}  // namespace ttnn::operations::pool

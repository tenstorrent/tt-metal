// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "pool_multi_core_program_factory.hpp"

#include "tt-metalium/constants.hpp"
#include "tt-metalium/tt_backend_api_types.hpp"
#include "tt-metalium/tile.hpp"
#include "tt-metalium/bfloat16.hpp"
#include "tt-metalium/math.hpp"
#include "tt-metalium/mesh_buffer.hpp"
#include "ttnn/operations/pool/pool_utils.hpp"
#include "tt-metalium/host_buffer.hpp"
#include "tt-metalium/buffer.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/storage.hpp"

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/tensor/tensor_apis.hpp>
#include <tt-metalium/experimental/tensor/mesh_tensor.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/hal.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::operations::pool::quasar {

namespace {

// ---------------------------------------------------------------------------
// Op-owned scalar-config tensor for avg pool (unchanged host computation).
// ---------------------------------------------------------------------------
struct ScalarInfo {
    uint16_t start;
    uint16_t value;
    uint16_t end;
};

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
        int h_start = (output_stick_x * config.stride_h) - config.pad_t;
        int w_start = (output_stick_y * config.stride_w) - config.pad_l;
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

        if (first_scalar || static_cast<uint32_t>(pool_area) != last_pool_area) {
            if (!scalars.empty()) {
                scalars.back().end = i;
            }
            scalars.push_back(
                {static_cast<uint16_t>(i),
                 std::bit_cast<uint16_t>(bfloat16::truncate(value)),
                 static_cast<uint16_t>(i)});
            first_scalar = false;
        }
        last_pool_area = static_cast<uint32_t>(pool_area);

        output_stick_y = (output_stick_y + 1) % config.out_w;
        if (output_stick_y == 0) {
            output_stick_x = (output_stick_x + 1) % config.out_h;
        }
    }

    scalars.back().end = config.max_out_nhw_per_core;
    return scalars;
}

void push_back_scalar_info_or_zero(
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

Tensor create_scalar_config_tensor(
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
        case TensorMemoryLayout::BLOCK_SHARDED: {
            for (const std::vector<ScalarInfo>& scalars : scalars_per_core) {
                uint32_t repeats = config_tensor_in_dram ? 1 : num_shards_c;
                push_back_scalar_info_or_zero(config_vector, scalars, max_scalars_cnt, repeats);
            }
            break;
        }
        case TensorMemoryLayout::WIDTH_SHARDED: {
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

// ---------------------------------------------------------------------------
// Metal 2.0 named resource handles (locals to avoid unity-build collisions).
// DFB names mirror the legacy CB-id variable names so the kernel-side dfb::<name>
// tokens line up 1:1 with the migrated kernels.
// ---------------------------------------------------------------------------
const TensorParamName INPUT_TENSOR{"input"};
const TensorParamName OUTPUT_TENSOR{"output"};
const TensorParamName OUTPUT_IDX_TENSOR{"output_idx"};
const TensorParamName READER_INDICES_TENSOR{"reader_indices"};
const TensorParamName CONFIG_TENSOR{"config"};

const DFBSpecName DFB_IN_SCALAR_0{"in_scalar_cb_0"};
const DFBSpecName DFB_IN_SCALAR_1{"in_scalar_cb_1"};
const DFBSpecName DFB_CLEAR_VALUE{"clear_value_cb"};
const DFBSpecName DFB_IN_SHARD{"in_shard_cb"};  // raw_in_cb (borrowed input)
const DFBSpecName DFB_READER_INDICES{"reader_indices_cb"};
const DFBSpecName DFB_IN_0{"in_cb_0"};
const DFBSpecName DFB_IN_1{"in_cb_1"};
const DFBSpecName DFB_IN_IDX{"in_idx_cb"};
const DFBSpecName DFB_PACK_TMP{"pack_tmp_cb"};
const DFBSpecName DFB_PACK_IDX_TMP{"pack_idx_tmp_cb"};
const DFBSpecName DFB_RIGHT_INC{"right_inc_cb"};
const DFBSpecName DFB_DOWN_LEFT{"down_left_wrap_inc_cb"};
const DFBSpecName DFB_UP_LEFT{"up_left_wrap_inc_cb"};
const DFBSpecName DFB_INTRA_RIGHT{"intra_kernel_right_inc_cb"};
const DFBSpecName DFB_INTRA_DOWN_LEFT{"intra_kernel_down_left_wrap_inc_cb"};
const DFBSpecName DFB_COMPUTE_TMP_IDX{"compute_tmp_idx_cb"};
const DFBSpecName DFB_PRE_TILIZE{"pre_tilize_cb"};
const DFBSpecName DFB_FAST_TILIZE{"fast_tilize_cb"};
const DFBSpecName DFB_OUT{"out_cb"};
const DFBSpecName DFB_OUT_IDX{"out_idx_cb"};
const DFBSpecName DFB_CONFIG{"config_cb"};

const KernelSpecName READER0_KERNEL{"reader0"};
const KernelSpecName READER1_KERNEL{"reader1"};
const KernelSpecName COMPUTE_KERNEL{"compute"};

constexpr const char* READER_POOL_PATH =
    "ttnn/cpp/ttnn/operations/experimental/quasar/pool_generic/device/kernels/dataflow/reader_pool_2d.cpp";
constexpr const char* READER_MPWI_PATH =
    "ttnn/cpp/ttnn/operations/experimental/quasar/pool_generic/device/kernels/dataflow/reader_mpwi.cpp";
constexpr const char* COMPUTE_POOL_PATH =
    "ttnn/cpp/ttnn/operations/experimental/quasar/pool_generic/device/kernels/compute/compute_pool_2d.cpp";
constexpr const char* COMPUTE_MPWI_PATH =
    "ttnn/cpp/ttnn/operations/experimental/quasar/pool_generic/device/kernels/compute/compute_mpwi.cpp";

// A local DFB (Program-lifetime L1 allocation).
DataflowBufferSpec local_dfb(
    const DFBSpecName& id,
    uint32_t page_size,
    uint32_t num_pages,
    tt::DataFormat df,
    std::optional<FaceGeometry> face = std::nullopt,
    std::optional<tt::tt_metal::Tile> tile = std::nullopt) {
    return DataflowBufferSpec{
        .unique_id = id,
        .entry_size = page_size,
        .num_entries = num_pages,
        .data_format_metadata = df,
        .tile_format_metadata = tile,
        .unpack_face_geometry_metadata = face,
    };
}

// A borrowed DFB (non-owning view over a TensorParameter's L1 storage).
DataflowBufferSpec borrowed_dfb(
    const DFBSpecName& id,
    uint32_t page_size,
    uint32_t num_pages,
    tt::DataFormat df,
    const TensorParamName& borrowed_from,
    std::optional<FaceGeometry> face = std::nullopt,
    std::optional<tt::tt_metal::Tile> tile = std::nullopt) {
    return DataflowBufferSpec{
        .unique_id = id,
        .entry_size = page_size,
        .num_entries = num_pages,
        .data_format_metadata = df,
        .tile_format_metadata = tile,
        .unpack_face_geometry_metadata = face,
        .borrowed_from = borrowed_from,
    };
}

}  // namespace

ttnn::device_operation::ProgramArtifacts pool2d_create_program_artifacts(
    const Pool2D::operation_attributes_t& op_attr,
    const Pool2D::tensor_args_t& tensor_args,
    Pool2D::tensor_return_value_t& output_tensors) {
    const Tensor& input = tensor_args.input_tensor_;
    const auto& sliding_window_config = op_attr.sliding_window_config_;
    PoolSetup setup = compute_pool_setup(op_attr, input);

    const Pool2DType pool_type = op_attr.pool_type_;
    const Layout output_layout = op_attr.output_layout_;
    const bool count_include_pad = op_attr.count_include_pad_;
    const std::optional<int32_t> divisor_override = op_attr.divisor_override_;
    const bool return_indices = op_attr.return_indices_;
    const bool config_tensor_in_dram = op_attr.config_tensor_in_dram;

    // -----------------------------------------------------------------------
    // Op-owned tensors: sliding-window reader-indices table (always) and the
    // avg-pool scalar config tensor (only when !one_scalar_per_core).
    // -----------------------------------------------------------------------
    std::vector<uint32_t> op_trace_metadata =
        ttnn::operations::sliding_window::generate_op_trace_metadata(sliding_window_config);
    std::vector<sliding_window::ShardBoundary> shard_boundaries =
        ttnn::operations::sliding_window::generate_shard_boundaries(sliding_window_config);
    std::vector<std::vector<uint16_t>> top_left_indices =
        sliding_window::generate_sliding_window_op_config(op_trace_metadata, shard_boundaries, setup.stride_w);

    auto* mesh_device = input.device();
    auto& cq = mesh_device->mesh_command_queue();

    // Reader-indices table: build on host, then upload to an OWNING MeshTensor (parked in
    // op_owned_tensors). The memory config mirrors sliding_window::move_config_tensor_to_device.
    Tensor reader_indices_host =
        sliding_window::construct_on_host_config_tensor(top_left_indices, setup.parallel_config, config_tensor_in_dram);
    MemoryConfig reader_indices_mem_config = [&]() -> MemoryConfig {
        if (config_tensor_in_dram) {
            return MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
        }
        const std::array<uint32_t, 2> ri_shard_shape{1, static_cast<uint32_t>(reader_indices_host.logical_shape()[-1])};
        const tt::tt_metal::ShardOrientation ri_orient =
            setup.is_block_sharded
                ? (setup.parallel_config.shard_orientation == tt::tt_metal::ShardOrientation::COL_MAJOR
                       ? tt::tt_metal::ShardOrientation::ROW_MAJOR
                       : tt::tt_metal::ShardOrientation::COL_MAJOR)
                : tt::tt_metal::ShardOrientation::ROW_MAJOR;
        const tt::tt_metal::ShardSpec ri_shard_spec(setup.parallel_config.grid, ri_shard_shape, ri_orient);
        return MemoryConfig{TensorMemoryLayout::HEIGHT_SHARDED, BufferType::L1_SMALL, ri_shard_spec};
    }();
    MeshTensor reader_indices_owned = tt::tt_metal::enqueue_write_tensor(
        cq, reader_indices_host.host_tensor(), *mesh_device, reader_indices_mem_config);
    const TensorSpec reader_indices_spec = reader_indices_owned.tensor_spec();
    const uint32_t reader_indices_page_size = reader_indices_owned.mesh_buffer().page_size();

    const uint32_t pad_h = setup.pad_t + setup.pad_b;
    const uint32_t pad_w = setup.pad_l + setup.pad_r;
    const bool one_scalar_per_core = is_pool_op_one_scalar_per_core(
        pool_type,
        setup.ceil_mode,
        setup.ceil_pad_h,
        setup.ceil_pad_w,
        count_include_pad,
        pad_h,
        pad_w,
        divisor_override);

    // Op-owned avg-pool scalar config tensor (only when !one_scalar_per_core).
    std::optional<MeshTensor> config_tensor_owned;
    std::optional<TensorSpec> config_tensor_spec;
    uint32_t config_buffer_page_size = 0;
    if (!one_scalar_per_core) {
        const uint32_t max_out_nhw_per_core = output_tensors[0].shard_spec()->shape[0];
        const uint32_t ncores = input.shard_spec().value().grid.num_cores();
        AvgPoolConfig avg_pool_config = {
            .kernel_h = setup.kernel_h,
            .kernel_w = setup.kernel_w,
            .in_h = setup.in_h,
            .in_w = setup.in_w,
            .out_h = setup.out_h,
            .out_w = setup.out_w,
            .stride_h = setup.stride_h,
            .stride_w = setup.stride_w,
            .ceil_mode = setup.ceil_mode,
            .ceil_h = setup.ceil_pad_h,
            .ceil_w = setup.ceil_pad_w,
            .count_include_pad = count_include_pad,
            .pad_t = setup.pad_t,
            .pad_b = setup.pad_b,
            .pad_l = setup.pad_l,
            .pad_r = setup.pad_r,
            .max_out_nhw_per_core = max_out_nhw_per_core,
            .divisor_override = divisor_override};
        Tensor config_tensor = create_scalar_config_tensor(
            avg_pool_config,
            input.memory_config().memory_layout(),
            setup.in_n,
            setup.num_shards_c,
            ncores,
            config_tensor_in_dram);

        MemoryConfig config_mem_config = [&]() -> MemoryConfig {
            if (config_tensor_in_dram) {
                return DRAM_MEMORY_CONFIG;
            }
            const std::array<uint32_t, 2> shard_shape{1, static_cast<uint32_t>(config_tensor.logical_shape()[-1])};
            const tt::tt_metal::ShardOrientation config_orient = input.shard_spec().value().orientation;
            const tt::tt_metal::ShardSpec config_shard_spec(
                input.shard_spec().value().grid, shard_shape, config_orient);
            return MemoryConfig{TensorMemoryLayout::HEIGHT_SHARDED, BufferType::L1_SMALL, config_shard_spec};
        }();
        config_tensor_owned =
            tt::tt_metal::enqueue_write_tensor(cq, config_tensor.host_tensor(), *mesh_device, config_mem_config);
        config_tensor_spec = config_tensor_owned->tensor_spec();
        config_buffer_page_size = config_tensor_owned->mesh_buffer().page_size();
    }

    // Park the op-owned tensors. The TensorArgument/borrowed_from references below point
    // at these parked (owning) MeshTensors, matched by identity.
    std::vector<MeshTensor> op_owned_tensors;
    op_owned_tensors.reserve(2);
    op_owned_tensors.push_back(std::move(reader_indices_owned));
    const MeshTensor& reader_indices_mt = op_owned_tensors.back();
    const MeshTensor* config_mt = nullptr;
    if (config_tensor_owned.has_value()) {
        op_owned_tensors.push_back(std::move(*config_tensor_owned));
        config_mt = &op_owned_tensors.back();
    }

    // -----------------------------------------------------------------------
    // Sizing (mirrors the legacy factory verbatim).
    // -----------------------------------------------------------------------
    const bool is_block_sharded = setup.is_block_sharded;
    const bool is_width_sharded = input.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED;
    const auto all_cores = input.shard_spec().value().grid;
    const uint32_t ncores = all_cores.num_cores();
    const uint32_t num_cores_x = input.memory_config().shard_spec()->grid.bounding_box().grid_size().x;
    const uint32_t rectangular_x = is_block_sharded ? all_cores.ranges()[0].end_coord.x + 1 : num_cores_x;
    const uint32_t max_out_nhw_per_core = output_tensors[0].shard_spec()->shape[0];
    const uint32_t max_in_nhw_per_core = input.shard_spec()->shape[0];
    TT_FATAL(
        max_in_nhw_per_core <= std::numeric_limits<uint16_t>::max(),
        "Input nhw per core {} exceeds uint16_t limit",
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

    const uint32_t eff_kernel_h = ((kernel_h - 1) * dilation_h) + 1;
    const uint32_t eff_kernel_w = ((kernel_w - 1) * dilation_w) + 1;
    const uint32_t in_h_padded = in_h + pad_h + setup.ceil_pad_h;
    const uint32_t in_w_padded = in_w + pad_w + setup.ceil_pad_w;

    auto output_shard_shape = output_tensors[0].shard_spec().value().shape;
    std::optional<uint32_t> reader_indices_actual_page_size;
    if (config_tensor_in_dram) {
        reader_indices_actual_page_size = reader_indices_page_size;
    }
    PoolCBSizes cb_sizes = calculate_pool_cb_sizes(
        params,
        one_scalar_per_core,
        return_indices,
        output_layout,
        output_tensors[0].dtype(),
        {output_shard_shape[0], output_shard_shape[1]},
        config_tensor_in_dram,
        reader_indices_actual_page_size);

    const auto& input_shape = input.padded_shape();
    const uint32_t shard_width = input.shard_spec()->shape[1];
    const uint32_t in_c_per_shard_ceil = in_c % shard_width != 0 && num_shards_c > 1
                                             ? (in_c - (in_c % shard_width)) / (num_shards_c - 1)
                                             : in_c / num_shards_c;
    const uint32_t in_nbytes_c = in_c_per_shard_ceil * params.nbytes;
    const uint32_t shard_width_bytes = input_shape[3] / num_shards_c * params.nbytes;
    TT_FATAL(
        input_shape[3] % num_shards_c == 0,
        "Input channels {} should be divisible by num shards {}",
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

    uint32_t in_nblocks_c = 1;
    if (return_indices || params.is_wide_reduction) {
        in_nblocks_c =
            static_cast<uint32_t>(std::ceil(static_cast<float>(params.in_ntiles_c) / params.MAX_TILES_PER_REDUCTION));
    }

    const bool is_output_tiled = output_layout == Layout::TILE;
    const bool is_output_block_format = is_block_float(output_tensors[0].dtype());
    const bool zero_pages = is_output_tiled && is_output_block_format;
    const bool indexes_32_bit = return_indices && params.index_format == tt::DataFormat::UInt32;

    // MPWI index increments (host computation, unchanged).
    uint32_t right_inc = 0, down_left_wrap_inc = 0, up_left_wrap_inc = 0;
    uint32_t intra_kernel_right_inc = 0, intra_kernel_down_left_wrap_inc = 0;
    if (return_indices) {
        right_inc = stride_w;
        down_left_wrap_inc = in_w * stride_h + (1 - out_w) * stride_w;
        up_left_wrap_inc = (1 - out_h) * stride_h * in_w + (1 - out_w) * stride_w;
        if (params.is_large_kernel) {
            uint32_t sticks_per_chunk =
                kernel_w <= params.max_rows_for_reduction ? kernel_w : params.max_rows_for_reduction;
            uint32_t w_chunks =
                kernel_w % sticks_per_chunk == 0 ? kernel_w / sticks_per_chunk : (kernel_w / sticks_per_chunk) + 1;
            uint32_t last_w_chunk_w_offset = (w_chunks - 1) * sticks_per_chunk * dilation_w;
            uint32_t last_top_left_kernel_index = ((kernel_h - 1) * dilation_h * in_w) + last_w_chunk_w_offset;
            uint32_t index_correction = last_top_left_kernel_index;  // first_top_left_kernel_index == 0
            right_inc -= index_correction;
            down_left_wrap_inc -= index_correction;
            up_left_wrap_inc -= index_correction;
            intra_kernel_right_inc = dilation_w * sticks_per_chunk;
            intra_kernel_down_left_wrap_inc = dilation_h * in_w - last_w_chunk_w_offset;
        }
    }

    // -----------------------------------------------------------------------
    // Tensor parameters.
    // -----------------------------------------------------------------------
    ProgramSpec spec;
    spec.name = "pool2d_multi_core";
    spec.tensor_parameters = {
        TensorParameter{.unique_id = INPUT_TENSOR, .spec = input.tensor_spec()},
        TensorParameter{.unique_id = OUTPUT_TENSOR, .spec = output_tensors[0].tensor_spec()},
        TensorParameter{.unique_id = READER_INDICES_TENSOR, .spec = reader_indices_spec},
    };
    if (return_indices) {
        spec.tensor_parameters.push_back(
            TensorParameter{.unique_id = OUTPUT_IDX_TENSOR, .spec = output_tensors[1].tensor_spec()});
    }
    if (config_tensor_spec.has_value()) {
        spec.tensor_parameters.push_back(TensorParameter{.unique_id = CONFIG_TENSOR, .spec = *config_tensor_spec});
    }

    // -----------------------------------------------------------------------
    // Dataflow buffers.
    // -----------------------------------------------------------------------
    const auto scalar_face = FaceGeometry{.face_r_dim = 1, .num_faces = 4};
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

    constexpr uint32_t pack_untilize_face_r_dim = 1;
    const bool last_tile_is_partial = in_c % tt::constants::TILE_WIDTH != 0;
    const bool single_partial_fits_in_face = last_tile_is_partial && in_c <= tt::constants::FACE_WIDTH;
    const uint32_t pack_untilize_num_faces = single_partial_fits_in_face ? 1u : 2u;
    const auto pack_untilize_face =
        FaceGeometry{.face_r_dim = pack_untilize_face_r_dim, .num_faces = pack_untilize_num_faces};
    const std::optional<tt::tt_metal::Tile> pack_untilize_tile =
        single_partial_fits_in_face ? std::optional{tt::tt_metal::Tile({1, tt::constants::FACE_WIDTH}, false)}
                                    : std::nullopt;

    std::vector<DataflowBufferSpec> dfbs;

    // scalar CB(s)
    // The pool2d split-reader compute kernel references dfb::in_cb_1 and dfb::in_scalar_cb_1
    // (under #ifdef SPLIT_READER) regardless of dtype, so both second-stream DFBs must exist
    // whenever pool2d uses a split reader. (mpwi has a single input/scalar stream — reader1 is
    // the writer face.)
    const bool has_second_input_cb = cb_sizes.has_split_reader && !return_indices;

    dfbs.push_back(local_dfb(
        DFB_IN_SCALAR_0, cb_sizes.scalar_cb_pagesize, cb_sizes.scalar_cb_npages, params.data_format, scalar_face));
    if (has_second_input_cb) {
        dfbs.push_back(local_dfb(
            DFB_IN_SCALAR_1, cb_sizes.scalar_cb_pagesize, cb_sizes.scalar_cb_npages, params.data_format, scalar_face));
    }
    // clear value CB
    dfbs.push_back(local_dfb(DFB_CLEAR_VALUE, cb_sizes.clear_value_cb_size, 1, params.data_format));
    // raw input shard CB (borrowed input)
    dfbs.push_back(
        borrowed_dfb(DFB_IN_SHARD, in_nbytes_c, input.shard_spec().value().shape[0], params.data_format, INPUT_TENSOR));
    // reader indices CB (borrowed L1 config tensor, or local scratch for DRAM path)
    const uint32_t in_reader_indices_cb_pagesize = tt::round_up(top_left_indices[0].size(), 4);
    if (config_tensor_in_dram) {
        dfbs.push_back(local_dfb(DFB_READER_INDICES, reader_indices_page_size, 1, tt::DataFormat::UInt16));
    } else {
        dfbs.push_back(borrowed_dfb(
            DFB_READER_INDICES, in_reader_indices_cb_pagesize, 1, tt::DataFormat::UInt16, READER_INDICES_TENSOR));
    }
    // input CB(s). The second input stream (in_cb_1) only exists for the pool2d split-reader
    // (mpwi has a single input stream — reader1 is the writer face, not a second producer).
    dfbs.push_back(
        local_dfb(DFB_IN_0, cb_sizes.in_cb_pagesize, cb_sizes.in_cb_npages, params.data_format, input_face_geometry));
    if (has_second_input_cb) {
        dfbs.push_back(local_dfb(
            DFB_IN_1, cb_sizes.in_cb_pagesize, cb_sizes.in_cb_npages, params.data_format, input_face_geometry));
    }

    // MPWI scratch / index CBs
    if (return_indices) {
        const uint32_t tile_elems = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
        const uint32_t idx_pagesize = params.index_nbytes * tile_elems;
        const uint32_t data_pagesize = params.nbytes * tile_elems;
        dfbs.push_back(local_dfb(DFB_IN_IDX, idx_pagesize, 1, params.index_format));
        dfbs.push_back(local_dfb(DFB_PACK_TMP, data_pagesize, 1, params.data_format));
        dfbs.push_back(local_dfb(DFB_PACK_IDX_TMP, idx_pagesize, 1, params.index_format));
        dfbs.push_back(local_dfb(DFB_RIGHT_INC, idx_pagesize, 1, params.index_format));
        dfbs.push_back(local_dfb(DFB_DOWN_LEFT, idx_pagesize, 1, params.index_format));
        dfbs.push_back(local_dfb(DFB_UP_LEFT, idx_pagesize, 1, params.index_format));
        dfbs.push_back(local_dfb(DFB_COMPUTE_TMP_IDX, idx_pagesize, 1, params.index_format));
        // intra-kernel increment CBs are referenced unconditionally at file scope by
        // both mpwi kernels (FIFO ops are gated by is_large_kernel, but the dfb:: token
        // must always exist), so always declare and bind them.
        dfbs.push_back(local_dfb(DFB_INTRA_RIGHT, idx_pagesize, 1, params.index_format));
        dfbs.push_back(local_dfb(DFB_INTRA_DOWN_LEFT, idx_pagesize, 1, params.index_format));
    }

    // pre_tilize / fast_tilize: two aliased DFBs over one L1 region (TILED output).
    if (cb_sizes.has_pre_tilize) {
        const uint32_t pre_tilize_total = cb_sizes.pre_tilize_cb_pagesize * cb_sizes.pre_tilize_cb_npages;
        const uint32_t fast_tilize_page_size = tt::tile_size(params.data_format);
        TT_FATAL(
            pre_tilize_total % fast_tilize_page_size == 0,
            "pre_tilize_cb total {} must be a multiple of fast_tilize tile {}",
            pre_tilize_total,
            fast_tilize_page_size);
        DataflowBufferSpec pre = local_dfb(
            DFB_PRE_TILIZE,
            cb_sizes.pre_tilize_cb_pagesize,
            cb_sizes.pre_tilize_cb_npages,
            params.data_format,
            pack_untilize_face,
            pack_untilize_tile);
        pre.advanced_options.alias_with = {DFB_FAST_TILIZE};
        DataflowBufferSpec fast = local_dfb(
            DFB_FAST_TILIZE, fast_tilize_page_size, pre_tilize_total / fast_tilize_page_size, params.data_format);
        fast.advanced_options.alias_with = {DFB_PRE_TILIZE};
        dfbs.push_back(std::move(pre));
        dfbs.push_back(std::move(fast));
    }

    // output CB (borrowed output) + optional output index CB (borrowed output_idx)
    dfbs.push_back(borrowed_dfb(
        DFB_OUT,
        cb_sizes.out_cb_pagesize,
        cb_sizes.out_cb_npages,
        params.output_data_format,
        OUTPUT_TENSOR,
        is_output_tiled ? std::nullopt : std::optional{pack_untilize_face},
        is_output_tiled ? std::nullopt : pack_untilize_tile));
    if (cb_sizes.has_out_idx) {
        TT_FATAL(output_tensors.size() == 2, "return_indices requires two outputs, got {}", output_tensors.size());
        dfbs.push_back(borrowed_dfb(
            DFB_OUT_IDX,
            cb_sizes.out_idx_cb_pagesize,
            cb_sizes.out_idx_cb_npages,
            params.index_format,
            OUTPUT_IDX_TENSOR));
    }

    // scalar config CB (avg pool, !one_scalar_per_core)
    if (!one_scalar_per_core) {
        TT_FATAL(config_mt != nullptr, "config tensor must be present when !one_scalar_per_core");
        constexpr tt::DataFormat config_df = tt::DataFormat::RawUInt32;
        const uint32_t max_config_tensor_size = max_out_nhw_per_core * 3 * sizeof(uint16_t);
        if (config_tensor_in_dram) {
            dfbs.push_back(local_dfb(DFB_CONFIG, max_config_tensor_size, 1, config_df));
        } else {
            dfbs.push_back(borrowed_dfb(DFB_CONFIG, config_buffer_page_size, 1, config_df, CONFIG_TENSOR));
        }
    }
    spec.dataflow_buffers = std::move(dfbs);

    // -----------------------------------------------------------------------
    // Compile-time args (named; values lifted verbatim from the legacy layout).
    // -----------------------------------------------------------------------
    KernelSpec::CompileTimeArgs reader_cta = {
        {"reader_nindices", max_out_nhw_per_core},
        {"kernel_h", kernel_h},
        {"kernel_w", kernel_w},
        {"pad_w", pad_w},
        {"in_nbytes_leftover", in_nbytes_leftover},
        {"in_w", in_w},
        {"in_c", in_c_per_shard_ceil},
        {"split_reader", params.split_reader},
        {"reader_id", 0u},
        {"bf16_scalar", bf16_scalar},
        {"bf16_init_value", bf16_init_value},
        {"in_nblocks_c", in_nblocks_c},
        {"in_cb_sz", cb_sizes.in_cb_raw_size},
        {"max_sticks_for_reduction", params.max_rows_for_reduction},
        {"ceil_pad_w", setup.ceil_pad_w},
        {"pool_type_is_avg", static_cast<uint32_t>(params.is_avg_pool)},
        {"one_scalar_per_core", static_cast<uint32_t>(one_scalar_per_core)},
        {"in_nbytes_c", in_nbytes_c},
        {"shard_width_bytes", shard_width_bytes},
        {"multi_buffering_factor", params.multi_buffering_factor},
        {"stride_w", stride_w},
        {"dilation_h", dilation_h},
        {"dilation_w", dilation_w},
        {"zero_pages", static_cast<uint32_t>(zero_pages)},
        {"config_in_dram", static_cast<uint32_t>(config_tensor_in_dram)},
        {"config_page_size", one_scalar_per_core ? 0u : config_buffer_page_size},
        {"reader_page_size", reader_indices_page_size},
    };
    if (return_indices) {
        reader_cta["pad_t"] = setup.pad_t;
        reader_cta["pad_l"] = setup.pad_l;
        reader_cta["right_inc"] = right_inc;
        reader_cta["down_left_wrap_inc"] = down_left_wrap_inc;
        reader_cta["up_left_wrap_inc"] = up_left_wrap_inc;
        reader_cta["intra_kernel_right_inc"] = intra_kernel_right_inc;
        reader_cta["intra_kernel_down_left_wrap_inc"] = intra_kernel_down_left_wrap_inc;
        reader_cta["indexes_32_bit"] = static_cast<uint32_t>(indexes_32_bit);
    }

    // -----------------------------------------------------------------------
    // Reader kernel(s).
    // -----------------------------------------------------------------------
    const std::filesystem::path reader_path{return_indices ? READER_MPWI_PATH : READER_POOL_PATH};

    // Endpoint roles must satisfy the per-node DFB invariant (exactly one producer and one
    // consumer per node). reader0 and reader1 share nodes, so a DFB cannot take the same role
    // on both. For mpwi the readers are asymmetric (reader0 = input/index producer, reader1 =
    // the writer face); for pool2d they are symmetric input producers.
    auto make_reader_bindings = [&](bool is_reader1) {
        Group<DFBBinding> b;
        // Shard input (fake CB, base-pointer read by both readers): reader0=P / reader1=C when
        // split, self-loop on reader0 otherwise.
        if (params.split_reader) {
            b.push_back(DFBBinding{
                .dfb_spec_name = DFB_IN_SHARD,
                .accessor_name = "in_shard_cb",
                .endpoint_type = is_reader1 ? DFBEndpointType::CONSUMER : DFBEndpointType::PRODUCER});
        } else {
            b.push_back(DFBBinding{
                .dfb_spec_name = DFB_IN_SHARD,
                .accessor_name = "in_shard_cb",
                .endpoint_type = DFBEndpointType::PRODUCER});
            b.push_back(DFBBinding{
                .dfb_spec_name = DFB_IN_SHARD,
                .accessor_name = "in_shard_cb",
                .endpoint_type = DFBEndpointType::CONSUMER});
        }
        // Reader-indices CB: reader0=P / reader1=C when split (DRAM push -> wait), else self-loop.
        if (params.split_reader) {
            b.push_back(DFBBinding{
                .dfb_spec_name = DFB_READER_INDICES,
                .accessor_name = "reader_indices_cb",
                .endpoint_type = is_reader1 ? DFBEndpointType::CONSUMER : DFBEndpointType::PRODUCER});
        } else {
            b.push_back(DFBBinding{
                .dfb_spec_name = DFB_READER_INDICES,
                .accessor_name = "reader_indices_cb",
                .endpoint_type = DFBEndpointType::PRODUCER});
            b.push_back(DFBBinding{
                .dfb_spec_name = DFB_READER_INDICES,
                .accessor_name = "reader_indices_cb",
                .endpoint_type = DFBEndpointType::CONSUMER});
        }
        if (!one_scalar_per_core) {
            if (params.split_reader) {
                b.push_back(DFBBinding{
                    .dfb_spec_name = DFB_CONFIG,
                    .accessor_name = "config_cb",
                    .endpoint_type = is_reader1 ? DFBEndpointType::CONSUMER : DFBEndpointType::PRODUCER});
            } else {
                b.push_back(DFBBinding{
                    .dfb_spec_name = DFB_CONFIG,
                    .accessor_name = "config_cb",
                    .endpoint_type = DFBEndpointType::PRODUCER});
                b.push_back(DFBBinding{
                    .dfb_spec_name = DFB_CONFIG,
                    .accessor_name = "config_cb",
                    .endpoint_type = DFBEndpointType::CONSUMER});
            }
        }

        if (return_indices) {
            // mpwi: asymmetric. reader0 produces input/index/inc CBs (consumed by compute) and
            // clear_value (consumed by compute); reader1 is the writer face.
            if (!is_reader1) {
                b.push_back(DFBBinding{
                    .dfb_spec_name = DFB_IN_0, .accessor_name = "in_cb", .endpoint_type = DFBEndpointType::PRODUCER});
                b.push_back(DFBBinding{
                    .dfb_spec_name = DFB_IN_SCALAR_0,
                    .accessor_name = "in_scalar_cb",
                    .endpoint_type = DFBEndpointType::PRODUCER});
                b.push_back(DFBBinding{
                    .dfb_spec_name = DFB_CLEAR_VALUE,
                    .accessor_name = "clear_value_cb",
                    .endpoint_type = DFBEndpointType::PRODUCER});
                b.push_back(DFBBinding{
                    .dfb_spec_name = DFB_IN_IDX,
                    .accessor_name = "in_idx_cb",
                    .endpoint_type = DFBEndpointType::PRODUCER});
                b.push_back(DFBBinding{
                    .dfb_spec_name = DFB_RIGHT_INC,
                    .accessor_name = "right_inc_cb",
                    .endpoint_type = DFBEndpointType::PRODUCER});
                b.push_back(DFBBinding{
                    .dfb_spec_name = DFB_DOWN_LEFT,
                    .accessor_name = "down_left_wrap_inc_cb",
                    .endpoint_type = DFBEndpointType::PRODUCER});
                b.push_back(DFBBinding{
                    .dfb_spec_name = DFB_UP_LEFT,
                    .accessor_name = "up_left_wrap_inc_cb",
                    .endpoint_type = DFBEndpointType::PRODUCER});
                b.push_back(DFBBinding{
                    .dfb_spec_name = DFB_INTRA_RIGHT,
                    .accessor_name = "intra_kernel_right_inc_cb",
                    .endpoint_type = DFBEndpointType::PRODUCER});
                b.push_back(DFBBinding{
                    .dfb_spec_name = DFB_INTRA_DOWN_LEFT,
                    .accessor_name = "intra_kernel_down_left_wrap_inc_cb",
                    .endpoint_type = DFBEndpointType::PRODUCER});
            } else {
                // reader1 (writer face): consumes the compute-produced pack tmps, and is the
                // producer-only writer into the borrowed output / output-index (self-loop).
                b.push_back(DFBBinding{
                    .dfb_spec_name = DFB_PACK_TMP,
                    .accessor_name = "pack_tmp_cb",
                    .endpoint_type = DFBEndpointType::CONSUMER});
                b.push_back(DFBBinding{
                    .dfb_spec_name = DFB_PACK_IDX_TMP,
                    .accessor_name = "pack_idx_tmp_cb",
                    .endpoint_type = DFBEndpointType::CONSUMER});
                b.push_back(DFBBinding{
                    .dfb_spec_name = DFB_OUT, .accessor_name = "out_cb", .endpoint_type = DFBEndpointType::PRODUCER});
                b.push_back(DFBBinding{
                    .dfb_spec_name = DFB_OUT, .accessor_name = "out_cb", .endpoint_type = DFBEndpointType::CONSUMER});
                b.push_back(DFBBinding{
                    .dfb_spec_name = DFB_OUT_IDX,
                    .accessor_name = "out_idx_cb",
                    .endpoint_type = DFBEndpointType::PRODUCER});
                b.push_back(DFBBinding{
                    .dfb_spec_name = DFB_OUT_IDX,
                    .accessor_name = "out_idx_cb",
                    .endpoint_type = DFBEndpointType::CONSUMER});
            }
        } else {
            // pool2d: each reader produces its own input + scalar stream (compute consumes both).
            b.push_back(DFBBinding{
                .dfb_spec_name = is_reader1 ? DFB_IN_1 : DFB_IN_0,
                .accessor_name = "in_cb",
                .endpoint_type = DFBEndpointType::PRODUCER});
            b.push_back(DFBBinding{
                .dfb_spec_name = is_reader1 ? DFB_IN_SCALAR_1 : DFB_IN_SCALAR_0,
                .accessor_name = "in_scalar_cb",
                .endpoint_type = DFBEndpointType::PRODUCER});
            // clear value: reader0=P / reader1=C when split, self-loop on reader0 otherwise.
            if (params.split_reader) {
                b.push_back(DFBBinding{
                    .dfb_spec_name = DFB_CLEAR_VALUE,
                    .accessor_name = "clear_value_cb",
                    .endpoint_type = is_reader1 ? DFBEndpointType::CONSUMER : DFBEndpointType::PRODUCER});
            } else {
                b.push_back(DFBBinding{
                    .dfb_spec_name = DFB_CLEAR_VALUE,
                    .accessor_name = "clear_value_cb",
                    .endpoint_type = DFBEndpointType::PRODUCER});
                b.push_back(DFBBinding{
                    .dfb_spec_name = DFB_CLEAR_VALUE,
                    .accessor_name = "clear_value_cb",
                    .endpoint_type = DFBEndpointType::CONSUMER});
            }
        }
        return b;
    };

    // Both reader binaries reference tensor::reader_indices / tensor::config at file scope
    // (the DRAM-path TensorAccessor sits in an `if constexpr (reader_id==0)` branch, which is
    // still name-looked-up), so both bind them. Tensor bindings carry no 1P/1C constraint.
    auto make_reader_tensor_bindings = [&](bool /*is_reader1*/) {
        Group<TensorBinding> tb;
        tb.push_back(TensorBinding{.tensor_parameter_name = READER_INDICES_TENSOR, .accessor_name = "reader_indices"});
        if (!one_scalar_per_core) {
            tb.push_back(TensorBinding{.tensor_parameter_name = CONFIG_TENSOR, .accessor_name = "config"});
        }
        return tb;
    };

    // Per-kernel RTA schemas: readers consume core_nhw_index (+ start_row/start_col for
    // mpwi); compute consumes out_nhw_this_core (+ start_row/start_col for mpwi).
    Group<std::string> reader_rta_names = {"core_nhw_index"};
    Group<std::string> compute_rta_names = {"out_nhw_this_core"};
    if (return_indices) {
        reader_rta_names.push_back("start_row");
        reader_rta_names.push_back("start_col");
        compute_rta_names.push_back("start_row");
        compute_rta_names.push_back("start_col");
    }

    KernelSpec::CompileTimeArgs reader1_cta = reader_cta;
    reader1_cta["reader_id"] = 1u;

    // reader_mpwi.cpp #ifdef-gates its per-role token references on READER_ID so each
    // reader binary references only the DFB tokens it actually drives. reader_pool_2d.cpp
    // #ifdef-gates dfb::config_cb / tensor::config on HAS_CONFIG (avg-pool, !one_scalar).
    KernelSpec::CompilerOptions::Defines reader0_defines{{"READER_ID", "0"}};
    KernelSpec::CompilerOptions::Defines reader1_defines{{"READER_ID", "1"}};
    if (!one_scalar_per_core) {
        reader0_defines.insert({"HAS_CONFIG", "1"});
        reader1_defines.insert({"HAS_CONFIG", "1"});
    }

    KernelSpec reader0{
        .unique_id = READER0_KERNEL,
        .source = reader_path,
        .compiler_options = {.defines = reader0_defines},
        .dfb_bindings = make_reader_bindings(false),
        .tensor_bindings = make_reader_tensor_bindings(false),
        .compile_time_args = reader_cta,
        .runtime_arg_schema = {.runtime_arg_names = reader_rta_names},
        .hw_config = ttnn::create_reader_datamovement_config(mesh_device->arch()),
    };

    std::optional<KernelSpec> reader1;
    if (params.split_reader) {
        reader1 = KernelSpec{
            .unique_id = READER1_KERNEL,
            .source = reader_path,
            .compiler_options = {.defines = reader1_defines},
            .dfb_bindings = make_reader_bindings(true),
            .tensor_bindings = make_reader_tensor_bindings(true),
            .compile_time_args = reader1_cta,
            .runtime_arg_schema = {.runtime_arg_names = reader_rta_names},
            .hw_config = ttnn::create_writer_datamovement_config(mesh_device->arch()),
        };
    }

    // -----------------------------------------------------------------------
    // Compute kernel.
    // -----------------------------------------------------------------------
    KernelSpec::CompileTimeArgs compute_cta = {
        {"in_ntiles_c", params.in_ntiles_c},
        {"window_size_hw", kernel_h * kernel_w},
        {"split_reader", params.split_reader},
        {"max_out_sticks_per_core", 0u},
        {"in_c", in_c_per_shard_ceil},
        {"in_nblocks_c", in_nblocks_c},
        {"max_sticks_for_reduction", params.max_rows_for_reduction},
        {"one_scalar_per_core", static_cast<uint32_t>(one_scalar_per_core)},
        {"is_output_tiled", static_cast<uint32_t>(is_output_tiled)},
        {"is_output_block_format", static_cast<uint32_t>(is_output_block_format)},
        {"force_max_tiles_per_reduction_4", 0u},
    };
    if (return_indices) {
        compute_cta["stride_h"] = stride_h;
        compute_cta["stride_w"] = stride_w;
        compute_cta["in_h_padded"] = in_h_padded;
        compute_cta["in_w_padded"] = in_w_padded;
        compute_cta["eff_kernel_h"] = eff_kernel_h;
        compute_cta["eff_kernel_w"] = eff_kernel_w;
        compute_cta["pad_l"] = setup.pad_l;
        compute_cta["kernel_h"] = kernel_h;
        compute_cta["kernel_w"] = kernel_w;
        compute_cta["indexes_32_bit"] = static_cast<uint32_t>(indexes_32_bit);
    }

    Group<DFBBinding> compute_bindings;
    compute_bindings.push_back(
        DFBBinding{.dfb_spec_name = DFB_IN_0, .accessor_name = "in_cb_0", .endpoint_type = DFBEndpointType::CONSUMER});
    compute_bindings.push_back(DFBBinding{
        .dfb_spec_name = DFB_IN_SCALAR_0,
        .accessor_name = "in_scalar_cb_0",
        .endpoint_type = DFBEndpointType::CONSUMER});
    // pool2d split-reader compute consumes the second input + scalar streams (both DFBs exist
    // whenever has_second_input_cb; the kernel references them under #ifdef SPLIT_READER).
    if (has_second_input_cb) {
        compute_bindings.push_back(DFBBinding{
            .dfb_spec_name = DFB_IN_1, .accessor_name = "in_cb_1", .endpoint_type = DFBEndpointType::CONSUMER});
        compute_bindings.push_back(DFBBinding{
            .dfb_spec_name = DFB_IN_SCALAR_1,
            .accessor_name = "in_scalar_cb_1",
            .endpoint_type = DFBEndpointType::CONSUMER});
    }
    if (!return_indices) {
        // pool2d: compute produces output directly into the borrowed output DFB.  The
        // result stays resident (the DFB is borrowed from OUTPUT_TENSOR and sized to the
        // full output shard, so the producer never wraps), so there is no real consumer.
        // Self-loop the compute as producer+consumer to satisfy the SPSC completeness
        // check (mirrors the mpwi writer-face self-loop on DFB_OUT); no kernel-side
        // pop is needed since the data is the final resident output.
        compute_bindings.push_back(DFBBinding{
            .dfb_spec_name = DFB_OUT, .accessor_name = "out_cb", .endpoint_type = DFBEndpointType::PRODUCER});
        compute_bindings.push_back(DFBBinding{
            .dfb_spec_name = DFB_OUT, .accessor_name = "out_cb", .endpoint_type = DFBEndpointType::CONSUMER});
        if (cb_sizes.has_pre_tilize) {
            compute_bindings.push_back(DFBBinding{
                .dfb_spec_name = DFB_PRE_TILIZE,
                .accessor_name = "pre_tilize_cb",
                .endpoint_type = DFBEndpointType::PRODUCER});
            compute_bindings.push_back(DFBBinding{
                .dfb_spec_name = DFB_PRE_TILIZE,
                .accessor_name = "pre_tilize_cb",
                .endpoint_type = DFBEndpointType::CONSUMER});
            compute_bindings.push_back(DFBBinding{
                .dfb_spec_name = DFB_FAST_TILIZE,
                .accessor_name = "fast_tilize_cb",
                .endpoint_type = DFBEndpointType::PRODUCER});
            compute_bindings.push_back(DFBBinding{
                .dfb_spec_name = DFB_FAST_TILIZE,
                .accessor_name = "fast_tilize_cb",
                .endpoint_type = DFBEndpointType::CONSUMER});
        }
    } else {
        // mpwi: compute consumes index/inc CBs (reader-produced), produces pack tmps + self-loops scratch idx.
        compute_bindings.push_back(DFBBinding{
            .dfb_spec_name = DFB_CLEAR_VALUE,
            .accessor_name = "clear_value_cb",
            .endpoint_type = DFBEndpointType::CONSUMER});
        compute_bindings.push_back(DFBBinding{
            .dfb_spec_name = DFB_IN_IDX, .accessor_name = "in_idx_cb", .endpoint_type = DFBEndpointType::CONSUMER});
        compute_bindings.push_back(DFBBinding{
            .dfb_spec_name = DFB_RIGHT_INC,
            .accessor_name = "right_inc_cb",
            .endpoint_type = DFBEndpointType::CONSUMER});
        compute_bindings.push_back(DFBBinding{
            .dfb_spec_name = DFB_DOWN_LEFT,
            .accessor_name = "down_left_wrap_inc_cb",
            .endpoint_type = DFBEndpointType::CONSUMER});
        compute_bindings.push_back(DFBBinding{
            .dfb_spec_name = DFB_UP_LEFT,
            .accessor_name = "up_left_wrap_inc_cb",
            .endpoint_type = DFBEndpointType::CONSUMER});
        compute_bindings.push_back(DFBBinding{
            .dfb_spec_name = DFB_INTRA_RIGHT,
            .accessor_name = "intra_kernel_right_inc_cb",
            .endpoint_type = DFBEndpointType::CONSUMER});
        compute_bindings.push_back(DFBBinding{
            .dfb_spec_name = DFB_INTRA_DOWN_LEFT,
            .accessor_name = "intra_kernel_down_left_wrap_inc_cb",
            .endpoint_type = DFBEndpointType::CONSUMER});
        compute_bindings.push_back(DFBBinding{
            .dfb_spec_name = DFB_PACK_TMP, .accessor_name = "pack_tmp_cb", .endpoint_type = DFBEndpointType::PRODUCER});
        compute_bindings.push_back(DFBBinding{
            .dfb_spec_name = DFB_PACK_IDX_TMP,
            .accessor_name = "pack_idx_tmp_cb",
            .endpoint_type = DFBEndpointType::PRODUCER});
        // compute_tmp_idx: self-loop accumulator on compute.
        compute_bindings.push_back(DFBBinding{
            .dfb_spec_name = DFB_COMPUTE_TMP_IDX,
            .accessor_name = "compute_tmp_idx_cb",
            .endpoint_type = DFBEndpointType::PRODUCER});
        compute_bindings.push_back(DFBBinding{
            .dfb_spec_name = DFB_COMPUTE_TMP_IDX,
            .accessor_name = "compute_tmp_idx_cb",
            .endpoint_type = DFBEndpointType::CONSUMER});
    }

    // Compute defines (REDUCE_OP / REDUCE_DIM, etc.) plus the conditional-binding gates the
    // compute kernel uses to preprocessor-guard optional DFB token references.
    const auto pool_defines_map = get_defines(pool_type);
    KernelSpec::CompilerOptions::Defines compute_defines;
    for (const auto& [k, v] : pool_defines_map) {
        compute_defines.insert({k, v});
    }
    if (cb_sizes.has_split_reader) {
        compute_defines.insert({"SPLIT_READER", "1"});
    }
    if (is_output_tiled) {
        compute_defines.insert({"OUTPUT_TILED", "1"});
    }

    auto device_arch = input.device()->arch();
    auto device_compute_kernel_config = init_device_compute_kernel_config(
        device_arch,
        op_attr.compute_kernel_config_,
        tt::tt_metal::MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/(params.is_avg_pool && params.is_large_kernel) || indexes_32_bit,
        /*default_l1_acc=*/false,
        /*default_dst_full_sync_en=*/(params.is_large_kernel && return_indices) || indexes_32_bit);

    ComputeHardwareConfig compute_hw = ttnn::to_compute_hardware_config(
        device_arch,
        ttnn::ComputeKernelConfig{
            .math_fidelity = get_math_fidelity(device_compute_kernel_config),
            .math_approx_mode = false,
            .fp32_dest_acc_en = get_fp32_dest_acc_en(device_compute_kernel_config),
            .dst_full_sync_en = get_dst_full_sync_en(device_compute_kernel_config)});

    KernelSpec compute{
        .unique_id = COMPUTE_KERNEL,
        .source = std::filesystem::path{return_indices ? COMPUTE_MPWI_PATH : COMPUTE_POOL_PATH},
        .dfb_bindings = std::move(compute_bindings),
        .compile_time_args = compute_cta,
        .runtime_arg_schema = {.runtime_arg_names = compute_rta_names},
        .hw_config = compute_hw,
    };
    compute.compiler_options.defines = std::move(compute_defines);

    spec.kernels = {reader0};
    if (reader1.has_value()) {
        spec.kernels.push_back(*reader1);
    }
    spec.kernels.push_back(compute);

    Group<KernelSpecName> wu_kernels = {READER0_KERNEL};
    if (reader1.has_value()) {
        wu_kernels.push_back(READER1_KERNEL);
    }
    wu_kernels.push_back(COMPUTE_KERNEL);
    spec.work_units = {WorkUnitSpec{.name = "pool2d_wu", .kernels = wu_kernels, .target_nodes = all_cores}};

    // -----------------------------------------------------------------------
    // Run args: per-core RTAs (mirrors the legacy per-core loop).
    // -----------------------------------------------------------------------
    std::vector<uint32_t> core_starting_indices;
    if (return_indices) {
        const TensorMemoryLayout shard_scheme = input.memory_config().memory_layout();
        core_starting_indices =
            generate_core_starting_indices(op_trace_metadata, shard_boundaries, shard_scheme, num_cores_x, ncores);
        TT_FATAL(core_starting_indices.size() == ncores, "core starting indices size should match number of cores");
    }

    KernelRunArgs reader0_run{.kernel = READER0_KERNEL};
    KernelRunArgs reader1_run{.kernel = READER1_KERNEL};
    KernelRunArgs compute_run{.kernel = COMPUTE_KERNEL};

    const uint32_t total_out_nhw = in_n * out_h * out_w;
    for (uint32_t core_i = 0; core_i < ncores; core_i++) {
        const uint32_t core_x_i = core_i % rectangular_x;
        const uint32_t core_y_i = core_i / rectangular_x;
        const NodeCoord node{core_x_i, core_y_i};

        uint32_t total_out_nhw_processed;
        uint32_t core_nhw_index;
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

        KernelRunArgs::RuntimeArgValues& reader0_rtas = reader0_run.runtime_arg_values;
        KernelRunArgs::RuntimeArgValues& reader1_rtas = reader1_run.runtime_arg_values;
        KernelRunArgs::RuntimeArgValues& compute_rtas = compute_run.runtime_arg_values;
        reader0_rtas["core_nhw_index"][node] = core_nhw_index;
        compute_rtas["out_nhw_this_core"][node] = out_nhw_this_core;
        if (reader1.has_value()) {
            reader1_rtas["core_nhw_index"][node] = core_nhw_index;
        }
        if (return_indices) {
            const uint32_t start_index = core_starting_indices[core_i];
            const uint32_t start_mod_batch = start_index % (in_w_padded * in_h_padded);
            const uint32_t start_row = start_mod_batch / in_w_padded;
            const uint32_t start_col = start_mod_batch % in_w_padded;
            AddRuntimeArgsForNode(
                reader0_rtas,
                node,
                {
                    {"start_row", start_row},
                    {"start_col", start_col},
                });
            AddRuntimeArgsForNode(
                compute_rtas,
                node,
                {
                    {"start_row", start_row},
                    {"start_col", start_col},
                });
            if (reader1.has_value()) {
                AddRuntimeArgsForNode(
                    reader1_rtas,
                    node,
                    {
                        {"start_row", start_row},
                        {"start_col", start_col},
                    });
            }
        }
    }

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {reader0_run};
    if (reader1.has_value()) {
        run_args.kernel_run_args.push_back(reader1_run);
    }
    run_args.kernel_run_args.push_back(compute_run);

    run_args.tensor_args = {
        {INPUT_TENSOR, TensorArgument{std::cref(input.mesh_tensor())}},
        {OUTPUT_TENSOR, TensorArgument{std::cref(output_tensors[0].mesh_tensor())}},
        {READER_INDICES_TENSOR, TensorArgument{std::cref(reader_indices_mt)}},
    };
    if (return_indices) {
        run_args.tensor_args.insert({OUTPUT_IDX_TENSOR, TensorArgument{std::cref(output_tensors[1].mesh_tensor())}});
    }
    if (config_mt != nullptr) {
        run_args.tensor_args.insert({CONFIG_TENSOR, TensorArgument{std::cref(*config_mt)}});
    }

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec), .run_params = std::move(run_args), .op_owned_tensors = std::move(op_owned_tensors)};
}

}  // namespace ttnn::operations::pool::quasar

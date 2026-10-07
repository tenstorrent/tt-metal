// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_decode_large_k_device_operation.hpp"
#include "matmul_decode_device_operation.hpp"
#include "tt-metalium/constants.hpp"
#include "tt-metalium/core_coord.hpp"
#include <tt-metalium/work_split.hpp>

#include <algorithm>
#include <string>
#include <string_view>
#include <vector>

namespace ttnn::operations::experimental::matmul_decode {

using namespace tt;
using namespace tt::tt_metal;

namespace {

// One core's place in the reduction tree. Cores are numbered in A's row-major shard order, so
// core i holds K-slice i. At level l (stride s = fan_in^l) every core whose index is a multiple of
// s * fan_in collects the partials of cores i + j*s (j = 1 .. fan_in-1); everyone else at that
// level sends its running sum to its parent and drops out. Core 0 is the root.
struct LargeKTreeNode {
    int parent = -1;
    uint32_t parent_level = 0;
    // Where this core's partial lands in the parent's receive CB, in partial-sized slots. Slots are
    // numbered level-major, the order in which the parent consumes them.
    uint32_t slot_in_parent = 0;
    std::vector<uint32_t> children_per_level;
    uint32_t num_children = 0;
};

std::vector<LargeKTreeNode> build_large_k_tree(uint32_t num_cores, uint32_t fan_in, uint32_t num_levels) {
    std::vector<LargeKTreeNode> nodes(num_cores);
    for (auto& node : nodes) {
        node.children_per_level.assign(num_levels, 0);
    }
    uint64_t stride = 1;
    for (uint32_t level = 0; level < num_levels; ++level, stride *= fan_in) {
        const uint64_t group = stride * fan_in;
        for (uint64_t i = 0; i < num_cores; i += group) {
            for (uint32_t j = 1; j < fan_in; ++j) {
                const uint64_t child = i + j * stride;
                if (child >= num_cores) {
                    break;
                }
                auto& parent = nodes[i];
                nodes[child].parent = static_cast<int>(i);
                nodes[child].parent_level = level;
                nodes[child].slot_in_parent = parent.num_children;
                parent.children_per_level[level]++;
                parent.num_children++;
            }
        }
    }
    return nodes;
}

}  // namespace

ProgramDescriptor MatmulDecodeLargeKDeviceOperation::TreeReduce::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;
    auto& output_tensor = tensor_return_value;
    auto* device = input_tensor_a.device();

    const tt::DataFormat in0_data_format = datatype_to_dataformat_converter(input_tensor_a.dtype());
    const tt::DataFormat in1_data_format = datatype_to_dataformat_converter(input_tensor_b.dtype());
    const tt::DataFormat out_data_format = datatype_to_dataformat_converter(output_tensor.dtype());

    // A's rows and every partial are consumed as 1x32 tiles: a row-major [M, W] buffer is exactly
    // M * W/32 such tiles in row-major order.
    const tt::tt_metal::Tile row_tile({1, tt::constants::TILE_WIDTH}, false);
    const tt::tt_metal::Tile& in1_tile = input_tensor_b.tensor_spec().tile();
    const uint32_t in0_tile_size = row_tile.get_tile_size(in0_data_format);
    const uint32_t in1_tile_size = in1_tile.get_tile_size(in1_data_format);
    const uint32_t out_tile_size = row_tile.get_tile_size(out_data_format);

    const auto& a_shard = input_tensor_a.memory_config().shard_spec().value();
    const CoreRangeSet& grid = a_shard.grid;
    const std::vector<CoreCoord> cores = corerange_to_cores(grid, std::nullopt, /*row_wise=*/true);
    const uint32_t num_cores = cores.size();
    const CoreCoord root = cores.front();

    const uint32_t M_tiles = operation_attributes.M;
    const uint32_t Kc_tiles = a_shard.shape[1] / tt::constants::TILE_WIDTH;
    const uint32_t N_tiles = operation_attributes.N / tt::constants::TILE_WIDTH;
    const uint32_t block_num_tiles = M_tiles * N_tiles;
    const uint32_t block_bytes = block_num_tiles * out_tile_size;

    const uint32_t fan_in = operation_attributes.reduce_fan_in;
    const uint32_t num_levels = large_k_tree_num_levels(num_cores, fan_in);
    const std::vector<LargeKTreeNode> tree = build_large_k_tree(num_cores, fan_in, num_levels);
    uint32_t max_children = 0;
    for (const auto& node : tree) {
        max_children = std::max(max_children, node.num_children);
    }

    const bool fp32_dest_acc_en = out_data_format == tt::DataFormat::Float32;
    // Half-sync DST holds 8 16-bit tiles or 4 fp32 ones.
    const uint32_t dst_num_tiles = fp32_dest_acc_en ? 4 : 8;
    const bool use_custom_mm = device->arch() == tt::ARCH::BLACKHOLE && M_tiles == 1 && !fp32_dest_acc_en &&
                               is_custom_mm_kt_dim(Kc_tiles) && is_custom_mm_ct_dim(N_tiles);
    if (!use_custom_mm) {
        std::string_view reason;
        if (device->arch() != tt::ARCH::BLACKHOLE) {
            reason = "custom_mm is Blackhole-only";
        } else if (M_tiles != 1) {
            reason = "more than one row of A is not contiguous for custom_mm";
        } else if (fp32_dest_acc_en) {
            reason = "a FLOAT32 output accumulates in fp32 DST";
        } else if (!is_custom_mm_kt_dim(Kc_tiles)) {
            reason = "the per-core K slice must contain an even number of tiles in [2, 256]";
        } else {
            reason = "N exceeds 16 tiles";
        }
        log_warning(tt::LogOp, "matmul_decode_large_k is falling back to the general block matmul LLKs: {}", reason);
    }
    // matmul_block keeps an out_subblock_h x out_subblock_w output block in DST.
    uint32_t out_subblock_h = 1;
    for (uint32_t h = std::min(M_tiles, dst_num_tiles); h >= 1; --h) {
        if (M_tiles % h == 0) {
            out_subblock_h = h;
            break;
        }
    }
    uint32_t out_subblock_w = 1;
    for (uint32_t w = N_tiles; w >= 1; --w) {
        if (N_tiles % w == 0 && out_subblock_h * w <= dst_num_tiles) {
            out_subblock_w = w;
            break;
        }
    }

    ProgramDescriptor desc;

    // Named "cb_*" compile-time args carry every index so op fusion can remap them.
    const uint32_t in0_cb_index = CBIndex::c_0;      // this core's K-slice of A
    const uint32_t in1_cb_index = CBIndex::c_1;      // this core's [Kc, N] slab of B
    const uint32_t partial_cb_index = CBIndex::c_2;  // local partial, on cores that reduce
    const uint32_t recv_cb_index = CBIndex::c_3;     // children's partials, filled by the children
    const uint32_t out_cb_index = CBIndex::c_4;      // root only: the output shard
    const uint32_t send_cb_index = CBIndex::c_5;     // non-root: the running sum sent to the parent

    std::vector<CoreCoord> non_root_cores(cores.begin() + 1, cores.end());

    desc.cbs.push_back(CBDescriptor{
        .total_size = M_tiles * Kc_tiles * in0_tile_size,
        .core_ranges = grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = in0_cb_index,
            .data_format = in0_data_format,
            .page_size = in0_tile_size,
            .tile = TileDescriptor{row_tile},
        }}},
        .buffer = input_tensor_a.buffer(),
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = Kc_tiles * N_tiles * in1_tile_size,
        .core_ranges = grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = in1_cb_index,
            .data_format = in1_data_format,
            .page_size = in1_tile_size,
            .tile = TileDescriptor{in1_tile},
        }}},
        .buffer = input_tensor_b.buffer(),
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = block_bytes,
        .core_ranges = CoreRangeSet(CoreRange(root, root)),
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = out_cb_index,
            .data_format = out_data_format,
            .page_size = out_tile_size,
            .tile = TileDescriptor{row_tile},
        }}},
        .buffer = output_tensor.buffer(),
    });
    if (max_children > 0) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = block_bytes,
            .core_ranges = grid,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = partial_cb_index,
                .data_format = out_data_format,
                .page_size = out_tile_size,
                .tile = TileDescriptor{row_tile},
            }}},
        });
        // One CB over the whole grid so it sits at the same L1 address on every core: a child
        // addresses its slot in the parent's CB from its own copy's base.
        desc.cbs.push_back(CBDescriptor{
            .total_size = max_children * block_bytes,
            .core_ranges = grid,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = recv_cb_index,
                .data_format = out_data_format,
                .page_size = out_tile_size,
                .tile = TileDescriptor{row_tile},
            }}},
        });
        desc.cbs.push_back(CBDescriptor{
            .total_size = block_bytes,
            .core_ranges = CoreRangeSet(non_root_cores),
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = send_cb_index,
                .data_format = out_data_format,
                .page_size = out_tile_size,
                .tile = TileDescriptor{row_tile},
            }}},
        });
    }

    // Semaphore l counts the level-l children that have delivered their partial to this core.
    for (uint32_t level = 0; level < num_levels; ++level) {
        desc.semaphores.push_back(SemaphoreDescriptor{.id = level, .core_ranges = grid, .initial_value = 0});
    }

    const std::string kernel_dir = "ttnn/cpp/ttnn/operations/experimental/matmul_decode/device/kernels/";

    // The reader only posts local CB credits, so it never touches the NoC.
    KernelDescriptor reader;
    reader.kernel_source = kernel_dir + "dataflow/reader_large_k_tree_reduce.cpp";
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = grid;
    reader.compile_time_args = {M_tiles * Kc_tiles, Kc_tiles * N_tiles, block_num_tiles, num_levels};
    reader.named_compile_time_args = {
        {"cb_in0", in0_cb_index},
        {"cb_in1", in1_cb_index},
        {"cb_recv", recv_cb_index},
    };
    reader.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = NOC::NOC_0,
    };

    // A parent always precedes its children in row-major order, i.e. sits left of or above them,
    // which is the direction NOC_1 routes.
    KernelDescriptor writer;
    writer.kernel_source = kernel_dir + "dataflow/writer_large_k_tree_reduce.cpp";
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = grid;
    writer.compile_time_args = {block_num_tiles, block_bytes};
    writer.named_compile_time_args = {
        {"cb_send", send_cb_index},
        {"cb_recv", recv_cb_index},
    };
    writer.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::NOC_1,
    };

    KernelDescriptor compute;
    compute.kernel_source = kernel_dir + "compute/compute_large_k_tree_reduce.cpp";
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = grid;
    compute.compile_time_args = {M_tiles, Kc_tiles, N_tiles, out_subblock_h, out_subblock_w, dst_num_tiles};
    compute.named_compile_time_args = {
        {"cb_in0", in0_cb_index},
        {"cb_in1", in1_cb_index},
        {"cb_partial", partial_cb_index},
        {"cb_recv", recv_cb_index},
        {"cb_out", out_cb_index},
        {"cb_send", send_cb_index},
    };
    if (use_custom_mm) {
        compute.defines.emplace_back("USE_CUSTOM_MM", "1");
    }
    compute.config = ComputeConfigDescriptor{
        .math_fidelity = use_custom_mm ? MathFidelity::LoFi : MathFidelity::HiFi4,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .math_approx_mode = false,
    };

    reader.runtime_args.reserve(num_cores);
    writer.runtime_args.reserve(num_cores);
    compute.runtime_args.reserve(num_cores);
    for (uint32_t i = 0; i < num_cores; ++i) {
        const LargeKTreeNode& node = tree[i];
        reader.runtime_args.emplace_back(cores[i], node.children_per_level);

        const bool has_parent = node.parent >= 0;
        const CoreCoord parent_phys =
            has_parent ? device->worker_core_from_logical_core(cores[node.parent]) : CoreCoord{0, 0};
        writer.runtime_args.emplace_back(
            cores[i],
            KernelDescriptor::CoreRuntimeArgs{
                static_cast<uint32_t>(has_parent),
                static_cast<uint32_t>(parent_phys.x),
                static_cast<uint32_t>(parent_phys.y),
                node.parent_level,
                node.slot_in_parent * block_bytes});

        compute.runtime_args.emplace_back(
            cores[i], KernelDescriptor::CoreRuntimeArgs{node.num_children, static_cast<uint32_t>(!has_parent)});
    }

    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(writer));
    desc.kernels.push_back(std::move(compute));

    log_debug(
        tt::LogOp,
        "matmul_decode_large_k: cores={}, M_tiles={}, Kc_tiles={}, N_tiles={}, fan_in={}, levels={}, custom_mm={}",
        num_cores,
        M_tiles,
        Kc_tiles,
        N_tiles,
        fan_in,
        num_levels,
        use_custom_mm);
    return desc;
}

}  // namespace ttnn::operations::experimental::matmul_decode

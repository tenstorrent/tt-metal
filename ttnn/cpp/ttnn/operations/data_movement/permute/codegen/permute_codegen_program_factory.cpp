// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/permute/codegen/permute_codegen_program_factory.hpp"

#include <algorithm>
#include <utility>
#include <vector>

#include <tt_stl/assert.hpp>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/tensor/tensor.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

namespace {

constexpr const char* kReaderStickUnified =
    "ttnn/cpp/ttnn/operations/data_movement/common/kernels/codegen/reader_stick_interleaved_unified.cpp";
constexpr const char* kWriterRowInvariant =
    "ttnn/cpp/ttnn/operations/data_movement/permute/codegen/kernels/writer_permute_rm_interleaved.cpp";
constexpr const char* kReaderBlocked =
    "ttnn/cpp/ttnn/operations/data_movement/permute/codegen/kernels/reader_permute_rm_blocked.cpp";
constexpr const char* kWriterBlocked =
    "ttnn/cpp/ttnn/operations/data_movement/permute/codegen/kernels/writer_permute_rm_blocked.cpp";
constexpr const char* kComputeBlocked =
    "ttnn/cpp/ttnn/operations/data_movement/permute/codegen/kernels/compute_permute_xw_rm.cpp";

// MODE_SEQUENCED and SEQ_IDENTITY, see common/kernels/codegen/sequencers.h.
constexpr uint32_t kModeSequenced = 4;
constexpr uint32_t kSeqIdentity = 0;

constexpr uint32_t kCbIn = 0;
constexpr uint32_t kCbTile = 1;
constexpr uint32_t kCbOut = 2;

constexpr uint32_t kXBlock = tt::constants::TILE_HEIGHT;
constexpr uint32_t kWBlock = tt::constants::TILE_WIDTH;
constexpr uint32_t kBlockedInDepth = 2 * kXBlock;
constexpr uint32_t kBlockedTileDepth = 2;
constexpr uint32_t kBlockedOutDepth = 2 * kWBlock;

constexpr uint32_t kUnusedNamedArg = 0;
constexpr uint32_t kSingleBatch = 1;
constexpr uint32_t kBytesPerWideElement = sizeof(uint32_t);

struct CoreSplit {
    CoreRangeSet all_cores;
    std::vector<CoreCoord> cores_in_order;
    CoreRangeSet core_group_1;
    CoreRangeSet core_group_2;
    uint32_t work_per_core_1 = 0;
    uint32_t work_per_core_2 = 0;
};

CoreSplit split_work(const Tensor& input, uint32_t total_work) {
    MeshDevice* device = input.device();
    auto grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, work_per_core_1, work_per_core_2] =
        tt::tt_metal::split_work_to_cores(grid_size, total_work, /*row_wise=*/false);
    return CoreSplit{
        .all_cores = all_cores,
        .cores_in_order = corerange_to_cores(all_cores, num_cores, /*row_wise=*/false),
        .core_group_1 = core_group_1,
        .core_group_2 = core_group_2,
        .work_per_core_1 = work_per_core_1,
        .work_per_core_2 = work_per_core_2,
    };
}

uint32_t work_for_core(const CoreSplit& split, const CoreCoord& core) {
    return split.core_group_1.contains(core) ? split.work_per_core_1 : split.work_per_core_2;
}

std::vector<uint32_t> shape_vector(const ttnn::Shape& shape) {
    std::vector<uint32_t> values;
    values.reserve(shape.rank());
    for (uint32_t i = 0; i < shape.rank(); ++i) {
        values.push_back(shape[i]);
    }
    return values;
}

// Strides in rows: the last two axes both step one row, since the last axis lives inside a page.
std::vector<uint32_t> row_strides(const std::vector<uint32_t>& shape) {
    std::vector<uint32_t> strides(shape.size(), 1);
    for (size_t i = shape.size() - kPermuteHeightAxisFromEnd; i > 0; --i) {
        strides[i - 1] = strides[i] * shape[i];
    }
    return strides;
}

uint64_t input_l1_bytes_per_bank(const Tensor& input) {
    return input.memory_config().buffer_type() == BufferType::L1 ? input.buffer()->aligned_size_per_bank() : 0;
}

KernelDescriptor make_kernel(const CoreRangeSet& cores) {
    KernelDescriptor kernel;
    kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    kernel.core_ranges = cores;
    return kernel;
}

CBDescriptor make_cb(
    uint32_t cb_id, tt::DataFormat data_format, uint32_t page_size, uint32_t depth, const CoreRangeSet& cores) {
    return CBDescriptor{
        .total_size = depth * page_size,
        .core_ranges = cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_id),
            .data_format = data_format,
            .page_size = page_size,
        }}},
    };
}

// Position of row `row` in the row space (every axis but the last), innermost axis last.
std::vector<uint32_t> row_index(uint32_t row, const std::vector<uint32_t>& row_shape) {
    std::vector<uint32_t> index(row_shape.size());
    for (size_t i = row_shape.size(); i > 0; --i) {
        index[i - 1] = row % row_shape[i - 1];
        row /= row_shape[i - 1];
    }
    return index;
}

ProgramDescriptor build_row_invariant(
    const Tensor& input,
    const Tensor& output,
    const std::vector<uint32_t>& input_shape,
    const std::vector<uint32_t>& dims,
    const std::vector<uint32_t>& output_shape,
    uint32_t cb_batch) {
    TT_FATAL(cb_batch > 0, "Permute codegen row-invariant CB does not fit below the lowest occupied L1 address");
    Buffer* src_buffer = input.buffer();
    Buffer* dst_buffer = output.buffer();
    const uint32_t width = input_shape[input_shape.size() - kPermuteWidthAxisFromEnd];
    const uint32_t row_axes = static_cast<uint32_t>(input_shape.size()) - kPermuteWidthAxisFromEnd;
    const uint32_t stick_bytes = width * input.element_size();
    const uint32_t src_page = static_cast<uint32_t>(src_buffer->aligned_page_size());
    const uint32_t slot_bytes = permute_rm_slot_bytes(input, output.memory_config());
    const uint32_t num_sticks = static_cast<uint32_t>(input.logical_shape().volume() / width);

    const CoreSplit split = split_work(input, num_sticks);
    const tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input.dtype());

    ProgramDescriptor desc;
    desc.cbs.push_back(make_cb(kCbIn, cb_data_format, slot_bytes, kPermuteRmCbTurns * cb_batch, split.all_cores));

    KernelDescriptor stick_reader = make_kernel(split.all_cores);
    stick_reader.kernel_source = kReaderStickUnified;
    TensorAccessorArgs(*src_buffer).append_to(stick_reader.compile_time_args);
    stick_reader.named_compile_time_args = {
        {"mode", kModeSequenced},
        {"cb_id", kCbIn},
        {"stick_bytes", stick_bytes},
        {"aligned_page_size", src_page},
        {"seq_id", kSeqIdentity},
        {"batch", cb_batch},
        // The shared reader reads every mode's named args even where the branch is discarded, and the
        // discarded pad branch still instantiates a fill templated on elem_size (2 or 4 bytes only).
        {"nabatch", kSingleBatch},
        {"hoist_cb_write_ptr", kUnusedNamedArg},
        {"elem_size", input.element_size()},
        {"tile_height", tt::constants::TILE_HEIGHT},
        {"tile_row_shift_bits", kUnusedNamedArg},
        {"num_pages_in_row", kSingleBatch},
        {"unpadded_X_bytes", kUnusedNamedArg},
        {"valid_last_page_bytes", kUnusedNamedArg},
        {"page_size", kUnusedNamedArg},
    };
    stick_reader.config = ReaderConfigDescriptor{};

    KernelDescriptor scatter_writer = make_kernel(split.all_cores);
    scatter_writer.kernel_source = kWriterRowInvariant;
    TensorAccessorArgs(*dst_buffer).append_to(scatter_writer.compile_time_args);
    scatter_writer.named_compile_time_args = {
        {"cb_id", kCbIn},
        {"write_size", stick_bytes},
        {"batch", cb_batch},
    };
    scatter_writer.config = WriterConfigDescriptor{};

    KernelDescriptor::RTArgList scatter_writer_common;
    scatter_writer_common.push_back(dst_buffer);
    scatter_writer.emplace_common_runtime_args(scatter_writer_common);

    // The writer walks input rows in order; the permute sequencer maps each to its output row.
    const std::vector<uint32_t> row_shape(input_shape.begin(), input_shape.begin() + row_axes);
    const std::vector<uint32_t> row_perm(dims.begin(), dims.begin() + row_axes);
    const std::vector<uint32_t> out_row_strides = row_strides(output_shape);

    uint32_t start = 0;
    for (const auto& core : split.cores_in_order) {
        const uint32_t n = work_for_core(split, core);
        stick_reader.emplace_runtime_args(core, {src_buffer, n, start});

        KernelDescriptor::RTArgList scatter_args;
        scatter_args.push_back(n);
        scatter_args.push_back(row_axes);
        scatter_args.append(std::vector<uint32_t>(out_row_strides.begin(), out_row_strides.begin() + row_axes));
        scatter_args.append(row_shape);
        scatter_args.append(row_perm);
        scatter_args.append(row_index(start, row_shape));
        scatter_writer.emplace_runtime_args(core, scatter_args);
        start += n;
    }

    desc.kernels.push_back(std::move(stick_reader));
    desc.kernels.push_back(std::move(scatter_writer));
    return desc;
}

ProgramDescriptor build_blocked(
    const Tensor& input,
    const Tensor& output,
    const std::vector<uint32_t>& input_shape,
    const std::vector<uint32_t>& dims,
    const std::vector<uint32_t>& output_shape) {
    Buffer* src_buffer = input.buffer();
    Buffer* dst_buffer = output.buffer();
    const uint32_t rank = static_cast<uint32_t>(input_shape.size());
    const uint32_t element_size = input.element_size();
    const uint32_t x_dim = dims[rank - kPermuteWidthAxisFromEnd];
    const uint32_t X = input_shape[x_dim];
    const uint32_t W = input_shape[rank - kPermuteWidthAxisFromEnd];
    const uint32_t num_rows = static_cast<uint32_t>(input.logical_shape().volume() / W);
    const uint32_t x_blocks = tt::div_up(X, kXBlock);
    const uint32_t w_blocks = tt::div_up(W, kWBlock);
    const uint32_t num_blocks_total = (num_rows / X) * x_blocks * w_blocks;

    const uint32_t in_cb_page = kWBlock * element_size;
    const uint32_t out_cb_page = kXBlock * element_size;
    const uint32_t tile_cb_page = kXBlock * kWBlock * element_size;

    const uint32_t in_page = static_cast<uint32_t>(src_buffer->aligned_page_size());
    const uint32_t out_page = static_cast<uint32_t>(dst_buffer->aligned_page_size());

    const CoreSplit split = split_work(input, num_blocks_total);

    // Four-byte datums only move through the compute kernel, so carry them as int32 to keep every bit.
    const tt::DataFormat cb_data_format =
        element_size == kBytesPerWideElement ? tt::DataFormat::Int32 : datatype_to_dataformat_converter(input.dtype());

    ProgramDescriptor desc;
    desc.cbs.push_back(make_cb(kCbIn, cb_data_format, in_cb_page, kBlockedInDepth, split.all_cores));
    desc.cbs.push_back(make_cb(kCbTile, cb_data_format, tile_cb_page, kBlockedTileDepth, split.all_cores));
    desc.cbs.push_back(make_cb(kCbOut, cb_data_format, out_cb_page, kBlockedOutDepth, split.all_cores));

    KernelDescriptor reader = make_kernel(split.all_cores);
    reader.kernel_source = kReaderBlocked;
    TensorAccessorArgs(*src_buffer).append_to(reader.compile_time_args);
    reader.named_compile_time_args = {
        {"N", rank},
        {"page_size", in_cb_page},
        {"num_rows", num_rows},
        {"x_dim", x_dim},
        {"num_blocks_total", num_blocks_total},
        {"x_blocks", x_blocks},
        {"w_blocks", w_blocks},
        {"x_block_size", kXBlock},
        {"w_block_size", kWBlock},
        {"element_size", element_size},
        {"in_page_size", in_page},
    };
    reader.config = ReaderConfigDescriptor{};

    KernelDescriptor writer = make_kernel(split.all_cores);
    writer.kernel_source = kWriterBlocked;
    TensorAccessorArgs(*dst_buffer).append_to(writer.compile_time_args);
    writer.named_compile_time_args = {
        {"N", rank},
        {"output_page_size", out_cb_page},
        {"num_rows", num_rows},
        {"X", X},
        {"x_dim", x_dim},
        {"x_blocks", x_blocks},
        {"w_blocks", w_blocks},
        {"x_block_size", kXBlock},
        {"w_block_size", kWBlock},
        {"W", W},
        {"element_size", element_size},
        {"out_page_size", out_page},
    };
    writer.config = WriterConfigDescriptor{};

    KernelDescriptor compute = make_kernel(split.all_cores);
    compute.kernel_source = kComputeBlocked;
    compute.named_compile_time_args = {{"x_block_size", kXBlock}, {"w_block_size", kWBlock}};
    compute.config = ComputeConfigDescriptor{.fp32_dest_acc_en = element_size == kBytesPerWideElement};

    KernelDescriptor::RTArgList reader_common;
    reader_common.reserve(1 + 2 * rank);
    reader_common.push_back(src_buffer);
    reader_common.append(input_shape);
    reader_common.append(row_strides(input_shape));
    reader.emplace_common_runtime_args(reader_common);

    KernelDescriptor::RTArgList writer_common;
    writer_common.reserve(1 + 3 * rank);
    writer_common.push_back(dst_buffer);
    writer_common.append(input_shape);
    writer_common.append(dims);
    writer_common.append(row_strides(output_shape));
    writer.emplace_common_runtime_args(writer_common);

    uint32_t start_block = 0;
    for (const auto& core : split.cores_in_order) {
        const uint32_t n_blocks = work_for_core(split, core);
        const uint32_t end_block = start_block + n_blocks;
        reader.emplace_runtime_args(core, {start_block, end_block});
        writer.emplace_runtime_args(core, {start_block, end_block});
        compute.emplace_runtime_args(core, {n_blocks, 0u, 0u});
        start_block = end_block;
    }

    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(compute));
    desc.kernels.push_back(std::move(writer));
    return desc;
}

}  // namespace

uint32_t permute_rm_slot_bytes(const Tensor& input, const MemoryConfig& output_mem_config) {
    const ttnn::Shape& shape = input.logical_shape();
    const uint32_t stick_bytes = shape[shape.rank() - kPermuteWidthAxisFromEnd] * input.element_size();
    const uint32_t output_alignment = input.device()->allocator()->get_alignment(output_mem_config.buffer_type());
    return std::max(
        static_cast<uint32_t>(input.buffer()->aligned_page_size()), tt::round_up(stick_bytes, output_alignment));
}

uint64_t permute_cb_l1_budget(const Tensor& input) {
    MeshDevice* device = input.device();
    const uint64_t window = device->l1_size_per_core() - device->allocator()->get_base_allocator_addr(HalMemType::L1);
    const uint64_t operands = input_l1_bytes_per_bank(input);
    return window > operands ? window - operands : 0;
}

uint64_t permute_live_cb_l1_budget(const Tensor& input) {
    MeshDevice* device = input.device();
    const uint64_t base = device->allocator()->get_base_allocator_addr(HalMemType::L1);
    uint64_t window = device->l1_size_per_core() - base;
    if (const auto lowest = device->lowest_occupied_compute_l1_address(); lowest.has_value()) {
        const uint64_t frontier = static_cast<uint64_t>(*lowest);
        window = std::min(window, frontier > base ? frontier - base : 0);
    }
    const uint64_t operands = input_l1_bytes_per_bank(input);
    return window > operands ? window - operands : 0;
}

uint32_t permute_live_rm_cb_batch(
    const Tensor& input,
    const MemoryConfig& output_mem_config,
    const std::array<uint32_t, kPermuteCodegenMaxDims>& dims) {
    // Runs before validation, so a tensor that validation will reject must not be dereferenced here.
    if (input.storage_type() != StorageType::DEVICE || input.buffer() == nullptr) {
        return 0;
    }
    const uint32_t rank = input.logical_shape().rank();
    if (rank < kPermuteHeightAxisFromEnd || rank > kPermuteCodegenMaxDims) {
        return 0;
    }
    // Only the row-invariant path sizes its CB from the batch; keying it elsewhere splits the cache for nothing.
    if (dims[rank - kPermuteWidthAxisFromEnd] != rank - kPermuteWidthAxisFromEnd) {
        return 0;
    }
    const std::optional<PermuteRmCbPlan> plan =
        plan_permute_rm_cb(permute_rm_slot_bytes(input, output_mem_config), permute_live_cb_l1_budget(input));
    return plan.has_value() ? plan->batch : 0;
}

std::optional<PermuteRmCbPlan> plan_permute_rm_cb(uint32_t slot_bytes, uint64_t l1_budget) {
    for (uint32_t batch = kPermuteRmMaxBatch; batch >= 1; batch /= 2) {
        if (uint64_t{kPermuteRmCbTurns} * batch * slot_bytes <= l1_budget) {
            return PermuteRmCbPlan{.batch = batch, .depth = kPermuteRmCbTurns * batch};
        }
    }
    return std::nullopt;
}

uint64_t permute_blocked_cb_bytes(uint32_t element_size) {
    const uint64_t in_bytes = uint64_t{kBlockedInDepth} * kWBlock * element_size;
    const uint64_t tile_bytes = uint64_t{kBlockedTileDepth} * kXBlock * kWBlock * element_size;
    const uint64_t out_bytes = uint64_t{kBlockedOutDepth} * kXBlock * element_size;
    return in_bytes + tile_bytes + out_bytes;
}

ProgramDescriptor PermuteCodegenProgramFactory::create_descriptor(
    const PermuteCodegenParams& operation_attributes,
    const PermuteCodegenInputs& tensor_args,
    Tensor& tensor_return_value) {
    const Tensor& input = tensor_args.input;
    const std::vector<uint32_t> input_shape = shape_vector(input.logical_shape());
    const uint32_t rank = static_cast<uint32_t>(input_shape.size());
    const std::vector<uint32_t> dims(operation_attributes.dims.begin(), operation_attributes.dims.begin() + rank);
    const std::vector<uint32_t> output_shape = shape_vector(tensor_return_value.logical_shape());

    const bool width_axis_fixed = dims[rank - kPermuteWidthAxisFromEnd] == rank - kPermuteWidthAxisFromEnd;
    return width_axis_fixed
               ? build_row_invariant(
                     input, tensor_return_value, input_shape, dims, output_shape, operation_attributes.rm_cb_batch)
               : build_blocked(input, tensor_return_value, input_shape, dims, output_shape);
}

}  // namespace ttnn::prim

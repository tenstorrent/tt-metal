// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "index_fill_device_operation.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::index_fill {

IndexFillOperation::MultiCore::cached_program_t IndexFillOperation::MultiCore::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const tt::tt_metal::Tensor& index = tensor_args.index;
    const tt::tt_metal::Tensor& input = tensor_args.input;
    uint32_t dim = operation_attributes.dim;

    const auto input_shape = input.padded_shape();
    const auto n = input_shape.rank();
    uint32_t num_rows_in_dim = 1;
    for (int i = n - 2; i > static_cast<int>(dim); --i) {
        num_rows_in_dim *= input_shape[i];
    }

    // Prepare fill_value to send as a uint32_t kernel arg
    auto fill_value_ = operation_attributes.value;
    uint32_t fill_value{};
    switch (input.dtype()) {
        case DataType::BFLOAT16:
            fill_value = pack_two_bfloat16_into_uint32({bfloat16(std::get<float>(fill_value_)), bfloat16(0.0f)});
            break;
        case DataType::FLOAT32: fill_value = std::bit_cast<uint32_t>(std::get<float>(fill_value_)); break;
        case DataType::INT32: fill_value = static_cast<uint32_t>(std::get<int>(fill_value_)); break;
        default: TT_FATAL(false, "Unsupported datatype"); break;
    }

    ////////////////////////////////////////////////////////////////////////////
    //                            Program Setup
    ////////////////////////////////////////////////////////////////////////////
    Program program{};

    // Distribute work across core grid
    uint32_t num_rows = static_cast<uint32_t>(input.physical_volume() / input.padded_shape()[-1]);

    CoreRangeSet all_cores{};
    CoreRangeSet core_group_1{};
    CoreRangeSet core_group_2{};
    uint32_t num_rows_per_core_group_1{};
    uint32_t num_rows_per_core_group_2{};

    auto input_mem_layout = input.memory_config().memory_layout();

    // Per-core shard parameters (populated in the branch below)
    struct CoreShardInfo {
        uint32_t start_row{};
        uint32_t end_row{};
        uint32_t col_shard_id{};     // column shard index (0 for HEIGHT/INTERLEAVED)
        uint32_t row_page_stride{};  // KW for WIDTH/BLOCK, 1 for HEIGHT/INTERLEAVED
    };
    std::vector<CoreShardInfo> core_shard_infos;

    // Ordering used to enumerate cores in the runtime-args loop below. For WIDTH/BLOCK the
    // per-core shard info is generated in shard-orientation order, so the runtime loop must
    // enumerate cores in the same order to pair each core with its own shard parameters.
    bool cores_row_wise = false;

    if (input_mem_layout == TensorMemoryLayout::INTERLEAVED) {
        auto compute_with_storage_grid_size = input.device()->compute_with_storage_grid_size();
        std::tie(
            std::ignore, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_rows);
        // col_shard_id=0, row_page_stride=1 for all cores — filled below
    } else if (input_mem_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
        const auto& shard_spec = input.shard_spec().value();
        all_cores = shard_spec.grid;
        core_group_1 = all_cores;
        num_rows_per_core_group_1 = shard_spec.shape[0];
        num_rows_per_core_group_2 = 0;
        // col_shard_id=0, row_page_stride=1 — filled below
    } else if (input_mem_layout == TensorMemoryLayout::WIDTH_SHARDED) {
        const auto& shard_spec = input.shard_spec().value();
        all_cores = shard_spec.grid;
        core_group_1 = all_cores;
        num_rows_per_core_group_1 = 0;  // unused for WIDTH_SHARDED — per-core infos set below
        num_rows_per_core_group_2 = 0;

        bool rm_orientation = (shard_spec.orientation == ShardOrientation::ROW_MAJOR);
        cores_row_wise = rm_orientation;
        auto cores_vec = corerange_to_cores(all_cores, std::nullopt, rm_orientation);
        uint32_t KW = cores_vec.size();
        core_shard_infos.reserve(KW);
        for (uint32_t k = 0; k < KW; ++k) {
            core_shard_infos.push_back({0, num_rows, k, KW});
        }
    } else {
        // BLOCK_SHARDED
        const auto& shard_spec = input.shard_spec().value();
        all_cores = shard_spec.grid;
        core_group_1 = all_cores;
        num_rows_per_core_group_1 = 0;  // unused — per-core infos set below
        num_rows_per_core_group_2 = 0;

        bool rm_orientation = (shard_spec.orientation == ShardOrientation::ROW_MAJOR);
        cores_row_wise = rm_orientation;
        auto cores_vec = corerange_to_cores(all_cores, std::nullopt, rm_orientation);

        uint32_t shard_h = shard_spec.shape[0];
        uint32_t shard_w = shard_spec.shape[1];
        uint32_t KW = (input.padded_shape()[-1] + shard_w - 1) / shard_w;
        uint32_t KH = (num_rows + shard_h - 1) / shard_h;

        core_shard_infos.reserve(cores_vec.size());
        for (uint32_t i = 0; i < cores_vec.size(); ++i) {
            uint32_t kh = rm_orientation ? (i / KW) : (i % KH);
            uint32_t kw = rm_orientation ? (i % KW) : (i / KH);
            uint32_t row_start = kh * shard_h;
            uint32_t row_end = std::min(row_start + shard_h, num_rows);
            core_shard_infos.push_back({row_start, row_end, kw, KW});
        }
    }

    // Create circular buffers
    auto input_dataformat = datatype_to_dataformat_converter(input.dtype());
    auto index_dataformat = datatype_to_dataformat_converter(index.dtype());

    uint32_t input_page_size = input.buffer()->aligned_page_size();
    uint32_t index_total_size = index.buffer()->aligned_size();

    uint32_t input_cb_depth = 2;

    // CB to store pages from input tensor
    auto cb_index = tt::CBIndex::c_0;
    CreateCircularBuffer(
        program,
        all_cores,
        tt::tt_metal::CircularBufferConfig(input_cb_depth * input_page_size, {{cb_index, input_dataformat}})
            .set_page_size(cb_index, input_page_size));

    // CB to store entire index tensor
    cb_index = tt::CBIndex::c_1;
    CreateCircularBuffer(
        program,
        all_cores,
        tt::tt_metal::CircularBufferConfig(index_total_size, {{cb_index, index_dataformat}})
            .set_page_size(cb_index, index_total_size));

    // CB to store an input page filled with fill_value
    cb_index = tt::CBIndex::c_2;
    CreateCircularBuffer(
        program,
        all_cores,
        tt::tt_metal::CircularBufferConfig(input_page_size, {{cb_index, input_dataformat}})
            .set_page_size(cb_index, input_page_size));

    // Create reader kernel
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)input_page_size,          // input tensor page size
        (std::uint32_t)index_total_size,         // index tensor total size
        (std::uint32_t)index.physical_volume(),  // num elements in index array
        (std::uint32_t)(dim == n - 1)            // is last dim
    };
    tt::tt_metal::TensorAccessorArgs(input.buffer()).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(index.buffer()).append_to(reader_compile_time_args);

    auto reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/index_fill/device/kernels/reader_index_fill.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    // Compute output write parameters (independent of input sharding).
    // For same-layout cases these equal the input params; for cross-layout they differ.
    // These are all loop-invariant, so they are computed once here rather than per-core.
    auto out_mem_layout = output.memory_config().memory_layout();
    bool out_is_col_sharded =
        (out_mem_layout == TensorMemoryLayout::WIDTH_SHARDED || out_mem_layout == TensorMemoryLayout::BLOCK_SHARDED);
    bool input_is_col_sharded =
        (input_mem_layout == TensorMemoryLayout::WIDTH_SHARDED ||
         input_mem_layout == TensorMemoryLayout::BLOCK_SHARDED);
    bool col_sharded_to_col_sharded = input_is_col_sharded && out_is_col_sharded;

    uint32_t out_write_size{};  // bytes written per NOC write to output
    uint32_t out_KW{};          // number of column writes per input row (1 unless splitting into WIDTH/BLOCK)

    if (out_is_col_sharded) {
        // NOTE: assumes the input shard page is tightly packed (page_size == shard_width *
        // element_size, no extra NOC-alignment padding), so that the WIDTH/BLOCK→INTERLEAVED/HEIGHT
        // byte offset (col_shard_id * input_page_size) and the row-splitting write size line up.
        // This holds for row-major L1 sharded buffers used here; revisit if padded page sizes are introduced.
        out_write_size = output.buffer()->aligned_page_size();
        out_KW = (input.padded_shape()[-1] * input.element_size()) / out_write_size;
    } else {
        // INTERLEAVED or HEIGHT_SHARDED output: write size = full input page
        out_write_size = input_page_size;
        out_KW = 1;
    }

    // Create writer kernel
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t)input_page_size,          // CB page size (= input shard row size)
        (std::uint32_t)index.physical_volume(),  // num elements in index array
        (std::uint32_t)input.element_size(),     // element size in bytes
        (std::uint32_t)(dim == n - 1)            // is last dim
    };
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_compile_time_args);

    auto writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/index_fill/device/kernels/writer_index_fill.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    // Set runtime args for each core. Enumerate in the same order used to build
    // core_shard_infos so each core is paired with its own shard parameters.
    auto cores = corerange_to_cores(all_cores, std::nullopt, cores_row_wise);
    bool use_per_core_shard_infos = !core_shard_infos.empty();

    uint32_t start_row_id = 0;
    for (uint32_t core_idx = 0; core_idx < cores.size(); ++core_idx) {
        const auto& core = cores[core_idx];

        uint32_t core_start_row{};
        uint32_t core_end_row{};
        uint32_t col_shard_id{};
        uint32_t row_page_stride{};

        if (use_per_core_shard_infos) {
            // WIDTH_SHARDED or BLOCK_SHARDED: use pre-computed shard info
            const auto& info = core_shard_infos[core_idx];
            core_start_row = info.start_row;
            core_end_row = info.end_row;
            col_shard_id = info.col_shard_id;
            row_page_stride = info.row_page_stride;
        } else {
            // INTERLEAVED or HEIGHT_SHARDED: row-based distribution
            uint32_t num_rows_per_core{};
            if (core_group_1.contains(core)) {
                num_rows_per_core = num_rows_per_core_group_1;
            } else if (core_group_2.contains(core)) {
                num_rows_per_core = num_rows_per_core_group_2;
            } else {
                TT_FATAL(false, "Core not in specified core ranges");
            }
            core_start_row = start_row_id;
            core_end_row = start_row_id + num_rows_per_core;
            col_shard_id = 0;
            row_page_stride = 1;
            start_row_id += num_rows_per_core;
        }

        SetRuntimeArgs(
            program,
            reader_kernel_id,
            core,
            {
                input.buffer()->address(),  // input tensor address
                index.buffer()->address(),  // index tensor address
                core_start_row,             // start row
                core_end_row,               // end row
                num_rows_in_dim,            // num rows in dim
                input_shape[dim],           // dim size
                col_shard_id,               // column shard index
                row_page_stride             // page stride across rows
            });

        // Output-specific write params:
        //   out_row_page_stride / out_col_shard_id: page_id = row_id * stride + shard_id
        //   in_col_shard_id:      input col shard; drives col_start for last-dim fill bounds
        //   out_col_byte_offset:  byte offset within the output page (WIDTH/BLOCK→INTERLEAVED/HEIGHT)
        //   out_num_col_shards:   number of output writes per input page (>1 when splitting rows)
        //   out_write_size:       bytes per output NOC write
        uint32_t out_col_shard_id{};
        uint32_t out_row_page_stride{};
        uint32_t out_col_byte_offset{};
        uint32_t out_num_col_shards{};  // writer loop count: 1 except INTERLEAVED/HEIGHT → WIDTH/BLOCK

        if (col_sharded_to_col_sharded) {
            // Column-sharded → column-sharded (WIDTH↔WIDTH, BLOCK↔BLOCK, WIDTH↔BLOCK).
            // Validation guarantees matching column shard width, so input KW == output KW and
            // the page formula row*KW+col is identical on both sides. One write per row; the
            // output TensorAccessor routes each page to the physically-owning core (which may
            // differ from the executing core when converting between WIDTH and BLOCK row layouts).
            out_row_page_stride = row_page_stride;  // = KW (same for input and output)
            out_col_shard_id = col_shard_id;        // same column shard index
            out_col_byte_offset = 0;
            out_num_col_shards = 1;
        } else if (out_is_col_sharded) {
            // INTERLEAVED/HEIGHT in → WIDTH/BLOCK out: one core splits each full input row
            // into out_KW fragments and writes each to its respective output shard.
            out_row_page_stride = out_KW;
            out_col_shard_id = 0;  // loop starts from shard 0, incrementing kout
            out_col_byte_offset = 0;
            out_num_col_shards = out_KW;
        } else {
            // WIDTH/BLOCK in → INTERLEAVED/HEIGHT out: write fragment at a byte offset
            // into the full output row; no loop needed.
            out_row_page_stride = 1;
            out_col_shard_id = 0;
            out_col_byte_offset = col_shard_id * input_page_size;  // 0 for INTERLEAVED/HEIGHT input
            out_num_col_shards = 1;
        }

        SetRuntimeArgs(
            program,
            writer_kernel_id,
            core,
            {
                output.buffer()->address(),  // output tensor address
                core_start_row,              // start row
                core_end_row,                // end row
                num_rows_in_dim,             // num rows in dim
                input_shape[dim],            // dim size
                fill_value,                  // fill value
                col_shard_id,                // input col shard (for last-dim bounds check)
                // --- output write params ---
                out_col_shard_id,     // output col shard base index
                out_row_page_stride,  // output page stride
                out_col_byte_offset,  // byte offset within output page
                out_num_col_shards,   // number of output writes per row
                out_write_size        // bytes per output write
            });
    }

    return {std::move(program), {reader_kernel_id, writer_kernel_id, cores}};
}

void IndexFillOperation::MultiCore::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& program = cached_program.program;
    auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    auto& cores = cached_program.shared_variables.cores;

    auto src_buffer = tensor_args.input.buffer()->address();
    auto index_buffer = tensor_args.index.buffer()->address();
    auto output_buffer = output.buffer()->address();

    for (const auto& core : cores) {
        {
            auto& runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = src_buffer;
            runtime_args[1] = index_buffer;
        }

        {
            auto& runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = output_buffer;
        }
    }
}

}  // namespace ttnn::operations::index_fill

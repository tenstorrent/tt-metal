// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "concat_block_sharded_program_factory.hpp"

#include <algorithm>
#include <array>
#include <numeric>
#include <string>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/buffer_distribution_spec.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/workload/run_args/program_run_args.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::prim {

// One rectangular copy: num_rows rows of copy_size bytes, read from shard shard_id of input
// input_index and written into this core's output shard. Offsets are bytes into the shard.
struct TransferDesc {
    uint32_t input_index;
    uint32_t shard_id;
    uint32_t src_offset;
    uint32_t src_stride;
    uint32_t dst_offset;
    uint32_t dst_stride;
    uint32_t copy_size;
    uint32_t num_rows;
};

ttnn::device_operation::ProgramArtifacts ConcatBlockShardedProgramFactory::create_program_artifacts(
    const ConcatParams& operation_attributes, const ConcatInputs& tensor_args, Tensor& tensor_return_value) {
    using namespace tt::constants;
    using namespace tt::tt_metal;
    using tt::tt_metal::experimental::KernelAdvancedOptions;

    const auto& input_tensors = tensor_args.input_tensors;
    const uint32_t dim = operation_attributes.dim;
    Tensor& output = tensor_return_value;

    const uint32_t rank = input_tensors[0].logical_shape().rank();
    // ConcatBlockShardedProgramFactory supports concat only on the last two dims (H, W)
    const bool width_concat = is_width_concat(rank, dim);
    // Height concat has to interleave per leading index (see num_leading_blocks); width concat
    // appends along the row and needs no block loop.
    const uint32_t num_blocks = width_concat ? 1u : num_leading_blocks(input_tensors[0]);

    const uint32_t num_input_tensors = input_tensors.size();

    const tt::DataFormat data_format = datatype_to_dataformat_converter(output.dtype());
    const bool rm_layout = output.layout() == Layout::ROW_MAJOR;
    const uint32_t element_size = input_tensors[0].element_size();

    const auto& first_shard_spec = input_tensors[0].shard_spec().value();
    const bool row_major_orient = first_shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    const auto all_cores = first_shard_spec.grid;
    TT_FATAL(
        all_cores.ranges().size() == 1,
        "Block-sharded concat requires a single contiguous rectangular CoreRange, got {} ranges",
        all_cores.ranges().size());
    const auto& core_range = *all_cores.ranges().begin();
    const uint32_t start_x = core_range.start_coord.x;
    const uint32_t start_y = core_range.start_coord.y;
    const uint32_t grid_cols = core_range.end_coord.x - start_x + 1;
    const uint32_t grid_rows = core_range.end_coord.y - start_y + 1;

    // For ROW_MAJOR: height distributed across grid rows (y), width across grid cols (x)
    // For COL_MAJOR: height distributed across grid cols (x), width across grid rows (y)
    const uint32_t shard_grid_h = row_major_orient ? grid_rows : grid_cols;
    const uint32_t shard_grid_w = row_major_orient ? grid_cols : grid_rows;

    uint32_t unit_h, unit_w, unit_size;
    if (rm_layout) {
        unit_h = 1;
        unit_w = 1;
        unit_size = element_size;
    } else {
        unit_h = TILE_HEIGHT;
        unit_w = TILE_WIDTH;
        unit_size = tt::tile_size(data_format);
    }

    std::vector<uint32_t> input_shard_h(num_input_tensors);
    std::vector<uint32_t> input_shard_w(num_input_tensors);

    const uint32_t l1_alignment = tt::tt_metal::hal::get_l1_alignment();

    for (uint32_t i = 0; i < num_input_tensors; i++) {
        const auto& ss = input_tensors[i].shard_spec().value();
        input_shard_h[i] = ss.shape[0];
        input_shard_w[i] = ss.shape[1];
        if (rm_layout) {
            const uint32_t row_bytes = input_shard_w[i] * element_size;
            TT_FATAL(
                row_bytes % l1_alignment == 0,
                "Input {} shard row size ({} bytes = {} elements x {} bytes) is not L1-aligned ({} bytes). "
                "Block-sharded RM concat requires aligned row sizes for correct NOC transfers.",
                i,
                row_bytes,
                input_shard_w[i],
                element_size,
                l1_alignment);
        } else {
            TT_FATAL(
                input_shard_h[i] % TILE_HEIGHT == 0 && input_shard_w[i] % TILE_WIDTH == 0,
                "Input {} shard shape ({}, {}) is not tile-aligned ({}x{})",
                i,
                input_shard_h[i],
                input_shard_w[i],
                TILE_HEIGHT,
                TILE_WIDTH);
        }
    }

    const auto& out_ss = output.shard_spec().value();
    const uint32_t output_shard_h = out_ss.shape[0];
    const uint32_t output_shard_w = out_ss.shape[1];
    if (rm_layout) {
        const uint32_t out_row_bytes = output_shard_w * element_size;
        TT_FATAL(
            out_row_bytes % l1_alignment == 0,
            "Output shard row size ({} bytes = {} elements x {} bytes) is not L1-aligned ({} bytes). "
            "Block-sharded RM concat requires aligned row sizes for correct NOC transfers.",
            out_row_bytes,
            output_shard_w,
            element_size,
            l1_alignment);
    } else {
        TT_FATAL(
            output_shard_h % TILE_HEIGHT == 0 && output_shard_w % TILE_WIDTH == 0,
            "Output shard shape ({}, {}) is not tile-aligned ({}x{})",
            output_shard_h,
            output_shard_w,
            TILE_HEIGHT,
            TILE_WIDTH);
    }

    auto to_units_h = [&](uint32_t h) { return h / unit_h; };
    auto to_units_w = [&](uint32_t w) { return w / unit_w; };

    const uint32_t out_units_w = to_units_w(output_shard_w);
    const uint32_t dst_stride_bytes = out_units_w * unit_size;

    // Rows each input contributes per leading index, the output rows one leading index spans, and
    // the output's real flattened height.
    //
    // These come from the tensors' own heights, not from input_shard_h * shard_grid_h. That
    // product is the allocated shard *capacity*, and block sharding lets the last height shard be
    // part padding -- flattened height 10 across two shards of 6 has a capacity of 12. Dividing
    // capacity by the block count folds that padding into every block, which shifts each block
    // after the first. Shard height is still what maps a row to its source core, just below.
    //
    // A tensor's flattened height is num_blocks * padded_shape[-2], so the per-block row count is
    // simply padded_shape[-2].
    std::vector<uint32_t> input_block_h(num_input_tensors);
    uint32_t output_block_h = 0;
    uint32_t out_total_h = 0;
    if (!width_concat) {
        for (uint32_t i = 0; i < num_input_tensors; i++) {
            input_block_h[i] = input_tensors[i].padded_shape()[-2];
            output_block_h += input_block_h[i];
        }
        out_total_h = output_block_h * num_blocks;
        // Height concat sums the inputs' heights, so this is a restatement of the output spec.
        TT_FATAL(
            output_block_h == output.padded_shape()[-2],
            "Height concat: inputs contribute {} rows per leading index but the output's height is "
            "{} (shape {}).",
            output_block_h,
            output.padded_shape()[-2],
            output.padded_shape());
        // The shards have to be able to hold the result; they may hold more (ragged tail).
        TT_FATAL(
            output_shard_h * shard_grid_h >= out_total_h,
            "Height concat: output shards span {} x {} rows, too few for the {} rows of shape {}.",
            output_shard_h,
            shard_grid_h,
            out_total_h,
            output.padded_shape());
    }

    // Columns each input contributes, logical rather than padded: a ragged width pads the last
    // dim out to whole shards -- 11 over three shards of 4 is stored as 12 -- so padded_shape()[-1]
    // equals input_shard_w * shard_grid_w for a tensor spanning every grid column, and using it
    // reproduces the capacity cursor exactly. Shard width still locates a column in its shard, below.
    std::vector<uint32_t> input_total_w(num_input_tensors);
    uint32_t out_total_w = 0;
    if (width_concat) {
        for (uint32_t i = 0; i < num_input_tensors; i++) {
            input_total_w[i] = input_tensors[i].logical_shape()[-1];
            out_total_w += input_total_w[i];
        }
        // Width concat sums the inputs' widths, so this restates the output spec.
        TT_FATAL(
            out_total_w == output.logical_shape()[-1],
            "Width concat: inputs contribute {} columns but the output's width is {} (shape {}).",
            out_total_w,
            output.logical_shape()[-1],
            output.logical_shape());
        // The shards have to hold the result; they may hold more (a padded tail).
        TT_FATAL(
            output_shard_w * shard_grid_w >= out_total_w,
            "Width concat: output shards span {} x {} columns, too few for the {} of shape {}.",
            output_shard_w,
            shard_grid_w,
            out_total_w,
            output.logical_shape());
        // The per-input start alignment is checked in validate_on_program_cache_miss, before a
        // factory is chosen, since it depends only on the input shapes, dtype and alignment.
    }

    // Shard ids. Each source shard's tensor-space indices (shard row, shard col) are already known
    // below; the kernel addresses that shard as shard_row * shards_per_row + shard_col, with
    // shards_per_row taken from that input's own shape.
    //
    // The distribution spec a legacy BLOCK_SHARDED tensor resolves to; its core list says which core
    // holds which shard id.
    auto distribution_spec = [](const Tensor& tensor) {
        const auto dist_spec = tensor.tensor_spec().compute_buffer_sharding_args().buffer_distribution_spec();
        TT_FATAL(dist_spec.has_value(), "Block-sharded concat: a tensor has no buffer distribution spec");
        return *dist_spec;
    };
    // Shards along one row of the shard grid: the step between shard ids one shard row apart. Shard
    // ids run along shard rows, but this is taken from the tensor's 2D physical shape rather than
    // the distribution spec's page shapes, because the spec may merge dimensions: a TILE tensor of
    // 2 x 2 one-tile shards is described as 4 pages in 1D, whose last dim is not the shard row.
    auto shards_per_row = [](const Tensor& tensor) {
        return tt::div_up(
            static_cast<uint32_t>(tensor.tensor_spec().physical_shape().width()), tensor.shard_spec().value().shape[1]);
    };
    std::vector<uint32_t> input_shards_per_row(num_input_tensors);
    for (uint32_t i = 0; i < num_input_tensors; i++) {
        input_shards_per_row[i] = shards_per_row(input_tensors[i]);
    }
    const BufferDistributionSpec output_dist_spec = distribution_spec(output);
    const uint32_t output_shards_per_row = shards_per_row(output);
    // Shard rows the output actually has. The grid can be bigger than the tensor needs, and the
    // distribution spec trims those surplus grid rows and columns: they hold no shard, so there is
    // nothing to copy into them and no source shard to copy from.
    const uint32_t output_shard_rows =
        tt::div_up(static_cast<uint32_t>(output.tensor_spec().physical_shape().height()), output_shard_h);

    auto input_shard_id = [&](uint32_t inp_id, uint32_t shard_row, uint32_t shard_col) {
        return (shard_row * input_shards_per_row[inp_id]) + shard_col;
    };

    // Pass 1: compute transfer descriptors for all cores
    // Index as [gy * grid_cols + gx]
    std::vector<std::vector<TransferDesc>> all_transfers(grid_rows * grid_cols);

    for (uint32_t gy = 0; gy < grid_rows; gy++) {
        for (uint32_t gx = 0; gx < grid_cols; gx++) {
            auto& transfers = all_transfers[gy * grid_cols + gx];

            // Map grid position to shard-space indices
            const uint32_t sh = row_major_orient ? gy : gx;  // shard height index
            const uint32_t sw = row_major_orient ? gx : gy;  // shard width index
            const CoreCoord logical_core(start_x + gx, start_y + gy);

            // A grid position past the output's shards has no output shard; leave it with no copies.
            if (sh >= output_shard_rows || sw >= output_shards_per_row) {
                continue;
            }

            if (width_concat) {
                // Clipped to the real width: the last shard can run past it into padding, which
                // has no source column.
                const uint32_t out_col_start = sw * output_shard_w;
                const uint32_t out_col_end = std::min(out_col_start + output_shard_w, out_total_w);
                const uint32_t num_rows_units = to_units_h(output_shard_h);

                uint32_t cum_w = 0;
                for (uint32_t inp_id = 0; inp_id < num_input_tensors; inp_id++) {
                    const uint32_t inp_total_w = input_total_w[inp_id];
                    const uint32_t inp_shard_w_val = input_shard_w[inp_id];

                    const uint32_t overlap_start = std::max(out_col_start, cum_w);
                    const uint32_t overlap_end = std::min(out_col_end, cum_w + inp_total_w);

                    if (overlap_start < overlap_end) {
                        const uint32_t local_col_start = overlap_start - cum_w;
                        const uint32_t local_col_end = overlap_end - cum_w;

                        const uint32_t first_src_c = local_col_start / inp_shard_w_val;
                        const uint32_t last_src_c = (local_col_end - 1) / inp_shard_w_val;

                        for (uint32_t src_c = first_src_c; src_c <= last_src_c; src_c++) {
                            const uint32_t src_shard_col_start = src_c * inp_shard_w_val;
                            const uint32_t needed_col_start = std::max(local_col_start, src_shard_col_start);
                            const uint32_t needed_col_end =
                                std::min(local_col_end, src_shard_col_start + inp_shard_w_val);
                            const uint32_t needed_cols = needed_col_end - needed_col_start;

                            const uint32_t src_col_off = needed_col_start - src_shard_col_start;
                            const uint32_t global_col = cum_w + needed_col_start;
                            const uint32_t dst_col_off = global_col - out_col_start;

                            const uint32_t src_stride_bytes = to_units_w(inp_shard_w_val) * unit_size;

                            transfers.push_back(TransferDesc{
                                .input_index = inp_id,
                                .shard_id = input_shard_id(inp_id, sh, src_c),
                                .src_offset = to_units_w(src_col_off) * unit_size,
                                .src_stride = src_stride_bytes,
                                .dst_offset = to_units_w(dst_col_off) * unit_size,
                                .dst_stride = dst_stride_bytes,
                                .copy_size = to_units_w(needed_cols) * unit_size,
                                .num_rows = num_rows_units,
                            });
                        }
                    }
                    cum_w += inp_total_w;
                }
            } else {
                // Height concat.
                //
                // The output rows for leading index blk are input 0's block-blk rows, then input
                // 1's, and so on, so the walk is over (blk, inp_id) segments in output order. One
                // segment per input would append whole flattened heights, which matches torch only
                // when every dim before rank-2 is 1 (#55342).
                //
                // A segment's output rows are a run of the *output* height; its source rows are a
                // run of input inp_id's flattened height starting at blk * input_block_h[inp_id].
                // Those source rows can straddle grid rows, which the src_r loop already splits.
                // Clipped to the real height: the last height shard can extend past it, and
                // those rows are padding with no source row to copy from.
                const uint32_t out_row_start = sh * output_shard_h;
                const uint32_t out_row_end = std::min(out_row_start + output_shard_h, out_total_h);
                const uint32_t copy_width = to_units_w(output_shard_w) * unit_size;

                for (uint32_t blk = 0; blk < num_blocks; blk++) {
                    uint32_t cum_h = blk * output_block_h;
                    for (uint32_t inp_id = 0; inp_id < num_input_tensors; inp_id++) {
                        const uint32_t block_h = input_block_h[inp_id];
                        const uint32_t inp_shard_h_val = input_shard_h[inp_id];

                        const uint32_t overlap_start = std::max(out_row_start, cum_h);
                        const uint32_t overlap_end = std::min(out_row_end, cum_h + block_h);

                        if (overlap_start < overlap_end) {
                            // Source rows, in input inp_id's flattened height.
                            const uint32_t src_block_base = blk * block_h;
                            const uint32_t flat_row_start = src_block_base + (overlap_start - cum_h);
                            const uint32_t flat_row_end = src_block_base + (overlap_end - cum_h);

                            const uint32_t first_src_r = flat_row_start / inp_shard_h_val;
                            const uint32_t last_src_r = (flat_row_end - 1) / inp_shard_h_val;

                            for (uint32_t src_r = first_src_r; src_r <= last_src_r; src_r++) {
                                const uint32_t src_shard_row_start = src_r * inp_shard_h_val;
                                const uint32_t needed_row_start = std::max(flat_row_start, src_shard_row_start);
                                const uint32_t needed_row_end =
                                    std::min(flat_row_end, src_shard_row_start + inp_shard_h_val);
                                const uint32_t needed_rows = needed_row_end - needed_row_start;

                                const uint32_t src_row_off = needed_row_start - src_shard_row_start;
                                const uint32_t src_stride_bytes = to_units_w(input_shard_w[inp_id]) * unit_size;
                                const uint32_t src_offset = to_units_h(src_row_off) * src_stride_bytes;

                                // Where this piece lands, as an output row, then made core-local.
                                const uint32_t dst_row = cum_h + (needed_row_start - src_block_base);

                                transfers.push_back(TransferDesc{
                                    .input_index = inp_id,
                                    .shard_id = input_shard_id(inp_id, src_r, sw),
                                    .src_offset = src_offset,
                                    .src_stride = src_stride_bytes,
                                    .dst_offset = to_units_h(dst_row - out_row_start) * dst_stride_bytes,
                                    .dst_stride = dst_stride_bytes,
                                    .copy_size = copy_width,
                                    .num_rows = to_units_h(needed_rows),
                                });
                            }
                        }
                        cum_h += block_h;
                    }
                }
            }

            // Each core writes its *local* output shard, so the core this loop calls (gx, gy) has to
            // be the one that holds output shard (sh, sw). A core with nothing to write holds no
            // output data and is not checked.
            if (!transfers.empty()) {
                const uint32_t output_shard_id = (sh * output_shards_per_row) + sw;
                const auto& output_cores = output_dist_spec.cores();
                TT_FATAL(
                    output_shard_id < output_cores.size() && output_cores[output_shard_id] == logical_core,
                    "Block-sharded concat: output shard ({}, {}) is not on core {}, which writes it as its local "
                    "shard",
                    sh,
                    sw,
                    logical_core.str());
            }
        }
    }

    // Find max transfers across all cores (should be uniform for equal-shaped inputs)
    uint32_t max_num_transfers = 0;
    for (const auto& t : all_transfers) {
        max_num_transfers = std::max(max_num_transfers, static_cast<uint32_t>(t.size()));
    }
    TT_FATAL(max_num_transfers > 0, "No transfers computed for block-sharded concat");

    constexpr uint32_t runtime_args_limit = 256;
    // One element of each of the eight per-copy arrays, plus the named num_copies.
    constexpr uint32_t args_per_transfer = 8;
    constexpr uint32_t args_overhead = 1;
    constexpr uint32_t max_transfers_per_risc = (runtime_args_limit - args_overhead) / args_per_transfer;
    const uint32_t reader_count_max = (max_num_transfers + 1) / 2;
    // Both concat directions share this budget but reach it differently, so each gets the count
    // that actually drives it and a remedy that helps.
    //
    // Height: a core walks only the blocks its *own* output rows cover, roughly
    // num_blocks / shard_grid_h, so adding grid rows lowers the per-core count and shrinking the
    // grid is what makes this fire. Measured, 2 inputs at 8 leading indices: 16 transfers per
    // core on one grid row, 8 on two, 4 on four, 2 on eight.
    //
    // Width: flat in the grid. output_shard_w is num_input_tensors * input_shard_w whatever the
    // grid width, so a core's columns always span about num_input_tensors input shards -- only
    // the input count moves it. With block-sharded concat capped at 16 inputs this branch cannot
    // actually reach the budget; it is kept so the failure is attributable if that cap changes.
    constexpr uint32_t max_transfers_per_core = max_transfers_per_risc * 2;
    if (width_concat) {
        TT_FATAL(
            reader_count_max <= max_transfers_per_risc,
            "Block-sharded width concat: {} transfers per core, past the {} the runtime-arg budget "
            "allows ({} per RISC at {} args each). A core's output columns span one transfer per "
            "input shard they cover, so the {} inputs drive this -- the grid width does not. "
            "Reduce the number of input tensors.",
            max_num_transfers,
            max_transfers_per_core,
            max_transfers_per_risc,
            args_per_transfer,
            num_input_tensors);
    } else {
        TT_FATAL(
            reader_count_max <= max_transfers_per_risc,
            "Block-sharded height concat: {} transfers per core, past the {} the runtime-arg "
            "budget allows ({} per RISC at {} args each). A core walks only the leading indices "
            "its own rows cover -- about {} / {} grid rows -- at {} inputs each, so *more* grid "
            "rows lowers this. Add grid rows, or reduce the leading dims or the input count.",
            max_num_transfers,
            max_transfers_per_core,
            max_transfers_per_risc,
            args_per_transfer,
            num_blocks,
            shard_grid_h,
            num_input_tensors);
    }

    // --- Kernels ---
    // Both RISCs run the same kernel but with different subsets of transfers,
    // doubling effective NOC bandwidth (reader uses NOC0, writer uses NOC1).
    //
    // Each core's transfers split reader-first, ceil(total / 2) to the reader. A per-copy array's
    // length is fixed per kernel, so each RISC's arrays are sized to its largest count over the
    // cores and shorter lists are padded; the kernel reads only num_copies entries. An array must
    // have at least one element, and the writer can have none on every core (one copy per core).
    const uint32_t reader_array_size = std::max(reader_count_max, 1u);
    const uint32_t writer_array_size = std::max(max_num_transfers - reader_count_max, 1u);

    // Program-scope resource names. Declared function-local (not at namespace scope) so that the
    // unity build, which concatenates every op's factory into one translation unit, cannot collide
    // these very generic identifiers across ops.
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const TensorParamName OUTPUT{"output"};
    std::vector<TensorParamName> INPUTS;
    INPUTS.reserve(num_input_tensors);
    for (uint32_t i = 0; i < num_input_tensors; ++i) {
        INPUTS.emplace_back(TensorParamName{"input_" + std::to_string(i)});
    }

    static constexpr const char* KERNEL_SOURCE =
        "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/"
        "reader_writer_block_sharded_concat.cpp";

    // Both instances bind every input and the output. Inputs are read from other cores by shard id;
    // the output is this core's own shard. The kernel picks an input by a per-copy index, so the
    // inputs are also bound positionally, as a sequence.
    Group<TensorBinding> tensor_bindings;
    std::vector<std::string> input_accessor_names;
    tensor_bindings.reserve(num_input_tensors + 1);
    input_accessor_names.reserve(num_input_tensors);
    for (uint32_t i = 0; i < num_input_tensors; ++i) {
        input_accessor_names.push_back("input_" + std::to_string(i));
        tensor_bindings.push_back(TensorBinding{
            .tensor_parameter_name = INPUTS[i],
            .accessor_name = input_accessor_names.back(),
        });
    }
    tensor_bindings.push_back(TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"});

    const KernelAdvancedOptions advanced_options{
        .tensor_binding_sequences =
            {
                KernelAdvancedOptions::TensorBindingSequence{
                    .sequence_name = "inputs",
                    .members = input_accessor_names,
                },
            },
    };

    static constexpr std::array<const char*, 8> COPY_ARRAYS = {
        "input_index", "shard_id", "src_offset", "src_stride", "dst_offset", "dst_stride", "copy_size", "num_rows"};
    auto make_schema = [&](uint32_t array_size) {
        KernelSpec::RuntimeArgSchema schema{.runtime_arg_names = {"num_copies"}};
        for (const char* name : COPY_ARRAYS) {
            schema.runtime_array_args.push_back({.name = name, .size = array_size});
        }
        return schema;
    };

    Group<KernelSpec> kernels;
    kernels.reserve(2);
    kernels.push_back(KernelSpec{
        .unique_id = READER,
        .source = KERNEL_SOURCE,
        .tensor_bindings = tensor_bindings,
        .runtime_arg_schema = make_schema(reader_array_size),
        .hw_config = ttnn::create_reader_datamovement_config(),
        .advanced_options = advanced_options,
    });
    kernels.push_back(KernelSpec{
        .unique_id = WRITER,
        .source = KERNEL_SOURCE,
        .tensor_bindings = std::move(tensor_bindings),
        .runtime_arg_schema = make_schema(writer_array_size),
        .hw_config = ttnn::create_writer_datamovement_config(),
        .advanced_options = advanced_options,
    });

    Group<TensorParameter> tensor_parameters;
    tensor_parameters.reserve(num_input_tensors + 1);
    for (uint32_t i = 0; i < num_input_tensors; ++i) {
        tensor_parameters.push_back(TensorParameter{
            .unique_id = INPUTS[i],
            .spec = input_tensors[i].mesh_tensor().tensor_spec(),
        });
    }
    tensor_parameters.push_back(TensorParameter{
        .unique_id = OUTPUT,
        .spec = output.mesh_tensor().tensor_spec(),
    });

    ProgramSpec spec{
        .name = "concat_block_sharded",
        .kernels = std::move(kernels),
        .tensor_parameters = std::move(tensor_parameters),
        .work_units =
            {
                WorkUnitSpec{
                    .name = "main",
                    .kernels = {READER, WRITER},
                    .target_nodes = all_cores,
                },
            },
    };

    // Writes transfers [start, start + count) into run_args for core, padded to array_size.
    auto set_copy_args = [](KernelRunArgs& run_args,
                            const CoreCoord& core,
                            const std::vector<TransferDesc>& descs,
                            uint32_t start,
                            uint32_t count,
                            uint32_t array_size) {
        run_args.runtime_arg_values["num_copies"][core] = count;
        std::array<std::vector<uint32_t>, 8> arrays;
        for (auto& array : arrays) {
            array.assign(array_size, 0);
        }
        for (uint32_t k = 0; k < count; k++) {
            const auto& td = descs[start + k];
            arrays[0][k] = td.input_index;
            arrays[1][k] = td.shard_id;
            arrays[2][k] = td.src_offset;
            arrays[3][k] = td.src_stride;
            arrays[4][k] = td.dst_offset;
            arrays[5][k] = td.dst_stride;
            arrays[6][k] = td.copy_size;
            arrays[7][k] = td.num_rows;
        }
        for (size_t a = 0; a < arrays.size(); a++) {
            run_args.runtime_array_arg_values[COPY_ARRAYS[a]][core] = std::move(arrays[a]);
        }
    };

    // --- Per-core runtime args ---
    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    for (uint32_t gy = 0; gy < grid_rows; gy++) {
        for (uint32_t gx = 0; gx < grid_cols; gx++) {
            const auto& transfers = all_transfers[gy * grid_cols + gx];
            const uint32_t total = static_cast<uint32_t>(transfers.size());
            const uint32_t reader_count = (total + 1) / 2;  // ceil(total / 2)
            const uint32_t writer_count = total - reader_count;

            const CoreCoord logical_core(start_x + gx, start_y + gy);
            set_copy_args(reader_run_args, logical_core, transfers, 0, reader_count, reader_array_size);
            set_copy_args(writer_run_args, logical_core, transfers, reader_count, writer_count, writer_array_size);
        }
    }

    ProgramRunArgs run_args;
    // push_back rather than brace-init: an initializer_list would copy the per-node arg tables.
    run_args.kernel_run_args.reserve(2);
    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));
    for (uint32_t i = 0; i < num_input_tensors; ++i) {
        run_args.tensor_args.emplace(INPUTS[i], input_tensors[i].mesh_tensor());
    }
    run_args.tensor_args.emplace(OUTPUT, output.mesh_tensor());

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::prim

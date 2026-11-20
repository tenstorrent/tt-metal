// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "concat_s2s_multi_program_factory.hpp"

#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/workload/run_args/program_run_args.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::prim {

namespace {

uint32_t find_greatest_common_page_size(std::vector<uint32_t>& stick_sizes, uint32_t alignment) {
    TT_FATAL(!stick_sizes.empty(), "Need at least one stick size to find page size");
    uint32_t page_size = tt::align(stick_sizes[0], alignment);
    for (size_t idx = 1; idx < stick_sizes.size(); idx++) {
        const uint32_t padded_stick_size = tt::align(stick_sizes[idx], alignment);
        page_size = std::gcd(page_size, padded_stick_size);
    }
    return page_size;
}

// How the two RISCs divide the leading blocks of a height concat.
struct RiscBlockPartition {
    bool split_blocks;
    uint32_t reader_block_count;
    uint32_t writer_block_start;
    uint32_t writer_block_count;
};

// Divide the work between the two RISCs on whichever axis balances better.
//
// Neither axis dominates. Splitting sticks within a block leaves one RISC with nothing when a
// block is one stick tall -- div_up(1, 2) is 1, so the writer gets 1 - 1 = 0 -- and skews for any
// odd stick count. Splitting blocks is just as lopsided for an odd block count whose sticks
// already halve evenly: 3 blocks of 4 sticks goes 8/4 by block but 6/6 by stick.
//
// So cost both and take the lower critical path, counting the busier RISC's sticks summed over
// the inputs. Each block's copy is independent, so any partition of blocks is correct, and the
// sticks within a block were already free to split.
RiscBlockPartition choose_risc_block_partition(
    uint32_t num_blocks, const std::vector<uint32_t>& input_num_sticks_per_block) {
    uint32_t total_sticks_per_block = 0;
    uint32_t halved_sticks_per_block = 0;
    for (const uint32_t sticks_per_block : input_num_sticks_per_block) {
        total_sticks_per_block += sticks_per_block;
        halved_sticks_per_block += tt::div_up(sticks_per_block, 2);
    }
    const uint32_t block_split_critical = tt::div_up(num_blocks, 2) * total_sticks_per_block;
    const uint32_t stick_split_critical = num_blocks * halved_sticks_per_block;
    const bool split_blocks = block_split_critical < stick_split_critical;
    const uint32_t reader_block_count = split_blocks ? tt::div_up(num_blocks, 2) : num_blocks;
    return {
        .split_blocks = split_blocks,
        .reader_block_count = reader_block_count,
        .writer_block_start = split_blocks ? reader_block_count : 0,
        .writer_block_count = num_blocks - (split_blocks ? reader_block_count : 0)};
}

}  // namespace

ttnn::device_operation::ProgramArtifacts ConcatS2SMultiProgramFactory::create_program_artifacts(
    const ConcatParams& operation_attributes, const ConcatInputs& tensor_args, Tensor& tensor_return_value) {
    using namespace tt::constants;
    using namespace tt::tt_metal;
    using tt::tt_metal::experimental::KernelAdvancedOptions;

    const auto& input_tensors = tensor_args.input_tensors;
    Tensor& output = tensor_return_value;
    const uint32_t rank = input_tensors[0].logical_shape().rank();
    const bool height_concat = is_height_concat(rank, operation_attributes.dim);
    // Height concat has to interleave per leading index (see num_leading_blocks). Width concat
    // does not: it appends along the stick, so one block is both correct and what the kernel did
    // before -- and for a height-sharded width concat a per-core block count is not even
    // well-defined, since a core's rows can straddle leading indices. Pinning it to 1 there keeps
    // width concat untouched by construction.
    const uint32_t num_blocks = height_concat ? num_leading_blocks(input_tensors[0]) : 1u;

    const uint32_t num_input_tensors = input_tensors.size();
    constexpr uint32_t max_input_tensors = 16;
    TT_FATAL(
        num_input_tensors <= max_input_tensors,
        "Sharded concat supports at most {} inputs, got {}.",
        max_input_tensors,
        num_input_tensors);
    const tt::DataFormat data_format = datatype_to_dataformat_converter(output.dtype());
    const bool rm_layout = output.layout() == Layout::ROW_MAJOR;

    // Program-scope resource names. Declared function-local (not at namespace scope) so that the
    // unity build, which concatenates every op's factory into one translation unit, cannot collide
    // these very generic identifiers across ops.
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const TensorParamName OUTPUT{"output"};
    // One TensorParameter per input; the count is an op attribute, so these are built rather than
    // declared.
    std::vector<TensorParamName> INPUTS;
    INPUTS.reserve(num_input_tensors);
    for (uint32_t i = 0; i < num_input_tensors; ++i) {
        INPUTS.emplace_back(TensorParamName{"input_" + std::to_string(i)});
    }

    // Assume inputs and output have the same element size and alignment.
    const uint32_t element_size = input_tensors[0].element_size();
    const uint32_t alignment = input_tensors[0].mesh_buffer().alignment();

    uint32_t page_size;
    uint32_t elements_per_page_width;
    uint32_t elements_per_page_height;
    if (rm_layout) {
        std::vector<uint32_t> all_stick_sizes;
        all_stick_sizes.reserve(input_tensors.size() + 1);
        all_stick_sizes.push_back(output.shard_spec().value().shape[1]);
        std::transform(
            input_tensors.begin(), input_tensors.end(), std::back_inserter(all_stick_sizes), [](const Tensor& tensor) {
                return tensor.element_size() * tensor.shard_spec().value().shape[1];
            });
        page_size = find_greatest_common_page_size(all_stick_sizes, alignment);
        elements_per_page_width = page_size / element_size;
        elements_per_page_height = 1;
    } else {
        page_size = tt::tile_size(data_format);
        elements_per_page_width = TILE_WIDTH;
        elements_per_page_height = TILE_HEIGHT;
    }

    std::vector<uint32_t> input_num_pages_per_stick;
    std::vector<uint32_t> input_num_sticks;
    std::vector<uint32_t> input_write_offsets;
    std::vector<uint32_t> input_num_sticks_per_block;
    std::vector<uint32_t> input_block_strides;
    input_num_pages_per_stick.reserve(num_input_tensors);
    input_num_sticks.reserve(num_input_tensors);
    input_write_offsets.reserve(num_input_tensors);
    input_num_sticks_per_block.reserve(num_input_tensors);
    input_block_strides.reserve(num_input_tensors);

    // Assume inputs and output have the same sharding grid.
    const auto all_cores = input_tensors[0].shard_spec().value().grid;

    // Per-input shard layout
    //
    // For height concat the write offset carries only this input's *prefix within one leading
    // block*: block b of input i starts b * output_block_stride further on, and the kernel adds
    // that term. Advancing by the whole shard instead lays the result out as
    // [all of input 0; all of input 1; ...] rather than interleaving per leading index (#55342).
    // At one block the two are the same expression, which is why width concat is unaffected.
    uint32_t curr_input_write_offset = 0;
    for (uint32_t input_id = 0; input_id < num_input_tensors; input_id++) {
        const auto shard_spec = input_tensors[input_id].shard_spec().value();
        input_num_pages_per_stick.push_back(tt::div_up(shard_spec.shape[1], elements_per_page_width));
        input_num_sticks.push_back(tt::div_up(shard_spec.shape[0], elements_per_page_height));
        input_write_offsets.push_back(curr_input_write_offset);

        // A width-sharded shard spans the whole flattened height, so every leading index
        // contributes the same number of rows. Anything else is a shard spec this factory cannot
        // describe as a strided copy. Vacuous at num_blocks == 1.
        TT_FATAL(
            input_num_sticks[input_id] % num_blocks == 0,
            "Height concat: input {} has {} shard rows, which does not divide into the {} leading "
            "indices of shape {}.",
            input_id,
            input_num_sticks[input_id],
            num_blocks,
            input_tensors[input_id].padded_shape());
        input_num_sticks_per_block.push_back(input_num_sticks[input_id] / num_blocks);
        input_block_strides.push_back(
            page_size * input_num_pages_per_stick[input_id] * input_num_sticks_per_block[input_id]);

        // Height concat: this input's rows within one block. Width concat: one stick.
        // input_block_strides is page_size * input_num_pages when num_blocks == 1.
        curr_input_write_offset +=
            height_concat ? input_block_strides[input_id] : page_size * input_num_pages_per_stick[input_id];
    }

    // Output shard layout
    const auto output_shard_spec = output.shard_spec().value();
    const uint32_t output_num_pages_per_stick = tt::div_up(output_shard_spec.shape[1], elements_per_page_width);
    const uint32_t output_num_sticks = tt::div_up(output_shard_spec.shape[0], elements_per_page_height);

    const uint32_t output_stride = page_size * output_num_pages_per_stick;

    TT_FATAL(
        output_num_sticks % num_blocks == 0,
        "Height concat: output has {} shard rows, which does not divide into the {} leading "
        "indices of shape {}.",
        output_num_sticks,
        num_blocks,
        output.padded_shape());
    // The write offsets above were accumulated in units of each input's own pages-per-stick, but
    // they index the output shard, so for height concat the two have to agree. They do: the inputs
    // differ only in the concat dim, which is not the width. Checked rather than assumed, because
    // a mismatch would skew every row silently.
    if (height_concat) {
        for (uint32_t input_id = 0; input_id < num_input_tensors; input_id++) {
            TT_FATAL(
                input_num_pages_per_stick[input_id] == output_num_pages_per_stick,
                "Height concat: input {} is {} pages wide but the output is {}; height concat "
                "requires equal widths.",
                input_id,
                input_num_pages_per_stick[input_id],
                output_num_pages_per_stick);
        }
    }
    // Rows one leading index contributes to the output shard, in bytes.
    const uint32_t output_block_stride = output_stride * (output_num_sticks / num_blocks);
    // Ties that back to the offsets accumulated above. curr_input_write_offset ended at
    // sum(input_block_strides) -- the bytes one leading index actually fills -- and the kernel
    // then steps output_block_stride to reach the next. If the output shard were taller than the
    // inputs' combined per-block rows, the blocks would be spaced further apart than they were
    // filled and every block after the first would leave unwritten rows inside the tensor's real
    // height, read back as garbage rather than as a failure.
    if (height_concat) {
        TT_FATAL(
            curr_input_write_offset == output_block_stride,
            "Height concat: inputs fill {} bytes per leading index but the output shard spaces "
            "them {} apart ({} rows over {} indices), which would leave gaps inside the tensor.",
            curr_input_write_offset,
            output_block_stride,
            output_num_sticks,
            num_blocks);
    }

    const auto [split_blocks, reader_block_count, writer_block_start, writer_block_count] =
        choose_risc_block_partition(num_blocks, input_num_sticks_per_block);

    // Each kernel carries its own block range as compile-time args so its loop bound is constant.
    const KernelSpec::CompileTimeArgs reader_compile_time_args = {
        {"page_size", page_size},
        {"output_stride", output_stride},
        {"output_block_stride", output_block_stride},
        {"block_start", 0},
        {"block_count", reader_block_count},
    };
    const KernelSpec::CompileTimeArgs writer_compile_time_args = {
        {"page_size", page_size},
        {"output_stride", output_stride},
        {"output_block_stride", output_block_stride},
        {"block_start", writer_block_start},
        {"block_count", writer_block_count},
    };

    std::vector<uint32_t> reader_num_sticks;
    std::vector<uint32_t> reader_write_offsets;
    std::vector<uint32_t> reader_read_offsets;
    std::vector<uint32_t> writer_num_sticks;
    std::vector<uint32_t> writer_write_offsets;
    std::vector<uint32_t> writer_read_offsets;
    reader_num_sticks.reserve(num_input_tensors);
    reader_write_offsets.reserve(num_input_tensors);
    reader_read_offsets.reserve(num_input_tensors);
    writer_num_sticks.reserve(num_input_tensors);
    writer_write_offsets.reserve(num_input_tensors);
    writer_read_offsets.reserve(num_input_tensors);
    for (uint32_t input_id = 0; input_id < num_input_tensors; input_id++) {
        const uint32_t sticks_per_block = input_num_sticks_per_block[input_id];
        // Under a block split each RISC copies every stick of the blocks it owns. Otherwise both
        // RISCs walk every block -- however many there are -- and halve each block's sticks, with
        // the writer starting where the reader stops.
        const uint32_t reader_sticks = split_blocks ? sticks_per_block : tt::div_up(sticks_per_block, 2);
        const uint32_t writer_sticks = split_blocks ? sticks_per_block : sticks_per_block - reader_sticks;
        const uint32_t writer_stick_offset = split_blocks ? 0 : reader_sticks;
        reader_num_sticks.push_back(reader_sticks);
        reader_write_offsets.push_back(input_write_offsets[input_id]);
        reader_read_offsets.push_back(0);
        writer_num_sticks.push_back(writer_sticks);
        writer_write_offsets.push_back(input_write_offsets[input_id] + (output_stride * writer_stick_offset));
        writer_read_offsets.push_back(page_size * input_num_pages_per_stick[input_id] * writer_stick_offset);
    }

    static constexpr const char* KERNEL_SOURCE =
        "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/reader_s2s_tensor_concat.cpp";

    // Both instances bind every input shard and the output shard, and reach them only as
    // node-local L1 (LocalTensorAccessor); nothing crosses cores. The kernel picks input i with its
    // loop counter, so the inputs are also bound positionally, as a sequence; the sequence carries
    // its own length, which is why no tensor-count argument is passed.
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

    // Five values per input, read by input index.
    const KernelSpec::RuntimeArgSchema runtime_arg_schema{
        .runtime_array_args =
            {
                {.name = "input_num_pages_per_stick", .size = num_input_tensors},
                {.name = "input_num_sticks", .size = num_input_tensors},
                {.name = "input_write_offset", .size = num_input_tensors},
                {.name = "input_read_offset", .size = num_input_tensors},
                {.name = "input_block_stride", .size = num_input_tensors},
            },
    };
    const KernelAdvancedOptions advanced_options{
        .tensor_binding_sequences =
            {
                KernelAdvancedOptions::TensorBindingSequence{
                    .sequence_name = "inputs",
                    .members = input_accessor_names,
                },
            },
    };

    Group<KernelSpec> kernels;
    kernels.reserve(2);
    kernels.push_back(KernelSpec{
        .unique_id = READER,
        .source = KERNEL_SOURCE,
        .tensor_bindings = tensor_bindings,
        .compile_time_args = reader_compile_time_args,
        .runtime_arg_schema = runtime_arg_schema,
        .hw_config = ttnn::create_reader_datamovement_config(),
        .advanced_options = advanced_options,
    });
    kernels.push_back(KernelSpec{
        .unique_id = WRITER,
        .source = KERNEL_SOURCE,
        .tensor_bindings = std::move(tensor_bindings),
        .compile_time_args = writer_compile_time_args,
        .runtime_arg_schema = runtime_arg_schema,
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
        .name = "concat_s2s_multi",
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

    // Per-node runtime arrays, even though every node gets identical values (the reader's and the
    // writer's differ). Switching to common runtime arrays is a candidate, but an earlier move to
    // common args caused unexplained precision failures in sharded concat, so it is left per-node.
    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    for (const auto& range : all_cores.ranges()) {
        for (const CoreCoord& core : range) {
            reader_run_args.runtime_array_arg_values["input_num_pages_per_stick"][core] = input_num_pages_per_stick;
            reader_run_args.runtime_array_arg_values["input_num_sticks"][core] = reader_num_sticks;
            reader_run_args.runtime_array_arg_values["input_write_offset"][core] = reader_write_offsets;
            reader_run_args.runtime_array_arg_values["input_read_offset"][core] = reader_read_offsets;
            reader_run_args.runtime_array_arg_values["input_block_stride"][core] = input_block_strides;
            writer_run_args.runtime_array_arg_values["input_num_pages_per_stick"][core] = input_num_pages_per_stick;
            writer_run_args.runtime_array_arg_values["input_num_sticks"][core] = writer_num_sticks;
            writer_run_args.runtime_array_arg_values["input_write_offset"][core] = writer_write_offsets;
            writer_run_args.runtime_array_arg_values["input_read_offset"][core] = writer_read_offsets;
            writer_run_args.runtime_array_arg_values["input_block_stride"][core] = input_block_strides;
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

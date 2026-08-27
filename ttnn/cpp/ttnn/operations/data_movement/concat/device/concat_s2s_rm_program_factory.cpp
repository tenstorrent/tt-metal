// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "concat_s2s_rm_program_factory.hpp"

#include <algorithm>
#include <functional>
#include <string>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::prim {

template <typename T>
static std::pair<std::vector<T>, std::vector<T>> split(std::vector<T> input, std::size_t index) {
    if (index > input.size()) {
        throw std::out_of_range{"split index out of range"};
    }
    std::vector<T> second{std::make_move_iterator(input.begin() + index), std::make_move_iterator(input.end())};
    input.erase(input.begin() + index, input.end());
    return {std::move(input), std::move(second)};
}

static CoreRangeSet cores_to_corerangeset(const std::vector<CoreCoord>& cores) {
    std::vector<CoreRange> core_ranges;
    core_ranges.reserve(cores.size());
    std::transform(cores.begin(), cores.end(), std::back_inserter(core_ranges), [](const CoreCoord& core) {
        return CoreRange(core);
    });
    return CoreRangeSet(core_ranges);
}

ttnn::device_operation::ProgramArtifacts ConcatS2SRMProgramFactory::create_program_artifacts(
    const ConcatParams& operation_attributes, const ConcatInputs& tensor_args, Tensor& tensor_return_value) {
    using namespace tt::constants;
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::experimental;

    const uint32_t groups = static_cast<uint32_t>(operation_attributes.groups);

    // Work against the Metalium device-tensor type throughout; the TTNN wrappers only carry us in.
    const auto& output = tensor_return_value.mesh_tensor();
    const uint32_t num_input_tensors = tensor_args.input_tensors.size();
    std::vector<std::reference_wrapper<const MeshTensor>> inputs;
    inputs.reserve(num_input_tensors);
    for (const auto& input_tensor : tensor_args.input_tensors) {
        inputs.emplace_back(input_tensor.mesh_tensor());
    }

    // Program-scope resource names. Declared function-local (not at namespace scope) so that the
    // unity build, which concatenates every op's factory into one translation unit, cannot collide
    // these very generic identifiers across ops.
    const DFBSpecName INPUT_0_DFB{"s2s_rm_input_0"};
    const DFBSpecName INPUT_1_DFB{"s2s_rm_input_1"};
    const DFBSpecName OUTPUT_DFB{"s2s_rm_output"};
    const TensorParamName INPUT_0{"input_0"};
    const TensorParamName INPUT_1{"input_1"};
    const TensorParamName OUTPUT{"output"};

    const auto& device = output.device();

    const uint32_t num_output_rows = output.padded_shape()[-2];
    const tt::DataFormat dfb_data_format = datatype_to_dataformat_converter(output.dtype());
    const CoreRangeSet all_cores = inputs[0].get().shard_spec().value().grid;

    // Each input's DFB borrows the input tensor's own shard memory: the kernel reaches tensor data
    // through the buffer's read pointer, so the DFB *is* the tensor access.
    const std::array<DFBSpecName, 2> input_dfb_names = {INPUT_0_DFB, INPUT_1_DFB};
    const std::array<TensorParamName, 2> input_param_names = {INPUT_0, INPUT_1};

    Group<DataflowBufferSpec> dataflow_buffers;
    dataflow_buffers.reserve(num_input_tensors + 1);
    for (uint32_t input_id = 0; input_id < num_input_tensors; input_id++) {
        constexpr uint32_t num_input_num_units_per_shard_width = 1;
        const ShardSpec shard_spec = inputs[input_id].get().shard_spec().value();
        const uint32_t num_input_num_units_per_shard_height = shard_spec.shape[0];
        const uint32_t num_input_units = num_input_num_units_per_shard_height * num_input_num_units_per_shard_width;
        const uint32_t input_unit_size = shard_spec.shape[1] * inputs[input_id].get().element_size();
        const uint32_t input_page_size = round_up_to_mul32(input_unit_size);

        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = input_dfb_names[input_id],
            .entry_size = input_page_size,
            .num_entries = num_input_units,
            .data_format_metadata = dfb_data_format,
            .borrowed_from = input_param_names[input_id],
        });
    }

    const uint32_t num_output_units = output.shard_spec().value().shape[0];
    const uint32_t output_unit_size = output.shard_spec().value().shape[1] * output.element_size();
    const uint32_t output_page_size = round_up_to_mul32(output_unit_size);
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = OUTPUT_DFB,
        .entry_size = output_page_size,
        .num_entries = num_output_units,
        .data_format_metadata = dfb_data_format,
        .borrowed_from = OUTPUT,
    });

    const ShardSpec output_shard_spec = output.shard_spec().value();
    const uint32_t output_stick_size = output_shard_spec.shape[1] * output.element_size();

    const ShardSpec input_0_shard_spec = inputs[0].get().shard_spec().value();
    const ShardSpec input_1_shard_spec = inputs[1].get().shard_spec().value();
    const uint32_t input_0_stick_size = input_0_shard_spec.shape[1] * inputs[0].get().element_size();
    const uint32_t input_1_stick_size = input_1_shard_spec.shape[1] * inputs[1].get().element_size();
    const uint32_t input_0_stride = output_stick_size - input_0_stick_size;
    const uint32_t input_1_stride = output_stick_size - input_1_stick_size;
    const uint32_t num_output_rows_per_core = inputs[0].get().shard_spec().value().shape[0];
    const uint32_t num_pages_per_risc = tt::div_up(num_output_rows_per_core, 2);

    const uint32_t num_output_rows_per_core_last = num_output_rows % num_output_rows_per_core;
    const uint32_t num_pages_per_risc_last = tt::div_up(num_output_rows_per_core_last, 2);

    // The four argument sets mirror the legacy positional lists one-for-one, so their values stay
    // diffable against the pre-port factory. The three buffer indices legacy carried in slots 0, 12
    // and 13 are gone — they are DFB bindings now.
    const KernelSpec::CompileTimeArgs compile_time_args_0 = {
        {"input_stick_size_0", input_0_stick_size},
        {"input_stick_size_1", input_1_stick_size},
        {"input_stride_0", input_0_stride},
        {"input_stride_1", input_1_stride},
        {"num_output_pages", num_output_rows_per_core * num_input_tensors},
        {"page_start", 0},
        {"page_end", num_pages_per_risc},
        {"output_stick_offset", 0},
        {"input_start_0", 0},
        {"input_start_1", 0},
        {"groups", groups},
    };
    const KernelSpec::CompileTimeArgs compile_time_args_1 = {
        {"input_stick_size_0", input_0_stick_size},
        {"input_stick_size_1", input_1_stick_size},
        {"input_stride_0", input_0_stride},
        {"input_stride_1", input_1_stride},
        {"num_output_pages", num_output_rows_per_core * num_input_tensors},
        {"page_start", num_pages_per_risc},
        {"page_end", num_output_rows_per_core},
        {"output_stick_offset", num_pages_per_risc * output_stick_size},
        {"input_start_0", num_pages_per_risc * input_0_stick_size},
        {"input_start_1", num_pages_per_risc * input_1_stick_size},
        {"groups", groups},
    };

    const KernelSpec::CompileTimeArgs compile_time_args_0_last = {
        {"input_stick_size_0", input_0_stick_size},
        {"input_stick_size_1", input_1_stick_size},
        {"input_stride_0", input_0_stride},
        {"input_stride_1", input_1_stride},
        {"num_output_pages", num_output_rows_per_core_last * num_input_tensors},
        {"page_start", 0},
        {"page_end", num_pages_per_risc_last},
        {"output_stick_offset", 0},
        {"input_start_0", 0},
        {"input_start_1", 0},
        {"groups", groups},
    };
    const KernelSpec::CompileTimeArgs compile_time_args_1_last = {
        {"input_stick_size_0", input_0_stick_size},
        {"input_stick_size_1", input_1_stick_size},
        {"input_stride_0", input_0_stride},
        {"input_stride_1", input_1_stride},
        {"num_output_pages", num_output_rows_per_core_last * num_input_tensors},
        {"page_start", num_pages_per_risc_last},
        {"page_end", num_output_rows_per_core_last},
        {"output_stick_offset", num_pages_per_risc_last * output_stick_size},
        {"input_start_0", num_pages_per_risc_last * input_0_stick_size},
        {"input_start_1", num_pages_per_risc_last * input_1_stick_size},
        {"groups", groups},
    };

    static constexpr const char* KERNEL_SOURCE =
        "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/"
        "reader_height_sharded_width_concat_two_tensors.cpp";

    // Both instances of the one source touch all three DFBs, and every touch is a raw read/write
    // pointer peek — the kernel contains no reserve_back / push_back / wait_front / pop_front at
    // all. Two role-free touchers per node is exactly enough to fill the validator's one-producer,
    // one-consumer requirement, so bind one instance producer and the other consumer. The roles
    // drive FIFO machinery this kernel never invokes, so on Gen1 the labels are cosmetic and the
    // kernel code is untouched by the choice.
    const auto make_dfb_bindings = [&](DFBEndpointType endpoint_type) {
        return Group<DFBBinding>{
            DFBBinding{
                .dfb_spec_name = OUTPUT_DFB,
                .accessor_name = "output",
                .endpoint_type = endpoint_type,
            },
            DFBBinding{
                .dfb_spec_name = INPUT_0_DFB,
                .accessor_name = "input_0",
                .endpoint_type = endpoint_type,
            },
            DFBBinding{
                .dfb_spec_name = INPUT_1_DFB,
                .accessor_name = "input_1",
                .endpoint_type = endpoint_type,
            },
        };
    };

    Group<KernelSpec> kernels;
    Group<WorkUnitSpec> work_units;

    const auto append_reader_writer_pair = [&](const std::string& suffix,
                                               const CoreRangeSet& core_ranges,
                                               const KernelSpec::CompileTimeArgs& reader_cta,
                                               const KernelSpec::CompileTimeArgs& writer_cta) {
        const KernelSpecName reader_name{"reader" + suffix};
        const KernelSpecName writer_name{"writer" + suffix};

        kernels.push_back(KernelSpec{
            .unique_id = reader_name,
            .source = KERNEL_SOURCE,
            .dfb_bindings = make_dfb_bindings(DFBEndpointType::PRODUCER),
            .compile_time_args = reader_cta,
            .hw_config = ttnn::create_reader_datamovement_config(device.arch()),
        });
        kernels.push_back(KernelSpec{
            .unique_id = writer_name,
            .source = KERNEL_SOURCE,
            .dfb_bindings = make_dfb_bindings(DFBEndpointType::CONSUMER),
            .compile_time_args = writer_cta,
            .hw_config = ttnn::create_writer_datamovement_config(device.arch()),
        });
        work_units.push_back(WorkUnitSpec{
            .name = "main" + suffix,
            .kernels = {reader_name, writer_name},
            .target_nodes = core_ranges,
        });
    };

    if (num_output_rows_per_core_last > 0) {
        // The per-core-group split stays a compile-time split: two work units over disjoint node
        // sets, each holding its own same-source pair with its own compile-time args. Demoting the
        // per-group bounds to runtime args would cost the kernel's compile-time loop unrolling.
        const bool rm_orientation = output_shard_spec.orientation == ShardOrientation::ROW_MAJOR;
        const std::vector<CoreCoord> cores = corerange_to_cores(all_cores, std::nullopt, rm_orientation);
        const auto [first, last] = split(cores, cores.size() - 1);
        const CoreRangeSet first_cores = cores_to_corerangeset(first);
        const CoreRangeSet last_cores = cores_to_corerangeset(last);
        append_reader_writer_pair("_first", first_cores, compile_time_args_0, compile_time_args_1);
        append_reader_writer_pair("_last", last_cores, compile_time_args_0_last, compile_time_args_1_last);
    } else {
        append_reader_writer_pair("", all_cores, compile_time_args_0, compile_time_args_1);
    }

    // No kernel builds a TensorAccessor here, so no tensor is bound on a KernelSpec. The three
    // parameters exist because the DFBs above borrow their memory, which is a use in its own right:
    // each DFB's L1 address resolves at run time from the matching tensor argument.
    ProgramSpec spec{
        .name = "concat_s2s_rm",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters =
            {
                TensorParameter{.unique_id = INPUT_0, .spec = inputs[0].get().tensor_spec()},
                TensorParameter{.unique_id = INPUT_1, .spec = inputs[1].get().tensor_spec()},
                TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()},
            },
        .work_units = std::move(work_units),
    };

    // The factory sets no runtime args at all, so ProgramRunArgs carries only the tensor arguments
    // the borrowed DFBs resolve their addresses from.
    ProgramRunArgs run_args;
    run_args.tensor_args.emplace(INPUT_0, inputs[0].get());
    run_args.tensor_args.emplace(INPUT_1, inputs[1].get());
    run_args.tensor_args.emplace(OUTPUT, output);

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::prim

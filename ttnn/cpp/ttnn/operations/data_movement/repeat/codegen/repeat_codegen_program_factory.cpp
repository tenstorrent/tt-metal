// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/repeat/codegen/repeat_codegen_program_factory.hpp"

#include <utility>
#include <vector>

#include <tt_stl/assert.hpp>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {

namespace {

// Pages a reader/writer moves per turn of its loop. The staging-buffer depth lives in the
// header as kRepeatCbDepth since repeat_codegen_supported.cpp's L1-capacity gate needs the
// same value.
constexpr uint32_t kReadBatch = 4;
constexpr uint32_t kWriteBatch = 4;

// SEQ_REPEAT, see common/kernels/codegen/sequencers.h.
constexpr uint32_t kSeqRepeat = 1;

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
    // row_wise=false (column-major core enumeration) to match the generator's
    // split_cores()/emit_per_core_rt(), which always calls ttnn.split_work_to_cores
    // and corerange_to_cores at their row_wise=False default. This is a no-op for
    // work counts that fill the whole grid, but for the small per-core-page RM
    // cases (a handful of sticks spread over a mostly-idle grid) the enumeration
    // order picks a different physical core per page range, which changes NOC
    // hop distance to the DRAM channel enough to show up as a device-time delta.
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
    if (split.core_group_1.contains(core)) {
        return split.work_per_core_1;
    }
    if (split.core_group_2.contains(core)) {
        return split.work_per_core_2;
    }
    return 0;
}

}  // namespace

ttnn::device_operation::ProgramArtifacts RepeatCodegenProgramFactory::create_program_artifacts(
    const RepeatCodegenParams& operation_attributes,
    const RepeatCodegenInputs& tensor_args,
    Tensor& tensor_return_value) {
    const Tensor& input = tensor_args.input;
    Tensor& output = tensor_return_value;
    Buffer* src_buffer = input.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_FATAL(src_buffer != nullptr, "RepeatCodegen input must be allocated on device!");
    TT_FATAL(dst_buffer != nullptr, "RepeatCodegen output must be allocated on device!");

    const bool is_row_major = input.layout() == ttnn::ROW_MAJOR_LAYOUT;
    const bool is_last_dim_rm = is_row_major && operation_attributes.rep_dim == 3;

    const CoreSplit split = split_work(input, operation_attributes.total_out_pages);
    tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());

    // Metal 2.0 named resource ids. Declared function-local so the unity build (this file shares a
    // translation unit with the repeat_interleave codegen factory) sees no duplicate
    // anonymous-namespace symbols.
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    // The one staging buffer every branch moves its pages through: the reader fills it (dfb::in on
    // the kernel side), the writer drains it (dfb::out). kRepeatCbDepth entries of one aligned page.
    const DFBSpecName PAGES{"pages"};
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};

    // Each branch binds its own kernel pair; the staging DFB, tensor parameters and work unit are
    // common and assembled once below.
    KernelSpec reader_spec;
    KernelSpec writer_spec;
    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    // Staging DFB entry size for the selected branch (the branch's aligned page).
    uint32_t page_size = 0;

    if (!is_row_major) {
        // TILE-interleaved path: shared pluggable sequencer reader (seq_id=1 == SEQ_REPEAT)
        // + interleaved writer, both bound through their Metal 2.0 forks in common/kernels/codegen.
        page_size = static_cast<uint32_t>(dst_buffer->aligned_page_size());

        reader_spec = KernelSpec{
            .unique_id = READER,
            .source =
                "ttnn/cpp/ttnn/operations/data_movement/common/kernels/codegen/"
                "reader_tile_interleaved_unified_metal2.cpp",
            .dfb_bindings = {DFBBinding{
                .dfb_spec_name = PAGES,
                .accessor_name = "in",
                .endpoint_type = DFBEndpointType::PRODUCER,
            }},
            .tensor_bindings = {TensorBinding{
                .tensor_parameter_name = INPUT,
                .accessor_name = "src",
            }},
            .compile_time_args =
                {{"seq_id", kSeqRepeat},
                 {"batch", kReadBatch},
                 // reader_tile_interleaved_unified_metal2.cpp reads this named arg unconditionally
                 // (not gated by seq_id): a non-zero value overrides the source page size used to
                 // bound each page transfer; 0 defers to the source binding's aligned page size.
                 // The repeat sequencer never needs the override, so it is supplied as 0.
                 {"src_page_pitch", 0u}},
            .runtime_arg_schema =
                {.runtime_arg_names = {"num_pages", "start_id", "num_repeats", "lower_pages", "rep_dim_pages"}},
            .hw_config = ttnn::create_reader_datamovement_config(),
        };

        writer_spec = KernelSpec{
            .unique_id = WRITER,
            .source = "ttnn/cpp/ttnn/operations/data_movement/common/kernels/codegen/writer_interleaved_metal2.cpp",
            .dfb_bindings = {DFBBinding{
                .dfb_spec_name = PAGES,
                .accessor_name = "out",
                .endpoint_type = DFBEndpointType::CONSUMER,
            }},
            .tensor_bindings = {TensorBinding{
                .tensor_parameter_name = OUTPUT,
                .accessor_name = "dst",
            }},
            .compile_time_args = {{"requested_write_size", page_size}, {"batch", kWriteBatch}},
            .runtime_arg_schema = {.runtime_arg_names = {"num_tiles", "start_id"}},
            .hw_config = ttnn::create_writer_datamovement_config(),
        };

        uint32_t start = 0;
        for (const auto& core : split.cores_in_order) {
            const uint32_t n = work_for_core(split, core);
            AddRuntimeArgsForNode(
                reader_run_args.runtime_arg_values,
                core,
                {{"num_pages", n},
                 {"start_id", start},
                 {"num_repeats", operation_attributes.num_repeats},
                 {"lower_pages", operation_attributes.lower_pages},
                 {"rep_dim_pages", operation_attributes.rep_dim_pages}});
            AddRuntimeArgsForNode(writer_run_args.runtime_arg_values, core, {{"num_tiles", n}, {"start_id", start}});
            start += n;
        }
    } else if (is_last_dim_rm) {
        // ROW_MAJOR last-dim (within-stick) path.
        const uint32_t in_stick_size = operation_attributes.stick_size;
        const uint32_t in_aligned = static_cast<uint32_t>(src_buffer->aligned_page_size());
        const uint32_t out_aligned = static_cast<uint32_t>(dst_buffer->aligned_page_size());
        page_size = out_aligned;

        reader_spec = KernelSpec{
            .unique_id = READER,
            .source = "ttnn/cpp/ttnn/operations/data_movement/repeat/codegen/kernels/reader_repeat_last_dim_rm.cpp",
            .dfb_bindings = {DFBBinding{
                .dfb_spec_name = PAGES,
                .accessor_name = "in",
                .endpoint_type = DFBEndpointType::PRODUCER,
            }},
            .tensor_bindings = {TensorBinding{
                .tensor_parameter_name = INPUT,
                .accessor_name = "src",
            }},
            .compile_time_args =
                {{"stick_size", in_stick_size},
                 {"in_read_size", in_aligned},
                 {"out_l1_stride", out_aligned},
                 {"num_repeats", operation_attributes.num_repeats},
                 {"batch", kReadBatch}},
            .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_page"}},
            .hw_config = ttnn::create_reader_datamovement_config(),
        };

        writer_spec = KernelSpec{
            .unique_id = WRITER,
            .source = "ttnn/cpp/ttnn/operations/data_movement/repeat/codegen/kernels/writer_repeat_rm.cpp",
            .dfb_bindings = {DFBBinding{
                .dfb_spec_name = PAGES,
                .accessor_name = "out",
                .endpoint_type = DFBEndpointType::CONSUMER,
            }},
            .tensor_bindings = {TensorBinding{
                .tensor_parameter_name = OUTPUT,
                .accessor_name = "dst",
            }},
            .compile_time_args = {{"xfer_size", out_aligned}, {"l1_stride", out_aligned}, {"batch", kWriteBatch}},
            .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
            .hw_config = ttnn::create_writer_datamovement_config(),
        };

        uint32_t start = 0;
        for (const auto& core : split.cores_in_order) {
            const uint32_t n = work_for_core(split, core);
            AddRuntimeArgsForNode(reader_run_args.runtime_arg_values, core, {{"num_pages", n}, {"start_page", start}});
            AddRuntimeArgsForNode(writer_run_args.runtime_arg_values, core, {{"num_pages", n}, {"start_id", start}});
            start += n;
        }
    } else {
        // ROW_MAJOR higher-dim path. Input and output share the same last-dim width on this branch (only a
        // non-last dim is repeated), so one aligned page pitch serves reader, writer, and the staging DFB.
        const uint32_t aligned_page_size = static_cast<uint32_t>(src_buffer->aligned_page_size());
        page_size = aligned_page_size;

        reader_spec = KernelSpec{
            .unique_id = READER,
            .source = "ttnn/cpp/ttnn/operations/data_movement/repeat/codegen/kernels/reader_repeat_higherdim_rm.cpp",
            .dfb_bindings = {DFBBinding{
                .dfb_spec_name = PAGES,
                .accessor_name = "in",
                .endpoint_type = DFBEndpointType::PRODUCER,
            }},
            .tensor_bindings = {TensorBinding{
                .tensor_parameter_name = INPUT,
                .accessor_name = "src",
            }},
            .compile_time_args =
                {{"xfer_size", aligned_page_size},
                 {"l1_stride", aligned_page_size},
                 {"num_repeats", operation_attributes.num_repeats},
                 {"lower_pages", operation_attributes.lower_pages},
                 {"rep_dim_pages", operation_attributes.rep_dim_pages},
                 {"batch", kReadBatch}},
            .runtime_arg_schema = {.runtime_arg_names = {"num_out_pages", "out_start_page"}},
            .hw_config = ttnn::create_reader_datamovement_config(),
        };

        writer_spec = KernelSpec{
            .unique_id = WRITER,
            .source = "ttnn/cpp/ttnn/operations/data_movement/repeat/codegen/kernels/writer_repeat_rm.cpp",
            .dfb_bindings = {DFBBinding{
                .dfb_spec_name = PAGES,
                .accessor_name = "out",
                .endpoint_type = DFBEndpointType::CONSUMER,
            }},
            .tensor_bindings = {TensorBinding{
                .tensor_parameter_name = OUTPUT,
                .accessor_name = "dst",
            }},
            .compile_time_args =
                {{"xfer_size", aligned_page_size}, {"l1_stride", aligned_page_size}, {"batch", kWriteBatch}},
            .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
            .hw_config = ttnn::create_writer_datamovement_config(),
        };

        uint32_t start = 0;
        for (const auto& core : split.cores_in_order) {
            const uint32_t n = work_for_core(split, core);
            AddRuntimeArgsForNode(
                reader_run_args.runtime_arg_values, core, {{"num_out_pages", n}, {"out_start_page", start}});
            AddRuntimeArgsForNode(writer_run_args.runtime_arg_values, core, {{"num_pages", n}, {"start_id", start}});
            start += n;
        }
    }

    ProgramSpec spec{
        .name = "repeat_codegen",
        .kernels = {std::move(reader_spec), std::move(writer_spec)},
        .dataflow_buffers = {DataflowBufferSpec{
            .unique_id = PAGES,
            .entry_size = page_size,
            .num_entries = kRepeatCbDepth,
            .data_format_metadata = data_format,
        }},
        .tensor_parameters =
            {TensorParameter{
                 .unique_id = INPUT,
                 .spec = input.tensor_spec(),
             },
             TensorParameter{
                 .unique_id = OUTPUT,
                 .spec = output.tensor_spec(),
             }},
        .work_units = {WorkUnitSpec{
            .name = "repeat_codegen",
            .kernels = {READER, WRITER},
            .target_nodes = split.all_cores,
        }},
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    run_args.tensor_args = {{INPUT, input.mesh_tensor()}, {OUTPUT, output.mesh_tensor()}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim

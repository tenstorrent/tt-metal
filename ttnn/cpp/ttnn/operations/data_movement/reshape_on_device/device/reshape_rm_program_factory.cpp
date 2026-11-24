// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reshape_rm_program_factory.hpp"

#include <tt-metalium/data_formats/tile_constants.hpp>
#include <tt-metalium/discovery/hal.hpp>
#include <tt-metalium/workload/legacy/host_api.hpp>
#include <tt-metalium/utils/work_split.hpp>
#include <tt-metalium/workload/run_args/program_run_args.hpp>
#include <tt-metalium/workload/specification/dataflow_buffer_spec.hpp>
#include <tt-metalium/workload/specification/kernel_spec.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>
#include <tt-metalium/workload/specification/tensor_parameter.hpp>

#include "ttnn/operation.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/math.hpp"

namespace ttnn::prim {

using namespace tt::tt_metal;

ttnn::device_operation::ProgramArtifacts ReshapeRMProgramFactory::create_program_artifacts(
    const ttnn::prim::ReshapeOnDeviceParams& /*operation_attributes*/,
    const ttnn::prim::ReshapeOnDeviceInputs& tensor_args,
    ttnn::Tensor& output_tensor) {
    const auto& input_tensor = tensor_args.input_tensor;
    TT_FATAL(
        input_tensor.dtype() == output_tensor.dtype(),
        "Input tensor dtype ({}) must match output tensor dtype ({})",
        input_tensor.dtype(),
        output_tensor.dtype());

    MeshDevice* device = input_tensor.device();

    auto output_shape = output_tensor.padded_shape();
    const MeshTensor& input_mesh_tensor = input_tensor.mesh_tensor();
    const MeshTensor& output_mesh_tensor = output_tensor.mesh_tensor();

    tt::DataFormat dfb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());

    uint32_t num_old_sticks =
        input_tensor.padded_shape()[0] * input_tensor.padded_shape()[1] * input_tensor.padded_shape()[2];
    uint32_t num_new_sticks = output_shape[0] * output_shape[1] * output_shape[2];

    uint32_t old_stick_size = input_tensor.padded_shape()[3] * input_tensor.element_size();
    uint32_t new_stick_size = output_shape[3] * output_tensor.element_size();

    TT_FATAL(
        std::max(old_stick_size, new_stick_size) % std::min(old_stick_size, new_stick_size) == 0,
        "Last dimension of the old shape ({}) should be divisible by the last dimension of the new shape ({}) or vice "
        "versa",
        old_stick_size,
        new_stick_size);

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;
    CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});
    const CoreRangeSet total_core_ranges{total_cores};

    bool split_work_by_old_sticks = old_stick_size > new_stick_size;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_sticks_per_core_group_1, num_sticks_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(
            compute_with_storage_grid_size, split_work_by_old_sticks ? num_old_sticks : num_new_sticks);

    auto num_pages = num_sticks_per_core_group_1 > num_sticks_per_core_group_2 ? num_sticks_per_core_group_1
                                                                               : num_sticks_per_core_group_2;
    auto max_page_size = old_stick_size > new_stick_size ? old_stick_size : new_stick_size;
    auto page_size = new_stick_size;

    const DFBSpecName SRC0{"src0"};
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};

    // DFB sized for the largest page across the whole grid; entry_size set to new_stick_size
    // so the writer drains output-sized chunks.
    // The check above guarantees that the larger stick size is a multiple of the smaller one.
    // That makes the entry count below a whole number.
    DataflowBufferSpec src0_dfb{
        .unique_id = SRC0,
        .entry_size = page_size,
        .num_entries = (num_pages * max_page_size) / page_size,
        .data_format_metadata = dfb_data_format,
    };

    KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/reshape_on_device/device/kernels/dataflow/"
            "reader_unary_reshape_stick_layout_interleaved_multi_core.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = SRC0,
                    .accessor_name = "in0",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "src"},
            },
        .compile_time_args = {{"old_stick_size", old_stick_size}},
        .runtime_arg_schema =
            {
                .runtime_arg_names =
                    {"num_sticks_per_core_read", "num_read_per_barrier", "num_sticks_per_dfb_push", "start_id"},
            },
        // The implicit-sync opt-out is left at its default. On Quasar this reader may need it, because it writes
        // sticks at running offsets into the DFB between one reserve_back and one push_back. Each stick is part of
        // an entry or spans several entries, depending on the old and new stick sizes. The Quasar uplift decides.
        .hw_config = ttnn::create_reader_datamovement_config(),
    };

    KernelSpec writer{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/reshape_on_device/device/kernels/dataflow/"
            "writer_unary_reshape_stick_layout_interleaved_multi_core.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = SRC0,
                    .accessor_name = "out0",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "dst"},
            },
        .compile_time_args = {{"new_stick_size", new_stick_size}},
        .runtime_arg_schema =
            {
                .runtime_arg_names =
                    {"num_sticks_per_core_read", "num_read_per_barrier", "num_sticks_per_dfb_push", "start_id"},
            },
        // The implicit-sync opt-out is left at its default here too. This writer issues one NoC write per DFB entry
        // between one wait_front and one pop_front. The Quasar uplift decides whether it needs the opt-out.
        .hw_config = ttnn::create_writer_datamovement_config(),
    };

    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};

    constexpr uint32_t max_read_size = 2048;
    uint32_t curr_sticks_read = 0;
    uint32_t curr_sticks_write = 0;
    for (uint32_t i = 0; i < num_cores_total; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_new_sticks_per_core = 0;
        uint32_t num_old_sticks_per_core = 0;
        if (split_work_by_old_sticks) {
            if (core_group_1.contains(core)) {
                num_old_sticks_per_core = num_sticks_per_core_group_1;
            } else if (core_group_2.contains(core)) {
                num_old_sticks_per_core = num_sticks_per_core_group_2;
            }
        } else {
            if (core_group_1.contains(core)) {
                num_new_sticks_per_core = num_sticks_per_core_group_1;
            } else if (core_group_2.contains(core)) {
                num_new_sticks_per_core = num_sticks_per_core_group_2;
            }
        }

        uint32_t old_new_stick_size_ratio =
            old_stick_size > new_stick_size ? (old_stick_size / new_stick_size) : (new_stick_size / old_stick_size);
        if (split_work_by_old_sticks) {
            num_new_sticks_per_core = num_old_sticks_per_core * old_new_stick_size_ratio;
        } else {
            num_old_sticks_per_core = num_new_sticks_per_core * old_new_stick_size_ratio;
        }

        // issue more reads before calling barrier
        uint32_t num_old_sticks_per_core_read = 0;
        uint32_t num_old_sticks_read_per_barrier = 0;
        uint32_t num_old_sticks_per_dfb_push = 0;
        uint32_t num_new_sticks_per_core_read = 0;
        uint32_t num_new_sticks_read_per_barrier = 0;
        uint32_t num_new_sticks_per_dfb_push = 0;
        if (old_stick_size > new_stick_size) {
            if (num_old_sticks_per_core != 0) {
                num_old_sticks_per_core_read =
                    tt::tt_metal::merge_num_sticks_to_read(num_old_sticks_per_core, old_stick_size, max_read_size);
                num_old_sticks_read_per_barrier = num_old_sticks_per_core / num_old_sticks_per_core_read;
                num_old_sticks_per_dfb_push = num_old_sticks_read_per_barrier * old_new_stick_size_ratio;

                num_new_sticks_per_dfb_push = num_old_sticks_per_dfb_push;
                num_new_sticks_read_per_barrier = num_old_sticks_per_dfb_push;
                num_new_sticks_per_core_read = num_new_sticks_per_core / num_new_sticks_read_per_barrier;
            }
        } else {
            if (num_new_sticks_per_core != 0) {
                num_new_sticks_per_core_read =
                    tt::tt_metal::merge_num_sticks_to_read(num_new_sticks_per_core, new_stick_size, max_read_size);
                num_new_sticks_read_per_barrier = num_new_sticks_per_core / num_new_sticks_per_core_read;
                num_new_sticks_per_dfb_push = num_new_sticks_read_per_barrier;

                num_old_sticks_per_dfb_push = num_new_sticks_per_dfb_push;
                num_old_sticks_read_per_barrier = num_new_sticks_per_dfb_push * old_new_stick_size_ratio;
                num_old_sticks_per_core_read = num_old_sticks_per_core / num_old_sticks_read_per_barrier;
            }
        }

        // Per-core runtime args. Every core of the grid gets an entry. Cores with no sticks scheduled
        // get zero counts, and their kernels never enter the read or write loops.
        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {{"num_sticks_per_core_read", num_old_sticks_per_core_read},
             {"num_read_per_barrier", num_old_sticks_read_per_barrier},
             {"num_sticks_per_dfb_push", num_old_sticks_per_dfb_push},
             {"start_id", curr_sticks_read}});
        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {{"num_sticks_per_core_read", num_new_sticks_per_core_read},
             {"num_read_per_barrier", num_new_sticks_read_per_barrier},
             {"num_sticks_per_dfb_push", num_new_sticks_per_dfb_push},
             {"start_id", curr_sticks_write}});

        curr_sticks_read += num_old_sticks_per_core;
        curr_sticks_write += num_new_sticks_per_core;
    }

    ProgramSpec spec{
        .name = "reshape_on_device_rm",
        .kernels = {std::move(reader), std::move(writer)},
        .dataflow_buffers = {std::move(src0_dfb)},
        .tensor_parameters =
            {
                // The program cache key (compute_program_hash in reshape_op.cpp) pins the input's padded
                // shape, dtype, page config and memory config. It does not pin the input's logical shape or
                // its alignment. One cached program can therefore serve inputs that differ only in those two.
                // This factory reads only the padded shape. That makes the sharing safe.
                // Both relaxations are needed. Freeing the logical shape alone would still require equal
                // alignment, and the key does not pin that.
                // The output spec is computed from attributes that the key already hashes. It stays an exact
                // match.
                TensorParameter{
                    .unique_id = INPUT,
                    .spec = input_tensor.tensor_spec(),
                    .relaxations = {.match_padded_shape_only = true, .relax_tensor_layout_alignment = true},
                },
                TensorParameter{.unique_id = OUTPUT, .spec = output_tensor.tensor_spec()},
            },
        .work_units =
            {
                WorkUnitSpec{
                    .name = "main",
                    .kernels = {READER, WRITER},
                    .target_nodes = total_core_ranges,
                },
            },
    };

    ProgramRunArgs run_params;
    run_params.kernel_run_args.push_back(std::move(reader_run_args));
    run_params.kernel_run_args.push_back(std::move(writer_run_args));
    run_params.tensor_args.emplace(INPUT, TensorArgument{input_mesh_tensor});
    run_params.tensor_args.emplace(OUTPUT, TensorArgument{output_mesh_tensor});

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_params),
    };
}

}  // namespace ttnn::prim

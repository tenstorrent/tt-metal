// SPDX-FileCopyrightText: © 2023 BOS Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "concat_device_operation.hpp"

namespace ttnn {
namespace operations {
namespace test_ops {
    Concat::cached_program_t Concat::MultiCore::create(const operation_attributes_t &operation_attributes, const tensor_args_t &input_tensors,Tensor &output) {

        TT_FATAL(dim == 3, "Sharded concat RM only supports dim=3");
        // TT_FATAL(, "Only Height sharded tensors allowed")

        tt_metal::Program program = tt_metal::CreateProgram();

        tt_metal::Device *device = output.device();
        auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
        uint32_t num_cores_x = compute_with_storage_grid_size.x;
        uint32_t num_cores_y = compute_with_storage_grid_size.y;

        uint32_t num_output_rows = output.get_legacy_shape()[-2];
        uint32_t num_input_tensors = input_tensors.size();

        vector<CBHandle> cb_input(num_input_tensors);
        vector<uint32_t> input_num_units_per_shard_height(num_input_tensors);

        tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(output.get_dtype());
        auto all_cores = input_tensors[0].shard_spec().value().grid;

        vector<uint32_t> cb_ids(num_input_tensors);
        uint32_t input_unit_size = input_tensors[0].shard_spec().value().shape[1] * input_tensors[0].element_size();
        auto input_page_size = round_up_to_mul32(input_unit_size);

        for (uint32_t input_id = 0; input_id < num_input_tensors; input_id++) {
            auto shard_spec = input_tensors[input_id].shard_spec().value();
            input_num_units_per_shard_height[input_id] = shard_spec.shape[0];
            auto num_input_units = input_num_units_per_shard_height[input_id];
            tt_metal::CircularBufferConfig input_cb_config =
                tt_metal::CircularBufferConfig(num_input_units * input_page_size, {{input_id, cb_data_format}})
                    .set_page_size(input_id, input_page_size)
                    .set_globally_allocated_address(*input_tensors[input_id].buffer());
            cb_input[input_id] = tt_metal::CreateCircularBuffer(program, all_cores, input_cb_config);
            cb_ids[input_id] = input_id;
        }

        uint32_t cb_dst_id = 16;
        auto num_output_units = input_num_units_per_shard_height[0] * num_input_tensors;
        uint32_t intermed_cb_id = 8;
        auto output_page_size = input_page_size;
        tt_metal::CircularBufferConfig output_cb_config =
            tt_metal::CircularBufferConfig(num_output_units * output_page_size, {{cb_dst_id, cb_data_format}})
                .set_page_size(cb_dst_id, output_page_size)
                .set_globally_allocated_address(*output.buffer());
        auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, output_cb_config);

        auto output_shard_spec = output.shard_spec().value();

        tt_metal::CircularBufferConfig intermed_cb_config =
        tt_metal::CircularBufferConfig(num_output_units * output_page_size, {{intermed_cb_id, cb_data_format}})
                .set_page_size(intermed_cb_id, output_page_size);
        auto cb_intermed = tt_metal::CreateCircularBuffer(program, all_cores, intermed_cb_config);

        std::map<string, string> defines;
        std::vector <uint32_t> compile_time_args = {num_input_tensors, cb_dst_id};

        tt_metal::KernelHandle unary_reader_kernel_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/test_ops/Concat/kernels/dataflow/reader_height_sharded_width_concat.cpp",
            all_cores,
            tt_metal::ReaderDataMovementConfig(compile_time_args, defines)
        );
        tt_metal::KernelHandle unary_writer_kernel_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/test_ops/Concat/kernels/dataflow/reader_height_sharded_width_concat.cpp",
            all_cores,
            tt_metal::WriterDataMovementConfig(compile_time_args, defines)
        );

        bool row_wise = input_tensors[0].shard_spec().value().orientation == ShardOrientation::ROW_MAJOR;
        auto cores = corerange_to_cores(all_cores, std::nullopt, row_wise);
        auto input_cores = input_tensors[0].shard_spec().value().grid;
        uint32_t num_output_rows_per_core = div_up(num_output_rows, input_cores.num_cores());
        auto input0_shard_spec = input_tensors[0].shard_spec().value();
        uint32_t output_stick_size = output_shard_spec.shape[1] * output.element_size();;

        uint32_t core_id = 0;
        for (auto core : cores) {
            uint32_t curr_num_input_tensors;
            uint32_t curr_num_output_rows;
            if (input_cores.core_coord_in_core_ranges(core)) {
                curr_num_input_tensors = num_input_tensors;
                curr_num_output_rows = num_output_rows_per_core;
            } else {
                curr_num_input_tensors = 0;
                curr_num_output_rows = 0;
            }

            uint32_t page_id = div_up(curr_num_output_rows, 2);

            vector<uint32_t> runtime_args_nrisc = {0, curr_num_output_rows * num_input_tensors, 0, div_up(curr_num_output_rows, 2)};
            vector<uint32_t> runtime_args_brisc = {page_id * output_stick_size, curr_num_output_rows * num_input_tensors, div_up(curr_num_output_rows, 2), curr_num_output_rows};

            uint32_t start_offset = 0;
            for(uint32_t input_id = 0; input_id < num_input_tensors; input_id++) {
                auto input_i_shard_spec = input_tensors[input_id].shard_spec().value();
                TT_FATAL(
                    input_i_shard_spec.shape[0] == input0_shard_spec.shape[0],
                    "Input shard_spec mismatch (must match in y dim)",
                    input_id,
                    input_i_shard_spec,
                    input0_shard_spec
                );

                auto input_stick_size = input_i_shard_spec.shape[1] * input_tensors[input_id].element_size();

                runtime_args_nrisc.push_back(input_stick_size);
                runtime_args_nrisc.push_back(output_stick_size - input_stick_size);
                runtime_args_nrisc.push_back(0);

                runtime_args_brisc.push_back(input_stick_size);
                runtime_args_brisc.push_back(output_stick_size - input_stick_size);
                runtime_args_brisc.push_back(page_id*input_stick_size);
                
                start_offset += input_stick_size;
            }
            
            tt_metal::SetRuntimeArgs(
                program,
                unary_reader_kernel_id,
                core,
                runtime_args_nrisc
            );

            tt_metal::SetRuntimeArgs(
                program,
                unary_writer_kernel_id,
                core,
                runtime_args_brisc
            );
            core_id++;
        }

        return { std::move(program), {.unary_reader_kernel_id = unary_reader_kernel_id, .unary_writer_kernel_id = unary_writer_kernel_id, .num_input_tensors = num_input_tensors, .num_output_rows_per_core = num_output_rows_per_core, .output_stick_size = output_stick_size} };
    }

    void Concat::MultiCore::override_runtime_arguments(Concat::cached_program_t& cached_program, const operation_attributes_t &operation_attributes, const tensor_args_t &input_tensors, Tensor &output) {
        
        auto& program = cached_program.program;
        auto& unary_reader_kernel_id = cached_program.shared_variables.unary_reader_kernel_id;
        auto& unary_writer_kernel_id = cached_program.shared_variables.unary_writer_kernel_id;
        auto& num_input_tensors = cached_program.shared_variables.num_input_tensors;
        uint32_t output_stick_size = cached_program.shared_variables.output_stick_size;

        bool row_wise = input_tensors[0].shard_spec().value().orientation == ShardOrientation::ROW_MAJOR;
        auto all_cores = input_tensors[0].shard_spec().value().grid;
        auto cores = corerange_to_cores(all_cores, std::nullopt, row_wise);
        auto input_cores = input_tensors[0].shard_spec().value().grid;

        for (auto core : cores) {
            uint32_t curr_num_input_tensors;
            uint32_t curr_num_output_rows;
            if (input_cores.core_coord_in_core_ranges(core)) {
                curr_num_input_tensors = num_input_tensors;
                curr_num_output_rows = num_output_rows_per_core;
            } else {
                curr_num_input_tensors = 0;
                curr_num_output_rows = 0;
            }

            uint32_t page_id = div_up(curr_num_output_rows, 2);
            vector<uint32_t> runtime_args_nrisc = {0, curr_num_output_rows * num_input_tensors, 0, div_up(curr_num_output_rows, 2)};
            vector<uint32_t> runtime_args_brisc = {page_id * output_stick_size, curr_num_output_rows * num_input_tensors, div_up(curr_num_output_rows, 2), curr_num_output_rows};
            
            uint32_t start_offset = 0;
            for(uint32_t input_id = 0; input_id < num_input_tensors; input_id++) {
                auto input_i_shard_spec = input_tensors[input_id].shard_spec().value();
                auto input_stick_size = input_i_shard_spec.shape[1] * input_tensors[input_id].element_size();

                runtime_args_nrisc.push_back(input_stick_size);
                runtime_args_nrisc.push_back(output_stick_size - input_stick_size);
                runtime_args_nrisc.push_back(0);

                runtime_args_brisc.push_back(input_stick_size);
                runtime_args_brisc.push_back(output_stick_size - input_stick_size);
                runtime_args_brisc.push_back(page_id*input_stick_size);
                
                start_offset += input_stick_size;
            }

            tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, runtime_args_nrisc);
            tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, runtime_args_brisc);
        }

    }



}
}
}

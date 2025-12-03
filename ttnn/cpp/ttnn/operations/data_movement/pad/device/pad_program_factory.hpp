// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/host_api.hpp>

#include "ttnn/device_operation.hpp"

namespace ttnn::operations::data_movement::pad::program {

struct PadSharedVariables {
    tt::tt_metal::KernelHandle unary_reader_kernel_id{};
    tt::tt_metal::KernelHandle unary_writer_kernel_id{};
};

struct PadProgramFactory {
    using shared_variables_t = PadSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static tt::tt_metal::operation::ProgramWithCallbacks pad_rm_reader_writer(
        const Tensor& a,
        Tensor& output,
        const ttnn::Shape& output_logical_shape,
        const ttnn::Shape& input_tensor_start,
        float pad_value);

    static tt::tt_metal::operation::ProgramWithCallbacks pad_tile(
        const Tensor& a,
        Tensor& output,
        const ttnn::Shape& output_padded_shape,
        const ttnn::Shape& input_tensor_start,
        float pad_value);

    static tt::tt_metal::operation::ProgramWithCallbacks pad_rm_reader_writer_multi_core(
        const Tensor& a,
        Tensor& output,
        const ttnn::Shape& output_padded_shape,
        const ttnn::Shape& input_tensor_start,
        float pad_value);

    static tt::tt_metal::operation::ProgramWithCallbacks pad_rm_reader_writer_multi_core_v2(
        const Tensor& a,
        Tensor& output,
        const ttnn::Shape& output_padded_shape,
        const ttnn::Shape& input_tensor_start,
        float pad_value);

    static tt::tt_metal::operation::ProgramWithCallbacks pad_rm_sharded_height_only(
        const Tensor& a,
        Tensor& output,
        const ttnn::Shape& output_padded_shape,
        const ttnn::Shape& input_tensor_start,
        float pad_value);

    static tt::tt_metal::operation::ProgramWithCallbacks pad_rm_sharded_width_only(
        const Tensor& a,
        Tensor& output,
        const ttnn::Shape& output_padded_shape,
        const ttnn::Shape& input_tensor_start,
        float pad_value);

    static tt::tt_metal::operation::ProgramWithCallbacks pad_tile_multicore(
        const Tensor& a,
        Tensor& output,
        const ttnn::Shape& output_padded_shape,
        const ttnn::Shape& input_tensor_start,
        float pad_value);
};

}  // namespace ttnn::operations::data_movement::pad::program

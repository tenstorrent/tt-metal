// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>

#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>
#include "ttnn/operation.hpp"

namespace ttnn::operations::experimental::detail {

using namespace tt::constants;

tt::tt_metal::operation::ProgramWithCallbacks plusone_single_core(
    const Tensor& input, const std::optional<CoreRangeSet>& sub_core_grids) {
    tt::tt_metal::Program program{};

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    uint32_t input_unit_size = input.element_size();

    CoreRangeSet all_cores = CoreRangeSet(std::vector{CoreRange({0, 0}, {0, 0})});
    uint32_t num_cores = 1;  // single-core

    if (sub_core_grids.has_value()) {
        all_cores = sub_core_grids.value();
        num_cores = all_cores.num_cores();
    }

    const auto& input_shape = input.padded_shape();
    uint32_t W = input_shape[-1];
    uint32_t H = 1;
    if (input_shape.size() > 1) {
        for (uint32_t i = 0; i < input_shape.size() - 1; ++i) {
            H *= input_shape[i];
        }
    }

    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t num_input_units = W;
    uint32_t aligned_input_unit_size = round_up_to_mul32(num_input_units * input_unit_size);
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(aligned_input_unit_size, {{src0_cb_index, input_cb_data_format}})
            .set_page_size(src0_cb_index, aligned_input_unit_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    auto src_buffer = input.buffer();
    bool src_is_dram = src_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;

    std::vector<uint32_t> reader_compile_time_args = {
        src0_cb_index,
        src_is_dram,
        aligned_input_unit_size,
        W,
        H,
    };

    std::map<std::string, std::string> kernel_defines;
    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/plusone/device/kernels/reader_plusone_interleaved.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args, kernel_defines));

    auto cores = corerange_to_cores(all_cores, num_cores, true);

    for (uint32_t i = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores.at(i);

        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, {src_buffer->address()});
    }

    auto override_runtime_args_callback = [reader_kernel_id, cores](
                                              const void* operation,
                                              tt::tt_metal::Program& program,
                                              const std::vector<Tensor>& input_tensors,
                                              const std::vector<std::optional<const Tensor>>& optional_tensors,
                                              const std::vector<Tensor>& output_tensors) {
        auto src_buffer = input_tensors.at(0).buffer();

        for (const auto& core : cores) {
            {
                auto& runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
                runtime_args[0] = src_buffer->address();
            }
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

}  // namespace ttnn::operations::experimental::detail

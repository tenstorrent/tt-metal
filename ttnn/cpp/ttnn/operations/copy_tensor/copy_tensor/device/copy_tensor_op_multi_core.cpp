// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/util.hpp>
#include "copy_tensor_op.hpp"
#include "ttnn/common/constants.hpp"

namespace ttnn::operations::copy_tensor {

using std::vector;
using namespace tt::constants;
using namespace tt::tt_metal;

operation::ProgramWithCallbacks copy_tensor_multi_core(const Tensor& src_tensor, const Tensor& dst_tensor) {
    /* Buffers */
    Buffer* src_buffer = src_tensor.buffer();
    Buffer* dst_buffer = dst_tensor.buffer();

    /* Tiles */
    tt::tt_metal::Tile tensor_tile = src_tensor.get_tensor_spec().tile();

    /* Dataformats */
    tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(src_tensor.get_dtype());

    Program program{};

    // In validate we make sure that all tensors are on the same device
    tt::tt_metal::IDevice* device = src_tensor.device();

    /* Cores setup */
    auto all_cores = src_buffer->shard_spec().grid();

    /* src cb setup */
    uint32_t src_cb_single_datum_size = tt::datum_size(data_format);
    uint32_t src_cb_size =
        src_buffer->shard_spec().shape()[0] * src_buffer->shard_spec().shape()[1] * src_cb_single_datum_size;

    uint32_t src_cb_index = tt::CBIndex::c_0;
    CircularBufferConfig src_cb_config = CircularBufferConfig(src_cb_size, {{src_cb_index, data_format}})
                                             .set_page_size(src_cb_index, src_cb_single_datum_size)
                                             .set_globally_allocated_address(*src_buffer);
    auto src_cb = CreateCircularBuffer(program, all_cores, src_cb_config);

    /* dst cb setup */
    uint32_t dst_cb_single_datum_size = src_cb_single_datum_size;
    uint32_t dst_cb_size = src_cb_size;
    uint32_t dst_cb_index = tt::CBIndex::c_1;
    CircularBufferConfig dst_cb_config = CircularBufferConfig(dst_cb_size, {{dst_cb_index, data_format}})
                                             .set_page_size(dst_cb_index, dst_cb_single_datum_size)
                                             .set_globally_allocated_address(*dst_buffer);
    auto dst_cb = CreateCircularBuffer(program, all_cores, dst_cb_config);

    /* Compile time args */
    std::vector<uint32_t> ct_args = {
        src_cb_size,
        src_cb_index,
        dst_cb_index,
    };

    auto copy_tensor_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/copy_tensor/copy_tensor/device/kernels/copy_tensor.cpp",
        all_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = ct_args});

    auto override_runtime_arguments_callback =
        [src_cb, dst_cb](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            auto src_buffer = input_tensors[0].buffer();
            auto dst_buffer = input_tensors[1].buffer();

            UpdateDynamicCircularBufferAddress(program, src_cb, *src_buffer);
            UpdateDynamicCircularBufferAddress(program, dst_cb, *dst_buffer);
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn::operations::copy_tensor

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/experimental/dataflow_buffer/dataflow_buffer.hpp>

namespace ttnn::prim::reduction_helpers {

namespace dfb = tt::tt_metal::experimental::dfb;

struct BufferIds {
    uint32_t input = 0;
    uint32_t scaler = 1;
    uint32_t output = 2;
    uint32_t acc = 3;
    uint32_t ineg = 4;
};

template <typename CoreSpec>
BufferIds create_reduction_buffers(
    tt::tt_metal::Program& program,
    const CoreSpec& cores,
    const dfb::DataflowBufferConfig& input_config,
    const dfb::DataflowBufferConfig& scaler_config,
    const dfb::DataflowBufferConfig& output_config,
    bool needs_negation = false,
    const dfb::DataflowBufferConfig& acc_config = {},
    const dfb::DataflowBufferConfig& ineg_config = {}) {
    BufferIds ids;
    ids.input = dfb::CreateDataflowBuffer(program, cores, input_config);
    ids.scaler = dfb::CreateDataflowBuffer(program, cores, scaler_config);
    ids.output = dfb::CreateDataflowBuffer(program, cores, output_config);

    if (needs_negation) {
        ids.acc = dfb::CreateDataflowBuffer(program, cores, acc_config);
        ids.ineg = dfb::CreateDataflowBuffer(program, cores, ineg_config);
    }

    return ids;
}

inline void bind_reduction_kernels(
    tt::tt_metal::Program& program,
    const BufferIds& ids,
    uint32_t reader_kernel,
    uint32_t compute_kernel,
    uint32_t writer_kernel,
    bool has_negation) {
    dfb::BindDataflowBufferToProducerConsumerKernels(program, ids.input, reader_kernel, compute_kernel);
    dfb::BindDataflowBufferToProducerConsumerKernels(program, ids.scaler, reader_kernel, compute_kernel);
    dfb::BindDataflowBufferToProducerConsumerKernels(program, ids.output, compute_kernel, writer_kernel);

    if (has_negation) {
        dfb::BindDataflowBufferToProducerConsumerKernels(program, ids.acc, compute_kernel, compute_kernel);
        dfb::BindDataflowBufferToProducerConsumerKernels(program, ids.ineg, compute_kernel, compute_kernel);
    }
}

}  // namespace ttnn::prim::reduction_helpers

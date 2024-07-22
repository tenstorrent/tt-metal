
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/experimental/queue/queue.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/run_operation.hpp"

namespace tt::tt_metal
{
void EnqueueHostToDeviceTransfer(
    CommandQueue& q, Tensor& dst, const void* src, const std::optional<std::size_t> transfer_size) {
    memcpy(q, dst, src, transfer_size);
}

void EnqueueDeviceToHostTransfer(
    CommandQueue& q, Tensor& src, void* dst, const std::optional<std::size_t> transfer_size, size_t src_offset) {
    TT_ASSERT(src_offset == 0, "src_offset is not supported");
    memcpy(q, dst, src, transfer_size);
}

void QueueSynchronize(CommandQueue& q) { Finish(q); }

template<class OutputTensors=tt_metal::operation::Tensors>
OutputTensors EnqueueOperation(
    CommandQueue& queue,
    operation::DeviceOperation<OutputTensors>& devop,
    const tt_metal::operation::Tensors& input_tensors,
    const tt_metal::operation::OptionalConstTensors& optional_input_tensors,
    const tt_metal::operation::OptionalTensors& optional_output_tensors) {
    return operation::run(queue, devop, input_tensors, optional_input_tensors, optional_output_tensors);
}
}

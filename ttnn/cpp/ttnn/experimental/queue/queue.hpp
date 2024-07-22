// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>
#include "ttnn/experimental/tensor/tensor.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "ttnn/operation.hpp"

namespace tt::tt_metal {

class Event;

void EnqueueHostToDeviceTransfer(
    CommandQueue&, Tensor& dst, const void* src, const std::optional<std::size_t> transfer_size = std::nullopt);

void EnqueueDeviceToHostTransfer(
    CommandQueue&,
    Tensor& src,
    void* dst,
    const std::optional<std::size_t> transfer_size = std::nullopt,
    size_t src_offset = 0);

void EnqueueRecordEvent(CommandQueue&, std::shared_ptr<Event>);
void EnqueueWaitForEvent(CommandQueue&, std::shared_ptr<Event>);
void EventSynchronize(std::shared_ptr<Event>);
bool EventQuery(std::shared_ptr<Event>);
void QueueSynchronize(CommandQueue&);

template<class OutputTensors=tt_metal::operation::Tensors>
OutputTensors EnqueueOperation(
    CommandQueue&,
    operation::DeviceOperation<OutputTensors>&,
    const tt_metal::operation::Tensors& input_tensors,
    const tt_metal::operation::OptionalConstTensors& optional_input_tensors = {},
    const tt_metal::operation::Tensors& optional_output_tensors = {});
}

// Future APIs to be added later
// void EnqueueAllocateDeviceBuffer(CommandQueue&, DeviceBuffer&); // Part 2
// void EnqueueAllocateHostBuffer(CommandQueue&, HostBuffer&); // Part 2


// void EnqueueDeallocate(CommandQueue&, Tensor&); // Part 2
// void EnqueueReallocate(CommandQueue&, Tensor&); // TBD
// void EnqueueAllocateHostMemory(CommandQueue& , owned_buffer::Buffer& ); // TBD

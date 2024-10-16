// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ttnn::graph {
// Vertex struct
constexpr auto kNodeType = "node_type";
constexpr auto kCounter = "counter";
constexpr auto kConnections = "connections";
constexpr auto kParams = "params";
// params keys
constexpr auto kName = "name";
constexpr auto kInputs = "inputs";
constexpr auto kTensorId = "tensor_id";
constexpr auto kType = "type";
constexpr auto kAddress = "address";
constexpr auto kSize = "size";
constexpr auto kLayout = "layout";
constexpr auto kShape = "shape";

// node names
constexpr auto kNodeBuffer = "buffer";
constexpr auto kNodeBufferAllocate = "buffer_allocate";
constexpr auto kNodeBufferDeallocate = "buffer_deallocate";
constexpr auto kNodeTensor = "tensor";
constexpr auto kNodeCBAllocate = "circular_buffer_allocate";
constexpr auto kNodeCBDeallocateAll = "circular_buffer_deallocate_all";
constexpr auto kNodeFunctionStart = "function_start";
constexpr auto kNodeFunctionEnd = "function_end";
constexpr auto kNodeCaptureStart = "capture_start";
constexpr auto kNodeCaptureEnd = "capture_end";
}  // namespace ttnn::graph

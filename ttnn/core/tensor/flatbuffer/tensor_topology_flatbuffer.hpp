// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor_topology_generated.h"

#include "ttnn/distributed/tensor_topology.hpp"

namespace ttnn {

// Converts TensorTopology to FlatBuffer representation
flatbuffers::Offset<ttnn::flatbuffer::TensorTopology> to_flatbuffer(
    const tt::tt_metal::TensorTopology& topology, flatbuffers::FlatBufferBuilder& builder);

// Converts FlatBuffer TensorTopology to C++ representation
tt::tt_metal::TensorTopology from_flatbuffer(const ttnn::flatbuffer::TensorTopology* fb_topology);

}  // namespace ttnn

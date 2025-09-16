// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_buffer.hpp>

#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor_attributes.hpp"
#include "ttnn/tensor/tensor_spec.hpp"

namespace tt::tt_metal {

TensorAttributes::TensorAttributes(Storage storage, TensorSpec tensor_spec, TensorTopology tensor_topology) :
    storage_(std::move(storage)), tensor_spec_(std::move(tensor_spec)), tensor_topology_(std::move(tensor_topology)) {}

const Storage& TensorAttributes::get_storage() const { return storage_; }
Storage& TensorAttributes::get_storage() { return storage_; }
const TensorSpec& TensorAttributes::get_tensor_spec() const { return tensor_spec_; }
const TensorTopology& TensorAttributes::get_tensor_topology() const { return tensor_topology_; }

}  // namespace tt::tt_metal

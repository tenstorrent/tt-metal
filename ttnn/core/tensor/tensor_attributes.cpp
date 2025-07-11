// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <utility>
#include <variant>

#include "ttnn/distributed/distributed_tensor_config.hpp"
#include <tt-metalium/host_buffer.hpp>
#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor_attributes.hpp"
#include "ttnn/tensor/tensor_spec.hpp"

namespace tt::tt_metal {

TensorAttributes::TensorAttributes(
    Storage storage, TensorSpec tensor_spec, DistributedTensorConfig distributed_tensor_config) :
    storage_(std::move(storage)),
    tensor_spec_(std::move(tensor_spec)),
    distributed_tensor_config_(std::move(distributed_tensor_config)) {}

const Storage& TensorAttributes::get_storage() const { return storage_; }
Storage& TensorAttributes::get_storage() { return storage_; }
const TensorSpec& TensorAttributes::get_tensor_spec() const { return tensor_spec_; }
const DistributedTensorConfig& TensorAttributes::get_distributed_tensor_config() const {
    return distributed_tensor_config_;
}

}  // namespace tt::tt_metal

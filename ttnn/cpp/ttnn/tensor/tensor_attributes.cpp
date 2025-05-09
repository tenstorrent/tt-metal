// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <utility>

#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor_attributes.hpp"
#include "ttnn/tensor/tensor_spec.hpp"

namespace tt::tt_metal {

TensorAttributes::TensorAttributes(TensorSpec tensor_spec) : tensor_spec_(std::move(tensor_spec)) {}

TensorAttributes::TensorAttributes(Storage storage, TensorSpec tensor_spec) :
    storage_(std::move(storage)), tensor_spec_(std::move(tensor_spec)) {}

const Storage& TensorAttributes::get_storage() const { return storage_; }
const TensorSpec& TensorAttributes::get_tensor_spec() const { return tensor_spec_; }
Storage& TensorAttributes::get_storage() { return storage_; }

}  // namespace tt::tt_metal

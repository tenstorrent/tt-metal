// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <utility>

#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor_attributes.hpp"
#include "ttnn/tensor/tensor_spec.hpp"

namespace tt::tt_metal {

TensorAttributes::TensorAttributes() :
    storage_(Storage(HostStorage())),
    tensor_spec_(
        ttnn::Shape(std::array<uint32_t, 4>{0xff, 0xff, 0xff, 0xff}),
        TensorLayout(DataType::INVALID, PageConfig(Layout::INVALID), MemoryConfig{})) {}

TensorAttributes::TensorAttributes(Storage storage, TensorSpec tensor_spec) :
    storage_(std::move(storage)), tensor_spec_(std::move(tensor_spec)) {}

const Storage& TensorAttributes::get_storage() const { return storage_; }
const TensorSpec& TensorAttributes::get_tensor_spec() const { return tensor_spec_; }
Storage& TensorAttributes::get_storage() { return storage_; }
TensorSpec& TensorAttributes::get_tensor_spec() { return tensor_spec_; }
void TensorAttributes::set_storage(const Storage& storage) { storage_ = storage; }
void TensorAttributes::set_tensor_spec(const TensorSpec& tensor_spec) { tensor_spec_ = tensor_spec; }

}  // namespace tt::tt_metal

// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor_spec.hpp"

namespace tt::tt_metal {

class TensorAttributes : public std::enable_shared_from_this<TensorAttributes> {
public:
    TensorAttributes(Storage storage, TensorSpec tensor_spec);

    // Creates `TensorAttributes` with default values.
    // TODO: remove the ability to do this. Tensor attributes should always be well-formed at creation time.
    TensorAttributes();

    // Getters and setters.
    const Storage& get_storage() const;
    const TensorSpec& get_tensor_spec() const;
    Storage& get_storage();
    TensorSpec& get_tensor_spec();
    void set_storage(const Storage& storage);
    void set_tensor_spec(const TensorSpec& tensor_spec);

private:
    Storage storage_;
    TensorSpec tensor_spec_;
};

}  // namespace tt::tt_metal

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
    explicit TensorAttributes(TensorSpec tensor_spec);
    TensorAttributes(Storage storage, TensorSpec tensor_spec);

    // Getters and setters.
    const Storage& get_storage() const;
    const TensorSpec& get_tensor_spec() const;
    Storage& get_storage();

private:
    Storage storage_;
    TensorSpec tensor_spec_;
};

}  // namespace tt::tt_metal

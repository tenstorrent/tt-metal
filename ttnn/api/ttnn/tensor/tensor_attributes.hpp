// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include "ttnn/tensor/storage.hpp"

namespace tt::tt_metal {

class TensorAttributes : public std::enable_shared_from_this<TensorAttributes> {
public:
    TensorAttributes(ttnn::HostStorage storage);
    TensorAttributes(ttnn::DeviceStorage storage);

    TensorAttributes(const TensorAttributes&) = default;
    TensorAttributes(TensorAttributes&&) = default;
    TensorAttributes& operator=(const TensorAttributes&) = default;
    TensorAttributes& operator=(TensorAttributes&&) = default;

    // Getters and setters.
    const ttnn::Storage& get_storage() const;
    ttnn::Storage& get_storage();
    const TensorSpec& get_tensor_spec() const;
    const TensorTopology& get_tensor_topology() const;

    void update_tensor_topology(const TensorTopology& tensor_topology);

private:
    ttnn::Storage storage_;
};

}  // namespace tt::tt_metal

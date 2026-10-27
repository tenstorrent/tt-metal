// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include "ttnn/tensor/storage.hpp"

namespace ttnn {

class TensorAttributes : public std::enable_shared_from_this<TensorAttributes> {
public:
    TensorAttributes(HostStorage storage);
    TensorAttributes(DeviceStorage storage);

    TensorAttributes(const TensorAttributes&) = default;
    TensorAttributes(TensorAttributes&&) = default;
    TensorAttributes& operator=(const TensorAttributes&) = default;
    TensorAttributes& operator=(TensorAttributes&&) = default;

    // Getters and setters.
    const Storage& get_storage() const;
    Storage& get_storage();
    const tt::tt_metal::TensorSpec& get_tensor_spec() const;
    const tt::tt_metal::TensorTopology& get_tensor_topology() const;

    void update_tensor_topology(const tt::tt_metal::TensorTopology& tensor_topology);

private:
    Storage storage_;
};

}  // namespace ttnn

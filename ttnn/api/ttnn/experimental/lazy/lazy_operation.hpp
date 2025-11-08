// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/experimental/lazy/lazy_tensor.hpp"

namespace ttnn::experimental::lazy {

template <typename operation_t>
constexpr tt::stl::hash::hash_t get_operation_type_id() {
    return tt::stl::hash::type_hash<operation_t>;
}

// TODO: Maybw we should add .get(idx) and size() methods
struct LazyOperationInputs {
    virtual void for_each(const std::function<void(const std::shared_ptr<LazyTensor>&)>& fn) const = 0;
    virtual std::any inputs() const = 0;
    virtual ~LazyOperationInputs() = default;
};

struct EmptyLazyOperationInputs : public LazyOperationInputs {
    void for_each(const std::function<void(const std::shared_ptr<LazyTensor>&)>& fn) const override {}
    std::any inputs() const override { return std::any(); }
};

struct LazyOperation {
    LazyOperation() = default;
    virtual std::vector<tt::tt_metal::metal_tensor::Tensor> invoke(const LazyOperationInputs& inputs) = 0;
    virtual std::string_view name() const = 0;
    virtual tt::stl::hash::hash_t operation_type_id() const = 0;
    // TODO: Do we need some attributes for serialization purposes?
    virtual ~LazyOperation() = default;
};

struct MaterializedLazyOperation : public LazyOperation {
    std::vector<tt::tt_metal::metal_tensor::Tensor> invoke(const LazyOperationInputs& inputs) override {
        return {};
    }
    std::string_view name() const override { return "MaterializedLazyOperation"; }
    tt::stl::hash::hash_t operation_type_id() const override { return tt::stl::hash::type_hash<MaterializedLazyOperation>; }
};

}  // namespace ttnn::experimental::lazy

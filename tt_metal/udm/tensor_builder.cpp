// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/udm/tensor_builder.hpp"
#include <tt_stl/assert.hpp>

namespace tt::tt_metal::udm {

class TensorBuilder::Impl {
public:
    explicit Impl(const ttnn::Tensor& input_tensor, const Grid& grid, const Block& block) :
        tensor_(input_tensor),
        block_builder_(
            input_tensor.device(),
            input_tensor.tensor_topology().distribution_shape(),
            input_tensor.tensor_topology().mesh_coords(),
            grid,
            block),
        tensor_accessor_(input_tensor, block_builder_) {
        // TensorBuilder extracts info from tensor and creates BlockBuilder + BlockTensorAccessor
        reconstruct_global_shape();
    }

    void reconstruct_global_shape() {
        // Reconstruct the global tensor shape from sharded tensor
        const auto& per_device_shape = tensor_.padded_shape();
        const auto& tensor_topology = tensor_.tensor_topology();
        const auto& placements = tensor_topology.placements();
        const auto& mesh_shape = tensor_topology.distribution_shape();

        ttsl::SmallVector<uint32_t> global_shape_vec(per_device_shape.cbegin(), per_device_shape.cend());

        for (size_t mesh_dim = 0; mesh_dim < placements.size(); ++mesh_dim) {
            if (std::holds_alternative<ttnn::distributed::MeshMapperConfig::Shard>(placements[mesh_dim])) {
                const auto& shard = std::get<ttnn::distributed::MeshMapperConfig::Shard>(placements[mesh_dim]);
                global_shape_vec[shard.dim] *= mesh_shape[mesh_dim];
            }
        }

        global_shape_ = tt::tt_metal::Shape(global_shape_vec);
    }

    const BlockBuilder& block_builder() const { return block_builder_; }

    BlockBuilder& block_builder() { return block_builder_; }

    const BlockTensorAccessor& tensor_accessor() const { return tensor_accessor_; }

    BlockTensorAccessor& tensor_accessor() { return tensor_accessor_; }

    const ttnn::Tensor& tensor() const { return tensor_; }

    const tt::tt_metal::Shape& global_shape() const { return global_shape_; }

private:
    const ttnn::Tensor& tensor_;
    tt::tt_metal::Shape global_shape_;
    BlockBuilder block_builder_;
    BlockTensorAccessor tensor_accessor_;
};

TensorBuilder::TensorBuilder(const ttnn::Tensor& input_tensor, const Grid& grid, const Block& block) :
    impl_(std::make_unique<Impl>(input_tensor, grid, block)) {}

TensorBuilder::~TensorBuilder() = default;

TensorBuilder::TensorBuilder(TensorBuilder&&) noexcept = default;
TensorBuilder& TensorBuilder::operator=(TensorBuilder&&) noexcept = default;

const BlockBuilder& TensorBuilder::block_builder() const { return impl_->block_builder(); }

BlockBuilder& TensorBuilder::block_builder() { return impl_->block_builder(); }

const BlockTensorAccessor& TensorBuilder::tensor_accessor() const { return impl_->tensor_accessor(); }

BlockTensorAccessor& TensorBuilder::tensor_accessor() { return impl_->tensor_accessor(); }

const ttnn::Tensor& TensorBuilder::tensor() const { return impl_->tensor(); }

const tt::tt_metal::Shape& TensorBuilder::global_shape() const { return impl_->global_shape(); }

TensorBuilder CreateTensorBuilder(const ttnn::Tensor& input_tensor, const Grid& grid, const Block& block) {
    return TensorBuilder(input_tensor, grid, block);
}

}  // namespace tt::tt_metal::udm

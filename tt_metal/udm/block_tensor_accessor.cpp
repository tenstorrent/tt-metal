// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/udm/block_tensor_accessor.hpp"
#include "tt_metal/udm/block_builder.hpp"
#include "tt_metal/api/tt-metalium/tensor_accessor_args.hpp"
#include <tt_stl/assert.hpp>

namespace tt::tt_metal::udm {

class BlockTensorAccessor::Impl {
public:
    Impl(const ttnn::Tensor& tensor, const BlockBuilder& block_builder) :
        tensor_(tensor), block_builder_(block_builder) {
        // Initialize using TensorAccessorArgs for now
        if (tensor_.buffer()) {
            accessor_args_ = tt::tt_metal::TensorAccessorArgs(*tensor_.buffer());
        }
    }

    std::vector<uint32_t> get_compile_time_args() const {
        if (!tensor_.buffer()) {
            return {};
        }

        // Get compile-time args from TensorAccessorArgs
        auto compile_time_args = accessor_args_.get_compile_time_args();

        // Add aligned page size
        compile_time_args.push_back(tensor_.buffer()->aligned_page_size());

        return compile_time_args;
    }

    uint64_t get_buffer_address() const {
        if (!tensor_.buffer()) {
            return 0;
        }
        return tensor_.buffer()->address();
    }

    uint32_t get_aligned_page_size() const {
        if (!tensor_.buffer()) {
            return 0;
        }
        return tensor_.buffer()->aligned_page_size();
    }

private:
    const ttnn::Tensor& tensor_;
    const BlockBuilder& block_builder_;
    tt::tt_metal::TensorAccessorArgs accessor_args_;
};

BlockTensorAccessor::BlockTensorAccessor(const ttnn::Tensor& tensor, const BlockBuilder& block_builder) :
    impl_(std::make_unique<Impl>(tensor, block_builder)) {}

BlockTensorAccessor::~BlockTensorAccessor() = default;

BlockTensorAccessor::BlockTensorAccessor(BlockTensorAccessor&&) noexcept = default;
BlockTensorAccessor& BlockTensorAccessor::operator=(BlockTensorAccessor&&) noexcept = default;

std::vector<uint32_t> BlockTensorAccessor::get_compile_time_args() const { return impl_->get_compile_time_args(); }

uint64_t BlockTensorAccessor::get_buffer_address() const { return impl_->get_buffer_address(); }

uint32_t BlockTensorAccessor::get_aligned_page_size() const { return impl_->get_aligned_page_size(); }

}  // namespace tt::tt_metal::udm

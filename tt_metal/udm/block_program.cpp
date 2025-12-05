// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/udm/block_program.hpp"
#include "tt_metal/udm/tensor_builder.hpp"
#include "tt_metal/api/tt-metalium/host_api.hpp"
#include <tt_stl/assert.hpp>

namespace tt::tt_metal::udm {

class BlockProgram::Impl {
public:
    explicit Impl(const TensorBuilder& builder) {
        // Create a single Program for now
        program_ = tt::tt_metal::CreateProgram();
    }

    tt::tt_metal::Program& program() { return program_; }
    const tt::tt_metal::Program& program() const { return program_; }

private:
    tt::tt_metal::Program program_;
};

BlockProgram::BlockProgram(const TensorBuilder& builder) : impl_(std::make_unique<Impl>(builder)) {}

BlockProgram::~BlockProgram() = default;

BlockProgram::BlockProgram(BlockProgram&&) noexcept = default;
BlockProgram& BlockProgram::operator=(BlockProgram&&) noexcept = default;

tt::tt_metal::Program& BlockProgram::program() { return impl_->program(); }

const tt::tt_metal::Program& BlockProgram::program() const { return impl_->program(); }

BlockProgram CreateBlockProgram(const TensorBuilder& builder) { return BlockProgram(builder); }

}  // namespace tt::tt_metal::udm

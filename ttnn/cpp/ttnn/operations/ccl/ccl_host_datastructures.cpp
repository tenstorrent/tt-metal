// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/ccl_host_datastructures.hpp"

namespace ttnn {
namespace ccl {

CCLOpConfig::CCLOpConfig(
    std::vector<Tensor>& input_tensors, const std::vector<Tensor>& output_tensors, Topology topology) :
    input_tensors(&input_tensors),
    output_tensors(&output_tensors),
    input_sharded(input_tensors.at(0).is_sharded()),
    output_sharded(output_tensors.at(0).is_sharded()),
    page_size(input_tensors.at(0).buffer()->page_size()),
    input_shard_size_bytes(
        input_tensors.at(0).is_sharded() ? static_cast<std::optional<uint32_t>>(
                                                (input_tensors.at(0).buffer()->page_size() *
                                                input_tensors.at(0).buffer()->shard_spec().tensor2d_shape[0] *
                                                input_tensors.at(0).buffer()->shard_spec().tensor2d_shape[1]) /
                                                input_tensors.at(0).shard_spec()->num_cores())
                                            : std::nullopt),
    output_shard_size_bytes(
        output_tensors.at(0).is_sharded() ? static_cast<std::optional<uint32_t>>(
                                                (output_tensors.at(0).buffer()->page_size() *
                                                    output_tensors.at(0).buffer()->shard_spec().tensor2d_shape[0] *
                                                    output_tensors.at(0).buffer()->shard_spec().tensor2d_shape[1]) /
                                                input_tensors.at(0).shard_spec()->num_cores())
                                            : std::nullopt),
    shard_grid_size(output_tensors.at(0).is_sharded() ? input_tensors.at(0).shard_spec()->num_cores() : 0),
    topology(topology),
    is_row_major(input_tensors.at(0).get_layout() == Layout::ROW_MAJOR) {
    TT_ASSERT(!this->is_input_sharded() || input_shard_size_bytes.has_value());
    TT_ASSERT(!this->is_output_sharded() || output_shard_size_bytes.has_value());
}

uint32_t CCLOpConfig::get_input_shard_size_bytes() const {
    TT_ASSERT(input_shard_size_bytes.has_value());
    return input_shard_size_bytes.value();
}

uint32_t CCLOpConfig::get_output_shard_size_bytes() const {
    TT_ASSERT(output_shard_size_bytes.has_value());
    return output_shard_size_bytes.value();
}

uint32_t CCLOpConfig::get_page_size() const { return this->page_size; }

Topology CCLOpConfig::get_topology() const { return this->topology; }

bool CCLOpConfig::is_input_sharded() const { return this->input_sharded; }

bool CCLOpConfig::is_output_sharded() const { return this->output_sharded; }

bool CCLOpConfig::get_shard_grid_size() const { return this->shard_grid_size; }

Tensor const& CCLOpConfig::get_input_tensor(std::size_t i) const { return input_tensors->at(i); }

Tensor const& CCLOpConfig::get_output_tensor(std::size_t i) const { return output_tensors->at(i); }

std::map<string, string> CCLOpConfig::emit_worker_defines() const {

    std::map<string, string> worker_defines;
    if (this->is_row_major) {
        worker_defines["ROW_MAJOR_LAYOUT"] = "1";
    } else {
        worker_defines["TILED_LAYOUT"] = "1";
    }
    if (this->input_sharded) {
        TT_ASSERT(this->output_sharded, "CCL Util functions currently don't  support a mix of input sharded with output interleaved or vice versa");
        worker_defines["SHARDED_MEM_LAYOUT"] = "1";
    } else {
        worker_defines["INTERLEAVED_MEM_LAYOUT"] = "1";
    }

    return worker_defines;
}

} // namespace ccl
} // namespace ttnn

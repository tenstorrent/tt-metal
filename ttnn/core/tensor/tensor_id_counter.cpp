// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal {

std::atomic<std::uint64_t> Tensor::tensor_id_counter{0};

std::uint64_t Tensor::get_tensor_id_counter() { return tensor_id_counter.load(std::memory_order_relaxed); }

void Tensor::set_tensor_id_counter(std::uint64_t id) { tensor_id_counter.store(id, std::memory_order_relaxed); }

std::uint64_t Tensor::next_tensor_id() { return tensor_id_counter.fetch_add(1, std::memory_order_relaxed); }

}  // namespace tt::tt_metal

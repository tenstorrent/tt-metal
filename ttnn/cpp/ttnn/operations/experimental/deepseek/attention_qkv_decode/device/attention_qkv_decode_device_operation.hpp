// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <variant>

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/prefetcher_pipe.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek::attention_qkv_decode {

// Core roles on the 12x10 Blackhole grid, disjoint:
//   q_b: x [0, 8) y [0, 8), one [q_lora, N_q / 64] slab each
//   kv:  x [0, 8) y [8, 10), one [K, 32] slab each (kv's 16 output tiles)
//   q_a: x [8, 12) y [0, 10), one [K, 32] slab each (q_a's 40 output tiles)
// q_a's and kv's results meet in one row on the hub (the first q_a core), which multicasts it over
// the q_b and kv cores: the 8x10 rectangle at the origin.
inline constexpr uint32_t kGridX = 12;
inline constexpr uint32_t kGridY = 10;
inline constexpr uint32_t kRowCoresX = 8;
inline constexpr uint32_t kQbCoresY = 8;
inline constexpr uint32_t kNumQbCores = kRowCoresX * kQbCoresY;

enum class WeightRole : uint32_t { QA, KV, QB };
inline constexpr uint32_t kNumWeightRoles = 3;

inline CoreRangeSet weight_role_cores(WeightRole role) {
    switch (role) {
        case WeightRole::QA: return CoreRangeSet(CoreRange({kRowCoresX, 0}, {kGridX - 1, kGridY - 1}));
        case WeightRole::KV: return CoreRangeSet(CoreRange({0, kQbCoresY}, {kRowCoresX - 1, kGridY - 1}));
        case WeightRole::QB: return CoreRangeSet(CoreRange({0, 0}, {kRowCoresX - 1, kQbCoresY - 1}));
    }
    return {};
}

struct AttentionQkvDecodeDeviceOperation {
    struct operation_attributes_t {
        float epsilon = 1e-6f;
        MemoryConfig output_mem_config;
        // Tensor-prefetcher pipes delivering the weights instead of DRAM reads: every pipe of the q_a, kv
        // and q_b rings, whose receivers are exactly those roles' cores and whose rings each hold one whole
        // slab (one entry per core per call). Empty reads the DRAM slabs.
        ttnn::PrefetcherPipeList prefetcher_pipes;
    };

    struct tensor_args_t {
        const Tensor& tokens;
        const Tensor& w_q_a;
        const Tensor& w_kv;
        const Tensor& w_q_b;
        const Tensor& q_a_gamma;
        const Tensor& kv_gamma;
        const Tensor& cos;
        const Tensor& sin;
        const Tensor& kv_cache;
        const Tensor& cache_pos;
    };

    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    struct ProgramFactory {
        static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<ProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::deepseek::attention_qkv_decode

namespace ttnn::prim {

ttnn::Tensor attention_qkv_decode(
    const ttnn::Tensor& tokens,
    const ttnn::Tensor& w_q_a,
    const ttnn::Tensor& w_kv,
    const ttnn::Tensor& w_q_b,
    const ttnn::Tensor& q_a_gamma,
    const ttnn::Tensor& kv_gamma,
    const ttnn::Tensor& cos,
    const ttnn::Tensor& sin,
    const ttnn::Tensor& kv_cache,
    const ttnn::Tensor& cache_pos,
    float epsilon,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const ttnn::PrefetcherPipeList& prefetcher_pipes);

}  // namespace ttnn::prim

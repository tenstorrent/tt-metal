// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::nomic_embed_text_v2_moe {

// Sizes of one call and the cores it runs on, derived from its inputs and attributes. M spreads over the config
// grid's columns and N over its rows, as minimal_matmul does with M > N, each padded to a multiple of its lanes.
struct PirotskiMatmulLayout {
    uint32_t m_tiles = 0;
    uint32_t k_tiles = 0;
    uint32_t n_tiles = 0;
    uint32_t columns = 0;
    uint32_t rows = 0;
    uint32_t m_tiles_per_core = 0;
    uint32_t n_tiles_per_core = 0;
    uint32_t m_blocks_per_core = 0;
    uint32_t n_blocks_per_core = 0;
    uint32_t k_blocks = 0;
    // Nonzero when the input is the split layer norm's block-sharded layout over the same columns: K block k of a
    // column's rows is then read straight from the shard holding it, shard_width K tiles a shard.
    uint32_t input_shard_width = 0;
    // The output is block-sharded with core (x, y)'s shard exactly its output, in one N block: compute packs into it.
    bool direct_pack = false;
    // The strip: the device grid's rows below the config grid. Strip row j computes the last strip_m_tiles rows of
    // every column's M share for N tiles [j * strip_n_tiles, (j + 1) * strip_n_tiles), strip_lanes grid rows' worth;
    // the config grid's cores keep the first main_m_tiles rows. With no strip, main_m_tiles is m_tiles_per_core.
    uint32_t main_m_tiles = 0;
    uint32_t strip_rows = 0;
    uint32_t strip_m_tiles = 0;
    uint32_t strip_lanes = 0;
    uint32_t strip_n_tiles = 0;
    uint32_t strip_subblock_h = 0;
    uint32_t strip_subblock_w = 0;
};

struct PirotskiMatmulDeviceOperation {
    struct operation_attributes_t {
        ttnn::experimental::prim::MinimalMatmulConfig config;
        tt::tt_metal::DataType dtype = tt::tt_metal::DataType::BFLOAT16;
        tt::tt_metal::MemoryConfig memory_config;
        DeviceComputeKernelConfig compute_kernel_config;
        bool resident_weights = false;
        uint32_t strip_m_tiles = 0;
        bool strip_read_ahead = false;
        bool read_ahead = false;
    };

    struct tensor_args_t {
        const Tensor& input;
        const Tensor& weight;
        std::optional<Tensor> bias;
    };

    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    // Validates the call and derives its sizes, on a program cache miss: a hit has the hash of a validated call.
    static PirotskiMatmulLayout compute_layout(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    // Every runtime argument that is not a buffer address comes from a hashed attribute or tensor spec, so the
    // default program hash keeps a cache hit correct; the framework re-binds the buffers.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
};

}  // namespace ttnn::experimental::nomic_embed_text_v2_moe

namespace ttnn::prim {

Tensor pirotski_matmul(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    const ttnn::experimental::prim::MinimalMatmulConfig& config,
    tt::tt_metal::DataType dtype,
    const tt::tt_metal::MemoryConfig& memory_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    bool resident_weights,
    uint32_t strip_m_tiles,
    bool strip_read_ahead,
    bool read_ahead);

}  // namespace ttnn::prim

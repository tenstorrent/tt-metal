// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "mlir_interface_api.hpp" // shard_spec_tuple, memory_config_tuple

#include <array>
#include <optional>
#include <tuple>
#include <string>
#include <vector>

namespace tt
{
namespace tt_metal
{
enum class DataType;            // tensor/types.hpp
enum class Layout;              // tensor/types.hpp
enum class StorageType;         // tensor/types.hpp
enum class TensorMemoryLayout;  // tt_metal/impl/buffers/buffer_constants.hpp
enum class ShardOrientation;    // tt_metal/impl/buffers/buffer_constants.hpp
enum class BufferType;          // tt_metal/impl/buffers/buffer.hpp
struct ShardSpec;               // tt_metal/impl/buffers/buffer.hpp
struct MemoryConfig;            // tensor/types.hpp
struct Shape;                   // tensor/types.hpp
} // namespace tt_metal
} // namespace tt

namespace ttnn {
    namespace types {
        struct Shape;           // tensor/types.hpp
    }
} // namespace ttnn

namespace ttnn::operations::unary {
    enum class UnaryOpType;     // unary_op_types.hpp
}

namespace ttnn::operations::matmul {
    struct MatmulMultiCoreReuseProgramConfig;
    struct MatmulMultiCoreReuseMultiCastProgramConfig;
    struct MatmulMultiCoreReuseMultiCast1DProgramConfig;
}

namespace ttnn::str_wrapper
{
    std::optional<tt::tt_metal::StorageType> to_storage_type(const std::string& storage_type_str);
    std::optional<tt::tt_metal::Layout> to_layout(const std::string& layout_str);
    std::optional<tt::tt_metal::TensorMemoryLayout> to_tensor_memory_layout(const std::string& tensor_memory_layout_str);
    std::optional<tt::tt_metal::DataType> to_data_type(const std::string& data_type_str);
    std::optional<tt::tt_metal::BufferType> to_buffer_type(const std::string& buffer_type_str);
    std::optional<tt::tt_metal::ShardOrientation> to_shard_orientation(const std::string& shard_str);
    std::optional<ttnn::operations::unary::UnaryOpType> to_unary_op_type(const std::string& unary_op_type_str);
} // namespace ttnn::str_wrapper

namespace ttnn::tuple_wrapper
{
    std::optional<tt::tt_metal::ShardSpec> to_shard_spec(const mlir_interface::shard_spec_tuple& shard_spec_tuple);
    std::optional<tt::tt_metal::MemoryConfig> to_memory_config(const mlir_interface::memory_config_tuple& memory_config_tuple);
    std::optional<ttnn::operations::matmul::MatmulMultiCoreReuseProgramConfig> to_program_config(const mlir_interface::matmul_multicore_reuse_config_tuple& program_config_tuple);
    std::optional<ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig> to_multicast_program_config(const mlir_interface::matmul_multicore_reuse_config_tuple &program_config_tuple, bool transpose_mcast, bool fuse_batch);
    std::optional<ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig> to_multicast_1d_program_config(const mlir_interface::matmul_multicore_reuse_config_tuple &program_config_tuple, bool fuse_batch, bool mcast_in0);
} // namespace ttnn::tuple_wrapper

namespace ttnn::vector_wrapper
{
    tt::tt_metal::Shape to_shape(const std::vector<uint32_t>& shape_vector);
} // namespace ttnn::vector_wrapper

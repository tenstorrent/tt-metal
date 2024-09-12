#pragma once

#include <cstdint>
#include <array>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace ttnn::mlir_interface
{
    int add(int a, int b);
    int subtract(int a, int b);

    // check if layout is dram interleaved or l1 sharded, returns false otherwise
    bool dummy_check(const std::string& tensor_memory_layout_str, const std::string& buffer_type_str);

    // shard_spec_tuple = core_range_set, shard_shape, shard_orientation, halo
    using shard_spec_tuple = std::tuple<std::vector<std::array<uint32_t, 4>>, std::array<uint32_t, 2>, std::string, bool>;
    // memory_config_typle = tensor_memory_layout, buffer_type, shard_spec_tuple
    using memory_config_tuple = std::tuple<std::string, std::string, std::optional<shard_spec_tuple>>;

    // struct MatmulMultiCoreReuseProgramConfig
    // compute_with_storage_grid_size, in0_block_w, out_subblock_h, out_subblock_w, per_core_M, per_core_N
    using matmul_multicore_reuse_config_tuple = std::tuple<std::array<uint32_t,2>, size_t, size_t, size_t, size_t, size_t>;

    bool does_binary_op_support_input_output_constraints(
        const std::vector<uint32_t>& shape_a,
        const memory_config_tuple& memory_config_a,
        const std::string& data_type_a,
        const std::vector<uint32_t>& shape_b,
        const memory_config_tuple& memory_config_b,
        const std::string& data_type_b,
        const memory_config_tuple& memory_config_o,
        const std::string& data_type_o);

    bool does_unary_op_support_input_output_constraints(
        const std::string op_type, // only RELU supported for now
        const std::vector<uint32_t>& input_shape_a,
        const memory_config_tuple& memory_config_a,
        const std::string& data_type_a,
        const std::vector<uint32_t>& input_shape_o,
        const memory_config_tuple& memory_config_o,
        const std::string& data_type_o);

    bool does_softmax_op_support_input_output_constraints(
        const std::vector<uint32_t>& input_shape_a,
        const memory_config_tuple& memory_config_a,
        const std::string& data_type_a,
        const std::vector<uint32_t>& input_shape_o,
        const memory_config_tuple& memory_config_o,
        const std::string& data_type_o);

    bool does_matmul_multicore_reuse_multicast_support_input_output_constraints(
        const std::vector<uint32_t>& shape_a,
        const memory_config_tuple& memory_config_a,
        const std::string& data_type_a,
        const std::vector<uint32_t>& shape_b,
        const memory_config_tuple& memory_config_b,
        const std::string& data_type_b,
        const memory_config_tuple& memory_config_o,
        const std::string& data_type_o,
        const matmul_multicore_reuse_config_tuple matmul_config,
        const bool transpose_mcast,
        const bool fuse_batch = true);

    bool does_matmul_multicore_reuse_multicast_1d_op_support_input_output_constraints(
        const std::vector<uint32_t>& shape_a,
        const memory_config_tuple& memory_config_a,
        const std::string& data_type_a,
        const std::vector<uint32_t>& shape_b,
        const memory_config_tuple& memory_config_b,
        const std::string& data_type_b,
        const memory_config_tuple& memory_config_o,
        const std::string& data_type_o,
        const matmul_multicore_reuse_config_tuple matmul_config,
        const bool fuse_batch,
        const bool mcast_in0);
}

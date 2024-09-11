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

    bool does_binary_op_support_input_output_constraints(
        const std::vector<uint32_t>& _shape_a,
        const memory_config_tuple& _memory_config_a,
        const std::string& _data_type_a,
        const std::vector<uint32_t>& _shape_b,
        const memory_config_tuple& _memory_config_b,
        const std::string& _data_type_b,
        const memory_config_tuple& _memory_config_o,
        const std::string& _data_type_o);
}

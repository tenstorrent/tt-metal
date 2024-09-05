#pragma once

#include <cstdint>
#include <string>

namespace ttnn::mlir_interface
{
    int add(int a, int b);
    int subtract(int a, int b);

    // check if layout is dram interleaved or l1 sharded, returns false otherwise
    bool dummy_check(const std::string& tensor_memory_layout_str, const std::string& buffer_type_str);
}

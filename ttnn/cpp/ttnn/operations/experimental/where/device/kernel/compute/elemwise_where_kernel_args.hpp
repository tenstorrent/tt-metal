#pragma once

#include <cstdint>

#include "cpp/kernel/kernel_host_utils.hpp"

namespace ttnn::kernel::eltwise::where_args {

struct ElemwiseComputeKernelArgs {
    uint32_t per_core_block_cnt;
    uint32_t per_core_block_size;
};

VALIDATE_KERNEL_STRUCT(ElemwiseComputeKernelArgs)

}  // namespace ttnn::kernel::eltwise::where_args

#pragma once

#include <cstdint>

#include "cpp/kernel/kernel_host_utils.hpp"

namespace ttnn::kernel::eltwise::where_args {

struct ElemwiseWriterKernelArgs {
    uint32_t dst_base_addr;
    uint32_t num_tiles;
    uint32_t tile_ofs;
};

struct CompileTimeWriterKernelArgs {
    uint32_t cb_dst;
    uint32_t is_dst_dram;
};

VALIDATE_KERNEL_STRUCT(ElemwiseWriterKernelArgs)
VALIDATE_KERNEL_STRUCT(CompileTimeWriterKernelArgs)
}  // namespace ttnn::kernel::eltwise::where_args

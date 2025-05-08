// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <tt-metalium/core_coord.hpp>
#include "cpp/ttnn/operations/ccl/common/types/ccl_types.hpp"

#include <vector>
#include <string>

namespace tt {
namespace tt_metal {
class Tensor;
class ShardSpec;

class IDevice;

}  // namespace tt_metal
}  // namespace tt

namespace ttnn {
namespace ccl {

using args_list_t = std::vector<uint32_t>;

args_list_t emit_runtime_args(WorkerEdmInterfaceArgs const& edm_interface_args);
args_list_t emit_compile_time(WorkerEdmInterfaceArgs const& edm_interface_args);
args_list_t log_runtime_args(WorkerEdmInterfaceArgs const& edm_interface_args, std::string_view const& prefix);
args_list_t log_compile_time(WorkerEdmInterfaceArgs const& edm_interface_args, std::string_view const& prefix);

template <typename T>
args_list_t emit_runtime_args(T const& args);
template <typename T>
args_list_t emit_compile_time(T const& args);

////////////
// Shape 4D
template <typename T>
args_list_t emit_runtime_args(Shape4D<T> const& shape) {
    return {shape.w, shape.z, shape.y, shape.x};
}

template <typename T>
args_list_t emit_compile_time(Shape4D<T> const& shape) {
    return {};
}

args_list_t emit_address_generator_runtime_args(
    tt::tt_metal::IDevice const* const d, tt::tt_metal::Tensor const& tensor);
args_list_t legacy_emit_address_generator_runtime_args(
    const tt::tt_metal::IDevice* const d, const tt::tt_metal::Tensor& tensor);
args_list_t emit_address_generator_compile_time_args(const tt::tt_metal::Tensor& t);
args_list_t legacy_emit_address_generator_compile_time_args(const tt::tt_metal::Tensor& tensor);

std::pair<CoreCoord, CoreCoord> shard_grid_from_shard_spec(const tt::tt_metal::ShardSpec& shard_spec);

struct ShardedAddrGenArgBuilder {
    static bool shard_grid_is_transposed(tt::tt_metal::Tensor const& t);
    static std::vector<uint32_t> emit_ct_args(tt::tt_metal::Tensor const& t);
    static std::vector<uint32_t> emit_rt_args(tt::tt_metal::IDevice const* d, tt::tt_metal::Tensor const& t);
    static void log_sharded_tensor_kernel_args(tt::tt_metal::Tensor const& t, std::string const& prefix);
};

}  // namespace ccl
}  // namespace ttnn

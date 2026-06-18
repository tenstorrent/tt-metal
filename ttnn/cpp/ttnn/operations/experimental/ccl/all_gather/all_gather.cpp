// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather.hpp"

#include "ttnn/operations/experimental/ccl/all_gather/device/all_gather_device_operation.hpp"
#include "ttnn/operations/experimental/ccl/composite_common.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"

namespace ttnn::experimental {

// Native implementation only handles cases where every output write is an aligned NoC
// write into the output buffer, computed with index math alone -- the bytes are never
// rearranged on-device first. If it needs a transpose, untilize, re-pad, or
// re-shard ("massaged op"), it goes to composite.
bool use_composite_all_gather(
    const ttnn::Tensor& /*input_tensor*/, int32_t /*dim*/, const std::optional<ttnn::MemoryConfig>& /*memory_config*/) {
    // Route to composite when the native iterator can't place the bytes:
    //
    // 1. Row-major, last-dim gather, unaligned pages (page_size != aligned_page_size:
    //    row/shard width in bytes isn't alignment-divisible). The concat seam lands
    //    on a sub-aligned address, which can't be DMA'd. (1D unaligned RM is the
    //    rank-1 form of this.)
    //
    // 2. Tiled, padded on the gather dim (logical[dim] != padded[dim]). The seam
    //    falls inside a 32x32 tile, stranding padding mid-tensor.
    //
    // 3. Output memory_config forces an unrelated re-shard -- e.g. an output shard
    //    width that isn't a whole multiple/divisor of the input's (no integer
    //    page-size ratio).
    //
    // Everything else stays native, including aligned multi-shard concat
    // (width/block-sharded input -> wider-shard or interleaved output): those writes
    // are aligned and index-computable, so they're a native enhancement, not a
    // composite case.
    //
    // TODO implement detection; default to native for now.
    return false;
}

ttnn::Tensor all_gather(
    const ttnn::Tensor& input_tensor,
    int32_t dim,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::Tensor>& persistent_output_tensor,
    std::optional<uint32_t> cluster_axis,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id,
    const std::optional<CoreRangeSet>& sub_core_grid) {
    // TODO fix this
    // if (composite_common::use_composite_all_gather(input_tensor, gather_dim, memory_config)) {
    if (false) {
        log_info(
            tt::LogOp,
            "Using slower composite all_gather since input_tensor has unaligned pages / padded tiles on gather dim "
            "xxx");

        // Query the Fabric setup
        auto* mesh_device = input_tensor.device();
        TT_FATAL(mesh_device != nullptr, "Input tensor should be on device for all_gather operation");
        uint32_t num_links = ttnn::operations::ccl::common::get_num_links(*mesh_device, cluster_axis);

        return composite_common::composite_all_gather(
            input_tensor, dim, num_links, memory_config, /*subdevice_id*/ std::nullopt, cluster_axis);
    }

    return ttnn::prim::all_gather_experimental(
        input_tensor, persistent_output_tensor, dim, memory_config, cluster_axis, subdevice_id, sub_core_grid);
}

}  // namespace ttnn::experimental

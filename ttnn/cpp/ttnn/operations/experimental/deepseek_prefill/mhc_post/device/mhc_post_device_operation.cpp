// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mhc_post_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/constants.hpp>
#include <cstdlib>

namespace ttnn::experimental::prim {

namespace {

// Every input is indexed by tile page, so an unpadded rank-4 [1,1,rows,cols] block with both
// tiled dims tile-aligned is what keeps host page arithmetic and kernel page arithmetic the
// same expression.
void check_flat_2d(const Tensor& t, const char* name, uint32_t rows, uint32_t cols) {
    TT_FATAL(t.dtype() == tt::tt_metal::DataType::FLOAT32, "{} must be FLOAT32", name);
    TT_FATAL(t.layout() == tt::tt_metal::Layout::TILE, "{} must be TILE layout", name);
    TT_FATAL(t.is_allocated(), "{} must be allocated on device", name);
    TT_FATAL(t.storage_type() == StorageType::DEVICE, "{} must be a device tensor", name);
    const auto& ls = t.logical_shape();
    TT_FATAL(ls.rank() >= 2, "{} must be rank>=2, got rank {}", name, ls.rank());
    for (int i = 0; i + 2 < static_cast<int>(ls.rank()); ++i) {
        TT_FATAL(ls[i] == 1, "{} leading dim {} must be 1 (got {})", name, i, ls[i]);
    }
    TT_FATAL(
        static_cast<uint32_t>(ls[-2]) == rows && static_cast<uint32_t>(ls[-1]) == cols,
        "{} must be [{}, {}], got [{}, {}]",
        name,
        rows,
        cols,
        ls[-2],
        ls[-1]);
}

}  // namespace

void MhcPostDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const uint32_t n = args.n;
    TT_FATAL(n > 0, "n must be positive");
    // comb is a per-token n x n matrix flattened into one tile row, so n*n columns must fit a tile.
    TT_FATAL(n * n <= tt::constants::TILE_WIDTH, "n*n ({}) must fit one tile width (32)", n * n);

    const auto& yls = tensor_args.y.logical_shape();
    TT_FATAL(yls.rank() >= 2, "y must be rank>=2, got rank {}", yls.rank());
    const uint32_t T = yls[-2];
    const uint32_t C = yls[-1];
    // The column tiles of y, residual and out must line up page for page, which they only do
    // when each stream is a whole number of tiles wide. T is free: a partial last token-tile
    // carries padding through the mix untouched, since every term is row-local.
    TT_FATAL(C % tt::constants::TILE_WIDTH == 0, "y width ({}) must be a multiple of 32", C);

    check_flat_2d(tensor_args.y, "y", T, C);
    check_flat_2d(tensor_args.residual, "residual", T, n * C);
    // post and comb come straight from mhc_split_sinkhorn, whose outputs are logically n / n*n
    // wide inside a full tile; the extraction matmul reads the tile, so only the row count binds.
    check_flat_2d(tensor_args.post, "post", T, n);
    check_flat_2d(tensor_args.comb, "comb", T, n * n);

    const auto& consts = tensor_args.consts;
    TT_FATAL(consts.dtype() == tt::tt_metal::DataType::FLOAT32, "consts must be FLOAT32");
    TT_FATAL(consts.layout() == tt::tt_metal::Layout::TILE, "consts must be TILE layout");
    TT_FATAL(consts.is_allocated(), "consts must be allocated on device");
    TT_FATAL(consts.storage_type() == StorageType::DEVICE, "consts must be a device tensor");
    TT_FATAL(
        consts.physical_volume() / tt::constants::TILE_HW == n * n,
        "consts must be n*n={} tiles [n*n,32,32], got {}",
        n * n,
        consts.physical_volume() / tt::constants::TILE_HW);

    auto* dev = tensor_args.y.device();
    TT_FATAL(
        tensor_args.residual.device() == dev && tensor_args.post.device() == dev && tensor_args.comb.device() == dev &&
            tensor_args.consts.device() == dev,
        "all inputs must be on the same device");
}

MhcPostDeviceOperation::spec_return_value_t MhcPostDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    const auto& r = tensor_args.residual;
    return tt::tt_metal::TensorSpec(
        ttnn::Shape({r.logical_shape()[-2], r.logical_shape()[-1]}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::FLOAT32,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM}));
}

MhcPostDeviceOperation::tensor_return_value_t MhcPostDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.y.device());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor mhc_post(
    const Tensor& y, const Tensor& residual, const Tensor& post, const Tensor& comb, const Tensor& consts, uint32_t n) {
    using OperationType = ttnn::experimental::prim::MhcPostDeviceOperation;
    // MHC_MAX_CORES=1 pins to a single core (the multi-core A/B baseline). Carried in the hashed
    // attributes so a changed value re-keys the program cache rather than reusing a program built
    // for the other core grid.
    uint32_t max_cores = 0;
    if (const char* mc = std::getenv("MHC_MAX_CORES"); mc != nullptr) {
        max_cores = static_cast<uint32_t>(std::atoi(mc));
    }
    auto operation_attributes = OperationType::operation_attributes_t{n, max_cores};
    auto tensor_args = OperationType::tensor_args_t{y, residual, post, comb, consts};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim

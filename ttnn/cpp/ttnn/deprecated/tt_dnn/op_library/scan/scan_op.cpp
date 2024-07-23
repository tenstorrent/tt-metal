// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/experimental/tt_dnn/op_library/scan/scan_op.hpp"

#include "ttnn/run_operation.hpp"

namespace tt::tt_metal {

using TCores = const std::variant<CoreCoord, CoreRange, CoreRangeSet>;

template <size_t N = 1>
std::array<CB, N> create_cb(
    Program &program,
    const TCores &core_spec,
    uint32_t page_size,
    uint32_t num_pages,
    const tt::DataFormat data_format,
    std::array<CB, N> cbs,
    Buffer *buffer = nullptr) {
    std::map<uint8_t, tt::DataFormat> data_format_spec = {};
    for (auto cb : cbs) {
        data_format_spec[cb] = data_format;
    }

    auto cb_config = CircularBufferConfig(num_pages * page_size, {data_format_spec});
    for (auto cb : cbs) {
        cb_config.set_page_size(cb, page_size);
    }

    if (buffer != nullptr) {
        cb_config.set_globally_allocated_address(*buffer);
    }

    tt_metal::CreateCircularBuffer(program, core_spec, cb_config);
    return cbs;
}

template <typename T, size_t N, typename... Rest>
auto aggregate_arrays(const std::array<T, N> &first, const Rest &...rest) {
    std::vector<uint32_t> result;

    result.reserve(N + (rest.size() + ...));
    result.insert(result.end(), first.begin(), first.end());
    (result.insert(result.end(), rest.begin(), rest.end()), ...);
    return result;
}

void Scan::validate(const std::vector<Tensor> &input_tensors) const {
    const Tensor &input_tensor = input_tensors.at(0);

    const Shape &input_shape = input_tensor.get_legacy_shape();

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Scan: Expect input tensor to be stored on device.");
    TT_FATAL(input_tensor.buffer() != nullptr, "Scan: Expect input tensor to be allocated on a device buffer.");
    TT_FATAL(input_tensor.get_layout() == Layout::TILE, "Scan: Expect input tensor in tile layout.");
    TT_FATAL(input_tensor.is_sharded(), "Scan: Expect input tensor to be sharded.");

    auto core_grid = input_tensor.shard_spec()->grid;
    TT_FATAL(core_grid.ranges().size() == 1, "Scan: Expect input tensor to be sharded along a single core range.");
}

operation::ProgramWithCallbacks scan_impl(const Tensor &input, ScanOpDirection direction) {
    Program program = Program();
    tt_metal::Device *device = input.device();
    Buffer *src_buffer = input.buffer();
    auto all_cores = input.shard_spec()->grid;

    auto data_format = tt_metal::datatype_to_dataformat_converter(input.get_dtype());
    uint32_t tile_size = tt_metal::detail::TileSize(data_format);
    uint32_t total_tiles = input.shard_spec()->numel() / TILE_HW;
    uint32_t tiles_per_row = input.shard_spec()->shape[1] / TILE_WIDTH;
    uint32_t tiles_per_col = input.shard_spec()->shape[0] / TILE_HEIGHT;
    uint32_t reshapes_per_row = input.shard_spec()->shape[1] / TILE_HW;

    float one = 1.;
    uint32_t bf16_one_u32 = *reinterpret_cast<uint32_t *>(&one);

    auto ct_args = aggregate_arrays(
        create_cb<2>(program, all_cores, tile_size, total_tiles, data_format, {CB::c_in0, CB::c_out0}, src_buffer),
        create_cb(program, all_cores, tile_size, 32, data_format, {CB::c_intermed0}),
        create_cb<3>(
            program, all_cores, tile_size, 32, data_format, {CB::c_intermed1, CB::c_intermed2, CB::c_intermed3}),
        create_cb(program, all_cores, tile_size, 8, data_format, {CB::c_intermed4}),
        create_cb(program, all_cores, tile_size, 8, data_format, {CB::c_intermed5}),
        create_cb<2>(program, all_cores, tile_size, reshapes_per_row, data_format, {CB::c_intermed6, CB::c_intermed7}));
    ct_args.push_back(bf16_one_u32);

    std::vector<uint32_t> runtime_args = {tiles_per_row, tiles_per_col, reshapes_per_row, total_tiles};

    // Reader kernel
    tt_metal::KernelHandle reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/experimental/tt_dnn/op_library/scan/kernels/dataflow/reader_scan_sharded.cpp",
        all_cores,
        WriterDataMovementConfig(ct_args));
    SetRuntimeArgs(program, reader_kernel_id, all_cores, runtime_args);

    // Compute kernel
    tt_metal::KernelHandle compute_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/experimental/tt_dnn/op_library/scan/kernels/compute/untilize_scan_tilize.cpp",
        all_cores,
        tt_metal::ComputeConfig{.compile_args = ct_args});
    tt_metal::SetRuntimeArgs(program, compute_kernel_id, all_cores, runtime_args);

    auto override_runtime_args_callback = [](const void *operation,
                                             Program &program,
                                             const std::vector<Tensor> &input_tensors,
                                             const std::vector<std::optional<const Tensor>> &optional_input_tensors,
                                             const std::vector<Tensor> &output_tensors) {};

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}

operation::ProgramWithCallbacks Scan::create_program(
    const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const {
    const Tensor &input_tensor = input_tensors.at(0);

    return scan_impl(input_tensor, direction);
}

Tensor scan(Tensor &a) {
    uint32_t n_tile_columns = a.shard_spec()->shape[1] / TILE_WIDTH;
    operation::run(Scan{.n_tile_columns = n_tile_columns}, {a});
    return a;
}

}  // namespace tt::tt_metal

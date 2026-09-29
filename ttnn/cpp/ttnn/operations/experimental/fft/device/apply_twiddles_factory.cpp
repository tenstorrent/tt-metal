// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ApplyTwiddlesFactory implementation — see header for op semantics.
//
// Dispatch model: M = product(input.shape[:-1]) rows total (each row =
// N1 elements).  We split rows across the same multi-core grid that
// BatchedStockhamFactory uses (pow-2 num_cores, picked so num_cores | M).
// Each core processes `rows_per_core` consecutive rows starting at
// `base_row`; the reader's inner loop computes `tw_row = row % N2` to
// broadcast the right twiddle row.
//
// Same ROW_MAJOR safety as BatchedStockhamFactory: input/output buffers use
// TensorAccessor with the ttnn buffer's layout; twiddle buffers use
// tile-sized pages.

#include "apply_twiddles_factory.hpp"

#include <cstdint>
#include <memory>

#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "apply_twiddles_shared.hpp"
#include "stockham_host.hpp"  // pick_batch_grid, max_cores_for_grid, batch_logical_core

namespace ttnn::experimental::prim {

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace {

constexpr uint32_t kTileElems_at_f = 32u * 32u;  // 1024

constexpr bool is_pow2_at(uint32_t n) { return n != 0u && (n & (n - 1u)) == 0u; }

const KernelSpecName AT_READER{"reader"};
const TensorParamName AT_IN_R{"in_real"};
const TensorParamName AT_IN_I{"in_imag"};
const TensorParamName AT_TW_R{"tw_real"};
const TensorParamName AT_TW_I{"tw_imag"};

}  // namespace

ttnn::device_operation::ProgramArtifacts ApplyTwiddlesFactory::create_program_artifacts(
    const ApplyTwiddlesParams& attrs,
    const ApplyTwiddlesTensorArgs& tensor_args,
    std::tuple<ttnn::Tensor, ttnn::Tensor>& tensor_return_value) {
    using namespace tt::tt_metal::distributed;

    const auto& in_r_tensor = tensor_args.input_real;
    const auto& in_i_tensor = tensor_args.input_imag;
    const auto& out_r_tensor = std::get<0>(tensor_return_value);
    const auto& out_i_tensor = std::get<1>(tensor_return_value);

    const uint32_t N1 = attrs.N1;
    const uint32_t N2 = attrs.N2;

    TT_FATAL(
        is_pow2_at(N1) && N1 >= 2u && N1 <= kTileElems_at_f,
        "ApplyTwiddlesFactory: N1 must be pow-2 in [2, 1024] (got {}).",
        N1);
    TT_FATAL(
        is_pow2_at(N2) && N2 >= 1u && N2 <= kTileElems_at_f,
        "ApplyTwiddlesFactory: N2 must be pow-2 in [1, 1024] (got {}).",
        N2);

    // ── Resolve M = total row count (product of leading dims) ──────────
    const auto& shape = in_r_tensor.padded_shape();
    TT_FATAL(
        static_cast<uint32_t>(shape[-1]) == N1,
        "ApplyTwiddlesFactory: last dim ({}) must equal N1 ({}).",
        static_cast<uint32_t>(shape[-1]),
        N1);
    uint32_t M = 1u;
    for (int d = 0; d < static_cast<int>(shape.size()) - 1; ++d) {
        M *= static_cast<uint32_t>(shape[d]);
    }
    TT_FATAL(M % N2 == 0u, "ApplyTwiddlesFactory: row count M ({}) must be a multiple of N2 ({}).", M, N2);

    const DataType dtype = in_r_tensor.dtype();
    TT_FATAL(
        dtype == DataType::FLOAT32 || dtype == DataType::BFLOAT16,
        "ApplyTwiddlesFactory: only fp32 / bf16 supported (got dtype {}).",
        static_cast<int>(dtype));
    TT_FATAL(in_i_tensor.dtype() == dtype, "ApplyTwiddlesFactory: input_real and input_imag dtypes must match.");
    const bool is_bf16 = (dtype == DataType::BFLOAT16);

    TT_FATAL(
        in_r_tensor.buffer() && in_i_tensor.buffer() && out_r_tensor.buffer() && out_i_tensor.buffer(),
        "ApplyTwiddlesFactory: all input/output tensors must be on device.");

    // ── MeshDevice (no-op deleter — tensor owns lifetime) ──────────────
    auto* device_raw = in_r_tensor.device();
    auto md = device_raw->get_mesh_device();

    // Twiddle table comes in as a declared input (see the tensor-args header
    // for why the factory cannot fetch it from the cache itself).
    const auto& tw_r_tensor = tensor_args.tw_real;
    const auto& tw_i_tensor = tensor_args.tw_imag;

    // ── Pick core grid: pow-2 num_cores that divides M ─────────────────
    const auto dev_grid = md->compute_with_storage_grid_size();
    const uint32_t max_cores = fft_stockham::max_cores_for_grid(dev_grid.x, dev_grid.y);
    uint32_t num_cores = (M < max_cores) ? M : max_cores;
    while (num_cores > 1u && (M % num_cores) != 0u) {
        num_cores >>= 1;  // halve until it divides
    }
    TT_FATAL(num_cores >= 1u && (M % num_cores) == 0u, "ApplyTwiddlesFactory: failed to pick num_cores for M={}.", M);
    const uint32_t rows_per_core = M / num_cores;
    auto [grid_cols, grid_rows] = fft_stockham::pick_batch_grid(num_cores, dev_grid.x);

    const CoreCoord first{0, 0};
    const CoreCoord last{grid_cols - 1u, grid_rows - 1u};
    const CoreRange cr(first, last);
    const CoreRangeSet crs({cr});

    namespace shared = apply_tw_shared;

    // The twiddle table is a device tensor owned by the per-device cache,
    // so it binds as an ordinary tensor parameter; only its row modulus N2
    // varies per launch and stays a runtime arg.
    KernelSpec reader{
        .unique_id = AT_READER,
        .source = "ttnn/cpp/ttnn/operations/experimental/fft/device/kernels/dataflow/apply_twiddles_reader.cpp",
        .compiler_options = {.defines = shared::reader_defines(is_bf16)},
        .dfb_bindings = shared::reader_dfb_bindings(is_bf16),
        .tensor_bindings =
            {TensorBinding{.tensor_parameter_name = AT_IN_R, .accessor_name = "in_r"},
             TensorBinding{.tensor_parameter_name = AT_IN_I, .accessor_name = "in_i"},
             TensorBinding{.tensor_parameter_name = AT_TW_R, .accessor_name = "tw_r"},
             TensorBinding{.tensor_parameter_name = AT_TW_I, .accessor_name = "tw_i"}},
        .compile_time_args = {{"n1", N1}},
        .runtime_arg_schema = {.runtime_arg_names = {"base_row", "num_rows", "n2"}},
        .hw_config = ttnn::create_reader_datamovement_config(device_raw->arch()),
    };

    KernelSpec writer = shared::make_writer(device_raw->arch(), N1, is_bf16);
    KernelSpec compute = shared::make_compute();

    KernelRunArgs reader_run_args{.kernel = AT_READER};
    KernelRunArgs writer_run_args{.kernel = shared::WRITER};
    KernelRunArgs compute_run_args{.kernel = shared::COMPUTE};

    for (uint32_t c = 0; c < num_cores; ++c) {
        const CoreCoord logical = fft_stockham::batch_logical_core(c, grid_cols);
        const uint32_t base = c * rows_per_core;

        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            logical,
            {{"base_row", base}, {"num_rows", rows_per_core}, {"n2", N2}});
        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values, logical, {{"base_row", base}, {"num_rows", rows_per_core}});
        AddRuntimeArgsForNode(compute_run_args.runtime_arg_values, logical, {{"num_tiles", rows_per_core}});
    }

    ProgramSpec spec{
        .name = "fft_apply_twiddles",
        .kernels = {std::move(reader), std::move(writer), std::move(compute)},
        .dataflow_buffers = shared::make_dataflow_buffers(is_bf16),
        .tensor_parameters =
            {TensorParameter{.unique_id = AT_IN_R, .spec = in_r_tensor.tensor_spec()},
             TensorParameter{.unique_id = AT_IN_I, .spec = in_i_tensor.tensor_spec()},
             TensorParameter{.unique_id = AT_TW_R, .spec = tw_r_tensor.tensor_spec()},
             TensorParameter{.unique_id = AT_TW_I, .spec = tw_i_tensor.tensor_spec()},
             TensorParameter{.unique_id = shared::OUT_R, .spec = out_r_tensor.tensor_spec()},
             TensorParameter{.unique_id = shared::OUT_I, .spec = out_i_tensor.tensor_spec()}},
        .work_units =
            {WorkUnitSpec{
                .name = "main",
                .kernels = {AT_READER, shared::WRITER, shared::COMPUTE},
                .target_nodes = crs,
            }},
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {
        std::move(reader_run_args), std::move(writer_run_args), std::move(compute_run_args)};
    run_args.tensor_args = {
        {AT_IN_R, TensorArgument{in_r_tensor.mesh_tensor()}},
        {AT_IN_I, TensorArgument{in_i_tensor.mesh_tensor()}},
        {AT_TW_R, TensorArgument{tw_r_tensor.mesh_tensor()}},
        {AT_TW_I, TensorArgument{tw_i_tensor.mesh_tensor()}},
        {shared::OUT_R, TensorArgument{out_r_tensor.mesh_tensor()}},
        {shared::OUT_I, TensorArgument{out_i_tensor.mesh_tensor()}},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::experimental::prim

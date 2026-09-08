// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ApplyTwiddlesXlFactory implementation.  Same dispatch / CB layout as
// ApplyTwiddlesFactory (so apply_twiddles_compute + apply_twiddles_writer
// can be reused verbatim); the only difference is the reader kernel,
// which builds the twiddle row in L1 from a per-(device, big_modulus,
// full_N) cached delta table (apply_twiddles_xl_host).
//
// Why a separate factory (not a flag on ApplyTwiddlesFactory)?
//   - Different runtime args + different reader compile-time args.
//   - Different cached host buffer (delta vs full twiddle table).
//   - Different validation envelope (twiddle_N2 cap is the whole reason
//     this op exists).  Keeping them separate preserves the original op's
//     simplicity and program-cache identity.

#include "apply_twiddles_xl_factory.hpp"

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

// `_xl` suffix avoids Unity-build ODR collision with anonymous-namespace
// symbols in apply_twiddles_factory.cpp / batched_stockham_factory.cpp.
constexpr uint32_t kTileElems_xl = 32u * 32u;  // 1024

constexpr bool is_pow2_xl(uint32_t n) { return n != 0u && (n & (n - 1u)) == 0u; }

const KernelSpecName XL_READER{"reader"};
const TensorParamName XL_IN_R{"in_real"};
const TensorParamName XL_IN_I{"in_imag"};
const TensorParamName XL_D_R{"delta_real"};
const TensorParamName XL_D_I{"delta_imag"};

}  // namespace

ttnn::device_operation::ProgramArtifacts ApplyTwiddlesXlFactory::create_program_artifacts(
    const ApplyTwiddlesXlParams& attrs,
    const ApplyTwiddlesXlTensorArgs& tensor_args,
    std::tuple<ttnn::Tensor, ttnn::Tensor>& tensor_return_value) {
    const auto& in_r_tensor = tensor_args.input_real;
    const auto& in_i_tensor = tensor_args.input_imag;
    const auto& out_r_tensor = std::get<0>(tensor_return_value);
    const auto& out_i_tensor = std::get<1>(tensor_return_value);

    const uint32_t P = attrs.P;
    const uint32_t big_modulus = attrs.big_modulus;
    const uint32_t full_N = attrs.full_N;

    TT_FATAL(
        is_pow2_xl(P) && P >= 2u && P <= kTileElems_xl,
        "ApplyTwiddlesXlFactory: P must be pow-2 in [2, 1024] (got {}).",
        P);
    TT_FATAL(
        is_pow2_xl(big_modulus) && big_modulus >= 1u,
        "ApplyTwiddlesXlFactory: big_modulus must be pow-2 and >= 1 (got {}).",
        big_modulus);
    TT_FATAL(
        is_pow2_xl(full_N) && full_N >= big_modulus,
        "ApplyTwiddlesXlFactory: full_N must be pow-2 and >= big_modulus "
        "(got full_N={} big_modulus={}).",
        full_N,
        big_modulus);

    // M = total rows.
    const auto& shape = in_r_tensor.padded_shape();
    TT_FATAL(
        static_cast<uint32_t>(shape[-1]) == P,
        "ApplyTwiddlesXlFactory: input last dim ({}) must equal P ({}).",
        static_cast<uint32_t>(shape[-1]),
        P);
    uint32_t M = 1u;
    for (int d = 0; d < static_cast<int>(shape.size()) - 1; ++d) {
        M *= static_cast<uint32_t>(shape[d]);
    }
    TT_FATAL(
        M >= 1u && (M % big_modulus) == 0u,
        "ApplyTwiddlesXlFactory: row count M ({}) must be a multiple of "
        "big_modulus ({}).",
        M,
        big_modulus);

    const DataType dtype = in_r_tensor.dtype();
    TT_FATAL(
        dtype == DataType::FLOAT32 || dtype == DataType::BFLOAT16,
        "ApplyTwiddlesXlFactory: only fp32 / bf16 supported (got dtype {}).",
        static_cast<int>(dtype));
    TT_FATAL(in_i_tensor.dtype() == dtype, "ApplyTwiddlesXlFactory: input_real and input_imag dtypes must match.");
    const bool is_bf16 = (dtype == DataType::BFLOAT16);

    TT_FATAL(
        in_r_tensor.buffer() && in_i_tensor.buffer() && out_r_tensor.buffer() && out_i_tensor.buffer(),
        "ApplyTwiddlesXlFactory: all input/output tensors must be on device.");

    auto* device_raw = in_r_tensor.device();
    auto md = device_raw->get_mesh_device();

    // Delta table comes in as a declared input (see the tensor-args header
    // for why the factory cannot fetch it from the cache itself).
    const auto& d_r_tensor = tensor_args.delta_real;
    const auto& d_i_tensor = tensor_args.delta_imag;

    // ── Pick core grid: pow-2 num_cores dividing M (matches apply_twiddles).
    const auto dev_grid = md->compute_with_storage_grid_size();
    const uint32_t max_cores = fft_stockham::max_cores_for_grid(dev_grid.x, dev_grid.y);
    uint32_t num_cores = (M < max_cores) ? M : max_cores;
    while (num_cores > 1u && (M % num_cores) != 0u) {
        num_cores >>= 1;
    }
    TT_FATAL(num_cores >= 1u && (M % num_cores) == 0u, "ApplyTwiddlesXlFactory: failed to pick num_cores for M={}.", M);
    const uint32_t rows_per_core = M / num_cores;
    auto [grid_cols, grid_rows] = fft_stockham::pick_batch_grid(num_cores, dev_grid.x);

    const CoreCoord first{0, 0};
    const CoreCoord last{grid_cols - 1u, grid_rows - 1u};
    const CoreRange cr(first, last);
    const CoreRangeSet crs({cr});

    namespace shared = apply_tw_shared;

    // Writer and compute are the same specs apply_twiddles uses; only the
    // reader differs, deriving each twiddle row from the cached delta
    // table instead of reading a precomputed one.
    KernelSpec reader{
        .unique_id = XL_READER,
        .source = "ttnn/cpp/ttnn/operations/experimental/fft/device/kernels/dataflow/apply_twiddles_xl_reader.cpp",
        .compiler_options = {.defines = shared::reader_defines(is_bf16)},
        .dfb_bindings = shared::reader_dfb_bindings(is_bf16),
        .tensor_bindings =
            {TensorBinding{.tensor_parameter_name = XL_IN_R, .accessor_name = "in_r"},
             TensorBinding{.tensor_parameter_name = XL_IN_I, .accessor_name = "in_i"},
             TensorBinding{.tensor_parameter_name = XL_D_R, .accessor_name = "d_r"},
             TensorBinding{.tensor_parameter_name = XL_D_I, .accessor_name = "d_i"}},
        .compile_time_args = {{"p", P}},
        .runtime_arg_schema = {.runtime_arg_names = {"base_row", "num_rows", "big_modulus"}},
        .hw_config = ttnn::create_reader_datamovement_config(device_raw->arch()),
    };

    KernelSpec writer = shared::make_writer(device_raw->arch(), P, is_bf16);
    KernelSpec compute = shared::make_compute();

    KernelRunArgs reader_run_args{.kernel = XL_READER};
    KernelRunArgs writer_run_args{.kernel = shared::WRITER};
    KernelRunArgs compute_run_args{.kernel = shared::COMPUTE};

    for (uint32_t c = 0; c < num_cores; ++c) {
        const CoreCoord logical = fft_stockham::batch_logical_core(c, grid_cols);
        const uint32_t base = c * rows_per_core;

        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            logical,
            {{"base_row", base}, {"num_rows", rows_per_core}, {"big_modulus", big_modulus}});
        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values, logical, {{"base_row", base}, {"num_rows", rows_per_core}});
        AddRuntimeArgsForNode(compute_run_args.runtime_arg_values, logical, {{"num_tiles", rows_per_core}});
    }

    ProgramSpec spec{
        .name = "fft_apply_twiddles_xl",
        .kernels = {std::move(reader), std::move(writer), std::move(compute)},
        .dataflow_buffers = shared::make_dataflow_buffers(is_bf16),
        .tensor_parameters =
            {TensorParameter{.unique_id = XL_IN_R, .spec = in_r_tensor.tensor_spec()},
             TensorParameter{.unique_id = XL_IN_I, .spec = in_i_tensor.tensor_spec()},
             TensorParameter{.unique_id = XL_D_R, .spec = d_r_tensor.tensor_spec()},
             TensorParameter{.unique_id = XL_D_I, .spec = d_i_tensor.tensor_spec()},
             TensorParameter{.unique_id = shared::OUT_R, .spec = out_r_tensor.tensor_spec()},
             TensorParameter{.unique_id = shared::OUT_I, .spec = out_i_tensor.tensor_spec()}},
        .work_units =
            {WorkUnitSpec{
                .name = "main",
                .kernels = {XL_READER, shared::WRITER, shared::COMPUTE},
                .target_nodes = crs,
            }},
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {
        std::move(reader_run_args), std::move(writer_run_args), std::move(compute_run_args)};
    run_args.tensor_args = {
        {XL_IN_R, TensorArgument{in_r_tensor.mesh_tensor()}},
        {XL_IN_I, TensorArgument{in_i_tensor.mesh_tensor()}},
        {XL_D_R, TensorArgument{d_r_tensor.mesh_tensor()}},
        {XL_D_I, TensorArgument{d_i_tensor.mesh_tensor()}},
        {shared::OUT_R, TensorArgument{out_r_tensor.mesh_tensor()}},
        {shared::OUT_I, TensorArgument{out_i_tensor.mesh_tensor()}},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::experimental::prim

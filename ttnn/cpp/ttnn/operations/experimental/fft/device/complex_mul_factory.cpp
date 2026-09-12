// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ComplexMulFactory implementation.  Same CB layout and writer/compute
// kernels as apply_twiddles_xl; the only difference is the reader
// kernel (complex_mul_reader.cpp) which reads BOTH input complex
// tensors A and B from DRAM (no on-the-fly twiddle generation).

#include "complex_mul_factory.hpp"

#include <cstdint>

#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "apply_twiddles_shared.hpp"
#include "stockham_host.hpp"  // pick_batch_grid, max_cores_for_grid, batch_logical_core

namespace ttnn::experimental::prim {

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace {

// `_cm` suffix avoids Unity-build ODR collision with anonymous-namespace
// symbols in apply_twiddles_factory.cpp / apply_twiddles_xl_factory.cpp.
const KernelSpecName CM_READER{"reader"};
const TensorParamName CM_A_R{"a_real"};
const TensorParamName CM_A_I{"a_imag"};
const TensorParamName CM_B_R{"b_real"};
const TensorParamName CM_B_I{"b_imag"};

}  // namespace

ttnn::device_operation::ProgramArtifacts ComplexMulFactory::create_program_artifacts(
    const ComplexMulParams& /*attrs*/,
    const ComplexMulTensorArgs& tensor_args,
    std::tuple<ttnn::Tensor, ttnn::Tensor>& tensor_return_value) {
    const auto& a_r_tensor = tensor_args.a_real;
    const auto& a_i_tensor = tensor_args.a_imag;
    const auto& b_r_tensor = tensor_args.b_real;
    const auto& b_i_tensor = tensor_args.b_imag;
    const auto& out_r_tensor = std::get<0>(tensor_return_value);
    const auto& out_i_tensor = std::get<1>(tensor_return_value);

    // M = total row count = product of all dims except the last.  P =
    // last dim = row length.  Both already validated in the device op.
    const auto& shape = a_r_tensor.padded_shape();
    const uint32_t P = static_cast<uint32_t>(shape[-1]);
    uint32_t M = 1u;
    for (int d = 0; d < static_cast<int>(shape.size()) - 1; ++d) {
        M *= static_cast<uint32_t>(shape[d]);
    }
    TT_FATAL(M >= 1u, "ComplexMulFactory: M must be >= 1 (got {}).", M);

    const DataType dtype = a_r_tensor.dtype();
    const bool is_bf16 = (dtype == DataType::BFLOAT16);

    TT_FATAL(
        a_r_tensor.buffer() && a_i_tensor.buffer() && b_r_tensor.buffer() && b_i_tensor.buffer() &&
            out_r_tensor.buffer() && out_i_tensor.buffer(),
        "ComplexMulFactory: all input/output tensors must be on device.");

    auto* device_raw = a_r_tensor.device();

    // ── Pick core grid: pow-2 num_cores dividing M.
    //   Unlike apply_twiddles[_xl] which guarantees M is a pow-2 multiple
    //   of big_modulus, complex_mul accepts arbitrary M (the chirp
    //   pre-multiply in Bluestein has last-dim P = N which can be any
    //   length).  We must therefore (a) FLOOR num_cores to a pow-2 first
    //   (else for M=37 we'd hit `num_cores = 37`, a non-pow-2 → invalid
    //   batch grid → dispatch-core placement TT_FATAL), then (b) shrink
    //   that pow-2 until it divides M.  Worst case (M odd prime)
    //   collapses to num_cores=1, which is correct.
    const auto dev_grid = device_raw->compute_with_storage_grid_size();
    const uint32_t max_cores = fft_stockham::max_cores_for_grid(dev_grid.x, dev_grid.y);
    const uint32_t cap = (M < max_cores) ? M : max_cores;
    uint32_t num_cores = 1u;
    while ((num_cores << 1) <= cap) {
        num_cores <<= 1;
    }
    while (num_cores > 1u && (M % num_cores) != 0u) {
        num_cores >>= 1;
    }
    TT_FATAL(num_cores >= 1u && (M % num_cores) == 0u, "ComplexMulFactory: failed to pick num_cores for M={}.", M);
    const uint32_t rows_per_core = M / num_cores;
    auto [grid_cols, grid_rows] = fft_stockham::pick_batch_grid(num_cores, dev_grid.x);

    const CoreCoord first{0, 0};
    const CoreCoord last{grid_cols - 1u, grid_rows - 1u};
    const CoreRange cr(first, last);
    const CoreRangeSet crs({cr});

    namespace shared = apply_tw_shared;

    // The reader feeds the second operand into the twiddle slots, so the
    // shared compute kernel's "input × twiddle" is exactly A × B here.
    // For bf16 it reuses one pair of staging buffers for both operands
    // (read A → expand into a_r/a_i, then read B → expand into t_r/t_i).
    KernelSpec reader{
        .unique_id = CM_READER,
        .source = "ttnn/cpp/ttnn/operations/experimental/fft/device/kernels/dataflow/complex_mul_reader.cpp",
        .compiler_options = {.defines = shared::reader_defines(is_bf16)},
        .dfb_bindings = shared::reader_dfb_bindings(is_bf16),
        .tensor_bindings =
            {TensorBinding{.tensor_parameter_name = CM_A_R, .accessor_name = "a_r"},
             TensorBinding{.tensor_parameter_name = CM_A_I, .accessor_name = "a_i"},
             TensorBinding{.tensor_parameter_name = CM_B_R, .accessor_name = "b_r"},
             TensorBinding{.tensor_parameter_name = CM_B_I, .accessor_name = "b_i"}},
        .compile_time_args = {{"p", P}},
        .runtime_arg_schema = {.runtime_arg_names = {"base_row", "num_rows"}},
        .hw_config = ttnn::create_reader_datamovement_config(device_raw->arch()),
    };

    KernelSpec writer = shared::make_writer(device_raw->arch(), P, is_bf16);
    KernelSpec compute = shared::make_compute();

    KernelRunArgs reader_run_args{.kernel = CM_READER};
    KernelRunArgs writer_run_args{.kernel = shared::WRITER};
    KernelRunArgs compute_run_args{.kernel = shared::COMPUTE};

    for (uint32_t c = 0; c < num_cores; ++c) {
        const CoreCoord logical = fft_stockham::batch_logical_core(c, grid_cols);
        const uint32_t base = c * rows_per_core;

        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values, logical, {{"base_row", base}, {"num_rows", rows_per_core}});
        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values, logical, {{"base_row", base}, {"num_rows", rows_per_core}});
        AddRuntimeArgsForNode(compute_run_args.runtime_arg_values, logical, {{"num_tiles", rows_per_core}});
    }

    ProgramSpec spec{
        .name = "fft_complex_mul",
        .kernels = {std::move(reader), std::move(writer), std::move(compute)},
        .dataflow_buffers = shared::make_dataflow_buffers(is_bf16),
        .tensor_parameters =
            {TensorParameter{.unique_id = CM_A_R, .spec = a_r_tensor.tensor_spec()},
             TensorParameter{.unique_id = CM_A_I, .spec = a_i_tensor.tensor_spec()},
             TensorParameter{.unique_id = CM_B_R, .spec = b_r_tensor.tensor_spec()},
             TensorParameter{.unique_id = CM_B_I, .spec = b_i_tensor.tensor_spec()},
             TensorParameter{.unique_id = shared::OUT_R, .spec = out_r_tensor.tensor_spec()},
             TensorParameter{.unique_id = shared::OUT_I, .spec = out_i_tensor.tensor_spec()}},
        .work_units =
            {WorkUnitSpec{
                .name = "main",
                .kernels = {CM_READER, shared::WRITER, shared::COMPUTE},
                .target_nodes = crs,
            }},
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {
        std::move(reader_run_args), std::move(writer_run_args), std::move(compute_run_args)};
    run_args.tensor_args = {
        {CM_A_R, TensorArgument{a_r_tensor.mesh_tensor()}},
        {CM_A_I, TensorArgument{a_i_tensor.mesh_tensor()}},
        {CM_B_R, TensorArgument{b_r_tensor.mesh_tensor()}},
        {CM_B_I, TensorArgument{b_i_tensor.mesh_tensor()}},
        {shared::OUT_R, TensorArgument{out_r_tensor.mesh_tensor()}},
        {shared::OUT_I, TensorArgument{out_i_tensor.mesh_tensor()}},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::experimental::prim

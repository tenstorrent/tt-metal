// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/eltwise/binary_ng/device/programs/binary_ng_dram_optimized_factory.hpp"
#include "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/programs/binary_ng_program_factory_utils.hpp"
#include "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_utils.hpp"
#include "ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/cpp/ttnn/operations/eltwise/binary/common/binary_op_utils.hpp"
#include "ttnn/cpp/ttnn/operations/cb_utils.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include <tt-metalium/work_split.hpp>

#include <tt-metalium/dispatch_core_query.hpp>

#include <cstdlib>
#include <limits>
#include <optional>
#include <unordered_set>
#include <algorithm>

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

// TODO: Move to utilities?
uint32_t get_noc_max_burst_size(const ttnn::MeshDevice& mesh_device) {
    // TODO: What about QUASAR?
    uint32_t noc_max_burst_size;
    const auto arch = mesh_device.arch();
    if (arch == tt::ARCH::BLACKHOLE) {
        noc_max_burst_size = 16384;
        // ttsim doesn't support 16384 bytes burst size( ERROR: UnimplementedFunctionality: tensix_execute_elw_op:
        // dst_row=1024)
        noc_max_burst_size /= 2;
    } else if (arch == tt::ARCH::WORMHOLE_B0) {
        noc_max_burst_size = 8192;
    } else {
        TT_THROW("Unsupported architecture for zero-init: {}", arch);
    }
    return noc_max_burst_size;
}

uint32_t compute_num_batches(const ttnn::operations::binary_ng::BinaryNgInputs& args) {
    (void)args;
    return 1;
}

uint32_t compute_num_tiles_per_batches(
    const ttnn::operations::binary_ng::BinaryNgParams& operation_attributes,
    const ttnn::operations::binary_ng::BinaryNgInputs& args,
    const ttnn::Tensor& output) {
    auto* device = args.input_tensor_a.device();
    auto dtype = tt::tt_metal::datatype_to_dataformat_converter(args.input_tensor_a.dtype());

    uint32_t single_tile_size = tt::tile_size(dtype);

    const auto b_data_format =
        args.input_tensor_b.has_value() ? datatype_to_dataformat_converter(args.input_tensor_b->dtype()) : dtype;
    const auto c_data_format = datatype_to_dataformat_converter(output.dtype());

    // With fp32_dest_acc_en the DST register file holds only 4 tiles (vs 16 for 16-bit).
    // The SFPU binary section interleaves LHS/RHS in DST using 2*n_tiles slots,
    // so large_chunk (= num_batches * num_tiles_per_batch) must stay <= 2 for 32-bit.
    // num_tiles_per_batch = 4 supposed to match NOC_MAX_BURST_SIZE (bytes)
    bool fp32_dest_acc_en =
        ttnn::operations::binary_ng::program::is_fp32_dest_acc_en(dtype, b_data_format, c_data_format);

    return fp32_dest_acc_en or operation_attributes.is_sfpu
               ? 1
               : CMAKE_UNIQUE_NAMESPACE::get_noc_max_burst_size(*(device->get_mesh_device())) / (single_tile_size);
}

template <bool initialize_args>
inline void set_eltwise_binary_runtime_args_for_dram_cores(
    tt::tt_metal::Program& program,
    const ttnn::Tensor& a_tensor,
    const ttnn::Tensor& b_tensor,
    const ttnn::Tensor& output,
    const tt::tt_metal::KernelHandle reader_kernel_id,
    const tt::tt_metal::KernelHandle writer_kernel_id,
    const tt::tt_metal::KernelHandle compute_kernel_id,
    const CoreRangeSet& all_device_cores,
    const uint32_t num_batches,
    const uint32_t num_tiles_per_batch) {
    using namespace tt;
    using namespace tt::tt_metal;
    using namespace tt::constants;

    uint32_t num_tiles = static_cast<uint32_t>(a_tensor.physical_volume() / TILE_HW);

    bool row_major = true;  // TODO: make this configurable
    uint32_t num_cores_total = all_device_cores.num_cores();

    // TODO: Move FATAL to validate function?
    TT_FATAL(
        a_tensor.padded_shape()[-1] % tt::constants::TILE_HEIGHT == 0,
        "num_tiles mismatch, {} % {} != 0",
        a_tensor.padded_shape()[-1],
        tt::constants::TILE_HEIGHT);
    TT_FATAL(
        a_tensor.padded_shape()[-2] % tt::constants::TILE_WIDTH == 0,
        "num_tiles mismatch, {} % {} != 0",
        a_tensor.padded_shape()[-2],
        tt::constants::TILE_WIDTH);

    // vector of cores
    auto cores = corerange_to_cores(all_device_cores, std::nullopt, row_major);

    std::vector<std::vector<uint32_t>> reader_args_array{cores.size()};
    std::vector<std::vector<uint32_t>> compute_args_array{cores.size()};
    std::vector<std::vector<uint32_t>> writer_args_array{cores.size()};

    auto& cached_reader_args = GetRuntimeArgs(program, reader_kernel_id);
    auto& cached_eltwise_args = GetRuntimeArgs(program, compute_kernel_id);
    auto& cached_writer_args = GetRuntimeArgs(program, writer_kernel_id);

    std::vector<uint32_t> core_ids;
    for (uint32_t core_id = 0; core_id < num_cores_total; ++core_id) {
        const CoreCoord& core = cores.at(core_id);

        uint32_t num_tiles_per_core = num_tiles / num_cores_total + (core_id < num_tiles % num_cores_total ? 1 : 0);

        if constexpr (!initialize_args) {
            // RuntimeArgsData
            auto& reader_args = cached_reader_args.at(core.x).at(core.y);
            reader_args[2] = 0;
            auto& eltwise_args = cached_eltwise_args.at(core.x).at(core.y);
            eltwise_args[0] = 0;
            auto& writer_args = cached_writer_args.at(core.x).at(core.y);
            writer_args[1] = 0;
        }

        uint32_t vc = core_id & 0x3;
        core_ids.push_back(core_id);
        for (uint32_t j = 0; j < core_id; ++j) {
            auto core_ = cores[j];

            if (core_.y == core.y and ((core_id & 0x3) == (core_ids[j] & 0x3))) {  // same vc and same row
                vc = (vc + 1) & 0x3;
                break;
            }
        }

        std::vector<uint32_t> reader_args_vec = {
            a_tensor.buffer()->address(),
            b_tensor.buffer()->address(),
            core_id,
            num_tiles_per_core,
            num_tiles_per_batch,
            num_batches};

        std::vector<uint32_t> compute_args_vec = {
            num_tiles_per_core,
            num_tiles_per_batch,
            num_batches,
        };
        std::vector<uint32_t> writer_args_vec = {
            output.buffer()->address(), core_id, num_tiles_per_core, num_tiles_per_batch, num_batches};

        reader_args_array[core_id] = std::move(reader_args_vec);
        compute_args_array[core_id] = std::move(compute_args_vec);
        writer_args_array[core_id] = std::move(writer_args_vec);
        if constexpr (!initialize_args) {
            auto& core_reader_args = cached_reader_args.at(core.x).at(core.y);
            std::ranges::copy(reader_args_array[core_id], core_reader_args.data());

            auto& core_eltwise_args = cached_eltwise_args.at(core.x).at(core.y);
            std::ranges::copy(compute_args_array[core_id], core_eltwise_args.data());

            auto& core_writer_args = cached_writer_args.at(core.x).at(core.y);
            std::ranges::copy(writer_args_array[core_id], core_writer_args.data());
        }
    }

    if constexpr (initialize_args) {
        SetRuntimeArgs(program, reader_kernel_id, cores, reader_args_array);
        SetRuntimeArgs(program, compute_kernel_id, cores, compute_args_array);
        SetRuntimeArgs(program, writer_kernel_id, cores, writer_args_array);
    }
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

namespace ttnn::operations::binary_ng::program {

const std::string kernel_prefix = "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/";

std::optional<std::string> BinaryNgDramOptimizedProgram::validate_program(
    const BinaryNgParams& operation_attributes, const BinaryNgInputs& tensor_args) {
    const auto& a = tensor_args.input_tensor_a;

    if (operation_attributes.subtile_broadcast_type != SubtileBroadcastType::NONE) {
        return "Subtile broadcast type is not supported for DRAM optimized program";
    }

    if (operation_attributes.dtype.has_value() && operation_attributes.dtype.value() != a.dtype()) {
        return "Output dtype mismatch";
    }
    auto* device = a.device();
    if (device->get_active_sub_device_manager_id() != device->get_default_sub_device_manager_id()) {
        return "Sub-device manager id mismatch. Buffer sub-device manager id: {}, Device active sub-device manager id";
    }

    /*
    The problem: that "optimal DRAM bank → logical worker" mapping is a fixed physical layout. When dispatch_core_axis =
    COL, the fast‑dispatch firmware reserves a column of Tensix cores; if any of the DRAM‑bank‑adjacent workers happen
    to land in that reserved column, the subsequent CreateKernel placement collides with a dispatch core and
    ProgramImpl::compile fatally asserts. In the ROW configuration those same workers are outside the dispatch row, so
    the identical program compiles fine. That's exactly why only the COL parametrization fails.
    */
    if (!operation_attributes.memory_config.is_dram()) {
        return "Memory config must be DRAM";
    }
    if (!a.memory_config().is_dram()) {
        return "Input tensor A must be on DRAM";
    }
    if (!tensor_args.input_tensor_b.has_value()) {
        return "Input tensor B must be set";
    }
    const auto& b = tensor_args.input_tensor_b.value();
    if (!b.memory_config().is_dram()) {
        return "Input tensor B must be on DRAM";
    }

    if (a.dtype() != b.dtype()) {
        return "Input tensors A and B must have the same dtype";
    }
    if (a.layout() != Layout::TILE) {
        return "Input tensor A must be in tile layout";
    }
    if (b.layout() != Layout::TILE) {
        return "Input tensor B must be in tile layout";
    }
    if (a.padded_shape() != b.padded_shape()) {
        return "Input tensors A and B must have the same padded shape";
    }

    if (!operation_attributes.rhs_activations.empty()) {
        return "Right hand side activations are not supported";
    }
    if (!operation_attributes.lhs_activations.empty()) {
        return "Left hand side activations are not supported";
    }
    if (!operation_attributes.post_activations.empty()) {
        return "Post activations are not supported";
    }

    if (operation_attributes.sub_core_grids.has_value()) {
        return "Sub core grids are not supported for DRAM optimized program";
    }

    // TODO: Fow now add support for interleaved tensors only
    if (a.memory_config().is_sharded() || b.memory_config().is_sharded()) {
        return "Sharded memory is not supported for DRAM optimized program";
    }
    // if ((a.memory_config().is_sharded() && !b.memory_config().is_sharded()) ||
    //     (!a.memory_config().is_sharded() && b.memory_config().is_sharded())) {
    //     return "Input tensors A and B must be either both sharded or both interleaved";
    // }

    // if (a.memory_config().is_sharded() && b.memory_config().is_sharded()) {
    //     if (a.memory_config().shard_spec().has_value() && b.memory_config().shard_spec().has_value()) {
    //         if (a.memory_config().shard_spec()->grid != b.memory_config().shard_spec()->grid) {
    //             return "Input tensors A and B must have the same shard spec";
    //         }
    //     } else if (a.memory_config().nd_shard_spec().has_value() && b.memory_config().nd_shard_spec().has_value()) {
    //         if (a.memory_config().nd_shard_spec()->grid != b.memory_config().nd_shard_spec()->grid) {
    //             return "Input tensors A and B must have the same ND shard spec";
    //         }
    //     }
    // }

    if (operation_attributes.is_where_op) {
        return "Where op is not supported for DRAM optimized program";
    }
    return std::nullopt;
}

BinaryNgDramOptimizedProgram::cached_program_t BinaryNgDramOptimizedProgram::create(
    const BinaryNgParams& operation_attributes, const BinaryNgInputs& args, Tensor& output) {
    auto op_type = operation_attributes.binary_op_type;
    const bool is_sfpu_op = operation_attributes.is_sfpu;
    const bool is_quant_op = operation_attributes.is_quant_op;
    // const bool is_where_op = operation_attributes.is_where_op;

    using namespace tt;
    using namespace tt::tt_metal;

    auto* device = args.input_tensor_a.device();

    auto all_worker_cores_ordered =
        device->get_optimal_dram_bank_to_logical_worker_assignment(tt::tt_metal::NOC::NOC_0);

    // The DRAM-bank-adjacent worker list is a fixed physical layout. Depending on DispatchCoreAxis
    // (ROW vs. COL), some of those cores can be reserved as dispatch cores and are therefore both
    // illegal for kernel placement and outside the compute-with-storage grid. Build the set of
    // available compute cores from compute_with_storage_grid_size() (which already excludes
    // dispatch-reserved rows/columns), optionally subtract any additional known dispatch cores,
    // and, for any DRAM-bank worker that is not usable or is a duplicate, substitute the nearest
    // free worker core so that every DRAM bank still has a 1:1 core assignment.
    const auto compute_grid = device->compute_with_storage_grid_size();
    const auto& dispatch_cores = tt::tt_metal::get_logical_dispatch_cores_on_user_chips();

    std::vector<CoreCoord> available_workers_vec;
    available_workers_vec.reserve(compute_grid.x * compute_grid.y);
    for (uint32_t y = 0; y < compute_grid.y; ++y) {
        for (uint32_t x = 0; x < compute_grid.x; ++x) {
            CoreCoord w{x, y};
            if (std::ranges::find(dispatch_cores, w) == dispatch_cores.end()) {
                available_workers_vec.push_back(w);
            }
        }
    }
    std::unordered_set<CoreCoord> available_set(available_workers_vec.begin(), available_workers_vec.end());

    auto find_nearest_free_worker = [&](const CoreCoord& target,
                                        const std::unordered_set<CoreCoord>& used) -> std::optional<CoreCoord> {
        std::optional<CoreCoord> best;
        int best_dist = std::numeric_limits<int>::max();
        int best_dx = std::numeric_limits<int>::max();
        for (const auto& w : available_workers_vec) {
            if (used.contains(w)) {
                continue;
            }
            int dx = std::abs(static_cast<int>(w.x) - static_cast<int>(target.x));
            int dy = std::abs(static_cast<int>(w.y) - static_cast<int>(target.y));
            int dist = dx + dy;
            // Prefer the closest core; on ties prefer the one with the smallest column delta
            // (DRAM banks run along columns on WH/BH, so staying in the same row preserves locality).
            if (dist < best_dist || (dist == best_dist && dx < best_dx)) {
                best_dist = dist;
                best_dx = dx;
                best = w;
            }
        }
        return best;
    };

    std::unordered_set<CoreCoord> used_cores;
    std::vector<CoreRange> core_ranges;
    core_ranges.reserve(all_worker_cores_ordered.size());
    for (const auto& c : all_worker_cores_ordered) {
        CoreCoord chosen = c;
        const bool is_reserved = !available_set.contains(c);
        const bool is_duplicate = used_cores.contains(c);
        if (is_reserved || is_duplicate) {
            auto replacement = find_nearest_free_worker(c, used_cores);
            TT_FATAL(
                replacement.has_value(),
                "No free worker core available to substitute for DRAM-bank-adjacent core ({}, {}){}.",
                c.x,
                c.y,
                is_reserved ? " (reserved as dispatch core)" : " (duplicate)");
            log_warning(
                tt::LogOp,
                "DRAM-bank-adjacent core ({}, {}) is {}; substituting nearest free worker ({}, {}).",
                c.x,
                c.y,
                is_reserved ? "reserved as dispatch" : "already used",
                replacement->x,
                replacement->y);
            chosen = *replacement;
        }
        used_cores.insert(chosen);
        core_ranges.emplace_back(chosen, chosen);
    }
    auto dram_optimal_cores = CoreRangeSet(core_ranges);

    Program program{};
    auto dtype = tt_metal::datatype_to_dataformat_converter(args.input_tensor_a.dtype());

    /***************   CIRCULAR BUFFERS ***************/

    uint32_t single_tile_size = tt::tile_size(dtype);

    constexpr auto intermediate_cb0_index = tt::CBIndex::c_3;
    constexpr auto intermediate_cb1_index = tt::CBIndex::c_4;

    const auto b_data_format =
        args.input_tensor_b.has_value() ? datatype_to_dataformat_converter(args.input_tensor_b->dtype()) : dtype;
    const auto c_data_format = datatype_to_dataformat_converter(output.dtype());
    bool fp32_dest_acc_en = is_fp32_dest_acc_en(dtype, b_data_format, c_data_format);

    /*
    WH NOC_MAX_BURST_SIZE = 8192 bytes (single‑packet limit).
    For bf8: single_tile_size = 1088 → num_tiles_per_batch = 8192/1088 = 7. ✓
    But large_chunk = num_batches * num_tiles_per_batch = 2 * 7 = 14 tiles = 15232 bytes — exceeds 8192 bytes. ✗
    The reader issues a single noc_async_read_one_packet for the whole large_chunk (see lines 97, 101‑102 of
    reader_interleaved_dram_optimized.cpp), but the API docstring says:

    Initiates an asynchronous read for a single packet with size <= NOC_MAX_BURST_SIZE.

    When the size exceeds the burst limit, the NOC transfers at most one burst worth and leaves the rest undefined. So
    for bf8 on WH, tiles 7‑13 of every large_chunk contain stale CB contents, which gets added to the matmul result →
    0.958 PCC.

    This would also be broken on bf16 (4*2=8 tiles = 16384 bytes > 8192), but
    test_addmm_square_matrices[matrix_size=512-dtype=bfloat16] was never actually run in your last session — only the
    bf8 case was.

    The writer has a different pattern — it issues n_tiles_proc separate noc_async_write(... tile_size ...) calls in a
    loop (line 48‑51), so it's fine.
    */

    const uint32_t num_batches = CMAKE_UNIQUE_NAMESPACE::compute_num_batches(args);

    const uint32_t num_tiles_per_batch =
        CMAKE_UNIQUE_NAMESPACE::compute_num_tiles_per_batches(operation_attributes, args, output);

    const uint32_t num_tiles_per_cb = 2 * num_tiles_per_batch * num_batches;
    log_info(
        tt::LogOp,
        "num_tiles_per_cb: {}, num_tiles_per_batch: {}, num_batches: {}",
        num_tiles_per_cb,
        num_tiles_per_batch,
        num_batches);

    auto [a_tensor_cb, a_tensor_cb_handle] =
        create_cb(tt::CBIndex::c_0, program, dram_optimal_cores, single_tile_size, num_tiles_per_cb, dtype);

    auto [b_tensor_cb, b_tensor_cb_handle] =
        create_cb(tt::CBIndex::c_1, program, dram_optimal_cores, single_tile_size, num_tiles_per_cb, dtype);

    auto [output_cb_index, cb_output] =
        create_cb(tt::CBIndex::c_2, program, dram_optimal_cores, single_tile_size, num_tiles_per_cb, dtype);

    /***************   READER KERNEL ***************/

    // TODO: We can't use num_batches and num_tiles_per_batch as compile time aruments, the op expect one kernels for
    // tensors with different shapes.
    std::vector<uint32_t> reader_compile_time_vec = {
        a_tensor_cb,
        b_tensor_cb,
    };
    tt::tt_metal::TensorAccessorArgs(args.input_tensor_a.buffer()).append_to(reader_compile_time_vec);
    tt::tt_metal::TensorAccessorArgs(args.input_tensor_b->buffer()).append_to(reader_compile_time_vec);

    std::map<std::string, std::string> reader_defines;
    KernelHandle reader_kernel_id = CreateKernel(
        program,
        kernel_prefix + "dataflow/reader_interleaved_dram_optimized.cpp",
        dram_optimal_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_vec, reader_defines));

    tt_metal::Buffer* dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    /***************   WRITER KERNEL ***************/
    // TODO: We can't use num_batches and num_tiles_per_batch as compile time aruments, the op expect one kernels for
    // tensors with different shapes.
    std::vector<uint32_t> writer_compile_time_vec = {output_cb_index};
    tt::tt_metal::TensorAccessorArgs(dst_buffer).append_to(writer_compile_time_vec);

    std::map<std::string, std::string> writer_defines;
    KernelHandle writer_kernel_id = CreateKernel(
        program,
        kernel_prefix + "dataflow/writer_interleaved_dram_optimized.cpp",
        dram_optimal_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_vec, writer_defines));

    /***************   COMPUTE KERNEL ***************/

    const auto a_dtype = args.input_tensor_a.dtype();
    // Always pass the more accurate fp32 when the quantization scale is passed as a scalar
    const auto b_dtype = args.input_tensor_b.has_value()    ? args.input_tensor_b->dtype()
                         : operation_attributes.is_quant_op ? DataType::FLOAT32
                         : operation_attributes.is_sfpu     ? a_dtype
                                                            : DataType::BFLOAT16;

    const auto op_config = is_sfpu_op ? OpConfig(op_type, std::in_place_type<OpConfig::SfpuBinaryOp>, a_dtype)
                                      : OpConfig(op_type, std::in_place_type<OpConfig::FpuBinaryOp>, a_dtype);

    auto compute_kernel_defines = op_config.as_defines(a_dtype);

    // TODO: create a common func and merge with op_config.as_defines(a_dtype);
    {
        const auto c_dtype = output.dtype();
        const auto input_dtype = operation_attributes.input_dtype;
        ttnn::SmallVector<unary::EltwiseUnaryWithParam> lhs_activations = operation_attributes.lhs_activations;
        ttnn::SmallVector<unary::EltwiseUnaryWithParam> rhs_activations = operation_attributes.rhs_activations;
        ttnn::SmallVector<unary::EltwiseUnaryWithParam> post_activations = operation_attributes.post_activations;

        if (op_config.process_lhs.has_value()) {
            lhs_activations.push_back(*op_config.process_lhs);
        }

        if (op_config.process_rhs.has_value()) {
            rhs_activations.push_back(*op_config.process_rhs);
        }

        if (op_config.postprocess.has_value()) {
            post_activations.insert(post_activations.begin(), *op_config.postprocess);
        }

        bool is_integer_division =
            (operation_attributes.binary_op_type == BinaryOpType::DIV && a_dtype == DataType::INT32 &&
             b_dtype == DataType::INT32);

        if (binary::utils::is_typecast(a_dtype, c_dtype) and !is_quant_op and !is_integer_division) {
            post_activations.push_back({
                unary::UnaryOpType::TYPECAST,
                {static_cast<int>(a_dtype), static_cast<int>(c_dtype)},
            });
        }

        add_activation_defines(compute_kernel_defines, lhs_activations, "LHS", a_dtype);
        add_activation_defines(compute_kernel_defines, rhs_activations, "RHS", b_dtype);

        if (lhs_activations.empty() and rhs_activations.empty() and post_activations.size() == 1) {
            compute_kernel_defines["PROCESS_POST_ACTIVATIONS(i)"] = "";
            if (post_activations[0].type() == unary::UnaryOpType::RELU) {
                compute_kernel_defines["PACK_RELU"] = "1";
                unary::utils::update_macro_defines(unary::UnaryOpType::RELU, compute_kernel_defines);
            } else if (post_activations[0].type() == unary::UnaryOpType::ZERO_POINT) {
                // Zero-point is passed as the 4th run-time kernel argument
                compute_kernel_defines["QUANT_ZERO_POINT_RT_ARGS_IDX"] = "3";
                unary::utils::update_macro_defines(unary::UnaryOpType::ZERO_POINT, compute_kernel_defines);
            } else {
                add_activation_defines(compute_kernel_defines, post_activations, "POST", input_dtype);
            }
        } else {
            add_activation_defines(compute_kernel_defines, post_activations, "POST", input_dtype);
        }

        /////////////////////   Alocate CB memory for intermediate activations   /////////////////////
        bool op_has_exp =
            op_type == BinaryOpType::LOGADDEXP || op_type == BinaryOpType::LDEXP || op_type == BinaryOpType::LOGADDEXP2;

        const auto a_data_format = datatype_to_dataformat_converter(a_dtype);
        if (not compute_kernel_defines["PROCESS_LHS_ACTIVATIONS(i)"].empty()) {
            auto a_intermediate_format = is_sfpu_op   ? a_data_format
                                         : op_has_exp ? tt::DataFormat::Float16_b
                                                      : a_data_format;
            uint32_t a_intermediate_single_tile_size = tt::tile_size(a_intermediate_format);
            create_cb(
                intermediate_cb0_index,
                program,
                dram_optimal_cores,
                a_intermediate_single_tile_size,
                num_tiles_per_cb,
                a_intermediate_format);
        }
        if (not compute_kernel_defines["PROCESS_RHS_ACTIVATIONS(i)"].empty()) {
            auto b_intermediate_format = is_sfpu_op   ? b_data_format
                                         : op_has_exp ? tt::DataFormat::Float16_b
                                                      : b_data_format;
            uint32_t b_intermediate_single_tile_size = tt::tile_size(b_intermediate_format);
            create_cb(
                intermediate_cb1_index,
                program,
                dram_optimal_cores,
                b_intermediate_single_tile_size,
                num_tiles_per_cb,
                b_intermediate_format);
        }
    }

    // TODO: Don't hardcode num_batches and num_tiles_per_batch, that should be runtime args, when device compute hash,
    // that info is not taken into account
    std::vector<uint32_t> compute_compile_time_vec = {

    };
    std::string compute_kernel_path = operation_attributes.is_sfpu ? "compute/eltwise_binary_sfpu_dram_optimized.cpp"
                                                                   : "compute/eltwise_binary_dram_optimized.cpp";
    // if (operation_attributes.is_where_op) {
    //     compute_kernel_path = "compute/where_compute_kernel.cpp";
    // }

    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (is_sfpu_op) {
        if (op_type != BinaryOpType::POWER) {
            unpack_to_dest_mode[tt::CBIndex::c_0] = UnpackToDestMode::UnpackToDestFp32;
            unpack_to_dest_mode[tt::CBIndex::c_1] = UnpackToDestMode::UnpackToDestFp32;
            unpack_to_dest_mode[tt::CBIndex::c_3] = UnpackToDestMode::UnpackToDestFp32;
            unpack_to_dest_mode[tt::CBIndex::c_4] = UnpackToDestMode::UnpackToDestFp32;
        } else {
            auto unpack_mode = [](DataType dt) {
                return (dt == DataType::FLOAT32) ? UnpackToDestMode::UnpackToDestFp32 : UnpackToDestMode::Default;
            };
            unpack_to_dest_mode[tt::CBIndex::c_0] = unpack_mode(a_dtype);
            unpack_to_dest_mode[tt::CBIndex::c_1] = unpack_mode(b_dtype);
            unpack_to_dest_mode[tt::CBIndex::c_3] = unpack_mode(a_dtype);
            unpack_to_dest_mode[tt::CBIndex::c_4] = unpack_mode(b_dtype);
        }
    }

    log_info(tt::LogOp, "compute_kernel_defines: {}", compute_kernel_defines);
    KernelHandle compute_kernel_id = CreateKernel(
        program,
        kernel_prefix + compute_kernel_path,
        dram_optimal_cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .unpack_to_dest_mode = std::move(unpack_to_dest_mode),
            .math_approx_mode = false,
            .compile_args = compute_compile_time_vec,
            .defines = std::move(compute_kernel_defines)});

    CMAKE_UNIQUE_NAMESPACE::set_eltwise_binary_runtime_args_for_dram_cores<true>(
        program,
        args.input_tensor_a,
        args.input_tensor_b.value(),
        output,
        reader_kernel_id,
        writer_kernel_id,
        compute_kernel_id,
        dram_optimal_cores,
        num_batches,
        num_tiles_per_batch);
    return {
        std::move(program),
        {reader_kernel_id,
         writer_kernel_id,
         compute_kernel_id,
         a_tensor_cb_handle,
         b_tensor_cb_handle,
         cb_output,
         dram_optimal_cores,
         single_tile_size,
         single_tile_size,
         single_tile_size}};
}

void BinaryNgDramOptimizedProgram::override_runtime_arguments(
    cached_program_t& cached_program,
    const BinaryNgParams& operation_attributes,
    const BinaryNgInputs& tensor_args,
    Tensor& tensor_return_value) {
    const auto& sh_var = cached_program.shared_variables;

    CMAKE_UNIQUE_NAMESPACE::set_eltwise_binary_runtime_args_for_dram_cores<false>(
        cached_program.program,
        tensor_args.input_tensor_a,
        tensor_args.input_tensor_b.value(),
        tensor_return_value,
        sh_var.reader_kernel_id,
        sh_var.writer_kernel_id,
        sh_var.eltwise_kernel_id,
        sh_var.dram_device_cores,
        CMAKE_UNIQUE_NAMESPACE::compute_num_batches(tensor_args),
        CMAKE_UNIQUE_NAMESPACE::compute_num_tiles_per_batches(operation_attributes, tensor_args, tensor_return_value));
}
}  // namespace ttnn::operations::binary_ng::program

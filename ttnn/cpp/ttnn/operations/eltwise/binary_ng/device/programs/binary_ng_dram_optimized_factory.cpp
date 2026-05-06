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
#include <algorithm>
#include <random>

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

using tt::tt_metal::CoreRange;
using tt::tt_metal::CoreRangeSet;
using tt::tt_metal::IDevice;

std::map<std::string, std::string> get_compute_defines(
    ttnn::DataType dtype, const ttnn::operations::binary_ng::OpConfig& op_config) {
    using namespace ttnn::operations::binary_ng;
    std::map<std::string, std::string> defines;

    std::optional<std::string> int_data_format;
    if (dtype == ttnn::DataType::INT32 || dtype == ttnn::DataType::UINT32 || dtype == ttnn::DataType::UINT16) {
        int_data_format = (dtype == ttnn::DataType::INT32)    ? "Int32"
                          : (dtype == ttnn::DataType::UINT32) ? "UInt32"
                                                              : "UInt16";
    }

    if (op_config.is_sfpu_op()) {
        auto op = std::get<OpConfig::SfpuBinaryOp>(op_config.binary_op);
        if (op == OpConfig::SfpuBinaryOp::MUL) {
            if (int_data_format) {
                defines["BINARY_SFPU_INIT"] = fmt::format(
                    "mul_int_tile_init<DataFormat::{}>(); mul_int_binary_init_replay<DataFormat::{}>();",
                    *int_data_format,
                    *int_data_format);
                defines["BINARY_SFPU_OP"] = fmt::format("mul_int_binary_tile_replay<DataFormat::{}>", *int_data_format);
                return defines;
            }
            defines["BINARY_SFPU_INIT"] = "mul_binary_tile_init_replay();";
            defines["BINARY_SFPU_OP"] = "mul_binary_tile_replay";
            return defines;
        }
    }

    return op_config.as_defines(dtype);
}

// For compute bound operations, we need to get multiple cores per bank for optimal performance.
CoreRangeSet get_optimal_wh_dram_core_coords(std::size_t num_cores_per_bank) {
    /*
    The problem: that "optimal DRAM bank → logical worker" mapping is a fixed physical layout. When
    dispatch_core_axis = COL, the fast‑dispatch firmware reserves a column of Tensix cores; if any of the
    DRAM‑bank‑adjacent workers happen to land in that reserved column, the subsequent CreateKernel placement
    collides with a dispatch core and ProgramImpl::compile fatally asserts. In the ROW configuration those same
    workers are outside the dispatch row, so the identical program compiles fine. That's exactly why only the COL
    parametrization fails.
    */
    std::vector<std::vector<CoreCoord>> dram_worker_cores_ordered = [num_cores_per_bank]() {
        if (num_cores_per_bank == 1) {
            return std::vector<std::vector<CoreCoord>>{// (y,x) physical core coords
                                                       // (11,0)
                                                       {{0, 1}},
                                                       // 1,0
                                                       {{0, 0}},
                                                       // 5,0
                                                       {{4, 0}},
                                                       // 7,0
                                                       {{5, 0}},
                                                       // 1,5
                                                       {{0, 4}},
                                                       // 11,5
                                                       {{0, 5}},
                                                       // 2,5
                                                       {{1, 4}},
                                                       // 9,5
                                                       {{7, 4}},
                                                       // 8,5
                                                       {{6, 4}},
                                                       // 3,5
                                                       {{2, 4}},
                                                       // 5,5
                                                       {{4, 4}},
                                                       // 7,5
                                                       {{5, 4}}};
        }
        if (num_cores_per_bank == 2) {
            return std::vector<std::vector<CoreCoord>>{// (y,x) physical core coords
                                                       // (11,0)
                                                       {{0, 1}, {0, 2}},
                                                       // 1,0
                                                       {{0, 0}, {1, 0}},
                                                       // 5,0
                                                       {{4, 0}, {4, 1}},
                                                       // 7,0
                                                       {{5, 0}, {5, 1}},
                                                       // 1,5
                                                       {{0, 4}, {0, 5}},
                                                       // 11,5
                                                       {{0, 6}, {0, 7}},
                                                       // 2,5
                                                       {{1, 4}, {1, 5}},
                                                       // 9,5
                                                       {{7, 4}, {7, 5}},
                                                       // 8,5
                                                       {{6, 4}, {6, 5}},
                                                       // 3,5
                                                       {{2, 4}, {2, 5}},
                                                       // 5,5
                                                       {{4, 4}, {4, 5}},
                                                       // 7,5
                                                       {{5, 4}, {5, 5}}};
        }
        if (num_cores_per_bank == 3) {
            return std::vector<std::vector<CoreCoord>>{// (y,x) physical core coords
                                                       // (11,0)
                                                       {{0, 1}, {0, 2}, {0, 3}},
                                                       // 1,0
                                                       {{0, 0}, {1, 0}, {2, 0}},
                                                       // 5,0
                                                       {{4, 0}, {4, 1}, {4, 2}},
                                                       // 7,0
                                                       {{5, 0}, {5, 1}, {5, 2}},
                                                       // 1,5 +
                                                       {{0, 4}, {0, 5}, {0, 6}},
                                                       // 11,5 +
                                                       {{1, 6}, {0, 7}, {1, 7}},
                                                       // 2,5 +
                                                       {{1, 4}, {1, 5}, {2, 5}},
                                                       // 9,5
                                                       {{7, 4}, {7, 5}, {7, 6}},
                                                       // 8,5
                                                       {{6, 4}, {6, 5}, {6, 6}},
                                                       // 3,5 +d
                                                       {{2, 4}, {3, 5}, {3, 4}},
                                                       // 5,5
                                                       {{4, 4}, {4, 5}, {4, 6}},
                                                       // 7,5
                                                       {{5, 4}, {5, 5}, {5, 6}}};
        }
        TT_THROW("Unsupported number of cores per bank: {}", num_cores_per_bank);
    }();

    auto num_of_cores_per_bank = dram_worker_cores_ordered[0].size();
    const uint32_t num_wh_banks = 12;
    std::vector<CoreRange> all_cores(num_wh_banks * num_of_cores_per_bank, CoreRange(CoreCoord(0, 0), CoreCoord(0, 0)));

    // TODO: Remove extra loop and provide coords in x,y format
    for (auto idx = 0; idx < num_of_cores_per_bank; ++idx) {
        for (auto bank_id = 0; bank_id < dram_worker_cores_ordered.size(); ++bank_id) {
            const auto& dram_cores = dram_worker_cores_ordered[bank_id];
            auto core = CoreCoord(dram_cores[idx].y, dram_cores[idx].x);
            all_cores[bank_id + idx * num_wh_banks] = CoreRange(core);
        }
    }

    return CoreRangeSet(all_cores);
}

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

uint32_t compute_num_tiles_per_batches(
    const ttnn::operations::binary_ng::BinaryNgParams& operation_attributes,
    const ttnn::operations::binary_ng::BinaryNgInputs& args,
    const ttnn::Tensor& output) {
    (void)operation_attributes;
    (void)output;
    auto* device = args.input_tensor_a.device();
    auto dtype = tt::tt_metal::datatype_to_dataformat_converter(args.input_tensor_a.dtype());

    uint32_t single_tile_size = tt::tile_size(dtype);

    // const auto b_data_format =
    //     args.input_tensor_b.has_value() ? datatype_to_dataformat_converter(args.input_tensor_b->dtype()) : dtype;
    // const auto c_data_format = datatype_to_dataformat_converter(output.dtype());

    // With fp32_dest_acc_en the DST register file holds only 4 tiles (vs 16 for 16-bit).
    // The SFPU binary section interleaves LHS/RHS in DST using 2*n_tiles slots,
    // so large_chunk (= num_batches * num_tiles_per_batch) must stay <= 2 for 32-bit.
    // num_tiles_per_batch = 4 supposed to match NOC_MAX_BURST_SIZE (bytes)
    // bool fp32_dest_acc_en =
    //     ttnn::operations::binary_ng::program::is_fp32_dest_acc_en(dtype, b_data_format, c_data_format);

    const uint32_t max_tiles_per_dst = dtype == tt::DataFormat::Bfp4_b ? 8 : 16;

    return std::min(
        max_tiles_per_dst,
        CMAKE_UNIQUE_NAMESPACE::get_noc_max_burst_size(*(device->get_mesh_device())) / single_tile_size);
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
    const uint32_t num_tiles_per_batch) {
    using namespace tt;
    using namespace tt::tt_metal;
    using namespace tt::constants;

    uint32_t num_tiles = static_cast<uint32_t>(a_tensor.physical_volume() / TILE_HW);

    uint32_t num_cores_total = all_device_cores.num_cores();

    // vector of cores
    bool row_major = true;  // TODO: make this configurable
    auto cores = corerange_to_cores(all_device_cores, std::nullopt, row_major);

    std::vector<std::vector<uint32_t>> reader_args_array{cores.size()};
    std::vector<std::vector<uint32_t>> compute_args_array{cores.size()};
    std::vector<std::vector<uint32_t>> writer_args_array{cores.size()};

    auto& cached_reader_args = GetRuntimeArgs(program, reader_kernel_id);
    auto& cached_eltwise_args = GetRuntimeArgs(program, compute_kernel_id);
    auto& cached_writer_args = GetRuntimeArgs(program, writer_kernel_id);

    std::vector<uint32_t> core_ids;

    const auto num_dram_banks =
        a_tensor.device()->get_mesh_device()->allocator()->get_num_banks(tt::tt_metal::BufferType::DRAM);
    std::vector<int> tiles_per_bank(num_dram_banks, 0);
    auto cores_per_bank = num_cores_total / num_dram_banks;

    for (uint32_t core_id = 0; core_id < num_cores_total; ++core_id) {
        const CoreCoord& core = cores.at(core_id);

        auto bank_id = core_id % num_dram_banks;
        auto slot_id = core_id / num_dram_banks;
        uint32_t pages_in_bank = num_tiles / num_dram_banks + (bank_id < num_tiles % num_dram_banks ? 1 : 0);
        uint32_t num_tiles_per_core =
            pages_in_bank / cores_per_bank + (slot_id < pages_in_bank % cores_per_bank ? 1 : 0);
        uint32_t bank_page_offset = 0;
        for (uint32_t slot = 0; slot < slot_id; ++slot) {
            bank_page_offset += pages_in_bank / cores_per_bank + (slot < pages_in_bank % cores_per_bank ? 1 : 0);
        }
        uint32_t tile_ofs = bank_id + bank_page_offset * num_dram_banks;

        if constexpr (!initialize_args) {
            // RuntimeArgsData
            auto& reader_args = cached_reader_args.at(core.x).at(core.y);
            reader_args[2] = 0;
            auto& eltwise_args = cached_eltwise_args.at(core.x).at(core.y);
            eltwise_args[0] = 0;
            auto& writer_args = cached_writer_args.at(core.x).at(core.y);
            writer_args[1] = 0;
        }

        std::vector<uint32_t> reader_args_vec = {
            a_tensor.buffer()->address(),
            b_tensor.buffer()->address(),
            tile_ofs,
            num_tiles_per_core,
            num_tiles_per_batch,
            tt::tt_metal::NOC::NOC_0};

        std::vector<uint32_t> compute_args_vec = {
            num_tiles_per_core,
            num_tiles_per_batch,
        };
        std::vector<uint32_t> writer_args_vec = {
            output.buffer()->address(), tile_ofs, num_tiles_per_core, num_tiles_per_batch, tt::tt_metal::NOC::NOC_1};

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

[[maybe_unused]] CoreRangeSet get_dram_optimal_cores(IDevice* device) {
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

    // To handle scenarios where some of the optimal cores are used as dispatch cores
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
        auto& w = chosen;
        if (w.x == 3 && w.y == 7) {
            w.x = 1;
            w.y = 0;
        }
        if (w.x == 7 && w.y == 3) {
            w.x = 5;
            w.y = 0;
        }
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

    return CoreRangeSet(core_ranges);
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

namespace ttnn::operations::binary_ng::program {

const std::string kernel_prefix = "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/";

std::optional<std::string> BinaryNgDramOptimizedProgram::validate_program(
    const BinaryNgParams& operation_attributes, const BinaryNgInputs& tensor_args) {
    const auto& a = tensor_args.input_tensor_a;

    if (a.device()->arch() != tt::ARCH::WORMHOLE_B0) {
        return "Only WH architecture is supported for DRAM optimized program";
    }

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

    if (a.memory_config().is_sharded() || b.memory_config().is_sharded()) {
        return "Sharded memory is not supported for DRAM optimized program";
    }

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

    using namespace tt;
    using namespace tt::tt_metal;

    //  auto dram_optimal_cores = CMAKE_UNIQUE_NAMESPACE::get_dram_optimal_cores(args.input_tensor_a.device());
    auto dram_optimal_cores = CMAKE_UNIQUE_NAMESPACE::get_optimal_wh_dram_core_coords(is_sfpu_op ? 3 : 1);

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

    const uint32_t num_tiles_per_batch =
        CMAKE_UNIQUE_NAMESPACE::compute_num_tiles_per_batches(operation_attributes, args, output);

    const uint32_t num_tiles_per_cb = 3 * num_tiles_per_batch;

    auto [a_tensor_cb, a_tensor_cb_handle] =
        create_cb(tt::CBIndex::c_0, program, dram_optimal_cores, single_tile_size, num_tiles_per_cb, dtype);

    auto [b_tensor_cb, b_tensor_cb_handle] =
        create_cb(tt::CBIndex::c_1, program, dram_optimal_cores, single_tile_size, num_tiles_per_cb, dtype);

    auto [output_cb_index, cb_output] =
        create_cb(tt::CBIndex::c_2, program, dram_optimal_cores, single_tile_size, num_tiles_per_cb, dtype);

    /***************   READER KERNEL ***************/

    std::vector<uint32_t> reader_compile_time_vec = {
        a_tensor_cb,
        b_tensor_cb,
    };
    tt::tt_metal::TensorAccessorArgs(args.input_tensor_a.buffer()).append_to(reader_compile_time_vec);
    tt::tt_metal::TensorAccessorArgs(args.input_tensor_b->buffer()).append_to(reader_compile_time_vec);

    // READER
    std::map<std::string, std::string> reader_defines;
    if (args.input_tensor_a.dtype() == DataType::BFLOAT4_B) {
        reader_defines["BFLOAT4_B_TILES"] = "1";
    }
    KernelHandle reader_kernel_id = CreateKernel(
        program,
        kernel_prefix + "dataflow/reader_interleaved_dram_optimized.cpp",
        dram_optimal_cores,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::NOC_0,
            .noc_mode = tt_metal::NOC_MODE::DM_DEDICATED_NOC,
            .compile_args = reader_compile_time_vec,
            .defines = reader_defines,
        });

    tt_metal::Buffer* dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    /***************   WRITER KERNEL ***************/

    std::vector<uint32_t> writer_compile_time_vec = {output_cb_index};
    tt::tt_metal::TensorAccessorArgs(dst_buffer).append_to(writer_compile_time_vec);

    std::map<std::string, std::string> writer_defines;
    KernelHandle writer_kernel_id = CreateKernel(
        program,
        kernel_prefix + "dataflow/writer_interleaved_dram_optimized.cpp",
        dram_optimal_cores,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::NOC_1,
            .noc_mode = tt_metal::NOC_MODE::DM_DEDICATED_NOC,
            .compile_args = writer_compile_time_vec,
            .defines = writer_defines,
        });

    /***************   COMPUTE KERNEL ***************/

    const auto a_dtype = args.input_tensor_a.dtype();
    // Always pass the more accurate fp32 when the quantization scale is passed as a scalar
    const auto b_dtype = args.input_tensor_b.has_value()    ? args.input_tensor_b->dtype()
                         : operation_attributes.is_quant_op ? DataType::FLOAT32
                         : operation_attributes.is_sfpu     ? a_dtype
                                                            : DataType::BFLOAT16;

    const auto op_config = is_sfpu_op ? OpConfig(op_type, std::in_place_type<OpConfig::SfpuBinaryOp>, a_dtype)
                                      : OpConfig(op_type, std::in_place_type<OpConfig::FpuBinaryOp>, a_dtype);

    auto compute_kernel_defines = CMAKE_UNIQUE_NAMESPACE::get_compute_defines(a_dtype, op_config);

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

    std::vector<uint32_t> compute_compile_time_vec = {};
    std::string compute_kernel_path = operation_attributes.is_sfpu ? "compute/eltwise_binary_sfpu_dram_optimized.cpp"
                                                                   : "compute/eltwise_binary_dram_optimized.cpp";

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
        CMAKE_UNIQUE_NAMESPACE::compute_num_tiles_per_batches(operation_attributes, tensor_args, tensor_return_value));
}
}  // namespace ttnn::operations::binary_ng::program

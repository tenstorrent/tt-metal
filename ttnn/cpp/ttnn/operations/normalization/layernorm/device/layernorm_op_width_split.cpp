// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <bit>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>

#include "ttnn/operations/normalization/layernorm/device/layernorm_device_operation.hpp"
#include "ttnn/operations/normalization/layernorm/device/layernorm_device_operation_types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/program_cache_l1.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

namespace m2 = tt::tt_metal::experimental;

// The interleaved factory gives each core whole tile rows, so its time grows with the row width; the
// split pays a fixed cross-core reduce plus per-core work that grows with the rows each core holds.
// Measured on Blackhole, the split wins once the width tiles Kt reach this many times the tile rows
// Mt.
constexpr std::uint32_t kMinWidthToHeightTileRatio = 4;
// Every core holds all tile rows of its width block, so past this many rows the interleaved
// factory's one-row-per-core work split catches up.
constexpr std::uint32_t kMaxTileRows = 8;
// Fewest width tiles per core the split starts from (each core holds them for every tile row). One
// width tile per core needs twice the cores, and the gather on the root core grows with them.
constexpr std::uint32_t kMinBlockWidthTiles = 2;
// Every core keeps its whole block resident: the input, its squares, the output, and x * gamma and
// the pre-beta result when those are fused. Past this many tiles per core the interleaved factory's
// streaming path handles the width instead; below it, selection still checks the block against the
// L1 left free.
constexpr std::uint32_t kMaxBlockTiles = 32;

const m2::KernelSpecName READER{"reader"};
const m2::KernelSpecName WRITER{"writer"};
const m2::KernelSpecName COMPUTE{"compute"};

const m2::DFBSpecName IN{"in"};                  // this core's input block
const m2::DFBSpecName SCALER{"scaler"};          // unit reduce scaler for the per-core partial
const m2::DFBSpecName SCALER_ONE{"scaler_one"};  // unit reduce scaler for the root's combine
const m2::DFBSpecName XSQ{"xsq"};                // x^2 of the block
const m2::DFBSpecName PARTIAL_T{"partial_t"};    // per-row partial sum of squares, as a row
const m2::DFBSpecName GATHER{"gather"};          // root: one row per core, written by every core
const m2::DFBSpecName RSTD{"rstd"};              // root: 1/sqrt(mean + eps) as a column
const m2::DFBSpecName GLOBAL{"global"};          // 1/sqrt(mean + eps) multicast from the root
const m2::DFBSpecName XG{"xg"};                  // x * gamma, formed while the statistics are combined
const m2::DFBSpecName FUSION{"fusion"};          // the normalised block before beta is added
const m2::DFBSpecName GAMMA{"gamma"};
const m2::DFBSpecName BETA{"beta"};
const m2::DFBSpecName OUT{"out"};

const m2::TensorParamName INPUT{"input"};
const m2::TensorParamName GAMMA_T{"weight"};
const m2::TensorParamName BETA_T{"bias"};
const m2::TensorParamName OUTPUT{"output"};

const m2::SemaphoreSpecName GATHER_SEM{"gather"};
const m2::SemaphoreSpecName BCAST_SEM{"bcast"};

void bind_dfb(m2::KernelSpec& kernel, const m2::DFBSpecName& dfb, std::string accessor_name, m2::DFBEndpointType role) {
    kernel.dfb_bindings.push_back(m2::DFBBinding{
        .dfb_spec_name = dfb,
        .accessor_name = std::move(accessor_name),
        .endpoint_type = role,
    });
}

void bind_self_loop(m2::KernelSpec& kernel, const m2::DFBSpecName& dfb, std::string accessor_name) {
    bind_dfb(kernel, dfb, accessor_name, m2::DFBEndpointType::PRODUCER);
    bind_dfb(kernel, dfb, std::move(accessor_name), m2::DFBEndpointType::CONSUMER);
}

void bind_tensor(m2::KernelSpec& kernel, const m2::TensorParamName& tensor, std::string accessor_name) {
    kernel.tensor_bindings.push_back(m2::TensorBinding{
        .tensor_parameter_name = tensor,
        .accessor_name = std::move(accessor_name),
    });
}

void bind_semaphore(m2::KernelSpec& kernel, const m2::SemaphoreSpecName& semaphore, std::string accessor_name) {
    kernel.semaphore_bindings.push_back(m2::SemaphoreBinding{
        .semaphore_spec_name = semaphore,
        .accessor_name = std::move(accessor_name),
    });
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

std::optional<LayerNormWidthSplit> LayerNormWidthSplitProgramFactory::select_width_split(
    const LayerNormParams& operation_attributes,
    const LayerNormInputs& tensor_args,
    const std::optional<CoreRangeSet>& core_range_set) {
    using namespace CMAKE_UNIQUE_NAMESPACE;

    const auto& a = tensor_args.input;
    if (a.is_sharded() || a.layout() != Layout::TILE || operation_attributes.output_mem_config.is_sharded()) {
        return std::nullopt;
    }
    if (operation_attributes.norm_type != LayerNormType::RMSNORM ||
        operation_attributes.distributed_norm_stage != DistributedLayerNormStage::NOT_DISTRIBUTED ||
        tensor_args.residual_input_tensor.has_value() || operation_attributes.fused_activation.has_value()) {
        return std::nullopt;
    }
    const auto* default_config = std::get_if<LayerNormDefaultProgramConfig>(&operation_attributes.program_config);
    // The split always reduces in the intermediate format, so it serves neither Welford nor the
    // legacy reduction numerics.
    if (default_config == nullptr || default_config->use_welford || default_config->legacy_reduction) {
        return std::nullopt;
    }
    auto* device = a.device();
    // Only measured and tested on Blackhole.
    if (device->arch() != tt::ARCH::BLACKHOLE) {
        return std::nullopt;
    }
    const auto tile = a.tensor_spec().tile();
    if (tile.get_height() != TILE_HEIGHT || tile.get_width() != TILE_WIDTH) {
        return std::nullopt;
    }
    // The per-core reduce sums whole tiles of the padded width while the mean divides by the logical
    // width, so any padding (inside the last tile or whole padding tiles) would fold into sum(x^2).
    if (a.logical_shape()[-1] % TILE_WIDTH != 0 || a.padded_shape()[-1] != a.logical_shape()[-1]) {
        return std::nullopt;
    }

    const auto& padded_shape = a.padded_shape();
    const std::uint32_t Kt = padded_shape[-1] / TILE_WIDTH;
    const std::uint32_t Mt = a.physical_volume() / padded_shape[-1] / TILE_HEIGHT;
    if (Mt == 0 || Mt > kMaxTileRows || Kt < kMinWidthToHeightTileRatio * Mt) {
        return std::nullopt;
    }

    // The cores come from the caller's restriction when there is one, else from the whole grid. The
    // rectangle must lie within a single range of the restriction, so the largest range bounds it.
    CoreRange region({0, 0}, {0, 0});
    if (core_range_set.has_value()) {
        if (core_range_set->empty()) {
            return std::nullopt;
        }
        region = *std::max_element(
            core_range_set->ranges().begin(), core_range_set->ranges().end(), [](const auto& lhs, const auto& rhs) {
                return lhs.size() < rhs.size();
            });
    } else {
        const CoreCoord device_grid = device->compute_with_storage_grid_size();
        region = CoreRange({0, 0}, {device_grid.x - 1, device_grid.y - 1});
    }
    const CoreCoord grid = region.grid_size();

    // L1 every core needs for a block of block_wt width tiles; mirrors the buffers the factory
    // declares.
    const bool fp32_dest_acc_en =
        std::get<2>(get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config));
    const std::uint64_t in_tile = tt::tile_size(datatype_to_dataformat_converter(a.dtype()));
    const std::uint64_t out_tile =
        tt::tile_size(datatype_to_dataformat_converter(operation_attributes.dtype.value_or(a.dtype())));
    const std::uint64_t interm_tile =
        tt::tile_size(fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b);
    const std::uint64_t bf16_tile = tt::tile_size(tt::DataFormat::Float16_b);
    const auto& gamma = tensor_args.weight;
    const auto& beta = tensor_args.bias;
    const std::uint64_t affine_tile_bytes =
        (gamma.has_value() ? tt::tile_size(datatype_to_dataformat_converter(gamma->dtype())) : 0) +
        (beta.has_value() ? tt::tile_size(datatype_to_dataformat_converter(beta->dtype())) : 0);
    const std::uint64_t interm_blocks = 1 + (gamma.has_value() ? 1 : 0) + (beta.has_value() ? 1 : 0);
    const std::uint64_t usable_l1 = ttnn::operations::core::usable_program_l1_capacity(device);
    auto fits_l1 = [&](std::uint32_t block_wt, std::uint32_t num_cores) {
        const std::uint64_t block_tiles = std::uint64_t{Mt} * block_wt;
        const std::uint64_t gather_tiles = std::uint64_t{Mt} * tt::div_up(num_cores, TILE_HEIGHT);
        const std::uint64_t bytes = block_tiles * (in_tile + out_tile + interm_blocks * interm_tile) +
                                    block_wt * affine_tile_bytes + (3 * Mt + gather_tiles) * interm_tile +
                                    2 * bf16_tile;
        return bytes <= usable_l1;
    };

    // Take the smallest block whose core count fills a rectangle of the region, so the root reaches
    // every core with one multicast. A count that fits in one row of the region is always
    // rectangular, so the search ends.
    for (std::uint32_t block_wt = kMinBlockWidthTiles; Mt * block_wt <= kMaxBlockTiles; ++block_wt) {
        const std::uint32_t num_cores = tt::div_up(Kt, block_wt);
        if (num_cores < 2) {
            return std::nullopt;
        }
        for (std::uint32_t x = std::min<std::uint32_t>(grid.x, num_cores); x > 0; --x) {
            if (num_cores % x == 0 && num_cores / x <= grid.y) {
                if (!fits_l1(block_wt, num_cores)) {
                    // Wider blocks only need more.
                    return std::nullopt;
                }
                return LayerNormWidthSplit{
                    .origin = region.start_coord,
                    .num_tile_rows = Mt,
                    .width_tiles = Kt,
                    .block_wt = block_wt,
                    .num_cores_x = x,
                    .num_cores_y = num_cores / x,
                };
            }
        }
    }
    return std::nullopt;
}

ttnn::device_operation::ProgramArtifacts LayerNormWidthSplitProgramFactory::create_program_artifacts(
    const LayerNormParams& operation_attributes,
    const LayerNormInputs& tensor_args,
    Tensor& tensor_return_value,
    const std::optional<CoreRangeSet>& core_range_set) {
    using namespace CMAKE_UNIQUE_NAMESPACE;

    // select_program_factory and the program hash see the whole grid. A caller's core restriction that
    // leaves no room for the split is an error here rather than a silent switch to another factory's
    // program under this factory's cache key.
    const auto split = select_width_split(operation_attributes, tensor_args, core_range_set);
    TT_FATAL(
        split.has_value(), "LayerNormWidthSplitProgramFactory: the requested core range cannot host the width split");

    const auto& a = tensor_args.input;
    const auto& gamma = tensor_args.weight;
    const auto& beta = tensor_args.bias;
    auto& output = tensor_return_value;
    MeshDevice* device = a.device();

    const std::uint32_t Mt = split->num_tile_rows;
    const std::uint32_t Kt = split->width_tiles;
    const std::uint32_t block_wt = split->block_wt;
    const std::uint32_t num_cores_x = split->num_cores_x;
    const std::uint32_t num_cores_y = split->num_cores_y;
    const std::uint32_t num_cores = num_cores_x * num_cores_y;
    // The root holds one gathered row per core, 32 rows to a tile, for each tile row of the input.
    const std::uint32_t gather_tiles_per_row = tt::div_up(num_cores, TILE_HEIGHT);

    const CoreCoord origin = split->origin;
    const CoreRange core_range(origin, {origin.x + num_cores_x - 1, origin.y + num_cores_y - 1});
    const CoreRangeSet all_cores(core_range);

    const auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    const tt::DataFormat in_format = datatype_to_dataformat_converter(a.dtype());
    const tt::DataFormat out_format = datatype_to_dataformat_converter(output.dtype());
    const tt::DataFormat interm_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    const tt::DataFormat gamma_format =
        gamma.has_value() ? datatype_to_dataformat_converter(gamma->dtype()) : tt::DataFormat::Float16_b;
    const tt::DataFormat beta_format =
        beta.has_value() ? datatype_to_dataformat_converter(beta->dtype()) : tt::DataFormat::Float16_b;
    const std::uint32_t in_tile_size = tt::tile_size(in_format);
    const std::uint32_t out_tile_size = tt::tile_size(out_format);
    const std::uint32_t interm_tile_size = tt::tile_size(interm_format);
    const std::uint32_t bf16_tile_size = tt::tile_size(tt::DataFormat::Float16_b);
    const std::uint32_t gamma_tile_size = tt::tile_size(gamma_format);
    const std::uint32_t beta_tile_size = tt::tile_size(beta_format);
    // One face row of an intermediate tile: the unit each core writes into the root's gather tile.
    const std::uint32_t interm_datum_size = interm_tile_size / (TILE_HEIGHT * TILE_WIDTH);
    const std::uint32_t face_row_bytes = FACE_WIDTH * interm_datum_size;
    const std::uint32_t face_bytes = FACE_HEIGHT * face_row_bytes;

    const bool row_major_affine = (gamma.has_value() && gamma->layout() == Layout::ROW_MAJOR) ||
                                  (beta.has_value() && beta->layout() == Layout::ROW_MAJOR);
    // Subblocks live in DEST, which holds half as many tiles with FP32 accumulation.
    const std::uint32_t max_subblock_wt = fp32_dest_acc_en ? 4 : 8;
    std::uint32_t subblock_wt = std::min(block_wt, max_subblock_wt);
    while (block_wt % subblock_wt != 0) {
        --subblock_wt;
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Dataflow buffers
    ////////////////////////////////////////////////////////////////////////////
    // Every buffer is declared on every core, including the ones only the root uses: the gather
    // and multicast writes address a buffer on another core by this core's address for it, which
    // holds only while all cores carry the same buffers.
    m2::ProgramSpec spec{.name = "layernorm_width_split"};
    auto add_dfb =
        [&spec](const m2::DFBSpecName& id, std::uint32_t num_entries, std::uint32_t entry_size, tt::DataFormat format) {
            spec.dataflow_buffers.push_back(m2::DataflowBufferSpec{
                .unique_id = id,
                .entry_size = entry_size,
                .num_entries = num_entries,
                .data_format_metadata = format,
            });
        };
    add_dfb(IN, Mt * block_wt, in_tile_size, in_format);
    add_dfb(SCALER, 1, bf16_tile_size, tt::DataFormat::Float16_b);
    add_dfb(SCALER_ONE, 1, bf16_tile_size, tt::DataFormat::Float16_b);
    add_dfb(XSQ, Mt * block_wt, interm_tile_size, interm_format);
    add_dfb(PARTIAL_T, Mt, interm_tile_size, interm_format);
    add_dfb(GATHER, Mt * gather_tiles_per_row, interm_tile_size, interm_format);
    add_dfb(RSTD, Mt, interm_tile_size, interm_format);
    add_dfb(GLOBAL, Mt, interm_tile_size, interm_format);
    if (gamma.has_value()) {
        add_dfb(XG, Mt * block_wt, interm_tile_size, interm_format);
    }
    if (beta.has_value()) {
        add_dfb(FUSION, Mt * block_wt, interm_tile_size, interm_format);
    }
    if (gamma.has_value()) {
        add_dfb(GAMMA, block_wt, gamma_tile_size, gamma_format);
    }
    if (beta.has_value()) {
        add_dfb(BETA, block_wt, beta_tile_size, beta_format);
    }
    add_dfb(OUT, Mt * block_wt, out_tile_size, out_format);

    spec.tensor_parameters.push_back(m2::TensorParameter{.unique_id = INPUT, .spec = a.tensor_spec()});
    spec.tensor_parameters.push_back(m2::TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()});
    if (gamma.has_value()) {
        spec.tensor_parameters.push_back(m2::TensorParameter{.unique_id = GAMMA_T, .spec = gamma->tensor_spec()});
    }
    if (beta.has_value()) {
        spec.tensor_parameters.push_back(m2::TensorParameter{.unique_id = BETA_T, .spec = beta->tensor_spec()});
    }
    spec.semaphores = {
        m2::SemaphoreSpec{.unique_id = GATHER_SEM, .target_nodes = all_cores},
        m2::SemaphoreSpec{.unique_id = BCAST_SEM, .target_nodes = all_cores},
    };

    ////////////////////////////////////////////////////////////////////////////
    //                      Kernels
    ////////////////////////////////////////////////////////////////////////////
    // The input read opens the critical path and the output write closes it, so each DRAM direction
    // gets the NoC the arch prefers for it: the reader streams the input in, the writer writes the
    // output back.
    const NOC reader_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch());
    const NOC writer_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(device->arch());
    const std::string kernel_dir = "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/";

    m2::KernelSpec reader{
        .unique_id = READER,
        .source = kernel_dir + "dataflow/reader_rmsnorm_width_split_gather.cpp",
        .compile_time_args =
            {{"num_tile_rows", Mt},
             {"gather_tiles_per_row", gather_tiles_per_row},
             {"num_cores", num_cores},
             {"face_row_bytes", face_row_bytes},
             {"face_bytes", face_bytes},
             {"block_w", block_wt},
             {"Kt", Kt}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"is_root",
                  "width_tile_start",
                  "num_valid_tiles",
                  "core_index",
                  "root_noc_x",
                  "root_noc_y",
                  "mcast_start_x",
                  "mcast_start_y",
                  "mcast_end_x",
                  "mcast_end_y"}},
        .hw_config =
            m2::DataMovementHardwareConfig{
                .config_1xx =
                    m2::DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = DataMovementProcessor::RISCV_0,
                        .noc = reader_noc,
                        .noc_mode = NOC_MODE::DM_DEDICATED_NOC,
                    },
            },
    };
    bind_dfb(reader, IN, "in", m2::DFBEndpointType::PRODUCER);
    bind_tensor(reader, INPUT, "src");
    bind_dfb(reader, PARTIAL_T, "partial_t", m2::DFBEndpointType::CONSUMER);
    bind_dfb(reader, GATHER, "gather", m2::DFBEndpointType::PRODUCER);
    bind_dfb(reader, RSTD, "rstd", m2::DFBEndpointType::CONSUMER);
    bind_dfb(reader, GLOBAL, "global", m2::DFBEndpointType::PRODUCER);
    bind_semaphore(reader, GATHER_SEM, "gather_sem");
    bind_semaphore(reader, BCAST_SEM, "bcast_sem");

    m2::KernelSpec writer{
        .unique_id = WRITER,
        .source = kernel_dir + "dataflow/writer_rmsnorm_width_split.cpp",
        .compile_time_args = {{"block_ht", Mt}, {"block_w", block_wt}, {"Kt", Kt}},
        .runtime_arg_schema = {.runtime_arg_names = {"is_root", "width_tile_start", "num_valid_tiles"}},
        .hw_config =
            m2::DataMovementHardwareConfig{
                .config_1xx =
                    m2::DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = DataMovementProcessor::RISCV_1,
                        .noc = writer_noc,
                        .noc_mode = NOC_MODE::DM_DEDICATED_NOC,
                    },
            },
    };
    bind_dfb(writer, SCALER, "scaler", m2::DFBEndpointType::PRODUCER);
    bind_dfb(writer, SCALER_ONE, "scaler_one", m2::DFBEndpointType::PRODUCER);
    bind_dfb(writer, OUT, "out", m2::DFBEndpointType::CONSUMER);
    bind_tensor(writer, OUTPUT, "dst");
    if (gamma.has_value()) {
        bind_dfb(writer, GAMMA, "gamma", m2::DFBEndpointType::PRODUCER);
        bind_tensor(writer, GAMMA_T, "gamma");
        writer.compiler_options.defines.emplace("FUSE_GAMMA", "1");
    }
    if (beta.has_value()) {
        bind_dfb(writer, BETA, "beta", m2::DFBEndpointType::PRODUCER);
        bind_tensor(writer, BETA_T, "beta");
        writer.compiler_options.defines.emplace("FUSE_BETA", "1");
    }
    if (row_major_affine) {
        writer.compiler_options.defines.emplace("AFFINE_ROW_MAJOR", "1");
    }

    m2::KernelSpec compute{
        .unique_id = COMPUTE,
        .source = kernel_dir + "compute/rmsnorm_width_split.cpp",
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .compile_time_args =
            {{"block_ht", Mt},
             {"block_w", block_wt},
             {"subblock_w", subblock_wt},
             {"gather_tiles_per_row", gather_tiles_per_row},
             {"inv_width", std::bit_cast<std::uint32_t>(1.0f / static_cast<float>(a.logical_shape()[-1]))},
             {"eps", std::bit_cast<std::uint32_t>(operation_attributes.eps)}},
        .runtime_arg_schema = {.runtime_arg_names = {"is_root", "num_valid_tiles"}},
        .hw_config = to_compute_hardware_config(operation_attributes.compute_kernel_config),
    };
    bind_dfb(compute, IN, "in", m2::DFBEndpointType::CONSUMER);
    bind_dfb(compute, SCALER, "scaler", m2::DFBEndpointType::CONSUMER);
    bind_dfb(compute, SCALER_ONE, "scaler_one", m2::DFBEndpointType::CONSUMER);
    bind_self_loop(compute, XSQ, "xsq");
    bind_dfb(compute, PARTIAL_T, "partial_t", m2::DFBEndpointType::PRODUCER);
    bind_dfb(compute, GATHER, "gather", m2::DFBEndpointType::CONSUMER);
    bind_dfb(compute, RSTD, "rstd", m2::DFBEndpointType::PRODUCER);
    bind_dfb(compute, GLOBAL, "global", m2::DFBEndpointType::CONSUMER);
    bind_dfb(compute, OUT, "out", m2::DFBEndpointType::PRODUCER);
    if (gamma.has_value()) {
        bind_self_loop(compute, XG, "xg");
    }
    if (beta.has_value()) {
        bind_self_loop(compute, FUSION, "fusion");
    }
    if (gamma.has_value()) {
        bind_dfb(compute, GAMMA, "gamma", m2::DFBEndpointType::CONSUMER);
        compute.compiler_options.defines.emplace("FUSE_GAMMA", "1");
    }
    if (beta.has_value()) {
        bind_dfb(compute, BETA, "beta", m2::DFBEndpointType::CONSUMER);
        compute.compiler_options.defines.emplace("FUSE_BETA", "1");
    }
    if (fp32_dest_acc_en) {
        // With a 32-bit DEST every Float32 buffer compute reads needs an explicit unpack mode; the
        // FPU consumers here all read through SrcA/SrcB.
        auto& modes = std::get<m2::ComputeHardwareConfig>(compute.hw_config).unpack_modes;
        for (const auto& binding : compute.dfb_bindings) {
            if (binding.endpoint_type != m2::DFBEndpointType::CONSUMER) {
                continue;
            }
            for (const auto& dfb : spec.dataflow_buffers) {
                if (dfb.unique_id == binding.dfb_spec_name && dfb.data_format_metadata == tt::DataFormat::Float32) {
                    modes.emplace(binding.dfb_spec_name, UnpackMode::UnpackToSrc);
                }
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Runtime arguments
    ////////////////////////////////////////////////////////////////////////////
    CoreCoord root_noc = device->worker_core_from_logical_core(core_range.start_coord);
    CoreCoord mcast_start = root_noc;
    CoreCoord mcast_end = device->worker_core_from_logical_core(core_range.end_coord);
    // A NOC_1 multicast rectangle is addressed from its high corner to its low one.
    if (reader_noc == NOC::NOC_1) {
        std::swap(mcast_start, mcast_end);
    }

    m2::KernelRunArgs reader_run_args{.kernel = READER};
    m2::KernelRunArgs writer_run_args{.kernel = WRITER};
    m2::KernelRunArgs compute_run_args{.kernel = COMPUTE};
    for (std::uint32_t core_index = 0; core_index < num_cores; ++core_index) {
        const CoreCoord core = {origin.x + core_index % num_cores_x, origin.y + core_index / num_cores_x};
        const std::uint32_t is_root = core_index == 0 ? 1 : 0;
        const std::uint32_t width_tile_start = core_index * block_wt;
        // The last block can overhang the width.
        const std::uint32_t num_valid_tiles = std::min(block_wt, Kt - width_tile_start);
        m2::AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {{"is_root", is_root},
             {"width_tile_start", width_tile_start},
             {"num_valid_tiles", num_valid_tiles},
             {"core_index", core_index},
             {"root_noc_x", root_noc.x},
             {"root_noc_y", root_noc.y},
             {"mcast_start_x", mcast_start.x},
             {"mcast_start_y", mcast_start.y},
             {"mcast_end_x", mcast_end.x},
             {"mcast_end_y", mcast_end.y}});
        m2::AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {{"is_root", is_root}, {"width_tile_start", width_tile_start}, {"num_valid_tiles", num_valid_tiles}});
        m2::AddRuntimeArgsForNode(
            compute_run_args.runtime_arg_values, core, {{"is_root", is_root}, {"num_valid_tiles", num_valid_tiles}});
    }

    spec.kernels = {std::move(reader), std::move(writer), std::move(compute)};
    spec.work_units = {m2::WorkUnitSpec{
        .name = "main",
        .kernels = {READER, WRITER, COMPUTE},
        .target_nodes = all_cores,
    }};

    m2::ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args), std::move(compute_run_args)};
    run_args.tensor_args.emplace(INPUT, a.mesh_tensor());
    run_args.tensor_args.emplace(OUTPUT, output.mesh_tensor());
    if (gamma.has_value()) {
        run_args.tensor_args.emplace(GAMMA_T, gamma->mesh_tensor());
    }
    if (beta.has_value()) {
        run_args.tensor_args.emplace(BETA_T, beta->mesh_tensor());
    }

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::prim

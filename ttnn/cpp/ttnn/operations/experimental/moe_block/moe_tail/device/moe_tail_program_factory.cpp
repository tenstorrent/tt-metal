// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The MoE tail's program, one for the mesh (TailEngineFactory: the latent engine, kernels/latent_engine/;
// TailAllToAllFactory: the all-to-all, kernels/all_to_all/):
//  * built a chip at a time, then every data-movement kernel's per-core args go to a per-chip DRAM table, read into
//    an args CB at the kernel's start (common/args_table.hpp): the chips' programs are then identical - one program;
//  * no per-call arena tensor: the engine's regions are program CBs; the landing CB comes first, on the engine and
//    the port cores, so it sits at the same address there and on every chip, and the kernels read the slots' bases
//    with get_write_ptr instead of taking them as runtime args;
//  * the fence's ready / arrival words are persistent word tensors of the resources (not global semaphores);
//  * the all-to-all creates no port on a direction without a neighbour.

#include <map>
#include <tracy/Tracy.hpp>
#include "moe_tail_program_factory.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <set>
#include <unordered_map>

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/workload_descriptor.hpp>
#include <tt_stl/assert.hpp>

#include <tt-metalium/constants.hpp>

#include "ttnn/operations/experimental/moe_block/common/kernels/tile_bytes.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn::operations::experimental::moe_block::moe_tail {

namespace moe_tail_factory_detail {

using tt::tt_metal::Buffer;
using tt::tt_metal::CBDescriptor;
using tt::tt_metal::CBFormatDescriptor;
using tt::tt_metal::ComputeConfigDescriptor;
using tt::tt_metal::DataMovementConfigDescriptor;
using tt::tt_metal::DataMovementProcessor;
using tt::tt_metal::KernelDescriptor;
using tt::tt_metal::NOC;
using tt::tt_metal::ProgramDescriptor;

constexpr const char* kKernelDirectory = "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_tail/device/kernels/";
using moe_block_tiles::kBf16TileBytes;
using moe_block_tiles::kFp32TileBytes;
using tt::constants::TILE_HEIGHT;

// The latent engine's CB ids (engine_common.hpp, engine_norm_projection_compute.cpp and gather_port_common.hpp use the
// same names).
namespace engine_cb {
enum : uint8_t {
    CB_UP_WEIGHTS = 0,        // W' (resident: its K ranges pushed in order, never popped)
    CB_RING = 1,              // the row's shares of a step: all of my K half (ring_slots slots)
    CB_ROW_SQUARES = 2,       // the row's squares tiles of a step (fp32, ring_slots slots)
    CB_MY_SQUARES = 3,        // my share's squares (fp32)
    CB_ONES = 4,              // the all-ones tile of the squares' row sums
    CB_FINISHED = 5,          // my finished columns' partials (fp32)
    CB_EXCHANGE_SEND = 6,     // the partner's columns' partials + my half's sum of squares (fp32)
    CB_SUM_OF_SQUARES = 7,    // my half's sum of squares (fp32)
    CB_EXCHANGE_RECEIVE = 8,  // the partner's exchange (fp32, a slot a step of a block)
    CB_SHARED_EXPERT = 9,
    CB_OUTPUT = 10,
    CB_LANDING = 11,   // the landing slots of every step: own steps tilized here, the others landed by the gather
    CB_OWN_ROWS = 12,  // my own shares, row-major
    CB_PORT_STAGING = 13,
    CB_PORT_PACKET_COUNTS = 14,
    CB_PORT_POLL = 15,
    CB_STEP_WORDS = 16,  // the engine's step words, this call's (zeroed each call): after the landing CB, one address
    CB_ARGS = 17,        // the data-movement kernels' args blocks (one_program): the last CB, sized to them
};
}  // namespace engine_cb
// The all-to-all's (the all_to_all kernels use the same names).
namespace all_to_all_cb {
enum : uint8_t {
    CB_ROWS = 0,  // a worker's owned rows, row-major
    CB_TILIZED = 1,
    CB_SHARED_EXPERT = 2,
    CB_OUTPUT = 3,
    CB_PORT_STAGING = 4,     // port_staging_reader -> port_fabric_sender
    CB_Z_READY = 5,          // the z-ready words, a row tile's 16 B apart
    CB_PADDING_SCRATCH = 6,  // a padded forward: the span's words, the workers' zero tile (padded_row_tiles.hpp)
    CB_LIVE_ROW_TILES = 7,   // a padded forward: the workers' reader -> compute hand-off of a block's live row tiles
    CB_ARGS = 8,             // the data-movement kernels' args blocks (one_program): the last CB, sized to them
};
}  // namespace all_to_all_cb

KernelDescriptor kernel(const std::string& file, const CoreRangeSet& cores, KernelDescriptor::ConfigDescriptor config) {
    KernelDescriptor descriptor;
    descriptor.kernel_source = std::string(kKernelDirectory) + file;
    descriptor.source_type = KernelDescriptor::SourceType::FILE_PATH;
    descriptor.core_ranges = cores;
    descriptor.config = std::move(config);
    return descriptor;
}

DataMovementConfigDescriptor data_movement(DataMovementProcessor risc, NOC noc) {
    return DataMovementConfigDescriptor{.processor = risc, .noc = noc};
}

CBDescriptor cb(
    uint8_t index, const CoreRangeSet& cores, uint32_t total_bytes, tt::DataFormat format, uint32_t page_bytes) {
    return CBDescriptor{
        .total_size = total_bytes,
        .core_ranges = cores,
        .format_descriptors = {CBFormatDescriptor{
            .buffer_index = index, .data_format = format, .page_size = page_bytes}},
    };
}

// (TailEngine's CB check: a block addressed from the CB's pointer must not straddle its end - every core pushes one
// size there and the dataflow ends read page by page, so each size must divide the CB)
void check_blocks(const char* name, uint32_t size, uint32_t page_bytes, const std::vector<uint32_t>& blocks) {
    for (uint32_t block_pages : std::set<uint32_t>(blocks.begin(), blocks.end())) {
        TT_FATAL(
            ring_pages(size / page_bytes, {block_pages}) == size / page_bytes,
            "moe_tail (engine): CB {} ({} pages) takes blocks of {}: one would straddle its end",
            name,
            size / page_bytes,
            block_pages);
    }
}

void append_accessor(KernelDescriptor& descriptor, const Buffer* buffer) {
    tt::tt_metal::TensorAccessorArgs(*buffer).append_to(descriptor.compile_time_args);
}

uint32_t as_uint32(size_t value) { return static_cast<uint32_t>(value); }
uint32_t packed_xy(const CoreCoord& core) { return as_uint32(core.x | core.y << 16); }

std::vector<CoreCoord> cores_of_set(const CoreRangeSet& core_ranges) {
    std::vector<CoreCoord> cores;
    for (const auto& range : core_ranges.ranges()) {
        for (auto y = range.start_coord.y; y <= range.end_coord.y; ++y) {
            for (auto x = range.start_coord.x; x <= range.end_coord.x; ++x) {
                cores.emplace_back(x, y);
            }
        }
    }
    return cores;
}

CoreRangeSet cells(const std::vector<CoreCoord>& cores) {
    std::vector<CoreRange> ranges;
    for (const auto& core : cores) {
        ranges.emplace_back(core, core);
    }
    return CoreRangeSet(ranges);
}

std::vector<ttnn::MeshCoordinate> coordinates_of(const ttnn::MeshCoordinateRangeSet& set) {
    std::vector<ttnn::MeshCoordinate> out;
    for (const auto& range : set.ranges()) {
        for (const auto& coordinate : range) {
            out.push_back(coordinate);
        }
    }
    return out;
}

uint32_t index_of(const std::vector<uint32_t>& values, uint32_t value) {
    return static_cast<uint32_t>(std::find(values.begin(), values.end(), value) - values.begin());
}

// The buffers the runtime args address, by source (address_patches.hpp; nullptr: absent).
std::array<Buffer*, 9> sources_of(const MoETailInputs& inputs, const Tensor& output) {
    const auto buffer_of = [](const std::optional<Tensor>& tensor) {
        return tensor.has_value() ? tensor->buffer() : nullptr;
    };
    return {
        inputs.y_own.buffer(),
        buffer_of(inputs.shared_expert),
        buffer_of(inputs.w_up),
        output.buffer(),
        inputs.ready.buffer(),
        buffer_of(inputs.arrival),
        buffer_of(inputs.staging),
        inputs.metadata.has_value() ? inputs.metadata->buffer() : buffer_of(inputs.actual_start),
        buffer_of(inputs.actual_end)};
}

// The cached workload of the program descriptors, with each program's address patches.
TailEngineFactory::cached_mesh_workload_t to_cached(
    tt::tt_metal::WorkloadDescriptor descriptors,
    const MoETailInputs& args,
    const Tensor& output,
    std::shared_ptr<const void> args_tables) {
    const auto sources = sources_of(args, output);
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, TailEngineFactory::shared_variables_t> shared;
    for (auto& [range, program_descriptor] : descriptors.programs) {
        TailEngineFactory::shared_variables_t variables{
            .patches = take_address_patches(program_descriptor, sources), .args_tables = args_tables};
        workload.add_program(range, tt::tt_metal::Program(program_descriptor));
        shared.emplace(range, std::move(variables));
    }
    return {std::move(workload), std::move(shared)};
}

void patch_cached(TailEngineFactory::cached_mesh_workload_t& cached, const MoETailInputs& args, const Tensor& output) {
    const auto sources = sources_of(args, output);
    for (auto& [range, variables] : cached.shared_variables) {
        apply_address_patches(cached.workload.get_programs().at(range), variables.patches, sources);
    }
}

// Row blocks in one call (attributes.row_blocks > 1): every kernel of both tails loops over them itself, block b's rows
// of y_own / shared_expert / the output b x block_stride further (compile-time args).
// A row block's rows of an interleaved DRAM tensor: whole pages of every bank, so block b's rows start l x the returned
// bytes further in every bank (0 with one block).
uint32_t block_stride(const Tensor& tensor, uint32_t row_blocks, const char* what) {
    if (row_blocks <= 1) {
        return 0;
    }
    const auto* buffer = tensor.buffer();
    const uint32_t pages = buffer->num_pages(), num_banks = buffer->allocator()->get_num_banks(buffer->buffer_type());
    TT_FATAL(
        !tensor.is_sharded() && pages % (row_blocks * num_banks) == 0,
        "moe_tail: row blocks in one call need {}'s pages ({}) interleaved, a whole number of every bank's ({} "
        "banks) a block ({} blocks)",
        what,
        pages,
        num_banks,
        row_blocks);
    return static_cast<uint32_t>(pages / row_blocks / num_banks * buffer->aligned_page_size());
}

// Two data-movement kernels on the same cores (their indices in the program).
struct DataMovementPair {
    size_t reader, writer;
    CoreRangeSet cores;
};

// The chips' programs as one program for the mesh (common/args_table.hpp): each data-movement kernel's per-core args
// go to a per-chip DRAM table; at start the kernel loads its core's block into CB `cb_id`, added here on each pair's
// cores and sized to that pair's blocks (the reader's at 0, the writer's right after). The compute kernels keep their
// runtime args (the same on every chip).
std::pair<tt::tt_metal::WorkloadDescriptor, std::shared_ptr<const void>> one_program(
    tt::tt_metal::WorkloadDescriptor workload,
    tt::tt_metal::distributed::MeshDevice& mesh,
    uint8_t cb_id,
    const std::vector<DataMovementPair>& pairs,
    const char* mode) {
    std::vector<std::pair<ttnn::MeshCoordinate, ProgramDescriptor>> chips;
    for (auto& program : workload.programs) {
        TT_FATAL(
            program.range.shape().mesh_size() == 1, "moe_tail ({}): one program a chip before the args move", mode);
        chips.emplace_back(program.range.start_coord(), std::move(program.descriptor));
    }
    // a kernel's block: its longest core's args on any chip, whole 64 B (as args_table.cpp pages it)
    auto block = [&](size_t kernel_index) {
        size_t words = 0;
        for (const auto& [coordinate, program] : chips) {
            for (const auto& [core, core_args] : program.kernels[kernel_index].runtime_args) {
                words = std::max(words, core_args.size());
            }
        }
        return std::max<uint32_t>(64, (4 * as_uint32(words) + 63) / 64 * 64);
    };
    std::vector<std::optional<ArgsScratch>> scratch(chips[0].second.kernels.size());
    for (const auto& [reader, writer, cores] : pairs) {
        const uint32_t reader_bytes = block(reader), writer_bytes = block(writer);
        scratch[reader] = ArgsScratch{.cb = cb_id, .offset = 0, .bytes = reader_bytes};
        scratch[writer] = ArgsScratch{.cb = cb_id, .offset = reader_bytes, .bytes = writer_bytes};
        log_debug(
            tt::LogOp,
            "moe_tail ({}): kernels {} / {}: args blocks {} + {} B",
            mode,
            reader,
            writer,
            reader_bytes,
            writer_bytes);
        for (auto& [coordinate, program] : chips) {
            program.cbs.push_back(
                cb(cb_id, cores, reader_bytes + writer_bytes, tt::DataFormat::UInt32, reader_bytes + writer_bytes));
        }
    }
    std::string why;
    auto tables = runtime_args_to_dram(mesh, chips, scratch, why);
    TT_FATAL(tables != nullptr, "moe_tail ({}): the kernels' args to DRAM: {}", mode, why);
    for (size_t i = 0; i < chips.size(); ++i) {
        workload.programs[i].descriptor = std::move(chips[i].second);
    }
    std::string merge_why;
    auto one = merge_chip_workload(std::move(workload), "moe_tail", &merge_why);
    TT_FATAL(
        one.programs.size() == 1,
        "moe_tail ({}): the chips' programs still differ with their args in DRAM: {}",
        mode,
        merge_why);
    return {std::move(one), std::move(tables)};
}

tt::tt_metal::WorkloadDescriptor engine_workload(
    const MoETailParams& attributes,
    const MoETailInputs& args,
    Tensor& output,
    const ttnn::MeshCoordinateRangeSet& tensor_coordinates);
tt::tt_metal::WorkloadDescriptor all_to_all_workload(
    const MoETailParams& attributes,
    const MoETailInputs& args,
    Tensor& output,
    const ttnn::MeshCoordinateRangeSet& tensor_coordinates);

}  // namespace moe_tail_factory_detail
using namespace moe_tail_factory_detail;

TailEngineFactory::cached_mesh_workload_t TailEngineFactory::create_mesh_workload(
    const MoETailParams& attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coordinates,
    const MoETailInputs& tensor_args,
    Tensor& output) {
    // (row blocks: the engine's kernels block themselves - no merged per-block programs)
    const auto& plan = *attributes.engine_plan;
    // (kernels: 0 reader, 1 writer, 2 compute on the engine; 3 port reader, 4 port writer on the ports)
    auto [workload, tables] = one_program(
        engine_workload(attributes, tensor_args, output, tensor_coordinates),
        *tensor_args.y_own.device(),
        engine_cb::CB_ARGS,
        {{0, 1, plan.engine_cores}, {3, 4, plan.port_cores}},
        "engine");
    return to_cached(std::move(workload), tensor_args, output, std::move(tables));
}

void TailEngineFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached,
    const MoETailParams& /*attributes*/,
    const MoETailInputs& tensor_args,
    Tensor& output) {
    ZoneScopedN("moe_tail: override (hit, engine)");
    patch_cached(cached, tensor_args, output);
}

TailAllToAllFactory::cached_mesh_workload_t TailAllToAllFactory::create_mesh_workload(
    const MoETailParams& attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coordinates,
    const MoETailInputs& tensor_args,
    Tensor& output) {
    // (row blocks: the all-to-all kernels block themselves - no merged per-block programs)
    const auto& plan = *attributes.all_to_all_plan;
    // (kernels: 0 reader, 1 compute, 2 writer on the workers; 3 port reader, 4 port writer on the ports' union)
    auto [workload, tables] = one_program(
        all_to_all_workload(attributes, tensor_args, output, tensor_coordinates),
        *tensor_args.y_own.device(),
        all_to_all_cb::CB_ARGS,
        {{0, 2, cells(plan.workers)}, {3, 4, plan.port_union}},
        "all-to-all");
    return to_cached(std::move(workload), tensor_args, output, std::move(tables));
}

void TailAllToAllFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached,
    const MoETailParams& /*attributes*/,
    const MoETailInputs& tensor_args,
    Tensor& output) {
    ZoneScopedN("moe_tail: override (hit)");
    patch_cached(cached, tensor_args, output);
}

// ---------------------------------------------------------------------------------------------------------------------
// TailEngine
// ---------------------------------------------------------------------------------------------------------------------

tt::tt_metal::WorkloadDescriptor moe_tail_factory_detail::engine_workload(
    const MoETailParams& attributes,
    const MoETailInputs& args,
    Tensor& output,
    const ttnn::MeshCoordinateRangeSet& tensor_coordinates) {
    using namespace engine_cb;
    const auto& plan = *attributes.engine_plan;
    const auto& geometry = plan.geometry;
    auto* mesh = args.y_own.device();
    const uint32_t engine_columns = geometry.engine_columns, engine_rows = geometry.engine_rows,
                   num_pairs = geometry.num_pairs;
    const uint32_t tp_size = geometry.num_chunks;
    const auto w_format = tt::tt_metal::datatype_to_dataformat_converter(args.w_up->dtype());
    const auto bfloat16_format = tt::DataFormat::Float16_b, float32_format = tt::DataFormat::Float32;

    Buffer* y_own_buffer = args.y_own.buffer();
    Buffer* w_up_buffer = args.w_up->buffer();
    Buffer* shared_expert_buffer = args.shared_expert->buffer();
    Buffer* output_buffer = output.buffer();
    Buffer* ready_buffer = args.ready.buffer();
    // row blocks: every kernel loop over them itself (block b's rows of y_own / the shared expert / the output b x
    // stride further)
    const uint32_t row_blocks = std::max<uint32_t>(1, attributes.row_blocks);
    const uint32_t rows_block_stride_bytes = block_stride(args.y_own, row_blocks, "y_own");
    const uint32_t shared_expert_block_stride_bytes = block_stride(*args.shared_expert, row_blocks, "shared_expert");
    const uint32_t output_block_stride_bytes = block_stride(output, row_blocks, "the output");

    auto virtual_core = [&](uint32_t column, uint32_t row) {
        return mesh->worker_core_from_logical_core(CoreCoord(column, row));
    };
    auto node = [&](uint32_t mesh_row, uint32_t position) {
        return mesh->get_fabric_node_id(mesh_coord(ChipCoord{mesh_row, position}, plan.topology));
    };
    auto schedule = [&](uint32_t position, TailDirection direction) {
        if (!row_neighbor(position, tp_size, plan.ring, direction)) {
            return std::vector<Unit>{};
        }
        const auto both_schedules = gather_schedule(position, tp_size, plan.ring);
        return direction == TailDirection::Forward ? both_schedules.first : both_schedules.second;
    };

    const CoreRangeSet& engine = plan.engine_cores;
    const CoreRangeSet& ports = plan.port_cores;
    const CoreRangeSet engine_and_ports = engine.merge(ports);
    // (a port's staging: three groups - one sending, one landing, one being issued)
    const uint32_t port_staging_pages = 3 * kPacketsPerGroup * plan.tiles_per_packet;

    // CBs: the landing slots first (engine and ports: one address everywhere), then the engine's, then the ports'.
    std::vector<CBDescriptor> cbs;
    auto engine_buffer = [&](uint8_t index,
                             const char* name,
                             uint32_t size,
                             tt::DataFormat format,
                             uint32_t page,
                             std::vector<uint32_t> blocks) {
        check_blocks(name, size, page, blocks);
        cbs.push_back(cb(index, index == CB_LANDING ? engine_and_ports : engine, size, format, page));
    };
    engine_buffer(
        CB_LANDING,
        "landing",
        geometry.max_steps * geometry.max_share_tiles * kBf16TileBytes,
        bfloat16_format,
        kBf16TileBytes,
        {geometry.max_share_tiles});
    // (the step words right after it, also on engine and ports: 16 B each - engine_common.hpp has the layout)
    const uint32_t step_words_bytes = 16 * (3 * geometry.max_steps + 1 + geometry.max_steps * engine_columns);
    cbs.push_back(cb(CB_STEP_WORDS, engine_and_ports, step_words_bytes, tt::DataFormat::UInt32, step_words_bytes));
    engine_buffer(
        CB_UP_WEIGHTS,
        "up_weights",
        geometry.k_half_tiles * geometry.max_column_tiles * geometry.w_tile_bytes,
        w_format,
        geometry.w_tile_bytes,
        {1});
    engine_buffer(
        CB_RING,
        "ring",
        geometry.ring_slots * geometry.k_half_tiles * kBf16TileBytes,
        bfloat16_format,
        kBf16TileBytes,
        {geometry.k_half_tiles});
    engine_buffer(
        CB_ROW_SQUARES,
        "row_squares",
        geometry.ring_slots * engine_columns * kFp32TileBytes,
        float32_format,
        kFp32TileBytes,
        {engine_columns});
    engine_buffer(CB_MY_SQUARES, "my_squares", kFp32TileBytes, float32_format, kFp32TileBytes, {1});
    engine_buffer(CB_ONES, "ones", kBf16TileBytes, bfloat16_format, kBf16TileBytes, {1});
    engine_buffer(
        CB_FINISHED,
        "finished",
        geometry.finished_ring_pages * kFp32TileBytes,
        float32_format,
        kFp32TileBytes,
        geometry.finished_counts);
    engine_buffer(
        CB_EXCHANGE_SEND,
        "exchange_send",
        2 * kExchangeTiles * kFp32TileBytes,
        float32_format,
        kFp32TileBytes,
        {kExchangeTiles});
    engine_buffer(CB_SUM_OF_SQUARES, "sum_of_squares", 2 * kFp32TileBytes, float32_format, kFp32TileBytes, {1});
    engine_buffer(
        CB_EXCHANGE_RECEIVE,
        "exchange_receive",
        geometry.max_steps * kExchangeTiles * kFp32TileBytes,
        float32_format,
        kFp32TileBytes,
        {kExchangeTiles});
    engine_buffer(
        CB_SHARED_EXPERT,
        "shared_expert",
        geometry.finished_ring_pages * kBf16TileBytes,
        bfloat16_format,
        kBf16TileBytes,
        geometry.finished_counts);
    engine_buffer(
        CB_OUTPUT,
        "output",
        geometry.finished_ring_pages * kBf16TileBytes,
        bfloat16_format,
        kBf16TileBytes,
        geometry.finished_counts);
    // (my own shares' row-major source: a whole max_share_tiles block a push (my share's tiles in it), so a block never
    // straddles the CB's end, block after block)
    engine_buffer(
        CB_OWN_ROWS,
        "own_rows",
        geometry.max_own_steps * geometry.max_share_tiles * kBf16TileBytes,
        bfloat16_format,
        kBf16TileBytes,
        {geometry.max_share_tiles});
    // the ports' staging, their group packet counts (a 16 B page each) and their poll words (16 B a share)
    cbs.push_back(cb(CB_PORT_STAGING, ports, port_staging_pages * kBf16TileBytes, bfloat16_format, kBf16TileBytes));
    cbs.push_back(cb(CB_PORT_PACKET_COUNTS, ports, 4 * 16, tt::DataFormat::UInt32, 16));
    cbs.push_back(
        cb(CB_PORT_POLL, ports, 16 + geometry.row_tiles_per_chunk * engine_columns * 16, tt::DataFormat::UInt32, 16));

    // the engine's dataflow kernels' compile-time args 0 - 10 (engine_common.hpp)
    const std::vector<uint32_t> engine_compile_args = {
        geometry.k_half_tiles,
        geometry.max_column_tiles,
        geometry.row_tiles_per_chunk,
        geometry.num_chunks,
        engine_columns,
        geometry.w_tile_bytes,
        geometry.ring_slots,
        geometry.latent_tiles,
        kExchangeTiles,
        num_pairs,
        geometry.max_steps};
    std::vector<uint32_t> w_k_bounds;
    for (const auto& [k_first, k_end] : geometry.w_k_ranges) {
        w_k_bounds.push_back(k_first);
    }
    w_k_bounds.push_back(geometry.k_half_tiles);
    std::vector<uint32_t> share_k_ranges;
    for (const auto& [k_first, k_end] : geometry.share_k_ranges) {
        share_k_ranges.push_back(k_first);
        share_k_ranges.push_back(k_end);
    }
    std::vector<uint32_t> column_xs;  // (the engine's NoC x by column)
    for (uint32_t column = 0; column < engine_columns; ++column) {
        column_xs.push_back(as_uint32(virtual_core(column, 0).x));
    }
    // (the gates: lead core (0, 0), the engine's NoC rectangle, its cores)
    const CoreCoord lead = virtual_core(0, 0), far_corner = virtual_core(engine_columns - 1, engine_rows - 1);
    const std::vector<uint32_t> gate = {
        packed_xy(lead),
        as_uint32(std::min(lead.x, far_corner.x)),
        as_uint32(std::min(lead.y, far_corner.y)),
        as_uint32(std::max(lead.x, far_corner.x)),
        as_uint32(std::max(lead.y, far_corner.y)),
        engine_columns * engine_rows};
    const uint32_t w_region_tiles = geometry.region_tiles();
    std::vector<uint32_t> port_xys;  // (the port cores the zero gate releases once each: x | y << 16)
    for (const auto& range : ports.ranges()) {
        for (const auto& core : range) {
            port_xys.push_back(packed_xy(virtual_core(core.x, core.y)));
        }
    }

    tt::tt_metal::WorkloadDescriptor workload;
    for (const auto& mesh_coordinate : coordinates_of(tensor_coordinates)) {
        const ChipCoord chip = chip_of(mesh_coordinate, plan.topology);
        const uint32_t mesh_row = chip.r, position = chip.c;
        const auto order = chunk_order(position, geometry.num_chunks, plan.ring);
        const auto watchers = engine_watchers(position, tp_size, plan.ring, geometry);

        ProgramDescriptor program_descriptor;
        program_descriptor.cbs = ProgramDescriptor::CBDescriptors(cbs.begin(), cbs.end());
        // (the ports' zero-gate go, PORT_READER_GO_SEM / PORT_WRITER_GO_SEM: ids 0 and 1, before the fabric
        // connections take the next free ones)
        for (uint32_t id = 0; id < 2; ++id) {
            program_descriptor.semaphores.push_back(
                tt::tt_metal::SemaphoreDescriptor{.id = id, .core_ranges = ports, .initial_value = 0});
        }

        auto reader = kernel(
            "latent_engine/engine_dram_reader.cpp", engine, data_movement(DataMovementProcessor::RISCV_0, NOC::NOC_0));
        reader.compile_time_args = engine_compile_args;
        append_accessor(reader, y_own_buffer);
        append_accessor(reader, shared_expert_buffer);
        reader.compile_time_args.insert(
            reader.compile_time_args.end(),
            {row_blocks, rows_block_stride_bytes, shared_expert_block_stride_bytes, geometry.max_share_tiles});
        auto writer = kernel(
            "latent_engine/engine_row_exchange_writer.cpp",
            engine,
            data_movement(DataMovementProcessor::RISCV_1, NOC::NOC_1));
        writer.compile_time_args = engine_compile_args;
        append_accessor(writer, output_buffer);
        writer.compile_time_args.insert(writer.compile_time_args.end(), {row_blocks, output_block_stride_bytes});
        ComputeConfigDescriptor compute_config{
            .math_fidelity = attributes.math_fidelity,
            .fp32_dest_acc_en = true,
            .dst_full_sync_en = true,
            .math_approx_mode = false,
        };
        compute_config.unpack_to_dest_mode.assign(NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
        for (uint8_t i : {CB_FINISHED, CB_SUM_OF_SQUARES, CB_EXCHANGE_RECEIVE}) {
            compute_config.unpack_to_dest_mode[i] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
        }
        auto compute = kernel("latent_engine/engine_norm_projection_compute.cpp", engine, compute_config);
        compute.compile_time_args = {
            geometry.k_half_tiles,
            engine_columns,
            geometry.max_share_tiles,
            std::bit_cast<uint32_t>(static_cast<float>(1.0 / geometry.latent_dim)),
            std::bit_cast<uint32_t>(plan.norm_eps),
            kExchangeTiles,
            row_blocks,
            geometry.max_steps};

        for (uint32_t column = 0; column < engine_columns; ++column) {
            const uint32_t column_tiles = geometry.column_tiles[column];
            const auto [share_k_first, share_k_end] = geometry.share_k_ranges[column];
            for (uint32_t row = 0; row < engine_rows; ++row) {
                const uint32_t pair = row % num_pairs, half = row / num_pairs;
                const auto [finished_offset, finished_count] = geometry.finished_columns[column][half];
                const CoreCoord core(column, row);
                const uint32_t noc_x = as_uint32(virtual_core(column, row).x),
                               noc_y = as_uint32(virtual_core(column, row).y);
                const uint32_t half_column_top = as_uint32(virtual_core(column, half * num_pairs).y);
                const uint32_t half_column_bottom = as_uint32(virtual_core(column, half * num_pairs + num_pairs - 1).y);
                const auto [w_k_first, w_k_end] = geometry.w_k_ranges[pair];
                const uint32_t region = (half * engine_columns + column) * num_pairs + pair;

                KernelDescriptor::RTArgList reader_args;
                for (uint32_t value : {pair, column_tiles, geometry.k_halves[half].first}) {
                    reader_args.push_back(value);
                }
                reader_args.push_back(w_up_buffer);
                for (uint32_t value :
                     {region % plan.num_dram_banks,
                      (region / plan.num_dram_banks) * w_region_tiles * geometry.w_tile_bytes,
                      w_k_first,
                      w_k_end,
                      noc_x,
                      half_column_top,
                      half_column_bottom}) {
                    reader_args.push_back(value);
                }
                reader_args.push_back(y_own_buffer);
                reader_args.push_back(share_k_first);
                reader_args.push_back(share_k_end);
                reader_args.push_back(shared_expert_buffer);
                for (uint32_t value :
                     {geometry.output_tiles,
                      geometry.column_first_tile[column] + finished_offset,
                      finished_count,
                      geometry.steps[pair],
                      geometry.own_steps[pair]}) {
                    reader_args.push_back(value);
                }
                reader_args.append(gate);
                reader_args.push_back(static_cast<uint32_t>(column == 0 && row == 0));
                if (column == 0 && row == 0) {  // (the zero gate's lead releases the port cores)
                    reader_args.push_back(static_cast<uint32_t>(port_xys.size()));
                    reader_args.append(port_xys);
                } else {
                    reader_args.push_back(0u);
                }
                reader_args.append(order);
                reader.emplace_runtime_args(core, reader_args);

                KernelDescriptor::RTArgList writer_args;
                for (uint32_t value :
                     {pair,
                      column_tiles,
                      geometry.column_first_tile[column] + finished_offset,
                      finished_count,
                      share_k_first,
                      share_k_end,
                      as_uint32(virtual_core(0, row).x),
                      as_uint32(virtual_core(engine_columns - 1, row).x),
                      column}) {
                    writer_args.push_back(value);
                }
                writer_args.push_back(output_buffer);
                for (uint32_t value :
                     {geometry.output_tiles,
                      geometry.max_share_tiles,
                      noc_x,
                      noc_y,
                      as_uint32(virtual_core(column, (row + num_pairs) % engine_rows).y)}) {
                    writer_args.push_back(value);
                }
                writer_args.push_back(geometry.steps[pair]);
                writer_args.push_back(geometry.own_steps[pair]);
                writer_args.append(order);
                for (uint32_t other_column = 0; other_column < engine_columns; ++other_column) {
                    writer_args.push_back(as_uint32(virtual_core(other_column, row).x));
                }
                writer_args.append(w_k_bounds);
                const auto watched_entry = watchers.find({column, row});
                std::vector<uint32_t> watched =
                    watched_entry == watchers.end() ? std::vector<uint32_t>{} : watched_entry->second;
                writer_args.push_back(static_cast<uint32_t>(watched.size()));
                watched.resize(geometry.max_steps, 0);  // (a fixed length)
                writer_args.append(watched);
                // (the gate's lead: every writer counts in there after each block but the last)
                writer_args.push_back(packed_xy(lead));
                writer.emplace_runtime_args(core, writer_args);

                std::vector<uint32_t> compute_args = {
                    column_tiles,
                    finished_offset,
                    finished_count,
                    share_k_first,
                    share_k_end,
                    geometry.steps[pair],
                    geometry.own_steps[pair],
                    0};
                compute_args.insert(compute_args.end(), w_k_bounds.begin() + 1, w_k_bounds.end());
                compute.runtime_args.emplace_back(core, std::move(compute_args));
            }
        }

        // ---- the gather's ports ---- (compile-time args: gather_port_common.hpp)
        const std::vector<uint32_t> port_compile_args = {
            geometry.row_tiles_per_chunk,
            engine_columns,
            plan.tiles_per_packet,
            geometry.max_share_tiles,
            geometry.latent_tiles,
            kPacketsPerGroup,
            num_pairs,
            kLandedGroupColumns,
            CB_PORT_STAGING,
            CB_PORT_PACKET_COUNTS,
            CB_PORT_POLL,
            CB_LANDING,
            row_blocks};
        auto port_reader = kernel(
            "latent_engine/gather_port_staging_reader.cpp",
            ports,
            data_movement(DataMovementProcessor::RISCV_1, NOC::NOC_0));
        port_reader.compile_time_args = port_compile_args;
        auto port_writer = kernel(
            "latent_engine/gather_port_fabric_sender.cpp",
            ports,
            data_movement(DataMovementProcessor::RISCV_0, NOC::NOC_1));
        port_writer.compile_time_args = port_compile_args;
        for (const auto& [key, core] : plan.ports) {
            const auto [direction, link] = key;
            const auto peer = row_neighbor(position, tp_size, plan.ring, direction);
            const auto sends = schedule(position, direction);
            std::vector<uint32_t> pair_ys;
            for (uint32_t pair = 0; pair < num_pairs; ++pair) {
                pair_ys.push_back(as_uint32(virtual_core(0, pair + num_pairs * link).y));
            }
            KernelDescriptor::RTArgList reader_args;
            reader_args.push_back(static_cast<uint32_t>(sends.size()));
            for (const auto& unit : sends) {
                reader_args.push_back(unit.part | index_of(order, unit.position) << 8);
            }
            reader_args.append(share_k_ranges);
            reader_args.append(column_xs);
            reader_args.append(pair_ys);
            // (the gate's lead: every port reader counts in there after each block but the last)
            reader_args.push_back(packed_xy(lead));
            port_reader.emplace_runtime_args(core, reader_args);

            // (my ready to the peer's port that writes into this chip; it waits for it before writing my landing slots)
            const bool send_ready = peer.has_value() && !schedule(*peer, opposite(direction)).empty();
            const CoreCoord peer_port = plan.ports.at({opposite(direction), link});
            const CoreCoord ready_core = send_ready ? virtual_core(peer_port.x, peer_port.y) : virtual_core(0, 0);
            KernelDescriptor::RTArgList sender_args;
            sender_args.push_back(ready_buffer);
            for (uint32_t value :
                 {static_cast<uint32_t>(send_ready), as_uint32(ready_core.x), as_uint32(ready_core.y)}) {
                sender_args.push_back(value);
            }
            const bool linked = peer.has_value() && (!sends.empty() || send_ready);
            std::vector<uint32_t> fabric_args;
            if (linked) {
                const auto peer_node = node(mesh_row, *peer);
                const auto peer_order = chunk_order(*peer, geometry.num_chunks, plan.ring);
                sender_args.push_back(*peer_node.mesh_id);
                sender_args.push_back(static_cast<uint32_t>(peer_node.chip_id));
                sender_args.push_back(static_cast<uint32_t>(sends.size()));
                for (const auto& unit : sends) {
                    sender_args.push_back(unit.part | index_of(peer_order, unit.position) << 8);
                }
                const auto& links = plan.links.at({ChipCoord{mesh_row, position}, direction});
                tt::tt_fabric::append_fabric_connection_rt_args<ProgramDescriptor>(
                    *mesh, node(mesh_row, position), peer_node, links.at(link), program_descriptor, core, fabric_args);
            } else {
                for (uint32_t value : {0u, 0u, 0u}) {
                    sender_args.push_back(value);
                }
            }
            sender_args.append(share_k_ranges);
            sender_args.append(column_xs);
            sender_args.append(pair_ys);
            sender_args.append(fabric_args);
            port_writer.emplace_runtime_args(core, sender_args);
        }

        program_descriptor.kernels.push_back(std::move(reader));
        program_descriptor.kernels.push_back(std::move(writer));
        program_descriptor.kernels.push_back(std::move(compute));
        program_descriptor.kernels.push_back(std::move(port_reader));
        program_descriptor.kernels.push_back(std::move(port_writer));
        // (engine_common.hpp: PARTNER_EXCHANGE, SLOTS_FREE, W_LOADED, W_GATE_COUNT, W_GATE_GO, BLOCK_GATE_COUNT,
        // BLOCK_GATE_GO)
        for (uint32_t id = 0; id < 7; ++id) {
            program_descriptor.semaphores.push_back(
                tt::tt_metal::SemaphoreDescriptor{.id = id, .core_ranges = engine, .initial_value = 0});
        }

        workload.programs.push_back(
            {ttnn::MeshCoordinateRange(mesh_coordinate, mesh_coordinate), std::move(program_descriptor)});
    }
    return workload;
}

// ---------------------------------------------------------------------------------------------------------------------
// all-to-all
// ---------------------------------------------------------------------------------------------------------------------
tt::tt_metal::WorkloadDescriptor moe_tail_factory_detail::all_to_all_workload(
    const MoETailParams& attributes,
    const MoETailInputs& args,
    Tensor& output,
    const ttnn::MeshCoordinateRangeSet& tensor_coordinates) {
    using namespace all_to_all_cb;
    const auto& plan = *attributes.all_to_all_plan;
    const auto& geometry = plan.geometry;
    const auto& schedule = plan.schedule;
    auto* mesh = args.y_own.device();
    const auto bfloat16_format = tt::DataFormat::Float16_b;

    Buffer* y_own_buffer = args.y_own.buffer();
    Buffer* shared_expert_buffer = args.shared_expert.has_value() ? args.shared_expert->buffer() : nullptr;
    Buffer* output_buffer = output.buffer();
    Buffer* staging_buffer = args.staging->buffer();
    Buffer* arrival_buffer = args.arrival->buffer();
    Buffer* ready_buffer = args.ready.buffer();

    auto virtual_core = [&](const CoreCoord& core) { return mesh->worker_core_from_logical_core(core); };
    auto node = [&](uint32_t mesh_row, uint32_t position) {
        return mesh->get_fabric_node_id(mesh_coord(ChipCoord{mesh_row, position}, plan.topology));
    };
    const CoreRangeSet workers = cells(plan.workers);
    std::vector<uint32_t> worker_xys;  // (x | y << 16 each: the carrying ports release them once a call)
    for (const auto& worker : plan.workers) {
        worker_xys.push_back(packed_xy(virtual_core(worker)));
    }
    TT_FATAL(
        worker_xys.size() == geometry.num_workers,
        "moe_tail (all-to-all): {} workers, {} planned",
        worker_xys.size(),
        geometry.num_workers);
    const uint32_t worker_columns = geometry.worker_columns,
                   has_shared_expert = shared_expert_buffer != nullptr ? 1u : 0u;
    // row blocks: every kernel loop over them itself (block b's rows of y_own / the shared expert / the output: b x
    // stride further)
    const uint32_t row_blocks = std::max<uint32_t>(1, attributes.row_blocks);
    const uint32_t rows_block_stride_bytes = block_stride(args.y_own, row_blocks, "y_own");
    const uint32_t shared_expert_block_stride_bytes =
        shared_expert_buffer != nullptr ? block_stride(*args.shared_expert, row_blocks, "shared_expert") : 0u;
    const uint32_t output_block_stride_bytes = block_stride(output, row_blocks, "the output");
    // a padded forward: the padding row tiles skipped (padded_row_tiles.hpp) - the span's buffers and each chip's
    // padding args
    Buffer* span_first = args.metadata.has_value()       ? args.metadata->buffer()
                         : args.actual_start.has_value() ? args.actual_start->buffer()
                                                         : nullptr;
    Buffer* span_end = args.metadata.has_value() || !args.actual_end.has_value() ? nullptr : args.actual_end->buffer();
    const bool skip_padding = span_first != nullptr;
    TT_FATAL(
        !skip_padding || geometry.owned_row_tiles <= 64,
        "moe_tail (all-to-all): a padded forward's {} row tiles a chip (at most 64)",
        geometry.owned_row_tiles);
    TT_FATAL(
        geometry.block_tiles % worker_columns == 0,
        "moe_tail (all-to-all): a worker's {} columns straddle a chip's block of {}",
        worker_columns,
        geometry.block_tiles);
    auto padding_args = [&](KernelDescriptor::RTArgList& list, uint32_t sp_row, uint32_t position) {
        list.push_back(span_first);
        if (span_end != nullptr) {
            list.push_back(span_end);
        } else {
            list.push_back(0u);
        }
        const uint32_t call_rows =
            TILE_HEIGHT * geometry.owned_row_tiles * geometry.tp_size;  // (a block's rows of my SP row)
        for (uint32_t value :
             {sp_row,
              plan.topology.R,
              attributes.row_tokens,
              attributes.first_block * call_rows + position * TILE_HEIGHT * geometry.owned_row_tiles,
              call_rows}) {
            list.push_back(value);
        }
    };
    // skip_padding: a named compile-time arg of every all-to-all kernel; the padded forward's chunks / metadata stay
    // defines (moe_head's padded_rows.hpp, shared with the head, reads them)
    auto padding_config = [&](KernelDescriptor& descriptor) {
        descriptor.named_compile_time_args.emplace_back("skip_padding", skip_padding ? 1u : 0u);
        if (skip_padding) {
            descriptor.defines.emplace_back("PADDED_FORWARD_CHUNKS", std::to_string(attributes.forward_chunks));
            if (args.metadata.has_value()) {
                descriptor.defines.emplace_back("PADDED_FORWARD_METADATA", "1");
            }
        }
    };

    tt::tt_metal::WorkloadDescriptor workload;
    for (const auto& mesh_coordinate : coordinates_of(tensor_coordinates)) {
        const ChipCoord chip = chip_of(mesh_coordinate, plan.topology);
        const uint32_t mesh_row = chip.r, position = chip.c;
        // (this chip's port cores; the port kernels and their CB go on every chip's - the union, the same program on
        // every chip (one program for the mesh): a core that is no port of this chip runs them idle)
        std::vector<std::pair<TailSchedule::Port, CoreCoord>> my_ports;
        std::set<std::pair<size_t, size_t>> my_port_cores;
        for (const auto& [key, core] : plan.ports) {
            if (std::get<0>(key) == chip) {
                my_ports.push_back({{position, std::get<1>(key), std::get<2>(key)}, core});
                my_port_cores.insert({core.x, core.y});
            }
        }
        const CoreRangeSet& port_cores = plan.port_union;

        ProgramDescriptor program_descriptor;
        // (the z-ready words, a row tile's 16 B apart: this call's, zeroed by each carrying port - the first CB on the
        // workers and the ports, so one address on all of them: a worker's get_write_ptr is the ports')
        program_descriptor.cbs.push_back(
            cb(CB_Z_READY,
               workers.merge(port_cores),
               16 * geometry.owned_row_tiles,
               tt::DataFormat::UInt32,
               16 * geometry.owned_row_tiles));
        // (every block worker_columns pages - the port's pages_per_run x runs_per_group: no block straddles a ring's
        // end)
        program_descriptor.cbs.push_back(
            cb(CB_ROWS,
               workers,
               ring_pages(2 * worker_columns, {worker_columns}) * kBf16TileBytes,
               bfloat16_format,
               kBf16TileBytes));
        program_descriptor.cbs.push_back(
            cb(CB_TILIZED,
               workers,
               ring_pages(worker_columns, {worker_columns}) * kBf16TileBytes,
               bfloat16_format,
               kBf16TileBytes));
        program_descriptor.cbs.push_back(
            cb(CB_SHARED_EXPERT,
               workers,
               ring_pages(2 * worker_columns, {worker_columns}) * kBf16TileBytes,
               bfloat16_format,
               kBf16TileBytes));
        program_descriptor.cbs.push_back(
            cb(CB_OUTPUT,
               workers,
               ring_pages(2 * worker_columns, {worker_columns}) * kBf16TileBytes,
               bfloat16_format,
               kBf16TileBytes));
        program_descriptor.cbs.push_back(
            cb(CB_PORT_STAGING,
               port_cores,
               ring_pages(
                   2 * geometry.runs_per_group * geometry.pages_per_run,
                   {geometry.runs_per_group * geometry.pages_per_run}) *
                   kBf16TileBytes,
               bfloat16_format,
               kBf16TileBytes));
        if (skip_padding) {
            // (128 B pages: the readers' metadata words, the writers' 256 B on, the workers' zero tile 512 B on -
            // padded_row_tiles.hpp)
            program_descriptor.cbs.push_back(
                cb(CB_PADDING_SCRATCH, workers.merge(port_cores), 512 + kBf16TileBytes, tt::DataFormat::UInt32, 128));
            program_descriptor.cbs.push_back(cb(CB_LIVE_ROW_TILES, workers, 128, tt::DataFormat::UInt32, 128));
        }

        auto worker_reader = kernel(
            "all_to_all/worker_input_reader.cpp", workers, data_movement(DataMovementProcessor::RISCV_1, NOC::NOC_1));
        worker_reader.compile_time_args = {worker_columns, has_shared_expert};
        append_accessor(worker_reader, y_own_buffer);
        append_accessor(worker_reader, shared_expert_buffer != nullptr ? shared_expert_buffer : y_own_buffer);
        worker_reader.compile_time_args.insert(
            worker_reader.compile_time_args.end(),
            {row_blocks, rows_block_stride_bytes, shared_expert_block_stride_bytes});
        auto worker_compute = kernel(
            "all_to_all/worker_tilize_add_compute.cpp",
            workers,
            ComputeConfigDescriptor{
                .math_fidelity = tt::tt_metal::MathFidelity::HiFi2,
                .fp32_dest_acc_en = false,
                .math_approx_mode = false});
        worker_compute.compile_time_args = {worker_columns, has_shared_expert, row_blocks};
        // (NOC0: the staging buffer lands in DRAM through the NOC0 endpoint the port reader reads it through - a read
        // through the other NoC endpoint can return a page's previous contents after the write's ack, the BH DRAM
        // cross-port stale read; the worker reader, NOC1, reads only an earlier program's y and shared expert)
        auto worker_writer = kernel(
            "all_to_all/worker_output_writer.cpp", workers, data_movement(DataMovementProcessor::RISCV_0, NOC::NOC_0));
        worker_writer.compile_time_args = {worker_columns};
        append_accessor(worker_writer, output_buffer);
        append_accessor(worker_writer, staging_buffer);
        worker_writer.compile_time_args.insert(
            worker_writer.compile_time_args.end(), {row_blocks, output_block_stride_bytes});
        // (the port kernels' compile-time args 0 - 5: port_common.hpp)
        const std::vector<uint32_t> port_compile_args = {
            CB_PORT_STAGING,
            kBf16TileBytes,
            geometry.pages_per_run,
            geometry.num_dram_banks,
            geometry.runs_per_group,
            kArrivalIncrementEvery};
        auto port_reader = kernel(
            "all_to_all/port_staging_reader.cpp",
            port_cores,
            data_movement(DataMovementProcessor::RISCV_1, NOC::NOC_0));
        port_reader.compile_time_args = port_compile_args;
        append_accessor(port_reader, staging_buffer);
        port_reader.compile_time_args.push_back(row_blocks);
        auto port_sender = kernel(
            "all_to_all/port_fabric_sender.cpp", port_cores, data_movement(DataMovementProcessor::RISCV_0, NOC::NOC_1));
        port_sender.compile_time_args = port_compile_args;
        append_accessor(port_sender, output_buffer);
        port_sender.compile_time_args.insert(
            port_sender.compile_time_args.end(), {row_blocks, output_block_stride_bytes});
        for (auto* kernel_descriptor : {&worker_reader, &worker_compute, &worker_writer, &port_reader, &port_sender}) {
            padding_config(*kernel_descriptor);
        }

        for (const auto& [port, core] : my_ports) {
            const auto [port_position, direction, link] = port;
            const auto destinations = schedule.destinations(position, direction);
            TT_FATAL(
                destinations.size() <= kMaxPortDestinations,
                "moe_tail (all-to-all): a port's {} destinations",
                destinations.size());
            KernelDescriptor::RTArgList reader_args;
            reader_args.push_back(staging_buffer);
            reader_args.push_back(geometry.num_workers);
            for (uint32_t value :
                 {geometry.owned_row_tiles,
                  geometry.hidden_tiles,
                  geometry.block_tiles,
                  link,
                  plan.num_links,
                  static_cast<uint32_t>(destinations.size())}) {
                reader_args.push_back(value);
            }
            if (skip_padding) {  // (at a fixed place: the same args bind the same buffers on every chip)
                padding_args(reader_args, mesh_row, position);
            }
            for (const auto& destination : destinations) {
                reader_args.push_back(destination.position | destination.part << 8);
            }
            reader_args.append(worker_xys);  // (the workers it releases once its z-ready words are zeroed)
            port_reader.emplace_runtime_args(core, reader_args);

            const auto readies = schedule.readies(port);
            std::set<uint32_t> destination_chips;
            for (const auto& destination : destinations) {
                destination_chips.insert(destination.position);
            }
            KernelDescriptor::RTArgList sender_args;
            sender_args.push_back(output_buffer);
            for (uint32_t value : {position, geometry.owned_row_tiles, geometry.block_tiles, link, plan.num_links}) {
                sender_args.push_back(value);
            }
            sender_args.push_back(arrival_buffer);
            sender_args.push_back(schedule.expected_arrivals(port));
            sender_args.push_back(ready_buffer);
            sender_args.push_back(static_cast<uint32_t>(destination_chips.size()));
            if (skip_padding) {
                padding_args(sender_args, mesh_row, position);
            }
            sender_args.push_back(static_cast<uint32_t>(readies.size()));
            for (const auto& [ready_position, ready_direction, ready_link] : readies) {
                const auto ready_node = node(mesh_row, ready_position);
                sender_args.push_back(packed_xy(
                    virtual_core(plan.ports.at({ChipCoord{mesh_row, ready_position}, ready_direction, ready_link}))));
                sender_args.push_back(static_cast<uint32_t>(ready_node.chip_id));
                sender_args.push_back(*ready_node.mesh_id);
                sender_args.push_back(schedule.hops(position, ready_position, direction));
            }
            sender_args.push_back(static_cast<uint32_t>(destinations.size()));
            for (const auto& destination : destinations) {
                const auto destination_node = node(mesh_row, destination.position);
                sender_args.push_back(destination.position | destination.part << 8);
                sender_args.push_back(static_cast<uint32_t>(destination_node.chip_id));
                sender_args.push_back(*destination_node.mesh_id);
                sender_args.push_back(schedule.hops(position, destination.position, direction));
                sender_args.push_back(geometry.owned_row_tiles * schedule.runs(destination.part, link));
                sender_args.push_back(packed_xy(virtual_core(
                    plan.ports.at({ChipCoord{mesh_row, destination.position}, opposite(direction), link}))));
            }
            if (!destinations.empty() || !readies.empty()) {
                const auto peer = row_neighbor(position, geometry.tp_size, plan.ring, direction);
                TT_FATAL(
                    peer.has_value(),
                    "moe_tail (all-to-all): ({}, {}) has no {} link",
                    mesh_row,
                    position,
                    direction == TailDirection::Forward ? "forward" : "backward");
                const auto& links = plan.links.at({chip, direction});
                std::vector<uint32_t> fabric_args;
                tt::tt_fabric::append_fabric_connection_rt_args<ProgramDescriptor>(
                    *mesh,
                    node(mesh_row, position),
                    node(mesh_row, *peer),
                    links.at(link),
                    program_descriptor,
                    core,
                    fabric_args);
                sender_args.append(fabric_args);
            }
            port_sender.emplace_runtime_args(core, sender_args);
        }
        // (the union's cores that are no port of this chip: idle - no destination, no ready, nothing expected)
        for (const auto& core : cores_of_set(port_cores)) {
            if (my_port_cores.contains({core.x, core.y})) {
                continue;
            }
            KernelDescriptor::RTArgList reader_args;
            reader_args.push_back(staging_buffer);
            reader_args.push_back(geometry.num_workers);
            for (uint32_t value :
                 {geometry.owned_row_tiles, geometry.hidden_tiles, geometry.block_tiles, 0u, plan.num_links, 0u}) {
                reader_args.push_back(value);
            }
            if (skip_padding) {
                padding_args(reader_args, mesh_row, position);
            }
            port_reader.emplace_runtime_args(core, reader_args);
            KernelDescriptor::RTArgList sender_args;
            sender_args.push_back(output_buffer);
            for (uint32_t value : {position, geometry.owned_row_tiles, geometry.block_tiles, 0u, plan.num_links}) {
                sender_args.push_back(value);
            }
            sender_args.push_back(arrival_buffer);
            sender_args.push_back(0u);
            sender_args.push_back(ready_buffer);
            sender_args.push_back(0u);
            if (skip_padding) {
                padding_args(sender_args, mesh_row, position);
            }
            sender_args.push_back(0u);  // (no ready)
            sender_args.push_back(0u);  // (no destination)
            port_sender.emplace_runtime_args(core, sender_args);
        }

        // (the ports that carry something: the workers' z-ready words; a port with no destination neither waits nor
        // re-arms)
        std::vector<uint32_t> carrying_port_xys;
        for (const auto& [port, core] : my_ports) {
            if (!schedule.destinations(position, std::get<1>(port)).empty()) {
                carrying_port_xys.push_back(packed_xy(virtual_core(core)));
            }
        }
        for (uint32_t worker_index = 0; worker_index < plan.workers.size(); ++worker_index) {
            const CoreCoord& core = plan.workers[worker_index];
            const uint32_t first_column = worker_index * worker_columns;
            KernelDescriptor::RTArgList reader_args;
            reader_args.push_back(y_own_buffer);
            if (shared_expert_buffer != nullptr) {
                reader_args.push_back(shared_expert_buffer);
            } else {
                reader_args.push_back(0u);
            }
            for (uint32_t value : {first_column, geometry.owned_row_tiles, geometry.hidden_tiles}) {
                reader_args.push_back(value);
            }
            if (skip_padding) {
                padding_args(reader_args, mesh_row, position);
            }
            worker_reader.emplace_runtime_args(core, reader_args);
            worker_compute.runtime_args.emplace_back(core, std::vector<uint32_t>{geometry.owned_row_tiles});
            KernelDescriptor::RTArgList writer_args;
            writer_args.push_back(output_buffer);
            writer_args.push_back(staging_buffer);
            for (uint32_t value :
                 {first_column, geometry.owned_row_tiles, geometry.hidden_tiles, geometry.block_tiles, position}) {
                writer_args.push_back(value);
            }
            if (skip_padding) {
                padding_args(writer_args, mesh_row, position);
            }
            writer_args.push_back(static_cast<uint32_t>(carrying_port_xys.size()));
            writer_args.append(carrying_port_xys);
            worker_writer.emplace_runtime_args(core, writer_args);
        }

        program_descriptor.kernels.push_back(std::move(worker_reader));
        program_descriptor.kernels.push_back(std::move(worker_compute));
        program_descriptor.kernels.push_back(std::move(worker_writer));
        program_descriptor.kernels.push_back(std::move(port_reader));
        program_descriptor.kernels.push_back(std::move(port_sender));
        // (the workers' zero gate, ZERO_GATE_SEM)
        program_descriptor.semaphores.push_back(
            tt::tt_metal::SemaphoreDescriptor{.id = 0, .core_ranges = workers, .initial_value = 0});
        workload.programs.push_back(
            {ttnn::MeshCoordinateRange(mesh_coordinate, mesh_coordinate), std::move(program_descriptor)});
    }
    return workload;
}
}  // namespace ttnn::operations::experimental::moe_block::moe_tail

// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pirotski_matmul_device_operation.hpp"

#include <algorithm>
#include <map>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_page_mapping.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/compute_throttle_utils.hpp"

namespace ttnn::experimental::nomic_embed_text_v2_moe {

namespace {

const std::string kKernelDirectory =
    "ttnn/cpp/ttnn/operations/experimental/nomic_embed_text_v2_moe/pirotski_matmul/device/kernels/";
// minimal_matmul's compute kernel, unchanged: its CB indices, argument order and defines are this op's.
const std::string kComputeKernel = "ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/compute.cpp";

// Two slots per input ring: the next block loads while the current one computes.
constexpr uint32_t kRingSlots = 2;

enum CircularBufferIndex : uint8_t {
    kInputRing = tt::CBIndex::c_0,
    kWeightRing = tt::CBIndex::c_1,
    kOutput = tt::CBIndex::c_2,
    kPartials = tt::CBIndex::c_3,
    kBias = tt::CBIndex::c_4,
};

enum SemaphoreId : uint32_t {
    kInputGranted,
    kInputArrived,
    kWeightGranted,
    kWeightArrived,
    kDrainPosted,
    kDrainDone,
};

tt::tt_metal::CBDescriptor ring(const CoreRangeSet& cores, uint8_t index, tt::DataFormat format, uint32_t tiles) {
    const uint32_t tile_bytes = tt::tile_size(format);
    return tt::tt_metal::CBDescriptor{
        .total_size = tile_bytes * tiles,
        .core_ranges = cores,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = index,
            .data_format = format,
            .page_size = tile_bytes,
        }}},
    };
}

// The kernels' marker of a padding row in a row table (stream_protocol.hpp kNoRow) and their list bounds.
constexpr uint32_t kNoRow = 0xFFFFFFFF;
constexpr uint32_t kMaxPieces = 64;

// Where each output page lives: its core (index into all_cores) and its page on that core.
struct OutputPages {
    std::vector<CoreCoord> cores;
    std::vector<int32_t> core_of_page;
    std::vector<uint32_t> device_page;
};

OutputPages output_pages(tt::tt_metal::Buffer& buffer, uint32_t pages) {
    const auto& mapping = *buffer.get_buffer_page_mapping();
    OutputPages out{
        .cores = mapping.all_cores,
        .core_of_page = std::vector<int32_t>(pages, -1),
        .device_page = std::vector<uint32_t>(pages, 0)};
    for (const auto mapped : mapping) {
        if (mapped.host_page < pages) {
            out.core_of_page[mapped.host_page] = static_cast<int32_t>(mapped.core_id);
            out.device_page[mapped.host_page] = mapped.device_page;
        }
    }
    return out;
}

// The rows each core column computes, m_share entries each, kNoRow-padded at the end. Contiguous: column x takes
// [x * m_share, (x + 1) * m_share). Strided: column x takes, in order, the rows whose tile (m, n_ref) lands in grid
// column x (mod the grid's columns), when no column then exceeds its share; with the generated layer norm's ND
// layout (row groups of r tile rows, G cores a group, round robin over the grid row-major) a column's rows then land
// in the grid columns around it rather than across one grid row.
std::vector<std::vector<uint32_t>> column_rows(
    uint32_t columns,
    uint32_t m_share,
    uint32_t m_tiles,
    uint32_t n_tiles,
    const OutputPages* pages,
    uint32_t n_ref,
    bool* strided) {
    std::vector<std::vector<uint32_t>> rows(columns);
    if (pages != nullptr && *strided) {
        bool fits = true;
        for (uint32_t m = 0; m < m_tiles && fits; ++m) {
            const int32_t core = pages->core_of_page[m * n_tiles + n_ref];
            if (core < 0) {
                fits = false;
                break;
            }
            auto& column = rows[pages->cores[core].x % columns];
            column.push_back(m);
            fits = column.size() <= m_share;
        }
        if (fits) {
            for (auto& column : rows) {
                column.resize(m_share, kNoRow);
            }
            return rows;
        }
        *strided = false;
        rows.assign(columns, {});
    }
    for (uint32_t x = 0; x < columns; ++x) {
        for (uint32_t i = 0; i < m_share; ++i) {
            const uint32_t m = x * m_share + i;
            rows[x].push_back(m < m_tiles ? m : kNoRow);
        }
    }
    return rows;
}

// One write of the drain: `count` tiles of local row `row` of a core's last block, `n_offset` tiles into its N
// block, all on core `dest`'s contiguous pages.
struct DrainPiece {
    uint32_t source = 0;  // index into the plan's sources
    uint32_t row = 0;
    uint32_t n_offset = 0;
    uint32_t count = 0;
    uint32_t dest = 0;  // index into OutputPages::cores
    uint32_t noc = 0;
};

// Blackhole's (and Wormhole's) NoCs are unidirectional tori: NOC0 runs toward +X and +Y, NOC1 toward -X and -Y (it
// takes NOC0's route between two cores backwards). A write's hops on each, in NoC coordinates on an nx x ny grid.
uint32_t noc_hops(uint32_t noc, CoreCoord src, CoreCoord dst, uint32_t nx, uint32_t ny) {
    const uint32_t forward = (dst.x + nx - src.x) % nx + (dst.y + ny - src.y) % ny;
    const uint32_t backward = (src.x + nx - dst.x) % nx + (src.y + ny - dst.y) % ny;
    return noc == 0 ? forward : backward;
}

// Gives each piece a NoC, so a RISC (NOC0 is RISCV_0's, NOC1 RISCV_1's), core by core: largest hop-weighted piece
// first, onto the RISC whose load (bytes x (hops + kIssueHops)) stays the lower. Measured (c13 probe), a RISC's drain
// time follows its bytes x hops (each hop's latency holds its outstanding writes longer), not the busiest link's
// bytes. Returns the largest load over every RISC, the plan's cost.
double plan_nocs(
    std::vector<DrainPiece>& pieces,
    uint32_t sources,
    const std::vector<CoreCoord>& source_noc,
    const std::vector<CoreCoord>& dest_noc,
    CoreCoord noc_grid,
    uint32_t tile_bytes) {
    constexpr double kIssueHops = 2.0;
    uint32_t nx = noc_grid.x;
    uint32_t ny = noc_grid.y;
    for (const auto& c : source_noc) {
        nx = std::max<uint32_t>(nx, c.x + 1);
        ny = std::max<uint32_t>(ny, c.y + 1);
    }
    for (const auto& c : dest_noc) {
        nx = std::max<uint32_t>(nx, c.x + 1);
        ny = std::max<uint32_t>(ny, c.y + 1);
    }
    const auto weight = [&](const DrainPiece& piece, uint32_t noc) {
        return static_cast<double>(piece.count) * tile_bytes *
               (noc_hops(noc, source_noc[piece.source], dest_noc[piece.dest], nx, ny) + kIssueHops);
    };
    std::vector<size_t> order(pieces.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        if (pieces[a].source != pieces[b].source) {
            return pieces[a].source < pieces[b].source;
        }
        return std::min(weight(pieces[a], 0), weight(pieces[a], 1)) >
               std::min(weight(pieces[b], 0), weight(pieces[b], 1));
    });
    std::vector<double> load(static_cast<size_t>(sources) * 2, 0.0);
    double cost = 0.0;
    for (size_t i : order) {
        auto& piece = pieces[i];
        double* risc = &load[static_cast<size_t>(piece.source) * 2];
        const double on0 = risc[0] + weight(piece, 0);
        const double on1 = risc[1] + weight(piece, 1);
        piece.noc = on0 <= on1 ? 0 : 1;
        risc[piece.noc] = piece.noc == 0 ? on0 : on1;
        cost = std::max(cost, risc[piece.noc]);
    }
    return cost;
}

// A piece as the kernels read it (stream_protocol.hpp unpack_piece).
uint32_t pack_piece(const DrainPiece& piece) { return (piece.row << 16) | (piece.n_offset << 8) | piece.count; }

}  // namespace

tt::tt_metal::ProgramDescriptor PirotskiMatmulDeviceOperation::create_descriptor(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    using tt::tt_metal::ComputeConfigDescriptor;
    using tt::tt_metal::DataMovementConfigDescriptor;
    using tt::tt_metal::DataMovementProcessor;
    using tt::tt_metal::KernelDescriptor;
    using tt::tt_metal::NOC;
    using tt::tt_metal::TensorAccessorArgs;

    const auto layout = compute_layout(attributes, tensor_args);
    const auto& input = tensor_args.input;
    const auto& weight = tensor_args.weight;
    const auto& config = attributes.config;
    const bool has_bias = tensor_args.bias.has_value();
    auto* device = input.device();
    const auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), attributes.compute_kernel_config);

    const uint32_t columns = layout.columns;
    const uint32_t rows = layout.rows;
    const bool has_strip = layout.strip_m_tiles > 0;
    const CoreRangeSet main_cores{CoreRange(CoreCoord(0, 0), CoreCoord(columns - 1, rows - 1))};
    const CoreRangeSet strip_cores =
        has_strip ? CoreRangeSet{CoreRange(CoreCoord(0, rows), CoreCoord(columns - 1, rows + layout.strip_rows - 1))}
                  : CoreRangeSet{};
    const CoreRangeSet all_cores = has_strip ? main_cores.merge(strip_cores) : main_cores;
    const uint32_t m_block = config.M_block_size;
    const uint32_t k_block = config.K_block_size;
    const uint32_t n_block = config.N_block_size;
    // drain_plan / strided_rows (validated in compute_layout: a sharded L1 output, not packed directly) work from
    // where each output page lives. A strided assignment that would overfill a column falls back to the contiguous
    // one, as does one over the split layer norm's input, which is read by contiguous rows.
    const bool drain_plan = attributes.drain_plan;
    bool strided = attributes.strided_rows && layout.input_shard_width == 0;
    std::optional<OutputPages> pages;
    if (drain_plan || attributes.strided_rows) {
        pages = output_pages(*output.buffer(), layout.m_tiles * layout.n_tiles);
    }
    // An interleaved output's last block goes out on both RISCs (even rows RISCV_1, odd RISCV_0), or per the plan;
    // a direct-packed one is never written.
    const bool split_drain = !layout.direct_pack && !drain_plan;

    const auto input_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    const auto weight_format = tt::tt_metal::datatype_to_dataformat_converter(weight.dtype());
    const auto output_format = tt::tt_metal::datatype_to_dataformat_converter(attributes.dtype);
    const auto partials_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    const auto bias_format = has_bias ? tt::tt_metal::datatype_to_dataformat_converter(tensor_args.bias->dtype())
                                      : tt::DataFormat::Float16_b;

    // The cores in drain order (the config grid column by column, then the strip), each with its rows (local row i
    // is output row table[i], kNoRow past its end), the N tiles of its last block, and its pieces of the drain.
    struct Source {
        CoreCoord core;
        std::vector<uint32_t> table;
        uint32_t last_block_row = 0;
        uint32_t m_block = 0;
        uint32_t n_start = 0;  // the core's N share (the strip's lane) starts here
        uint32_t n_tile = 0;   // its last block's first N tile
        uint32_t n_count = 0;
        std::vector<uint32_t> pieces[2];  // packed, per NoC
    };
    // One candidate arrangement: rows strided by output tile column n_ref (when strided), and the config grid's
    // rows taking the N shares in reverse (reverse_n): row y the share of row rows - 1 - y, strip row j the lane of
    // strip row strip_rows - 1 - j. Either changes only where a tile is computed, never its K order.
    struct Arrangement {
        bool strided = false;
        bool reverse_n = false;
        std::vector<Source> sources;
        double cost = 0.0;
    };
    std::vector<CoreCoord> dest_noc;
    if (pages) {
        for (const auto& core : pages->cores) {
            dest_noc.push_back(device->worker_core_from_logical_core(core));
        }
    }
    const auto arrange = [&](uint32_t n_ref, bool want_strided, bool reverse_n) {
        Arrangement arrangement{.strided = want_strided, .reverse_n = reverse_n};
        const auto rows_of_column = column_rows(
            columns,
            layout.m_tiles_per_core,
            layout.m_tiles,
            layout.n_tiles,
            pages ? &*pages : nullptr,
            n_ref,
            &arrangement.strided);
        auto& sources = arrangement.sources;
        for (uint32_t x = 0; x < columns; ++x) {
            const auto& column = rows_of_column[x];
            for (uint32_t y = 0; y < rows; ++y) {
                const uint32_t share = reverse_n ? rows - 1 - y : y;
                sources.push_back(Source{
                    .core = CoreCoord(x, y),
                    .table = std::vector<uint32_t>(column.begin(), column.begin() + layout.main_m_tiles),
                    .last_block_row = (layout.m_blocks_per_core - 1) * m_block,
                    .m_block = m_block,
                    .n_start = share * layout.n_tiles_per_core,
                    .n_tile = share * layout.n_tiles_per_core + (layout.n_blocks_per_core - 1) * n_block,
                    .n_count = n_block});
            }
            for (uint32_t j = 0; has_strip && j < layout.strip_rows; ++j) {
                const uint32_t lane = reverse_n ? layout.strip_rows - 1 - j : j;
                sources.push_back(Source{
                    .core = CoreCoord(x, rows + j),
                    .table = std::vector<uint32_t>(column.begin() + layout.main_m_tiles, column.end()),
                    .last_block_row = 0,
                    .m_block = layout.strip_m_tiles,
                    .n_start = lane * layout.strip_n_tiles,
                    .n_tile = lane * layout.strip_n_tiles,
                    .n_count = layout.strip_n_tiles});
            }
        }
        if (!drain_plan) {
            return arrangement;
        }
        // The last block's pieces: runs of a row's tiles on one core's consecutive pages, one write each.
        std::vector<DrainPiece> pieces;
        for (uint32_t s = 0; s < sources.size(); ++s) {
            const auto& source = sources[s];
            for (uint32_t row = 0; row < source.m_block; ++row) {
                const uint32_t local = source.last_block_row + row;
                const uint32_t m = local < source.table.size() ? source.table[local] : kNoRow;
                if (m == kNoRow) {
                    break;  // a block's valid rows are a prefix of it
                }
                for (uint32_t n = 0; n < source.n_count; ++n) {
                    const uint32_t page = m * layout.n_tiles + source.n_tile + n;
                    const int32_t core = pages->core_of_page[page];
                    TT_FATAL(
                        core >= 0, "pirotski_matmul: output page {} has no core in the output's page mapping", page);
                    const bool extends = n > 0 && pieces.back().source == s && pieces.back().row == row &&
                                         pieces.back().dest == static_cast<uint32_t>(core) &&
                                         pages->device_page[page] == pages->device_page[page - 1] + 1;
                    if (extends) {
                        ++pieces.back().count;
                    } else {
                        pieces.push_back(DrainPiece{
                            .source = s, .row = row, .n_offset = n, .count = 1, .dest = static_cast<uint32_t>(core)});
                    }
                }
            }
        }
        std::vector<CoreCoord> source_noc;
        for (const auto& source : sources) {
            source_noc.push_back(device->worker_core_from_logical_core(source.core));
        }
        arrangement.cost =
            plan_nocs(pieces, sources.size(), source_noc, dest_noc, device->grid_size(), tt::tile_size(output_format));
        // Each RISC takes its pieces in row order, as compute pushes the rows.
        std::stable_sort(pieces.begin(), pieces.end(), [](const DrainPiece& a, const DrainPiece& b) {
            return std::tie(a.source, a.row, a.n_offset) < std::tie(b.source, b.row, b.n_offset);
        });
        for (const auto& piece : pieces) {
            TT_FATAL(
                piece.row < (1u << 16) && piece.n_offset < 256 && piece.count < 256,
                "pirotski_matmul: a drain piece does not fit its word");
            sources[piece.source].pieces[piece.noc].push_back(pack_piece(piece));
        }
        for (const auto& source : sources) {
            TT_FATAL(
                source.pieces[0].size() <= kMaxPieces && source.pieces[1].size() <= kMaxPieces,
                "pirotski_matmul: core {} has {} / {} drain pieces, at most {} a RISC",
                source.core,
                source.pieces[0].size(),
                source.pieces[1].size(),
                kMaxPieces);
        }
        return arrangement;
    };
    // With a drain plan the cheapest arrangement (planned cost, ties to the first) of: the rows strided by the
    // output's middle tile column, its first or its last (when strided rows are asked for), each with the grid rows'
    // N shares in order or reversed; without one, the rows as asked and the N shares in order. With a strip, only
    // the last tile column's: at 128 tile rows of the generated norm's ND layout the cost picks the middle column,
    // whose drain left some grid rows' K loops up to 0.7 us behind the rest, which the cost does not see (out_proj
    // 35.25 us against 34.97, fc2 83.50 against 83.10).
    const uint32_t n_mid = layout.n_tiles >= 2 ? layout.n_tiles / 2 - 1 : 0;
    std::vector<uint32_t> n_refs{n_mid};
    if (drain_plan && strided) {
        n_refs =
            has_strip ? std::vector<uint32_t>{layout.n_tiles - 1} : std::vector<uint32_t>{n_mid, 0, layout.n_tiles - 1};
    }
    Arrangement chosen = arrange(n_refs.front(), strided, false);
    if (drain_plan) {
        for (uint32_t n_ref : n_refs) {
            for (bool reverse_n : {false, true}) {
                if (n_ref == n_refs.front() && !reverse_n) {
                    continue;  // the first candidate, already arranged
                }
                auto candidate = arrange(n_ref, strided, reverse_n);
                if (candidate.cost < chosen.cost) {
                    chosen = std::move(candidate);
                }
            }
        }
    }
    strided = chosen.strided;
    const auto& sources = chosen.sources;

    std::map<CoreCoord, size_t> source_of_core;
    for (size_t i = 0; i < sources.size(); ++i) {
        source_of_core[sources[i].core] = i;
    }

    tt::tt_metal::ProgramDescriptor descriptor;
    descriptor.cbs.push_back(ring(main_cores, kInputRing, input_format, kRingSlots * m_block * k_block));
    descriptor.cbs.push_back(ring(
        main_cores,
        kWeightRing,
        weight_format,
        (attributes.resident_weights ? layout.k_blocks * layout.n_blocks_per_core : kRingSlots) * k_block * n_block));
    if (layout.direct_pack) {
        // Compute packs straight into the output's shard; rows past M land in the last column's shard padding.
        auto out = ring(main_cores, kOutput, output_format, layout.m_tiles_per_core * n_block);
        out.buffer = output.buffer();
        descriptor.cbs.push_back(out);
    } else {
        descriptor.cbs.push_back(ring(main_cores, kOutput, output_format, m_block * n_block));
    }
    descriptor.cbs.push_back(ring(main_cores, kPartials, partials_format, m_block * n_block));
    if (has_bias) {
        // One N block: with one a core it is read once and re-pushed unread for every output block.
        descriptor.cbs.push_back(ring(main_cores, kBias, bias_format, n_block));
    }
    if (has_strip) {
        const uint32_t strip_block = layout.strip_m_tiles * layout.strip_n_tiles;
        descriptor.cbs.push_back(
            ring(strip_cores, kInputRing, input_format, kRingSlots * layout.strip_m_tiles * k_block));
        descriptor.cbs.push_back(
            ring(strip_cores, kWeightRing, weight_format, kRingSlots * k_block * layout.strip_n_tiles));
        descriptor.cbs.push_back(ring(strip_cores, kOutput, output_format, strip_block));
        descriptor.cbs.push_back(ring(strip_cores, kPartials, partials_format, strip_block));
        if (has_bias) {
            descriptor.cbs.push_back(ring(strip_cores, kBias, bias_format, layout.strip_n_tiles));
        }
    }
    for (uint32_t id : {kInputGranted, kInputArrived, kWeightGranted, kWeightArrived, kDrainPosted, kDrainDone}) {
        descriptor.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
            .id = id,
            .core_type = tt::CoreType::WORKER,
            .core_ranges = all_cores,
            .initial_value = 0,
        });
    }

    // One region's blocking: the config grid's, or the strip's (one M and one N block a core).
    struct Region {
        CoreRangeSet cores;
        uint32_t m_block;
        uint32_t m_blocks;
        uint32_t n_block;
        uint32_t subblock_h;
        uint32_t subblock_w;
        bool split_drain;
        bool direct_pack;
        bool shard_drain;
        bool read_ahead;
        bool row_table;
        bool drain_plan;
    };

    // RISCV_0 on NOC0 carries the activations down each column from its top core, RISCV_1 on NOC1 the weights west
    // along each row from its right core. A NOC1 multicast runs from the far corner of its rectangle to the near one.
    const auto in0_kernel_for = [&](const Region& region) {
        KernelDescriptor kernel;
        kernel.kernel_source = kKernelDirectory + "dataflow/in0_stream.cpp";
        kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
        kernel.core_ranges = region.cores;
        kernel.named_compile_time_args = {
            {"K_tiles", layout.k_tiles},
            {"M_block_tiles", region.m_block},
            {"K_block_tiles", k_block},
            {"M_blocks_per_core", region.m_blocks},
            {"N_blocks_per_core", layout.n_blocks_per_core},
            {"K_num_blocks", layout.k_blocks},
            {"in0_cb", kInputRing},
            {"shard_width", layout.input_shard_width},
            {"split_drain", region.split_drain ? 1u : 0u},
            {"row_table", region.row_table ? 1u : 0u},
            {"drain_plan", region.drain_plan ? 1u : 0u},
            {"N_tiles", layout.n_tiles},
            {"N_block_tiles", region.n_block},
            {"out_cb", kOutput},
            {"in0_granted", kInputGranted},
            {"in0_arrived", kInputArrived},
            {"drain_posted", kDrainPosted},
            {"drain_done", kDrainDone},
        };
        TensorAccessorArgs(*input.buffer()).append_to(kernel.compile_time_args);
        TensorAccessorArgs(*output.buffer()).append_to(kernel.compile_time_args);
        kernel.config = DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0};
        return kernel;
    };
    const auto in1_kernel_for = [&](const Region& region) {
        KernelDescriptor kernel;
        kernel.kernel_source = kKernelDirectory + "dataflow/in1_stream.cpp";
        kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
        kernel.core_ranges = region.cores;
        kernel.named_compile_time_args = {
            {"N_tiles", layout.n_tiles},
            {"M_block_tiles", region.m_block},
            {"K_block_tiles", k_block},
            {"N_block_tiles", region.n_block},
            {"M_blocks_per_core", region.m_blocks},
            {"N_blocks_per_core", layout.n_blocks_per_core},
            {"K_num_blocks", layout.k_blocks},
            {"has_bias", has_bias ? 1u : 0u},
            {"resident", attributes.resident_weights ? 1u : 0u},
            {"split_drain", region.split_drain ? 1u : 0u},
            {"direct_pack", region.direct_pack ? 1u : 0u},
            {"shard_drain", region.shard_drain ? 1u : 0u},
            {"read_ahead", region.read_ahead ? 1u : 0u},
            {"row_table", region.row_table ? 1u : 0u},
            {"drain_plan", region.drain_plan ? 1u : 0u},
            {"in1_cb", kWeightRing},
            {"out_cb", kOutput},
            {"bias_cb", kBias},
            {"in1_granted", kWeightGranted},
            {"in1_arrived", kWeightArrived},
            {"drain_posted", kDrainPosted},
            {"drain_done", kDrainDone},
        };
        TensorAccessorArgs(*weight.buffer()).append_to(kernel.compile_time_args);
        // Without a bias the accessor is never built; the weight's arguments hold its place.
        TensorAccessorArgs(*(has_bias ? tensor_args.bias->buffer() : weight.buffer()))
            .append_to(kernel.compile_time_args);
        TensorAccessorArgs(*output.buffer()).append_to(kernel.compile_time_args);
        kernel.config = DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::NOC_1};
        return kernel;
    };
    std::map<std::string, std::string> compute_defines;
    if (has_bias) {
        compute_defines["FUSE_BIAS"] = "1";
    }
    ttnn::operations::compute_throttle_utils::throttle_mm_perf(
        device->arch(),
        static_cast<int>(columns * (rows + layout.strip_rows)),
        compute_defines,
        ttnn::get_throttle_level(attributes.compute_kernel_config));
    const auto compute_kernel_for = [&](const Region& region) {
        KernelDescriptor kernel;
        kernel.kernel_source = kComputeKernel;
        kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
        kernel.core_ranges = region.cores;
        kernel.compile_time_args = {
            layout.k_blocks,
            region.m_block,
            k_block,
            region.n_block,
            region.m_blocks,
            layout.n_blocks_per_core,
            region.subblock_h,
            region.subblock_w};
        kernel.defines.assign(compute_defines.begin(), compute_defines.end());
        // As minimal_matmul's factory builds it: a double-buffered destination whatever dst_full_sync_en says, and
        // fp32 partials unpacked to the source registers (the default mode). The legacy descriptor compiles at O3.
        kernel.config = ComputeConfigDescriptor{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = false,
            .math_approx_mode = math_approx_mode,
        };
        return kernel;
    };

    const Region main_region{
        .cores = main_cores,
        .m_block = m_block,
        .m_blocks = layout.m_blocks_per_core,
        .n_block = n_block,
        .subblock_h = config.subblock_h,
        .subblock_w = config.subblock_w,
        .split_drain = split_drain,
        .direct_pack = layout.direct_pack,
        .shard_drain = false,
        .read_ahead = attributes.read_ahead,
        .row_table = strided,
        .drain_plan = drain_plan};
    auto in0_kernel = in0_kernel_for(main_region);
    auto in1_kernel = in1_kernel_for(main_region);
    auto compute = compute_kernel_for(main_region);

    const auto noc_coordinates = [&](uint32_t x, uint32_t y) { return device->worker_core_from_logical_core({x, y}); };
    // The column's activations: multicast south from its top core on NOC0 (start <= end), or read by the core
    // itself when it has no receivers.
    const auto in0_args = [&](uint32_t x,
                              uint32_t y,
                              bool is_sender,
                              uint32_t m_start,
                              uint32_t m_end,
                              uint32_t receivers,
                              uint32_t n_start) {
        const auto sender = noc_coordinates(x, is_sender ? y : 0);
        const auto first = noc_coordinates(x, receivers > 0 ? 1 : y);
        const auto last = noc_coordinates(x, receivers > 0 ? rows - 1 : y);
        KernelDescriptor::RTArgList args;
        args.push_back(input.buffer());
        args.push_back(is_sender ? 1u : 0u);
        args.push_back(m_start);
        args.push_back(m_end);
        args.push_back(static_cast<uint32_t>(sender.x));
        args.push_back(static_cast<uint32_t>(sender.y));
        args.push_back(static_cast<uint32_t>(first.x));
        args.push_back(static_cast<uint32_t>(first.y));
        args.push_back(static_cast<uint32_t>(last.x));
        args.push_back(static_cast<uint32_t>(last.y));
        args.push_back(receivers);
        if (layout.input_shard_width > 0) {
            for (uint32_t shard = 0; shard < layout.k_tiles / layout.input_shard_width; ++shard) {
                args.push_back(static_cast<uint32_t>(noc_coordinates(x, shard).y));
            }
        }
        args.push_back(output.buffer());
        args.push_back(n_start);
        return args;
    };
    // The row's weights: multicast west from its right core on NOC1, start at the far corner.
    const auto in1_args =
        [&](uint32_t x, uint32_t y, uint32_t m_start, uint32_t m_end, uint32_t n_start, uint32_t defer_write_k_block) {
            const auto sender = noc_coordinates(columns - 1, y);
            const auto far = noc_coordinates(columns > 1 ? columns - 2 : 0, y);
            const auto near = noc_coordinates(0, y);
            KernelDescriptor::RTArgList args;
            args.push_back(weight.buffer());
            args.push_back(has_bias ? tensor_args.bias->buffer() : weight.buffer());
            args.push_back(output.buffer());
            for (uint32_t value :
                 {x == columns - 1 ? 1u : 0u,
                  m_start,
                  m_end,
                  n_start,
                  defer_write_k_block,
                  static_cast<uint32_t>(sender.x),
                  static_cast<uint32_t>(sender.y),
                  static_cast<uint32_t>(far.x),
                  static_cast<uint32_t>(far.y),
                  static_cast<uint32_t>(near.x),
                  static_cast<uint32_t>(near.y),
                  columns - 1}) {
                args.push_back(value);
            }
            return args;
        };

    // The kernels' trailing arguments: the row table, then the RISC's pieces.
    const auto append_plan = [&](KernelDescriptor::RTArgList& args, const Source& source, uint32_t noc) {
        if (strided) {
            args.push_back(static_cast<uint32_t>(source.table.size()));
            for (uint32_t row : source.table) {
                args.push_back(row);
            }
        }
        if (drain_plan) {
            args.push_back(static_cast<uint32_t>(source.pieces[noc].size()));
            for (uint32_t word : source.pieces[noc]) {
                args.push_back(word);
            }
        }
    };

    // minimal_matmul's rule: block m - 1 goes out at a K block that grows with the core's row, never K block 0.
    const uint32_t k_blocks_per_row = tt::div_up(layout.k_blocks, rows);
    for (uint32_t x = 0; x < columns; ++x) {
        for (uint32_t y = 0; y < rows; ++y) {
            const CoreCoord core(x, y);
            const uint32_t m_start = x * layout.m_tiles_per_core;
            const uint32_t m_end = std::min(m_start + layout.main_m_tiles, layout.m_tiles);
            const auto& source = sources[source_of_core.at(core)];
            const uint32_t n_start = source.n_start;
            auto in0 = in0_args(x, y, y == 0, m_start, m_end, rows - 1, n_start);
            append_plan(in0, source, 0);
            in0_kernel.emplace_runtime_args(core, in0);
            uint32_t defer_write_k_block = std::min(y * k_blocks_per_row, layout.k_blocks - 1);
            if (layout.k_blocks > 1) {
                defer_write_k_block = std::max(defer_write_k_block, 1u);
            }
            auto in1 = in1_args(x, y, m_start, m_end, n_start, defer_write_k_block);
            append_plan(in1, source, 1);
            in1_kernel.emplace_runtime_args(core, in1);
            // As minimal_matmul passes them: the core's M share unclamped, its padding rows computed.
            compute.emplace_runtime_args(
                core, {m_start, m_start + layout.main_m_tiles, n_start, n_start + layout.n_tiles_per_core});
        }
    }
    descriptor.kernels.push_back(std::move(in0_kernel));
    descriptor.kernels.push_back(std::move(in1_kernel));
    descriptor.kernels.push_back(std::move(compute));

    if (has_strip) {
        // Each strip core reads its own rows; each strip row's weights come from its right core, as a main row's.
        const Region strip_region{
            .cores = strip_cores,
            .m_block = layout.strip_m_tiles,
            .m_blocks = 1,
            .n_block = layout.strip_n_tiles,
            .subblock_h = layout.strip_subblock_h,
            .subblock_w = layout.strip_subblock_w,
            .split_drain = false,
            .direct_pack = false,
            .shard_drain = layout.direct_pack,
            .read_ahead = attributes.strip_read_ahead,
            .row_table = strided,
            .drain_plan = drain_plan};
        auto strip_in0 = in0_kernel_for(strip_region);
        auto strip_in1 = in1_kernel_for(strip_region);
        auto strip_compute = compute_kernel_for(strip_region);
        for (uint32_t x = 0; x < columns; ++x) {
            for (uint32_t j = 0; j < layout.strip_rows; ++j) {
                const CoreCoord core(x, rows + j);
                const uint32_t m_start = x * layout.m_tiles_per_core + layout.main_m_tiles;
                const uint32_t m_end = std::min(m_start + layout.strip_m_tiles, layout.m_tiles);
                const auto& source = sources[source_of_core.at(core)];
                const uint32_t n_start = source.n_start;
                auto in0 = in0_args(x, rows + j, true, m_start, m_end, 0, n_start);
                append_plan(in0, source, 0);
                strip_in0.emplace_runtime_args(core, in0);
                auto args = in1_args(x, rows + j, m_start, m_end, n_start, 0);
                if (layout.direct_pack) {
                    // The rows go into the shards of grid rows [j * strip_lanes, (j + 1) * strip_lanes) of column x.
                    args.push_back(layout.main_m_tiles);
                    args.push_back(layout.n_tiles_per_core);
                    args.push_back(layout.strip_lanes);
                    for (uint32_t lane = 0; lane < layout.strip_lanes; ++lane) {
                        const auto shard = noc_coordinates(x, j * layout.strip_lanes + lane);
                        args.push_back(static_cast<uint32_t>(shard.x));
                        args.push_back(static_cast<uint32_t>(shard.y));
                    }
                }
                append_plan(args, source, 1);
                strip_in1.emplace_runtime_args(core, args);
                strip_compute.emplace_runtime_args(
                    core, {m_start, m_start + layout.strip_m_tiles, n_start, n_start + layout.strip_n_tiles});
            }
        }
        descriptor.kernels.push_back(std::move(strip_in0));
        descriptor.kernels.push_back(std::move(strip_in1));
        descriptor.kernels.push_back(std::move(strip_compute));
    }
    return descriptor;
}

}  // namespace ttnn::experimental::nomic_embed_text_v2_moe

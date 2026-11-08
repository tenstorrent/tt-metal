// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Parts of the GGUF matmul programs that both factories use (gguf_matmul_program_factory.cpp,
// gguf_matmul_prefill_program_factory.cpp): circular buffers, the weight reader (kernels/gguf/gguf_matmul_reader.cpp)
// and the reader / writer pairs.

#pragma once

#include <array>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "ttnn/operations/matmul/device/gguf/gguf_matmul_program_factory.hpp"

namespace ttnn::prim::gguf_program {

using namespace tt::tt_metal;

inline constexpr auto kKernels = "ttnn/cpp/ttnn/operations/matmul/device/kernels/gguf/";
inline constexpr uint32_t kGroupsPerRead = 4;  // one-page groups: records per unit of reads
inline constexpr uint32_t kPageUnits = 4;      // units of pages c_0 holds
inline constexpr uint32_t kRecordSlots = 3;    // multi-page groups: records staged in L1, two in flight from DRAM

inline void add_cb(
    Program& program,
    const CoreRangeSet& cores,
    tt::CBIndex index,
    tt::DataFormat format,
    uint32_t page_bytes,
    uint32_t pages,
    std::optional<tt::tt_metal::Tile> tile = std::nullopt) {
    CircularBufferConfig config =
        CircularBufferConfig(page_bytes * pages, {{index, format}}).set_page_size(index, page_bytes);
    if (tile) {
        config.set_tile_dims(index, *tile);
    }
    CreateCircularBuffer(program, cores, config);
}

// Records per unit of the weight reader's reads: a multi-page group's pages, else kGroupsPerRead one-page groups.
inline uint32_t reader_unit(const GgufTraits& f) {
    const uint32_t pages_per_group = f.group_tiles / f.page_tiles;
    return pages_per_group > 1 ? pages_per_group : kGroupsPerRead;
}

// L1 bytes of the weight reader's circular buffers (add_weight_reader_cbs).
inline uint32_t weight_reader_l1_bytes(const GgufTraits& f) {
    return kPageUnits * reader_unit(f) * f.l1_page_bytes() +
           (f.group_tiles / f.page_tiles > 1 ? kRecordSlots * f.page_bytes() : 0);
}

// The weight reader's circular buffers on `cores`: c_0, the decoder's L1 pages (UInt32 tiles of the page's size), and
// for multi-page groups c_2, the staging ring.
inline void add_weight_reader_cbs(Program& program, const CoreRangeSet& cores, const GgufTraits& f) {
    const uint32_t l1_page = f.l1_page_bytes();
    const tt::tt_metal::Tile page_tile(
        l1_page == 1024   ? std::array<uint32_t, 2>{16, 16}
        : l1_page == 2048 ? std::array<uint32_t, 2>{16, 32}
                          : std::array<uint32_t, 2>{32, 32});
    add_cb(program, cores, tt::CBIndex::c_0, tt::DataFormat::UInt32, l1_page, kPageUnits * reader_unit(f), page_tile);
    if (f.group_tiles / f.page_tiles > 1) {
        add_cb(program, cores, tt::CBIndex::c_2, tt::DataFormat::UInt32, kRecordSlots * f.page_bytes(), 1);
    }
}

// The weight reader's compile args: format id, c_0 pages, one-page groups per unit, staging slots, the lookup request,
// answer and page address semaphores, reads in flight, the weight's TensorAccessorArgs.
inline std::vector<uint32_t> weight_reader_args(
    const Tensor& weight, uint32_t request_sem, uint32_t answer_sem, uint32_t address_sem) {
    const GgufTraits& f = gguf_traits(weight.dtype());
    // Two reads in flight for Q4_0, Q4_1, MXFP4 and NVFP4, one for the others, which a second one slows down
    // (measured, decode).
    const DataType dtype = weight.dtype();
    const bool two_in_flight = dtype == DataType::GGUF_Q4_0 || dtype == DataType::GGUF_Q4_1 ||
                               dtype == DataType::GGUF_MXFP4 || dtype == DataType::GGUF_NVFP4;
    std::vector<uint32_t> args = {
        f.format_id,
        kPageUnits * reader_unit(f),
        kGroupsPerRead,
        kRecordSlots,
        request_sem,
        answer_sem,
        address_sem,
        two_in_flight ? 2u : 1u};
    TensorAccessorArgs(*weight.buffer()).append_to(args);
    return args;
}

// Codebook formats: each page's index words in GGUF_LOOKUP_PARTS equal parts, looked up by the reader, the writer, and
// the compute kernel's math and unpack threads; the pack thread takes the rest. An expanded codebook layout holds the
// level codes: no lookups.
inline std::map<std::string, std::string> lookup_defines(const Tensor& weight) {
    const GgufTraits& f = gguf_traits(weight.dtype());
    const bool expanded = is_gguf_expanded(weight.dtype()) && f.code_offset > 0;
    struct LookupSplit {
        uint32_t parts, reader, writer, math, unpack;
    };
    const LookupSplit split = expanded            ? LookupSplit{4, 0, 0, 0, 0}
                              : f.code_offset > 0 ? LookupSplit{4, 1, 1, 1, 1}
                                                  : LookupSplit{2, 1, 1, 0, 0};
    return {
        {"GGUF_EXPANDED", expanded ? "1" : "0"},
        {"GGUF_LOOKUP_PARTS", std::to_string(split.parts)},
        {"GGUF_READER_PARTS", std::to_string(split.reader)},
        {"GGUF_WRITER_PARTS", std::to_string(split.writer)},
        {"GGUF_MATH_PARTS", std::to_string(split.math)},
        {"GGUF_UNPACK_PARTS", std::to_string(split.unpack)}};
}

// The reader and the writer on the cores of `ranges`, all in columns of one parity: the reader uses one NoC and the
// writer the other, alternating by column so that both NoCs carry DRAM reads.
inline void add_movers(
    Program& program,
    const CoreRangeSet& ranges,
    uint32_t parity,
    const std::string& reader,
    const std::string& writer,
    const std::vector<uint32_t>& reader_ct,
    const std::vector<uint32_t>& writer_ct,
    const std::map<std::string, std::string>& reader_defines,
    const std::map<std::string, std::string>& writer_defines,
    GgufMatmulSharedVariables& shared) {
    shared.reader[parity] = CreateKernel(
        program,
        std::string(kKernels) + reader,
        ranges,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = parity ? NOC::NOC_1 : NOC::NOC_0,
            .compile_args = reader_ct,
            .defines = reader_defines});
    shared.writer[parity] = CreateKernel(
        program,
        std::string(kKernels) + writer,
        ranges,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = parity ? NOC::NOC_0 : NOC::NOC_1,
            .compile_args = writer_ct,
            .defines = writer_defines});
}

// override_runtime_arguments of both programs: the weight address is the reader's runtime arg 0, the input and output
// addresses the writer's args writer_input and writer_input + 1.
template <typename CachedProgram>
void override_addresses(
    CachedProgram& cached_program, const GgufMatmulInputs& inputs, const Tensor& output, uint32_t writer_input) {
    const GgufMatmulSharedVariables& shared = cached_program.shared_variables;
    for (const CoreCoord& core : shared.cores) {
        const uint32_t parity = core.x % 2;
        GetRuntimeArgs(cached_program.program, shared.reader[parity], core)[0] = inputs.weight.buffer()->address();
        auto& writer_args = GetRuntimeArgs(cached_program.program, shared.writer[parity], core);
        writer_args[writer_input] = inputs.input.buffer()->address();
        writer_args[writer_input + 1] = output.buffer()->address();
    }
}

}  // namespace ttnn::prim::gguf_program

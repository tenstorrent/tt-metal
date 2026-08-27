// SPDX-License-Identifier: Apache-2.0
//
// GATE A -- the structural question, with no unified library involved.
//
// ONE source, compiled for all five baby-RISCV threads by THREE KernelSpecs in one
// Metal 2.0 ProgramSpec, moving a block through two dataflow buffers whose endpoints
// straddle data movement and compute:
//
//     reader (NCRISC)  DRAM  -> in   DFB      producer of `in`
//     compute (TRISCs) in    -> out  DFB      consumer of `in`, producer of `out`
//     writer (BRISC)   out   -> DRAM         consumer of `out`
//
// The compute pass SQUARES each tile, so a wrong result is distinguishable from a
// buffer that was merely copied through.
//
// Buffer slots arrive as NAMED COMPILE-TIME ARGS, deliberately not as `dfb::` binding
// tokens: a token is emitted only into the kernels that bind it, so `dfb::out` does
// not exist on the reader's build -- and a unified kernel declares every Storage on
// every projection. See unified_metal2_spec.md.

#include <cstdint>

#if defined(COMPILE_FOR_BRISC)
#define GATE_WRITER 1
#elif defined(COMPILE_FOR_NCRISC)
#define GATE_READER 1
#elif defined(UCK_CHLKC_UNPACK) || defined(UCK_CHLKC_MATH) || defined(UCK_CHLKC_PACK)
#define GATE_COMPUTE 1
#else
#error "gate_a.cpp: no thread-identity define present"
#endif

#if defined(GATE_COMPUTE)
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#else
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#endif

#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t cb_in = get_arg(args::cb_in);
    constexpr uint32_t cb_out = get_arg(args::cb_out);
    constexpr uint32_t num_tiles = get_arg(args::num_tiles);

    // BOTH buffers are declared on EVERY projection, which is the property the unified
    // model depends on and the one this gate is really testing. The reader never binds
    // `out` and the writer never binds `in`; constructing the handle anyway must be
    // harmless, because only the guarded regions below ever touch it.
    DataflowBuffer in(static_cast<uint16_t>(cb_in));
    DataflowBuffer out(static_cast<uint16_t>(cb_out));

#if defined(GATE_READER)
    {
        const uint32_t src_addr = get_arg(args::src_addr);
        const uint32_t entry_bytes = in.get_entry_size();
        Noc noc;
        AllocatorBank<AllocatorBankType::DRAM> dram;
        for (uint32_t t = 0; t < num_tiles; ++t) {
            in.reserve_back(1);
            noc.async_read(dram, in, entry_bytes, {.bank_id = 0, .addr = src_addr + t * entry_bytes}, {});
            noc.async_read_barrier();
            in.push_back(1);
        }
    }
#elif defined(GATE_WRITER)
    {
        const uint32_t dst_addr = get_arg(args::dst_addr);
        const uint32_t entry_bytes = out.get_entry_size();
        Noc noc;
        AllocatorBank<AllocatorBankType::DRAM> dram;
        for (uint32_t t = 0; t < num_tiles; ++t) {
            out.wait_front(1);
            noc.async_write(out, dram, entry_bytes, {}, {.bank_id = 0, .addr = dst_addr + t * entry_bytes});
            noc.async_write_barrier();
            out.pop_front(1);
        }
    }
#else
    {
        compute_kernel_hw_startup(cb_in, cb_in, cb_out);
        mul_tiles_init(cb_in, cb_in);
        for (uint32_t t = 0; t < num_tiles; ++t) {
            in.wait_front(1);
            out.reserve_back(1);
            tile_regs_acquire();
            mul_tiles(cb_in, cb_in, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out);
            tile_regs_release();
            out.push_back(1);
            in.pop_front(1);
        }
    }
#endif
}

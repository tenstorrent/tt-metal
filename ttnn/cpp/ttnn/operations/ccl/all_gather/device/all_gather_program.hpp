// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "all_gather_plan.hpp"

// Program pieces both all_gather factories share: the workload prologue (plan and semaphores), the slot CB, the
// kernels and the runtime argument tables they read. Every value comes from the plan or the tensors, so a cached
// program depends only on hashed params and Buffer* bindings.
namespace ttnn::operations::ccl::all_gather_program {

inline constexpr const char* kUnicastWriterKernel =
    "ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/unicast_writer.cpp";
inline constexpr const char* kMulticastWriterKernel =
    "ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/multicast_writer.cpp";

// Kernel order in each program: the reader, then the writer.
inline constexpr uint32_t kReaderIdx = 0;
inline constexpr uint32_t kWriterIdx = 1;

// The reader and writer NOCs (Tuning::writer_noc; the two RISCs of a core use different NOCs).
struct KernelNocs {
    tt::tt_metal::NOC reader, writer;
};
KernelNocs kernel_nocs(const all_gather_plan::AgPlan& plan);

// The plan of one call, re-expanded from the hashed choice, and a workload holding its two GlobalSemaphores (init,
// then data_valid / done) on the plan's worker cores.
struct WorkloadStart {
    all_gather_plan::AgPlan plan;
    tt::tt_metal::WorkloadDescriptor workload;
    uint32_t init_sem = 0, data_sem = 0;
};
WorkloadStart start_workload(const AllGatherParams& params, const Tensor& input, const Tensor& output);

// Worker cores of the plan: unicast group workers (not muxes), or the multicast cores.
CoreRangeSet worker_cores(const all_gather_plan::AgPlan& plan);

// The slot CB: kCbSlots pages of plan.slot_bytes on `cores`.
tt::tt_metal::CBDescriptor slot_cb(const all_gather_plan::AgPlan& plan, const CoreRangeSet& cores);

// The reader kernel (ag_reader.cpp) on `cores`. Named CT args: A, CAP, K, SLOT_BYTES, CB_ID, P_A, UI, UO, S, N, E,
// DST_CROSS, SRC_CROSS, RDELAY; positional CT args: TensorAccessorArgs(input), TensorAccessorArgs(output).
tt::tt_metal::KernelDescriptor reader_kernel(
    const all_gather_plan::AgPlan& plan,
    const AgForce& force,
    const Tensor& input,
    const Tensor& output,
    const CoreRangeSet& cores);

// A writer kernel on `cores`. Named CT args: A, CAP, K, SLOT_BYTES, CB_ID, WDELAY, then `extra`.
tt::tt_metal::KernelDescriptor writer_kernel(
    const all_gather_plan::AgPlan& plan,
    const AgForce& force,
    const char* source,
    const CoreRangeSet& cores,
    const tt::tt_metal::KernelDescriptor::NamedCompileTimeArgs& extra);

// Reader runtime args (ag_reader.cpp reads them in this order):
//   input (Buffer*), output (Buffer*), e0, ne, iters, stripe0, stripe_step, last_skip, last_take, iter_units,
//   end_wait, local_copy, data_sem. The init handshake is the writer's.
void add_reader_rt_args(
    tt::tt_metal::KernelDescriptor& reader,
    const CoreCoord& core,
    const Tensor& input,
    const Tensor& output,
    const all_gather_plan::WorkerSched& ws,
    uint32_t data_sem);

}  // namespace ttnn::operations::ccl::all_gather_program

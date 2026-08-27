// SPDX-License-Identifier: Apache-2.0
//
// GATE B -- the unified library, unmodified, under a Metal 2.0 ProgramSpec.
//
// Structurally this is unified_kernels/unary.cpp. The only differences are the two
// that Metal 2.0 forces:
//
//   * compile-time args are NAMED -- KernelSpec::CompileTimeArgs is a
//     Table<string, uint32_t>, so there is no positional list to index.
//   * the tensor accessors come from BINDING TOKENS rather than a positional
//     TensorAccessorArgs<N> block, which is what retires hazard D18.
//
// The circular-buffer slots still arrive as compile-time VALUES, not as `dfb::`
// tokens: a DFB token exists only in the kernels that bind that buffer, and a
// unified kernel declares every Storage on every projection. See gate_a_tokens.cpp,
// which fails to compile for exactly that reason.

#include <tt/unified/core>
#include "experimental/kernel_args.h"

namespace u = tt::unified;

void kernel_main() {
    constexpr uint32_t kCbIn = get_arg(args::cb_in);
    constexpr uint32_t kCbOut = get_arg(args::cb_out);
    constexpr uint32_t num_blocks = get_arg(args::num_blocks);
    constexpr uint32_t tiles_per_block = get_arg(args::tiles_per_block);

    u::compute_init(kCbIn, kCbOut);

    using Block1D = u::Shape<1, tiles_per_block>;
    u::Storage<Block1D> in_storage(kCbIn);
    u::Storage<Block1D> out_storage(kCbOut);

    const auto in = TensorAccessor(tensor::in);
    const auto out = TensorAccessor(tensor::out);

    for (uint32_t b = 0; b < num_blocks; ++b) {
        u::ComputeBlock a = u::noc_load<0>(in_storage, in, b).wait();
        u::Block result = out_storage.store(u::recip(a));
        u::noc_store<1>(std::move(result), out, b);
    }
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The matmul-core kernels' macros, built from host defines that carry no address.
#pragma once

// GROW(m, lr): the global row tile of chunk m's chunk-local row tile lr. With row chunks over TP shards (NCH > 1 and
// shards > 1) chunk m is block m of every owner, so the owners' rows stay contiguous: the host defines
// SE_GROW_OWN_CR = CR / shards and SE_GROW_OWN_RT = rows / 32 / shards; else chunk-major (SE_CR = CR, row tiles a
// chunk).
#ifdef SE_GROW_OWN_CR
#define GROW(m, lr) (((lr) / SE_GROW_OWN_CR) * SE_GROW_OWN_RT + (m) * SE_GROW_OWN_CR + (lr) % SE_GROW_OWN_CR)
#else
#define GROW(m, lr) ((m) * SE_CR + (lr))
#endif

// SE_EW: the router-phase words (a persistent L1 tensor a core). The BRISC / NCRISC read its address from the common
// runtime args right past the shared span: their common args are [span start, span length n, n words] ([0, 0] when
// nothing is shared), then this address at index 2 + n. Only expanded by the dataflow kernels (the compute never names
// it).
#define SE_EW (get_common_arg_val<uint32_t>(2 + get_common_arg_val<uint32_t>(1)))

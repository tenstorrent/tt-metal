// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#define ACTIVATION_INIT_RELU relu_tile_init
#define ACTIVATION_APPLY_RELU(i) relu_tile(i)

#define ACTIVATION_INIT_SQUARE square_tile_init
#define ACTIVATION_APPLY_SQUARE(i) square_tile(i)

#define ACTIVATION_INIT_GTZ gtz_tile_init
#define ACTIVATION_APPLY_GTZ(i) gtz_tile(i)

#define ACTIVATION_INIT_LTZ ltz_tile_init
#define ACTIVATION_APPLY_LTZ(i) ltz_tile(i)

#define ACTIVATION_INIT_GEZ gez_tile_init
#define ACTIVATION_APPLY_GEZ(i) gez_tile(i)

#define ACTIVATION_INIT_LEZ lez_tile_init
#define ACTIVATION_APPLY_LEZ(i) lez_tile(i)

#define ACTIVATION_INIT_EQZ eqz_tile_init
#define ACTIVATION_APPLY_EQZ(i) eqz_tile(i)

#define ACTIVATION_INIT_NEZ nez_tile_init
#define ACTIVATION_APPLY_NEZ(i) nez_tile(i)

#define ACTIVATION_INIT_LOG log_tile_init
#define ACTIVATION_APPLY_LOG(i) log_tile(i)

#define ACTIVATION_INIT_LOG2 log_with_base_tile_init
#define ACTIVATION_APPLY_LOG2(i) log_with_base_tile(i, 0x3dc5u)

#define DO_QUOTE(x) #x
#define QUOTE(x) DO_QUOTE(x)

#if defined(PREPROCESS_A_INIT)
#define PREPROCESS_A 1
#else
#define PREPROCESS_A 0
#endif

#if defined(PREPROCESS_B_INIT)
#define PREPROCESS_B 1
#else
#define PREPROCESS_B 0
#endif

#ifdef PREPROCESS_A_INCLUDE
#include QUOTE(PREPROCESS_A_INCLUDE)
#endif

#ifdef PREPROCESS_B_INCLUDE
#include QUOTE(PREPROCESS_B_INCLUDE)
#endif

#define ACTIVATION_INIT(elem) ACTIVATION_INIT_##elem()
#define ACTIVATION_APPLY(elem, i) ACTIVATION_APPLY_##elem(i)

#define PROCESS_ACTIVATION(elem, i) \
    ACTIVATION_INIT(elem);          \
    ACTIVATION_APPLY(elem, i)

// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

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

#if defined(POSTPROCESS_INIT)
#define POSTPROCESS 1
#else
#define POSTPROCESS 0
#endif

#ifdef PREPROCESS_A_INCLUDE
#include QUOTE(PREPROCESS_A_INCLUDE)
#endif

#ifdef PREPROCESS_B_INCLUDE
#include QUOTE(PREPROCESS_B_INCLUDE)
#endif

#ifdef POSTPROCESS_INCLUDE
#include QUOTE(POSTPROCESS_INCLUDE)
#endif

// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include "nlp_concat_heads_decode_device_operation.hpp"
#include "multi_core_nlp_concat_heads_decode_program_factory.hpp"
#include "multi_core_nlp_concat_heads_decode_subcoregrids_program_factory.hpp"
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::experimental::transformer {

using namespace tt;
using namespace tt::constants;

}  // namespace ttnn::operations::experimental::transformer

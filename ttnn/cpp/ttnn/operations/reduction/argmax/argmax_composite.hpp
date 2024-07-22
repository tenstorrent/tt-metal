// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


namespace ttnn {

namespace operations {

namespace unary {

Tensor _argmax(const Tensor& input_t, int64_t _dim, bool all, const std::optional<MemoryConfig>& output_mem_config);
}
}
}

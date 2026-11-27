// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt-metalium/data_formats/data_format.hpp>

namespace ttnn::prim::qsr {

// typecast_tile / typecast_tile_init take the input and output formats as template arguments and cast them to the
// Quasar device DataFormat. The codes of MxInt8, MxInt4 and MxInt2 differ from the host enumerators (host 12, 16, 17;
// Quasar 2, 3, 11, tensix_types.h). Passed as host values they name other formats (host MxInt4 = 16 is Quasar Fp8P,
// host MxInt2 = 17 is Quasar UInt8) or none (12), so the kernel would not treat the tensor as MX. The other MX
// formats (MxFp8R/P, MxFp6R/P, MxFp4) have the same code on both sides. Every Quasar-fork factory is Quasar-only, so
// no arch check is needed.
inline std::uint32_t typecast_kernel_data_format(tt::DataFormat format) {
    switch (format) {
        case tt::DataFormat::MxInt8: return 2;
        case tt::DataFormat::MxInt4: return 3;
        case tt::DataFormat::MxInt2: return 11;
        default: return static_cast<std::uint32_t>(format);
    }
}

}  // namespace ttnn::prim::qsr

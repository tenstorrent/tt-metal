# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


fill_cache_for_user_ = ttnn.register_operation()(ttnn._ttnn.operations.kv_cache.fill_cache_for_user_)
update_cache_for_token_ = ttnn.register_operation()(ttnn._ttnn.operations.kv_cache.update_cache_for_token_)

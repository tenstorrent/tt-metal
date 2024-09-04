# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import List


# Base class for OP
class TTPyOp(ABC):
    # Generate op config variabes and tensors
    def set_op_configs(self):
        pass

    # Construct pytorch tensors for op weights and bias. Moves those tensors to device
    def set_op_weights_biases(self, weight_tensor: List, bias_tensor: List):
        pass

    # Return stats on op's L1 buffers
    def get_l1_buffer_stats(self):
        pass

    @abstractmethod
    def __call__(self):
        pass

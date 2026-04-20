"""TTNN VAE Decoder model for Tongyi-MAI/Z-Image-Turbo."""

import torch
import ttnn
from vae.model_pt import SCALING_FACTOR, SHIFT_FACTOR, VaeDecoderPT
from vae.params import load_weights

DRAM_RM = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)


class LightweightModule:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class VaeDecoderTTNN(LightweightModule):
    """TTNN VAE Decoder - direct translation of the generated _main function."""

    def __init__(self, mesh_device):
        self.device = mesh_device
        pt = VaeDecoderPT()
        self.ce_cache, self.attn_args = load_weights(pt.state_dict, self.device)
        del pt

    def forward(self, raw_latents):
        """Decode raw (pre-scaling) latents → [1, 3, 512, 512] float32 CPU tensor."""
        z = (raw_latents.float() / SCALING_FACTOR) + SHIFT_FACTOR
        latent = ttnn.from_torch(
            z.bfloat16(),
            dtype=ttnn.DataType.BFLOAT16,
            layout=ttnn.Layout.ROW_MAJOR,
            device=self.device,
            memory_config=DRAM_RM,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )
        var_0 = latent
        var_1 = self.ce_cache["main_const_eval_13"][0]
        var_2 = self.ce_cache["main_const_eval_81"]
        var_3 = var_2[0]
        var_4 = var_2[1]
        var_5 = self.ce_cache["main_const_eval_91"][0]
        var_6 = self.ce_cache["main_const_eval_94"][0]
        var_7 = self.ce_cache["main_const_eval_103"][0]
        var_8 = self.ce_cache["main_const_eval_104"][0]
        var_9 = self.ce_cache["main_const_eval_106"][0]
        var_10 = self.ce_cache["main_const_eval_119"][0]
        var_11 = self.ce_cache["main_const_eval_124"][0]
        var_12 = self.ce_cache["main_const_eval_126"][0]
        device = self.device
        ttnn_to_layout_130 = ttnn.to_layout(var_0, ttnn.Layout.TILE, None, memory_config=None)
        ttnn.deallocate(var_0, False)
        ttnn_permute_161 = ttnn.permute(
            ttnn_to_layout_130,
            [0, 2, 3, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_to_layout_130, False)
        ttnn_reshape_173 = ttnn.reshape(
            ttnn_permute_161,
            [1, 1, 4096, 16],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_161, False)
        ttnn_conv2d_0 = ttnn.conv2d(
            input_tensor=ttnn_reshape_173,
            weight_tensor=self.ce_cache["main_const_eval_0"][0],
            device=device,
            in_channels=16,
            out_channels=512,
            batch_size=1,
            input_height=64,
            input_width=64,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=[1, 1],
            groups=1,
            bias_tensor=self.ce_cache["main_const_eval_55"][0],
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=ttnn.DataType.BFLOAT16,
                deallocate_activation=True,
                config_tensors_in_dram=True,
                act_block_h_override=0,
                enable_kernel_stride_folding=False,
            ),
            compute_config=None,
            slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_reshape_173, False)
        ttnn_typecast_63 = ttnn.typecast(
            ttnn_conv2d_0,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_reshape_174 = ttnn.reshape(
            ttnn_typecast_63,
            [1, 64, 64, 512],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_63, False)
        ttnn_permute_162 = ttnn.permute(
            ttnn_reshape_174,
            [0, 3, 1, 2],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_174, False)
        ttnn_reshape_175 = ttnn.reshape(
            ttnn_permute_162,
            [1, 32, 16, 4096],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_162, False)
        ttnn_sum_0 = ttnn.sum(
            ttnn_reshape_175,
            [2, 3],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=None,
        )
        ttnn_multiply_0 = ttnn.multiply(
            ttnn_sum_0,
            var_9,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sum_0, False)
        ttnn_subtract_0 = ttnn.subtract(
            ttnn_reshape_175,
            ttnn_multiply_0,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_0, False)
        ttnn.deallocate(ttnn_reshape_175, False)
        ttnn_multiply_1 = ttnn.multiply(
            ttnn_subtract_0,
            ttnn_subtract_0,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_sum_1 = ttnn.sum(
            ttnn_multiply_1,
            [2, 3],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_multiply_1, False)
        ttnn_multiply_2 = ttnn.multiply(
            ttnn_sum_1,
            var_9,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sum_1, False)
        ttnn_add_0 = ttnn.add(
            ttnn_multiply_2,
            var_5,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_2, False)
        ttnn_rsqrt_0 = ttnn.rsqrt(
            ttnn_add_0,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_0, False)
        ttnn_multiply_3 = ttnn.multiply(
            ttnn_subtract_0,
            ttnn_rsqrt_0,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_rsqrt_0, False)
        ttnn.deallocate(ttnn_subtract_0, False)
        ttnn_multiply_4 = ttnn.multiply(
            ttnn_multiply_3,
            self.ce_cache["main_const_eval_75"][0],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_3, False)
        ttnn_add_1 = ttnn.add(
            ttnn_multiply_4,
            self.ce_cache["main_const_eval_120"][0],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_4, False)
        ttnn_silu_0 = ttnn.silu(
            ttnn_add_1,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_1, False)
        ttnn_typecast_64 = ttnn.typecast(
            ttnn_silu_0,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_silu_0, False)
        ttnn_reshape_176 = ttnn.reshape(
            ttnn_typecast_64,
            [1, 512, 64, 64],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_64, False)
        ttnn_permute_163 = ttnn.permute(
            ttnn_reshape_176,
            [0, 2, 3, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_176, False)
        ttnn_reshape_177 = ttnn.reshape(
            ttnn_permute_163,
            [1, 1, 4096, 512],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_163, False)
        ttnn_conv2d_1 = ttnn.conv2d(
            input_tensor=ttnn_reshape_177,
            weight_tensor=self.ce_cache["main_const_eval_40"][0],
            device=device,
            in_channels=512,
            out_channels=512,
            batch_size=1,
            input_height=64,
            input_width=64,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=[1, 1],
            groups=1,
            bias_tensor=self.ce_cache["main_const_eval_136"][0],
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=ttnn.DataType.BFLOAT16,
                deallocate_activation=True,
                config_tensors_in_dram=True,
                act_block_h_override=0,
                enable_kernel_stride_folding=False,
            ),
            compute_config=None,
            slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_reshape_177, False)
        ttnn_typecast_65 = ttnn.typecast(
            ttnn_conv2d_1,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_conv2d_1, False)
        ttnn_reshape_178 = ttnn.reshape(
            ttnn_typecast_65,
            [1, 64, 64, 512],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_65, False)
        ttnn_permute_164 = ttnn.permute(
            ttnn_reshape_178,
            [0, 3, 1, 2],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_178, False)
        ttnn_reshape_179 = ttnn.reshape(
            ttnn_permute_164,
            [1, 32, 16, 4096],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_164, False)
        ttnn_sum_2 = ttnn.sum(
            ttnn_reshape_179,
            [2, 3],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=None,
        )
        ttnn_multiply_5 = ttnn.multiply(
            ttnn_sum_2,
            var_9,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sum_2, False)
        ttnn_subtract_1 = ttnn.subtract(
            ttnn_reshape_179,
            ttnn_multiply_5,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_5, False)
        ttnn.deallocate(ttnn_reshape_179, False)
        ttnn_multiply_6 = ttnn.multiply(
            ttnn_subtract_1,
            ttnn_subtract_1,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_sum_3 = ttnn.sum(
            ttnn_multiply_6,
            [2, 3],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_multiply_6, False)
        ttnn_multiply_7 = ttnn.multiply(
            ttnn_sum_3,
            var_9,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sum_3, False)
        ttnn_add_2 = ttnn.add(
            ttnn_multiply_7,
            var_5,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_7, False)
        ttnn_rsqrt_1 = ttnn.rsqrt(
            ttnn_add_2,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_2, False)
        ttnn_multiply_8 = ttnn.multiply(
            ttnn_subtract_1,
            ttnn_rsqrt_1,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_rsqrt_1, False)
        ttnn.deallocate(ttnn_subtract_1, False)
        ttnn_multiply_9 = ttnn.multiply(
            ttnn_multiply_8,
            self.ce_cache["main_const_eval_14"][0],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_8, False)
        ttnn_add_3 = ttnn.add(
            ttnn_multiply_9,
            self.ce_cache["main_const_eval_23"][0],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_9, False)
        ttnn_silu_1 = ttnn.silu(
            ttnn_add_3,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_3, False)
        ttnn_typecast_66 = ttnn.typecast(
            ttnn_silu_1,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_silu_1, False)
        ttnn_reshape_180 = ttnn.reshape(
            ttnn_typecast_66,
            [1, 512, 64, 64],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_66, False)
        ttnn_permute_165 = ttnn.permute(
            ttnn_reshape_180,
            [0, 2, 3, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_180, False)
        ttnn_reshape_181 = ttnn.reshape(
            ttnn_permute_165,
            [1, 1, 4096, 512],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_165, False)
        ttnn_conv2d_2 = ttnn.conv2d(
            input_tensor=ttnn_reshape_181,
            weight_tensor=self.ce_cache["main_const_eval_102"][0],
            device=device,
            in_channels=512,
            out_channels=512,
            batch_size=1,
            input_height=64,
            input_width=64,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=[1, 1],
            groups=1,
            bias_tensor=self.ce_cache["main_const_eval_80"][0],
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=ttnn.DataType.BFLOAT16,
                deallocate_activation=True,
                config_tensors_in_dram=True,
                act_block_h_override=0,
                enable_kernel_stride_folding=False,
            ),
            compute_config=None,
            slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_reshape_181, False)
        ttnn_add_4 = ttnn.add(
            ttnn_conv2d_0,
            ttnn_conv2d_2,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_conv2d_2, False)
        ttnn.deallocate(ttnn_conv2d_0, False)
        ttnn_reshape_182 = ttnn.reshape(
            ttnn_add_4,
            [1, 64, 64, 512],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_4, False)
        ttnn_permute_166 = ttnn.permute(
            ttnn_reshape_182,
            [0, 3, 1, 2],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_182, False)
        ttnn_divide_0 = ttnn.divide(
            ttnn_permute_166,
            var_3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_166, False)
        ttnn_typecast_67 = ttnn.typecast(
            ttnn_divide_0,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_reshape_183 = ttnn.reshape(
            ttnn_typecast_67,
            [1, 32, 16, 4096],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_67, False)
        ttnn_sum_4 = ttnn.sum(
            ttnn_reshape_183,
            [2, 3],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=None,
        )
        ttnn_multiply_10 = ttnn.multiply(
            ttnn_sum_4,
            var_9,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sum_4, False)
        ttnn_subtract_2 = ttnn.subtract(
            ttnn_reshape_183,
            ttnn_multiply_10,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_10, False)
        ttnn.deallocate(ttnn_reshape_183, False)
        ttnn_multiply_11 = ttnn.multiply(
            ttnn_subtract_2,
            ttnn_subtract_2,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_sum_5 = ttnn.sum(
            ttnn_multiply_11,
            [2, 3],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_multiply_11, False)
        ttnn_multiply_12 = ttnn.multiply(
            ttnn_sum_5,
            var_9,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sum_5, False)
        ttnn_add_5 = ttnn.add(
            ttnn_multiply_12,
            var_5,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_12, False)
        ttnn_rsqrt_2 = ttnn.rsqrt(
            ttnn_add_5,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_5, False)
        ttnn_multiply_13 = ttnn.multiply(
            ttnn_subtract_2,
            ttnn_rsqrt_2,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_rsqrt_2, False)
        ttnn.deallocate(ttnn_subtract_2, False)
        ttnn_multiply_14 = ttnn.multiply(
            ttnn_multiply_13,
            self.ce_cache["main_const_eval_63"][0],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_13, False)
        ttnn_add_6 = ttnn.add(
            ttnn_multiply_14,
            self.ce_cache["main_const_eval_133"][0],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_14, False)
        ttnn_typecast_68 = ttnn.typecast(
            ttnn_add_6,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_6, False)
        ttnn_reshape_184 = ttnn.reshape(
            ttnn_typecast_68,
            [1, 512, 4096],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_68, False)
        ttnn_permute_167 = ttnn.permute(
            ttnn_reshape_184,
            [0, 2, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_184, False)
        ttnn_reshape_185 = ttnn.reshape(
            ttnn_permute_167,
            [4096, 512],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_167, False)
        ttnn_linear_0 = ttnn.linear(
            ttnn_reshape_185,
            self.attn_args[134],
            bias=self.attn_args[133],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=None,
        )
        ttnn_typecast_69 = ttnn.typecast(
            ttnn_linear_0,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_linear_0, False)
        ttnn_reshape_186 = ttnn.reshape(
            ttnn_typecast_69,
            [1, 1, 4096, 512],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_69, False)
        ttnn_linear_1 = ttnn.linear(
            ttnn_reshape_185,
            self.attn_args[132],
            bias=self.attn_args[131],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=None,
        )
        ttnn_typecast_70 = ttnn.typecast(
            ttnn_linear_1,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_linear_1, False)
        ttnn_reshape_187 = ttnn.reshape(
            ttnn_typecast_70,
            [1, 1, 4096, 512],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_70, False)
        ttnn_linear_2 = ttnn.linear(
            ttnn_reshape_185,
            self.attn_args[128],
            bias=self.attn_args[127],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_reshape_185, False)
        ttnn_typecast_71 = ttnn.typecast(
            ttnn_linear_2,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_linear_2, False)
        ttnn_reshape_188 = ttnn.reshape(
            ttnn_typecast_71,
            [1, 1, 4096, 512],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_71, False)
        ttnn_typecast_72 = ttnn.typecast(
            ttnn_reshape_186,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_reshape_186, False)
        ttnn_typecast_73 = ttnn.typecast(
            ttnn_reshape_187,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_reshape_187, False)
        ttnn_typecast_74 = ttnn.typecast(
            ttnn_reshape_188,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_reshape_188, False)
        ttnn_transformer_scaled_dot_product_attention_0 = ttnn.transformer.scaled_dot_product_attention(
            ttnn_typecast_72,
            ttnn_typecast_73,
            ttnn_typecast_74,
            attn_mask=None,
            is_causal=False,
            scale=0.04419417679309845,
            sliding_window_size=None,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_74, False)
        ttnn.deallocate(ttnn_typecast_73, False)
        ttnn.deallocate(ttnn_typecast_72, False)
        ttnn_typecast_75 = ttnn.typecast(
            ttnn_transformer_scaled_dot_product_attention_0,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_0, False)
        ttnn_typecast_76 = ttnn.typecast(
            ttnn_typecast_75,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_75, False)
        ttnn_reshape_189 = ttnn.reshape(
            ttnn_typecast_76,
            [4096, 512],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_76, False)
        ttnn_linear_3 = ttnn.linear(
            ttnn_reshape_189,
            self.attn_args[126],
            bias=self.attn_args[125],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_reshape_189, False)
        ttnn_reshape_190 = ttnn.reshape(
            ttnn_linear_3,
            [1, 4096, 512],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_linear_3, False)
        ttnn_permute_168 = ttnn.permute(
            ttnn_reshape_190,
            [0, 2, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_190, False)
        ttnn_reshape_191 = ttnn.reshape(
            ttnn_permute_168,
            [1, 512, 64, 64],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_168, False)
        ttnn_add_7 = ttnn.add(
            ttnn_reshape_191,
            ttnn_divide_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_reshape_191, False)
        ttnn.deallocate(ttnn_divide_0, False)
        ttnn_divide_1 = ttnn.divide(
            ttnn_add_7,
            var_3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_7, False)
        ttnn_typecast_77 = ttnn.typecast(
            ttnn_divide_1,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_reshape_192 = ttnn.reshape(
            ttnn_typecast_77,
            [1, 32, 16, 4096],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_77, False)
        ttnn_sum_6 = ttnn.sum(
            ttnn_reshape_192,
            [2, 3],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=None,
        )
        ttnn_multiply_15 = ttnn.multiply(
            ttnn_sum_6,
            var_9,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sum_6, False)
        ttnn_subtract_3 = ttnn.subtract(
            ttnn_reshape_192,
            ttnn_multiply_15,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_15, False)
        ttnn.deallocate(ttnn_reshape_192, False)
        ttnn_multiply_16 = ttnn.multiply(
            ttnn_subtract_3,
            ttnn_subtract_3,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_sum_7 = ttnn.sum(
            ttnn_multiply_16,
            [2, 3],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_multiply_16, False)
        ttnn_multiply_17 = ttnn.multiply(
            ttnn_sum_7,
            var_9,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sum_7, False)
        ttnn_add_8 = ttnn.add(
            ttnn_multiply_17,
            var_5,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_17, False)
        ttnn_rsqrt_3 = ttnn.rsqrt(
            ttnn_add_8,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_8, False)
        ttnn_multiply_18 = ttnn.multiply(
            ttnn_subtract_3,
            ttnn_rsqrt_3,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_rsqrt_3, False)
        ttnn.deallocate(ttnn_subtract_3, False)
        ttnn_multiply_19 = ttnn.multiply(
            ttnn_multiply_18,
            self.ce_cache["main_const_eval_18"][0],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_18, False)
        ttnn_add_9 = ttnn.add(
            ttnn_multiply_19,
            self.ce_cache["main_const_eval_29"][0],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_19, False)
        ttnn_silu_2 = ttnn.silu(
            ttnn_add_9,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_9, False)
        ttnn_typecast_78 = ttnn.typecast(
            ttnn_silu_2,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_silu_2, False)
        ttnn_reshape_193 = ttnn.reshape(
            ttnn_typecast_78,
            [1, 512, 64, 64],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_78, False)
        ttnn_permute_169 = ttnn.permute(
            ttnn_reshape_193,
            [0, 2, 3, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_193, False)
        ttnn_reshape_194 = ttnn.reshape(
            ttnn_permute_169,
            [1, 1, 4096, 512],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_169, False)
        ttnn_conv2d_3 = ttnn.conv2d(
            input_tensor=ttnn_reshape_194,
            weight_tensor=self.ce_cache["main_const_eval_30"][0],
            device=device,
            in_channels=512,
            out_channels=512,
            batch_size=1,
            input_height=64,
            input_width=64,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=[1, 1],
            groups=1,
            bias_tensor=self.ce_cache["main_const_eval_48"][0],
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=ttnn.DataType.BFLOAT16,
                deallocate_activation=True,
                config_tensors_in_dram=True,
                act_block_h_override=0,
                enable_kernel_stride_folding=False,
            ),
            compute_config=None,
            slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_reshape_194, False)
        ttnn_typecast_79 = ttnn.typecast(
            ttnn_conv2d_3,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_conv2d_3, False)
        ttnn_reshape_195 = ttnn.reshape(
            ttnn_typecast_79,
            [1, 64, 64, 512],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_79, False)
        ttnn_permute_170 = ttnn.permute(
            ttnn_reshape_195,
            [0, 3, 1, 2],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_195, False)
        ttnn_reshape_196 = ttnn.reshape(
            ttnn_permute_170,
            [1, 32, 16, 4096],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_170, False)
        ttnn_sum_8 = ttnn.sum(
            ttnn_reshape_196,
            [2, 3],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=None,
        )
        ttnn_multiply_20 = ttnn.multiply(
            ttnn_sum_8,
            var_9,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sum_8, False)
        ttnn_subtract_4 = ttnn.subtract(
            ttnn_reshape_196,
            ttnn_multiply_20,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_20, False)
        ttnn.deallocate(ttnn_reshape_196, False)
        ttnn_multiply_21 = ttnn.multiply(
            ttnn_subtract_4,
            ttnn_subtract_4,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_sum_9 = ttnn.sum(
            ttnn_multiply_21,
            [2, 3],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_multiply_21, False)
        ttnn_multiply_22 = ttnn.multiply(
            ttnn_sum_9,
            var_9,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sum_9, False)
        ttnn_add_10 = ttnn.add(
            ttnn_multiply_22,
            var_5,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_22, False)
        ttnn_rsqrt_4 = ttnn.rsqrt(
            ttnn_add_10,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_10, False)
        ttnn_multiply_23 = ttnn.multiply(
            ttnn_subtract_4,
            ttnn_rsqrt_4,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_rsqrt_4, False)
        ttnn.deallocate(ttnn_subtract_4, False)
        ttnn_multiply_24 = ttnn.multiply(
            ttnn_multiply_23,
            self.ce_cache["main_const_eval_73"][0],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_23, False)
        ttnn_add_11 = ttnn.add(
            ttnn_multiply_24,
            self.ce_cache["main_const_eval_66"][0],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_24, False)
        ttnn_silu_3 = ttnn.silu(
            ttnn_add_11,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_11, False)
        ttnn_typecast_80 = ttnn.typecast(
            ttnn_silu_3,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_silu_3, False)
        ttnn_reshape_197 = ttnn.reshape(
            ttnn_typecast_80,
            [1, 512, 64, 64],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_80, False)
        ttnn_permute_171 = ttnn.permute(
            ttnn_reshape_197,
            [0, 2, 3, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_197, False)
        ttnn_reshape_198 = ttnn.reshape(
            ttnn_permute_171,
            [1, 1, 4096, 512],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_171, False)
        ttnn_conv2d_4 = ttnn.conv2d(
            input_tensor=ttnn_reshape_198,
            weight_tensor=self.ce_cache["main_const_eval_110"][0],
            device=device,
            in_channels=512,
            out_channels=512,
            batch_size=1,
            input_height=64,
            input_width=64,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=[1, 1],
            groups=1,
            bias_tensor=self.ce_cache["main_const_eval_22"][0],
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=ttnn.DataType.BFLOAT16,
                deallocate_activation=True,
                config_tensors_in_dram=True,
                act_block_h_override=0,
                enable_kernel_stride_folding=False,
            ),
            compute_config=None,
            slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_reshape_198, False)
        ttnn_reshape_199 = ttnn.reshape(
            ttnn_conv2d_4,
            [1, 64, 64, 512],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_conv2d_4, False)
        ttnn_permute_172 = ttnn.permute(
            ttnn_reshape_199,
            [0, 3, 1, 2],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_199, False)
        ttnn_add_12 = ttnn.add(
            ttnn_divide_1,
            ttnn_permute_172,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_172, False)
        ttnn.deallocate(ttnn_divide_1, False)
        ttnn_divide_2 = ttnn.divide(
            ttnn_add_12,
            var_3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_12, False)
        ttnn_typecast_81 = ttnn.typecast(
            ttnn_divide_2,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_reshape_200 = ttnn.reshape(
            ttnn_typecast_81,
            [1, 32, 16, 4096],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_81, False)
        ttnn_sum_10 = ttnn.sum(
            ttnn_reshape_200,
            [2, 3],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=None,
        )
        ttnn_multiply_25 = ttnn.multiply(
            ttnn_sum_10,
            var_9,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sum_10, False)
        ttnn_subtract_5 = ttnn.subtract(
            ttnn_reshape_200,
            ttnn_multiply_25,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_25, False)
        ttnn.deallocate(ttnn_reshape_200, False)
        ttnn_multiply_26 = ttnn.multiply(
            ttnn_subtract_5,
            ttnn_subtract_5,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_sum_11 = ttnn.sum(
            ttnn_multiply_26,
            [2, 3],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_multiply_26, False)
        ttnn_multiply_27 = ttnn.multiply(
            ttnn_sum_11,
            var_9,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sum_11, False)
        ttnn_add_13 = ttnn.add(
            ttnn_multiply_27,
            var_5,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_27, False)
        ttnn_rsqrt_5 = ttnn.rsqrt(
            ttnn_add_13,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_13, False)
        ttnn_multiply_28 = ttnn.multiply(
            ttnn_subtract_5,
            ttnn_rsqrt_5,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_rsqrt_5, False)
        ttnn.deallocate(ttnn_subtract_5, False)
        ttnn_multiply_29 = ttnn.multiply(
            ttnn_multiply_28,
            self.ce_cache["main_const_eval_2"][0],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_28, False)
        ttnn_add_14 = ttnn.add(
            ttnn_multiply_29,
            self.ce_cache["main_const_eval_24"][0],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_29, False)
        ttnn_silu_4 = ttnn.silu(
            ttnn_add_14,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_14, False)
        ttnn_typecast_82 = ttnn.typecast(
            ttnn_silu_4,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_silu_4, False)
        ttnn_reshape_201 = ttnn.reshape(
            ttnn_typecast_82,
            [1, 512, 64, 64],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_82, False)
        ttnn_permute_173 = ttnn.permute(
            ttnn_reshape_201,
            [0, 2, 3, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_201, False)
        ttnn_reshape_202 = ttnn.reshape(
            ttnn_permute_173,
            [1, 1, 4096, 512],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_173, False)
        ttnn_conv2d_5 = ttnn.conv2d(
            input_tensor=ttnn_reshape_202,
            weight_tensor=self.ce_cache["main_const_eval_52"][0],
            device=device,
            in_channels=512,
            out_channels=512,
            batch_size=1,
            input_height=64,
            input_width=64,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=[1, 1],
            groups=1,
            bias_tensor=self.ce_cache["main_const_eval_123"][0],
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=ttnn.DataType.BFLOAT16,
                deallocate_activation=True,
                config_tensors_in_dram=True,
                act_block_h_override=0,
                enable_kernel_stride_folding=False,
            ),
            compute_config=None,
            slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_reshape_202, False)
        ttnn_typecast_83 = ttnn.typecast(
            ttnn_conv2d_5,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_conv2d_5, False)
        ttnn_reshape_203 = ttnn.reshape(
            ttnn_typecast_83,
            [1, 64, 64, 512],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_83, False)
        ttnn_permute_174 = ttnn.permute(
            ttnn_reshape_203,
            [0, 3, 1, 2],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_203, False)
        ttnn_reshape_204 = ttnn.reshape(
            ttnn_permute_174,
            [1, 32, 16, 4096],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_174, False)
        ttnn_sum_12 = ttnn.sum(
            ttnn_reshape_204,
            [2, 3],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=None,
        )
        ttnn_multiply_30 = ttnn.multiply(
            ttnn_sum_12,
            var_9,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sum_12, False)
        ttnn_subtract_6 = ttnn.subtract(
            ttnn_reshape_204,
            ttnn_multiply_30,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_30, False)
        ttnn.deallocate(ttnn_reshape_204, False)
        ttnn_multiply_31 = ttnn.multiply(
            ttnn_subtract_6,
            ttnn_subtract_6,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_sum_13 = ttnn.sum(
            ttnn_multiply_31,
            [2, 3],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_multiply_31, False)
        ttnn_multiply_32 = ttnn.multiply(
            ttnn_sum_13,
            var_9,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sum_13, False)
        ttnn_add_15 = ttnn.add(
            ttnn_multiply_32,
            var_5,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_32, False)
        ttnn_rsqrt_6 = ttnn.rsqrt(
            ttnn_add_15,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_15, False)
        ttnn_multiply_33 = ttnn.multiply(
            ttnn_subtract_6,
            ttnn_rsqrt_6,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_rsqrt_6, False)
        ttnn.deallocate(ttnn_subtract_6, False)
        ttnn_multiply_34 = ttnn.multiply(
            ttnn_multiply_33,
            self.ce_cache["main_const_eval_72"][0],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_33, False)
        ttnn_add_16 = ttnn.add(
            ttnn_multiply_34,
            self.ce_cache["main_const_eval_122"][0],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_34, False)
        ttnn_silu_5 = ttnn.silu(
            ttnn_add_16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_16, False)
        ttnn_typecast_84 = ttnn.typecast(
            ttnn_silu_5,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_silu_5, False)
        ttnn_reshape_205 = ttnn.reshape(
            ttnn_typecast_84,
            [1, 512, 64, 64],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_84, False)
        ttnn_permute_175 = ttnn.permute(
            ttnn_reshape_205,
            [0, 2, 3, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_205, False)
        ttnn_reshape_206 = ttnn.reshape(
            ttnn_permute_175,
            [1, 1, 4096, 512],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_175, False)
        ttnn_conv2d_6 = ttnn.conv2d(
            input_tensor=ttnn_reshape_206,
            weight_tensor=self.ce_cache["main_const_eval_41"][0],
            device=device,
            in_channels=512,
            out_channels=512,
            batch_size=1,
            input_height=64,
            input_width=64,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=[1, 1],
            groups=1,
            bias_tensor=self.ce_cache["main_const_eval_36"][0],
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=ttnn.DataType.BFLOAT16,
                deallocate_activation=True,
                config_tensors_in_dram=True,
                act_block_h_override=0,
                enable_kernel_stride_folding=False,
            ),
            compute_config=None,
            slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_reshape_206, False)
        ttnn_reshape_207 = ttnn.reshape(
            ttnn_conv2d_6,
            [1, 64, 64, 512],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_conv2d_6, False)
        ttnn_permute_176 = ttnn.permute(
            ttnn_reshape_207,
            [0, 3, 1, 2],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_207, False)
        ttnn_add_17 = ttnn.add(
            ttnn_divide_2,
            ttnn_permute_176,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_176, False)
        ttnn.deallocate(ttnn_divide_2, False)
        ttnn_divide_3 = ttnn.divide(
            ttnn_add_17,
            var_3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_17, False)
        ttnn_typecast_85 = ttnn.typecast(
            ttnn_divide_3,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_reshape_208 = ttnn.reshape(
            ttnn_typecast_85,
            [1, 32, 16, 4096],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_85, False)
        ttnn_sum_14 = ttnn.sum(
            ttnn_reshape_208,
            [2, 3],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=None,
        )
        ttnn_multiply_35 = ttnn.multiply(
            ttnn_sum_14,
            var_9,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sum_14, False)
        ttnn_subtract_7 = ttnn.subtract(
            ttnn_reshape_208,
            ttnn_multiply_35,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_35, False)
        ttnn.deallocate(ttnn_reshape_208, False)
        ttnn_multiply_36 = ttnn.multiply(
            ttnn_subtract_7,
            ttnn_subtract_7,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_sum_15 = ttnn.sum(
            ttnn_multiply_36,
            [2, 3],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_multiply_36, False)
        ttnn_multiply_37 = ttnn.multiply(
            ttnn_sum_15,
            var_9,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sum_15, False)
        ttnn_add_18 = ttnn.add(
            ttnn_multiply_37,
            var_5,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_37, False)
        ttnn_rsqrt_7 = ttnn.rsqrt(
            ttnn_add_18,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_18, False)
        ttnn_multiply_38 = ttnn.multiply(
            ttnn_subtract_7,
            ttnn_rsqrt_7,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_rsqrt_7, False)
        ttnn.deallocate(ttnn_subtract_7, False)
        ttnn_multiply_39 = ttnn.multiply(
            ttnn_multiply_38,
            self.ce_cache["main_const_eval_5"][0],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_38, False)
        ttnn_add_19 = ttnn.add(
            ttnn_multiply_39,
            self.ce_cache["main_const_eval_58"][0],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_39, False)
        ttnn_silu_6 = ttnn.silu(
            ttnn_add_19,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_19, False)
        ttnn_typecast_86 = ttnn.typecast(
            ttnn_silu_6,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_silu_6, False)
        ttnn_reshape_209 = ttnn.reshape(
            ttnn_typecast_86,
            [1, 512, 64, 64],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_86, False)
        ttnn_permute_177 = ttnn.permute(
            ttnn_reshape_209,
            [0, 2, 3, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_209, False)
        ttnn_reshape_210 = ttnn.reshape(
            ttnn_permute_177,
            [1, 1, 4096, 512],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_177, False)
        ttnn_conv2d_7 = ttnn.conv2d(
            input_tensor=ttnn_reshape_210,
            weight_tensor=self.ce_cache["main_const_eval_137"][0],
            device=device,
            in_channels=512,
            out_channels=512,
            batch_size=1,
            input_height=64,
            input_width=64,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=[1, 1],
            groups=1,
            bias_tensor=self.ce_cache["main_const_eval_60"][0],
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=ttnn.DataType.BFLOAT16,
                deallocate_activation=True,
                config_tensors_in_dram=True,
                act_block_h_override=0,
                enable_kernel_stride_folding=False,
            ),
            compute_config=None,
            slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_reshape_210, False)
        ttnn_typecast_87 = ttnn.typecast(
            ttnn_conv2d_7,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_conv2d_7, False)
        ttnn_reshape_211 = ttnn.reshape(
            ttnn_typecast_87,
            [1, 64, 64, 512],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_87, False)
        ttnn_permute_178 = ttnn.permute(
            ttnn_reshape_211,
            [0, 3, 1, 2],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_211, False)
        ttnn_reshape_212 = ttnn.reshape(
            ttnn_permute_178,
            [1, 32, 16, 4096],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_178, False)
        ttnn_sum_16 = ttnn.sum(
            ttnn_reshape_212,
            [2, 3],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=None,
        )
        ttnn_multiply_40 = ttnn.multiply(
            ttnn_sum_16,
            var_9,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sum_16, False)
        ttnn_subtract_8 = ttnn.subtract(
            ttnn_reshape_212,
            ttnn_multiply_40,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_40, False)
        ttnn.deallocate(ttnn_reshape_212, False)
        ttnn_multiply_41 = ttnn.multiply(
            ttnn_subtract_8,
            ttnn_subtract_8,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_sum_17 = ttnn.sum(
            ttnn_multiply_41,
            [2, 3],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_multiply_41, False)
        ttnn_multiply_42 = ttnn.multiply(
            ttnn_sum_17,
            var_9,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sum_17, False)
        ttnn_add_20 = ttnn.add(
            ttnn_multiply_42,
            var_5,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_42, False)
        ttnn_rsqrt_8 = ttnn.rsqrt(
            ttnn_add_20,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_20, False)
        ttnn_multiply_43 = ttnn.multiply(
            ttnn_subtract_8,
            ttnn_rsqrt_8,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_rsqrt_8, False)
        ttnn.deallocate(ttnn_subtract_8, False)
        ttnn_multiply_44 = ttnn.multiply(
            ttnn_multiply_43,
            self.ce_cache["main_const_eval_86"][0],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_43, False)
        ttnn_add_21 = ttnn.add(
            ttnn_multiply_44,
            self.ce_cache["main_const_eval_97"][0],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_44, False)
        ttnn_silu_7 = ttnn.silu(
            ttnn_add_21,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_21, False)
        ttnn_typecast_88 = ttnn.typecast(
            ttnn_silu_7,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_silu_7, False)
        ttnn_reshape_213 = ttnn.reshape(
            ttnn_typecast_88,
            [1, 512, 64, 64],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_88, False)
        ttnn_permute_179 = ttnn.permute(
            ttnn_reshape_213,
            [0, 2, 3, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_213, False)
        ttnn_reshape_214 = ttnn.reshape(
            ttnn_permute_179,
            [1, 1, 4096, 512],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_179, False)
        ttnn_conv2d_8 = ttnn.conv2d(
            input_tensor=ttnn_reshape_214,
            weight_tensor=self.ce_cache["main_const_eval_32"][0],
            device=device,
            in_channels=512,
            out_channels=512,
            batch_size=1,
            input_height=64,
            input_width=64,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=[1, 1],
            groups=1,
            bias_tensor=self.ce_cache["main_const_eval_11"][0],
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=ttnn.DataType.BFLOAT16,
                deallocate_activation=True,
                config_tensors_in_dram=True,
                act_block_h_override=0,
                enable_kernel_stride_folding=False,
            ),
            compute_config=None,
            slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_reshape_214, False)
        ttnn_reshape_215 = ttnn.reshape(
            ttnn_conv2d_8,
            [1, 64, 64, 512],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_conv2d_8, False)
        ttnn_permute_180 = ttnn.permute(
            ttnn_reshape_215,
            [0, 3, 1, 2],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_215, False)
        ttnn_add_22 = ttnn.add(
            ttnn_divide_3,
            ttnn_permute_180,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_180, False)
        ttnn.deallocate(ttnn_divide_3, False)
        ttnn_divide_4 = ttnn.divide(
            ttnn_add_22,
            var_3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_22, False)
        ttnn_typecast_89 = ttnn.typecast(
            ttnn_divide_4,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_reshape_216 = ttnn.reshape(
            ttnn_typecast_89,
            [1, 32, 16, 4096],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_89, False)
        ttnn_sum_18 = ttnn.sum(
            ttnn_reshape_216,
            [2, 3],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=None,
        )
        ttnn_multiply_45 = ttnn.multiply(
            ttnn_sum_18,
            var_9,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sum_18, False)
        ttnn_subtract_9 = ttnn.subtract(
            ttnn_reshape_216,
            ttnn_multiply_45,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_45, False)
        ttnn.deallocate(ttnn_reshape_216, False)
        ttnn_multiply_46 = ttnn.multiply(
            ttnn_subtract_9,
            ttnn_subtract_9,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_sum_19 = ttnn.sum(
            ttnn_multiply_46,
            [2, 3],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_multiply_46, False)
        ttnn_multiply_47 = ttnn.multiply(
            ttnn_sum_19,
            var_9,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sum_19, False)
        ttnn_add_23 = ttnn.add(
            ttnn_multiply_47,
            var_5,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_47, False)
        ttnn_rsqrt_9 = ttnn.rsqrt(
            ttnn_add_23,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_23, False)
        ttnn_multiply_48 = ttnn.multiply(
            ttnn_subtract_9,
            ttnn_rsqrt_9,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_rsqrt_9, False)
        ttnn.deallocate(ttnn_subtract_9, False)
        ttnn_multiply_49 = ttnn.multiply(
            ttnn_multiply_48,
            self.ce_cache["main_const_eval_10"][0],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_48, False)
        ttnn_add_24 = ttnn.add(
            ttnn_multiply_49,
            self.ce_cache["main_const_eval_132"][0],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_49, False)
        ttnn_silu_8 = ttnn.silu(
            ttnn_add_24,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_24, False)
        ttnn_typecast_90 = ttnn.typecast(
            ttnn_silu_8,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_silu_8, False)
        ttnn_reshape_217 = ttnn.reshape(
            ttnn_typecast_90,
            [1, 512, 64, 64],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_90, False)
        ttnn_permute_181 = ttnn.permute(
            ttnn_reshape_217,
            [0, 2, 3, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_217, False)
        ttnn_reshape_218 = ttnn.reshape(
            ttnn_permute_181,
            [1, 1, 4096, 512],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_181, False)
        ttnn_conv2d_9 = ttnn.conv2d(
            input_tensor=ttnn_reshape_218,
            weight_tensor=self.ce_cache["main_const_eval_45"][0],
            device=device,
            in_channels=512,
            out_channels=512,
            batch_size=1,
            input_height=64,
            input_width=64,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=[1, 1],
            groups=1,
            bias_tensor=self.ce_cache["main_const_eval_82"][0],
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=ttnn.DataType.BFLOAT16,
                deallocate_activation=True,
                config_tensors_in_dram=True,
                act_block_h_override=0,
                enable_kernel_stride_folding=False,
            ),
            compute_config=None,
            slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_reshape_218, False)
        ttnn_typecast_91 = ttnn.typecast(
            ttnn_conv2d_9,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_conv2d_9, False)
        ttnn_reshape_219 = ttnn.reshape(
            ttnn_typecast_91,
            [1, 64, 64, 512],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_91, False)
        ttnn_permute_182 = ttnn.permute(
            ttnn_reshape_219,
            [0, 3, 1, 2],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_219, False)
        ttnn_reshape_220 = ttnn.reshape(
            ttnn_permute_182,
            [1, 32, 16, 4096],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_182, False)
        ttnn_sum_20 = ttnn.sum(
            ttnn_reshape_220,
            [2, 3],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=None,
        )
        ttnn_multiply_50 = ttnn.multiply(
            ttnn_sum_20,
            var_9,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sum_20, False)
        ttnn_subtract_10 = ttnn.subtract(
            ttnn_reshape_220,
            ttnn_multiply_50,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_50, False)
        ttnn.deallocate(ttnn_reshape_220, False)
        ttnn_multiply_51 = ttnn.multiply(
            ttnn_subtract_10,
            ttnn_subtract_10,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_sum_21 = ttnn.sum(
            ttnn_multiply_51,
            [2, 3],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_multiply_51, False)
        ttnn_multiply_52 = ttnn.multiply(
            ttnn_sum_21,
            var_9,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sum_21, False)
        ttnn_add_25 = ttnn.add(
            ttnn_multiply_52,
            var_5,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_52, False)
        ttnn_rsqrt_10 = ttnn.rsqrt(
            ttnn_add_25,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_25, False)
        ttnn_multiply_53 = ttnn.multiply(
            ttnn_subtract_10,
            ttnn_rsqrt_10,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_rsqrt_10, False)
        ttnn.deallocate(ttnn_subtract_10, False)
        ttnn_multiply_54 = ttnn.multiply(
            ttnn_multiply_53,
            self.ce_cache["main_const_eval_98"][0],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_53, False)
        ttnn_add_26 = ttnn.add(
            ttnn_multiply_54,
            self.ce_cache["main_const_eval_42"][0],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_54, False)
        ttnn_silu_9 = ttnn.silu(
            ttnn_add_26,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_26, False)
        ttnn_typecast_92 = ttnn.typecast(
            ttnn_silu_9,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_silu_9, False)
        ttnn_reshape_221 = ttnn.reshape(
            ttnn_typecast_92,
            [1, 512, 64, 64],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_92, False)
        ttnn_permute_183 = ttnn.permute(
            ttnn_reshape_221,
            [0, 2, 3, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_221, False)
        ttnn_reshape_222 = ttnn.reshape(
            ttnn_permute_183,
            [1, 1, 4096, 512],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_183, False)
        ttnn_conv2d_10 = ttnn.conv2d(
            input_tensor=ttnn_reshape_222,
            weight_tensor=self.ce_cache["main_const_eval_117"][0],
            device=device,
            in_channels=512,
            out_channels=512,
            batch_size=1,
            input_height=64,
            input_width=64,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=[1, 1],
            groups=1,
            bias_tensor=self.ce_cache["main_const_eval_4"][0],
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=ttnn.DataType.BFLOAT16,
                deallocate_activation=True,
                config_tensors_in_dram=True,
                act_block_h_override=0,
                enable_kernel_stride_folding=False,
            ),
            compute_config=None,
            slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_reshape_222, False)
        ttnn_reshape_223 = ttnn.reshape(
            ttnn_conv2d_10,
            [1, 64, 64, 512],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_conv2d_10, False)
        ttnn_permute_184 = ttnn.permute(
            ttnn_divide_4,
            [0, 1, 3, 2],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_divide_4, False)
        ttnn_permute_185 = ttnn.permute(
            ttnn_reshape_223,
            [0, 3, 2, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_223, False)
        ttnn_add_27 = ttnn.add(
            ttnn_permute_184,
            ttnn_permute_185,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_185, False)
        ttnn.deallocate(ttnn_permute_184, False)
        ttnn_reshape_224 = ttnn.reshape(
            ttnn_add_27,
            [32768, 64],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_27, False)
        ttnn_divide_5 = ttnn.divide(
            ttnn_reshape_224,
            var_4,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_reshape_224, False)
        ttnn_matmul_0 = ttnn.matmul(
            ttnn_divide_5,
            var_12,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_divide_5, False)
        ttnn_reshape_225 = ttnn.reshape(
            ttnn_matmul_0,
            [1, 512, 64, 128],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_matmul_0, False)
        ttnn_permute_186 = ttnn.permute(
            ttnn_reshape_225,
            [0, 1, 3, 2],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_225, False)
        ttnn_reshape_226 = ttnn.reshape(
            ttnn_permute_186,
            [65536, 64],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_186, False)
        ttnn_matmul_1 = ttnn.matmul(
            ttnn_reshape_226,
            var_12,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_reshape_226, False)
        ttnn_reshape_227 = ttnn.reshape(
            ttnn_matmul_1,
            [1, 512, 128, 128],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_matmul_1, False)
        ttnn_permute_187 = ttnn.permute(
            ttnn_reshape_227,
            [0, 2, 3, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_227, False)
        ttnn_reshape_228 = ttnn.reshape(
            ttnn_permute_187,
            [1, 1, 16384, 512],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_187, False)
        ttnn_conv2d_11 = ttnn.conv2d(
            input_tensor=ttnn_reshape_228,
            weight_tensor=self.ce_cache["main_const_eval_49"][0],
            device=device,
            in_channels=512,
            out_channels=512,
            batch_size=1,
            input_height=128,
            input_width=128,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=[1, 1],
            groups=1,
            bias_tensor=self.ce_cache["main_const_eval_95"][0],
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=ttnn.DataType.BFLOAT16,
                deallocate_activation=True,
                config_tensors_in_dram=True,
                act_block_h_override=1024,
                enable_kernel_stride_folding=False,
            ),
            compute_config=None,
            slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_reshape_228, False)
        ttnn_typecast_93 = ttnn.typecast(
            ttnn_conv2d_11,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_reshape_229 = ttnn.reshape(
            ttnn_typecast_93,
            [1, 128, 128, 512],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_93, False)
        ttnn_permute_188 = ttnn.permute(
            ttnn_reshape_229,
            [0, 3, 1, 2],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_229, False)
        ttnn_reshape_230 = ttnn.reshape(
            ttnn_permute_188,
            [1, 32, 16, 16384],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_188, False)
        ttnn_sum_22 = ttnn.sum(
            ttnn_reshape_230,
            [2, 3],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=None,
        )
        ttnn_multiply_55 = ttnn.multiply(
            ttnn_sum_22,
            var_7,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sum_22, False)
        ttnn_subtract_11 = ttnn.subtract(
            ttnn_reshape_230,
            ttnn_multiply_55,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_55, False)
        ttnn.deallocate(ttnn_reshape_230, False)
        ttnn_multiply_56 = ttnn.multiply(
            ttnn_subtract_11,
            ttnn_subtract_11,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_sum_23 = ttnn.sum(
            ttnn_multiply_56,
            [2, 3],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_multiply_56, False)
        ttnn_multiply_57 = ttnn.multiply(
            ttnn_sum_23,
            var_7,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sum_23, False)
        ttnn_add_28 = ttnn.add(
            ttnn_multiply_57,
            var_5,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_57, False)
        ttnn_rsqrt_11 = ttnn.rsqrt(
            ttnn_add_28,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_28, False)
        ttnn_multiply_58 = ttnn.multiply(
            ttnn_subtract_11,
            ttnn_rsqrt_11,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_rsqrt_11, False)
        ttnn.deallocate(ttnn_subtract_11, False)
        ttnn_multiply_59 = ttnn.multiply(
            ttnn_multiply_58,
            self.ce_cache["main_const_eval_88"][0],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_58, False)
        ttnn_add_29 = ttnn.add(
            ttnn_multiply_59,
            self.ce_cache["main_const_eval_107"][0],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_59, False)
        ttnn_silu_10 = ttnn.silu(
            ttnn_add_29,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_29, False)
        ttnn_typecast_94 = ttnn.typecast(
            ttnn_silu_10,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_silu_10, False)
        ttnn_reshape_231 = ttnn.reshape(
            ttnn_typecast_94,
            [1, 512, 128, 128],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_94, False)
        ttnn_permute_189 = ttnn.permute(
            ttnn_reshape_231,
            [0, 2, 3, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_231, False)
        ttnn_reshape_232 = ttnn.reshape(
            ttnn_permute_189,
            [1, 1, 16384, 512],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_189, False)
        ttnn_conv2d_12 = ttnn.conv2d(
            input_tensor=ttnn_reshape_232,
            weight_tensor=self.ce_cache["main_const_eval_53"][0],
            device=device,
            in_channels=512,
            out_channels=512,
            batch_size=1,
            input_height=128,
            input_width=128,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=[1, 1],
            groups=1,
            bias_tensor=self.ce_cache["main_const_eval_7"][0],
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=ttnn.DataType.BFLOAT16,
                deallocate_activation=True,
                config_tensors_in_dram=True,
                act_block_h_override=1024,
                enable_kernel_stride_folding=False,
            ),
            compute_config=None,
            slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_reshape_232, False)
        ttnn_typecast_95 = ttnn.typecast(
            ttnn_conv2d_12,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_conv2d_12, False)
        ttnn_reshape_233 = ttnn.reshape(
            ttnn_typecast_95,
            [1, 128, 128, 512],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_95, False)
        ttnn_permute_190 = ttnn.permute(
            ttnn_reshape_233,
            [0, 3, 1, 2],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_233, False)
        ttnn_reshape_234 = ttnn.reshape(
            ttnn_permute_190,
            [1, 32, 16, 16384],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_190, False)
        ttnn_sum_24 = ttnn.sum(
            ttnn_reshape_234,
            [2, 3],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=None,
        )
        ttnn_multiply_60 = ttnn.multiply(
            ttnn_sum_24,
            var_7,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sum_24, False)
        ttnn_subtract_12 = ttnn.subtract(
            ttnn_reshape_234,
            ttnn_multiply_60,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_60, False)
        ttnn.deallocate(ttnn_reshape_234, False)
        ttnn_multiply_61 = ttnn.multiply(
            ttnn_subtract_12,
            ttnn_subtract_12,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_sum_25 = ttnn.sum(
            ttnn_multiply_61,
            [2, 3],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_multiply_61, False)
        ttnn_multiply_62 = ttnn.multiply(
            ttnn_sum_25,
            var_7,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sum_25, False)
        ttnn_add_30 = ttnn.add(
            ttnn_multiply_62,
            var_5,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_62, False)
        ttnn_rsqrt_12 = ttnn.rsqrt(
            ttnn_add_30,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_30, False)
        ttnn_multiply_63 = ttnn.multiply(
            ttnn_subtract_12,
            ttnn_rsqrt_12,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_rsqrt_12, False)
        ttnn.deallocate(ttnn_subtract_12, False)
        ttnn_multiply_64 = ttnn.multiply(
            ttnn_multiply_63,
            self.ce_cache["main_const_eval_51"][0],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_63, False)
        ttnn_add_31 = ttnn.add(
            ttnn_multiply_64,
            self.ce_cache["main_const_eval_105"][0],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_64, False)
        ttnn_silu_11 = ttnn.silu(
            ttnn_add_31,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_31, False)
        ttnn_typecast_96 = ttnn.typecast(
            ttnn_silu_11,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_silu_11, False)
        ttnn_reshape_235 = ttnn.reshape(
            ttnn_typecast_96,
            [1, 512, 128, 128],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_96, False)
        ttnn_permute_191 = ttnn.permute(
            ttnn_reshape_235,
            [0, 2, 3, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_235, False)
        ttnn_reshape_236 = ttnn.reshape(
            ttnn_permute_191,
            [1, 1, 16384, 512],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_191, False)
        ttnn_conv2d_13 = ttnn.conv2d(
            input_tensor=ttnn_reshape_236,
            weight_tensor=self.ce_cache["main_const_eval_93"][0],
            device=device,
            in_channels=512,
            out_channels=512,
            batch_size=1,
            input_height=128,
            input_width=128,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=[1, 1],
            groups=1,
            bias_tensor=self.ce_cache["main_const_eval_83"][0],
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=ttnn.DataType.BFLOAT16,
                deallocate_activation=True,
                config_tensors_in_dram=True,
                act_block_h_override=1024,
                enable_kernel_stride_folding=False,
            ),
            compute_config=None,
            slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_reshape_236, False)
        ttnn_add_32 = ttnn.add(
            ttnn_conv2d_11,
            ttnn_conv2d_13,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_conv2d_13, False)
        ttnn.deallocate(ttnn_conv2d_11, False)
        ttnn_reshape_237 = ttnn.reshape(
            ttnn_add_32,
            [1, 128, 128, 512],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_32, False)
        ttnn_permute_192 = ttnn.permute(
            ttnn_reshape_237,
            [0, 3, 1, 2],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_237, False)
        ttnn_divide_6 = ttnn.divide(
            ttnn_permute_192,
            var_3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_192, False)
        ttnn_typecast_97 = ttnn.typecast(
            ttnn_divide_6,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_reshape_238 = ttnn.reshape(
            ttnn_typecast_97,
            [1, 32, 16, 16384],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_97, False)
        ttnn_sum_26 = ttnn.sum(
            ttnn_reshape_238,
            [2, 3],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=None,
        )
        ttnn_multiply_65 = ttnn.multiply(
            ttnn_sum_26,
            var_7,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sum_26, False)
        ttnn_subtract_13 = ttnn.subtract(
            ttnn_reshape_238,
            ttnn_multiply_65,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_65, False)
        ttnn.deallocate(ttnn_reshape_238, False)
        ttnn_multiply_66 = ttnn.multiply(
            ttnn_subtract_13,
            ttnn_subtract_13,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_sum_27 = ttnn.sum(
            ttnn_multiply_66,
            [2, 3],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_multiply_66, False)
        ttnn_multiply_67 = ttnn.multiply(
            ttnn_sum_27,
            var_7,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sum_27, False)
        ttnn_add_33 = ttnn.add(
            ttnn_multiply_67,
            var_5,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_67, False)
        ttnn_rsqrt_13 = ttnn.rsqrt(
            ttnn_add_33,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_33, False)
        ttnn_multiply_68 = ttnn.multiply(
            ttnn_subtract_13,
            ttnn_rsqrt_13,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_rsqrt_13, False)
        ttnn.deallocate(ttnn_subtract_13, False)
        ttnn_multiply_69 = ttnn.multiply(
            ttnn_multiply_68,
            self.ce_cache["main_const_eval_127"][0],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_68, False)
        ttnn_add_34 = ttnn.add(
            ttnn_multiply_69,
            self.ce_cache["main_const_eval_67"][0],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_69, False)
        ttnn_silu_12 = ttnn.silu(
            ttnn_add_34,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_34, False)
        ttnn_typecast_98 = ttnn.typecast(
            ttnn_silu_12,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_silu_12, False)
        ttnn_reshape_239 = ttnn.reshape(
            ttnn_typecast_98,
            [1, 512, 128, 128],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_98, False)
        ttnn_permute_193 = ttnn.permute(
            ttnn_reshape_239,
            [0, 2, 3, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_239, False)
        ttnn_reshape_240 = ttnn.reshape(
            ttnn_permute_193,
            [1, 1, 16384, 512],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_193, False)
        ttnn_conv2d_14 = ttnn.conv2d(
            input_tensor=ttnn_reshape_240,
            weight_tensor=self.ce_cache["main_const_eval_128"][0],
            device=device,
            in_channels=512,
            out_channels=512,
            batch_size=1,
            input_height=128,
            input_width=128,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=[1, 1],
            groups=1,
            bias_tensor=self.ce_cache["main_const_eval_12"][0],
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=ttnn.DataType.BFLOAT16,
                deallocate_activation=True,
                config_tensors_in_dram=True,
                act_block_h_override=1024,
                enable_kernel_stride_folding=False,
            ),
            compute_config=None,
            slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_reshape_240, False)
        ttnn_typecast_99 = ttnn.typecast(
            ttnn_conv2d_14,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_conv2d_14, False)
        ttnn_reshape_241 = ttnn.reshape(
            ttnn_typecast_99,
            [1, 128, 128, 512],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_99, False)
        ttnn_permute_194 = ttnn.permute(
            ttnn_reshape_241,
            [0, 3, 1, 2],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_241, False)
        ttnn_reshape_242 = ttnn.reshape(
            ttnn_permute_194,
            [1, 32, 16, 16384],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_194, False)
        ttnn_sum_28 = ttnn.sum(
            ttnn_reshape_242,
            [2, 3],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=None,
        )
        ttnn_multiply_70 = ttnn.multiply(
            ttnn_sum_28,
            var_7,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sum_28, False)
        ttnn_subtract_14 = ttnn.subtract(
            ttnn_reshape_242,
            ttnn_multiply_70,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_70, False)
        ttnn.deallocate(ttnn_reshape_242, False)
        ttnn_multiply_71 = ttnn.multiply(
            ttnn_subtract_14,
            ttnn_subtract_14,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_sum_29 = ttnn.sum(
            ttnn_multiply_71,
            [2, 3],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_multiply_71, False)
        ttnn_multiply_72 = ttnn.multiply(
            ttnn_sum_29,
            var_7,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sum_29, False)
        ttnn_add_35 = ttnn.add(
            ttnn_multiply_72,
            var_5,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_72, False)
        ttnn_rsqrt_14 = ttnn.rsqrt(
            ttnn_add_35,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_35, False)
        ttnn_multiply_73 = ttnn.multiply(
            ttnn_subtract_14,
            ttnn_rsqrt_14,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_rsqrt_14, False)
        ttnn.deallocate(ttnn_subtract_14, False)
        ttnn_multiply_74 = ttnn.multiply(
            ttnn_multiply_73,
            self.ce_cache["main_const_eval_31"][0],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_73, False)
        ttnn_add_36 = ttnn.add(
            ttnn_multiply_74,
            self.ce_cache["main_const_eval_111"][0],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_74, False)
        ttnn_silu_13 = ttnn.silu(
            ttnn_add_36,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_36, False)
        ttnn_typecast_100 = ttnn.typecast(
            ttnn_silu_13,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_silu_13, False)
        ttnn_reshape_243 = ttnn.reshape(
            ttnn_typecast_100,
            [1, 512, 128, 128],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_100, False)
        ttnn_permute_195 = ttnn.permute(
            ttnn_reshape_243,
            [0, 2, 3, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_243, False)
        ttnn_reshape_244 = ttnn.reshape(
            ttnn_permute_195,
            [1, 1, 16384, 512],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_195, False)
        ttnn_conv2d_15 = ttnn.conv2d(
            input_tensor=ttnn_reshape_244,
            weight_tensor=self.ce_cache["main_const_eval_78"][0],
            device=device,
            in_channels=512,
            out_channels=512,
            batch_size=1,
            input_height=128,
            input_width=128,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=[1, 1],
            groups=1,
            bias_tensor=self.ce_cache["main_const_eval_131"][0],
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=ttnn.DataType.BFLOAT16,
                deallocate_activation=True,
                config_tensors_in_dram=True,
                act_block_h_override=1024,
                enable_kernel_stride_folding=False,
            ),
            compute_config=None,
            slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_reshape_244, False)
        ttnn_reshape_245 = ttnn.reshape(
            ttnn_conv2d_15,
            [1, 128, 128, 512],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_conv2d_15, False)
        ttnn_permute_196 = ttnn.permute(
            ttnn_reshape_245,
            [0, 3, 1, 2],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_245, False)
        ttnn_add_37 = ttnn.add(
            ttnn_divide_6,
            ttnn_permute_196,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_196, False)
        ttnn.deallocate(ttnn_divide_6, False)
        ttnn_divide_7 = ttnn.divide(
            ttnn_add_37,
            var_3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_37, False)
        ttnn_typecast_101 = ttnn.typecast(
            ttnn_divide_7,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_reshape_246 = ttnn.reshape(
            ttnn_typecast_101,
            [1, 32, 16, 16384],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_101, False)
        ttnn_sum_30 = ttnn.sum(
            ttnn_reshape_246,
            [2, 3],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=None,
        )
        ttnn_multiply_75 = ttnn.multiply(
            ttnn_sum_30,
            var_7,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sum_30, False)
        ttnn_subtract_15 = ttnn.subtract(
            ttnn_reshape_246,
            ttnn_multiply_75,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_75, False)
        ttnn.deallocate(ttnn_reshape_246, False)
        ttnn_multiply_76 = ttnn.multiply(
            ttnn_subtract_15,
            ttnn_subtract_15,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_sum_31 = ttnn.sum(
            ttnn_multiply_76,
            [2, 3],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_multiply_76, False)
        ttnn_multiply_77 = ttnn.multiply(
            ttnn_sum_31,
            var_7,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sum_31, False)
        ttnn_add_38 = ttnn.add(
            ttnn_multiply_77,
            var_5,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_77, False)
        ttnn_rsqrt_15 = ttnn.rsqrt(
            ttnn_add_38,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_38, False)
        ttnn_multiply_78 = ttnn.multiply(
            ttnn_subtract_15,
            ttnn_rsqrt_15,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_rsqrt_15, False)
        ttnn.deallocate(ttnn_subtract_15, False)
        ttnn_multiply_79 = ttnn.multiply(
            ttnn_multiply_78,
            self.ce_cache["main_const_eval_65"][0],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_78, False)
        ttnn_add_39 = ttnn.add(
            ttnn_multiply_79,
            self.ce_cache["main_const_eval_115"][0],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_79, False)
        ttnn_silu_14 = ttnn.silu(
            ttnn_add_39,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_39, False)
        ttnn_typecast_102 = ttnn.typecast(
            ttnn_silu_14,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_silu_14, False)
        ttnn_reshape_247 = ttnn.reshape(
            ttnn_typecast_102,
            [1, 512, 128, 128],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_102, False)
        ttnn_permute_197 = ttnn.permute(
            ttnn_reshape_247,
            [0, 2, 3, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_247, False)
        ttnn_reshape_248 = ttnn.reshape(
            ttnn_permute_197,
            [1, 1, 16384, 512],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_197, False)
        ttnn_conv2d_16 = ttnn.conv2d(
            input_tensor=ttnn_reshape_248,
            weight_tensor=self.ce_cache["main_const_eval_6"][0],
            device=device,
            in_channels=512,
            out_channels=512,
            batch_size=1,
            input_height=128,
            input_width=128,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=[1, 1],
            groups=1,
            bias_tensor=self.ce_cache["main_const_eval_37"][0],
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=ttnn.DataType.BFLOAT16,
                deallocate_activation=True,
                config_tensors_in_dram=True,
                act_block_h_override=1024,
                enable_kernel_stride_folding=False,
            ),
            compute_config=None,
            slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_reshape_248, False)
        ttnn_typecast_103 = ttnn.typecast(
            ttnn_conv2d_16,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_conv2d_16, False)
        ttnn_reshape_249 = ttnn.reshape(
            ttnn_typecast_103,
            [1, 128, 128, 512],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_103, False)
        ttnn_permute_198 = ttnn.permute(
            ttnn_reshape_249,
            [0, 3, 1, 2],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_249, False)
        ttnn_reshape_250 = ttnn.reshape(
            ttnn_permute_198,
            [1, 32, 16, 16384],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_198, False)
        ttnn_sum_32 = ttnn.sum(
            ttnn_reshape_250,
            [2, 3],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=None,
        )
        ttnn_multiply_80 = ttnn.multiply(
            ttnn_sum_32,
            var_7,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sum_32, False)
        ttnn_subtract_16 = ttnn.subtract(
            ttnn_reshape_250,
            ttnn_multiply_80,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_80, False)
        ttnn.deallocate(ttnn_reshape_250, False)
        ttnn_multiply_81 = ttnn.multiply(
            ttnn_subtract_16,
            ttnn_subtract_16,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_sum_33 = ttnn.sum(
            ttnn_multiply_81,
            [2, 3],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_multiply_81, False)
        ttnn_multiply_82 = ttnn.multiply(
            ttnn_sum_33,
            var_7,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sum_33, False)
        ttnn_add_40 = ttnn.add(
            ttnn_multiply_82,
            var_5,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_82, False)
        ttnn_rsqrt_16 = ttnn.rsqrt(
            ttnn_add_40,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_40, False)
        ttnn_multiply_83 = ttnn.multiply(
            ttnn_subtract_16,
            ttnn_rsqrt_16,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_rsqrt_16, False)
        ttnn.deallocate(ttnn_subtract_16, False)
        ttnn_multiply_84 = ttnn.multiply(
            ttnn_multiply_83,
            self.ce_cache["main_const_eval_108"][0],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_83, False)
        ttnn_add_41 = ttnn.add(
            ttnn_multiply_84,
            self.ce_cache["main_const_eval_59"][0],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_84, False)
        ttnn_silu_15 = ttnn.silu(
            ttnn_add_41,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_41, False)
        ttnn_typecast_104 = ttnn.typecast(
            ttnn_silu_15,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_silu_15, False)
        ttnn_reshape_251 = ttnn.reshape(
            ttnn_typecast_104,
            [1, 512, 128, 128],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_104, False)
        ttnn_permute_199 = ttnn.permute(
            ttnn_reshape_251,
            [0, 2, 3, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_251, False)
        ttnn_reshape_252 = ttnn.reshape(
            ttnn_permute_199,
            [1, 1, 16384, 512],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_199, False)
        ttnn_conv2d_17 = ttnn.conv2d(
            input_tensor=ttnn_reshape_252,
            weight_tensor=self.ce_cache["main_const_eval_139"][0],
            device=device,
            in_channels=512,
            out_channels=512,
            batch_size=1,
            input_height=128,
            input_width=128,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=[1, 1],
            groups=1,
            bias_tensor=self.ce_cache["main_const_eval_62"][0],
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=ttnn.DataType.BFLOAT16,
                deallocate_activation=True,
                config_tensors_in_dram=True,
                act_block_h_override=1024,
                enable_kernel_stride_folding=False,
            ),
            compute_config=None,
            slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_reshape_252, False)
        ttnn_reshape_253 = ttnn.reshape(
            ttnn_conv2d_17,
            [1, 128, 128, 512],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_conv2d_17, False)
        ttnn_permute_200 = ttnn.permute(
            ttnn_divide_7,
            [0, 1, 3, 2],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_divide_7, False)
        ttnn_permute_201 = ttnn.permute(
            ttnn_reshape_253,
            [0, 3, 2, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_253, False)
        ttnn_add_42 = ttnn.add(
            ttnn_permute_200,
            ttnn_permute_201,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_201, False)
        ttnn.deallocate(ttnn_permute_200, False)
        ttnn_reshape_254 = ttnn.reshape(
            ttnn_add_42,
            [65536, 128],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_42, False)
        ttnn_divide_8 = ttnn.divide(
            ttnn_reshape_254,
            var_4,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_reshape_254, False)
        ttnn_matmul_2 = ttnn.matmul(
            ttnn_divide_8,
            var_10,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_divide_8, False)
        ttnn_reshape_255 = ttnn.reshape(
            ttnn_matmul_2,
            [1, 512, 128, 256],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_matmul_2, False)
        ttnn_permute_202 = ttnn.permute(
            ttnn_reshape_255,
            [0, 1, 3, 2],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_255, False)
        ttnn_reshape_256 = ttnn.reshape(
            ttnn_permute_202,
            [131072, 128],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_202, False)
        ttnn_matmul_3 = ttnn.matmul(
            ttnn_reshape_256,
            var_10,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_reshape_256, False)
        ttnn_reshape_257 = ttnn.reshape(
            ttnn_matmul_3,
            [1, 512, 256, 256],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_matmul_3, False)
        ttnn_permute_203 = ttnn.permute(
            ttnn_reshape_257,
            [0, 2, 3, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_257, False)
        ttnn_reshape_258 = ttnn.reshape(
            ttnn_permute_203,
            [1, 1, 65536, 512],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_203, False)
        ttnn_conv2d_18 = ttnn.conv2d(
            input_tensor=ttnn_reshape_258,
            weight_tensor=self.ce_cache["main_const_eval_21"][0],
            device=device,
            in_channels=512,
            out_channels=512,
            batch_size=1,
            input_height=256,
            input_width=256,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=[1, 1],
            groups=1,
            bias_tensor=self.ce_cache["main_const_eval_3"][0],
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=ttnn.DataType.BFLOAT16,
                deallocate_activation=True,
                config_tensors_in_dram=True,
                act_block_h_override=1024,
                enable_kernel_stride_folding=False,
            ),
            compute_config=None,
            slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dDRAMSliceWidth, num_slices=0),
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_reshape_258, False)
        ttnn_conv2d_19 = ttnn.conv2d(
            input_tensor=ttnn_conv2d_18,
            weight_tensor=self.ce_cache["main_const_eval_19"][0],
            device=device,
            in_channels=512,
            out_channels=256,
            batch_size=1,
            input_height=256,
            input_width=256,
            kernel_size=[1, 1],
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=[1, 1],
            groups=1,
            bias_tensor=self.ce_cache["main_const_eval_38"][0],
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=ttnn.DataType.BFLOAT16,
                deallocate_activation=True,
                config_tensors_in_dram=True,
                act_block_h_override=0,
                enable_kernel_stride_folding=False,
            ),
            compute_config=None,
            slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_typecast_105 = ttnn.typecast(
            ttnn_conv2d_18,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_conv2d_18, False)
        ttnn_reshape_259 = ttnn.reshape(
            ttnn_typecast_105,
            [1, 256, 256, 512],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_105, False)
        ttnn_permute_204 = ttnn.permute(
            ttnn_reshape_259,
            [0, 3, 1, 2],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_259, False)
        ttnn_reshape_260 = ttnn.reshape(
            ttnn_permute_204,
            [1, 32, 16, 65536],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_204, False)
        ttnn_sum_34 = ttnn.sum(
            ttnn_reshape_260,
            [2, 3],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=None,
        )
        ttnn_multiply_85 = ttnn.multiply(
            ttnn_sum_34,
            var_11,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sum_34, False)
        ttnn_subtract_17 = ttnn.subtract(
            ttnn_reshape_260,
            ttnn_multiply_85,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_85, False)
        ttnn.deallocate(ttnn_reshape_260, False)
        ttnn_multiply_86 = ttnn.multiply(
            ttnn_subtract_17,
            ttnn_subtract_17,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_sum_35 = ttnn.sum(
            ttnn_multiply_86,
            [2, 3],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_multiply_86, False)
        ttnn_multiply_87 = ttnn.multiply(
            ttnn_sum_35,
            var_11,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sum_35, False)
        ttnn_add_43 = ttnn.add(
            ttnn_multiply_87,
            var_5,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_87, False)
        ttnn_rsqrt_17 = ttnn.rsqrt(
            ttnn_add_43,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_43, False)
        ttnn_multiply_88 = ttnn.multiply(
            ttnn_subtract_17,
            ttnn_rsqrt_17,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_rsqrt_17, False)
        ttnn.deallocate(ttnn_subtract_17, False)
        ttnn_multiply_89 = ttnn.multiply(
            ttnn_multiply_88,
            self.ce_cache["main_const_eval_70"][0],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_88, False)
        ttnn_add_44 = ttnn.add(
            ttnn_multiply_89,
            self.ce_cache["main_const_eval_90"][0],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_89, False)
        ttnn_silu_16 = ttnn.silu(
            ttnn_add_44,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_44, False)
        ttnn_typecast_106 = ttnn.typecast(
            ttnn_silu_16,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_silu_16, False)
        ttnn_reshape_261 = ttnn.reshape(
            ttnn_typecast_106,
            [1, 512, 256, 256],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_106, False)
        ttnn_permute_205 = ttnn.permute(
            ttnn_reshape_261,
            [0, 2, 3, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_261, False)
        ttnn_reshape_262 = ttnn.reshape(
            ttnn_permute_205,
            [1, 1, 65536, 512],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_205, False)
        ttnn_conv2d_20 = ttnn.conv2d(
            input_tensor=ttnn_reshape_262,
            weight_tensor=self.ce_cache["main_const_eval_35"][0],
            device=device,
            in_channels=512,
            out_channels=256,
            batch_size=1,
            input_height=256,
            input_width=256,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=[1, 1],
            groups=1,
            bias_tensor=self.ce_cache["main_const_eval_25"][0],
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=ttnn.DataType.BFLOAT16,
                deallocate_activation=True,
                config_tensors_in_dram=True,
                act_block_h_override=1024,
                enable_kernel_stride_folding=False,
            ),
            compute_config=None,
            slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dDRAMSliceWidth, num_slices=0),
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_reshape_262, False)
        ttnn_typecast_107 = ttnn.typecast(
            ttnn_conv2d_20,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_conv2d_20, False)
        ttnn_reshape_263 = ttnn.reshape(
            ttnn_typecast_107,
            [1, 256, 256, 256],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_107, False)
        ttnn_permute_206 = ttnn.permute(
            ttnn_reshape_263,
            [0, 3, 1, 2],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_263, False)
        ttnn_reshape_264 = ttnn.reshape(
            ttnn_permute_206,
            [1, 32, 8, 65536],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_206, False)
        ttnn_sum_36 = ttnn.sum(
            ttnn_reshape_264,
            [2, 3],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=None,
        )
        ttnn_multiply_90 = ttnn.multiply(
            ttnn_sum_36,
            var_1,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sum_36, False)
        ttnn_subtract_18 = ttnn.subtract(
            ttnn_reshape_264,
            ttnn_multiply_90,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_90, False)
        ttnn.deallocate(ttnn_reshape_264, False)
        ttnn_multiply_91 = ttnn.multiply(
            ttnn_subtract_18,
            ttnn_subtract_18,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_sum_37 = ttnn.sum(
            ttnn_multiply_91,
            [2, 3],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_multiply_91, False)
        ttnn_multiply_92 = ttnn.multiply(
            ttnn_sum_37,
            var_1,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sum_37, False)
        ttnn_add_45 = ttnn.add(
            ttnn_multiply_92,
            var_5,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_92, False)
        ttnn_rsqrt_18 = ttnn.rsqrt(
            ttnn_add_45,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_45, False)
        ttnn_multiply_93 = ttnn.multiply(
            ttnn_subtract_18,
            ttnn_rsqrt_18,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_rsqrt_18, False)
        ttnn.deallocate(ttnn_subtract_18, False)
        ttnn_multiply_94 = ttnn.multiply(
            ttnn_multiply_93,
            self.ce_cache["main_const_eval_61"][0],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_93, False)
        ttnn_add_46 = ttnn.add(
            ttnn_multiply_94,
            self.ce_cache["main_const_eval_135"][0],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_94, False)
        ttnn_silu_17 = ttnn.silu(
            ttnn_add_46,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_46, False)
        ttnn_typecast_108 = ttnn.typecast(
            ttnn_silu_17,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_silu_17, False)
        ttnn_reshape_265 = ttnn.reshape(
            ttnn_typecast_108,
            [1, 256, 256, 256],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_108, False)
        ttnn_permute_207 = ttnn.permute(
            ttnn_reshape_265,
            [0, 2, 3, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_265, False)
        ttnn_reshape_266 = ttnn.reshape(
            ttnn_permute_207,
            [1, 1, 65536, 256],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_207, False)
        ttnn_conv2d_21 = ttnn.conv2d(
            input_tensor=ttnn_reshape_266,
            weight_tensor=self.ce_cache["main_const_eval_84"][0],
            device=device,
            in_channels=256,
            out_channels=256,
            batch_size=1,
            input_height=256,
            input_width=256,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=[1, 1],
            groups=1,
            bias_tensor=self.ce_cache["main_const_eval_76"][0],
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=ttnn.DataType.BFLOAT16,
                deallocate_activation=True,
                config_tensors_in_dram=True,
                act_block_h_override=1024,
                enable_kernel_stride_folding=False,
            ),
            compute_config=None,
            slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_reshape_266, False)
        ttnn_add_47 = ttnn.add(
            ttnn_conv2d_19,
            ttnn_conv2d_21,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_conv2d_21, False)
        ttnn.deallocate(ttnn_conv2d_19, False)
        ttnn_reshape_267 = ttnn.reshape(
            ttnn_add_47,
            [1, 256, 256, 256],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_47, False)
        ttnn_permute_208 = ttnn.permute(
            ttnn_reshape_267,
            [0, 3, 1, 2],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_267, False)
        ttnn_divide_9 = ttnn.divide(
            ttnn_permute_208,
            var_3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_208, False)
        ttnn_typecast_109 = ttnn.typecast(
            ttnn_divide_9,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_reshape_268 = ttnn.reshape(
            ttnn_typecast_109,
            [1, 32, 8, 65536],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_109, False)
        ttnn_sum_38 = ttnn.sum(
            ttnn_reshape_268,
            [2, 3],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=None,
        )
        ttnn_multiply_95 = ttnn.multiply(
            ttnn_sum_38,
            var_1,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sum_38, False)
        ttnn_subtract_19 = ttnn.subtract(
            ttnn_reshape_268,
            ttnn_multiply_95,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_95, False)
        ttnn.deallocate(ttnn_reshape_268, False)
        ttnn_multiply_96 = ttnn.multiply(
            ttnn_subtract_19,
            ttnn_subtract_19,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_sum_39 = ttnn.sum(
            ttnn_multiply_96,
            [2, 3],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_multiply_96, False)
        ttnn_multiply_97 = ttnn.multiply(
            ttnn_sum_39,
            var_1,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sum_39, False)
        ttnn_add_48 = ttnn.add(
            ttnn_multiply_97,
            var_5,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_97, False)
        ttnn_rsqrt_19 = ttnn.rsqrt(
            ttnn_add_48,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_48, False)
        ttnn_multiply_98 = ttnn.multiply(
            ttnn_subtract_19,
            ttnn_rsqrt_19,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_rsqrt_19, False)
        ttnn.deallocate(ttnn_subtract_19, False)
        ttnn_multiply_99 = ttnn.multiply(
            ttnn_multiply_98,
            self.ce_cache["main_const_eval_96"][0],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_98, False)
        ttnn_add_49 = ttnn.add(
            ttnn_multiply_99,
            self.ce_cache["main_const_eval_44"][0],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_99, False)
        ttnn_silu_18 = ttnn.silu(
            ttnn_add_49,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_49, False)
        ttnn_typecast_110 = ttnn.typecast(
            ttnn_silu_18,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_silu_18, False)
        ttnn_reshape_269 = ttnn.reshape(
            ttnn_typecast_110,
            [1, 256, 256, 256],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_110, False)
        ttnn_permute_209 = ttnn.permute(
            ttnn_reshape_269,
            [0, 2, 3, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_269, False)
        ttnn_reshape_270 = ttnn.reshape(
            ttnn_permute_209,
            [1, 1, 65536, 256],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_209, False)
        ttnn_conv2d_22 = ttnn.conv2d(
            input_tensor=ttnn_reshape_270,
            weight_tensor=self.ce_cache["main_const_eval_9"][0],
            device=device,
            in_channels=256,
            out_channels=256,
            batch_size=1,
            input_height=256,
            input_width=256,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=[1, 1],
            groups=1,
            bias_tensor=self.ce_cache["main_const_eval_64"][0],
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=ttnn.DataType.BFLOAT16,
                deallocate_activation=True,
                config_tensors_in_dram=True,
                act_block_h_override=1024,
                enable_kernel_stride_folding=False,
            ),
            compute_config=None,
            slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_reshape_270, False)
        ttnn_typecast_111 = ttnn.typecast(
            ttnn_conv2d_22,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_conv2d_22, False)
        ttnn_reshape_271 = ttnn.reshape(
            ttnn_typecast_111,
            [1, 256, 256, 256],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_111, False)
        ttnn_permute_210 = ttnn.permute(
            ttnn_reshape_271,
            [0, 3, 1, 2],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_271, False)
        ttnn_reshape_272 = ttnn.reshape(
            ttnn_permute_210,
            [1, 32, 8, 65536],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_210, False)
        ttnn_sum_40 = ttnn.sum(
            ttnn_reshape_272,
            [2, 3],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=None,
        )
        ttnn_multiply_100 = ttnn.multiply(
            ttnn_sum_40,
            var_1,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sum_40, False)
        ttnn_subtract_20 = ttnn.subtract(
            ttnn_reshape_272,
            ttnn_multiply_100,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_100, False)
        ttnn.deallocate(ttnn_reshape_272, False)
        ttnn_multiply_101 = ttnn.multiply(
            ttnn_subtract_20,
            ttnn_subtract_20,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_sum_41 = ttnn.sum(
            ttnn_multiply_101,
            [2, 3],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_multiply_101, False)
        ttnn_multiply_102 = ttnn.multiply(
            ttnn_sum_41,
            var_1,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sum_41, False)
        ttnn_add_50 = ttnn.add(
            ttnn_multiply_102,
            var_5,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_102, False)
        ttnn_rsqrt_20 = ttnn.rsqrt(
            ttnn_add_50,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_50, False)
        ttnn_multiply_103 = ttnn.multiply(
            ttnn_subtract_20,
            ttnn_rsqrt_20,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_rsqrt_20, False)
        ttnn.deallocate(ttnn_subtract_20, False)
        ttnn_multiply_104 = ttnn.multiply(
            ttnn_multiply_103,
            self.ce_cache["main_const_eval_112"][0],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_103, False)
        ttnn_add_51 = ttnn.add(
            ttnn_multiply_104,
            self.ce_cache["main_const_eval_50"][0],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_104, False)
        ttnn_silu_19 = ttnn.silu(
            ttnn_add_51,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_51, False)
        ttnn_typecast_112 = ttnn.typecast(
            ttnn_silu_19,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_silu_19, False)
        ttnn_reshape_273 = ttnn.reshape(
            ttnn_typecast_112,
            [1, 256, 256, 256],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_112, False)
        ttnn_permute_211 = ttnn.permute(
            ttnn_reshape_273,
            [0, 2, 3, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_273, False)
        ttnn_reshape_274 = ttnn.reshape(
            ttnn_permute_211,
            [1, 1, 65536, 256],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_211, False)
        ttnn_conv2d_23 = ttnn.conv2d(
            input_tensor=ttnn_reshape_274,
            weight_tensor=self.ce_cache["main_const_eval_99"][0],
            device=device,
            in_channels=256,
            out_channels=256,
            batch_size=1,
            input_height=256,
            input_width=256,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=[1, 1],
            groups=1,
            bias_tensor=self.ce_cache["main_const_eval_100"][0],
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=ttnn.DataType.BFLOAT16,
                deallocate_activation=True,
                config_tensors_in_dram=True,
                act_block_h_override=1024,
                enable_kernel_stride_folding=False,
            ),
            compute_config=None,
            slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_reshape_274, False)
        ttnn_reshape_275 = ttnn.reshape(
            ttnn_conv2d_23,
            [1, 256, 256, 256],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_conv2d_23, False)
        ttnn_permute_212 = ttnn.permute(
            ttnn_reshape_275,
            [0, 3, 1, 2],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_275, False)
        ttnn_add_52 = ttnn.add(
            ttnn_divide_9,
            ttnn_permute_212,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_212, False)
        ttnn.deallocate(ttnn_divide_9, False)
        ttnn_divide_10 = ttnn.divide(
            ttnn_add_52,
            var_3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_52, False)
        ttnn_typecast_113 = ttnn.typecast(
            ttnn_divide_10,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_reshape_276 = ttnn.reshape(
            ttnn_typecast_113,
            [1, 32, 8, 65536],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_113, False)
        ttnn_sum_42 = ttnn.sum(
            ttnn_reshape_276,
            [2, 3],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=None,
        )
        ttnn_multiply_105 = ttnn.multiply(
            ttnn_sum_42,
            var_1,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sum_42, False)
        ttnn_subtract_21 = ttnn.subtract(
            ttnn_reshape_276,
            ttnn_multiply_105,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_105, False)
        ttnn.deallocate(ttnn_reshape_276, False)
        ttnn_multiply_106 = ttnn.multiply(
            ttnn_subtract_21,
            ttnn_subtract_21,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_sum_43 = ttnn.sum(
            ttnn_multiply_106,
            [2, 3],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_multiply_106, False)
        ttnn_multiply_107 = ttnn.multiply(
            ttnn_sum_43,
            var_1,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sum_43, False)
        ttnn_add_53 = ttnn.add(
            ttnn_multiply_107,
            var_5,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_107, False)
        ttnn_rsqrt_21 = ttnn.rsqrt(
            ttnn_add_53,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_53, False)
        ttnn_multiply_108 = ttnn.multiply(
            ttnn_subtract_21,
            ttnn_rsqrt_21,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_rsqrt_21, False)
        ttnn.deallocate(ttnn_subtract_21, False)
        ttnn_multiply_109 = ttnn.multiply(
            ttnn_multiply_108,
            self.ce_cache["main_const_eval_47"][0],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_108, False)
        ttnn_add_54 = ttnn.add(
            ttnn_multiply_109,
            self.ce_cache["main_const_eval_130"][0],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_109, False)
        ttnn_silu_20 = ttnn.silu(
            ttnn_add_54,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_54, False)
        ttnn_typecast_114 = ttnn.typecast(
            ttnn_silu_20,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_silu_20, False)
        ttnn_reshape_277 = ttnn.reshape(
            ttnn_typecast_114,
            [1, 256, 256, 256],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_114, False)
        ttnn_permute_213 = ttnn.permute(
            ttnn_reshape_277,
            [0, 2, 3, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_277, False)
        ttnn_reshape_278 = ttnn.reshape(
            ttnn_permute_213,
            [1, 1, 65536, 256],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_213, False)
        ttnn_conv2d_24 = ttnn.conv2d(
            input_tensor=ttnn_reshape_278,
            weight_tensor=self.ce_cache["main_const_eval_68"][0],
            device=device,
            in_channels=256,
            out_channels=256,
            batch_size=1,
            input_height=256,
            input_width=256,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=[1, 1],
            groups=1,
            bias_tensor=self.ce_cache["main_const_eval_125"][0],
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=ttnn.DataType.BFLOAT16,
                deallocate_activation=True,
                config_tensors_in_dram=True,
                act_block_h_override=1024,
                enable_kernel_stride_folding=False,
            ),
            compute_config=None,
            slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_reshape_278, False)
        ttnn_typecast_115 = ttnn.typecast(
            ttnn_conv2d_24,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_conv2d_24, False)
        ttnn_reshape_279 = ttnn.reshape(
            ttnn_typecast_115,
            [1, 256, 256, 256],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_115, False)
        ttnn_permute_214 = ttnn.permute(
            ttnn_reshape_279,
            [0, 3, 1, 2],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_279, False)
        ttnn_reshape_280 = ttnn.reshape(
            ttnn_permute_214,
            [1, 32, 8, 65536],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_214, False)
        ttnn_sum_44 = ttnn.sum(
            ttnn_reshape_280,
            [2, 3],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=None,
        )
        ttnn_multiply_110 = ttnn.multiply(
            ttnn_sum_44,
            var_1,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sum_44, False)
        ttnn_subtract_22 = ttnn.subtract(
            ttnn_reshape_280,
            ttnn_multiply_110,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_110, False)
        ttnn.deallocate(ttnn_reshape_280, False)
        ttnn_multiply_111 = ttnn.multiply(
            ttnn_subtract_22,
            ttnn_subtract_22,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_sum_45 = ttnn.sum(
            ttnn_multiply_111,
            [2, 3],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_multiply_111, False)
        ttnn_multiply_112 = ttnn.multiply(
            ttnn_sum_45,
            var_1,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sum_45, False)
        ttnn_add_55 = ttnn.add(
            ttnn_multiply_112,
            var_5,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_112, False)
        ttnn_rsqrt_22 = ttnn.rsqrt(
            ttnn_add_55,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_55, False)
        ttnn_multiply_113 = ttnn.multiply(
            ttnn_subtract_22,
            ttnn_rsqrt_22,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_rsqrt_22, False)
        ttnn.deallocate(ttnn_subtract_22, False)
        ttnn_multiply_114 = ttnn.multiply(
            ttnn_multiply_113,
            self.ce_cache["main_const_eval_39"][0],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_113, False)
        ttnn_add_56 = ttnn.add(
            ttnn_multiply_114,
            self.ce_cache["main_const_eval_89"][0],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_114, False)
        ttnn_silu_21 = ttnn.silu(
            ttnn_add_56,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_56, False)
        ttnn_typecast_116 = ttnn.typecast(
            ttnn_silu_21,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_silu_21, False)
        ttnn_reshape_281 = ttnn.reshape(
            ttnn_typecast_116,
            [1, 256, 256, 256],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_116, False)
        ttnn_permute_215 = ttnn.permute(
            ttnn_reshape_281,
            [0, 2, 3, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_281, False)
        ttnn_reshape_282 = ttnn.reshape(
            ttnn_permute_215,
            [1, 1, 65536, 256],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_215, False)
        ttnn_conv2d_25 = ttnn.conv2d(
            input_tensor=ttnn_reshape_282,
            weight_tensor=self.ce_cache["main_const_eval_77"][0],
            device=device,
            in_channels=256,
            out_channels=256,
            batch_size=1,
            input_height=256,
            input_width=256,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=[1, 1],
            groups=1,
            bias_tensor=self.ce_cache["main_const_eval_54"][0],
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=ttnn.DataType.BFLOAT16,
                deallocate_activation=True,
                config_tensors_in_dram=True,
                act_block_h_override=1024,
                enable_kernel_stride_folding=False,
            ),
            compute_config=None,
            slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_reshape_282, False)
        ttnn_reshape_283 = ttnn.reshape(
            ttnn_conv2d_25,
            [1, 256, 256, 256],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_conv2d_25, False)
        ttnn_permute_216 = ttnn.permute(
            ttnn_divide_10,
            [0, 1, 3, 2],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_divide_10, False)
        ttnn_permute_217 = ttnn.permute(
            ttnn_reshape_283,
            [0, 3, 2, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_283, False)
        ttnn_add_57 = ttnn.add(
            ttnn_permute_216,
            ttnn_permute_217,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_217, False)
        ttnn.deallocate(ttnn_permute_216, False)
        ttnn_reshape_284 = ttnn.reshape(
            ttnn_add_57,
            [65536, 256],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_57, False)
        ttnn_divide_11 = ttnn.divide(
            ttnn_reshape_284,
            var_4,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_reshape_284, False)
        ttnn_matmul_4 = ttnn.matmul(
            ttnn_divide_11,
            var_8,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_divide_11, False)
        ttnn_reshape_285 = ttnn.reshape(
            ttnn_matmul_4,
            [1, 256, 256, 512],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_matmul_4, False)
        ttnn_permute_218 = ttnn.permute(
            ttnn_reshape_285,
            [0, 1, 3, 2],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_285, False)
        ttnn_reshape_286 = ttnn.reshape(
            ttnn_permute_218,
            [131072, 256],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_218, False)
        ttnn_matmul_5 = ttnn.matmul(
            ttnn_reshape_286,
            var_8,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_reshape_286, False)
        ttnn_reshape_287 = ttnn.reshape(
            ttnn_matmul_5,
            [1, 256, 512, 512],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_matmul_5, False)
        ttnn_permute_219 = ttnn.permute(
            ttnn_reshape_287,
            [0, 2, 3, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_287, False)
        ttnn_reshape_288 = ttnn.reshape(
            ttnn_permute_219,
            [1, 1, 262144, 256],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_219, False)
        ttnn_conv2d_26 = ttnn.conv2d(
            input_tensor=ttnn_reshape_288,
            weight_tensor=self.ce_cache["main_const_eval_56"][0],
            device=device,
            in_channels=256,
            out_channels=256,
            batch_size=1,
            input_height=512,
            input_width=512,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=[1, 1],
            groups=1,
            bias_tensor=self.ce_cache["main_const_eval_16"][0],
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=ttnn.DataType.BFLOAT16,
                deallocate_activation=True,
                config_tensors_in_dram=True,
                act_block_h_override=1024,
                enable_kernel_stride_folding=False,
            ),
            compute_config=None,
            slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dDRAMSliceWidth, num_slices=0),
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_reshape_288, False)
        ttnn_conv2d_27 = ttnn.conv2d(
            input_tensor=ttnn_conv2d_26,
            weight_tensor=self.ce_cache["main_const_eval_26"][0],
            device=device,
            in_channels=256,
            out_channels=128,
            batch_size=1,
            input_height=512,
            input_width=512,
            kernel_size=[1, 1],
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=[1, 1],
            groups=1,
            bias_tensor=self.ce_cache["main_const_eval_101"][0],
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=ttnn.DataType.BFLOAT16,
                deallocate_activation=True,
                config_tensors_in_dram=True,
                act_block_h_override=0,
                enable_kernel_stride_folding=False,
            ),
            compute_config=None,
            slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_typecast_117 = ttnn.typecast(
            ttnn_conv2d_26,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_conv2d_26, False)
        ttnn_reshape_289 = ttnn.reshape(
            ttnn_typecast_117,
            [1, 512, 512, 256],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_117, False)
        ttnn_permute_220 = ttnn.permute(
            ttnn_reshape_289,
            [0, 3, 1, 2],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_289, False)
        ttnn_reshape_290 = ttnn.reshape(
            ttnn_permute_220,
            [1, 32, 8, 262144],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_220, False)
        ttnn_sum_46 = ttnn.sum(
            ttnn_reshape_290,
            [2, 3],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=None,
        )
        ttnn_multiply_115 = ttnn.multiply(
            ttnn_sum_46,
            var_6,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sum_46, False)
        ttnn_subtract_23 = ttnn.subtract(
            ttnn_reshape_290,
            ttnn_multiply_115,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_115, False)
        ttnn.deallocate(ttnn_reshape_290, False)
        ttnn_multiply_116 = ttnn.multiply(
            ttnn_subtract_23,
            ttnn_subtract_23,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_sum_47 = ttnn.sum(
            ttnn_multiply_116,
            [2, 3],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_multiply_116, False)
        ttnn_multiply_117 = ttnn.multiply(
            ttnn_sum_47,
            var_6,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sum_47, False)
        ttnn_add_58 = ttnn.add(
            ttnn_multiply_117,
            var_5,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_117, False)
        ttnn_rsqrt_23 = ttnn.rsqrt(
            ttnn_add_58,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_58, False)
        ttnn_multiply_118 = ttnn.multiply(
            ttnn_subtract_23,
            ttnn_rsqrt_23,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_rsqrt_23, False)
        ttnn.deallocate(ttnn_subtract_23, False)
        ttnn_multiply_119 = ttnn.multiply(
            ttnn_multiply_118,
            self.ce_cache["main_const_eval_74"][0],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_118, False)
        ttnn_add_59 = ttnn.add(
            ttnn_multiply_119,
            self.ce_cache["main_const_eval_138"][0],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_119, False)
        ttnn_silu_22 = ttnn.silu(
            ttnn_add_59,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_59, False)
        ttnn_typecast_118 = ttnn.typecast(
            ttnn_silu_22,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_silu_22, False)
        ttnn_reshape_291 = ttnn.reshape(
            ttnn_typecast_118,
            [1, 256, 512, 512],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_118, False)
        ttnn_permute_221 = ttnn.permute(
            ttnn_reshape_291,
            [0, 2, 3, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_291, False)
        ttnn_reshape_292 = ttnn.reshape(
            ttnn_permute_221,
            [1, 1, 262144, 256],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_221, False)
        ttnn_conv2d_28 = ttnn.conv2d(
            input_tensor=ttnn_reshape_292,
            weight_tensor=self.ce_cache["main_const_eval_43"][0],
            device=device,
            in_channels=256,
            out_channels=128,
            batch_size=1,
            input_height=512,
            input_width=512,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=[1, 1],
            groups=1,
            bias_tensor=self.ce_cache["main_const_eval_17"][0],
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=ttnn.DataType.BFLOAT16,
                deallocate_activation=True,
                config_tensors_in_dram=True,
                act_block_h_override=1024,
                enable_kernel_stride_folding=False,
            ),
            compute_config=None,
            slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dDRAMSliceWidth, num_slices=0),
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_reshape_292, False)
        ttnn_typecast_119 = ttnn.typecast(
            ttnn_conv2d_28,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_conv2d_28, False)
        ttnn_reshape_293 = ttnn.reshape(
            ttnn_typecast_119,
            [1, 512, 512, 128],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_119, False)
        ttnn_permute_222 = ttnn.permute(
            ttnn_reshape_293,
            [0, 3, 1, 2],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_293, False)
        ttnn_reshape_294 = ttnn.reshape(
            ttnn_permute_222,
            [1, 32, 4, 262144],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_222, False)
        ttnn_sum_48 = ttnn.sum(
            ttnn_reshape_294,
            [2, 3],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=None,
        )
        ttnn_multiply_120 = ttnn.multiply(
            ttnn_sum_48,
            var_11,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sum_48, False)
        ttnn_subtract_24 = ttnn.subtract(
            ttnn_reshape_294,
            ttnn_multiply_120,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_120, False)
        ttnn.deallocate(ttnn_reshape_294, False)
        ttnn_multiply_121 = ttnn.multiply(
            ttnn_subtract_24,
            ttnn_subtract_24,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_sum_49 = ttnn.sum(
            ttnn_multiply_121,
            [2, 3],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_multiply_121, False)
        ttnn_multiply_122 = ttnn.multiply(
            ttnn_sum_49,
            var_11,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sum_49, False)
        ttnn_add_60 = ttnn.add(
            ttnn_multiply_122,
            var_5,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_122, False)
        ttnn_rsqrt_24 = ttnn.rsqrt(
            ttnn_add_60,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_60, False)
        ttnn_multiply_123 = ttnn.multiply(
            ttnn_subtract_24,
            ttnn_rsqrt_24,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_rsqrt_24, False)
        ttnn.deallocate(ttnn_subtract_24, False)
        ttnn_multiply_124 = ttnn.multiply(
            ttnn_multiply_123,
            self.ce_cache["main_const_eval_33"][0],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_123, False)
        ttnn_add_61 = ttnn.add(
            ttnn_multiply_124,
            self.ce_cache["main_const_eval_27"][0],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_124, False)
        ttnn_silu_23 = ttnn.silu(
            ttnn_add_61,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_61, False)
        ttnn_typecast_120 = ttnn.typecast(
            ttnn_silu_23,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_silu_23, False)
        ttnn_reshape_295 = ttnn.reshape(
            ttnn_typecast_120,
            [1, 128, 512, 512],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_120, False)
        ttnn_permute_223 = ttnn.permute(
            ttnn_reshape_295,
            [0, 2, 3, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_295, False)
        ttnn_reshape_296 = ttnn.reshape(
            ttnn_permute_223,
            [1, 1, 262144, 128],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_223, False)
        ttnn_conv2d_29 = ttnn.conv2d(
            input_tensor=ttnn_reshape_296,
            weight_tensor=self.ce_cache["main_const_eval_113"][0],
            device=device,
            in_channels=128,
            out_channels=128,
            batch_size=1,
            input_height=512,
            input_width=512,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=[1, 1],
            groups=1,
            bias_tensor=self.ce_cache["main_const_eval_79"][0],
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=ttnn.DataType.BFLOAT16,
                deallocate_activation=True,
                config_tensors_in_dram=True,
                act_block_h_override=1024,
                enable_kernel_stride_folding=False,
            ),
            compute_config=None,
            slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dDRAMSliceWidth, num_slices=0),
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_reshape_296, False)
        ttnn_add_62 = ttnn.add(
            ttnn_conv2d_27,
            ttnn_conv2d_29,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_conv2d_29, False)
        ttnn.deallocate(ttnn_conv2d_27, False)
        ttnn_reshape_297 = ttnn.reshape(
            ttnn_add_62,
            [1, 512, 512, 128],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_62, False)
        ttnn_permute_224 = ttnn.permute(
            ttnn_reshape_297,
            [0, 3, 1, 2],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_297, False)
        ttnn_reshape_298 = ttnn.reshape(
            ttnn_permute_224,
            [1, 32, 4, 262144],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_224, False)
        ttnn_divide_12 = ttnn.divide(
            ttnn_reshape_298,
            var_3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_reshape_298, False)
        ttnn_typecast_121 = ttnn.typecast(
            ttnn_divide_12,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_sum_50 = ttnn.sum(
            ttnn_typecast_121,
            [2, 3],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=None,
        )
        ttnn_multiply_125 = ttnn.multiply(
            ttnn_sum_50,
            var_11,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sum_50, False)
        ttnn_subtract_25 = ttnn.subtract(
            ttnn_typecast_121,
            ttnn_multiply_125,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_125, False)
        ttnn.deallocate(ttnn_typecast_121, False)
        ttnn_multiply_126 = ttnn.multiply(
            ttnn_subtract_25,
            ttnn_subtract_25,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_sum_51 = ttnn.sum(
            ttnn_multiply_126,
            [2, 3],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_multiply_126, False)
        ttnn_multiply_127 = ttnn.multiply(
            ttnn_sum_51,
            var_11,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sum_51, False)
        ttnn_add_63 = ttnn.add(
            ttnn_multiply_127,
            var_5,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_127, False)
        ttnn_rsqrt_25 = ttnn.rsqrt(
            ttnn_add_63,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_63, False)
        ttnn_multiply_128 = ttnn.multiply(
            ttnn_subtract_25,
            ttnn_rsqrt_25,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_rsqrt_25, False)
        ttnn.deallocate(ttnn_subtract_25, False)
        ttnn_multiply_129 = ttnn.multiply(
            ttnn_multiply_128,
            self.ce_cache["main_const_eval_134"][0],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_128, False)
        ttnn_add_64 = ttnn.add(
            ttnn_multiply_129,
            self.ce_cache["main_const_eval_46"][0],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_129, False)
        ttnn_silu_24 = ttnn.silu(
            ttnn_add_64,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_64, False)
        ttnn_typecast_122 = ttnn.typecast(
            ttnn_silu_24,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_silu_24, False)
        ttnn_reshape_299 = ttnn.reshape(
            ttnn_typecast_122,
            [1, 128, 512, 512],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_122, False)
        ttnn_permute_225 = ttnn.permute(
            ttnn_reshape_299,
            [0, 2, 3, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_299, False)
        ttnn_reshape_300 = ttnn.reshape(
            ttnn_permute_225,
            [1, 1, 262144, 128],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_225, False)
        ttnn_conv2d_30 = ttnn.conv2d(
            input_tensor=ttnn_reshape_300,
            weight_tensor=self.ce_cache["main_const_eval_116"][0],
            device=device,
            in_channels=128,
            out_channels=128,
            batch_size=1,
            input_height=512,
            input_width=512,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=[1, 1],
            groups=1,
            bias_tensor=self.ce_cache["main_const_eval_34"][0],
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=ttnn.DataType.BFLOAT16,
                deallocate_activation=True,
                config_tensors_in_dram=True,
                act_block_h_override=1024,
                enable_kernel_stride_folding=False,
            ),
            compute_config=None,
            slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dDRAMSliceWidth, num_slices=0),
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_reshape_300, False)
        ttnn_typecast_123 = ttnn.typecast(
            ttnn_conv2d_30,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_conv2d_30, False)
        ttnn_reshape_301 = ttnn.reshape(
            ttnn_typecast_123,
            [1, 512, 512, 128],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_123, False)
        ttnn_permute_226 = ttnn.permute(
            ttnn_reshape_301,
            [0, 3, 1, 2],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_301, False)
        ttnn_reshape_302 = ttnn.reshape(
            ttnn_permute_226,
            [1, 32, 4, 262144],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_226, False)
        ttnn_sum_52 = ttnn.sum(
            ttnn_reshape_302,
            [2, 3],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=None,
        )
        ttnn_multiply_130 = ttnn.multiply(
            ttnn_sum_52,
            var_11,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sum_52, False)
        ttnn_subtract_26 = ttnn.subtract(
            ttnn_reshape_302,
            ttnn_multiply_130,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_130, False)
        ttnn.deallocate(ttnn_reshape_302, False)
        ttnn_multiply_131 = ttnn.multiply(
            ttnn_subtract_26,
            ttnn_subtract_26,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_sum_53 = ttnn.sum(
            ttnn_multiply_131,
            [2, 3],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_multiply_131, False)
        ttnn_multiply_132 = ttnn.multiply(
            ttnn_sum_53,
            var_11,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sum_53, False)
        ttnn_add_65 = ttnn.add(
            ttnn_multiply_132,
            var_5,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_132, False)
        ttnn_rsqrt_26 = ttnn.rsqrt(
            ttnn_add_65,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_65, False)
        ttnn_multiply_133 = ttnn.multiply(
            ttnn_subtract_26,
            ttnn_rsqrt_26,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_rsqrt_26, False)
        ttnn.deallocate(ttnn_subtract_26, False)
        ttnn_multiply_134 = ttnn.multiply(
            ttnn_multiply_133,
            self.ce_cache["main_const_eval_28"][0],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_133, False)
        ttnn_add_66 = ttnn.add(
            ttnn_multiply_134,
            self.ce_cache["main_const_eval_118"][0],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_134, False)
        ttnn_silu_25 = ttnn.silu(
            ttnn_add_66,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_66, False)
        ttnn_typecast_124 = ttnn.typecast(
            ttnn_silu_25,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_silu_25, False)
        ttnn_reshape_303 = ttnn.reshape(
            ttnn_typecast_124,
            [1, 128, 512, 512],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_124, False)
        ttnn_permute_227 = ttnn.permute(
            ttnn_reshape_303,
            [0, 2, 3, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_303, False)
        ttnn_reshape_304 = ttnn.reshape(
            ttnn_permute_227,
            [1, 1, 262144, 128],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_227, False)
        ttnn_conv2d_31 = ttnn.conv2d(
            input_tensor=ttnn_reshape_304,
            weight_tensor=self.ce_cache["main_const_eval_57"][0],
            device=device,
            in_channels=128,
            out_channels=128,
            batch_size=1,
            input_height=512,
            input_width=512,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=[1, 1],
            groups=1,
            bias_tensor=self.ce_cache["main_const_eval_71"][0],
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=ttnn.DataType.BFLOAT16,
                deallocate_activation=True,
                config_tensors_in_dram=True,
                act_block_h_override=1024,
                enable_kernel_stride_folding=False,
            ),
            compute_config=None,
            slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dDRAMSliceWidth, num_slices=0),
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_reshape_304, False)
        ttnn_reshape_305 = ttnn.reshape(
            ttnn_conv2d_31,
            [1, 512, 512, 128],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_conv2d_31, False)
        ttnn_permute_228 = ttnn.permute(
            ttnn_reshape_305,
            [0, 3, 1, 2],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_305, False)
        ttnn_reshape_306 = ttnn.reshape(
            ttnn_permute_228,
            [1, 32, 4, 262144],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_228, False)
        ttnn_add_67 = ttnn.add(
            ttnn_divide_12,
            ttnn_reshape_306,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_reshape_306, False)
        ttnn.deallocate(ttnn_divide_12, False)
        ttnn_divide_13 = ttnn.divide(
            ttnn_add_67,
            var_3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_67, False)
        ttnn_typecast_125 = ttnn.typecast(
            ttnn_divide_13,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_sum_54 = ttnn.sum(
            ttnn_typecast_125,
            [2, 3],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=None,
        )
        ttnn_multiply_135 = ttnn.multiply(
            ttnn_sum_54,
            var_11,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sum_54, False)
        ttnn_subtract_27 = ttnn.subtract(
            ttnn_typecast_125,
            ttnn_multiply_135,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_135, False)
        ttnn.deallocate(ttnn_typecast_125, False)
        ttnn_multiply_136 = ttnn.multiply(
            ttnn_subtract_27,
            ttnn_subtract_27,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_sum_55 = ttnn.sum(
            ttnn_multiply_136,
            [2, 3],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_multiply_136, False)
        ttnn_multiply_137 = ttnn.multiply(
            ttnn_sum_55,
            var_11,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sum_55, False)
        ttnn_add_68 = ttnn.add(
            ttnn_multiply_137,
            var_5,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_137, False)
        ttnn_rsqrt_27 = ttnn.rsqrt(
            ttnn_add_68,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_68, False)
        ttnn_multiply_138 = ttnn.multiply(
            ttnn_subtract_27,
            ttnn_rsqrt_27,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_rsqrt_27, False)
        ttnn.deallocate(ttnn_subtract_27, False)
        ttnn_multiply_139 = ttnn.multiply(
            ttnn_multiply_138,
            self.ce_cache["main_const_eval_85"][0],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_138, False)
        ttnn_add_69 = ttnn.add(
            ttnn_multiply_139,
            self.ce_cache["main_const_eval_114"][0],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_139, False)
        ttnn_silu_26 = ttnn.silu(
            ttnn_add_69,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_69, False)
        ttnn_typecast_126 = ttnn.typecast(
            ttnn_silu_26,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_silu_26, False)
        ttnn_reshape_307 = ttnn.reshape(
            ttnn_typecast_126,
            [1, 128, 512, 512],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_126, False)
        ttnn_permute_229 = ttnn.permute(
            ttnn_reshape_307,
            [0, 2, 3, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_307, False)
        ttnn_reshape_308 = ttnn.reshape(
            ttnn_permute_229,
            [1, 1, 262144, 128],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_229, False)
        ttnn_conv2d_32 = ttnn.conv2d(
            input_tensor=ttnn_reshape_308,
            weight_tensor=self.ce_cache["main_const_eval_20"][0],
            device=device,
            in_channels=128,
            out_channels=128,
            batch_size=1,
            input_height=512,
            input_width=512,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=[1, 1],
            groups=1,
            bias_tensor=self.ce_cache["main_const_eval_1"][0],
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=ttnn.DataType.BFLOAT16,
                deallocate_activation=True,
                config_tensors_in_dram=True,
                act_block_h_override=1024,
                enable_kernel_stride_folding=False,
            ),
            compute_config=None,
            slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dDRAMSliceWidth, num_slices=0),
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_reshape_308, False)
        ttnn_typecast_127 = ttnn.typecast(
            ttnn_conv2d_32,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_conv2d_32, False)
        ttnn_reshape_309 = ttnn.reshape(
            ttnn_typecast_127,
            [1, 512, 512, 128],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_127, False)
        ttnn_permute_230 = ttnn.permute(
            ttnn_reshape_309,
            [0, 3, 1, 2],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_309, False)
        ttnn_reshape_310 = ttnn.reshape(
            ttnn_permute_230,
            [1, 32, 4, 262144],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_230, False)
        ttnn_sum_56 = ttnn.sum(
            ttnn_reshape_310,
            [2, 3],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=None,
        )
        ttnn_multiply_140 = ttnn.multiply(
            ttnn_sum_56,
            var_11,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sum_56, False)
        ttnn_subtract_28 = ttnn.subtract(
            ttnn_reshape_310,
            ttnn_multiply_140,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_140, False)
        ttnn.deallocate(ttnn_reshape_310, False)
        ttnn_multiply_141 = ttnn.multiply(
            ttnn_subtract_28,
            ttnn_subtract_28,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_sum_57 = ttnn.sum(
            ttnn_multiply_141,
            [2, 3],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_multiply_141, False)
        ttnn_multiply_142 = ttnn.multiply(
            ttnn_sum_57,
            var_11,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sum_57, False)
        ttnn_add_70 = ttnn.add(
            ttnn_multiply_142,
            var_5,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_142, False)
        ttnn_rsqrt_28 = ttnn.rsqrt(
            ttnn_add_70,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_70, False)
        ttnn_multiply_143 = ttnn.multiply(
            ttnn_subtract_28,
            ttnn_rsqrt_28,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_rsqrt_28, False)
        ttnn.deallocate(ttnn_subtract_28, False)
        ttnn_multiply_144 = ttnn.multiply(
            ttnn_multiply_143,
            self.ce_cache["main_const_eval_121"][0],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_143, False)
        ttnn_add_71 = ttnn.add(
            ttnn_multiply_144,
            self.ce_cache["main_const_eval_69"][0],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_144, False)
        ttnn_silu_27 = ttnn.silu(
            ttnn_add_71,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_71, False)
        ttnn_typecast_128 = ttnn.typecast(
            ttnn_silu_27,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_silu_27, False)
        ttnn_reshape_311 = ttnn.reshape(
            ttnn_typecast_128,
            [1, 128, 512, 512],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_128, False)
        ttnn_permute_231 = ttnn.permute(
            ttnn_reshape_311,
            [0, 2, 3, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_311, False)
        ttnn_reshape_312 = ttnn.reshape(
            ttnn_permute_231,
            [1, 1, 262144, 128],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_231, False)
        ttnn_conv2d_33 = ttnn.conv2d(
            input_tensor=ttnn_reshape_312,
            weight_tensor=self.ce_cache["main_const_eval_129"][0],
            device=device,
            in_channels=128,
            out_channels=128,
            batch_size=1,
            input_height=512,
            input_width=512,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=[1, 1],
            groups=1,
            bias_tensor=self.ce_cache["main_const_eval_87"][0],
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=ttnn.DataType.BFLOAT16,
                deallocate_activation=True,
                config_tensors_in_dram=True,
                act_block_h_override=1024,
                enable_kernel_stride_folding=False,
            ),
            compute_config=None,
            slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dDRAMSliceWidth, num_slices=0),
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_reshape_312, False)
        ttnn_reshape_313 = ttnn.reshape(
            ttnn_conv2d_33,
            [1, 512, 512, 128],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_conv2d_33, False)
        ttnn_permute_232 = ttnn.permute(
            ttnn_reshape_313,
            [0, 3, 1, 2],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_313, False)
        ttnn_reshape_314 = ttnn.reshape(
            ttnn_permute_232,
            [1, 32, 4, 262144],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_232, False)
        ttnn_add_72 = ttnn.add(
            ttnn_divide_13,
            ttnn_reshape_314,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_reshape_314, False)
        ttnn.deallocate(ttnn_divide_13, False)
        ttnn_divide_14 = ttnn.divide(
            ttnn_add_72,
            var_3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_72, False)
        ttnn_typecast_129 = ttnn.typecast(
            ttnn_divide_14,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_divide_14, False)
        ttnn_sum_58 = ttnn.sum(
            ttnn_typecast_129,
            [2, 3],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=None,
        )
        ttnn_multiply_145 = ttnn.multiply(
            ttnn_sum_58,
            var_11,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sum_58, False)
        ttnn_subtract_29 = ttnn.subtract(
            ttnn_typecast_129,
            ttnn_multiply_145,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_145, False)
        ttnn.deallocate(ttnn_typecast_129, False)
        ttnn_multiply_146 = ttnn.multiply(
            ttnn_subtract_29,
            ttnn_subtract_29,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_sum_59 = ttnn.sum(
            ttnn_multiply_146,
            [2, 3],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_multiply_146, False)
        ttnn_multiply_147 = ttnn.multiply(
            ttnn_sum_59,
            var_11,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sum_59, False)
        ttnn_add_73 = ttnn.add(
            ttnn_multiply_147,
            var_5,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_147, False)
        ttnn_rsqrt_29 = ttnn.rsqrt(
            ttnn_add_73,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_73, False)
        ttnn_multiply_148 = ttnn.multiply(
            ttnn_subtract_29,
            ttnn_rsqrt_29,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_rsqrt_29, False)
        ttnn.deallocate(ttnn_subtract_29, False)
        ttnn_multiply_149 = ttnn.multiply(
            ttnn_multiply_148,
            self.ce_cache["main_const_eval_92"][0],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_148, False)
        ttnn_add_74 = ttnn.add(
            ttnn_multiply_149,
            self.ce_cache["main_const_eval_109"][0],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_149, False)
        ttnn_silu_28 = ttnn.silu(
            ttnn_add_74,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_74, False)
        ttnn_typecast_130 = ttnn.typecast(
            ttnn_silu_28,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_silu_28, False)
        ttnn_reshape_315 = ttnn.reshape(
            ttnn_typecast_130,
            [1, 128, 512, 512],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_130, False)
        ttnn_permute_233 = ttnn.permute(
            ttnn_reshape_315,
            [0, 2, 3, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_315, False)
        ttnn_reshape_316 = ttnn.reshape(
            ttnn_permute_233,
            [1, 1, 262144, 128],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_233, False)
        ttnn_conv2d_34 = ttnn.conv2d(
            input_tensor=ttnn_reshape_316,
            weight_tensor=self.ce_cache["main_const_eval_15"][0],
            device=device,
            in_channels=128,
            out_channels=3,
            batch_size=1,
            input_height=512,
            input_width=512,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=[1, 1],
            groups=1,
            bias_tensor=self.ce_cache["main_const_eval_8"][0],
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=ttnn.DataType.BFLOAT16,
                deallocate_activation=True,
                config_tensors_in_dram=True,
                act_block_h_override=192,
                enable_kernel_stride_folding=False,
            ),
            compute_config=None,
            slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_reshape_316, False)
        ttnn_reshape_317 = ttnn.reshape(
            ttnn_conv2d_34,
            [1, 512, 512, 3],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_conv2d_34, False)
        ttnn_permute_234 = ttnn.permute(
            ttnn_reshape_317,
            [0, 3, 1, 2],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_317, False)
        out = ttnn.to_torch(
            ttnn.from_device(ttnn_permute_234),
            mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0),
        )
        return out[: out.shape[0] // 4].float()

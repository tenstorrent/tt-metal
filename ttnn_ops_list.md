# TTNN Operations — Full List

Sources: [docs/source/ttnn/ttnn/api.rst](docs/source/ttnn/ttnn/api.rst), C++ nanobind registrations under [ttnn/cpp/ttnn/operations/](ttnn/cpp/ttnn/operations/), and Python registrations in [ttnn/ttnn/operations/](ttnn/ttnn/operations/). Moreh ops excluded.

## Device & Tensor Lifecycle
`open_device`, `close_device`, `manage_device`, `synchronize_device`, `SetDefaultDevice`, `GetDefaultDevice`, `pad_to_tile_shape`, `allocate_tensor_on_device`, `allocate_tensor_on_host`, `as_tensor`, `copy_device_to_host_tensor`, `copy_host_to_device_tensor`, `copy_host_to_device_tensor_partial`, `deallocate`, `dump_tensor`, `from_device`, `from_torch`, `get_device_tensors`, `get_optimal_worker_cores_for_sharded_tensor`, `load_tensor`, `reallocate`, `split_work_to_cores`, `to_device`, `to_dtype`, `to_layout`, `to_memory_config`, `to_torch`, `typecast`, `create_sharded_memory_config`

## Tensor Creation
`arange`, `bernoulli`, `complex_tensor`, `empty`, `empty_like`, `from_buffer`, `full`, `full_like`, `index_fill`, `ones`, `ones_like`, `rand`, `randn`, `uniform`, `zeros`, `zeros_like`

## Matrix Multiplication
`matmul`, `linear`, `addmm`, `sparse_matmul`, `matmul_batched_weights`

## Pointwise Unary
`abs`, `acos`, `acosh`, `alt_complex_rotate90`, `angle`, `asin`, `asinh`, `atan`, `atanh`, `bitcast`, `bitwise_left_shift`, `bitwise_not`, `bitwise_right_shift`, `cbrt`, `ceil`, `celu`, `clamp`, `clip`, `clone`, `conj`, `cos`, `cosh`, `deg2rad`, `digamma`, `eqz`, `erf`, `erfc`, `erfinv`, `exp`, `exp2`, `experimental.dropout`, `elu`, `expm1`, `fill`, `floor`, `frac`, `geglu`, `gelu`, `gez`, `glu`, `gtz`, `hardmish`, `hardshrink`, `hardsigmoid`, `hardswish`, `hardtanh`, `heaviside`, `i0`, `i1`, `identity`, `imag`, `is_imag`, `is_real`, `isfinite`, `isinf`, `isnan`, `isneginf`, `isposinf`, `leaky_relu`, `lez`, `lgamma`, `log`, `log10`, `log1p`, `log2`, `log_sigmoid`, `logical_left_shift`, `logical_not`, `logical_not_`, `logical_right_shift`, `logit`, `ltz`, `mish`, `multigammaln`, `neg`, `nez`, `normalize_global`, `normalize_hw`, `polar`, `polygamma`, `prelu`, `rad2deg`, `rdiv`, `real`, `reciprocal`, `reglu`, `relu`, `relu6`, `relu_max`, `relu_min`, `remainder`, `round`, `rsqrt`, `selu`, `sigmoid`, `sigmoid_accurate`, `sign`, `signbit`, `silu`, `sin`, `sinh`, `softplus`, `softshrink`, `softsign`, `sqrt`, `square`, `std_hw`, `swiglu`, `swish`, `tan`, `tanh`, `tanhshrink`, `threshold`, `tril`, `triu`, `trunc`, `unary_chain`, `var_hw`, `xielu`

## Pointwise Binary
`add`, `add_`, `addalpha`, `atan2`, `bias_gelu`, `bias_gelu_`, `bitwise_and`, `bitwise_or`, `bitwise_xor`, `div`, `div_no_nan`, `divide`, `divide_`, `eq`, `eq_`, `floor_div`, `fmod`, `gcd`, `ge`, `ge_`, `gt`, `gt_`, `hypot`, `isclose`, `lcm`, `ldexp`, `ldexp_`, `le`, `le_`, `logaddexp`, `logaddexp2`, `logaddexp2_`, `logaddexp_`, `logical_and`, `logical_and_`, `logical_or`, `logical_or_`, `logical_xor`, `logical_xor_`, `lt`, `lt_`, `maximum`, `minimum`, `multiply`, `multiply_`, `ne`, `ne_`, `nextafter`, `outer`, `polyval`, `pow`, `remainder`, `rpow`, `rsub`, `rsub_`, `squared_difference`, `squared_difference_`, `subalpha`, `subtract`, `subtract_`, `xlogy`

## Pointwise Ternary
`addcdiv`, `addcmul`, `lerp`, `mac`, `where`

## Quantization
`dequantize`, `quantize`, `requantize`

## Losses
`l1_loss`, `mse_loss`

## Reduction
`argmax`, `cumprod`, `cumsum`, `ema`, `manual_seed`, `max`, `mean`, `min`, `moe`, `prod`, `sampling`, `std`, `sum`, `topk`, `var`

## Data Movement
`assign`, `bcast`, `chunk`, `concat`, `copy`, `expand`, `fill_implicit_tile_padding`, `fill_ones_rm`, `fill_rm`, `fold`, `gather`, `indexed_fill`, `interleaved_to_sharded`, `interleaved_to_sharded_partial`, `moe_expert_token_remap`, `moe_routing_remap`, `move`, `narrow`, `nonzero`, `pad`, `permute`, `repeat`, `repeat_interleave`, `reshape`, `reshape_on_device`, `reshard`, `roll`, `scatter`, `scatter_add`, `sharded_to_interleaved`, `sharded_to_interleaved_partial`, `slice`, `sort`, `split`, `squeeze`, `stack`, `tilize`, `tilize_with_val_padding`, `tilize_with_zero_padding`, `tosa_gather`, `tosa_scatter`, `transpose`, `unsqueeze`, `unsqueeze_to_4D`, `untilize`, `untilize_with_unpadding`, `view`

## Normalization
`batch_norm`, `group_norm`, `layer_norm`, `layer_norm_post_all_gather`, `layer_norm_pre_all_gather`, `rms_norm`, `rms_norm_post_all_gather`, `rms_norm_pre_all_gather`, `scale_causal_mask_hw_dims_softmax_in_place`, `scale_mask_softmax`, `scale_mask_softmax_in_place`, `softmax`, `softmax_in_place`, `fused_rms_minimal`

## Transformer
`transformer.attention_softmax`, `transformer.attention_softmax_`, `transformer.chunked_flash_mla_prefill`, `transformer.chunked_scaled_dot_product_attention`, `transformer.concatenate_heads`, `transformer.exp_ring_joint_scaled_dot_product_attention`, `transformer.flash_mla_prefill`, `transformer.flash_multi_latent_attention_decode`, `transformer.joint_scaled_dot_product_attention`, `transformer.paged_flash_multi_latent_attention_decode`, `transformer.paged_scaled_dot_product_attention_decode`, `transformer.ring_distributed_scaled_dot_product_attention`, `transformer.ring_joint_scaled_dot_product_attention`, `transformer.scaled_dot_product_attention`, `transformer.scaled_dot_product_attention_decode`, `transformer.split_query_key_value_and_split_heads`, `transformer.windowed_scaled_dot_product_attention`

## KV Cache
`kv_cache.fill_cache_for_user_`, `kv_cache.update_cache_for_token_`, `kv_cache.zero_cache_range`, `fill_cache`, `update_cache`

## CCL (Collective Communication)
`all_broadcast`, `all_gather`, `all_reduce`, `all_to_all_combine`, `all_to_all_dispatch`, `broadcast`, `mesh_partition`, `point_to_point`, `reduce_scatter`, `reduce_to_root`

## Convolution
`conv1d`, `conv2d`, `conv_transpose2d`, `experimental.conv3d`, `prepare_conv_bias`, `prepare_conv_transpose2d_bias`, `prepare_conv_transpose2d_weights`, `prepare_conv_weights`

## Pooling
`adaptive_avg_pool2d`, `adaptive_max_pool2d`, `avg_pool2d`, `global_avg_pool2d`, `max_pool2d`

## Vision
`grid_sample`, `upsample`, `rotate`

## Embedding
`embedding`

## Prefetcher
`dram_prefetcher`

## Generic / Examples / Test
`generic_op`, `composite_example`, `composite_example_multiple_return`, `plus_one`, `test_hang_device_operation`

## Utility / Comparison
`pearson_correlation_coefficient`

## Backward — Unary
`abs_bw`, `acos_bw`, `acosh_bw`, `asin_bw`, `asinh_bw`, `atan_bw`, `atanh_bw`, `ceil_bw`, `celu_bw`, `clamp_bw`, `clip_bw`, `cos_bw`, `cosh_bw`, `deg2rad_bw`, `digamma_bw`, `div_no_nan_bw`, `elu_bw`, `erf_bw`, `erfc_bw`, `erfinv_bw`, `exp2_bw`, `exp_bw`, `experimental.gelu_bw`, `expm1_bw`, `fill_bw`, `fill_zero_bw`, `floor_bw`, `frac_bw`, `gelu_bw`, `hardshrink_bw`, `hardsigmoid_bw`, `hardswish_bw`, `hardtanh_bw`, `i0_bw`, `leaky_relu_bw`, `lgamma_bw`, `log10_bw`, `log1p_bw`, `log2_bw`, `log_bw`, `log_sigmoid_bw`, `logit_bw`, `logiteps_bw`, `multigammaln_bw`, `neg_bw`, `polygamma_bw`, `pow_bw`, `prod_bw`, `rad2deg_bw`, `rdiv_bw`, `reciprocal_bw`, `relu6_bw`, `relu_bw`, `repeat_bw`, `round_bw`, `rpow_bw`, `rsqrt_bw`, `selu_bw`, `sigmoid_bw`, `sign_bw`, `silu_bw`, `sin_bw`, `sinh_bw`, `softplus_bw`, `softshrink_bw`, `softsign_bw`, `sqrt_bw`, `square_bw`, `tan_bw`, `tanh_bw`, `tanhshrink_bw`, `threshold_bw`, `trunc_bw`

## Backward — Binary
`add_bw`, `addalpha_bw`, `assign_bw`, `atan2_bw`, `bias_gelu_bw`, `concat_bw`, `div_bw`, `fmod_bw`, `hypot_bw`, `ldexp_bw`, `logaddexp2_bw`, `logaddexp_bw`, `max_bw`, `min_bw`, `mul_bw`, `remainder_bw`, `rsub_bw`, `squared_difference_bw`, `sub_bw`, `subalpha_bw`, `xlogy_bw`

## Backward — Ternary
`addcdiv_bw`, `addcmul_bw`, `lerp_bw`, `where_bw`

## Backward — Complex / Embedding
`angle_bw`, `conj_bw`, `embedding_bw`, `imag_bw`, `polar_bw`, `real_bw`

## Hooks / Reports / Model Conversion
`set_printoptions`, `register_pre_operation_hook`, `register_post_operation_hook`, `model_preprocessing.preprocess_model`, `model_preprocessing.preprocess_model_parameters`

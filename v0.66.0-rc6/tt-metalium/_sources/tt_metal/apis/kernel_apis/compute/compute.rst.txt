Compute APIs
============

This page lists the APIs available in the compute kernels. These APIs cover:

* Synchronization between the three cores running the compute kernel cooperatively
* Copying data between SRAM and the registers used by the compute engines
* Performing computations

For details on function parameters and their meanings, see the :ref:`Compute Engines and Data Flow within Tensix<compute_engines_and_dataflow_within_tensix>` documentation.

Synchronization
---------------
.. toctree::

  acquire_dst
  release_dst
  reg_api

Compute
-------

Initialization
~~~~~~~~~~~~~~

.. toctree::

  init_functions
  binary_op_init_funcs

Compute (FPU/matrix engine)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::

  add_tiles
  sub_tiles
  mul_tiles
  add_tiles_bcast
  sub_tiles_bcast
  mul_tiles_bcast

  matmul_tiles
  matmul_block
  reduce_tile
  transpose_wh_tile

  tilize
  untilize

.. only:: not html

Compute (SFPU/vector engine)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Basic arithmetic operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::

  add_binary_tile
  sub_binary_tile
  mul_binary_tile
  div_binary_tile
  rsub_tile
  rsub_binary_tile
  power_tile
  power_binary_tile
  square_tile
  sqrt_tile
  rsqrt_tile
  recip_tile
  abs_tile
  negative_tile
  binop_with_scalar_tile
  tiled_prod_tile

Integer operations
^^^^^^^^^^^^^^^^^^

.. toctree::

  add_int_tile
  sub_int_tile
  mul_int_tile
  rsub_int_tile
  gcd_tile
  lcm_tile
  remainder_tile
  fmod_tile

Exponential and logarithmic functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::

  exp_tile
  exp2_tile
  expm1_tile
  log_tile
  log1p_tile
  log_with_base_tile
  xlogy_binary_tile

Comparison and logical operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::

  unary_eq_tile
  unary_ne_tile
  unary_gt_tile
  unary_ge_tile
  unary_lt_tile
  unary_le_tile
  unary_max_tile
  unary_min_tile
  max_tile
  binary_max_tile
  binary_min_tile
  ltz_tile
  lez_tile
  gtz_tile
  gez_tile
  eqz_tile
  nez_tile
  logical_not_unary_tile
  isinf_tile
  isnan_tile
  heaviside_tile

Bitwise operations
^^^^^^^^^^^^^^^^^^

.. toctree::

  bitwise_and_tile
  bitwise_or_tile
  bitwise_xor_tile
  bitwise_not_tile
  bitwise_and_binary_tile
  bitwise_or_binary_tile
  bitwise_xor_binary_tile
  left_shift_tile
  right_shift_tile
  binary_shift_tile

Rounding and ceiling functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::

  round_tile
  clamp_tile
  threshold_tile
  sign_tile
  signbit_tile

Hyperbolic functions
^^^^^^^^^^^^^^^^^^^^

.. toctree::

  asinh_tile
  atanh_tile

Special mathematical functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::

  i0_tile
  i1_tile
  cumsum_tile
  alt_complex_rotate90_tile

Initialization and utilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::

  compute_kernel_hw_startup

Trigonometric functions
^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::

  erf_tile
  erfc_tile
  erfinv_tile

  tanh_tile
  tan_tile
  sin_tile
  cos_tile
  asin_tile
  atan_tile
  acos_tile
  acosh_tile
  selu_tile

Activation functions
^^^^^^^^^^^^^^^^^^^^

.. toctree::

  relu_tile
  elu_tile
  gelu_tile
  sigmoid_tile
  celu_tile
  silu_tile
  prelu_tile
  softplus_tile
  softsign_tile
  hardsigmoid_tile
  hardtanh_tile

Data manipulation and processing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::

  fill_tile
  identity_tile
  mask_tile
  where_tile
  addcmul_tile
  reshuffle_rows_tile
  typecast_tile
  dropout_tile
  rand_tile

Quantization operations
^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::

  quant_tile
  requant_tile
  dequant_tile

import ttnn


class DeviceGetter:
    _instance = None
    _mesh_shape = None
    l1_small_size = 1 << 15

    def __init__(self):
        raise RuntimeError("This is Singleton, invoke get_device() instead.")

    @classmethod
    def get_device(cls, mesh_shape):
        if cls._instance == None:
            if (
                not isinstance(mesh_shape, (list, tuple))
                or len(mesh_shape) == 0
                or not all(isinstance(x, int) and x > 0 for x in mesh_shape)
            ):
                raise ValueError(f"mesh_shape must be a non-empty list or tuple of positive integers, got {mesh_shape}")
            cls._mesh_shape = mesh_shape
            cls._instance = ttnn.open_mesh_device(
                mesh_shape=ttnn.MeshShape(mesh_shape),
                l1_small_size=cls.l1_small_size,
            )
            print(f"Device: {cls._instance}")

        # Compare requested mesh_shape with _mesh_shape used to initialize the device
        if tuple(cls._mesh_shape) != tuple(mesh_shape):
            raise ValueError(
                f"Device already initialized with mesh_shape={cls._mesh_shape}, but got mesh_shape={mesh_shape}"
            )

        return cls._instance


def test_reshape_pattern(v1):
    v2 = v1[0]
    v3 = v1[1]
    v4 = DeviceGetter.get_device((1, 1))
    v5 = ttnn.silu(v2, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
    ttnn.deallocate(v2, False)
    v6 = ttnn.permute(
        v5,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    print(f"v6: {v6.shape}")
    ttnn.deallocate(v5, False)
    v7 = ttnn.global_avg_pool2d(
        input_tensor=v6,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=ttnn.DataType.BFLOAT16,
    )
    print(f"v7: {v7.shape}")
    ttnn.deallocate(v6, False)
    v8 = ttnn.reshape(
        v7,
        [1, 1, 12, 144],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v7, False)
    v9 = ttnn.prepare_conv_weights(
        weight_tensor=v3,
        input_memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=144,
        out_channels=6,
        batch_size=12,
        input_height=1,
        input_width=1,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=False,
        groups=1,
        device=v4,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.DataType.BFLOAT16),
        compute_config=None,
        slice_config=None,
    )
    ttnn.deallocate(v3, False)
    v10 = ttnn.conv2d(
        input_tensor=v8,
        weight_tensor=v9,
        device=v4,
        in_channels=144,
        out_channels=6,
        batch_size=12,
        input_height=1,
        input_width=1,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=None,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.DataType.BFLOAT16),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v9, False)
    ttnn.deallocate(v8, False)
    v11 = ttnn.to_memory_config(v10, ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
    ttnn.deallocate(v10, False)
    v12 = ttnn.reshape(
        v11,
        [12, 1, 1, 6],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v11, False)
    v13 = ttnn.permute(
        v12,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(v12, False)
    v14 = [v13]
    return v14


def create_inputs_for_test_reshape_pattern():
    v1 = DeviceGetter.get_device((1, 1))
    v2 = ttnn.ones(
        shape=ttnn.Shape([12, 144, 56, 56]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=v1
    )
    v3 = ttnn.ones(
        shape=ttnn.Shape([6, 144, 1, 1]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=None
    )
    v4 = [v2, v3]
    return v4


def main():
    v1 = create_inputs_for_test_reshape_pattern()
    v2 = test_reshape_pattern(v1)
    v3 = 0
    return v3


if __name__ == "__main__":
    main()

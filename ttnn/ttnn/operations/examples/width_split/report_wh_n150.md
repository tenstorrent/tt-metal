# width_split reduce case — WH B0 n150 sweeps

Measured on `bgd-lab-22-special-adamjanovic-for-reservation-169096` · WH B0 n150 · 8×8 = 64 Tensix cores · AICLK 1000 MHz · 2026-11-17 · commit `c2079cdc7ed` (combine fenced with a write barrier). `DEVICE KERNEL DURATION [ns]`, mean of 20 launches per cell after warm-up; one sweep run each.

Reproduce:

```bash
WS_SWEEPS=1 WSR_ROWS=1,4,16,64,128 WSR_WTS=4,8,16,32,64,128,256 scripts/run_safe_pytest.sh \
  tests/ttnn/unit_tests/operations/examples/test_width_split.py::test_width_split_reduce_device_perf
WS_SWEEPS=1 scripts/run_safe_pytest.sh \
  tests/ttnn/unit_tests/operations/examples/test_width_split.py::test_width_split_reduce_rm_device_perf
```

## Layout::TILE

```
=== width_split reduce case (y = x - mean_W(x)) — row_split_reduce vs width_split_reduce ===
    dtype=bfloat16 Layout::TILE, TensorMemoryLayout::INTERLEAVED DRAM in/out  grid=8x8=64 Tensix cores  arch=Arch.WORMHOLE_B0  iters=1  trials=20
    cores = active Tensix cores; tiles/slice = tiles per Tensix core per tile-row
    DRAM GB/s = (input + output bytes) / time = 2 * R_t * Wt * 2048 B / ns
     R_t   Wt  variant                k  cores  tiles/slice       ns/op    GB/s   vs k=1
       1    4  row_split_reduce       1      1            4      4056.5     4.0  (base)
       1    4  width_split_reduce     2      2            2      4259.8     3.8   0.95x
       1    4  width_split_reduce     4      4            1      4426.9     3.7   0.92x
       1    8  row_split_reduce       1      1            8      5485.9     6.0  (base)
       1    8  width_split_reduce     2      2            4      4876.9     6.7   1.12x
       1    8  width_split_reduce     4      4            2      4691.9     7.0   1.17x
       1    8  width_split_reduce     8      8            1      6070.4     5.4   0.90x
       1   16  row_split_reduce       1      1           16      8390.2     7.8  (base)
       1   16  width_split_reduce     2      2            8      6363.1    10.3   1.32x
       1   16  width_split_reduce     4      4            4      5308.1    12.3   1.58x
       1   16  width_split_reduce     8      8            2      6355.9    10.3   1.32x
       1   16  width_split_reduce    16     16            1      9769.7     6.7   0.86x
       1   32  row_split_reduce       1      1           32     14261.9     9.2  (base)
       1   32  width_split_reduce     2      2           16      9362.6    14.0   1.52x
       1   32  width_split_reduce     4      4            8      6798.8    19.3   2.10x
       1   32  width_split_reduce     8      8            4      7081.6    18.5   2.01x
       1   32  width_split_reduce    16     16            2     10065.2    13.0   1.42x
       1   32  width_split_reduce    32     32            1     17397.8     7.5   0.82x
       1   64  row_split_reduce       1      1           64     25696.0    10.2  (base)
       1   64  width_split_reduce     2      2           32     15127.7    17.3   1.70x
       1   64  width_split_reduce     4      4           16      9832.5    26.7   2.61x
       1   64  width_split_reduce     8      8            8      8673.9    30.2   2.96x
       1   64  width_split_reduce    16     16            4     10987.8    23.9   2.34x
       1   64  width_split_reduce    32     32            2     18106.8    14.5   1.42x
       1   64  width_split_reduce    64     64            1     35861.6     7.3   0.72x
       1  128  row_split_reduce       1      1          128     48920.1    10.7  (base)
       1  128  width_split_reduce     2      2           64     26823.9    19.5   1.82x
       1  128  width_split_reduce     4      4           32     15911.9    32.9   3.07x
       1  128  width_split_reduce     8      8           16     11671.0    44.9   4.19x
       1  128  width_split_reduce    16     16            8     12716.0    41.2   3.85x
       1  128  width_split_reduce    32     32            4     19229.8    27.3   2.54x
       1  128  width_split_reduce    64     64            2     36411.7    14.4   1.34x
       4    4  row_split_reduce       1      4            4      4300.4    15.2  (base)
       4    4  width_split_reduce     2      8            2      4493.2    14.6   0.96x
       4    4  width_split_reduce     4     16            1      4846.7    13.5   0.89x
       4    8  row_split_reduce       1      4            8      5708.1    23.0  (base)
       4    8  width_split_reduce     2      8            4      5300.1    24.7   1.08x
       4    8  width_split_reduce     4     16            2      5304.9    24.7   1.08x
       4    8  width_split_reduce     8     32            1      6288.0    20.8   0.91x
       4   16  row_split_reduce       1      4           16      8774.4    29.9  (base)
       4   16  width_split_reduce     2      8            8      6834.2    38.4   1.28x
       4   16  width_split_reduce     4     16            4      6454.9    40.6   1.36x
       4   16  width_split_reduce     8     32            2      6973.5    37.6   1.26x
       4   16  width_split_reduce    16     64            1     11251.6    23.3   0.78x
       4   32  row_split_reduce       1      4           32     15221.9    34.4  (base)
       4   32  width_split_reduce     2      8           16      9967.0    52.6   1.53x
       4   32  width_split_reduce     4     16            8      8331.6    62.9   1.83x
       4   32  width_split_reduce     8     32            4      8795.6    59.6   1.73x
       4   32  width_split_reduce    16     64            2     12144.6    43.2   1.25x
       4   32  width_split_reduce    32     64            1     36234.7    14.5   0.42x
       4   64  row_split_reduce       1      4           64     26890.2    39.0  (base)
       4   64  width_split_reduce     2      8           32     16080.8    65.2   1.67x
       4   64  width_split_reduce     4     16           16     11881.0    88.3   2.26x
       4   64  width_split_reduce     8     32            8     12027.1    87.2   2.24x
       4   64  width_split_reduce    16     64            4     15270.2    68.7   1.76x
       4   64  width_split_reduce    32     64            2     37231.4    28.2   0.72x
       4  128  row_split_reduce       1      4          128     50120.2    41.8  (base)
       4  128  width_split_reduce     2      8           64     28195.5    74.4   1.78x
       4  128  width_split_reduce     4     16           32     19117.0   109.7   2.62x
       4  128  width_split_reduce     8     32           16     17631.2   118.9   2.84x
       4  128  width_split_reduce    16     64            8     20687.3   101.4   2.42x
       4  128  width_split_reduce    32     64            4     40844.7    51.3   1.23x
      16    4  row_split_reduce       1     16            4      5046.5    51.9  (base)
      16    4  width_split_reduce     2     32            2      5233.1    50.1   0.96x
      16    4  width_split_reduce     4     64            1      5438.1    48.2   0.93x
      16    8  row_split_reduce       1     16            8      6806.9    77.0  (base)
      16    8  width_split_reduce     2     32            4      7142.8    73.4   0.95x
      16    8  width_split_reduce     4     64            2      6947.6    75.5   0.98x
      16    8  width_split_reduce     8     64            1     10121.0    51.8   0.67x
      16   16  row_split_reduce       1     16           16     10681.4    98.2  (base)
      16   16  width_split_reduce     2     32            8     10230.8   102.5   1.04x
      16   16  width_split_reduce     4     64            4     10853.1    96.6   0.98x
      16   16  width_split_reduce     8     64            2     11918.5    88.0   0.90x
      16   32  row_split_reduce       1     16           32     17752.0   118.1  (base)
      16   32  width_split_reduce     2     32           16     16221.4   129.3   1.09x
      16   32  width_split_reduce     4     64            8     16896.7   124.1   1.05x
      16   32  width_split_reduce     8     64            4     17238.2   121.7   1.03x
      16   64  row_split_reduce       1     16           64     32075.6   130.8  (base)
      16   64  width_split_reduce     2     32           32     28396.8   147.7   1.13x
      16   64  width_split_reduce     4     64           16     28817.0   145.5   1.11x
      16   64  width_split_reduce     8     64            8     27929.5   150.2   1.15x
      16  128  row_split_reduce       1     16          128     61038.7   137.4  (base)
      16  128  width_split_reduce     2     32           64     53224.9   157.6   1.15x
      16  128  width_split_reduce     4     64           32     51471.4   163.0   1.19x
      16  128  width_split_reduce     8     64           16     49510.8   169.4   1.23x
      64    4  row_split_reduce       1     64            4      8734.6   120.0  (base)
      64    4  width_split_reduce     2     64            2      8722.6   120.2   1.00x
      64    8  row_split_reduce       1     64            8     15131.5   138.6  (base)
      64    8  width_split_reduce     2     64            4     14482.5   144.8   1.04x
      64   16  row_split_reduce       1     64           16     27369.4   153.2  (base)
      64   16  width_split_reduce     2     64            8     26096.0   160.7   1.05x
      64   32  row_split_reduce       1     64           32     51545.5   162.7  (base)
      64   32  width_split_reduce     2     64           16     49359.9   169.9   1.04x
      64   64  row_split_reduce       1     64           64     99940.2   167.9  (base)
      64   64  width_split_reduce     2     64           32     94406.9   177.7   1.06x
      64  128  row_split_reduce       1     64          128    196090.7   171.1  (base)
      64  128  width_split_reduce     2     64           64    185006.0   181.4   1.06x
     128    4  row_split_reduce       1     64            4     14009.5   149.7  (base)
     128    8  row_split_reduce       1     64            8     26323.1   159.3  (base)
     128   16  row_split_reduce       1     64           16     48332.8   173.6  (base)
     128   32  row_split_reduce       1     64           32     94971.8   176.7  (base)
     128   64  row_split_reduce       1     64           64    189229.5   177.3  (base)
     128  128  row_split_reduce       1     64          128    369563.4   181.6  (base)
```

Missing cells (R_t = 128, Wt = 256, wide rows at R_t ≥ 16, and k beyond the grid) are the ones `_sweep_cells` drops: it keeps a cell only when R_t ≤ the grid, R_t · k ≤ 2 × the grid, and its circular buffers fit L1.

## Layout::ROW_MAJOR

```
=== width_split reduce case on ROW_MAJOR sticks (y = x - mean_W(x)) — row_split_reduce vs width_split_reduce ===
    dtype=bfloat16 Layout::ROW_MAJOR, TensorMemoryLayout::INTERLEAVED DRAM in/out  grid=8x8=64 Tensix cores  arch=Arch.WORMHOLE_B0  iters=1  trials=20
    cores = active Tensix cores
    stick B = W * 2; piece B = stick B / k (bytes per DRAM transfer); transfers = 2 * 32 * R_t * k (read + write)
    DRAM GB/s = 2 * R_t * 32 * stick B / ns;  Mxfer/s = transfers / ns * 1000
     R_t      W  stick B  variant               k  cores  piece B   xfers       ns/op    GB/s  Mxfer/s   vs k=1
       1     64      128  row_split_reduce      1      1      128      64      7417.4     1.1        9  (base)
       1     64      128  width_split_reduce    2      2       64     128      7956.8     1.0       16   0.93x
       1    128      256  row_split_reduce      1      1      256      64      8090.1     2.0        8  (base)
       1    128      256  width_split_reduce    2      2      128     128      8307.3     2.0       15   0.97x
       1    128      256  width_split_reduce    4      4       64     256      8309.4     2.0       31   0.97x
       1    256      512  row_split_reduce      1      1      512      64      9485.3     3.5        7  (base)
       1    256      512  width_split_reduce    2      2      256     128      8971.2     3.7       14   1.06x
       1    256      512  width_split_reduce    4      4      128     256      8771.0     3.7       29   1.08x
       1    512     1024  row_split_reduce      1      1     1024      64     11894.0     5.5        5  (base)
       1    512     1024  width_split_reduce    2      2      512     128     10399.6     6.3       12   1.14x
       1    512     1024  width_split_reduce    4      4      256     256      9364.2     7.0       27   1.27x
       1   1024     2048  row_split_reduce      1      1     2048      64     17650.8     7.4        4  (base)
       1   1024     2048  width_split_reduce    2      2     1024     128     12784.5    10.3       10   1.38x
       1   1024     2048  width_split_reduce    4      4      512     256     10638.4    12.3       24   1.66x
       1   2048     4096  row_split_reduce      1      1     4096      64     31616.2     8.3        2  (base)
       1   2048     4096  width_split_reduce    2      2     2048     128     18564.7    14.1        7   1.70x
       1   2048     4096  width_split_reduce    4      4     1024     256     13222.4    19.8       19   2.39x
       4     64      128  row_split_reduce      1      4      128     256      7562.9     4.3       34  (base)
       4     64      128  width_split_reduce    2      8       64     512      8078.5     4.1       63   0.94x
       4    128      256  row_split_reduce      1      4      256     256      8243.7     7.9       31  (base)
       4    128      256  width_split_reduce    2      8      128     512      8426.0     7.8       61   0.98x
       4    128      256  width_split_reduce    4     16       64    1024      8743.4     7.5      117   0.94x
       4    256      512  row_split_reduce      1      4      512     256      9601.0    13.7       27  (base)
       4    256      512  width_split_reduce    2      8      256     512      9082.1    14.4       56   1.06x
       4    256      512  width_split_reduce    4     16      128    1024      9163.2    14.3      112   1.05x
       4    512     1024  row_split_reduce      1      4     1024     256     12057.0    21.7       21  (base)
       4    512     1024  width_split_reduce    2      8      512     512     10476.7    25.0       49   1.15x
       4    512     1024  width_split_reduce    4     16      256    1024      9807.2    26.7      104   1.23x
       4   1024     2048  row_split_reduce      1      4     2048     256     18278.3    28.7       14  (base)
       4   1024     2048  width_split_reduce    2      8     1024     512     12989.0    40.4       39   1.41x
       4   1024     2048  width_split_reduce    4     16      512    1024     11135.8    47.1       92   1.64x
       4   2048     4096  row_split_reduce      1      4     4096     256     32587.4    32.2        8  (base)
       4   2048     4096  width_split_reduce    2      8     2048     512     20368.4    51.5       25   1.60x
       4   2048     4096  width_split_reduce    4     16     1024    1024     15674.8    66.9       65   2.08x
      16     64      128  row_split_reduce      1     16      128    1024      7643.7    17.1      134  (base)
      16     64      128  width_split_reduce    2     32       64    2048      8135.0    16.1      252   0.94x
      16    128      256  row_split_reduce      1     16      256    1024      8324.1    31.5      123  (base)
      16    128      256  width_split_reduce    2     32      128    2048      8470.2    30.9      242   0.98x
      16    128      256  width_split_reduce    4     64       64    4096      8776.8    29.9      467   0.95x
      16    256      512  row_split_reduce      1     16      512    1024      9790.0    53.6      105  (base)
      16    256      512  width_split_reduce    2     32      256    2048      9218.2    56.9      222   1.06x
      16    256      512  width_split_reduce    4     64      128    4096      9284.0    56.5      441   1.05x
      16    512     1024  row_split_reduce      1     16     1024    1024     14194.7    73.9       72  (base)
      16    512     1024  width_split_reduce    2     32      512    2048     12440.6    84.3      165   1.14x
      16    512     1024  width_split_reduce    4     64      256    4096     12029.8    87.2      340   1.18x
      16   1024     2048  row_split_reduce      1     16     2048    1024     24225.7    86.6       42  (base)
      16   1024     2048  width_split_reduce    2     32     1024    2048     19922.2   105.3      103   1.22x
      16   1024     2048  width_split_reduce    4     64      512    4096     17513.7   119.7      234   1.38x
      16   2048     4096  row_split_reduce      1     16     4096    1024     45733.2    91.7       22  (base)
      16   2048     4096  width_split_reduce    2     32     2048    2048     36192.2   115.9       57   1.26x
      16   2048     4096  width_split_reduce    4     64     1024    4096     29401.2   142.7      139   1.56x
      47     64      128  row_split_reduce      1     47      128    3008      7625.6    50.5      394  (base)
      47     64      128  width_split_reduce    2     64       64    6016     11802.5    32.6      510   0.65x
      47    128      256  row_split_reduce      1     47      256    3008      8882.2    86.7      339  (base)
      47    128      256  width_split_reduce    2     64      128    6016     12335.7    62.4      488   0.72x
      47    256      512  row_split_reduce      1     47      512    3008     13313.6   115.7      226  (base)
      47    256      512  width_split_reduce    2     64      256    6016     15181.3   101.4      396   0.88x
      47    512     1024  row_split_reduce      1     47     1024    3008     21795.0   141.3      138  (base)
      47    512     1024  width_split_reduce    2     64      512    6016     21302.8   144.6      282   1.02x
      47   1024     2048  row_split_reduce      1     47     2048    3008     39751.4   155.0       76  (base)
      47   1024     2048  width_split_reduce    2     64     1024    6016     35948.4   171.4      167   1.11x
      47   2048     4096  row_split_reduce      1     47     4096    3008     78631.1   156.7       38  (base)
      47   2048     4096  width_split_reduce    2     64     2048    6016     73205.6   168.3       82   1.07x
      64     64      128  row_split_reduce      1     64      128    4096      7796.5    67.2      525  (base)
      64     64      128  width_split_reduce    2     64       64    8192     11815.2    44.4      693   0.66x
      64    128      256  row_split_reduce      1     64      256    4096     10716.7    97.8      382  (base)
      64    128      256  width_split_reduce    2     64      128    8192     12428.2    84.4      659   0.86x
      64    256      512  row_split_reduce      1     64      512    4096     16117.4   130.1      254  (base)
      64    256      512  width_split_reduce    2     64      256    8192     16776.3   125.0      488   0.96x
      64    512     1024  row_split_reduce      1     64     1024    4096     27346.0   153.4      150  (base)
      64    512     1024  width_split_reduce    2     64      512    8192     25503.5   164.5      321   1.07x
      64   1024     2048  row_split_reduce      1     64     2048    4096     52439.6   160.0       78  (base)
      64   1024     2048  width_split_reduce    2     64     1024    8192     46529.3   180.3      176   1.13x
      64   2048     4096  row_split_reduce      1     64     4096    4096    106876.8   157.0       38  (base)
      64   2048     4096  width_split_reduce    2     64     2048    8192     94625.4   177.3       87   1.13x
     128     64      128  row_split_reduce      1     64      128    8192     10136.0   103.5      808  (base)
     128    128      256  row_split_reduce      1     64      256    8192     14376.7   145.9      570  (base)
     128    256      512  row_split_reduce      1     64      512    8192     25388.8   165.2      323  (base)
     128    512     1024  row_split_reduce      1     64     1024    8192     47689.4   175.9      172  (base)
     128   1024     2048  row_split_reduce      1     64     2048    8192     94763.6   177.0       86  (base)
     128   2048     4096  row_split_reduce      1     64     4096    8192    196666.0   170.6       42  (base)
```

At R_t ≥ 64 a k = 2 split cannot add Tensix cores on this 64-core grid (64 → 64), so those rows compare the same core count with twice the transfers.

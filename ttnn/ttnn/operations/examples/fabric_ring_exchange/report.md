# fabric_ring_exchange — device report

Box: qb2-120-p02t03 (Blackhole QuietBox 2, 4× p300c), 1×4 ring, FABRIC_2D_TORUS_XY, router payload 14400 B.
Measured 2026-10-23 with `test_fabric_ring_exchange.py` (RX_IDLE=0,100, defaults otherwise: 7 untraced and 8 traced calls per case).
Columns, µs: untraced last-launched / untraced slowest / traced slowest [traced range].

```
links=1 idle=  0 empty  slot=      -     0.02 /    0.04 /    0.03 [0.02-0.04]
links=1 idle=  0 fence  slot=      -     4.31 /    6.90 /    5.16 [4.83-5.81]
links=1 idle=  0 relay  slot=   128B     4.45 /    6.50 /    5.38 [5.14-5.90]
links=1 idle=  0 direct slot=   128B     6.05 /    6.57 /    6.65 [6.31-6.90]
links=1 idle=100 empty  slot=      -     0.23 /    0.26 /    0.19 [0.18-0.26]
links=1 idle=100 fence  slot=      -     4.58 /    7.95 /    5.04 [4.96-6.41]
links=1 idle=100 relay  slot=   128B     4.71 /    7.38 /    5.38 [5.21-6.13]
links=1 idle=100 direct slot=   128B     6.35 /    9.15 /    6.48 [6.40-6.59]
links=1 idle=  0 relay  slot=  2560B     4.57 /    7.90 /    5.64 [5.50-6.22]
links=1 idle=  0 direct slot=  2560B     6.23 /    8.26 /    6.80 [6.49-6.91]
links=1 idle=100 relay  slot=  2560B     4.83 /    9.04 /    5.52 [5.41-6.47]
links=1 idle=100 direct slot=  2560B     6.53 /    9.78 /    6.74 [6.61-6.99]
links=1 idle=  0 relay  slot=  8192B     4.97 /    7.96 /    5.98 [5.78-6.59]
links=1 idle=  0 direct slot=  8192B     6.78 /    8.99 /    7.25 [7.20-7.63]
links=1 idle=100 relay  slot=  8192B     5.17 /    8.38 /    6.06 [5.86-6.89]
links=1 idle=100 direct slot=  8192B     7.05 /    9.29 /    7.34 [7.27-7.70]
links=1 idle=  0 relay  slot= 32768B     6.09 /    9.48 /    7.33 [7.18-8.05]
links=1 idle=  0 direct slot= 32768B     8.06 /    9.81 /    9.00 [8.58-9.10]
links=1 idle=100 relay  slot= 32768B     6.38 /   10.86 /    7.60 [7.17-8.17]
links=1 idle=100 direct slot= 32768B     8.37 /   10.74 /    8.71 [8.64-8.87]
links=2 idle=  0 empty  slot=      -     0.04 /    0.07 /    0.05 [0.04-0.06]
links=2 idle=  0 fence  slot=      -     4.27 /    7.55 /    5.02 [4.76-5.75]
links=2 idle=  0 relay  slot=   128B     4.44 /    7.94 /    5.48 [5.34-6.84]
links=2 idle=  0 direct slot=   128B     6.01 /    7.90 /    6.67 [6.44-6.93]
links=2 idle=100 empty  slot=      -     0.24 /    0.27 /    0.19 [0.18-0.30]
links=2 idle=100 fence  slot=      -     4.53 /    7.98 /    4.98 [4.92-5.79]
links=2 idle=100 relay  slot=   128B     4.67 /    8.43 /    5.39 [5.21-6.32]
links=2 idle=100 direct slot=   128B     6.32 /    9.16 /    6.78 [6.48-6.92]
links=2 idle=  0 relay  slot=  2560B     4.53 /    7.69 /    5.74 [5.41-6.36]
links=2 idle=  0 direct slot=  2560B     6.29 /    8.33 /    7.00 [6.83-7.12]
links=2 idle=100 relay  slot=  2560B     4.85 /    7.97 /    5.90 [5.58-6.67]
links=2 idle=100 direct slot=  2560B     6.54 /    8.50 /    6.86 [6.70-7.48]
links=2 idle=  0 relay  slot=  8192B     4.98 /    8.63 /    6.32 [6.06-6.64]
links=2 idle=  0 direct slot=  8192B     7.06 /    8.93 /    7.46 [7.38-7.68]
links=2 idle=100 relay  slot=  8192B     5.22 /    8.77 /    6.52 [6.22-7.13]
links=2 idle=100 direct slot=  8192B     7.22 /    8.34 /    7.39 [7.30-7.77]
links=2 idle=  0 relay  slot= 32768B     6.40 /    9.99 /    8.42 [7.96-8.70]
links=2 idle=  0 direct slot= 32768B     9.26 /   10.12 /    9.71 [9.36-9.94]
links=2 idle=100 relay  slot= 32768B     6.68 /   10.44 /    8.54 [8.20-8.88]
links=2 idle=100 direct slot= 32768B     9.19 /   11.67 /    9.85 [9.68-10.27]
```

fence vs fence_direct (same session, 1 link, 100 idle cores, 9 untraced / 12 traced calls):

```
fence         4.53 /  7.66 / 5.09 [4.88-6.02]
fence_direct  6.11 /  9.01 / 6.35 [6.21-6.45]
relay 128B    4.78 /  8.70 / 5.36 [5.13-6.31]
direct 128B   6.33 /  8.31 / 6.58 [6.44-6.70]
```

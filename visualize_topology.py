#!/usr/bin/env python3
"""Visualize the physical topology to understand connectivity"""

# Physical connections
connections = [(0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (6, 7)]

print("\n🔍 Your T3K Physical Topology Visualization")
print("=" * 60)
print("\nConnectivity Map:")
print()
print("    0 ←→ 1 ←→ 2 ←→ 3")
print("    ↕         ↕    ↕    ↕")
print("    4 ←→ 5         6 ←→ 7")
print()
print("Notice: Chips 4,5 are NOT connected to chips 6,7!")
print("        There's a gap in the bottom row.")
print()

print("\n📊 Degree of Each Chip (number of connections):")
degree = {}
for a, b in connections:
    degree[a] = degree.get(a, 0) + 1
    degree[b] = degree.get(b, 0) + 1

for chip in range(8):
    d = degree.get(chip, 0)
    bar = "█" * d
    print(f"  Chip {chip}: {bar} ({d} connections)")

print("\n❌ Why 2×4 Doesn't Work:")
print("-" * 60)
print("For a 2×4 mesh, you need:")
print("  Row 0: A ←→ B ←→ C ←→ D")
print("         ↕    ↕    ↕    ↕")
print("  Row 1: E ←→ F ←→ G ←→ H")
print()
print("This requires ALL of these connections:")
print("  • A↔B, B↔C, C↔D  (top row horizontal)")
print("  • E↔F, F↔G, G↔H  (bottom row horizontal)")
print("  • A↔E, B↔F, C↔G, D↔H  (vertical connections)")
print()
print("Your physical topology has:")
print("  ✓ Good horizontal chains: 0-1-2-3 and 4-5 and 6-7")
print("  ❌ BUT: Chips 4-5 are NOT connected to 6-7")
print("  ❌ Missing: The bottom row (4,5,6,7) can't form a continuous chain!")
print()

print("✓ Why 2×2 DOES Work:")
print("-" * 60)
print("Option 1: Chips 0,1,4,5")
print("  0 ←→ 1    ✓ Top row connected")
print("  ↕    ↕    ✓ Vertical connections exist")
print("  4 ←→ 5    ✓ Bottom row connected")
print()
print("Option 2: Chips 2,3,6,7")
print("  2 ←→ 3    ✓ Top row connected")
print("  ↕    ↕    ✓ Vertical connections exist")
print("  6 ←→ 7    ✓ Bottom row connected")
print()

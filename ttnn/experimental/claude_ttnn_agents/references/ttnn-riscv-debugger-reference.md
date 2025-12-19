# TTNN RISC-V Debugger Quick Reference

Minimal reference for watcher interpretation and DPRINT syntax.

For CB debugging strategies, see `.claude/references/cb-debugging-strategy.md`

## Watcher Log Interpretation

Logs at `generated/watcher/watcher.log`.

### Log Line Format
```
Device 1 worker core(x= 0,y= 0) virtual(x=18,y=18): BRW, D,TR0W, D, D rmsg:D0G|BNT h_id:0 smsg:DWDD k_ids:1|2|3
```

### Waypoint Columns (5 RISC-V cores per Tensix)
| Position | Core | Role |
|----------|------|------|
| 1st | BRISC (RISCV_0) | Reader |
| 2nd | NCRISC (RISCV_1) | Writer |
| 3rd | TRISC0 | Unpack |
| 4th | TRISC1 | Math |
| 5th | TRISC2 | Pack |

### Waypoint Status Codes
| Code | Meaning |
|------|---------|
| `W` | Waiting |
| `R` | Running |
| `D` | Done |
| `CRBW` | CB Reserve Back Wait (producer blocked) |
| `CWFW` | CB Wait Front Wait (consumer blocked) |
| `NRW` | NoC Read Wait |
| `NWW` | NoC Write Wait |

### smsg Field
Format: `smsg:<ncrisc><trisc0><trisc1><trisc2>` where G=Go, D=Done, W=Wait

### Common Hang Patterns
| Pattern | Diagnosis |
|---------|-----------|
| `CRBW,D,W,W,W` | Reader blocked - consumer not releasing CB pages |
| `D,D,CWFW,D,D` | Compute blocked - producer not pushing to CB |
| `CRBW,CRBW,W,W,W` | Both readers blocked - CB full, no pops |

## DPRINT Syntax

```cpp
// Basic output
DPRINT << "msg" << ENDL();

// With variables
DPRINT << "iter=" << i << " val=" << val << ENDL();

// CB state check
DPRINT << "CB" << cb_id
       << " avail=" << cb_pages_available_at_front(cb_id, 1)
       << " reservable=" << cb_pages_reservable_at_back(cb_id, 1)
       << ENDL();

// Tile slice visualization
DPRINT << TSLICE(cb_id, 0, SliceRange::hw0_32_16()) << ENDL();
```

## Environment Setup

```bash
# Reset device before testing (IMPORTANT: use tt-smi -r without device ID)
tt-smi -r  # Resets all devices allocated to you - NEVER use tt-smi -r 0 or other specific device IDs

# Enable watcher (polls every N ms)
export TT_METAL_WATCHER=10

# Enable DPRINT for specific cores
export TT_METAL_DPRINT_CORES="(0,0)-(0,0)"

# Enable DPRINT for specific RISC-V processors
export TT_METAL_DPRINT_RISCVS=BR       # BRISC (reader)
export TT_METAL_DPRINT_RISCVS=NC       # NCRISC (writer)
export TT_METAL_DPRINT_RISCVS=TR0,TR1,TR2  # TRISCs (compute)
```

## Semaphore Debugging Quick Reference

### Semaphore APIs
| API | Purpose |
|-----|---------|
| `noc_semaphore_wait_min(addr, val)` | Block until sem >= val |
| `noc_semaphore_set(addr, val)` | Set local semaphore |
| `noc_semaphore_inc(noc_addr, val)` | Atomic increment on remote core |
| `noc_semaphore_set_multicast(...)` | Set semaphore on multiple cores |

### Waypoint Patterns (Semaphore-Related)
| Pattern | Diagnosis |
|---------|-----------|
| `W,W,W,W,W` (no CB code) | Semaphore deadlock |
| `NRW` stuck | NoC read stall (may be semaphore) |
| `NWW` stuck | NoC write stall (may be semaphore) |
| `R,R,W,W,W` | Readers running, compute waiting on semaphore? |

### DPRINT for Semaphores
```cpp
// Read semaphore value
volatile tt_l1_ptr uint32_t* sem = get_semaphore(sem_id);
DPRINT << "SEM" << sem_id << "=" << *sem << ENDL();

// Track signals sent
DPRINT << "INC sem" << sem_id << " to core(" << x << "," << y << ")" << ENDL();

// Track waits completed
DPRINT << "WAIT_DONE sem" << sem_id << " val=" << *sem << ENDL();
```

### Multicast Coordination Check
```bash
# Find expected receiver count
grep -n "mcast_num_dests\|num_receivers" program_factory.cpp

# Find semaphore wait threshold
grep -n "noc_semaphore_wait.*num" kernel.cpp
```

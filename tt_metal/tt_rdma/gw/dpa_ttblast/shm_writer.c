// SPDX-License-Identifier: Apache-2.0
//
// A3.3a cross-program seam test producer. Stands in for the RoCE responder: mmaps the same file-backed shared
// region the DPA re-head program maps (TTDPA_SHMFILE), writes a payload into the landing buffer, and drives the
// `produced` doorbell. Layout must match tt_run_doorbell(): [u64 produced @0][pad][landing buffer @4096].
//
// usage: shm_writer <file> <shmsize> <count> <plen> [chunk] [us]
//   file    shared mmap path (e.g. /dev/shm/tt_rehead) — same as the DPA program's TTDPA_SHMFILE
//   shmsize bytes to ftruncate/mmap — same as the DPA program's TTDPA_SHMSIZE
//   count   frames to "land" (final produced value)
//   plen    payload bytes to write into the landing buffer
//   chunk   produced increment per step (default = count: one shot / preset)
//   us      microseconds between steps (default 0). chunk<count + us>0 => arrival-driven producer.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <file> <shmsize> <count> <plen> [chunk] [us]\n", argv[0]);
        return 1;
    }
    const char* f = argv[1];
    size_t sz = strtoull(argv[2], NULL, 0);
    uint64_t count = strtoull(argv[3], NULL, 0);
    uint32_t plen = (uint32_t)strtoul(argv[4], NULL, 0);
    uint64_t chunk = argc > 5 ? strtoull(argv[5], NULL, 0) : count;
    unsigned us = argc > 6 ? (unsigned)strtoul(argv[6], NULL, 0) : 0;
    const char* tag = "A33-SHM-ROCE-LAND-";
    size_t tl = strlen(tag), i;
    volatile uint64_t* produced;
    uint64_t p = 0;
    char* land;
    void* base;
    int fd;

    if (!chunk) {
        chunk = count;
    }
    fd = open(f, O_RDWR | O_CREAT, 0666);
    if (fd < 0 || ftruncate(fd, (off_t)sz)) {
        perror("open/ftruncate");
        return 1;
    }
    base = mmap(NULL, sz, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (base == MAP_FAILED) {
        perror("mmap");
        return 1;
    }
    produced = (volatile uint64_t*)base;
    land = (char*)base + 4096;

    /* one landing slot, reused for every frame (the DPA re-heads it `count` times) */
    for (i = 0; i < plen; i++) {
        land[i] = (i < tl) ? tag[i] : (char)i;
    }
    __sync_synchronize();
    *produced = 0;
    __sync_synchronize();

    printf(
        "shm_writer: %s sz=%zu count=%lu plen=%u chunk=%lu us=%u\n",
        f,
        sz,
        (unsigned long)count,
        plen,
        (unsigned long)chunk,
        us);
    while (p < count) {
        p += chunk;
        if (p > count) {
            p = count;
        }
        *produced = p;
        __sync_synchronize();
        if (us) {
            usleep(us);
        }
    }
    printf("shm_writer: done, produced=%lu\n", (unsigned long)*produced);
    usleep(300000); /* keep the mapping briefly so a concurrent DPA drain can finish */
    munmap(base, sz);
    return 0;
}

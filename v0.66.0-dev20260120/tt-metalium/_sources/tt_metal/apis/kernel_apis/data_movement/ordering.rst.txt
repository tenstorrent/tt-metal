

Ordering (NOC, CMD_BUF, VC)
===========================

NoC requests/responses are usually asynchronous, but they can have an implicit ordering, or you could enforce an explicit ordering. In this section, we will go over
command buffers (`CMD_BUF`), virtual channels (`VC`), and allocation of virtual channels.

There are two NoCs (NoC-0 and NoC-1) that are completely phyiscally replicated and separated.
The only communication between these NoCs is through software or L1 memory.
For each NoC, we have 4 command buffers: `RD_CMD_BUF`, `WR_CMD_BUF`, `WR_REG_CMD_BUF`, `AT_CMD_BUF`.
We usually use the `WR_CMD_BUF` for all writes, except for the atomic ones where we use `AT_CMD_BUF`, e.g.
`noc_semaphore_inc` uses `AT_CMD_BUF`.

For each NoC, we have 6 `VC` (numbered 0-5). Each `VC` is usually used for a different purpose. For example,
all unicast writes go on `NOC_UNICAST_WRITE_VC`, which is `VC` 1, and all multicast writes go on `NOC_MULTICAST_WRITE_VC`, which is `VC` 4.

We can allocate `VC` either statically or dynamically. We can allocate `VC` statically using `NOC_CMD_STATIC_VC`, which is usually the case
for noc writes. For noc reads, the read requests can use a statically allocated `VC`, as for read responses, we always use
dynamically allocated `VC`. So there's really no way to control ordering for data reads.

As for ordering of the NoC writes:

* If writes are on different NoCs, there is no ordering guarantees.

* If writes are on the same NoC but different VCs, there's also no ordering guarantees. You might as well use different CMD_BUFs to avoid serialization.

* If writes are on the same NoC and same VC, they will be ordered based on program order, regardless whether you use the same CMD_BUF or different ones.

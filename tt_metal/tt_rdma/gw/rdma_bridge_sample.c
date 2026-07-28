/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * TT-RDMA gateway BRIDGE (Architecture B, Phase B1) — Tenstorrent, 2026.
 *
 * A modification of the NVIDIA DOCA sample `rdma_write_immediate_responder`. The stock sample terminates a
 * RoCEv2 RC QP, receives one RDMA WRITE_WITH_IMM, prints the written string, and stops. This bridge turns it
 * into the inbound leg of the DPU gateway: on every WRITE_IMM completion it FORWARDS the payload landed in
 * the responder mmap to the Blackhole over the TT rail as a native TT-RDMA-v1 WRITE frame (ethertype 0x1AF6
 * + 32B tt_rdma_hdr), then re-posts the receive task so it runs continuously.
 *
 * So: ConnectX HW terminates full-spec RoCEv2 (PSN/ICRC/ACK, all in silicon); this bridge does only the lean
 * TT-RDMA re-origination; the BH drainer pool lands it (unchanged, validated 200G lossless). B1 uses a raw
 * AF_PACKET socket on the uplink (p0) for egress — simple + proven from tt_rdma_bf3_send; B3 swaps this for
 * the DOCA Eth-Tx datapath (doca_ttblast) / DPA for line rate.
 *
 * Config via env (keeps the stock argp untouched):
 *   TTBRIDGE_IFACE   egress netdev toward the BH   (default p0)
 *   TTBRIDGE_DMAC    BH RXQ2 dest MAC              (default 02:00:00:00:00:02)
 *   TTBRIDGE_RKEY    TT-RDMA rkey (-> MR slot)     (default 0x00CAFE42)
 *   TTBRIDGE_PLEN    bytes to forward per WRITE    (default 256)
 *   TTBRIDGE_MAX     stop after N forwards         (default 1)
 * Build/run: deploy_rdma_bridge.sh (vendors this + builds against the stock DOCA rdma sample sources).
 */

#include <arpa/inet.h>
#include <errno.h>
#include <linux/if_packet.h>
#include <net/ethernet.h>
#include <net/if.h>
#include <netinet/in.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

#include <doca_ctx.h>
#include <doca_error.h>
#include <doca_log.h>

#include "rdma_common.h"

#define MAX_BUFF_SIZE (4096) /* responder mmap size — big enough for a jumbo forward */

DOCA_LOG_REGISTER(TT_RDMA_BRIDGE::SAMPLE);

/* ---- TT-RDMA egress state (raw AF_PACKET on the uplink toward the BH) ---- */
static int g_fd = -1;
static struct sockaddr_ll g_sa;
static unsigned char g_frame[4200];
static uint32_t g_rkey = 0x00CAFE42u;
static unsigned g_plen = 256u;
static unsigned long g_max_fwd = 1u;
static unsigned long g_fwded = 0u;
static uint32_t g_seq = 0u;

static void put_u16(unsigned char *p, uint16_t v)
{
	p[0] = v & 0xff;
	p[1] = (v >> 8) & 0xff;
}
static void put_u32(unsigned char *p, uint32_t v)
{
	for (int i = 0; i < 4; i++)
		p[i] = (v >> (8 * i)) & 0xff;
}
static void put_u64(unsigned char *p, uint64_t v)
{
	for (int i = 0; i < 8; i++)
		p[i] = (v >> (8 * i)) & 0xff;
}

/* CRC-32 (reflected 0xEDB88320) — matches tt_rdma_crc32 / the BH ETH-CTRL ROCE_ICRC poly; the RX kernel
 * drops frames whose header_cksum mismatches. */
static uint32_t tt_crc32(const unsigned char *p, unsigned n)
{
	uint32_t crc = 0xFFFFFFFFu;
	for (unsigned i = 0; i < n; i++) {
		crc ^= p[i];
		for (int b = 0; b < 8; b++) {
			uint32_t mask = (uint32_t)(-(int32_t)(crc & 1u));
			crc = (crc >> 1) ^ (0xEDB88320u & mask);
		}
	}
	return crc ^ 0xFFFFFFFFu;
}

/* Open the raw egress socket toward the BH and pre-build the L2 header. Returns 0 on success. */
static int egress_init(void)
{
	const char *iface = getenv("TTBRIDGE_IFACE");
	const char *dmac_s = getenv("TTBRIDGE_DMAC");
	const char *rkey_s = getenv("TTBRIDGE_RKEY");
	const char *plen_s = getenv("TTBRIDGE_PLEN");
	const char *max_s = getenv("TTBRIDGE_MAX");
	if (!iface)
		iface = "p0";
	if (!dmac_s)
		dmac_s = "02:00:00:00:00:02";
	if (rkey_s)
		g_rkey = (uint32_t)strtoul(rkey_s, NULL, 0);
	if (plen_s)
		g_plen = (unsigned)strtoul(plen_s, NULL, 0);
	if (max_s)
		g_max_fwd = strtoul(max_s, NULL, 0);
	if (g_plen > 4080u)
		g_plen = 4080u;

	unsigned dm[6] = {0};
	if (sscanf(dmac_s, "%x:%x:%x:%x:%x:%x", &dm[0], &dm[1], &dm[2], &dm[3], &dm[4], &dm[5]) != 6) {
		DOCA_LOG_ERR("TTBRIDGE_DMAC parse failed: %s", dmac_s);
		return -1;
	}

	g_fd = socket(AF_PACKET, SOCK_RAW, htons(ETH_P_ALL));
	if (g_fd < 0) {
		DOCA_LOG_ERR("egress raw socket: %s", strerror(errno));
		return -1;
	}
	struct ifreq ifr;
	memset(&ifr, 0, sizeof(ifr));
	strncpy(ifr.ifr_name, iface, IFNAMSIZ - 1);
	if (ioctl(g_fd, SIOCGIFINDEX, &ifr) < 0) {
		DOCA_LOG_ERR("SIOCGIFINDEX %s: %s", iface, strerror(errno));
		return -1;
	}
	int ifindex = ifr.ifr_ifindex;
	unsigned char smac[6] = {0};
	if (ioctl(g_fd, SIOCGIFHWADDR, &ifr) == 0)
		memcpy(smac, ifr.ifr_hwaddr.sa_data, 6);

	memset(&g_sa, 0, sizeof(g_sa));
	g_sa.sll_family = AF_PACKET;
	g_sa.sll_ifindex = ifindex;
	g_sa.sll_halen = 6;
	memset(g_frame, 0, sizeof(g_frame));
	for (int i = 0; i < 6; i++) {
		g_frame[i] = (unsigned char)dm[i];
		g_sa.sll_addr[i] = (unsigned char)dm[i];
	}
	memcpy(g_frame + 6, smac, 6);
	g_frame[12] = 0x1a;
	g_frame[13] = 0xf6;
	DOCA_LOG_INFO("TT-RDMA egress ready: iface=%s ifindex=%d dmac=%s rkey=0x%08x plen=%u max=%lu",
		      iface, ifindex, dmac_s, g_rkey, g_plen, g_max_fwd);
	return 0;
}

/* Build + send one TT-RDMA WRITE frame from `payload` (plen bytes) at remote offset `roff`. */
static void egress_send_ttrdma(const unsigned char *payload, unsigned plen, uint32_t roff, uint32_t imm)
{
	if (plen > 4080u)
		plen = 4080u;
	unsigned char *h = g_frame + 14;
	h[0] = 0x10; /* WRITE opcode */
	h[1] = 0x01; /* version_flags: ver=1, no IMM */
	put_u16(h + 2, 0);
	put_u32(h + 4, plen);           /* length */
	put_u32(h + 8, ++g_seq);        /* seq (BH does not order-check the pool path) */
	put_u32(h + 12, g_rkey);        /* rkey -> MR slot */
	put_u64(h + 16, roff);          /* remote offset within the MR */
	put_u32(h + 24, imm);           /* carry the RoCE imm through for traceability */
	put_u32(h + 28, tt_crc32(h, 28)); /* header_cksum over [0..27] */
	memcpy(h + 32, payload, plen);
	unsigned flen = 14u + 32u + plen;
	if (sendto(g_fd, g_frame, flen, 0, (struct sockaddr *)&g_sa, sizeof(g_sa)) < 0)
		DOCA_LOG_ERR("egress sendto: %s", strerror(errno));
}

/*
 * Write the connection details and the mmap details for the requester to read,
 * and read the connection details of the requester.
 */
static doca_error_t write_read_connection(struct rdma_config *cfg, struct rdma_resources *resources)
{
	doca_error_t result;

	result = write_file(cfg->local_connection_desc_path,
			    (char *)resources->rdma_conn_descriptor,
			    resources->rdma_conn_descriptor_size);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to write the RDMA connection details: %s", doca_error_get_descr(result));
		return result;
	}
	result = write_file(cfg->remote_resource_desc_path,
			    (char *)resources->mmap_descriptor,
			    resources->mmap_descriptor_size);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to write the RDMA mmap details: %s", doca_error_get_descr(result));
		return result;
	}
	DOCA_LOG_INFO("You can now copy %s and %s to the requester",
		      cfg->local_connection_desc_path,
		      cfg->remote_resource_desc_path);
	if (cfg->transport_type == DOCA_RDMA_TRANSPORT_TYPE_DC)
		return result;
	DOCA_LOG_INFO("Please copy %s from the requester and then press enter", cfg->remote_connection_desc_path);
	wait_for_enter();
	result = read_file(cfg->remote_connection_desc_path,
			   (char **)&resources->remote_rdma_conn_descriptor,
			   &resources->remote_rdma_conn_descriptor_size);
	if (result != DOCA_SUCCESS)
		DOCA_LOG_ERR("Failed to read the remote RDMA connection details: %s", doca_error_get_descr(result));
	return result;
}

static doca_error_t rdma_receive_prepare_and_submit_task(struct rdma_resources *resources);

/*
 * RDMA receive task completed callback — the BRIDGE hook. On each RoCEv2 WRITE_WITH_IMM, forward the payload
 * the requester landed in our mmap to the BH as a TT-RDMA WRITE, then re-post to keep bridging.
 */
static void rdma_receive_completed_callback(struct doca_rdma_task_receive *rdma_receive_task,
					    union doca_data task_user_data,
					    union doca_data ctx_user_data)
{
	struct rdma_resources *resources = (struct rdma_resources *)ctx_user_data.ptr;
	doca_be32_t immediate_data;
	enum doca_rdma_opcode op_code;
	doca_error_t *first_encountered_error = (doca_error_t *)task_user_data.ptr;
	doca_error_t result = DOCA_SUCCESS;

	op_code = doca_rdma_task_receive_get_result_opcode(rdma_receive_task);
	if (op_code != DOCA_RDMA_OPCODE_RECV_WRITE_WITH_IMM) {
		result = DOCA_ERROR_UNEXPECTED;
		DOCA_LOG_ERR("Got incorrect opcode (want RECV_WRITE_WITH_IMM)");
		goto free_task;
	}
	immediate_data = doca_rdma_task_receive_get_result_immediate_data(rdma_receive_task);

	/* Forward: the requester's RDMA WRITE landed the payload directly in resources->mmap_memrange. Re-head it
	 * as a TT-RDMA WRITE toward the BH (offset 0 for B1 byte-exact correctness). */
	egress_send_ttrdma((const unsigned char *)resources->mmap_memrange, g_plen, 0u, (uint32_t)immediate_data);
	g_fwded++;
	if ((g_fwded & 0x3FFu) == 0u || g_fwded <= 4u)
		DOCA_LOG_INFO("bridged WRITE_IMM #%lu -> TT-RDMA (imm=0x%x, %u bytes)", g_fwded,
			      (unsigned)immediate_data, g_plen);

free_task:
	doca_task_free(doca_rdma_task_receive_as_task(rdma_receive_task));
	DOCA_ERROR_PROPAGATE(*first_encountered_error, result);

	/* Keep bridging: re-post a receive task until we hit the forward cap (then let the ctx drain to idle). */
	if (result == DOCA_SUCCESS && g_fwded < g_max_fwd)
		(void)rdma_receive_prepare_and_submit_task(resources);

	resources->num_remaining_tasks--;
	if (resources->num_remaining_tasks == 0) {
		if (resources->cfg->use_rdma_cm == true)
			(void)rdma_cm_disconnect(resources);
		(void)doca_ctx_stop(resources->rdma_ctx);
	}
}

static void rdma_receive_error_callback(struct doca_rdma_task_receive *rdma_receive_task,
					union doca_data task_user_data,
					union doca_data ctx_user_data)
{
	struct rdma_resources *resources = (struct rdma_resources *)ctx_user_data.ptr;
	struct doca_task *task = doca_rdma_task_receive_as_task(rdma_receive_task);
	doca_error_t *first_encountered_error = (doca_error_t *)task_user_data.ptr;
	doca_error_t result = doca_task_get_status(task);

	DOCA_ERROR_PROPAGATE(*first_encountered_error, result);
	DOCA_LOG_ERR("RDMA receive task failed: %s", doca_error_get_descr(result));
	doca_task_free(task);
	resources->num_remaining_tasks--;
	if (resources->num_remaining_tasks == 0) {
		if (resources->cfg->use_rdma_cm == true)
			(void)rdma_cm_disconnect(resources);
		(void)doca_ctx_stop(resources->rdma_ctx);
	}
}

static doca_error_t rdma_write_immediate_export_and_connect(struct rdma_resources *resources)
{
	doca_error_t result;

	if (resources->cfg->use_rdma_cm == true)
		return rdma_cm_connect(resources);

	result = doca_rdma_export(resources->rdma,
				  &(resources->rdma_conn_descriptor),
				  &(resources->rdma_conn_descriptor_size),
				  &(resources->connections[0]));
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to export RDMA: %s", doca_error_get_descr(result));
		return result;
	}
	result = doca_mmap_export_rdma(resources->mmap,
				       resources->doca_device,
				       (const void **)&(resources->mmap_descriptor),
				       &(resources->mmap_descriptor_size));
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to export DOCA mmap for RDMA: %s", doca_error_get_descr(result));
		return result;
	}
	result = write_read_connection(resources->cfg, resources);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to write/read connection details: %s", doca_error_get_descr(result));
		return result;
	}
	if (resources->cfg->transport_type == DOCA_RDMA_TRANSPORT_TYPE_DC)
		return result;
	result = doca_rdma_connect(resources->rdma,
				   resources->remote_rdma_conn_descriptor,
				   resources->remote_rdma_conn_descriptor_size,
				   resources->connections[0]);
	if (result != DOCA_SUCCESS)
		DOCA_LOG_ERR("Failed to connect RDMA: %s", doca_error_get_descr(result));
	return result;
}

static doca_error_t rdma_receive_prepare_and_submit_task(struct rdma_resources *resources)
{
	struct doca_rdma_task_receive *rdma_receive_task = NULL;
	union doca_data task_user_data = {0};
	doca_error_t result;

	task_user_data.ptr = &(resources->first_encountered_error);
	/* NULL dst buffer: the receive task only surfaces the IMM completion; the data itself is written by the
	 * requester straight into our exported mmap (resources->mmap_memrange). */
	result = doca_rdma_task_receive_allocate_init(resources->rdma, NULL, task_user_data, &rdma_receive_task);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to allocate RDMA receive task: %s", doca_error_get_descr(result));
		return result;
	}
	resources->num_remaining_tasks++;
	result = doca_task_submit(doca_rdma_task_receive_as_task(rdma_receive_task));
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to submit RDMA receive task: %s", doca_error_get_descr(result));
		doca_task_free(doca_rdma_task_receive_as_task(rdma_receive_task));
	}
	return result;
}

static void rdma_write_imm_responder_state_change_callback(const union doca_data user_data,
							   struct doca_ctx *ctx,
							   enum doca_ctx_states prev_state,
							   enum doca_ctx_states next_state)
{
	struct rdma_resources *resources = (struct rdma_resources *)user_data.ptr;
	struct rdma_config *cfg = resources->cfg;
	doca_error_t result = DOCA_SUCCESS;
	(void)prev_state;
	(void)ctx;

	switch (next_state) {
	case DOCA_CTX_STATE_STARTING:
		DOCA_LOG_INFO("RDMA context entered starting state");
		break;
	case DOCA_CTX_STATE_RUNNING:
		DOCA_LOG_INFO("RDMA context is running");
		result = rdma_write_immediate_export_and_connect(resources);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("export_and_connect failed: %s", doca_error_get_descr(result));
			break;
		}
		if (cfg->use_rdma_cm == true)
			break;
		result = rdma_receive_prepare_and_submit_task(resources);
		if (result != DOCA_SUCCESS)
			DOCA_LOG_ERR("prepare_and_submit_task failed: %s", doca_error_get_descr(result));
		break;
	case DOCA_CTX_STATE_STOPPING:
		DOCA_LOG_INFO("RDMA context stopping; inflight tasks flushed");
		break;
	case DOCA_CTX_STATE_IDLE:
		DOCA_LOG_INFO("RDMA context stopped");
		resources->run_pe_progress = false;
		break;
	default:
		break;
	}
	if (result != DOCA_SUCCESS) {
		DOCA_ERROR_PROPAGATE(resources->first_encountered_error, result);
		(void)doca_ctx_stop(ctx);
	}
}

/*
 * Bridge entry — same name/signature as the stock sample so the stock *_main.c drives it.
 */
doca_error_t rdma_write_immediate_responder(struct rdma_config *cfg)
{
	struct rdma_resources resources = {0};
	union doca_data ctx_user_data = {0};
	const uint32_t mmap_permissions = DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_WRITE;
	const uint32_t rdma_permissions = DOCA_ACCESS_FLAG_RDMA_WRITE;
	struct timespec ts = {.tv_sec = 0, .tv_nsec = SLEEP_IN_NANOS};
	doca_error_t result, tmp_result;

	if (egress_init() != 0)
		return DOCA_ERROR_INITIALIZATION;

	result = allocate_rdma_resources(cfg, mmap_permissions, rdma_permissions,
					 doca_rdma_cap_task_receive_is_supported, &resources);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to allocate RDMA Resources: %s", doca_error_get_descr(result));
		return result;
	}
	result = doca_rdma_task_receive_set_conf(resources.rdma, rdma_receive_completed_callback,
						 rdma_receive_error_callback, NUM_RDMA_TASKS);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Unable to set RDMA receive task conf: %s", doca_error_get_descr(result));
		goto destroy_resources;
	}
	result = doca_ctx_set_state_changed_cb(resources.rdma_ctx, rdma_write_imm_responder_state_change_callback);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Unable to set state change cb: %s", doca_error_get_descr(result));
		goto destroy_resources;
	}
	ctx_user_data.ptr = &(resources);
	result = doca_ctx_set_user_data(resources.rdma_ctx, ctx_user_data);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set context user data: %s", doca_error_get_descr(result));
		goto destroy_resources;
	}
	if (cfg->use_rdma_cm == true) {
		resources.is_requester = false;
		resources.require_remote_mmap = true;
		resources.task_fn = rdma_receive_prepare_and_submit_task;
		result = config_rdma_cm_callback_and_negotiation_task(&resources, true, false);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to config RDMA CM: %s", doca_error_get_descr(result));
			goto destroy_resources;
		}
	}
	result = doca_ctx_start(resources.rdma_ctx);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to start RDMA context: %s", doca_error_get_descr(result));
		goto destroy_resources;
	}
	while (resources.run_pe_progress) {
		if (doca_pe_progress(resources.pe) == 0)
			nanosleep(&ts, &ts);
	}
	result = resources.first_encountered_error;
	DOCA_LOG_INFO("Bridge done: forwarded %lu WRITE_IMM -> TT-RDMA", g_fwded);

destroy_resources:
	if (g_fd >= 0)
		close(g_fd);
	if (resources.buf_inventory != NULL) {
		tmp_result = doca_buf_inventory_stop(resources.buf_inventory);
		if (tmp_result != DOCA_SUCCESS)
			DOCA_ERROR_PROPAGATE(result, tmp_result);
		tmp_result = doca_buf_inventory_destroy(resources.buf_inventory);
		if (tmp_result != DOCA_SUCCESS)
			DOCA_ERROR_PROPAGATE(result, tmp_result);
	}
	tmp_result = destroy_rdma_resources(&resources, cfg);
	if (tmp_result != DOCA_SUCCESS)
		DOCA_ERROR_PROPAGATE(result, tmp_result);
	return result;
}

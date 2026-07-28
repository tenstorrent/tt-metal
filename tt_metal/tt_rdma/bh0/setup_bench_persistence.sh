#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Make the TT-RDMA bench connectivity durable across DPU blasts / link bounces. Every heavy DPU blast used
# to (a) drop the host tmfifo IP (192.168.100.1) and (b) reset the DPU-side jumbo MTUs to 1500 -> 4096/jumbo
# frames then silently eSwitch-dropped and the pool saw no frames. This installs two keepers:
#   - HOST: a --user systemd service that re-adds 192.168.100.1/30 on tmfifo_net0 every 2s (uses the
#     passwordless-allowlisted `sudo -n ip`; no root needed to install).
#   - DPU:  a systemd service (re-applies MTU 9000 on p0/p1/pf0hpf/pf1hpf every 2s) + a udev rule that
#     re-applies on every net add/change (link-up) -> self-heals within ~2s of any reset.
# Verified: forcing p0 to 1500 self-heals to 9000 in <3s; both keepers survive a 200G blast.
#
# Run once on the bench host (desktop-0). For the DPU it uses the askpass ssh helper in this dir.
set -uo pipefail
SP="$(cd "$(dirname "$0")" && pwd)"
sshdpu(){ SSH_ASKPASS="$SP/askpass.sh" SSH_ASKPASS_REQUIRE=force DISPLAY=:0 setsid -w \
  ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=12 ubuntu@192.168.100.2 "$@"; }

echo "== HOST tmfifo-IP keeper (user systemd) =="
mkdir -p ~/.local/bin ~/.config/systemd/user
cat > ~/.local/bin/tt-tmfifo-ip.sh <<'EOF'
#!/bin/bash
while true; do
  ip addr show tmfifo_net0 2>/dev/null | grep -q "192.168.100.1" || sudo -n ip addr add 192.168.100.1/30 dev tmfifo_net0 2>/dev/null
  sudo -n ip link set tmfifo_net0 up 2>/dev/null
  sleep 2
done
EOF
chmod +x ~/.local/bin/tt-tmfifo-ip.sh
cat > ~/.config/systemd/user/tt-tmfifo-ip.service <<'EOF'
[Unit]
Description=TT tmfifo IP keeper (DPU management link)
[Service]
ExecStart=%h/.local/bin/tt-tmfifo-ip.sh
Restart=always
RestartSec=2
[Install]
WantedBy=default.target
EOF
systemctl --user daemon-reload
systemctl --user enable --now tt-tmfifo-ip.service
echo "  host keeper: $(systemctl --user is-active tt-tmfifo-ip.service)"
# Optional: survive logout (needs a one-time root sudo the agent can't do non-interactively):
#   sudo loginctl enable-linger "$USER"

echo "== DPU jumbo-MTU keeper (systemd + udev) =="
WSH=$(printf '%s\n' '#!/bin/bash' 'while true; do for i in p0 p1 pf0hpf pf1hpf; do ip link set dev "$i" mtu 9000 2>/dev/null; done; sleep 2; done' | base64 -w0)
UNIT=$(printf '%s\n' '[Unit]' 'Description=TT jumbo MTU keeper for RDMA rails' 'After=network.target' '[Service]' 'ExecStart=/usr/local/bin/tt-jumbo-mtu.sh' 'Restart=always' 'RestartSec=2' '[Install]' 'WantedBy=multi-user.target' | base64 -w0)
RULE=$(printf '%s\n' 'ACTION=="add|change", SUBSYSTEM=="net", KERNEL=="p0|p1|pf0hpf|pf1hpf", RUN+="/sbin/ip link set %k mtu 9000"' | base64 -w0)
sshdpu "echo $WSH | base64 -d > /tmp/tt-jumbo-mtu.sh; echo $UNIT | base64 -d > /tmp/tt-jumbo-mtu.service; echo $RULE | base64 -d > /tmp/99-tt-jumbo.rules; echo ubuntu | sudo -S bash -c 'install -m755 /tmp/tt-jumbo-mtu.sh /usr/local/bin/tt-jumbo-mtu.sh && install -m644 /tmp/tt-jumbo-mtu.service /etc/systemd/system/tt-jumbo-mtu.service && install -m644 /tmp/99-tt-jumbo.rules /etc/udev/rules.d/99-tt-jumbo.rules && systemctl daemon-reload && systemctl enable --now tt-jumbo-mtu.service && udevadm control --reload-rules && echo DPU_INSTALLED'" 2>&1 | grep -viE "askpass|Warning|Permanently|password" | tail -1
echo "done."

curl -fsSL https://tailscale.com/install.sh | sh
tailscaled --tun=userspace-networking --socks5-server=localhost:54355 --outbound-http-proxy-listen=localhost:54355 &
tailscale up
tailscale funnel 6006

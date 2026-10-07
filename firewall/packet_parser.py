"""
firewall/packet_parser.py
─────────────────────────
Fast, resilient network packet parser supporting IPv4, IPv6, TCP, UDP, ICMP.
Extracts canonical 5-tuples, TCP flags, timing, and packet sizes into a
structured ParsedPacket dataclass.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class ParsedPacket:
    """Immutable representation of an extracted network packet."""

    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str  # "tcp", "udp", "icmp", "other"
    ip_proto: int  # 6=TCP, 17=UDP, 1=ICMP, 58=ICMPv6, etc.
    packet_len: int
    payload_len: int
    tcp_flags: dict[str, bool] = field(default_factory=dict)
    raw_flags: int = 0
    flag_str: str = "OTH"  # NSL-KDD compatible: "SF", "S0", "REJ", "RSTO", etc.
    ttl: int = 64
    timestamp: float = field(default_factory=time.time)
    is_ipv4: bool = True
    is_ipv6: bool = False

    @property
    def flow_key(self) -> tuple[str, str, int, int, str]:
        """Directional 5-tuple: (src_ip, dst_ip, src_port, dst_port, protocol)."""
        return (self.src_ip, self.dst_ip, self.src_port, self.dst_port, self.protocol)

    @property
    def canonical_flow_key(self) -> tuple[str, str, int, int, str]:
        """
        Bidirectional canonical 5-tuple.
        Orders endpoints deterministically so traffic in both directions maps together.
        """
        ep1 = (self.src_ip, self.src_port)
        ep2 = (self.dst_ip, self.dst_port)
        if ep1 <= ep2:
            return (self.src_ip, self.dst_ip, self.src_port, self.dst_port, self.protocol)
        return (self.dst_ip, self.src_ip, self.dst_port, self.src_port, self.protocol)

    def is_syn_only(self) -> bool:
        """Check if packet is a SYN without ACK (connection request)."""
        return bool(self.tcp_flags.get("SYN") and not self.tcp_flags.get("ACK"))

    def is_syn_ack(self) -> bool:
        """Check if packet is SYN-ACK (handshake response)."""
        return bool(self.tcp_flags.get("SYN") and self.tcp_flags.get("ACK"))

    def is_fin(self) -> bool:
        """Check if packet has FIN flag set."""
        return bool(self.tcp_flags.get("FIN"))

    def is_rst(self) -> bool:
        """Check if packet has RST flag set."""
        return bool(self.tcp_flags.get("RST"))


class PacketParser:
    """
    Parses raw Scapy packets, raw byte buffers, or synthetic packet dictionaries
    into ParsedPacket instances with robust error handling.
    """

    # TCP flag bits
    FLAG_FIN = 0x01
    FLAG_SYN = 0x02
    FLAG_RST = 0x04
    FLAG_PSH = 0x08
    FLAG_ACK = 0x10
    FLAG_URG = 0x20
    FLAG_ECE = 0x40
    FLAG_CWR = 0x80

    def parse(self, packet: Any, timestamp: Optional[float] = None) -> Optional[ParsedPacket]:
        """
        Parse a packet object (Scapy packet or mock/dict).
        Returns None if packet is non-IP or malformed.
        """
        if packet is None:
            return None

        # Allow passing mock/dictionary objects for testing
        if isinstance(packet, dict):
            return self._parse_from_dict(packet, timestamp)

        try:
            from scapy.layers.inet import ICMP, IP, TCP, UDP
            from scapy.layers.inet6 import IPv6
        except ImportError:
            return None

        pkt_time = float(getattr(packet, "time", timestamp or time.time()))
        pkt_len = len(packet)

        # Check IP layer
        is_ipv4 = packet.haslayer(IP)
        is_ipv6 = packet.haslayer(IPv6) if not is_ipv4 else False

        if not (is_ipv4 or is_ipv6):
            return None

        if is_ipv4:
            ip_layer = packet[IP]
            src_ip = str(ip_layer.src)
            dst_ip = str(ip_layer.dst)
            ip_proto = int(ip_layer.proto)
            ttl = int(ip_layer.ttl)
        else:
            ip_layer = packet[IPv6]
            src_ip = str(ip_layer.src)
            dst_ip = str(ip_layer.dst)
            ip_proto = int(ip_layer.nh)
            ttl = int(ip_layer.hlim)

        src_port = 0
        dst_port = 0
        protocol = "other"
        payload_len = 0
        raw_flags = 0
        flags_dict: dict[str, bool] = {
            "FIN": False, "SYN": False, "RST": False, "PSH": False,
            "ACK": False, "URG": False, "ECE": False, "CWR": False,
        }
        flag_str = "OTH"

        if packet.haslayer(TCP):
            protocol = "tcp"
            tcp_layer = packet[TCP]
            src_port = int(tcp_layer.sport)
            dst_port = int(tcp_layer.dport)
            payload_len = len(tcp_layer.payload)

            raw_flags = int(tcp_layer.flags)
            flags_dict = {
                "FIN": bool(raw_flags & self.FLAG_FIN),
                "SYN": bool(raw_flags & self.FLAG_SYN),
                "RST": bool(raw_flags & self.FLAG_RST),
                "PSH": bool(raw_flags & self.FLAG_PSH),
                "ACK": bool(raw_flags & self.FLAG_ACK),
                "URG": bool(raw_flags & self.FLAG_URG),
                "ECE": bool(raw_flags & self.FLAG_ECE),
                "CWR": bool(raw_flags & self.FLAG_CWR),
            }
            flag_str = self._resolve_kdd_flag(raw_flags, flags_dict)

        elif packet.haslayer(UDP):
            protocol = "udp"
            udp_layer = packet[UDP]
            src_port = int(udp_layer.sport)
            dst_port = int(udp_layer.dport)
            payload_len = len(udp_layer.payload)
            flag_str = "SF"  # UDP is connectionless; default normal flag

        elif packet.haslayer(ICMP):
            protocol = "icmp"
            src_port = 0
            dst_port = 0
            payload_len = len(packet[ICMP].payload)
            flag_str = "SF"

        return ParsedPacket(
            src_ip=src_ip,
            dst_ip=dst_ip,
            src_port=src_port,
            dst_port=dst_port,
            protocol=protocol,
            ip_proto=ip_proto,
            packet_len=pkt_len,
            payload_len=payload_len,
            tcp_flags=flags_dict,
            raw_flags=raw_flags,
            flag_str=flag_str,
            ttl=ttl,
            timestamp=pkt_time,
            is_ipv4=is_ipv4,
            is_ipv6=is_ipv6,
        )

    def _resolve_kdd_flag(self, raw_flags: int, flags: dict[str, bool]) -> str:
        """Map TCP flags to NSL-KDD flag categories."""
        if flags["RST"]:
            if flags["ACK"]:
                return "RSTR"
            return "RSTO"
        if flags["SYN"] and not flags["ACK"]:
            return "S0"
        if flags["SYN"] and flags["ACK"]:
            return "S1"
        if flags["FIN"] and flags["ACK"]:
            return "SF"
        if flags["ACK"]:
            return "SF"
        return "OTH"

    def _parse_from_dict(self, d: dict, timestamp: Optional[float] = None) -> ParsedPacket:
        """Helper to create ParsedPacket directly from dictionary (for tests & mocks)."""
        flags_dict = d.get("tcp_flags", {
            "FIN": False, "SYN": False, "RST": False, "PSH": False,
            "ACK": False, "URG": False, "ECE": False, "CWR": False,
        })
        raw_flags = d.get("raw_flags", 0)
        flag_str = d.get("flag_str") or self._resolve_kdd_flag(raw_flags, flags_dict)

        return ParsedPacket(
            src_ip=d.get("src_ip", "127.0.0.1"),
            dst_ip=d.get("dst_ip", "127.0.0.1"),
            src_port=d.get("src_port", 0),
            dst_port=d.get("dst_port", 0),
            protocol=d.get("protocol", "tcp"),
            ip_proto=d.get("ip_proto", 6),
            packet_len=d.get("packet_len", 64),
            payload_len=d.get("payload_len", 0),
            tcp_flags=flags_dict,
            raw_flags=raw_flags,
            flag_str=flag_str,
            ttl=d.get("ttl", 64),
            timestamp=d.get("timestamp", timestamp or time.time()),
            is_ipv4=d.get("is_ipv4", True),
            is_ipv6=d.get("is_ipv6", False),
        )

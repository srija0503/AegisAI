"""
datasets/feature_extraction.py
────────────────────────────────
Extracts network flow features from raw packets (via Scapy) or from processed
DataFrames. Produces the 41-dimensional feature vector used by the RL agent.

Two modes:
  - offline_from_df(df) : convert an already-loaded DataFrame row → feature vector
  - online_from_packet(packet) : extract partial features from a live Scapy packet
                                  (used by FirewallController for live capture)

The 41 features mirror NSL-KDD for dataset compatibility:
  [0..3]   - protocol_type, service, flag (encoded), land
  [4..8]   - src_bytes, dst_bytes, wrong_fragment, urgent, hot
  [9..13]  - num_failed_logins, logged_in, num_compromised, root_shell, su_attempted
  [14..20] - num_root, num_file_creations, num_shells, num_access_files,
              num_outbound_cmds, is_host_login, is_guest_login
  [21..28] - count, srv_count, serror_rate, srv_serror_rate, rerror_rate,
              srv_rerror_rate, same_srv_rate, diff_srv_rate
  [29..36] - srv_diff_host_rate, dst_host_count, dst_host_srv_count,
              dst_host_same_srv_rate, dst_host_diff_srv_rate,
              dst_host_same_src_port_rate, dst_host_srv_diff_host_rate,
              dst_host_serror_rate
  [37..40] - dst_host_srv_serror_rate, dst_host_rerror_rate,
              dst_host_srv_rerror_rate, duration
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ─── Feature vector definition ────────────────────────────────────────────────

FEATURE_NAMES: list[str] = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

FEATURE_DIM = len(FEATURE_NAMES)  # 41

# Mappings used during online extraction
PROTOCOL_MAP = {"tcp": 0, "udp": 1, "icmp": 2, "other": 3}
FLAG_MAP = {
    "SF": 0, "S0": 1, "REJ": 2, "RSTO": 3, "RSTR": 4,
    "SH": 5, "S1": 6, "S2": 7, "S3": 8, "OTH": 9,
}
# Common service port → service name mapping (subset)
PORT_SERVICE_MAP: dict[int, str] = {
    21: "ftp", 22: "ssh", 23: "telnet", 25: "smtp",
    53: "domain", 80: "http", 110: "pop_3", 143: "imap4",
    443: "https", 3306: "sql_net", 3389: "rdp",
}


@dataclass
class FlowStats:
    """Accumulates per-flow statistics used to build the feature vector."""

    src_ip: str = ""
    dst_ip: str = ""
    src_port: int = 0
    dst_port: int = 0
    protocol: str = "other"
    service: str = "other"
    flag: str = "OTH"

    start_time: float = field(default_factory=time.time)
    last_time: float = field(default_factory=time.time)

    src_bytes: int = 0
    dst_bytes: int = 0
    packet_count: int = 0
    wrong_fragment: int = 0

    # Flags seen
    syn_count: int = 0
    fin_count: int = 0
    rst_count: int = 0

    @property
    def duration(self) -> float:
        return max(0.0, self.last_time - self.start_time)


# ─── Offline extractor (from DataFrame) ───────────────────────────────────────

class OfflineFeatureExtractor:
    """
    Converts rows from a preprocessed NSL-KDD / CICIDS2017 DataFrame
    into numpy feature vectors ready for the RL agent.
    """

    def __init__(self, feature_names: list[str] = FEATURE_NAMES) -> None:
        self.feature_names = feature_names

    def extract(self, row: pd.Series) -> np.ndarray:
        """Extract a single feature vector from a DataFrame row."""
        vec = np.zeros(FEATURE_DIM, dtype=np.float32)
        for i, fname in enumerate(self.feature_names):
            if fname in row.index:
                vec[i] = float(row[fname])
        return vec

    def batch_extract(self, df: pd.DataFrame) -> np.ndarray:
        """Extract all feature vectors from a DataFrame at once."""
        available = [f for f in self.feature_names if f in df.columns]
        arr = df[available].values.astype(np.float32)
        if arr.shape[1] < FEATURE_DIM:
            # Pad missing columns with zeros
            pad = np.zeros((len(df), FEATURE_DIM - arr.shape[1]), dtype=np.float32)
            arr = np.hstack([arr, pad])
        return arr


# ─── Online extractor (from Scapy packets) ────────────────────────────────────

class OnlineFeatureExtractor:
    """
    Extracts features from live Scapy packets.
    Maintains flow state internally.
    """

    def __init__(self, window_size: int = 100) -> None:
        self.window_size = window_size
        # recent flows for computing rate features
        self._recent_flows: list[FlowStats] = []
        self._dst_host_stats: dict[str, list] = defaultdict(list)

    def _get_service(self, port: int) -> str:
        return PORT_SERVICE_MAP.get(port, "other")

    def _get_protocol(self, pkt) -> str:
        try:
            from scapy.layers.inet import TCP, UDP, ICMP
            if pkt.haslayer(TCP):
                return "tcp"
            elif pkt.haslayer(UDP):
                return "udp"
            elif pkt.haslayer(ICMP):
                return "icmp"
        except ImportError:
            pass
        return "other"

    def _get_flag(self, pkt) -> str:
        try:
            from scapy.layers.inet import TCP
            if pkt.haslayer(TCP):
                flags = pkt[TCP].flags
                # Simplified flag determination
                if flags & 0x01 and flags & 0x10:  # FIN+ACK
                    return "SF"
                elif flags & 0x04:  # RST
                    return "RSTO"
                elif flags & 0x02:  # SYN only
                    return "S0"
                elif flags & 0x01:  # FIN
                    return "SF"
        except (ImportError, AttributeError):
            pass
        return "OTH"

    def extract_from_flow(self, flow: FlowStats, all_flows: list[FlowStats]) -> np.ndarray:
        """
        Compute the 41-dimensional feature vector from a FlowStats object
        plus the window of recent flows for rate computations.
        """
        vec = np.zeros(FEATURE_DIM, dtype=np.float32)

        vec[0]  = min(flow.duration, 58329.0) / 58329.0   # duration (normalized)
        vec[1]  = PROTOCOL_MAP.get(flow.protocol, 3)
        vec[2]  = hash(flow.service) % 70 / 70.0          # service (rough encode)
        vec[3]  = FLAG_MAP.get(flow.flag, 9) / 9.0
        vec[4]  = min(flow.src_bytes, 1e9) / 1e9          # src_bytes
        vec[5]  = min(flow.dst_bytes, 1e9) / 1e9          # dst_bytes
        vec[6]  = 1.0 if flow.src_ip == flow.dst_ip else 0.0  # land
        vec[7]  = min(flow.wrong_fragment, 3.0) / 3.0

        # --- Rate features using recent window ---
        window = all_flows[-self.window_size:] if all_flows else []
        same_host = [f for f in window if f.dst_ip == flow.dst_ip]
        same_srv  = [f for f in window if f.service == flow.service]

        n_win = len(window) or 1
        vec[22] = len(same_host) / n_win   # count
        vec[23] = len(same_srv) / n_win    # srv_count

        syn_err   = sum(1 for f in same_host if f.flag in ("S0", "S1", "S2", "S3"))
        rst_err   = sum(1 for f in same_host if f.flag in ("REJ", "RSTO", "RSTR"))
        vec[24] = syn_err / (len(same_host) or 1)   # serror_rate
        vec[26] = rst_err / (len(same_host) or 1)   # rerror_rate

        diff_srv = len({f.service for f in same_host} - {flow.service})
        vec[29] = diff_srv / (len(same_host) or 1)  # diff_srv_rate

        # dst_host_count
        dst_count = sum(1 for f in all_flows[-256:] if f.dst_ip == flow.dst_ip)
        vec[31] = min(dst_count, 255) / 255.0

        return vec

    def extract_from_packet(self, pkt, flow_stats: "FlowStats") -> np.ndarray:
        """Quick extraction for a single raw packet (updates flow stats)."""
        # Update flow stats
        try:
            from scapy.layers.inet import IP, TCP, UDP
            if pkt.haslayer(IP):
                flow_stats.protocol = self._get_protocol(pkt)
                if pkt.haslayer(TCP):
                    flow_stats.dst_port = pkt[TCP].dport
                    flow_stats.src_port = pkt[TCP].sport
                elif pkt.haslayer(UDP):
                    flow_stats.dst_port = pkt[UDP].dport
                    flow_stats.src_port = pkt[UDP].sport
                flow_stats.service = self._get_service(flow_stats.dst_port)
                flow_stats.flag    = self._get_flag(pkt)
                flow_stats.src_bytes += len(pkt)
                flow_stats.last_time  = time.time()
        except Exception:
            pass

        return self.extract_from_flow(flow_stats, self._recent_flows)


# ─── Public helpers ───────────────────────────────────────────────────────────

def get_feature_names() -> list[str]:
    """Return the ordered list of 41 feature names."""
    return FEATURE_NAMES.copy()


def df_to_tensors(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Convenience function: extract (X, y) arrays from a preprocessed DataFrame.
    Returns:
        X : float32 array of shape (N, 41)
        y : int32 array of shape (N,) — 0=normal, 1=attack
    """
    extractor = OfflineFeatureExtractor()
    X = extractor.batch_extract(df)
    y = df["label"].values.astype(np.int32) if "label" in df.columns else np.zeros(len(df), np.int32)
    return X, y

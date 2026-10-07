"""
firewall/packet_capture.py
──────────────────────────
Packet capture engine supporting:
1. Live network interface sniffing (via Scapy)
2. PCAP file replay (with optional rate pacing)
3. Synthetic / in-memory packet injection (for tests, simulation, and CI)
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Generator, Iterator, Optional


@dataclass
class CaptureMetrics:
    packets_captured: int = 0
    packets_dropped: int = 0
    bytes_captured: int = 0
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def duration(self) -> float:
        now = self.end_time if self.end_time > 0 else time.time()
        return max(0.0, now - self.start_time) if self.start_time > 0 else 0.0

    @property
    def packets_per_second(self) -> float:
        d = self.duration
        return (self.packets_captured / d) if d > 0.001 else 0.0


class PacketCapture:
    """
    Asynchronous packet capture engine. Feeds packets into an internal thread-safe
    queue consumed by the firewall controller.
    """

    def __init__(
        self,
        interface: Optional[str] = "eth0",
        bpf_filter: str = "ip",
        queue_maxsize: int = 50000,
        promisc: bool = True,
        snaplen: int = 65535,
    ) -> None:
        self.interface = interface
        self.bpf_filter = bpf_filter
        self.promisc = promisc
        self.snaplen = snaplen

        self._queue: queue.Queue[Any] = queue.Queue(maxsize=queue_maxsize)
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._sniffer: Any = None
        self.metrics = CaptureMetrics()

    def start_live(self) -> None:
        """Start asynchronous live packet sniffing on configured interface."""
        if self._running:
            return

        self._running = True
        self.metrics.start_time = time.time()
        self.metrics.end_time = 0.0

        try:
            from scapy.sendrecv import AsyncSniffer

            self._sniffer = AsyncSniffer(
                iface=self.interface,
                filter=self.bpf_filter,
                prn=self._on_packet,
                store=False,
                promisc=self.promisc,
            )
            self._sniffer.start()
        except Exception:
            # Fallback for environments lacking raw sockets or valid interface
            self._running = False
            raise

    def start_pcap_replay(
        self,
        pcap_path: str,
        rate_pps: Optional[float] = None,
        loop: bool = False,
    ) -> None:
        """Start PCAP file replay in a background thread."""
        if self._running:
            return

        path = Path(pcap_path)
        if not path.exists():
            raise FileNotFoundError(f"PCAP file not found: {pcap_path}")

        self._running = True
        self.metrics.start_time = time.time()
        self.metrics.end_time = 0.0

        self._worker_thread = threading.Thread(
            target=self._pcap_worker,
            args=(str(path), rate_pps, loop),
            daemon=True,
        )
        self._worker_thread.start()

    def inject_packet(self, packet: Any) -> bool:
        """
        Inject a synthetic packet into the capture queue (used by Simulation / tests).
        Returns True if queued, False if queue is full.
        """
        try:
            self._queue.put_nowait(packet)
            self.metrics.packets_captured += 1
            self.metrics.bytes_captured += len(packet) if hasattr(packet, "__len__") else 64
            if self.metrics.start_time == 0:
                self.metrics.start_time = time.time()
            return True
        except queue.Full:
            self.metrics.packets_dropped += 1
            return False

    def get_packet(self, timeout: float = 0.1) -> Optional[Any]:
        """Retrieve next packet from capture queue. Returns None on timeout."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stream_packets(self, max_packets: Optional[int] = None) -> Iterator[Any]:
        """Generator yielding packets continuously until stopped or max_packets reached."""
        count = 0
        while self._running or not self._queue.empty():
            pkt = self.get_packet(timeout=0.1)
            if pkt is not None:
                yield pkt
                count += 1
                if max_packets is not None and count >= max_packets:
                    break

    def stop(self) -> None:
        """Halt packet capture and workers."""
        self._running = False
        self.metrics.end_time = time.time()

        if self._sniffer is not None:
            try:
                self._sniffer.stop()
            except Exception:
                pass
            self._sniffer = None

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
            self._worker_thread = None

    def is_running(self) -> bool:
        return self._running

    def queue_size(self) -> int:
        return self._queue.qsize()

    def _on_packet(self, pkt: Any) -> None:
        """Callback invoked by Scapy sniffer for each live packet."""
        if not self._running:
            return
        try:
            self._queue.put_nowait(pkt)
            self.metrics.packets_captured += 1
            self.metrics.bytes_captured += len(pkt)
        except queue.Full:
            self.metrics.packets_dropped += 1

    def _pcap_worker(self, path: str, rate_pps: Optional[float], loop: bool) -> None:
        """Replay worker thread function."""
        try:
            from scapy.utils import PcapReader

            delay = (1.0 / rate_pps) if rate_pps and rate_pps > 0 else 0.0

            while self._running:
                with PcapReader(path) as reader:
                    for pkt in reader:
                        if not self._running:
                            break
                        self._on_packet(pkt)
                        if delay > 0:
                            time.sleep(delay)

                if not loop or not self._running:
                    break
        except Exception:
            pass
        finally:
            self._running = False

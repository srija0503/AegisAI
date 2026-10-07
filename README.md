# Quantum-Secure Federated Firewall 🛡️

> A multi-layered, quantum-secure network architecture where individual edge firewalls collaboratively learn from each other while keeping their local network data completely hidden.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                         EDGE FIREWALL NODE                           │
│                                                                      │
│  [Packet Capture] → [Flow Manager] → [Feature Extractor]            │
│         ↓                                                            │
│  [RL Agent] ──(anomaly/zero-day)──→ [Agentic RAG] → [Rule Update]  │
│         ↓                                                            │
│  [Firewall Controller] → ALLOW / BLOCK                              │
│         ↓ (periodic weight export)                                   │
└──────────────────┬───────────────────────────────────────────────────┘
                   │  PQC-encrypted (Kyber + Dilithium)
                   ▼
         ┌──────────────────┐
         │ Central Aggregator│  ← Federated Learning (FedAvg)
         │  (FedAvg / WAFL) │
         └──────────────────┘
```

## Team Structure

| Member | Role | Modules |
|--------|------|---------|
| Member 1 | Edge + RL | `firewall/`, `rl_agent/`, `simulation/`, `datasets/` |
| Member 2 | RAG + KB  | `agentic_rag/`, `knowledge_base/` |
| Member 3 | FL + PQC + Backend | `federated_learning/`, `pqc_security/`, `communication/`, `backend/`, `dashboard/` |

## Quick Start

```bash
# 1. Clone & setup environment
git clone <repo-url> && cd quantum_secure_federated_firewall
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt

# 2. Copy environment config
cp .env.example .env
# Edit .env with your settings

# 3. Download datasets
python datasets/download_dataset.py --dataset nsl-kdd

# 4. Preprocess
python datasets/preprocess.py --dataset nsl-kdd
python datasets/train_test_split.py --dataset nsl-kdd

# 5. Train RL agent
python scripts/train_initial_model.py

# 6. Start simulation
python scripts/start_simulation.py

# 7. Start all services (Docker)
docker-compose up
```

## Key Technologies

| Layer | Technology |
|-------|-----------|
| Packet capture | Scapy |
| RL framework | PyTorch + Gymnasium |
| RL algorithm | DQN (Dueling + PER) |
| RAG | ChromaDB + LlamaIndex + Ollama |
| Federated Learning | Flower (flwr) |
| Post-Quantum Crypto | liboqs (Kyber768, Dilithium3) |
| Backend API | FastAPI |
| Dashboard | React + Vite |
| Containers | Docker + Kubernetes |

## Dataset

- **NSL-KDD** — Primary training dataset (125,973 train / 22,544 test samples, 41 features, 5 classes)
- **CICIDS2017** — Supplementary dataset with modern attack types (DDoS, infiltration, web attacks)

## License

MIT License — see [LICENSE](LICENSE)

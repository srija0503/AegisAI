# System Architecture

## Overview

The Quantum-Secure Federated Firewall (QSFF) is a distributed intrusion detection and prevention system with three primary layers:

1. **Edge Layer** — RL-powered firewalls at each network node
2. **Intelligence Layer** — Agentic RAG for contextual threat analysis
3. **Learning Layer** — Federated Learning with PQC-secured communications

## Component Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                     NETWORK PERIMETER                               │
│                                                                     │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐ │
│  │   EDGE NODE 01   │  │   EDGE NODE 02   │  │   EDGE NODE 03   │ │
│  │                  │  │                  │  │                  │ │
│  │ [Scapy Capture]  │  │ [Scapy Capture]  │  │ [Scapy Capture]  │ │
│  │       ↓          │  │       ↓          │  │       ↓          │ │
│  │ [Flow Manager]   │  │ [Flow Manager]   │  │ [Flow Manager]   │ │
│  │       ↓          │  │       ↓          │  │       ↓          │ │
│  │ [Feature Extract]│  │ [Feature Extract]│  │ [Feature Extract]│ │
│  │       ↓          │  │       ↓          │  │       ↓          │ │
│  │  [RL Agent DQN]  │  │  [RL Agent DQN]  │  │  [RL Agent DQN]  │ │
│  │  ALLOW/BLOCK/RAG │  │  ALLOW/BLOCK/RAG │  │  ALLOW/BLOCK/RAG │ │
│  │       ↓ (RAG)    │  │       ↓ (RAG)    │  │       ↓ (RAG)    │ │
│  │ [Agentic RAG]    │  │ [Agentic RAG]    │  │ [Agentic RAG]    │ │
│  │ [Local VectorDB] │  │ [Local VectorDB] │  │ [Local VectorDB] │ │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘ │
│           │ PQC-TLS              │ PQC-TLS              │ PQC-TLS   │
└───────────┼──────────────────────┼──────────────────────┼───────────┘
            │   (weights only)     │                      │
            ▼                      ▼                      ▼
        ┌──────────────────────────────────────────────────┐
        │            CENTRAL AGGREGATOR                    │
        │                                                  │
        │  [Flower FL Server] → [FedAvg Aggregation]       │
        │  [Weight Validator] → [Global Model]             │
        │  [PQC Key Manager]  → [Key Rotation]             │
        └──────────────────────────────────────────────────┘
                        │
                        ▼
        ┌──────────────────────────────────────────────────┐
        │            BACKEND API (FastAPI)                 │
        │  /api/firewall  /api/threats  /api/federated     │
        └──────────────────────────────────────────────────┘
                        │
                        ▼
        ┌──────────────────────────────────────────────────┐
        │            DASHBOARD (React/Vite)                │
        │  Real-time threat map, RL metrics, FL rounds     │
        └──────────────────────────────────────────────────┘
```

## Data Flow

1. **Packet Ingestion**: Scapy captures raw packets on the NIC
2. **Flow Construction**: FlowManager groups packets into bidirectional 5-tuple flows
3. **Feature Extraction**: 41-dimensional NSL-KDD-compatible feature vector
4. **RL Decision**: DQN agent selects ALLOW / BLOCK / TRIGGER_RAG
5. **RAG Loop** (conditional): ChromaDB retrieval → LLM reasoning → rule generation
6. **Rule Application**: FirewallController applies iptables-compatible rules
7. **FL Export** (periodic): Model weights serialized, PQC-encrypted, sent to aggregator
8. **Global Update**: Aggregator runs FedAvg, broadcasts updated global weights

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Packet Capture | Scapy 2.5+ |
| RL | PyTorch + Gymnasium |
| Algorithm | DQN (Dueling + Prioritized Replay) |
| RAG | ChromaDB + LlamaIndex + Ollama |
| FL | Flower (flwr) |
| PQC | liboqs — Kyber768 (KEM), Dilithium3 (signatures) |
| Backend | FastAPI + SQLAlchemy |
| Dashboard | React + Vite + Recharts |
| Containers | Docker + Kubernetes |
| Dataset | NSL-KDD, CICIDS2017 |

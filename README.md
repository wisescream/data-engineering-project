# 🏗️ Insurance Claims Pipeline — Data Engineering & MLOps

> Pipeline ETL industrialisé + MLOps complet + API Production Quantifiée — Airflow + ClickHouse + dbt + MLflow + FastAPI + Prometheus

---

## 📐 Architecture Globale

```
                        ┌──── JOUR 1 : DATA ENGINEERING ────┐
                        │                                    │
  CSV Dataset ──→ MinIO (S3) ──→ Airflow DAG                │
                                    │                        │
                          Great Expectations (validation)     │
                                    │                        │
                               dbt (transformation)          │
                                    │                        │
                            ClickHouse (DWH)                 │
                                    │                        │
                         Slack Alerts + Dashboard            │
                        └────────────────────────────────────┘

                        ┌──── JOUR 2 : MLOps ───────────────┐
                        │                                    │
  fraud_claims.csv ──→ train.py (3 modèles)                  │
                             │                                │
                       MLflow Tracking                        │
                             │                                │
                       Model Registry (Dev/Staging/Prod)      │
                             │                                │
                       GitHub Actions CI/CD                   │
                        └────────────────────────────────────┘

                        ┌──── JOUR 3 : PRODUCTION ──────────┐
                        │                                    │
  Model Registry ──→ quantize.py (FP32 -> INT8)               │
                             │                                │
                       FastAPI Production API                 │
                             │                                │
                       Prometheus & Grafana (Monitoring)      │
                             │                                │
                       Load Testing (1000+ req/s)             │
                        └────────────────────────────────────┘

                        ┌──── JOUR 4 : GOUVERNANCE & TRUST ─┐
                        │                                    │
  Privacy Guard ──→ PII Masking (RGPD)                       │
                             │                                │
  Ethics Monitor ──→ Bias Detection                          │
                             │                                │
  Rule Engine ──→ Business Constraints                       │
                        └────────────────────────────────────┘

                        ┌──── JOUR 5 : SYSTEME D'AGENTS ────┐
                        │                                    │
  User Input ──→ CrewAI Orchestrator                         │
                             │                                │
  Tools ──→ [ClickHouse, FastAPI J3, Prom, RGPD J4]          │
                             │                                │
  Output ──→ Justified & Secure Response                     │
                        └────────────────────────────────────┘
```

---

## 📁 Structure du Projet

```
insurance-pipeline/
├── dags/
│   └── airflow_pipeline.py            # DAG ETL (5 tasks)
├── api/                                # ◀ JOUR 3 : Production
│   ├── main.py                        # FastAPI Production API
│   ├── quantize.py                    # Script de Quantification
│   ├── benchmark.py                   # Benchmark Performance
│   ├── schemas.py                     # Pydantic Models
│   └── Dockerfile
├── mlops/                              # ◀ JOUR 2 : CI/CD & Training
│   ├── train.py                       # Training Pipeline (3 modèles)
│   ├── deploy.py                      # Canary Deployment Simulator
│   └── generate_fraud_dataset.py      # Générateur dataset fraud
├── governance/                         # ◀ JOUR 4 : Gouvernance
│   ├── pii_masker.py                  # Anonymisation RGPD
│   └── bias_monitor.py                # Détection de biais
├── monitoring/
│   └── prometheus.yml                 # Config Monitoring
├── kubernetes/
│   └── deployment.yaml                # Manifests K8s
├── tests/
│   ├── test_mlops.py                  # Tests unitaires & intégration
│   └── load_test.jmx                  # JMeter Load Test
├── docker-compose.yml                 # Stack Full (J1-J5)
├── JOUR3_PRODUCTION_GUIDE.md          # Guide détaillé Jour 3
└── README.md

insurance-ai-agent/                     # ◀ JOUR 5 : AI Agent
├── api/
│   └── main.py                        # API de l'Agent (CrewAI)
├── ui/
│   └── app.py                         # Interface Streamlit
└── core/
    └── agents.py                      # Définition des agents & tools
```

---

## 🚀 Installation & Lancement Quickstart

### 1. Prérequis
*   Docker & Docker Compose
*   Python 3.11+
*   Compte Slack (pour les alertes)

### 2. Démarrage de l'Infrastructure (Jour 1)
```bash
docker compose up -d
```

### 3. Pipeline MLOps (Jour 2)
```bash
pip install -r mlops/requirements.txt
python -m mlops.generate_fraud_dataset
python -m mlops.train
```

### 4. Mise en Production (Jour 3)
```bash
pip install -r api/requirements.txt
python -m api.quantize
docker compose -f docker-compose-api.yml up -d
```

---

## 🔗 Accès aux Services

| Service | URL | Identifiants |
| :--- | :--- | :--- |
| **Airflow UI** | [http://localhost:8080](http://localhost:8080) | `admin` / `admin` |
| **MinIO Console** | [http://localhost:9001](http://localhost:9001) | `admin` / `password` |
| **MLflow UI** | [http://localhost:5000](http://localhost:5000) | (Accès libre) |
| **FastAPI J3** | [http://localhost:8000/docs](http://localhost:8000/docs) | (Swagger Prediction) |
| **Agent API J5** | [http://localhost:8001/docs](http://localhost:8001/docs) | (Swagger Agent) |
| **Streamlit Agent** | [http://localhost:8501](http://localhost:8501) | (UI Utilisateur) |
| **Prometheus** | [http://localhost:9090](http://localhost:9090) | (Metrics) |
| **Grafana** | [http://localhost:3000](http://localhost:3000) | `admin` / `admin` |

---

## 📊 Performances & Monitoring

### Quantification FP32 -> INT8
La quantification réduit la taille du modèle et accélère l'inférence sans perte d'accuracy.
*   **Compression** : ~2.0x plus léger (6MB -> 3MB)
*   **Accélération** : ~1.3x plus rapide (Latence réduite)
*   **Accuracy Delta** : < 0.1% (Négligeable)

### Monitoring Prometheus
Métriques exposées sur `/metrics` :
*   `fraud_api_request_count` : Volume de requêtes par endpoint/modèle.
*   `fraud_api_request_latency_seconds` : Distribution des latences.
*   `fraud_api_prediction_distribution` : Suivi du taux de fraude détecté en temps réel.

---

## 🧪 Tests & Qualité

```bash
# Lancer les 19 tests MLOps (Data, Preproc, Model, Integr)
pytest tests/test_mlops.py -v

# Lancer le benchmark de performance comparatif
python -m api.benchmark

# Load Testing (JMeter)
# Ouvrir tests/load_test.jmx dans JMeter
```

---

## 📝 Licence
Projet académique — TP Industrialisation de l'IA dans le Cloud.
Réalisé par l'équipe MLOps Assurance.

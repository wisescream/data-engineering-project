# 🏗️ Insurance Claims Pipeline — Data Engineering & MLOps

> Pipeline ETL industrialisé + MLOps complet pour l'assurance — Airflow + ClickHouse + dbt + Great Expectations + MLflow + CI/CD

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
                       (params, metrics, artifacts)          │
                            │                                │
                      Model Registry                         │
                       (Dev → Staging → Prod)                │
                            │                                │
                      deploy.py (Canary 5→25→50→100%)        │
                            │                                │
                      GitHub Actions CI/CD                   │
                        └────────────────────────────────────┘
```

### Stack Technique

| Outil              | Rôle                              |
| ------------------ | --------------------------------- |
| Apache Airflow     | Orchestration du pipeline ETL     |
| ClickHouse         | Data Warehouse analytique         |
| MinIO              | Stockage S3 local                 |
| dbt                | Transformation SQL                |
| Great Expectations | Validation qualité des données    |
| **MLflow**         | **Tracking & Model Registry**     |
| **scikit-learn**   | **Entraînement ML (RF, SVM)**     |
| **XGBoost**        | **Gradient Boosting**             |
| **GitHub Actions** | **CI/CD pipeline**                |
| Docker Compose     | Conteneurisation                  |
| Slack              | Notifications                     |

---

## 📁 Structure du Projet

```
insurance-pipeline/
├── dags/
│   └── airflow_pipeline.py            # DAG ETL (5 tasks)
├── data/
│   ├── sample_claims.csv              # Dataset ETL
│   └── fraud_claims.csv               # Dataset ML (fraud detection)
├── dbt/
│   ├── models/
│   │   ├── claims_transformed.sql
│   │   ├── schema.yml
│   │   └── sources.yml
│   ├── dbt_project.yml
│   └── profiles.yml
├── great_expectations/
│   └── great_expectations.yml
├── mlops/                              # ◀ JOUR 2
│   ├── __init__.py
│   ├── generate_fraud_dataset.py      # Générateur dataset fraud
│   ├── train.py                       # Pipeline d'entraînement (3 modèles)
│   ├── deploy.py                      # Déploiement canary + API
│   └── requirements.txt
├── scripts/
│   ├── generate_dataset.py
│   ├── create_table.sql
│   └── dashboard.py                   # Dashboard Streamlit
├── tests/
│   ├── test_pipeline.py               # Tests ETL (Jour 1)
│   └── test_mlops.py                  # Tests MLOps (Jour 2)
├── .github/
│   └── workflows/
│       └── ml-pipeline.yml            # CI/CD GitHub Actions
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🚀 Installation & Lancement

### Prérequis

- [Docker Desktop](https://www.docker.com/products/docker-desktop)
- Python 3.11+
- Git

### 1. Cloner le projet

```bash
git clone <repo-url>
cd insurance-pipeline
```

### 2. Lancer l'infrastructure Docker (Jour 1)

```bash
docker compose up -d
```

### 3. Installer les dépendances MLOps (Jour 2)

```bash
pip install -r mlops/requirements.txt
```

### 4. Vérifier les services

| Service         | URL                    | Credentials            |
| --------------- | ---------------------- | ---------------------- |
| Airflow UI      | http://localhost:8080   | admin / admin          |
| MinIO Console   | http://localhost:9001   | minioadmin / minioadmin|
| ClickHouse HTTP | http://localhost:8123   | default / (vide)       |
| Dashboard       | http://localhost:8501   | (Accès libre)          |
| MLflow UI       | http://localhost:5000   | (après `mlflow ui`)    |

---

# 📅 JOUR 1 — Data Engineering

## 🔄 Pipeline DAG

```
extract_from_s3 → validate_schema → transform_data → load_to_clickhouse → send_notification
```

| Task               | Détail                                         |
| ------------------ | ---------------------------------------------- |
| Extract from S3    | Télécharge CSV depuis MinIO                    |
| Validate Schema    | Great Expectations (nulls, ranges, statuts)    |
| Transform Data     | dbt + Python (uppercase, filtre, catégories)   |
| Load to ClickHouse | Insertion `insurance.claims_transformed`        |
| Send Notification  | Slack avec métriques (rows, null%, duration)   |

## 📊 Monitoring

- **Freshness SLA** : Alerte Slack si pipeline > 1h (`sla_miss_callback`)
- **Completeness** : `(rows_loaded / rows_extracted) * 100`
- **Quality metrics** : `null_percentage`, `invalid_percentage`
- **Dashboard** : Streamlit branché sur ClickHouse (port 8501)

---

# 📅 JOUR 2 — MLOps

## 🧠 Use Case : Détection de Fraude sur Sinistres

Target : `fraud_reported` (0 = légitime, 1 = fraude)

### 1. Training Pipeline (`mlops/train.py`)

```bash
# Générer le dataset fraud
python -m mlops.generate_fraud_dataset

# Lancer l'entraînement
python -m mlops.train
```

**3 modèles entraînés :**

| Modèle        | Accuracy | F1-Score | ROC-AUC |
| ------------- | -------- | -------- | ------- |
| RandomForest  | 0.9380   | 0.8000   | 0.9789  |
| XGBoost       | 0.9310   | 0.7877   | 0.9775  |
| SVM           | 0.8770   | 0.6284   | 0.9077  |

**Champion sélectionné automatiquement** sur le meilleur F1-Score.

### 2. MLflow Tracking

Chaque modèle enregistre dans MLflow :

```python
mlflow.set_experiment("insurance-fraud")
mlflow.log_param("model_type", "RandomForest")
mlflow.log_param("n_estimators", 200)
mlflow.log_metric("accuracy", 0.938)
mlflow.log_metric("f1_score", 0.80)
mlflow.log_metric("roc_auc", 0.979)
mlflow.log_artifact("confusion_matrix.png")
mlflow.sklearn.log_model(model, "model")
```

Lancer l'interface MLflow :

```bash
mlflow ui --port 5000
```

### 3. Model Registry

```
                    ┌─────────────┐
                    │ Development │
                    └──────┬──────┘
                           ↓
                    ┌─────────────┐
                    │   Staging   │
                    └──────┬──────┘
                           ↓
                    ┌─────────────┐
                    │ Production  │  ← Champion automatique
                    └─────────────┘
```

- Le meilleur modèle est automatiquement enregistré sous `insurance-fraud-detector`
- Transition automatique : Development → Staging → Production

### 4. CI/CD GitHub Actions (`.github/workflows/ml-pipeline.yml`)

```
┌──────────┐     ┌──────────┐     ┌────────────────────┐
│   Lint   │ ──→ │  Tests   │ ──→ │ Train & Register   │
│ (flake8) │     │ (pytest) │     │ (3 models + gate)  │
└──────────┘     └──────────┘     └────────────────────┘
```

Le pipeline CI/CD :
1. **Lint** — Validation du code avec flake8
2. **Tests unitaires** — pytest (data, preprocessing, modèles)
3. **Training** — Entraîne les 3 modèles
4. **Accuracy Gate** — Bloque si accuracy < 0.75
5. **Register** — Enregistre le champion dans MLflow

### 5. Canary Deployment (`mlops/deploy.py`)

```bash
python -m mlops.deploy
```

Déploiement progressif simulé :

```
 5% traffic ──→ vérification accuracy ──→ OK
25% traffic ──→ vérification accuracy ──→ OK
50% traffic ──→ vérification accuracy ──→ OK
100% traffic ──→ DEPLOYED
```

**Rollback automatique** si accuracy < 0.70 à n'importe quel palier.

### 6. Testing MLOps

```bash
# Tous les tests (19 tests)
pytest tests/test_mlops.py -v

# Tests unitaires uniquement
pytest tests/test_mlops.py -v -k "not TestIntegration"

# Tests d'intégration
pytest tests/test_mlops.py -v -k "TestIntegration"
```

| Suite                      | Tests | Description                         |
| -------------------------- | ----- | ----------------------------------- |
| TestDataGeneration         | 6     | Validité du dataset fraud           |
| TestPreprocessing          | 4     | Scaling, features, balance          |
| TestModelTraining          | 4     | RF, SVM, confusion matrix           |
| TestIntegrationPipeline    | 2     | Pipeline E2E complet                |
| TestModelAccuracyValidation| 3     | Accuracy > 0.75, F1, AUC gates      |

---

## 🔐 Exécution Complète

### Jour 1 — ETL
1. `docker compose up -d`
2. Ouvrir http://localhost:8080
3. Activer le DAG `insurance_pipeline` → Trigger

### Jour 2 — MLOps
1. `pip install -r mlops/requirements.txt`
2. `python -m mlops.generate_fraud_dataset`
3. `python -m mlops.train`
4. `mlflow ui --port 5000` → Ouvrir http://localhost:5000
5. `python -m mlops.deploy`
6. `pytest tests/test_mlops.py -v`

---

## 🛑 Arrêt

```bash
docker compose down           # Arrêter les services
docker compose down -v        # Arrêter + supprimer les volumes
```

---

## 📝 Licence

Projet académique — TP Industrialisation de l'IA dans le Cloud.

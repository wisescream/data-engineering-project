# 📅 JOUR 3 — Mise en Production & Performance (MLOps Senior)

Ce document résume l'industrialisation finale du modèle de détection de fraude assurance, passant d'un modèle entraîné à un service de production scalable et monitoré.

## 📐 Architecture de Production

L'infrastructure est conçue pour la haute performance et l'observabilité :

1.  **FastAPI Application** : Serveur asynchrone haute performance gérant l'inférence.
2.  **MLflow Model Registry** : Source de vérité pour les modèles (`champion` model).
3.  **Quantization Engine** : Conversion FP32 (Full Precision) → INT8 (Quantized) pour réduire la latence et l'empreinte mémoire.
4.  **Prometheus & Grafana** : Monitoring en temps réel des performances techniques et métiers.
5.  **Docker Compose / Kubernetes** : Orchestration pour la scalabilité horizontale.

---

## 🛠️ Composants Livrés

### 1. API Production (`api/main.py`)
*   **Multi-modèles** : Supporte l'inférence en FP32 et INT8.
*   **Validation Pydantic** : Typage strict des entrées/sorties pour éviter les données corrompues.
*   **Instrumentation Prometheus** : Metrics natives (`/metrics`) exposant le débit, la latence et le taux d'erreur.

### 2. Quantization & Benchmarking (`api/quantize.py`, `api/benchmark.py`)
*   **Technique** : Utilisation de la réduction de profondeur et de la précision des seuils (float16 simulation) pour le RandomForest.
*   **Gain Performance** : Réduction de la taille du modèle de ~2x à 4x et accélération de l'inférence.
*   **Rapport de comparaison** : Génération d'un JSON comparant l'accuracy vs la vitesse pour aider au choix du modèle.

### 3. Stack Monitoring (`monitoring/prometheus.yml`)
*   Configuration de Prometheus pour scraper l'API toutes les 5 secondes.
*   Prêt pour Grafana (Admin pass: `admin`) pour visualiser les P95 latencies.

### 4. Déploiement & Scalabilité
*   **Dockerfile** : Image optimisée `python:3.11-slim` avec healthchecks intégrés.
*   **Docker Compose** : Déploiement local de toute la stack en une commande.
*   **Kubernetes** : Manifests de déploiement avec 3 replicas et auto-healing.

---

## 🚀 Guide d'Exécution

### 1. Préparer les modèles
Assurez-vous que le modèle est entraîné (Jour 2), puis lancez la quantification :
```bash
python -m api.quantize
```

### 2. Lancer la stack de production (Docker)
```bash
docker-compose -f docker-compose-api.yml up -d
```

### 3. Benchmark de performance
Testez la différence entre FP32 et INT8 en local :
```bash
python -m api.benchmark
```

### 4. Monitoring
*   **API Health** : [http://localhost:8000/health](http://localhost:8000/health)
*   **Prometheus** : [http://localhost:9090](http://localhost:9090)
*   **Grafana** : [http://localhost:3000](http://localhost:3000)

---

## 📈 Checklist pour la Démo Finale (Professeur)

- [ ] **Inférence Temps Réel** : Faire un POST sur `/predict` avec un JSON.
- [ ] **Quantization Demo** : Montrer que `/predict-int8` est plus rapide que `/predict-fp32`.
- [ ] **Observabilité** : Montrer les metrics monter dans Prometheus après un test de charge.
- [ ] **Scalabilité** : Montrer les 3 replicas tournant dans Kubernetes (ou simulé par Docker).
- [ ] **Load Test** : Montrer que l'API tient > 1000 req/s avec JMeter.

---

## 💡 Notes pour le Professeur (Justification)
*   **Pourquoi FastAPI ?** Pour sa nature asynchrone qui permet de gérer des milliers de requêtes concurrentes sans bloquer le CPU sur l'E/S.
*   **Pourquoi la Quantization ?** Dans un contexte d'assurance avec des millions de sinistres, réduire la latence de 5ms à 1ms permet des économies d'infrastructure massives.
*   **Pourquoi Prometheus ?** Pour détecter le "Model Drift" et les ralentissements de l'API avant qu'ils ne touchent les utilisateurs.

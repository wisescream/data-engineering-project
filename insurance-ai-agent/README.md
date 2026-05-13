# 🤖 SANLAM AI Agent - Intelligent Orchestration (Jour 5)

Ce module constitue le cerveau de la plateforme **SANLAM**. Il utilise des agents autonomes pour orchestrer les données (J1), les modèles (J2/J3) et les règles de gouvernance (J4).

---

## 🧠 Architecture de l'Agent
L'agent est construit avec **CrewAI** et **LangChain**, lui permettant de raisonner et d'utiliser des outils pour répondre aux besoins métier complexes :

1.  **Orchestrateur (CrewAI)** : Gère la délégation des tâches entre agents.
2.  **Tools (Outils)** :
    *   `get_claim_history` : Connexion directe à **ClickHouse** pour l'historique SQL.
    *   `predict_fraud_j3` : Appel à l'API **FastAPI** optimisée (INT8).
    *   `compliance_check_j4` : Validation via le moteur de gouvernance RGPD.
    *   `system_health` : Récupération des métriques depuis **Prometheus**.

---

## 🔗 Intégration Holistique (Full Cycle)
L'agent ne travaille pas en isolation. Il est le point de convergence des 4 jours précédents :
*   **Data (J1)** : Requêtes analytiques sur les données transformées par dbt.
*   **MLOps (J2)** : Sélection automatique des meilleurs modèles via MLflow.
*   **Serving (J3)** : Consommation de l'inférence ultra-rapide (Quantization).
*   **Trust (J4)** : Filtrage PII et détection de biais systématique avant toute réponse.

---

## 🚀 Installation & Lancement
L'agent peut être lancé via Docker (recommandé) ou manuellement.

### Option 1 : Docker (Stack Complète)
```bash
# Dans la racine du projet
docker compose up -d
```

### Option 2 : Manuel
```bash
# Dans le dossier insurance-ai-agent
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8001
streamlit run ui/app.py --server.port 8501
```

---

## 🎯 Démonstration / Pitch
> "Notre agent ne se contente pas de prédire. Il **justifie** ses décisions en croisant l'historique ClickHouse, valide sa prédiction via notre modèle de production, et garantit la conformité RGPD avant de s'adresser à l'utilisateur."

---

## 📝 Licence & Auteurs
Composant du projet **Insurance AI Pipeline**.
Réalisé par l'équipe Data & AI.

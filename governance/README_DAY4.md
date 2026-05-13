# 🏛️ Day 4: Governance, Security & Responsible AI

Ce module implémente la couche finale de protection et de conformité pour le pipeline de détection de fraude.

## 🔒 Cloud Security & Zero Trust
- **IAM Policies**: Définition de rôles par métier (DataScientist, DevOps, Analyst) avec accès limité.
- **SSE-KMS**: Configuration du chiffrement côté serveur pour les buckets S3.
- **RBAC**: Contrôle d'accès basé sur les rôles simulé pour l'API.

## 🇪🇺 RGPD & Privacy
Le script `anonymization.py` assure :
1. **Data Minimization**: Suppression des colonnes PII (Nom, Email).
2. **Pseudonymisation**: Hashing salé (SHA256) des adresses IP.
3. **Differential Privacy**: Ajout de bruit de Laplace sur les montants de sinistres pour empêcher la ré-identification par corrélation.

## 🧠 Responsible AI (XAI & Fairness)
- **SHAP**: Importance globale des variables pour justifier les décisions du modèle aux régulateurs.
- **LIME**: Explication locale d'une prédiction spécifique (ex: "Pourquoi ce dossier a été marqué comme fraude ?").
- **Fairness Audit**: Analyse des disparités de prédiction par région pour éviter les biais discriminatoires.

## 💰 FinOps
- **Cost Monitoring**: Configuration de budgets et de tags pour le suivi des dépenses Cloud.
- **Alerting**: Seuils à 50%, 80% et 100% du budget mensuel (500$).

---
*Projet réalisé dans le cadre du TP Industriel MLOps - Jour 4*

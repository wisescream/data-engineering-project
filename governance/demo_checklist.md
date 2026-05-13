# ✅ Checklist de Démo Finale

Utilisez cette liste pour valider votre présentation devant le professeur.

### 1. Sécurité & IAM
- [ ] Afficher les fichiers JSON dans `governance/iam/policies/`.
- [ ] Expliquer pourquoi le DataScientist n'a pas le droit de supprimer un modèle en Prod (`Deny` sur `DeleteModel`).

### 2. RGPD / Privacy
- [ ] Exécuter `python governance/privacy/anonymization.py`.
- [ ] Montrer le fichier `anonymized_dataset.csv` et souligner que l'IP est devenue un hash illisible.

### 3. Explainability (XAI)
- [ ] Ouvrir l'image `governance/xai/reports/shap_summary.png`.
- [ ] Expliquer quelle feature (ex: `claim_amt`) a le plus d'impact sur la fraude.
- [ ] Ouvrir le fichier HTML `lime_explanation.html` pour montrer une explication locale.

### 4. Fairness Audit
- [ ] Ouvrir `governance/fairness/fairness_report.md`.
- [ ] Montrer le tableau des disparités par région.

### 5. FinOps
- [ ] Montrer `governance/cost/cost_config.json` et expliquer la stratégie d'alerte à 80%.

### 6. Docker Security
- [ ] Montrer dans le `Dockerfile` (Jour 3) l'utilisation d'un utilisateur `non-root` (`USER appuser`).

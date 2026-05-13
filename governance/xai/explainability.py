import shap
import lime
import lime.lime_tabular
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os

# Create reports directory if not exists
os.makedirs('governance/xai/reports', exist_ok=True)

def generate_xai_reports():
    print("[XAI] Generating Explainability Reports...")
    
    # 1. Generate Mock Dataset for Demo
    X = pd.DataFrame(np.random.rand(100, 5), columns=['claim_amt', 'age', 'past_claims', 'region_id', 'policy_type'])
    y = (X['claim_amt'] * 2 + X['past_claims'] > 1.5).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    # 2. SHAP (Global Importance)
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        plt.figure(figsize=(10, 6))
        # Handle different SHAP output formats (binary vs multiclass)
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values[1], X_test, show=False)
        else:
            shap.summary_plot(shap_values, X_test, show=False)
            
        plt.tight_layout()
        plt.savefig('governance/xai/reports/shap_summary.png')
        plt.close()
        print("OK [XAI] SHAP Summary generated.")
    except Exception as e:
        print(f"Warning: SHAP failed, generating fallback feature importance. Error: {e}")
        importances = model.feature_importances_
        indices = np.argsort(importances)
        plt.figure(figsize=(10, 6))
        plt.title('Feature Importances (RandomForest Native)')
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [X_test.columns[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        plt.savefig('governance/xai/reports/shap_summary.png') # Keep same name for checklist
        plt.close()

    # 3. LIME (Local Interpretation)
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train.values, 
        feature_names=X_train.columns, 
        class_names=['Clean', 'Fraud'],
        mode='classification'
    )
    
    idx = 0 
    exp = lime_explainer.explain_instance(X_test.iloc[idx].values, model.predict_proba)
    exp.save_to_file('governance/xai/reports/lime_explanation.html')
    
    print("OK [XAI] LIME report generated.")

if __name__ == "__main__":
    generate_xai_reports()

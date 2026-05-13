import requests
import time
import concurrent.futures
import json

BASE_URL = "http://localhost:8000"
SAMPLE_DATA = {
    "age": 42.0, "claim_amount": 25000.0, "policy_annual_premium": 1450.0,
    "months_as_customer": 240.0, "num_previous_claims": 2.0,
    "police_report_filed": 1.0, "witnesses": 3.0,
    "injury_claim": 8000.0, "vehicle_claim": 17000.0, "total_claim_amount": 25000.0,
    "insurance_type_encoded": 1.0, "region_encoded": 3.0
}

def run_step(name, func):
    print(f"\n🚀 STEP: {name}")
    func()

def test_single_prediction():
    print("Envoi d'une requête sur /predict...")
    response = requests.post(f"{BASE_URL}/predict", json=SAMPLE_DATA)
    print(f"Résultat: {json.dumps(response.json(), indent=2)}")

def compare_quantization():
    print("Comparaison FP32 vs INT8 (10 requêtes)...")
    latencies_fp32 = [requests.post(f"{BASE_URL}/predict-fp32", json=SAMPLE_DATA).json()['latency_ms'] for _ in range(10)]
    latencies_int8 = [requests.post(f"{BASE_URL}/predict-int8", json=SAMPLE_DATA).json()['latency_ms'] for _ in range(10)]
    print(f"  Latence Moyenne FP32 : {sum(latencies_fp32)/10:.2f} ms")
    print(f"  Latence Moyenne INT8 : {sum(latencies_int8)/10:.2f} ms")

def stress_test():
    total_reqs = 50  # Version allégée
    workers = 5      # Plus fluide
    print(f"Lancement du Stress Test ({total_reqs} requêtes)...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(lambda _: requests.post(f"{BASE_URL}/predict", json=SAMPLE_DATA), range(total_reqs)))
    print("✅ Terminé ! Allez voir Grafana.")

if __name__ == "__main__":
    run_step("Inférence", test_single_prediction)
    run_step("Benchmark", compare_quantization)
    run_step("Stress Test", stress_test)

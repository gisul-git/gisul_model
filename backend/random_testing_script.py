import requests
import time
import random
import json
from datetime import datetime

# =========================
# CONFIG
# =========================
BASE_URL = "http://192.168.1.11:9000/api/v1"
GENERATE_URL = f"{BASE_URL}/generate-aiml-library"

OUTPUT_FILE = "random_test_results.json"

POLL_INTERVAL = 3

# =========================
# RANDOM POOL
# =========================
TOPICS = [
    "Fraud Detection",
    "Stock Price Prediction",
    "Customer Segmentation",
    "Medical Diagnosis",
    "Speech Recognition",
    "Recommendation System",
    "Autonomous Driving",
    "Object Detection",
    "Chatbot NLP",
    "Time Series Forecasting",
    "Anomaly Detection",
    "Face Recognition",
    "Sentiment Analysis",
    "Image Captioning",
    "Machine Translation",
]

CONCEPTS = [
    "classification",
    "clustering",
    "deep learning",
    "nlp",
    "computer vision",
    "time series",
    "anomaly detection",
    "reinforcement learning"
]

DIFFICULTIES = ["easy", "medium", "hard"]

# =========================
# LOGGER
# =========================
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# =========================
# LOAD DATA
# =========================
def load_data():
    try:
        with open(OUTPUT_FILE, "r") as f:
            return json.load(f)
    except:
        return []

# =========================
# SAVE DATA
# =========================
def save_data(data):
    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=2)

# =========================
# CREATE JOB
# =========================
def create_job(payload):
    try:
        res = requests.post(GENERATE_URL, json=payload, timeout=120)

        if res.status_code != 200:
            log(f"❌ HTTP {res.status_code}: {res.text}")
            return None

        return res.json()

    except Exception as e:
        log(f"❌ Request failed: {e}")
        return None

# =========================
# POLL JOB
# =========================
def poll_job(job_id):
    url = f"{BASE_URL}/job/{job_id}"

    while True:
        try:
            res = requests.get(url, timeout=60)

            if res.status_code == 404:
                time.sleep(POLL_INTERVAL)
                continue

            data = res.json()
            status = data.get("status")

            if status == "complete":
                log("✅ Completed")
                return data.get("result")

            elif status in ("failed", "error"):
                log(f"❌ Failed: {data}")
                return None

            else:
                log("⏳ Processing...")

        except requests.exceptions.ReadTimeout:
            log("⏳ still processing...")

        except Exception as e:
            log(f"⚠️ Poll error: {e}")

        time.sleep(POLL_INTERVAL)

# =========================
# MAIN TEST
# =========================
def run_random_tests(n=10):
    log(f"🚀 Running {n} random tests\n")

    results = load_data()

    for i in range(n):
        topic = random.choice(TOPICS)
        concept = random.choice(CONCEPTS)
        difficulty = random.choice(DIFFICULTIES)

        log(f"👉 Test {i+1}: {topic} | {concept} | {difficulty}")

        payload = {
            "topic": topic,
            "difficulty": difficulty,
            "concepts": [concept],
            "use_cache": False
        }

        response = create_job(payload)

        if not response:
            continue

        if "job_id" in response:
            job_id = response["job_id"]
            log(f"🆔 Job: {job_id}")
            result = poll_job(job_id)
        else:
            log("⚡ Direct response")
            result = response.get("result")

        # =========================
        # SAVE RESULT
        # =========================
        if result:
            results.append({
                "topic": topic,
                "concept": concept,
                "difficulty": difficulty,
                "response": result
            })

            save_data(results)
            log("💾 Saved")

        else:
            log("❌ No result")

        log("--------------------------------------------------")


# =========================
# RUN
# =========================
if __name__ == "__main__":
    run_random_tests(n=15)
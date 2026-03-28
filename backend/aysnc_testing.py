import requests
import time
import json
from datetime import datetime

# =========================
# CONFIG
# =========================
BASE_URL = "http://192.168.1.11:9000/api/v1"
GENERATE_URL = f"{BASE_URL}/generate-aiml-library"

OUTPUT_FILE = "aiml_dataset2.json"

# =========================
# TOPIC → CONCEPT MAPPING (🔥 IMPORTANT)
# =========================
TOPIC_CONCEPT_MAP = {
    "Iris": ["classification", "clustering"],
    "MNIST": ["classification", "deep learning"],
    "CIFAR-10": ["classification", "computer vision"],
    "Image Classification": ["computer vision", "deep learning"],
    "Text Classification": ["nlp", "classification"],
    "Sentiment Analysis": ["nlp", "classification"],
    "K-Means": ["clustering"],
    "Decision Trees": ["classification", "regression"],
    "Random Forest": ["classification", "regression"],
    "Neural Networks": ["deep learning"],
    "CNN": ["deep learning", "computer vision"],
    "RNN": ["deep learning", "nlp"],
    "Transformers": ["deep learning", "nlp"],
    "BERT": ["nlp"],
    "GAN": ["deep learning"],
    "Autoencoders": ["deep learning"],
    "PCA": ["dimensionality reduction"],
    "t-SNE": ["dimensionality reduction"],
    "Feature Engineering": ["feature engineering"],
    "Model Evaluation": ["model evaluation"],
    "Hyperparameter Tuning": ["hyperparameter tuning"],
    "Time Series Forecasting": ["time series"],
    "Anomaly Detection": ["anomaly detection"],
    "Reinforcement Learning": ["reinforcement learning"],
}

DIFFICULTIES = ["easy", "medium", "hard"]

POLL_INTERVAL = 3

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
    start = time.time()

    while True:
        try:
            res = requests.get(url, timeout=60)

            if res.status_code == 404:
                time.sleep(POLL_INTERVAL)
                continue

            data = res.json()
            status = data.get("status")

            elapsed = round(time.time() - start, 2)

            if status == "complete":
                log(f"✅ Completed in {elapsed}s")
                return data.get("result")

            elif status in ("failed", "error"):
                log(f"❌ Failed: {data}")
                return None

            else:
                log(f"⏳ [{elapsed}s] processing...")

        except requests.exceptions.ReadTimeout:
            log("⏳ still processing (long task)...")

        except Exception as e:
            log(f"⚠️ Poll error: {e}")

        time.sleep(POLL_INTERVAL)

# =========================
# MAIN
# =========================
def main():
    results = load_data()

    total_tasks = sum(len(v) for v in TOPIC_CONCEPT_MAP.values()) * len(DIFFICULTIES)
    count = 0

    log(f"🚀 Starting AIML dataset generation ({total_tasks} tasks)\n")

    for topic, concepts_list in TOPIC_CONCEPT_MAP.items():
        for concept in concepts_list:
            for difficulty in DIFFICULTIES:

                count += 1
                log(f"👉 [{count}/{total_tasks}] {topic} | {concept} | {difficulty}")

                payload = {
                    "topic": topic,
                    "difficulty": difficulty,
                    "concepts": [concept],
                    "use_cache": False
                }

                response = create_job(payload)

                if not response:
                    continue

                # HANDLE RESPONSE
                if response.get("status") == "complete":
                    result = response.get("result")
                    log("⚡ Direct result")

                elif "job_id" in response:
                    job_id = response["job_id"]
                    log(f"🆔 Job created: {job_id}")
                    result = poll_job(job_id)

                else:
                    log(f"❌ Unexpected response: {response}")
                    continue

                # SAVE
                if result:
                    results.append({
                        "topic": topic,
                        "concept": concept,
                        "difficulty": difficulty,
                        "response": result
                    })

                    save_data(results)
                    log("💾 Saved")

                log("--------------------------------------------------")

    log("🎉 ALL DATA GENERATED!")

# =========================
# RUN
# =========================
if __name__ == "__main__":
    main()
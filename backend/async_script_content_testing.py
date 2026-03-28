import requests
import time
import json
from datetime import datetime

# =========================
# CONFIG
# =========================
BASE_URL = "http://192.168.1.11:9000/api/v1"
GENERATE_URL = f"{BASE_URL}/generate-aiml-library"

OUTPUT_FILE = "aiml_dataset.json"

# 🔥 FULL TOPIC LIST
TOPICS = [
    "Iris", "MNIST", "CIFAR-10", "Image Classification",
    "Object Detection", "Regression", "Clustering",
    "K-Means", "DBSCAN", "Hierarchical Clustering",
    "Decision Trees", "Random Forest", "XGBoost",
    "Neural Networks", "CNN", "RNN", "LSTM",
    "Transformers", "BERT", "GPT",
    "Recommendation Systems", "Time Series Forecasting",
    "Anomaly Detection", "Feature Engineering",
    "Dimensionality Reduction", "PCA", "t-SNE",
    "Autoencoders", "GAN", "Reinforcement Learning",
    "Q-Learning", "Deep Q Networks",
    "Natural Language Processing", "Sentiment Analysis",
    "Named Entity Recognition", "Text Classification",
    "Speech Recognition", "Computer Vision",
    "Transfer Learning", "Fine Tuning",
    "Hyperparameter Tuning", "Model Evaluation",
    "Cross Validation", "Overfitting", "Regularization"
]

DIFFICULTIES = ["easy", "medium", "hard"]

POLL_INTERVAL = 3

# =========================
# LOGGER
# =========================
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# =========================
# LOAD EXISTING DATA
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
# POLL JOB (STREAMLIT STYLE)
# =========================
def poll_job(job_id):
    url = f"{BASE_URL}/job/{job_id}"
    start = time.time()

    while True:
        elapsed = int(time.time() - start)

        try:
            res = requests.get(url, timeout=30)

            if res.status_code == 404:
                log("⚠️ Job not ready yet...")
                time.sleep(POLL_INTERVAL)
                continue

            data = res.json()
            status = data.get("status")

            if status == "complete":
                log(f"✅ Completed in {elapsed}s")
                return data.get("result")

            elif status in ("failed", "error"):
                log(f"❌ Failed: {data}")
                return None

            else:
                log(f"⏳ [{elapsed}s] processing...")

        except Exception as e:
            log(f"⚠️ Poll error: {e}")

        time.sleep(POLL_INTERVAL)

# =========================
# MAIN
# =========================
def main():
    results = load_data()

    total = len(TOPICS) * len(DIFFICULTIES)
    count = 0

    log(f"🚀 Starting AIML dataset generation ({total} tasks)\n")

    for topic in TOPICS:
        for difficulty in DIFFICULTIES:

            count += 1
            log(f"👉 [{count}/{total}] {topic} | {difficulty}")

            payload = {
                "topic": topic,
                "difficulty": difficulty,
                "concepts": ["classification"],
                "use_cache": False   # 🔥 dataset generation
            }

            response = create_job(payload)

            if not response:
                continue

            # =========================
            # HANDLE RESPONSE
            # =========================
            if response.get("status") == "complete":
                result = response.get("result")
                log("⚡ Direct result received")

            elif "job_id" in response:
                job_id = response["job_id"]
                log(f"🆔 Job created: {job_id}")
                result = poll_job(job_id)

            else:
                log(f"❌ Unexpected response: {response}")
                continue

            # =========================
            # SAVE RESULT
            # =========================
            if result:
                results.append({
                    "topic": topic,
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
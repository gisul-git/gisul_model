FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

WORKDIR /app

COPY . .

RUN apt update && apt install -y python3 python3-pip

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 9000

CMD ["uvicorn", "qwen_model_server_script_multiple:app", "--host", "0.0.0.0", "--port", "9000"]

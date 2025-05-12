model_path=mistralai/Ministral-8B-Instruct-2410
tensor_parallel_size=2
cd /root/autodl-tmp
python -m vllm.entrypoints.openai.api_server \
  --model $model_path \
  --tensor-parallel-size $tensor_parallel_size \
  --port 6006 \
  --dtype=half
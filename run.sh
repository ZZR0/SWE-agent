# python run_docker_builder.py --image_name zzr/swe-env:latest
# docker images | grep '^zzr/swe-env--' | awk '{print $1 ":" $2}' | xargs -I {} docker rmi {}

python run.py --model_name vllm-llama3-70b \
  --container_name sweagent-swe-agent-latest-240420 \
  --per_instance_cost_limit 2.00 \
  --config_file ./config/default.yaml \
  --instance_filter "^(?!pvlib__pvlib-python).*"  \
  # --instance_filter marshmallow-code__marshmallow-1343  \

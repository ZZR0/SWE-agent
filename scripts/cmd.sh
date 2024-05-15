python run_docker_builder.py --image_name zzr/swe-env:latest
docker images | grep '^zzr/swe-env--' | awk '{print $1 ":" $2}' | xargs -I {} docker rmi {}
docker images | grep '<none>' | awk '{print $3}' | xargs -I {} docker rmi {}
docker build --no-cache --network host --build-arg all_proxy=http://127.0.0.1:10809 -t zzr/swe-env:latest -f docker/swe-env.Dockerfile .

python scripts/run_with_image.py --data_path /hdd2/zzr/SWE-agent/dataset/swebench_lite_dev.json
python scripts/valid_with_image.py --data_path /hdd2/zzr/SWE-agent/dataset/swebench_lite_dev.json

python /data1/zengzhengran/SWE-agent/evaluation/evaluation.py --predictions_path "/data1/zengzhengran/SWE-agent/trajectories/zengzhengran/vllm-llama3-70b__SWE-bench_Lite__default__t-0.00__p-0.95__c-2.00__install-1/all_preds.jsonl" --swe_bench_tasks "/data1/zengzhengran/SWE-agent/dataset/swebench_lite_dev.json" --log_dir "/data1/zengzhengran/SWE-agent/evaluation/log" --testbed "/data1/zengzhengran/SWE-agent/evaluation/testbed" --skip_existing --timeout 900 --verbose


python run_docker_builder.py --image_name zzr/swe-env:latest
docker images | grep '^zzr/swe-env--' | awk '{print $1 ":" $2}' | xargs -I {} docker rmi {}
docker images | grep '<none>' | awk '{print $3}' | xargs -I {} docker rmi {}
docker build --no-cache --network host --build-arg all_proxy=http://192.168.100.211:10809 -t zzr/swe-env:latest -f docker/swe-env.Dockerfile .

python scripts/run_with_image.py --data_path /hdd2/zzr/SWE-agent/dataset/swebench_lite_dev.json

# engine_validation
python /data1/zengzhengran/sweTrans_yang/SWE-agent/scripts/valid_with_image.py --data_path /data1/zengzhengran/sweTrans_yang/SWE-agent/dataset/swebench_lite_dev.json

python /data1/zengzhengran/sweTrans_yang/SWE-agent/evaluation/run_evaluation.py --predictions_path "/data1/zengzhengran/sweTrans_yang/SWE-agent/trajectories/zengzhengran/vllm-llama3-70b__SWE-bench_Lite__default__t-0.00__p-0.95__c-2.00__install-1/all_preds.jsonl" --swe_bench_tasks "/data1/zengzhengran/sweTrans_yang/SWE-agent/dataset/swebench_lite_dev.json" --log_dir "/data1/zengzhengran/sweTrans_yang/SWE-agent/evaluation/log" --testbed "/data1/zengzhengran/sweTrans_yang/SWE-agent/evaluation/testbed" --skip_existing --timeout 900 --verbose


python /data1/zengzhengran/sweTrans_yang/SWE-agent/evaluation/evaluation.py --predictions_path "/data1/zengzhengran/sweTrans_yang/SWE-agent/trajectories/zengzhengran/vllm-llama3-70b__SWE-bench_Lite__default__t-0.00__p-0.95__c-2.00__install-1/all_preds.jsonl" --swe_bench_tasks "/data1/zengzhengran/sweTrans_yang/SWE-agent/dataset/swebench_lite_dev.json" --log_dir "/data1/zengzhengran/sweTrans_yang/SWE-agent/evaluation/log" --testbed "/data1/zengzhengran/sweTrans_yang/SWE-agent/evaluation/testbed" --skip_existing --timeout 900 --verbose

docker run --rm -it --network host -e ALL_PROXY=http://192.168.100.211:10809 -v /data1/zengzhengran/sweTrans_yang/SWE-agent:/SWE-agent -v /data1/zengzhengran/sweTrans_yang/SWE-bench:/SWE-bench zzr/swe-env--sqlfluff__sqlfluff__0.6
# docker内部运行engine_evaluation.py
python /data1/zengzhengran/sweTrans_yang/SWE-agent/evaluation/engine_evaluation.py --path_conda /root/miniconda3 --testbed /testbed --num_workers 1 --log_dir /data1/zengzhengran/sweTrans_yang/SWE-agent/evaluation/log --predictions_path /data1/zengzhengran/sweTrans_yang/SWE-agent/evaluation/testbed/sqlfluff__sqlfluff-1625/all_preds_filtered.json --temp_dir /data1/zengzhengran/sweTrans_yang/SWE-agent/evaluation/testbed/sqlfluff__sqlfluff-1625 --timeout 900 --verbose --skip_existing

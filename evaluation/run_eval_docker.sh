docker run --rm -it \
    --network host -e ALL_PROXY=http://127.0.0.1:10809 \
    -v /data1/zengzhengran/SWE-agent:/SWE-agent \
    sweagent/swe-eval:latest \
    python evaluation.py \
    --predictions_path /SWE-agent/trajectories/zengzhengran/gpt3-0125__SWE-bench_Lite__default__t-0.00__p-0.95__c-0.50__install-1/all_preds.jsonl \
    --swe_bench_tasks princeton-nlp/SWE-bench \
    --log_dir /SWE-agent/evaluation/results \
    --testbed /SWE-agent/evaluation/testbed \
    --skip_existing \
    --timeout 900 \
    --verbose
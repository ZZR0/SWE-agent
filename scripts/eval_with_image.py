import argparse
import json
import os

def main(args):
    data = json.load(open(args.predictions_path))
    for item in data:
        instance_id = f"{item['instance_id']}"
        if not "pyvista__pyvista-4315" in instance_id:
            continue
        image_name = f"zzr/swe-env--{item['repo'].replace('/', '__')}__{item['version']}"
        cmd = f"""
            docker run --rm -it \
            --network host -e ALL_PROXY=http://192.168.100.211:10809 \
            -v /hdd2/zzr/SWE-agent:/SWE-agent \
            -v /hdd2/zzr/SWE-bench:/SWE-bench \
            {image_name} \
            python /SWE-agent/evaluation/engine_validation.py \
                --instances_path /SWE-agent/dataset/swebench_lite_dev.json \
                --instance_filter {instance_id} \
                --path_conda /root/miniconda3 \
                --testbed /testbed \
                --log_dir /SWE-agent/evaluation/log \
                --temp_dir /SWE-agent/evaluation/tmp \
                --num_workers 1 \
                --verbose 
                
                python /SWE-agent/evaluation/evaluation.py \
                --predictions_path "$predictions_path" \
                --swe_bench_tasks "$dataset_name_or_path" \
                --log_dir "$results_dir" \
                --testbed "$testbed_dir" \
                --skip_existing \
                --timeout 900 \
                --verbose
                
        """
        cmd = " ".join(cmd.strip().split())
        print("==="*10)
        print(cmd)
        print("==="*10)
        os.system(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, help="Path to log directory", required=True
    )
    args = parser.parse_args()
    
    main(args)
    
    
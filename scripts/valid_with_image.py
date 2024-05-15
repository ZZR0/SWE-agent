import argparse
import json
import os

def main(args):
    data = json.load(open(args.data_path))
    for item in data:
        instance_id = f"{item['instance_id']}"
        # if not "pyvista__pyvista-4315" in instance_id:
        #     continue
        image_name = f"zzr/swe-env--{item['repo'].replace('/', '__')}__{item['version']}"
        cmd = f"""
            docker run --rm -it \
            --network host -e ALL_PROXY=http://127.0.0.1:10809 \
            -v /data1/zengzhengran/SWE-agent:/SWE-agent \
            -v /data1/zengzhengran/SWE-bench:/SWE-bench \
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
    
    
import argparse
import json
import os

def main(args):
    data = json.load(open(args.data_path))
    for item in data:
        instance_id = f"{item['instance_id']}"
        # if not "sqlfluff" in instance_id:
        #     continue
        image_name = f"zzr/swe-env--{item['repo'].replace('/', '__')}__{item['version']}"
        cmd = f"python run.py --model_name vllm-llama3-70b --image_name {image_name} --instance_filter {instance_id} --per_instance_cost_limit 3.00 --config_file ./config/default.yaml"
        print(cmd)
        os.system(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, help="Path to log directory", required=True
    )
    args = parser.parse_args()
    
    main(args)
    
    
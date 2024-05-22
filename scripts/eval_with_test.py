import argparse
import json
import os
import logging
import re

from swebench import (
    KEY_INSTANCE_ID,
    get_logs_gold,
)

from swebench.harness.constants import INSTALL_FAIL
from swebench.harness.utils import has_attribute_or_import_error

from swebench.metrics.log_parsers import MAP_REPO_TO_PARSER

get_file_name_from_lp = lambda x: x.rsplit("/", 1)[-1]
get_id_from_lp = lambda x: get_file_name_from_lp(x).split(".")[0]
get_repo_from_lp = lambda x: get_id_from_lp(x).rsplit("-", 1)[0].replace("__", "/")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("evaluation")



def main(args):
    data = json.load(open(args.data_path))
    data = [data[0]]
    for item in data:
        instance_id = f"{item['instance_id']}"
        image_name = f"zzr/swe-env--{item['repo'].replace('/', '__')}__{item['version']}"
        cmd = f"""
            docker run --rm -it \
            --network host -e ALL_PROXY=http://192.168.100.211:10809 \
            -v /data1/zengzhengran/sweTrans_yang/SWE-agent:/SWE-agent \
            -v /data1/zengzhengran/sweTrans_yang/SWE-bench:/SWE-bench \
            {image_name} \
            python /SWE-agent/evaluation/engine_validation_test.py \
                --instances_path /SWE-agent/dataset/swebench_lite_dev.json \
                --instance_filter {instance_id} \
                --path_conda /root/miniconda3 \
                --testbed /testbed \
                --log_dir /SWE-agent/evaluation/log \
                --temp_dir /SWE-agent/evaluation/tmp \
                --num_workers 1 \
                --verbose \
                --coverage_report
                
        """
        cmd = " ".join(cmd.strip().split())
        print("==="*10)
        print(cmd)
        print("==="*10)
        os.system(cmd)
        
    logger.info("✅ Finished evaluation")
    logger.info("==================================")
    log_dir = '/data1/zengzhengran/sweTrans_yang/SWE-agent/evaluation/log'
    logger.info(f"Log directory for evaluation run: {log_dir}") # 使用logger记录日志，打印包含评估运行日志目录的消息
    
    scorecards = [] # 总体结果
    for p in data:
        scorecard = {KEY_INSTANCE_ID: p[KEY_INSTANCE_ID], "statuses": [], "success_rate": [], "report": ""}
        log_file_name = f"{p[KEY_INSTANCE_ID]}.log" # 形如'pvlib__pvlib-python-1072.log'
        log_path = os.path.join(
            log_dir, log_file_name
        )
        if not os.path.exists(log_path):
            scorecard["statuses"].append(f"no log file {log_path}")
            scorecards.append(scorecard)
            continue
        
        # 用>>>>> applied patch划分结果，得到使用golden patch前后的log内容
        log_before, log_after = get_logs_gold(log_path)
        if has_attribute_or_import_error(log_before): # 之前的环境有问题
            scorecard["statuses"].append("log_before has attribute or import error")
            scorecards.append(scorecard)
            continue
        
        # 使用正则表达式匹配覆盖率报告
        match = re.search(r"(Name.*?TOTAL.*?\d+%)", log_after, re.DOTALL)
        if not match:
            scorecard["report"] = "can't match coverage report"
        else:
            scorecard["report"] = match.group(0)
        
        with open(log_path, "r") as f:
            log_contents = f.read()
            if INSTALL_FAIL in log_contents: # "install_fail"在log中出现，则将"install_fail"状态添加到评估结果的"statuses"字段中
                scorecard["statuses"].append("install_fail")
                
        scorecard["statuses"].append("has evaluated") # 之前的环境没有问题，将"has evaluated"状态添加到评估结果的"statuses"字段中
        
        repo = get_repo_from_lp(log_path) # 从日志路径中获取仓库信息
        log_parser = MAP_REPO_TO_PARSER[repo] # 根据仓库信息获取对应的日志解析器
        
        content_before = log_parser(log_before) # 从log_before中获取状态映射图
        content_after = log_parser(log_after) # 从log_after中获取状态映射图
        
        success_test = [key for key in content_before if content_before[key] == "FAILED" and content_after[key] == "PASSED"] # 找到符合条件的测试用例
        
        scorecard["eval_test"] = {"success": [],"failure": []}
        scorecard["eval_test"]["success"] = success_test
        scorecard["eval_test"]["failure"] = [key for key in content_after if key not in success_test]
        
        success_count = len(scorecard["eval_test"]["success"]) # 计算成功的测试用例数量
        total_count = success_count + len(scorecard["eval_test"]["failure"]) # 计算所有测试用例的总数
        scorecard["success_rate"] = success_count / total_count if total_count > 0 else 0 # 计算并添加成功率到评分卡
        
        scorecards.append(scorecard)
    
    directory = os.path.dirname(args.data_path) # 得到'/data1/zengzhengran/sweTrans_yang/SWE-agent/dataset'
    path_scorecards = os.path.join(directory, "test_eval_scorecards.json") # 形如'/data1/zengzhengran/sweTrans_yang/SWE-agent/dataset/test_eval_scorecards.json'
    with open(path_scorecards, "w") as f:
        json.dump(scorecards, fp=f, indent=2)
    logger.info(f"- Saved scorecards to {path_scorecards}")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, help="Path to log directory", required=True
    )
    args = parser.parse_args()
    
    main(args)
    
# python /data1/zengzhengran/sweTrans_yang/SWE-agent/scripts/eval_with_test.py --data_path /data1/zengzhengran/sweTrans_yang/SWE-agent/dataset/swebench_lite_dev.json

import argparse
import json
import os
import traceback
import logging
import os
import shutil
import subprocess
from multiprocessing import Pool
from collections import Counter

from swebench import (
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTION,
    get_eval_report,
    get_logs_eval,
    get_model_report,
    get_resolution_status,
    get_eval_refs,
)
from swebench.harness.constants import INSTALL_FAIL
from swebench.harness.utils import get_instances

from unidiff import PatchSet

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("evaluation")

WORK_DIR = "/data1/zengzhengran/sweTrans_yang"

def validate_predictions(predictions_path, tasks_ids):
    # 检查预测文件是否存在
    if not any([predictions_path.endswith(x) for x in [".json", ".jsonl"]]):
        raise ValueError("Predictions path must be .json or .jsonl file")
    predictions = get_instances(predictions_path) # 列表，每个元素是一个字典，每个字典形如{'model_name_or_path': 'vllm-llama3-70b', 'instance_id': 'pvlib__pvlib-python-1072', 'model_patch': '\ndiff --git a/pvlib/temperature.py ...'}
    not_in_tasks = []
    # 检查预测是否正确格式化
    for pred in predictions:
        if any([x not in pred for x in [KEY_INSTANCE_ID, KEY_MODEL, KEY_PREDICTION]]):
            raise ValueError(f"Every prediction must have {KEY_INSTANCE_ID}, {KEY_MODEL}, and {KEY_PREDICTION} fields")
        if pred[KEY_INSTANCE_ID] not in tasks_ids:
            not_in_tasks.append(pred[KEY_INSTANCE_ID])
    # Check that instance IDs specified by predictions exist
    if len(not_in_tasks) > 0:
        logger.warning(
            "Predictions for the following instance_ids were not "
            + "found in the tasks file and will not be considered: "
            + ", ".join(not_in_tasks)
        )

def eval_engine_docker(args):
    # instance_id = f"{item['instance_id']}"
    # if not "pyvista__pyvista-4315" in instance_id:
    #     return
    args.log_dir = args.log_dir.replace(f"{WORK_DIR}", "/") # 把工作目录替换成/，因后面docker要做映射，即把工作目录映射到宿主机
    args.temp_dir = args.temp_dir.replace(f"{WORK_DIR}", "/")
    args.predictions_path = args.predictions_path.replace(f"{WORK_DIR}", "/")
    image_name = f"zzr/swe-env--{args.repo.replace('/', '__')}__{args.version}"
    cmd = f"""
        docker run --rm -it \
        --network host -e ALL_PROXY=http://192.168.100.211:10809 \
        -v {WORK_DIR}/SWE-agent:/SWE-agent \
        -v {WORK_DIR}/SWE-bench:/SWE-bench \
        {image_name} \
        python /SWE-agent/evaluation/engine_evaluation.py \
            --path_conda /root/miniconda3 \
            --testbed /testbed \
            --num_workers 1 \
            --log_dir {args.log_dir} \
            --predictions_path {args.predictions_path} \
            --temp_dir {args.temp_dir} \
            --timeout {args.timeout} \
    """
    if args.verbose: # 是否显示日志
        cmd += " --verbose"
    if args.skip_existing: # 是否跳过已经存在日志的预测结果的评估
        cmd += " --skip_existing"
    cmd = " ".join(cmd.strip().split())
    logger.info("==="*10)
    logger.info(cmd)
    logger.info("==="*10)
    os.system(cmd)

def run_evaluation(
    predictions_path: str,
    swe_bench_tasks: str,
    log_dir: str,
    testbed: str,
    conda_link: str,
    log_suffix: str,
    skip_existing: bool,
    timeout: int,
    verbose: bool,
    num_processes: int = -1,
    path_conda: str = None,
):
    """
    对每个模型/库/版本组合的预测结果运行评估。

    Args:
        predictions_path (str): Path to the predictions file.
        swe_bench_tasks (str): Path to the SWE-bench tasks file OR HF dataset name.
        log_dir (str): 保存日志的目录路径。
        testbed (str): 保存测试结果的目录路径。
        skip_existing (bool): 是否跳过已经存在日志的预测结果的评估。
        timeout (int): 每个评估的超时时间。
        verbose (bool): 是否打印详细输出。
        path_conda (str): conda 环境文件的路径。

    Raises:
        ValueError: 如果 log_dir 不是目录，testbed 不是目录，或 swe_bench_tasks 不存在。
    """
    # 验证参数
    if not os.path.exists(log_dir) or not os.path.isdir(log_dir):
        raise ValueError("--log_dir must exist and point at a directory")
    if not os.path.exists(testbed) or not os.path.isdir(testbed):
        raise ValueError("--testbed must exist and point at a directory")
    
    tasks = list(get_eval_refs(swe_bench_tasks).values())

    # 验证参数的格式是否正确
    if not isinstance(tasks, list):
        raise ValueError(f"{swe_bench_tasks} must contain an array of tasks")
    tasks_map = {t[KEY_INSTANCE_ID]: t for t in tasks} # 字典，格式形如{'pyvista__pyvista-4315': {'instance_id': 'pyvista__pyvista-4315', 'model': 'pyvista', 'version': '4315'}, ...}
    predictions_path = os.path.abspath(predictions_path) # 获取绝对路径
    validate_predictions(predictions_path, [t[KEY_INSTANCE_ID] for t in tasks]) # 检查是否有非法格式，是否有predictions无法对应tasks

    # 按模型对预测进行分组
    predictions = get_instances(predictions_path)
    logger.info(f"Found {len(predictions)} predictions in predictions file")

    # For each model, split predictions by repo + save to folder
    eval_args = [] # 保存所有参数的列表
    temp_dirs = [] # 保存文件夹路径的列表，文件夹形如'/data1/zengzhengran/sweTrans_yang/SWE-agent/evaluation/testbed/pvlib__pvlib-python-1072'
    for p in predictions:
        # Group predictions by repository, version
        repo = p[KEY_INSTANCE_ID].rsplit("-", 1)[0]
        t = tasks_map[p[KEY_INSTANCE_ID]]
        p.update(t) # 将t中的键值对添加到p中
        version = t["version"]

        # 创建针对instance_id的testbed文件夹
        testbed_save_dir = os.path.join(testbed, p[KEY_INSTANCE_ID])
        os.makedirs(testbed_save_dir, exist_ok=True)

        # 创建用于存储model/repo/version的预测文件
        file_name = f"{predictions_path.split('/')[-1]}"
        file_path = os.path.join(testbed_save_dir, file_name)
        if file_path.endswith(".jsonl"):
            file_path = file_path[:-1] # 把jsonl改为json

        # Create evaluation args
        args = argparse.Namespace()
        args.repo = repo
        args.version = version
        args.log_dir = log_dir
        args.log_suffix = log_suffix
        args.num_workers = 1
        args.predictions_path = file_path
        args.skip_existing = skip_existing
        args.temp_dir = testbed_save_dir
        args.timeout = timeout
        args.verbose = verbose
        args.conda_link = conda_link
        args.path_conda = path_conda

        # Save predictions to file
        with open(file_path, "w") as f:
            json.dump([p], f, indent=4)

        eval_args.append(args)
        temp_dirs.append(testbed_save_dir)

    eval_args = eval_args[1:2] # 提取索引为1的元素，形如[Namespace(repo='sqlfluff__sqlfluff', version='0.6', log_dir='/data1/zengzhengran/sweTrans_yang/SWE-agent/evaluation/log', log_suffix=None, num_workers=1, predictions_path='/data1/zengzhengran/sweTrans_yang/SWE-agent/evaluation/testbed/sqlfluff__sqlfluff-1625/all_preds_filtered.json', skip_existing=True, temp_dir='/data1/zengzhengran/sweTrans_yang/SWE-agent/evaluation/testbed/sqlfluff__sqlfluff-1625', timeout=900, verbose=True, conda_link=None, path_conda=None)]
    temp_dirs = temp_dirs[1:2] # 提取索引为1的元素
    if len(eval_args) == 0: # 没有预测结果
        logger.info("No predictions to evaluate")
        return

    # Run evaluation on each model/repo
    # 如果num_processes大于0，则选择较小的值，以确保不超过eval_args的长度
    # 否则，num_processes等于eval_args的长度
    num_processes = min(len(eval_args), num_processes) if num_processes > 0 else len(eval_args)
    try:
        if num_processes == 1:
            for args in eval_args:
                eval_engine_docker(args)
        else:
            pool = Pool(processes=num_processes)
            pool.map(eval_engine_docker, eval_args)
            pool.close()
            pool.join()
    finally:
        # Clean up
        for temp_dir in temp_dirs:
            # Kill all processes that are using the temp directory
            subprocess.run(f"lsof +D {temp_dir} | awk 'NR>1 {{print $2}}' | xargs kill", shell=True, capture_output=True)
            # Remove temp directory
            # shutil.rmtree(temp_dir, ignore_errors=True) # 删掉tmp文件夹

def main(predictions_path, log_dir, swe_bench_tasks, testbed, 
         skip_existing, timeout, verbose, conda_link, 
         log_suffix, num_processes, path_conda, instance_filter):
    # Check if paths exist
    if not os.path.exists(predictions_path):
        raise FileNotFoundError(f"Predictions path {predictions_path} does not exist")
    eval_refs = get_eval_refs(swe_bench_tasks)
    for k, v in eval_refs.items():
        eval_refs[k] = {key: v[key] for key in [KEY_INSTANCE_ID, "FAIL_TO_PASS", "PASS_TO_PASS"]}
    """
    转换后的格式
        eval_refs = {'sqlfluff__sqlfluff-1625': 
        {'instance_id': 'sqlfluff__sqlfluff-1625', 
         'FAIL_TO_PASS': ['test/cli/commands_test.py::test__cli__command_directed'], 
         'PASS_TO_PASS': ['test/cli/commands_test.py::test_encoding[utf-32-UTF-32]', ...]
         },}
    """

    # Change model_name_or_patch field to directory name for all predictions
    directory = os.path.dirname(predictions_path) # 获取给定文件路径的目录路径
    directory_name = directory.rsplit("/", 1)[-1] # 通过将目录路径按照"/"进行拆分，获取目录的名称
    pred_path_orig = predictions_path # 备份原始的文件路径
    pred_path_temp = predictions_path.replace(".jsonl", "_filtered.jsonl") # 将原始文件路径中的".jsonl"替换为"_filtered.jsonl"，生成一个新的文件路径

    pred_total, pred_will_eval = 0, 0 # 初始化预测总数和将要评估的预测数为0
    with open(pred_path_temp, "w") as f:
        for l in open(pred_path_orig, "r").readlines():
            pred_total += 1
            p = json.loads(l)
            # 排除预测结果为空字符串的情况，写入_filtered.jsonl中
            if p[KEY_PREDICTION] is not None and p[KEY_PREDICTION].strip() != "":
                p[KEY_MODEL] = directory_name
                json.dump(p, f)
                f.write("\n")
                pred_will_eval += 1
    logger.info(
        f"Found {pred_total} total predictions, will evaluate {pred_will_eval} ({pred_total-pred_will_eval} are empty)"
    )

    # Run evaluation
    predictions_path = pred_path_temp
    try:
        logger.info("🏃 Beginning evaluation...")
        run_evaluation(
            predictions_path=predictions_path,
            log_dir=log_dir,
            swe_bench_tasks=swe_bench_tasks,
            testbed=testbed,
            skip_existing=skip_existing,
            timeout=timeout,
            verbose=verbose,
            conda_link=conda_link,
            log_suffix=log_suffix,
            num_processes=num_processes
        )
        logger.info("✅ Finished evaluation")
    except Exception as e:
        logger.info(f"❌ Evaluation failed: {e}\n{traceback.format_exc()}")
    logger.info("==================================")
    # os.remove(pred_path_temp)

    # Get predictions, define log_dir
    predictions = [json.loads(l) for l in open(pred_path_orig, "r").readlines()] # 获取预测结果，将每行字符串解析为JSON对象，并存储在列表predictions中
    logger.info(f"Log directory for evaluation run: {log_dir}") # 使用logger记录日志，打印包含评估运行日志目录的消息

    # Iterate through predictions
    scorecards = [] # 总体结果
    for p in predictions:
        scorecard = {KEY_INSTANCE_ID: p[KEY_INSTANCE_ID], "statuses": [], "stats": {}}

        # 如果存在traj_path，添加trajectory统计信息
        traj_path = os.path.join(directory, f"{p[KEY_INSTANCE_ID]}.traj")
        if os.path.exists(traj_path):
            traj_data = json.load(open(traj_path, "r")) # 加载数据
            scorecard["stats"]["traj_num_steps"] = len(traj_data["trajectory"]) # 记录"trajectory"的长度
            scorecard["stats"]["traj_action_dist"] = dict( # 统计"history"中"assistant"角色的action的分布情况
                Counter( # 统计role为assistant的action的情况，结果形如{'create': 5, 'edit': 2}
                    [
                        entry["action"].strip().split()[0] # 只取第一个词，动词，形如'create'，'edit'
                        if entry["role"] == "assistant" and "action" in entry and len(entry["action"]) > 0
                        else None
                        for entry in traj_data["history"]
                    ]
                )
            )
            scorecard["exit_status"] = ( # 记录exit_status的状况
                traj_data["info"]["exit_status"] # 如果info里有，则取info的exit_status
                if "exit_status" in traj_data["info"]
                else "n/a" # 否则取n/a
            )

        # 检查是否生成了预测，即预测是否为空
        if p[KEY_PREDICTION] is None or p[KEY_PREDICTION].strip() == "":
            scorecard["statuses"].append("not_generated") # 如果预测结果为空，则将"not_generated"状态添加到评估结果的"statuses"字段中
            scorecards.append(scorecard)
            continue
        scorecard["statuses"].append("generated") # 有预测，则将"generated"状态添加到评估结果的"statuses"字段中

        # Get log file
        log_file_name = f"{p[KEY_INSTANCE_ID]}.{p[KEY_MODEL]}.eval.log" # 形如'pvlib__pvlib-python-1072.vllm-llama3-70b__SWE-bench_Lite__default__t-0.00__p-0.95__c-2.00__install-1.eval.log'
        if args.log_suffix is not None:
            log_file_name = f"{p[KEY_INSTANCE_ID]}.{p[KEY_MODEL]}.{args.log_suffix}.eval.log"
        log_path = os.path.join(
            log_dir, log_file_name
        ) # 形如'/data1/zengzhengran/sweTrans_yang/SWE-agent/evaluation/log/pvlib__pvlib-python-1072.vllm-llama3-70b__SWE-bench_Lite__default__t-0.00__p-0.95__c-2.00__install-1.eval.log'
        if not os.path.exists(log_path):
            scorecard["statuses"].append("build_failure")
            scorecards.append(scorecard)
            continue

        # 状态映射图（只关心 "APPLY_PATCH_PASS (pred)"之后的内容），log中信息是否完整
        eval_sm, found = get_logs_eval(log_path)

        # Check that the prediction generated
        if not found:
            scorecards.append(scorecard)
            continue
        scorecard["statuses"].append("applied")

        with open(log_path, "r") as f:
            log_contents = f.read()
            if INSTALL_FAIL in log_contents: # "install_fail"在log中出现，则将"install_fail"状态添加到评估结果的"statuses"字段中
                scorecard["statuses"].append("install_fail")

        # Get resolution status
        report = get_eval_report(eval_sm, eval_refs[p[KEY_INSTANCE_ID]]) # 结合log文件和原来构建的（Fail-Pass, Pass-Pass）对，得到评估结果
        scorecard["test_results"] = {
            "failure": {
                "FAIL_TO_PASS": report["FAIL_TO_PASS"]["failure"],
                "PASS_TO_PASS": report["PASS_TO_PASS"]["failure"],
            },
            "success": {
                "FAIL_TO_PASS": report["FAIL_TO_PASS"]["success"],
                "PASS_TO_PASS": report["PASS_TO_PASS"]["success"],
            }
        }
        resolution_status = get_resolution_status(report)
        scorecard["statuses"].append(resolution_status)

        try:
            diff_obj = PatchSet(p[KEY_PREDICTION])
            scorecard["patch_files"] = [
                x.path
                for x in diff_obj.modified_files
                + diff_obj.added_files
                + diff_obj.removed_files
            ]
            scorecard["patch_lines_add"] = sum([f.added for f in diff_obj])
            scorecard["patch_lines_del"] = sum([f.removed for f in diff_obj])
        except Exception as e:
            logger.info(f"[{p[KEY_INSTANCE_ID]}] Error parsing prediction diff: {e}")
            scorecard["patch_files"] = []
            scorecard["patch_lines_add"] = 0
            scorecard["patch_lines_del"] = 0
        scorecards.append(scorecard)

    # Save to summary, scorecard json
    path_scorecards = os.path.join(directory, "scorecards.json") # 形如'/data1/zengzhengran/sweTrans_yang/SWE-agent/trajectories/zengzhengran/vllm-llama3-70b__SWE-bench_Lite__default__t-0.00__p-0.95__c-2.00__install-1/scorecards.json'
    with open(path_scorecards, "w") as f:
        json.dump(scorecards, fp=f, indent=2)
    logger.info(f"- Wrote per-instance scorecards to {path_scorecards}")

    # Get results and write to file
    logger.info(f"Reference Report:")
    report = get_model_report(directory_name, pred_path_orig, swe_bench_tasks, log_dir)
    for k, v in report.items():
        logger.info(f"- {k}: {len(v)}")

    path_results = os.path.join(directory, "results.json") # 形如'/data1/zengzhengran/sweTrans_yang/SWE-agent/trajectories/zengzhengran/vllm-llama3-70b__SWE-bench_Lite__default__t-0.00__p-0.95__c-2.00__install-1/results.json'
    with open(path_results, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"- Wrote summary of run to {path_results}")

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_path", type=str, help="Path to predictions file (.jsonl)", required=True,)
    parser.add_argument("--log_dir", type=str, help="Path to log directory", required=True)
    parser.add_argument("--swe_bench_tasks", type=str, help="Path to SWE-bench task instances file", required=True,)
    parser.add_argument("--testbed", type=str, help="Path to testbed directory", required=True)
    parser.add_argument("--skip_existing", action="store_true", help="(Optional) Skip existing logs")
    parser.add_argument("--timeout", type=int, help="(Optional) Timeout in seconds (default: 900)", default=900,)
    parser.add_argument("--verbose", action="store_true", help="(Optional) Verbose mode")
    parser.add_argument("--conda_link", default=None, type=str, help="(Optional) URL to conda installation to use")
    parser.add_argument("--log_suffix", default=None, type=str, help="(Optional) Log suffix")
    parser.add_argument("--num_processes", default=-1, type=int, help="Num processes")
    parser.add_argument("--path_conda", type=str, default=None, help="(Optional) Path to miniconda3 or anaconda installation")
    parser.add_argument("--instance_filter", type=str, default=None, help="(Optional) Number of workers")
    
    args = parser.parse_args()
    main(**vars(args))


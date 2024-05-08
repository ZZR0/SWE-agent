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

WORK_DIR = "/data1/zengzhengran/"

def validate_predictions(predictions_path, tasks_ids):
    # Check that predictions file exists
    if not any([predictions_path.endswith(x) for x in [".json", ".jsonl"]]):
        raise ValueError("Predictions path must be .json or .jsonl file")
    predictions = get_instances(predictions_path)
    not_in_tasks = []
    # Check that predictions are correctly formatted
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
    args.log_dir = args.log_dir.replace(f"{WORK_DIR}", "/")
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
    if args.verbose:
        cmd += " --verbose"
    if args.skip_existing:
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
    Runs evaluation on predictions for each model/repo/version combination.

    Args:
        predictions_path (str): Path to the predictions file.
        swe_bench_tasks (str): Path to the SWE-bench tasks file OR HF dataset name.
        log_dir (str): Path to the directory where logs will be saved.
        testbed (str): Path to the directory where testbeds will be saved.
        skip_existing (bool): Whether to skip evaluations for predictions that already have logs.
        timeout (int): Timeout for each evaluation.
        verbose (bool): Whether to print verbose output.
        path_conda (str): Path to the conda environment file.

    Raises:
        ValueError: If log_dir is not a directory, testbed is not a directory, or swe_bench_tasks does not exist.
    """
    # Validate arguments
    if not os.path.exists(log_dir) or not os.path.isdir(log_dir):
        raise ValueError("--log_dir must exist and point at a directory")
    if not os.path.exists(testbed) or not os.path.isdir(testbed):
        raise ValueError("--testbed must exist and point at a directory")
    
    tasks = list(get_eval_refs(swe_bench_tasks).values())

    # Verify arguments are formatted correctly
    if not isinstance(tasks, list):
        raise ValueError(f"{swe_bench_tasks} must contain an array of tasks")
    tasks_map = {t[KEY_INSTANCE_ID]: t for t in tasks}
    predictions_path = os.path.abspath(predictions_path)
    validate_predictions(predictions_path, [t[KEY_INSTANCE_ID] for t in tasks])

    # Group predictions by model
    predictions = get_instances(predictions_path)
    logger.info(f"Found {len(predictions)} predictions in predictions file")

    # For each model, split predictions by repo + save to folder
    eval_args = []
    temp_dirs = []
    for p in predictions:
        # Group predictions by repository, version
        repo = p[KEY_INSTANCE_ID].rsplit("-", 1)[0]
        t = tasks_map[p[KEY_INSTANCE_ID]]
        p.update(t)
        version = t["version"]

        # Create instance_id specific testbed folder
        testbed_save_dir = os.path.join(testbed, p[KEY_INSTANCE_ID])
        os.makedirs(testbed_save_dir, exist_ok=True)

        # Create predictions file for model/repo/version
        file_name = f"{predictions_path.split('/')[-1]}"
        file_path = os.path.join(testbed_save_dir, file_name)
        if file_path.endswith(".jsonl"):
            file_path = file_path[:-1]

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

    eval_args = eval_args[1:2]
    temp_dirs = temp_dirs[1:2]
    if len(eval_args) == 0:
        logger.info("No predictions to evaluate")
        return

    # Run evaluation on each model/repo
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
            shutil.rmtree(temp_dir, ignore_errors=True)

def main(predictions_path, log_dir, swe_bench_tasks, testbed, 
         skip_existing, timeout, verbose, conda_link, 
         log_suffix, num_processes, path_conda, instance_filter):
    # Check if paths exist
    if not os.path.exists(predictions_path):
        raise FileNotFoundError(f"Predictions path {predictions_path} does not exist")
    eval_refs = get_eval_refs(swe_bench_tasks)
    for k, v in eval_refs.items():
        eval_refs[k] = {key: v[key] for key in [KEY_INSTANCE_ID, "FAIL_TO_PASS", "PASS_TO_PASS"]}

    # Change model_name_or_patch field to directory name for all predictions
    directory = os.path.dirname(predictions_path)
    directory_name = directory.rsplit("/", 1)[-1]
    pred_path_orig = predictions_path
    pred_path_temp = predictions_path.replace(".jsonl", "_filtered.jsonl")

    pred_total, pred_will_eval = 0, 0
    with open(pred_path_temp, "w") as f:
        for l in open(pred_path_orig, "r").readlines():
            pred_total += 1
            p = json.loads(l)
            # Exclude predictions w/ empty strings
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
    os.remove(pred_path_temp)

    # Get predictions, define log_dir
    predictions = [json.loads(l) for l in open(pred_path_orig, "r").readlines()]
    logger.info(f"Log directory for evaluation run: {log_dir}")

    # Iterate through predictions
    scorecards = []
    for p in predictions:
        scorecard = {KEY_INSTANCE_ID: p[KEY_INSTANCE_ID], "statuses": [], "stats": {}}

        # Add trajectory statistics if traj_path exists
        traj_path = os.path.join(directory, f"{p[KEY_INSTANCE_ID]}.traj")
        if os.path.exists(traj_path):
            traj_data = json.load(open(traj_path, "r"))
            scorecard["stats"]["traj_num_steps"] = len(traj_data["trajectory"])
            scorecard["stats"]["traj_action_dist"] = dict(
                Counter(
                    [
                        entry["action"].strip().split()[0]
                        if entry["role"] == "assistant" and "action" in entry and len(entry["action"]) > 0
                        else None
                        for entry in traj_data["history"]
                    ]
                )
            )
            scorecard["exit_status"] = (
                traj_data["info"]["exit_status"]
                if "exit_status" in traj_data["info"]
                else "n/a"
            )

        # Check that a prediction was generated
        if p[KEY_PREDICTION] is None or p[KEY_PREDICTION].strip() == "":
            scorecard["statuses"].append("not_generated")
            scorecards.append(scorecard)
            continue
        scorecard["statuses"].append("generated")

        # Get log file
        log_file_name = f"{p[KEY_INSTANCE_ID]}.{p[KEY_MODEL]}.eval.log"
        if args.log_suffix is not None:
            log_file_name = f"{p[KEY_INSTANCE_ID]}.{p[KEY_MODEL]}.{args.log_suffix}.eval.log"
        log_path = os.path.join(
            log_dir, log_file_name
        )
        if not os.path.exists(log_path):
            scorecard["statuses"].append("build_failure")
            scorecards.append(scorecard)
            continue

        # Get evaluation logs
        eval_sm, found = get_logs_eval(log_path)

        # Check that the prediction generated
        if not found:
            scorecards.append(scorecard)
            continue
        scorecard["statuses"].append("applied")

        with open(log_path, "r") as f:
            log_contents = f.read()
            if INSTALL_FAIL in log_contents:
                scorecard["statuses"].append("install_fail")

        # Get resolution status
        report = get_eval_report(eval_sm, eval_refs[p[KEY_INSTANCE_ID]])
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
    path_scorecards = os.path.join(directory, "scorecards.json")
    with open(path_scorecards, "w") as f:
        json.dump(scorecards, fp=f, indent=2)
    logger.info(f"- Wrote per-instance scorecards to {path_scorecards}")

    # Get results and write to file
    logger.info(f"Reference Report:")
    report = get_model_report(directory_name, pred_path_orig, swe_bench_tasks, log_dir)
    for k, v in report.items():
        logger.info(f"- {k}: {len(v)}")

    path_results = os.path.join(directory, "results.json")
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

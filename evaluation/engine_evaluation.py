import argparse, os, re

from multiprocessing import Pool, cpu_count
from swebench.harness.constants import (
    APPLY_PATCH_FAIL,
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTION,
    PatchType,
)
from swebench.harness.context_manager import TaskEnvContextManager
from swebench.harness.engine_validation import setup_testbed
from swebench.harness.utils import (
    extract_minimal_patch,
    get_instances,
    split_instances,
    DotDict
)
from tqdm.auto import tqdm


def overwrite_ablation(tcm: TaskEnvContextManager, task_instance: dict):
    """
    运行消融实验的代码，比较生成完整文件与补丁的效果

    Args:
        tcm: TaskEnvContextManager
        task_instance: 包含任务实例的字典
    """
    # 如果完整输出为None，写入日志并完全跳过
    if 'full_output' not in task_instance:
        print(f'[{task_instance[KEY_INSTANCE_ID]}] No `full_output` field, skipping')
        with open(tcm.log_file, 'a') as f_log:
            f_log.write(f'{APPLY_PATCH_FAIL}; No `full_output` field\n')
        return
    if task_instance['full_output'] is None:
        print(f'[{task_instance[KEY_INSTANCE_ID]}] `full_output` is None, skipping')
        with open(tcm.log_file, 'a') as f_log:
            f_log.write(f'{APPLY_PATCH_FAIL}; `full_output` is None\n')
        return

    # 尝试使用任务+应用测试补丁来设置环境
    if not tcm.reset_task_env(task_instance):
        return
    
    filename_pat = re.compile(r'\[start of ([\w\.\-\/]+)\]\n(.+?)\n\[end of \1\]', re.DOTALL)
    # 运行安装
    if (
        not tcm.run_install_task(task_instance)
        or not tcm.apply_patch(task_instance["test_patch"], patch_type=PatchType.PATCH_TEST.value)
    ):
        return
    
    # 覆盖文件
    for filename, contents in filename_pat.findall(task_instance['full_output']):
        correct_filename = './' + filename.lstrip('/')
        correct_filename = os.path.abspath(correct_filename)
        if not correct_filename.startswith(os.getcwd()):
            print(f"[{task_instance[KEY_INSTANCE_ID]}] Generation attempted to create file outside of working directory")
            return

        # if os.path.exists(correct_filename):
        if not os.path.exists(correct_filename):
            folder = '/'.join(correct_filename.split('/')[:-1])
            if not os.path.exists(folder):
                os.makedirs(folder)
        with open(correct_filename, 'w') as f:
            f.write(contents)
            with open(tcm.log_file, 'a') as f_log:
                f_log.write(f'Overwrote {correct_filename}\n')
    
    # 运行测试脚本
    if not tcm.run_tests_task(task_instance):
        return
    
    return


def evaluate_predictions(data: dict):
    """
    设置任务环境上下文管理器。然后在上下文管理器中评估每个预测。

    Args:
        data: 包含任务实例和其他数据的字典
            task_instances: 要评估的[任务实例，预测]对列表
            + setup_testbed args
    """
    data_dict = DotDict(data)
    for task_instance in tqdm(
        data_dict.task_instances,
        disable=data_dict.verbose,
        desc=f"Evaluating predictions for {data_dict.log_dir}"
    ):
        with TaskEnvContextManager(
            task_instance,
            data_dict.testbed,
            data_dict.venv,
            data_dict.log_dir,
            data_dict.conda_path,
            verbose=data_dict.verbose,
            timeout=data_dict.timeout,
            is_eval=True,
            log_suffix=data_dict.log_suffix,
        ) as tcm:
            # 尝试使用任务实例设置环境
            if not tcm.reset_task_env(task_instance):
                continue

            # 尝试应用预测
            patch_type = PatchType.PATCH_PRED_TRY.value # "pred_try"

            # 如果预测补丁无法应用，尝试进行一些小的补丁重构然后再试一次
            if not tcm.apply_patch(task_instance[KEY_PREDICTION], patch_type=patch_type) \
                and task_instance[KEY_PREDICTION] is not None \
                and task_instance[KEY_PREDICTION] != "":
                task_instance[KEY_PREDICTION] = extract_minimal_patch(task_instance[KEY_PREDICTION])
                patch_type = PatchType.PATCH_PRED_MINIMAL_TRY.value
                if not tcm.apply_patch(task_instance[KEY_PREDICTION], patch_type=patch_type):
                    # 如果编辑后的补丁仍然无法应用，继续
                    continue
            tcm.apply_patch(task_instance[KEY_PREDICTION], patch_type=patch_type, revert=True) # 重置为原始状态

            # 根据补丁是否被编辑设置预测补丁标签
            if patch_type == PatchType.PATCH_PRED_MINIMAL_TRY.value: # "pred_minimal_try"
                patch_type = PatchType.PATCH_PRED_MINIMAL.value # "pred_minimal"
            else:
                patch_type = PatchType.PATCH_PRED.value # "pred"

            # 运行安装 + 测试脚本，返回值是True，前面+not就是False
            if (
                not tcm.run_install_task(task_instance) # 初始化环境
                or not tcm.apply_patch(task_instance[KEY_PREDICTION], patch_type=patch_type) # 应用patch，此时type为"pred"
                or not tcm.apply_patch(task_instance["test_patch"], patch_type=PatchType.PATCH_TEST.value) # 应用test_patch，此时type为"test"
                or not tcm.run_tests_task(task_instance) # 运行测试
            ):
                continue

def filter_instances(instance_filter, task_instances: list) -> list:
    new_instances = []
    for instance in task_instances:
        # Skip instances that don't match the instance filter
        if re.match(instance_filter, instance['instance_id']) is None:
            continue
        new_instances.append(instance)
    return new_instances

def main(args):
    """
    如果 num_workers > 1，将预测分成多个组。然后并行评估每个组。
    """
    if args.num_workers is None:
        args.num_workers = cpu_count()  # 如果没有指定工作进程数，则使用CPU的核心数

    predictions = get_instances(args.predictions_path) # 获取预测实例
    # 形如[
        # 'model_name_or_path': 'vllm-llama3-70b', 
        # 'instance_id': 'sqlfluff__sqlfluff-1625', 
        # 'model_patch': '\ndiff --git a/pvlib/temperature.py ...', 
        # 'repo':'sqlfluff/sqlfluff', 
        # 'base_commit':'14e1a23a3166b9a645a16de96f694c77a5d4abb7',
        # 'patch': 'diff --git a/pvlib/temperature.py ...', 
        # 'test_patch': '\ndiff --git a/pvlib/temperature.py ...', 
        # 'problem_statement':'TSQL - L031 But there is no join condition',
        # 'hints_text':'Actually, re-reading the docs I think this is the intended behaviour... closing',
        # 'created_at': '2021-10-13T11:35:29Z',
        # 'version': '0.6']
    if args.instance_filter is not None:
        task_instances = filter_instances(args.instance_filter, task_instances) # 如果有过滤条件，则进行过滤
    # 移除已经被评估过的预测
    if args.skip_existing:
        predictions_filtered = []
        for p in predictions:
            log_file_name = f"{p[KEY_INSTANCE_ID]}.{p[KEY_MODEL]}.eval.log"
            if args.log_suffix is not None:
                log_file_name = f"{p[KEY_INSTANCE_ID]}.{p[KEY_MODEL]}.{args.log_suffix}.eval.log"
            path_log = os.path.join(args.log_dir, log_file_name)
            if not os.path.exists(path_log): # 如果日志文件不存在，说明该预测还未被评估过
                predictions_filtered.append(p)
        if len(predictions_filtered) == 0: # 待评估的预测都已经被评估过，直接返回
            return
        else:
            predictions = predictions_filtered

    predictions_groups = split_instances(predictions, args.num_workers) # 将预测分组，按照 num_workers 条分成多个组

    data_groups = [
        {
            "task_instances": g,
            "func": evaluate_predictions,
            **vars(args),
        }
        for g in predictions_groups
    ] # data_group形如['task_instances': [每个待评估都是一个大字典，组成一个列表], 'func': evaluate_predictions, 后面跟着函数参数]

    if args.num_workers == 1: # 如果只有一个工作进程，则直接设置测试环境并返回
        setup_testbed(data_groups[0])
        return

    pool = Pool(processes=args.num_workers) # 创建进程池
    pool.map(setup_testbed, data_groups) # 并行设置测试环境
    pool.close()
    pool.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_path", type=str, help="Path to predictions instances file", required=True)
    parser.add_argument("--instance_filter", type=str, default=None, help="(Optional) Number of workers")
    parser.add_argument("--log_dir", type=str, help="Path to log directory", required=True)
    parser.add_argument("--conda_link", type=str, default=None, help="(Optional) URL to conda installation to use")
    parser.add_argument("--log_suffix", type=str, default=None, help="(Optional) Suffix to append to log file names")
    parser.add_argument("--num_workers", type=int, default=1, help="(Optional) Number of workers")
    parser.add_argument("--path_conda", type=str, help="(Optional) Path to miniconda3 or anaconda installation")
    parser.add_argument("--skip_existing", action="store_true", help="(Optional) Skip existing logs")
    parser.add_argument("--testbed", type=str, help="(Optional) Path to testbed directory")
    parser.add_argument("--temp_dir", type=str, help="(Optional) Path to temporary directory for storing virtual envs")
    parser.add_argument("--timeout", type=int, default=None, help="(Optional) Timeout (seconds) for testing script execution")
    parser.add_argument("--verbose", action="store_true", help="(Optional) Verbose mode")
    args = parser.parse_args()
    main(args)

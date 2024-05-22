import argparse, os

from multiprocessing import Pool, cpu_count
import re
from swebench.harness.constants import PatchType
from swebench.harness.context_manager import TaskEnvContextManager, TestbedContextManager
from swebench.harness.utils import get_instances, split_instances, DotDict


SKIP_INSTANCES = {"pytest-dev/pytest": ["6387", "7956", "3805"]}

global_coverage = False # 默认不使用coverage run -m pytest

def validate_args(args):
    """
    验证命令行参数
    """
    if not os.path.exists(args.instances_path):
        raise ValueError(f"Could not find instances file at {args.instances_path}")
    if not os.path.exists(args.log_dir):
        raise ValueError(f"Could not find log directory at {args.log_dir}")

    # If value is provided, check that the paths exist
    if args.path_conda is not None and not os.path.exists(args.path_conda):
        raise ValueError(f"Could not find conda installation at {args.path_conda}")
    if args.testbed is not None and not os.path.exists(args.testbed):
        raise ValueError(f"Could not find testbed at {args.testbed}")
    if args.temp_dir is not None and not os.path.exists(args.temp_dir):
        raise ValueError(f"Could not find temporary directory at {args.temp_dir}")

    # If value is provided, check that it is valid
    if args.timeout is not None and args.timeout < 0:
        raise ValueError(f"Timeout must be a positive integer")
    if args.num_workers is not None and args.num_workers < 1:
        raise ValueError(f"Number of workers must be a positive integer")

def verify_task_instances(data: dict):
    """
    设置任务环境上下文管理器。然后在上下文管理器中安装并验证每个任务实例。

    Args:
        data: 包含任务实例和其他数据的字典
            task_instances: 任务实例列表
            + setup_testbed args
    """
    # 将输入的字典转换为DotDict，使其可以像访问属性一样访问字典元素
    data_dict = DotDict(data)
    for task_instance in data_dict.task_instances: # 遍历任务实例
        # 使用任务环境上下文管理器，设置任务实例的环境
        with TaskEnvContextManager(
            task_instance,
            data_dict.testbed,
            data_dict.venv,
            data_dict.log_dir,
            data_dict.conda_path,
            verbose=data_dict.verbose,
            timeout=data_dict.timeout,
            log_suffix=data_dict.log_suffix,
        ) as tcm:
            # 如果任务实例的仓库在跳过的实例中，并且任务实例的拉取号也在跳过的实例中，则跳过此次循环
            if (
                task_instance["repo"] in SKIP_INSTANCES
                and task_instance["pull_number"]
                in SKIP_INSTANCES[task_instance["repo"]]
            ):
                continue
            # 如果以下任一操作失败，则跳过此次循环
            # 1. 重置任务环境
            # 2. 运行安装任务
            # 3. 应用test patch（符合我们要求的test用例）
            # 4. 运行测试任务
            # 5. 应用golden patch
            # 6. 再次运行测试任务
            print("-"*20)
            print(task_instance["FAIL_TO_PASS"])
            print("-"*20)
            if (
                not tcm.reset_task_env(task_instance)
                or not tcm.run_install_task(task_instance)
                or not tcm.apply_patch(task_instance["test_patch"], patch_type=PatchType.PATCH_TEST.value) # type是‘test'
                ########### 这里也可以apply其他patch，包括我们自己生成的test_patch
                ########### 之后要修改run_tests_task里的指令，使它只运行我们想要的test用例，即，对instance["test_cmd"]指令重构（175行-177行）
                or not tcm.run_tests_task(task_instance)
                or not tcm.apply_patch(task_instance["patch"], patch_type=PatchType.PATCH_GOLD.value) # type是‘gold'
                or not tcm.run_tests_task(task_instance, global_coverage)
            ):
                continue


def setup_testbed(data: dict):
    """
    创建 testbed 上下文管理器并并行运行 verify_task_instances

    Args:
        data: 包含任务实例和其他数据的字典
        conda_link: 要使用的 conda 安装的 URL
        task_instances: 任务实例列表
        log_dir: 日志目录的路径
        path_conda: miniconda3 或 anaconda 安装的路径
        testbed: testbed 目录的路径
        temp_dir: 用于存储虚拟环境的临时目录的路径
        timeout: 测试脚本执行的超时时间（秒）
        verbose: 详细模式
    """
    data_dict = DotDict(data)
    with TestbedContextManager(
        data_dict.task_instances,
        data_dict.log_dir,
        conda_link=data_dict.conda_link,
        path_conda=data_dict.path_conda,
        testbed=data_dict.testbed,
        temp_dir=data_dict.temp_dir,
        timeout=data_dict.timeout,
        verbose=data_dict.verbose,
        coverage=global_coverage,
    ) as tcm:
        distributed_task_list = tcm.get_distributed_tasks()
        for task_list in distributed_task_list:
            print(
                f"{task_list['testbed']}: {len(task_list['task_instances'])} instances"
            )

        if len(distributed_task_list) == 1:
            data_dict.func(distributed_task_list[0])
            return

        pool = Pool(processes=len(distributed_task_list))
        pool.map(data_dict.func, distributed_task_list)
        pool.close()
        pool.join()

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
    Splits task instances into multiple groups if num_workers > 1
    """
    if args.num_workers is None:
        args.num_workers = cpu_count()

    task_instances = get_instances(args.instances_path)
    if args.instance_filter is not None:
        task_instances = filter_instances(args.instance_filter, task_instances)
    task_instances_groups = split_instances(task_instances, args.num_workers)
    
    if args.coverage_report == True:
        global global_coverage
        global_coverage = True

    data_groups = [
        {
            "task_instances": g,
            "func": verify_task_instances,
            **vars(args),
        }
        for g in task_instances_groups
    ]

    for group in data_groups:
        del group["instances_path"]

    if args.num_workers == 1:
        setup_testbed(data_groups[0])
        return

    pool = Pool(processes=args.num_workers)
    pool.map(setup_testbed, data_groups)
    pool.close()
    pool.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances_path", type=str, help="Path to candidate task instances file", required=True)
    parser.add_argument("--instance_filter", type=str, default=None, help="(Optional) Number of workers")
    parser.add_argument("--log_dir", type=str, help="Path to log directory", required=True)
    parser.add_argument("--conda_link", type=str, default=None, help="(Optional) URL to conda installation to use")
    parser.add_argument("--log_suffix", type=str, default=None, help="(Optional) Suffix to append to log file names")
    parser.add_argument("--path_conda", type=str, default=None, help="(Optional) Path to miniconda3 or anaconda installation")
    parser.add_argument("--testbed", type=str, help="(Optional) Path to testbed directory")
    parser.add_argument("--temp_dir", type=str, help="(Optional) Path to temporary directory for storing virtual envs")
    parser.add_argument("--timeout", type=int, default=None, help="(Optional) Timeout (seconds) for testing script execution")
    parser.add_argument("--verbose", action="store_true", help="(Optional) Verbose mode")
    parser.add_argument("--num_workers", type=int, default=None, help="(Optional) Number of workers")
    parser.add_argument("--coverage_report", action="store_true", help="(Optional) evaluate the test coverage")
    
    args = parser.parse_args()
    validate_args(args)
    main(args)

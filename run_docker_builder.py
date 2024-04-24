import logging
import re
import docker
import traceback

from sweagent.environment.swe_env import LONG_TIMEOUT

try:
    from rich_argparse import RichHelpFormatter
except ImportError:
    msg = (
        "Please install the rich_argparse package with `pip install rich_argparse`."
    )
    raise ImportError(msg)
import yaml
from rich.markdown import Markdown
from dataclasses import dataclass
from pathlib import Path
from rich.logging import RichHandler
from simple_parsing import parse
from simple_parsing.helpers.serialization.serializable import FrozenSerializable
from simple_parsing.helpers.flatten import FlattenedAccess
from sweagent import (
    AgentArguments,
    EnvironmentArguments,
    ModelArguments,
    SWEEnv,
    get_data_path_name,
)


__doc__: str = """ Run inference. Usage examples:

```bash
# Run over a github issue:
python run.py --model_name "gpt4" --data_path "https://github.com/pvlib/pvlib-python/issues/1603" --config_file "config/default_from_url.yaml"
# Apply a patch in a local repository to an issue specified as Markdown file and run a custom installer script in the container
python run.py --model_name "gpt4" --data_path "/path/to/my_issue.md" --repo_path "/path/to/my/local/repo" --environment_setup "/path/to/setup.sh" --config_file "config/default_from_url.yaml" --apply_patch_locally
```
"""

handler = RichHandler(show_time=False, show_path=False)
handler.setLevel(logging.DEBUG)
logger = logging.getLogger("run_dev")
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)
logger.propagate = False
logging.getLogger("simple_parsing").setLevel(logging.WARNING)


@dataclass(frozen=True)
class ActionsArguments(FlattenedAccess, FrozenSerializable):
    """Run real-life actions (opening PRs, etc.) if we can solve the issue."""
    # Open a PR with the patch if we can solve the issue
    open_pr: bool = False  
    # When working with local repository: Apply patch
    apply_patch_locally: bool = False
    # Option to be used with open_pr: Skip action if there are already commits claiming 
    # to fix the issue. Please only set this to False if you are sure the commits are 
    # not fixes or if this is your own repository!
    skip_if_commits_reference_issue: bool = True  
    # OBSOLETE. Do not use, will raise error. Please specify --repo_path instead.
    push_gh_repo_url: str = ""

    def __post_init__(self):
        if self.push_gh_repo_url:
            raise ValueError("push_gh_repo_url is obsolete. Use repo_path instead")

@dataclass(frozen=True)
class ScriptArguments(FlattenedAccess, FrozenSerializable):
    """Configure the control flow of the run.py script"""
    environment: EnvironmentArguments
    agent: AgentArguments
    actions: ActionsArguments
    instance_filter: str = ".*"  # Only run instances that completely match this regex
    skip_existing: bool = True  # Skip instances with existing trajectories
    suffix: str = ""
    # Raise unhandled exceptions during the run (useful for debugging)
    raise_exceptions: bool = False

    @property
    def run_name(self):
        """Generate a unique name for this run based on the arguments."""
        model_name = self.agent.model.model_name.replace(":", "-")
        data_stem = get_data_path_name(self.environment.data_path)
        assert self.agent.config_file is not None  # mypy
        config_stem = Path(self.agent.config_file).stem

        temp = self.agent.model.temperature
        top_p = self.agent.model.top_p

        per_instance_cost_limit = self.agent.model.per_instance_cost_limit
        install_env = self.environment.install_environment

        return (
            f"{model_name}__{data_stem}__{config_stem}__t-{temp:.2f}__p-{top_p:.2f}"
            + f"__c-{per_instance_cost_limit:.2f}__install-{int(install_env)}"
            + (f"__{self.suffix}" if self.suffix else "")
        )


class _ContinueLoop(Exception):
    """Used for internal control flow"""
    ...


class Main:
    def __init__(self, args: ScriptArguments):
        logger.info(f"📙 Arguments: {args.dumps_yaml()}")
        self.args = args
        self.env = SWEEnv(self.args.environment)

    def run(self, index):
        logger.info("▶️  Beginning task " + str(index))
        self.env = SWEEnv(self.args.environment)
        env_name = f"{self.env.data[index]['repo'].replace('/', '__')}__{self.env.data[index]['version']}"
        image_name = f"zzr/swe-env--{env_name}"
        tag = "latest"
        observation, info = self.env.reset(index)
        if info is None:
            raise _ContinueLoop
        self.env.communicate_with_handling(
            f"mkdir /testbed  && cp -r /{self.env.data[index]['repo'].replace('/', '__')} /testbed/{env_name}",
            error_msg="Failed to copy github repo.",
            timeout_duration=LONG_TIMEOUT
        )
        new_image = self.env.container_obj.commit(repository=image_name, tag=tag)
        logger.info(f"📦 New Docker image created: {image_name}:{tag}")
        self.env.close()
        

    def main(self):
        instrance_len = len(self.env.data)
        for index in range(instrance_len):
            try:
                if self.should_skip(index):
                    raise _ContinueLoop
                self.run(index)
            except _ContinueLoop:
                continue
            except KeyboardInterrupt:
                logger.info("Exiting InterCode environment...")
                self.env.close()
                break
            except Exception as e:
                traceback.print_exc()
                if self.args.raise_exceptions:
                    raise e
                if self.env.record:
                    logger.warning(f"❌ Failed on {self.env.record['instance_id']}: {e}")
                else:
                    logger.warning(f"❌ Failed on unknown instance")
                self.env.reset_container()
                continue


    def should_skip(self, index: int) -> bool:
        env_name = f"{self.env.data[index]['repo'].replace('/', '__')}__{self.env.data[index]['version']}"
        image_name_and_tag = f"zzr/swe-env--{env_name}:latest"
        if re.match(self.args.instance_filter, self.env.data[index]['instance_id']) is None:
            logger.info(f"Instance filter not matched. Skipping instance {self.env.data[index]['instance_id']}")
            return True
        
        try:
            client = docker.from_env()
        except docker.errors.DockerException as e:
            if "Error while fetching server API version" in str(e):
                raise RuntimeError(
                    "Docker is not running. Please start Docker and try again."
                ) 
        # Check if the image already exists
        try:
            existing_images = client.images.list(name=image_name_and_tag)
            if existing_images:
                logger.info(f"🔍 Image {image_name_and_tag} already exists. Skipping save.")
                return True
        except Exception as e:
            logger.info(f"🔍 Image {image_name_and_tag} does not exist: {str(e)}")

        return False


def get_args(args=None) -> ScriptArguments:
    """Parse command line arguments and return a ScriptArguments object.
    
    Args:
        args: Optional list of arguments to parse. If not provided, uses sys.argv.
    """
    defaults = ScriptArguments(
        suffix="",
        environment=EnvironmentArguments(
            image_name="sweagent/swe-agent:latest",
            data_path="princeton-nlp/SWE-bench_Lite",
            split="dev",
            verbose=True,
            install_environment=True,
        ),
        skip_existing=True,
        agent=AgentArguments(
            model=ModelArguments(
                model_name="gpt4",
                total_cost_limit=0.0,
                per_instance_cost_limit=3.0,
                temperature=0.0,
                top_p=0.95,
            ),
            config_file=Path("config/default.yaml"),
        ),
        actions=ActionsArguments(open_pr=False, skip_if_commits_reference_issue=True),
    )

    # Nicer yaml dumping of multiline strings
    def multiline_representer(dumper, data):
        """configures yaml for dumping multiline strings
        Ref: https://stackoverflow.com/questions/8640959/how-can-i-control-what-scalar-form-pyyaml-uses-for-my-data
        """
        if data.count("\n") > 0:  # check for multiline string
            return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
        return dumper.represent_scalar("tag:yaml.org,2002:str", data)

    yaml.add_representer(str, multiline_representer)

    return parse(ScriptArguments, default=defaults, add_config_path_arg=False, args=args, formatter_class=RichHelpFormatter, description=Markdown(__doc__))



def main(args: ScriptArguments):
    Main(args).main()


if __name__ == "__main__":
    main(get_args())

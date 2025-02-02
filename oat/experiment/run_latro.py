# ./oat/experiment/run_latro.py
import launchpad as lp
from oat.algorithms.latro import LaTROLearner, LaTROActor, LaTROArgs
from oat.args import default_args_validation, get_default_args
from oat.interface import get_program

def run_latro(args: LaTROArgs):
    learner_cls = LaTROLearner
    actor_cls = LaTROActor
    program, local_resources = get_program(args, learner_cls, actor_cls)
    lp.launch(program, launch_type=args.launch_type, local_resources=local_resources)

if __name__ == "__main__":
    # parse CLI arguments
    # 1) get defaults from get_default_args, but pass your LaTROArgs as the type
    args = get_default_args(LaTROArgs)
    args = default_args_validation(args)
    run_latro(args)

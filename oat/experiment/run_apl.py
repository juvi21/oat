# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import launchpad as lp

from oat.args import default_args_validation, get_default_args
from oat.baselines.apl import APLActor, APLArgs, APLLearner
from oat.interface import get_program


def run_apl(args):
    program, local_resources = get_program(args, APLLearner, APLActor)
    lp.launch(
        program,
        launch_type="local_mp",
        local_resources=local_resources,
        terminal="current_terminal",
    )


if __name__ == "__main__":
    args = get_default_args(APLArgs)
    args = default_args_validation(args)
    if args.apl_pref_certainty_only:
        args.num_samples = 2
    run_apl(args)

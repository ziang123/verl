# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from verl.utils.reward_score import gsm8k


def test_gsm8k_reward_components_sum_to_two():
    response = (
        "<think>\n"
        "April has 48 clips. May has 48 / 2 = 24 clips. The total is 48 + 24 = 72.\n"
        "</think>\n"
        "<answer>\n"
        "#### 72\n"
        "</answer>\n"
    )

    assert gsm8k.hard_format_reward(response) == 0.3
    assert gsm8k.mark_reward(response) == 0.1
    assert gsm8k.process_reward(response) == 0.2
    assert gsm8k.digit_reward(response) == 0.1
    assert gsm8k.correctness_reward(response, "72") == 1.3
    assert gsm8k.compute_score(response, "72") == 2.0


def test_gsm8k_mark_reward_counts_each_tag_once():
    response = "<think>\nwork\n</think>\n<answer>\n#### 72"

    assert gsm8k.mark_reward(response) == 0.075
    assert gsm8k.hard_format_reward(response) == 0.0


def test_gsm8k_process_reward_requires_calculation_in_think_block():
    response = "<think>\nThe answer is obvious.\n</think>\n<answer>\n#### 72\n</answer>\n"

    assert gsm8k.process_reward(response) == 0.0


def test_gsm8k_correctness_requires_hash_answer():
    response = (
        "<think>\n"
        "April has 48 clips. May has 48 / 2 = 24 clips. The total is 48 + 24 = 72.\n"
        "</think>\n"
        "<answer>\n"
        "\\boxed{72}\n"
        "</answer>\n"
    )

    assert gsm8k.digit_reward(response) == 0.0
    assert gsm8k.correctness_reward(response, "72") == 0.0
    assert gsm8k.compute_score(response, "72") == 0.6

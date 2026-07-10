# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import re

_SOLUTION_CLIP_CHARS = 300
_HARD_FORMAT_PATTERN = re.compile(r"<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n?", re.DOTALL)
_THINK_PATTERN = re.compile(r"<think>\n(.*?)\n</think>", re.DOTALL)
_NUMERIC_ANSWER_PATTERN = re.compile(r"-?(?:\d+(?:\.\d+)?|\.\d+)")
_CALCULATION_SIGNAL_PATTERN = re.compile(
    r"[+\-*/=×÷]|"
    r"\b(?:add|added|altogether|difference|divide|divided|each|equal|equals|half|less|minus|more|"
    r"multiply|multiplied|plus|remaining|sum|total|times)\b|"
    r"[加减乘除共总还剩]",
    re.IGNORECASE,
)


def _clean_numeric_answer(answer):
    return answer.replace(",", "").replace("$", "").strip()


def _extract_boxed_solution(solution_str):
    boxed_answers = re.findall(r"\\boxed\{([^{}]+)\}", solution_str)
    if not boxed_answers:
        return None

    for boxed_answer in reversed(boxed_answers):
        numbers = re.findall(r"\-?[0-9\.\,]+", boxed_answer)
        for number in reversed(numbers):
            if number not in ["", "."]:
                return _clean_numeric_answer(number)
    return None


def extract_solution(solution_str, method="strict"):
    assert method in ["strict", "flexible"]

    # Optimization: Regular expression matching on very long strings can be slow.
    # For math problems, the final answer is usually at the end.
    # We only match on the last 300 characters, which is a safe approximation for 300 tokens.
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    if method == "strict":
        solutions = re.findall("#### (\\-?[0-9\\.\\,]+)", solution_str)
        if len(solutions) == 0:
            final_answer = _extract_boxed_solution(solution_str)
        else:
            # take the last solution
            final_answer = _clean_numeric_answer(solutions[-1])
    elif method == "flexible":
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            # no reward is there is no answer
            pass
        else:
            invalid_str = ["", "."]
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer


def hard_format_reward(solution_str, ground_truth=None, method="strict"):
    return 0.3 if _HARD_FORMAT_PATTERN.fullmatch(solution_str) else 0.0


def mark_reward(solution_str, ground_truth=None, method="strict"):
    reward = 0.0
    if solution_str.count("<think>\n") == 1:
        reward += 0.025
    if solution_str.count("\n</think>\n") == 1:
        reward += 0.025
    if solution_str.count("<answer>\n") == 1:
        reward += 0.025
    if solution_str.count("\n</answer>") == 1:
        reward += 0.025
    return reward


def process_reward(solution_str, ground_truth=None, method="strict"):
    match = _THINK_PATTERN.search(solution_str)
    if match is None:
        return 0.0

    thought = match.group(1).strip()
    if not thought:
        return 0.0

    numbers = _NUMERIC_ANSWER_PATTERN.findall(thought)
    if not numbers:
        return 0.0

    has_process_length = len(thought.split()) >= 3 or len(thought) >= 8
    has_calculation = len(numbers) >= 2 or _CALCULATION_SIGNAL_PATTERN.search(thought) is not None
    return 0.2 if has_process_length and has_calculation else 0.0


def digit_reward(solution_str, ground_truth=None, method="strict"):
    answer = extract_solution(solution_str=solution_str, method=method)
    if answer is None:
        return 0.0
    return 0.1 if _NUMERIC_ANSWER_PATTERN.fullmatch(answer) else 0.0


def correctness_reward(solution_str, ground_truth, method="strict"):
    answer = extract_solution(solution_str=solution_str, method=method)
    if answer is None:
        return 0.0

    if isinstance(ground_truth, list | tuple | set):
        ground_truths = ground_truth
    else:
        ground_truths = [ground_truth]

    cleaned_answer = _clean_numeric_answer(answer)
    cleaned_ground_truths = {_clean_numeric_answer(str(item)) for item in ground_truths}
    return 1.3 if cleaned_answer in cleaned_ground_truths else 0.0


reward_funcs = [
    hard_format_reward,
    mark_reward,
    process_reward,
    digit_reward,
    correctness_reward,
]


def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for GSM8k.

    The total reward is capped at 2.0 across five dimensions:
    strict format (0.3), tag marks (0.1), calculation process (0.2),
    numeric answer (0.1), and answer correctness (1.3).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: retained for API compatibility
        score: retained for API compatibility
    """
    del format_score, score

    total_score = sum(reward_func(solution_str, ground_truth, method=method) for reward_func in reward_funcs)
    return round(total_score, 10)

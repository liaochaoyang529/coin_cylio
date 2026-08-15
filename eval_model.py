import numpy as np
import json
from PIL import Image
from utils import (
    GeminiLLM,
    ClientBasedLLM,
    create_doubao_client,
    load_api_keys,
    encode_image_b64,
)
from env import QAEnv
import time
import gzip
import argparse
import os
from pathlib import Path
from Questioner import YourQuestioner

ORACLE_MODEL_ID = "gemini-3-flash"

parser = argparse.ArgumentParser(prog="eval-QA-model")
parser.add_argument("start_idx", type=int, help="First trajectory index")
parser.add_argument("end_idx", type=int, help="Last trajectory index")
parser.add_argument(
    "--description-type",
    default="category",
    help="Type of description. Choose one among 'all', 'category', 'color', 'context', 'color_context_feature', 'color_feature', 'color_context'.",
)
parser.add_argument(
    "--split",
    choices=("train", "val", "test"),
    default="train",
    help="Episode split to evaluate. Defaults to train for backward compatibility.",
)
parser.add_argument(
    "--local",
    type=int,
    default=0,
    help="Deprecated compatibility switch for a local VLLM Oracle.",
)
parser.add_argument(
    "--oracle-provider",
    choices=("doubao", "gemini", "local"),
    default=None,
    help="Oracle backend. Defaults to doubao; ORACLE_PROVIDER is used when set.",
)

args = parser.parse_args()

local = args.local

run_type = args.split

ALLOWED_DESCPRITION_TYPES = [
    "all",
    "category",
    "color",
    "context",
    "color_context_feature",
    "color_feature",
    "color_context",
]

assert args.description_type in ALLOWED_DESCPRITION_TYPES, f"--description-type should be one among {ALLOWED_DESCPRITION_TYPES}"



def _add_obs_and_question_to_log(
    obs, new_obs, action, observations, actions, answers, reasonings
):
    image = Image.fromarray(obs["image"])
    if len(observations) == 0:
        observations.append(image)
        actions.append([])
        answers.append([])
        reasonings.append([])
    elif image != observations[-1]:
        observations.append(image)
        if action["question"] is not None:
            actions.append([f"Q: {action['question'][:200]}"])
            answers.append([f"A: {new_obs['answer']}"])
        else:
            actions.append([f"C: {bool(action['conclusion'])}"])
            answers.append([])
        reasonings.append([action["reasoning"]])
    else:
        if action["question"] is not None:
            assert new_obs["answer"] is not None
            s = f"Q: {action['question'][:200]}"
            answers[-1].append(f"A: {new_obs['answer']}")
        else:
            s = f"C: {bool(action['conclusion'])}"
        actions[-1].append(s)
        reasonings[-1].append(action["reasoning"])


def _print_summary(log_data, task_type, run_type, start_idx, end_idx):
    """Print aggregate metrics for one description type."""
    episode_count = len(log_data["id"])
    if not episode_count:
        print(f"[SUMMARY] {task_type}/{run_type}: no completed episodes")
        return

    total_successes = sum(log_data["n_successes"])
    total_conclusions = sum(log_data["n_conclusions"])
    total_questions = sum(log_data["n_questions"])
    total_reward = sum(log_data["episode_reward"])
    average_time = sum(log_data["time_required"]) / episode_count
    full_match_rate = 100 * sum(log_data["episode_success"]) / episode_count

    print(f"\n[SUMMARY] {task_type}/{run_type} episodes {start_idx}:{end_idx}")
    print(f"  Completed episodes: {episode_count}")
    accuracy = 100 * total_successes / total_conclusions if total_conclusions else 0.0
    print(f"  Decision accuracy: {accuracy:.1f}% ({total_successes}/{total_conclusions})")
    print(f"  Oracle questions: {total_questions} total, {total_questions / episode_count:.2f} average")
    print(f"  Environment reward: {total_reward:.2f} total, {total_reward / episode_count:.2f} average")
    print(f"  Questioner time: {average_time:.2f}s average")
    print(f"  Full-match rate: {full_match_rate:.1f}%")


already_done_ids = set()

# Example usage:
if __name__ == "__main__":
    now = str(time.time_ns())

    load_api_keys()
    # model = "gemini-3-flash-preview"

    # -------------------------------------- ORACLE ----------------------------------------
    # if you want to use a custom oracle, you have to change these lines. You can
    # implement it however you want, but it should inher it from the class OracleInterace
    # (see file Oracle.py)
    oracle_provider = args.oracle_provider or os.getenv(
        "ORACLE_PROVIDER", "local" if local else "doubao"
    ).lower()
    if oracle_provider == "doubao":
        oracle_client = create_doubao_client(temperature=1e-6)
        print(f"[INFO] Using Doubao Oracle model: {oracle_client.model_id}")
    elif oracle_provider == "gemini":
        oracle_client = GeminiLLM(model_id=ORACLE_MODEL_ID, temperature=1e-6)
        print(f"[INFO] Using oracle model: {ORACLE_MODEL_ID}")
    elif oracle_provider == "local":
        oracle_model_id = os.environ["ORACLE_MODEL_ID"]
        oracle_client = ClientBasedLLM(
            model_id=oracle_model_id,
            temperature=1e-6,
            max_output_length=int(os.getenv("LOCAL_MAX_OUTPUT_LENGTH", "800")),
        )
        ## Or you can use your oracle here
        # oracle_client = YourOracle
        print(f"[INFO] Using oracle model: {oracle_model_id}")
        # TODO You can also use your oracle here
    else:
        raise ValueError("ORACLE_PROVIDER must be one of: gemini, doubao, local")

    print(f"[INFO] Using oracle: {oracle_client}")
    # --------------------------------------------------------------------------------------
    if args.description_type.lower() == "all":
        task_types = ALLOWED_DESCPRITION_TYPES[1:]
    else:
        task_types = [args.description_type]

    for task_type in task_types:
        env = QAEnv(
            oracle_client,
            f"episodes_{run_type}.jsonl",
            render_mode="rgb",
            task_type=task_type,
        )

        log_data = dict(
            id=[],
            target_image=[],
            task=[],
            observations=[],
            questions=[],
            answers=[],
            reasonings=[],
            n_successes=[],
            n_conclusions=[],
            n_questions=[],
            time_required=[],
            episode_success=[],
            episode_reward=[],
        )
        for episode in range(args.start_idx, args.end_idx):
            _observations = []
            _actions = []
            _answers = []
            _reasonings = []
            _episode_reward = 0.0
            _n_conclusions = 0
            try:
                old_obs, info = env.reset(options={"episode_idx": episode})
            except IndexError as e:
                continue
            print(f"EPISODE: {episode}, ID: {env.current_episode_data['id']}\n")
            if env.current_episode_data["id"] in already_done_ids:
                print(
                    f"I did already run episode with id '{env.current_episode_data['id']}' for subset '{task_type}'"
                )
                continue
            try:
                _add_obs_and_question_to_log(
                    old_obs, {}, {}, _observations, _actions, _answers, _reasonings
                )
            except Exception as e:  # noqa
                print("[ERROR] Error in episode number: ", episode)
                print(str(e))
                continue

            try:
                questioner = YourQuestioner(info)  # TODO: YOUR QUESTIONER HERE
            except:  # noqa
                raise NotImplementedError(
                    "Insert here your Questioner class. See the README for more details."
                )

            questioner.reset_time()
            print(f"Task is: {info['target_description']}")

            for step in range(200):
                print("=============")
                # action = ACTIONS[step]
                try:
                    action = questioner.ask_or_conclude(old_obs)
                except Exception as e:  # noqa
                    print("[ERROR] Error in episode number: ", episode)
                    print(str(e))
                    break

                print(f"Current action: {action}")
                obs, reward, terminated, truncated, info = env.step(action)
                _episode_reward += reward
                if action["conclusion"] is not None:
                    _n_conclusions += 1

                # TODO for some reason sometimes the env doesn't switch observation after a conclusion, especially in the training set.
                # The bug is fixed for the heldout set. For the time being, we just break the loop as we consider this an error
                # (Nothing will be logged if we break here)
                # Conditions: the answer was a conclusion; the two images are the same but neither terminated nor truncated was set
                if action["conclusion"] is not None and (
                    np.all(obs["image"] == old_obs["image"])
                    and not terminated
                    and not truncated
                ):
                    print(
                        "[ERROR] for some reason, no new image was supplied and this is an error"
                    )
                    break

                if action["question"] is not None:
                    a = ""
                    if obs["answer"] is not None:
                        a = obs["answer"]
                    questioner.add_answer(a)
                try:
                    _add_obs_and_question_to_log(
                        old_obs,
                        obs,
                        action,
                        _observations,
                        _actions,
                        _answers,
                        _reasonings,
                    )
                except Exception as e:  # noqa
                    print("[ERROR] Error in episode number: ", episode)
                    print(str(e))
                    _add_obs_and_question_to_log(
                        old_obs,
                        obs,
                        action,
                        _observations,
                        _actions,
                        _answers,
                        _reasonings,
                    )
                    break

                # env.render()

                old_obs = obs
                if terminated or truncated:
                    times = round(time.time() - env.initial_time, 2)
                    try:
                        print(f"Episode finished after {step + 1} steps")
                        _id = env.current_episode_data["id"]
                        _target_image = encode_image_b64(
                            Image.open(env.current_episode_data["path"]), format="png"
                        )
                        _task = env.current_episode_data["tasks"][task_type]
                        _n_successes = env.n_successes
                        _n_questions = questioner.n_questions
                        _time_required = round(questioner.time_required, 2)
                        _episode_success = _n_successes == len(
                            env.current_episode_data["distractors"]
                        )

                        # Append
                        log_data["id"].append(_id)
                        log_data["target_image"].append(_target_image)
                        log_data["task"].append(_task)
                        log_data["n_successes"].append(_n_successes)
                        log_data["n_conclusions"].append(_n_conclusions)
                        log_data["n_questions"].append(_n_questions)
                        log_data["time_required"].append(_time_required)
                        log_data["episode_success"].append(_episode_success)
                        log_data["episode_reward"].append(_episode_reward)
                        log_data["observations"].append([
                            encode_image_b64(o, format="png") for o in _observations
                        ])
                        log_data["questions"].append(_actions)
                        log_data["answers"].append(_answers)
                        log_data["reasonings"].append(_reasonings)
                    except Exception as e:  # noqa
                        print("[ERROR] Error in episode number: ", episode)
                        print(str(e))
                    break
            print("\n\n")

        if len(log_data["id"]) != 0:
            Path("results").mkdir(exist_ok=True)
            print(
                f"~~~~~~~~~~ Finished {task_type}_{run_type}_{str(args.start_idx)}_{str(args.end_idx)} ~~~~~~~~~~"
            )
            data_to_save = json.dumps(log_data).encode()

            with gzip.GzipFile(
                f"results/{questioner.__class__.__name__}_{task_type}_{run_type}_{str(args.start_idx)}_{str(args.end_idx)}.gzip.json",
                "w",
            ) as file:
                file.write(data_to_save)

        _print_summary(
            log_data,
            task_type,
            run_type,
            args.start_idx,
            args.end_idx,
        )

        env.close()

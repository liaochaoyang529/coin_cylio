from abc import ABC, abstractmethod
import hashlib
import json
import os
import re
import numpy as np
from utils import ClientBasedLLM, GeminiLLM, create_doubao_client
from retrying import retry
import time

# Example of a questioner prompt
QUESTIONER_EXAMPLE_PROMPT = (
    "An oracle has a fixed target image and has given you this description of the object in the target image: {TARGET_DESCRIPTION}. "
    "You are given the above image, which may or may not picturing the same object as the target image owned by the oracle. Your goal is to decide whether the "
    "image that you see corresponds to the same image target image owned by the oracle. "
    "Use, to guide your decision, the description of the object, the give image, and, if they exists, the questions that you previously asked to the oracle "
    "and the associated answers. Provide a reasoning about your conclusion, or why you are uncertain and asked a question. "
    "If you are sure that the two image match, return the score 2, if you are sure that they don't match, return "
    "the score 0. If you are unsure either way, return the score 1 and ask an informative question to the oracle about what it might appears in the target image, "
    "to dispel your doubts. You can always trust the oracle's answers and the initial description. Do not ask questions that can be directly answered by "
    "reading the initial description, or questions about the image that you are provided. Be careful: the target image and the given image might differ only in some small details, "
    "like the color of the object, its texture, or the presence of other objects. For example, the target image might picture a bed with a blue comforter, "
    "and the given image a bed with a red comforter, or a blue bed but with a white comforter. "
    "The image might have distortions or digital artifacts: *NEVER* mention them in the question. Prefer asking question if the description is very generic. "
    "Strictly follow this output format: "
    "<motivation>Your reasoning here (under 60 words, do NOT use double quotes \")</motivation><score>0, 1, or 2</score><question>Your question or '' (if score is not 1)</question>"
)


def _validate_observation(observation):
    assert isinstance(observation["image"], np.ndarray) and (
        not observation["answer"] or isinstance(observation["answer"], str)
    ), (
        "Wrong observation format: it must be a dictionary with keys 'image' and 'answer', where 'answer' is a numpy array and 'answer' is either a string or None"
    )
    assert (
        len(observation["image"].shape) == 3 and observation["image"].shape[2] == 3
    ), "Wrong image format: must be a numpy array of shape (H,W,3) --- an rgb image."


class QuestionerInterface(ABC):
    """Abstract Questioner class. Your questioner should inherit from this."""

    def __init__(self, info, *args):
        self.info = info  # required info like the task description
        self.target_description = info["target_description"]

    @abstractmethod
    def ask_or_conclude(self, observation):
        # TODO: this is what you have to implement
        pass

    def add_answer(self, answer):
        self.answers.append(answer)

    def reset_questions(self):
        self.questions = []
        self.answers = []

    def reset_time(self):
        self.time_required = 0


class QuestionerLocalVLM(QuestionerInterface):
    """Simple class that can use a local VLM (run via VLLM) as the questioner."""

    def __init__(self, info, model_id: str):
        # info will contain the target object description: info["target_description"]
        # This is also saved in self.target_description
        super().__init__(info)
        self.client = ClientBasedLLM(model_id=model_id)  # Handle the VLLM connection
        self.questions = []
        self.reasonings = []
        self.answers = []
        self.time_required = 0
        self.n_questions = 0

    @retry(
        stop_max_attempt_number=5,
        wait_exponential_multiplier=2000,
        wait_exponential_max=60000,
    )
    def ask_or_conclude(self, observation):
        _validate_observation(observation)
        start_time = time.time()

        # Define `prompt_to_use`` and other stuff here. The prompt can ask the model to evaluate the observation vs the description,
        # in order to reason about it and conclude (or not) whether the object in the image matches the description.
        # You can also handle resource-keeping tasks, like keeping track of previous questions and answers (although you don't need to)
        prompt_to_use = "TODO"  # TODO:

        # prompt_to_use = QUESTIONER_EXAMPLE_PROMPT.format(TARGET_DESCRIPTION=self.target_description)

        response = self.client.ask(
            prompt=prompt_to_use,
            images=[observation["image"]],
        )

        end_time = time.time()

        # Parse and return a question or a conclusion
        # TODO: parse the question/conclusion and return it.
        ## Return either (if uncertain whether the observation corresponds to the target description or not)
        # return dict(question=question, conclusion=None, reasoning=reasoning)
        ## if certain that is a match
        # return dict(question=None, conclusion=1, reasoning=reasoning)
        ## or if certaint that is NOT a match
        # return dict(question=None, conclusion=0, reasoning=reasoning)
        raise NotImplementedError("Impement this function.")


class YourQuestioner(QuestionerInterface):
    """VLM questioner with bounded, structured target and candidate memory."""

    _MAX_REASONING_LENGTH = 300
    _MAX_FACTS = 24
    _LOW_DISCRIMINATION_ATTRIBUTES = {
        "category",
        "installation_style",
        "object_category",
        "object_type",
        "same_category",
        "subtype",
        "type",
    }
    _ATTRIBUTE_ALIASES = {
        "body_color": "finish_color",
        "cabinet_color": "cabinetry_color",
        "cabinet_finish": "cabinetry_color",
        "cabinetry_finish": "cabinetry_color",
        "context": "surrounding_context",
        "door_count": "component_count",
        "drawer_count": "component_count",
        "exterior_color": "finish_color",
        "exterior_finish": "finish_color",
        "number_of_components": "component_count",
        "contents": "accessory",
        "placement": "surrounding_context",
        "placement_surface": "surrounding_context",
        "pot_color": "finish_color",
        "shape": "distinctive_shape",
        "side_color": "trim_color",
        "surroundings": "surrounding_context",
    }

    def __init__(self, info, *args, client=None):
        super().__init__(info)
        self.questions = []
        self.answers = []
        self.time_required = 0
        self.n_questions = 0

        self.target_profile = {
            "description": self.target_description,
            "facts": {},
        }
        self.oracle_evidence = []
        self.target_evidence = {}
        self.candidate_profile = {}
        self.candidate_comparison = {}
        self._candidate_fingerprint = None
        self._candidate_question_count = 0
        self._asked_question_keys = set()
        self._asked_attribute_keys = set()
        self._answered_attribute_keys = set()
        self._question_attribute_by_key = {}
        self._question_candidate_by_key = {}
        self.max_questions_per_candidate = max(
            0, int(os.getenv("QUESTIONER_MAX_QUESTIONS_PER_CANDIDATE", "2"))
        )
        self.max_evidence_items = max(
            self.max_questions_per_candidate,
            int(os.getenv("QUESTIONER_MAX_EVIDENCE_ITEMS", "12")),
        )
        self.max_memory_chars = min(
            12000,
            max(4000, int(os.getenv("QUESTIONER_MAX_MEMORY_CHARS", "6000"))),
        )
        self.api_attempts = min(
            3, max(1, int(os.getenv("QUESTIONER_API_ATTEMPTS", "2")))
        )
        self.api_retry_delay = min(
            10.0, max(0.0, float(os.getenv("QUESTIONER_API_RETRY_DELAY", "1")))
        )

        if client is not None:
            self.client = client
            return

        configured_provider = os.getenv("QUESTIONER_PROVIDER")
        if configured_provider:
            provider = configured_provider.lower()
        elif os.getenv("QUESTIONER_LOCAL", "0") == "1":
            provider = "local"
        else:
            provider = "doubao"

        if provider == "doubao":
            self.client = create_doubao_client(
                model_id=os.getenv("QUESTIONER_MODEL_ID"), temperature=1e-6
            )
        elif provider == "local":
            model_id = os.getenv("QUESTIONER_MODEL_ID")
            if not model_id:
                raise ValueError("Local questioner requires QUESTIONER_MODEL_ID.")
            self.client = ClientBasedLLM(
                model_id=model_id,
                temperature=1e-6,
                max_output_length=int(os.getenv("LOCAL_MAX_OUTPUT_LENGTH", "800")),
            )
        elif provider == "gemini":
            model_id = os.getenv("QUESTIONER_MODEL_ID", "gemini-3-flash")
            self.client = GeminiLLM(model_id=model_id, temperature=1e-6)
        else:
            raise ValueError(
                "QUESTIONER_PROVIDER must be one of: doubao, gemini, local"
            )

    def reset_questions(self):
        super().reset_questions()
        self.n_questions = 0
        self.target_profile["facts"] = {}
        self.target_evidence = {}
        self.oracle_evidence = []
        self.candidate_profile = {}
        self.candidate_comparison = {}
        self._candidate_fingerprint = None
        self._candidate_question_count = 0
        self._asked_question_keys = set()
        self._asked_attribute_keys = set()
        self._answered_attribute_keys = set()
        self._question_attribute_by_key = {}
        self._question_candidate_by_key = {}

    @classmethod
    def _answer_polarity(cls, answer):
        """Classify answers to the atomic yes/no questions used by the agent."""
        normalized = cls._normalize_text(answer)
        if not normalized:
            return "uncertain"
        uncertain_markers = (
            "cannot determine",
            "can t determine",
            "not visible",
            "unclear",
            "unknown",
            "unsure",
        )
        if any(marker in normalized for marker in uncertain_markers):
            return "uncertain"
        first_word = normalized.split(maxsplit=1)[0]
        if first_word in {"yes", "yeah", "correct", "true"}:
            return "yes"
        if first_word in {"no", "incorrect", "false"}:
            return "no"
        return "uncertain"

    @classmethod
    def _answer_has_target_detail(cls, answer):
        """Return whether an answer adds detail beyond a bare yes/no."""
        normalized = cls._normalize_text(answer)
        words = normalized.split()
        return len(words) >= 4 and words[0] in {
            "yes",
            "yeah",
            "correct",
            "true",
            "no",
            "incorrect",
            "false",
        }

    def add_answer(self, answer):
        """Store bounded, source-attributed evidence about the hidden target."""
        answer = str(answer or "").strip()[:300]
        super().add_answer(answer)
        answer_index = len(self.answers) - 1
        question = (
            self.questions[answer_index]
            if answer_index < len(self.questions)
            else "Unknown question"
        )
        question_key = self._normalize_text(question)
        attribute_key = self._question_attribute_by_key.get(question_key)
        polarity = self._answer_polarity(answer)
        self.oracle_evidence.append({
            "question": question,
            "answer": answer,
            "attribute": attribute_key or "unknown",
            "polarity": polarity,
            "candidate_fingerprint": self._question_candidate_by_key.get(
                question_key
            ),
            "has_target_detail": self._answer_has_target_detail(answer),
        })
        self.oracle_evidence = self.oracle_evidence[-self.max_evidence_items :]
        if attribute_key:
            has_target_detail = self._answer_has_target_detail(answer)
            self.target_evidence[attribute_key] = {
                "polarity": polarity,
                "proposition": question[:200],
                "answer": answer,
                "has_target_detail": has_target_detail,
            }
            while len(self.target_evidence) > self.max_evidence_items:
                self.target_evidence.pop(next(iter(self.target_evidence)))
            self._answered_attribute_keys.add(attribute_key)
            if polarity == "yes":
                fact = f"Oracle confirmed proposition: {question}"
            elif polarity == "no":
                fact = f"Oracle rejected proposition: {question}; {answer}"
            else:
                fact = (
                    f"Oracle was uncertain about proposition: {question}; {answer}"
                )
            self.target_profile["facts"][attribute_key] = fact[:500]

    @staticmethod
    def _image_fingerprint(image):
        digest = hashlib.blake2b(digest_size=12)
        digest.update(str(image.shape).encode("ascii"))
        digest.update(str(image.dtype).encode("ascii"))
        digest.update(image.tobytes())
        return digest.hexdigest()

    def _prepare_candidate(self, image):
        fingerprint = self._image_fingerprint(image)
        if fingerprint != self._candidate_fingerprint:
            self._candidate_fingerprint = fingerprint
            self._candidate_question_count = 0
            self.candidate_profile = {}
            self.candidate_comparison = {}

    @staticmethod
    def _normalize_text(value):
        return re.sub(r"[^\w]+", " ", str(value).casefold()).strip()

    @classmethod
    def _normalize_attribute_key(cls, value):
        normalized = cls._normalize_text(value).replace(" ", "_")[:48]
        return cls._ATTRIBUTE_ALIASES.get(normalized, normalized)

    def _is_generic_description(self):
        description = self._normalize_text(self.target_description)
        category = self._normalize_text(self.info.get("category", ""))
        if category:
            return description == category
        return len(description.split()) <= 2

    @classmethod
    def _bounded_value(cls, value):
        if isinstance(value, (str, int, float, bool)) or value is None:
            return str(value)[:120] if isinstance(value, str) else value
        if isinstance(value, list):
            return [cls._bounded_value(item) for item in value[:8]]
        return str(value)[:160]

    @classmethod
    def _bounded_mapping(cls, value):
        if not isinstance(value, dict):
            return {}
        bounded = {}
        for key, item in list(value.items())[: cls._MAX_FACTS]:
            normalized_key = cls._normalize_text(key).replace(" ", "_")[:48]
            if normalized_key:
                bounded[normalized_key] = cls._bounded_value(item)
        return bounded

    @staticmethod
    def _decode_response(response):
        """Find the first action-bearing JSON object without greedy matching."""
        if isinstance(response, dict):
            return response
        if not isinstance(response, str):
            return None

        decoder = json.JSONDecoder()
        fallback = None
        for start_index, character in enumerate(response):
            if character != "{":
                continue
            try:
                value, _ = decoder.raw_decode(response[start_index:])
            except json.JSONDecodeError:
                continue
            if not isinstance(value, dict):
                continue
            action_type = str(value.get("action", "")).lower()
            if action_type in ("question", "conclusion"):
                return value
            if fallback is None:
                fallback = value
        return fallback

    @staticmethod
    def _coerce_conclusion(value):
        if isinstance(value, bool):
            return int(value)
        if value in (0, 1):
            return value
        if isinstance(value, str) and value.strip() in ("0", "1"):
            return int(value.strip())
        return None

    def _merge_model_state(self, result):
        target_updates = self._bounded_mapping(result.get("target_updates"))
        self.target_profile["facts"].update(target_updates)
        if len(self.target_profile["facts"]) > self._MAX_FACTS:
            self.target_profile["facts"] = dict(
                list(self.target_profile["facts"].items())[-self._MAX_FACTS :]
            )
        self.candidate_profile = self._bounded_mapping(
            result.get("candidate_profile")
        )
        candidate_category = result.get("candidate_category")
        if candidate_category and "category" not in self.candidate_profile:
            self.candidate_profile["category"] = str(candidate_category)[:120]

        comparison = result.get("comparison")
        if not isinstance(comparison, dict):
            comparison = {}
        match_attributes = result.get("match_attributes")
        if match_attributes is None:
            match_attributes = result.get("matches")
        if match_attributes is None:
            match_attributes = comparison.get(
                "match_attributes", comparison.get("matches", [])
            )
        conflicts = result.get("conflicts", comparison.get("conflicts", []))
        unknown = result.get("unknown", comparison.get("unknown", []))
        self.candidate_comparison = self._bounded_mapping({
            "match_attributes": match_attributes,
            "conflicts": conflicts,
            "unknown": unknown,
        })

    def _parse_response(self, response):
        result = self._decode_response(response)
        if not isinstance(result, dict):
            return None, None

        self._merge_model_state(result)
        reasoning = str(
            result.get("reasoning", result.get("reason", "No reasoning provided."))
        )[
            : self._MAX_REASONING_LENGTH
        ]
        action_type = str(result.get("action", "")).lower()
        fallback_conclusion = self._coerce_conclusion(
            result.get("fallback_conclusion", result.get("fallback"))
        )

        if action_type == "question":
            question = str(result.get("question", "")).strip()
            if question:
                question_key = self._normalize_text(question)
                attribute_key = self._normalize_attribute_key(
                    result.get("question_attribute", result.get("attribute", ""))
                )
                self._question_attribute_by_key[question_key] = attribute_key
                return {
                    "question": question[:300],
                    "conclusion": None,
                    "reasoning": reasoning,
                }, fallback_conclusion

        if action_type == "conclusion":
            conclusion = self._coerce_conclusion(result.get("conclusion"))
            if conclusion is not None:
                return {
                    "question": None,
                    "conclusion": conclusion,
                    "reasoning": reasoning,
                }, conclusion
        return None, fallback_conclusion

    def _current_candidate_evidence(self):
        return [
            item
            for item in self.oracle_evidence
            if item.get("candidate_fingerprint") == self._candidate_fingerprint
            and item.get("attribute") not in (None, "", "unknown")
        ]

    def _direct_evidence_attributes(self):
        attributes = {"yes": set(), "no": set(), "uncertain": set()}
        for item in self._current_candidate_evidence():
            polarity = item.get("polarity", "uncertain")
            if polarity not in attributes:
                polarity = "uncertain"
            attribute = self._normalize_attribute_key(item.get("attribute", ""))
            if attribute and attribute not in self._LOW_DISCRIMINATION_ATTRIBUTES:
                attributes[polarity].add(attribute)
        return {
            polarity: sorted(values) for polarity, values in attributes.items()
        }

    def _grounded_attribute_keys(self):
        grounded = set()
        for attribute, item in self.target_evidence.items():
            normalized = self._normalize_attribute_key(attribute)
            if (
                not normalized
                or normalized in self._LOW_DISCRIMINATION_ATTRIBUTES
            ):
                continue
            polarity = item.get("polarity")
            if polarity == "yes" or (
                polarity == "no" and item.get("has_target_detail", False)
            ):
                grounded.add(normalized)

        for item in self.oracle_evidence:
            attribute = self._normalize_attribute_key(item.get("attribute", ""))
            if not attribute or attribute in self._LOW_DISCRIMINATION_ATTRIBUTES:
                continue
            polarity = item.get("polarity")
            if polarity == "yes" or (
                polarity == "no" and item.get("has_target_detail", False)
            ):
                grounded.add(attribute)
        return grounded

    def _deterministic_evidence_conclusion(self):
        direct = self._direct_evidence_attributes()
        if direct["no"]:
            return 0
        if self._is_generic_description() and len(direct["yes"]) >= 2:
            return 1
        return None

    def _evidence_action(self, model_action):
        conclusion = self._deterministic_evidence_conclusion()
        if conclusion is None:
            return None
        if (
            model_action is not None
            and model_action["question"] is None
            and model_action["conclusion"] == conclusion
        ):
            return model_action
        direct = self._direct_evidence_attributes()
        if conclusion == 1:
            reason = (
                "Controller accepted two direct, independent Oracle matches "
                f"with no conflict: {', '.join(direct['yes'])}."
            )
        else:
            reason = (
                "Controller found a direct Oracle conflict for this candidate: "
                f"{', '.join(direct['no'])}."
            )
        return self._fallback_action(conclusion, reason)

    def _memory_payload(self):
        direct = self._direct_evidence_attributes()
        return {
            "target": {
                "description": self.target_description[:500],
                "evidence": dict(self.target_evidence),
                "facts": dict(self.target_profile["facts"]),
            },
            "oracle_evidence": [
                {
                    "question": item["question"][:200],
                    "answer": item["answer"][:300],
                    "attribute": item.get("attribute", "unknown"),
                    "polarity": item.get("polarity", "uncertain"),
                    "scope": (
                        "current_candidate"
                        if item.get("candidate_fingerprint")
                        == self._candidate_fingerprint
                        else "prior_candidate"
                    ),
                }
                for item in self.oracle_evidence
            ],
            "current_oracle_support": {
                "matches": direct["yes"],
                "conflicts": direct["no"],
                "uncertain": direct["uncertain"],
            },
            "current_candidate": dict(self.candidate_profile),
            "current_comparison": dict(self.candidate_comparison),
            "already_asked": [
                question[:200]
                for question in self.questions[-self.max_evidence_items :]
            ],
            "already_asked_attributes": sorted(self._asked_attribute_keys),
            "answered_attributes": sorted(self._answered_attribute_keys),
        }

    def _memory_json(self):
        """Fit structured memory into a predictable prompt budget."""
        payload = self._memory_payload()

        def encode():
            return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

        memory = encode()
        while len(memory) > self.max_memory_chars:
            if len(payload["already_asked"]) > 4:
                payload["already_asked"].pop(0)
            elif len(payload["oracle_evidence"]) > 2:
                payload["oracle_evidence"].pop(0)
            elif payload["current_comparison"]:
                payload["current_comparison"].pop(
                    next(iter(payload["current_comparison"]))
                )
            elif len(payload["target"]["facts"]) > 8:
                payload["target"]["facts"].pop(
                    next(iter(payload["target"]["facts"]))
                )
            elif len(payload["target"]["evidence"]) > 2:
                payload["target"]["evidence"].pop(
                    next(iter(payload["target"]["evidence"]))
                )
            elif len(payload["current_candidate"]) > 8:
                payload["current_candidate"].pop(
                    next(iter(payload["current_candidate"]))
                )
            elif payload["already_asked"]:
                payload["already_asked"].pop(0)
            elif payload["oracle_evidence"]:
                payload["oracle_evidence"].pop(0)
            elif payload["target"]["evidence"]:
                payload["target"]["evidence"].pop(
                    next(iter(payload["target"]["evidence"]))
                )
            else:
                break
            memory = encode()
        return memory

    def _build_prompt(
        self,
        force_conclusion=False,
        force_question=False,
        previous_response=None,
        decision_feedback=None,
    ):
        description_mode = (
            "generic category only" if self._is_generic_description() else "detailed"
        )
        remaining_questions = max(
            0, self.max_questions_per_candidate - self._candidate_question_count
        )
        memory = self._memory_json()
        forced_instruction = ""
        if force_conclusion:
            forced_instruction = (
                "The previous response was invalid or repeated a question. "
                "You MUST return action=conclusion now; do not ask a question."
            )
        elif force_question:
            forced_instruction = (
                "The prior conclusion was unsupported while questions remain. "
                "You MUST ask one new atomic high-discrimination question now; "
                "do not conclude merely because target details are unknown."
            )
        prior_output = ""
        if previous_response is not None:
            prior_output = f"\nInvalid prior output (do not copy it): {str(previous_response)[:600]}"
        feedback_instruction = ""
        if decision_feedback:
            feedback_instruction = (
                f"\nController feedback: {decision_feedback} "
                "Reconsider the decision using a different evidence dimension."
            )

        return f"""You are the questioner in a hidden-target image matching game.
You see only the candidate image. Never use or request the hidden target image.
Target description mode: {description_mode}
Questions remaining for this candidate: {remaining_questions}
Structured memory: {memory}

Decision policy:
- Conclude 0 on a clear category mismatch or any reliable target conflict.
- Conclude 1 only with very high confidence and no unresolved distinctive detail.
- Oracle evidence has an explicit polarity. A current-candidate yes is a direct
  match, a current-candidate no is a direct conflict, and uncertain is neither.
  Never treat an answered attribute by itself as a match.
- Under a generic category description, conclusion 1 requires at least two
  independent grounded matches. Category, subtype, installation_style, and other
  common class traits do not count. Never claim target evidence that was not
  learned from an oracle answer or a target fact.
- For a same-category candidate under a generic description, ask one atomic
  yes/no question about its most discriminative visible feature unless target
  memory already resolves it. Phrase the question as a proposition that is true
  of the visible candidate, so the Oracle's yes/no directly tests the candidate.
- Every question must include one question_attribute key. Never reuse an
  already_asked_attributes key for the same target. Any concise snake_case visual
  attribute is allowed except category, subtype, type, object_type, and
  installation_style. Prefer rare candidate-specific details such as trim/body
  color, control layout, component count, border/numeral style, cabinetry color,
  contents, placement, or surrounding objects. Avoid common class traits.
- Ask only about the hidden target. Never repeat an already_asked question, ask
  about artifacts, or combine several attributes in one question.
- Because a wrong conclusion costs 10 and a question costs 1, ask when confidence
  is below 0.95 and a remaining question can resolve the uncertainty.
{forced_instruction}{feedback_instruction}{prior_output}

Return one compact JSON object only. Keep reasoning under 25 words. Attribute
lists contain snake_case keys only. Include a best-guess fallback (0 or 1) when
asking:
{{"action":"question","attribute":"finish_color","question":"...","fallback":0,"reasoning":"..."}}
or
{{"action":"conclusion","conclusion":0,"matches":["finish_color"],"conflicts":[],"reasoning":"..."}}
"""

    @staticmethod
    def _is_transient_api_error(error):
        error_text = f"{type(error).__name__}: {error}".casefold()
        transient_markers = (
            "connection error",
            "connection reset",
            "connection refused",
            "temporarily unavailable",
            "timed out",
            "timeout",
        )
        return isinstance(error, (ConnectionError, TimeoutError)) or any(
            marker in error_text for marker in transient_markers
        )

    def _ask_model(self, prompt, image):
        model_name = getattr(self.client, "model_id", type(self.client).__name__)
        for attempt in range(1, self.api_attempts + 1):
            start_time = time.time()
            print(
                f"[INFO] Asking Questioner ({model_name}), "
                f"attempt {attempt}/{self.api_attempts}...",
                flush=True,
            )
            try:
                response = self.client.ask(prompt=prompt, images=[image])
            except Exception as error:
                elapsed = time.time() - start_time
                self.time_required += elapsed
                if (
                    attempt == self.api_attempts
                    or not self._is_transient_api_error(error)
                ):
                    raise
                print(
                    f"[WARN] Transient Questioner error: {error}. Retrying...",
                    flush=True,
                )
                time.sleep(self.api_retry_delay * attempt)
                continue

            elapsed = time.time() - start_time
            self.time_required += elapsed
            print(
                f"[INFO] Questioner ({model_name}) responded in {elapsed:.1f}s",
                flush=True,
            )
            return response
        raise AssertionError("Questioner retry loop exited unexpectedly")

    def _question_rejection_reason(self, action):
        if action is None or action["question"] is None:
            return None
        if self._candidate_question_count >= self.max_questions_per_candidate:
            return "question limit reached"
        question_key = self._normalize_text(action["question"])
        if not question_key:
            return "empty question"
        if question_key in self._asked_question_keys:
            return "question was already asked"
        attribute_key = self._question_attribute_by_key.get(question_key, "")
        if not attribute_key:
            return "question is missing question_attribute"
        if attribute_key in self._asked_attribute_keys:
            return f"attribute {attribute_key} was already asked"
        if (
            self._is_generic_description()
            and attribute_key in self._LOW_DISCRIMINATION_ATTRIBUTES
        ):
            return f"attribute {attribute_key} is too generic for category mode"
        return None

    @staticmethod
    def _comparison_items(value):
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if value:
            return [str(value).strip()]
        return []

    def _distinctive_matches(self):
        distinctive = []
        seen = set()
        for match in self._comparison_items(
            self.candidate_comparison.get("match_attributes", [])
        ):
            normalized = self._normalize_attribute_key(match)
            if (
                not normalized
                or normalized in self._LOW_DISCRIMINATION_ATTRIBUTES
                or normalized in seen
            ):
                continue
            seen.add(normalized)
            distinctive.append(normalized)
        return distinctive

    def _positive_rejection_reason(self, action):
        if (
            action is None
            or action["conclusion"] != 1
            or not self._is_generic_description()
        ):
            return None

        direct = self._direct_evidence_attributes()
        if direct["no"]:
            return "positive conclusion contradicts direct Oracle evidence"

        conflicts = self._comparison_items(
            self.candidate_comparison.get("conflicts", [])
        )
        if conflicts:
            return "positive conclusion contains unresolved target conflicts"

        grounded = self._grounded_attribute_keys()
        supported_matches = {
            match for match in self._distinctive_matches() if match in grounded
        }
        supported_matches.update(direct["yes"])
        match_count = len(supported_matches)
        if match_count < 2:
            return (
                "generic positive conclusion needs two grounded non-category "
                f"matches; received {match_count}"
            )
        return None

    def _negative_rejection_reason(self, action):
        if (
            action is None
            or action["conclusion"] != 0
            or not self._is_generic_description()
            or self._candidate_question_count >= self.max_questions_per_candidate
        ):
            return None

        conflicts = self._comparison_items(
            self.candidate_comparison.get("conflicts", [])
        )
        if conflicts:
            return None

        target_category = self._normalize_text(self.info.get("category", ""))
        candidate_category = self._normalize_text(
            self.candidate_profile.get("category", "")
        )
        if (
            target_category
            and candidate_category
            and target_category != candidate_category
        ):
            return None
        return (
            "generic same-category non-match needs a concrete conflict while "
            "questions remain"
        )

    def _record_question(self, question):
        self.questions.append(question)
        question_key = self._normalize_text(question)
        self._asked_question_keys.add(question_key)
        attribute_key = self._question_attribute_by_key.get(question_key)
        self._question_candidate_by_key[question_key] = self._candidate_fingerprint
        if attribute_key:
            self._asked_attribute_keys.add(attribute_key)
        self._candidate_question_count += 1
        self.n_questions += 1

    @staticmethod
    def _fallback_action(conclusion, reason):
        if conclusion is None:
            conclusion = 0
        return {
            "question": None,
            "conclusion": conclusion,
            "reasoning": reason[: YourQuestioner._MAX_REASONING_LENGTH],
        }

    def _supported_fallback_action(self, conclusion, reason):
        fallback_action = self._fallback_action(conclusion, reason)
        if self._positive_rejection_reason(fallback_action):
            return None
        if self._negative_rejection_reason(fallback_action):
            return None
        return fallback_action

    def ask_or_conclude(self, observation):
        _validate_observation(observation)
        image = observation["image"]
        self._prepare_candidate(image)

        response = self._ask_model(self._build_prompt(), image)
        action, fallback_conclusion = self._parse_response(response)
        question_rejection = self._question_rejection_reason(action)
        evidence_action = self._evidence_action(action)
        if evidence_action is not None:
            return evidence_action

        positive_rejection = self._positive_rejection_reason(action)
        negative_rejection = self._negative_rejection_reason(action)

        if (
            action is not None
            and question_rejection is None
            and positive_rejection is None
            and negative_rejection is None
        ):
            if action["question"] is not None:
                self._record_question(action["question"])
            return action

        if (
            question_rejection
            and fallback_conclusion is not None
            and self._candidate_question_count >= self.max_questions_per_candidate
        ):
            fallback_action = self._supported_fallback_action(
                fallback_conclusion,
                f"Used the model's fallback because {question_rejection}.",
            )
            if fallback_action is not None:
                return fallback_action
            positive_rejection = (
                "positive fallback did not contain two independent "
                "non-category matches"
            )

        rejection_reason = (
            positive_rejection
            or negative_rejection
            or question_rejection
            or "model output was invalid or unsupported"
        )
        print(f"[WARN] Rejected Questioner action: {rejection_reason}", flush=True)
        print(f"[WARN] Raw Questioner output: {str(response)[:500]}", flush=True)
        questions_remain = (
            self._candidate_question_count < self.max_questions_per_candidate
        )
        force_conclusion = action is None or not questions_remain
        force_question = action is not None and questions_remain
        repair_response = self._ask_model(
            self._build_prompt(
                force_conclusion=force_conclusion,
                force_question=force_question,
                previous_response=response,
                decision_feedback=rejection_reason,
            ),
            image,
        )
        repaired_action, repaired_fallback = self._parse_response(repair_response)
        repaired_question_rejection = self._question_rejection_reason(repaired_action)
        repaired_positive_rejection = self._positive_rejection_reason(repaired_action)
        repaired_negative_rejection = self._negative_rejection_reason(repaired_action)
        if (
            repaired_action is not None
            and repaired_question_rejection is None
            and repaired_positive_rejection is None
            and repaired_negative_rejection is None
        ):
            if repaired_action["question"] is not None:
                self._record_question(repaired_action["question"])
            return repaired_action

        repaired_rejection = (
            repaired_positive_rejection
            or repaired_negative_rejection
            or repaired_question_rejection
            or "repaired model output was invalid or unsupported"
        )
        print(
            f"[WARN] Rejected repaired Questioner action: {repaired_rejection}",
            flush=True,
        )
        print(
            f"[WARN] Raw repaired output: {str(repair_response)[:500]}",
            flush=True,
        )

        if repaired_question_rejection and repaired_fallback is not None:
            fallback_action = self._supported_fallback_action(
                repaired_fallback,
                f"Used the repaired model fallback because {repaired_question_rejection}.",
            )
            if fallback_action is not None:
                return fallback_action

        final_conclusion = (
            repaired_fallback if repaired_fallback is not None else fallback_conclusion
        )
        fallback_action = self._supported_fallback_action(
            final_conclusion,
            "Decision remained unsupported after one repair attempt; chose the conservative non-match fallback.",
        )
        if fallback_action is not None:
            return fallback_action
        return self._fallback_action(
            0,
            "Positive fallback lacked two independent matches; chose the conservative non-match fallback.",
        )

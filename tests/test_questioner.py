import json
import unittest

import numpy as np

from Questioner import YourQuestioner


class FakeClient:
    model_id = "fake-vlm"

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def ask(self, *, prompt, images):
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError("FakeClient ran out of responses")
        assert len(images) == 1
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def response(**values):
    defaults = {
        "target_updates": {},
        "candidate_profile": {},
        "comparison": {"matches": [], "conflicts": [], "unknown": []},
        "reasoning": "test",
    }
    defaults.update(values)
    return json.dumps(defaults)


class YourQuestionerTests(unittest.TestCase):
    def setUp(self):
        self.info = {
            "target_description": "Wardrobe",
            "category": "Wardrobe",
        }
        self.first_image = np.zeros((4, 4, 3), dtype=np.uint8)
        self.second_image = np.ones((4, 4, 3), dtype=np.uint8)

    def observation(self, image=None, answer=None):
        return {
            "image": self.first_image if image is None else image,
            "answer": answer,
        }

    def test_target_memory_survives_candidate_switch(self):
        client = FakeClient([
            response(
                target_updates={"category": "wardrobe"},
                candidate_profile={
                    "category": "wardrobe",
                    "door_style": "mirrored sliding doors",
                },
                comparison={"unknown": ["door_style"]},
                action="question",
                question_attribute="door_style",
                question="Does the target wardrobe have mirrored sliding doors?",
                fallback_conclusion=0,
            ),
            response(
                target_updates={"door_style": "mirrored sliding doors"},
                candidate_profile={
                    "category": "wardrobe",
                    "door_style": "mirrored sliding doors",
                    "installation": "built-in",
                },
                comparison={"unknown": ["surrounding_context"]},
                action="question",
                question_attribute="surrounding_context",
                question="Is the target wardrobe built into white cabinetry?",
                fallback_conclusion=0,
            ),
            response(
                target_updates={
                    "door_style": "mirrored sliding doors",
                    "surrounding_context": "white cabinetry",
                },
                candidate_profile={
                    "category": "wardrobe",
                    "door_style": "mirrored sliding doors",
                    "installation": "built-in",
                },
                comparison={
                    "match_attributes": ["door_style", "surrounding_context"],
                    "matches": ["door_style", "built-in"],
                },
                action="conclusion",
                conclusion=1,
            ),
            response(
                candidate_profile={"category": "chair"},
                comparison={"conflicts": ["category"]},
                action="conclusion",
                conclusion=0,
            ),
        ])
        questioner = YourQuestioner(self.info, client=client)

        first_action = questioner.ask_or_conclude(self.observation())
        self.assertIsNotNone(first_action["question"])
        questioner.add_answer("Yes, it has mirrored sliding doors.")

        second_action = questioner.ask_or_conclude(
            self.observation(answer="Yes, it has mirrored sliding doors.")
        )
        self.assertIsNotNone(second_action["question"])
        questioner.add_answer("Yes, it is built into white cabinetry.")

        third_action = questioner.ask_or_conclude(
            self.observation(answer="Yes, it is built into white cabinetry.")
        )
        self.assertEqual(third_action["conclusion"], 1)
        self.assertEqual(
            questioner.target_profile["facts"]["door_style"],
            "mirrored sliding doors",
        )
        self.assertIn("mirrored sliding doors", client.prompts[1])

        fourth_action = questioner.ask_or_conclude(
            self.observation(image=self.second_image)
        )
        self.assertEqual(fourth_action["conclusion"], 0)
        self.assertEqual(questioner._candidate_question_count, 0)
        self.assertIn("mirrored sliding doors", client.prompts[3])

    def test_repeated_question_uses_model_fallback(self):
        duplicate_question = "Does the target wardrobe have mirrored doors?"
        client = FakeClient([
            response(
                action="question",
                question_attribute="door_style",
                question=duplicate_question,
                fallback_conclusion=0,
            ),
            response(
                action="question",
                question_attribute="door_style",
                question=duplicate_question,
                fallback_conclusion=0,
            ),
        ])
        questioner = YourQuestioner(self.info, client=client)
        questioner.max_questions_per_candidate = 1

        first_action = questioner.ask_or_conclude(self.observation())
        second_action = questioner.ask_or_conclude(self.observation())

        self.assertEqual(first_action["question"], duplicate_question)
        self.assertEqual(second_action["conclusion"], 0)
        self.assertIn("fallback", second_action["reasoning"])
        self.assertEqual(questioner.n_questions, 1)
        self.assertEqual(len(client.prompts), 2)

    def test_question_limit_is_per_candidate(self):
        client = FakeClient([
            response(
                action="question",
                question_attribute="door_style",
                question="Does the target have mirrored doors?",
                fallback_conclusion=0,
            ),
            response(
                action="question",
                question_attribute="installation_style",
                question="Is the target built into a wall?",
                fallback_conclusion=0,
            ),
            response(
                action="question",
                question_attribute="finish_color",
                question="Is the target white?",
                fallback_conclusion=0,
            ),
        ])
        questioner = YourQuestioner(self.info, client=client)
        questioner.max_questions_per_candidate = 1

        self.assertIsNotNone(
            questioner.ask_or_conclude(self.observation())["question"]
        )
        limited_action = questioner.ask_or_conclude(self.observation())
        self.assertEqual(limited_action["conclusion"], 0)
        self.assertIn("limit reached", limited_action["reasoning"])

        new_candidate_action = questioner.ask_or_conclude(
            self.observation(image=self.second_image)
        )
        self.assertIsNotNone(new_candidate_action["question"])
        self.assertEqual(questioner.n_questions, 2)

    def test_invalid_output_gets_one_forced_conclusion_repair(self):
        client = FakeClient([
            "not valid json",
            """```json
{"candidate_profile":{"doors":"mirrored","installation":"built-in"},"comparison":{"match_attributes":["door_style","surrounding_context"],"matches":["mirrored doors","built-in"],"conflicts":[]},"action":"conclusion","conclusion":"1","reasoning":"repaired"}
```""",
        ])
        questioner = YourQuestioner(self.info, client=client)
        questioner.oracle_evidence = [
            {
                "question": "Does it have mirrored doors?",
                "answer": "Yes.",
                "attribute": "door_style",
                "polarity": "yes",
                "candidate_fingerprint": "prior",
                "has_target_detail": False,
            },
            {
                "question": "Is it built into white cabinetry?",
                "answer": "Yes.",
                "attribute": "surrounding_context",
                "polarity": "yes",
                "candidate_fingerprint": "prior",
                "has_target_detail": False,
            },
        ]

        action = questioner.ask_or_conclude(self.observation())

        self.assertEqual(action["conclusion"], 1)
        self.assertEqual(len(client.prompts), 2)
        self.assertIn("MUST return action=conclusion", client.prompts[1])

    def test_generic_positive_requires_two_distinctive_matches(self):
        client = FakeClient([
            response(
                candidate_profile={"category": "wardrobe", "doors": "mirrored"},
                comparison={"matches": ["mirrored doors"], "conflicts": []},
                action="conclusion",
                conclusion=1,
            ),
            response(
                candidate_profile={"category": "wardrobe", "doors": "mirrored"},
                comparison={
                    "matches": ["mirrored doors"],
                    "unknown": ["trim color"],
                },
                action="question",
                question_attribute="trim_color",
                question="Does the target wardrobe have silver trim?",
                fallback_conclusion=0,
            ),
        ])
        questioner = YourQuestioner(self.info, client=client)

        action = questioner.ask_or_conclude(self.observation())

        self.assertIsNotNone(action["question"])
        self.assertIn("requires at least two", client.prompts[1])
        self.assertEqual(questioner.n_questions, 1)

    def test_generic_positive_accepts_two_distinctive_matches(self):
        client = FakeClient([
            response(
                candidate_profile={
                    "category": "wardrobe",
                    "doors": "mirrored",
                    "installation": "built-in",
                },
                comparison={
                    "match_attributes": ["door_style", "surrounding_context"],
                    "matches": ["same category", "mirrored doors", "built-in"],
                    "conflicts": [],
                },
                action="conclusion",
                conclusion=1,
            )
        ])
        questioner = YourQuestioner(self.info, client=client)
        questioner.oracle_evidence = [
            {
                "question": "Does it have mirrored doors?",
                "answer": "Yes.",
                "attribute": "door_style",
                "polarity": "yes",
                "candidate_fingerprint": "prior",
                "has_target_detail": False,
            },
            {
                "question": "Is it built into white cabinetry?",
                "answer": "Yes.",
                "attribute": "surrounding_context",
                "polarity": "yes",
                "candidate_fingerprint": "prior",
                "has_target_detail": False,
            },
        ]

        action = questioner.ask_or_conclude(self.observation())

        self.assertEqual(action["conclusion"], 1)
        self.assertEqual(len(client.prompts), 1)

    def test_generic_same_category_negative_requires_conflict_or_question(self):
        client = FakeClient([
            response(
                candidate_profile={"category": "wardrobe"},
                comparison={"conflicts": []},
                action="conclusion",
                conclusion=0,
            ),
            response(
                candidate_profile={"category": "wardrobe", "color": "white"},
                comparison={"unknown": ["finish_color"]},
                action="question",
                question_attribute="finish_color",
                question="Is the target wardrobe white?",
                fallback_conclusion=0,
            ),
        ])
        questioner = YourQuestioner(self.info, client=client)

        action = questioner.ask_or_conclude(self.observation())

        self.assertIsNotNone(action["question"])
        self.assertIn("MUST ask one new atomic", client.prompts[1])

    def test_generic_negative_with_concrete_conflict_is_accepted(self):
        client = FakeClient([
            response(
                candidate_profile={"category": "wardrobe", "color": "white"},
                comparison={"conflicts": ["target is black"]},
                action="conclusion",
                conclusion=0,
            )
        ])
        questioner = YourQuestioner(self.info, client=client)

        action = questioner.ask_or_conclude(self.observation())

        self.assertEqual(action["conclusion"], 0)
        self.assertEqual(len(client.prompts), 1)

    def test_low_discrimination_question_attribute_is_repaired(self):
        client = FakeClient([
            response(
                action="question",
                question_attribute="installation_style",
                question="Is the target built into the wall?",
                fallback_conclusion=0,
            ),
            response(
                action="question",
                question_attribute="trim_color",
                question="Does the target have silver trim?",
                fallback_conclusion=0,
            ),
        ])
        questioner = YourQuestioner(self.info, client=client)

        action = questioner.ask_or_conclude(self.observation())

        self.assertEqual(action["question"], "Does the target have silver trim?")
        self.assertIn("too generic", client.prompts[1])

    def test_transient_questioner_error_retries_once(self):
        client = FakeClient([
            ConnectionError("Connection error."),
            response(
                candidate_profile={"category": "chair"},
                comparison={"conflicts": ["category"]},
                action="conclusion",
                conclusion=0,
            ),
        ])
        questioner = YourQuestioner(self.info, client=client)
        questioner.api_retry_delay = 0

        action = questioner.ask_or_conclude(self.observation())

        self.assertEqual(action["conclusion"], 0)
        self.assertEqual(len(client.prompts), 2)

    def test_positive_fallback_cannot_bypass_evidence_gate(self):
        duplicate_question = "Does the target have mirrored doors?"
        client = FakeClient([
            response(
                comparison={"matches": ["mirrored doors"]},
                action="question",
                question_attribute="door_style",
                question=duplicate_question,
                fallback_conclusion=0,
            ),
            response(
                comparison={"matches": ["mirrored doors"]},
                action="question",
                question_attribute="door_style",
                question=duplicate_question,
                fallback_conclusion=1,
            ),
            response(
                comparison={"conflicts": ["installation style"]},
                action="conclusion",
                conclusion=0,
            ),
        ])
        questioner = YourQuestioner(self.info, client=client)

        self.assertIsNotNone(
            questioner.ask_or_conclude(self.observation())["question"]
        )
        action = questioner.ask_or_conclude(self.observation())

        self.assertEqual(action["conclusion"], 0)
        self.assertEqual(len(client.prompts), 3)

    def test_decoder_skips_non_action_json(self):
        decoded = YourQuestioner._decode_response(
            'metadata {"request_id":"abc"} final '
            '{"action":"CONCLUSION","conclusion":0}'
        )

        self.assertEqual(decoded["action"], "CONCLUSION")
        self.assertEqual(decoded["conclusion"], 0)

    def test_attribute_aliases_are_normalized(self):
        self.assertEqual(
            YourQuestioner._normalize_attribute_key("exterior finish"),
            "finish_color",
        )
        self.assertEqual(
            YourQuestioner._normalize_attribute_key("cabinet_color"),
            "cabinetry_color",
        )

    def test_compact_response_schema_is_supported(self):
        client = FakeClient([
            json.dumps({
                "action": "question",
                "attribute": "border_style",
                "question": "Does the target clock have a gold border?",
                "fallback": 0,
                "reasoning": "Need border evidence.",
            })
        ])
        questioner = YourQuestioner(self.info, client=client)

        action = questioner.ask_or_conclude(self.observation())
        questioner.add_answer("Yes")

        self.assertEqual(
            action["question"], "Does the target clock have a gold border?"
        )
        self.assertIn("border_style", questioner._answered_attribute_keys)

    def test_direct_no_overrides_unsupported_positive_conclusion(self):
        client = FakeClient([
            json.dumps({
                "action": "question",
                "attribute": "finish_color",
                "question": "Is the target wardrobe bright white?",
                "fallback": 0,
                "reasoning": "Check color.",
            }),
            json.dumps({
                "action": "conclusion",
                "conclusion": 1,
                "matches": ["finish_color", "door_style"],
                "conflicts": [],
                "reasoning": "Looks similar.",
            }),
        ])
        questioner = YourQuestioner(self.info, client=client)

        first_action = questioner.ask_or_conclude(self.observation())
        questioner.add_answer("No, it has a dark wood finish.")
        action = questioner.ask_or_conclude(self.observation())

        self.assertIsNotNone(first_action["question"])
        self.assertEqual(action["conclusion"], 0)
        self.assertIn("direct Oracle conflict", action["reasoning"])
        self.assertEqual(questioner.oracle_evidence[0]["polarity"], "no")
        self.assertEqual(
            questioner.oracle_evidence[0]["candidate_fingerprint"],
            questioner._candidate_fingerprint,
        )
        self.assertIn("finish_color", questioner.target_profile["facts"])
        self.assertEqual(questioner.target_evidence["finish_color"]["polarity"], "no")
        questioner.oracle_evidence = []
        self.assertIn("finish_color", questioner._grounded_attribute_keys())


    def test_two_direct_yes_answers_override_conservative_negative(self):
        client = FakeClient([
            json.dumps({
                "action": "question",
                "attribute": "finish_color",
                "question": "Is the target wardrobe bright white?",
                "fallback": 0,
                "reasoning": "Check color.",
            }),
            json.dumps({
                "action": "question",
                "attribute": "door_style",
                "question": "Does the target wardrobe have mirrored doors?",
                "fallback": 0,
                "reasoning": "Check doors.",
            }),
            json.dumps({
                "action": "conclusion",
                "conclusion": 0,
                "matches": [],
                "conflicts": [],
                "reasoning": "Conservative fallback.",
            }),
        ])
        questioner = YourQuestioner(self.info, client=client)

        questioner.ask_or_conclude(self.observation())
        questioner.add_answer("Yes, it is bright white.")
        questioner.ask_or_conclude(self.observation())
        questioner.add_answer("Yes, it has mirrored doors.")
        action = questioner.ask_or_conclude(self.observation())

        self.assertEqual(action["conclusion"], 1)
        self.assertIn("two direct", action["reasoning"])
        self.assertEqual(len(client.prompts), 3)

    def test_uncertain_answer_is_not_counted_as_match_or_conflict(self):
        client = FakeClient([
            json.dumps({
                "action": "question",
                "attribute": "finish_color",
                "question": "Is the target wardrobe bright white?",
                "fallback": 0,
                "reasoning": "Check color.",
            }),
            response(
                candidate_profile={"category": "chair"},
                comparison={"conflicts": ["category"]},
                action="conclusion",
                conclusion=0,
            ),
        ])
        questioner = YourQuestioner(self.info, client=client)

        questioner.ask_or_conclude(self.observation())
        questioner.add_answer("Cannot determine from the target image.")
        action = questioner.ask_or_conclude(self.observation())

        direct = questioner._direct_evidence_attributes()
        self.assertEqual(action["conclusion"], 0)
        self.assertEqual(direct["yes"], [])
        self.assertEqual(direct["no"], [])
        self.assertEqual(direct["uncertain"], ["finish_color"])
        self.assertNotIn("finish_color", questioner._grounded_attribute_keys())

    def test_structured_memory_respects_total_character_budget(self):
        questioner = YourQuestioner(self.info, client=FakeClient([]))
        questioner.max_memory_chars = 4000
        questioner.target_profile["facts"] = {
            f"fact_{index}": "x" * 120 for index in range(24)
        }
        questioner.questions = [f"Question {index} " + "q" * 180 for index in range(12)]
        questioner.target_evidence = {
            f"attribute_{index}": {
                "polarity": "yes",
                "proposition": "p" * 200,
                "answer": "a" * 300,
                "has_target_detail": True,
            }
            for index in range(12)
        }
        questioner.oracle_evidence = [
            {
                "question": f"Question {index} " + "q" * 180,
                "answer": "a" * 300,
            }
            for index in range(12)
        ]
        questioner.candidate_profile = {
            f"attribute_{index}": "c" * 120 for index in range(20)
        }

        memory = questioner._memory_json()

        self.assertLessEqual(len(memory), questioner.max_memory_chars)
        self.assertIsInstance(json.loads(memory), dict)

    def test_reset_questions_clears_all_episode_memory(self):
        questioner = YourQuestioner(self.info, client=FakeClient([]))
        questioner.questions = ["Does it have mirrored doors?"]
        questioner.answers = ["Yes"]
        questioner.n_questions = 1
        questioner.target_profile["facts"] = {"door_style": "mirrored"}
        questioner.oracle_evidence = [{"question": "q", "answer": "a"}]
        questioner._asked_question_keys = {"does it have mirrored doors"}
        questioner.target_evidence = {
            "door_style": {"polarity": "yes", "answer": "Yes"}
        }
        questioner._candidate_fingerprint = "candidate"
        questioner._candidate_question_count = 1
        questioner._question_candidate_by_key = {"question": "candidate"}

        questioner.reset_questions()

        self.assertEqual(questioner.questions, [])
        self.assertEqual(questioner.answers, [])
        self.assertEqual(questioner.n_questions, 0)
        self.assertEqual(questioner.target_profile["facts"], {})
        self.assertEqual(questioner.oracle_evidence, [])
        self.assertEqual(questioner._asked_question_keys, set())
        self.assertIsNone(questioner._candidate_fingerprint)
        self.assertEqual(questioner.target_evidence, {})
        self.assertEqual(questioner._question_candidate_by_key, {})


if __name__ == "__main__":
    unittest.main()

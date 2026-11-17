from __future__ import annotations

import pathlib
import sys
import unittest

# Allow both ``python tests/test_integration.py`` and unittest discovery.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from longtraj.integration import (
    AttributionEngine,
    AttributionInputError,
    prepare_trajectory,
    profile_config,
)


def sample():
    return {
        "task_id": "integration-test",
        "task_description": "Complete the task",
        "steps": [
            {"step": "request", "role": "user", "content": "Do it"},
            {"step": "plan", "role": "assistant", "name": "Planner",
             "content": "Use the wrong option"},
            {"step": "result", "role": "tool", "content": "FAILED"},
        ],
    }


class IntegrationTests(unittest.TestCase):
    def test_external_ids_round_trip(self):
        prepared = prepare_trajectory(sample())
        self.assertEqual(prepared.external_step_ids, ("request", "plan", "result"))
        self.assertEqual([s.index for s in prepared.trajectory.steps], [0, 1, 2])

    def test_label_side_input_is_rejected(self):
        row = sample()
        row["ground_truth"] = {"critical_step": 1}
        with self.assertRaises(AttributionInputError):
            prepare_trajectory(row)

    def test_nested_annotation_is_rejected(self):
        row = sample()
        row["meta"] = {"annotation": {"critical_error_step": 1}}
        with self.assertRaises(AttributionInputError):
            prepare_trajectory(row)

    def test_other_evaluation_side_fields_are_rejected(self):
        for key in ("failure_reason", "critical_failure_step", "gold_answer"):
            with self.subTest(key=key):
                row = sample()
                row[key] = "evaluation-only"
                with self.assertRaises(AttributionInputError):
                    prepare_trajectory(row)

    def test_profiles_pin_the_recorded_flags(self):
        rootse = profile_config("rootse_qwen_best")
        self.assertTrue(rootse.span_prior)
        self.assertFalse(rootse.candidate_set)
        self.assertEqual(rootse.max_prompt_chars, 700000)

        sp = profile_config("sp_qwen_clean")
        self.assertTrue(sp.prior_compress)
        self.assertTrue(sp.candidate_strict)
        self.assertTrue(sp.verify)
        self.assertTrue(sp.verify_outdeg)
        self.assertTrue(sp.full_keep)
        self.assertFalse(sp.build_outline)

    def test_dry_run_returns_json_safe_public_result(self):
        engine = AttributionEngine(profile="generic", dry_run=True, workers=1)
        result = engine.locate(sample())
        self.assertEqual(result["task_id"], "integration-test")
        self.assertIn(result["critical_step_id"], {"plan", "result"})
        self.assertEqual(result["selected_step"]["step_id"],
                         result["critical_step_id"])
        self.assertNotIn("diagnostics", result)


if __name__ == "__main__":
    unittest.main()


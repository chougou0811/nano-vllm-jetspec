from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from jetspec_flash_report import compact_report


class FlashReportTest(unittest.TestCase):
    def test_removes_only_bulk_arrays_and_keeps_all_repeat_metrics_and_proofs(self):
        sample = {"tokens_per_second": 123.0, "request_metrics": {"offered_ttft_s": 0.1},
                  "after_cleanup": {"used_blocks": 0}, "requests": ["bulky"],
                  "verified_request_details": ["bulky"], "repeat": 1}
        report = {"passed": True, "status": "complete", "summary": {"speedup": 1.66},
                  "correctness_qualification": {"measured_production_binding": True},
                  "upstream_worker": {"source": "clean", "samples": [sample] * 3, "warmups": [sample]},
                  "artifacts": [{"sha256": "raw-hash"}]}
        before = deepcopy(report)
        result = compact_report(report)
        self.assertEqual(report, before)
        self.assertEqual(len(result["upstream_worker"]["samples"]), 3)
        self.assertNotIn("requests", result["upstream_worker"]["samples"][0])
        self.assertEqual(result["upstream_worker"]["samples"][0]["request_metrics"], sample["request_metrics"])
        self.assertEqual(result["summary"], report["summary"])
        self.assertEqual(result["artifacts"], report["artifacts"])
        self.assertEqual(result["correctness_qualification"], report["correctness_qualification"])

    def test_never_publishes_failed_or_incomplete_run(self):
        for report in ({"passed": False, "status": "complete"}, {"passed": True, "status": "running"}):
            with self.assertRaisesRegex(ValueError, "incomplete"):
                compact_report(report)


if __name__ == "__main__":
    unittest.main()

"""Publish measured medians/raw repeat metrics without bulky delivery arrays."""
import argparse
import hashlib
import json
from pathlib import Path


SAMPLE_BULK_FIELDS = {
    "requests", "nonempty_batch_delivery_times_s", "unique_nonempty_batch_event_gaps_s",
    "verified_request_details",
}


def compact_worker(worker):
    result = {key: value for key, value in worker.items() if key not in ("samples", "warmups")}
    for phase in ("samples", "warmups"):
        result[phase] = [{key: value for key, value in sample.items() if key not in SAMPLE_BULK_FIELDS}
                         for sample in worker[phase]]
    return result


def compact_report(report):
    if report.get("passed") is not True or report.get("status") != "complete":
        raise ValueError("refusing to publish an incomplete comparison")
    result = dict(report)
    for name in ("upstream_worker", "jetspec_flashattn_worker", "jetspec_sdpa_worker"):
        if name in result:
            result[name] = compact_worker(result[name])
    result["publication_note"] = (
        "All warmup and three-repeat metric records and medians are retained. "
        "Bulky per-delivery/token arrays are omitted only from this public copy; "
        "the original full workers are preserved at the artifact paths with SHA256. "
        "No samples are removed, reweighted, or selected by performance."
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, target = Path(args.input).resolve(), Path(args.output).resolve()
    if source == target:
        raise ValueError("publication must not overwrite the full raw comparison")
    report = compact_report(json.loads(source.read_text()))
    report["full_comparison_artifact"] = {"path": str(source),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    report["publisher_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

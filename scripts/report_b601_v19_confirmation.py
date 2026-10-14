"""Summarize the pre-registered fresh-seed V18/V19 confirmation test."""
import json
from pathlib import Path

from report_b601_v19 import compare

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
V18 = ARTIFACTS / "b601-bc-v18-intervention-dagger"
V19 = ARTIFACTS / "b601-bc-v19-blue-weight"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    blocks = range(641000, 651000, 1000)
    runs18 = [load(V18 / f"fresh-v19-confirmation-{seed}.json") for seed in blocks]
    runs19 = [load(V19 / f"fresh-v19-confirmation-{seed}.json") for seed in blocks]
    records18 = [row for run in runs18 for row in run["policy"]["records"]]
    records19 = [row for run in runs19 for row in run["policy"]["records"]]
    assert len(records18) == len(records19) == 120
    assert [row["seed"] for row in records18] == [row["seed"] for row in records19]
    result = compare(records19, records18)
    result["candidate_joint_limit_violations"] = sum(
        not row.get("joint_limits_respected", True) for row in records19)
    result["reference_joint_limit_violations"] = sum(
        not row.get("joint_limits_respected", True) for row in records18)
    result["seed_start"] = 641000
    result["seed_end"] = 650011

    report = [
        "# B601 V19 confirmation results", "",
        "This is the pre-registered follow-up on fresh seeds 641000–650011. V18 and V19 used the same scenes and fixed checkpoints. The paired sign test excludes ties.", "",
        "| Policy | Cubes | Full sorts |", "|---|---:|---:|",
        f"| V19 | {result['candidate_cubes']}/360 | {result['candidate_full_sorts']}/120 |",
        f"| V18 | {result['reference_cubes']}/360 | {result['reference_full_sorts']}/120 |", "",
        f"V19 had {result['wins']} wins, {result['losses']} losses, and {result['ties']} ties (exact two-sided p = {result['exact_sign_test_two_sided_p']:.3f}).",
        f"Joint-limit violations: V19 {result['candidate_joint_limit_violations']}/120; V18 {result['reference_joint_limit_violations']}/120.",
        "", "| Cube | V19 | V18 |", "|---|---:|---:|",
    ]
    for name, counts in result["by_color"].items():
        report.append(f"| {name} | {counts['candidate']}/{counts['trials']} | {counts['reference']}/{counts['trials']} |")
    report += ["", "This follow-up is separate from the first 120-scene comparison. It does not change either model or select a new checkpoint.",
               "All results are from the B601-RS PyBullet simulator. No physical arm was used.", ""]
    (V19 / "CONFIRMATION.md").write_text("\n".join(report), encoding="utf-8")
    (V19 / "confirmation-comparison.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    master_path = ROOT / "portfolio" / "b601-bc-results.json"
    master = load(master_path)
    version = next(item for item in master["versions"] if item["id"] == "b601-bc-v19-blue-weight")
    version["confirmation_comparison_to_v18"] = result
    version["confirmation_runs"] = runs19
    master_path.write_text(json.dumps(master, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

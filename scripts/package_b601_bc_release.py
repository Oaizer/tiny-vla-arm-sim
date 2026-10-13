"""Package compact B601 policy checkpoints and their test reports for download."""
import argparse
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
VERSION_DIRS = [
    "b601-bc-v1-all", "b601-bc-v1-success-only", "b601-bc-v2-192-all",
    "b601-bc-v2-192-success-only", "b601-bc-v3-phase192", "b601-bc-v4-dagger",
    "b601-bc-v5-384", "b601-bc-v5b-384-grid8", "b601-bc-v6-cartesian-canonical",
    "b601-bc-v7-overhead-cartesian", "b601-bc-v8-overhead-dagger",
    "b601-bc-v9-overhead-visual-targets", "b601-bc-v10-overhead-visual-dagger",
    "b601-bc-v11-overhead-visual-yaw",
    "b601-bc-v12-overhead-visual-dagger", "b601-bc-v13-overhead-visual-success-only",
    "b601-bc-v14-overhead-visual-grasp-weighted",
    "b601-bc-v15-overhead-visual-delivery-weighted",
    "b601-bc-v16-overhead-visual-more-demos",
    "b601-bc-v17-overhead-visual-more-demos-dagger",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "b601-policy-models-v6.zip")
    args = parser.parse_args()
    included = []
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in VERSION_DIRS:
            directory = args.artifacts / name
            file_names = ["policy.pt", "RESULTS.json", "heldout-evaluation.json"]
            file_names.extend(sorted(path.name for path in directory.glob("heldout-replication-*.json")))
            for file_name in file_names:
                source = directory / file_name
                if source.is_file():
                    archive.write(source, f"models/{name}/{file_name}")
                    included.append(f"models/{name}/{file_name}")
            report_path = directory / "RESULTS.json"
            if report_path.is_file():
                report = json.loads(report_path.read_text(encoding="utf-8"))
                dataset = Path(report["dataset"])
                if not dataset.is_absolute():
                    dataset = ROOT / dataset
                manifest = dataset / "manifest.json"
                manifest_name = f"datasets/{dataset.name}/manifest.json"
                if manifest.is_file() and manifest_name not in included:
                    archive.write(manifest, manifest_name)
                    included.append(manifest_name)
                extra_dataset = report.get("extra_training_dataset")
                if extra_dataset:
                    extra_datasets = extra_dataset if isinstance(extra_dataset, list) else [extra_dataset]
                    for extra_dataset_name in extra_datasets:
                        extra_path = Path(extra_dataset_name)
                        if not extra_path.is_absolute():
                            extra_path = ROOT / extra_path
                        extra_manifest = extra_path / "manifest.json"
                        extra_name = f"datasets/{extra_path.name}/manifest.json"
                        if extra_manifest.is_file() and extra_name not in included:
                            archive.write(extra_manifest, extra_name)
                            included.append(extra_name)
        for dagger_name in ("b601-dagger-v8-overhead-40", "b601-dagger-v11-yaw-40"):
            dagger_manifest = args.artifacts / dagger_name / "manifest.json"
            manifest_name = f"datasets/{dagger_name}/manifest.json"
            if dagger_manifest.is_file() and manifest_name not in included:
                archive.write(dagger_manifest, manifest_name)
                included.append(manifest_name)
        portfolio_results = ROOT / "portfolio" / "b601-bc-results.json"
        if portfolio_results.is_file():
            archive.write(portfolio_results, "b601-bc-results.json")
            included.append("b601-bc-results.json")
        readme = (
            "B601-RS compact policy models\n"
            "\n"
            "All checkpoints and evaluations in this bundle use the Seeed reBot B601-RS URDF in PyBullet.\n"
            "No physical arm was used.\n"
            "V11 corrects the overhead image-yaw direction. Across 36 paired seeds it placed 27/108 cubes.\n"
            "V9 placed 14/108, V12 placed 23/108, and V13 placed 4/108 on those same seeds.\n"
            "V14 weighted grasp phases 3x and placed 13/108, below V11's 27/108.\n"
            "V15 weighted carry/place/release phases 3x and placed 5/108, below V11's 27/108.\n"
            "V16 adds 100 fresh teacher episodes and placed 66/216 cubes across 72 paired seeds.\n"
            "V11 placed 45/216 cubes on the same seeds. V16 completed one full sort; V11 completed none.\n"
            "V16 had 31 wins, 13 losses, and 28 ties (exact paired sign test p=0.0096).\n"
            "The V16 result is a simulation gain; full sorting remains rare. The V12 correction manifest is included.\n"
            "V17 adds 40 V16-state correction episodes and completed 3/72 sorts. Placements did not improve\n"
            "reliably over V16 (22 wins, 27 losses, 23 ties; exact paired sign test p=0.568).\n"
            "The JSON reports contain seed-level outcomes, model size, and validation measures.\n"
            "The manifests describe generated training data; image arrays are not included. Recreate data\n"
            "with scripts/collect_b601_demos.py and scripts/collect_b601_dagger.py in the source repository.\n"
            "See docs/B601_SIMULATION.md for methods and limits.\n"
        )
        archive.writestr("README.txt", readme)
        included.append("README.txt")
    print(json.dumps({"output": str(args.output), "bytes": args.output.stat().st_size,
                      "files": len(included), "included": included}, indent=2))


if __name__ == "__main__":
    main()

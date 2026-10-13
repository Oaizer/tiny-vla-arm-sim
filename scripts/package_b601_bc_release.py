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
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "b601-policy-models-v2.zip")
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
        dagger_manifest = args.artifacts / "b601-dagger-v8-overhead-40" / "manifest.json"
        if dagger_manifest.is_file():
            archive.write(dagger_manifest, "datasets/b601-dagger-v8-overhead-40/manifest.json")
            included.append("datasets/b601-dagger-v8-overhead-40/manifest.json")
        portfolio_results = ROOT / "portfolio" / "b601-bc-results.json"
        if portfolio_results.is_file():
            archive.write(portfolio_results, "b601-bc-results.json")
            included.append("b601-bc-results.json")
        readme = (
            "B601-RS compact policy models\n"
            "\n"
            "All checkpoints and evaluations in this bundle use the Seeed reBot B601-RS URDF in PyBullet.\n"
            "No physical arm was used. The learned policies have not completed a full three-cube sort.\n"
            "V11 corrects the overhead image-yaw direction. Across 36 paired seeds it placed 27/108 cubes.\n"
            "V9 placed 14/108 cubes on those same seeds. Neither policy completed a full sort.\n"
            "The paired seed sign test is not conclusive (15 wins, 6 losses, 15 ties; p=0.078).\n"
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

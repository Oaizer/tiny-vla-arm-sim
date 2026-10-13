"""Refresh the B601 policy entries and trial clips on the static portfolio page."""
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "portfolio" / "b601-bc-results.json"
ITERATIONS = ROOT / "portfolio" / "iterations.json"


def main():
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    page = json.loads(ITERATIONS.read_text(encoding="utf-8"))
    existing = {item["id"]: index for index, item in enumerate(page["iterations"])}
    for version in results["versions"]:
        runs = version["runs"]
        policy_sorts = sum(run["policy"]["successes"] for run in runs)
        policy_objects = sum(run["policy"]["objects_placed"] for run in runs)
        objects_attempted = sum(run["policy"]["objects_attempted"] for run in runs)
        trials = sum(run["trials"] for run in runs)
        teacher_sorts = sum(run["teacher"]["successes"] for run in runs)
        teacher_objects = sum(run["teacher"]["objects_placed"] for run in runs)
        teacher_attempted = sum(run["teacher"]["objects_attempted"] for run in runs)
        version_number = version["id"].split("-bc-v")[-1].split("-")[0]
        iteration_id = f"b601-bc-v{version_number}"
        if "success-only" in version["id"]:
            iteration_id += "-successful-only"
        elif "v5b" in version["id"]:
            iteration_id = "b601-bc-v5b"
        question = "Can this change improve the small policy's real sorting results?"
        if version_number == "8":
            question = "Can teacher corrections on policy-visited states improve sorting?"
        elif version_number == "9":
            question = "Can camera-derived target features improve cube placement?"
        elif version_number == "10":
            question = "Do camera-derived targets and DAgger corrections work better together?"
        elif version_number == "11":
            question = "Does correcting the overhead camera's yaw direction improve face-aligned grasps?"
        elif version_number == "12":
            question = "Can teacher corrections on V11-visited states improve grasping?"
        elif version_number == "13":
            question = "Does filtering to successful teacher sorts improve the learned policy?"
        paired_text = ("matched held-out seeds" if len(runs) == 1 else
                       f"{len(runs)} paired held-out seed groups")
        evidence = (f"{trials} trials, {paired_text}; training used {version['training_samples']:,} samples "
                    f"from {version['training_episodes']} episodes.")
        measure = (f"Policy {policy_sorts}/{trials} full sorts; {policy_objects}/{objects_attempted} cubes. "
                   f"Teacher {teacher_sorts}/{trials} sorts; {teacher_objects}/{teacher_attempted} cubes.")
        detail = (f"{version['change']}. The selected checkpoint is epoch {version['best_epoch']} "
                  f"with {version['parameters']:,} parameters and {version['cpu_latency_ms']:.2f} ms "
                  "CPU inference per action. Waypoint scores are in the data file; they do not replace task success.")
        if version.get("dagger_data"):
            data = version["dagger_data"]
            detail += (f" The correction set had {data['episodes']} episodes and "
                       f"{data['successful_episodes']} complete teacher-guided sorts; "
                       f"{data['training_samples']} correction samples entered training.")
        if version_number == "9":
            detail += " This is an early object-level gain; neither V7 nor V9 completed a full sort in these 24 trials."
        elif version_number == "10":
            detail += (" Combining the visual cues with DAgger did not improve V9 overall: both placed "
                       "8/72 cubes and completed 0/24 full sorts. On the fresh seed group, V10 placed "
                       "3/36 cubes and V9 placed 6/36.")
        elif version_number == "11":
            detail += (" The overhead image-angle feature previously reversed world yaw. After correcting it, "
                       "V11 placed 27/108 cubes across 36 paired seeds; V9 placed 14/108 on those same seeds. "
                       "Neither completed a full sort. The paired trial sign test is promising but not conclusive "
                       "(15 wins, 6 losses, 15 ties; p = 0.078).")
        elif version_number == "12":
            detail += (" V12 added 768 teacher corrections from 40 V11-visited episodes. It placed 23/108 cubes "
                       "across the same 36 paired seeds, compared with V11's 27/108. Neither completed a full sort; "
                       "there were 11 V12-favored trials, 11 V11-favored trials, and 14 ties. DAgger did not improve "
                       "the result.")
        elif version_number == "13":
            detail += (" Training only on 56 successful demonstration episodes lowered validation waypoint error, "
                       "but placed just 4/108 cubes across the same 36 paired seeds. V11 placed 27/108. "
                       "The lower waypoint error did not predict task success.")
        item = {
            "id": iteration_id,
            "title": f"{version['label']} · {version['change']}",
            "kind": "B601-RS learned policy",
            "question": question,
            "evidence": evidence,
            "measure": measure,
            "detail": detail,
            "result": f"{policy_objects}/{objects_attempted} cubes; {policy_sorts}/{trials} full sorts",
            "status": "warn" if policy_sorts == 0 else "good",
            "source": "../docs/B601_SIMULATION.md#learned-policy-iterations",
            "data": "b601-bc-results.json",
            "archive": "https://github.com/Oaizer/tiny-vla-arm-sim/releases/download/b601-policy-models-v3/b601-policy-models-v3.zip",
        }
        if iteration_id in existing:
            page["iterations"][existing[iteration_id]] = item
        else:
            existing[iteration_id] = len(page["iterations"])
            page["iterations"].append(item)

    videos = [
        {
            "title": "Face-aligned teacher · overhead view · seed 607000",
            "scene": "Scripted B601-RS teacher run in PyBullet. The teacher uses cube yaw to turn the fingers toward a face. All three cubes reach their pads. 1280 × 768, 10 fps, 47.7 seconds.",
            "videos": [{"name": "Face-aligned B601 teacher", "outcome": "SUCCESS",
                        "file": "media/b601-rs-sort-v7-face-aligned-seed-607000.mp4"}],
        },
        {
            "title": "Learned V9 policy · seed 622009",
            "scene": "Held-out B601-RS simulation example. The visual-target policy places two cubes and misses the third. The full paired test scored 8/72 cubes and 0/24 complete sorts. This selected clip is one example, not the whole result. 1280 × 768, 10 fps.",
            "videos": [{"name": "V9 visual-target policy", "outcome": "PARTIAL",
                        "file": "media/b601-rs-bc-v9-seed-622009.mp4"}],
        },
        {
            "title": "Learned V11 policy · seed 623004",
            "scene": "Held-out B601-RS simulation example using the corrected world-yaw estimate. V11 places two cubes and misses the third. Across 36 paired seeds it placed 27/108 cubes, compared with V9's 14/108. Neither completed a full sort. The paired result is still uncertain. 1280 × 768, 10 fps.",
            "videos": [{"name": "V11 yaw-corrected policy", "outcome": "PARTIAL",
                        "file": "media/b601-rs-bc-v11-seed-623004.mp4"}],
        },
    ]
    trial_titles = {trial["title"] for trial in page["trials"]}
    for trial in videos:
        if trial["title"] not in trial_titles:
            page["trials"].append(trial)
    page["updated"] = date.today().isoformat()
    ITERATIONS.write_text(json.dumps(page, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"iterations": len(page["iterations"]), "trials": len(page["trials"]),
                      "updated": page["updated"]}, indent=2))


if __name__ == "__main__":
    main()

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
        elif version_number == "14":
            question = "Does extra training weight on grasp phases improve object placements?"
        elif version_number == "15":
            question = "Does extra training weight on carry, place, and release phases improve sorting?"
        elif version_number == "16":
            question = "Do 100 fresh teacher episodes improve policy generalization?"
        elif version_number == "17":
            question = "Do teacher corrections from V16 policy states improve the expanded-data model?"
        elif version_number == "18":
            question = "Can error-triggered teacher takeovers make DAgger data more useful?"
        elif version_number == "19":
            question = "Does weighting blue-cube instructions improve the balance across colors?"
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
        elif version_number == "14":
            detail += (" Training multiplied descend, close, and lift waypoint loss by 3. On the same 36 held-out "
                       "seeds, V14 placed 13/108 cubes and V11 placed 27/108; neither completed a full sort. "
                       "V14 had more placements in 4 trials, fewer in 13, and tied in 19 (exact paired sign test "
                       "p = 0.049). The grasp-phase weighting reduced task performance, so V11 remains the lead policy.")
        elif version_number == "15":
            detail += (" Training multiplied carry, place, and release waypoint loss by 3. On the same 36 held-out "
                       "seeds, V15 placed 5/108 cubes and V11 placed 27/108; neither completed a full sort. "
                       "V15 had more placements in 3 trials, fewer in 19, and tied in 14 (exact paired sign test "
                       "p = 0.00086). Delivery-phase weighting reduced task performance.")
        elif version_number == "16":
            detail += (" The model adds 80 training episodes from a fresh 100-episode set; its validation split "
                       "remains unchanged. Across 120 matched held-out seeds, V16 placed 109/360 cubes and completed "
                       "1/120 full sorts. V11 placed 72/360 cubes and completed none. V16 had more placements in "
                       "49 trials, fewer in 17, and tied in 54 (exact paired sign test p = 0.00010).")
        elif version_number == "17":
            detail += (" V17 adds 768 teacher corrections from 40 V16-visited episodes to the V16 training data. "
                       "Across 120 matched held-out seeds, V17 placed 125/360 cubes and completed 5/120 full sorts; "
                       "V16 placed 109/360 and completed 1/120. V17 had more placements in 47 trials, fewer in 40, "
                       "and tied in 33 (exact paired sign test p = 0.520). V17 has the best observed count, but its "
                       "gain over V16 is not conclusive.")
        elif version_number == "18":
            comparison = version["comparison_to_v17"]
            data = version["dagger_data"]
            colors = comparison["by_object"]
            detail += (f" V18 adds {data['training_samples']} intervention-guided correction samples from "
                       f"{data['episodes']} episodes, including {data['successful_episodes']} complete sorts. "
                       f"Across 120 matched seeds, V18 placed {comparison['candidate_cubes']}/360 cubes and "
                       f"completed {comparison['candidate_full_sorts']}/120 full sorts. V17 placed "
                       f"{comparison['reference_cubes']}/360 cubes and completed "
                       f"{comparison['reference_full_sorts']}/120. V18 had {comparison['wins']} wins, "
                       f"{comparison['losses']} losses, and {comparison['ties']} ties "
                       f"(exact paired sign test p = {comparison['exact_sign_test_two_sided_p']:.3f}). "
                       f"By color, green placements rose from {colors['green cube']['reference']} to "
                       f"{colors['green cube']['candidate']}, while blue placements fell from "
                       f"{colors['blue cube']['reference']} to {colors['blue cube']['candidate']}. "
                       "This is the highest observed count, but the gain over V17 is not conclusive.")
        elif version_number == "19":
            v18 = version["comparison_to_v18"]
            v17 = version["comparison_to_v17"]
            colors = v18["by_color"]
            detail += (f" On 120 fresh matched seeds, V19 placed {v18['candidate_cubes']}/360 cubes and completed "
                       f"{v18['candidate_full_sorts']}/120 full sorts. V18 placed {v18['reference_cubes']}/360 "
                       f"and completed {v18['reference_full_sorts']}/120. V19 had {v18['wins']} wins, "
                       f"{v18['losses']} losses, and {v18['ties']} ties (exact sign test p = "
                       f"{v18['exact_sign_test_two_sided_p']:.3f}); this gain is uncertain. Against V17, "
                       f"V19 placed {v17['candidate_cubes']}/360 versus {v17['reference_cubes']}/360 "
                       f"(p = {v17['exact_sign_test_two_sided_p']:.3f}). By color against V18, placements "
                       f"were red {colors['red cube']['candidate']} vs {colors['red cube']['reference']}, "
                       f"blue {colors['blue cube']['candidate']} vs {colors['blue cube']['reference']}, and "
                       f"green {colors['green cube']['candidate']} vs {colors['green cube']['reference']}. "
                       "The V19 lead over V17 is promising, but more fresh tests are needed to confirm it.")
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
            "archive": "https://github.com/Oaizer/tiny-vla-arm-sim/releases/download/b601-policy-models-v9/b601-policy-models-v9.zip",
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
            "scene": "Held-out B601-RS simulation example using the corrected world-yaw estimate. The large view is overhead; the inset is a synchronized oblique camera. V11 places two cubes and misses the third. Across 36 paired seeds it placed 27/108 cubes, compared with V9's 14/108. Neither completed a full sort. The paired result is still uncertain. 1280 × 768, 10 fps.",
            "videos": [{"name": "V11 yaw-corrected policy", "outcome": "PARTIAL",
                        "file": "media/b601-rs-bc-v11-seed-623004-dual.mp4"}],
        },
        {
            "title": "Learned V16 expanded data · seed 625004",
            "scene": "Held-out B601-RS simulation trial from seed group 625. V16 completes the full three-cube sort. The large view is overhead and the inset is a synchronized oblique camera. Across 120 matched seeds, V16 placed 109/360 cubes and completed 1/120 full sorts; V11 placed 72/360 and completed none. 1280 × 768, 10 fps.",
            "videos": [{"name": "V16 full-sort policy", "outcome": "SUCCESS",
                        "file": "media/b601-rs-bc-v16-seed-625004-dual.mp4"}],
        },
        {
            "title": "Learned V17 DAgger candidate · seed 624003",
            "scene": "Held-out B601-RS simulation trial from seed group 624. V17 completes all three placements. Across 120 matched seeds, it placed 125/360 cubes and completed 5/120 full sorts. V16 placed 109/360 cubes and completed 1/120 full sorts, but the paired placement comparison is inconclusive (p = 0.520). The large view is overhead; the inset is a synchronized oblique camera. 1280 × 768, 10 fps.",
            "videos": [{"name": "V17 DAgger candidate", "outcome": "SUCCESS",
                        "file": "media/b601-rs-bc-v17-seed-624003-dual.mp4"}],
        },
        {
            "title": "Learned V18 intervention-guided DAgger · seed 626006",
            "scene": "Held-out B601-RS simulation trial from seed group 626. V18 completes all three placements. Across 120 matched seeds, it placed 131/360 cubes and completed 6/120 full sorts. V17 placed 125/360 cubes and completed 5/120. The paired comparison is uncertain (41 wins, 35 losses, 44 ties; p = 0.567). The overhead view includes a synchronized oblique camera inset. 1280 × 768, 10 fps.",
            "videos": [{"name": "V18 full-sort policy", "outcome": "SUCCESS",
                        "file": "media/b601-rs-bc-v18-seed-626006-dual.mp4"}],
        },
        {
            "title": "Learned V19 blue-weighted policy · seed 632001",
            "scene": "Held-out B601-RS simulation trial from the fresh seed set. V19 completes all three placements. Across 120 matched seeds, V19 placed 150/360 cubes and completed 14/120 full sorts; V18 placed 130/360 and completed 8/120. The placement gain over V18 is uncertain (41 wins, 30 losses, 49 ties; p = 0.235). V19 improved over V17 on this test (52 wins, 31 losses, 37 ties; p = 0.028). The overhead view includes a synchronized oblique camera inset. 1280 × 768, 10 fps.",
            "videos": [{"name": "V19 blue-weighted policy", "outcome": "SUCCESS",
                        "file": "media/b601-rs-bc-v19-seed-632001-dual.mp4"}],
        },
    ]
    trial_titles = {trial["title"]: index for index, trial in enumerate(page["trials"])}
    for trial in videos:
        if trial["title"] in trial_titles:
            page["trials"][trial_titles[trial["title"]]] = trial
        else:
            page["trials"].append(trial)
    page["updated"] = date.today().isoformat()
    ITERATIONS.write_text(json.dumps(page, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"iterations": len(page["iterations"]), "trials": len(page["trials"]),
                      "updated": page["updated"]}, indent=2))


if __name__ == "__main__":
    main()

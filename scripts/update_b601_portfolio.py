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
            confirmation = version.get("confirmation_comparison_to_v18")
            if confirmation:
                detail += (f" On a separate confirmation set, V19 placed {confirmation['candidate_cubes']}/360 "
                           f"cubes and completed {confirmation['candidate_full_sorts']}/120 sorts; V18 placed "
                           f"{confirmation['reference_cubes']}/360 and completed "
                           f"{confirmation['reference_full_sorts']}/120. The paired comparison was "
                           f"{confirmation['wins']} wins, {confirmation['losses']} losses, and "
                           f"{confirmation['ties']} ties (p = "
                           f"{confirmation['exact_sign_test_two_sided_p']:.3f}). Neither policy had a joint-limit violation.")
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
            "archive": "https://github.com/Oaizer/tiny-vla-arm-sim/releases/download/b601-policy-models-v10/b601-policy-models-v10.zip",
        }
        if iteration_id in existing:
            page["iterations"][existing[iteration_id]] = item
        else:
            existing[iteration_id] = len(page["iterations"])
            page["iterations"].append(item)

    grasp_pilot_path = ROOT / "portfolio" / "b601-grasp-height-pilot.json"
    if grasp_pilot_path.is_file():
        pilot = json.loads(grasp_pilot_path.read_text(encoding="utf-8"))
        baseline = pilot["summaries"]["0.08"]
        raised = pilot["summaries"]["0.1"]
        comparison = pilot["paired_vs_0.08m"]["0.1"]
        item = {
            "id": "b601-grasp-height-pilot",
            "title": "Teacher grasp height · 12-scene pilot",
            "kind": "B601-RS teacher waypoint pilot",
            "question": "Does a higher pick waypoint improve stable grasps and placements?",
            "evidence": "12 matched simulation scenes; one pick attempt per cube; face-aligned wrist yaw.",
            "measure": (f"0.10 m: {raised['objects_placed']}/36 cubes and {raised['full_sorts']}/12 sorts. "
                        f"0.08 m: {baseline['objects_placed']}/36 cubes and {baseline['full_sorts']}/12 sorts."),
            "detail": (f"The 0.10 m waypoint had {comparison['wins']} wins, {comparison['losses']} losses, "
                       f"and {comparison['ties']} ties by seed (exact p = "
                       f"{comparison['exact_sign_test_two_sided_p']:.4f}). It improved pilot task outcomes. "
                       f"Mean corner proximity changed from {baseline['mean_corner_proximity']:.3f} to "
                       f"{raised['mean_corner_proximity']:.3f}; most contacts still landed near the top or "
                       "bottom rim. This is pilot evidence, not a learned-policy result."),
            "result": f"{raised['objects_placed']}/36 cubes; {raised['full_sorts']}/12 full sorts",
            "status": "warn",
            "source": "../docs/B601_V20_PROTOCOL.md",
            "data": "b601-grasp-height-pilot.json",
        }
        if item["id"] in existing:
            page["iterations"][existing[item["id"]]] = item
        else:
            existing[item["id"]] = len(page["iterations"])
            page["iterations"].append(item)

    v20_path = ROOT / "portfolio" / "b601-v20-evaluation.json"
    if v20_path.is_file():
        result = json.loads(v20_path.read_text(encoding="utf-8"))
        v19, v20 = result["v19"], result["v20"]
        paired = result["paired_placements_v20_vs_v19"]
        yaw19 = v19["contact_audit"]["mean_face_yaw_error_deg"]
        yaw20 = v20["contact_audit"]["mean_face_yaw_error_deg"]
        corner19 = v19["contact_audit"]["mean_corner_proximity"]
        corner20 = v20["contact_audit"]["mean_corner_proximity"]
        no_contact19 = v19["contact_audit"]["objects_with_no_finger_contact"]
        no_contact20 = v20["contact_audit"]["objects_with_no_finger_contact"]
        item = {
            "id": "b601-bc-v20",
            "title": "BC V20 · Higher teacher pick height",
            "kind": "B601-RS learned policy",
            "question": "Does training on a higher grasp waypoint improve the learned policy's contact and placement results?",
            "evidence": "120 matched held-out simulation scenes; 5,440 training samples from 200 train episodes; contact audit enabled.",
            "measure": (f"V20: {v20['cubes_placed']}/360 cubes; {v20['full_sorts']}/120 sorts. "
                        f"V19: {v19['cubes_placed']}/360 cubes; {v19['full_sorts']}/120 sorts."),
            "detail": (f"V20 had {paired['wins']} wins, {paired['losses']} losses, and {paired['ties']} ties "
                       f"by scene (exact paired sign test p = {paired['exact_sign_test_two_sided_p']:.3f}). "
                       f"Its observed count was lower by {abs(paired['difference_cubes'])} cubes; this test "
                       "does not show a clear difference at the 0.05 level. Mean contact-to-corner score "
                       f"improved from {corner19:.3f} to {corner20:.3f}, but mean wrist-to-face yaw error "
                       f"rose from {yaw19:.2f}° to {yaw20:.2f}°. No-contact objects rose from "
                       f"{no_contact19} to {no_contact20}. Most contacts remained near the top or bottom rim. "
                       "The V20 set also used fewer training samples than V19, so this is not a clean test of "
                       "grasp height alone. The teacher pilot did not translate into a learned-policy improvement. Next, hold the "
                       "V19 data and grasp height fixed while testing extra yaw loss on grasp waypoints."),
            "result": f"{v20['cubes_placed']}/360 cubes; {v20['full_sorts']}/120 full sorts",
            "status": "warn",
            "source": "../docs/B601_V20_RESULTS.md",
            "data": "b601-v20-evaluation.json",
        }
        if item["id"] in existing:
            page["iterations"][existing[item["id"]]] = item
        else:
            existing[item["id"]] = len(page["iterations"])
            page["iterations"].append(item)

    v22_path = ROOT / "portfolio" / "b601-v22-evaluation.json"
    if v22_path.is_file():
        result = json.loads(v22_path.read_text(encoding="utf-8"))
        v19, v22 = result["v19"], result["v22"]
        paired = result["paired_placements_v22_vs_v19"]
        full_sorts = result["paired_full_sorts_v22_vs_v19"]
        audit19, audit22 = v19["contact_audit"], v22["contact_audit"]
        confirmation = result.get("independent_confirmation")
        confirm_text = ""
        if confirmation:
            c19, c22 = confirmation["v19"], confirmation["v22"]
            cp = confirmation["paired_placements_v22_vs_v19"]
            cs = confirmation["paired_full_sorts_v22_vs_v19"]
            ca19, ca22 = c19["contact_audit"], c22["contact_audit"]
            confirm_text = (f" Independent confirmation (seeds {confirmation['seed_range'][0]}–"
                            f"{confirmation['seed_range'][1]}): V22 placed {c22['cubes_placed']}/360 cubes "
                            f"and completed {c22['full_sorts']}/120 sorts; V19 placed "
                            f"{c19['cubes_placed']}/360 and completed {c19['full_sorts']}/120. "
                            f"V22 had {cp['wins']} placement wins, {cp['losses']} losses, and {cp['ties']} ties "
                            f"(p = {cp['exact_sign_test_two_sided_p']:.2g}); full-sort outcomes were "
                            f"{cs['wins']} wins, {cs['losses']} losses, and {cs['ties']} ties "
                            f"(p = {cs['exact_sign_test_two_sided_p']:.3f}). Yaw error fell from "
                            f"{ca19['mean_face_yaw_error_deg']:.2f}° to {ca22['mean_face_yaw_error_deg']:.2f}°; "
                            f"no-contact objects fell from {ca19['objects_with_no_finger_contact']} to "
                            f"{ca22['objects_with_no_finger_contact']}.")
        item = {
            "id": "b601-bc-v22",
            "title": "BC V22 · Camera-corrected grasp poses",
            "kind": "B601-RS hybrid VLA policy",
            "question": "Can calibrated overhead pixels improve the policy's grasp alignment?",
            "evidence": "120 fresh matched held-out scenes; frozen V19 checkpoint; contact audit enabled.",
            "measure": (f"V22: {v22['cubes_placed']}/360 cubes; {v22['full_sorts']}/120 sorts. "
                        f"V19: {v19['cubes_placed']}/360 cubes; {v19['full_sorts']}/120 sorts."),
            "detail": (f"V22 had {paired['wins']} wins, {paired['losses']} losses, and {paired['ties']} ties "
                       f"by scene (exact paired sign test p = {paired['exact_sign_test_two_sided_p']:.6f}). "
                       f"Mean wrist-to-face yaw error fell from {audit19['mean_face_yaw_error_deg']:.2f}° "
                       f"to {audit22['mean_face_yaw_error_deg']:.2f}°. Objects with no finger contact fell "
                       f"from {audit19['objects_with_no_finger_contact']} to "
                       f"{audit22['objects_with_no_finger_contact']}; mean corner proximity fell from "
                       f"{audit19['mean_corner_proximity']:.3f} to {audit22['mean_corner_proximity']:.3f}. "
                       f"The first test's paired full-sort result was uncertain ({full_sorts['wins']} wins, "
                       f"{full_sorts['losses']} losses, {full_sorts['ties']} ties; "
                       f"p = {full_sorts['exact_sign_test_two_sided_p']:.3f}).{confirm_text}"),
            "result": f"{v22['cubes_placed']}/360 cubes; {v22['full_sorts']}/120 full sorts",
            "status": "good",
            "source": "../docs/B601_V22_RESULTS.md",
            "data": "b601-v22-evaluation.json",
        }
        if item["id"] in existing:
            page["iterations"][existing[item["id"]]] = item
        else:
            existing[item["id"]] = len(page["iterations"])
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
            "title": "Matched learned policies · confirmation seed 644004",
            "scene": "Paired held-out B601-RS simulation scene from the second fresh evaluation. V19 completes all three placements; V18 places none. Across 120 confirmation seeds, V19 placed 145/360 cubes and completed 9/120 sorts; V18 placed 124/360 and also completed 9/120. The paired placement comparison is uncertain (49 wins, 34 losses, 37 ties; p = 0.124). Both clips use the same scene, overhead view, and synchronized oblique camera inset. 1280 × 768, 10 fps.",
            "videos": [{"name": "V19 confirmation policy", "outcome": "SUCCESS",
                        "file": "media/b601-rs-bc-v19-confirmation-seed-644004-dual.mp4"},
                       {"name": "V18 matched policy", "outcome": "FAILED",
                        "file": "media/b601-rs-bc-v18-confirmation-seed-644004-dual.mp4"}],
        },
        {
            "title": "Matched V19 and V20 grasp · seed 661001",
            "scene": "Same held-out B601-RS simulation scene. V19 places all three cubes; V20 places one. Across the full matched set, V19 placed 165/360 cubes and completed 16/120 sorts; V20 placed 131/360 and completed 8/120. V20 had 30 wins, 46 losses, and 44 ties (p = 0.085), so the lower count is not conclusive at the 0.05 level. Both clips use the overhead view with synchronized oblique inset. 1280 × 768, 10 fps.",
            "videos": [{"name": "V19 · 3/3 placements", "outcome": "SUCCESS",
                        "file": "media/b601-rs-v19-v20-matched-seed-661001-v19.mp4"},
                       {"name": "V20 · 1/3 placements", "outcome": "PARTIAL",
                        "file": "media/b601-rs-v19-v20-matched-seed-661001-v20.mp4"}],
        },
        {
            "title": "Matched V19 and V22 grasp · seed 673003",
            "scene": "Same first-test B601-RS simulation scene. V19 places none of the cubes; V22's calibrated camera grasp correction places all three. Across the 120 first-test scenes, V19 placed 156/360 cubes and completed 18/120 sorts; V22 placed 222/360 and completed 27/120. V22 had 60 wins, 26 losses, and 34 ties (p = 0.000317). An independent test on seeds 674000–674119 confirmed the placement gain: V19 placed 141/360, V22 placed 219/360; the paired result was 67 wins, 16 losses, and 37 ties (p = 1.39e-8). Both clips use the overhead view with a synchronized oblique inset. 1280 × 768, 10 fps.",
            "videos": [{"name": "V19 · 0/3 placements", "outcome": "FAILED",
                        "file": "media/b601-rs-v19-v22-matched-seed-673003-v19.mp4"},
                       {"name": "V22 · 3/3 placements", "outcome": "SUCCESS",
                        "file": "media/b601-rs-v19-v22-matched-seed-673003-v22.mp4"}],
        },
        {
            "title": "V22 confirmation · matched seed 674007",
            "scene": "Independent matched B601-RS simulation scene from the confirmation set. V19 places none of the cubes; V22 places all three. Across the 120 confirmation scenes, V19 placed 141/360 cubes and completed 8/120 sorts; V22 placed 219/360 cubes and completed 27/120. V22 had 67 placement wins, 16 losses, and 37 ties (p = 1.39e-8). Full-sort outcomes were 26 wins, 7 losses, and 87 ties (p = 0.0013). Both clips use the overhead view with synchronized oblique inset. 1280 × 768, 10 fps.",
            "videos": [{"name": "V19 · 0/3 placements", "outcome": "FAILED",
                        "file": "media/b601-rs-v19-v22-confirmation-seed-674007-v19.mp4"},
                       {"name": "V22 · 3/3 placements", "outcome": "SUCCESS",
                        "file": "media/b601-rs-v19-v22-confirmation-seed-674007-v22.mp4"}],
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

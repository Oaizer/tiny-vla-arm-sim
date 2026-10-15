"""Create compact portfolio data for the matched V19/V21 yaw-loss test."""
import json
from pathlib import Path

from report_b601_v22 import exact_sign_p, summarize_contacts

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts" / "b601-v21-evaluation"


def main():
    baseline = json.loads((SOURCE / "v19-671000-120.json").read_text(encoding="utf-8"))
    candidate = json.loads((SOURCE / "v21-671000-120.json").read_text(encoding="utf-8"))
    old, new = baseline["policy"]["records"], candidate["policy"]["records"]
    wins = losses = ties = 0
    paired = []
    for left, right in zip(old, new):
        assert left["seed"] == right["seed"]
        delta = right["objects_placed"] - left["objects_placed"]
        wins += delta > 0
        losses += delta < 0
        ties += delta == 0
        paired.append({"seed": left["seed"], "v19_cubes": left["objects_placed"],
                       "v21_cubes": right["objects_placed"]})
    summary = {
        "title": "B601 V21 grasp-yaw loss-weight test",
        "protocol": "120 matched B601-RS simulation scenes; seeds 671000–671119; V19 data and schedule fixed; 3× yaw loss on descend, close, and lift waypoints.",
        "simulator_only": True,
        "candidate": {"version": "V21", "training": True, "base_checkpoint": "V19",
                      "change": "Multiply wrist-yaw loss by 3 on descend, close, and lift waypoints.",
                      "parameters": 427798},
        "v19": {"trials": len(old), "full_sorts": baseline["policy"]["successes"],
                "cubes_placed": baseline["policy"]["objects_placed"],
                "contact_audit": summarize_contacts(old)},
        "v21": {"trials": len(new), "full_sorts": candidate["policy"]["successes"],
                "cubes_placed": candidate["policy"]["objects_placed"],
                "contact_audit": summarize_contacts(new)},
        "paired_placements_v21_vs_v19": {"wins": wins, "losses": losses, "ties": ties,
            "exact_sign_test_two_sided_p": exact_sign_p(wins, losses),
            "difference_cubes": candidate["policy"]["objects_placed"] - baseline["policy"]["objects_placed"]},
        "paired_seed_results": paired,
    }
    destination = ROOT / "portfolio" / "b601-v21-evaluation.json"
    destination.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(destination),
                      "v19_cubes": summary["v19"]["cubes_placed"],
                      "v21_cubes": summary["v21"]["cubes_placed"],
                      "paired": summary["paired_placements_v21_vs_v19"]}, indent=2))


if __name__ == "__main__":
    main()

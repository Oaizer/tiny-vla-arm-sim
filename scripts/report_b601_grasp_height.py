"""Report the matched B601 teacher grasp-height pilot."""
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts" / "b601-grasp-height-pilot" / "seed-651000.json"
OUTPUT = SOURCE.parent / "COMPARISON.md"
PORTFOLIO = ROOT / "portfolio" / "b601-grasp-height-pilot.json"


def exact_sign(wins, losses):
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2**n
    return min(1.0, 2 * tail)


def summarize(rows):
    objects = [obj for row in rows for obj in row["objects"]]
    contacts = [obj["grasp_contact"] for obj in objects if obj["grasp_contact"]]
    points = [point for contact in contacts for point in contact["contacts"]]
    colors = {name: sum(obj["placed"] for obj in objects if obj["object"] == name)
              for name in ("red cube", "blue cube", "green cube")}
    return {
        "seeds": len(rows),
        "objects_attempted": len(objects),
        "objects_grasped": sum(obj["grasped"] for obj in objects),
        "objects_placed": sum(obj["placed"] for obj in objects),
        "full_sorts": sum(row["placed"] == 3 for row in rows),
        "by_color": colors,
        "mean_corner_proximity": mean(
            contact["mean_corner_proximity"] for contact in contacts
            if contact["mean_corner_proximity"] is not None),
        "contact_points_near_top_or_bottom": sum(
            abs(point["cube_local_contact_m"][2]) >= .015 for point in points),
        "contact_points": len(points),
    }


def main():
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    heights = {}
    for height in data["heights_m"]:
        heights[height] = [row for row in data["records"]
                           if row["grasp_height_m"] == height]
    summaries = {str(height): summarize(rows) for height, rows in heights.items()}
    baseline = {row["seed"]: row for row in heights[.08]}
    comparisons = {}
    for height in (.09, .10):
        candidate = {row["seed"]: row for row in heights[height]}
        wins = losses = ties = 0
        for seed in baseline:
            difference = candidate[seed]["placed"] - baseline[seed]["placed"]
            wins += difference > 0
            losses += difference < 0
            ties += difference == 0
        comparisons[str(height)] = {
            "wins": wins, "losses": losses, "ties": ties,
            "exact_sign_test_two_sided_p": exact_sign(wins, losses),
        }
    report = {
        "protocol": data["protocol"], "seed_range": data["seed_range"],
        "summaries": summaries, "paired_vs_0.08m": comparisons,
        "runs": data["records"],
        "simulator_only": True,
    }
    PORTFOLIO.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# B601 teacher grasp-height pilot", "",
        "The test changed only the TCP height for the pick descend and close waypoints. Each of 12 matched scenes used one attempt per cube. Wrist yaw stayed aligned to the cube faces.", "",
        "| Pick height | Grasped | Placed | Full sorts | Red | Blue | Green |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for height in data["heights_m"]:
        summary = summaries[str(height)]
        lines.append(
            f"| {height:.2f} m | {summary['objects_grasped']}/36 | "
            f"{summary['objects_placed']}/36 | {summary['full_sorts']}/12 | "
            f"{summary['by_color']['red cube']} | {summary['by_color']['blue cube']} | "
            f"{summary['by_color']['green cube']} |")
    test = comparisons["0.1"]
    lines += [
        "", f"At 0.10 m, the teacher placed {summaries['0.1']['objects_placed']}/36 cubes, versus "
        f"{summaries['0.08']['objects_placed']}/36 at 0.08 m. By seed, 0.10 m had "
        f"{test['wins']} wins, {test['losses']} losses, and {test['ties']} ties "
        f"(exact two-sided p = {test['exact_sign_test_two_sided_p']:.4f}). It completed "
        f"{summaries['0.1']['full_sorts']}/12 sorts, compared with {summaries['0.08']['full_sorts']}/12.",
        "", f"The average corner-proximity score was {summaries['0.08']['mean_corner_proximity']:.3f} "
        f"at 0.08 m and {summaries['0.1']['mean_corner_proximity']:.3f} at 0.10 m "
        "(0 is centered on a face; 1 is at a corner). Contact points remained near the top or bottom "
        f"rim in {summaries['0.1']['contact_points_near_top_or_bottom']}/"
        f"{summaries['0.1']['contact_points']} contacts at 0.10 m. The higher waypoint improved "
        "task outcomes in this pilot, but it did not move contact points to the middle of cube faces.",
        "", "All tests use B601-RS PyBullet. No physical arm was used.", "",
    ]
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

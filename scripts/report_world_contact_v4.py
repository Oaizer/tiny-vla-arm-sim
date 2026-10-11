"""Build the human-readable report for the version 4 contact study."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/world-v4/contact'


def pct(value):
    return f'{100 * value:.1f}%'


def main():
    report = json.loads((OUT / 'report.json').read_text())
    rank = json.loads((OUT / 'ranking_results.json').read_text())
    replication = json.loads((OUT / 'ranking_replication_results.json').read_text())
    target = report['targeted_red_cylinder_test']['objects']['red cylinder']
    v3 = report['version_3_head_on_targeted_red_cylinder_test']['objects']['red cylinder']
    old = report['targeted_red_cylinder_old_score_test']['objects']['red cylinder']
    rows = rank['rows']
    paired_differences = sum(row['head_top1_contact'] != row['unchanged_contact'] for row in rows)
    text = f'''# Targeted red-cylinder contact study

## Scope

We added simulated red-cylinder contact and slip scenes after version 3 showed weak red-cylinder recall. We trained the new head on 103 targeted scenes, selected the checkpoint with 24 targeted validation scenes, and measured it on 40 fresh targeted test scenes. The version 3 model and arm controller stayed fixed. The contact head did not control the arm.

The targeted set had 8,643 training transitions, 2,059 validation transitions, and 3,236 test transitions. The training split contained 81 tipped scenes and 51 forced-slip scenes. These rates are close to the planned 80% and 45%. The validation and test splits had 20/24 and 29/40 tipped scenes, and 12/24 and 13/40 forced-slip scenes. The test seeds were 186200–186239.

## Red-cylinder test

| Measure | New targeted head | Version 3 head | Old geometric score |
|---|---:|---:|---:|
| Average precision | {target['average_precision']:.3f} | {v3['average_precision']:.3f} | {old['average_precision']:.3f} |
| Brier score | {target['brier']:.3f} | {v3['brier']:.3f} | {old['brier']:.3f} |
| Calibration error, 10 bins | {target['expected_calibration_error_10_bins']:.3f} | {v3['expected_calibration_error_10_bins']:.3f} | {old['expected_calibration_error_10_bins']:.3f} |
| Precision at 0.5 | {target['precision']:.3f} | {v3['precision']:.3f} | {old['precision']:.3f} |
| Recall at 0.5 | {target['recall']:.3f} | {v3['recall']:.3f} | {old['recall']:.3f} |

The new head improved average precision from 0.732 to 0.777 and recall from 7.2% to 29.8% on these fresh red-cylinder scenes. Its Brier score improved from 0.144 to 0.112. Calibration remains weak (error 0.097), and it still misses about seven in ten contact observations at the 0.5 threshold. Precision is high, but is not enough on its own to support a grasp decision.

Across the combined fresh targeted and prior held-out test episodes, the new head had macro average precision {report['test_calibrated']['macro_average_precision']:.3f} and macro Brier score {report['test_calibrated']['macro_brier']:.3f}. The main targeted red-cylinder result above is the more direct test of this data change.

## Fresh candidate ranking

We ran 20 new red-cylinder scenes and 500 offline candidate branches. The new head selected a contacting candidate in {sum(row['head_top1_contact'] for row in rows)}/20 scenes ({pct(rank['head_top1_rate'])}). The unchanged command made contact in {sum(row['unchanged_contact'] for row in rows)}/20 ({pct(rank['unchanged_rate'])}); the old score did so in {sum(row['old_top1_contact'] for row in rows)}/20 ({pct(rank['old_top1_rate'])}). At least one candidate made contact in {sum(row['oracle_contact'] for row in rows)}/20 scenes. Candidate average precision was {rank['head_mean_ap']:.3f} for the new head and {rank['old_mean_ap']:.3f} for the old score.

This ranking result favors the new score over the old geometric score, but it ties the unchanged command's top-choice contact rate. The new head and unchanged command had the same contact outcome in all {len(rows) - paired_differences} scenes. The sample is small. The ranking check measures finger contact after closure, not object lift, grasp retention, or task completion.

## Independent ranking replication

We froze the head and ran a separate check on 40 new scenes (seeds 187020–187059), with 1000 candidate branches. The new head's top candidate contacted in {sum(row['head_top1_contact'] for row in replication['rows'])}/40 scenes ({pct(replication['head_top1_rate'])}). The unchanged command contacted in {sum(row['unchanged_contact'] for row in replication['rows'])}/40 ({pct(replication['unchanged_rate'])}); the old score contacted in {sum(row['old_top1_contact'] for row in replication['rows'])}/40 ({pct(replication['old_top1_rate'])}). Candidate average precision was {replication['head_mean_ap']:.3f} for the new head and {replication['old_mean_ap']:.3f} for the old score. The new head had {replication['paired_wins_vs_unchanged']} wins and {replication['paired_losses_vs_unchanged']} losses against the unchanged command.

The replication keeps the slight candidate-ranking advantage over the old score, but the new head's top candidate contacted less often than the unchanged command. This confirms that the version 4 score does not yet provide a reliable grasp-selection gain.

## Decision

Keep the new head offline. Red-cylinder contact prediction improved on the new test scenes, but the independent ranking replication did not beat the unchanged command. Calibration remains weak. Collect more diverse grasp and slip scenes and improve the selection result before planning a controller comparison. No real robot was used.

The checkpoint was selected at training step {report['selected_step']}; it has {report['parameters']:,} parameters. Read `report.json`, both ranking result and protocol pairs, the episode metadata, and `manifest.json` for details and hashes.
'''
    path = OUT / 'RESULTS.md'
    path.write_text(text)
    print(json.dumps({'report': str(path.relative_to(ROOT)), 'target_red_cylinder_ap': target['average_precision'],
                      'head_top1': rank['head_top1_rate'], 'unchanged_top1': rank['unchanged_rate']}))


if __name__ == '__main__':
    main()

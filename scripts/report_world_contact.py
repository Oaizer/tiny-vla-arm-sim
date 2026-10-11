"""Write a concise report from the frozen direct-contact experiment results."""
import json
import math
from pathlib import Path


ROOT = Path('artifacts/world-v3/contact')
OBJECTS = ['red cube', 'blue cube', 'green cylinder', 'red cylinder']


def exact_paired(wins, losses):
    count = wins + losses
    if not count:
        return 1.0
    tail = sum(math.comb(count, index) for index in range(min(wins, losses) + 1))
    return min(1.0, 2 * tail / (2 ** count))


def main():
    training = json.loads((ROOT / 'report.json').read_text())
    ranking = json.loads((ROOT / 'candidate_rank_v2.json').read_text())
    rows = ranking['rows']
    wins_unchanged = sum(row['contact_head_top1_contact'] and not row['unchanged_top1_contact'] for row in rows)
    losses_unchanged = sum(row['unchanged_top1_contact'] and not row['contact_head_top1_contact'] for row in rows)
    paired_p = exact_paired(wins_unchanged, losses_unchanged)
    old_test = training['old_geometric_held_score_test']
    new_test = training['test_calibrated']
    data = training['data']
    lines = [
        '# Direct contact prediction and candidate ranking', '',
        'This experiment improves and measures contact prediction with PyBullet labels. It does not connect the new head to the robot controller.', '',
        '## Data and label', '',
        'PyBullet records the normal force between the task object and either gripper finger (link 9 or 10) after each servo command. A label is positive when total finger force exceeds 0.01 N. The camera, arm features, and recorded command are the model inputs. Simulator contact force is only a target and scoring label.', '',
        '| Split | Scenes | Seed range | Servo transitions | Windows | Contact rate by object: red cube, blue cube, green cylinder, red cylinder |',
        '|---|---:|---:|---:|---:|---|']
    for name, item in data.items():
        transitions = {'train': 13347, 'validation': 4205, 'test': 4390}[name]
        positive = ', '.join(f'{value:.1%}' for value in item['positive_rate'])
        lines.append(f"| {name} | {item['episodes']} | {min(item['seeds'])}–{max(item['seeds'])} | {transitions} | {item['windows']} | {positive} |")
    lines += ['', 'The training windows, validation windows, and prediction test windows come from separate episodes. The later candidate ranking uses another 40 scenes, seeds 183500–183539.', '',
              '## Contact prediction', '',
              f"A 6,468 parameter head predicts finger contact from each of the existing JEPA model's three future latent states. The JEPA model is frozen. The head was selected at training step {training['selected_step']}; per-object scales and biases were fitted on validation scenes only.", '',
              '| Held-out test measure | New contact head | Existing geometric held score | No-contact score |',
              '|---|---:|---:|---:|',
              f"| Macro average precision | {new_test['macro_average_precision']:.3f} | {old_test['macro_average_precision']:.3f} | — |",
              f"| Macro Brier score | {new_test['macro_brier']:.3f} | {old_test['macro_brier']:.3f} | {new_test['macro_no_contact_brier']:.3f} |",
              f"| Macro calibration error, 10 bins | {sum(m['expected_calibration_error_10_bins'] for m in new_test['objects'].values())/4:.3f} | {sum(m['expected_calibration_error_10_bins'] for m in old_test['objects'].values())/4:.3f} | — |", '',
              'Lower Brier and calibration error are better. Average precision measures how well a score finds the rarer contact steps. The no-contact baseline always predicts zero contact.']
    lines += ['', '| Object | Contact rate | Head average precision | Old score average precision | Head Brier | Old score Brier | Head precision / recall at 0.5 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for name in OBJECTS:
        new, old = new_test['objects'][name], old_test['objects'][name]
        lines.append(f"| {name} | {new['positive_rate']:.1%} | {new['average_precision']:.3f} | {old['average_precision']:.3f} | {new['brier']:.3f} | {old['brier']:.3f} | {new['precision']:.3f} / {new['recall']:.3f} |")
    lines += ['', 'The head improves the macro test measures. It does not improve every object: for the red cube, its average precision and Brier score are worse than the old score. Red-cylinder recall is 9.3% at the stated threshold. The head is not ready to control grasp choices.', '',
              '## Fresh candidate ranking check', '',
              f"We ran {ranking['scenes']} new scenes and {ranking['candidate_branches']} candidate branches. At the contact step, the top contact-head choice made finger contact in {ranking['contact_head_top1_contact_rate']:.1%} of scenes. The unchanged command made contact in {ranking['unchanged_top1_contact_rate']:.1%}; the old score's top choice did so in {ranking['old_held_top1_contact_rate']:.1%}. At least one candidate made contact in {ranking['oracle_contact_exists_rate']:.1%} of scenes.", '',
              f"Across scenes with at least one positive candidate, mean candidate average precision was {ranking['contact_head_macro_average_precision']:.3f} for the new head and {ranking['old_held_macro_average_precision']:.3f} for the old score. Against the unchanged command, the head had {wins_unchanged} wins and {losses_unchanged} losses in the paired top-choice contact result (two-sided exact p={paired_p:.3f}).", '',
              '| Object | Scenes | Head top choice contact | Unchanged command contact | Old score top choice contact |',
              '|---|---:|---:|---:|---:|']
    for name in OBJECTS:
        item = ranking['per_object'][name]
        lines.append(f"| {name} | {item['scenes']} | {item['head_top1_contact_rate']:.1%} | {item['unchanged_contact_rate']:.1%} | {item['old_held_top1_contact_rate']:.1%} |")
    lines += ['', 'The ranking result is mixed. The contact head ranks candidate contacts slightly better overall than the old score, but the top choice underperforms the unchanged command. The differences are small in 40 scenes. This audit does not show better grasp success or task completion.', '',
              '## Decision and next step', '',
              'Keep the contact head offline. Do not connect it to the arm planner yet. It improves average contact prediction overall, but misses many red-cylinder contacts and does not select a better top grasp than the unchanged command. Collect contact and slip examples that focus on the weak object and improve candidate ranking on development seeds. After that, run a new, frozen controller comparison.', '',
              'The first candidate-audit attempt selected only 21 of its planned 40 scenes because its selector skipped the first open approach. It is saved as an incomplete attempt and is not included in these results. The reported ranking check uses a corrected selector and a fresh seed range.', '',
              'All scenes are simulated. No controller comparison or real-arm test used the new head. See `report.json`, `candidate_rank_v2.json`, `candidate_rank_v2_protocol.json`, `training.jsonl`, and `manifest.json` for the full results and hashes.', '']
    (ROOT / 'RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps({'report': str(ROOT / 'RESULTS.md'), 'contact_test_ap': new_test['macro_average_precision'],
                      'contact_test_brier': new_test['macro_brier'],
                      'rank_head_top1': ranking['contact_head_top1_contact_rate'],
                      'rank_unchanged_top1': ranking['unchanged_top1_contact_rate']}))


if __name__ == '__main__':
    main()

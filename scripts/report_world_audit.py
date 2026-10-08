import json
from pathlib import Path
import numpy as np
root=Path('artifacts/world-v2')
rows=json.loads((root/'prediction_audit.json').read_text())
lines=['# Candidate prediction audit','', 'This audit uses seven old version 1 trials where JEPA and feedback had different task outcomes. It checks the first planning decision in each trial. Each of the 25 candidate sequences runs from the same saved physics state. Simulation labels are used only for this offline test. They are not given to the controller.', '', '| Seed | Scene | Old task outcome vs feedback | Position RMSE, mm | Held-score MAE | Actual selected cost gain |', '|---|---|---|---:|---:|---:|']
errors=[]
for r in rows:
    lines.append(f"| {r['seed']} | {r['suite']} | {r['original_paired_outcome']} | {r['position_rmse_mm']:.1f} | {r['held_mae']:.3f} | {r['selected_actual_cost_gain']:+.3f} |")
    actual=np.array([v['position'] for v in r['actual']]); predicted=np.array(r['predicted_positions'])
    mask=((actual>=[.25,-.4,0])&(actual<=[.8,.4,.4])).all(1)
    errors.extend((actual[mask]-predicted[mask]).tolist())
lines += ['', 'Positive actual cost gain means the selected three-command sequence scored better than the unchanged sequence. Negative gain means it scored worse. This cost combines distance, object height, held-object label, and workspace presence. It is a short-horizon score, not a measure of final task success.', '', f"Across the {len(errors)} candidate endpoints within the supported workspace, per-axis position RMSE was {np.sqrt(np.mean(np.array(errors)**2))*1000:.1f} mm. The table also includes objects thrown outside the workspace. Its unbounded errors cannot be compared directly with the workspace-filtered errors in the version 1 report.", '', 'The selected sequence had positive measured gain in both old wins and negative gain in all five old losses. This indicates that candidate ranking is a problem in these selected cases. It does not establish a rate across all trials. The held output is a clipped regression score; it is not a calibrated probability.', '', 'The repeated baseline branch reproduced its endpoint within 0.000001 m in every case. The audit ran 175 distinct candidate branches plus seven repeat checks. Full actions, predicted positions, held scores, measured endpoints, and branch checks are in prediction_audit.json.', '', 'The frozen version 2 test changes only how grasp offsets are executed. It does not repair the model predictions. Its outcome will show whether command consistency helps despite these prediction errors. A later experiment should collect contact and slip examples, calibrate the held score, and test candidate ranking on separate development scenes before another fresh confirmation.', '']
(root/'PREDICTION_AUDIT.md').write_text('\n'.join(lines))
print('\n'.join(lines))

# Explain the shared log field without changing any measured result.
report_path = root / 'RESULTS.md'
if report_path.exists():
    report_text = report_path.read_text().replace('| Changed commands |', '| Accepted planner choices |')
    report_text = report_text.replace(
        'Coherent grasp completed -5 more tasks than feedback and +0 more than original JEPA on these 200 task trials.',
        'The consistent-grasp variant completed five fewer tasks than feedback. It matched the original JEPA total of 145 successes in 200 task trials.')
    report_text = report_text.replace(
        'Future work should check predicted held-object scores against actual simulated outcomes for matched candidate actions, then improve contact prediction if those scores are poorly calibrated.',
        'The offline candidate audit found errors in position and held-object prediction. Improve those predictions and check candidate ranking on separate development scenes before another confirmation run.')
    note = ('The log field `changed_commands` counts accepted planner choices. '
            'The consistent-grasp variant also applies each retained offset to closure; '
            'that second application is not a new planner choice. '
            'Read [the candidate prediction audit](PREDICTION_AUDIT.md) for the separate offline prediction checks.')
    if note not in report_text:
        report_text += '\n## Measurement note\n\n' + note + '\n'
    report_path.write_text(report_text)

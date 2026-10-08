"""Save matched videos of successful and failed planner interventions."""
import json
from pathlib import Path

from .controller import Agent
from .evaluate import annotate_video, load_policy
from .language import parse
from .sim import ArmSim
from .world_data import VisualFeatures, tip_object
from .world_model import load_world
from .world_planner import GraspPlanner, PlanningAgent, SensorArm


def replay(config, root, policy_path):
    root = Path(root)
    output = root / 'replays'
    output.mkdir(exist_ok=True)
    policy, _ = load_policy(policy_path)
    extractor = VisualFeatures(policy)
    settings = json.loads(Path('configs/project.json').read_text())['controller']
    saved = []
    for mode in ['fixed', 'jepa']:
        choices = []
        for suite in ['slip', 'tipped', 'object_moved', 'nominal', 'dynamics']:
            path = root / 'evaluation' / suite / f'{mode}.jsonl'
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            choices.extend(r for r in rows if any(d['accepted'] for d in r['planning']))
        for success in [True, False]:
            original = next((row for row in choices if row['success'] == success), None)
            if original is None:
                continue
            suite, seed = original['suite'], original['seed']
            task = parse(original['instruction'])
            for controller in ['feedback', mode]:
                model, _ = load_world(root / 'model' / f'{mode}.pt')
                sim = ArmSim()
                try:
                    sim.reset(seed, 'dynamics' if suite == 'dynamics' else 'nominal')
                    if suite == 'tipped':
                        tip_object(sim, task)
                    planner = None if controller == 'feedback' else GraspPlanner(model, config['planner'])
                    view = SensorArm(sim, extractor, planner)
                    agent = Agent(policy, task.instruction, settings) if planner is None else PlanningAgent(policy, task.instruction, settings)
                    events = []
                    def inject(stage):
                        if not events and ((suite == 'slip' and stage == 'after_lift') or
                                           (suite == 'object_moved' and stage == 'after_approach')):
                            sim.perturb(suite, task)
                            view.refresh()
                            events.append(stage)
                    frames = [sim.image(384, True)] * 10
                    result = agent.run(view, 'feedback', inject, frames)
                    outcome = sim.score(task)
                    frames.extend([sim.image(384, True)] * 20)
                    label = 'success' if success else 'failure'
                    video = output / f'{mode}_{label}_{seed}_{controller}.mp4'
                    annotate_video(video, frames, task.instruction, f'{suite} | {controller} | {outcome["success"]}')
                    source = root / 'evaluation' / suite / f'{controller}.jsonl'
                    expected = next(json.loads(line) for line in source.read_text().splitlines() if json.loads(line)['seed'] == seed)
                    row = {'suite': suite, 'seed': seed, 'controller': controller, 'video': str(video),
                           **result, **outcome, 'matches_original_success': outcome['success'] == expected['success'],
                           'planning': planner.decisions if planner else []}
                    video.with_suffix('.json').write_text(json.dumps(row, indent=2))
                    saved.append(row)
                    print(json.dumps({key: row[key] for key in ['seed', 'controller', 'success', 'matches_original_success']}), flush=True)
                finally:
                    sim.close()
    (output / 'summary.json').write_text(json.dumps(saved, indent=2))
    return saved

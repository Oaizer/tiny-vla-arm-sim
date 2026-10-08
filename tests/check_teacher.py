"""Run a few teacher examples and save their simulation results."""
import json
from tinyvla.sim import ArmSim
from tinyvla.evaluate import instruction_for
from tinyvla.language import parse

sim=ArmSim();rows=[]
try:
    for i in range(16):
        task=parse(instruction_for(i));sim.reset(12000+i);sim.execute_chunk(sim.teacher(task));sim.step(100)
        rows.append({'seed':12000+i,'instruction':task.instruction,**sim.score(task)})
finally:sim.close()
print(json.dumps({'teacher_successes':sum(r['success'] for r in rows),'episodes':len(rows),'trials':rows},indent=2))

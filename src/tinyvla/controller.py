"""Replan between steps and check grasps with the arm's finger sensors.

This supervisor is scripted. Perception, instruction selection and Cartesian
waypoints are learned. It never calls simulator object labels or scoring APIs.
"""
import time
import numpy as np
import torch
from .language import tokenize

class Agent:
    def __init__(self,model,instruction,settings):
        self.model=model;self.tokens=torch.from_numpy(tokenize(instruction))[None]
        self.settings=settings;self.calls=0;self.latencies=[];self.trace=[]

    def observe(self,sim,phase):
        if self.calls>=self.settings['max_inference_calls']:raise RuntimeError('The model reached its call limit.')
        obs=sim.observation();start=time.perf_counter()
        with torch.inference_mode():
            action,info=self.model(torch.from_numpy(obs['rgb']).permute(2,0,1)[None].float()/255,
                self.tokens,torch.from_numpy(obs['state'])[None],torch.from_numpy(obs['calibration'])[None])
        self.calls+=1;self.latencies.append((time.perf_counter()-start)*1000)
        action=action[0].numpy();result={k:float(info[k][0]) for k in ['source_confidence','pad_confidence']}
        result['task']=int(info['task_logits'][0].argmax());result['state']=obs['state']
        self.trace.append({'phase':phase,'call':self.calls,'source_confidence':result['source_confidence'],
            'pad_confidence':result['pad_confidence'],'gripper_m':float(obs['state'][3]),'task':result['task']})
        if not np.isfinite(action).all():raise RuntimeError('The model returned invalid action values.')
        return action,result

    def valid(self,action,info,source=True):
        threshold=self.settings['presence_threshold']
        if source and info['source_confidence']<threshold:return False,'source_not_visible'
        if info['task']==0 and info['pad_confidence']<threshold:return False,'destination_not_visible'
        points=action[:4,:2] if source else action[4:,:2]
        if np.any(points<np.array(self.settings['workspace_min'][:2])-.02) or np.any(points>np.array(self.settings['workspace_max'][:2])+.02):return False,'outside_workspace'
        return True,'ready'

    def run(self,sim,mode='feedback',event=None,frames=None):
        retries=0;recovered=False;threshold=self.settings['empty_gripper_threshold_m']
        carry=getattr(sim,'move_smooth',sim.move)
        try:
            action,info=self.observe(sim,'initial');valid,reason=self.valid(action,info)
            if not valid:return self.report('stopped_'+reason,retries,recovered)
            if mode=='one_shot':
                for i,w in enumerate(action):
                    motion=carry if i in [3,4,5] else sim.move
                    motion(w[:3],w[3],160 if i in [2,6] else 150,frames)
                    if event and i==0:event('after_approach')
                    if event and i==3:event('after_lift')
                sim.step(100,frames);return self.report('executed',0,False)
            for attempt in range(self.settings['max_attempts']):
                if attempt:
                    retries+=1;sim.move(sim.park,.04,180,frames)
                    action,info=self.observe(sim,'recovery');valid,reason=self.valid(action,info)
                    if not valid:return self.report('stopped_'+reason,retries,recovered)
                # A bounded approach segment. Reobserve from a clear view before grasping.
                sim.move(action[0,:3],action[0,3],120,frames)
                if event:event('after_approach')
                sim.move(sim.park,.04,150,frames)
                revised,new_info=self.observe(sim,'pregrasp_replan');valid,reason=self.valid(revised,new_info)
                if not valid:return self.report('stopped_'+reason,retries,recovered)
                correction=float(np.linalg.norm(revised[0,:2]-action[0,:2]))
                if correction>.025:recovered=True
                self.trace.append({'phase':'visual_correction','distance_m':correction})
                action,info=revised,new_info
                sim.move(action[0,:3],action[0,3],130,frames)
                sim.move(action[1,:3],action[1,3],150,frames)
                sim.move(action[2,:3],action[2,3],190,frames)
                carry(action[3,:3],action[3,3],160,frames)
                if event:event('after_lift')
                # Uses finger position encoders, not object poses or contact labels.
                if sim.state()[3]<threshold:
                    self.trace.append({'phase':'empty_gripper_retry','attempt':attempt+1})
                    recovered=True;continue
                if info['task']==1:return self.report('lifted',retries,recovered)
                # Destination remains visible from the lift pose; avoid an extra carry detour.
                updated,pad_info=self.observe(sim,'destination_replan');valid,reason=self.valid(updated,pad_info,source=False)
                if not valid:
                    # Keep grip, hold above the bounded workspace, and stop.
                    return self.report('stopped_'+reason,retries,recovered)
                lost=False
                for i in [4,5]:
                    carry(updated[i,:3],updated[i,3],150,frames)
                    if sim.state()[3]<threshold:
                        lost=True;recovered=True;self.trace.append({'phase':'transport_slip_retry','attempt':attempt+1});break
                if lost:continue
                for i in [6,7]:sim.move(updated[i,:3],updated[i,3],170 if i==6 else 150,frames)
                sim.step(100,frames);return self.report('placed',retries,recovered)
            sim.move(sim.park,.04,150,frames);return self.report('stopped_attempt_budget',retries,recovered)
        except (RuntimeError,ValueError) as error:
            return self.report('stopped_invalid_prediction',retries,recovered,str(error))

    def report(self,status,retries,recovered,error=None):
        return {'status':status,'retries':retries,'recovery_triggered':recovered,'inference_calls':self.calls,
            'inference_ms':self.latencies,'trace':self.trace,'error':error}

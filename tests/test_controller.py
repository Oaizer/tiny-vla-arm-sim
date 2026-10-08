"""Check failed-grip retries and stopping when an object is missing."""
import unittest
import numpy as np
import torch
from tinyvla.controller import Agent

SETTINGS={'max_inference_calls':24,'presence_threshold':.65,'empty_gripper_threshold_m':.008,
          'max_attempts':3,'workspace_min':[.28,-.33,.015],'workspace_max':[.78,.37,.38]}

class SensorOnlySim:
    park=np.array([.27,-.35,.34])
    def __init__(self,empty=False):self.grip=.04;self.moves=0;self.empty=empty
    def observation(self):return {'rgb':np.zeros((128,128,3),np.uint8),'state':self.state(),'calibration':np.repeat(np.eye(3)[None],7,0).astype(np.float32)}
    def state(self):return np.array([.5,0,.24,self.grip],np.float32)
    def move(self,xyz,grip,steps=120,frames=None):self.moves+=1;self.grip=(0. if self.empty else .018) if grip<.008 else grip
    def step(self,n,frames=None):pass

class FixedPolicy:
    def __init__(self,presence=.99):self.presence=presence
    def __call__(self,rgb,tokens,state,calibration):
        a=torch.tensor([[[.5,-.1,.20,.04],[.5,-.1,.026,.04],[.5,-.1,.026,0.],
             [.5,-.1,.24,0.],[.53,.27,.24,0.],[.53,.27,.065,0.],[.53,.27,.065,.04],[.53,.27,.25,.04]]])
        return a,{'source_confidence':torch.tensor([self.presence]),'pad_confidence':torch.tensor([.99]),'task_logits':torch.tensor([[10.,-10.]])}

class ControllerContracts(unittest.TestCase):
    def test_missing_source_stops_before_motion(self):
        sim=SensorOnlySim();agent=Agent(FixedPolicy(.1),'put the red cube on the cyan pad',SETTINGS)
        result=agent.run(sim);self.assertEqual(sim.moves,0);self.assertEqual(result['status'],'stopped_source_not_visible')

    def test_empty_gripper_retries_then_stops(self):
        sim=SensorOnlySim(empty=True);agent=Agent(FixedPolicy(),'put the red cube on the cyan pad',SETTINGS)
        result=agent.run(sim);self.assertEqual(result['status'],'stopped_attempt_budget')
        self.assertEqual(result['retries'],2);self.assertLessEqual(result['inference_calls'],24)

    def test_slip_retries_from_camera_and_encoders(self):
        sim=SensorOnlySim();events=[]
        def slip(stage):
            if stage=='after_lift' and not events:sim.grip=0.;events.append(stage)
        agent=Agent(FixedPolicy(),'put the red cube on the cyan pad',SETTINGS)
        result=agent.run(sim,event=slip);self.assertEqual(result['status'],'placed');self.assertEqual(result['retries'],1)
        self.assertTrue(result['recovery_triggered'])

if __name__=='__main__':unittest.main()

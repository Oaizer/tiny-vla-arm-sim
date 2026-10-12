"""B601-RS visual preview on top of the original Panda physics simulator."""
from pathlib import Path
import numpy as np
import pybullet as p

from .sim import ArmSim


class B601VisualSim(ArmSim):
    """Render the B601-RS model while retaining Panda collision physics."""

    def reset(self, seed, suite='nominal', omissions=False):
        observation = super().reset(seed, suite, omissions)
        for link in range(-1, p.getNumJoints(self.arm)):
            p.changeVisualShape(self.arm, link, rgbaColor=[1, 1, 1, 0])

        model = (Path(__file__).with_name('assets') / 'rebot_b601_rs' / 'RS' /
                 'urdf' / 'ReBot_Arm_RS.urdf')
        self.visual_arm = p.loadURDF(str(model), basePosition=[.18, 0, 0],
                                     useFixedBase=True,
                                     flags=p.URDF_MAINTAIN_LINK_ORDER)
        self.visual_joints = []
        self.visual_gripper = []
        self.visual_tool = -1
        for joint in range(p.getNumJoints(self.visual_arm)):
            info = p.getJointInfo(self.visual_arm, joint)
            name = info[1].decode()
            p.setCollisionFilterGroupMask(self.visual_arm, joint, 0, 0)
            if name.startswith('joint') and name[5:].isdigit():
                self.visual_joints.append(joint)
            elif name.startswith('gripper_joint'):
                self.visual_gripper.append(joint)
            if info[12].decode() == 'gripper_end':
                self.visual_tool = joint
        p.setCollisionFilterGroupMask(self.visual_arm, -1, 0, 0)
        if (len(self.visual_joints) != 6 or len(self.visual_gripper) != 2 or
                self.visual_tool < 0):
            raise RuntimeError('The B601-RS visual model has an unexpected joint layout.')
        self._visual_dirty = True
        return self.observation()

    def _sync_visual_arm(self):
        target = p.getLinkState(self.arm, 11, computeForwardKinematics=True)
        joints = (*self.visual_joints, *self.visual_gripper)
        current = [p.getJointState(self.visual_arm, joint)[0] for joint in joints]
        lower = [-2.8, 0, 0, -1.57, -1.57, -3.14, 0, 0]
        upper = [2.8, 3.14, 3.14, 1.57, 1.57, 3.14, .05, .0715]
        ranges = [hi - lo for lo, hi in zip(lower, upper)]
        solution = p.calculateInverseKinematics(
            self.visual_arm, self.visual_tool, target[4], target[5],
            lowerLimits=lower, upperLimits=upper, jointRanges=ranges,
            restPoses=current, jointDamping=[.08] * 8,
            maxNumIterations=80, residualThreshold=1e-4)
        for index, joint in enumerate(self.visual_joints):
            p.resetJointState(self.visual_arm, joint, solution[index], 0)
        grip = float(np.mean([p.getJointState(self.arm, joint)[0] for joint in [9, 10]]))
        for joint, opening in zip(self.visual_gripper, [grip, grip * 1.43]):
            info = p.getJointInfo(self.visual_arm, joint)
            opening = float(np.clip(opening, info[8], info[9]))
            p.resetJointState(self.visual_arm, joint, opening, 0)
        self._visual_dirty = False

    def image(self, size=None, overview=False):
        if hasattr(self, 'visual_arm') and self._visual_dirty:
            self._sync_visual_arm()
        return super().image(size, overview)

    def step(self, n, frames=None):
        if not hasattr(self, 'ticks'):
            self.ticks = 0
        for i in range(n):
            p.stepSimulation()
            self.ticks += 1
            if hasattr(self, 'visual_arm'):
                self._visual_dirty = True
            if frames is not None and i % 12 == 0:
                frames.append(self.image(384, True))
            if hasattr(self, 'objects') and hasattr(self, 'max_heights'):
                for name, body in self.objects.items():
                    self.max_heights[name] = max(
                        self.max_heights.get(name, 0),
                        p.getBasePositionAndOrientation(body)[0][2])

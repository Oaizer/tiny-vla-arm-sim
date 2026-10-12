"""B601-RS simulation using the licensed Seeed URDF and PyBullet physics."""
from pathlib import Path
import numpy as np
import pybullet as pb
import pybullet_data

ROOT = Path(__file__).with_name("assets") / "rebot_b601_rs" / "RS"
JOINTS = tuple(range(6))
LIMITS = ((-2.8, 2.8), (0.0, 3.14), (0.0, 3.14), (-1.57, 1.57),
          (-1.57, 1.57), (-3.14, 3.14), (0.0, 0.05), (0.0, 0.0715))
TORQUES = (36, 36, 36, 14, 14, 14)
MAX_TARGET_SPEED_RAD_S = 1.5
QUINTIC_PEAK_SPEED_FACTOR = 1.875

class B601Sim:
    """Six-axis B601-RS arm with its two prismatic gripper joints."""
    def __init__(self, size=384, frame_stride=12):
        self.cid = pb.connect(pb.DIRECT)
        self.size = size
        self.frame_stride = int(frame_stride)
        self.base = (0, 0, 0)
        self.tcp_link = 6
        self.top_down = pb.getQuaternionFromEuler([0, np.pi / 2, 0])

    def close(self):
        if pb.isConnected(self.cid):
            pb.disconnect(self.cid)

    def reset(self, seed=0):
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        pb.resetSimulation()
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        pb.setGravity(0, 0, -9.81)
        pb.setTimeStep(1 / 240)
        pb.setPhysicsEngineParameter(numSolverIterations=120, deterministicOverlappingPairs=1)
        pb.loadURDF("plane.urdf")
        self.table = pb.createMultiBody(0,
            pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[.25, .45, .015]),
            pb.createVisualShape(pb.GEOM_BOX, halfExtents=[.25, .45, .015], rgbaColor=[.28,.31,.29,1]),
            [.38, 0, .085])
        # Keep mesh collision checks tractable in the small CPU-only pilot.
        self.arm = pb.loadURDF(str((ROOT / "urdf" / "ReBot_Arm_RS.urdf").resolve()),
                               self.base, useFixedBase=True, flags=0)
        self.arm_joints = tuple(range(6))
        self.gripper_joints = (7, 8)
        for joint in (6, 7, 8):
            pb.changeDynamics(self.arm, joint, lateralFriction=2.5,
                              spinningFriction=.01, rollingFriction=.001)
        self.max_heights = {}
        self.objects = {}
        self.pads = {}
        self.ticks = 0
        self.command_count = 0
        self.peak_reference_speed_rad_s = 0.0
        # Start from a repeatable, folded pose inside the B601 joint limits.
        self.q = np.array([0, .7, .8, -1.5, 0, 0], dtype=float)
        self.home_q = self.q.copy()
        for joint, value in zip(self.arm_joints, self.q):
            pb.resetJointState(self.arm, joint, value)
        for joint, value in zip(self.gripper_joints, (.05, .0715)):
            pb.resetJointState(self.arm, joint, value)
        self._make_scene()
        self.eye = [.91, -1.05, .92]
        self.aim = [.34, 0, .11]
        self.view = pb.computeViewMatrix(self.eye, self.aim, [0, 0, 1])
        return self

    def _make_scene(self):
        self.objects = {}
        self.pads = {}
        for color, x, y in (("red", .24, -.16), ("blue", .34, .00), ("green", .33, .20)):
            half = .021
            col = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[half, half, .02])
            vis = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[half, half, .02], rgbaColor={"red":[.86,.08,.07,1],"blue":[.05,.22,.86,1],"green":[.08,.65,.16,1]}[color])
            body = pb.createMultiBody(.045, col, vis, [x,y,.125])
            pb.changeDynamics(body,-1,lateralFriction=1.4,spinningFriction=.002,restitution=0)
            self.objects[color + " cube"] = body
            self.max_heights[color + " cube"] = .125
        for color, x in (("yellow", .34), ("purple", .34), ("cyan", .34)):
            vis = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[.045,.045,.001], rgbaColor={"yellow":[.91,.72,.04,1],"purple":[.55,.12,.69,1],"cyan":[.02,.68,.74,1]}[color])
            self.pads[color] = pb.createMultiBody(0,-1,vis,[x,(-.16,.0,.16)[("yellow","purple","cyan").index(color)],.102])
        for _ in range(12): pb.stepSimulation()

    def image(self, size=None):
        resolution = self.size if size is None else size
        if isinstance(resolution, int):
            width = height = resolution
        else:
            width, height = (int(value) for value in resolution)
        projection = pb.computeProjectionMatrixFOV(42, width / height, .03, 3)
        image = pb.getCameraImage(width, height, self.view, projection,
                                  renderer=pb.ER_TINY_RENDERER, shadow=1)
        return np.asarray(image[2], np.uint8).reshape(height, width, 4)[:, :, :3].copy()

    def object_position(self, name):
        return np.asarray(pb.getBasePositionAndOrientation(self.objects[name])[0],float)

    def tool_position(self):
        return np.asarray(pb.getLinkState(self.arm,self.tcp_link,computeForwardKinematics=True)[4],float)

    def _ik(self, xyz):
        lower=[x[0] for x in LIMITS]; upper=[x[1] for x in LIMITS]
        ranges=[b-a for a,b in LIMITS]
        rest=[*self.q,.05,.0715]
        solution=pb.calculateInverseKinematics(self.arm,self.tcp_link,xyz,
            targetOrientation=self.top_down,lowerLimits=lower,upperLimits=upper,
            jointRanges=ranges,restPoses=rest,maxNumIterations=250,residualThreshold=1e-5)
        target=np.asarray(solution[:6],float)
        for j,(lo,hi) in enumerate(LIMITS[:6]): target[j]=np.clip(target[j],lo,hi)
        return target

    def move(self, xyz, finger_travel_mm=121.5, frames=None):
        """Move using URDF torque limits and a smooth quintic joint reference."""
        xyz=np.asarray(xyz,float)
        if xyz.shape!=(3,) or not np.isfinite(xyz).all(): raise ValueError("Invalid B601 pose")
        target=self._ik(xyz)
        start=self.q.copy()
        max_delta=float(np.max(np.abs(target-start)))
        duration=max(.35,QUINTIC_PEAK_SPEED_FACTOR*max_delta/MAX_TARGET_SPEED_RAD_S)
        ticks=max(1,int(np.ceil(duration*240)))
        opening=float(np.clip(finger_travel_mm/1000,0,.1215))
        finger=[opening*(.05/.1215),opening*(.0715/.1215)]
        self.command_count+=1
        for t in range(1,ticks+1):
            u=t/ticks
            blend=10*u**3-15*u**4+6*u**5
            q=start+(target-start)*blend
            blend_rate=30*u**2-60*u**3+30*u**4
            reference_speed=np.abs(target-start)*blend_rate/duration
            self.peak_reference_speed_rad_s=max(self.peak_reference_speed_rad_s,float(reference_speed.max()))
            pb.setJointMotorControlArray(self.arm,self.arm_joints,pb.POSITION_CONTROL,
                targetPositions=q.tolist(),forces=TORQUES,positionGains=[.18]*6,velocityGains=[1.0]*6)
            pb.setJointMotorControlArray(self.arm,self.gripper_joints,pb.POSITION_CONTROL,
                targetPositions=finger,forces=[500,500],positionGains=[.5,.5],velocityGains=[1,1])
            pb.stepSimulation(); self.ticks+=1
            self.q=np.asarray([pb.getJointState(self.arm,j)[0] for j in self.arm_joints])
            for name,body in self.objects.items():
                self.max_heights[name]=max(self.max_heights[name],pb.getBasePositionAndOrientation(body)[0][2])
            if frames is not None and t % self.frame_stride == 0: frames.append(self.image(self.size))
        return target

    def move_joints(self, target_q, frames=None):
        """Move the arm to a known joint pose with the same capped profile."""
        target = np.asarray(target_q, dtype=float)
        if target.shape != (6,) or not np.isfinite(target).all():
            raise ValueError("Invalid B601 joint target")
        for value, (lo, hi) in zip(target, LIMITS[:6]):
            if not lo <= value <= hi:
                raise ValueError("Joint target exceeds the B601-RS limit")
        start = self.q.copy()
        max_delta = float(np.max(np.abs(target - start)))
        duration = max(.35, QUINTIC_PEAK_SPEED_FACTOR * max_delta / MAX_TARGET_SPEED_RAD_S)
        ticks = max(1, int(np.ceil(duration * 240)))
        finger = [pb.getJointState(self.arm, joint)[0] for joint in self.gripper_joints]
        self.command_count += 1
        for t in range(1, ticks + 1):
            u = t / ticks
            blend = 10 * u**3 - 15 * u**4 + 6 * u**5
            blend_rate = 30 * u**2 - 60 * u**3 + 30 * u**4
            q = start + (target - start) * blend
            reference_speed = np.abs(target - start) * blend_rate / duration
            self.peak_reference_speed_rad_s = max(self.peak_reference_speed_rad_s, float(reference_speed.max()))
            pb.setJointMotorControlArray(self.arm, self.arm_joints, pb.POSITION_CONTROL,
                targetPositions=q.tolist(), forces=TORQUES, positionGains=[.18] * 6, velocityGains=[1.0] * 6)
            pb.setJointMotorControlArray(self.arm, self.gripper_joints, pb.POSITION_CONTROL,
                targetPositions=finger, forces=[500, 500], positionGains=[.5, .5], velocityGains=[1, 1])
            pb.stepSimulation()
            self.ticks += 1
            self.q = np.asarray([pb.getJointState(self.arm, joint)[0] for joint in self.arm_joints])
            if frames is not None and t % self.frame_stride == 0:
                frames.append(self.image(self.size))
        return target

    def teacher_trial(self, object_name="red cube", pad="yellow", frames=None,
                      approach_height=.22, carry_height=.20, return_to_home=False, check_grasp_before_transfer=False):
        source = self.object_position(object_name)
        destination = np.asarray(pb.getBasePositionAndOrientation(self.pads[pad])[0])
        self.move([source[0], source[1], approach_height], 121.5, frames)
        self.move([source[0], source[1], .08], 121.5, frames)
        self.move([source[0], source[1], .08], 45, frames)
        self.move([source[0], source[1], carry_height], 45, frames)

        # Read the simulated object state before moving over the table. A failed
        # grasp must not turn the transfer motion into a sweep through loose cubes.
        lifted = self.object_position(object_name)
        grasped = bool(lifted[2] > .15 and np.linalg.norm(lifted[:2] - source[:2]) < .06)
        if not check_grasp_before_transfer:
            grasped = True
        if check_grasp_before_transfer and not grasped:
            tool = self.tool_position()
            self.move(tool, 121.5, frames)
            self.move([tool[0], tool[1], approach_height], 121.5, frames)
            for _ in range(120):
                pb.stepSimulation()
                self.ticks += 1
            if return_to_home:
                self.move_joints(self.home_q, frames)
            final = self.object_position(object_name)
            distance = float(np.linalg.norm(final[:2] - destination[:2]))
            return {"success": False, "grasped": False, "distance_m": distance,
                    "final_height_m": float(final[2]), "commands": self.command_count,
                    "joint_limits_respected": bool(all(lo - 1e-5 <= v <= hi + 1e-5
                        for v, (lo, hi) in zip(self.q, LIMITS[:6])))}

        self.move([destination[0], destination[1], carry_height], 45, frames)
        self.move([destination[0], destination[1], .08], 45, frames)
        self.move([destination[0], destination[1], .08], 121.5, frames)
        self.move([destination[0], destination[1], approach_height], 121.5, frames)
        for _ in range(240):
            pb.stepSimulation()
            self.ticks += 1
        if return_to_home:
            self.move_joints(self.home_q, frames)
        final = self.object_position(object_name)
        distance = float(np.linalg.norm(final[:2] - destination[:2]))
        return {"success": bool(distance < .055 and .10 < final[2] < .18),
                "grasped": True, "distance_m": distance,
                "final_height_m": float(final[2]), "commands": self.command_count,
                "joint_limits_respected": bool(all(lo - 1e-5 <= v <= hi + 1e-5
                    for v, (lo, hi) in zip(self.q, LIMITS[:6])))}



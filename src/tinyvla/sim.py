"""Seeded Panda physics. Privileged state is restricted to teachers and scoring."""
from pathlib import Path
import numpy as np
import pybullet as p
import pybullet_data
from .language import OBJECTS,PADS

RGB={'red':(.88,.07,.08),'blue':(.07,.18,.90),'green':(.06,.70,.17),
     'yellow':(.92,.77,.06),'purple':(.64,.12,.76),'cyan':(.04,.78,.84)}

class ArmSim:
    def __init__(self,size=128):
        self.cid=p.connect(p.DIRECT);self.size=size
        self.rest=[0,-.45,0,-2.3,0,1.85,.78]
        self.quat=p.getQuaternionFromEuler([np.pi,0,0]);self.park=np.array([.27,-.35,.34])

    def close(self):
        if p.isConnected(self.cid):p.disconnect(self.cid)

    def reset(self,seed,suite='nominal',omissions=False):
        self.seed=seed;self.suite=suite;rng=np.random.default_rng(seed);self.rng=rng
        p.resetSimulation();self.objects={};self.pads={};self.max_heights={};self.ticks=0;self.command_count=0
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-9.81);p.setTimeStep(1/240)
        p.setPhysicsEngineParameter(numSolverIterations=100,deterministicOverlappingPairs=1)
        floor=p.loadURDF('plane.urdf');grey=float(rng.uniform(.68,.92))
        p.changeVisualShape(floor,-1,textureUniqueId=-1,rgbaColor=[grey]*3+[1])
        self.arm=p.loadURDF('franka_panda/panda.urdf',useFixedBase=True)
        for i,q in enumerate(self.rest):p.resetJointState(self.arm,i,q,0)
        for j in [9,10]:p.changeDynamics(self.arm,j,lateralFriction=1.5,spinningFriction=.002)
        self.move(self.park,.04,steps=180)
        # Known calibrated camera parameters are policy inputs, not object labels.
        cam_range=.12 if suite=='camera' else .045
        eye=np.array([.53,.01,1.18])+rng.uniform([-cam_range,-cam_range,-.07],[cam_range,cam_range,.07])
        aim=np.array([.53,.01,0])+rng.uniform([-.025,-.025,0],[.025,.025,0])
        fov=float(rng.uniform(35,48) if suite=='camera' else rng.uniform(40,45))
        self.view=p.computeViewMatrix(eye,aim,[0,1,0]);self.proj=p.computeProjectionMatrixFOV(fov,1,.05,3)
        self.light=np.array([*rng.uniform(-2,2,2),4.])
        self.brightness=float(rng.uniform(.4,.65) if suite=='lighting' else rng.uniform(.75,1.05))
        self.noise=.025 if suite=='appearance' else .006
        self.camera_info={'eye':eye.tolist(),'aim':aim.tolist(),'fov':fov}
        self.objects={};self.pads={};self.properties={};positions=[]
        # One reusable procedural grayscale texture, randomized independently of goals.
        texture_path=Path(__file__).with_name('floor.png')
        if texture_path.exists():
            tex=p.loadTexture(str(texture_path));pixels=rng.integers(155,230,(16,16,3),dtype=np.uint8)
            pixels[:]=pixels.mean(-1,keepdims=True)
            p.changeTexture(tex,pixels.reshape(-1).tolist(),16,16);p.changeVisualShape(floor,-1,textureUniqueId=tex)
        for name in OBJECTS:
            if omissions and rng.random()<.10:continue
            color,shape=name.split();size=float(rng.uniform(.017,.023))
            if suite=='appearance':size=float(rng.uniform(.015,.025))
            while True:
                xy=rng.uniform([.34,-.27] if suite=='positions' else [.36,-.24],
                               [.73,.12] if suite=='positions' else [.70,.07])
                if all(np.linalg.norm(xy-x)>.095 for x in positions):break
            positions.append(xy)
            mass=float(rng.uniform(.08,.13) if suite=='dynamics' else rng.uniform(.035,.065))
            friction=float(rng.uniform(.55,.8) if suite=='dynamics' else rng.uniform(.9,1.3))
            tint=np.clip(np.array(RGB[color])*rng.uniform(.75,1.10),0,1)
            if shape=='cube':
                collision=p.createCollisionShape(p.GEOM_BOX,halfExtents=[size,size,.02])
                visual=p.createVisualShape(p.GEOM_BOX,halfExtents=[size,size,.02],rgbaColor=[*tint,1])
            else:
                collision=p.createCollisionShape(p.GEOM_CYLINDER,radius=size,height=.04)
                visual=p.createVisualShape(p.GEOM_CYLINDER,radius=size,length=.04,rgbaColor=[*tint,1])
            yaw=float(rng.uniform(-.3,.3) if suite=='appearance' else rng.uniform(-.06,.06))
            b=p.createMultiBody(mass,collision,visual,[*xy,.022],p.getQuaternionFromEuler([0,0,yaw]))
            p.changeDynamics(b,-1,lateralFriction=friction,spinningFriction=.002,restitution=0)
            self.objects[name]=b;self.properties[name]={'half_width':size,'mass':mass,'friction':friction,'yaw':yaw}
        for i,name in enumerate(PADS):
            if omissions and rng.random()<.035:continue
            xy=np.array([.35+i*.18,.27])+rng.uniform([-.015,-.018],[.015,.018])
            if suite=='positions':xy += rng.uniform([-.015,-.025],[.015,.025])
            visual=p.createVisualShape(p.GEOM_BOX,halfExtents=[.052,.052,.001],rgbaColor=[*RGB[name],1])
            self.pads[name]=p.createMultiBody(0,-1,visual,[*xy,.001])
        self.max_heights={name:0. for name in OBJECTS}
        self.step(30)
        return self.observation()

    def state(self):
        xyz=p.getLinkState(self.arm,11,computeForwardKinematics=True)[4]
        grip=np.mean([p.getJointState(self.arm,j)[0] for j in [9,10]])
        return np.array([*xyz,grip],np.float32)

    def image(self,size=None,overview=False):
        n=size or self.size
        view=p.computeViewMatrix([1.25,-1.10,1.05],[.44,.01,.12],[0,0,1]) if overview else self.view
        result=p.getCameraImage(n,n,viewMatrix=view,projectionMatrix=self.proj,renderer=p.ER_TINY_RENDERER,
            shadow=0,lightDirection=self.light.tolist(),lightAmbientCoeff=.65,lightDiffuseCoeff=.35,lightSpecularCoeff=.1)
        rgb=np.asarray(result[2],np.uint8).reshape(n,n,4)[:,:,:3].copy()
        if not overview:
            # Pixel noise is deterministically keyed to the physics tick.
            rng=np.random.default_rng(self.seed*10000+self.ticks)
            rgb=np.clip(rgb.astype(np.float32)*self.brightness+rng.normal(0,self.noise*255,rgb.shape),0,255).astype(np.uint8)
        return rgb

    def homography(self,z):
        matrix=np.array(self.proj).reshape(4,4,order='F')@np.array(self.view).reshape(4,4,order='F')
        rows=np.stack([.5*(matrix[0]+matrix[3]),.5*(matrix[3]-matrix[1]),matrix[3]])
        return np.stack([rows[:,0],rows[:,1],rows[:,2]*z+rows[:,3]],axis=1)

    def calibration(self):
        return np.stack([np.linalg.inv(self.homography(.04))]*len(OBJECTS)+
                        [np.linalg.inv(self.homography(.001))]*len(PADS)).astype(np.float32)

    def observation(self):
        return {'rgb':self.image(),'state':self.state(),'calibration':self.calibration()}

    def labels(self):
        """Training-only supervision for all visible objects and pads."""
        points=[];pixels=[];present=[]
        for name in (*OBJECTS,*PADS):
            bodies=self.objects if name in OBJECTS else self.pads
            valid=name in bodies;xyz=np.array(p.getBasePositionAndOrientation(bodies[name])[0]) if valid else np.zeros(3)
            z=.04 if name in OBJECTS else .001
            clip=self.homography(z)@np.array([*xyz[:2],1.]);uv=clip[:2]/clip[2]
            points.append(xyz);pixels.append(uv);present.append(valid)
        return np.array(points,np.float32),np.array(pixels,np.float32),np.array(present,np.float32)

    def teacher(self,task):
        src=np.array(p.getBasePositionAndOrientation(self.objects[task.object])[0])
        dst=src if task.kind=='lift' else np.array(p.getBasePositionAndOrientation(self.pads[task.pad])[0])
        action=np.array([[*src[:2],.20,.04],[*src[:2],.026,.04],[*src[:2],.026,0.],
                         [*src[:2],.24,0.],[*dst[:2],.24,0.],[*dst[:2],.065,0.],
                         [*dst[:2],.065,.04],[*dst[:2],.25,.04]],np.float32)
        if task.kind=='lift':action[4:]=[*src[:2],.24,0.]
        return action

    def step(self,n,frames=None):
        if not hasattr(self,'ticks'):self.ticks=0
        for i in range(n):
            p.stepSimulation();self.ticks+=1
            if frames is not None and i%12==0:frames.append(self.image(384,True))
            if hasattr(self,'objects') and hasattr(self,'max_heights'):
                for name,b in self.objects.items():
                    self.max_heights[name]=max(self.max_heights.get(name,0),p.getBasePositionAndOrientation(b)[0][2])

    def move(self,xyz,grip,steps=120,frames=None):
        self.command_count=getattr(self,'command_count',0)+1
        xyz=np.clip(np.array(xyz,float),[.28,-.33,.015],[.78,.37,.38]);grip=float(np.clip(grip,0,.04))
        if not np.isfinite(xyz).all() or not np.isfinite(grip):raise ValueError('Non-finite robot command')
        q=p.calculateInverseKinematics(self.arm,11,xyz,self.quat,maxNumIterations=150,residualThreshold=1e-5)
        p.setJointMotorControlArray(self.arm,list(range(7)),p.POSITION_CONTROL,targetPositions=q[:7],forces=[87]*7,positionGains=[.12]*7)
        p.setJointMotorControlArray(self.arm,[9,10],p.POSITION_CONTROL,targetPositions=[grip]*2,forces=[30]*2,positionGains=[.3]*2)
        self.step(steps,frames)

    def execute_chunk(self,action,frames=None):
        for i,w in enumerate(action):self.move(w[:3],w[3],160 if i in [2,6] else 150,frames)

    def perturb(self,event,task):
        """Evaluation harness only. These events do not expose labels to controllers."""
        if event=='object_moved' and task.object in self.objects:
            b=self.objects[task.object];xyz,orn=p.getBasePositionAndOrientation(b)
            shifted=np.array(xyz);shifted[:2]+=np.array([.07,-.07]);shifted[:2]=np.clip(shifted[:2],[.33,-.28],[.75,.14]);shifted[2]=.022
            p.resetBasePositionAndOrientation(b,shifted,orn);p.resetBaseVelocity(b,[0,0,0],[0,0,0]);self.step(20)
        elif event=='slip':
            xyz=self.state()[:3];self.move(xyz,.04,steps=80);self.move(xyz,0.,steps=60)
        elif event=='missing' and task.object in self.objects:
            p.removeBody(self.objects.pop(task.object))

    def score(self,task):
        if task.object not in self.objects:return {'success':False,'distance_m':None,'lift_height_m':0.}
        src=np.array(p.getBasePositionAndOrientation(self.objects[task.object])[0]);lift=self.max_heights[task.object]
        if task.kind=='lift':
            distance=float(np.linalg.norm(src[:2]-self.state()[:2]));ok=src[2]>.14 and distance<.045 and self.state()[3]>.008
        else:
            if task.pad not in self.pads:return {'success':False,'distance_m':None,'lift_height_m':lift}
            dst=np.array(p.getBasePositionAndOrientation(self.pads[task.pad])[0]);distance=float(np.linalg.norm(src[:2]-dst[:2]))
            ok=distance<.045 and .007<src[2]<.065 and lift>.10
        return {'success':bool(ok),'distance_m':distance,'lift_height_m':float(lift)}

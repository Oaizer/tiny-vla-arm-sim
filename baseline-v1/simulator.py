"""CPU Panda pick/place simulator. Object state is exposed only to teacher/scorer."""
from __future__ import annotations
import numpy as np
import pybullet as p
import pybullet_data

COLORS = {'red': (0.88, .08, .08), 'green': (.08,.72,.18), 'blue': (.08,.20,.92)}
PADS = {'yellow': (.92,.78,.10), 'purple': (.65,.15,.80)}
VOCAB = ['<pad>','put','the','red','green','blue','block','on','yellow','purple','pad','place','move','to']

def tokenize(instruction):
    words=instruction.lower().replace('.','').split()
    unknown=set(words)-set(VOCAB)
    if unknown: raise ValueError(f'Unsupported words: {unknown}; vocabulary: {VOCAB}')
    return np.array([VOCAB.index(w) for w in words]+[0]*(12-len(words)),dtype=np.int64)

class PandaSim:
    def __init__(self, size=96):
        self.cid=p.connect(p.DIRECT)
        self.size=size
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-9.81)
        p.setTimeStep(1/240)
        p.setPhysicsEngineParameter(numSolverIterations=100)
        p.loadURDF('plane.urdf')
        self.arm=p.loadURDF('franka_panda/panda.urdf',useFixedBase=True)
        self.rest=[0,-.45,0,-2.3,0,1.85,.78]
        for i,q in enumerate(self.rest): p.resetJointState(self.arm,i,q)
        self.quat=p.getQuaternionFromEuler([np.pi,0,0])
        self.view=p.computeViewMatrix([.53,.01,1.18],[.53,.01,0],[0,1,0])
        self.proj=p.computeProjectionMatrixFOV(43,1,.05,3)
        self.blocks={};self.pads={}
        self._box_shape=p.createCollisionShape(p.GEOM_BOX,halfExtents=[.02,.02,.02])
        self.park=np.array([.27,-.35,.34])
        self.move(self.park,.04,steps=180)

    def close(self): p.disconnect(self.cid)

    def reset(self, seed):
        rng=np.random.default_rng(seed)
        for b in list(self.blocks.values())+list(self.pads.values()):p.removeBody(b)
        self.blocks={};self.pads={}
        # Reset arm before placing new objects.
        for i,q in enumerate(self.rest):p.resetJointState(self.arm,i,q)
        self.move(self.park,.04,steps=120)
        positions=[]
        for color,rgb in COLORS.items():
            while True:
                xy=rng.uniform([.37,-.24],[.68,.08])
                if all(np.linalg.norm(xy-other)>.095 for other in positions):break
            positions.append(xy)
            tint=np.clip(np.array(rgb)*rng.uniform(.85,1.05),0,1)
            vs=p.createVisualShape(p.GEOM_BOX,halfExtents=[.02]*3,rgbaColor=[*tint,1])
            b=p.createMultiBody(.04,self._box_shape,vs,[*xy,.022])
            p.changeDynamics(b,-1,lateralFriction=1.1,spinningFriction=.002,restitution=0)
            self.blocks[color]=b
        for name,rgb in PADS.items():
            x=(.40 if name=='yellow' else .65)+rng.uniform(-.025,.025)
            y=.265+rng.uniform(-.02,.02)
            vs=p.createVisualShape(p.GEOM_BOX,halfExtents=[.055,.055,.001],rgbaColor=[*rgb,1])
            self.pads[name]=p.createMultiBody(0,-1,vs,[x,y,.001])
        for _ in range(30):p.stepSimulation()
        self.max_heights={c:float(p.getBasePositionAndOrientation(b)[0][2]) for c,b in self.blocks.items()}
        return self.image(),self.state()

    def state(self):
        xyz=p.getLinkState(self.arm,11,computeForwardKinematics=True)[4]
        grip=sum(p.getJointState(self.arm,j)[0] for j in [9,10])/2
        return np.array([*xyz,grip],np.float32)

    def image(self, size=None, overview=False):
        n=size or self.size
        view=self.view
        if overview:view=p.computeViewMatrix([1.25,-1.10,1.05],[.42,.02,.12],[0,0,1])
        out=p.getCameraImage(n,n,viewMatrix=view,projectionMatrix=self.proj,
             renderer=p.ER_TINY_RENDERER,shadow=0,lightDirection=[1,-2,4])
        return np.asarray(out[2],np.uint8).reshape(n,n,4)[:,:,:3].copy()

    def pixel(self,xyz):
        view=np.array(self.view).reshape(4,4,order='F')
        proj=np.array(self.proj).reshape(4,4,order='F')
        point=proj@view@np.array([*xyz,1])
        point=point[:3]/point[3]
        return np.array([(point[0]+1)/2,(1-point[1])/2],np.float32)

    def truth(self,color,pad):
        src=np.array(p.getBasePositionAndOrientation(self.blocks[color])[0])
        dst=np.array(p.getBasePositionAndOrientation(self.pads[pad])[0])
        return src,dst

    def teacher(self,color,pad):
        src,dst=self.truth(color,pad)
        # Cartesian grasp frame link 11, metres; finger opening is per finger.
        return np.array([
            [*src[:2],.20,.04], [*src[:2],.025,.04],
            [*src[:2],.025,0.0], [*src[:2],.24,0.0],
            [*dst[:2],.24,0.0], [*dst[:2],.065,0.0],
            [*dst[:2],.065,.04], [*dst[:2],.25,.04],
        ],np.float32)

    def move(self,xyz,grip,steps=100,frames=None):
        xyz=np.array(xyz)
        q=p.calculateInverseKinematics(self.arm,11,xyz,self.quat,
              maxNumIterations=150,residualThreshold=1e-5)
        p.setJointMotorControlArray(self.arm,list(range(7)),p.POSITION_CONTROL,
              targetPositions=q[:7],forces=[87]*7,positionGains=[.12]*7)
        p.setJointMotorControlArray(self.arm,[9,10],p.POSITION_CONTROL,
              targetPositions=[float(grip)]*2,forces=[30]*2,positionGains=[.3]*2)
        for t in range(steps):
            p.stepSimulation()
            if frames is not None and t%12==0:frames.append(self.image(384,overview=True))

    def execute(self,waypoints,frames=None):
        # IK and position servos are the low-level controller; no object lookup here.
        for i,w in enumerate(waypoints):
            goal=np.clip(w[:3],[.28,-.32,.015],[.77,.36,.38])
            grip=float(np.clip(w[3],0,.04))
            self.move(goal,grip,steps=160 if i in [2,6] else 150,frames=frames)
            # Outcome logging only: object positions never affect the executed commands.
            for c,b in self.blocks.items():
                self.max_heights[c]=max(self.max_heights[c],p.getBasePositionAndOrientation(b)[0][2])
        for _ in range(100):p.stepSimulation()

    def score(self,color,pad):
        src,dst=self.truth(color,pad)
        return bool(np.linalg.norm(src[:2]-dst[:2])<.045 and .008<src[2]<.055
                    and self.max_heights[color]>.10),float(np.linalg.norm(src[:2]-dst[:2]))

if __name__=='__main__':
    import argparse,json
    from PIL import Image
    ap=argparse.ArgumentParser();ap.add_argument('--episodes',type=int,default=6)
    ap.add_argument('--video');args=ap.parse_args()
    sim=PandaSim();results=[];frames=[]
    for i in range(args.episodes):
        color=list(COLORS)[i%3];pad=list(PADS)[i%2]
        img,state=sim.reset(i)
        if i==0:Image.fromarray(img).save('camera.png')
        sim.execute(sim.teacher(color,pad),frames if args.video and i==0 else None)
        success,dist=sim.score(color,pad)
        row={'seed':i,'color':color,'pad':pad,'success':success,'distance':dist}
        print(json.dumps(row),flush=True);results.append(row)
    if args.video:
        import imageio.v2 as imageio
        imageio.mimsave(args.video,frames,fps=20)
    sim.close()

"""Check the bundled B601-RS description without loading a physics engine."""
from pathlib import Path
import xml.etree.ElementTree as ET


MODEL = Path(__file__).with_name('assets') / 'rebot_b601_rs' / 'RS'
URDF = MODEL / 'urdf' / 'ReBot_Arm_RS.urdf'


def main():
    robot = ET.parse(URDF).getroot()
    names = [joint.get('name') for joint in robot.findall('joint')]
    arm = [name for name in names if name and name.startswith('joint') and name[5:].isdigit()]
    gripper = [name for name in names if name and name.startswith('gripper_joint')]
    meshes = [node.get('filename') for node in robot.findall('.//mesh')]
    missing = [str((URDF.parent / name).resolve()) for name in meshes
               if not (URDF.parent / name).is_file()]
    if len(arm) != 6 or len(gripper) != 2 or missing:
        raise SystemExit(f'Invalid B601-RS package: {len(arm)} arm joints, '
                         f'{len(gripper)} gripper joints, {len(missing)} missing meshes.')
    print(f'B601-RS model OK: {len(arm)} arm joints, {len(gripper)} gripper joints, '
          f'{len(meshes)} mesh references.')


if __name__ == '__main__':
    main()

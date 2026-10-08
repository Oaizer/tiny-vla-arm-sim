"""Check the language rules and repeatable simulator behavior."""
import unittest
import numpy as np
from tinyvla.language import commands,parse,tokenize,OBJECTS
from tinyvla.sim import ArmSim

class ProjectContracts(unittest.TestCase):
    def test_supported_language(self):
        for text in commands()+commands(True):
            self.assertIn(parse(text).object,OBJECTS);self.assertEqual(tokenize(text).shape,(16,))
        with self.assertRaises(ValueError):parse('put the robot in the oven')

    def test_seed_and_camera_contract(self):
        sim=ArmSim()
        try:
            first=sim.reset(13);points,pixels,present=sim.labels();second=sim.reset(13)
            np.testing.assert_array_equal(first['rgb'],second['rgb'])
            for i in range(len(points)):
                ray=first['calibration'][i]@np.array([*pixels[i],1.])
                np.testing.assert_allclose(ray[:2]/ray[2],points[i,:2],atol=1e-5)
            self.assertEqual(set(first),{'rgb','state','calibration'})
        finally:sim.close()

if __name__=='__main__':unittest.main()

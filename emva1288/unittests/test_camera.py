import unittest
import emva1288.camera as cam
import numpy as np
from emva1288.camera import routines


class CameraTestCase(unittest.TestCase):
    def setUp(self):
        self.cam = cam.Camera()

    def tearDown(self):
        del self.cam

    def test_img(self):
        img = self.cam.grab(0)
        self.assertEqual((self.cam.height, self.cam.width), np.shape(img))

    def test_radiance(self):
        img1 = self.cam.grab(0)
        img2 = self.cam.grab(self.cam.get_radiance_for(mean=250))
        self.assertLess(img1.mean(), img2.mean())


class CameraTestBayer(unittest.TestCase):
    def bayer_layer(self):
        self.height = cam.Camera().height
        self.width = cam.Camera().width
        b_layer = routines.get_bayer_filter(1, 0.15, 0.02, 1,
                                            self.height, self.width)
        self.cam = cam.Cam(radiance_factor=bayer_layer)
    # in production
    pass

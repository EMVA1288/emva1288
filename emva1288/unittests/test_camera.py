import unittest
import emva1288.camera as cam
import numpy as np


class CameraTestCase(unittest.TestCase):
    def setUp(self):
        self.cam = cam.Camera()

    def tearDown(self):
        del self.cam

    def test_img(self):
        img = self.cam.grab()
        self.assertEqual((self.cam._img_x, self.cam._img_y), np.shape(img))

    def test_radiance(self):
        self.cam.radiance = 0
        img1 = self.cam.grab()
        self.cam.radiance = self.cam.saturation_radiance
        img2 = self.cam.grab()
        self.assertLess(img1.mean(), img2.mean())

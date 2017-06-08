import unittest
from emva1288.camera.camera import Camera
import numpy as np
from emva1288.camera import routines


class CameraTestCase(unittest.TestCase):
    def setUp(self):
        self.cam = Camera()

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
    def test_get_bayer(self):
        pass

    def test_bayer_layer(self):
        h = 480
        w = 640
        transmition_red = 0.15
        transmition_blue = 0.02
        transmition_green = 1.
        b_layer = routines.get_bayer_filter(transmition_green,
                                            transmition_red,
                                            transmition_blue,
                                            transmition_green, w, h)
        self.assertEqual((h, w), b_layer.shape)
        cam = Camera(width=w, height=h, radiance_factor=b_layer)
        target = cam.img_max / 2
        radiance = cam.get_radiance_for(mean=target)
        img = cam.grab(radiance)
        green_filter = routines.get_bayer_filter(0, 1, 1, 0, w, h)
        blue_filter = routines.get_bayer_filter(1, 1, 0, 1, w, h)
        red_filter = routines.get_bayer_filter(1, 0, 1, 1, w, h)
        self.assertAlmostEqual(np.ma.masked_array(
            img,
            mask=green_filter).mean(),
            target, delta=5.0,
            msg="green not in range")
        self.assertAlmostEqual(np.ma.masked_array(
            img,
            mask=red_filter).mean(),
            target*transmition_red, delta=5.0,
            msg="red not in range")
        self.assertAlmostEqual(np.ma.masked_array(
            img,
            mask=blue_filter).mean(),
            target*transmition_blue, delta=5.0,
            msg="blue not in range")

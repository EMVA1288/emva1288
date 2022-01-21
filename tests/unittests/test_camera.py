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

    def test_bayer_layer(self):
        # Init the parameters
        h, w = [480, 640]
        wavelength = np.linspace(400, 800, 100)
        transmission_red = 670
        transmission_blue = 450
        transmission_green = 550
        b_layer = routines.get_bayer_filter(transmission_green,
                                            transmission_red,
                                            transmission_blue,
                                            transmission_green,
                                            w, h, wavelength)
        qe = routines.Qe(filter=b_layer)
        cam = Camera(width=w, height=h, qe=qe)
        # Initiata a cam without a bayer filter
        cam_d = Camera(width=w, height=h)
        # Set the camera for testing the layer
        target = cam.img_max / 2
        # Get the radiance to grab from the second cam. The output radiance
        # is affected by qe, so the bayer_filter as well
        radiance = cam_d.get_radiance_for(mean=target)
        img = cam.grab(radiance)
        green_filter = np.tile([[0, 1], [1, 0]], (int(h/2), int(w/2)))
        blue_filter = np.tile([[1, 0], [1, 1]], (int(h/2), int(w/2)))
        red_filter = np.tile([[1, 1], [0, 1]], (int(h/2), int(w/2)))

        gf = b_layer[0, 0, :].mean()
        rf = b_layer[0, 1, :].mean()
        bf = b_layer[1, 0, :].mean()
        # Test if the mean of the green it's 100% of the target +/- 5
        self.assertAlmostEqual(np.ma.masked_array(
            img,
            mask=green_filter).mean(),
            target * gf, delta=10,
            msg="green not in range")
        # Test if the mean of the red it's 15% of the target +/- 5
        self.assertAlmostEqual(np.ma.masked_array(
            img,
            mask=red_filter).mean(),
            target * rf, delta=10,
            msg="red not in range")
        # Test if the mean of the blue it's 2% of the target +/- 5
        self.assertAlmostEqual(np.ma.masked_array(
            img,
            mask=blue_filter).mean(),
            target * bf, delta=10,
            msg="blue not in range")


class CameraTestPrnuDsnu(unittest.TestCase):
    def test_prnu(self):
        # Init the parameters
        h, w = [480, 640]
        rep = 200
        value8 = 3
        # create the pattern of the prnu
        prnu_array = np.ones((8))
        prnu_array[-1] = value8
        prnu = routines.get_tile(prnu_array, h, w)
        # Set the camera for testing the prnu
        cam = Camera(width=w, height=h, prnu=prnu)
        var = np.sqrt(cam._sigma2_dark_0)
        target = cam.img_max / 2
        # The target (top_target) is the multiplication of the target
        # (what we expect without prnu) and the value8(prnu). We can do it
        # because if we look at the _u_e function in emva1288.camera.camera
        # the prnu affect the QE with a multiplication. So if we just
        # multiplied the target by the prnu it's the same thing.
        # But this value can go over the maximal value for one pixel, this
        # is why we use the min() function to take the maximal value than the
        # camera can take.
        top_target = min(target * value8, cam.img_max)
        radiance = cam.get_radiance_for(mean=target)
        img = cam.grab(radiance)
        # create the mask
        prnu_mask = np.zeros((8))
        prnu_mask[-1] = 1
        prnu_mask_resize = routines.get_tile(prnu_mask, h, w)
        prnu_non_mask = np.ones((8))
        prnu_non_mask[-1] = 0
        prnu_non_mask_resize = routines.get_tile(prnu_non_mask, h, w)
        # Test if the mean it's 100% of the target +/- variance
        self.assertAlmostEqual(np.ma.masked_array(
            img,
            mask=prnu_mask_resize).mean(),
            target, delta=var,
            msg="values are not in range")
        # Test if the mean of the 8th value it's value8
        # multiplied be the target +/- variance
        self.assertAlmostEqual(np.ma.masked_array(
            img,
            mask=prnu_non_mask_resize).mean(),
            top_target, delta=var,
            msg="8th value it's not in range")

    def test_dsnu(self):
        # Init the parameters
        h, w = [480, 640]
        value8 = 5
        rep = 200
        # create the pattern of the dsnu
        dsnu_array = np.ones((8))
        dsnu_array[-1] = value8
        dsnu = routines.get_tile(dsnu_array, h, w)
        # Set the camera for testing the dsnu
        cam = Camera(width=w, height=h, dsnu=dsnu)
        var = np.sqrt(cam._sigma2_dark_0)
        # The target is the number of electrons who are not affected
        # by the dsnu. To resume, we suppose to observe is a combinaison of
        # electrons from the dark signal and the temperature. The total need
        # to be multiplied by the gain of the system (K).
        # for more eplination see the grab function in emva1288.camera.camera
        target = cam.K * (cam._dark_signal_0 + cam._u_therm())
        # Here the target (top_target) is the part who is affected by
        # the dsnu. Physicaly, the same phenomen append but this time the
        # dark signal is NonUniform so thw value who represent the dsnu is
        # added to the dark signal befor the multiplication of the gain.
        top_target = cam.K * (cam._dark_signal_0 + cam._u_therm() + value8)
        img = cam.grab(0)
        # create the mask
        dsnu_mask = np.zeros((8))
        dsnu_mask[-1] = 1
        dsnu_mask_resize = routines.get_tile(dsnu_mask, h, w)
        dsnu_non_mask = np.ones((8))
        dsnu_non_mask[-1] = 0
        dsnu_non_mask_resize = routines.get_tile(dsnu_non_mask, h, w)
        # Test if the mean it's 100% of the target +/- variance
        self.assertAlmostEqual(np.ma.masked_array(
            img,
            mask=dsnu_mask_resize).mean(),
            target, delta=var,
            msg="values are not in range")
        # Test if the mean of the 8th value it's value8
        # multiplied be the target +/- variance
        self.assertAlmostEqual(np.ma.masked_array(
            img,
            mask=dsnu_non_mask_resize).mean(),
            top_target, delta=var,
            msg="8th value it's not in range")


class CameraTestRoutines(unittest.TestCase):
    def test_get_tile(self):
        # TEST 1D
        # Init the parameters
        h, w = [1, 24]
        dim1 = np.zeros((8))
        # Supposed Results
        res_array = routines.get_tile(dim1, h, w)
        res_dim1 = np.zeros((24))
        # Test to see if the layer come right
        # shape[0] will give us the number in (width,)
        self.assertEqual(w, res_array.shape[0])
        self.assertEqual(res_dim1.tolist(), res_array.tolist())

        # TEST 2D
        # Init the parameters
        h, w = [5, 7]
        dim2 = np.zeros((3))
        # Supposed Results
        res_array = routines.get_tile(dim2, h, w)
        res_dim2 = np.zeros((5, 7))
        # Test to see if the layer come right
        self.assertEqual((h, w), res_array.shape)
        self.assertEqual(res_dim2.tolist(), res_array.tolist())

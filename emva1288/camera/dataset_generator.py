# This module contains a generator that generates a descriptor file and the
# corresponding images using the implemented camera.

from emva1288.camera.camera import Camera as Cam
import numpy as np
import tempfile
import os
from PIL import Image


def _get_emva_gain(cam):
    """Find the gain to satisfy EMVA1288 requirements"""
    gini = cam.K
    # Find gain with a minum temporal noise of 0.5DN
    g = cam.Ks[0]
    for gain in cam.Ks:
        cam.K = gain
        g = gain
        img1 = cam.grab(0).astype(np.int64)
        img2 = cam.grab(0).astype(np.int64)
        if (img1 - img2).std() > 0.5:
            break
    cam.K = gini
    return g


def _get_emva_blackoffset(cam):
    """Find the blackoffset to satifsfy EMVA1288 requirements"""
    bini = cam.blackoffset
    # Find black offset with a maximum of 0.5% of values at Zero
    bo = cam.blackoffsets[0]
    pixels = cam.img_x * cam.img_y
    for i in cam.blackoffsets:
        cam.blackoffset = i
        img = cam.grab(0)
        bo = i
        if np.count_nonzero(img) > pixels * .995:
            break
    cam.blackoffset = bini
    return bo


class DatasetGenerator:
    """Dataset generator.

    Creates a descriptor file and the corresponding linked images for a
    a exposure variant test example according to the emva1288 standart.
    The images are created using the implemented camera in the emva module.
    """

    def __init__(self,
                 steps=100,
                 L=50,
                 version='3.0',
                 image_format='png',  # best memory consumption
                 outdir=None,  # directory where to save the dataset
                 **kwargs
                 ):
        """Dataset generator init method.

        It creates the :class:`~emva1288.camera.camera.Camera` object that
        will generate the images. The camera is intialized according to the
        given kwargs. Then, it :meth:`makes <run_test>` the images
        with it by changing its
        exposure time and :meth:`saves <save_images>` the images
        and the descriptor file.

        Parameters
        ----------
        L : int, optional
            The number of image taken during a spatial test point.
        version : str, optional
                  Data version to add in descriptor file.
        image_format : str, optional
                       The image's format when they are saved.
        outdir : str, optional
                 The output directory where the descriptor file and the images
                 will be saved. If None, it will create a tempory directory
                 that will be deleted (and its contents) when the dataset
                 generator object is deleted.
        kwargs : All other kwargs are passed to the camera.
        """
        self.cam = Cam(**kwargs)
        # set the camera parameters for the test
        self.cam.exposure = self.cam.exposure_max
        self.cam.blackoffset = _get_emva_blackoffset(self.cam)
        self.cam.K = _get_emva_gain(self.cam)
        self._saturation_radiance = self.cam.get_radiance_for()

        self._steps = steps  # number of points to take
        self._L = L  # number of images to take for a spatial test
        self._version = version  # data version
        # store image format
        self._image_format = image_format

        # images will be saved one at a time during the generation into outdir
        self.outdir = outdir
        # create temporary directory to store the dataset
        if outdir is None:
            self.tempdir = tempfile.TemporaryDirectory()
            self.outdir = self.tempdir.name
        # create dir where images will be saved
        os.makedirs(os.path.join(self.outdir, 'images'))

        # create test points
        self._points = self._get_points()
        # run test
        self._descriptor_path = self.run_test()

    @property
    def points(self):
        """The test points suite."""
        return self._points

    @property
    def descriptor_path(self):
        """The absolute path to the descriptor file."""
        return self._descriptor_path

    def _get_points(self):
        """Get the test points list to iterate on."""
        # compute the test points parameters
        bright = []
        dark = []
        exposures = np.round(np.linspace(self.cam.exposure_min,
                                         self.cam.exposure_max,
                                         self._steps),
                             decimals=1)

        for i, exposure in enumerate(exposures):
            if self._is_point_spatial_test(i):
                # spatial test
                L = self._L
            else:
                # temporal test
                L = 2

            d_bright = {'exposure': exposure,  # exposure variation
                        'radiance': self._saturation_radiance,
                        'photons':
                        round(self.cam.get_photons(self._saturation_radiance,
                                                   exposure),
                              1),
                        'names':
                        self._get_image_names(i, L, self._saturation_radiance),
                        }
            bright.append(d_bright)

            d_dark = {'exposure': exposure,
                      'radiance': 0.0,
                      'photons': 0.0,
                      'names': self._get_image_names(i, L, 0.0),
                      }
            dark.append(d_dark)

        return bright + dark

    def _is_point_spatial_test(self, i):
        """Check if a point index should be a spatial test.

        Spatial points are done at midpoint of bright and dark series.
        """
        v = self._steps // 2
        if i in (v, self._steps + v):
            return True
        return False

    def _get_descriptor_line(self, exposure, photons):
        """Create the line introducing a test point images in descriptor."""
        if photons == 0.0:
            # dark image
            return "d %.1f" % exposure
        # bright image
        return "b %.1f %.1f" % (exposure, photons)

    def _get_image_names(self, i, L, radiance):
        """Create an image filename."""
        names = []
        for j in range(L):
            # i is the point number
            # j is the image number for the point i
            if radiance == 0.0:
                # dark image
                prefix = "d"
            else:
                # bright image
                prefix = "b"
            if L == self._L:
                # spatial point
                prefix += "_s"
                i = 0  # For spatial image, they are at a '000' test point

            # The modulo beside the i is because dark image points have same
            # number than the corresponding bright one
            names.append(prefix + "_%03d_snap_%03d.%s" % (i % self._steps,
                                                          j,
                                                          self._image_format))
        return names

    def _get_imgs(self, n, radiance):
        """Create a list of dictionaries that contains one image.

        The dictionaries contains the data and name of one image.
        The list is the images for one test point.
        """
        # computes an array of dict whose keys are the name of the file
        # and the data of the image to save
        imgs = []
        for j in range(n):
            imgs.append(self.cam.grab(radiance))
        return imgs

    def run_test(self):
        """Run the test points, save the images and generate descriptor."""
        for i, point in enumerate(self.points):
            self.cam.exposure = point['exposure']
            imgs = self._get_imgs(len(point['names']), point['radiance'])
            # save images now to prevent too much memory consumption
            self.save_images(imgs, point['names'])

        # generate descriptor file
        return self._generate_descriptor_file()

    def save_images(self, imgs, names):
        """Save the images one test point at a time.
        """
        # save the images contained in d
        dtype = np.uint32
        mode = 'I'
        if self.cam.bit_depth <= 8:
            # 8 bit images have special format for PIL
            dtype = np.uint8
            mode = 'L'
        for i, name in enumerate(names):
            im = Image.fromarray(imgs[i].astype(dtype), mode)
            path = os.path.join(self.outdir, 'images', name)
            # (filename already contains image format)
            im.save(path)
            # erase image from dict to reduce memory consuption

    def _generate_descriptor_file(self):
        """Generate and save the descriptor file."""
        path = os.path.join(self.outdir, 'EMVA1288descriptor.txt')
        with open(path, 'w') as f:
            # first, write the version line
            f.write("v %s\n" % self._version)
            # second, write camera's properties
            f.write("n %i %i %i\n" % (self.cam.bit_depth,
                                      self.cam.img_x,
                                      self.cam.img_y))
            # Then write the rest of the file
            # iterate over all data points
            for i, point in enumerate(self.points):
                f.write("%s\n" % self._get_descriptor_line(point['exposure'],
                                                           point['photons']))

                # iterate over this point's images
                for name in point['names']:
                    f.write("i %s\n" % ('images\\' + name))
        return path

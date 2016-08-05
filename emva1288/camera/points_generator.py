import numpy as np
from collections import OrderedDict


class PointsGenerator:
    """Class that generates a dictionary of operation points for an emva test.

    The points are stored in the :attr:`points` attribute as a dictionary.
    The points are ordered if they are a 'spatial' or 'temporal' operation
    point. Under each keys there is a dictionary whose keys are the
    different exposure times and their values are the list of radiances
    under which the camera is illuminated.
    """
    def __init__(self,
                 cam,
                 exposure_min=None,
                 exposure_max=None,
                 exposure_fixed=None,
                 radiance_min=None,
                 radiance_max=None,
                 gain=None,
                 blackref=None,
                 steps=100):
        """Point generator init method.

        Parameters
        ----------
        cam : The camera object that will be taking the images.
        exposure_min : float, optional
                       The minimal exposure time (in ns).
        exposure_max : float, optional
                       The maximal exposure time (in ns).
        exposure_fixed : float, optional
                         By default, the points given are for an exposure time
                         variation test (if this is None). If a value is given
                         to this kwarg, this will be the camera's exposure
                         time (in ns) at which the operation points will be
                         set for an illumination variation test.
        radiance_min : float, optional
                       The minimal radiance (in W/cm^2/sr). If None, a value
                       above dark illumination will be automatically chosen.
        radiance_max : float, optional
                       The maximal radiance (in W/cm^2/sr).
                       If None, the maximal
                       radiation will be taken as the saturation radiation for
                       the exposition time given in the exposure_fixed kwarg.
        gain : float, optional
               The camera's gain at which we want the test to run.
        blackref : float, optional
                   The camera's blackoffset at which we want the test to run.
        steps : int, optional
                The number of points in the test.
        """
        self._cam = cam
        self._steps = steps
        self._exposure_min = exposure_min or self._cam.exposure_min
        self._exposure_max = exposure_max or self._cam.exposure_max
        self._exposure = exposure_fixed
        self._radiance_min = radiance_min
        self._radiance_max = radiance_max
        self._gain = gain or self._cam.K
        self._blackref = blackref or self._cam.blackoffset

        if self._exposure is None:
            # Get radiance for saturation at maximal exposure time
            # Only if it is for an exposure time variation test
            self._cam.exposure = self._exposure_max
            self._radiance = self._cam.get_radiance_for()

        else:
            # get radiances for radiation variation
            self._cam.exposure = self._exposure
            self._cam.K = self._gain
            self._cam.blackoffset = self._blackref
            m = self._cam.grab(0.0).mean()
            target = (self._cam.img_max - m) / self._steps + m
            self._radiance_min = self._cam.get_radiance_for(mean=target)
            self._radiance_max = self._cam.get_radiance_for()

        # By default, an exposure time variation data points
        # If exposure fixed is given, it is photons variation
        self._points = self._get_points()

    def _get_points(self):
        spatial = OrderedDict()
        temporal = OrderedDict()

        if self._exposure is None:
            # Exposure time variation
            # round to only have one decimal
            exposures = np.round(np.linspace(self._exposure_min,
                                             self._exposure_max,
                                             self._steps), 1)
            # only one radiance
            radiances = [self._radiance, 0.0]

            # Main loop to compute points
            for n, texp in enumerate(exposures):
                if self._is_point_spatial(n):
                    spatial[texp] = radiances
                temporal[texp] = radiances

        else:
            # Photons variations
            # only one exposure time
            radiances = np.linspace(self._radiance_min,
                                    self._radiance_max,
                                    self._steps).tolist()
            # round to only have one decimal
            self._exposure = round(self._exposure)
            radiances.append(0.0)
            spatial[self._exposure] = [radiances[self._steps // 2], 0.0]
            temporal[self._exposure] = radiances

        return {'spatial': spatial, 'temporal': temporal}

    @property
    def points(self):
        """The operation points."""
        return self._points

    def _is_point_spatial(self, n):
        """Checks if a spatial test must be done at this point.

        Spatial test are executed at mid run for both dark and bright tests.
        """
        middle = self._steps // 2
        if n in (middle, self._steps + middle):
            return True
        return False

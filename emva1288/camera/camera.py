import numpy as np
from emva1288.camera import routines


class Camera(object):
    """Camera simulator. It creates images according to the given parameters.
    """
    def __init__(self,
                 f_number=8,  # F-number of the light source/camera setup
                 pixel_area=25,  # um^2
                 bit_depth=8,  # Bit depth of the image [8, 10, 12, 14]
                 img_x=640,
                 img_y=480,

                 temperature=22,  # Sensor temperature in ^oC
                 temperature_ref=30,  # Reference temperature
                 temperature_doubling=8,  # Doubling temperature

                 wavelength=525,  # illumination wavelength
                 qe=None,  # Quantum efficiency for the given wavelength

                 exposure=1000000,  # Exposure time in ns
                 exposure_min=10000,  # Minimum exposure time in ns
                 exposure_max=500000000,  # Maximum exposure time in ns

                 K=0.01,  # Overall system gain
                 K_min=0.01,
                 K_max=17.,
                 K_steps=255,

                 blackoffset=0,
                 blackoffset_min=0,
                 blackoffset_max=16,
                 blackoffset_steps=255,

                 dark_current_ref=0,
                 dark_signal_0=0,
                 sigma2_dark_0=10
                 ):
        """Camera simulator init method.

        Parameters
        ----------
        f_number : float, optional
                   The emva1288 f_number for the camera.
        pixel_area : float, optional
                     The area of one pixel (in um ^ 2)
        bit_depth : int, optiona
                    The number of bits allowed for one pixel value.
        img_x : int, optional
                The number of pixel in the x axis.
        img_y : int, optional
                The number of pixels in the y axis.
        temperature : float, optional
                      The camera's sensor temperature in degrees Celsius.
        temperature_ref : float, optional
                          The reference temperature (at which the dark current
                          is equal to the reference dark current).
        temperature_doubling: float, optional
                              The doubling temperature (at which the dark
                              current is two times the reference dark
                              current).
        wavelength : float, optional
                     The light wavelength hitting the sensor (in nm).
        qe : float, optional
             Quantum efficiency (between 0 and 1). If None, a simulated
             quantum efficiency is choosen with the
             :func:`~emva1288.camera.routines.qe` function.
        exposure : float, optional
                   The camera's exposure time in ns.
        exposure_min : float, optional
                       The camera's minimal exposure time in ns.
        exposure_max : float, optional
                       The camera's maximal exposure time in ns.
        K : float, optional
            The overall system gain (in DN/e^-).
        K_min : float, optional
                The overall minimal system gain (in DN/e^-).
        K_max : float, optional
                The overall maximal system gain (in DN/e^-).
        K_steps : int, optional
                  The number of available intermediate overall system gains
                  between K_min and K_max.
        blackoffset : float, optional
                      The dark signal offset for each pixel (in DN).
        blackoffset_min: float, optional
                         The minimal dark signal offset for each pixel (in DN).
        blackoffset_max : float, optional
                          The maximal dark signal offset for each pixel
                          (in DN).
        blackoffset_steps : int, optional
                            The number of available blackoffsets between the
                            mimimal and maximal blackoffsets.
        dark_current_ref : float, optional
                           The reference dark current used for computing the
                           total dark current.
        dark_signal_0 : float, optional
                        The offset for the computation of the mean number
                        of electrons in the dark for one pixel (in DN/s).
        sigma2_dark_0 : float, optional
                        The offset for the computation of the dark signal
                        (the dark signal standart deviation).
        """

        self._f_number = f_number
        self._pixel_area = pixel_area
        self._bit_depth = bit_depth
        self._img_max = 2 ** int(bit_depth) - 1
        self._img_x = img_x
        self._img_y = img_y

        self._temperature = temperature
        self._temperature_ref = temperature_ref
        self._temperature_doubling = temperature_doubling

        self._wavelength = wavelength
        self._qe = qe
        # When no specific qe is provided we simulate one
        if qe is None:
            self._qe = routines.qe(wavelength)

        self._dark_current_ref = dark_current_ref
        self._dark_signal_0 = dark_signal_0
        self._sigma2_dark_0 = sigma2_dark_0

        self._exposure = exposure
        self._exposure_min = exposure_min
        self._exposure_max = exposure_max

        self.__Ks = np.linspace(K_min, K_max, num=K_steps)
        self._K = None
        self.K = K

        self.__blackoffsets = np.linspace(blackoffset_min, blackoffset_max,
                                          num=blackoffset_steps)
        self._blackoffset = None
        self.blackoffset = blackoffset

    @property
    def bit_depth(self):
        """The number of bits allowed for a gray value for one pixel."""
        return self._bit_depth

    @property
    def pixel_area(self):
        """The area of one pixel (in um ^ 2)."""
        return self._pixel_area

    @property
    def img_max(self):
        """The maximal value for one pixel (in DN)."""
        return self._img_max

    @property
    def img_x(self):
        """The number of pixel in the x axis."""
        return self._img_x

    @property
    def img_y(self):
        """The number of pixel in the y axis."""
        return self._img_y

    @property
    def exposure(self):
        """The camera's exposure time (in ns)."""
        return self._exposure

    @exposure.setter
    def exposure(self, value):
        self._exposure = value

    @property
    def exposure_min(self):
        """The camera's minimal exposure time (in ns)."""
        return self._exposure_min

    @property
    def exposure_max(self):
        """The camera's maximal exposure time (in ns)."""
        return self._exposure_max

    @property
    def K(self):
        """The overall system gain (in DN/e^-).

        :Setter: The setter uses the
                 :func:`~emva1288.camera.routines.nearest_value` function
                 to set the system gain to the nearest value given to the
                 setter.
                 This is because not all system gains are possible but rather a
                 linear sample between the minimal and maximal value.
        """
        return self._K

    @K.setter
    def K(self, value):
        self._K = routines.nearest_value(value, self.__Ks)

    @property
    def Ks(self):
        """The array of all the available system gains (in DN/e^-)."""
        return self.__Ks

    @property
    def blackoffset(self):
        """The system dark signal offset (in DN).

        :Setter: The setter uses the
                 :func:`~emva1288.camera.routines.nearest_value` function
                 to set the black signal offset to the nearest value given
                 to the setter. This is because not all black offsets are
                 possible but rather a linear sample between the minimal
                 and maximal value.
        """
        return self._blackoffset

    @blackoffset.setter
    def blackoffset(self, value):
        self._blackoffset = routines.nearest_value(value,
                                                   self.__blackoffsets)

    @property
    def blackoffsets(self):
        """The array of all blackoffsets (in DN)."""
        return self.__blackoffsets

    def grab(self, radiance):
        """
        Create an image based on the mean and standard deviation from the
        EMVA1288 parameters.

        The image is generated using a normal distribution for each pixel.
        """
        clipping_point = int(self.img_max)
        u_y = self._u_y(radiance)
        s2_y = np.sqrt(self._s2_y(radiance))
        img = np.random.normal(loc=u_y, scale=s2_y,
                               size=(self._img_x, self._img_y))
        img += self.blackoffset
        np.rint(img, img)
        np.clip(img, 0, clipping_point, img)
        return img.astype(np.uint64)

    def _u_y(self, radiance):
        """
        Mean digital value (in DN) of the image.
        """
        uy = self.K * (self._u_d() + self._u_e(radiance))
        return uy

    def _u_e(self, radiance):
        """
        Mean number of electrons per pixel during exposure time.
        """
        u_e = self._qe * self.get_photons(radiance)
        return u_e

    def _s2_e(self, radiance):
        """
        Variance of the number of electrons.

        Same as u_e because the number of electrons is supposed to
        be distributed by a Poisson distribution where the mean
        equals the variance.
        """
        return self._u_e(radiance)

    def _u_d(self):
        """
        Mean number of electrons without light.
        """
        u_d = (self._u_i() * self.exposure / (10 ** 9)) + self._dark_signal_0
        return u_d

    def _s2_q(self):
        """
        Variance of the quantization noise.
        """
        return 1.0 / 12.0

    def _s2_y(self, radiance):
        """
        Variance of the digital signal (= temporal noise).
        """
        s2_y = ((self.K ** 2) * (self._s2_d() + self._s2_e(radiance)) +
                self._s2_q())
        return s2_y

    def _u_i(self):
        """
        Dark current (in DN/s).
        """
        u_i = 1. * self._dark_current_ref * 2 ** (
            (self._temperature - self._temperature_ref) /
            self._temperature_doubling)
        return u_i

    def _s2_d(self):
        """
        Variance of the dark signal =  Dark temporal noise.
        """
        s2_d = self._sigma2_dark_0 + (self._u_i() * self.exposure / (10 ** 9))
        return s2_d

    def get_radiance_for(self, mean=None, exposure=None):
        """Radiance to achieve saturation.

        Calls the :func:`~emva1288.camera.routines.get_radiance` function
        to get the radiance for saturation.

        Parameters
        ----------
        mean : float, optional
               The saturation value of the camera. If None, this value
               is set to the :attr:`img_max` attribute.
        exposure : float, optional
                   The camera's exposure time at which the radiance
                   for saturation value will be computed. If None, the
                   exposure time taken will be the camera's actual exposure
                   time.

        Returns
        -------
        float :
            The radiance at which, for the given saturation value and
            the given exposure time, the camera saturates.
        """
        if not mean:
            mean = self.img_max
        if not exposure:
            exposure = self.exposure
        ud = self._u_d()
        ue = (mean / self.K) - ud
        up = ue / self._qe
        radiance = routines.get_radiance(exposure,
                                         self._wavelength,
                                         up,
                                         self.pixel_area,
                                         self._f_number)
        return radiance

    def get_photons(self, radiance, exposure=None):
        """Computes the number of photons received by one pixel.

        Uses the :func:`~emva1288.camera.routines.get_photons` function
        to compute this number.

        Parameters
        ----------
        radiance : float
                   The radiance exposed to the camera (in Wsr^-1cm^-2).
        exposure : float, optional
                   The pixel's exposure time in ns.

        Returns
        -------
        float :
            The number of photons received by one pixel.
        """
        if exposure is None:
            exposure = self.exposure
        return routines.get_photons(exposure,
                                    self._wavelength,
                                    radiance,
                                    self.pixel_area,
                                    self._f_number)

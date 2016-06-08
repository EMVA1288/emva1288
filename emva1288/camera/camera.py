import numpy as np
from emva1288.camera import routines


class Camera(object):
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
        return self._bit_depth

    @property
    def pixel_area(self):
        return self._pixel_area

    @property
    def img_max(self):
        return self._img_max

    @property
    def img_x(self):
        return self._img_x

    @property
    def img_y(self):
        return self._img_y

    @property
    def exposure(self):
        return self._exposure

    @exposure.setter
    def exposure(self, value):
        self._exposure = value

    @property
    def exposure_min(self):
        return self._exposure_min

    @property
    def exposure_max(self):
        return self._exposure_max

    @property
    def K(self):
        return self._K

    @K.setter
    def K(self, value):
        self._K = routines.nearest_value(value, self.__Ks)

    @property
    def Ks(self):
        return self.__Ks

    @property
    def blackoffset(self):
        return self._blackoffset

    @blackoffset.setter
    def blackoffset(self, value):
        self._blackoffset = routines.nearest_value(value,
                                                   self.__blackoffsets)

    @property
    def blackoffsets(self):
        return self.__blackoffsets

    def grab(self, radiance):
        '''
        Create an image based on the mean and standard deviation from the
        EMVA1288 parameters
        '''
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
        '''
        Mean of the image
        '''
        uy = self.K * (self._u_d() + self._u_e(radiance))
        return uy

    def _u_e(self, radiance):
        '''
        Mean of electrons per pixel during exposure time
        '''
        u_e = self._qe * self.get_photons(radiance)
        return u_e

    def _s2_e(self, radiance):
        '''
        variance of the number of electrons....Poisson distribution
        '''
        return self._u_e(radiance)

    def _u_d(self):
        '''
        Mean number of electrons without light
        '''
        u_d = (self._u_i() * self.exposure / (10 ** 9)) + self._dark_signal_0
        return u_d

    def _s2_q(self):
        '''
        Quantization noise
        '''
        return 1.0 / 12.0

    def _s2_y(self, radiance):
        '''
        Temporal noise
        '''
        s2_y = ((self.K ** 2) * (self._s2_d() + self._s2_e(radiance)) +
                self._s2_q())
        return s2_y

    def _u_i(self):
        '''
        Dark current
        '''
        u_i = 1. * self._dark_current_ref * 2 ** (
            (self._temperature - self._temperature_ref) /
            self._temperature_doubling)
        return u_i

    def _s2_d(self):
        '''
        Variance of the dark signal =  Dark temporal noise
        '''
        s2_d = self._sigma2_dark_0 + (self._u_i() * self.exposure / (10 ** 9))
        return s2_d

    def get_radiance_for(self, mean=None, exposure=None):
        """Radiance to achieve saturation"""
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
                                         self._pixel_area,
                                         self._f_number)
        return radiance

    def get_photons(self, radiance):
        return routines.get_photons(self.exposure,
                                    self._wavelength,
                                    radiance,
                                    self._pixel_area,
                                    self._f_number)

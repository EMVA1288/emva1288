import numpy as np


def nearest_value(value, array):
    """Returns the nearest value in vals.

    Parameters
    ----------
    value : float
            The value we want to get as near as possible.
    array : array_like
            The array containing the available values to get near the value.

    Returns
    -------
    The nearest element of `array` to `value`.
    """
    # http://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def get_photons(exposure, radiance, pixel_area, wavelength, f_number):
    """Get the number of photons hitting one pixel.

    Parameters
    ----------
    exposure : float
               The pixel exposure time (in ns) to the light.
    radiance : float or array_like
               The radiance hitting the sensor (in W/sr/cm^2).
    pixel_area : float
                 The pixel area in um ^ 2
    f_number : float
               The f number of the setup.

    Returns
    -------
    float :
        The number of photons that hit the pixel.
    """
    h = 6.63e-34
    c = 3.00e8
    w = wavelength * 1e-9
    t = exposure * 1e-9
    a = pixel_area * 1e-12
    # Actual mathematics:-
    # irradiance = radiance * ((np.pi / (1 + ((2 * f_number) ** 2))) * w)
    # photons = irradiance * a * t * w / (h * c)
    # To optimize the calculation time first all the scalars are seperately
    # calculated
    # The matrix (w and radiance) multiplication order is the same.
    scalar = ((a * t) * np.pi / ((h * c) * (1 + ((2 * f_number) ** 2))))
    return radiance * (scalar * (w * w))


def get_radiance(exposure, photons, pixel_area, f_number, wavelength):
    """From the number of photons, get the radiance hitting a pixel.

    Parameters
    ----------
    exposure : float
               The pixel exposure time to the light (in ns).
    wavelength : float
                 The photons' wavelength (in nm).
    photons : float
              The number of photons that hit the pixel.
    pixel_area : float
                 The pixel area in um^2.
    f_number : float
               The f number of the setup.

    Returns
    -------
    float :
        The radiance that hit the pixel and gave the number of photons.
    """
    h = 6.63e-34
    c = 3.00e8
    w = wavelength * 1e-9
    t = exposure * 1e-9
    a = pixel_area * 1e-12

    #  p = j * a * t * w / (h * c)
    # j = np.pi * radiance / (1 + ((2 * f) ** 2)) * delta_w

    # Actual mathematics:-
    # irradiance = photons * h * c / (a * t * w)
    # radiance = irradiance * (1 + ((2 * f_number) ** 2)) / np.pi / w
    # To optimize the calculation time first all the scalars are seperately
    # calculated
    # The matrix (w and photon) multiplication order is the same.
    scalar = ((h * c) * (1 + ((2 * f_number) ** 2)) / (np.pi * a * t))
    return photons * (scalar / (w * w))


def get_tile(arr, height, width):
    """From an array with a pattern, repeat the array to fill an 2D array
       with the given dimention.

    Parameters
    ----------
    arr : array
          The pattern wanted to be replecate in a bigger shape.
    height : int
             The height of the array wanted in return.
    width : int
            The width of the array wanted in return.

    Returns
    -------
    array :
           The pattern given(arr) replecate in the size given.
    """

    #  To reduce the execution time, we will reduce the width an height of
    #  arguments given in the np.tile function. We take the width and height
    #  wanted, dived by size of the array dimention. By careful, for an 2D
    #  array, the shape[1] is the width and the shape[0] is the height, but
    #  for an 1D array the height is 1 so the shape[0] is the width.
    #  The size need to be a int, so we use np.floor to arroud the division
    #  and add one to be sure than the size is not to short. To be sure to
    #  return the array with the good shape we use [:height, :width].

    # if the array given is an 1D array.
    if len(arr.shape) == 1:
        # the height suppose to be 1
        h_r = height
        # only the width change
        w_r = int(np.floor(width / arr.shape[0]) + 1)

    # if the array given is a 2D array.
    if len(arr.shape) == 2:
        # number of repetition needed
        h_r = int(np.floor(height / arr.shape[0]) + 1)
        w_r = int(np.floor(width / arr.shape[1]) + 1)

    # the case if the array given is an multidimentinal array.
    # else:
        # TODO: make a error message for to many dimention.

    # create a array with the dimension given and the array given
    tile = np.tile(arr, (h_r, w_r))[:height, :width]

    # if we want an 1D array so with a shape of (width,) and not (1, width)
    if height == 1:
        # tile = array([[0,0,0,...]]) so we only want the first line.
        return tile[0]
    return tile


def get_bayer_filter(t00, t01, t10, t11, width, height, w):
    """From different values of transmition and the size, get a bayer filter.

    +-----------+-----------+-----------+-----------+-----------+-----------+
    |              Pattern                                                  |
    +===========+===========+===========+===========+===========+===========+
    |    This case          |     Example           | Suggested values      |
    +-----------+-----------+-----------+-----------+-----------+-----------+
    |   t00     |  t01      |   G1      |   R       |    1      |  0.15     |
    +-----------+-----------+-----------+-----------+-----------+-----------+
    |   t10     |  t11      |   B       |   G2      |  0.02     |   1       |
    +-----------+-----------+-----------+-----------+-----------+-----------+

    Where *G1* and *G2* are transmition values for a green filter, *R* for a
    red filter and *B* for a blue filter. The suggested values are values
    for a standard bayer filter like the example.

    Parameters
    ----------
    t00 : int
               The wavelength of the first pixel.
    t01 : int
               The wavelength of the second pixel.
    t10 : int
               The wavelength of the third pixel.
    t11 : int
               The wavelength of the fourth pixel.
    width : int
            The number of columns in the the image.
    height : int
            The number of rows in the image.
    w : array_like
        Array of the wavelengths to be returned

    Returns
    -------
    array :
        The array with the bayer filter of the size gived.
    """
    t00 = poisson(w, int(t00))
    t01 = poisson(w, int(t01))
    t10 = poisson(w, int(t10))
    t11 = poisson(w, int(t11))
    pattern_rep = np.array([[t00/np.max(t00), t01/np.max(t01)],
                            [t10/np.max(t10), t11/np.max(t11)]])
    b_filter = array_tile(pattern_rep, height, width)
    return b_filter


def array_tile(array, height, width):
    h, w = ((array.shape[:2]) if array.ndim == 3 else (1, array.shape[-1])
            if array.ndim == 2 else (1, 1))
    return np.tile(array, (int(height/h), int(width/w), 1))


def poisson(x, loc, mu=1000):
    x = x.astype(int) + (mu - loc)
    ln = [v * np.log(mu) - mu - np.sum([np.log(i) for i in range(1, v)])
          for v in x]
    return np.exp(ln)


class Qe(object):

    def __init__(self,
                 qe=None,
                 wavelength=np.linspace(400, 800, 100),
                 width=640,
                 height=480,
                 filter=None):

        self._width = width
        self._height = height
        self._qe = qe
        self._w = wavelength

        if self._qe is None:
            self._qe = self.gen_qe()

        if self._qe.shape != (self._height, self._width, len(self._w)):
            raise ValueError('qe dimensions mismatch %s != %s' %
                             self._qe.shape,
                             (self._height, self._width, len(self._w)))

        if filter is not None:
            self._qe *= filter

    @property
    def w(self):
        return self._w

    @property
    def qe(self):
        return self._qe

    def gen_qe(self):
        """Simulate quantum efficiency for a specific wavelengths.

        Returns
        -------
        float :
            The simulated quantum efficiency.
        """
        # For the time being we just simulate a simple gaussian
        s = 0.5
        u = 0.0
        min_ = 350
        max_ = 800

        def gauss(w):
            w = -1 + (w) / (max_ - min_)
            qe = -.1 + np.exp((-(w - u) ** 2) /
                              (2 * s ** 2)) / (np.sqrt(np.pi * 2 * s ** 2))
            return np.max([qe, 0])
        qe_ = np.array([(gauss(w)) for w in self._w])
        qe = np.zeros((self._height, self._width, len(self._w)))
        qe[:, :] = qe_
        return qe

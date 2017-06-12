import numpy as np


def qe(wavelength):
    """Simulate quantum efficiency for a specific wavelengths.

    Parameters
    ----------
    wavelength : float
                 The wavelength to compute the quantum efficency for.

    Returns
    -------
    float :
        The simulated quantum efficiency.
    """
    # For the time being we just simulate a simple gaussian
    s = 0.5
    u = 0.
    min_ = 350
    max_ = 800
    w = -1 + (wavelength) / (max_ - min_)
    qe = -.1 + np.exp((-(w - u) ** 2) /
                      (2 * s ** 2)) / (np.sqrt(np.pi * 2 * s ** 2))
    if qe < 0:
        return 0
    return qe


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


def get_irradiance(radiance, f):
    """Get The irradiance, in w/cm^2.

    Parameters
    ----------
    radiance : float
               The radiance (in W/sr/cm^2) to compute the irradiance from.
    f : float
        The f number of the setup.

    Returns
    -------
    float :
        The irradiance in W/cm^2
    """
    j = np.pi * radiance / (1 + ((2 * f) ** 2))
    return j


def get_photons(exposure, wavelength, radiance, pixel_area, f_number):
    """Get the number of photons hitting one pixel.

    Parameters
    ----------
    exposure : float
               The pixel exposure time (in ns) to the light.
    wavelength : float
                 The light wavelength hitting the pixel (in nm).
    radiance : float
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
    j = get_irradiance(radiance, f_number)
    return j * a * t * w / (h * c)


def get_radiance(exposure, wavelength, photons, pixel_area, f_number):
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
    # j = np.pi * radiance / (1 + ((2 * f) ** 2))

    j = photons * h * c / (a * t * w)
    r = j * (1 + ((2 * f_number) ** 2)) / np.pi
    return r


def get_bayer_filter(t00, t01, t10, t11, width, height):
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
    t00 : float
               The transmition, in pourcentage, of the first pixel.
    t01 : float
               The transmition, in pourcentage, of the second pixel.
    t10 : float
               The transmition, in pourcentage, of the third pixel.
    t11 : float
               The transmition, in pourcentage, of the fourth pixel.
    width : int
            The number of columns in the the image.
    height : int
            The number of rows in the image.

    Returns
    -------
    array :
        The array with the bayer filter of the size gived.
    """
    pattern_rep = np.array([[t00, t01], [t10, t11]])
    size = (int(np.floor(height / pattern_rep.shape[1]))+1,
            int(np.floor(width / pattern_rep.shape[0]))+1)
    b_filter = np.tile(pattern_rep, size)[:height, :width]
    return b_filter

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

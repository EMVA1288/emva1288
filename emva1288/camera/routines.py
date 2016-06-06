import numpy as np


def qe(wavelength):
    """Simulate quantum efficiency for different wavelengths"""
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
    """Return the nearest value in vals"""
    # http://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def get_irradiance(radiance, f):
    # Get The irradiance, in w/cm^2
    j = np.pi * radiance / (1 + ((2 * f) ** 2))
    return j


def get_photons(exposure, wavelength, radiance, pixel_area, f_number):
        h = 6.63e-34
        c = 3.00e8

        w = wavelength * 1e-9
        t = exposure * 1e-9
        a = pixel_area * 1e-12

#        j = math.pi * radiance / (1 + ((2 * f) ** 2))
        j = get_irradiance(radiance, f_number)

        return j * a * t * w / (h * c)


def get_radiance(exposure, wavelength, photons, pixel_area, f_number):
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

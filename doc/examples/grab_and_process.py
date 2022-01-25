import numpy as np
from emva1288.camera import Camera
from emva1288.process.routines import get_int_imgs
from emva1288 import process


def get_emva_blackoffset(cam):
    """Find the blackoffset to satifsfy EMVA1288 requirements"""
    bini = cam.blackoffset
    # Find black offset with a maximum of 0.5% of values at Zero
    bo = cam.blackoffsets[0]
    pixels = cam.width * cam.height
    for i in cam.blackoffsets:
        cam.blackoffset = i
        img = cam.grab(0)
        bo = i
        if np.count_nonzero(img) > pixels * .995:
            break
    cam.blackoffset = bini
    return bo


def get_emva_gain(cam):
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


def get_temporal(cam, radiance):
    img1 = cam.grab(radiance)
    img2 = cam.grab(radiance)
    imgs = get_int_imgs((img1, img2))
    return {'sum': np.sum(imgs['sum']), 'pvar': np.sum(imgs['pvar'])}


def get_spatial(cam, radiance, L=50):
    imgs = []
    for i in range(L):
        imgs.append(cam.grab(radiance))
    return get_int_imgs(imgs)


data = {'temporal': {},
        'spatial': {},
        'width': None, 'height': None}


# Intialize the camera, here we can specify different image size
# or any other parameter that Camera allows
c = Camera(bit_depth=10,
           width=100,
           height=50)

# Fill the information
data['width'] = c.width
data['height'] = c.height

# Maximum exposure for test
exposure_max = 9000000

# Find the camera parameters for the test
c.exposure = exposure_max
c.blackoffset = get_emva_blackoffset(c)
c.K = get_emva_gain(c)

# Find the radiance that will saturate the camera at our maximum exposure time
saturation_radiance = c.get_radiance_for()

# Initialize the exposure for the spatial measure
exposure_spatial = None

# Loop through the exposures
for exposure in np.linspace(c.exposure_min, exposure_max, 100):
    c.exposure = exposure
    data['temporal'][exposure] = {}

    # For each exposure, take to measurements (bright, dark)
    for radiance in (saturation_radiance, 0.0):
        photons = c.get_photons(radiance)

        # Get the temporal data
        data['temporal'][exposure].setdefault(photons, {})
        data['temporal'][exposure][photons] = get_temporal(c, radiance)

        # Check if we are at the middle of the range, to set the spatial exp
        img = c.grab(radiance)
        if not exposure_spatial and (img.mean() > c.img_max / 2.):
            exposure_spatial = exposure

        # Get the spatial data
        if exposure_spatial == exposure:
            data['spatial'].setdefault(exposure, {})
            data['spatial'][exposure].setdefault(photons, {})
            data['spatial'][exposure][photons] = get_spatial(c, radiance)


# Process the collected data
dat = process.Data1288(data)
res = process.Results1288(dat.data, pixel_area=c.pixel_area)
res.print_results()
plot = process.Plotting1288(res)
plot.plot()

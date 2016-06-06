import numpy as np
from camera1288 import Camera

c = Camera()

radiances = np.linspace(0, c.saturation_radiance, num=20)
for radiance in radiances:
    c.radiance = radiance
    img = c.grab()
    print(img.mean(), img.std())

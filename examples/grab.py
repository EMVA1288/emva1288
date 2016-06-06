import numpy as np
from emva1288.camera import Camera

c = Camera()

radiances = np.linspace(0, c.get_radiance_for(mean=200), num=20)
for radiance in radiances:
    c.set_radiance(radiance)
    img = c.grab()
    print(img.mean(), img.std())

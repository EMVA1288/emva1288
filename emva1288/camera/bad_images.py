import numpy as np


def bad_pixel(cam, num_bad_pixel, bad_value=1):
    """Generation of bad pixel in a camera.

    Parameters
    ----------
    cam: Object
         The camera object who influence the simulation of
         bad pixel in the image.
    num_bad_pixel: int
                    Number of bad pixels desired in the image.
    bad_value: float, optional
               Pourcentage, 0 to 1, of the satured e^- value for each
               bad_pixel in the the camera.
               If None, the value will be the Full well capacity.

    """
    # TODO: add error message if num_bad_pixel is higger than pixel_num
    # Set the bad value at the Full well capacity for the camera if 1.
    bad_value = bad_value * cam._u_esat
    # Count the number of pixel in the camera
    pixel_num = cam.height * cam.width
    # Select the number of bad pixel in a 1D with number
    # for they futur position.
    bad_pixel_position = np.random.choice(pixel_num, num_bad_pixel)
    # change the position in 1D to 2D.
    for i in bad_pixel_position:
        # Change the DSNU of the camera with the bad values
        # at the random position generated above.
        cam.DSNU[i // cam.width, i % cam.width] = bad_value

def cart2pol(x, y):
    """
    Cartesian coordinates to polar coordinates

    Parameters
    ----------
    x: floats or integers of horizontal cartesian axis
    y: floats or integers of vertical cartesian axis


    Return
    ----------
    rho: radius of coordinates
    phi: angle of coordinates in radii  [0-2*pi]

    """
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return (rho, phi)


def pol2cart(rho, phi):
    """
    Polar coordinates (numpy array) to cartesian coordinates

    Parameters
    ----------
    rho: radius of coordinates
    phi: angle of coordinates in radii [0-2*pi]


    Return
    ----------
    x: floats or integers of horizontal cartesian axis
    y: floats or integers of vertical cartesian axis

    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)

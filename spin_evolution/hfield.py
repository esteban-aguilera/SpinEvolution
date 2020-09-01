import numpy as np


# --------------------------------------------------------------------------------
# functions
# --------------------------------------------------------------------------------
def cos(xarr, k=2*np.pi, Bx=0, Bz=1):
    """Sinusoidal magnetic field for each site of a unidimensional lattice.

    Parameters
    ----------
    xarr: np.ndarray(?)
        Unidimensional array with the positions of each site.

    k: float
        Wave number.  If the period is L, then k = 2*pi/L.
    
    Bx: float
        Magnetic field in the x-axis.

    Bz: float
        Magnetic field in the z-axis.

    Returns
    -------
    harr: np.ndarray(?, 3)
        Magnetic field in each of each site site.

    dharr = np.ndarray(?, 3, 3)
        Magnetic field gradient of each site.
    """
    assert(len(xarr.shape) == 1)

    num = xarr.shape[0]

    harr = np.zeros((num, 3))
    dharr = np.zeros((num, 3, 3))

    harr[:,0] = Bx * np.cos(k * xarr)
    harr[:,2] = Bz

    dharr[:,0,0] = -k * Bx * np.sin(k * xarr)

    return harr, dharr


def sin(xarr, k=2*np.pi, Bx=0, Bz=1):
    """Sinusoidal magnetic field for each site of a unidimensional lattice.

    Parameters
    ----------
    xarr: np.ndarray(?)
        Unidimensional array with the positions of each site.

    k: float
        Wave number.  If the period is L, then k = 2*pi/L.
    
    Bx: float
        Magnetic field in the x-axis.

    Bz: float
        Magnetic field in the z-axis.

    Returns
    -------
    harr: np.ndarray(?, 3)
        Magnetic field in each of each site site.

    dharr = np.ndarray(?, 3, 3)
        Magnetic field gradient of each site.
    """
    assert(len(xarr.shape) == 1)

    num = xarr.shape[0]

    harr = np.zeros((num, 3))
    dharr = np.zeros((num, 3, 3))

    harr[:,0] = Bx * np.sin(k * xarr)
    harr[:,2] = Bz

    dharr[:,0,0] = k * Bx * np.cos(k * xarr)

    return harr, dharr

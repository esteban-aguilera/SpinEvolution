"""Python function made to interact easily with class SpinLattice, that is
programed in Cython.
"""

import numpy as np

# package imports
from spin_evolution import SpinLattice


# --------------------------------------------------------------------------------
# functions
# --------------------------------------------------------------------------------
def unidimensional_lattice(num=2, m=1, S=1, a=1, Phi=0, J=1, alpha=1e-5,
                           gamma=1e-7, dt=1e-3, icond='rand'):
    """Returns a unidimensional lattice.

    Parameters
    ----------
    num: int, optional
        number of sites in the unidimensional lattice.

    m: float, optional
        mass of the sites.  Each site has the same mass.

    S: float, optional
        Spin of each site.  Each site has the same spin.

    a: float, optional
        Nearest-neighbor distance.

    Phi: float or np.ndarray(3, 3), optional
        Elasticity matrix.  If it is a number, the matrix is assumed
        to be diagonal.

    J: float, optional
        Heisenberg exchange.

    alpha: float, optional
        Gilbert damping coefficient.

    gamma: float, optional
        Viscous damping coefficient

    dt: float, optional
        time difference between each time step

    icond: str, optional
        Type of initial condition.  Can be 'rand' or 'const'
    
    Returns
    -------
    lat: spin_evolution.SpinLattice
        Unidimensional lattice created with the given parameters.
    """
    # add two sites for the periodic boundary conditions
    num = num + 2

    # each site will have 2 nearest neighbors
    num_nn = 2

    if((type(Phi) in [int, float]) and Phi == 0):
        Phi = np.zeros((3, 3))
    
    # initialize matrices
    m_arr = np.zeros(num)  # mass array
    r_arr = np.zeros((num,3))  # position array
    dr_arr = np.zeros((num,3))  # displacement array
    v_arr = np.zeros((num,3))  # velocity array
    M_arr = np.zeros((num,3))  # mass array
    idto = -np.ones(num, dtype=np.int32)  # identical to array

    alpha_arr = alpha * np.ones(num)  # Gilbert damping array
    gamma_arr = gamma * np.ones(num)  # Viscous damping array

    nn_arr = -np.ones((num,num_nn), dtype=np.int32)  # nearest neighbors array

    B_arr = np.zeros((num,3))  # magnetic field array
    J_arr = J * np.ones((num,num_nn))  # Heisenberg exchange array
    Phi_arr = np.zeros((num,num_nn,3,3))  # Elastic coupling array

    m_arr[:] = m
    r_arr[:,0] = a * np.arange(-1,num-1)
    dr_arr[:,0] = 0.2*a * (np.random.rand(num)-0.5)
    
    if(icond == 'rand'):
        # random spin initial condition
        theta, phi = np.pi*np.random.rand(num), 2*np.pi*np.random.rand(num)
        M_arr[:,0] = S*np.sin(theta)*np.cos(phi)
        M_arr[:,1] = S*np.sin(theta)*np.sin(phi)
        M_arr[:,2] = S*np.cos(theta)
    elif(icond == 'const'):
        # constant spin initial condition
        M_arr[:,2] = S
    else:
        raise Exception('Invalid icond')
    
    # set periodic boundary conditions
    idto[0] = num-2
    idto[num-1] = 1
    
    # set magnetic field, nearest neighbors and Elastic coupling.
    B_arr[:,2] = 1
    for n in range(1,num-1):
        nn_arr[n,:] = np.array([n-1, n+1])
        for l in range(num_nn):
            Phi_arr[n,l,:,:] = Phi[:,:]
    
    # pass everything to create the SpinLattice object.
    lat = SpinLattice(m_arr=m_arr, r_arr=r_arr, dr_arr=dr_arr, M_arr=M_arr, idto=idto,
                      alpha_arr=alpha_arr, gamma_arr=gamma_arr, B_arr=B_arr,
                      nn_arr=nn_arr, J_arr=J_arr, Phi_arr=Phi_arr,
                      dt=dt)

    return lat

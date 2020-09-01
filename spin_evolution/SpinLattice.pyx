import matplotlib.pyplot as plt
import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, free

from .utils cimport linalg3D


# --------------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------------
cdef double hbar = 6.582e-13  # Plank's reduced constant
cdef double muB = 5.788e-2  # Bohr's magneton
cdef double g = 2  # g-factor


# --------------------------------------------------------------------------------
# class
# --------------------------------------------------------------------------------
cdef class SpinLattice:

    # constants
    cdef double **delta
    cdef double ***L

    # general properties
    cdef int memory_allocated
    cdef double t
    cdef double dt

    # properties of each site
    cdef int num
    cdef double *m_arr  # mass
    cdef double *q_arr  # charge
    cdef double **r_arr  # position
    cdef double **dr_arr  # position
    cdef double **v_arr  # position
    cdef double **M_arr  # magnetization
    cdef int *idto  # identical tu

    cdef double *alpha_arr  # magnetization damping
    cdef double *gamma_arr  # kinetic damping

    # lattice properties
    cdef int num_nn
    cdef int **nn_arr  # nearest neighbors

    cdef double **f_arr  # external magnetic field
    cdef double ****Phi_arr  # elastic interaction

    cdef double **B_arr  # external magnetic field
    cdef double **h_arr  # effective magnetic field
    cdef double ***dB_arr  # external magnetic field gradient
    cdef double *K_arr  # anisotropy
    cdef double **J_arr  # Heisenberg exchange
    cdef double ***D_arr  # DMI interaction

    def __cinit__(self, **kwargs):
        cdef int i=0, j=0, k=0

        cdef double[:] m_arr
        cdef double[:] q_arr
        cdef double[:,:] r_arr
        cdef double[:,:] dr_arr
        cdef double[:,:] v_arr
        cdef double[:,:] M_arr
        cdef int[:] idto
        cdef double[:] alpha_arr
        cdef double[:] gamma_arr
        cdef int[:,:] nn_arr
        cdef double[:,:] B_arr
        cdef double[:,:,:] dB_arr
        cdef double[:] K_arr
        cdef double[:,:] J_arr
        cdef double[:,:,:] D_arr
        cdef double[:,:,:,:] Phi_arr

        self.t = kwargs.pop('t', 0)
        self.dt = kwargs.pop('dt', 1e-5)
        for key in ['t', 'dt']:
            if(key in kwargs):
                kwargs.pop(key)

        self.num = 10
        self.num_nn = 2
        for key, value in kwargs.items():
            if(key in ['m_arr', 'q_arr', 'r_arr', 'dr_arr', 'v_arr', 'M_arr',
                       'idto', 'alpha_arr', 'gamma_arr', 'nn_arr', 'B_arr',
                       'dB_arr', 'K_arr', 'J_arr', 'D_arr', 'Phi_arr']):
                self.num = value.shape[0]
            if(key in ['J_arr', 'D_arr', 'Phi_arr']):
                self.num_nn = value.shape[1]

        m_arr = kwargs.pop('m_arr', np.zeros(self.num))
        q_arr = kwargs.pop('q_arr', np.zeros(self.num))
        r_arr = kwargs.pop('r_arr', np.zeros([self.num,3]))
        dr_arr = kwargs.pop('dr_arr', np.zeros([self.num,3]))
        v_arr = kwargs.pop('v_arr', np.zeros([self.num,3]))
        M_arr = kwargs.pop('M_arr', np.zeros([self.num,3]))
        idto = kwargs.pop('idto', -np.ones(self.num, dtype=np.int32))
        alpha_arr = kwargs.pop('alpha_arr', np.zeros(self.num))
        gamma_arr = kwargs.pop('gamma_arr', np.zeros(self.num))
        nn_arr = kwargs.pop('nn_arr', -np.ones([self.num,self.num_nn], dtype=np.int32))
        B_arr = kwargs.pop('B_arr', np.zeros([self.num,3]))
        dB_arr = kwargs.pop('dB_arr', np.zeros([self.num,3,3]))
        K_arr = kwargs.pop('K_arr', np.zeros(self.num))
        J_arr = kwargs.pop('J_arr', np.zeros([self.num,self.num_nn]))
        D_arr = kwargs.pop('D_arr', np.zeros([self.num,self.num_nn,3]))
        Phi_arr = kwargs.pop('Phi_arr', np.zeros([self.num,self.num_nn,3,3]))

        if(len(kwargs) > 0):
            print(f'{list(kwargs)} is (are) invalid argument(s).')

        # Angular momentum matrix
        self.L = <double ***> malloc(3 * sizeof(double **))
        for i in range(3):
            self.L[i] = <double **> malloc(3 * sizeof(double *))
            for j in range(3):
                self.L[i][j] = <double *> malloc(3 * sizeof(double))
                for k in range(3):
                    self.L[i][j][k] = 0
        self.L[0][1][2], self.L[0][2][1] = -1., 1.
        self.L[1][0][2], self.L[1][2][0] = 1., -1.
        self.L[2][0][1], self.L[2][1][0] = -1., 1.

        # Delta Kronecker
        self.delta = <double **> malloc(3 * sizeof(double *))
        for i in range(3):
            self.delta[i] = <double *> malloc(3 * sizeof(double))
            for j in range(3):
                if(i == j):
                    self.delta[i][j] = 1
                else:
                    self.delta[i][j] = 0

        # setting sites
        self.set_m(m_arr)
        self.set_q(q_arr)
        self.set_r(r_arr)
        self.set_dr(dr_arr)
        self.set_v(v_arr)
        self.set_M(M_arr)

        self.set_alpha(alpha_arr)
        self.set_gamma(gamma_arr)

        self.set_identicals(idto)

        # setting nearest neighbors
        self.set_nn(nn_arr)

        # setting interactions
        self.set_B(B_arr)
        self.set_dB(dB_arr)
        self.set_K(K_arr)
        self.set_J(J_arr)
        self.set_D(D_arr)
        self.set_Phi(Phi_arr)

        self.f_arr = <double **> malloc( self.num*sizeof(double *) )
        self.h_arr = <double **> malloc( self.num*sizeof(double *) )
        for n in range(self.num):
            self.f_arr[n] = <double *> malloc( 3*sizeof(double) )
            self.h_arr[n] = <double *> malloc( 3*sizeof(double) )

        self.update_identicals()

        self.update_harr()
        self.update_farr()

    def __init__(self, *args, **kwargs):
        self.memory_allocated = 1

    def __dealloc__(self):
        cdef int i=0, j=0, k=0, l=0, n=0

        for i in range(3):
            free(self.delta[i])
            for j in range(3):
                free(self.L[i][j])
            free(self.L[i])
        free(self.delta)
        free(self.L)

        # deleting properties of each site
        free(self.m_arr)
        free(self.q_arr)
        free(self.idto)
        free(self.alpha_arr)
        free(self.gamma_arr)
        for n in range(self.num):
            free(self.r_arr[n])
            free(self.dr_arr[n])
            free(self.v_arr[n])
            free(self.M_arr[n])
        free(self.r_arr)
        free(self.dr_arr)
        free(self.v_arr)
        free(self.M_arr)

        # deleting lattice properties
        free(self.K_arr)
        for n in range(self.num):
            free(self.nn_arr[n])
            free(self.f_arr[n])
            free(self.B_arr[n])
            free(self.h_arr[n])
            free(self.J_arr[n])
            for l in range(self.num_nn):
                free(self.D_arr[n][l])
                for k in range(3):
                    free(self.Phi_arr[n][l][k])
                free(self.Phi_arr[n][l])
            free(self.D_arr[n])
            free(self.Phi_arr[n])
            for k in range(3):
                free(self.dB_arr[n][k])
            free(self.dB_arr[n])

        free(self.nn_arr)
        free(self.f_arr)
        free(self.B_arr)
        free(self.h_arr)
        free(self.J_arr)

        free(self.D_arr)
        free(self.Phi_arr)

        free(self.dB_arr)

    # ----------------------------------------------------------------------------
    # functions
    # ----------------------------------------------------------------------------
    cpdef void time_step(self):
        self.time_step_c()

    # ----------------------------------------------------------------------------
    # C functions
    # ----------------------------------------------------------------------------
    cdef void time_step_c(self):
        cdef int i=0, j=0, k=0, n=0
        cdef double Lh_ij=0
        cdef double *h = <double *> malloc(3*sizeof(double))
        cdef double *b = <double *> malloc(3*sizeof(double))
        cdef double **A = <double **> malloc(3*sizeof(double *))
        cdef double **A_inv = <double **> malloc(3*sizeof(double *))

        for i in range(3):
            A[i] = <double *> malloc(3*sizeof(double))
            A_inv[i] = <double *> malloc(3*sizeof(double))

        self.update_harr()
        self.update_farr()
        for n in range(self.num):
            if(self.idto[n] == -1):
                linalg3D.cross(self.M_arr[n], self.h_arr[n], h)
                for k in range(3):
                    h[k] = self.h_arr[n][k] + self.alpha_arr[n]*h[k]
                    
                    # displacement time step.
                    self.dr_arr[n][k] += self.v_arr[n][k]*self.dt
                    self.v_arr[n][k] += self.f_arr[n][k]*self.dt/self.m_arr[n]
                
                # spin time step.
                for i in range(3):
                    b[i] = 0
                    for j in range(3):
                        Lh_ij = self.L[0][i][j]*h[0] + self.L[1][i][j]*h[1] + \
                            self.L[2][i][j]*h[2]
                        A[i][j] = self.delta[i][j] - 0.5*self.dt*Lh_ij/hbar
                        b[i] += (self.delta[i][j] + 0.5*self.dt*Lh_ij/hbar)*self.M_arr[n][j]

                linalg3D.inverse(A, A_inv)
                for i in range(3):
                    self.M_arr[n][i] = A_inv[i][0]*b[0] + A_inv[i][1]*b[1] + \
                        A_inv[i][2]*b[2]
            else:
                pass  # identicals are updated later.

        self.update_identicals()
        self.t = self.t + self.dt

        free(h)
        free(b)
        for n in range(3):
            free(A[n])
            free(A_inv[n])
        free(A_inv)
        free(A)

    cdef void update_harr(self):
        cdef int n=0, j=0, l=0

        for n in range(self.num):
            self.h_arr[n][0] = muB*g*self.B_arr[n][0]
            self.h_arr[n][1] = muB*g*self.B_arr[n][1]
            self.h_arr[n][2] = muB*g*self.B_arr[n][2] + 2*self.K_arr[n]*self.M_arr[n][2]

            for l in range(self.num_nn):
                j = self.nn_arr[n][l]
                if(j != -1):
                    self.h_arr[n][0] += self.J_arr[n][l]*self.M_arr[j][0] + \
                        self.D_arr[n][l][1]*self.M_arr[j][2] - \
                        self.D_arr[n][l][2]*self.M_arr[j][1]
                    self.h_arr[n][1] += self.J_arr[n][l]*self.M_arr[j][1] + \
                        self.D_arr[n][l][2]*self.M_arr[j][0] - \
                        self.D_arr[n][l][0]*self.M_arr[j][2]
                    self.h_arr[n][2] += self.J_arr[n][l]*self.M_arr[j][2] + \
                        self.D_arr[n][l][0]*self.M_arr[j][1] - \
                        self.D_arr[n][l][1]*self.M_arr[j][0]

    cdef void update_farr(self):
        cdef int n=0, j=0, l=0, alpha=0, beta=0

        for n in range(1, self.num-1):
            for alpha in range(3):
                self.f_arr[n][alpha] = -self.gamma_arr[n]*self.v_arr[n][alpha]
                for l in range(self.num_nn):
                    j = self.nn_arr[n][l]
                    if(j != -1):
                        for beta in range(3):
                            self.f_arr[n][alpha] += -self.Phi_arr[n][l][alpha][beta] * \
                            (self.r_arr[n][beta] + self.dr_arr[n][beta] -
                             self.r_arr[j][beta] - self.dr_arr[j][beta])
                for beta in range(3):
                    self.f_arr[n][alpha] += muB*g*self.dB_arr[n][beta][alpha]*self.M_arr[n][beta]

    cdef void update_identicals(self):
        cdef int j=0, n=0

        for n in range(self.num):
            j = self.idto[n]
            if(j != -1):
                for k in range(3):
                    self.dr_arr[n][k] = self.dr_arr[j][k]
                    self.v_arr[n][k] = self.v_arr[j][k]
                    self.M_arr[n][k] = self.M_arr[j][k]

    # ----------------------------------------------------------------------------
    # plotters
    # ----------------------------------------------------------------------------
    def plot(self, *args, **kwargs):

        a = kwargs.get('a', 1)
        S = kwargs.get('S', 1)
        data = kwargs.get('data', [])
        show_h = kwargs.get('show_h', False)
        show_f = kwargs.get('show_f', False)
        title = kwargs.get('title', '')
        xlim = kwargs.get('xlim', None)
        show_grid = kwargs.get('show_grid', False)
        show = kwargs.get('show', False)
        fn = kwargs.get('filename', 'SpinLattice')

        r_arr = (np.array(self.get_r()) + np.array(self.get_dr()))/a
        M_arr = np.array(self.get_M())

        B_arr = np.array(self.get_B())
        f_arr = np.array(self.get_farr())

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.scatter(r_arr[1:-1,0], r_arr[1:-1,2], color='black')
        ax.scatter([r_arr[0,0],r_arr[-1,0]], [r_arr[0,2],r_arr[-1,2]], color='orange')
        ax.quiver(r_arr[:,0], r_arr[:,2], M_arr[:,0], M_arr[:,2], scale=S,
                  color='darkblue')

        for x, y in data:
            ax.plot(x, y)

        if(show_h is True):
            ax.quiver(r_arr[:,0], r_arr[:,2], B_arr[:,0], B_arr[:,2], scale=S,
                      color='yellow')
        if(show_f is True):
            ax.quiver(r_arr[:,0], r_arr[:,2], f_arr[:,0], f_arr[:,2], scale=S,
                      color='green')

        ax.set_title(title)
        if(xlim is not None):
            ax.set_xlim(xlim)
        if(show_grid is True):
            ax.grid()

        fig.tight_layout()

        if(show is True):
            plt.show()
        fig.savefig(fn)

        plt.close(fig)

    # ----------------------------------------------------------------------------
    # setters
    # ----------------------------------------------------------------------------
    cpdef void set_m(self, double[:] m_arr):
        cdef int n=0

        if(self.memory_allocated == 0):
            self.m_arr = <double *> malloc( self.num*sizeof(double) )
        elif(m_arr.shape[0] != self.num):
            raise ValueError('length of m_arr[:] (%d) does not match number of'
                             ' sites (%d)' % (m_arr.shape[0], self.num))

        for n in range(self.num):
            self.m_arr[n] = m_arr[n]

    cpdef void set_q(self, double[:] q_arr):
        cdef int n=0

        if(self.memory_allocated == 0):
            self.q_arr = <double *> malloc( self.num*sizeof(double) )
        elif(q_arr.shape[0] != self.num):
            raise ValueError('length of q_arr[:] (%d) does not match number of'
                             ' sites (%d)' % (q_arr.shape[0], self.num))

        for n in range(self.num):
            self.q_arr[n] = q_arr[n]

    cpdef void set_r(self, double[:,:] r_arr):
        cdef int n=0, k=0

        if(self.memory_allocated == 0):
            self.r_arr = <double **> malloc( self.num*sizeof(double *) )
            for n in range(self.num):
                self.r_arr[n] = <double *> malloc( 3*sizeof(double) )
        elif(r_arr.shape[0] != self.num):
            raise ValueError('length of r_arr[:,k] (%d) does not match number of'
                             ' sites (%d)' % (r_arr.shape[0], self.num))
        elif(r_arr.shape[1] != 3):
            raise ValueError('length of r_arr[n,:] (%d) is not three' % r_arr.shape[1])

        for n in range(self.num):
            for k in range(3):
                self.r_arr[n][k] = r_arr[n,k]

    cpdef void set_dr(self, double[:,:] dr_arr):
        cdef int n=0, k=0

        if(self.memory_allocated == 0):
            self.dr_arr = <double **> malloc( self.num*sizeof(double *) )
            for n in range(self.num):
                self.dr_arr[n] = <double *> malloc( 3*sizeof(double) )
        elif(dr_arr.shape[0] != self.num):
            raise ValueError('length of dr_arr[:,k] (%d) does not match number of'
                             ' sites (%d)' % (dr_arr.shape[0], self.num))
        elif(dr_arr.shape[1] != 3):
            raise ValueError('length of dr_arr[n,:] (%d) is not three' % dr_arr.shape[1])

        for n in range(self.num):
            for k in range(3):
                self.dr_arr[n][k] = dr_arr[n,k]

    cpdef void set_v(self, double[:,:] v_arr):
        cdef int n=0, k=0

        if(self.memory_allocated == 0):
            self.v_arr = <double **> malloc( self.num*sizeof(double *) )
            for n in range(self.num):
                self.v_arr[n] = <double *> malloc( 3*sizeof(double) )
        elif(v_arr.shape[0] != self.num):
            raise ValueError('length of v_arr[:,k] (%d) does not match number of'
                             ' sites (%d)' % (v_arr.shape[0], self.num))
        elif(v_arr.shape[1] != 3):
            raise ValueError('length of v_arr[n,:] (%d) is not three' % v_arr.shape[1])

        for n in range(self.num):
            for k in range(3):
                self.v_arr[n][k] = v_arr[n,k]

    cpdef void set_M(self, double[:,:] M_arr):
        cdef int n=0, k=0

        if(self.memory_allocated == 0):
            self.M_arr = <double **> malloc( self.num*sizeof(double *) )
            for n in range(self.num):
                self.M_arr[n] = <double *> malloc( 3*sizeof(double) )
        elif(M_arr.shape[0] != self.num):
            raise ValueError('length of M_arr[:,k] (%d) does not match number of sites (%d)'
                             % (M_arr.shape[0], self.num))
        elif(M_arr.shape[1] != 3):
            raise ValueError('length of M_arr[n,:] (%d) is not three' % M_arr.shape[1])

        for n in range(self.num):
            for k in range(3):
                self.M_arr[n][k] = M_arr[n,k]

    cpdef void set_alpha(self, double[:] alpha_arr):
        cdef int n=0

        if(self.memory_allocated == 0):
            self.alpha_arr = <double *> malloc( self.num*sizeof(double) )
        elif(alpha_arr.shape[0] != self.num):
            raise ValueError('length of alpha_arr[:] (%d) does not match number'
                             ' of sites (%d)' % (alpha_arr.shape[0], self.num))

        for n in range(self.num):
            self.alpha_arr[n] = alpha_arr[n]

    cpdef void set_gamma(self, double[:] gamma_arr):
        cdef int n=0

        if(self.memory_allocated == 0):
            self.gamma_arr = <double *> malloc( self.num*sizeof(double) )
        elif(gamma_arr.shape[0] != self.num):
            raise ValueError('length of gamma_arr[:] (%d) does not match number'
                             ' of sites (%d)' % (gamma_arr.shape[0], self.num))

        for n in range(self.num):
            self.gamma_arr[n] = gamma_arr[n]

    cpdef void set_identicals(self, int[:] idto):
        cdef int n=0

        if(self.memory_allocated == 0):
            self.idto = <int *> malloc( self.num*sizeof(int) )
        elif(idto.shape[0] != self.num):
            raise ValueError('length of idto[:] (%d) does not match number'
                             ' of sites (%d)' % (idto.shape[0], self.num))

        for n in range(self.num):
            self.idto[n] = idto[n]

    cpdef void set_nn(self, int[:,:] nn_arr):
        cdef int n=0, l=0

        if(self.memory_allocated == 0):
            self.nn_arr = <int **> malloc( self.num*sizeof(int *) )
            for n in range(self.num):
                self.nn_arr[n] = <int *> malloc( self.num_nn*sizeof(int) )
        elif(nn_arr.shape[0] != self.num):
            raise ValueError('length of nn_arr[:,k] (%d) does not match number of sites (%d)'
                             % (nn_arr.shape[0], self.num))
        elif(nn_arr.shape[1] != self.num_nn):
            raise ValueError('length of nn_arr[:,l] (%d) does not match number '
                             'of neighbors (%d)' % (nn_arr.shape[1], self.num_nn))

        for n in range(self.num):
            for l in range(self.num_nn):
                self.nn_arr[n][l] = nn_arr[n,l]

    cpdef void set_Phi(self, double[:,:,:,:] Phi_arr):
        cdef int n=0, l=0, alpha=0, beta=0

        if(self.memory_allocated == 0):
            self.Phi_arr = <double ****> malloc( self.num*sizeof(double ***) )
            for n in range(self.num):
                self.Phi_arr[n] = <double ***> malloc( self.num_nn*sizeof(double **) )
                for l in range(self.num_nn):
                    self.Phi_arr[n][l] = <double **> malloc( 3*sizeof(double *) )
                    for alpha in range(3):
                        self.Phi_arr[n][l][alpha] = <double *> malloc( 3*sizeof(double) )
        elif(Phi_arr.shape[0] != self.num):
            raise ValueError('length of Phi_arr[:,l,alpha,beta] (%d) does not match number of'
                             ' sites (%d)' % (Phi_arr.shape[0], self.num))
        elif(Phi_arr.shape[1] != self.num_nn):
            raise ValueError('length of Phi_arr[n,:,alpha,beta] (%d) does not match number'
                             ' of neighbors (%d)' % (Phi_arr.shape[1], self.num_nn))
        elif(Phi_arr.shape[2] != 3):
            raise ValueError('length of Phi_arr[n,l,:,beta] (%d) is not three' % Phi_arr.shape[2])
        elif(Phi_arr.shape[3] != 3):
            raise ValueError('length of Phi_arr[n,l,alpha,:] (%d) is not three' % Phi_arr.shape[3])

        for n in range(self.num):
            for l in range(self.num_nn):
                for alpha in range(3):
                    for beta in range(3):
                        self.Phi_arr[n][l][alpha][beta] = Phi_arr[n,l,alpha,beta]

    cpdef void set_B(self, double[:,:] B_arr):
        cdef int n=0, k=0

        if(self.memory_allocated == 0):
            self.B_arr = <double **> malloc( self.num*sizeof(double *) )
            for n in range(self.num):
                self.B_arr[n] = <double *> malloc( 3*sizeof(double) )
        elif(B_arr.shape[0] != self.num):
            raise ValueError('length of B_arr[:,k] (%d) does not match number of sites (%d)'
                             % (B_arr.shape[0], self.num))
        elif(B_arr.shape[1] != 3):
            raise ValueError('length of B_arr[n,:] (%d) is not three' % B_arr.shape[1])

        for n in range(self.num):
            for k in range(3):
                self.B_arr[n][k] = B_arr[n,k]

    cpdef void set_dB(self, double[:,:,:] dB_arr):
        cdef int n=0, alpha=0, beta=0

        if(self.memory_allocated == 0):
            self.dB_arr = <double ***> malloc( self.num*sizeof(double **) )
            for i in range(self.num):
                self.dB_arr[i] = <double **> malloc( 3*sizeof(double *) )
                for alpha in range(3):
                    self.dB_arr[i][alpha] = <double *> malloc( 3*sizeof(double) )
        elif(dB_arr.shape[0] != self.num):
            raise ValueError('length of dB_arr[:,alpha,beta] (%d) does not '
                             'match number of sites (%d)'
                             % (dB_arr.shape[0], self.num))
        elif(dB_arr.shape[1] != 3):
            raise ValueError('length of dB_arr[n,:,beta] (%d) is not three' % dB_arr.shape[1])
        elif(dB_arr.shape[2] != 3):
            raise ValueError('length of dB_arr[n,alpha,:] (%d) is not three' % dB_arr.shape[2])

        for i in range(self.num):
            for alpha in range(3):
                for beta in range(3):
                    self.dB_arr[i][alpha][beta] = dB_arr[i,alpha,beta]

    cpdef void set_K(self, double[:] K_arr):
        cdef int n=0

        if(self.memory_allocated == 0):
            self.K_arr = <double *> malloc( self.num*sizeof(double) )
        elif(K_arr.shape[0] != self.num):
            raise ValueError('length of K_arr[:] (%d) does not match number of sites (%d)'
                             % (K_arr.shape[0], self.num))

        for n in range(self.num):
            self.K_arr[n] = K_arr[n]

    cpdef void set_J(self, double[:,:] J_arr):
        cdef int n=0, l=0

        if(self.memory_allocated == 0):
            self.J_arr = <double **> malloc( self.num*sizeof(double *) )
            for n in range(self.num):
                self.J_arr[n] = <double *> malloc( self.num_nn*sizeof(double) )
        elif(J_arr.shape[0] != self.num):
            raise ValueError('length of B_arr[:,l] (%d) does not match number of sites (%d)'
                             % (J_arr.shape[0], self.num))
        elif(J_arr.shape[1] != self.num_nn):
            raise ValueError('length of J_arr[n,:] (%d) does not match number'
                             'of neighbors (%d)' % (J_arr.shape[1], self.num_nn))

        for n in range(self.num):
            for l in range(self.num_nn):
                self.J_arr[n][l] = J_arr[n,l]

    cpdef void set_D(self, double[:,:,:] D_arr):
        cdef int n=0, l=0, k=0

        if(self.memory_allocated == 0):
            self.D_arr = <double ***> malloc( self.num*sizeof(double **) )
            for n in range(self.num):
                self.D_arr[n] = <double **> malloc( self.num_nn*sizeof(double *) )
                for l in range(self.num_nn):
                    self.D_arr[n][l] = <double *> malloc( 3*sizeof(double) )
        elif(D_arr.shape[0] != self.num):
            raise ValueError('length of D_arr[:,l,k] (%d) does not match number'
                             ' of sites (%d)' % (D_arr.shape[0], self.num))
        elif(D_arr.shape[1] != self.num_nn):
            raise ValueError('length of D_arr[n,:,k] (%d) does not match number'
                             ' of neighbors (%d)' % (D_arr.shape[1], self.num_nn))
        elif(D_arr.shape[2] != 3):
            raise ValueError('length of D_arr[n,l,:] (%d) is not three' % D_arr.shape[2])

        for n in range(self.num):
            for l in range(self.num_nn):
                for k in range(3):
                    self.D_arr[n][l][k] = D_arr[n,l,k]


    # ----------------------------------------------------------------------------
    # getters
    # ----------------------------------------------------------------------------
    cpdef int get_num(self):
        return self.num

    cpdef int get_num_nn(self):
        return self.num_nn

    cpdef double[:] get_m(self):
        cdef double[:] m_arr = np.zeros(self.num)
        cdef int n=0

        for n in range(self.num):
            m_arr[n] = self.m_arr[n]

        return m_arr

    cpdef double[:] get_q(self):
        cdef double[:] q_arr = np.zeros(self.num)
        cdef int n=0

        for n in range(self.num):
            q_arr[n] = self.q_arr[n]

        return q_arr

    cpdef double[:,:] get_r(self):
        cdef double[:,:] r_arr = np.zeros((self.num, 3))
        cdef int n=0, k=0

        for n in range(self.num):
            for k in range(3):
                r_arr[n,k] = self.r_arr[n][k]

        return r_arr

    cpdef double[:,:] get_dr(self):
        cdef double[:,:] dr_arr = np.zeros((self.num, 3))
        cdef int n=0, k=0

        for n in range(self.num):
            for k in range(3):
                dr_arr[n,k] = self.dr_arr[n][k]

        return dr_arr

    cpdef double[:,:] get_v(self):
        cdef double[:,:] v_arr = np.zeros((self.num, 3))
        cdef int n=0, k=0

        for n in range(self.num):
            for k in range(3):
                v_arr[n,k] = self.v_arr[n][k]

        return v_arr

    cpdef double[:,:] get_M(self):
        cdef double[:,:] M_arr = np.zeros((self.num, 3))
        cdef int n=0, k=0

        for n in range(self.num):
            for k in range(3):
                M_arr[n,k] = self.M_arr[n][k]

        return M_arr

    cpdef double[:] get_alpha(self):
        cdef double[:] alpha_arr = np.zeros(self.num)
        cdef int n=0

        for n in range(self.num):
            alpha_arr[n] = self.alpha_arr[n]

        return alpha_arr

    cpdef double[:] get_gamma(self):
        cdef double[:] gamma_arr = np.zeros(self.num)
        cdef int n=0

        for n in range(self.num):
            gamma_arr[n] = self.gamma_arr[n]

        return gamma_arr

    cpdef int[:] get_idto(self):
        cdef int[:] idto = np.zeros(self.num, dtype=np.int32)
        cdef int n=0

        for n in range(self.num):
            idto[n] = self.idto[n]

        return idto

    cpdef int[:,:] get_nn(self):
        cdef int[:,:] nn_arr = np.zeros([self.num,self.num_nn], dtype=np.int32)
        cdef int n=0, l=0

        for n in range(self.num):
            for l in range(self.num_nn):
                nn_arr[n,l] = self.nn_arr[n][l]

        return nn_arr

    cpdef double[:,:,:,:] get_Phi(self):
        cdef double[:,:,:,:] Phi_arr = np.zeros((self.num, self.num_nn, 3, 3))
        cdef int n=0, l=0, alpha, beta

        for n in range(self.num):
            for l in range(self.num_nn):
                for alpha in range(3):
                    for beta in range(3):
                        Phi_arr[n,l,alpha,beta] = self.Phi_arr[n][l][alpha][beta]

        return Phi_arr

    cpdef double[:,:] get_h(self):
        cdef double[:,:] h_arr = np.zeros((self.num, 3))
        cdef int n=0, k=0

        for n in range(self.num):
            for k in range(3):
                h_arr[n,k] = self.h_arr[n][k]

        return h_arr

    cpdef double[:,:] get_B(self):
        cdef double[:,:] B_arr = np.zeros((self.num, 3))
        cdef int n=0, k=0

        for n in range(self.num):
            for k in range(3):
                B_arr[n,k] = self.B_arr[n][k]

        return B_arr

    cpdef double[:,:] get_farr(self):
        cdef double[:,:] f_arr = np.zeros((self.num, 3))
        cdef int n=0, k=0

        for n in range(self.num):
            for k in range(3):
                f_arr[n,k] = self.f_arr[n][k]

        return f_arr

    cpdef double[:,:,:] get_dB(self):
        cdef double[:,:,:] dB_arr = np.zeros((self.num, 3, 3))
        cdef int n=0, alpha=0, beta=0

        for n in range(self.num):
            for alpha in range(3):
                for beta in range(3):
                    dB_arr[n,alpha, beta] = self.dB_arr[n][alpha][beta]

        return dB_arr

    cpdef double[:] get_K(self):
        cdef double[:] K_arr = np.zeros(self.num)
        cdef int n=0

        for n in range(self.num):
            K_arr[n] = self.K_arr[n]

        return K_arr

    cpdef double[:,:] get_J(self):
        cdef double[:,:] J_arr = np.zeros([self.num,self.num_nn])
        cdef int n=0, l=0

        for n in range(self.num):
            for l in range(self.num_nn):
                J_arr[n,l] = self.J_arr[n][l]

        return J_arr

    cpdef double[:,:,:] get_D(self):
        cdef double[:,:,:] D_arr = np.zeros((self.num, self.num_nn, 3))
        cdef int n=0, l=0, k=0

        for n in range(self.num):
            for l in range(self.num_nn):
                for k in range(3):
                    D_arr[n,l,k] = self.D_arr[n][l][k]

        return D_arr

    cpdef double get_t(self):
        return self.t

    cpdef double get_dt(self):
        return self.dt

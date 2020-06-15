import numpy as np

def scal_matr_prod(A,B):
    return np.trace( np.dot(A, B) )


# defining lambda_x matrices
def GGM_matr(N):    

    N = 2 * N
    lam_x = []
    # symmetric
    for i in range(N):
        for j in range(N):
            ar = np.zeros(shape=(N, N))
            if ( j >= (N/2) and i < (N/2) ):
                ar[i,j] = ar[j,i] = 1
                lam_x.append(ar)

    # antisymmetric
    for i in range(N):
        for j in range(N):
            ar = np.zeros(shape=(N, N))
            if ( j >= (N/2) and i < (N/2) ):
                ar[i,j] = -1
                ar[j, i] = 1
                lam_x.append(1j*ar)
            
    lambda_x = np.array(lam_x, dtype = complex) # array of matrises lambda_x





    # defining lambda_z matrices
    lam_z = []

    # symmetric
    for i in range(N):
        for j in range(N):
            ar1 = np.zeros(shape=(N, N))
            if ( 0 <= i and i < j and j <= (N/2)-1):
                ar1[i,j] = ar1[j,i] = 1
                lam_z.append(ar1)
            if ( (N/2) <= i and i < j and j <= N-1):
                ar1[i,j] = ar1[j,i] = 1
                lam_z.append(ar1)

    # antisymmetric
    for i in range(N):
        for j in range(N):
            ar2 = np.zeros(shape=(N, N))
            if ( 0 <= i and i < j and j <= (N/2)-1):
                ar2[i,j] = -1
                ar2[j,i] = 1
                lam_z.append(1j*ar2)
            if ( (N/2) <= i and i < j and j <= N-1):
                ar2[i,j] = -1
                ar2[j,i] = 1
                lam_z.append(1j*ar2)

    # diagonal
    for j in range(N):
        if j < N-1:
            j = j + 1
            ar3 = np.zeros(shape=(N, N))
            for i in range(j):
                ar3[i,i] = 1 
                ar3[i+1,i+1] = - (i + 1)
            lam_z.append( ar3 * np.sqrt(2./((i + 1)*(i + 2))))

    lambda_z = np.array(lam_z, dtype = complex) # array of matrises lambda_z


    lx = len(lambda_x)
    lz = len(lambda_z)        


    lam = np.concatenate((lambda_x, lambda_z), axis=0) # joint array of both lambda_x and lambda_z matrices
    
    n_lam = lam.shape[0]
    # n_lam - number of all GGM matrices lambda_x and lambda_z
    # first n_x = lambda_x.shape[0] elements in lam correspond to lambda_x matrices,
    # while next n_z = lambda_z.shape[1] elements in lam correspond to lambda_z matrices

    f = np.empty( [n_lam, n_lam, n_lam], dtype = complex )
    g = np.empty( [n_lam, n_lam, n_lam], dtype = complex )


    for i in range(n_lam):
        for j in range(n_lam):
            for k in range(n_lam):
                com = np.dot(lam[i], lam[j]) - np.dot(lam[j], lam[i]) 
                anticom = np.dot(lam[i], lam[j]) + np.dot(lam[j], lam[i]) 
                f[i,j,k] = - 0.25 * 1j * scal_matr_prod(com, lam[k])
                g[i,j,k] = 0.25 * scal_matr_prod(anticom, lam[k])
    
    
    
    
    return lambda_x, lambda_z, lx, lz, lam, f, g


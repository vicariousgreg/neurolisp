import itertools as it
import numpy as np
from gnetwork.hadamard_base_matrices import H_20, H_12, H_4

def give_new_number(n):
    if n < 4:
        return (n,4)
    if multiple(n):
        result = calculate_half(n)
        if result == -1:
            n = get_next_multiple(n + 1)
            return give_new_number(n)

    else:
        n = get_next_multiple(n)
        return give_new_number(n)

    return (n,result)

def multiple(n):
    if n % 4 == 0:
        return True
    else:
        return False

def get_next_multiple(n):
    if n % 4 == 0:
        return n
    else:
        return get_next_multiple(n+1)

def calculate_half(n):
    r = 0;
    q = n;

    while q > 1 and r == 0:
        if q == 20:
            return 20
        elif q == 12:
            return 12
        elif q == 4:
            return 4;
        else:
            r = q % 2
            q = q // 2

    return -1;

def nearest_valid_hadamard_size(i):
    multiple = give_new_number(i)
    return multiple[0]
    # return nearest_power_of_2(i)

def nearest_power_of_2(i):
    return 2**int(np.ceil(np.log2(i)))

def _expand_hadamard_non_sylvester(N):

    start_matrix = give_new_number(N)
    if start_matrix[1] == 4:
        M = H_4
    if start_matrix[1] == 12:
        M = H_12
    if start_matrix[1] == 20:
        M = H_20

    # Persistent Hadamard matrix
    H = getattr(_expand_hadamard_non_sylvester, "H_%d" % start_matrix[1], M)

    # construction out to N
    while H.shape[0] < N:
        H = np.concatenate((
            np.concatenate((H, H), axis=1),
            np.concatenate((H, -H), axis=1),
        ), axis=0)

    # Save for sequel
    if start_matrix[1] == 4:
        _expand_hadamard_non_sylvester.H_4 = H
    if start_matrix[1] == 12:
        _expand_hadamard_non_sylvester.H_12 = H
    if start_matrix[1] == 20:
        _expand_hadamard_non_sylvester.H_20 = H

    return H

def _expand_hadamard(N):
    """
    Expand to NxN Hadamard matrix using Sylvester construction.
    N must be a power of two.
    """

    # Persistent Hadamard matrix
    H = getattr(_expand_hadamard, "H", np.array([[1]]))

    # Check for power of 2
    if not np.log2(N) == int(np.log2(N)):
        raise(Exception("N=%d is not a power of 2"%N))

    # Sylvester construction out to N
    while H.shape[0] < N:
        H = np.concatenate((
                np.concatenate((H, H),axis=1),
                np.concatenate((H,-H),axis=1),
            ), axis=0)
    
    # Save for sequel
    _expand_hadamard.H = H

    return H

def random_hadamard(N, P):
    """
    Create randomized hadamard matrix of size NxP.
    N must be a valid Hadamard size.
    If P > N, only N columns are returned.
    """
    
    # Expand as necessary
    # H = _expand_hadamard(N)[:N,:min(N,P)]
    H = _expand_hadamard_non_sylvester(N)[:N,:min(N,P)]

    # Randomly negate rows
    R = np.sign(np.random.randn(H.shape[0],1)) * H

    # Randomly interchange N pairs of rows
    for _ in range(N):
        m, n = np.random.randint(N), np.random.randint(N)
        R[n,:], R[m,:] = R[m,:].copy(), R[n,:].copy()
    
    # # Interchange every pair of rows with some probability (N^2 time)
    # for (m,n) in it.combinations(range(R.shape[0]),2):
    #     if np.random.randn() > 0:
    #         R[n,:], R[m,:] = R[m,:].copy(), R[n,:].copy()

    return R

def random_orthogonal_patterns(N, P):
    """
    Create an NxP matrix of roughly orthogonal patterns.
    N must be a valid Hadamard size.
    If P > N, then orthogonality is only preserved within successive groups of N columns.
    """
    R = random_hadamard(N, P)

    while R.shape[1] < P:
        R = np.concatenate(
            (R, random_hadamard(N, P - R.shape[1])), axis=1)

    return R

if __name__ == "__main__":

    print((H_12.dot(H_12.T)).astype(int))
    print((H_20.dot(H_20.T)).astype(int))

    H = random_orthogonal_patterns(4,2)
    print((H.T.dot(H)).astype(int))

    H = random_orthogonal_patterns(4,4)
    print((H.T.dot(H)).astype(int))

    H = random_orthogonal_patterns(8,3)
    print((H.T.dot(H)).astype(int))

    H = random_orthogonal_patterns(12,12)
    print((H.T.dot(H)).astype(int))

    H = random_orthogonal_patterns(20,20)
    print((H.T.dot(H)).astype(int))

    H = random_orthogonal_patterns(24,24)
    print((H.T.dot(H)).astype(int))

    H = random_orthogonal_patterns(40,40)
    print((H.T.dot(H)).astype(int))

    H = random_orthogonal_patterns(12*20,12*20)
    print((H.T.dot(H)).astype(int))

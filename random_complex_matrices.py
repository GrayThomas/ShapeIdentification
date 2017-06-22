




import numpy as np




def randomUniformSubUnityComplexScalar():
    # generate a sub-unity magnitude complex number by trial and error
    while 1:
        a = 2*(np.random.random() - 0.5)
        b = 2*(np.random.random() - 0.5)
        if abs(complex(a,b))<=1:
            return complex(a,b)

def randomStandardComplexNormalVector():
	# standard complex normal vectors have identity expected value of the complex outer product.
    return np.sqrt(0.5)*np.array([np.random.normal() + complex(0, 1.) * np.random.normal() for j in range(0, 2)])

def randomComplexUnitVector():
    v = randomStandardComplexNormalVector()
    while(np.linalg.norm(v) < 1e-4):
        v = randomStandardComplexNormalVector()
    v *= 1.0 / np.linalg.norm(v)
    return v

def randomContractiveMatrix():
    # this samples over all contractions, with eigenvalues sampled evenly from zero to one, and directions random.
    # The actual distribution is not necessarily mathematically simple.
    # 4/100,000 seconds
    A = np.array([[np.random.normal() + complex(0, 1.) * np.random.normal()
                   for j in range(0, 2)] for i in range(0, 2)])
    u, s, v = np.linalg.svd(A)
    sprime = [si / abs(si) * np.random.random() for si in s]
    unity_A2 = u.dot(np.diagflat(sprime)).dot(v)
    return unity_A2

def randomStandardComplexNormalMatrix():
    # samples matrices via independent standard gaussian elements
    return np.sqrt(0.5)*np.array([[np.random.normal() + complex(0, 1.) * np.random.normal()
                      for j in range(0, 2)] for i in range(0, 2)])

def randomUnitaryMatrix():
    # this samples over all diagonalizable matrices with unit length singular
    # values
    A = np.array([[np.random.normal() + complex(0, 1.) * np.random.normal()
                   for j in range(0, 2)] for i in range(0, 2)])
    return unitarify(A)

def randomUniformContractiveMatrix():
    # this samples "evenly" over the space of contractions
    # 9.5/10,000 seconds
    while 1:
        A = np.array([[random_complex() for j in range(0, 2)]
                      for i in range(0, 2)])
        u, s, v = np.linalg.svd(A)

        if all([abs(si) <= 1.0 for si in s]):
            return A

def zeroMatrix():
    return complex(0.0,0.0)*np.zeros(2,2)


def unitarify(A):
    u, s, v = np.linalg.svd(A)
    sprime = [si / abs(si) for si in s]
    assert np.linalg.norm(A - u.dot(np.diagflat(s)).dot(v)) < 1e-12
    unity_A2 = u.dot(np.diagflat(sprime)).dot(v)
    return unity_A2

# Test randomUniformContractiveMatrix for speed.
if __name__ == '__main__':
    for i in range(10000):
        # print abs(random_complex())
        randomUniformContractiveMatrix()
    exit()
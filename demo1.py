"""
demo1 --- a script which implements the algorithm described in
"Uncertainty--Shape Identification in the Frequency Domain" 
by Gray C. Thomas and Luis Sentis.
"""
import numpy as np
import cvxpy as cvx # CVXOPT needs to work
from collections import namedtuple
# from grayplotlib import *
import matplotlib.pyplot as plt 
import sys
import traceback
plt.style.use(["seaborn-muted", "seaborn-paper"]) # requires the fancy new version of matplotlib

Model = namedtuple("Model", ["P", "E", "J", "KSigma_eta"])
np.random.seed(200)
np.random.seed(201)

USE_NOISE = False


def getRandomContraction():
    # this samples over all contractions, with eigenvalues sampled evenly from zero to one, and directions random.
    # The actual distribution is not necessarily mathematically simple.
    # 4/100,000 seconds
    A = np.array([[np.random.normal() + complex(0, 1.) * np.random.normal()
                   for j in range(0, 2)] for i in range(0, 2)])
    u, s, v = np.linalg.svd(A)
    sprime = [si / abs(si) * np.random.random() for si in s]
    unity_A2 = u.dot(np.diagflat(sprime)).dot(v)
    return unity_A2


def random_complex():
    a = (np.random.random() - 0.5) + 0.5
    b = (np.random.random() - 0.5) + 0.5
    return complex(a, b)


def getEvenRandomContraction():
    # this samples "evenly" over the space of contractions
    # 8/10,000 seconds
    while 1:
        A = np.array([[random_complex() for j in range(0, 2)]
                      for i in range(0, 2)])
        u, s, v = np.linalg.svd(A)
        if all([abs(si) <= 1.0 for si in s]):
            return A

# # Test getEvenRandomContraction for speed.
# if __name__ == '__main__':
#     for i in range(10000):
#         getEvenRandomContraction()
#     exit()


def getRandomPseudoContraction():
    # samples matrices via independent standard gaussian elements
    return np.array([[np.random.normal() + complex(0, 1.) * np.random.normal()
                      for j in range(0, 2)] for i in range(0, 2)])


def getRandomBoarderlineContraction():
    # this samples over all diagonalizable matrices with unit length singular
    # values
    A = np.array([[np.random.normal() + complex(0, 1.) * np.random.normal()
                   for j in range(0, 2)] for i in range(0, 2)])
    return cast_to_boarderline_contraction(A)

def getZeroContraction():
    # this samples over all diagonalizable matrices with unit length singular
    # values
    A = np.array([[np.random.normal() + complex(0, 1.) * np.random.normal()
                   for j in range(0, 2)] for i in range(0, 2)])
    return 0.0*cast_to_boarderline_contraction(A)


def cast_to_boarderline_contraction(A):
    u, s, v = np.linalg.svd(A)
    sprime = [si / abs(si) for si in s]
    assert np.linalg.norm(A - u.dot(np.diagflat(s)).dot(v)) < 1e-12
    unity_A2 = u.dot(np.diagflat(sprime)).dot(v)
    return unity_A2


def get_random_complex_vector():
    return np.array([np.random.normal() + complex(0, 1.) * np.random.normal() for j in range(0, 2)])


def get_random_complex_unit_vector():
    v = get_random_complex_vector()
    while(np.linalg.norm(v) < 1e-4):
        v = get_random_complex_vector()
    v *= 1.0 / np.linalg.norm(v)
    return v


def generateConditionGroupAverage(model, KSigma_u, N=100, ugenerator=get_random_complex_vector, generator=getRandomContraction):
    local_P = model.P + model.E.dot(generator()).dot(model.J)
    # local_U = KSigma_u.dot(get_random_complex_vector())
    local_U = KSigma_u.dot(ugenerator())
    Y = []
    S = 0
    for i in range(N):
        local_eta = model.KSigma_eta.dot(get_random_complex_vector())
        Y.append(local_eta + local_P.dot(local_U))
    Y = sum(Y) / len(Y)
    return local_U, Y



def H(mat):
    return mat.conjugate().T


def cone_matrix(m):
    # constructs the "Cone Form matrix M"
    # m is a model
    Einv = np.linalg.inv(m.E)
    A = -H(Einv).dot(Einv)
    B = -A.dot(m.P)
    C = H(m.J).dot(m.J) + H(m.P).dot(A).dot(m.P)
    return np.concatenate(
        (
            np.concatenate((A,    B), axis=1),
            np.concatenate((H(B), C), axis=1)
        ), axis=0)


def contraction_to_contraction(mod1, mod2):
    def mymap(Delta2):
        Pprime = mod2.P + mod2.E.dot(Delta2).dot(mod2.J)
        Delta1 = np.linalg.inv(mod1.E).dot(
            Pprime - mod1.P).dot(np.linalg.inv(mod1.J))
        return Delta1
    return mymap

def standardize(mod):
    K = mod.KSigma_eta
    if not USE_NOISE:
        K=np.eye(2)
    E = mod.E
    J = mod.J
    scale = np.linalg.norm(J,"fro")
    mod=Model(P=mod.P, E=mod.E*scale, J=mod.J/scale, KSigma_eta=K)
    return mod

def cost_function(mod1):
    E = mod1.E
    ne=E.shape[0]
    J = mod1.J
    # nj=J.shape[1]

    characteristic_radius = pow(abs(np.prod(np.linalg.eig(E)[0])),1./ne)
    expected_reciprocal_distance = np.linalg.norm(abs(np.linalg.eig(J)[0]))
    expected_characteristic_cone_ratio = characteristic_radius*expected_reciprocal_distance
    # print expected_characteristic_cone_ratio
    # print np.log(expected_characteristic_cone_ratio)
    return np.log(expected_characteristic_cone_ratio)

    # trJhJ = 2*np.trace(H(J).dot(J))
    # detEEh = np.linalg.det(E.dot(H(E)))
    # detEEh = detEEh*detEEh.conjugate()
    # log_det = np.log(detEEh)
    # # valuePrime =  2*log_det + trJhJ
    # # cvx.Minimize(-cvx.log_det(Mneg[0:4, 0:4]) + cvx.trace(Mpos)
    # # cvx.Minimize(-log(det(X_E)det(X_E*)) + 2*cvx.trace(X_J)
    # # return 2*log_det+trJhJ
    # # detEEh = detEEh*detEEh.conjugate()
    # log_det = np.log(detEEh)
    # # valuePrime =  2*log_det + trJhJ
    # # cvx.Minimize(-cvx.log_det(Mneg[0:4, 0:4]) + cvx.trace(Mpos)
    # # cvx.Minimize(-log(det(X_E)det(X_E*)) + 2*cvx.trace(X_J)
    # return log_det

def get_true_model():
    E = np.eye(2)
    E[0,0]=1.2
    # E[1,0]=0.3
    J=np.eye(2)*1.1
    J[1,1]=0.5
    J[0,1]=0.2
    trueModel = Model(
        P=np.eye(2) * 30,
        E=E,
        J=J,
        KSigma_eta=np.eye(2) * 0.0
    )
    return trueModel
# def get_true_model():
#     E = np.eye(2)
#     E[1,1]=10.
#     J = np.eye(2)
#     trueModel = Model(
#         P=np.eye(2) * 10,
#         E=E,
#         J=J,
#         KSigma_eta=np.eye(2) * 0.0
#     )
#     return trueModel

trueModel = get_true_model()

# print trueModel
# print standardize(trueModel)
# print trueModel
# print trueModel.E
print cost_function(trueModel)



# def learn_model(dat):
#     # a dummy -> tests should fail.
#     return Model(
#         P=np.eye(2) + 1e-4 * getRandomContraction(),
#         E=np.eye(2) * 1.000,
#         J=np.eye(2) * 1e-2,
#         KSigma_eta=np.eye(2) * 1e-1
#     )

# def learn_model(dat):
#     # a dummy -> tests should pass.
#     scale = 10.0/len(dat)**2
#     return Model(
#         P=np.eye(2) + 1e-2 * scale * getRandomContraction(),
#         E=np.eye(2)*(1+scale)+ 1e-2 * scale * getRandomContraction(),
#         J=np.eye(2)*1e-2+ 1e-6 * scale * getRandomContraction(),
#         KSigma_eta=np.eye(2)*1e-1)


# how to deal with complex matrices:
    # x = a + bj
    # x' = aT - bTj
    # M=M', M=A+B j, A=AT, B=-BT
    # x'Mx = x'Ax+x'Bx
    # = aTAa+aTAbj-jbTAa-jjbTAb  +   jaTBa+aTBbjj-jjbTBa-jjjbTBb
    # = aTAa+bTAb  + -2aTBb
    # = aTAa  +bTBa-aTBb +bTAb
    # = [aT bT][A -B; B A][a;b]

def semidef_complex_2x2(cons):
    # represents a 2x2 complex semidefinite matrix as a 4x4 real semidefinite
    # matrix subject to constraints
    meta_matrix = cvx.Variable(4, 4)
    possemidef = cvx.Semidef(4)
    cons.append(meta_matrix[0:2, 0:2] == meta_matrix[2:4, 2:4])
    cons.append(meta_matrix[0:2, 2:4] == -meta_matrix[2:4, 0:2])
    cons.append(meta_matrix - cvx.Constant(np.eye(4) * 2e-5) == possemidef)
    return meta_matrix


def convert_complex_2x2(mat):
    # converts a 2x2 complex matrix as a 4x4 real matrix
    # this is the truth, confirmed via a test
    new_mat = np.zeros((4, 4))
    new_mat[0:2, 0:2] = mat.real
    new_mat[0:2, 2:4] = -mat.imag
    new_mat[2:4, 0:2] = mat.imag
    new_mat[2:4, 2:4] = mat.real
    return new_mat

    # if this matrix is symmetric as well, then it implies that mat.real is symmetric and that (-mat.imag)^T = mat.imag
    # as we would expect---mat is hermetian.

    # if this matrix is semidefinite as well then it should mean that the inner product of any complex number is positive,
    # but the complex number inner product uses a conjugate transpose in stead of a simple transpose for the left vector.
    # We test this as well, and it turns out that since the matrix is semidefinite the result of inner products with the same
    # vecotr will be real. This is a helpful property, since computing the real part of the inner product is simpler than computing
    # both the real and imaginary parts. The real part of the inner product is simply the real vector equivalent inner product:
    # Re (a-bj)' (c+dj) = a'c+ b'd = [a;b]' [c;d]   On the other hand Im (a-bj)' (c+dj) = -b'c+a'd = [-b;a]' [c;d]
    # fortunately, we never need to calculate this.


def test_convert_complex_2x2():
    # test with commutitive diagram
    complex_matrix = np.array(
        [[complex(np.random.random(), np.random.random()) for i in range(2)] for j in range(2)])
    vectors = np.array([[complex(np.random.random(), np.random.random())
                         for i in range(20)] for j in range(2)])
    print complex_matrix
    print vectors[:, 0], vectors[:, 1]
    real_matrix = convert_complex_2x2(complex_matrix)
    real_vectors = np.vstack([vectors.real, vectors.imag])
    print real_vectors[:, 0], real_vectors[:, 1]
    print real_matrix
    real_results1 = real_matrix.dot(real_vectors)
    complex_results2 = complex_matrix.dot(vectors)
    real_results2 = np.vstack([complex_results2.real, complex_results2.imag])
    assert(np.linalg.norm(real_results2 - real_results1) < 1e-7)
    re_inner_product1 = real_vectors.T.dot(real_results1)
    re_inner_product2 = (vectors.conjugate().T.dot(complex_results2)).real
    # print inner_product1.shape, inner_product2.shape
    assert(np.linalg.norm(re_inner_product1 - re_inner_product2) < 1e-7)
    exit()

# if __name__ == '__main__':
#     test_convert_complex_2x2()

    # assert that Pcomplex.T is equivalent to convert_complex_2x2(P.conjugate().T):
    # if M = A+Bi
    # return [A,-B;B, A]
    # if M= A'-B'i
    # return [A',+B';-B', A'] = [A,-B;B, A]' confirmed!


def guarantee_complex_structure(mat4by4, cons):
    cons.append(mat4by4[0:2, 2:4] == -mat4by4[2:4, 0:2])
    cons.append(mat4by4[0:2, 0:2] == mat4by4[2:4, 2:4])

def validate_complex_structure(mat4by4, tol=1e-7):
    print np.linalg.norm(mat4by4[0:2, 2:4]+mat4by4[2:4, 0:2])
    print np.linalg.norm(mat4by4[0:2, 2:4])
    assert np.linalg.norm(mat4by4[0:2, 2:4]+mat4by4[2:4, 0:2])<tol
    assert np.linalg.norm(mat4by4[0:2, 0:2]-mat4by4[2:4, 2:4])<tol


def back_to_array(cvxmat_complex):
    return np.array(cvxmat_complex[0:2, 0:2]) - complex(0, 1) * np.array(cvxmat_complex[0:2, 2:4])


def triang(n):
    return n * (n + 1) / 2


def SkewSym(n, cons):
    N = n * (n + 1) / 2 - n
    var = cvx.Variable(N)
    mat = cvx.Variable(n, n)
    # print n
    # print [h for h in range(1,n)]
    for h in range(0, n - 1):
        js = range(triang(h) + 1, triang(h + 1) + 1)
        for j in js:
            cons.append(mat[j - triang(h) - 1, n - 1 - h +
                            j - triang(h) - 1] == var[j - 1])
            cons.append(mat[n - 1 - h + j - triang(h) - 1,
                            j - triang(h) - 1] == -var[j - 1])
        [j for j in js]
        lc = [[(j - triang(h) - 1), (n - 1 - h + j - triang(h) - 1), (j)]
              for j in js]
        lc2 = [[(n - 1 - h + j - triang(h) - 1), (j - triang(h) - 1), (-j)]
               for j in js]
        # print lc ,  "at", h
        # print lc2 ,  "at", h
    [cons.append(mat[i, i] == 0) for i in range(n)]
    # print cons
    # print [[(j-trang(h)-1,n-1-h+j-trang(h)-1)] for j in
    # range(triang(h)+1,triang(h+1)+1)] for h in range(0, n)]


if __name__ == '__main__':
    SkewSym(2, [])
    SkewSym(3, [])
    SkewSym(4, [])
    # exit()
# Strategy Discussion: lambda conjecture, and conservatism function.

# Pros for solving lambda conjecture:
#   * Impressive Theorem
#   * Rigorous Foundation for Conservatism Test
# Cons for solving lambda conjecture (and insteady relying on numerics):
#   * Time Investement would push back paper publication
#   * Complex proof would dilute message of paper, which is about the convex program

XE0=convert_complex_2x2(trueModel.E)
def learn_model(dat, alpha=-0.0002, verbose=False, learn_P=True, bias_comp=lambda N: 1.0002):
    # assumes the plant is known perfectly

    # assumes dat is a list of (u,y) pairs representing cg averages.
    # alpha is the noise-compensating factor.

    # TODO: estimate covariance properly.
    KSigma_eta = get_true_model().KSigma_eta

    # exit()

    cons = []

    # Setup hermetian 2x2 vars as symmetric 4x4 real vars
    EihEicomplex = semidef_complex_2x2(cons)
    if not learn_P:
        JhJcomplex = semidef_complex_2x2(cons)
        Pcomplex = cvx.Constant(convert_complex_2x2(P))

    # Construct the "Cone Form matrix M"
    # M = [-EihEi     EihEiP       ] = [ 0   0  ] - Mneg
    #     [PhEihEi    JhJ-PhEihEiP ]   [ 0  Mpos]

    # EihEi_r = Symmetric(2)
    # EihEi_i = Symmetric(2)
    # EihEiP_r = Variable(2,2)
    # PhEihEiP_r = Symmetric(2)
    # JhJ_r = Symmetric(2)
    # JhJ_i = SkewSym(2)
    Mneg = cvx.Semidef(8)

    guarantee_complex_structure(Mneg[0:4, 0:4], cons)
    guarantee_complex_structure(Mneg[0:4, 4:8], cons)
    # guarantee_complex_structure(Mneg[4:8, 0:4], cons) # unneccesary since Mneg is symmetric
    guarantee_complex_structure(Mneg[4:8, 4:8], cons)
    Mpos = cvx.Semidef(4)
    guarantee_complex_structure(Mpos, cons)

    # Add all the inclusion constraints. One per condition group.
    for u, y in dat:
        # convert complex data into pseudo-complex real vector of twice the
        # size
        prexi = np.hstack([y.real, y.imag, u.real, u.imag]).reshape((-1, 1))
        # convention is a little strange because it maintains 2x2 block
        # structure
        xi_u = cvx.Constant(np.hstack([u.real, u.imag]).reshape((-1, 1)))
        xi = cvx.Constant(prexi)
        # TODO: determine alpha based on noise
        cons.append(cvx.quad_form(xi_u, Mpos) >=
                     cvx.quad_form(xi, Mneg)) 

    # Add a final constraint on overall E versus J size
    # TODO: relate this constraint to noise
    # cons.append(cvx.trace(Mneg[0:4, 0:4]) == 4.0)
    cons.append(cvx.trace(Mpos) == 4.0*np.sqrt(.25))

    # Solve the convex problem

    # 0.5 for complex numbers
    # 0.5 for number of outputs
    # 0.5 for complex to real conversion
    # global XE0
    # prob = cvx.Problem(cvx.Minimize(-0.5*0.5*(1./2.)*cvx.trace(
    #     cvx.Constant(np.linalg.inv(XE0))*Mneg[0:4, 0:4])), cons)
    prob = cvx.Problem(cvx.Minimize(-0.5*0.5*(1./2.)*cvx.log_det(
        Mneg[0:4, 0:4])), cons)


    for solver_attempt in range(4):
        try:
            res = prob.solve(
                solver="CVXOPT",verbose=True,
                abstol=1e-9, reltol=1e-9, feastol = 1.0e-9,
                refinement=[5,20,100,300][solver_attempt], kktsolver="robust", # {"chol", "robust"} robust=LDL
                max_iters=200)
        except cvx.error.SolverError as e:
            _, _, tb = sys.exc_info()
            traceback.print_tb(tb)
            if solver_attempt==3:
                return Model() , -float('NaN'), - float('NaN')
        else:
            break
    # try:

    #     # There are only two options in CVX which can solve such complex problems:
    #     # res = prob.solve(
    #     #     solver="SCS", verbose=True,
    #     #     eps=1e-7, alpha=1.5, scale=1.5, normalize=False, use_indirect=False, warm_start=True, max_iters=25000)
    #     # res = prob.solve(
    #     #     solver="SCS", verbose=True)
    #     # res = prob.solve(
    #     #     solver="CVXOPT",verbose=verbose,
    #     #     abstol=1e-7, reltol=1e-7, feastol = 1.0e-7,
    #     #     refinement=20, kktsolver="robust", # {"chol", "robust"} robust=LDL
    #     #     max_iters=2000) # worked well with a particular true model

    #     res = prob.solve(
    #         solver="CVXOPT",verbose=True,
    #         abstol=1e-9, reltol=1e-9, feastol = 1.0e-9,
    #         refinement=100, kktsolver="robust", # {"chol", "robust"} robust=LDL
    #         max_iters=200)

    #     # res = prob.solve(
    #     #     solver="CVXOPT",verbose=True,
    #     #     abstol=1e-9, reltol=1e-9, feastol = 1.0e-9,
    #     #     refinement=10, kktsolver="robust", # {"chol", "robust"} robust=LDL
    #     #     max_iters=200)
    # except cvx.error.SolverError as e:
    #     _, _, tb = sys.exc_info()
    #     traceback.print_tb(tb)
    #     return Model() , -float('NaN'), - float('NaN')
    enforce_solver_tolerance = False
    eft = enforce_solver_tolerance

    # -cvx.log_det(Mneg[0:4, 0:4]) + cvx.trace(Mpos)
    valuePrime =  -np.log(np.linalg.det(Mneg.value[0:4,0:4])) + np.trace(Mpos.value) # good
    valuePrime =  -2*np.log(np.abs(np.linalg.det(Mneg.value[0:2,0:2]+complex(0,1)*Mneg.value[2:4,0:2]))) + 2*np.trace(Mpos.value[0:2,0:2])# good
    print np.trace(Mneg.value[0:2,0:2]+complex(0,1)*Mneg.value[2:4,0:2])
    if eft: assert abs(np.trace(Mpos.value[0:2,0:2]+complex(0,1)*Mpos.value[2:4,0:2]))<1.01
    if eft: assert abs(np.trace(Mpos.value[0:2,0:2]+complex(0,1)*Mpos.value[2:4,0:2]))>0.99
    print prob.status
    try:
        M = -np.array(Mneg.value)
        if eft: assert np.linalg.norm(M.T - M) < 1e-7
        M[4:8, 4:8] += np.array(Mpos.value)

        # Cast inaccurate results to guarantee valid Hermetian format
        if eft: assert np.linalg.norm(M.T - M) < 1e-7
        if eft: validate_complex_structure(M[0:4,0:4], tol=1e-2)
        if eft: validate_complex_structure(M[0:4,4:8], tol=1e-2)
        if eft: validate_complex_structure(M[4:8,0:4], tol=1e-2)
        if eft: validate_complex_structure(M[4:8,4:8], tol=1e-2)

        M = 0.5 * (M.T + M)  # cast to symmetric (not really necessary?)

        valuePrime =  -2*np.log(np.abs(np.linalg.det(-M[0:2,0:2]-complex(0,1)*M[2:4,0:2]))) + 2*np.trace(Mpos.value[0:2,0:2]) # good

        checktol = 1e-2
        # real part of -EihEi
        if eft: assert np.linalg.norm(M[0:2, 0:2] - M[2:4, 2:4]) < checktol
        M[0:2, 0:2] = 0.5 * (M[0:2, 0:2] + M[2:4, 2:4])
        M[2:4, 2:4] = M[0:2, 0:2]

        # imag part of -EihEi
        if eft: assert np.linalg.norm(M[0:2, 2:4] + M[2:4, 0:2]) < checktol
        M[0:2, 2:4] = 0.25 * (M[0:2, 2:4] - M[0:2, 2:4].T -
                              M[2:4, 0:2] + M[2:4, 0:2].T)
        M[2:4, 0:2] = -M[0:2, 2:4]

        # real part of PhEihEi (& EihEiP)
        if eft: assert np.linalg.norm(M[4:6, 0:2] - M[6:8, 2:4]) < checktol
        M[4:6, 0:2] = 0.5 * (M[4:6, 0:2] + M[6:8, 2:4])
        M[6:8, 2:4] = M[4:6, 0:2]
        M[0:2, 4:6] = M[4:6, 0:2].T
        M[2:4, 6:8] = M[4:6, 0:2].T

        # imag part of PhEihEi (& EihEiP)
        if eft: assert np.linalg.norm(M[4:6, 2:4] + M[6:8, 0:2]) < checktol
        M[4:6, 2:4] = 0.5 * (M[4:6, 2:4] - M[6:8, 0:2])
        M[6:8, 0:2] = - M[4:6, 2:4]
        M[0:2, 6:8] = M[6:8, 0:2].T
        M[2:4, 4:6] = M[4:6, 2:4].T

        # real part of JhJ-PhEihEiP
        if eft: assert np.linalg.norm(M[4:6, 4:6] - M[6:8, 6:8]) < checktol
        M[4:6, 4:6] = 0.5 * (M[4:6, 4:6] + M[6:8, 6:8])
        M[6:8, 6:8] = M[4:6, 4:6]

        # imag part of JhJ-PhEihEiP
        if eft: assert np.linalg.norm(M[4:6, 6:8] + M[6:8, 4:6]) < checktol
        if eft: assert np.linalg.norm(M[4:6, 6:8] + M[4:6, 6:8].T) < checktol
        M[4:6, 6:8] = 0.25 * (M[4:6, 6:8] - M[4:6, 6:8].T -
                              M[6:8, 4:6] + M[6:8, 4:6].T)
        M[6:8, 4:6] = - M[4:6, 6:8]

        # cast to postive definite:


        valuePrime =  -2*np.log(np.abs(np.linalg.det(-M[0:2,0:2]-complex(0,1)*M[2:4,0:2]))) + 2*np.trace(Mpos.value[0:2,0:2]) # good


        # validate afterwards
        assert np.linalg.norm(M.T - M) < 1e-7
        validate_complex_structure(M[0:4,0:4], tol=1e-7)
        validate_complex_structure(M[0:4,4:8], tol=1e-7)
        validate_complex_structure(M[4:8,0:4], tol=1e-7)
        validate_complex_structure(M[4:8,4:8], tol=1e-7)


        EihEi = -back_to_array(M[0:4, 0:4])
        assert np.linalg.norm(EihEi.conjugate().T - EihEi) < 1e-7
        JhJmPhEihEiP = back_to_array(M[4:8, 4:8])
        EihEiP = back_to_array(M[0:4, 4:8])
        P = np.linalg.solve(EihEi, EihEiP)
        JhJ = JhJmPhEihEiP + H(P).dot(EihEi).dot(P)
        JhJ = back_to_array(Mpos.value)

        # cast JhJ to positive semi-definite
        eig_JhJ, u_JhJ = np.linalg.eigh(JhJ)
        min_eig = min(eig_JhJ)
        if min_eig<0:
            print "buffing low eigenvalues"
            JhJ += np.eye(JhJ.shape[0])*(abs(min_eig)+1e-6)

        # cast EihEi to P.S.D.
        eig_EihEi, _ = np.linalg.eigh(EihEi)
        min_eig = min(eig_EihEi)
        if min_eig<0:
            print "buffing low eigenvalues in EihEi"
            EihEi += np.eye(EihEi.shape[0])*(abs(min_eig)+1e-6)

        # Extract J and E via cholesky decomposition
        J = np.linalg.cholesky(JhJ).conjugate().T
        E = np.linalg.inv(np.linalg.cholesky(EihEi).conjugate().T)


        trJhJ = 2*np.trace(H(J).dot(J))
        detEEh = np.linalg.det(E.dot(H(E)))
        detEEh = detEEh*detEEh.conjugate()
        log_det = np.log(detEEh)
        valuePrime =  log_det + 2*np.trace(back_to_array(Mpos.value)) # good
        valuePrime =  log_det + trJhJ # good


        print J

        print eig_JhJ
        print u_JhJ
        print JhJ - u_JhJ.dot(np.diagflat(eig_JhJ)).dot(u_JhJ.conjugate().T)
        print JhJ
        print np.linalg.norm(JhJ)
        print np.linalg.norm(J.conjugate().T.dot(J) - JhJ)
        print np.linalg.norm(J.dot(J.conjugate().T) - JhJ)
        print np.linalg.norm(J.conjugate().dot(J.T) - JhJ)
        print np.linalg.norm(J.T.dot(J.conjugate()) - JhJ)
        if eft: assert np.linalg.norm(J.conjugate().T.dot(J) - JhJ) < 1e-5
        if eft: assert np.linalg.norm(np.linalg.inv(E.conjugate().T).dot(
            np.linalg.inv(E)) - EihEi) < 1e-5

        # apply simple bias compensation
        # E *= bias_comp(len(dat))

        # Point-wise confirmation check
        assert np.linalg.norm(KSigma_eta) < 1e-7
        max_scale=1.
        for u, y in dat:
            # y = P u + E Delta J u + 0
            # Delta J u = Ei (y - P u)
            q = np.linalg.solve(E, y - P.dot(u))
            v = J.dot(u)
            scale = np.linalg.norm(q)/(np.linalg.norm(v)-1e-3)
            if scale>max_scale:
                max_scale = scale
            if verbose:
                print "|q|, |v|", np.linalg.norm(q), np.linalg.norm(v)
            # q = Ei (y - P u), v = J u, |q|<|v|
        print "max_scale", max_scale
        # E *= max_scale

        # Ship it.
        return Model(
            P=P,
            E=E,
            J=J,
            KSigma_eta=KSigma_eta
        ), res, valuePrime
    except AssertionError as e:
        print "assertion error"
        _, _, tb = sys.exc_info()
        traceback.print_tb(tb)
        return Model() , -float('NaN'), - float('NaN')


def model_distance(mod1, mod2):
    # calculate distance between models.
    eP = mod1.P - mod2.P
    #
    # only compare ETE, JTJ, Sigma matrices. E and J are not unique.
    ETE1 = mod1.E.T.dot(mod1.E)
    ETE2 = mod2.E.T.dot(mod2.E)

    JTJ1 = mod1.J.T.dot(mod1.J)
    JTJ2 = mod2.J.T.dot(mod2.J)
    eE = ETE1 / np.trace(ETE1) - ETE2 / np.trace(ETE2)
    eJ = JTJ1 / np.trace(JTJ1) - JTJ2 / np.trace(JTJ2)
    eS = mod1.KSigma_eta.T.dot(mod1.KSigma_eta) - \
        mod2.KSigma_eta.T.dot(mod2.KSigma_eta)
    # sum of squared frobenius norms---penalizes all elements of all error
    # matrices equally.
    return np.sqrt(np.linalg.norm(eP)**2 + np.linalg.norm(eE)**2 + np.linalg.norm(eJ)**2 + np.linalg.norm(eS)**2)

def model_distance(mod1, mod2):
    # calculate distance between models.
    eP = mod1.P - mod2.P
    #
    # only compare ETE, JTJ, Sigma matrices. E and J are not unique.
    ETE1 = mod1.E.dot(H(mod1.E))
    ETE2 = mod2.E.dot(H(mod2.E))
    
    JTJ1 = H(mod1.J).dot(mod1.J)
    JTJ2 = H(mod2.J).dot(mod2.J)
    eE = ETE1 / np.trace(ETE1) - ETE2 / np.trace(ETE2)
    eJ = JTJ1 * np.trace(ETE1) - JTJ2 * np.trace(ETE2)
    eS = H(mod1.KSigma_eta).dot(mod1.KSigma_eta) - \
        H(mod2.KSigma_eta).dot(mod2.KSigma_eta)
    # sum of squared frobenius norms---penalizes all elements of all error
    # matrices equally.
    return np.sqrt(np.linalg.norm(eP)**2 + np.linalg.norm(eE)**2 + np.linalg.norm(eJ)**2 + np.linalg.norm(eS)**2)






def model_contains(mod1, mod2, generator=getRandomBoarderlineContraction, verbose=False):
    # checks if mod1 contains mod2
    # generates random boarderline elements of mod2
    # and finds the worst case error margin
    maxsing = 1e-99
    for i in range(10):
        Delta2 = generator()
        Pprime = mod2.P + mod2.E.dot(Delta2).dot(mod2.J)
        Delta1 = np.linalg.inv(mod1.E).dot(
            Pprime - mod1.P).dot(np.linalg.inv(mod1.J))
        # assert np.linalg.norm(
        #     mod1.P + mod1.E.dot(Delta1).dot(mod1.J) - Pprime) < 1e-6
        # assert np.linalg.norm(
        #     mod2.P + mod2.E.dot(Delta2).dot(mod2.J) - Pprime) < 1e-6
        sing, X = np.linalg.eigh(H(Delta1).dot(Delta1))
        # these are the singular values of Delta2.
        # If any singular value is greater than one, then Delta2 is not a
        # contraction
        if verbose:
            print "eigh (Delta2HDelta2", sing
        maxsing = max(maxsing, max(sing))
    if verbose:
        print np.log(maxsing)

    return np.log(maxsing)


def loud_assert(condition, string):
    print "Assert: %s" % string
    if condition:
        print "\tPass."
    else:
        print "\tFail."


def test_inclusion_function(func):
    # tests func(mbig, msmall) which compares two models and returns a
    # negative number if mbig fails to contain msmall.
    # The test has several assertions, for simple cases which are easy to verify.
    # func(M1,M2)>0 if M1 does not include M2
    # func(M1,M2)<0 if M1 includes M2
    M1 = Model(P=np.eye(2), E=np.eye(2), J=np.eye(2), KSigma_eta=np.eye(2))
    M2 = Model(P=M1.P, E=M1.E, J=M1.J * 0.9999, KSigma_eta=M1.KSigma_eta)
    loud_assert(func(M1, M2) < 0.0, "Shrink J in model 2 --> included")
    M2 = Model(P=M1.P, E=M1.E * 0.9999, J=M1.J, KSigma_eta=M1.KSigma_eta)
    loud_assert(func(M1, M2) < 0.0, "Shrink E in model 2 --> included")
    M2 = Model(P=M1.P, E=M1.E, J=M1.J * 1.00001, KSigma_eta=M1.KSigma_eta)
    loud_assert(func(M1, M2) > 0.0, "Grow J in model 2 --> not included")
    M2 = Model(P=M1.P, E=M1.E * 1.00001, J=M1.J, KSigma_eta=M1.KSigma_eta)
    loud_assert(func(M1, M2) > 0.0, "Grow E in model 2 --> not included")
    # model mismatch
    M1 = Model(P=np.eye(2) * 0.0, E=np.eye(2),
               J=np.eye(2), KSigma_eta=np.eye(2))
    M2 = Model(P=np.eye(2) * 1.0, E=M1.E, J=M1.J, KSigma_eta=M1.KSigma_eta)
    loud_assert(func(M1, M2) > 0.0, "Mismatch P --> not included")
    M2 = Model(P=np.eye(2) * 0.5, E=M1.E * 0.4999,
               J=M1.J, KSigma_eta=M1.KSigma_eta)
    loud_assert(func(M1, M2,) < 0.0, "Mismatch P, but tight E --> included")
    M2 = Model(P=np.eye(2) * 0.5, E=M1.E, J=M1.J *
               0.4999, KSigma_eta=M1.KSigma_eta)
    loud_assert(func(M1, M2,) < 0.0, "Mismatch P, but tight J --> included")


# if __name__ == "__main__":
#     test_inclusion_function(model_contains)
#     exit()


###########################################################################
##############  MAIN TEST  ############  MAIN TEST  #######################
###########################################################################

def getConvenientContraction():
    if np.random.random() > 0.5:
        return getRandomBoarderlineContraction()
    else:
        return getRandomContraction()


if __name__ == "__main__":
    trueModel = get_true_model()
    num_trials = 1
    KSigma_u = np.array([[1.0, 0.0], [0.0, 1.0]]) * 1e1
    # fig = plt.figure()
    # ax1 = plt.gca()
    # fig2 = plt.figure()
    # ax2 = plt.gca()
    fig, (ax1, ax2) = plt.subplots(2,sharex=True)
    for trial in range(num_trials):
        cgDat = []
        sampN = 2  # effectively decreases sigma, but sigma is zero
        # cgNs is numbers of condition groups
        # cgNs = range(40, 300, 10)  # more realistic
        cgNs = range(6, 60, 10)  # For minimimum convergence time testing

        # contraction_gen = getRandomContraction  # realistic mode (solver fails sometimes)
        contraction_gen=getConvenientContraction  # super nice mode
        # contraction_gen=getRandomBoarderlineContraction # most
        # contraction_gen=getRandomPseudoContraction # Gaussian mode

        dists, good_cgNs, costs, costs3 = [], [], [], []
        for cgN in cgNs:
            try:
                cgDat.extend([generateConditionGroupAverage(
                    trueModel, KSigma_u, N=sampN,
                    ugenerator=get_random_complex_vector,
                    generator=contraction_gen
                    # informative data mode
                ) for i in range(cgN - len(cgDat))])
                guessModel, costcvx, cost2 = learn_model(cgDat)
                dist = model_distance(trueModel, guessModel)
                # print model_contains(guessModel, trueModel)
                dists.append(dist)
                costs.append(cost_function(guessModel))#cost_function(guessModel))
                costs3.append(costcvx)
                good_cgNs.append(cgN)

            # except (np.linalg.linalg.LinAlgError, cvx.error.SolverError):
            except (TypeError):
                good_cgNs.append(cgN)
                dists.append(float('NaN'))
                costs.append(float('NaN'))
                costs3.append(float('NaN'))
        try:
            ax1.semilogy(good_cgNs, dists,'-o')
        except Exception:
            pass
        try:
            ax2.plot(good_cgNs, costs, '-o',lw=1.5)
        except Exception:
            pass
        ax2.plot(cgNs, costs3,'ko-',lw=1.)
    ax2.plot(cgNs, [cost_function(trueModel) for cgN in cgNs], 'k:', lw=3)
    ax1.set_title("Consistency")
    ax1.set_ylabel("Distance from True Model")
    # ax1.set_xlabel("number of condition groups")

    ax2.set_title("Cost Function")
    ax2.set_ylabel("Log of Generalized Cone Width")
    ax2.set_xlabel("Number of Condition Groups")
    plt.show()
    exit()
    # dat is a list of (u,y) pairs, where u and y are both (2,) vectors of
    # complex floats.
    for u, y in dat:
        print u[0].real, u[0].imag, u[1].real, u[1].imag, y[0].real, y[0].imag, y[1].real, y[1].imag
    print dat
    exit()

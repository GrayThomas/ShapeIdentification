from collections import namedtuple
import numpy as np



SingleFrequencyQuadricModelSet = namedtuple("SingleFrequencyQuadricModelSet", ["P", "E", "J"])
# note, re-defined functions like this one are basically deleted.
def generalized_cone_width(mod1):
    # This is numerically terrible, causing a visible disparity between the
    # cost of the returned model and the cost returned by the solver. This is
    # responsible for the blue line in the initial submission of the paper.
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


def generalized_cone_width(mod1):
    # This cost function is vastly more reliable. Differences due to
    # conversion are on the order of the solver tolerance 1e-9. Moral of this
    # story: trust only numpy.linalg.eigh.
    E = mod1.E
    ne=E.shape[0]
    J = mod1.J
    EE = E.dot(H(E))
    JJ = H(J).dot(J)
    char_radius = pow(np.prod(np.linalg.eigh(EE)[0]), 0.5/ne)
    char_recip_dist = np.sqrt(sum(np.linalg.eigh(JJ)[0]))
    return np.log(char_radius*char_recip_dist)

def validate_result(final_model, dat):
    # checks whether a model-set includes all the data
    max_scale=-0.1
    for u, y in dat:
        # y = P u + E q
        # q = Delta v
        # v = J u
        q = np.linalg.solve(final_model.E, y.reshape((-1,1)) - final_model.P.dot(u.reshape((-1,1))))
        v = final_model.J.dot(u.reshape((-1,1)))
        scale = np.linalg.norm(q)/(np.linalg.norm(v))
        if scale>max_scale:
            max_scale = scale

    return max_scale

def safe_extract_model(X_E, X_P, X_PP, X_J, enforce_solver_tolerance = 1e-2):
    # checks some complex number as real number matrix constraints, and extracts the complex model

    # shorthand
    eft = enforce_solver_tolerance

    # enter try environment which should properly follow stack traces for
    # failed assertions.
    try:
        # Note: This method of using the hermetian eigendecomposition improves
        # over the Cholesky decomposition used in the journal paper, improving
        # numerical stability and the accuracy of the solution.

        l_J, U_J = np.linalg.eigh(X_J)
        print l_J
        l_E, U_E = np.linalg.eigh(X_E)
        print l_E
        sqrt_l_J = [np.sqrt(l) if l>1e-12 else 1e-6 for l in l_J]
        inv_sqrt_l_E = [1./np.sqrt(l) if l>1e-12 else 1e6 for l in l_E]
        assert np.linalg.norm(X_E-U_E.dot(np.diagflat(l_E)).dot(U_E.H),"fro")<1e-7
        assert np.linalg.norm(X_J-U_J.dot(np.diagflat(l_J)).dot(U_J.H),"fro")<1e-7
        J = np.diagflat(sqrt_l_J).dot(U_J.H)
        E = U_E.dot(np.diagflat(inv_sqrt_l_E))
        P = E.dot(E.H).dot(X_P)

        print np.linalg.inv(E.dot(E.H))-X_E

        # unique SS-DD assertion:
        assert np.linalg.norm(X_PP - P.H.dot(X_P),'fro')<1e-7

        # Ship it.
        return SingleFrequencyQuadricModelSet(
            P=P,
            E=E,
            J=J
        )
    except AssertionError as e:
        print "assertion error"
        _, _, tb = sys.exc_info()
        traceback.print_tb(tb)
        return SingleFrequencyQuadricModelSet()

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

    # sum of squared frobenius norms---penalizes all elements of all error
    # matrices equally.
    return np.sqrt(np.linalg.norm(eP)**2 + np.linalg.norm(eE)**2 + np.linalg.norm(eJ)**2)

def H(mat):
    return mat.conjugate().T
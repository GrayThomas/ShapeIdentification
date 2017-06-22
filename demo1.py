"""
demo1 --- a script which implements the algorithm described in
"Uncertainty--Shape Identification in the Frequency Domain" 
by Gray C. Thomas and Luis Sentis.
"""
import numpy as np
import cvxpy as cvx # CVXOPT needs to work

# from grayplotlib import *
import matplotlib.pyplot as plt 
import sys
import traceback

# project specific imports
from random_complex_matrices import (
    randomUniformSubUnityComplexScalar, # by trial and error, magnitude less than one.
    randomStandardComplexNormalVector, # elementwise standard complex normal
    randomComplexUnitVector, # scales standard complex normal to have unit inner product
    randomContractiveMatrix, # uniform [0,1] distributed singular values guarantee contractivity.
    randomStandardComplexNormalMatrix, # generates elements from standard complex normal distribution
    randomUnitaryMatrix, # scales singular values to unity
    randomUniformContractiveMatrix, # produces contractive matrices by trial and error, using even distributions for the elements
    zeroMatrix, # a zero matrix of the same size as the contraction-generating functions produce
    unitarify, # casts singular values to unity
)
from phi_transform_utilities import (
    phi_image_structure, # ensures that a cvxpy matrix has appropriate structure to be the phi transform of a complex matrix
    cvx_block_matrix, # stacks cvx matrix types to make a block matrix. Usage similar to numpy.bmat
    inverse_phi, # returns a complex matrix from a larger real matrix
    vec_phi
)
from shape_identification_tools import (
    SingleFrequencyQuadricModelSet, # basic model class --- a namedtuple P, E, J
    generalized_cone_width, # calculates the generalized cone width for a model
    validate_result, # returns worst case delta magnitude required to include all the data. 
    safe_extract_model, # extracts quadric model-sets from X_E, X_J, X_P, X_JJ
    model_distance, # an alternative distance metric between quadric model sets
    H # short for hermite or conjugate transpose
)

plt.style.use(["seaborn-muted", "seaborn-paper"]) # requires the fancy new version of matplotlib

# The solvers are finnicky. The selected seed doesn't exhibit too much solver
# failure, but other seeds will.
np.random.seed(201) # seed to generate paper plots
# np.random.seed(200) # demonstrates more solver failure

def main():
    true_ms = get_true_ms()
    KSigma_u = np.array([[1.0, 0.0], [0.0, 1.0]]) * 1e1

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)

    cgDat = [] # condition-group averages data-set (to be populated)
    sampN = 2  # effectively decreases sigma. meaningless in the absence of noise
    # cgNs is numbers of condition groups
    # cgNs = range(40, 300, 10)  # more realistic
    cgNs = range(6, 60, 10)  # fast condition (good for demo)
    # cgNs = range(6, 60, 1) # paper condition

    # contraction_gen = randomContractiveMatrix # realistic mode (solver fails sometimes)
    contraction_gen=getConvenientContraction # super nice mode
    # contraction_gen=randomUnitaryMatrix # fast convergence
    # contraction_gen=randomStandardComplexNormalMatrix # Gaussian mode

    dists, good_cgNs, costs, costs3 = [], [], [], []
    for cgN in cgNs:
        try:
            cgDat.extend([generateConditionGroupAverage(
                true_ms, KSigma_u, N=sampN,
                ugenerator=randomStandardComplexNormalVector,
                generator=contraction_gen
                # informative data mode
            ) for i in range(cgN - len(cgDat))])
            guess_ms, costcvx = learn_model(cgDat)
            dist = model_distance(true_ms, guess_ms)
            dists.append(dist)
            costs.append(generalized_cone_width(guess_ms))
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
    ax2.plot(cgNs, [generalized_cone_width(true_ms) for cgN in cgNs], 'k:', lw=3)
    ax1.set_title("Consistency")
    ax1.set_ylabel("Distance from True Model-Set")
    # ax1.set_xlabel("number of condition groups")

    ax2.set_title("Cost Function")
    ax2.set_ylabel("Log of Generalized Cone Width")
    ax2.set_xlabel("Number of Condition Groups")
    plt.show()


def learn_model(dat):
    # Fits a minimal model-set to explain a list of condition group averages

    # setup empty constraints list to populate as we go
    cons = []
    
    # we will use 4x4 real matrixes as a proxy for complex 2x2 matrices
    if True:
        # I suspect this method of defining the variables to be better
        # computationally.
        M_neg = cvx.Semidef(8)
        X_E = phi_image_structure(M_neg[0:4, 0:4], cons)
        X_P = -phi_image_structure(M_neg[0:4, 4:8], cons)
        X_PP = phi_image_structure(M_neg[4:8, 4:8], cons)
        X_J = phi_image_structure(cvx.Semidef(4), cons)

    else:
        # This method seems more intuitive, but introduces subtle differences
        # between the matrix M_neg and the matrix which must be positive semi
        # definate.
        X_E = phi_image_structure(cvx.Variable(4,4), cons)
        X_J = phi_image_structure(cvx.Variable(4,4), cons)
        X_P = phi_image_structure(cvx.Variable(4,4), cons)
        X_PP = phi_image_structure(cvx.Variable(4,4), cons)

        M_neg = cvx_block_matrix([[X_E, -X_P],[-X_P.T, X_PP]])
        cons.append(M_neg>>0)
        cons.append(X_J>>0)

    # Add all the inclusion constraints. One per condition group.
    for u, y in dat:
        # convert complex data into pseudo-complex real vector of twice the
        # size
        phi_xi = cvx.Constant(np.vstack([vec_phi(y), vec_phi(u)]))
        # convention is a little strange because it maintains 2x2 block
        # structure
        phi_u = cvx.Constant(vec_phi(u))
        
        cons.append(cvx.quad_form(phi_u, X_J) >= cvx.quad_form(phi_xi, M_neg)) 

    cons.append(cvx.trace(X_J) == 2.0)

    # Solve the convex problem

    prob = cvx.Problem(cvx.Minimize(-0.125*cvx.log_det(X_E)), cons)


    for solver_attempt in range(4):
        # This tries four different refinement levels, in hopes of getting away with the faster ones the majority of the time.
        # We can make no guarantees about the actual solvability of the problem.
        try:
            # There are only two options in CVX which can solve SD + EXP cone
            # problems: SCS and CVXOPT. However SCS never finds optimal
            # solutions, perhaps due to the unusual sub-problem we face in
            # uncertainty shape identification. The performance of CVXOPT
            # depends heavily on the refinement parameter, but it is unclear
            # what this actually does or why it is so important.
            res = prob.solve(
                solver="CVXOPT", verbose=True,
                abstol=1e-9, reltol=1e-9, feastol = 1.0e-9,
                refinement=[5,20,100,300][solver_attempt], kktsolver="robust", # {"chol", "robust"} robust=LDL
                max_iters=200)
            X_E_c = inverse_phi(X_E.value, enf_tol=1e-5, enf_hrm=True)
            # X_E = inverse_phi(phi_X_E, enf_tol=eft, enf_hrm=True)
            X_J_c = inverse_phi(X_J.value, enf_tol=1e-5, enf_hrm=True)
            # X_J = inverse_phi(phi_X_J, enf_tol=eft, enf_hrm=True)
            X_P_c = inverse_phi(X_P.value, enf_tol=1e-5, enf_hrm=False)
            # X_P = inverse_phi(phi_X_P, enf_tol=eft, enf_hrm=False)
            X_PP_c = inverse_phi(X_PP.value, enf_tol=1e-5, enf_hrm=True)
            # X_PP = inverse_phi(phi_X_PP, enf_tol=eft, enf_hrm=True)
        except (cvx.error.SolverError, AssertionError) as e:
            _, _, tb = sys.exc_info()
            traceback.print_tb(tb)
            if solver_attempt==3:
                return SingleFrequencyQuadricModelSet() , -float('NaN')
        else:
            break

    final_model= safe_extract_model(X_E_c, X_P_c, X_PP_c, X_J_c)
    worst_case_uncertainty_magnitude = validate_result(final_model, dat)
    print "worst case mag", worst_case_uncertainty_magnitude
    return final_model, res


def generateConditionGroupAverage(model, KSigma_u, N=100, ugenerator=randomStandardComplexNormalVector, generator=randomContractiveMatrix):
    # This is the data spoofing function, which generates random data using
    # the true model and some random sampler functions.

    # Note that the local P will only ever be measured with a single input.
    local_P = model.P + model.E.dot(generator()).dot(model.J)
    local_U = KSigma_u.dot(ugenerator())
    Y = []
    S = 0
    KSigma_eta = np.eye(2)*0.0 # noise is disabled for the moment.
    for i in range(N):
        local_eta = KSigma_eta.dot(randomStandardComplexNormalVector())
        Y.append(local_eta + local_P.dot(local_U))
    Y = sum(Y) / len(Y)
    return local_U, Y


def get_true_ms():
    # Here we just come up with an aribrary choice of true model. Clearly
    # tweaking it will change the result a bit, of course the random seed
    # changes the actual samples the most, but changing this can reaveal the
    # numerical behavior under different scalings. I've tweaked it so that it
    # is not as convenient as the identity everything case. I figured it is
    # important to demonstrate ability to learn off diagonal terms. Change it
    # too much and it will probably fail to solve unless you tweak the solver
    # parameters again.
    E = np.eye(2)
    E[0,0]=1.2
    # E[1,0]=0.3
    J=np.eye(2)*1.1
    J[1,1]=0.5
    J[0,1]=0.2
    true_ms = SingleFrequencyQuadricModelSet(
        P=np.eye(2) * 30,
        E=E,
        J=J
    )
    return true_ms


def getConvenientContraction():
    # chooses a randomly between unitary (exactly unity singular values) and
    # contractive (singular values less than unity) test matrices. This choice
    # is convenient in the sence that it causes the model to converge to the
    # true model in finite time, unlike a more realistic distribution.
    if np.random.random() > 0.5:
        return randomUnitaryMatrix()
    else:
        return randomContractiveMatrix()

if __name__ == "__main__":
    main()
    

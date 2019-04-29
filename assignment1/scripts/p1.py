import pandas
import numpy as np
import scipy
from scipy import special


def euclid_norm(vec):
    return np.sqrt(np.sum(vec ** 2))


def modified_bessel_function(v, z):
    return scipy.special.iv(v, z)


def cal_grad_pyf(f, T, N):
    normed2_f = []
    for i in range(0, N):
        cur_f = np.asarray([f[i], f[i+N]])
        normed2_f.append(euclid_norm(cur_f))
    assert len(normed2_f) == N
    normed2_f = np.asarray(normed2_f)

    I0 = special.iv(np.zeros(N), normed2_f)
    I1 = special.iv(np.ones(N), normed2_f)


    v1 = I0 * normed2_f
    f1_grad = np.cos(T) - (I1 * f[:N]) / v1
    f2_grad = np.sin(T) - (I1 * f[N:]) / v1

    return np.concatenate((f1_grad, f2_grad))


def kernel_func(x1, x2, ktype):
    if ktype == 'rbf':
        return np.exp(-1 * np.sum((x1 - x2)**2))

def generate_kernel(X, N, ktype):
    kmatrix = np.zeros((2 * N, 2 * N))
    for i in range(0, N):
        for j in range(0, N):
            kval = kernel_func(X[i], X[j], ktype)


def cal_hessian_matrix(f, N):
    normed2_f = []
    for i in range(0, N):
        cur_f = np.asarray([f[i], f[i+N]])
        normed2_f.append(euclid_norm(cur_f))
    assert len(normed2_f) == N
    normed2_f = np.asarray(normed2_f)

    I0 = special.iv(np.zeros(N), normed2_f)
    I1 = special.iv(np.ones(N), normed2_f)
    
    # f11_grad = - (((I0 ** 2) - (I1 ** 2)) * (f[:N] ** 2) + (I0 * I1) * (normed2_f - 2 * (f[:N] ** 2 / normed2_f))) / denominator
    # f22_grad = - (((I0 ** 2) - (I1 ** 2)) * (f[N:] ** 2) + (I0 * I1) * (normed2_f - 2 * (f[N:] ** 2 / normed2_f))) / denominator
    # f1221_grad = - (((I0 ** 2) - (I1 ** 2)) * f[:N] * f[N:] - 2 * I0 * I1 * (f[:N] * f[N:]) / normed2_f) / denominator

    v1 = (I0 ** 2) - (I1 ** 2)
    v2 = I0 * I1
    f1_pow2 = f[:N] ** 2
    f2_pow2 = f[N:] ** 2
    f1_f2 = f[:N] * f[N:]
    denominator = (I0 ** 2) * (normed2_f ** 2)

    f11_grad = - (v1 * f1_pow2 + v2 * (normed2_f - 2 * (f1_pow2 / normed2_f))) / denominator
    f22_grad = - (v1 * f2_pow2 + v2 * (normed2_f - 2 * (f2_pow2 / normed2_f))) / denominator
    f1221_grad = - (v1 * f1_f2 - 2 * v2 * f1_f2 / normed2_f) / denominator

    hessian = np.zeros((2 * N, 2 * N))

    for i in range(0, N):
        hessian[i][i] = f11_grad[i]
        hessian[i+N][i] = f1221_grad[i]
        hessian[i][i+N] = f1221_grad[i]
        hessian[i+N][i+N] = f22_grad[i] 
    
    return hessian


if __name__ == '__main__':
    # set data path
    user_id = 1
    dpath = './canvas/social-data/user_{0}.csv'.format(user_id)
    ktype = 'rbf'

    #　load data
    df = pandas.read_csv(dpath)
    X, Y, T = [], [], []
    
    for index, row in df.iterrows():
        X.append(row.tolist()[0])
        Y.append(row.tolist()[1])
        T.append(row.tolist()[2])
    N = len(X)

    # init f randomly
    f = np.random.rand(int(2 * N))

    # kernel function
    kernel = generate_kernel(X, N, ktype)

    # caculate Hessian Matrix
    W = cal_hessian_matrix(f, N)

    # calculate grad(p(y|f))
    grad_vec = cal_grad_pyf(f, T, N)


    # calculate grad(p(f|X, Y))

    # iter f -> get f*

    # calculate A

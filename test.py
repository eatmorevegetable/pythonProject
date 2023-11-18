import numpy as np
from pyDOE import lhs
import tensorflow as tf

if __name__ == '__main__':
    # A = np.array([[0., 1., 2., 3., 4.],
    #               [0., 1., 2., 3., 4.],
    #               [0., 1., 2., 3., 4.],
    #               [0., 1., 2., 3., 4.]])
    # B = np.array([[0., 0., 0., 0., 0.],
    #               [1., 1., 1., 1., 1.],
    #               [2., 2., 2., 2., 2.],
    #               [3., 3., 3., 3., 3.]])
    # xx1 = np.hstack((A[0:1, :].T, B[0:1, :].T))
    # xx2 = np.hstack((A[:, 0:1], B[:, 0:1]))
    # xx3 = np.hstack((A[:, -1:], B[:, -1:]))
    # X_u_train = np.vstack([xx1, xx2, xx3])
    # idx = np.random.choice(X_u_train.shape[0], 5, replace=False)
    # a = lhs(2, 5)
    # # print(xx1)
    # # print(xx2)
    # # print(xx3)
    # print(X_u_train)
    # X_u_train = X_u_train[idx, :]
    # print(X_u_train)
    # print(idx)
    # # print(a)
    # x_u = np.array([[1], [2], [3], [4]])
    # print(x_u[0])
    # cache = [[-1]*i for i in range(1, 7+2)]
    # cache[6][6] = 7
    num_equipment = [7, 4, 6, 8]
    num_people =[7, 6, 6, 6, 6, 6]
    krn = len(num_equipment)
    kpn = len(num_people)
    mkr = max(num_equipment)
    mkp = max(num_people)
    distribution_operation = np.zeros((kpn, mkp), dtype=int)
    for per_type_people in range(kpn):
        for i in range(num_people[per_type_people], mkp):
            distribution_operation[per_type_people][i] = 200

    print(distribution_operation)


import cv2
import numpy as np
import os
import time
import scipy.io as sio
import matplotlib.pyplot as plt
import math
import main_functions as main

def im2col(x,hh,ww,stride):

    """
    Args:
      x: image matrix to be translated into columns, (C,H,W)
      hh: filter height
      ww: filter width
      stride: stride
    Returns:
      col: (new_h*new_w,hh*ww*C) matrix, each column is a cube that will convolve with a filter
            new_h = (H-hh) // stride + 1, new_w = (W-ww) // stride + 1
    """

    c,h,w = x.shape
    new_h = (h-hh) // stride + 1
    new_w = (w-ww) // stride + 1
    col = np.zeros([new_h*new_w,c*hh*ww])

    for i in range(new_h):
       for j in range(new_w):
           patch = x[...,i*stride:i*stride+hh,j*stride:j*stride+ww]
           col[i*new_w+j,:] = np.reshape(patch,-1)
    return col

def col2im(mul,h_prime,w_prime,C):
    """
      Args:
      mul: (h_prime*w_prime*w,F) matrix, each col should be reshaped to C*h_prime*w_prime when C>0, or h_prime*w_prime when C = 0
      h_prime: reshaped filter height
      w_prime: reshaped filter width
      C: reshaped filter channel, if 0, reshape the filter to 2D, Otherwise reshape it to 3D
    Returns:
      if C == 0: (F,h_prime,w_prime) matrix
      Otherwise: (F,C,h_prime,w_prime) matrix
    """
    F = mul.shape[1]
    if(C == 1):
        out = np.zeros([F,h_prime,w_prime])
        for i in range(F):
            col = mul[:,i]
            out[i,:,:] = np.reshape(col,(h_prime,w_prime))
    else:
        out = np.zeros([F,C,h_prime,w_prime])
        for i in range(F):
            col = mul[:,i]
            out[i,:,:] = np.reshape(col,(C,h_prime,w_prime))

    return out

def get_mini_batch(im_train, label_train, batch_size):
    print("Training shape:", im_train.shape, label_train.shape)
    t, n = im_train.shape
    print(t, n)
    combined = np.hstack((im_train.T, label_train.T))
    np.random.shuffle(combined)

    print(combined.shape)
    # combined = combined.T
    # random_train, random_label = np.vsplit(combined, n-1)
    random_train = combined[:, :t].T
    # features -= features.mean(axis=0)
    random_label = combined[:,t].reshape((1, n))
    print(random_label)
    print(random_train.shape, random_label.shape)

    num_batches = (n // batch_size)
    remainder = n % batch_size
    incomplete = False


    mini_batch_x = []
    mini_batch_y = []
    if remainder != 0:
        incomplete = True

    for i in range(0, num_batches):
        mini_batch_x.append(random_train[:, (i*batch_size):((i+1)*batch_size)].T)
        # print(mini_batch_x[i].shape)
        mini_batch_y.append(random_label[:, (i*batch_size):((i+1)*batch_size)].T)

    if incomplete:
        mini_batch_x.append(random_train[:, ((num_batches-1)*batch_size):n].T)
        mini_batch_y.append(random_label[:, ((num_batches-1)*batch_size):n].T)

    return mini_batch_x, mini_batch_y


def fc(x, w, b):
    y = (w @ x).reshape(b.shape) + b
    return y


def fc_backward(dl_dy, x, w, b, y):
    try:
        dl_dx = dl_dy.T @ w
    except:
        dl_dx = dl_dy @ w
    # print(dl_dy.shape, x.shape)
    dl_dw = dl_dy @ x.T
    dl_db = dl_dy
    return dl_dx, dl_dw, dl_db


def loss_euclidean(y_tilde, y):
    y_real = np.zeros((10,))
    y_real[int(y)] = 1
    l = np.sum((y_real-y_tilde)**2)
    dl_dy = y_tilde - y_real.reshape((10,1))

    return l, dl_dy

def loss_cross_entropy_softmax(x, y):
    y_real = np.zeros((10,))
    y_real[int(y)] = 1
    y_tilde = np.zeros((10,))
    x = x.reshape((10,))
    total = 0

    total = np.sum(np.exp(np.copy(x)))
    y_tilde = np.exp(x) / total
    log = y_real*np.log(y_tilde)
    l = np.sum(log)
    dl_dy = y_tilde - y_real

    return l, dl_dy.reshape((10, 1))

def relu(x):
    # y = np.maximum(x, 0)
    y = np.where(x > 0, x, x * 0.01)
    # y = np.where(x > 0, x, 0)
    return y


def relu_backward(dl_dy, x, y):
    dx = np.ones_like(y)
    dx[y < 0] = 0.01
    return dx*dl_dy

# def conv_2(x, w_conv, b_conv):
#     k, k, c1, c2 = w_conv.shape
#     H, W, c1 = x.shape
#     y = np.zeros((H, W, c2))
#     x_n = x.reshape((14, 14))
#     x_pad = np.pad(x_n, (1,), 'constant', constant_values = (0))
#     # print(x_n, x_pad)
#     x_im = x_pad.reshape((1, 16, 16))
#     x_col = im2col(x_im, 3, 3, 1)
#
#     filter_col = np.reshape(w_conv, (3, -1))
#     # print(x_col.shape, filter_col.shape)
#     mul = x_col.dot(filter_col.T).reshape((14, 14, 3))
#     for i in range(3):
#         mul[i] += b_conv[i]
#     # print(mul.shape)
#     return y

def conv(x, w_conv, b_conv):
    # k, k, c1, c2 = w_conv.shape
    # H, W, c1 = x.shape
    # y = np.zeros((H, W, c2))
    # w1 = w_conv[:,:,:,0]
    # w2 = w_conv[:,:,:,1]
    # w3 = w_conv[:,:,:,2]
    # x_n = x.reshape((14, 14))
    # x_pad = np.pad(x_n, (1,), 'constant', constant_values = (0))
    # # print(x_n, x_pad)
    # x_im = x_pad.reshape((1, 16, 16))
    # x_col = im2col(x_im, 3, 3, 1)
    # # print(x_col.shape)
    # y1 = (w1.flatten() @ x_col.T).reshape((14, 14)) + b_conv[0]
    # # print(y1, (w1.flatten() @ x_col.T).reshape((14, 14)), b_conv[0])
    # y2 = (w2.flatten() @ x_col.T).reshape((14, 14)) + b_conv[1]
    # y3 = (w3.flatten() @ x_col.T).reshape((14, 14)) + b_conv[2]
    # # print("yay", y1.shape, y2.shape, y3.shape)
    # # tmp = y1[10, 10]
    # ys = []
    # ys.append(y1)
    # ys.append(y2)
    # ys.append(y3)
    # y = np.dstack(ys).reshape((14, 14, 3))
    # # print(y.shape)
    # # print(y[10,10,0], y1[10,10])
    # # x = 5/0
    # # print("Tmp:", tmp, y[10, 10, 2])
    # # print(y.shape)
    # return y

    # print(w_conv[0].reshape((3,3)), w_conv[1].reshape((3,3)), w_conv[2].reshape((3,3)))
    k, k, c1, c2 = w_conv.shape
    H, W, c1 = x.shape
    y = np.zeros((H, W, c2))
    x = x.reshape((H, W))
    x_pad = np.pad(x, (1,), 'constant', constant_values = (0))
    # print("H, W", H, W)
    # print(x_pad.shape)
    # x=5/0
    # x_pad = x_pad.reshape((H+2, W+2, 1))
    # print(w_conv[:,:,:,0].shape)
    # x = 5/0
    for i in range(H):
        for j in range(W):
            tmp = x_pad[i:i+3, j:j+3]
            for k in range(c2):
                tmp2 = np.dot(tmp.flatten(), w_conv[:,:,:,k].reshape((3,3)).flatten()) + b_conv[k]
                # print("tmp2:", tmp2.shape)
                y[i][j][k] = tmp2
                # break
        #     break
        # break
    # print(y, y.shape)
    return y

# def conv_backward_2(dl_dy, x, w_conv, b_conv, y):
#     # print(dl_dy.shape, x.shape, w_conv.shape, b_conv.shape, y.shape)
#     k, k, c1, c2 = w_conv.shape
#     H, W, c1 = x.shape
#     dl_dw = np.zeros(w_conv.shape)
#     dl_db = np.zeros(b_conv.shape)
#     xx, y, z, w = w_conv.shape
#     x_n = x.reshape((14, 14))
#     x_pad = np.pad(x_n, (1,), 'constant', constant_values = (0))
#
#     for i in range(H):
#         for j in range(W):
#             tmp = x_pad[i:i+3, j:j+3]
#             for k in range(c2):
#                 tmp2 = np.dot(tmp.flatten(), w_conv[k].reshape((3,3)).flatten()) + b_conv[k]
#                 # print("tmp2:", tmp2.shape)
#                 y[i][j][k] = tmp2
#                 # break
#         #     break
#         # break
#     # print(y, y.shape)
#     return y


def conv_backward(dl_dy, x, w_conv, b_conv, y):
    # print(dl_dy.shape, x.shape, w_conv.shape, b_conv.shape, y.shape)
    dl_dw = np.zeros(w_conv.shape)
    dl_db = np.zeros(b_conv.shape)
    xx, y, z, w = w_conv.shape
    H, W, c1 = x.shape
    # print("Dl_dy:", dl_dy.shape, x.shape)
    # print("H, W", xx, y, z)
    x_n = x.reshape((H, W))
    x_pad = np.pad(x_n, (1,), 'constant', constant_values = (0))
    x_im = x_pad.reshape((1, H+2, W+2))
    x_col = im2col(x_im, xx, y, z)
    # print(x_col, x_col.shape)
    f = H*W #196

    for i in range(w):
        # dl_dys.append(dl_dy[:,:,w].reshape((1, f)))
        # flats
        dl_dy_t = dl_dy[:,:,i].reshape((1, f))
        flat_t = (dl_dy_t @ x_col).reshape((xx,y,z))
        dl_dw[:,:,:,i] = flat_t


    # dl_dy1 = dl_dy[:,:,0].reshape((1, f))
    # dl_dy2 = dl_dy[:,:,1].reshape((1, f))
    # dl_dy3 = dl_dy[:,:,2].reshape((1, f))
    # # print("S:", dl_dy1.shape)
    # flat1 = (dl_dy1 @ x_col).reshape((xx,y,z))
    # flat2 = (dl_dy2 @ x_col).reshape((xx,y,z))
    # flat3 = (dl_dy3 @ x_col).reshape((xx,y,z))
    # dl_dw[:,:,:,0] = flat1
    # dl_dw[:,:,:,1] = flat2
    # dl_dw[:,:,:,2] = flat3



    # print("F1:", flat1)
    # print("F2:", flat2)
    # print("F3:", flat3)
    # print("dl_dw", dl_dw)
    # x=5/0
    # print(dl_dw)

    # old start
    # print(w_conv.shape)
    # x_old = x.reshape((14, 14))
    # for i in range(xx):
    #     for j in range(y):
    #         total = 0
    #         for k in range(z):
    #             for l in range(w):
    #                 # print(i, j, k, l)
    #                 # print(dl_dy.shape, dl_dy[i, j, k], x[k+i, j+l].shape)
    #                 total += dl_dy[i,j,k] * x[int(k+i), int(j+l), 0]
    #         dl_dw[i][j] = total
    # dl_db = np.zeros((b_conv.shape))
    # old end
    for i in range(w):
        # print("Shape test:", dl_dy[:,:,i])
        dl_db[i] = np.sum(dl_dy[:,:,i])
    # dl_db = np.sum(np.sum(dl_dy, axis = 0), axis = 0)
    # print("dl_db", dl_db)
    return dl_dw, dl_db

def pool2x2(x): # (14, 14, 3)
    pool_size = 2
    a, b, c = x.shape
    a = int(a / pool_size)
    b = int(b / pool_size)
    y = np.zeros((a, b, c))
    for i in range(a):
        for j in range(b):
            for k in range(c):
                tmp = x[(2*i):(2*(i+1)),(2*j):(2*(j+1)),k]
                # print(tmp.shape)
                max = np.max(tmp)
                # print(tmp, max)
                # x = 5/0
                y[i][j][k] = max
    return y # (7, 7, 3)

def pool2x2_backward(dl_dy, x, y): # (7, 7, 3), (14, 14, 3), (7, 7, 3)
    pool_size = 2
    a, b, c = x.shape
    a_new = int(a / pool_size)
    b_new = int(b / pool_size)
    y = np.zeros((a_new, b_new, c))
    dl_dx = np.zeros((a, b, c))
    for k in range(c):
        for i in range(a_new):
            for j in range(b_new):
                tmp = x[(2*i):(2*(i+1)),(2*j):(2*(j+1)),k]
                max = np.argmax(tmp)
                max_val = np.max(tmp)
                # print(tmp, max,  max_val, x[2*i, 2*j, k], x[2*i, (2*j) + 1, k],x[(2*i) + 1, 2*j, k], x[(2*i) + 1, (2*j) + 1, k])
                # x = 5/0
                if max == 0:
                    dl_dx[2*i, 2*j, k] = dl_dy[i, j, k]
                elif max == 1:
                    dl_dx[2*i, (2*j) + 1, k] = dl_dy[i, j, k]
                elif max == 2:
                    dl_dx[(2*i) + 1, 2*j, k] = dl_dy[i, j, k]
                elif max == 3:
                    dl_dx[(2*i) + 1, (2*j) + 1, k] = dl_dy[i, j, k]
                else:
                    x = 5/0
    return dl_dx #(14, 14, 3)


def flattening(x):
    # dl_dx = x.flatten(order='F').reshape(147, 1)
    dl_dx = x.flatten(order='F').reshape(-1, 1)
    # print(dl_dx.shape)
    return dl_dx


def flattening_backward(dl_dy, x, y):
    # print(x.shape, y.shape, dl_dy.shape)
    a, b, c = x.shape
    # print(a, b, c, dl_dy.shape)
    # return dl_dy.reshape((a, b, c))
    return dl_dy.reshape((a, b, c), order='F')


def train_slp_linear(mini_batch_x, mini_batch_y):
    learning_rate = .1
    decay_rate = .9
    n_iters = 10000
    w = np.random.normal(0, 1, size=(10, 196))
    b = np.zeros((10, 1))
    k = 0
    num_batches = len(mini_batch_x)
    print(num_batches)
    batch_size, _ = mini_batch_x[0].shape
    print(batch_size)
    for iter in range(n_iters):
        if iter % 1000 == 999:
            learning_rate *= decay_rate
        dL_dw = np.zeros((10, 196))
        dL_db = np.zeros((10, 1))
        for i in range(batch_size):
            x = mini_batch_x[k][i]
            y = mini_batch_y[k][i]

            pred = fc(x, w, b)

            l, dl_dy = loss_euclidean(pred, y)
            dl_dx, dl_dw, dl_db = fc_backward(dl_dy, x.reshape((196, 1)), w, b, y)
            dL_dw += dl_dw
            dL_db += dl_db
        # print("DL_dw", dL_dw[5], dL_db)
        k = (k+1) % num_batches
        w -= (dL_dw/batch_size) * learning_rate
        b -= (dL_db/batch_size) * learning_rate
    return w, b

def train_slp(mini_batch_x, mini_batch_y):
    learning_rate = .6
    decay_rate = .9
    n_iters = 8000
    w = np.random.normal(0, 1, size=(10, 196))
    b = np.zeros((10,1))
    k = 0
    num_batches = len(mini_batch_x)
    print(num_batches)
    batch_size, _ = mini_batch_x[0].shape
    print(batch_size)
    for iter in range(n_iters):
        if iter % 1000 == 999:
            learning_rate *= decay_rate
        dL_dw = np.zeros((10, 196))
        dL_db = np.zeros((10, 1))

        for i in range(batch_size):
            x = mini_batch_x[k][i]
            y = mini_batch_y[k][i]

            pred = fc(x, w, b)

            l, dl_dy = loss_cross_entropy_softmax(pred, y)
            dl_dx, dl_dw, dl_db = fc_backward(dl_dy, x.reshape((196, 1)), w, b, y)
            dL_dw += dl_dw
            dL_db += dl_db
        k = (k+1) % num_batches
        w -= (dL_dw/batch_size) * learning_rate
        b -= (dL_db/batch_size) * learning_rate
    return w, b

def train_mlp(mini_batch_x, mini_batch_y):
    learning_rate = .5
    decay_rate = .9
    n_iters = 10000 # 10000
    w1 = np.random.normal(0, 1, size=(30, 196))
    w2 = np.random.normal(0, 1, size=(10, 30))
    b1 = np.zeros((30,1))
    b2 = np.zeros((10, 1))
    k = 0
    num_batches = len(mini_batch_x)
    print(num_batches)
    batch_size, _ = mini_batch_x[0].shape
    print(batch_size)
    for iter in range(n_iters):
        if iter % 1000 == 999:
            learning_rate *= decay_rate
        dL_dw1 = np.zeros((30, 196))
        dL_db1 = np.zeros((30, 1))
        dL_dw2 = np.zeros((10, 30))
        dL_db2 = np.zeros((10, 1))

        for i in range(batch_size):
            x = mini_batch_x[k][i]
            y = mini_batch_y[k][i]

            pred1 = fc(x, w1, b1)
            # print("Pred1:", pred1.shape)
            pred2 = relu(pred1)
            # print("Pred2:", pred2.shape)
            y_pred = fc(pred2, w2, b2)
            # print("Y_pred:", y_pred.shape)

            l, dl_dy = loss_cross_entropy_softmax(y_pred, y)
            dl_dx2, dl_dw2, dl_db2 = fc_backward(dl_dy, pred2, w2, b2, y_pred)
            # print("2", dl_dx2.shape, pred1.shape, pred2.shape)
            dl_dx1 = relu_backward(dl_dx2.reshape((30,1)), pred1, pred2)
            # print("dl",dl_dx1.shape)
            dl_dx0, dl_dw1, dl_db1 = fc_backward(dl_dx1, x.reshape((196, 1)), w1, b1, pred2)
            # print('dl_dx2', dl_dx2.shape)
            # dl_dx1 = relu_backward(dl_dy, pred1, y) #orig good
            #dl_dx1 = relu_backward(dl_dy, dl_dx2, y)
            # dl_dx0, dl_dw1, dl_db1 = fc_backward(dl_dx2.reshape((30, 1)), x.reshape((196, 1)), w1, b1, y) #orig good
            #dl_dx0, dl_dw1, dl_db1 = fc_backward(dl_dx1.reshape((30, 1)), x.reshape((196, 1)), w1, b1, y)
            dL_dw1 += dl_dw1
            dL_db1 += dl_db1
            dL_dw2 += dl_dw2
            dL_db2 += dl_db2
        k = (k+1) % num_batches
        w1 -= (dL_dw1/batch_size) * learning_rate
        b1 -= (dL_db1/batch_size) * learning_rate
        w2 -= (dL_dw2/batch_size) * learning_rate
        b2 -= (dL_db2/batch_size) * learning_rate
    return w1, b1, w2, b2


def train_cnn(mini_batch_x, mini_batch_y):
    learning_rate = 0.5
    conv_learning_rate = 0.01
    decay_rate = .9
    n_iters = 1000
    w_conv = np.random.normal(0, 1, size=(3, 3, 1, 3))
    b_conv = np.random.normal(0, 1, size=(3,1))
    w_fc = np.random.normal(0, 1, size=(10, 147))
    b_fc = np.zeros((10,1))
    k = 0
    num_batches = len(mini_batch_x)
    print(num_batches)
    batch_size, _ = mini_batch_x[0].shape
    print(batch_size)
    for iter in range(n_iters):
        if iter % 1000 == 999:
            learning_rate *= decay_rate
        dL_dw_conv = np.zeros((3, 3, 1, 3))
        dL_db_conv = np.zeros((3, 1))
        dL_dw_fc = np.zeros((10, 147))
        dL_db_fc = np.zeros((10, 1))

        for i in range(batch_size):
            x = mini_batch_x[k][i]
            y = mini_batch_y[k][i]
            # print("W, b:", w_conv, b_conv)
            # import time
            t0 = time.time()
            pred1 = conv(x.reshape((14, 14, 1)), w_conv, b_conv)  # (14, 14, 3)
            t1 = time.time()
            # print("Pred1:", pred1)
            pred2 = relu(pred1)  # (14, 14, 3)
            t2 = time.time()
            # print("Pred2:", pred2)
            pred3 = pool2x2(pred2)  # (7, 7, 3)
            t3 = time.time()
            # print("Pred3:", pred3)
            pred4 = flattening(pred3)  # (147, 1)
            t4 = time.time()
            # print("Pred4:", pred4)
            y_pred = fc(pred4, w_fc, b_fc)  # (10, 1)
            t5 = time.time()
            # print("Y_pred:", y_pred)
            l_pred = np.argmax(y_pred)
            t6 = time.time()
            # print(l_pred.shape)

            l, dl_dy = loss_cross_entropy_softmax(y_pred, y)
            t7 = time.time()
            # print("dl_dy",dl_dy)
            dl_dx5, dl_dw_fc, dl_db_fc = fc_backward(dl_dy, pred4, w_fc, b_fc, y_pred)
            t8 = time.time()
            # print("dx5:", dl_dx5)
            # print("dw_fc", dl_dw_fc)
            # print("db_fc", dl_db_fc)
            dl_dx4 = flattening_backward(dl_dx5, pred3, pred4)
            t9 = time.time()
            # print("dx4:", dl_dx4)
            dl_dx3 = pool2x2_backward(dl_dx4, pred2, pred3)
            t10 = time.time()
            # print("dx3:", dl_dx3)
            dl_dx2 = relu_backward(dl_dx3, pred1, pred2)
            t11 = time.time()

            # print("dx2:", dl_dx2)

            dl_dw_conv, dl_db_conv = conv_backward(dl_dx2, x.reshape((14, 14, 1)), w_conv, b_conv, pred1)
            t12 = time.time()
            # print("dw", dl_dw_conv)
            # print("db", dl_db_conv)
            # print(t1-t0, t2-t1, t3-t2, t4-t3, t5-t4, t6-t5, t7-t6, t8-t7, t9-t8, t10-t9, t11-t10, t12-t11)
            # x = 5/0

            dL_dw_conv += dl_dw_conv
            dL_db_conv += dl_db_conv
            dL_dw_fc += dl_dw_fc
            dL_db_fc += dl_db_fc
        k = (k+1) % num_batches
        w_conv -= (dL_dw_conv/batch_size) * learning_rate
        b_conv -= (dL_db_conv/batch_size) * learning_rate
        w_fc -= (dL_dw_fc/batch_size) * learning_rate
        b_fc -= (dL_db_fc/batch_size) * learning_rate
    return w_conv, b_conv, w_fc, b_fc


def test_conv():
    filter = np.zeros((3, 3, 1, 2))
    count = 1
    for k in range(2):
        for i in range(3):
            for j in range(3):
                    filter[i,j,0,k] = count
                    count += 1
    print("Filter:", filter[:,:,:,1])
    b_conv = np.zeros((2,))
    b_conv[0] = 2
    b_conv[1] = -1
    image = np.zeros((3, 3, 1))
    image[0][0] = 1
    image[0][1] = 2
    image[0][2] = 3
    image[1][0] = 4
    image[1][1] = 5
    image[1][2] = 6
    image[2][0] = 7
    image[2][1] = 8
    image[2][2] = 9
    res = conv(image, filter, b_conv)
    print(res, res.shape)
    print(res[:,:,0])

def test_conv_backward():
    x = np.zeros((4,4,1))
    c = 1
    for i in range(4):
        for j in range(4):
            x[i,j,0] = c
            c += 1
    c=1
    dl_dy = np.zeros((4,4,3))
    for i in range(3):
        for j in range(4):
            for k in range(4):
                dl_dy[j,k,i] = c
        c += 1
    c = 1
    w_conv = np.zeros((3,3,1,3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                w_conv[j,k,0,i] = c
        c += 1
    b_conv = np.zeros((3,))
    b_conv[0] = 0
    b_conv[1] = 5
    b_conv[2] = 10
    y = None
    # dl_dy =
    dl_dw_conv, dl_db_conv = conv_backward(dl_dy, x, w_conv, b_conv, y)
    print(dl_dw_conv)
    print(dl_db_conv)
    print(dl_dw_conv[:,:,:,0])

def test_flattening():
    x = np.zeros((4, 4, 2))
    c = 1
    for i in range(2):
        for j in range(4):
            for k in range(4):
                x[j,k,i] = c
                c += 1
    print(x)
    res = flattening(x)
    print(res)
    y = None
    res2 = flattening_backward(res, x, y)
    print(res2)


if __name__ == '__main__':
    # main.main_slp_linear()
    # main.main_slp()
    # main.main_mlp()
    main.main_cnn()
    # test_flattening()
    # test_conv_backward()
    # test_conv()
    # pool_test = np.zeros((2, 2, 1))
    # pool_test[0, 0, 0] = 11
    # pool_test[0,1,0] = 8
    # pool_test[1,0,0] = 5
    # pool_test[1,1,0] = 4
    # after = pool2x2(pool_test)
    # print(after)
    # dl_dx = np.zeros((1,1,1))
    # dl_dx[0][0][0] = 10
    # back = pool2x2_backward(dl_dx, pool_test, after)
    # print(pool_test)
    # print(back, back.shape)

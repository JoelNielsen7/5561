import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy import interpolate
import random
from matplotlib import cm


def get_gradient(im):
    filter_x = np.array([[0,0,0],[1,0,-1],[0,0,0]])
    filter_y = np.array([[0,1,0],[0,0,0],[0,-1,0]])

    # pad the image with zeros around the boundary
    m, n = im.shape
    padded = np.zeros((m+2, n+2))
    # print(im.shape, padded.shape)
    padded[1:m+1,1:n+1] = im
    # print(padded)

    im_filtered_x = np.zeros((m,n))
    im_filtered_y = np.zeros((m,n))

    # print(padded[0:3, 0:3])

    for x in range(0, m):
        for y in range(0, n):
            # print(padded[m:m+4, n:n+4])
            im_filtered_x[x][y] = np.tensordot(padded[x:x+3, y:y+3], filter_x)
            im_filtered_y[x][y] = np.tensordot(padded[x:x+3, y:y+3], filter_y)


    return im_filtered_x, im_filtered_y


def find_match(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create()
    # get SIFTS
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # img1=cv2.drawKeypoints(img1,kp,img1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # kp2, des2 = sift.detect(img2, None)
    # img2 = cv2.drawKeypoints(img2, kp2, img2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imwrite('sift_keypoints2.jpg',img2)

    # KNN from first image to second
    neighbors1 = NearestNeighbors(n_neighbors = 2).fit(des1)

    distances1, indicies1 = neighbors1.kneighbors(des2)
    print(indicies1.shape)

    x1 = np.empty((0, 2), int)
    x2 = np.empty((0, 2), int)

    threshold = 0.8
    g = 0
    b = 0

    matches = []

    for i in range(0, len(indicies1)):
        # check for threshold
        if distances1[i][0]/distances1[i][1] < threshold:
            # need to do bidirectional so check this later
            matches.append((i, indicies1[i][0]))
            g += 1
        else:
            b += 1

    # KNN in reverse
    neighbors2 = NearestNeighbors(n_neighbors = 2).fit(des2)

    distances2, indicies2 = neighbors2.kneighbors(des1)
    print(indicies2.shape)

    x1 = np.empty((0, 2), int)
    x2 = np.empty((0, 2), int)

    threshold = 0.8
    g = 0
    b = 0


    for i in range(0, len(indicies2)):
        if distances2[i][0]/distances2[i][1] < threshold:
            # check if threshold is good AND it was good in reverse
            for x in matches:
                if x[0] == indicies2[i][0] and x[1] == i:
                    print("Bidi match")
                    g += 1
                    point2 = kp2[indicies2[i][0]].pt
                    point1 = kp1[i].pt

                    x1 = np.vstack((x1, point1))
                    x2 = np.vstack((x2, point2))
                else:
                    b += 1

    return x1, x2

def align_image_using_feature(x1, x2, ransac_thr, ransac_iter):
    l = len(x1)
    best_good = -1
    best_H = None

    best_x1 = None
    best_x2 = None

    # iterate
    for iter in range(ransac_iter):
        x1tmp = np.empty((0, 2), int)
        x2tmp = np.empty((0, 2), int)
        r1 = random.randint(0, l-1)
        r2 = random.randint(0, l-1)
        r3 = random.randint(0, l-1)
        # make sure points aren't the same
        while r1 == r2:
            r2 = random.randint(0, l-1)
        while r3 == r1 or r3 == r2:
            r3 = random.randint(0, l-1)
        p1 = x1[r1]
        p2 = x1[r2]
        p3 = x1[r3]
        p1p = x2[r1]
        p2p = x2[r2]
        p3p = x2[r3]
        # solve for p matrix
        A = np.array([  [p1[0], p1[1], 1, 0, 0, 0],
                        [0, 0, 0, p1[0], p1[1], 1],
                        [p2[0], p2[1], 1, 0, 0, 0],
                        [0, 0, 0, p2[0], p2[1], 1],
                        [p3[0], p3[1], 1, 0, 0, 0],
                        [0, 0, 0, p3[0], p3[1], 1]
                        ])

        b = np.array([p1p[0], p1p[1], p2p[0], p2p[1], p3p[0], p3p[1]]).T



        # try:
        #     x = np.linalg.inv(A.T @ A) @ A.T @ b
        # except:
        #     x = np.linalg.pinv(A.T @ A) @ A.T @ b

        x = np.linalg.pinv(A) @ b

        # make H from p
        H = np.array([  [x[0], x[1], x[2]],
                        [x[3], x[4], x[5]],
                        [0, 0, 1]
        ])
        # print("H:", H.shape)

        good = 0
        for i in range(l):
            # check all points to see if they are in the range
            pt = np.array([[x1[i][0], x1[i][1], 1]]).T
            pred = (H @ pt)
            if (pred[0][0] - x2[i][0])**2 + (pred[1][0] - x2[i][1])**2 < ransac_thr:
                x1tmp = np.vstack((x1tmp, x1[i]))
                x2tmp = np.vstack((x2tmp, x2[i]))
                good += 1
        # check if this configuration is the best
        if good > best_good:
            best_good = good
            best_H = H
            best_x1 = x1tmp
            best_x2 = x2tmp

    print("Best good:", best_good, l)
    return best_H, best_x1, best_x2

def warp_image(img, A, output_size):
    shape = img.shape
    # print(img.shape)
    # print(output_size)
    img_warped = np.empty((output_size[0], output_size[1]))

    for i in range(output_size[0]):
        for j in range(output_size[1]):
            # compute reverse warped pixels one at a time
            out = A @ np.array([j, i, 1])
            x = int(out[0])
            y = int(out[1])
            img_warped[i][j] = img[y][x]

    return img_warped

def validate_warped_image(img_warped, template):
    # this just prints the average error per pixel of the warped image
    size = img_warped.shape
    print(size)
    diff = abs(template - img_warped)
    plt.imshow(diff, cmap='coolwarm')
    plt.show()
    diff = np.sum(diff)
    print("Diff:", diff)
    # print(diff[0][2])
    avg_diff = diff / (size[0]*size[1])
    print("Average:", avg_diff)


def align_image(template, target, A):
    size = template.size
    # intialize variables
    targ_m, targ_n = target.shape
    m, n = template.shape
    img_warped = np.empty((m, n))
    delta_p = np.ones((6,1))

    # get gradient
    grad_x, grad_y = get_gradient(template)
    Hessian = np.zeros((6,6))

    for x in range(m):
        for y in range(n):
            #initialize jacobian
            jacobian = np.array([[x, y, 1, 0, 0, 0],
                                [0, 0, 0, x, y, 1]])
            # make gradient vector
            grad = np.array([grad_x[x][y], grad_y[x][y]])
            # compute Hessian
            steep = np.reshape(grad @ jacobian, (1, 6))
            add = steep.T @ steep
            Hessian += add

    # intialize variables
    thresh = 0.22
    errors = np.empty((1,0))
    err = 999
    last_err = 1000
    inc = 0
    max_inc = 5
    tmp_A = None
    its = 0

    # max of 100 iterations
    while np.linalg.norm(delta_p) > thresh and its < 100:
        its += 1
        img_warped = np.empty((m, n))
        # compute warped image
        for i in range(m):
            for j in range(n):
                out = np.matmul(A, np.array([j, i, 1]))
                x = int(out[0])
                y = int(out[1])
                img_warped[i][j] = target[y][x]


        # find the error
        im_err = template - img_warped #?good
        F = np.zeros((6, 1))
        # compute F
        for x in range(m):
            for y in range(n):
                jacobian = np.array([[x, y, 1, 0, 0, 0],
                                    [0, 0, 0, x, y, 1]])
                # make gradient vector
                grad = np.array([grad_x[x][y], grad_y[x][y]])
                # multiply by jacobian and error at that pixel
                # print(im_err[x][y])
                tmp = (np.matmul(grad, jacobian)).T * im_err[x][y]
                # add to F
                F += tmp.reshape((6, 1))

        # calculate delta p
        delta_p = np.matmul(np.linalg.inv(Hessian), F)
        print(delta_p)

        # make update matrix using delta_p values
        update = np.zeros((3,3))
        update[0][0] = delta_p[0] + 1
        update[0][1] = delta_p[1]
        update[0][2] = delta_p[2]
        update[1][0] = delta_p[3]
        update[1][1] = delta_p[4] + 1
        update[1][2] = delta_p[5]
        update[2][2] = 1

        # update A
        A = np.matmul(A, np.linalg.inv(update))

        # calculate average image error
        err = np.sum(np.abs(im_err)) / size
        # check if the error is increasing. If yes for 5 iterations, terminate
        if err > last_err:
            inc += 1
            # save the A matrix from the minimum before it increases
            if inc == 1:
                tmp_A = A
            elif inc == max_inc:
                return tmp_A, errors
        else:
            inc = 0
        last_err = err
        print("It, err:", its, err)
        errors = np.append(errors, err)

    return A, errors


def track_multi_frames(template, img_list):
    # To do
    ransac_thr = 1000
    ransac_iter = 1000
    A_list = []
    # find the initial A
    x1, x2 = find_match(template, img_list[0])
    A, x1new, x2new = align_image_using_feature(x1, x2, ransac_thr, ransac_iter)
    A_list.append(A)
    for img in img_list[1:]:
        # align and warp for each image
        A_refined, errors = align_image(template, img, A)
        template = warp_image(img, A_refined, template.shape).astype('uint8')
        A_list.append(A_refined)

    return A_list


def visualize_find_match(img1, img2, x1, x2, img_h=500):
    assert x1.shape == x2.shape, 'x1 and x2 should have same shape!'
    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    x1 = x1 * scale_factor1
    x2 = x2 * scale_factor2
    x2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    for i in range(x1.shape[0]):
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'b')
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'bo')
    plt.axis('off')
    plt.show()

def visualize_align_image(template, target, A, A_refined, errors=None):
    img_warped_init = warp_image(target, A, template.shape)
    img_warped_optim = warp_image(target, A_refined, template.shape)
    err_img_init = np.abs(img_warped_init - template)
    err_img_optim = np.abs(img_warped_optim - template)
    img_warped_init = np.uint8(img_warped_init)
    img_warped_optim = np.uint8(img_warped_optim)
    overlay_init = cv2.addWeighted(template, 0.5, img_warped_init, 0.5, 0)
    overlay_optim = cv2.addWeighted(template, 0.5, img_warped_optim, 0.5, 0)
    plt.subplot(241)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(242)
    plt.imshow(img_warped_init, cmap='gray')
    plt.title('Initial warp')
    plt.axis('off')
    plt.subplot(243)
    plt.imshow(overlay_init, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(244)
    plt.imshow(err_img_init, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.subplot(245)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(246)
    plt.imshow(img_warped_optim, cmap='gray')
    plt.title('Opt. warp')
    plt.axis('off')
    plt.subplot(247)
    plt.imshow(overlay_optim, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(248)
    plt.imshow(err_img_optim, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.show()

    if errors is not None:
        plt.plot(errors * 255)
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.show()


def visualize_track_multi_frames(template, img_list, A_list):
    bbox_list = []
    for A in A_list:
        boundary_t = np.hstack((np.array([[0, 0], [template.shape[1], 0], [template.shape[1], template.shape[0]],
                                        [0, template.shape[0]], [0, 0]]), np.ones((5, 1)))) @ A[:2, :].T
        bbox_list.append(boundary_t)

    plt.subplot(221)
    plt.imshow(img_list[0], cmap='gray')
    plt.plot(bbox_list[0][:, 0], bbox_list[0][:, 1], 'r')
    plt.title('Frame 1')
    plt.axis('off')
    plt.subplot(222)
    plt.imshow(img_list[1], cmap='gray')
    plt.plot(bbox_list[1][:, 0], bbox_list[1][:, 1], 'r')
    plt.title('Frame 2')
    plt.axis('off')
    plt.subplot(223)
    plt.imshow(img_list[2], cmap='gray')
    plt.plot(bbox_list[2][:, 0], bbox_list[2][:, 1], 'r')
    plt.title('Frame 3')
    plt.axis('off')
    plt.subplot(224)
    plt.imshow(img_list[3], cmap='gray')
    plt.plot(bbox_list[3][:, 0], bbox_list[3][:, 1], 'r')
    plt.title('Frame 4')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    template = cv2.imread('./Hyun_Soo_template.jpg', 0)  # read as grey scale image
    target_list = []
    for i in range(4):
        target = cv2.imread('./Hyun_Soo_target{}.jpg'.format(i+1), 0)  # read as grey scale image
        target_list.append(target)
    print(target_list)

    x1, x2 = find_match(template, target_list[0])
    visualize_find_match(template, target_list[0], x1, x2)

    ransac_thr = 100
    ransac_iter = 1000

    A, x1new, x2new = align_image_using_feature(x1, x2, ransac_thr, ransac_iter)

    visualize_find_match(template, target_list[0], x1new, x2new)


    img_warped = warp_image(target_list[0], A, template.shape)
    # cv2.imwrite('warped.jpg',img_warped)
    # print("HEREEE")
    plt.imshow(img_warped, cmap='gray', vmin=0, vmax=255)
    # plt.imshow(template, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.show()

    validate_warped_image(img_warped, template)

    # x = 5/0
    A_refined, errors = align_image(template, target_list[0], A)
    # A_refined, errors = test(template, target_list[0], A)
    visualize_align_image(template, target_list[0], A, A_refined, errors)

    A_list = track_multi_frames(template, target_list)
    visualize_track_multi_frames(template, target_list, A_list)

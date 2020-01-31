import cv2
import numpy as np
import matplotlib.pyplot as plt
import functools

def get_differential_filter():
    # To do
    filter_x = np.array([[0,0,0],[-1,0,1],[0,0,0]])
    filter_y = np.array([[0,1,0],[0,0,0],[0,-1,0]])
    return filter_x, filter_y


def filter_image(im, filter):
    # To do
    # pad the image with zeros around the boundary
    m, n = im.shape
    padded = np.zeros((m+2, n+2))
    print(im.shape, padded.shape)
    padded[1:m+1,1:n+1] = im
    print(padded)

    im_filtered = np.zeros((m,n))

    print(padded[0:3, 0:3])

    for x in range(0, m):
        for y in range(0, n):
            # print(padded[m:m+4, n:n+4])
            im_filtered[x][y] = np.tensordot(padded[x:x+3, y:y+3], filter)


    # threshold = functools.reduce(lambda:x, x if x > 0.5 else 0, im_filtered)
    # plt.imshow(im_filtered, cmap='gray')
    # plt.show()
    return im_filtered


def get_gradient(im_dx, im_dy):
    # To do

    grad_mag = np.sqrt(im_dx**2 + im_dy**2)
    print(grad_mag)
    # print("Max", np.max(grad_mag), np.min(grad_mag))
    ratio = im_dy/im_dx
    print("Ratio:", ratio.shape, ratio)
    grad_angle = np.arctan(ratio)
    grad_angle = np.nan_to_num(grad_angle, np.pi/2)
    # print("Angle max", np.max(grad_angle), np.min(grad_angle))

    # np.save("grad_mag.txt", grad_mag)
    # np.save("grad_angle.txt", grad_angle)
    return grad_mag, grad_angle


def build_histogram(grad_mag, grad_angle, cell_size):
    # To do
    # grad_angle = np.degrees(grad_angle)
    grad_angle = (np.degrees(grad_angle) + 180) % 180
    print("Grad angle", grad_angle.shape, grad_angle)
    print("Max:", np.max(grad_angle), np.min(grad_angle))
    m, n = grad_mag.shape
    M = int(m//cell_size)
    N = int(n//cell_size)
    leftover_M = m % cell_size
    leftover_N = n % cell_size
    print("M, N , m, n", M, N, m, n)
    ori_histo = np.zeros((M, N, 6))
    print("Ori", ori_histo.shape)
    for i in range(m - leftover_M):
        for j in range(n - leftover_N):
            a = grad_angle[i][j]
            if a >= 165 or a < 15:
                ind = 0
            elif a >= 15 and a < 45:
                ind = 1
            elif a >=45 and a < 75:
                ind = 2
            elif a >= 75 and a < 105:
                ind = 3
            elif a >= 105 and a < 135:
                ind = 4
            else:
                ind = 5
            ori_histo[i//cell_size, j//cell_size, ind] += grad_mag[i, j]
    return ori_histo


def get_block_descriptor(ori_histo, block_size):
    # To do
    e = 0.001
    M, N, b = ori_histo.shape
    ori_histo_normalized = np.zeros((M-1, N-1, 24))
    print(ori_histo)
    print(ori_histo.shape)
    print("TEST", ori_histo[0:2,0:2].shape)
    for i in range(M-1):
        for j in range(N-1):
            # print(ori_histo[i:i+block_size][j:j+block_size].shape)
            tmp = np.reshape(ori_histo[i:i+block_size, j:j+block_size], (6*(block_size**2),))
            denom = np.sqrt((np.sum(tmp**2)) + (e**2))
            for k in range(24):
                ori_histo_normalized[i][j][k] = tmp[k] / denom
            # tmp = np.stack(ori_histo[i:i+block_size][j:j+block_size])
            # tmp = np.concatenate(ori_histo[i][j], ori_histo[i+1][j], ori_histo[i][j+1], ori_histo[i+1][j+1])
            # print(tmp, tmp.shape)
    return ori_histo_normalized


def extract_hog(im):
    # convert grey-scale image to double format
    im = im.astype('float') / 255.0
    # print(type(im))
    # # To do
    filter_x, filter_y = get_differential_filter()
    im_x = filter_image(im, filter_x)
    im_y = filter_image(im, filter_y)



    grad_mag, grad_angle = get_gradient(im_x, im_y)

    # grad_mag = np.load("grad_mag.txt.npy")
    # grad_angle = np.load("grad_angle.txt.npy")

    print("Grad angle", grad_angle)

    ori_histo = build_histogram(grad_mag, grad_angle, 8)


    hog = get_block_descriptor(ori_histo, 2)

    # print(h for h in hog)
    # visualize to verify
    visualize_hog(im, hog, 8, 2)
    #
    return hog


# visualize histogram of each block
def visualize_hog(im, hog, cell_size, block_size):
    num_bins = 6
    max_len = 7  # control sum of segment lengths for visualized histogram bin of each block
    im_h, im_w = im.shape
    num_cell_h, num_cell_w = int(im_h / cell_size), int(im_w / cell_size)
    num_blocks_h, num_blocks_w = num_cell_h - block_size + 1, num_cell_w - block_size + 1
    histo_normalized = hog.reshape((num_blocks_h, num_blocks_w, block_size**2, num_bins))
    histo_normalized_vis = np.sum(histo_normalized**2, axis=2) * max_len  # num_blocks_h x num_blocks_w x num_bins
    angles = np.arange(0, np.pi, np.pi/num_bins)
    mesh_x, mesh_y = np.meshgrid(np.r_[cell_size: cell_size*num_cell_w: cell_size], np.r_[cell_size: cell_size*num_cell_h: cell_size])
    mesh_u = histo_normalized_vis * np.sin(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    mesh_v = histo_normalized_vis * -np.cos(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    for i in range(num_bins):
        plt.quiver(mesh_x - 0.5 * mesh_u[:, :, i], mesh_y - 0.5 * mesh_v[:, :, i], mesh_u[:, :, i], mesh_v[:, :, i],
                   color='white', headaxislength=0, headlength=0, scale_units='xy', scale=1, width=0.002, angles='xy')
    plt.show()


if __name__=='__main__':
    im = cv2.imread('einstein.jpg', 0)
    # print("hey")
    # plt.imshow(im, cmap='gray')#, vmin=0, vmax=1)
    # plt.show()


    hog = extract_hog(im)

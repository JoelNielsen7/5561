import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_differential_filter():
    filter_x = np.array([[0,0,0],[-1,0,1],[0,0,0]])
    filter_y = np.array([[0,1,0],[0,0,0],[0,-1,0]])
    return filter_x, filter_y


def filter_image(im, filter):
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


    return im_filtered


def get_gradient(im_dx, im_dy):
    grad_mag = np.sqrt(im_dx**2 + im_dy**2)
    print(grad_mag)

    ratio = im_dy/im_dx
    print("Ratio:", ratio.shape, ratio)
    grad_angle = np.arctan(ratio)
    grad_angle = np.nan_to_num(grad_angle, np.pi/2)

    return grad_mag, grad_angle


def build_histogram(grad_mag, grad_angle, cell_size):
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

    return ori_histo_normalized


def extract_hog(im):
    # convert grey-scale image to double format
    im = im.astype('float') / 255.0

    filter_x, filter_y = get_differential_filter()
    im_x = filter_image(im, filter_x)
    im_y = filter_image(im, filter_y)



    grad_mag, grad_angle = get_gradient(im_x, im_y)


    print("Grad angle", grad_angle)

    ori_histo = build_histogram(grad_mag, grad_angle, 8)


    hog = get_block_descriptor(ori_histo, 2)

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





def face_recognition(I_target, I_template):
    # extract hogs and visualize
    target_hog = extract_hog(I_target)
    template_hog = extract_hog(I_template)

    bounded_boxes = np.empty((0,3), int)

    iou_bounded_boxes = np.empty((0,3), int)



    x1, y1, z1 = target_hog.shape
    x2, y2, z2 = template_hog.shape


    print(x1, y1, z1, x2, y2, z2)


    # normalize
    for i in range(x1):
        for j in range(y1):
            target_mean = np.mean(target_hog[i][j])
            target_hog[i][j] -= target_mean
    for i in range(x2):
        for j in range(y2):
            template_mean = np.mean(template_hog[i][j])
            template_hog[i][j] -= template_mean

    # flatten and calculate NCC
    for i in range(x1-x2):
        for j in range(y1-y2):
            tmp1 = target_hog[i:i+x2, j:j+y2, :].flatten()
            tmp2 = template_hog.flatten()

            sum = np.dot(tmp1, tmp2) / (np.linalg.norm(tmp1) * np.linalg.norm(tmp2))

            # threshold
            bound = 0.29
            if abs(sum) >= bound:
                bounded_boxes = np.append(bounded_boxes, np.array([[j*8, i*8, abs(sum)]]), axis=0)

    total_pixels = x2*8 *y2*8

    # IoU non-max suppression
    while(True):
        if len(bounded_boxes) == 0:
            break
        max = -1
        ind = -1
        tmp = None
        # get max
        for i, box in enumerate(bounded_boxes):
            if abs(box[2]) > max:
                tmp = box
                max = abs(box[2])
                ind = i

        # add max to iou
        iou_bounded_boxes = np.append(iou_bounded_boxes, np.reshape(bounded_boxes[ind], (1, 3)), axis=0)

        # remove from original array
        bounded_boxes = np.delete(bounded_boxes, (ind),  axis=0)

        # calculat IoU and figure out if we need to suppress
        deletes = []
        print("Len:", len(bounded_boxes))
        for i, box in enumerate(bounded_boxes):
            xA = np.maximum(box[0], tmp[0])
            yA = np.maximum(box[1], tmp[1])
            xB = np.minimum(box[0]+(8*x2), tmp[0]+(8*x2))
            yB = np.minimum(box[1]+(8*y2), tmp[1]+(8*y2))


            areaI = np.maximum(0, xB - xA) * np.maximum(0, yB - yA)

            iou = areaI / float((total_pixels*2) - areaI)
            # print("IOU", iou, box[2], tmp[2],box[0], box[1], tmp[0], tmp[1])
            if iou > 0.4:
                # print("Zoinks", xA, yA, xB, yB, areaI, total_pixels, iou, box[2], tmp[2])
                deletes.append(i)
        deletes.reverse()

        # delete necessary boxes
        for d in deletes:
            bounded_boxes = np.delete(bounded_boxes, (d), axis=0)

    return iou_bounded_boxes


def box_visualization(I_target,bounding_boxes,box_size):

    hh,ww,cc=I_target.shape
    print("WW", ww)

    fimg=I_target.copy()
    for ii in range(bounding_boxes.shape[0]):

        # x1 = bounding_boxes[ii,0]  - box_size / 2
        # x2 = bounding_boxes[ii, 0] + box_size / 2
        # y1 = bounding_boxes[ii, 1] - box_size / 2
        # y2 = bounding_boxes[ii, 1] + box_size / 2
        # I changed this because I don't think it was correct, wanted box to start
        # with top left corner in the coordinates provided
        x1 = bounding_boxes[ii, 0]
        x2 = bounding_boxes[ii, 0] + box_size
        y1 = bounding_boxes[ii, 1]
        y2 = bounding_boxes[ii, 1] + box_size

        if x1<0:
            x1=0
        if x1>ww-1:
            x1=ww-1
        if x2<0:
            x2=0
        if x2>ww-1:
            x2=ww-1
        if y1<0:
            y1=0
        if y1>hh-1:
            y1=hh-1
        if y2<0:
            y2=0
        if y2>hh-1:
            y2=hh-1
        fimg = cv2.rectangle(fimg, (int(x1),int(y1)), (int(x2),int(y2)), (255, 0, 0), 1)
        cv2.putText(fimg, "%.2f"%bounding_boxes[ii,2], (int(x1)+1, int(y1)+2), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0, 255, 0), 2, cv2.LINE_AA)


    plt.figure(3)
    plt.imshow(fimg, vmin=0, vmax=1)
    plt.show()




if __name__=='__main__':
    I_target= cv2.imread('target.png', 0)
    #MxN image

    I_template = cv2.imread('template.png', 0)
    #mxn  face template

    bounding_boxes=face_recognition(I_target, I_template)

    I_target_c= cv2.imread('target.png')
    # MxN image (just for visualization)
    print("Template shape:", I_template.shape[0])
    box_visualization(I_target_c, bounding_boxes, I_template.shape[0]) #template.shape[0])
    #this is visualization code.

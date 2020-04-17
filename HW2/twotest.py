import numpy as np
from scipy.ndimage import shift, affine_transform
from scipy.interpolate import RectBivariateSpline
import cv2
# def InverseCompositionAffine(It, It1, threshold=0.005, iters=50):
'''
[input]
* It - Template image
* It1 - Current image
* threshold - Threshold for error convergence (default: 0.005)
* iters - Number of iterations for error convergence (default: 50)

[output]
* M - Affine warp matrix [2x3 numpy array]
'''

def test(It, It1, A):

    # It =  cv2.imread('./Hyun_Soo_template.jpg', 0)
    # It1 = cv2.imread('./warped.jpg', 0)
    iters = 50
    threshold = 0.05

    # Initial parameters
    # M = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    M = np.array([[A[0][0], A[0][1], A[0][2]],
                    [A[1][0], A[1][1], A[1][2]]])
    I = M

    # Step 3 - Compute the gradient for template
    gradient = np.dstack(np.gradient(It)[::-1])
    gradient = gradient.reshape(gradient.shape[0] * gradient.shape[1], 2).T

    # Step 4 - Evaluate jacobian parameters
    H, W = It.shape
    Jx = np.tile(np.linspace(0, W-1, W), (H, 1)).flatten()
    Jy = np.tile(np.linspace(0, H-1, H), (W, 1)).T.flatten()

    # Step 5 - Compute the steepest descent images
    steepest_descent = np.vstack([gradient[0] * Jx, gradient[0] * Jy,
        gradient[0], gradient[1] * Jx, gradient[1] * Jy, gradient[1]]).T

    # Step 6 - Compute the Hessian matrix
    hessian = np.matmul(steepest_descent.T, steepest_descent)

    # Iterate
    for i in range(iters):
        print(i)
        # Step 1 - Warp image
        warp_img = affine_transform(It1, np.flip(M)[..., [1, 2, 0]])

        # Step 2 - Compute error image with common pixels
        mask = affine_transform(np.ones(It1.shape), np.flip(M)[..., [1, 2, 0]])
        error_img = (mask * warp_img) - (mask * It)

        # Step 7/8 - Compute delta P
        delta_p = np.matmul(np.linalg.inv(hessian), np.matmul(steepest_descent.T, error_img.flatten()))

        # Step 9 - Update the parameters
        dM = np.vstack([delta_p.reshape(2, 3) + I, [0, 0, 1]])
        M = np.matmul(M, np.linalg.inv(dM))

        # Test for convergence
        if np.linalg.norm(delta_p) <= threshold: break

    return M, None

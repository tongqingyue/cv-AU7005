import cv2
import numpy as np

def gradient_xy(image):
    grad_x = np.zeros((3,3), dtype=np.float32)
    grad_y = np.zeros((3,3), dtype=np.float32)
    for i in [1,2,3]:
        for j in [1,2,3]:
            # compute gradient
            grad_x[i-1,j-1] = image[i+1,j]-image[i-1,j]
            grad_y[i-1,j-1] = image[i,j+1]-image[i,j-1]
    return grad_x, grad_y

def gradient_t(image0, image1):
    grad_t= np.zeros((3,3), dtype=np.float32)
    for i in [1,2,3]:
        for j in [1,2,3]:
            # compute gradient
            grad_t[i-1,j-1] = image1[i,j] - image0[i,j]
    return grad_t

def LK(image0, image1):
    grad_x, grad_y = gradient_xy(image0)
    grad_t = gradient_t(image0, image1)
    A = np.hstack((grad_x.reshape(-1,1), grad_y.reshape(-1,1)))
    b = -1*grad_t.reshape(-1,1)
    # two methods lead to same result
    motion = np.linalg.pinv(A)@ b
    # motion = np.linalg.inv(A.T@A)@A.T@b
    return motion

image_01 = np.array([[1,2,3,2,1],[1,2,3,2,1],[2,3,2,1,0],[2,3,2,1,0],[3,2,1,0,0]])
image_02 = np.array([[0,8,8,8,0],[8,9,9,9,8],[8,9,10,9,8],[8,9,9,9,8],[0,8,8,8,0]])
image_11 = np.array([[0,1,2,3,2],[0,1,2,3,2],[1,2,3,2,1],[1,2,3,2,1],[2,3,2,1,0]])
image_12 = np.array([[0,0,0,0,0],[8,8,8,0,0],[9,9,9,8,0],[9,10,9,8,0],[9,9,9,8,0]])

print(LK(image_01, image_11))
print(LK(image_02, image_12))
import cv2
import numpy as np

def structure_tensor(image):
    M = np.zeros((2,2), dtype=np.float32)
    for i in [1,2,3]:
        for j in [1,2,3]:
            # compute gradient
            grad_x = image[i+1,j]-image[i-1,j]
            grad_y = image[i,j+1]-image[i,j-1]
            M += np.array([[grad_x**2, grad_x*grad_y], [grad_x*grad_y, grad_y**2]])
    return M

def cornerness(M, k):
    return np.linalg.det(M) - k*np.trace(M)**2


image_01 = np.array([[1,2,3,2,1],[1,2,3,2,1],[2,3,2,1,0],[2,3,2,1,0],[3,2,1,0,0]])
image_02 = np.array([[0,8,8,8,0],[8,9,9,9,8],[8,9,10,9,8],[8,9,9,9,8],[0,8,8,8,0]])

M1 = structure_tensor(image_01)
eigenvalues1, _ = np.linalg.eig(M1)
print(M1, eigenvalues1)

M2 = structure_tensor(image_02)
eigenvalues2, _ = np.linalg.eig(M2)
print(M2, eigenvalues2)

C1 = cornerness(M1, 0.05) # set k=0.05
C2 = cornerness(M2, 0.05)

print(C1, C2)
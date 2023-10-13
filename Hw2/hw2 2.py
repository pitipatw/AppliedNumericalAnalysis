import polyscope as ps
import polyscope.imgui as psim
import numpy as np
import gpytoolbox as gpy
import random
import scipy as sp

# Polyscope user interface initial steps
ps.set_program_name("HW2:  ARAP")
ps.set_verbosity(0)
ps.set_use_prefs_file(False)
ps.init() # only call once

# Load triangle mesh
vertices, faces = gpy.read_mesh("Hw2/armadillo.obj")
# vertices, faces = gpy.read_mesh("Hw2/test.obj")
# print(vertices)

# print(vertices)
ps_mesh0 = ps.register_surface_mesh("Rest mesh", vertices, faces, transparency=0.1) # display mesh
ps_mesh = ps.register_surface_mesh("Deformed mesh", vertices, faces) # display mesh
nv = vertices.shape[0] # number of vertices
print(nv)
# Handles: hands, feet, chest, nose, back, tail of armadillo.obj
handles =  [96,8,1487,839,671,30,1120,1309] 
# handles = [0,20]
ps_cloud = ps.register_point_cloud("Point cloud", vertices[handles,:]) # display handles

# Area weights -- weights w_ij in the ARAP model
A = gpy.doublearea(vertices,faces)
row = np.concatenate((faces[:,0],faces[:,1],faces[:,2]))
col = np.concatenate((faces[:,1],faces[:,2],faces[:,0]))
A3 = np.concatenate((A,A,A))
W = sp.sparse.csr_matrix((A3, (row,col)),shape=(nv,nv))
W = W + W.T

# Homework problem 3c:
#    Input:
#        restPositions     --- nv x 3 matrix of vertex positions before deformation (p)
#        deformedPositions --- nv x 3 matrix of vertex positions after deformation (p')
#    Output:
#        R --- nv x 3 x 3 tensor of 3x3 matrices, one per vertex, with best-fit ARAP rotations
#    Hint:  check out numpy.linalg.svd(...)
def findARAPRotations(restPositions,deformedPositions):
    R = np.zeros((nv,3,3))
    ##### ANSWER GOES HERE #####
    for i in range(nv):
        Pi = restPositions[i,:] - restPositions
        Pi_prime = deformedPositions[i,:] - deformedPositions

        Di = np.diagflat(W.toarray()[i,:])
        Si = np.matmul(np.matmul(Pi.T,Di),Pi_prime)
        # perform singular value decomposition Si = UiSViT
        U,S,Vh = np.linalg.svd(Si)
        # R[i,:,:] = Vh*U.T
        R[i,:,:] = np.matmul(Vh.T,U.T)
        # print("R")
        # print(R[i,:,:].T @ R[i,:,:])
        # print("ENDR")
    ##### END ANSWER #####
    return R

# Homework problem 4e:
#    Input:  None (note "global" variables W, handles above)
#    Output:
#        A --- nv x nv sparse matrix so that A^{-1}B (together with ARAPRHS below) solves the ARAP global problem
#    Hint:  check out scipy.sparse.diags(...)
def ARAPMatrix(): # the LHS
    ##### ANSWER GOES HERE #####
    sumW = np.sum(W.toarray(),axis=1).squeeze()
    L = sp.sparse.diags(sumW)
    L = L - W.toarray()
    print(np.shape(L))
    return L 

# Homework problem 4e:
#    Input:  None (note "global" variables W, handles above)
#    Output:
#        B --- nv x 3 sparse matrix so that A^{-1}B (together with ARAPMatrix above) solves the ARAP global problem
def ARAPRHS(restPositions,R,handlePositions):
    B = np.zeros((nv,3))
    ##### ANSWER GOES HERE #####
    # b = np.repeat(a[:, :, np.newaxis], 3, axis=2)

    for i in range(nv):
        Bi = np.zeros((1,3)) 
        for j in range(nv):
            Bi += W[i,j]/2 * np.matmul( R[i,:,:] + R[j,:,:] , restPositions[i,:] - restPositions[j,:])
        B[i,:] = Bi.copy()

    ##### END ANSWER #####
    return B

def solveARAP(handlePositions):
    deformedPositions = np.copy(vertices)
    restPositions = np.copy(vertices)

    ##### ANSWER GOES HERE: PRECOMPUTATION #####
    print("Precomputing ARAP matrix...")
    L = ARAPMatrix()
    newL = np.delete(L,handles,0)
    newL = np.delete(newL,handles,1)
    # print(np.linalg.slogdet(A))
    # print("A")
    # print(A)
    print("Precomputing ARAP matrix...done")
    ##### END ANSWER: PRECOMPUTATION #####
    
    for i in range(10): # one ARAP iteration
        print("ARAP iteration",i)

        ##### ANSWER GOES HERE: ARAP ITERATION #####
        R = findARAPRotations(restPositions,deformedPositions)
        B = ARAPRHS(restPositions,R,handlePositions)
        print(np.shape(L))
        B = B - (L[:,handles] @ handlePositions)

        deformedPositions[~np.isin(np.arange(nv), handles)] = np.linalg.solve(newL, B[~np.isin(np.arange(nv), handles)])
        deformedPositions[handles] = handlePositions
        ##### END ANSWER: ARAP ITERATION #####

    return deformedPositions

def callback():
    if (psim.Button("Randomly deform")):
        # Randomly move the handles
        newhandles = vertices[handles,:]
        # np.random.seed(42)
        newhandles = newhandles + np.random.normal(size=newhandles.shape)*10
        
        # Solve the ARAP problem and update the 3d model
        ps_mesh.update_vertex_positions(solveARAP(newhandles))
        ps_cloud.update_point_positions(newhandles)
        print("DONE")
ps.set_user_callback(callback)
ps.show()
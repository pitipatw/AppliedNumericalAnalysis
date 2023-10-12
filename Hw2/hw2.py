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
vertices, faces = gpy.read_mesh("armadillo.obj")
ps_mesh0 = ps.register_surface_mesh("Rest mesh", vertices, faces, transparency=0.1) # display mesh
ps_mesh = ps.register_surface_mesh("Deformed mesh", vertices, faces) # display mesh
nv = vertices.shape[0] # number of vertices

# Handles: hands, feet, chest, nose, back, tail of armadillo.obj
handles =  [96,8,1487,839,671,30,1120,1309] 
ps_cloud = ps.register_point_cloud("Point cloud", vertices[handles,:]) # display handles

# Area weights -- weights w_ij in the ARAP model
A = gpy.doublearea(vertices,faces)
row = np.concatenate((faces[:,0],faces[:,1],faces[:,2]))
col = np.concatenate((faces[:,1],faces[:,2],faces[:,0]))
A3 = np.concatenate((A,A,A))
W = sp.sparse.csr_matrix((A3, (row,col)),shape=(nv,nv))
W = W + W.T
# print("W")
# print(W)
# print("---")
# print(W[(0,3)])
# print("END W")
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
        pi = restPositions[i,:]
        pi_prime = deformedPositions[i,:]
        eij = pi - restPositions
        eij_prime = pi_prime - deformedPositions

        Pi = eij
        Pi_prime = eij_prime
        #construct Di
        Di = np.zeros((nv,nv))
        for j in range(nv): 
            Di[j,j] = W[(i,j)]

        Si = Pi*Di*Pi_prime
# perform singular value decomposition Si = UiViT
        U,S,Vh =  np.linalg.svd(Si)
        R[i,:,:] = U*Vh.T
    
    ##### END ANSWER #####

    return R

# Homework problem 4e:
#    Input:  None (note "global" variables W, handles above)
#    Output:
#        A --- nv x nv sparse matrix so that A^{-1}B (together with ARAPRHS below) solves the ARAP global problem
#    Hint:  check out scipy.sparse.diags(...)
def ARAPMatrix():
    ##### ANSWER GOES HERE #####
    return sp.sparse.eye(nv) # replace this code!
    ##### END ANSWER #####


# Homework problem 4e:
#    Input:  None (note "global" variables W, handles above)
#    Output:
#        B --- nv x 3 sparse matrix so that A^{-1}B (together with ARAPMatrix above) solves the ARAP global problem
def ARAPRHS(restPositions,R,handlePositions):
    B = np.zeros((nv,3))
    
    ##### ANSWER GOES HERE #####
    
    ##### END ANSWER #####
    
    return B

def solveARAP(handlePositions):
    deformedPositions = vertices
    
    ##### ANSWER GOES HERE: PRECOMPUTATION #####
    
    ##### END ANSWER: PRECOMPUTATION #####
    
    for i in range(10): # one ARAP iteration
        print("ARAP iteration",i)
        
        ##### ANSWER GOES HERE: ARAP ITERATION #####
        
        ##### END ANSWER: ARAP ITERATION #####
    
    return deformedPositions

def callback():
    if (psim.Button("Randomly deform")):
        # Randomly move the handles
        newhandles = vertices[handles,:]
        newhandles = newhandles + np.random.normal(size=newhandles.shape)*10
        
        # Solve the ARAP problem and update the 3d model
        ps_mesh.update_vertex_positions(solveARAP(newhandles))
        ps_cloud.update_point_positions(newhandles)

ps.set_user_callback(callback)
ps.show()
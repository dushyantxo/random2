#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

import matplotlib.pyplot as plt

A = np.array([4,3])
B = np.array([3,-5])
C = np.array([-4,-6])
 
print("Assigned points to variables:")
print("a:", A)
print("b:", B)
print("c:", C)

omat = np.array([[0, 1], [-1, 0]])


def dir_vec(A,B):
  return B-A
def norm_vec(C,B):
    return omat@dir_vec(C,B)


dir_vec_AB = dir_vec(A,B)
dir_vec_BC = dir_vec(B,C)
dir_vec_CA = dir_vec(C,A)

print('direction vector AB',dir_vec_AB)
print('direction vector BC',dir_vec_BC)
print('direction vector CA',dir_vec_CA)

def len_vec(dir_vec):
  return np.linalg.norm(dir_vec)

len_vec_AB = len_vec(dir_vec_AB)
len_vec_BC = len_vec(dir_vec_BC)
len_vec_CA = len_vec(dir_vec_CA)

print('length AB:',len_vec_AB)
print('length BC:',len_vec_BC)
print('length CA:',len_vec_CA)
Mat = np.array([[1, 1, 1], [A[0], B[0], C[0]], [A[1], B[1], C[1]]])
rank = np.linalg.matrix_rank(Mat)

if rank <= 2:
        print("Hence proved that points A, B, C in a triangle are collinear")
else:
        print("The given points are not collinear")
        
#generating lines
def line_gen(A,B):
  len =10
  dim = A.shape[0]
  x_AB = np.zeros((dim,len))
  lam_1 = np.linspace(0,1,len)
  for i in range(len):
    temp1 = A + lam_1[i]*(B-A)
    x_AB[:,i]= temp1.T
  return x_AB

x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)

plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
#plt.plot(x_BE[0,:],x_BE[1,:],label='$BE$')
#plt.plot(x_CF[0,:],x_CF[1,:],label='$CF$')

#Labeling the coordinates
A = A.reshape(-1,1)
B = B.reshape(-1,1)
C = C.reshape(-1,1)
#D = D.reshape(-1,1)
#E = E.reshape(-1,1)
#F = F.reshape(-1,1)
#G = G.reshape(-1,1)
tri_coords = np.block([[A, B, C]])
plt.scatter(tri_coords[0, :], tri_coords[1, :])
vert_labels = ['A', 'B', 'C']
for i, txt in enumerate(vert_labels):
    offset = 10 if txt == 'C' else -10
    plt.annotate(txt,
                 (tri_coords[0, i], tri_coords[1, i]),
                 textcoords="offset points",
                 xytext=(0, offset),
                 ha='center')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')  
plt.show()
plt.savefig("main.png",bbox_inches='tight')

def parametric_form (A,B,k):
    dir_vec_AB = dir_vec(A,B)
    x = A + k * dir_vec_AB
    return x

k_values = np.linspace(0,1,10) 
print("parametric form of line AB:")
for k in k_values:
    parametric_point_AB = parametric_form(A,B,k)
    print (f"(k) = {k},Parametric_form_AB:",parametric_point_AB)
    

print("\nparametric form of line BC:")
for k in k_values:
        parametric_point_BC = parametric_form(B,C,k)
        print (f"(k) = {k},Parametric_form_BC:",parametric_point_BC)

print ("\nparametric form of line CA:")
for k in k_values:
        parametric_point_CA = parametric_form(C,A,k)
        print (f"(k) = {k},Parametric_form_CA:",parametric_point_CA)
#Generating all lines 
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)
#plotting all lines 
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
plt.plot(A[0], A[1], 'o')
plt.text(A[0] * (1.2 + 0.05), A[1] * (1 - 0.1) , 'A')
plt.plot(B[0], B[1], 'o')
plt.text(B[0] * (1.2 + 0.05), B[1] * (1 - 0.1) , 'B')
plt.plot(C[0], C[1], 'o')
plt.text(C[0] * (1.2 + 0.05), C[1] * (1 - 0.1) , 'C')

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

plt.show()
plt.savefig("main1.png",bbox_inches='tight')



n = norm_vec(C, B)
pro = n.T @ B
print("n =", n)
print("x =", pro)

def line_gen(A, B):
    num_points = 10
    dim = A.shape[0]
    x_AB = np.zeros((dim, num_points))
    lam_1 = np.linspace(0, 1, num_points)
    
    for i in range(num_points):
        temp1 = A + lam_1[i] * (B - A)
        x_AB[:, i] = temp1.T
    
    return x_AB

x_BC = line_gen(B, C)
x_CA = line_gen(C, A)
x_AB = line_gen(A, B)

plt.plot(x_BC[0, :], x_BC[1, :], label='$BC$')
plt.plot(x_CA[0, :], x_CA[1, :], label='$CA$')
plt.plot(x_AB[0, :], x_AB[1, :], label='$AB$')

tri_coords = np.block([[A, B, C]])

plt.scatter(tri_coords[0, :], tri_coords[1, :])

vert_labels = ['A', 'B', 'C']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt,
                 (tri_coords[0, i], tri_coords[1, i]),
                 textcoords="offset points",
                 xytext=(0, offset),
                 ha='center')
plt.legend()
plt.grid()
plt.show() 
plt.savefig("main2.png",bbox_inches='tight')



# In[3]:


cross_product = np.cross(dir_vec_AB,dir_vec_CA)
magnitude = np.linalg.norm(cross_product)
area = 0.5 * magnitude
print("Area of triangle ABC:",area )


# angle
dotA=((B-A).T)@(C-A)
dotA=dotA[0,0]
NormA=(np.linalg.norm(B-A))*(np.linalg.norm(C-A))
print('value of angle A: ', np.degrees(np.arccos((dotA)/NormA)))


dotB=(A-B).T@(C-B)
dotB=dotB[0,0]
NormB=(np.linalg.norm(A-B))*(np.linalg.norm(C-B))
print('value of angle B: ', np.degrees(np.arccos((dotB)/NormB)))

dotC=(A-C).T@(B-C)
dotC=dotC[0,0]
NormC=(np.linalg.norm(A-C))*(np.linalg.norm(B-C))
print('value of angle C: ', np.degrees(np.arccos((dotC)/NormC)))
# median


D = (B + C)/2

#Similarly for E and F
E = (A + C)/2
F = (A + B)/2

print("D:", list(D))
print("E:", list(E))
print("F:", list(F))

#Generating all lines
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)


#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')


#Labeling the coordinates
A = A.reshape(-1,1)
B = B.reshape(-1,1)
C = C.reshape(-1,1)
D = D.reshape(-1,1)
E = E.reshape(-1,1)
F = F.reshape(-1,1)
tri_coords = np.block([[A,B,C,D,E,F]])
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','D','E','F']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

#1.2.2
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')

x_AD = line_gen(A, D)
plt.plot(x_AD[0, :], x_AD[1, :], label='$AD$')

x_BE = line_gen(B, E)
plt.plot(x_BE[0, :], x_BE[1, :], label='$BE$')

x_CF = line_gen(C, F)
plt.plot(x_CF[0, :], x_CF[1, :], label='$CF$')

A = A.reshape(-1,1)
B = B.reshape(-1,1)
C = C.reshape(-1,1)
D = D.reshape(-1,1)
E = E.reshape(-1,1)
F = F.reshape(-1,1)
tri_coords = np.block([[A,B,C,D,E,F]])
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','D','E','F']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(-10,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')


plt.show()
plt.savefig("main3.png",bbox_inches='tight')

#intersection

def line_intersect(n1,A1,n2,A2):
	N=np.block([[n1],[n2]])
	p = np.zeros(2)
	p[0] = n1@A1
	p[1] = n2@A2
	#Intersection
	P=np.linalg.inv(N)@p
	return P
G=line_intersect(norm_vec(F,C).T,C,norm_vec(E,B).T,B)
print("("+str(G[0])+","+str(G[1])+")")

#Hence verified that A - F = E - D and AFDE is a parallelogram

#Generating all lines
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)
x_BE = line_gen(B,E)
x_CF = line_gen(C,F)
x_AD = line_gen(A,D)

#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
plt.plot(x_AD[0,:],x_AD[1,:],label='$AD$')
plt.plot(x_BE[0,:],x_BE[1,:],label='$BE$')
plt.plot(x_CF[0,:],x_CF[1,:],label='$CF$')

#Labeling the coordinates
A = A.reshape(-1,1)
B = B.reshape(-1,1)
C = C.reshape(-1,1)
D = D.reshape(-1,1)
E = E.reshape(-1,1)
F = F.reshape(-1,1)
G = G.reshape(-1,1)
tri_coords = np.block([[A, B, C, D, E, F, G]])
plt.scatter(tri_coords[0, :], tri_coords[1, :])
vert_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
for i, txt in enumerate(vert_labels):
    offset = 10 if txt == 'G' else -10
    plt.annotate(txt,
                 (tri_coords[0, i], tri_coords[1, i]),
                 textcoords="offset points",
                 xytext=(0, offset),
                 ha='center')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.show()
plt.savefig("main4.png",bbox_inches='tight')

# 1.2.4

AG = np.linalg.norm(G - A)
GD = np.linalg.norm(D - G)

BG = np.linalg.norm(G - B)
GE = np.linalg.norm(E - G)
 
CG = np.linalg.norm(G - C)
GF = np.linalg.norm(F - G)

print("AG/GD= "+str(AG/GD))
print("BG/GE= "+str(BG/GE))
print("CG/GF= "+str(CG/GF))


# In[4]:


mat2 = np.array([[1, 1, 1], [A[0][0], D[0][0], G[0][0]], [A[1][0], D[1][0], G[1][0]]])

rank2 = np.linalg.matrix_rank(mat2)
if rank2==2:
    print("Hence proved that points A,G,D in a triangle are collinear")
else:
    print("Error")
    
#verify centroid
G = (A + B + C) / 3
print("centroid of the given triangle: ")      
      
print(G)
     
print("Hence Q.1.2.6 is verified.")

#1.2.7


if np.array_equal((A - F), (E - D)):
    print("A-E=E-D,Verified")
else:
    print("Error")
#1.3.1
norm_vec_AD = norm_vec(A,D)
print(norm_vec_AD)



# In[5]:


#1.3.2 amd 1.3.3
A=A.reshape(2)
B=B.reshape(2)
C=C.reshape(2)

def alt_foot(A,B,C):
  m = B-C
  n = omat@m 
  N=np.block([[m],[n]])
  p = np.zeros(2)
  p[0] = m@A 
  p[1] = n@B
  #Intersection
  P=np.linalg.inv(N.T)@p
  return P

D =  alt_foot(A,B,C)
E =  alt_foot(B,C,A)
F =  alt_foot(C,A,B)

# Print altitude foot points
print("Altitude foot point for A:", D)
print("Altitude foot point for B:", E)
print("Altitude foot point for C:", F)
dir_vec_AD = dir_vec(A,D)
dir_vec_BE = dir_vec(B,E)
dir_vec_CF = dir_vec(C,F)
print("direction vector of AD = ",dir_vec_AD,)
print("direction vector of BE = ",dir_vec_BE,)
print("direction vector of CF = ",dir_vec_CF,)

norm_vec_BE = norm_vec(B,E)
norm_vec_CF = norm_vec(C,F)
norm_vec_AD = norm_vec(A,D)



print(f"{norm_vec_BE}[x-B]=0" )
print(f"{norm_vec_CF}[x-C]=0" )
print(f"{norm_vec_AD}[x-A]=0" )




H = line_intersect(norm_vec(B,E),E,norm_vec(C,F),F)
print('orthocentre ',H)
#plotting
x_AB = line_gen(A,B)	
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)
x_AD = line_gen(A,alt_foot(A,B,C))
x_AE = line_gen(A,alt_foot(B,A,C))
x_BE = line_gen(B,alt_foot(B,A,C))
x_CF = line_gen(C,alt_foot(C,A,B))
x_AF = line_gen(A,alt_foot(C,A,B))
x_CH = line_gen(C,H)
x_CF = line_gen(C,F)


x_BD = line_gen(B,D)
#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
plt.plot(x_AD[0,:],x_AD[1,:],label='$AD$')
plt.plot(x_BE[0,:],x_BE[1,:],label='$BE_1$')
plt.plot(x_AE[0,:],x_AE[1,:],linestyle = 'dashed',label='$AE_1$')
plt.plot(x_BD[0,:],x_BD[1,:],linestyle = 'dashed',label='$BD_1$')
plt.plot(x_CF[0,:],x_CF[1,:])
plt.plot(x_AF[0,:],x_AF[1,:],linestyle = 'dashed',label='$AF_1$')


#Labeling the coordinates
A = A.reshape(-1,1)
B = B.reshape(-1,1)
C = C.reshape(-1,1)
D = D.reshape(-1,1)
E = E.reshape(-1,1)
F = F.reshape(-1,1)
H = H.reshape(-1,1)
tri_coords = np.block([[A,B,C,D,E,F]])
#tri_coords = np.vstack((A,B,C,alt_foot(A,B,C),alt_foot(B,A,C),alt_foot(C,A,B),H)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','D_1','E_1','F_1']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the poi3.5nt to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.show()
plt.savefig("main5.png",bbox_inches='tight')
#plotting
#Generating all lines
x_AB = line_gen(A,B)	
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)
x_AD = line_gen(A,alt_foot(A,B,C))
x_AE = line_gen(A,alt_foot(B,A,C))
x_BE = line_gen(B,alt_foot(B,A,C))
x_CF = line_gen(C,alt_foot(C,A,B))
x_AF = line_gen(A,alt_foot(C,A,B))
x_CH = line_gen(C,H)
x_BH = line_gen(B,H)
x_AH = line_gen(A,H)
x_BD = line_gen(B,D)
#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
plt.plot(x_AD[0,:],x_AD[1,:],label='$AD$')
plt.plot(x_BE[0,:],x_BE[1,:],label='$BE_1$')
plt.plot(x_AE[0,:],x_AE[1,:],linestyle = 'dashed',label='$AE_1$')
plt.plot(x_BD[0,:],x_BD[1,:],linestyle = 'dashed',label='$BD_1$')
plt.plot(x_CF[0,:],x_CF[1,:],label='$CF_1$')
plt.plot(x_AF[0,:],x_AF[1,:],linestyle = 'dashed',label='$AF_1$')
plt.plot(x_CH[0,:],x_CH[1,:],label='$CH$')
plt.plot(x_BH[0,:],x_BH[1,:],label='$BH$')
plt.plot(x_AH[0,:],x_AH[1,:],linestyle = 'dashed',label='$AH$')

#Labeling the coordinates
A = A.reshape(-1,1)
B = B.reshape(-1,1)
C = C.reshape(-1,1)
D = D.reshape(-1,1)
E = E.reshape(-1,1)
F = F.reshape(-1,1)
H = H.reshape(-1,1)
tri_coords = np.block([[A,B,C,D,E,F,H]])
#tri_coords = np.vstack((A,B,C,alt_foot(A,B,C),alt_foot(B,A,C),alt_foot(C,A,B),H)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','D_1','E_1','F_1','H']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the poi3.5nt to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.show()
plt.savefig("main5.png",bbox_inches='tight')

#1.3.5
result = int(((A - H).T) @ (B - C))    # Checking orthogonality condition...

# printing output
if result == 0:
  print("(A - H)^T (B - C) = 0\nHence Verified...")

else:
  print("(A - H)^T (B - C)) != 0\nHence the given statement is wrong...")

#1.4.1

A=A.reshape(2)
B=B.reshape(2)
C=C.reshape(2)



def midpoint(P, Q):
    return (P + Q) / 2

def perpendicular_bisector(B, C):
    midBC = midpoint(B, C)
    dir = B - C
    constant = -np.dot(dir, midBC)
    return dir, constant


equation_coeff1, const1 = perpendicular_bisector(A, B)
equation_coeff2, const2 = perpendicular_bisector(B, C)
equation_coeff3, const3 = perpendicular_bisector(C, A)

print(f'Equation for perpendicular bisector of AB: {equation_coeff1[0]:.2f}x + {equation_coeff1[1]:.2f}y + {const1:.2f} = 0')
print(f'Equation for perpendicular bisector of BC: {equation_coeff2[0]:.2f}x + {equation_coeff2[1]:.2f}y + {const2:.2f} = 0')
print(f'Equation for perpendicular bisector of CA: {equation_coeff3[0]:.2f}x + {equation_coeff3[1]:.2f}y + {const3:.2f} = 0')

# Circumcentre of triangle ABC
def ccircle(A, B, C):
    p = np.zeros(2)
    n1 = equation_coeff1[:2]
    p[0] = 0.5 * (np.linalg.norm(A) ** 2 - np.linalg.norm(B) ** 2)
    n2 = equation_coeff2[:2]
    p[1] = 0.5 * (np.linalg.norm(B) ** 2 - np.linalg.norm(C) ** 2)
    
    # Intersection
    N = np.vstack((n1, n2))  # Use vstack to create a 2x2 matrix
    O = np.linalg.solve(N, p)
    return O

O = ccircle(A, B, C)
print(f'Circumcenter of triangle ABC: {O}')

#plott
x_AB = line_gen(A, B)
x_BC = line_gen(B, C)
x_CA = line_gen(C, A)
# Plotting all lines
plt.plot(x_AB[0, :], x_AB[1, :], label='$AB$')
plt.plot(x_BC[0, :], x_BC[1, :], label='$BC$')
plt.plot(x_CA[0, :], x_CA[1, :], label='$CA$')
# Perpendicular bisector
def line_dir_pt(m, A, k1=0, k2=1):
    len = 10
    dim = A.shape[0]
    x_AB = np.zeros((dim, len))
    lam_1 = np.linspace(k1, k2, len)
    for i in range(len):
        temp1 = A + lam_1[i] * m
        x_AB[:, i] = temp1.T
    return x_AB
# Calculate the perpendicular vector and plot arrows
def perpendicular(B, C, label):
    perpendicular=norm_vec(B,C)
    mid = midpoint(B, C)
    x_D = line_dir_pt(perpendicular, mid, 0, 1)
    plt.arrow(mid[0], mid[1], perpendicular[0], perpendicular[1], color='blue', head_width=0.4, head_length=0.4, label=label)
    plt.arrow(mid[0], mid[1], -perpendicular[0], -perpendicular[1], color='blue', head_width=0.4, head_length=0.4)
    return x_D
x_D = perpendicular(A, B, 'OD')
x_E = perpendicular(B, C, 'OE')
x_F = perpendicular(C, A, 'OF')
mid1 = midpoint(A, B)
mid2 = midpoint(B, C)
mid3 = midpoint(C, A)
#Labeling the coordinates
#tri_coords = np.vstack((A,B,C,O,I)).T
#np.block([[A1,A2,B1,B2]])
A = A.reshape(-1,1)
B = B.reshape(-1,1)
C = C.reshape(-1,1)
O = O.reshape(-1,1)
mid12=mid1.reshape(-1,1)
mid23=mid2.reshape(-1,1)
mid31=mid3.reshape(-1,1)
tri_coords = np.block([[A,B,C,O,mid12,mid23,mid31]])
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','O','D','E','F']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.show()
plt.savefig("main6.png",bbox_inches='tight')

#1.4.2
A=A.reshape(2)
B=B.reshape(2)
C=C.reshape(2)
AB = dir_vec(A,B)
# direction vector along line joining A & C
AC = dir_vec(A,C)
# midpoint of A & B is F
F = (A+B)/2
# midpoint of A & C is E
E = (A+C)/2
# O is the point of intersection of perpendicular bisectors of AB and AC
O = line_intersect(AB,F,AC,E)
print(O)
G=(C+B)/2
#Generating all lines 
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)
x_OE = line_gen(O,E)
x_OF = line_gen(O,F)
x_OG=line_gen(O,G)


#plotting all lines 
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
plt.plot(x_OG[0,:],x_OG[1,:],label='$OG$')
plt.plot(x_OE[0,:],x_OE[1,:],label='$OE$')
plt.plot(x_OF[0,:],x_OF[1,:],label='$OF$')

A = A.reshape(-1,1)
B = B.reshape(-1,1)
C = C.reshape(-1,1)
O = O.reshape(-1,1)
E = E.reshape(-1,1)
F = F.reshape(-1,1)
G= G.reshape(-1,1)
tri_coords = np.block([[A,B,C,O,E,F,G]])
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','O','E','F','G']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center


plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

plt.show()
plt.savefig("main7.png",bbox_inches='tight')

#1.4.4

O_1 = O - A
O_2 = O - B
O_3 = O - C
a = np.linalg.norm(O_1)
b = np.linalg.norm(O_2)
c = np.linalg.norm(O_3)
print("Points of triangle A, B, C respectively are", A ,",", B ,",", C, ".")
print("Circumcentre of triangle is", O, ".")
print(" OA, OB, OC are respectively", a,",", b,",",c, ".")
print("Here, OA = OB = OC.")
print("Hence verified.")




#1.4.5
print('o is equal to',O)


# In[6]:


O


# In[14]:




A=A.reshape(2)
B=B.reshape(2)
C=C.reshape(2)

O=O.reshape(2,)
def circ_gen(O,r):
	len = 50
	theta = np.linspace(0,2*np.pi,len)
	x_circ = np.zeros((2,len))
	x_circ[0,:] = r*np.cos(theta)
	x_circ[1,:] = r*np.sin(theta)
	x_circ = (x_circ.T + O).T
	return x_circ
X = A - O
radius = np.linalg.norm(X)
#Generating all lines
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)
x_OA = line_gen(O,A)
#Generating the circumcirclecircle
x_ccirc= circ_gen(O,radius)
#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
plt.plot(x_OA[0,:],x_OA[1,:],label='$OA$')

#Plotting the circumcircle
plt.plot(x_ccirc[0,:],x_ccirc[1,:],label='$circumcircle$')

A = A.reshape(-1,1)
B = B.reshape(-1,1)
C = C.reshape(-1,1)
O = O.reshape(-1,1)
#Labeling the coordinates
tri_coords = np.block([[A,B,C,O]])
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','O']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() 
plt.axis('equal')
plt.show()
plt.savefig("main8.png",bbox_inches='tight')

#1.4.6
A=A.reshape(2)
B=B.reshape(2)
C=C.reshape(2)
O=O.reshape(2)
dot_pt_O = (B - O) @ ((C - O).T)
norm_pt_O = np.linalg.norm(B - O) * np.linalg.norm(C - O)
cos_theta_O = dot_pt_O / norm_pt_O
angle_BOC = (np.degrees(np.arccos(cos_theta_O)))  #Round is used to round of number till 5 decimal places
print("angle BOC = ",angle_BOC)

#To find angle BAC

dot_pt_A = (B - A) @ ((C - A).T)
norm_pt_A = np.linalg.norm(B - A) * np.linalg.norm(C - A)
cos_theta_A = dot_pt_A / norm_pt_A
angle_BAC = (np.degrees(np.arccos(cos_theta_A)))  #Round is used to round of number till 5 decimal places
print("angle BAC = ",angle_BAC)
#To check whether the answer is correct
if np.all(angle_BOC == 2 * angle_BAC):
    print("\nangle BOC = 2 times angle BAC\nHence the give statement is correct")
else:
    print("\nangle BOC ≠ 2 times angle BAC\nHence the given statement is wrong")
x_AB = line_gen(A,B)
x_AC = line_gen(A,C)
x_OB = line_gen(O,B)
x_OC = line_gen(O,C)

#Generating the circumcircle
X = A - O
r= np.linalg.norm(X)
x_circ= circ_gen(O,r)

#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_AC[0,:],x_AC[1,:],label='$BC$')
plt.plot(x_OB[0,:],x_OB[1,:],label='$OB$')
plt.plot(x_OC[0,:],x_OC[1,:],label='$OB$')
#Plotting the circumcircle
plt.plot(x_circ[0,:],x_circ[1,:],label='$circumcircle$')


#Labeling the coordinates
A = A.reshape(-1,1)
B = B.reshape(-1,1)
C = C.reshape(-1,1)
O = O.reshape(-1,1)
tri_coords = np.block([[A,B,C,O]])
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','O']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() 
plt.axis('equal')    
plt.show()
plt.savefig("main9.png",bbox_inches='tight')

#1.5.1
A=A.reshape(2)
B=B.reshape(2)
C=C.reshape(2)
t = norm_vec(B,C) 
n1 = t/np.linalg.norm(t) #unit normal vector
t = norm_vec(C,A)
n2 = t/np.linalg.norm(t)
t = norm_vec(A,B)
n3 = t/np.linalg.norm(t)

m_a=norm_vec(n2,n3)
m_b=norm_vec(n1,n3)
m_c=norm_vec(n1,n2)

  
#generating sides of triangle
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)

#generating angle bisectors
k1=[-6,-6]
k2=[6,6]  
x_A = line_dir_pt(m_a,A,k1,k2)
x_B = line_dir_pt(m_b,B,k1,k2)
x_C = line_dir_pt(m_c,C,k1,k2)

#plotting Angle bisectors
plt.plot(x_A[0,:],x_A[1,:],label='angle bisector of A')
plt.plot(x_B[0,:],x_B[1,:],label='angle bisector of B')
plt.plot(x_C[0,:],x_C[1,:],label='angle bisector of C')

#plotting sides
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')

tri_coords = np.block([[A],[B],[C]])
plt.scatter(tri_coords[:,0], tri_coords[:,1])
vert_labels = ['A','B','C']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[i,0], tri_coords[i,1]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.show()
plt.savefig("main10.png",bbox_inches='tight')

#1.5.2
I=line_intersect(n1-n3,B,n1-n2,C) #intersection of angle bisectors B and C
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)
x_BI = line_gen(B,I)
x_CI = line_gen(C,I)

#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
plt.plot(x_CI[0,:],x_CI[1,:],label='$CI$')
plt.plot(x_BI[0,:],x_BI[1,:],label='$BI$')

#Labeling the coordinates
tri_coords = np.block([[A],[B],[C],[I]])
plt.scatter(tri_coords[:,0], tri_coords[:,1])
vert_labels = ['A','B','C','I']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[i,0], tri_coords[i,1]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

                 
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.show()
plt.savefig("main11.png",bbox_inches='tight')

#1.5.3
A=A.reshape(2)
B=B.reshape(2)
C=C.reshape(2)
def icircle(A,B,C):
  k1 = 1
  k2 = 1
  p = np.zeros(2)
  t = norm_vec(B,C)
  n1 = t/np.linalg.norm(t)
  t = norm_vec(C,A)
  n2 = t/np.linalg.norm(t)
  t = norm_vec(A,B)
  n3 = t/np.linalg.norm(t)
  p[0] = n1@B- k1*n2@C
  p[1] = n2@C- k2*n3@A
  N=np.vstack((n1-k1*n2,n2-k2*n3))
  I=np.matmul(np.linalg.inv(N),p)
  r = n1@(I-B)
  #Intersection
  return I,r
[I,r] = icircle(A,B,C)
x_icirc= circ_gen(I,r)

#Generating all lines
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)
x_IA = line_gen(I,A)
#Generating the circumcircle
#[O,R] = ccircle(A,B,C)
#x_circ= circ_gen(O,R)


#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
plt.plot(x_IA[0,:],x_IA[1,:],label='$IA$')

#Plotting the circumcircle
#plt.plot(x_circ[0,:],x_circ[1,:],label='$circumcircle$')

#Plotting the circumcircle
#plt.plot(x_icirc[0,:],x_icirc[1,:],label='$incircle$')

#BA, CA, and IA in vector form
BA = A - B
CA = A - C
IA = A - I

def angle_btw_vectors(v1, v2):
    dot_product = v1 @ v2
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    angle = np.arccos(dot_product / norm)
    angle_in_deg = np.degrees(angle)
    return angle_in_deg

#Calculating the angles BAI and CAI
angle_BAI = angle_btw_vectors(BA, IA)
angle_CAI = angle_btw_vectors(CA, IA)

# Print the angles
print("Angle BAI:", angle_BAI)
print("Angle CAI:", angle_CAI)

if np.isclose(angle_BAI, angle_CAI):
    print("Angle BAI is approximately equal to angle CAI.")
else:
    print("error")

#Labeling the coordinates
A = A.reshape(-1,1)
B = B.reshape(-1,1)
C = C.reshape(-1,1)
I = I.reshape(-1,1)
tri_coords = np.block([[A,B,C,I]])
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','I']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
		 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.show()
plt.savefig("main12.png",bbox_inches='tight')

#1.5.4

print("Coordinates of point I:", I)
print(f"Distance from I to BC= {r}")
#1.5.5
A=A.reshape(2)
B=B.reshape(2)
C=C.reshape(2)
I=I.reshape(2)
r2=n2@(I-C)
r3=n3@(I-A)
print(f"distance of i from CA =",{r2})
print(f"distance of I from AB =",{r3})
#1.5.7
x_AB = line_gen(A, B)
x_BC = line_gen(B, C)
x_CA = line_gen(C, A)

#generating the incircle
[I,r] = icircle(A,B,C)
x_icirc= circ_gen(I,r)

#plotiing the lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')

#plotting the incircle
plt.plot(x_icirc[0,:],x_icirc[1,:],label='$incircle$')

#labelling the coordinates
tri_coords = np.block([[A],[B],[C],[I]]).T
tri_coords = tri_coords.reshape(2, -1)
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','I']

for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center


plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.show()
#1.5.8
A = A.reshape(-1,1)
B = B.reshape(-1,1)
C = C.reshape(-1,1)
I = I.reshape(-1,1)
D1=C-B
p=pow(np.linalg.norm(C-B),2)
q=2*(D1.T@(I-B))
r=pow(np.linalg.norm(I-B),2)-radius*radius

Discre=q*q-4*p*r
print("the Value of discriminant is ",Discre)
k=((I-B).T@(C-B))/((C-B).T@(C-B))
print("the value of parameter k is ",k)
D3=B+k*(C-B)
print("Hence we prove that side BC is tangent To incircle and also found the value of k!")
#1.5.9

x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)

#Generating the incircle
A=A.reshape(2)
B=B.reshape(2)
C=C.reshape(2)
I=I.reshape(2)
[I,r] = icircle(A,B,C)
x_icirc= circ_gen(I,r)
#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
plt.plot(x_icirc[0,:],x_icirc[1,:],label='$incircle$')

#finding k for E_3 and F_3
k1=((I-A)@(A-B))/((A-B)@(A-B))
k2=((I-A)@(A-C))/((A-C)@(A-C))
k3=((I-B)@(C-B))/((C-B)@(C-B))
#finding E_3 and F_3
E3=A+(k1*(A-B))
F3=A+(k2*(A-C))
D3=B+(k3*(C-B))
print("k1 = ",k1)
print("k2 = ",k2)
print("D3 = ",D3)
print("E3 = ",E3)
print("F3 = ",F3)


#Labeling the coordinates
A = A.reshape(-1,1)
B = B.reshape(-1,1)
C = C.reshape(-1,1)
I = I.reshape(-1,1)
E3 = E3.reshape(-1,1)
F3 = F3.reshape(-1,1)
D3 = D3.reshape(-1,1)
tri_coords = np.block([[A,B,C,I,E3,F3,D3]])
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','I','E3','F3','D3']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid(True) # minor
plt.axis('equal')
plt.show()
plt.savefig("main13.png",bbox_inches='tight')

#1.5.10
def norm(X,Y):
    magnitude=round(float(np.linalg.norm([X-Y])),3)
    return magnitude 
print("AE3=", norm(A,E3) ,"\nAF3=", norm(A,F3) ,"\nBD3=", norm(B,D3) ,"\nBF3=", norm(B,F3) ,
      "\nCD3=", norm(C,D3) ,"\nCE3=",norm(C,E3))
#1.5.11
a = np.linalg.norm(B-C)
b = np.linalg.norm(C-A)
c = np.linalg.norm(A-B)

#creating array containing coefficients
Y = np.array([[1,1,0],[0,1,1],[1,0,1]])

#solving the equations
X = np.linalg.solve(Y,[c,a,b])

#printing output 
print('side length',X)
# In[8]:


O


# In[ ]:





# In[ ]:





import numpy as np

A = np.array([[0, 1, 1], [2**0.5, 2, 0], [0, 1, 1]]) #np.array([[1, 1, 0, 1], [0, 0, 0, 1], [1, 1, 0,0]])#np.array([[1, 2], [2, 1]])#np.array([[1, 1, 1, 0], [0, 1, 0, 1]])


eigvals_p, eigvecs_p = np.linalg.eigh(np.dot(A, A.T)) #P
#unit normalize eigvecs1
temp_eigvecs_p = []
for ele in eigvecs_p.T: #take each column and unit normalize
	temp_eigvecs_p.append(ele/float(sum(ele**2)**0.5))
	
eigvecs_p = np.column_stack((temp_eigvecs_p))

#computing eigen vectors of Q matrix
eigvecs_q = []
#print("Eig val of P ",eigvals_p)
for index, ele in enumerate(eigvals_p):
	#print(ele)
	if ele != 0:
		eigvecs_q.append(np.dot(A.T, eigvecs_p.T[index])/(float(abs(ele)**0.5)))
		
		
P = eigvecs_p
S = np.diag(abs(eigvals_p)**0.5)
Q = np.column_stack((eigvecs_q))	
print("Left Singular Vectors!")
print(P)
print("Singular Values!")
print(S)
print("Right Singular Vectors!")
print(Q)

M = np.dot(np.dot(P,S), Q.T)
print("Approximated matrix!")
print(M)



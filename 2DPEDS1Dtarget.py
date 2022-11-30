#Solve 2D PEDS for 1D target system
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

#dimension of PEDS
N = 2
Omega = 1./N*np.ones((N,N))
alpha = 0.5

def fs(x):
    a1 = 0
    a2 = -12.0
    a3 = 2.0/3.0
    a4 = 0.25
    return -4*a4*x**3 -3*a3*x**2 -2*a2*x -a1

def Fv(Xv):
    return fs(Xv)

def RHS(t,Xv):
    return np.matmul(Omega,Fv(Xv)) - alpha*np.matmul((np.identity(N)-Omega),Xv)

#vectorised X
Xv = np.array([1,1])

x1 = np.linspace(-8, 8, 30)
x2 = np.linspace(-8, 8, 30)

X1, X2 = np.meshgrid(x1,x2)
Xf = np.array([X1,X2])
F = fs(Xf)
print(F.shape)
print(Omega.shape)
Fo = np.tensordot(Omega,F,axes =((1),(0)))

R = np.tensordot(Omega,fs(Xf),axes =((1),(0))) \
    - alpha*np.tensordot((np.identity(N)-Omega),Xf,axes=((1),(0)))

a1 = 0
a2 = -12.0
a3 = 2.0/3.0
a4 = 0.25
fig = plt.figure()
ax = fig.add_subplot(111)
x = np.arange(-8,8,0.1)
ax.plot(x,a4*x**4+a3*x**3+a2*x**2+a1*x)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.quiver(X1, X2, F[0], F[1])
#ax.xaxis.set_ticks([])
#ax.yaxis.set_ticks([])
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.axis('equal')

fig = plt.figure()
ax2 = fig.add_subplot(111)
ax2.quiver(X1, X2, Fo[0], Fo[1])
#ax.xaxis.set_ticks([])
#ax.yaxis.set_ticks([])
ax2.set_xlabel('X1')
ax2.set_ylabel('X2')

fig = plt.figure()
ax3 = fig.add_subplot(111)                                                                                                                            
ax3.quiver(X1, X2, R[0], R[1])
#ax.xaxis.set_ticks([])
#ax.yaxis.set_ticks([])
#ax3.set_xlabel('X1')
#ax3.set_ylabel('X2')
#ax3.axis('equal')



t_eval = np.arange(0, 1000, 1)
sol = integrate.solve_ivp(RHS, [0, 1000], np.array([4,6]), t_eval=t_eval)
print(sol.y)
ax3.plot(sol.y[0],sol.y[1])
ax3.set_xlabel('X1')
ax3.set_ylabel('X2')
ax3.axis('equal')

plt.show()

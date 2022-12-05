#Solve 2D PEDS for 1D target system
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

#dimension of PEDS
N = 2

#Omega = 1./N*np.ones((N,N))
Omega = np.array([[0,1],[0,1]]) # looks good
#th = np.pi/4
#Rot = np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]])
#Omega = np.matmul(np.matmul(Rot,Omega),Rot.transpose())

alpha = 0.1

a1 = 0
a2 = -12.0
a3 = 2.0/3.0
a4 = 0.25
'''
a1 = -9.85#a1 = 0
a2 =-10 #a2 = -12.0
a3 = -2 #a3 = 2.0/3.0
a4 = 0.395 #a4 = 0.25
'''
def fs(x):
    return -4*a4*x**3 -3*a3*x**2 -2*a2*x -a1

def Fv(Xv):
    return fs(Xv)

def RHS_C(t,Xv):
    return np.matmul(Omega,Fv(Xv)) - alpha*np.matmul((np.identity(N)-Omega),Xv)

def RHS_NC(t,Xv):
    OX = np.matmul(Omega,np.diag(Xv))
    OX2 = np.matmul(OX,OX)
    OX3 = np.matmul(OX2,OX)
    Fm = -4*a4*OX3 -3*a3*OX2 -2*a2*OX -a1
    Fv = np.matmul(Fm,np.array([1,1]))
    return np.matmul(Omega,Fv) - alpha*np.matmul((np.identity(N)-Omega),Xv)

#vectorised X
Xv = np.array([1,1])
L = 30
x1 = np.linspace(-8, 8, L)
x2 = np.linspace(-8, 8, L)

X1, X2 = np.meshgrid(x1,x2)
Xf = np.array([X1,X2])
# for c
Fc = fs(Xf)
# for nc 
OMEGA = 0.5*np.block([[np.identity(L),np.identity(L)],[np.identity(L),np.identity(L)]])
XDIAG = np.block([[X1,np.zeros((L,L))],[np.zeros((L,L)),X2]])

OX = np.matmul(OMEGA,XDIAG)
OX2 = np.matmul(OX,OX)
OX3 = np.matmul(OX2,OX)

FMAT = -(4.0*a4*OX3+3.0*a3*OX2+2.0*a2*OX + a1)
Fnc = np.matmul(FMAT, np.block([[np.identity(L)],[np.identity(L)]]))

Fnc = Fnc.reshape(2,L,L)
Rc = np.tensordot(Omega,Fc,axes =((1),(0))) \
    - alpha*np.tensordot((np.identity(N)-Omega),Xf,axes=((1),(0)))

Rnc = np.tensordot(Omega,Fnc,axes =((1),(0))) \
    - alpha*np.tensordot((np.identity(N)-Omega),Xf,axes=((1),(0)))

for i in range(L):
    for j in range(L):
        #print(np.array([X1[i,j],X2[j]]))
        R = RHS_NC(0,np.array([X1[i,j],X2[i,j]]))
        Rnc[0,i,j] = R[0]
        Rnc[1,i,j] = R[1]

fig = plt.figure()
ax1 = fig.add_subplot(111)
xmin = -8
xmax = 8
x = np.arange(-8,8,0.1)
ax1.plot(x,a4*x**4+a3*x**3+a2*x**2+a1*x)

fig = plt.figure()
ax2 = fig.add_subplot(111)
mag = np.sqrt(Fc[0]**2 + Fc[1]**2)
scale = np.power(mag,2.0/3.0)
ax2.quiver(X1, X2, Fc[0]/scale, Fc[1]/scale)
#ax.xaxis.set_ticks([])
#ax.yaxis.set_ticks([])
ax2.set_xlabel('X1')
ax2.set_ylabel('X2')
ax2.set_title('Decoupled Function')


fig = plt.figure()
ax3 = fig.add_subplot(111)   
mag = np.sqrt(Rc[0]**2 + Rc[1]**2)
scale = np.power(mag,2.0/3.0)
print(scale)                                                                                                                   
ax3.quiver(X1, X2, Rc[0]/scale, Rc[1]/scale)
ax3.set_title('comuntative Function')
#ax.xaxis.set_ticks([])
#ax.yaxis.set_ticks([])
#ax3.set_xlabel('X1')
#ax3.set_ylabel('X2')
#ax3.axis('equal')

fig = plt.figure()
ax4 = fig.add_subplot(111)  
ax4.set_title('Non-comuntative Function')
mag = np.sqrt(Rnc[0]**2 + Rnc[1]**2)
scale = np.power(mag,2.0/3.0)                                                                                                                          
ax4.quiver(X1, X2, Rnc[0]/scale, Rnc[1]/scale)



t_eval = np.arange(0, 50, 1)
Xi = np.array([-1.6,3])
sol = integrate.solve_ivp(RHS_C, [0, 50], Xi, t_eval=t_eval)
ax3.plot(Xi[0],Xi[1],marker="o", markersize=4)
ax3.plot(sol.y[0],sol.y[1])
print(sol.y[0][-1],sol.y[1][-1])

t_eval = np.arange(0, 50, 1)
Xi = np.array([-1.6,-1])
sol = integrate.solve_ivp(RHS_C, [0, 50], Xi, t_eval=t_eval)
ax3.plot(Xi[0],Xi[1],marker="o", markersize=4)
ax3.plot(sol.y[0],sol.y[1])
print(sol.y[0][-1],sol.y[1][-1])


t_eval = np.arange(0, 50, 1)
Xi = np.array([-4,6])
sol = integrate.solve_ivp(RHS_C, [0, 50], Xi, t_eval=t_eval)
ax3.plot(Xi[0],Xi[1],marker="o", markersize=4)
ax3.plot(sol.y[0],sol.y[1])
print(sol.y[0][-1],sol.y[1][-1])

t_eval = np.arange(0, 50, 1)
Xi = np.array([7,7])
sol = integrate.solve_ivp(RHS_C, [0, 50], Xi, t_eval=t_eval)
ax3.plot(Xi[0],Xi[1],marker="o", markersize=4)
ax3.plot(sol.y[0],sol.y[1])
print(sol.y[0][-1],sol.y[1][-1])

for ax in [ax2,ax3,ax4]:
    ax.set_xlim([xmin,xmax])
    ax.set_ylim([xmin,xmax])
    ax.axes.set_aspect('equal', adjustable='box')

plt.show()

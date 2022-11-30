import numpy as np
from numpy.polynomial.polynomial import polyval
from scipy import integrate
class solusion:
    def __init__(self,ti,tf,nsteps):
        self.t = np.linspace(ti,tf,num = nsteps)
        self.y = None

class solver:
    def __init__(self,N,m,f,ti,tf,nsteps,Omega = None,alpha=0,Xi = None,delta = 2e-8) -> None:
        # N -- dimension of the PEDS
        # m -- dimension of the traget system
        # f -- is a callable function of size m
        #      returning [f1(t,{xm}),f2(t,{xm}) ... fm(t,{xm})] 
        #      in the target system
        # Omega -- projection matrix
        # alpha -- decay rate
        # Xi -- the initial condition for PEDS 
        #       a list of inital value [X11,X12 ... X1N,
        #                               X21,X22 ... X2N
        #                               ...
        #                               Xm1,Xm2 ... XmN]
        #       of size m*N 
        self.N = N # dimension of the PEDS
        self.m = m # dimension of the traget system
        self.ti = ti
        self.tf= tf
        self.nsteps = nsteps
        if not Omega:
            # uniform mean field projector is used in default
            self.Omega = 1.0/N*np.ones((N,N))
        else:
            self.Omega = Omega
        self.alpha = alpha
        self.f = f
        if not Xi:
            self.Xi = Xi
        else:
            self.Xi = np.zeros((N,m))
        self.fvec_type = 'c' # or nc
        self._delta = delta # the difference to measure the gradient
        
    
    def potential_grad_i(self,x,i):
        # x is of size m 
        # x is the set of input, {xm} for the target system
        x1 = x
        x1[i] += self._delta
        x2 = x
        x2[i] -= self._delta
        return (self.potential(x) - self.potential(x))/(2.0*self._delta)
    
    #def minus_potential_grad_i(self,x,i):
    #    return -self.potential_grad_i(self,x,i)
    
    def minus_potential_grad(self,t,x):
        # x is the input vaialbe of size m in the target system
        minusGradV = np.zeros(self.m)
        for i in range(self.m):
            minusGradV[i] = -self.potential_grad_i(x,i)
        return minusGradV
      
    def set_pontential(self,V):
        # V({xm}) is a callable function in the target system
        self.potential = V
        self.f = self.minus_potential_grad

    def rhs(self,t,Xv):
        # variable Xv of length m*N 
        # variable Xmat of size N*m 
        Xmat = Xv.reshape(self.N,self.m)
        rhs = np.zeros(self.N,self.m)
        if self.fvec_type == 'c':
            for i in range(self.N):
                rhs[i,:] = np.matmul(self.Omega,self.f(t,Xmat[i,:])) \
                    - self.alpha*np.matmul(np.identity(self.N)-self.Omega,Xmat[i,:])
        return rhs.reshape(self.m*self.N)
    
    def ivp_ode(self):
        #rhs = lambda t,Xv: self.rhs(t,Xv)
        sol = integrate.solve_ivp(self.rhs, [self.ti,self.tf], self.Xi, 
                        t_eval=np.linspace(self.ti,self.tf,self.nsteps))
        return sol

    def ivp_sde(self,sigma = 1.0, method = 'EM'):
        #https://math.gmu.edu/~tsauer/pre/sde.pdf
        dt = (self.tf - self.ti)/self.nsteps
        sqt_dt = np.sqrt(dt)
        sol = solusion(self.ti,self.tf,self.nsteps)
        sol.y = np.array(self.m*self.N,self.nsteps)
        sol.y[:,0] = self.Xi
        X = self.Xi
        if method == 'EM': #0.5 order Euler-Maruyama 
            for i in range(1,self.nsteps):
                t = t + dt
                dW = np.random.normal(self.N*self.m)*sqt_dt
                X_new = X + self.rhs(t,X)*dt + sigma*dW
                sol.y[:,i] = X_new
                X = X_new
        elif method =='RK': #1 order Runge-Kutta
            for i in range(1,self.nsteps):
                t = t + dt
                dW = np.random.normal(self.N*self.m)*sqt_dt
                X_new = X + self.rhs(t,X)*dt + sigma*dW \
                        + 0.5*(sigma*(X + sqt_dt) -X)*(dW*dW-dt)/sqt_dt
                sol.y[:,i] = X_new
                X = X_new
        return sol

    def mean_sol(self,sol):
        return np.mean(sol.y,axis=0)
    
    
        
        

            
    



        



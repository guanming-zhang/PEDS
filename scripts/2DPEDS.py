import sys
sys.path.insert(0,'../')
#import peds
from peds import peds_solver
import numpy as np


def V(x):
    a1 = 0
    a2 = -12.0
    a3 = 2.0/3.0
    a4 = 0.25
    #a4 = 0.395
    #a3 = -2
    #a2 = -10
    #a1 = -9.85
    return a4*x**4 + a3*x**3 + a2*x*x +a1*x

m = 1
N = 2
alpha = 0.1
solver = peds_solver.solver(N,m,0,80,nsteps = 100,alpha=alpha)
nruns = 5
gmin_x = -6
lmin_x = 4
stat_c =  {'gmin_count':0,'lmin_count':0,'psudo_min_count':0,'other_count':0}
stat_nc = {'gmin_count':0,'lmin_count':0,'psudo_min_count':0,'other_count':0}
stat_dc = {'gmin_count':0,'lmin_count':0,'psudo_min_count':0,'other_count':0}
dd = 0.3
for i in range(nruns):
    print(i)
    Xi = np.random.uniform(low=-10.0, high= 10.0, size=m*N)
    solver.Xi = Xi
    solver.Omega = 1.0/N*np.ones((N,N))
    solver.set_pontential(V)
    # commutative function
    solver.fvec_type = 'c'
    sol = solver.ivp_ode()
    x_mean = solver.mean_sol(sol)
    if np.abs(x_mean[-1] - gmin_x)< dd:
        stat_c['gmin_count'] += 1
    elif np.abs(x_mean[-1] - lmin_x)< dd:
        stat_c['lmin_count'] += 1
    elif np.abs(x_mean[-1] - 0.5*(lmin_x+gmin_x))< dd:
        stat_c['psudo_min_count'] += 1
    else:
        stat_c['other_count'] += 1

    # noncommutative 
    solver.Omega = 1.0/N*np.ones((N,N))
    solver.fvec_type = 'nc'
    sol = solver.ivp_ode()
    solver.set_pontential(V)
    x_mean = solver.mean_sol(sol)
    if np.abs(x_mean[-1] - gmin_x)< dd:
        stat_nc['gmin_count'] += 1
    elif np.abs(x_mean[-1] - lmin_x)< dd:
        stat_nc['lmin_count'] += 1
    elif np.abs(x_mean[-1] - 0.5*(lmin_x+gmin_x))< dd:
        stat_nc['psudo_min_count'] += 1
    else:
        stat_nc['other_count'] += 1
    # decoupled function
    solver.fvec_type = 'c'
    solver.Omega = np.identity(N)
    sol = solver.ivp_ode()
    x_mean = solver.mean_sol(sol)
    if np.abs(x_mean[-1] - gmin_x)< dd:
        stat_dc['gmin_count'] += 1
    elif np.abs(x_mean[-1] - lmin_x)< dd:
        stat_dc['lmin_count'] += 1
    elif np.abs(x_mean[-1] - 0.5*(lmin_x+gmin_x))< dd:
        stat_dc['psudo_min_count'] += 1
    else:
        stat_dc['other_count'] += 1

print('no noise')
print(stat_c)
print(stat_nc)
print(stat_dc)

import json
with open('peds_poly_ode.txt', 'w') as f:
     f.write('commutative representation \n')
     f.write(json.dumps(stat_c))
     f.write('\n')
     f.write('non-commutative representation \n')
     f.write(json.dumps(stat_nc))
     f.write('\n')
     f.write('decoupled functions \n')
     f.write(json.dumps(stat_dc))
      
      
      

#solver = peds_solver.solver(N,m,0,400,nsteps = 10000,alpha=0.01,Xi=Xi)

#sol = solver.ivp_sde(sigma=0.0,method='EM')


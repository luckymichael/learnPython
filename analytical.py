# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 16:36:58 2016

@author: Michael Ou
"""

# These are the van Genuchten (1980) equations
# The input is matric potential, psi and the hydraulic parameters.
# psi must be sent in as a numpy array.
# The pars variable is like a MATLAB structure.

 #%%
def S2Q(S, pars):
    return (pars['thetaS'] - pars['thetaR']) * min(1.0, S) + pars['thetaR']

def Q2S(Q, pars):
    return (Q - pars['thetaR']) / (pars['thetaS'] - pars['thetaR'])

def vanG_H2S(H, pars):
    return (1+abs(H*pars['alpha'])**pars['n'])**(-pars['m'])

def vanG_S2H(S, pars):
    return -(S ** (-1 / pars['m']) - 1) ** (1 / pars['n']) / pars['alpha']

def vanG_S2K(S, pars):
    Kr = S ** pars['neta'] * (1 - (1 - S ** (1 / pars['m'])) ** pars['m'])**2
    return pars['Ks'] *  Kr

def Sand():
  pars={}
  pars['thetaR']=0.045
  pars['thetaS']=0.43
  pars['alpha']=0.15
  pars['n']=3
  pars['m']=1-1/pars['n']
  pars['Ks']=1000
  pars['neta']=0.5
  pars['Ss']=0.000001
  return pars

def Clay():
  pars={}
  pars['thetaR']=0.1
  pars['thetaS']=0.4
  pars['alpha']=0.01
  pars['n']=1.1
  pars['m']=1-1/pars['n']
  pars['Ks']=10
  pars['neta']=0.5
  pars['Ss']=0.000001
  return pars

def Loam():
  pars={}
  pars['thetaR']=0.08
  pars['thetaS']=0.43
  pars['alpha']=0.04
  pars['n']=1.6
  pars['m']=1-1/pars['n']
  pars['Ks']=50
  pars['neta']=0.5
  pars['Ss']=0.000001
  return pars


def kirchoff(psi, flux, pars):
    '''
    the function inside the integral
    '''
    K = vanG_S2K(vanG_H2S(psi, pars), pars)
    try:
        return 1.0 / (flux / K - 1.0)
    except ZeroDivisionError:
        print('float division by zero', '\npsi = ', psi, '\nflux = ', flux, '\nK = ', K, '\n', pars)
        raise

def integrate_kirchoff(pars, psi1, psi2, flux):
    '''
    area of the trapezoid in the integral
    '''
    k1 = kirchoff(psi1, flux, pars)
    k2 = kirchoff(psi2, flux, pars)
    return (psi2 - psi1) * (k1 + k2) * 0.5


def H2K_zero(H, pars, flux):
    '''
    difference between flux and hydraulic conductivity,
    used to root-searching the head when hydraulic conductivity is equal to flux
    '''
    S = vanG_H2S(H, pars)
    K = vanG_S2K(S, pars)
    return flux - K


import scipy.optimize
import matplotlib.pyplot as plt
import pandas as pd

# flux > 0 and dx > 0
def integrate_profile(xlen, h0, flux, pars, dh=0.01, dxmax=2, dxmin=0.5, reduce=0.3, increase=1.5):
    x = [0.0]
    h = [h0]
    dxs = [0.0]
    x1 = 0
    while abs(x1) <= abs(xlen):
        repeat = True
        while repeat:
            dx = integrate_kirchoff(pars, h0, h0 + dh, flux)
            if dx > dxmax:
                dh = dh * reduce
            else:
                repeat = False
            if dx < dxmin: dh = dh * increase
            if dx * xlen < 0:
                print('dx = ', dx, '\n',
                      'h0 = ', h0, '\n',
                      'dh = ', dh)
                raise ValueError('`dx` is increasing in different direction of `xlen`.')
        x1 = x1 + dx
        h0 = h0 + dh
        print(dx, x1, h0)
        x.append(x1)
        h.append(h0)
        dxs.append(dx)
    #return {'head': h, 'depth': x, 'dx': dxs}
    return {'head': h, 'depth': x}


flux = 0.5
######## set 1 ########
top = Loam()
bottom = Sand()
H_op = scipy.optimize.newton(H2K_zero, -20, args=(bottom, flux))
profile = integrate_profile(xlen=50, h0=H_op, dh =-0.01, flux=flux, pars=top)
profile['head'].append(H_op)
profile['depth'].append(-150)
df = pd.DataFrame(profile)
df['depth'] = df['depth'] - 50.0
df.sort_values(by = 'depth', inplace=True, ascending=False)
df.to_csv(r'D:\Cloud\Dropbox\postdoc\summa\summaData\summaTestCases\settings_org\syntheticTestCases\vanderborght2005\analytical_solution_set1.csv', index=False)

######## set 2 ########
top = Sand()
bottom = Loam()
H_op = scipy.optimize.newton(H2K_zero, -20, args=(bottom, flux))
profile = integrate_profile(xlen=50, h0=H_op, dh=0.01, flux=flux, pars=top)
profile['head'].append(H_op)
profile['depth'].append(-150)
df = pd.DataFrame(profile)
df['depth'] = df['depth'] - 50.0
df.sort_values(by = 'depth', inplace=True, ascending=False)
df.to_csv(r'D:\Cloud\Dropbox\postdoc\summa\summaData\summaTestCases\settings_org\syntheticTestCases\vanderborght2005\analytical_solution_set2.csv', index=False)

######## set 3 ########
top = Clay()
bottom = Sand()
H_op = scipy.optimize.newton(H2K_zero, -20, args=(bottom, flux))
profile = integrate_profile(xlen=50, h0=H_op, dh=0.01, flux=flux, pars=top)
profile['head'].append(H_op)
profile['depth'].append(-150)
df = pd.DataFrame(profile)
df['depth'] = df['depth'] - 50.0
df.sort_values(by = 'depth', inplace=True, ascending=False)
df.to_csv(r'D:\Cloud\Dropbox\postdoc\summa\summaData\summaTestCases\settings_org\syntheticTestCases\vanderborght2005\analytical_solution_set3.csv', index=False)


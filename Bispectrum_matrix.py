import numpy as np
import scipy.interpolate as interpolate
from scipy.integrate import simps
from numpy.linalg import inv
from math import pi
import matplotlib.pyplot as plt
from decimal import *
from BFisherutils import *
getcontext().prec=28
#from tempfile import TemporaryFile
#PowerFisher = TemporaryFile()

apar = 1.01
aper = 0.99
alp=(apar**2)*aper
f = 0.4
b1 = 1.18 
b2 = -0.76 
navg = 1  
Vs = 1  
eps = 1e-6

parc = (apar, aper, f, b1, b2)
pars = (navg, Vs)

kmax = 0.3
kmin = 0.01     
#Load power spectrum
PP = np.loadtxt("camb_03775393_matterpower_z0.dat")
K = PP[:,0]
P = PP[:,1]
K[0] = 0
fPk = interpolate.interp1d(K, P, kind='cubic') 

kk=np.linspace(kmin,kmax,50)
k123=np.zeros((125000,3)) # stores all possible combinations of k1,k2,k3 values
count=0
bi0=[]
#print(kk)

'''Calculating Bispectrum monopole'''
mu1=np.linspace(-0.98,0.98,50)
phi12=np.linspace(0,np.pi,50)
for i in range (0,len(kk)):
    for j in range (0,len(kk)):
        for k in range (0,len(kk)):
            k1=kk[i]
            k2=kk[j]
            k3=kk[k]
            k123[count][0]=k1
            k123[count][1]=k2
            k123[count][2]=k3
            #print((k3**2 - k1**2 - k2**2)/(2*k1*k2))
            #func=lambda mu1,phi12:Bisp((k1, k2, k3, mu1, phi12), parc, pars)
            func=Bisp((k1, k2, k3, mu1, phi12), parc, pars)
            res,err=simps((func,phi12),mu1)
            #res,err=integrate.quad(func,0,np.pi)
            bi0.append(res)
            count=count+1
            #print(res,err)
            #print (Bisp((k1, k2, k3, mu1, phi12), parc, pars),k1,k2,k3)
            #Bisp((k1, k2, k3, mu1, phi12), parc, pars)
            
'''Interpolating Bispectrum monopole over k1, k2,k3'''
bi0=np.array(bi0)
bi_0=bi0.reshape((50,50,50))
k1=np.array(kk)
k2=np.array(kk)
k3=np.array(kk)
from scipy.interpolate import RegularGridInterpolator
fBi= RegularGridInterpolator((k1,k2,k3),bi_0,method='nearest',bounds_error=False, fill_value=None)

import scipy.integrate as integrate 
def Pk(k,mu):
    k = k*np.sqrt(aper**2+mu**2*(apar**2-aper**2))
    alp=(apar**2)*aper
    mu = mu*apar/np.sqrt(aper**2+mu**2*(apar**2-aper**2))
    return (b1 + f*mu)**2*fPk(alp*k)
def Bfunc(k123):
    k1=k123[:,0]
    k2=k123[:,1]
    k3=k123[:,2]
    func=fBi((alp*k1,alp*k2,alp*k3))
    return func
    
def avg(M,k123):
    '''Preparing the averaging matrix'''
    div=np.linspace(kmin,kmax,M+1)
    avg=np.zeros((125000,M))
    for i in range(M):
        for j in range(125000):
            if div[i]<=k123[j,0]<=div[i+1]:
                if div[i]<=k123[j,1]<=div[i+1]:
                    if div[i]<=k123[j,2]<=div[i+1]:
                        avg[j,i]=1
    return avg

def Bavg(M,k123):
    return np.dot(avg(M,k123).T,Bfinal)

def dBdalp(M,k123):
    global alp
    eps = 0.4
    B0 = Bavg(M,k123)
    alp += eps
    B1 = Bavg(M,k123)
    alp -= eps
    return B1-B0/eps

def CovB(k1,k2,k3, mu1, phi12):
    '''Covariance matrix of Bispectrum'''
    mu12 = (k3**2 - k1**2 - k2**2)/(2*k1*k2)
    mu2 = mu2 = mu1*mu12 - np.sqrt(1 - mu1**2)*np.sqrt(1 - mu12**2+0.001 if (np.any(mu12)**2==1) else 0)*np.cos(phi12)
    mu3 = -(mu1*k1 + mu2*k2)/k3
    C = Pk(k1,mu1) + 1/navg
    C *= Pk(k2,mu2) + 1/navg
    C *= Pk(k3,mu3) + 1/navg
    return C

def CovB0(k1,k2,k3):
    '''Covariance matrix of Bispectrum monopole'''
    fun=[]
    mu1=np.linspace(-0.98,0.98,100)
    phi12=np.linspace(0,np.pi,100)
    for i in range(0,125000):
        k1a=k1[i]
        k2a=k2[i]
        k3a=k3[i]
        res,_=simps((CovB(k1a, k2a, k3a, mu1, phi12),phi12),mu1)
        fun.append(res)
    return np.array(fun)

def Cavg(M,k123):
    k1=k123[:,0]
    k2=k123[:,1]
    k3=k123[:,2]
    return np.dot(avg(M,k123).T,Covfinal)

def final(M,k123):
    '''Discarding all nan values from Covariance matrix'''
    k1=k123[:,0]
    k2=k123[:,1]
    k3=k123[:,2]
    Covfinal=CovB0(k1,k2,k3)
    Bfinal=Bfunc(k123)
    for i in range(0,125000):
        if np.isnan(Covfinal[i])==True:
            Covfinal[i]=0
            Bfinal[i]=0    
    return Covfinal,Bfinal

'''Calculating variance of alpha for M=50'''
Covfinal,Bfinal = final(50,k123)
#Fisher=np.zeros((1,1))
Cov=Cavg(50,k123)
dalp=dBdalp(50,k123)
var_a=np.sum(dalp**2/Cov)
print(np.sqrt(var_a))
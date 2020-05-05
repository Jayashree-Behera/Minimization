import scipy.interpolate as itp
import numpy as np
np.seterr(all='ignore')
from interpolation import interp
import pdb

PS = np.loadtxt('camb_03775393_matterpower_z0.dat')
Pbao = np.loadtxt('camb_03775393_matterpower_z0.dat')
Psmooth = np.loadtxt('camb_03775393_matterpower_z0.dat')
Piso = itp.interp1d(PS[:,0],PS[:,1], kind='cubic', fill_value=0)
Pbao_itp = itp.interp1d(Pbao[:,0], Pbao[:,1], kind='cubic', fill_value=0)
Psmooth_itp = itp.interp1d(Psmooth[:,0], Psmooth[:,1], kind='cubic', fill_value=0)


def Pk_new(var, parc):
    k, mu = var
    A, am1, am2, a0, a1, a2, apar, aper = parc

    ak = k*np.sqrt(aper**2+mu**2*(apar**2-aper**2))
    return A*(0.64*Psmooth_itp(k) + am2/k**2 + am1/k + a0 + a1*k + a2*k**2)*Pbao_itp(ak)

def Pk_vec(var, parc):
    K, Mu = var
    A, am1, am2, a0, a1, a2, apar, aper = parc
    K = K.reshape(len(K),1)
    Mu = Mu.reshape(len(Mu),1)

    aK = np.dot(K, np.sqrt(aper**2+Mu.T**2*(apar**2-aper**2)))
    Pbao_vec = Pbao_itp(aK)
    Psmooth_vec = Psmooth_itp(K)
    
    return A*(0.64*Psmooth_vec + am2/K**2 + am1/K + a0 + a1*K + a2*K**2)*Pbao_vec


def Pk_aniso(var,parc):
    '''
    Anisotropic power spectrum.
    var = k, mu
    parc = f, b1
    '''
    k, mu = var
    apar, aper, f, b1, b2 = parc

    k = k.reshape(len(k),1)
    mu = mu.reshape(len(mu),1)

    k = np.dot(k, np.sqrt(aper**2+mu.T**2*(apar**2-aper**2)))
    mu = mu*apar/np.sqrt(aper**2+mu**2*(apar**2-aper**2))
    return (b1 + f*mu.T)**2*Piso(k)

def Pk(var,parc):
    '''
    Anisotropic power spectrum.
    var = k, mu
    parc = f, b1
    '''
    k, mu = var
    apar, aper, f, b1, b2 = parc

    k = k*np.sqrt(aper**2+mu**2*(apar**2-aper**2))
    mu = mu*apar/np.sqrt(aper**2+mu**2*(apar**2-aper**2))
    return (b1 + f*mu)**2*Piso(k)


def Bisp_vec(var,parc,pars):
    '''
    Bispectrum as a function of 5 variables - k1,k2,k3,mu1,phi12 and cosmological
    parameters parc.
    This doesn't check for triangular condition.
    '''

    k1_v, k2_v, k3_v, mu1_v, phi12_v = var
    apar, aper, f, b1, b2 = parc
    navg, Vs = pars
    lk1, lk2, lk3, lmu, lphi = len(k1_v), len(k2_v), len(k3_v), len(mu1_v), len(phi12_v)

    k1 = k1_v.reshape(lk1,1,1,1,1) * np.ones((lk1,lk2,lk3,lmu,lphi))
    k2 = k2_v.reshape(1,lk2,1,1,1) * np.ones((lk1,lk2,lk3,lmu,lphi))
    k3 = k3_v.reshape(1,1,lk3,1,1) * np.ones((lk1,lk2,lk3,lmu,lphi))
    mu1 = mu1_v.reshape(1,1,1,lmu,1) * np.ones((lk1, lk2, lk3, lmu, lphi))
    phi12 = phi12_v.reshape(1,1,1,1,lphi) * np.ones((lk1, lk2, lk3, lmu, lphi))

    mu12 = (k3**2 - k1**2 - k2**2)/2/k1/k2
    mu2 = mu1*mu12 - np.sqrt(1 - mu1**2)*np.sqrt(1 - mu12**2)*np.cos(phi12)
    mu3 = -(mu1*k1 + mu2*k2)/k3

    # rescale for AP
    k1 = k1*np.sqrt(aper**2+mu1**2*(apar**2-aper**2))
    k2 = k2*np.sqrt(aper**2+mu2**2*(apar**2-aper**2))
    k3 = k3*np.sqrt(aper**2+mu3**2*(apar**2-aper**2))
    mu1 = mu1*apar/np.sqrt(aper**2+mu1**2*(apar**2-aper**2))
    mu2 = mu2*apar/np.sqrt(aper**2+mu2**2*(apar**2-aper**2))
    mu3 = mu3*apar/np.sqrt(aper**2+mu3**2*(apar**2-aper**2))

    mu12 = (k3**2 - k1**2 - k2**2)/2/k1/k2
    mu31 = -(k1 + k2*mu12)/k3
    mu23 = -(k1*mu12 + k2)/k3

    k12 = np.sqrt(k1**2 + k2**2 + 2*k1*k2*mu12)
    k23 = np.sqrt(k2**2 + k3**2 + 2*k2*k3*mu23)
    k31 = np.sqrt(k3**2 + k1**2 + 2*k3*k1*mu31)

    Z1k1 = b1 + f*mu1**2
    Z1k2 = b1 + f*mu2**2
    Z1k3 = b1 + f*mu3**2

    F12 = 5./7. + mu12/2*(k1/k2 + k2/k1) + 2./7.*mu12**2
    F23 = 5./7. + mu23/2*(k2/k3 + k3/k2) + 2./7.*mu23**2
    F31 = 5./7. + mu31/2*(k3/k1 + k1/k3) + 2./7.*mu31**2

    G12 = 3./7. + mu12/2*(k1/k2 + k2/k1) + 4./7.*mu12**2
    G23 = 3./7. + mu23/2*(k2/k3 + k3/k2) + 4./7.*mu23**2
    G31 = 3./7. + mu31/2*(k3/k1 + k1/k3) + 4./7.*mu31**2

    mu1p2 = (mu1*k1 + mu2*k2)/k12
    mu2p3 = (mu2*k2 + mu3*k3)/k23
    mu3p1 = (mu3*k3 + mu1*k1)/k31

    Z2k12 = b2/2. + b1*F12 + f*mu1p2**2*G12
    Z2k12 += f*mu1p2*k12/2.*(mu1/k1*Z1k2 + mu2/k2*Z1k1)
    Z2k23 = b2/2. + b1*F23 + f*mu2p3**2*G23
    Z2k23 += f*mu2p3*k23/2.*(mu2/k2*Z1k3 + mu3/k3*Z1k2)
    Z2k31 = b2/2. + b1*F31 + f*mu3p1**2*G31
    Z2k31 += f*mu3p1*k31/2.*(mu3/k3*Z1k1 + mu1/k1*Z1k3)

    Bi = 2*Z2k12*Z1k1*Z1k2*Piso(k1)*Piso(k2)
    Bi += 2*Z2k23*Z1k2*Z1k3*Piso(k2)*Piso(k3)
    Bi += 2*Z2k31*Z1k3*Z1k1*Piso(k3)*Piso(k1)
    '''
    I think I should add this.
    '''
    Bi += (Pk([k1, mu1],parc) + Pk([k2,mu2], parc) + Pk([k3,mu3], parc))/navg

    return Bi


def Bisp(var,parc,pars):
    '''
    Bispectrum as a function of 5 variables - k1,k2,k3,mu1,phi12 and cosmological
    parameters parc.
    This doesn't check for triangular condition.
    '''

    k1, k2, k3, mu1, phi12 = var
    apar, aper, f, b1, b2 = parc
    navg, Vs = pars

    mu12 = (k3**2 - k1**2 - k2**2)/(2*k1*k2)
    mu2 = mu1*mu12 - np.sqrt(1 - mu1**2)*np.sqrt(1 - mu12**2+0.001 if (mu12**2==1) else 0)*np.cos(phi12)
    mu3 = -(mu1*k1 + mu2*k2)/k3
    #print(np.isnan(mu3))

    # rescale for AP
    k1 = k1*np.sqrt(aper**2+mu1**2*(apar**2-aper**2)+0.01)
    k2 = k2*np.sqrt(aper**2+mu2**2*(apar**2-aper**2)+0.01)
    k3 = k3*np.sqrt(aper**2+mu3**2*(apar**2-aper**2)+0.01)
    #print(np.isnan(mu1),np.isnan(mu3),np.isnan(mu2))
    mu1 = mu1*apar/np.sqrt(aper**2+mu1**2*(apar**2-aper**2)+(0.01 if (aper**2+np.any(mu1)**2*(apar**2-aper**2)<=0) else 0))
    mu2 = mu2*apar/np.sqrt(aper**2+mu2**2*(apar**2-aper**2)+(0.01 if (aper**2+np.any(mu1)**2*(apar**2-aper**2)<=0) else 0))
    mu3 = mu3*apar/np.sqrt(aper**2+mu3**2*(apar**2-aper**2)+(0.01 if (aper**2+np.any(mu1)**2*(apar**2-aper**2)<=0) else 0))
    #print(np.isnan(mu1),np.isnan(mu3),np.isnan(mu2))

    mu12 = (k3**2 - k1**2 - k2**2)/2/k1/k2
    mu31 = -(k1 + k2*mu12)/k3
    mu23 = -(k1*mu12 + k2)/k3
    #print(np.isnan(mu12),np.isnan(mu31),np.isnan(mu23))


    k12 = np.sqrt(k1**2 + k2**2 + 2*k1*k2*mu12)
    k23 = np.sqrt(k2**2 + k3**2 + 2*k2*k3*mu23)
    k31 = np.sqrt(k3**2 + k1**2 + 2*k3*k1*mu31)
    #print(np.isnan(k12),np.isnan(k31),np.isnan(k23))

    Z1k1 = b1 + f*mu1**2
    Z1k2 = b1 + f*mu2**2
    Z1k3 = b1 + f*mu3**2

    F12 = 5./7. + mu12/2*(k1/k2 + k2/k1) + 2./7.*mu12**2
    F23 = 5./7. + mu23/2*(k2/k3 + k3/k2) + 2./7.*mu23**2
    F31 = 5./7. + mu31/2*(k3/k1 + k1/k3) + 2./7.*mu31**2

    G12 = 3./7. + mu12/2*(k1/k2 + k2/k1) + 4./7.*mu12**2
    G23 = 3./7. + mu23/2*(k2/k3 + k3/k2) + 4./7.*mu23**2
    G31 = 3./7. + mu31/2*(k3/k1 + k1/k3) + 4./7.*mu31**2

    mu1p2 = (mu1*k1 + mu2*k2)/(k12+0.01)
    mu2p3 = (mu2*k2 + mu3*k3)/(k23+0.01)
    mu3p1 = (mu3*k3 + mu1*k1)/(k31+0.01)

    Z2k12 = b2/2. + b1*F12 + f*mu1p2**2*G12
    Z2k12 += f*mu1p2*k12/2.*(mu1/k1*Z1k2 + mu2/k2*Z1k1)
    Z2k23 = b2/2. + b1*F23 + f*mu2p3**2*G23
    Z2k23 += f*mu2p3*k23/2.*(mu2/k2*Z1k3 + mu3/k3*Z1k2)
    Z2k31 = b2/2. + b1*F31 + f*mu3p1**2*G31
    Z2k31 += f*mu3p1*k31/2.*(mu3/k3*Z1k1 + mu1/k1*Z1k3)

    Bi = 2*Z2k12*Z1k1*Z1k2*Piso(k1)*Piso(k2)
    Bi += 2*Z2k23*Z1k2*Z1k3*Piso(k2)*Piso(k3)
    Bi += 2*Z2k31*Z1k3*Z1k1*Piso(k3)*Piso(k1)
    '''
    I think I should add this.
    '''
    
    Bi += (Pk([k1,mu1],parc) + Pk([k2,mu2],parc) + Pk([k3,mu3],parc))/navg

    return np.sin(phi12)*Bi

def dP(var,parc):
    '''
    Derivative of Power spectrum with respect to cosmological parameters - parc.
    var are the 5D variables k1,k2,k3,mu1,phi12 k1, k2, k3, mu1, phi12 = var
    '''
    eps = 1e-6
    derP = np.zeros(np.shape(parc))
    P0 = Pk(var,parc)
    for i in range(np.shape(parc)[0]):
        parc1 = np.copy(parc)
        parc1[i,:] += eps
        P1 = Pk(var,parc1)
        derP[i,:] = (P1 - P0)/eps
    return derP
    
def dB(var,parc,pars):
    '''
    Derivative of Bispectrum with respect to cosmological parameters - parc. var are
    the 5D variables k1,k2,k3,mu1,phi12
    k1, k2, k3, mu1, phi12 = var
    '''
    eps = 1e-6
    derB = np.zeros(np.shape(parc))
    B0 = Bisp(var,parc,pars)
    for i in range(np.shape(parc)[0]):
        parc1 = np.copy(parc)
        parc1[i,:] += eps
        B1 = Bisp(var,parc1,pars)
        derB[i,:] = (B1 - B0)/eps
    return derB

def CovP(var,parc,pars):
    navg, Vs = pars
    C = (Pk(var,parc) + 1/navg)**2
    return C

def CovP_new(var,parc,pars):
    navg, Vs = pars
    C = (Pk_new(var,parc) + 1/navg)**2
    return C

def CovP_vec(var, parc, pars):
    navg, Vs = pars
    C = (Pk_vec(var,parc) + 1/navg)**2
    return C

def CovB(var,parc,pars):
    '''
    Covariance of Bispectrum at k1,k2,k3,mu1,phi12 for fiducial cosmological
    parameters - parc, and survey parameters - pars. 
    var = k1,k2,k3,mu1,phi12
    pars = navg, V
    parc = whatever goes into Pk
    This doesn't check for triangular condition.
    '''
 
    k1, k2, k3, mu1, phi12 = var
    navg, Vs = pars

    mu12 = (k3**2 - k1**2 - k2**2)/2/k1/k2
    mu2 = mu1*mu12 - np.sqrt(1 - mu1**2)*np.sqrt(1 - mu12**2)*np.cos(phi12)
    mu3 = -(mu1*k1 + mu2*k2)/k3
    
    C = Pk([k1,mu1],parc) + 1/navg
    C *= Pk([k2,mu2],parc) + 1/navg
    C *= Pk([k3,mu3],parc) + 1/navg
    C *= Vs 

    return C

def CovB_vec(var,parc,pars):
    '''
    Covariance of Bispectrum at k1,k2,k3,mu1,phi12 for fiducial cosmological
    parameters - parc, and survey parameters - pars. 
    var = k1,k2,k3,mu1,phi12
    pars = navg, V
    parc = whatever goes into Pk
    This doesn't check for triangular condition.
    '''
 
    k1_v, k2_v, k3_v, mu1_v, phi12_v = var
    navg, Vs = pars

    lk1, lk2, lk3, lmu, lphi = len(k1_v), len(k2_v), len(k3_v), len(mu1_v), len(phi12_v)

    k1 = k1_v.reshape(lk1,1,1,1,1) * np.ones((lk1,lk2,lk3,lmu,lphi))
    k2 = k2_v.reshape(1,lk2,1,1,1) * np.ones((lk1,lk2,lk3,lmu,lphi))
    k3 = k3_v.reshape(1,1,lk3,1,1) * np.ones((lk1,lk2,lk3,lmu,lphi))
    mu1 = mu1_v.reshape(1,1,1,lmu,1) * np.ones((lk1, lk2, lk3, lmu, lphi))
    phi12 = phi12_v.reshape(1,1,1,1,lphi) * np.ones((lk1, lk2, lk3, lmu, lphi))

    mu12 = (k3**2 - k1**2 - k2**2)/2/k1/k2
    mu2 = mu1*mu12 - np.sqrt(1 - mu1**2)*np.sqrt(1 - mu12**2)*np.cos(phi12)
    mu3 = -(mu1*k1 + mu2*k2)/k3
    
    C = Pk([k1,mu1],parc) + 1/navg
    C *= Pk([k2,mu2],parc) + 1/navg
    C *= Pk([k3,mu3],parc) + 1/navg
    C *= Vs 

    return C

def CrossPB(var1,var2,parc,pars):

    k1, k2, k3, mu1, phi12 = var2

    navg, Vs = pars

    mu12 = (k3**2 - k1**2 - k2**2)/2/k1/k2
    mu2 = mu1*mu12 - np.sqrt(1 - mu1**2)*np.sqrt(1 - mu12**2)*np.cos(phi12)
    mu3 = -(mu1*k1 + mu2*k2)/k3
    
    C = Bisp(var2,parc,pars)
    C += Pk([k1,mu1],parc)/navg  
    C += Pk([k2,mu2],parc)/navg  
    C += Pk([k2,mu2],parc)/navg  
    C += 1/navg**2

    return C

def Fishz(pars,parc):
    '''
    Compute Bispectrum Fisher Matrix at a fixed redshift for survey parameters -
    pars (that are known/fixed), and cosmological parameters - parc (that are
    unknown).
    pars = [nbar, Volume]
    parc = [Omega_m, sigma_8, n_s, w_0, w_a, Omega_b, h0]
    '''
    # Bispectrum in redshift space is a 5D function. Integrate over 5D with MC.
    RR = np.random.rand(NMC,5)
    for i in range(NMC):
        eta1 = etamax*RR[i,0]
        eta2 = etamax*RR[i,1]
        eta3 = etamax*RR[i,2]
        k1 = np.sqrt(2*eta1)
        k2 = np.sqrt(2*eta2)
        k3 = np.sqrt(2*eta3)
        mu1 = 2*mumax*RR[i,3] - 1
        phi12 = 2*np.pi*RR[i,4]
        var = [k1, k2, k3, mu1, phi12]
        CB = CovB()
        derB = dB()
        FM += np.outer(dB,dB)/CB

    return FM

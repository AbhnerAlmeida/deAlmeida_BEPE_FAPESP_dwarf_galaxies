#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.integrate import odeint
from scipy import interpolate
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('text', usetex=True)
import sys as sys
import argparse
import time

"""
Created on Fri Jan 15, 2021
Modified on Fri Mar 17, 2023
N-body code
@author: gam
"""

def MPlummer_tilde(x):
    """dimensionless mass profile of Plummer (1911) model
    argument: r/a
    returns: M(r)/M_infinity
    author: Gary Mamon"""

    return x*x*x/(1+x*x)**1.5

def MHernquist_tilde(x):
    """dimensionless mass profile of Hernquist (1990) model
    argument: r/a
    returns: M(r)/M_infinity
    author: Gary Mamon"""
    
    return x*x/(1+x)**2

def M_tilde(x,model):
    """dimensionless mass profile
    arguments: r/a, model
    returns: M(r)/M_infinity
    author: Gary Mamon"""

    if model == "Hernquist":
        return MHernquist_tilde(x)
    elif model == "Plummer":
        return MPlummer_tilde(x)
    else:
        raise ValueError("cannot recognize model " + model)

def sigmaisoPlummer_tilde(x):
    """dimensionless isotropic velocity dispersion profile of Plummer model
    argument: r/a
    returns: sigma_iso(r) / sqrt[GM_inf/a]
    author: Gary Mamon"""

    sigma2 = 1/6 / np.sqrt(x*x+1)
    return np.sqrt(sigma2)

def sigmaisoHernquist_tilde(x):
    """dimensionless isotropic velocity dispersion profile of Hernquist model
    argument: r/a
    returns: sigma_iso(r) / sqrt[GM_inf/a]
    author: Gary Mamon"""

    x1 = x+1
    # need series expansion to avoid numerical errors at large x
    sigma2 = np.where(x>1000,0.2/x-7/30/(x*x),x*x1**3*np.log(x1/x) - x*(25+2*x*(26+3*x*(7+2*x)))/(12*x1))
    return np.sqrt(sigma2)

def sigmaiso_tilde(x,model):
    """dimensionless isotropic velocity dispersion profile
    arguments: r/a, model
    returns: sigma_iso(r)/sqrt[GM_inf/a]
    author: Gary Mamon"""

    if model == "Hernquist":
        return sigmaisoHernquist_tilde(x)
    elif model == "Plummer":
        return sigmaisoPlummer_tilde(x)
    else:
        raise ValueError("cannot recognize model " + model)

def Phi_tilde_Hernquist(x):
    """dimensionless gravitational potential of Hernquist model
    argument: r/a
    returns: -Phi(r) / G M_tot/a
    """
    return 1/(x+1)

def Phi_tilde_Plummer(x):
    """dimensionless gravitational potential of Plummer model
    argument: r/a
    returns: -Phi(r) / G M_tot/a
    """
    return 1/np.sqrt(x*x+1)

def Phi_tilde(x,model):
    """dimensionless gravitational potential
    arguments: 
        x: r/a
        model: density model
    returns: -Phi(r) / G M_tot/a
    """
    if model == 'Hernquist':
        return Phi_tilde_Hernquist(x)
    if model == 'Plummer':
        return Phi_tilde_Plummer(x)
    else:
        raise ValueError("Cannot recognize model " + model)
        
def derivsNbody(w,t,*args):
    """6N array of deriatives of positions and velocities for N-body simulation
    called by scipy.odeint 
    arguments:
        w: 6*N array x_1 ... x_n, y_1 ... y_n, z_1 ... z_n, vx1 ... vxn, ... (in working units)
        t: time (working units)
        args: auxiliary arguments: G, N, M (working units), epsilonSoft (working units), model
        returns dwdt array of time derivatives of w array (in working units)
        where 'working units' means rescaled to pc, km/s, 1000 M_Sun, and 0.9778 Myr
    author: Gary Mamon
    """

    # auxiliary parameters
    G,N_tot,M,m_array,scaleRadius,epsilonSoft_over_a,model = args
    epsilonSoft = epsilonSoft_over_a * scaleRadius

    # setup acceleration array
    dwdt = np.zeros(6*N_tot)

    # positions
    x = w[0:N_tot]
    y = w[N_tot:2*N_tot]
    z = w[2*N_tot:3*N_tot]

    # derivatives of positions = velocities
    dwdt[0:3*N_tot] = w[3*N_tot:6*N_tot] 
    
    # prepare accelerations: accoverG = acceleration/G
    accoverG_X = np.zeros(N_tot)
    accoverG_Y = np.zeros(N_tot)
    accoverG_Z = np.zeros(N_tot)
    if m_array is None:
        m_array = M/N_tot * np.ones(N_tot)

    # derivatives of velocities = accelerations
    for k,(X,Y,Z) in enumerate(zip(x,y,z)):
        dx = x-X
        dy = y-Y
        dz = z-Z
        seps2 = dx*dx + dy*dy + dz*dz
        seps2modwSoft = seps2 + epsilonSoft*epsilonSoft
        accoverG_X[k] = np.sum(m_array * dx / seps2modwSoft**1.5)
        accoverG_Y[k] = np.sum(m_array * dy / seps2modwSoft**1.5)
        accoverG_Z[k] = np.sum(m_array * dz / seps2modwSoft**1.5)
    dwdt[3*N_tot:4*N_tot] = G * accoverG_X
    dwdt[4*N_tot:5*N_tot] = G * accoverG_Y
    dwdt[5*N_tot:6*N_tot] = G * accoverG_Z
    
    return dwdt

def derivs3body(w,t,*args):
    """6N array of deriatives of positions and velocities for restricted 3-body simulation
    called by scipy.odeint 
    arguments:
        w: 6*N array x_1 ... x_n, y_1 ... y_n, z_1 ... z_n, vx1 ... vxn, ... (in working units)
        t: time (working units)
        args: auxiliary arguments: G, N, M (working units), 
                epsilonSoft (working units), model, scale_radius
        returns dwdt array of time derivatives of w array (in working units)
        where 'working units' means rescaled to pc, km/s, 1000 M_Sun, and 0.9778 Myr
    author: Gary Mamon
    """

    # auxiliary parameters
    G,N_tot,M,m_array,scaleRadius,epsilonSoft_over_a,model = args
    epsilonSoft = epsilonSoft_over_a * scaleRadius

    # setup acceleration array
    dwdt = np.zeros(6*N_tot)

    # positions
    x = w[0:N_tot]
    y = w[N_tot:2*N_tot]
    z = w[2*N_tot:3*N_tot]
    
    # subtract mean positions for positions relative to mean
    # x = x - x.mean()
    # y = y - y.mean()
    # z = z - z.mean()
    
    r = np.sqrt(x*x + y*y + z*z + epsilonSoft*epsilonSoft)
    r2 = r*r
    r3 = r*r*r

    # derivatives of positions = velocities
    dwdt[0:3*N_tot] = w[3*N_tot:6*N_tot] 
    
    if m_array is None:
        m_array = M/N_tot * np.ones(N_tot)

    # derivatives of velocities = accelerations
    factor = -1 * G * M * M_tilde(r/scaleRadius,model)/r3
    dwdt[3*N_tot:4*N_tot] = factor * x
    dwdt[4*N_tot:5*N_tot] = factor * y
    dwdt[5*N_tot:6*N_tot] = factor * z
    
    return dwdt

def InitialConditions(N=20,M=1e5,scaleRadius=1,model="Plummer",
                      vFactor=1,MextraFrac=0,rExtra=None,seed=100):
    """Initial conditions for scipy.odeint
    arguments:
        N: number of particles
        M: mass of object (in solar masses)
        scaleRadius: scale radius (a) in pc
        model: density model of cluster (default Plummer)
        vfactor: multiplicative factor for velocities 
            (e.g. 0.1 for collapse, default 1)
        MextraFrac: fraction of extra mass in circular orbit (default 0)
        rextra: radius of orbit of extra mass (in units of scale radius) 
        seed: seed for random number generator (None for random seed)
    returns x,y,z,vx,vy,vz (positions in pc, velocities in km/s)
    dependencies: numpy as np
    author: Gary Mamon
    """
    # work in rescaled units
    print("in ICs...")
    M_rescaled = M / massUnit
    scaleRadius_rescaled = scaleRadius / radiusUnit

    # random generator according to value of seed
    rg = np.random.default_rng(seed)
    
    # uniform [0-1] random arrays for r, theta and phi
    qr = rg.random(N)
    qtheta = rg.random(N)
    qphi = rg.random(N)

    # spherical coordinates:
    # r in physical units
    if model == "Plummer":
        r_rescaled = scaleRadius_rescaled * qr**(1/3)/np.sqrt(1-qr**(2/3))
    elif model == "Hernquist":
        r_rescaled = scaleRadius_rescaled * np.sqrt(qr)/(1-np.sqrt(qr))
    else:
        raise ValueError("Cannot recognize model = " + model)

    # angular spherical coordinates (radians)
    theta = np.arccos(1-2*qtheta)
    phi = 2*np.pi*qphi
    
    # cartesian positions in working (rescaled) units
    x = r_rescaled*np.sin(theta)*np.cos(phi)
    y = r_rescaled*np.sin(theta)*np.sin(phi)
    z = r_rescaled*np.cos(theta)
    
    # dimensionless radii
    u = r_rescaled/scaleRadius_rescaled
    
    # isotropic velocity dispersion in working units
    sigma = np.sqrt(G*M_rescaled/r_rescaled) * sigmaiso_tilde(u,model)

    # cartesian velocities (isotropic orbits) in 100 km/s
    vx = vFactor * rg.normal(0,sigma,N)
    vy = vFactor * rg.normal(0,sigma,N)
    vz = vFactor * rg.normal(0,sigma,N)
    
    # subtract bulk motion (equal mass particles)
    if N > 1:
        vx = vx - np.mean(vx)
        vy = vy - np.mean(vy)
        vz = vz - np.mean(vz)
    
    # extra mass
    if MextraFrac > 0:
        x = np.append(x,rExtra/radiusUnit)
        y = np.append(y,0)
        z = np.append(z,0)
        vx = np.append(vx,0)
        vy = np.append(vy,np.sqrt(G*MextraFrac*M_rescaled*M_tilde(rExtra/scaleRadius,model)))
        # rExtra is in units of scaleRadius, hence dimensionless
        vz = np.append(vz,0)
        
    return x,y,z,vx,vy,vz

def EnergyNBody(t,sol,m_array,epsilonSoft):
    """Energy of N-body system
    arguments:
        t: time array for Plot (physical units)
        sol: solution of ODE (working units by construction)
        m_array: array of particle masses (physical units)
        epsilonSoft (physical units)
    returns kinetic, potential and total energies (in physical units)
    dependencies: numpy as np
    author: Gary Mamon
    """
    N = int(len(sol[0,])/6)

    # kinetic energy in physical units
    mravel = np.ravel((m_array,m_array,m_array)) # shape 3*nParticles
    velocities = sol[:,3*N:6*N] # shape nTimes,3*nParticles
    Ekin = 0.5 * np.sum(mravel*velocities*velocities,axis=1)

    # convert to physical units (noting that m_array is already in physical units)
    Ekin = velocityUnit*velocityUnit * Ekin

    # potential energy: first in working units
    Epot = 0*sol[:,0]
    x = sol[:,0:N]
    y = sol[:,N:2*N]
    z = sol[:,2*N:3*N]
    m_array_rescaled = m_array / massUnit
    epsilonSoftSq_rescaled = (epsilonSoft/radiusUnit)**2
    # r_softened = np.zeros((N,N,len(sol[:,0])))
    rSoft0Arr = []

    # loop over time
    for i,(X,Y,Z) in enumerate(zip(x,y,z)):
        # loop over particles
        for k,(X0,Y0,Z0) in enumerate(zip(X,Y,Z)):
            dx = np.delete(X,k) - X0
            dy = np.delete(Y,k) - Y0
            dz = np.delete(Z,k) - Z0
            m_others = np.delete(m_array_rescaled,k)
            sep2 = dx*dx + dy*dy + dz*dz
            r_softened = np.sqrt(sep2+epsilonSoftSq_rescaled)
            rSoft0Arr.append(r_softened)
            Epot[i] = Epot[i] - m_array_rescaled[k] \
                          *np.sum(m_others/r_softened)
    Epot = G * Epot/2 # divide by 2 because of double counting

    # rescale potential energy to physical units (same as kinetic energy: M_sun (km/s)^2)
    # here insert massUnit because we used rescale units in the calculation in the loop above
    Epot = massUnit * velocityUnit*velocityUnit * Epot

    # total energy
    Etot = Ekin + Epot
    
    # plot
    plt.figure(figsize=(6,6))
    plt.plot(t,Ekin,'r',label='kinetic energy')
    plt.plot(t,Epot,'g',label='potential energy')
    plt.plot(t,Etot,'k',label='total energy')
    plt.xlabel('time (Myr)')
    plt.ylabel('energies ($\mathrm{M}_\odot\,\mathrm{km/s}^2$)')
    relativeError = np.abs(Etot[len(Etot)-1]/Etot[0]-1)
    plt.legend(loc='best')
    plt.title('Energy conserved to ' 
               + "%.2g"%(100*relativeError) + "\%")
    plt.show()
    
    # plot of virial ratio: 2 Ekin / -Epot
    plt.figure(figsize=(6,6))
    plt.plot(t,2*Ekin/(-1*Epot),'r')
    plt.plot(t,0*t+1,'k',ls='dashed')
    plt.xlabel('time (Myr)')
    plt.ylabel('virial ratio')

    return Etot, Ekin, Epot

def Energy3Body(t,sol,m_array,scale_radius,epsilonSoft):
    """Energy of restricted 3-body system
    arguments:
        t: time array for Plot (physical units)
        sol: solution of ODE (working units by construction)
        m_array: array of particle masses (physical units)
        scale_radius 
        epsilonSoft (physical units)
    returns kinetic, potential and total energies (in physical units)
    dependencies: numpy as np
    author: Gary Mamon
    """
    N = int(len(sol[0,])/6)

    # kinetic energy in physical units
    mravel = np.ravel((m_array,m_array,m_array)) # shape 3*nParticles
    velocities = sol[:,3*N:6*N] # shape nTimes,3*nParticles
    Ekin = 0.5 * np.sum(mravel*velocities*velocities,axis=1)

    # convert to physical units (noting that m_array is already in physical units)
    Ekin = velocityUnit*velocityUnit * Ekin

    # potential energy: first in working units
    x = sol[:,0:N]
    y = sol[:,N:2*N]
    z = sol[:,2*N:3*N]
    m_array_rescaled = m_array / massUnit
    epsilonSoftSq_rescaled = (epsilonSoft/radiusUnit)**2
    r = np.sqrt(x*x + y*y + z*z + epsilonSoftSq_rescaled)
    scale_radius_rescaled = scale_radius/radiusUnit
    Epot = -1*G*m_array_rescaled.sum()/scale_radius \
             *np.sum(m_array_rescaled*Phi_tilde(r/scale_radius_rescaled,model),
                     axis=1)

    # rescale potential energy to physical units (same as kinetic energy: M_sun (km/s)^2)
    # here insert massUnit because we used rescale units in the calculation in the loop above
    Epot = massUnit * velocityUnit*velocityUnit * Epot

    # total energy
    Etot = Ekin + Epot
    
    # plot
    plt.figure(figsize=(6,6))
    plt.plot(t,Ekin,'r',label='kinetic energy')
    plt.plot(t,Epot,'g',label='potential energy')
    plt.plot(t,Etot,'k',label='total energy')
    plt.xlabel('time (Myr)')
    plt.ylabel('energies ($\mathrm{M}_\odot\,\mathrm{km/s}^2$)')
    relativeError = np.abs(Etot[len(Etot)-1]/Etot[0]-1)
    plt.legend(loc='best')
    plt.title('Energy conserved to ' 
               + "%.2g"%(100*relativeError) + "\%")
    plt.show()
    
    # plot of virial ratio: 2 Ekin / -Epot
    plt.figure(figsize=(6,6))
    plt.plot(t,2*Ekin/(-1*Epot),'r')
    plt.plot(t,0*t+1,'k',ls='dashed')
    plt.xlabel('time (Myr)')
    plt.ylabel('virial ratio')

    return Etot, Ekin, Epot

def Momentum(sol):
    """Evaluate linear and angular momentum, and center of mass 
        all in working units
    """
    N = int(len(sol[0,])/6)
    
    # 6D coordinates
    x  = sol[:,0:N]
    y  = sol[:,N:2*N]
    z  = sol[:,2*N:3*N]
    vx = sol[:,3*N:4*N]
    vy = sol[:,4*N:5*N]
    vz = sol[:,5*N:6*N]

    # Linear momentum
    LinearMomentumX = np.mean(vx,axis=1)
    LinearMomentumY = np.mean(vy,axis=1)
    LinearMomentumZ = np.mean(vz,axis=1)
    LinearMomentum = np.transpose([LinearMomentumX,LinearMomentumY,
                                    LinearMomentumZ])
    
    # Angular momentum
    AngularMomentumX = np.sum(y*vz-z*vy,axis=1)
    AngularMomentumY = np.sum(z*vx-x*vz,axis=1)
    AngularMomentumZ = np.sum(x*vy-y*vx,axis=1)
    AngularMomentum = np.transpose([AngularMomentumX,AngularMomentumY,
                                     AngularMomentumZ])
    
    # Center of mass
    CenterofMass = np.transpose([np.mean(x,axis=1),np.mean(y,axis=1),
                                  np.mean(z,axis=1)])
    return LinearMomentum, AngularMomentum, CenterofMass

def PlotTrajectory(sol,xlab='x',ylab='x',rmax=None,device=None,lim=None):
    N = int(len(sol[0,])/6)
    if N not in [10,20,50,100,200,500]:
        Nlight = N-1
    else:
        Nlight = N
    plt.figure(figsize=(6,6))
    ax = plt.gca()
    if lim is not None:
        plt.xlim((-lim,lim))
        plt.ylim((-lim,lim))
    else:
        ax.set_aspect('equal')
    plt.xlabel('x (pc)')
    plt.ylabel('y (pc)')
    for k in range(Nlight):
        p = plt.plot(radiusUnit*sol[:,k],radiusUnit*sol[:,N+k])
        color = p[0].get_color()
        # plot initial & final positions
        plt.scatter(radiusUnit*sol[0,k],radiusUnit*sol[0,N+k],marker="$\mathrm{S}$",
                    c=color)
        plt.scatter(radiusUnit*sol[-1,k],radiusUnit*sol[-1,N+k],marker="$\mathrm{E}$",
                    c=color)
    if N > Nlight:
        # plot extra mass (for dynamical friction studies)
        plt.plot(radiusUnit*sol[:,N-1],radiusUnit*sol[:,2*N-1],color='k',lw=3)
        plt.scatter(radiusUnit*sol[0,N-1],radiusUnit*sol[0,2*N-1],color='k',
                    lw=3,marker="$\mathrm{S}$")
        plt.scatter(radiusUnit*sol[-1,N-1],radiusUnit*sol[-1,2*N-1],color='k',
                    lw=3,marker="$\mathrm{E}$")
    if rmax is not None:
        plt.xlim(-1*rmax,rmax)
        plt.ylim(-1*rmax,rmax)
        
    plt.show()

def Plot_xoft(t,sol,connect=True,tlims=None,xlims=None,device=None):
    plt.figure(figsize=(6,6))
    if tlims is not None:
        plt.xlim(tlims)
    if xlims is not None:
        plt.ylim(xlims)
    plt.xlabel('t (Myr)')
    plt.ylabel('x (pc)')
    for k in range(N):
        if connect:
            plt.plot(t,radiusUnit*sol[:,k])
        else:
            plt.scatter(t,radiusUnit*sol[:,k])
    plt.show()
    
def Plot_vxoft(t,sol,connect=True,tlims=None,device=None):
    plt.figure(figsize=(6,6))
    if tlims is not None:
        plt.xlim(tlims)
    plt.xlabel('t (Myr)')
    plt.ylabel('vx (km/s)')
    for k in range(N):
        if connect:
            plt.plot(t,velocityUnit*sol[:,3*N+k]) # v in km/s
        else:
            plt.scatter(t,velocityUnit*sol[:,3*N+k])
    plt.show()
    
def Plot_vxofx(sol,connect=True,xlims=None,device=None):
    N = int(len(sol[0,])/6)
    plt.figure(figsize=(6,6))
    if xlims is not None:
        plt.xlim(xlims)
    plt.xlabel('x (kpc)')
    plt.ylabel('vx (km/s)')
    for k in range(N):
        if connect:
            plt.plot(sol[:,k],velocityUnit*sol[:,3*N+k])
        else:
            plt.scatter(sol[:,k],velocityUnit*sol[:,3*N+k]) # v in km/s
    plt.show()
    
def Plot_vrofr(sol,step,xlog=True,device=None):
    N = int(len(sol[0,])/6)
    plt.figure(figsize=(6,6))
    plt.xlabel(r'$r$ (kpc)',usetex=True)
    plt.ylabel(r'$v_r$ (km/s)',usetex=True)
    if step < 0:
        step2 = slice(0,len(sol))
    else:
        step2 = step
    x = sol[step2,0:N]
    y = sol[step2,N:2*N]
    z = sol[step2,2*N:3*N]
    vx = sol[step2,3*N:4*N]
    vy = sol[step2,4*N:5*N]
    vz = sol[step2,5*N:6*N]
    r = np.sqrt(x*x + y*y + z*z)
    vr = (x*vx + y*vy + z*vz) / r
    if xlog:
        plt.xscale('log')
    if step >= 0:
        plt.scatter(r,velocityUnit*vr)  
        plt.title("step " + str(step))
    else:
        plt.plot(r,velocityUnit*vr)
        plt.title("all steps")
    plt.show()
    
def Plot_roft(t,sol,connect=True,tlims=None,rlims=None,yLog=False,device=None,Mextra=None):
    N = int(len(sol[0,])/6)
    plt.figure(figsize=(6,6))
    if yLog:
        plt.yscale('log')
    if tlims is not None:
        plt.xlim(tlims)
    if rlims is not None:
        plt.ylim(rlims)
    plt.xlabel('$t$ (Myr)')
    plt.ylabel('$r$ (pc)')
    
    # plot evolution of geometric mean radius in thick blue
    rsq = np.zeros((len(sol),N))
    r = np.zeros((len(sol),N))
    if N not in [10,20,50,100,200,500]:
        Nlight = N-1
    else:
        Nlight = N
    for k in range(Nlight):
        rsq[:,k] = sol[:,k]**2 + sol[:,N+k]**2 + sol[:,2*N+k]**2
        r[:,k] = radiusUnit * np.sqrt(rsq[:,k])
        if connect:
            plt.plot(t,r[:,k])
        else:
            plt.scatter(t,r[:,k])

    # geometric mean
    if connect:
        rGeomean = stats.mstats.gmean(r[:,0:Nlight],axis=1)
        rMedian = np.median(r[:,0:Nlight],axis=1)
        plt.plot(t,rGeomean,color='b',lw=3,label='Geometric mean')
        plt.plot(t,rMedian,color='b',lw=3,ls='dashed',label='Median')
        
    # add massive particle in thick black
    del rsq
    if N > Nlight:
        rsq = sol[:,N-1]**2 + sol[:,2*N-1]**2 + sol[:,3*N-1]**2
        rMassive = radiusUnit * np.sqrt(rsq)
        if connect:
            plt.plot(t,rMassive,color='k',lw=3,label='Massive particle')
        else:
            plt.scatter(t,rMassive,c='k')
    else:
        print("N = ", N)
    plt.legend(loc='best')
    plt.show()
    
def doAll(N=20,M=1e5,scaleRadius=2,epsilonSoft_over_a=0.05,model="Plummer",MextraFrac=0,
        vFactor=1,rExtra=1, tEnd=3,nSteps=2000,sim='NBody',seed=100,plot=True):
    """Solve ODE and make plots
    arguments:
        N: number of particles
        M: cluster mass (solar units, default 10^5)
        scaleRadius: scale radius (pc, default 2)
        model: density model (Hernquist or Plummer, default Plummer)
        epsilonSoft_over_a: ratio of softening scale to cluster scale radius (default 0.05)
        MextraFrac: fraction of extra mass in circular orbit (default 0)
        vFactor: multiplicative factor for velocities (default 1)
        rExtra: radius of extra massive particle (in units of rscale, default 1)
        tEnd: final time (Myr, default 3)
        nSteps: maximum number of internal timesteps (default 2000)
        seed: seed for random number generator (None for random seed, default 100)
        plot: make plot (boolean, default=True)
    returns:
        sol: solution of ODE ([nSteps,6*N_tot] numpy array) 
    dependencies: numpy as np, rminus2overa
    author: Gary Mamon
    """

    # softening length
    epsilonSoft = epsilonSoft_over_a * scaleRadius

    # rescaled parameters
    M_rescaled = M / massUnit
    a_rescaled = scaleRadius / radiusUnit
    tEnd_rescaled = tEnd / timeUnit
    epsilonSoft_rescaled = epsilonSoft_over_a * a_rescaled
    rExtra_rescaled = rExtra / radiusUnit

    # Initial conditions in working (astro) units
    x,y,z,vx,vy,vz = InitialConditions(N=N,M=M,scaleRadius=scaleRadius,
                                        model=model,
                                        vFactor=vFactor,MextraFrac=MextraFrac,
                                        rExtra=rExtra,seed=seed)

    print("median radius = ",np.median(np.sqrt(x*x + y*y + z*z)))
    print("velocity dispersion in x y z = %4.2f"%np.std(vx),'%4.2f'%np.std(vy),
          '%4.2f'%np.std(vz),"km/s")
    # exit if any particles are deeper inside core of cluster than softening scale
    r0 = np.sqrt(x*x + y*y + z*z)
    if np.min(r0) < epsilonSoft_rescaled:
        raise ValueError("min(r(t=0))=%.3f"%(radiusUnit*np.min(r0)) 
                         + " smaller than epsilonSoft =%.3f"%(radiusUnit*epsilonSoft_rescaled)
                         + "(both in physical units): exiting ...")

    if nSteps == 0:
        # plot x,y position if nSteps = 0
        fig,ax = plt.subplots()
        plt.scatter(radiusUnit*x, radiusUnit*y,s=1000/len(x))
        limits = [-5,25]
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_aspect('equal')
        plt.xlabel('$x$ (pc)')
        plt.ylabel('$y$ (pc)')
        plt.show()
        sys.exit(0)

    # Vector of initial conditions
    w0 = np.ravel((x,y,z,vx,vy,vz))   # flatten to 1D array

    # particle masses (in rescaled units)
    m_array = np.ones(N) * M / N
    m_array_rescaled = m_array / massUnit
    
    # add extra particle for dynamical friction studies
    if MextraFrac > 0:
        m_array_rescaled = np.append(m_array_rescaled,M_rescaled*MextraFrac)
        N_tot = N+1
    else:
        N_tot = N
    m_array = massUnit * m_array_rescaled
    
    # advance particles
    t = np.linspace(0,tEnd,1+nSteps)
    t_rescaled = t / timeUnit
    args = (G,N_tot,M_rescaled,m_array_rescaled,a_rescaled,epsilonSoft_rescaled,model)
    start_time = time.time() # to compute elapsed time for full solution of ODE
    if sim == 'NBody':
        print("entering ODEINT for N-body...")
        sol = odeint(derivsNbody,w0,t_rescaled,args,atol=atol,rtol=rtol,mxstep=nSteps) 
    elif sim == '3Body':
        print("entering ODEINT for 3-body...")
        sol = odeint(derivs3body,w0,t_rescaled,args,atol=atol,rtol=rtol,mxstep=nSteps) 
    print("elapsed time taken:", "% .2f seconds" % (time.time()-start_time))

    if plot:
        # r vs t
        Plot_roft(t,sol,yLog=True)

    #     # v_r vs r
    #     Plot_vrofr(sol,0,xlog=True)
    #     Plot_vrofr(sol,nSteps-1)

        # x,y trajectories
        PlotTrajectory(sol,device=None)
        PlotTrajectory(sol,device=None,rmax=scaleRadius)

        if sim == 'NBody':
            Etot, Ekin, Epot = EnergyNBody(t,sol,m_array,epsilonSoft)
        elif sim == '3Body':
            Etot, Ekin, Epot = Energy3Body(t,sol,m_array,scaleRadius,epsilonSoft)
        print("energy conserved to" + "% .3f"% (100*np.abs(Etot[len(Etot)-1]/Etot[0]-1))
            + "%")

        p, L, CofM = Momentum(sol)
        print("initial time: p=",p[0,:],"L=",L[0,:],"CofM=",CofM[0,:])
        print("final time:   p=",p[nSteps-1,:],"L=",L[nSteps-1,:],"CofM=",CofM[nSteps-1,:])

    return sol

# MAIN

# constants
TINY = 1e-12
HUGE = 1e30
G = 4.3010  # Newton's gravitational constant in astrophysical units
radiusUnit = 1 # pc
velocityUnit = 1 # km/s
massUnit = 1000 # solar masses
timeUnit = 0.9778 # Myr

# default arguments
scaleRadius = 2 # pc
M = 1e5       # solar masses
atol = 1e-9
rtol = 0.0001   # relative tolerance for solving differential equations
plot = True
model = "Plummer"
epsilon_over_a = 0.05
epsilonSoft = 0.05*scaleRadius

# Test of standard equilibrium run
# solStd = doAll(N=20,vFactor=1,tEnd=5)

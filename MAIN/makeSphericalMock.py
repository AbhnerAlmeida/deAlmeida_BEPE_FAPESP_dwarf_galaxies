import agama
import sys # for debugging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
plt.rc('text', usetex=True)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})

# 1st component: stars, 2nd component: stellar-mass BH;
# both described by any spherically-symmetric density profile available in Agama
# (Table 1 in reference.pdf lists possible models and their parameters),
# and the DF is derived from the density profile in the combined total potential
# with the Cuddeford-Osipkov-Merritt inversion formula, which has two free params:
# beta0 is the central value of velocity anisotropy (-0.5 <= beta0 < 1), and
# r_a is the anisotropy radius (may be infinite)

def MakeMock(name=None,model=None,mass=None,meanmass=None,radius=None,param2=None,
             beta0=0,rbeta=np.inf,
             sizeunit='kpc',fileprefix=None,plot=False,title=None,
             plotcolors=None,device=None):
    """

    Parameters
    ----------
    name : str or tuple of strings
        Name of component. Default None
    model : str or tuple of strings
        Name of density model. 
        Allowed models:
            Plummer, NFW, Sersic, gPlummer
            Default None.
    mass : float or tuple of floats
        Total mass of component (M_Sun). Default None.
    meanmass : float or or tuple of floats
        Mean mass of particles of component (M_Sun). Default None.
    radius : float or tuple of floats
        Scale radius of component (kpc). Default None.
    param2 : float or tuple of floats
        2nd parameter of components (e.g. for Sersic, gNFW, pc). Default None.
    beta0 : float or tuple of floats
        Velocity anisotropy at center. Default 0.
    rbeta : float or tuple of floats
        Radius of transition from central to outer velocity anisotropy (sizeunit). Default np.inf
    sizeunit : str or tuple of strings
        unit of sizes ('pc', 'kpc', or 'Mpc')
    fileprefix : str or tuple of strings
        file prefix for mock file and for figures. Default None.
    plot : boolean
        Plot profiles? The default is False.
    plotcolors : str or tuple of strings
        colors for plots. Default None.
    device : str or tuple of strings
        prefix for plot device ('0' for fileprefix). Default None.

    Returns
    -------
    None
    
    Author: Gary Mamon (gam AAT iap.fr) with much help from Eugene Vasiliev

    """
    
    # convert scalars to tuples if necessary
    if np.isscalar(name):
        name = (name,)
        model = (model,)
        mass = (mass,)
        meanmass = (meanmass,)
        radius = (radius,)
        param2 = (param2,)
        beta0 = (beta0,)
        rbeta = (rbeta,)
        plotcolors = (plotcolors,)
    else:
        Nc = len(name)
        if param2 is None:
            param2 = tuple([param2 for i in range(Nc)])
        if beta0 == 0:
            beta0 = tuple([beta0 for i in range(Nc)])
        if rbeta == np.inf:
            rbeta = tuple([rbeta for i in range(Nc)])
        if plotcolors is None:
            plotcolors = tuple([plotcolors for i in range(Nc)])
            
    # size unit
    if sizeunit == 'pc':
        length = 0.001
    elif sizeunit == 'kpc':
        length  = 1
    elif sizeunit == 'Mpc':
        length = 1000
    else: 
        ValueError("Cannot recognize sizeunit=" + sizeunit)
    agama.setUnits(length=length, velocity=1, mass=1) # working units: 1 pc/kpc/Mpc, 1 km/s, 1 Msun
    
    # loop over components 
    DensAll = ()
    print("\nDensity models ...")
    for i, Name in enumerate(name):
        # number of particles
        print("model = ", model[i],"...")
        Number = int(mass[i]/meanmass[i])        
        print("Component", Name, "N=", Number, "...")
        
        # density model
        if model[i] == 'Sersic':
            Dens = agama.Potential(type=model[i], scaleRadius=radius[i], sersicIndex=param2[i], mass=mass[i])
        elif model[i] == "gPlummer":
            Dens = agama.Potential(type="Spheroid", scaleRadius=radius[i], mass=mass[i],
                                   gamma=param2[i],beta=5,alpha=2)
            print("gPlummer: mass = ", mass[i])
            print("gPlummer a = ", radius[i])
            print("gPlummer param2 = ",param2[i])
            print("gPlummer: dens 0.1 1 = ", Dens.density(0.1,0,0),Dens.density(1,0,0))
        else:
            Dens = agama.Potential(type=model[i], scaleRadius=radius[i], mass=mass[i])
        print("Dens = ", Dens)
        DensAll = DensAll + tuple(Dens)
        
    # (total) gravitational potential
    print("Gravitational potential ...")
    potential = agama.Potential(*DensAll)
    print("potential = ", potential)
    
    # prepare plot
    if plot:
        rbins=agama.nonuniformGrid(100, 0.01, 100)
        xyz=np.column_stack((rbins, rbins*0, rbins*0))
        ax=plt.subplots(1, 2, figsize=(15,8))[1]
        if plotcolors is None:
            plotcolors = 'r','g','b'
        
    # distribution functions and moments
    xvall = np.empty(shape=[0,6])
    mall = np.empty(shape=[0])

    for i, Name in enumerate(name):
        Number = int(mass[i]/meanmass[i])   
        print("Distribution function for", Name, "...")
        print("DensAll[i]=",DensAll[i])
        DistribFunc = agama.DistributionFunction(type='quasispherical', 
                                                 beta0=beta0[i], r_a=rbeta[i], 
                                                 density=DensAll[i], potential=potential)
        print("positions and velocities for ", Name, "...")
        xv, m = agama.GalaxyModel(potential, DistribFunc).sample(Number)
        xvall = np.append(xvall,xv,axis=0)
        mall = np.append(mall,m)
                    
        if plot:
            print("Moments of DF for", Name,"...")
            # print("r =",rbins)
            dens,sigma = agama.GalaxyModel(potential, DistribFunc).moments(xyz)
            # print("dens=",dens)
            # print("sigma=",sigma)
            ax[0].plot(rbins,dens, label=Name, color=plotcolors[i])
            ax[1].plot(rbins,sigma[:,0]**0.5, label=Name,color=plotcolors[i])
            
    # min max and median radius
    r = np.sqrt(xvall[:,0]*xvall[:,0] + xvall[:,1]*xvall[:,1] + xvall[:,2]*xvall[:,2])
    print("r statistics: min median max = ",np.min(r),np.median(r),np.max(r))

    # write to file
    mvxall = np.column_stack((mall,xvall))
    # agama.writeSnapshot(filePrefix + ".txt",mvxall)
    print("saving to ", fileprefix + ".txt ...")
    np.savetxt(fileprefix + ".txt",mvxall)
    
    print("Finishing plot ...")
    if plot:
        ax[0].set_xlabel('$r$ [pc]',fontsize=20)
        ax[0].set_ylabel('density ($\mathrm{M}_\odot\,\mathrm{pc}^{-3}$)',fontsize=20)
        ax[1].set_xlabel('$r$ [pc]',fontsize=20)
        ax[1].set_ylabel('velocity (dispersion) [km$\,$s$^{-1}$]',fontsize=20)
        ax[1].plot(rbins, (-potential.force(xyz)[:,0]*rbins)**0.5, label='$v_\mathrm{circ}$', color='k')
        ax[0].legend(fontsize=20)
        ax[1].legend(fontsize=20)
        ax[0].set_xscale('log')
        ax[0].set_yscale('log')
        ax[1].set_xscale('log')
        ax[1].set_yscale('log')
        ax[0].set_xlim(0.01,100)
        ax[1].set_xlim(0.01,100)
        ax[0].xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
        ax[1].xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
        ax[1].yaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
        ax[1].yaxis.set_major_locator(mticker.FixedLocator(np.array([1,2,5,10,20,50,100,200,500,1000])))
        ax[0].tick_params(axis="x",labelsize=18)
        ax[0].tick_params(axis="y",labelsize=18)
        ax[1].tick_params(axis="x",labelsize=18)
        ax[1].tick_params(axis="y",labelsize=18)   
        if title is not None:
            plt.title(title,fontsize=20)
        if device is None:
            plt.show()
        else:
            if device == '0':
                file = fileprefix + ".pdf"
            else:
                file = device + ".pdf"
            plt.savefig(file)
    return()

# examples 


MakeMock(name="stars",model="Plummer",mass=1e4,meanmass=10,
         radius=2,sizeunit='pc',plot=True,fileprefix="test",title="Plummer")

print("")

MakeMock(name=("stars","CUO"),model=("Plummer","Plummer"),mass=(1e4,100),meanmass=(10,1),
          radius=(2,0.5),param2=(0,0.),sizeunit='pc',plot=True,fileprefix="test",
          title="Plummer+Plummer")

print("")
MakeMock(name=("stars","CUO"),model=("gPlummer","Plummer"),mass=(1e4,100),meanmass=(10,1),
          radius=(2,0.5),param2=(0.9,0),sizeunit='pc',plot=True,fileprefix="test",
          title="gPlummer+Plummer")


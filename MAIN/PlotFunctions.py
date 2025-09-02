import os

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
np.seterr(divide='ignore') # ignore divide by zero

from scipy.stats import spearmanr
import ExtractTNG as ETNG   
import TNGFunctions as TNG
import MATH
import warnings
import h5py
import shutil
import matplotlib.colors as mplcol   
from matplotlib.patches import Patch          # for log colorbars
import matplotlib.colors as mcolors

from scipy.signal import argrelextrema
from scipy import stats
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import FixedLocator
from matplotlib.ticker import FixedFormatter
from matplotlib.offsetbox import AnchoredText
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
from scipy.interpolate import make_interp_spline
from mpl_toolkits.axes_grid1 import AxesGrid
from sphviewer.tools import QuickView
from matplotlib.collections import LineCollection

warnings.filterwarnings( "ignore" )
plt.style.use('abhner.mplstyle')

# cosmological parameters
Omegam0 = 0.3089
h = 0.6774

#Paths
SaveSubhaloPath = os.getenv("HOME")+'/TNG_Analyzes/SubhaloHistory/'
SIMTNG = 'TNG50'
Nsim = '-1'
dfTime = pd.read_csv(os.getenv("HOME")+'/TNG_Analyzes/SubhaloHistory/SNAPS_TIME.csv')

colors = {
        #Classes
        'Normal': 'darkorange',
        'SBC': 'forestgreen',
        'MBC': 'royalblue',
        
        'SBCEmpty': 'white',
        'MBCEmpty': 'white',
        'NormalEmpty': 'white',

        'Diffuse': 'firebrick',
        'SubDiffuse': 'violet',
        
        'NormalCentral': 'darkorange',
        'DiffuseCentral': 'firebrick',
        'SubDiffuseCentral': 'violet',
        
        'NormalSatelliteNotInteract': 'darkorange',
        'DiffuseSatelliteNotInteract': 'firebrick',
        'SubDiffuseSatelliteNotInteract': 'violet',
        
        'NormalSatelliteInteract': 'darkorange',
        'DiffuseSatelliteInteract': 'firebrick',
        'SubDiffuseSatelliteInteract': 'violet',
    
        'NormalSatelliteFalse': 'darkorange',
        'SBCSatelliteFalse': 'forestgreen',
        'MBCSatelliteTrue': 'royalblue',
        
        'NormalSatelliteTrue': 'darkorange',
        'SBCSatelliteTrue': 'forestgreen',
        'MBCSatelliteFalse': 'royalblue',
        
        'NormalSatellite': 'darkorange',
        'SBCSatellite': 'forestgreen',
        'MBCSatellite': 'royalblue',
        
        
        'NormalSatelliteDMrich': 'darkorange',
        'SBCSatelliteDMrich': 'forestgreen',
        'MBCSatelliteDMrich':  'royalblue',
        
        'NormalSatelliteDMpoor': 'darkorange',
        'SBCSatelliteDMpoor': 'forestgreen',
        'MBCSatelliteDMpoor':  'royalblue',
        'SBCSatelliteDMpoorError': 'tab:green',

        'SBCSatelliteDMpoorMetalRich': 'forestgreen',

        'SBCSatelliteDMpoorMetalRichError': 'tab:green',

        'SatelliteDMpoorEntryToNoGas':  'blue',
        'SatelliteDMpoorNoGasToFinal':  'red',

        'SatelliteEMNew':  'blue',
        'SatelliteESNew':  'red',

        'NormalWithoutBH': 'darkorange',
        'SBCWithoutBH': 'forestgreen',
        'MBCWithoutBH': 'royalblue',
        'NormalWithBH': 'darkorange',
        'SBCWithBH': 'forestgreen',
        'MBCWithBH': 'royalblue',

        'SBCBornYoung': 'lime',


        
        'NormalCentral': 'darkorange',
        'SBCCentral': 'forestgreen',
        'MBCCentral': 'royalblue',

        'TNGrage':  'gray',
        'Selected':  'none',
        'SatelliteSelected':  'black',
        'CentralSelectedEmpty':  'none',
        'BadFlag':  'none',
        'Satellite': 'none',
        'GMM': 'red',
        

        'SBCGamaColor': 'darkseagreen', 
        'MBCGamaColor': 'lightblue', 
        'NormalGamaColor': 'navajowhite', 
        'GAMAColor': 'crimson', 
        
        'SatelliteDMrich':  'blue',
        'SatelliteDMpoor':  'red',
        'SatelliteNotInteract':  'blue',
        'SatelliteInteract':  'red',
        'Central':  'black',
        
        'SatelliteDMrichError':  'tab:blue',
        'SatelliteDMpoorError':  'tab:red',
        
        'SatelliteNotInteractError':  'tab:blue',
        'SatelliteInteractError':   'tab:red',
        'CentralError':  'gray',
        
        #Error
        'NormalError': 'tab:orange',
        'SBCError': 'tab:green',
        'MBCError': 'tab:blue',
        'DiffuseError': 'tab:red',
        'SubDiffuseError': 'tab:pink',

        'NormalWithoutBHError': 'tab:orange',
        'SBCWithoutBHError': 'tab:green',
        'MBCWithoutBHError':  'tab:blue',
        'NormalWithBHError': 'tab:orange',
        'SBCWithBHError': 'tab:green',
        'MBCWithBHError':  'tab:blue',
        
        #Colorbar
        'NormalColorbar': 'darkorange',
        'SBCColorbar': 'forestgreen',
        'MBCColorbar': 'royalblue',
        
        'SBCColorbarEmpty': 'white',
        'MBCColorbarEmpty': 'white',
        
        'NormalColorbarEdge': 'darkorange',
        'SBCColorbarEdge': 'forestgreen',
        'MBCColorbarEdge': 'royalblue',
        
        'SubDiffuseColorbar': 'violet',
        'DiffuseColorbar': 'firebrick',
        
        'SubDiffuseColorbarEmpty': 'white',
        'DiffuseColorbarEmpty': 'white',
        'NormalColorbarEmpty': 'white',

        'GAMAColorbar': 'crimson', 
        
        'GAMAColorbar': 'crimson', 
        
        
        'LoseTheirGasColorbar': 'black',
        'DontLoseTheirGasColorbar': 'black',
        
        'DontLoseTheirGasSatelliteColorbar': 'black',
        'LoseTheirGasSatelliteColorbar': 'black',
        'CentralColorbar': 'black',

        
        'SBCLoseTheirGasColorbarLegend': 'white',
        'MBCLoseTheirGasColorbarLegend':  'white',
        'NormalLoseTheirGasColorbarLegend':  'white',
        
        'SBCLoseTheirGasSatelliteColorbarLegend': 'white',
        'MBCLoseTheirGasSatelliteColorbarLegend':  'white',
        'NormalLoseTheirGasSatelliteColorbarLegend':  'white',
        
        'SBCCentralColorbar':   'forestgreen',
        'MBCCentralColorbar':  'royalblue',
        'NormalCentralColorbar':  'darkorange',

        #Random
        '0': 'red', 
        '1': 'sienna',
        '2': 'darkorange',
        '3': 'purple',
        '4': 'lime',
        '5': 'g',
        '6': 'dodgerblue',
        '7':  'b',
        '8': 'brown',
        '9': 'lawngreen', 
        '10': 'turquoise',
        '11': 'steelblue',
        '12':  'indigo',
        '13': 'tab:blue',
        '14': 'gold',   
        
        #Subhalos
        'SubhaloHalfmassRadType0': 'green',
        'SubhaloHalfmassRadType1': 'purple',
        'SubhaloHalfmassRadType4': 'blue',

        'Mgas_Norm_Max': 'green',
        'MDM_Norm_Max': 'purple',
        'Mstar_Norm_Max': 'blue',
        
        'SubhalosSFRInHalfRad': 'darkblue',
        'SubhalosSFRwithinHalfandRad': 'darkred',


        'r_over_R_Crit200': 'darkorange',
        'r_over_R_Crit200_FirstGroup': 'red',
        
        'Group_M_Crit200FinalGroup': 'darkorange',
        'Group_M_Crit200': 'red',
        
        'SubhaloStellarMass_in_Rhpkpc': 'royalblue',
        'SubhaloStellarMass_Above_Rhpkpc': 'blue',
        'DMMass_In_Rhpkpc': 'royalblue',
        'DMMass_Above_Rhpkpc': 'blue',
        'StarMass_In_Rhpkpc': 'darkblue',
        'StarMass_Above_Rhpkpc': 'tab:blue',
        'GasMassInflow_In_Rhpkpc': 'limegreen',
        'GasMassInflow_Above_Rhpkpc': 'darkgreen',
        
        'GasMass_In_Rhpkpc': 'royalblue',
        'GasMass_Above_Rhpkpc': 'blue',
        
        'GasMass_In_TrueRhpkpc': 'black',
        'GasMass_Above_TrueRhpkpc': 'black',
        
        'sSFR_In_TrueRhpkpc': 'royalblue',
        'sSFR_Above_TrueRhpkpc': 'blue',
        
        'SBCLoseTheirGasColorbar': 'forestgreen',
        'MBCLoseTheirGasColorbar':  'royalblue',
        
        'SBCLoseTheirGas': 'forestgreen',
        'MBCLoseTheirGas': 'royalblue',
        
        'SBCDontLoseTheirGas': 'forestgreen',
        'MBCDontLoseTheirGas': 'royalblue',
        
        'NormalLoseTheirGas': 'darkorange',
        
        'NormalLoseTheirGasColorbar': 'darkorange',
        'NormalDontLoseTheirGasColorbar': 'darkorange',
        
        'NormalLoseTheirGasSatelliteColorbar': 'darkorange',
        'NormalDontLoseTheirGasSatelliteColorbar': 'darkorange',
        'SBCLoseTheirGasSatelliteColorbar': 'forestgreen',
        'MBCLoseTheirGasSatelliteColorbar':  'royalblue',
        'SBCDontLoseTheirGasSatelliteColorbar': 'forestgreen',
        'MBCDontLoseTheirGasSatelliteColorbar':  'royalblue',
        
        
        

        
        
        'NormalDontLoseTheirGas': 'darkorange',

        'StarMass_In_MultRhpkpc_plus000dex': 'limegreen',
        'StarMass_In_MultRhpkpc_plus015dex': 'darkgreen',
        'StarMass_In_MultRhpkpc_plus025dex': 'royalblue',
        'StarMass_In_MultRhpkpc_plus050dex': 'darkblue',
        'StarMass_In_MultRhpkpc_plus075dex': 'purple',

        'Starvrad_In_MultRhpkpc_plus000dex': 'limegreen',
        'Starvrad_In_MultRhpkpc_plus015dex': 'darkgreen',
        'Starvrad_In_MultRhpkpc_plus025dex': 'royalblue',
        'Starvrad_In_MultRhpkpc_plus050dex': 'darkblue',
        'Starvrad_In_MultRhpkpc_plus075dex': 'purple',

        'GasMass_In_MultRhpkpc_plus000dex': 'limegreen',
        'GasMass_In_MultRhpkpc_plus015dex': 'darkgreen',
        'GasMass_In_MultRhpkpc_plus025dex': 'royalblue',
        'GasMass_In_MultRhpkpc_plus050dex': 'darkblue',
        'GasMass_In_MultRhpkpc_plus075dex': 'purple',

        'Gasvrad_In_MultRhpkpc_plus000dex': 'limegreen',
        'Gasvrad_In_MultRhpkpc_plus015dex': 'darkgreen',
        'Gasvrad_In_MultRhpkpc_plus025dex': 'royalblue',
        'Gasvrad_In_MultRhpkpc_plus050dex': 'darkblue',
        'Gasvrad_In_MultRhpkpc_plus075dex': 'purple',

        'GasMassInflow_In_MultRhpkpc_plus000dex': 'limegreen',
        'GasMassInflow_In_MultRhpkpc_plus015dex': 'darkgreen',
        'GasMassInflow_In_MultRhpkpc_plus025dex': 'royalblue',
        'GasMassInflow_In_MultRhpkpc_plus050dex': 'darkblue',
        'GasMassInflow_In_MultRhpkpc_plus075dex': 'purple',

        'sSFR_In_MultRhpkpc_plus000dex': 'limegreen',
        'sSFR_In_MultRhpkpc_plus015dex': 'darkgreen',
        'sSFR_In_MultRhpkpc_plus025dex': 'royalblue',
        'sSFR_In_MultRhpkpc_plus050dex': 'darkblue',
        'sSFR_In_MultRhpkpc_plus075dex': 'purple',

        'SFR_In_MultRhpkpc_plus000dex': 'limegreen',
        'SFR_In_MultRhpkpc_plus015dex': 'darkgreen',
        'SFR_In_MultRhpkpc_plus025dex': 'royalblue',
        'SFR_In_MultRhpkpc_plus050dex': 'darkblue',
        'SFR_In_MultRhpkpc_plus075dex': 'purple',
        
      
        'StarMass_In_Rhpkpc_entry_minus200dex': 'salmon',
        'StarMass_In_Rhpkpc_entry_minus100dex': 'darkorange',
        'StarMass_In_Rhpkpc_entry_minus150dex': 'red',
        'StarMass_In_Rhpkpc_entry_plus100dex': 'darkviolet',
        
        'GasMass_In_Rhpkpc_entry_minus200dex': 'salmon',
        'GasMass_In_Rhpkpc_entry_minus100dex': 'darkorange',
        'GasMass_In_Rhpkpc_entry_minus150dex': 'red',
        'GasMass_In_Rhpkpc_entry_plus100dex': 'darkviolet',
        
        'Starvrad_In_Rhpkpc_entry_minus200dex': 'salmon',
        'Starvrad_In_Rhpkpc_entry_minus100dex': 'darkorange',
        'Starvrad_In_Rhpkpc_entry_minus150dex': 'red',
        'Starvrad_In_Rhpkpc_entry_plus100dex': 'darkviolet',

        'Gasvrad_In_Rhpkpc_entry_minus200dex': 'salmon',
        'Gasvrad_In_Rhpkpc_entry_minus100dex': 'darkorange',
        'Gasvrad_In_Rhpkpc_entry_minus150dex': 'red',
        'Gasvrad_In_Rhpkpc_entry_plus100dex': 'darkviolet',

        'GasMassInflow_In_Rhpkpc_entry_minus200dex': 'salmon',
        'GasMassInflow_In_Rhpkpc_entry_minus100dex': 'darkorange',
        'GasMassInflow_In_Rhpkpc_entry_minus150dex': 'red',
        'GasMassInflow_In_Rhpkpc_entry_plus100dex': 'darkviolet',
        
        'sSFR_In_Rhpkpc_entry_minus200dex': 'salmon',
        'sSFR_In_Rhpkpc_entry_minus100dex': 'darkorange',
        'sSFR_In_Rhpkpc_entry_minus150dex': 'red',
        'sSFR_In_Rhpkpc_entry_plus100dex': 'darkviolet',

        'SFR_In_Rhpkpc_entry_minus200dex': 'salmon',
        'SFR_In_Rhpkpc_entry_minus100dex': 'darkorange',
        'SFR_In_Rhpkpc_entry_minus150dex': 'red',
        'SFR_In_Rhpkpc_entry_plus100dex': 'darkviolet',

      'Starrho_In_Rhpkpc_entry_minus200dex': 'salmon',
      'Starrho_In_Rhpkpc_entry_minus100dex': 'darkorange',
      'Starrho_In_Rhpkpc_entry_minus150dex': 'red',
      'Starrho_In_Rhpkpc_entry_plus100dex': 'darkviolet',
      'Starrho_In_Rhpkpc_entry_minus050dex':'limegreen',
      'Starrho_In_Rhpkpc_entry_minus025dex': 'limegreen',
      'Starrho_In_Rhpkpc_entry': 'darkgreen',
      'Starrho_In_Rhpkpc_entry_plus025dex': 'dodgerblue',
      'Starrho_In_Rhpkpc_entry_plus050dex': 'darkblue',
      
      'Gasrho_In_Rhpkpc_entry_minus200dex':  'salmon',
      'Gasrho_In_Rhpkpc_entry_minus100dex': 'darkorange',
      'Gasrho_In_Rhpkpc_entry_minus150dex': 'red',
      'Gasrho_In_Rhpkpc_entry_plus100dex':  'darkviolet',
      'Gasrho_In_Rhpkpc_entry_minus050dex': 'gold',
      'Gasrho_In_Rhpkpc_entry_minus025dex': 'limegreen',
      'Gasrho_In_Rhpkpc_entry': 'darkgreen',
      'Gasrho_In_Rhpkpc_entry_plus025dex': 'dodgerblue',
      'Gasrho_In_Rhpkpc_entry_plus050dex': 'darkblue',


        'StarMass_In_Rhpkpc_entry_minus050dex': 'gold',
        'StarMass_In_Rhpkpc_entry_minus025dex': 'limegreen',
        'StarMass_In_Rhpkpc_entry_plus025dex': 'dodgerblue',
        'StarMass_In_Rhpkpc_entry_plus050dex': 'darkblue',
        
        'Starvrad_In_Rhpkpc_entry_minus050dex': 'gold',
        'Starvrad_In_Rhpkpc_entry_minus025dex': 'limegreen',
        'Starvrad_In_Rhpkpc_entry_plus025dex': 'dodgerblue',
        'Starvrad_In_Rhpkpc_entry_plus050dex': 'darkblue',
        
        'GasMass_In_Rhpkpc_entry_minus050dex': 'gold',
        'GasMass_In_Rhpkpc_entry_minus025dex': 'limegreen',
        'GasMass_In_Rhpkpc_entry_plus025dex': 'dodgerblue',
        'GasMass_In_Rhpkpc_entry_plus050dex': 'darkblue',
        
        'Gasvrad_In_Rhpkpc_entry_minus050dex': 'gold',
        'Gasvrad_In_Rhpkpc_entry_minus025dex': 'limegreen',
        'Gasvrad_In_Rhpkpc_entry_plus025dex': 'dodgerblue',
        'Gasvrad_In_Rhpkpc_entry_plus050dex': 'darkblue',
        
        'GasMassInflow_In_Rhpkpc_entry_minus050dex': 'gold',
        'GasMassInflow_In_Rhpkpc_entry_minus025dex': 'limegreen',
        'GasMassInflow_In_Rhpkpc_entry_plus025dex': 'dodgerblue',
        'GasMassInflow_In_Rhpkpc_entry_plus050dex': 'darkblue',
        
        'sSFR_In_Rhpkpc_entry_minus050dex': 'gold',
        'sSFR_In_Rhpkpc_entry_minus025dex': 'limegreen',
        'sSFR_In_Rhpkpc_entry_plus025dex': 'dodgerblue',
        'sSFR_In_Rhpkpc_entry_plus050dex': 'darkblue',
        
        'SFR_In_Rhpkpc_entry_minus050dex': 'gold',
        'SFR_In_Rhpkpc_entry_minus025dex': 'limegreen',
        'SFR_In_Rhpkpc_entry_plus025dex': 'dodgerblue',
        'SFR_In_Rhpkpc_entry_plus050dex': 'darkblue',
        
        'StarMass_In_Rhpkpc_entry': 'darkgreen',
        'Starvrad_In_Rhpkpc_entry': 'darkgreen',
        'GasMass_In_Rhpkpc_entry': 'darkgreen',
        'Gasvrad_In_Rhpkpc_entry': 'darkgreen',
        'GasMassInflow_In_Rhpkpc_entry': 'darkgreen',
        'sSFR_In_Rhpkpc_entry': 'darkgreen',
        'SFR_In_Rhpkpc_entry': 'darkgreen',
        
        'StarMass_minus050dex_r_Rhpkpc_entry':  'darkgreen',
        'StarMass_Rhpkpc_entry_r_plus050dex': 'royalblue',
        'Starvrad_minus050dex_r_Rhpkpc_entry': 'darkgreen',
        'Starvrad_Rhpkpc_entry_r_plus050dexex': 'royalblue',
        
        'GasMass_minus050dex_r_Rhpkpc_entry': 'darkgreen',
        'GasMass_Rhpkpc_entry_r_plus050dexex':  'royalblue',
        'Gasvrad_minus050dex_r_Rhpkpc_entry': 'darkgreen',
        'Gasvrad_Rhpkpc_entry_r_plus050dex': 'royalblue',
        'GasMassInflow_minus050dex_r_Rhpkpc_entry': 'darkgreen',
        'GasMassInflow_Rhpkpc_entry_r_plus050dex': 'royalblue',
        
        'sSFR_minus050dex_r_Rhpkpc_entry': 'darkgreen',
        'sSFR_Rhpkpc_entry_r_plus050dex': 'royalblue',
        'SFR_minus050dex_r_Rhpkpc_entry': 'darkgreen',
        'SFR_Rhpkpc_entry_r_plus050dex': 'royalblue',
        
        'StarMass_Above_Rhpkpc_entry_plus050dex': 'darkblue',
        'Starvrad_Above_Rhpkpc_entry_plus050dex': 'darkblue',
        'GasMass_Above_Rhpkpc_entry_plus050dex': 'darkblue',
        'Gasvrad_Above_Rhpkpc_entry_plus050dex': 'darkblue',
        'GasMassInflow_Above_Rhpkpc_entry_plus050dex': 'darkblue',
        'sSFR_Above_Rhpkpc_entry_plus050dex': 'darkblue',
        'SFR_Above_Rhpkpc_entry_plus050dex': 'darkblue',

        
        'SFGasMass_In_Rhpkpc': 'black',
        'SFGasMass_Above_Rhpkpc': 'black',
        
        'GasMass_in_07Rhpkpc': 'dodgerblue',
        'GasMass_07_r_14Rhpkpc': 'limegreen',
        'GasMass_14_r_21Rhpkpc': 'darkorange',
        'GasMass_Above_21Rhpkpc': 'red',
        
        'GasMass_Inflow_in_07Rhpkpc': 'dodgerblue',
        'GasMass_Inflow_07_r_14Rhpkpc': 'limegreen',
        'GasMass_Inflow_14_r_21Rhpkpc': 'darkorange',
        'GasMass_Inflow_Above_21Rhpkpc': 'red',
        
        'sSFR_In_Rhpkpc': 'darkblue',
        'sSFR_Above_Rhpkpc': 'tab:blue',
        
        'GasMass_Inflow_In_Rhpkpc': 'royalblue',
        'GasMass_Outflow_In_Rhpkpc':   'crimson',
        'GasMass_Inflow_Above_Rhpkpc':  'royalblue',
        'GasMass_Outflow_Above_Rhpkpc':  'crimson',
        
        'nn2_distance': 'blue',
        'nn5_distance': 'green',
        'nn10_distance': 'red',
        
        'nn2_distance_massive': 'blue',
        'nn5_distance_massive': 'green',
        'nn10_distance_massive': 'red',
        
        'N_aper_1_Mpc': 'blue',
        'N_aper_2_Mpc': 'green',
        'N_aper_5_Mpc': 'red',
        
        'N_aper_1_Mpc_massive': 'blue',
        'N_aper_2_Mpc_massive': 'green',
        'N_aper_5_Mpc_massive': 'red',
        
    'SBCDontLoseTheirGasColorbar': 'forestgreen',
    'MBCDontLoseTheirGasColorbar': 'royalblue',        
}

edgecolors = {
    
        #Classes
        'Selected':  'black',
        'SatelliteSelected':  'black',
        'CentralSelectedEmpty':  'black',
        'BadFlag':  'red',
        'Central': 'black',

        #Colorbar
        'NormalColorbar': 'darkorange',
        'SBCColorbar': 'black',
        'MBCColorbar': 'black',
        
        
        'SBC': 'forestgreen',
        'MBC':  'royalblue',
        
        'SBCColorbarEmpty': 'forestgreen',
        'MBCColorbarEmpty':  'royalblue',
        'NormalColorbarEmpty':  'darkorange',
        
        'SubDiffuseColorbarEmpty': 'violet',
        'DiffuseColorbarEmpty': 'firebrick',
        'NormalColorbarEmpty': 'darkorange',
        
        
        'SubDiffuseColorbar': 'violet',
        'DiffuseColorbar': 'firebrick',

        
        'NormalColorbarEdge': 'black',
        'SBCColorbarEdge': 'black',
        'MBCColorbarEdge': 'black',
        
        'DiffuseColorbar': 'firebrick',
        'GAMAColorbar': 'crimson', 
        
        
        
        'SBCLoseTheirGasColorbar': 'forestgreen',
        'MBCLoseTheirGasColorbar': 'royalblue',
        
        'NormalLoseTheirGas': 'darkorange',
        
        'NormalDontLoseTheirGas': 'darkorange',
        
        'NormalLoseTheirGasColorbar': 'darkorange',
        'NormalDontLoseTheirGasColorbar': 'darkorange',

        'SBCLoseTheirGasSatelliteColorbar': 'forestgreen',
        'MBCLoseTheirGasSatelliteColorbar': 'royalblue',
        'SBCDontLoseTheirGasSatelliteColorbar': 'forestgreen',
        'MBCDontLoseTheirGasSatelliteColorbar': 'royalblue',
        'NormalLoseTheirGasSatelliteColorbar': 'darkorange',
        'NormalDontLoseTheirGasSatelliteColorbar': 'darkorange',
        
        'SBCCentralColorbar':   'forestgreen',
        'MBCCentralColorbar':  'royalblue',
        'NormalCentralColorbar':  'darkorange',
        
        'SBCDontLoseTheirGasColorbar': 'forestgreen',
        'MBCDontLoseTheirGasColorbar': 'royalblue',
        
        'SBCLoseTheirGas': 'forestgreen',
        'MBCLoseTheirGas': 'royalblue',
        
        'SBCDontLoseTheirGas': 'forestgreen',
        'MBCDontLoseTheirGas': 'royalblue',
        
        'SBCEmpty': 'forestgreen',
        'MBCEmpty': 'royalblue',
        'NormalEmpty': 'darkorange',
        'SBCLoseTheirGasColorbarLegend':  'forestgreen',
        'MBCLoseTheirGasColorbarLegend':  'royalblue',
        
      
}
    
lines = {
        #Classes
        'Normal': 'solid',
        'SBC': 'solid',
        'MBC': 'solid',
        'Diffuse': 'solid',
        'SubDiffuse': 'solid',
        'NormalSatelliteFalse': 'solid',
        'SBCSatelliteFalse': 'solid',
        'MBCSatelliteTrue': 'solid',
        
        'NormalSatelliteTrue': (0, (10, 8)),
        'SBCSatelliteTrue': (0, (10, 8)),
        'MBCSatelliteFalse': (0, (10, 8)),
        
        
        'NormalSatelliteDMrich': 'solid',
        'SBCSatelliteDMrich': 'solid',
        'MBCSatelliteDMrich': 'solid',
        
        'NormalSatelliteDMpoor': (0, (10,4)),
        'SBCSatelliteDMpoor': (0, (10, 4)),
        'MBCSatelliteDMpoor': (0, (10, 4)),


        'NormalWithoutBH': (0, (10, 8)),
        'SBCWithoutBH': (0, (10, 8)),
        'MBCWithoutBH': (0, (10, 8)),
        'WithoutBH': (0, (10, 8)),
        'WithBH': 'solid',
        'NormalWithBH': 'solid',
        'SBCWithBH': 'solid',
        'MBCWithBH': 'solid',
        
        'SatelliteDMrich':  'solid',
        'SatelliteDMpoor':  (0, (10, 4)),
        'Central':  'solid',
        
        'SatelliteDMpoorEntryToNoGas':  'solid',
        'SatelliteDMpoorNoGasToFinal': (0, (10, 4)),
        
        'SatelliteEMNew':  'solid',
        'SatelliteESNew':  (0, (10, 4)),
        
        'DMrich': 'solid',
        'DMpoor':  (0, (10, 8)),

        #Subhalo
        'SubhaloHalfmassRadType0': (0, (10, 8)),
        'SubhaloHalfmassRadType1':  (0,(0.1,2)),
        'SubhaloHalfmassRadType4':  'solid',
        
        'CompareSubhaoHalfmassRadType1': (0, (0.1, 2)),
        'CompareSubhaoHalfmassRadType4': 'solid',
        
        

        'SubhaloMassInRadType0': (0, (10, 8)),
        'SubhaloMassInRadType1':  (0,(0.1,2)),
        'SubhaloMassInRadType4':  'solid',

        'SubhaloMassType0': (0, (10, 8)),
        'SubhaloMassType1':   (0,(0.1,2)),
        'SubhaloMassType4': 'solid',
        
        'StarMassExSitu_Above_5rpkpc': (0, (10, 8)),
        'StarMassExSitu_24_r_5rpkpc':  (0,(0.1,2)),
        'StarMassExSitu_In_24rpkpc':  'solid',
        
        'StarMassExSitu_Above_RhpkpcDiffuse': (0, (10, 8)),
        'StarMassExSitu_In_RhpkpcDiffuse':  'solid',

        #Component
        'Type0': (0, (10, 8)),
        'Type1': (0,(0.1,2)),
        'Type4': 'solid',
        
        
        'SubhalosSFRInHalfRad': 'solid',
        'sSFR_Outer': (0, (10, 8)),

        #ExSitu Contribution
        'MassExNormalize': 'solid',
        'MassExNormalizeAll': (0, (10, 8)),
        'StellarMassExSituMinor': (0,(0.1,2)),
        'StellarMassExSituIntermediate': (0, (10, 8)),
        'StellarMassExSituMajor': 'solid',
        
        'StellarMassExSitu': (0, (10, 8)),
        'StellarMassInSitu': 'solid',

        'ExMassType0Evolution': (0, (10, 8)),
        'ExMassType1Evolution': (0,(0.1,2)),
        'ExMassType4Evolution': 'solid',
        
        #Others
        'SubhalosSFRwithinHalfandRad': (0, (10, 8)),

        'r_over_R_Crit200': 'solid',
        'r_over_R_Crit200_FirstGroup': (0, (10, 6)),
        
        'Group_M_Crit200FinalGroup': 'solid',
        'Group_M_Crit200': (0, (10, 6)),
        
        'SubhaloStellarMass_in_Rhpkpc': 'solid',
        
        'SubhaloStellarMass_Above_Rhpkpc': (0, (10, 8)),
        
        'StellarMass_in_Rhpkpc': 'solid',
        'StellarMass_Above_Rhpkpc': (0, (10, 8)),
        'GasMass_in_Rhpkpc': 'solid',
        
        'GasMass_in_07Rhpkpc':  'solid',
        'GasMass_07_r_14Rhpkpc':  'solid',
        'GasMass_14_r_21Rhpkpc':  'solid',
        'GasMass_Above_21Rhpkpc':   'solid',
        
        'DMMass_In_Rhpkpc': 'solid',
        'StarMass_In_Rhpkpc': 'solid',
        'GasMass_In_Rhpkpc': 'solid',
        
        
        'GasMass_In_TrueRhpkpc': 'solid',
        'GasMass_Above_TrueRhpkpc': (0, (10, 8)),
        'sSFR_In_TrueRhpkpc': 'solid',
        'sSFR_Above_TrueRhpkpc': (0, (10, 8)),
        
        'SFR_In_Rhpkpc': 'solid',
        'SFR_Above_Rhpkpc': (0, (10, 8)),
        
        'SubhaloSFRinRad': 'solid',
        'SubhaloSFRouterRad': (0, (10, 8)),
        
     
        'GasMassInflow_In_Rhpkpc': 'solid',
        'GasMassInflow_Above_Rhpkpc': (0, (10, 8)),
        
        
        'DMMass_Above_Rhpkpc': (0, (10, 8)),
        'StarMass_Above_Rhpkpc': (0, (10, 8)),
        'GasMass_Above_Rhpkpc': (0, (10, 8)),
        
        'SFGasMass_Above_Rhpkpc': (0, (10, 8)),
        
        'sSFR_In_Rhpkpc': 'solid',
        'sSFR_Above_Rhpkpc': (0, (10, 8)),
        
        'logStar_GFM_Metallicity_In_Rhpkpc': 'solid',
        'logStar_GFM_Metallicity_Above_Rhpkpc': (0, (10, 8)),
        
        
        'sSFR_In_Rhpkpc_entry': 'solid',
        'sSFR_Above_Rhpkpc_entry': (0, (10, 8)),
        
        'GasMass_Inflow_In_Rhpkpc': 'solid',
        'GasMass_Outflow_In_Rhpkpc':   (0, (10, 8)),
        'GasMass_Inflow_Above_Rhpkpc':  'solid',
        'GasMass_Outflow_Above_Rhpkpc':   (0, (10, 8)),
        
        'nn2_distance': (0, (10, 8)),
        'nn5_distance': 'dotted',
        'nn10_distance': 'solid',
        
        'nn2_distance_massive': (0, (10, 8)),
        'nn5_distance_massive':  'dotted',
        'nn10_distance_massive': 'solid',
        
        'N_aper_1_Mpc': (0, (10, 8)),
        'N_aper_2_Mpc':  'dotted',
        'N_aper_5_Mpc': 'solid',
        
        'N_aper_1_Mpc_massive': (0, (10, 8)),
        'N_aper_2_Mpc_massive':  'dotted',
        'N_aper_5_Mpc_massive': 'solid',
}


scales = {
    # Subhalo
    'SigmasSFRRatio': 'log',
    'FracType1': 'log',
    # Group
    'GroupNsubs': 'linear',
    'GroupNsubsFirstGroup':  'log',
    'GroupNsubsFinalGroup':  'log',
    'J200': 'log',
    'FracStarLoss': 'log',
    #Frac
    'GasFrac_99':  'log',
    'StarFrac_99':  'log',
    'DMFrac_99': 'linear',
    'DMFrac_Birth': 'logit',
    
    'Mgas_Norm_Max':   'log',
    'MDM_Norm_Max':   'log',
    'Mstar_Norm_Max':  'log',

    #ExSitu Contribution
    'MassExNormalize':  'log',
    'MassInNormalize': 'log',

    'MassExNormalizeAll':  'log',

    #Profile
    'RadVelocity': 'linear',
    'j': 'log',
    'DensityGas': 'log',
    'SFR': 'log',
    'DensityStar': 'log',

    'joverR': 'log',
    'DensityGasOverR2': 'log',
    'sSFR': 'log',
    'DensityStarOverR2': 'log',

    'rad': 'log',
    
    #Orbit
    'rToRNearYoung': 'log',
    'r_over_R_Crit200': 'log',
    'rOverR200Mean': 'log',
    'rOverR200Mean_New': 'log',

    'rOverR200_99': 'log',
    #Scatter
    'z_Birth': 'log',
    'MDM_Norm_Max_99': 'log',
    'Loss_MDM_Norm_Max_99': 'log',

    'rOverR200Min': 'log',

    'nn2_distance': 'log',
    'nn5_distance': 'log',
    'nn10_distance': 'log',
    
    'nn2_distance_massive': 'log',
    'nn5_distance_massive': 'log',
    'nn10_distance_massive': 'log',
    
    'N_aper_1_Mpc': 'log',
    'N_aper_2_Mpc': 'log',
    'N_aper_5_Mpc': 'log',
    
    'N_aper_1_Mpc_massive': 'log',
    'N_aper_2_Mpc_massive': 'log',
    'N_aper_5_Mpc_massive': 'log',
    
    'deltaInnersSFR_afterEntry':'linear',
    'derStellarExSitu': 'linear',
    'pVEXMassWithRh': 'log',
    'pVInMassWithRh': 'log',
    'DeltaStarMass_Above_Normalize_99': 'log',
    'FracExSitu': 'log',
    None: 'linear'
}

linesthicker = {
        #Classes
        #'Normal': 0.2,
        #'SBC': 0.2,
        #'MBC': 0.2,
        'Diffuse': 1.1,
        'SubDiffuse': 1.1,
        'Central': 1.1,
        'Satellite': 1.1,
        
        'NormalSatelliteDMrich': 1.,
        'SBCSatelliteDMrich': 1.,
        'MBCSatelliteDMrich': 1.,
        
        'NormalSatelliteDMpoor':1.1,
        'SBCSatelliteDMpoor': 1.1,
        'MBCSatelliteDMpoor':  1.1,
        
        
        'SatelliteDMrich':  1.1,
        'SatelliteDMpoor':  1.1,
        
        'SatelliteEMNew': 1.1, 
        'SatelliteESNew':  1.1,
    
        'SatelliteDMpoorEntryToNoGas':  1.1,
        'SatelliteDMpoorNoGasToFinal':  1.1,
    


        #Subhalo
        'SubhaloHalfmassRadType0': 1.1,
        'SubhaloHalfmassRadType1': 1.5,
        'SubhaloHalfmassRadType4': 1.1,
        
        
        'CompareSubhaoHalfmassRadType1': 1.5,
        'CompareSubhaoHalfmassRadType4': 1.1,
        
        'Mgas_Norm_Max': 1.1,
        'MDM_Norm_Max': 1.1,
        'Mstar_Norm_Max': 1.1,

        'SubhaloMassInRadType0': 1.1,
        'SubhaloMassInRadType1': 1.5,
        'SubhaloMassInRadType4': 1.1,

        'SubhaloMassType0': 1.1,
        'SubhaloMassType1': 1.5,
        'SubhaloMassType4': 1.1,

        #Component
        'Type0': 1.1,
        'Type1': 1.5,
        'Type4': 1.1,

        #ExSitu Contribution
        'MassExNormalize': 1.1,
        #'MassExNormalizeAll': 1.1,
        'StellarMassExSituMinor': 1.5,
        'StellarMassExSituIntermediate': 1.1,
        'StellarMassExSituMajor': 1.1,
        
        'StellarMassExSitu': 1.1,
        'StellarMassInSitu': 1.1,

        'ExMassType0Evolution': 1.1,
        'ExMassType1Evolution': 1.5,
        'ExMassType4Evolution': 1.1,
        
        'nn2_distance': 1.1,
        'nn5_distance': 1.9,
        'nn10_distance': 1.1,
        
        'nn2_distance_massive': 1.1,
        'nn5_distance_massive': 1.9,
        'nn10_distance_massive': 1.1,
        
        'N_aper_1_Mpc':  1.1,
        'N_aper_2_Mpc':  1.9,
        'N_aper_5_Mpc':  1.1,
        
        'N_aper_1_Mpc_massive':  1.1,
        'N_aper_2_Mpc_massive':  1.9,
        'N_aper_5_Mpc_massive':  1.1,

}

markers = {
    #Classes
    'Normal': 'o',
    'SBC': 'o',
    'MBC': 'o',
    
    'SBCEmpty': 'o',
    'MBCEmpty': 'o',
    'NormalEmpty': 'o',
    
    'SubDiffuse': 'o',
    'Diffuse': 'o',
    'SBCBornYoung': '^',

    'TNGrage': 'o',
    'Selected': 'o',

    'SatelliteSelected':  'o',
    'CentralSelectedEmpty':  'o',
    'BadFlag': '^',

    'SBCGamaColor': 'D', 
    'MBCGamaColor': 'o', 
    'NormalGamaColor': '^', 
    'GAMAColor': '*',
    
    #Colorbar
    'NormalColorbar': '^',
    'SBCColorbar': 'D',
    'MBCColorbar': 'o',
    
  
    'NormalColorbarEmpty':'^',

    'SBCColorbarEmpty': 'D',
    'MBCColorbarEmpty':  'o',
    
    'SubDiffuseColorbarEmpty': 'H',
    'DiffuseColorbarEmpty': 'D',

    
    'NormalColorbarEdge': '^',
    'SBCColorbarEdge': 'D',
    'MBCColorbarEdge': 'o',
    
    'DiffuseColorbar': 'D',
    'SubDiffuseColorbar': 'H',
    'GAMAColorbar': '*', 
    
    'SBCLoseTheirGasColorbarLegend': 'o',
    'MBCLoseTheirGasColorbarLegend': 'o',
    
    


    'SBCLoseTheirGasColorbar': 's',
    'MBCLoseTheirGasColorbar': 's',
    
    'SBCDontLoseTheirGasColorbar': '*',
    'MBCDontLoseTheirGasColorbar': '*',
    
    'DontLoseTheirGasColorbar': '*',
    'LoseTheirGasColorbar': 's',
    
    'DontLoseTheirGasSatelliteColorbar': '*',
    'LoseTheirGasSatelliteColorbar': 's',
    'CentralColorbar': 'o',

    
    'NormalLoseTheirGas': 's',
    'SBCLoseTheirGas': 's',
    'MBCLoseTheirGas': 's',
    
    'SBCDontLoseTheirGas': '*',
    'MBCDontLoseTheirGas': '*',
    
    'NormalDontLoseTheirGas': '*',
    'NormalDontLoseTheirGasColorbar': '*',
    'NormalLoseTheirGasColorbar':  's',
    
    'SBCLoseTheirGasSatelliteColorbar': 's',
    'MBCLoseTheirGasSatelliteColorbar': 's',
    
    'SBCDontLoseTheirGasSatelliteColorbar': '*',
    'MBCDontLoseTheirGasSatelliteColorbar': '*',
    'NormalDontLoseTheirGasSatelliteColorbar': '*',
    'NormalLoseTheirGasSatelliteColorbar':  's',
    
    'SBCCentralColorbar':   'o',
    'MBCCentralColorbar':  'o',
    'NormalCentralColorbar':  'o',
    'Central': 'o',
    

} 

msize = {
    #Classes
    'Normal': 3.3,
    'SBC': 8,
    'MBC': 8,
    
    'SatelliteDMrich': 8,
    'SatelliteDMpoor': 8,
    
    'SatelliteEMNew':  8,
    'SatelliteESNew':  8,
    
    'SBCEmpty': 8,
    'MBCEmpty': 8,
    'NormalEmpty': 8,
    
    'Diffuse': 8,
    'SubDiffuse': 8,
    'Central': 8,
    'Satellite': 8,
    'SBCBornYoung': 11,


    'TNGrage':  3,
    'Selected':  8,
    'BadFlag':  11,

    'SatelliteSelected':  8,
    'CentralSelectedEmpty':  8,

    'SBCGamaColor': 8, 
    'MBCGamaColor': 8, 
    'NormalGamaColor': 8, 
    'GAMAColor': 9.5, 


    #Colorbar
    'NormalColorbar': 6,
    'SBCColorbar': 8, 
    'MBCColorbar': 8, 
    
    'NormalColorbarEmpty': 6,
    'SBCColorbarEmpty': 8,
    'MBCColorbarEmpty':  8,
    
    'SubDiffuseColorbarEmpty': 8,
    'DiffuseColorbarEmpty': 8,

    
    'NormalColorbarEdge': 6,
    'SBCColorbarEdge': 8, 
    'MBCColorbarEdge': 8, 
    
    
    'DiffuseColorbar':  8, 
    'SubDiffuseColorbar': 8,
    'GAMAColorbar': 9.5, 
    
    'SBCLoseTheirGasColorbarLegend': 8,
    'MBCLoseTheirGasColorbarLegend': 8,
    
    
    'SBCDontLoseTheirGasColorbar': 36,
    'MBCDontLoseTheirGasColorbar': 36,
    
    'SBCLoseTheirGasColorbar': 12,
    'MBCLoseTheirGasColorbar': 12,
    
    'NormalLoseTheirGas': 12,
    'SBCLoseTheirGas': 12,
    'MBCLoseTheirGas': 12,
    

    
    'NormalDontLoseTheirGas': 36,
    'SBCDontLoseTheirGas': 36,
    'MBCDontLoseTheirGas': 36,
    
    'NormalDontLoseTheirGasColorbar': 24,
    'NormalLoseTheirGasColorbar': 6,
    
    'NormalDontLoseTheirGasSatelliteColorbar': 24,
    'NormalLoseTheirGasSatelliteColorbar': 6,
    
    'SBCDontLoseTheirGasSatelliteColorbar': 36,
    'MBCDontLoseTheirGasSatelliteColorbar': 36,
    
    'SBCLoseTheirGasSatelliteColorbar': 12,
    'MBCLoseTheirGasSatelliteColorbar': 12,
    
    'DontLoseTheirGasColorbar': 12,
    'LoseTheirGasColorbar': 8,
    
    'DontLoseTheirGasSatelliteColorbar': 12,
    'LoseTheirGasSatelliteColorbar': 8,
    'CentralColorbar': 8,

    
    
    'SBCCentralColorbar':   8,
    'MBCCentralColorbar':  8,
    'NormalCentralColorbar':  6,


}

capstyles = {}


titles = {
    #Classes
    'Normal': r'Normals',
    'SBC': r'Compacts$_\mathrm{SB}$',
    'SBCBornYoung': 'Young \n Compacts$_\mathrm{SB} $',
    'MBC': r'Compacts$_\mathrm{MB}$',
    
    'SBCEmpty':  r'Compacts$_\mathrm{SB}$',
    'MBCEmpty': r'Compacts$_\mathrm{MB}$',
    'NormalEmpty': r'Normals',

    'Diffuse': r'Diffuses',
    'SubDiffuse': r'Sub-Diffuses',
    'TNGrage':  'All \n galaxies',
    'Selected':  'Selected',
    'SatelliteSelected':  'Satellite',
    'CentralSelectedEmpty':  'Central',
    'BadFlag':  'Bad flags',
    'GMM': 'GMM',

    'SBCGamaColor': r'$\mathrm{Compacts_{SB}}$',
    'MBCGamaColor': r'$\mathrm{Compacts_{MB}}$',
    'NormalGamaColor': 'Normals', 
    'GAMAColor': 'GAMA',

    'DMrich': r'$f_\mathrm{DM} > 0.7$',
    'DMpoor': r'$f_\mathrm{DM} < 0.7$',
    
    'DMFracHigher': r'$f_\mathrm{DM} > 0.93$',
    'DMFracLower': r'$f_\mathrm{DM} \leq 0.93$',
    
    'DMFracHigher': r'$f_\mathrm{DM} > 0.93$',
    'DMFracLower': r'$f_\mathrm{DM} \leq 0.93$',
    
    'SatelliteNotInteract': r'Sattelites$_\mathrm{EM}$',
    'SatelliteInteract': r'Sattelites$_\mathrm{ES}$',

    'WithBH': 'With BH',
    'WithoutBH': 'Without BH',
    
    'SatelliteDMrich': r'Satellites $f_\mathrm{DM} > 0.7$',
    'SatelliteDMpoor': r'Satellites $f_\mathrm{DM} < 0.7$',
    'Central': 'Centrals',
    
    'SatelliteDMpoorEntryToNoGas':  'With gas',
    'SatelliteDMpoorNoGasToFinal':  'After gas loss',


    #Colorbar
    'NormalColorbar': r'Normals',
    'SBCColorbar':  r'Compacts$_\mathrm{SB}$',
    'MBCColorbar':  r'Compacts$_\mathrm{MB}$',
    
    'SBCColorbarEmpty':  r'Compacts$_\mathrm{SB}$',
    'MBCColorbarEmpty':   r'Compacts$_\mathrm{MB}$',
    
    'SubDiffuseColorbarEmpty': r'Diffuses',
    'DiffuseColorbarEmpty': r'Sub-Diffuses',
    'NormalColorbarEmpty': r'Normals',
    
    'NormalColorbarEdge':  r'Normals',
    'SBCColorbarEdge': r'Compacts$_\mathrm{SB}$',
    'MBCColorbarEdge': r'Compacts$_\mathrm{MB}$',
    
    'DiffuseColorbar': r'Diffuses',
    'SubDiffuseColorbar': r'Sub-Diffuses',
    'GAMAColorbar': 'GAMA',
    
    
    'SBCLoseTheirGasColorbarLegend': r'Compacts$_\mathrm{SB}$',
    'MBCLoseTheirGasColorbarLegend': r'Compacts$_\mathrm{MB}$',
    
    'DontLoseTheirGasColorbar': 'Retain their gas',
    'LoseTheirGasColorbar': 'Lose their gas',
    
    'DontLoseTheirGasSatelliteColorbar': 'Satellites:  \n Retain their gas',
    'LoseTheirGasSatelliteColorbar': 'Lose their gas',
    'CentralColorbar': 'Central',


    #Component
    'Type0': 'Gas',
    'Type1': 'DM',
    'Type4': 'Stars',
    'SubhaloMassType0': 'Gas',
    'SubhaloMassType1': 'DM',
    'SubhaloMassType4': 'Stars',

    #Subhalos
    'SubhaloHalfmassRadType0': 'Gas',
    'SubhaloHalfmassRadType1': 'DM',
    'SubhaloHalfmassRadType4': 'Stars',
    
    
    'CompareSubhaoHalfmassRadType1': 'DM',
    'CompareSubhaoHalfmassRadType4': 'Stars',
    
    'Mgas_Norm_Max': 'Gas',
    'MDM_Norm_Max': 'DM',
    'Mstar_Norm_Max': 'Stars',
    
    'SubhalosSFRInHalfRad': r'$r < r_{1/2}$',
    'SubhalosSFRwithinHalfandRad': r'$r_{1/2} < r < 2r_{1/2}$',
    'sSFR_Outer': r'$r > r_{1/2}$',

    'r_over_R_Crit200': r'Final host',
    'r_over_R_Crit200_FirstGroup': r'Running host',

    #ExSitu Contribution
    'MassExNormalize': 'by \n $M_{\mathrm{ex-situ}, \; z = 0}$',
    'MassExNormalizeAll': r'by $M_{\star, \; z = 0}$',
    'StellarMassExSituMinor': r'Minor mergers',
    'StellarMassExSituIntermediate': r'Intermediate mergers',
    'StellarMassExSituMajor': r'Major mergers',
    
    'StellarMassExSitu': 'Ex-Situ',
    'StellarMassInSitu': 'In-Situ',
    
    'SubhaloStellarMass_in_Rhpkpc': r'$r < r_{1/2,\; z=0}$',
    
    'SubhaloStellarMass_Above_Rhpkpc': r'$r > r_{1/2,\; z=0}$',

    'DMMass_In_Rhpkpc': r'$r \leq 2 r_{1/2,\; z=0}$',
    'DMMass_Above_Rhpkpc': r'$r > 2 r_{1/2,\; z=0}$',
    'StarMass_In_Rhpkpc':  r'$r \leq r_{1/2,\; z=0}$',
    'StarMass_Above_Rhpkpc': r'$r > r_{1/2,\; z=0}$',
    'GasMass_In_Rhpkpc':  r'$r \leq  r_{1/2,\; z=0}$',
    'GasMass_Above_Rhpkpc': r'$r > r_{1/2,\; z=0}$',
    
    'GasMass_In_TrueRhpkpc':   r'$r \leq  r_{1/2,\; z=0}$',
    'GasMass_Above_TrueRhpkpc':   r'$r >  r_{1/2,\; z=0}$',
    
    'GasMassInflow_In_Rhpkpc':  r'$r \leq 2 r_{1/2,\; z=0}$',
    'GasMassInflow_Above_Rhpkpc': r'$r > 2 r_{1/2,\; z=0}$',
    
    'SFGasMass_In_Rhpkpc':  r'$r \leq 2 r_{1/2,\; z=0}$',
    'SFGasMass_Above_Rhpkpc': r'$r >  2 r_{1/2,\; z=0}$',
    
    'sSFR_In_Rhpkpc': r'$r <  2 r_{1/2,\; z=0}$',
    'sSFR_Above_Rhpkpc': r'$r > 2 r_{1/2,\; z=0}$',
    
   
    'sSFR_In_TrueRhpkpc': r'$r <  r_{1/2,\; z=0}$',
    'sSFR_Above_TrueRhpkpc': r'$r > r_{1/2,\; z=0}$',
    
    'SubhaloSFRinRad': r'$r < 2 r_{1/2}$',
    'SubhaloSFRouterRad': r'$r > 2 r_{1/2}$',
    
    'StarMass_In_MultRhpkpc_plus000dex':  r'$\log r < \log r_{1/2,\; z=0} $',
    'StarMass_In_MultRhpkpc_plus015dex': r'$\log r < \log r_{1/2,\; z=0} + 0.15 $',
    'StarMass_In_MultRhpkpc_plus025dex': r'$\log r < \log r_{1/2,\; z=0} + 0.25 $',
    'StarMass_In_MultRhpkpc_plus050dex': r'$\log r < \log r_{1/2,\; z=0} + 0.5 $',
    'StarMass_In_MultRhpkpc_plus075dex': r'$\log r < \log r_{1/2,\; z=0} + 0.75 $',

    'Starvrad_In_MultRhpkpc_plus000dex':  r'$\log r < \log r_{1/2,\; z=0} $',
    'Starvrad_In_MultRhpkpc_plus015dex': r'$\log r < \log r_{1/2,\; z=0} + 0.15 $',
    'Starvrad_In_MultRhpkpc_plus025dex': r'$\log r < \log r_{1/2,\; z=0} + 0.25 $',
    'Starvrad_In_MultRhpkpc_plus050dex': r'$\log r < \log r_{1/2,\; z=0} + 0.5 $',
    'Starvrad_In_MultRhpkpc_plus075dex': r'$\log r < \log r_{1/2,\; z=0} + 0.75 $',

    'GasMass_In_MultRhpkpc_plus000dex':  r'$\log r < \log r_{1/2,\; z=0}  $',
    'GasMass_In_MultRhpkpc_plus015dex': r'$\log r < \log r_{1/2,\; z=0} + 0.15 $',
    'GasMass_In_MultRhpkpc_plus025dex': r'$\log r < \log r_{1/2,\; z=0} + 0.25 $',
    'GasMass_In_MultRhpkpc_plus050dex': r'$\log r < \log r_{1/2,\; z=0} + 0.5 $',
    'GasMass_In_MultRhpkpc_plus075dex': r'$\log r < \log r_{1/2,\; z=0} + 0.75 $',

    'Gasvrad_In_MultRhpkpc_plus000dex':  r'$\log r < \log r_{1/2,\; z=0}  $',
    'Gasvrad_In_MultRhpkpc_plus015dex': r'$\log r < \log r_{1/2,\; z=0} + 0.15 $',
    'Gasvrad_In_MultRhpkpc_plus025dex': r'$\log r < \log r_{1/2,\; z=0} + 0.25 $',
    'Gasvrad_In_MultRhpkpc_plus050dex': r'$\log r < \log r_{1/2,\; z=0} + 0.5 $',
    'Gasvrad_In_MultRhpkpc_plus075dex': r'$\log r < \log r_{1/2,\; z=0} + 0.75 $',

    'GasMassInflow_In_MultRhpkpc_plus000dex':  r'$\log r < \log r_{1/2,\; z=0}  $',
    'GasMassInflow_In_MultRhpkpc_plus015dex': r'$\log r < \log r_{1/2,\; z=0} + 0.15 $',
    'GasMassInflow_In_MultRhpkpc_plus025dex': r'$\log r < \log r_{1/2,\; z=0} + 0.25 $',
    'GasMassInflow_In_MultRhpkpc_plus050dex': r'$\log r < \log r_{1/2,\; z=0} + 0.5 $',
    'GasMassInflow_In_MultRhpkpc_plus075dex': r'$\log r < \log r_{1/2,\; z=0} + 0.75 $',

    'sSFR_In_MultRhpkpc_plus000dex':  r'$\log r < \log r_{1/2,\; z=0}  $',
    'sSFR_In_MultRhpkpc_plus015dex': r'$\log r < \log r_{1/2,\; z=0} + 0.15 $',
    'sSFR_In_MultRhpkpc_plus025dex': r'$\log r < \log r_{1/2,\; z=0} + 0.25 $',
    'sSFR_In_MultRhpkpc_plus050dex': r'$\log r < \log r_{1/2,\; z=0} + 0.5 $',
    'sSFR_In_MultRhpkpc_plus075dex': r'$\log r < \log r_{1/2,\; z=0} + 0.75 $',

    'SFR_In_MultRhpkpc_plus000dex':  r'$\log r < \log r_{1/2,\; z=0}  $',
    'SFR_In_MultRhpkpc_plus015dex': r'$\log r < \log r_{1/2,\; z=0} + 0.15 $',
    'SFR_In_MultRhpkpc_plus025dex': r'$\log r < \log r_{1/2,\; z=0} + 0.25 $',
    'SFR_In_MultRhpkpc_plus050dex': r'$\log r < \log r_{1/2,\; z=0} + 0.5 $',
    'SFR_In_MultRhpkpc_plus075dex': r'$\log r < \log r_{1/2,\; z=0} + 0.75 $',
    
    'StarMass_In_Rhpkpc_entry_minus200dex':  r'$\log r < \log r_{1/2,\; \mathrm{entry}} - 2 $',
    'StarMass_In_Rhpkpc_entry_minus100dex':  r'$\log r < \log r_{1/2,\; \mathrm{entry}} - 1 $',
    'StarMass_In_Rhpkpc_entry_minus150dex':  r'$\log r < \log r_{1/2,\; \mathrm{entry}} - 1.5 $',
    'StarMass_In_Rhpkpc_entry_plus100dex':  r'$\log r < \log r_{1/2,\; \mathrm{entry}} + 1.0 $',

    'StarMass_In_Rhpkpc_entry_minus050dex':  r'$\log r < \log r_{1/2,\; \mathrm{entry}} - 0.5 $',
    'StarMass_In_Rhpkpc_entry_minus025dex': r'$\log r < \log r_{1/2,\; \mathrm{entry}} - 0.25 $',
    'StarMass_In_Rhpkpc_entry_plus025dex': r'$\log r < \log r_{1/2,\; \mathrm{entry}} + 0.25 $',
    'StarMass_In_Rhpkpc_entry_plus050dex': r'$\log r < \log r_{1/2,\; \mathrm{entry}} + 0.5 $',
    
    'Starvrad_In_Rhpkpc_entry_minus050dex':  r'$\log r < \log r_{1/2,\; \mathrm{entry}} - 0.5 $',
    'Starvrad_In_Rhpkpc_entry_minus025dex': r'$\log r < \log r_{1/2,\; \mathrm{entry}} - 0.25 $',
    'Starvrad_In_Rhpkpc_entry_plus025dex': r'$\log r < \log r_{1/2,\; \mathrm{entry}} + 0.25 $',
    'Starvrad_In_Rhpkpc_entry_plus050dex': r'$\log r < \log r_{1/2,\; \mathrm{entry}} + 0.5 $',
    
    'GasMass_In_Rhpkpc_entry_minus050dex':  r'$\log r < \log r_{1/2,\; \mathrm{entry}} - 0.5 $',
    'GasMass_In_Rhpkpc_entry_minus025dex': r'$\log r < \log r_{1/2,\; \mathrm{entry}} - 0.25 $',
    'GasMass_In_Rhpkpc_entry_plus025dex': r'$\log r < \log r_{1/2,\; \mathrm{entry}} + 0.25 $',
    'GasMass_In_Rhpkpc_entry_plus050dex': r'$\log r < \log r_{1/2,\; \mathrm{entry}} + 0.5 $',
    
    'Gasvrad_In_Rhpkpc_entry_minus050dex':  r'$\log r < \log r_{1/2,\; \mathrm{entry}} - 0.5 $',
    'Gasvrad_In_Rhpkpc_entry_minus025dex': r'$\log r < \log r_{1/2,\; \mathrm{entry}} - 0.25 $',
    'Gasvrad_In_Rhpkpc_entry_plus025dex': r'$\log r < \log r_{1/2,\; \mathrm{entry}} + 0.25 $',
    'Gasvrad_In_Rhpkpc_entry_plus050dex': r'$\log r < \log r_{1/2,\; \mathrm{entry}} + 0.5 $',
    
    'GasMassInflow_In_Rhpkpc_entry_minus050dex':  r'$\log r < \log r_{1/2,\; \mathrm{entry}} - 0.5 $',
    'GasMassInflow_In_Rhpkpc_entry_minus025dex': r'$\log r < \log r_{1/2,\; \mathrm{entry}} - 0.25 $',
    'GasMassInflow_In_Rhpkpc_entry_plus025dex': r'$\log r < \log r_{1/2,\; \mathrm{entry}} + 0.25 $',
    'GasMassInflow_In_Rhpkpc_entry_plus050dex': r'$\log r < \log r_{1/2,\; \mathrm{entry}} + 0.5 $',
    
    'sSFR_In_Rhpkpc_entry_minus050dex':  r'$\log r < \log r_{1/2,\; \mathrm{entry}} - 0.5 $',
    'sSFR_In_Rhpkpc_entry_minus025dex': r'$\log r < \log r_{1/2,\; \mathrm{entry}} - 0.25 $',
    'sSFR_In_Rhpkpc_entry_plus025dex': r'$\log r < \log r_{1/2,\; \mathrm{entry}} + 0.25 $',
    'sSFR_In_Rhpkpc_entry_plus050dex': r'$\log r < \log r_{1/2,\; \mathrm{entry}} + 0.5 $',
    
    'SFR_In_Rhpkpc_entry_minus050dex':  r'$\log r < \log r_{1/2,\; \mathrm{entry}} - 0.5 $',
    'SFR_In_Rhpkpc_entry_minus025dex': r'$\log r < \log r_{1/2,\; \mathrm{entry}} - 0.25 $',
    'SFR_In_Rhpkpc_entry_plus025dex': r'$\log r < \log r_{1/2,\; \mathrm{entry}} + 0.25 $',
    'SFR_In_Rhpkpc_entry_plus050dex': r'$\log r < \log r_{1/2,\; \mathrm{entry}} + 0.5 $',
    
    'StarMass_Above_Rhpkpc_entry_plus050dex': r'$\log r > \log r_{1/2,\; \mathrm{entry}} + 0.5 $',
    'Starvrad_Above_Rhpkpc_entry_plus050dex': r'$\log r > \log r_{1/2,\; \mathrm{entry}} + 0.5 $',
    'GasMass_Above_Rhpkpc_entry_plus050dex': r'$\log r > \log r_{1/2,\; \mathrm{entry}} + 0.5 $',
    'Gasvrad_Above_Rhpkpc_entry_plus050dex': r'$\log r > \log r_{1/2,\; \mathrm{entry}} + 0.5 $',
    'GasMassInflow_Above_Rhpkpc_entry_plus050dex': r'$\log r > \log r_{1/2,\; \mathrm{entry}} + 0.5 $',
    'sSFR_Above_Rhpkpc_entry_plus050dex': r'$\log r > \log r_{1/2,\; \mathrm{entry}} + 0.5 $',
    'SFR_Above_Rhpkpc_entry_plus050dex': r'$\log r > \log r_{1/2,\; \mathrm{entry}} + 0.5 $',
    
    'StarMass_minus050dex_r_Rhpkpc_entry': r'$\log r_{1/2,\; \mathrm{entry}} - 0.5 < \log r < \log r_{1/2,\; \mathrm{entry}} $',
    'StarMass_Rhpkpc_entry_r_plus050dexex': r'$\log r_{1/2,\; \mathrm{entry}}  < \log r < \log r_{1/2,\; \mathrm{entry}} + 0.5 $',
    'Starvrad_minus050dex_r_Rhpkpc_entry':r'$\log r_{1/2,\; \mathrm{entry}} - 0.5 < \log r < \log r_{1/2,\; \mathrm{entry}} $',
    'Starvrad_Rhpkpc_entry_r_plus050dexex': r'$\log r_{1/2,\; \mathrm{entry}}  < \log r < \log r_{1/2,\; \mathrm{entry}} + 0.5 $',
    
    'GasMass_minus050dex_r_Rhpkpc_entry':  r'$\log r_{1/2,\; \mathrm{entry}} - 0.5 < \log r < \log r_{1/2,\; \mathrm{entry}} $',
    'GasMass_Rhpkpc_entry_r_plus050dexex':  r'$\log r_{1/2,\; \mathrm{entry}}  < \log r < \log r_{1/2,\; \mathrm{entry}} + 0.5 $',
    'Gasvrad_minus050dex_r_Rhpkpc_entry': r'$\log r_{1/2,\; \mathrm{entry}} - 0.5 < \log r < \log r_{1/2,\; \mathrm{entry}} $',
    'Gasvrad_Rhpkpc_entry_r_plus050dex': r'$\log r_{1/2,\; \mathrm{entry}}  < \log r < \log r_{1/2,\; \mathrm{entry}} + 0.5 $',
    'GasMassInflow_minus050dex_r_Rhpkpc_entry': r'$\log r_{1/2,\; \mathrm{entry}} - 0.5 < \log r < \log r_{1/2,\; \mathrm{entry}} $',
    'GasMassInflow_Rhpkpc_entry_r_plus050dex': r'$\log r_{1/2,\; \mathrm{entry}}  < \log r < \log r_{1/2,\; \mathrm{entry}} + 0.5 $',
    
    'sSFR_minus050dex_r_Rhpkpc_entry': r'$\log r_{1/2,\; \mathrm{entry}} - 0.5 < \log r < \log r_{1/2,\; \mathrm{entry}} $',
    'sSFR_Rhpkpc_entry_r_plus050dex': r'$\log r_{1/2,\; \mathrm{entry}}  < \log r < \log r_{1/2,\; \mathrm{entry}} + 0.5 $',
    'SFR_minus050dex_r_Rhpkpc_entry': r'$\log r_{1/2,\; \mathrm{entry}} - 0.5 < \log r < \log r_{1/2,\; \mathrm{entry}} $',
    'SFR_Rhpkpc_entry_r_plus050dex': r'$\log r_{1/2,\; \mathrm{entry}}  < \log r < \log r_{1/2,\; \mathrm{entry}} + 0.5 $',
    
    'GasMass_in_07Rhpkpc':  r'$r/\mathrm{kpc} < 0.7 $',
    'GasMass_07_r_14Rhpkpc':  r'$0.7 < r/\mathrm{kpc} < 1.4 $',
    'GasMass_14_r_21Rhpkpc':   r'$1.4 < r/\mathrm{kpc} < 2.1 $',
    'GasMass_Above_21Rhpkpc':   r'$r/\mathrm{kpc} > 2.1 $',
    
    'GasMass_Inflow_in_07Rhpkpc':  r'$r/\mathrm{kpc} < 0.7 $',
    'GasMass_Inflow_07_r_14Rhpkpc': r'$0.7 < r/\mathrm{kpc} < 1.4 $',
    'GasMass_Inflow_14_r_21Rhpkpc':   r'$1.4 < r/\mathrm{kpc} < 2.1 $',
    'GasMass_Inflow_Above_21Rhpkpc':  r'$r/\mathrm{kpc} > 2.1 $',
    
    
    'GasMass_Inflow_In_Rhpkpc':  r'$v_\mathrm{rad} < -5 \mathrm{km/s}$',
    'GasMass_Outflow_In_Rhpkpc':  r'$v_\mathrm{rad} > 5 \mathrm{km/s} $',
    'GasMass_Inflow_Above_Rhpkpc':  r'$v_\matlimegreenhrm{rad} < -5 \mathrm{km/s}$',
    'GasMass_Outflow_Above_Rhpkpc':  r'$v_\mathrm{rad} > 5 \mathrm{km/s}$',
    
    'StarMass_In_Rhpkpc_entry':  r'$\log r < \log r_{1/2,\; \mathrm{entry}}$',
    'Starvrad_In_Rhpkpc_entry':  r'$\log r < \log r_{1/2,\; \mathrm{entry}}$',
    'GasMass_In_Rhpkpc_entry':   r'$\log r < \log r_{1/2,\; \mathrm{entry}}$',
    'Gasvrad_In_Rhpkpc_entry':  r'$\log r < \log r_{1/2,\; \mathrm{entry}}$',
    'GasMassInflow_In_Rhpkpc_entry':  r'$\log r < \log r_{1/2,\; \mathrm{entry}}$',
    'sSFR_In_Rhpkpc_entry': r'$r < r_{1/2,\; \mathrm{entry}}$',
    'sSFR_Above_Rhpkpc_entry': r'$ r < r_{1/2,\; \mathrm{entry}}$',
    'SubhaloMassInRadType4': r'Total',
    'nn2_distance': '$n = 2$',
    'nn5_distance': '$n = 5$',
    'nn10_distance': '$n = 10$',
    
    'nn2_distance_massive': '$n = 2$',
    'nn5_distance_massive': '$n = 5$',
    'nn10_distance_massive': '$n = 10$',
    
    'N_aper_1_Mpc':  '$ R = 1 [\mathrm{Mpc}]$',
    'N_aper_2_Mpc': '$ R = 2 [\mathrm{Mpc}]$',
    'N_aper_5_Mpc':  '$ R = 5 [\mathrm{Mpc}]$',
    
    'N_aper_1_Mpc_massive':  '$ R = 1 [\mathrm{Mpc}]$',
    'N_aper_2_Mpc_massive':  '$ R = 2 [\mathrm{Mpc}]$',
    'N_aper_5_Mpc_massive':  '$ R = 5 [\mathrm{Mpc}]$',

}

labels = {
    #Subhalo
    'Relative_logZ_At_Entry': r'$[\log Z_\star -\left<\log Z_\star\right>]^{\mathrm{entry}}$ ',
    'Relative_logInnerZ_At_Entry': r'$[\log Z_\star -\left<\log Z_\star\right>]^{\mathrm{entry}}_\mathrm{inner}$ ',
    'logMstar_Entry': r'$\log(M_\star/\mathrm{M}_\odot)_\mathrm{at-entry}$',
    'AgeBorn': r'SubFind Age [Gyr]',
    'StarAgeAll': r'Stellar Age [Gyr]',
    
    'RadIn': r'$\overline{r}_\mathrm{in-situ} [kpc]$',
    'RadEx': r'$\overline{r}_\mathrm{ex-situ}  [kpc]$',
    'SigmaIn': r'$\sigma_\mathrm{v, in-situ}  [\mathrm{km s}^{-1}]$',

    
    'logSUM_Mstar_merger_Corotate' : r'$\log(M_{\star,\,\mathrm{mergers}} / \mathrm{M}_\odot)$',
    'logSUM_Mstar_merger_Perpendicular' : r'$\log(M_{\star,\,\mathrm{mergers}} / \mathrm{M}_\odot)$',
    'logSUM_Mstar_merger_Counterotating' : r'$\log(M_{\star,\,\mathrm{mergers}} / \mathrm{M}_\odot)$',

    
    'FracStarLoss': r'$(M_{\mathrm{loss}}/ M_{z = 0})_\star$',
    'FracStarAfterEntry_Inner': r'$(M_\mathrm{inner}^\mathrm{after \, entry}/ M_{z = 0})_\star$',
    'FracStarAfterEntry': r'$(M^\mathrm{after \, entry}/ M_{z = 0})_\star$',

    'FracNew_Loss': r'$M_{\star}^{\mathrm{entry-to-gas-loss}} / M_{\star,\,\mathrm{loss}}$',

    'SnapBorn': r'$t_\mathrm{birth}\, [$Gyr$]$',
    'logMassDMAtBirth': r'$\log(M_\mathrm{DM}/\mathrm{M}_\odot)_\mathrm{at-birth}$',
    'logMassStarAtBirth': r'$\log(M_\star/\mathrm{M}_\odot)_\mathrm{at-birth}$',

    'logSizeAtBirth': r'$\log(r_{1/2}/\mathrm{kpc})_\mathrm{at-birth}$',
    'FracType1': r'$M_{\mathrm{DM}} / M_{\mathrm{tot}} $',
    'FracExSitu': r'$M_{\star,\; \mathrm{ex-situ}} / M_\star$',
    'SubhaloStarMetallicity': r'$\log( Z_\star / Z_\odot)$',
    'logStarZ_99': r'$\log( Z_\star / Z_\odot)_{z = 0}$',

    'logHalfRadstar_99': r'$\log(r_{1/2, \, z = 0}/\mathrm{kpc})$',
    'logMstarRad_99': r'$\log(M_{\star}/\mathrm{M}_\odot)_{z = 0}$',
    'SubhaloHalfmassRadType0': r'$\log(r_{1/2, \mathrm{gas}}/\mathrm{kpc})$',
    'SubhaloHalfmassRadType1': r'$\log(r_{1/2, \mathrm{DM}}/\mathrm{kpc})$',
    'SubhaloHalfmassRadType4': r'$\log(r_{1/2, \star}/\mathrm{kpc})$',

    'CompareSubhaoHalfmassRadType1': r'$r_{1/2}/r_{\mathrm{entry}}$',
    'CompareSubhaoHalfmassRadType4': r'$r_{1/2}/r_{\mathrm{entry}}$',


    'Star_GFM_Metallicity_In_Rhpkpc': r'Stellar Metallicity',

    'SubhaloMassInRadType0': r'$\log(M_{\mathrm{gas}, r < 2 r_{1/2}}/\mathrm{M}_\odot)$',
    'SubhaloMassInRadType1': r'$\log(M_{\mathrm{DM}, r < 2 r_{1/2}}/\mathrm{M}_\odot)$',
    'SubhaloMassInRadType4': r'$\log(M_{\star, r < 2 r_{1/2}}/\mathrm{M}_\odot)$',

    'SubhaloMassType0': r'$\log(M_{\mathrm{gas}}/\mathrm{M}_\odot)$',
    'SubhaloMassType1': r'$\log(M_{\mathrm{DM}}/\mathrm{M}_\odot)$',
    'SubhaloMassType4': r'$\log(M_{\star}/\mathrm{M}_\odot)$',

    'SubhalosSFRInHalfRad': r'$\log(\mathrm{sSFR}_{r < r_{1/2}}/\mathrm{yr}^{-1})$',
    'SubhalosSFRwithinHalfandRad': r'$\log sSFR_{r_{1/2} < r < 2r_{1/2}} \, \, [\mathrm{M_\odot \; yr^{-1}}]$',
    'SubhalosSFRinRad': r'$\log(\mathrm{sSFR}_{r < 2 r_{1/2}}/\mathrm{yr}^{-1})$',
    'sSFR_Outer': r'$\log(\mathrm{sSFR}_{r > r_{1/2}}/\mathrm{yr}^{-1})$',
    'SubhalosSFR': r'$\log(\mathrm{sSFR}/\mathrm{yr}^{-1})$',
    'SubhaloSFR': r'$\log(\mathrm{SFR}/\mathrm{M_\odot\; yr}^{-1})$',
    
    'SubhaloSFRinRad': r'$\log(\mathrm{SFR}_{r < 2 r_{1/2}}/\mathrm{\mathrm{M}_\odot\, yr}^{-1})$',
    'SubhaloSFRouterRad': r'$\log(\mathrm{SFR}_{r > 2 r_{1/2}}/\mathrm{\mathrm{M}_\odot\, yr}^{-1})$',

    'sSFRCoreRatio': r'$\mathrm{sSFR}_{r < r_{1/2}} / \mathrm{sSFR}_{r > r_{1/2}}$',
    'SigmasSFRRatio': r'$\Sigma \mathrm{sSFR}_{r < r_{1/2}} / \Sigma  \mathrm{sSFR}_{r > r_{1/2}}$',

    'SubhaloStellarMass_in_Rhpkpc': r'$\log(M_{\star}/\mathrm{M_\odot})$',
    
    'SubhaloStellarMass_Above_Rhpkpc': r'$\log(M_{\star}/\mathrm{M_\odot})$',

    'SubhaloSpin': r'$\log(j /\mathrm{km \; kpc \; s^{-1}})$',

    'J200': r'$j_{200} \mathrm{ \; km \; kpc \; s^{-1}}$',
    
    'StellarMassExSitu':  r'$\log(M_{\star,\; \mathrm{ex-situ}}/\mathrm{M_\odot})$',
    'StellarMassInSitu':  r'$\log(M_{\star,\; \mathrm{in-situ}}/\mathrm{M_\odot})$',
    
    'DMMass_In_Rhpkpc':  r'$\log(M_{\mathrm{DM}}/\mathrm{M_\odot})$',
    'DMMass_Above_Rhpkpc':  r'$\log(M_{\mathrm{DM}}/\mathrm{M_\odot})$',
    
    'GasMass_In_Rhpkpc':  r'$\log(M_{\mathrm{gas}}/\mathrm{M_\odot})$',
    'GasMass_Above_Rhpkpc':  r'$\log(M_{\mathrm{gas}}/\mathrm{M_\odot})$',
    
    'GasMass_In_TrueRhpkpc':  r'$\log(M_{\mathrm{gas}}/\mathrm{M_\odot})$',
    'GasMass_Above_TrueRhpkpc':  r'$\log(M_{\mathrm{gas}}/\mathrm{M_\odot})$',
    
    'StarMass_In_Rhpkpc':  r'$\log(M_{\star}/\mathrm{M_\odot})_{r \leq r_{1/2,\; z=0}}$',
    'StarMass_Above_Rhpkpc':  r'$\log(M_{\star}/\mathrm{M_\odot})$',
    
    'GasMassInflow_In_Rhpkpc': r'$\log(M_{\mathrm{gas,\, inflow}}/\mathrm{M_\odot})$',
    'GasMassInflow_Above_Rhpkpc':r'$\log(M_{\mathrm{gas,\, inflow}}/\mathrm{M_\odot})$',
    
    
    'SFGasMass_In_Rhpkpc':  r'$\log(M_{\mathrm{sf-gas}}/\mathrm{M_\odot}})$',
    'SFGasMass_Above_Rhpkpc':  r'$\log(M_{\mathrm{sf-gas}}/\mathrm{M_\odot})$',
    
    'GasMass_in_07Rhpkpc':  r'$\log(M_{\mathrm{gas}}/\mathrm{M_\odot})$',
    'GasMass_07_r_14Rhpkpc':   r'$\log(M_{\mathrm{gas}}/\mathrm{M_\odot})$',
    'GasMass_14_r_21Rhpkpc':  r'$\log(M_{\mathrm{gas}}/\mathrm{M_\odot})$',
    'GasMass_Above_21Rhpkpc':  r'$\log(M_{\mathrm{gas}}/\mathrm{M_\odot})$',
    
    'GasMass_Inflow_in_07Rhpkpc':  r'$\log(M_{\mathrm{gas,\;}v_\mathrm{rad} < 0}/\mathrm{M_\odot})$',
    'GasMass_Inflow_07_r_14Rhpkpc':  r'$\log(M_{\mathrm{gas,\;}v_\mathrm{rad} < 0}/\mathrm{M_\odot})$',
    'GasMass_Inflow_14_r_21Rhpkpc':  r'$\log(M_{\mathrm{gas,\;}v_\mathrm{rad} < 0}/\mathrm{M_\odot})$',
    'GasMass_Inflow_Above_21Rhpkpc':   r'$\log(M_{\mathrm{gas,\;}v_\mathrm{rad} < 0}/\mathrm{M_\odot})$',
    
    
    'sSFR_In_Rhpkpc': r'$\log(\mathrm{sSFR}/\mathrm{yr}^{-1})$',
    'sSFR_Above_Rhpkpc': r'$\log(\mathrm{sSFR}/\mathrm{yr}^{-1})$',
    
    'sSFR_In_TrueRhpkpc': r'$\log(\mathrm{sSFR}/\mathrm{yr}^{-1})$',
    'sSFR_Above_TrueRhpkpc': r'$\log(\mathrm{sSFR}/\mathrm{yr}^{-1})$',
    
    
    'GasMass_Inflow_In_Rhpkpc':  r'$\log(M_{\mathrm{gas,\;}r < 2r_{1/2}}/\mathrm{M_\odot})$',
    'GasMass_Outflow_In_Rhpkpc':  r'$\log(M_{\mathrm{gas,\;}r < 2r_{1/2}}/\mathrm{M_\odot})$',
    'GasMass_Inflow_Above_Rhpkpc':  r'$\log(M_{\mathrm{gas,\;}r > 2r_{1/2}}/\mathrm{M_\odot})$',
    'GasMass_Outflow_Above_Rhpkpc':  r'$\log(M_{\mathrm{gas,\;}r > 2r_{1/2}}/\mathrm{M_\odot})$',
    'U-r': r'$u-r \; [\mathrm{mag}]$',
    #Fracs
    'GasFrac_99': r'$(M_\mathrm{gas}/ M)_{z = 0}$',
    'StarFrac_99': r'$(M_\star/ M)_{z = 0}$',
    'DMFrac_99': r'$(M_\mathrm{DM}/ M)_{z = 0}$',
    
    'Mgas_Norm_Max_99': r'$M_{z = 0}/ M_\mathrm{max}$',
    'MDM_Norm_Max_99': r'$(M_{z = 0}/ M_\mathrm{max})_\mathrm{DM}$',
    'Mstar_Norm_Max_99': r'$M_{z = 0}/ M_\mathrm{max}$',

    'Loss_MDM_Norm_Max_99': r'DM-loss fraction',

    'Mgas_Norm_Max':  r'$(M/ M_\mathrm{max})_\mathrm{gas}$',
    'MDM_Norm_Max':  r'$(M/ M_\mathrm{max})_\mathrm{DM}$', 
    'Mstar_Norm_Max':r'$(M/ M_\mathrm{max})_\star$', 
    
    # Group
    'Group_M_Crit200': r'$\log(M_{200}/\mathrm{M}_\odot)$',
    'GroupM200_99': r'$\log(M_{200}/\mathrm{M}_\odot)_{z = 0}$',
    'rOverR200_99': r'$(R/R_{200})_{z = 0}$',
    
    
    'logM200_99': r'$\log(M_{200}/\mathrm{M}_\odot)$',
    'Group_M_Crit200FirstGroup': r'$\log(M_{200, \mathrm{first\; host}}/\mathrm{M}_\odot)$',
    'Group_M_Crit200FinalGroup': r'$\log(M_{200}/\mathrm{M}_\odot)$',
    'M200': r'$\log(M_{200}/\mathrm{M}_\odot)$',
    'R200': r'$\log(R/\mathrm{kpc})$',
    
    'CorotateFrac_GasInflow_Above_AbovestdRhpkpcCurrent': r'$(M_\mathrm{co-rotate}/M)_{\mathrm{gas-inflow}}$',
    'CumulativeCorotateFraction_at_8' : 'Accreted COROTATE material \n fraction',
    'l200': r'$\lambda_{200}$',
    'l200': r'$\lambda_{200}$',
    'lamMeanAfter1Gyr': r'$\overline{\lambda}_{200,\; \mathrm{1 \; Gyr\; After-Birth}}$',
    'lamMeanAfter2Gyr': r'$\overline{\lambda}_{200,\; \mathrm{2 \; Gyr\; After-Birth}}$',
    'lamAtFirst': r'$\lambda_{200,\; \mathrm{Birth}}$',
    'lamMeanAfter1Gyr': r'$\overline{\lambda}_{200,\; \mathrm{1 \; Gyr\; After-Birth}}$',
    'lamMeanAfter2Gyr': r'$\overline{\lambda}_{200,\; \mathrm{2 \; Gyr\; After-Birth}}$',
    'Lambda_at_99': r'$\lambda_{200,\; z = 0}$',
    'l200_99': r'$\lambda_{200,\; z = 0}$',

    'l200_at_Birth': r'$\lambda_{200,\; \mathrm{Birth}}$',
    'fEx_at_99': r'$(\mathrm{M}_{\star, \mathrm{ex-situ}} / \mathrm{M}_{\star, \mathrm{in-situ}})_{z = 0}$',
    'Ex_at_99': r'$\log(\mathrm{M}_{\star, \mathrm{ex-situ}} / M_\odot)_{z = 0}$',

    'meanl200_after_05Gyr_Birth': r'$\overline{\lambda}_{200,\; \mathrm{0.5 \; Gyr\; After-Birth}}$',

    'l200_NewMeanAfter1Gyr': r'$\overline{\lambda}_{200,\; \mathrm{1 \; Gyr\; After-Birth}}$',

    'meanl200_after_2Gyr_Birth': r'$\overline{\lambda}_{200,\; \mathrm{2 \; Gyr\; After-Birth}}$',
    'meanl200_after_5Gyr_Birth': r'$\overline{\lambda}_{200,\; \mathrm{5 \; Gyr\; After-Birth}}$',
    'meanl200_after_8Gyr_Birth': r'$\overline{\lambda}_{200,\; \mathrm{8 \; Gyr\; After-Birth}}$',
    
    'meanl200_after_2Gyr_Birth': r'$\overline{\lambda}_{200,\; \mathrm{2 \; Gyr\; After-Birth}}$',
    'l200_NewMeanAfter5Gyr': r'$\overline{\lambda}_{200,\; \mathrm{5 \; Gyr\; After-Birth}}$',
    'l200_NewMeanAfter8Gyr': r'$\overline{\lambda}_{200,\; \mathrm{8 \; Gyr\; After-Birth}}$',
    'l200_New_at_99': r'$\lambda_{200,\; z = 0}$',

    'l200_NewAtFirst': r'$\lambda_{200,\; \mathrm{Birth}}$',

    'Ex_after_05Gyr_Birth': r'$\Delta (\log {M}_{\star,\; \mathrm{ex-situ}})_\mathrm{0.5 \; Gyr\; After-Birth}$',
    'In_after_05Gyr_Birth': r'$\Delta (\log {M}_{\star,\; \mathrm{in-situ}})_\mathrm{0.5 \; Gyr\; After-Birth}$',
    'l200_after_05Gyr_Birth': r'$\Delta ({\lambda}_{200})_\mathrm{0.5 \; Gyr\; After-Birth}$',
    'Rh_after_05Gyr_Birth': r'$\Delta (\log {r}_{05/2})_\mathrm{0.5 \; Gyr\; After-Birth}$',
    
    'Ex_after_1Gyr_Birth': r'$\Delta (\log {M}_{\star,\; \mathrm{ex-situ}})_\mathrm{1 \; Gyr\; After-Birth}$',
    'In_after_1Gyr_Birth': r'$\Delta (\log {M}_{\star,\; \mathrm{in-situ}})_\mathrm{1 \; Gyr\;\ After-Birth}$',
    'l200_after_1Gyr_Birth': r'$\Delta ({\lambda}_{200})_\mathrm{1 \; Gyr\; After-Birth}$',
    'Rh_after_1Gyr_Birth': r'$\Delta (\log {r}_{1/2})_\mathrm{1 \; Gyr\; After-Birth}$',


     'Ex_after_8Gyr_Birth': r'$\Delta (\log {M}_{\star,\; \mathrm{ex-situ}})_\mathrm{8 \; Gyr\;  After-Birth}$',
     'In_after_8Gyr_Birth': r'$\Delta (\log {M}_{\star,\; \mathrm{in-situ}})_\mathrm{8 \; Gyr\;  After-Birth}$',
     'l200_after_8Gyr_Birth': r'$\Delta ({\lambda}_{200})_\mathrm{8 \; Gyr\;  After-Birth}$',
     'Rh_after_8Gyr_Birth': r'$\Delta (\log {r}_{8/2})_\mathrm{8 \; Gyr\;  After-Birth}$',
     
     'Ex_after_2Gyr_Birth': r'$\Delta (\log {M}_{\star,\; \mathrm{ex-situ}})_\mathrm{2 \; Gyr\;  After-Birth}$',
     'In_after_2Gyr_Birth': r'$\Delta (\log {M}_{\star,\; \mathrm{in-situ}})_\mathrm{2 \; Gyr \; After-Birth}$',
     'l200_after_2Gyr_Birth': r'$\Delta ({\lambda}_{200})_\mathrm{2 \; Gyr\;  After-Birth}$',
     'Rh_after_2Gyr_Birth': r'$\Delta (\log {r}_{2/2})_\mathrm{2 \; Gyr\;  After-Birth}$',
     
     'Ex_after_5Gyr_Birth': r'$\Delta (\log {M}_{\star,\; \mathrm{ex-situ}})_\mathrm{5 \; Gyr\;  After-Birth}$',
     'In_after_5Gyr_Birth': r'$\Delta (\log {M}_{\star,\; \mathrm{in-situ}})_\mathrm{5 \; Gyr \; After-Birth}$',
     'l200_after_5Gyr_Birth': r'$\Delta ({\lambda}_{200})_\mathrm{5 \; Gyr\;  After-Birth}$',
     'Rh_after_5Gyr_Birth': r'$\Delta (\log {r}_{5/2})_\mathrm{5 \; Gyr\;  After-Birth}$',
     
    'Group_R_Crit200': r'$\log(R_{200}/\mathrm{kpc})$',
    'Group_R_Crit200FirstGroup': r'$\log(R_{200, \mathrm{first\; host}}/\mathrm{kpc})$',
    'Group_R_Crit200FinalGroup': r'$\log(R_{200, \mathrm{final\; host}}/\mathrm{kpc})$',

    'Group_M_Crit500': r'$\log(M_{500}/\mathrm{M}_\odot)$',
    'Group_R_Crit500': r'$\log(R_{500}/\mathrm{kpc})$',

    'GroupNsubs': r'Number of satellites',
    'GroupNsubsFirstGroup': 'Number of satellites \n First host',
    'GroupNsubsFinalGroup': 'Number of satellites \n Final host',
    'GroupNsubsPriorGroup': r'Number of subhalos',

    #ExSitu Contribution
    'MassExNormalize': r'$(M_{\mathrm{ex-situ}} / M_{\mathrm{ex-situ},\; z = 0})$',
    'MassInNormalize': r'$(M_{\mathrm{in-situ}} / M_{\mathrm{ex-situ},\; z = 0})$',

    'MassExNormalizeAll': r'$(M_{\mathrm{ex-situ}} / M_{\star,\; z = 0})$',
    'StellarMassExSituMinor': r'$\log(M_{\mathrm{ex-situ,\; minor\; merger}}/\mathrm{M}_\odot)$',
    'StellarMassExSituIntermediate': r'$\log(M_{\mathrm{ex-situ,\; intermediate\; merger}}/\mathrm{M}_\odot)$',
    'StellarMassExSituMajor': r'$\log(M_{\mathrm{ex-situ,\; major\; merger}}/\mathrm{M}_\odot)$',

    'logStellarMassExSituMinor_99': r'$\log(M_\mathrm{ex-situ}/\mathrm{M}_\odot)_{z = 0}$',
    'logStellarMassExSituIntermediate_99': r'$\log(M_\mathrm{ex-situ}/\mathrm{M}_\odot)_{z = 0}$',
    'logStellarMassExSituMajor_99': r'$\log(M_\mathrm{ex-situ}/\mathrm{M}_\odot)_{z = 0}$',

    'LBTimeMajorMerger': 'Lookback time \n Major Merger [Gyr]',
    'LBTimeMinorMerger': 'Lookback time \n  Minor Merger [Gyr]',
    'LBTimeIntermediateMerger': 'Lookback time \n Intermediate Merger [Gyr]',

    'ExMassType0Evolution': r'$\log(M_{\mathrm{gas, \; ex-situ}}/\mathrm{M}_\odot)$',
    'ExMassType1Evolution': r'$\log(M_{\mathrm{DM, \; ex-situ}}/\mathrm{M}_\odot)$',
    'ExMassType4Evolution': r'$\log(M_{\mathrm{\star, \; ex-situ}}/\mathrm{M}_\odot)$',


    'SnapLastMerger_Before_Entry': 'Lookback time \n Last Major or Intermediate \n merger before Entry [Gyr]',
    'deltaSize_BeforeEntry': r'$(\overline{r_{1/2,\;\mathrm{max}}} - r_{1/2,\;\mathrm{entry}}) [\mathrm{kpc}]$',
    'GasjInflow_BeforeEntry': r'$\overline{\log(\mathrm{j} /\mathrm{km \; kpc \; s^{-1}} )}_{\mathrm{gas}}^{\mathrm{entry}}$',

    'deltaT_Merger_Entry': r'$(t_\mathrm{entry} - t_\mathrm{last M+I merger})$',
    #Orbits
    'r_over_R_Crit200': r'$R/R_{200}$',
    'tsincebirth': r'$t - t_\mathrm{birth}$',
    'rOverR200Min': r'$(R/R_{200})_\mathrm{min}$',
    'zInfall': r'$z$ of first infall',
    'zInfall_New': r'$z$ of first infall',

    'deltaStarInAbove': r'$(M_\mathrm{\star,\; r > 5 \; kpc} - M_\mathrm{\star,\; r < 2 \; kpc}) / M_\star$',

    #Others:
    'MassTensorEigenVals': r'$\mu_1 / \sqrt{\mu_2 \mu_3}$',
    'logjProfile': r'$\log (j_{\mathrm{gas}} / \, \, [\mathrm{kpc \; km  \; s^{-1}}])$',
    'rToRNearYoung': r'$d_{\mathrm{NNB}}$ [kpc]', 
    'Gasj_Rhpkpc_Diffuse_r_plus050dex': r'$j_{\mathrm{gas}} \, \, [\mathrm{kpc \; km  \; s^{-1}}]$',
    'sSFRCoreRatioAfterz5': r'$\overline{(\mathrm{sSFR}_{r < r_{1/2}} / \mathrm{sSFR}_{r > r_{1/2}})}_{z < 5}$',
    'MassIn_Infall_to_GasLost': r'$(\Delta M_\star)_{\mathrm{inner}}^\mathrm{entry-to-gas-loss} / M_\star^\mathrm{entry}$', #'Relative inner stellar mass \n change during period', #
    'MassAboveAfterInfall_Lost': r'$(\Delta M_\star)_{\mathrm{outer}} ^\mathrm{no-gas} / M_{\star}^{\mathrm{gas-loss}}$', #r'$(\Delta M_\star)_{r > r_{1/2, z = 0},  M_\mathrm{gas, \, min} \mathrm{\, to \,} z = 0} / M_{\star, M_\mathrm{gas, \, min} }$',
    'MassAboveAfter_Infall_to_GasLost': r'$(\Delta M_\star)_{\mathrm{outer}}^\mathrm{entry-to-gas-loss} / M_\star^{\mathrm{entry}}$', #'Relative inner stellar mass \n change during period', #
    

    'dMIn_Entry_to_Max': r'$(\Delta M_\star)_{\mathrm{inner}}^\mathrm{entry-to-max-star} / M_\star^\mathrm{entry}$',
    'dMAbove_Entry_to_Max': r'$(\Delta M_\star)_{\mathrm{outer}}^\mathrm{entry-to-max-star} / M_\star^\mathrm{entry}$',

    'dMIn_Max_to_Nogas': r'$(\Delta M_\star)_{\mathrm{inner}}^\mathrm{max-star-to-no-gas} / M_\star^\mathrm{max-star}$',
    'dMAbove_Max_to_Nogas': r'$(\Delta M_\star)_{\mathrm{outer}}^\mathrm{max-star-to-no-gas} / M_\star^\mathrm{max-star}$',

    'dMIn_NoGas_Final': r'$(\Delta M_\star)_{\mathrm{inner}}^\mathrm{no-gas-to-final} / M_\star^\mathrm{no-gas}$',
    'dMAbove_NoGas_Final': r'$(\Delta M_\star)_{\mathrm{outer}}^\mathrm{no-gas-to-final} / M_\star^\mathrm{no-gas}$',

    'dSize_Entry_to_Max': r'$(\Delta r_{1/2} / \Delta t )^\mathrm{entry-to-max-star} $',
    'dSize_Max_to_Nogas': r'$(\Delta r_{1/2} / \Delta t)^\mathrm{max-star-to-no-gas}$',
    'dSize_NoGas_Final': r'$(\Delta r_{1/2} / \Delta t)^\mathrm{no-gas-to-final}$',


    'StarMass_Above_Normalize_99': r'$(M_\star / M_{\star, \mathrm{max}})_{\mathrm{outer}, z = 0}$', #'Relative inner stellar mass \n change during period', #
    'StarMass_In_Normalize_99': r'$(M_\star / M_{\star, \mathrm{max}})_{\mathrm{inner}, z = 0}$', #'Relative inner stellar mass \n change during period', #
    'DeltaStarMass_Above_Normalize_99': r'$(M_{\star, \mathrm{max}} - M_{\star, \mathrm{z \;=\; 0}} )_{\mathrm{outer}} /  M_{\star, \mathrm{z \;=\; 0}}$', #'Relative inner stellar mass \n change during period', #
    'DeltaStarMass_In_Normalize_99':  r'$(M_{\star, \mathrm{max}} - M_{\star, \mathrm{z \;=\; 0}} )_{\mathrm{inner}} /  M_{\star, \mathrm{z \;=\; 0}}$', #'Relative inner stellar mass \n change during period', #
    
    'DeltaMassIn_Infall_In_Normalize_99':r'$(\Delta M_\star)_{\mathrm{inner}}^\mathrm{entry-to-gas-loss} /  M_{\star, \mathrm{z \;=\; 0}}$', #'Relative inner stellar mass \n change during period', #
    'DecreaseBetweenGasStar_Over_starFinal': r'$(\Delta \log r_{1/2})^{\mathrm{no-gas}} - (\Delta \log r_{1/2})^\mathrm{entry-to-gas-loss}$',
    'SnapLostGas': 'Gas loss lookback time  [Gyr]', #$M_\mathrm{gas} = 0$',
    'SnapAtEntry_First': 'Entry lookback time  [Gyr]', #$M_\mathrm{gas} = 0$',
    'Rhalf_MaxProfile_Minus_HalfRadstar_99': r'$r_{1/2,\; \mathrm{sf}} - r_{1/2, z = 0}$',  
    'Rhalf_MinProfile_Minus_HalfRadstar_99': r'$r_{1/2,\; \mathrm{ts}} - r_{1/2, z = 0}$',

    'Rhalf_MaxProfile_Minus_HalfRadstar_Entry': r'$(r_{1/2,\; \mathrm{sf}} - r_{1/2, z_\mathrm{entry}}) [\mathrm{kpc}]$',
    'Rhalf_MinProfile_Minus_HalfRadstar_Entry': r'$(r_{1/2,\; \mathrm{ts}} - r_{1/2, z_\mathrm{entry}}) [\mathrm{kpc}]$',

    'Relative_Rhalf_MaxProfile_Minus_HalfRadstar_Entry': r'$(r_{1/2,\; \mathrm{sf}} - r_{1/2, z_\mathrm{entry}})/  r_{1/2, z_\mathrm{entry}}$',
    'Relative_Rhalf_MinProfile_Minus_HalfRadstar_Entry': r'$(r_{1/2,\; \mathrm{ts}} - r_{1/2, z_\mathrm{entry}}) / r_{1/2, z_\mathrm{entry}}$',

    'Ratio_half_mass_SF': r'$(r_{1/2,\; \mathrm{sf}} - r_{1/2, z_\mathrm{entry}})/  r_{1/2, z_\mathrm{entry}}$',
    'Ratio_half_mass_TS': r'$(r_{1/2,\; \mathrm{ts}} - r_{1/2, z_\mathrm{entry}}) / r_{1/2, z_\mathrm{entry}}$',


    'Rhalf_MaxProfile_Over_HalfRadstar_99': r'$r_{1/2,\; \mathrm{sf}} / r_{1/2, z = 0}$',  
    'Rhalf_MinProfile_Over_HalfRadstar_99': r'$r_{1/2,\; \mathrm{ts}} / r_{1/2, z = 0}$',

    'deltaInnersSFR_afterEntry': r'$(\mathrm{sSFR}_{\mathrm{max,\;entry-to-gas-loss}}/\mathrm{sSFR}_{\mathrm{entry}})_{r < r_{1/2, z = 0}}$',
    'deltaInnersSFR_afterEntry_all': r'$(\overline{\mathrm{sSFR}}_\mathrm{max}^\mathrm{entry-to-gas-loss}/\overline{\mathrm{sSFR}}_\mathrm{max}^\mathrm{before-entry})_{r < 2 r_{1/2, z = 0}}$',
    'deltaInnersSFR_afterEntry_all_EntryRh': r'$(\overline{\mathrm{sSFR}_{\mathrm{entry-to-gas-loss}}/\mathrm{sSFR}_{\mathrm{entry}}})_{3 - max,\; r < r_{1/2,\mathrm{entry}}}$',

    'deltaTrueInnersSFR_afterEntry': r'$(\mathrm{sSFR}_{\mathrm{max,\;entry-to-gas-loss}}/\mathrm{sSFR}_{\mathrm{entry}})_{\mathrm{inner}}$',
    'deltaTrueInnersSFR_afterEntry_all': r'$(\overline{\mathrm{sSFR}}_\mathrm{max}^\mathrm{entry-to-gas-loss}/\overline{\mathrm{sSFR}}_\mathrm{max}^\mathrm{before-entry})_{\mathrm{inner}}$',
    'deltaTrueInnersSFR_afterEntry_all_EntryRh': r'$(\overline{\mathrm{sSFR}_{\mathrm{entry-to-gas-loss}}/\mathrm{sSFR}_{\mathrm{entry}}})_{3 - max,\; r < r_{1/2,\mathrm{entry}}}$',

    'MassStarIn_Over_Above_absolutevalue':r'$((\Delta M_\star)_{\mathrm{inner}} / (\Delta M_\star)_{\mathrm{outer}})^\mathrm{entry-to-gas-loss} $',
    'BeforesSFR_Entry': r'$\overline{\log{(\mathrm{sSFR}}/\mathrm{yr^{-1}})}_{r < 2r_{1/2, z = 0}}^{\mathrm{entry}}$',
    'meanlogSFR_BeforeEntry': r'$\overline{\log{(\mathrm{SFR}}/\mathrm{M_\odot \, yr^{-1}})}^{\mathrm{entry}}$',
    'logSFR_AfterEntry': r'$\overline{\log{(\mathrm{SFR}}/\mathrm{M_\odot \, yr^{-1}})}^{\mathrm{entry-to-gas-loss}}$',
    'InnersSFR_Entry_to_quench': r'$\overline{\log{(\mathrm{sSFR}}/\mathrm{yr^{-1}})}_{r < 2r_{1/2, z = 0}}^{\mathrm{entry-to-gas-loss}}$',
    'sSFRAbove_Entry': r'$\overline{\log{(\mathrm{sSFR}_{r > r_{1/2, z = 0}}/\mathrm{yr^{-1}})}}_{\mathrm{entry}}$',

    'RatioSFR': r'$\overline{\mathrm{SFR}}^{\mathrm{entry-to-no-gas}}/ \overline{\mathrm{SFR}}^{\mathrm{entry}}$',

    'meanlogZ_BeforeEntry': r'$\overline{\log{(Z_\star/Z_\odot)}}^{\mathrm{entry}}$',

    'sSFRInner_BeforeEntry': r'$\log{(\overline{\mathrm{sSFR}}/\mathrm{yr^{-1}}})_{r < 2r_{1/2, z = 0}}^{\mathrm{ before-entry}}$',
    'sSFRInner_Entry_to_Nogas': r'$\log{(\overline{\mathrm{sSFR}}/\mathrm{yr^{-1}}})_{r < 2r_{1/2, z = 0}}^{\mathrm{entry-to-gas-loss}}$',
    'sSFRRatio_Entry_to_Nogas':  r'$(\overline{\mathrm{sSFR}_{r < 2r_{1/2, z = 0}}/\mathrm{sSFR}_{r > 2r_{1/2, z = 0}}})^{\mathrm{entry-to-gas-loss}}$',

    'sSFRInner_BeforeEntry_Rentr': r'$\log{(\overline{\mathrm{sSFR}}/\mathrm{yr^{-1}}})_{r < r_{1/2, \mathrm{entry}}}^{\mathrm{ before-entry}}$',
    'sSFRInner_Entry_to_Nogas_Rentry': r'$\log{(\overline{\mathrm{sSFR}}/\mathrm{yr^{-1}}})_{r < r_{1/2, \mathrm{entry}}}^{\mathrm{entry-to-gas-loss}}$',
    
    'sSFRTrueInner_BeforeEntry': r'$\log{(\overline{\mathrm{sSFR}}/\mathrm{yr^{-1}}})_{\mathrm{inner}}^{\mathrm{entry}}$',
    'sSFRTrueInner_Entry_to_Nogas': r'$\log{(\overline{\mathrm{sSFR}}/\mathrm{yr^{-1}}})_{\mathrm{inner}}^{\mathrm{entry-to-gas-loss}}$',
    'sSFRTrueRatio_Entry_to_Nogas':  r'$(\overline{\mathrm{sSFR}_\mathrm{inner}/\mathrm{sSFR}_{\mathrm{outer}}})^{\mathrm{entry-to-gas-loss}}$',

    'sSFRRatio_ETOGAS_Before':r'$\overline{\mathrm{sSFR}}_{\mathrm{inner}}^{\mathrm{entry-to-gas-loss}} /\overline{\mathrm{sSFR}}_{\mathrm{inner}}^{\mathrm{entry}}$' ,
    'sSFRCoreRatioAfterz5': r'$\overline{(\mathrm{sSFR}_{r < r_{1/2}} / \mathrm{sSFR}_{r > r_{1/2}})}_{z < 5}$',
    'sSFRinHalfRadAfterz5': r'$\overline{\log{(\mathrm{sSFR}_{r < r_{1/2}}/\mathrm{yr^{-1}})}}_{z < 5}$',
    
    'Decrease_Entry_To_NoGas': r'$(\Delta r_{1/2})^\mathrm{entry-to-gas-loss}/\mathrm{kpc}$',
    'Decrease_NoGas_To_Final': r'$(\Delta r_{1/2})^\mathrm{no-gas}/\mathrm{kpc}$',

    'Decrease_Entry_To_NoGas_Norm': r'$(\Delta r_{1/2})^\mathrm{entry-to-gas-loss} / r_{1/2}^\mathrm{entry}$',
    'Decrease_NoGas_To_Final_Norm': r'$(\Delta r_{1/2})^\mathrm{no-gas}/ r_{1/2}^\mathrm{no-gas}$',

    'Decrease_Entry_To_NoGas_Norm_Delta': r'$(\Delta r_{1/2} / (r_{1/2}^\mathrm{entry}  \Delta t))^\mathrm{entry-to-gas-loss}\, \mathrm{[Gyr^{-1}]}$',
    'Decrease_NoGas_To_Final_Norm_Delta': r'$(\Delta r_{1/2} / (r_{1/2}^\mathrm{no-gas}  \Delta t))^\mathrm{no-gas}\, \mathrm{[Gyr^{-1}]}$', 
   

    'Decrease_Entry_To_NoGas_Norm_Norb': r'$(\Delta r_{1/2} / (r_{1/2}^\mathrm{entry}  \mathrm{N_{orb}}))^\mathrm{entry-to-gas-loss}$',
    'Decrease_NoGas_To_Final_Norm_Norb': r'$(\Delta r_{1/2} / (r_{1/2}^\mathrm{no-gas}   \mathrm{N_{orb}}))^\mathrm{no-gas}$', 
    'Norbit_Entry_To_NoGas': r'$\mathrm{N_{per}}^\mathrm{entry-to-gas-loss}$',

    
    #Profile
    'RadVelocity':  r'$v_\mathrm{r} (r) \, \, [\mathrm{km \, s}^{-1}]$',
    'j': r'$j_{\mathrm{gas}} (r)  \; \, \, [\mathrm{km \; kpc \; s^{-1}}]$',
    'RadVelocityPartType0':  r'$v_\mathrm{r, \; gas} (r) \, \, [\mathrm{km \, s}^{-1}]$',
    'RadVelocityPartType4':  r'$v_\mathrm{r, \; \star} (r) \, \, [\mathrm{km \, s}^{-1}]$',
    'jPartType0': r'$j_{\mathrm{gas}} (r)  \; \, \, [\mathrm{km \; kpc \; s^{-1}}]$',
    'jPartType4': r'$j_{\star} (r)  \; \, \, [\mathrm{km \; kpc \; s^{-1}}]$',
    'DensityGas': r'$\rho_{\mathrm{gas}} (r)  \; \, \, [\mathrm{M_\odot  \; kpc^{-3}}]$',
    'SFR': r'$\mathrm{SFR} \, \, [\mathrm{M_\odot \; yr^{-1}}]$',
    'DensityStar': r'$\rho_{\star} (r)  \; \, \, [\mathrm{M_\odot  \; kpc^{-3}}]$',


    'joverR': r'$j (r) / r  \; \, \, [\mathrm{km  \; s^{-1}}]$',
    'joverRPartType0': r'$j_{\mathrm{gas}} (r) / r  \; \, \, [\mathrm{km  \; s^{-1}}]$',
    'joverRPartType4': r'$j_{\star} (r) / r  \; \, \, [\mathrm{km  \; s^{-1}}]$',
    'DensityGasOverR2':  r'$\rho_{\mathrm{gas}} (r)  r^2 \; \, \, [\mathrm{M_\odot  \; kpc^{-1}}]$',
    'sSFR': r'$\mathrm{sSFR} \, \, [\mathrm{M_\odot \; yr^{-1}}]$',
    'DensityStarOverR2':  r'$\rho_{\star} (r)  r^2 \; \, \, [\mathrm{M_\odot  \; kpc^{-1}}]$',

    'rad': r'$r \, \, [\mathrm{kpc}]$',

    #Scatter
    'z_Birth': r'$z_\mathrm{birth}$',
    'DMFrac_Birth': r'$(M_\mathrm{DM}/M)_\mathrm{birth}$',
    'fracEx_99': r'$(f_\mathrm{ex-situ})_\mathrm{z = 0}$',


    'nn2_distance': '$r_{nn} [\mathrm{kpc}]$',
    'nn5_distance': '$r_{nn} [\mathrm{kpc}]$',
    'nn10_distance': '$r_{nn} [\mathrm{kpc}]$',
    
    'nn2_distance_massive': '$r_{nn, \mathrm{massive}} [\mathrm{kpc}]$',
    'nn5_distance_massive': '$r_{nn, \mathrm{massive}} [\mathrm{kpc}]$',
    'nn10_distance_massive': '$r_{nn, \mathrm{massive}} [\mathrm{kpc}]$',


    'N_aper_1_Mpc':  '$N \mathrm{neighbours}$',
    'N_aper_2_Mpc': '$N \mathrm{neighbours}$',
    'N_aper_5_Mpc':  '$N \mathrm{neighbours}$',
    
    'N_aper_1_Mpc_massive':  '$N \mathrm{massive \, neighbours}$',
    'N_aper_2_Mpc_massive':  '$N \mathrm{massive \,neighbours}$',
    'N_aper_5_Mpc_massive': '$N \mathrm{massive \, neighbours}$',
    
    #
    
    'deltaN_z5': r'$\Delta(\mathrm{N_{sub}})_{_\mathrm{birth - to - z = 5}}$',
    'deltaN_z5z2': r'$\Delta(\mathrm{N_{sub}})_{_\mathrm{z = 2 - to - z = 5}}$',
    'NsubsPrior': r'$\mathrm{N_{sub, \; z = 0}}$',



    # PAPERII
    'M200Mean': r'$\overline{\log(M_{200} / \mathrm{M}_\odot)}$',
    'rOverR200Mean': r'$\overline{(\mathrm{R} / \mathrm{R}_{200})}$',
    'rOverR200Mean_New': r'$\overline{(\mathrm{R} / \mathrm{R}_{200})}$',


    'LBTime_Loss_Gass': 'Gas loss lookback time  [Gyr]',

    'z_At_FinalEntry':  r'$z_{\mathrm{infall}}$ in final host',
    'z_At_FirstEntry':  r'$z_{\mathrm{infall}}$ in first host',
    'deltaFirst_to_Final':  r'$\Delta \mathrm{t}_{\mathrm{first-to-final}}$ [Gyr]',
    
    'deltaSize_at_Entry':  r'$[(r_{{1/2}} -\left<r_{{1/2}}\right>) / \sigma_{r_{{1/2}}}]^{\mathrm{entry}}$ ',
    'deltalogSize_At_Entry':r'$(\log r_{{1/2}} -\left<\log r_{{1/2}}\right>)$',

    'DeltaGasMass_In_Entry_to_gas_loss':  r'$(\overline{\Delta M_\mathrm{gas} / M_\mathrm{gas}})_\mathrm{r<2r_{1/2, \; z = 0}}^\mathrm{entry-to-gas-loss}$ ',
    'DeltaGasMass_Above_Entry_to_gas_loss': r'$(\overline{\Delta M_\mathrm{gas} / M_\mathrm{gas}})_\mathrm{r>2r_{1/2, \; z = 0}}^\mathrm{entry-to-gas-loss}$ ',

    'GasInner_Entry_to_Nogas':  r'$(\overline{\Delta M_\mathrm{gas} / M_\mathrm{gas}})_\mathrm{inner}^\mathrm{entry-to-gas-loss}$ ',
    'GasAbove_Entry_to_Nogas': r'$(\overline{\Delta M_\mathrm{gas} / M_\mathrm{gas}})_\mathrm{outer}^\mathrm{entry-to-gas-loss}$ ',
    'TimeLossGass': r'$\Delta t^\mathrm{entry-to-gas-loss} \; [\mathrm{Gyr}]$ ',
    
    'GasTrueInner_Entry_to_Nogas':  r'$(\overline{\Delta M_\mathrm{gas} / M_\mathrm{gas}})_\mathrm{inner}^\mathrm{entry-to-gas-loss}$ ',
    'GasTrueAbove_Entry_to_Nogas': r'$(\overline{\Delta M_\mathrm{gas} / M_\mathrm{gas}})_\mathrm{outer}^\mathrm{entry-to-gas-loss}$ ',
    
    'DeltasSFR_Ratio': r'$(\overline{\Delta \mathrm{sSFR}_\mathrm{outer}} / \overline{\Delta \mathrm{sSFR}_\mathrm{inner}})^\mathrm{entry-to-gas-loss}$ ',
    'sSFRRatioPericenter': r'$(\overline{\mathrm{sSFR}_\mathrm{After}} / \overline{\mathrm{sSFR}_\mathrm{Before}})^\mathrm{1st\,pericenter}_\mathrm{inner}$ ',

    'StellarMassExSitu_At_Entry': r'$\log(M_{\star, \; \mathrm{ex-situ}}/M_\odot)$',
    'dLastMajor_Before_Entry': r'$\Delta t_\mathrm{since \;last \;merger} [Gyr]$',
    'dLastIntermediate_Before_Entry': r'$\Delta t_\mathrm{since \;last \;merger} [Gyr]$',
    'dLastMinor_Before_Entry': r'$\Delta t_\mathrm{since \;last \;merger} [Gyr]$',
    
    
    'StarMass_GasLoss_Over_EntryToGas': r'$\overline{(\Delta M_\star)}^{\mathrm{no-gas}} / \overline{(\Delta M_\star)}^{\mathrm{entry-to-gas-loss}}$',
    None: 'Any'
}

texts = { 
    #Subhalo
    
    #Subhalo
    'l200_NewMeanAfter1Gyr': r'$1$ Gyr after birth',
    'l200_NewMeanAfter5Gyr': r'$5$ Gyr after birth',
    'l200_NewMeanAfter8Gyr': r'$8$ Gyr after birth',

    'l200_New_at_99': r'$z = 0$',
    
    
    'logHalfRadstar_99': r'$\log(r_{1/2, \, z = 0}/\mathrm{kpc})$',

    'SubhaloHalfmassRadType0': r'$r_{1/2, \mathrm{gas}}$',
    'SubhaloHalfmassRadType1': r'$r_{1/2, \mathrm{DM}}$',
    'SubhaloHalfmassRadType4': r'$r_{1/2, \star}$',
    
    'CompareSubhaoHalfmassRadType1': r'$r_{1/2}/r_{\mathrm{entry}}$',
    'CompareSubhaoHalfmassRadType4': r'$r_{1/2}/r_{\mathrm{entry}}$',



    'SubhaloMassInRadType0': r'$M_{\mathrm{gas}, r < 2 r_{1/2}}$',
    'SubhaloMassInRadType1': r'$M_{\mathrm{DM}, r < 2 r_{1/2}}$',
    'SubhaloMassInRadType4': r'$M_{\star, r < 2 r_{1/2}}$',

    'SubhaloMassType0': r'$M_{\mathrm{gas}}$',
    'SubhaloMassType1': r'$M_{\mathrm{DM}}$',
    'SubhaloMassType4': r'$M_{\star}$',

    'SubhalosSFRInHalfRad': r'$\mathrm{sSFR}_{r < r_{1/2}}$',
    'sSFR_Outer': r'$\mathrm{sSFR}_{r > r_{1/2}}$',
    'SubhalosSFRwithinHalfandRad': r'$\mathrm{sSFR}_{r_{1/2} < r < 2r_{1/2}}$',
    'SubhalosSFRinRad': r'$\mathrm{sSFR}_{r < 2 r_{1/2}}$',
    'SubhalosSFR': r'$\mathrm{sSFR}$',

    'sSFRCoreRatio': r'$\mathrm{sSFR}_{r < r_{1/2}} / \mathrm{sSFR}_{r > r_{1/2}}$',
    'SigmasSFRRatio': r'$\Sigma \mathrm{sSFR}_{r < r_{1/2}} / \Sigma  \mathrm{sSFR}_{r > r_{1/2}}$',

    'SubhaloStellarMass_in_Rhpkpc': r'$\log(M_{\star}/\mathrm{M_\odot})$',
    
    'SubhaloStellarMass_Above_Rhpkpc': r'$\log(M_{\star}/\mathrm{M_\odot})$',


    'DMMass_In_Rhpkpc':  r'$\log(M_{\mathrm{DM}}/\mathrm{M_\odot})$',
    'DMMass_Above_Rhpkpc':  r'$\log(M_{\mathrm{DM}}/\mathrm{M_\odot})$',
    
    'GasMass_In_Rhpkpc':  r'$\log(M_{\mathrm{gas}}/\mathrm{M_\odot})$',
    'GasMass_Above_Rhpkpc':  r'$\log(M_{\mathrm{gas}}/\mathrm{M_\odot})$',
    'GasMass_In_TrueRhpkpc':  r'$\log(M_{\mathrm{gas}}/\mathrm{M_\odot})$',
    'GasMass_Above_TrueRhpkpc':  r'$\log(M_{\mathrm{gas}}/\mathrm{M_\odot})$',
    'StarMass_In_Rhpkpc':  r'$\log(M_{\star}/\mathrm{M_\odot})$',
    'StarMass_Above_Rhpkpc':  r'$\log(M_{\star}/\mathrm{M_\odot})$',
    'GasMassInflow_In_Rhpkpc': r'$\log(M_{\mathrm{gas,\, inflow}}/\mathrm{M_\odot})$',
    'GasMassInflow_Above_Rhpkpc':r'$\log(M_{\mathrm{gas,\, inflow}}/\mathrm{M_\odot})$',
    'SFGasMass_in_Rhpkpc':  r'$\log(M_{\mathrm{sf-gas}}/\mathrm{M_\odot})$',
    'SFGasMass_Above_Rhpkpc':  r'$\log(M_{\mathrm{sf-gas}}/\mathrm{M_\odot})$',
    
    'logStar_GFM_Metallicity_In_Rhpkpc': r'$\log( Z_\star / Z_\odot)$',
    'logStar_GFM_Metallicity_Above_Rhpkpc': r'$\log( Z_\star / Z_\odot)$',


    'GasMass_in_07Rhpkpc':  r'$\log(M_{\mathrm{gas}}/\mathrm{M_\odot})$',
    'GasMass_07_r_14Rhpkpc':  r'$\log(M_{\mathrm{gas}}/\mathrm{M_\odot})$',
    'GasMass_14_r_21Rhpkpc':  r'$\log(M_{\mathrm{gas}}/\mathrm{M_\odot})$',
    'GasMass_Above_21Rhpkpc':  r'$\log(M_{\mathrm{gas}}/\mathrm{M_\odot})$',
    
    'GasMass_Inflow_in_07Rhpkpc':  r'$\log(M_{\mathrm{gas,\;}v_\mathrm{rad} < 0}/\mathrm{M_\odot})$',
    'GasMass_Inflow_07_r_14Rhpkpc':  r'$\log(M_{\mathrm{gas,\;}v_\mathrm{rad} < 0}/\mathrm{M_\odot})$',
    'GasMass_Inflow_14_r_21Rhpkpc':  r'$\log(M_{\mathrm{gas,\;}v_\mathrm{rad} < 0}/\mathrm{M_\odot})$',
    'GasMass_Inflow_Above_21Rhpkpc':   r'$\log(M_{\mathrm{gas,\;}v_\mathrm{rad} < 0}/\mathrm{M_\odot})$',
    
    'StellarMassExSitu':  r'$\log(M_{\star}/\mathrm{M_\odot})$',
    'StellarMassInSitu':  r'$\log(M_{\star}/\mathrm{M_\odot})$',
    
    'SubhaloSpin': r'$\log(j /\mathrm{km \; kpc \; s^{-1}})$',
    
    'J200': r'$\log(j_{200} /\mathrm{km \; kpc \; s^{-1}})$',


    'sSFR_In_Rhpkpc': r'$\log(\mathrm{sSFR}/\mathrm{yr}^{-1})$',
    'sSFR_Above_Rhpkpc': r'$\log(\mathrm{sSFR}/\mathrm{yr}^{-1})$',
    
    'sSFR_In_TrueRhpkpc': r'$\log(\mathrm{sSFR}/\mathrm{yr}^{-1})$',
    'sSFR_Above_TrueRhpkpc': r'$\log(\mathrm{sSFR}/\mathrm{yr}^{-1})$',
    
    'SFR_In_Rhpkpc': r'$\log(\mathrm{SFR}/\mathrm{M_\odot \,yr}^{-1})$',
    'SFR_Above_Rhpkpc': r'$\log(\mathrm{SFR}/\mathrm{M_\odot \,yr}^{-1})$',
    
    'SubhaloStarMetallicity': r'$\log( Z_\star / Z_\odot)$',

    'GasMass_Inflow_In_Rhpkpc':  r'$\log(M_{\mathrm{gas,\;}r < 2r_{1/2}}/\mathrm{M_\odot})$',
    'GasMass_Outflow_In_Rhpkpc':  r'$\log(M_{\mathrm{gas,\;}r < 2r_{1/2}}/\mathrm{M_\odot})$',
    'GasMass_Inflow_Above_Rhpkpc':  r'$\log(M_{\mathrm{gas,\;}r > 2r_{1/2}}/\mathrm{M_\odot})$',
    'GasMass_Outflow_Above_Rhpkpc':  r'$\log(M_{\mathrm{gas,\;}r > 2r_{1/2}}/\mathrm{M_\odot})$',
    
    #Fracs
    'GasFrac_99': r'$(M_\mathrm{gas}/ M)_{z = 0}$',
    'StarFrac_99': r'$(M_\star/ M)_{z = 0}$',
    'DMFrac_99': r'$(M_\mathrm{DM}/ M)_{z = 0}$',
    'fEx_at_99': r'$(\mathrm{M}_{\star, \mathrm{ex-situ}} / \mathrm{M}_{\star, \mathrm{in-situ}})_{z = 0}$',
    'Ex_at_99': r'$\log(\mathrm{M}_{\star, \mathrm{ex-situ}} / M_\odot)_{z = 0}$',

    'Mgas_Norm_Max':  r'$(M/ M_\mathrm{max})_\mathrm{gas}$',
    'MDM_Norm_Max':  r'$(M/ M_\mathrm{max})_\mathrm{DM}$', 
    'Mstar_Norm_Max':r'$(M/ M_\mathrm{max})_\star$', 
    
    # Group
    'Group_M_Crit200': r'$M_{200}$',
    'Group_M_Crit200FirstGroup': r'$M_{200, \mathrm{first\; host}}$',
    'Group_M_Crit200FinalGroup': r'$M_{200}$',
    
    'logM200_99':  r'$M_{200}$',

    'M200': r'$\log(M_{200}/\mathrm{M}_\odot)$',
    'R200': r'$\log(R/\mathrm{kpc})$',
    'l200': r'$\lambda_{200}$',
    'lamAtFirst': r'$\lambda_{200,\; \mathrm{Birth}}$',
    'Lambda_at_99': r'$\lambda_{200,\; z = 0}$',
    'l200_at_Birth': r'$\lambda_{200,\; \mathrm{Birth}}$',
    
    'Ex_after_05Gyr_Birth': r'$\Delta (\log {M}_{\star,\; \mathrm{ex-situ}})_\mathrm{0.5 \; Gyr\; After-Birth}$',
    'In_after_05Gyr_Birth': r'$\Delta (\log {M}_{\star,\; \mathrm{in-situ}})_\mathrm{0.5 \; Gyr\; After-Birth}$',
    'l200_after_05Gyr_Birth': r'$\Delta ({\lambda}_{200})_\mathrm{0.5 \; Gyr\; After-Birth}$',
    'Rh_after_05Gyr_Birth': r'$\Delta (\log {r}_{05/2})_\mathrm{0.5 \; Gyr\; After-Birth}$',
      
    'Ex_after_1Gyr_Birth': r'$\Delta (\log {M}_{\star,\; \mathrm{ex-situ}})_\mathrm{1 \; Gyr \; After-Birth}$',
    'In_after_1Gyr_Birth': r'$\Delta (\log {M}_{\star,\; \mathrm{in-situ}})_\mathrm{1 \; Gyr \; After-Birth}$',
    'l200_after_1Gyr_Birth': r'$\Delta ({\lambda}_{200})_\mathrm{1 \; Gyr\;  After-Birth}$',
    'Rh_after_1Gyr_Birth': r'$\Delta (\log {r}_{1/2})_\mathrm{1 \; Gyr\;  After-Birth}$',
    
    'Ex_after_8Gyr_Birth': r'$\Delta (\log {M}_{\star,\; \mathrm{ex-situ}})_\mathrm{8 \; Gyr\;  After-Birth}$',
    'In_after_8Gyr_Birth': r'$\Delta (\log {M}_{\star,\; \mathrm{in-situ}})_\mathrm{8 \; Gyr\;  After-Birth}$',
    'l200_after_8Gyr_Birth': r'$\Delta ({\lambda}_{200})_\mathrm{8 \; Gyr\;  After-Birth}$',
    'Rh_after_8Gyr_Birth': r'$\Delta (\log {r}_{8/2})_\mathrm{8 \; Gyr\;  After-Birth}$',
    
    'Ex_after_2Gyr_Birth': r'$\Delta (\log {M}_{\star,\; \mathrm{ex-situ}})_\mathrm{2 \; Gyr\;  After-Birth}$',
    'In_after_2Gyr_Birth': r'$\Delta (\log {M}_{\star,\; \mathrm{in-situ}})_\mathrm{2 \; Gyr \; After-Birth}$',
    'l200_after_2Gyr_Birth': r'$\Delta ({\lambda}_{200})_\mathrm{2 \; Gyr\;  After-Birth}$',
    'Rh_after_2Gyr_Birth': r'$\Delta (\log {r}_{2/2})_\mathrm{2 \; Gyr\;  After-Birth}$',
    
    'Ex_after_5Gyr_Birth': r'$\Delta (\log {M}_{\star,\; \mathrm{ex-situ}})_\mathrm{5 \; Gyr\;  After-Birth}$',
    'In_after_5Gyr_Birth': r'$\Delta (\log {M}_{\star,\; \mathrm{in-situ}})_\mathrm{5 \; Gyr \; After-Birth}$',
    'l200_after_5Gyr_Birth': r'$\Delta ({\lambda}_{200})_\mathrm{5 \; Gyr\;  After-Birth}$',
    'Rh_after_5Gyr_Birth': r'$\Delta (\log {r}_{5/2})_\mathrm{5 \; Gyr\;  After-Birth}$',


    'Group_R_Crit200': r'$R_{200}$',
    'Group_R_Crit200FirstGroup': r'$R_{200, \mathrm{first\; host}}$',
    'Group_R_Crit200FinalGroup': r'$R_{200, \mathrm{final\; host}}$',

    'Group_M_Crit500': r'$M_{500}$',
    'Group_R_Crit500': r'$R_{500}$',

    'GroupNsubs': r'Number of satellites',
    'GroupNsubsFirstGroup': 'Number of satellites \n First host',
    'GroupNsubsFinalGroup': 'Number of satellites \n Final host',
    'GroupNsubsPriorGroup': r'Number of subhalos',

    #ExSitu Contribution
    'MassExNormalize': r'(M_{\star, \mathrm{ex-situ}} / M_{\star, \mathrm{ex-situ}, z = 0})',
    'MassInNormalize': r'$(M_{\mathrm{in-situ}} / M_{\mathrm{ex-situ},\; z = 0})$',

    'MassExNormalizeAll': r'(M_{\star, \mathrm{ex-situ}} / M_{\star})',
    'StellarMassExSituMinor': r'$M_{\mathrm{ex-situ,\; minor\; merger}}$',
    'StellarMassExSituIntermediate': r'$M_{\mathrm{ex-situ,\; intermediate\; merger}}$',
    'StellarMassExSituMajor': r'$M_{\mathrm{ex-situ,\; major\; merger}}$',
    
    'logStellarMassExSituMinor_99': r'$\log(M_\mathrm{ex-situ}/\mathrm{M}_\odot)_{z = 0}$',
    'logStellarMassExSituIntermediate_99': r'$\log(M_\mathrm{ex-situ}/\mathrm{M}_\odot)_{z = 0}$',
    'logStellarMassExSituMajor_99': r'$\log(M_\mathrm{ex-situ}/\mathrm{M}_\odot)_{z = 0}$',




    'LBTimeMajorMerger': 'Lookback time \n Major Merger [Gyr]',
    'LBTimeMinorMerger': 'Lookback time \n  Minor Merger [Gyr]',
    'LBTimeIntermediateMerger': 'Lookback time \n Intermediate Merger [Gyr]',

    'ExMassType0Evolution': r'$M_{\mathrm{gas, \; ex-situ}}$',
    'ExMassType1Evolution': r'$M_{\mathrm{DM, \; ex-situ}}$',
    'ExMassType4Evolution': r'$M_{\mathrm{\star, \; ex-situ}}$',

    #Orbits
    'r_over_R_Crit200': r'$R/R_{200}$',
    'tsincebirth': r'$t - t_\mathrm{birth}$',
    'rOverR200Min': r'$(R/R_{200})_\mathrm{min}$',
    'zInfall': r'$z$ of first infall',
    'GroupM200_99': r'$\log(M_{200}/\mathrm{M}_\odot)_{z = 0}$',
    'zInfall_New': r'$z$ of first infall',

    #Others:
    'MassTensorEigenVals': r'$\mu_1 / \sqrt{\mu_2 \mu_3}$',
    'logjProfile': r'$\log (j_{\mathrm{gas}} / \, \, [\mathrm{kpc \; km  \; s^{-1}}])$',
    'rToRNearYoung': r'$d_{\mathrm{NNB}}$ [kpc]',
    
    'sSFRCoreRatioAfterz5': r'$\overline{(\mathrm{sSFR}_{r < r_{1/2}} / \mathrm{sSFR}_{r > r_{1/2}})}_{z < 5}$',
    'MassIn_Infall_to_GasLost': r'$(\Delta M_\star)_{\mathrm{inner}}^\mathrm{entry-to-gas-loss} / M_\star^\mathrm{entry}$', #'Relative inner stellar mass \n change during period', #
    'MassAboveAfterInfall_Lost': r'$(\Delta M_\star)_{\mathrm{outer}} ^\mathrm{no-gas} / M_{\star}^{\mathrm{gas-loss}}$', #r'$(\Delta M_\star)_{r > r_{1/2, z = 0},  M_\mathrm{gas, \, min} \mathrm{\, to \,} z = 0} / M_{\star, M_\mathrm{gas, \, min} }$',
    'MassAboveAfter_Infall_to_GasLost': r'$(\Delta M_\star)_{\mathrm{outer}}^\mathrm{entry-to-gas-loss} / M_\star^{\mathrm{entry}}$', #'Relative inner stellar mass \n change during period', #
    'DecreaseBetweenGasStar_Over_starFinal': r'$(\Delta \log r_{1/2})^{\mathrm{no-gas}} - (\Delta \log r_{1/2})^\mathrm{entry-to-gas-loss}$',
    'SnapLostGas': 'Gas loss lookback time  [Gyr]', #$M_\mathrm{gas} = 0$',
    'Rhalf_MaxProfile_Minus_HalfRadstar_99': r'$r_{1/2,\; \mathrm{sf}} - r_{1/2, z = 0}$',
    'Rhalf_MinProfile_Minus_HalfRadstar_99': r'$r_{1/2,\; \mathrm{ts}} - r_{1/2, z = 0}$',

    'Rhalf_MaxProfile_Minus_HalfRadstar_Entry': r'$r_{1/2,\; \mathrm{sf}} - r_{1/2, z_\mathrm{entry}}$',
    'Rhalf_MinProfile_Minus_HalfRadstar_Entry': r'$r_{1/2,\; \mathrm{ts}} - r_{1/2, z_\mathrm{entry}}$',

    'Relative_Rhalf_MaxProfile_Minus_HalfRadstar_Entry': r'$(r_{1/2,\; \mathrm{sf}} - r_{1/2, z_\mathrm{entry}})/  r_{1/2, z_\mathrm{entry}}$',
    'Relative_Rhalf_MinProfile_Minus_HalfRadstar_Entry': r'$(r_{1/2,\; \mathrm{ts}} - r_{1/2, z_\mathrm{entry}}) / r_{1/2, z_\mathrm{entry}}$',

    'Rhalf_MaxProfile_Over_HalfRadstar_99': r'$r_{1/2,\; \mathrm{sf}} / r_{1/2, z = 0}$',  
    'Rhalf_MinProfile_Over_HalfRadstar_99': r'$r_{1/2,\; \mathrm{ts}} / r_{1/2, z = 0}$',
    'StarMass_Above_Normalize_99': r'$(M_\star / M_{\star, \mathrm{max}})_{\mathrm{outer}, z = 0}$', #'Relative inner stellar mass \n change during period', #
    'StarMass_In_Normalize_99': r'$(M_\star / M_{\star, \mathrm{max}})_{\mathrm{inner}, z = 0}$', #'Relative inner stellar mass \n change during period', #
    
    'DeltaStarMass_Above_Normalize_99': r'$(M_{\star, \mathrm{max}} - M_{\star, \mathrm{z \;=\; 0}} )_{\mathrm{outer}} /  M_{\star, \mathrm{z \;=\; 0}}$', #'Relative inner stellar mass \n change during period', #
    'DeltaStarMass_In_Normalize_99':  r'$(M_{\star, \mathrm{max}} - M_{\star, \mathrm{z \;=\; 0}} )_{\mathrm{inner}} /  M_{\star, \mathrm{z \;=\; 0}}$', #'Relative inner stellar mass \n change during period', #
    'DeltaMassIn_Infall_In_Normalize_99':r'$(\Delta M_\star)_{\mathrm{inner}}^\mathrm{entry-to-gas-loss} /  M_{\star, \mathrm{z \;=\; 0}}$', #'Relative inner stellar mass \n change during period', #

    'deltaInnersSFR_afterEntry': r'$(\mathrm{sSFR}_{\mathrm{max,\;entry-to-gas-loss}}/\mathrm{sSFR}_{\mathrm{entry}})_{r < r_{1/2, z = 0}}$',
    'deltaInnersSFR_afterEntry_all': r'$(\overline{\mathrm{sSFR}_{\mathrm{entry-to-gas-loss}}/\mathrm{sSFR}_{\mathrm{entry}}})_{3 - max,\; r < 2 r_{1/2, z = 0}}$',
    'deltaInnersSFR_afterEntry_all_EntryRh': r'$(\overline{\mathrm{sSFR}_{\mathrm{entry-to-gas-loss}}/\mathrm{sSFR}_{\mathrm{entry}}})_{3 - max,\; r < r_{1/2,\mathrm{entry}}}$',

    'MassStarIn_Over_Above_absolutevalue':r'$((\Delta M_\star)_{\mathrm{inner}} / (\Delta M_\star)_{\mathrm{outer}})^\mathrm{entry-to-gas-loss} $',
    'BeforesSFR_Entry': r'$\overline{\log{(\mathrm{sSFR}}/\mathrm{yr^{-1}})}_{r < 2r_{1/2, z = 0}}^{\mathrm{entry}}$',
    'InnersSFR_Entry_to_quench': r'$\overline{\log{(\mathrm{sSFR}}/\mathrm{yr^{-1}})}_{r < 2r_{1/2, z = 0}}^{\mathrm{entry-to-gas-loss}}$',
    
    'sSFRCoreRatioAfterz5': r'$\overline{(\mathrm{sSFR}_{r < r_{1/2}} / \mathrm{sSFR}_{r > r_{1/2}})}_{z < 5}$',
    'sSFRinHalfRadAfterz5': r'$\overline{\log{(\mathrm{sSFR}_{r < r_{1/2}}/\mathrm{yr^{-1}})}}_{z < 5}$',
    
    'Decrease_Entry_To_NoGas': r'$(\Delta r_{1/2})^\mathrm{entry-to-gas-loss}$',
    'Decrease_NoGas_To_Final': r'$(\Delta r_{1/2})^\mathrm{no-gas}$',
    'Norbit_Entry_To_NoGas': r'$\mathrm{N_{orb}}^\mathrm{entry-to-gas-loss}$',

    #Profile
    'RadVelocity':  r'$v_\mathrm{r} (r) \, \, [\mathrm{km \, s}^{-1}]$',
    'j': r'$j_{\mathrm{gas}} (r)  \; \, \, [\mathrm{km \; kpc \; s^{-1}}]$',
    'RadVelocityPartType0':  r'$v_\mathrm{r, \; gas} (r) \, \, [\mathrm{km \, s}^{-1}]$',
    'RadVelocityPartType4':  r'$v_\mathrm{r, \; \star} (r) \, \, [\mathrm{km \, s}^{-1}]$',
    'jPartType0': r'$j_{\mathrm{gas}} (r)  \; \, \, [\mathrm{km \; kpc \; s^{-1}}]$',
    'jPartType4': r'$j_{\star} (r)  \; \, \, [\mathrm{km \; kpc \; s^{-1}}]$',
    'DensityGas': r'$\rho_{\mathrm{gas}} (r)  \; \, \, [\mathrm{M_\odot  \; kpc^{-3}}]$',
    'SFR': r'$\mathrm{SFR} \, \, [\mathrm{M_\odot \; yr^{-1}}]$',
    'DensityStar': r'$\rho_{\star} (r)  \; \, \, [\mathrm{M_\odot  \; kpc^{-3}}]$',

    'joverR': r'$j (r) / r  \; \, \, [\mathrm{km  \; s^{-1}}]$',
    'joverRPartType0': r'$j_{\mathrm{gas}} (r) / r  \; \, \, [\mathrm{km  \; s^{-1}}]$',
    'joverRPartType4': r'$j_{\star} (r) / r  \; \, \, [\mathrm{km  \; s^{-1}}]$',
    'DensityGasOverR2':  r'$\rho_{\mathrm{gas}} (r)  r^2 \; \, \, [\mathrm{M_\odot  \; kpc^{-1}}]$',
    'sSFR': r'$\mathrm{sSFR} \, \, [\mathrm{M_\odot \; yr^{-1}}]$',
    'DensityStarOverR2':  r'$\rho_{\star} (r)  r^2 \; \, \, [\mathrm{M_\odot  \; kpc^{-1}}]$',

    'rad': r'$r \, \, [\mathrm{kpc}]$',
    'deltaStarInAbove':  r'$(M_\mathrm{\star,\; r > 5 \; kpc} - M_\mathrm{\star,\; r < 2 \; kpc}) / M_\star$',

    #Scatter
    'z_Birth': r'$z_\mathrm{birth}$',
    'DMFrac_Birth': r'$(M_\mathrm{DM}/M)_\mathrm{birth}$',
    
    'nn2_distance': '$r_{nn} [\mathrm{kpc}]$',
    'nn5_distance': '$r_{nn} [\mathrm{kpc}]$',
    'nn10_distance': '$r_{nn} [\mathrm{kpc}]$',
    
    'nn2_distance_massive': '$r_{nn, \mathrm{massive}} [\mathrm{kpc}]$',
    'nn5_distance_massive': '$r_{nn, \mathrm{massive}} [\mathrm{kpc}]$',
    'nn10_distance_massive': '$r_{nn, \mathrm{massive}} [\mathrm{kpc}]$',
    
    
    'N_aper_1_Mpc':  '$N \mathrm{neighbours}$',
    'N_aper_2_Mpc': '$N \mathrm{neighbours}$',
    'N_aper_5_Mpc':  '$N \mathrm{neighbours}$',
    
    'N_aper_1_Mpc_massive':  '$N \mathrm{massive \, neighbours}$',
    'N_aper_2_Mpc_massive':  '$N \mathrm{massive \,neighbours}$',
    'N_aper_5_Mpc_massive': '$N \mathrm{massive \, neighbours}$',

    # PAPERII
    'M200Mean': r'$\overline{\log(M_{200} / \mathrm{M}_\odot)}$',
    'rOverR200Mean': r'$\overline{(\mathrm{R} / \mathrm{R}_{200})}$',
    'rOverR200Mean_New': r'$\overline{(\mathrm{R} / \mathrm{R}_{200})}$',

    'LBTime_Loss_Gass': 'Gas loss lookback time  [Gyr]',

    'z_At_FinalEntry':  r'$z_{\mathrm{infall}}$ in final host',
    'z_At_FirstEntry':  r'$z_{\mathrm{infall}}$ in first host',
    'deltaFirst_to_Final':  r'$\Delta \mathrm{t}_{\mathrm{first-to-final}}$ [Gyr]',



    None: 'Any'
}

labelsequal = {
    #Subhalo
    'logHalfRadstar_99': r'$\log(r_{1/2, \, z = 0}/\mathrm{kpc})$',
    'SubhaloHalfmassRadType0': r'$\log(r_{1/2}/\mathrm{kpc})$',
    'SubhaloHalfmassRadType1': r'$\log(r_{1/2}/\mathrm{kpc})$',
    'SubhaloHalfmassRadType4': r'$\log(r_{1/2}/\mathrm{kpc})$',
    
    'CompareSubhaoHalfmassRadType1': r'$r_{1/2}/r_{\mathrm{entry}}$',
    'CompareSubhaoHalfmassRadType4': r'$r_{1/2}/r_{\mathrm{entry}}$',

     'l200_NewMeanAfter1Gyr': r'$\lambda_{200}$',
     'l200_NewMeanAfter5Gyr': r'$\lambda_{200}$',
     'l200_NewMeanAfter8Gyr': r'$\lambda_{200}$',

     'l200_New_at_99': r'$\lambda_{200}$',
     
     'CumulativeCorotateFraction_at_1': r'$f_{\mathrm{acc,\, Corotate}}$',
     'CumulativeCorotateFraction_at_5': r'$f_{\mathrm{acc,\, Corotate}}$',
     'CumulativeCorotateFraction_at_8': r'$f_{\mathrm{acc,\, Corotate}}$',

     'CumulativeCorotateFraction': r'$f_{\mathrm{acc,\, Corotate}}$',

    'SubhaloMassInRadType0': r'$\log(M_{r < 2 r_{1/2}}/\mathrm{M}_\odot)$',
    'SubhaloMassInRadType1': r'$\log(M_{r < 2 r_{1/2}}/\mathrm{M}_\odot)$',
    'SubhaloMassInRadType4': r'$\log(M_{r < 2 r_{1/2}}/\mathrm{M}_\odot)$',

    'SubhaloMassType0': r'$\log(M/\mathrm{M}_\odot)$',
    'SubhaloMassType1': r'$\log(M/\mathrm{M}_\odot)$',
    'SubhaloMassType4': r'$\log(M/\mathrm{M}_\odot)$',

    'SubhalosSFRInHalfRad': r'$\log(\mathrm{sSFR}/\mathrm{yr}^{-1})$',
    'sSFR_Outer':  r'$\log(\mathrm{sSFR}/\mathrm{yr}^{-1})$',
    'SubhalosSFRwithinHalfandRad':r'$\log(\mathrm{sSFR}/\mathrm{yr}^{-1})$',
    'SubhalosSFRinRad': r'$\log(\mathrm{sSFR}/\mathrm{yr}^{-1})$',
    'SubhalosSFR': r'$\log(\mathrm{sSFR}/\mathrm{yr}^{-1})$',
    'SubhaloSFR': r'$\log(\mathrm{SFR}/\mathrm{M_\odot\; yr}^{-1})$',
    
    'SubhaloSFRinRad': r'$\log(\mathrm{SFR}/\mathrm{\mathrm{M}_\odot\, yr}^{-1})$',
    'SubhaloSFRouterRad': r'$\log(\mathrm{SFR}/\mathrm{\mathrm{M}_\odot\, yr}^{-1})$',

    'StellarMassExSitu':  r'$\log(M_{\star}/\mathrm{M_\odot})$',
    'StellarMassInSitu':  r'$\log(M_{\star}/\mathrm{M_\odot})$',
    
    'SubhaloSpin': r'$\log(j /\mathrm{km \; kpc \; s^{-1}})$',
    
    'J200': r'$\log(j_{200} /\mathrm{km \; kpc \; s^{-1}})$',
    
    'Ex_after_05Gyr_Birth': r'$\Delta (\log {M}_{\star,\; \mathrm{ex-situ}})_\mathrm{0.5 \; Gyr\; After-Birth}$',
    'In_after_05Gyr_Birth': r'$\Delta (\log {M}_{\star,\; \mathrm{in-situ}})_\mathrm{0.5 \; Gyr\; After-Birth}$',
    'l200_after_05Gyr_Birth': r'$\Delta ({\lambda}_{200})_\mathrm{0.5 \; Gyr\; After-Birth}$',
    'Rh_after_05Gyr_Birth': r'$\Delta (\log {r}_{05/2})_\mathrm{0.5 \; Gyr\; After-Birth}$',
      
    'Ex_after_1Gyr_Birth': r'$\Delta (\log {M}_{\star,\; \mathrm{ex-situ}})_\mathrm{1 \; Gyr\; After-Birth}$',
    'In_after_1Gyr_Birth': r'$\Delta (\log {M}_{\star,\; \mathrm{in-situ}})_\mathrm{1 \; Gyr\; After-Birth}$',
    'l200_after_1Gyr_Birth': r'$\Delta ({\lambda}_{200})_\mathrm{1 \; Gyr\; After-Birth}$',
    'Rh_after_1Gyr_Birth': r'$\Delta (\log {r}_{1/2})_\mathrm{1 \; Gyr\; After-Birth}$',

     'Ex_after_8Gyr_Birth': r'$\Delta (\log {M}_{\star,\; \mathrm{ex-situ}})_\mathrm{8 \; Gyr\;  After-Birth}$',
     'In_after_8Gyr_Birth': r'$\Delta (\log {M}_{\star,\; \mathrm{in-situ}})_\mathrm{8 \; Gyr\;  After-Birth}$',
     'l200_after_8Gyr_Birth': r'$\Delta ({\lambda}_{200})_\mathrm{8 \; Gyr\;  After-Birth}$',
     'Rh_after_8Gyr_Birth': r'$\Delta (\log {r}_{8/2})_\mathrm{8 \; Gyr\;  After-Birth}$',
     
     'Ex_after_2Gyr_Birth': r'$\Delta (\log {M}_{\star,\; \mathrm{ex-situ}})_\mathrm{2 \; Gyr\;  After-Birth}$',
     'In_after_2Gyr_Birth': r'$\Delta (\log {M}_{\star,\; \mathrm{in-situ}})_\mathrm{2 \; Gyr \; After-Birth}$',
     'l200_after_2Gyr_Birth': r'$\Delta ({\lambda}_{200})_\mathrm{2 \; Gyr\;  After-Birth}$',
     'Rh_after_2Gyr_Birth': r'$\Delta (\log {r}_{2/2})_\mathrm{2 \; Gyr\;  After-Birth}$',
     
     'Ex_after_5Gyr_Birth': r'$\Delta (\log {M}_{\star,\; \mathrm{ex-situ}})_\mathrm{5 \; Gyr\;  After-Birth}$',
     'In_after_5Gyr_Birth': r'$\Delta (\log {M}_{\star,\; \mathrm{in-situ}})_\mathrm{5 \; Gyr \; After-Birth}$',
     'l200_after_5Gyr_Birth': r'$\Delta ({\lambda}_{200})_\mathrm{5 \; Gyr\;  After-Birth}$',
     'Rh_after_5Gyr_Birth': r'$\Delta (\log {r}_{5/2})_\mathrm{5 \; Gyr\;  After-Birth}$',

    'sSFRCoreRatio': r'$\mathrm{sSFR}_{r < r_{1/2}} / \mathrm{sSFR}_{r > r_{1/2}}$',
    'SigmasSFRRatio': r'$\Sigma \mathrm{sSFR}_{r < r_{1/2}} / \Sigma  \mathrm{sSFR}_{r > r_{1/2}}$',

    'SubhaloStellarMass_in_Rhpkpc': r'$\log(M_{\star}/\mathrm{M_\odot})$',
    
    'SubhaloStellarMass_Above_Rhpkpc': r'$\log(M_{\star}/\mathrm{M_\odot})$',

    'DMMass_In_Rhpkpc':  r'$\log(M_{\mathrm{DM}}/\mathrm{M_\odot})$',
    'DMMass_Above_Rhpkpc':  r'$\log(M_{\mathrm{DM}}/\mathrm{M_\odot})$',
    
    'GasMass_In_Rhpkpc':  r'$\log(M_{\mathrm{gas}}/\mathrm{M_\odot})$',
    'GasMass_Above_Rhpkpc':  r'$\log(M_{\mathrm{gas}}/\mathrm{M_\odot})$',
    'GasMass_In_TrueRhpkpc':  r'$\log(M_{\mathrm{gas}}/\mathrm{M_\odot})$',
    'GasMass_Above_TrueRhpkpc':  r'$\log(M_{\mathrm{gas}}/\mathrm{M_\odot})$',
    'StarMass_In_Rhpkpc':  r'$\log(M_{\star}/\mathrm{M_\odot})$',
    'StarMass_Above_Rhpkpc':  r'$\log(M_{\star}/\mathrm{M_\odot})$',
    
    'logStar_GFM_Metallicity_In_Rhpkpc': r'$\log( Z_\star / Z_\odot)$',
    'logStar_GFM_Metallicity_Above_Rhpkpc': r'$\log( Z_\star / Z_\odot)$',



    'GasMassInflow_In_Rhpkpc': r'$\log(M_{\mathrm{gas,\, inflow}}/\mathrm{M_\odot})$',
    'GasMassInflow_Above_Rhpkpc':r'$\log(M_{\mathrm{gas,\, inflow}}/\mathrm{M_\odot})$',
    'SFGasMass_in_Rhpkpc':  r'$\log(M_{\mathrm{sf-gas}}/\mathrm{M_\odot})$',
    'SFGasMass_Above_Rhpkpc':  r'$\log(M_{\mathrm{sf-gas}}/\mathrm{M_\odot})$',
    
    'GasMass_in_07Rhpkpc':  r'$\log(M_{\mathrm{gas}}/\mathrm{M_\odot})$',
    'GasMass_07_r_14Rhpkpc':  r'$\log(M_{\mathrm{gas}}/\mathrm{M_\odot})$',
    'GasMass_14_r_21Rhpkpc':  r'$\log(M_{\mathrm{gas}}/\mathrm{M_\odot})$',
    'GasMass_Above_21Rhpkpc':  r'$\log(M_{\mathrm{gas}}/\mathrm{M_\odot})$',
    
    'GasMass_Inflow_in_07Rhpkpc':  r'$\log(M_{\mathrm{gas,\;}v_\mathrm{rad} < 0}/\mathrm{M_\odot})$',
    'GasMass_Inflow_07_r_14Rhpkpc':  r'$\log(M_{\mathrm{gas,\;}v_\mathrm{rad} < 0}/\mathrm{M_\odot})$',
    'GasMass_Inflow_14_r_21Rhpkpc':  r'$\log(M_{\mathrm{gas,\;}v_\mathrm{rad} < 0}/\mathrm{M_\odot})$',
    'GasMass_Inflow_Above_21Rhpkpc':   r'$\log(M_{\mathrm{gas,\;}v_\mathrm{rad} < 0}/\mathrm{M_\odot})$',
    
    'sSFR_In_Rhpkpc': r'$\log(\mathrm{sSFR}/\mathrm{yr}^{-1})$',
    'sSFR_Above_Rhpkpc': r'$\log(\mathrm{sSFR}/\mathrm{yr}^{-1})$',
    'sSFR_In_TrueRhpkpc': r'$\log(\mathrm{sSFR}/\mathrm{yr}^{-1})$',
    'sSFR_Above_TrueRhpkpc': r'$\log(\mathrm{sSFR}/\mathrm{yr}^{-1})$',
    'SFR_In_Rhpkpc': r'$\log(\mathrm{SFR}/\mathrm{M_\odot \,yr}^{-1})$',
    'SFR_Above_Rhpkpc': r'$\log(\mathrm{SFR}/\mathrm{M_\odot \,yr}^{-1})$',
    
    'SubhaloStarMetallicity': r'$\log( Z_\star / Z_\odot)$',

    'GasMass_Inflow_In_Rhpkpc':  r'$\log(M_{\mathrm{gas,\;}r < 2r_{1/2}}/\mathrm{M_\odot})$',
    'GasMass_Outflow_In_Rhpkpc':  r'$\log(M_{\mathrm{gas,\;}r < 2r_{1/2}}/\mathrm{M_\odot})$',
    'GasMass_Inflow_Above_Rhpkpc':  r'$\log(M_{\mathrm{gas,\;}r > 2r_{1/2}}/\mathrm{M_\odot})$',
    'GasMass_Outflow_Above_Rhpkpc':  r'$\log(M_{\mathrm{gas,\;}r > 2r_{1/2}}/\mathrm{M_\odot})$',
    
    'StarMass_In_MultRhpkpc_plus000dex': r'$\log(M_{\star}/\mathrm{M_\odot})$',
    'StarMass_In_MultRhpkpc_plus015dex': r'$\log(M_{\star}/\mathrm{M_\odot})$',
    'StarMass_In_MultRhpkpc_plus025dex': r'$\log(M_{\star}/\mathrm{M_\odot})$',
    'StarMass_In_MultRhpkpc_plus050dex': r'$\log(M_{\star}/\mathrm{M_\odot})$', 
    'StarMass_In_MultRhpkpc_plus075dex': r'$\log(M_{\star}/\mathrm{M_\odot})$', 

    'Starvrad_In_MultRhpkpc_plus000dex':  r'$v_{\mathrm{rad,\; \star}} [\mathrm{km \, s^{-1}}]$',
    'Starvrad_In_MultRhpkpc_plus015dex': r'$v_{\mathrm{rad,\; \star}} [\mathrm{km \, s^{-1}}]$',
    'Starvrad_In_MultRhpkpc_plus025dex': r'$v_{\mathrm{rad,\; \star}} [\mathrm{km \, s^{-1}}]$',
    'Starvrad_In_MultRhpkpc_plus050dex': r'$v_{\mathrm{rad,\; \star}} [\mathrm{km \, s^{-1}}]$',
    'Starvrad_In_MultRhpkpc_plus075dex': r'$v_{\mathrm{rad,\; \star}} [\mathrm{km \, s^{-1}}]$',

    'GasMass_In_MultRhpkpc_plus000dex':  r'$\log(M_{\mathrm{gas}}/\mathrm{M_\odot})$',
    'GasMass_In_MultRhpkpc_plus015dex':  r'$\log(M_{\mathrm{gas}}/\mathrm{M_\odot})$',
    'GasMass_In_MultRhpkpc_plus025dex':  r'$\log(M_{\mathrm{gas}}/\mathrm{M_\odot})$',
    'GasMass_In_MultRhpkpc_plus050dex':  r'$\log(M_{\mathrm{gas}}/\mathrm{M_\odot})$',
    'GasMass_In_MultRhpkpc_plus075dex':  r'$\log(M_{\mathrm{gas}}/\mathrm{M_\odot})$',

    'Gasvrad_In_MultRhpkpc_plus000dex': r'$v_{\mathrm{rad,\; gas}} [\mathrm{km \, s^{-1}}]$',
    'Gasvrad_In_MultRhpkpc_plus015dex': r'$v_{\mathrm{rad,\; gas}} [\mathrm{km \, s^{-1}}]$',
    'Gasvrad_In_MultRhpkpc_plus025dex': r'$v_{\mathrm{rad,\; gas}} [\mathrm{km \, s^{-1}}]$',
    'Gasvrad_In_MultRhpkpc_plus050dex': r'$v_{\mathrm{rad,\; gas}} [\mathrm{km \, s^{-1}}]$',
    'Gasvrad_In_MultRhpkpc_plus075dex': r'$v_{\mathrm{rad,\; gas}} [\mathrm{km \, s^{-1}}]$',

    'GasMassInflow_In_MultRhpkpc_plus000dex':   r'$\log(M_{\mathrm{gas,\; inflow}}/\mathrm{M_\odot})$',
    'GasMassInflow_In_MultRhpkpc_plus015dex': r'$\log(M_{\mathrm{gas,\; inflow}}/\mathrm{M_\odot})$',
    'GasMassInflow_In_MultRhpkpc_plus025dex': r'$\log(M_{\mathrm{gas,\; inflow}}/\mathrm{M_\odot})$',
    'GasMassInflow_In_MultRhpkpc_plus050dex': r'$\log(M_{\mathrm{gas,\; inflow}}/\mathrm{M_\odot})$',
    'GasMassInflow_In_MultRhpkpc_plus075dex': r'$\log(M_{\mathrm{gas,\; inflow}}/\mathrm{M_\odot})$',

    'sSFR_In_MultRhpkpc_plus000dex':  r'$\log(\mathrm{sSFR}/\mathrm{yr}^{-1})$',
    'sSFR_In_MultRhpkpc_plus015dex':r'$\log(\mathrm{sSFR}/\mathrm{yr}^{-1})$',
    'sSFR_In_MultRhpkpc_plus025dex': r'$\log(\mathrm{sSFR}/\mathrm{yr}^{-1})$',
    'sSFR_In_MultRhpkpc_plus050dex': r'$\log(\mathrm{sSFR}/\mathrm{yr}^{-1})$',
    'sSFR_In_MultRhpkpc_plus075dex': r'$\log(\mathrm{sSFR}/\mathrm{yr}^{-1})$',

    'SFR_In_MultRhpkpc_plus000dex':  r'$\log(\mathrm{SFR}/\mathrm{M_\odot \; yr^{-1}})$',
    'SFR_In_MultRhpkpc_plus015dex': r'$\log(\mathrm{SFR}/\mathrm{M_\odot \; yr^{-1}})$',
    'SFR_In_MultRhpkpc_plus025dex': r'$\log(\mathrm{SFR}/\mathrm{M_\odot \; yr^{-1}})$',
    'SFR_In_MultRhpkpc_plus050dex': r'$\log(\mathrm{SFR}/\mathrm{M_\odot \; yr^{-1}})$',
    'SFR_In_MultRhpkpc_plus075dex': r'$\log(\mathrm{SFR}/\mathrm{M_\odot \; yr^{-1}})$',
    
   
    'StarMass_In_Rhpkpc_entry_minus200dex':  r'$\log(M_{\star}/\mathrm{M_\odot})$',
    'StarMass_In_Rhpkpc_entry_minus100dex': r'$\log(M_{\star}/\mathrm{M_\odot})$',
    'StarMass_In_Rhpkpc_entry_minus150dex':  r'$\log(M_{\star}/\mathrm{M_\odot})$',
    'StarMass_In_Rhpkpc_entry_plus100dex': r'$\log(M_{\star}/\mathrm{M_\odot})$',
    
    'GasMass_In_Rhpkpc_entry_minus200dex': r'$\log(M_{\mathrm{gas}}/\mathrm{M_\odot})$',
    'GasMass_In_Rhpkpc_entry_minus100dex': r'$\log(M_{\mathrm{gas}}/\mathrm{M_\odot})$',
    'GasMass_In_Rhpkpc_entry_minus150dex': r'$\log(M_{\mathrm{gas}}/\mathrm{M_\odot})$',
    'GasMass_In_Rhpkpc_entry_plus100dex': r'$\log(M_{\mathrm{gas}}/\mathrm{M_\odot})$',
    
    'Starvrad_In_Rhpkpc_entry_minus200dex': r'$v_{\mathrm{rad,\; \star}} [\mathrm{km \, s^{-1}}]$',
    'Starvrad_In_Rhpkpc_entry_minus100dex': r'$v_{\mathrm{rad,\; \star}} [\mathrm{km \, s^{-1}}]$',
    'Starvrad_In_Rhpkpc_entry_minus150dex': r'$v_{\mathrm{rad,\; \star}} [\mathrm{km \, s^{-1}}]$',
    'Starvrad_In_Rhpkpc_entry_plus100dex': r'$v_{\mathrm{rad,\; \star}} [\mathrm{km \, s^{-1}}]$',

    'Gasvrad_In_Rhpkpc_entry_minus200dex': r'$v_{\mathrm{rad,\; gas}} [\mathrm{km \, s^{-1}}]$',
    'Gasvrad_In_Rhpkpc_entry_minus100dex':r'$v_{\mathrm{rad,\; gas}} [\mathrm{km \, s^{-1}}]$',
    'Gasvrad_In_Rhpkpc_entry_minus150dex': r'$v_{\mathrm{rad,\; gas}} [\mathrm{km \, s^{-1}}]$',
    'Gasvrad_In_Rhpkpc_entry_plus100dex': r'$v_{\mathrm{rad,\; gas}} [\mathrm{km \, s^{-1}}]$',

    'GasMassInflow_In_Rhpkpc_entry_minus200dex': r'$\log(M_{\mathrm{gas,\; inflow}}/\mathrm{M_\odot})$',
    'GasMassInflow_In_Rhpkpc_entry_minus100dex': r'$\log(M_{\mathrm{gas,\; inflow}}/\mathrm{M_\odot})$',
    'GasMassInflow_In_Rhpkpc_entry_minus150dex': r'$\log(M_{\mathrm{gas,\; inflow}}/\mathrm{M_\odot})$',
    'GasMassInflow_In_Rhpkpc_entry_plus100dex': r'$\log(M_{\mathrm{gas,\; inflow}}/\mathrm{M_\odot})$',
    
    'sSFR_In_Rhpkpc_entry_minus200dex':  r'$\log(\mathrm{sSFR}/\mathrm{yr}^{-1})$',
    'sSFR_In_Rhpkpc_entry_minus100dex':  r'$\log(\mathrm{sSFR}/\mathrm{yr}^{-1})$',
    'sSFR_In_Rhpkpc_entry_minus150dex':  r'$\log(\mathrm{sSFR}/\mathrm{yr}^{-1})$',
    'sSFR_In_Rhpkpc_entry_plus100dex':  r'$\log(\mathrm{sSFR}/\mathrm{yr}^{-1})$',

    'SFR_In_Rhpkpc_entry_minus200dex': r'$\log(\mathrm{SFR}/\mathrm{M_\odot \; yr^{-1}})$',
    'SFR_In_Rhpkpc_entry_minus100dex': r'$\log(\mathrm{SFR}/\mathrm{M_\odot \; yr^{-1}})$',
    'SFR_In_Rhpkpc_entry_minus150dex': r'$\log(\mathrm{SFR}/\mathrm{M_\odot \; yr^{-1}})$',
    'SFR_In_Rhpkpc_entry_plus100dex': r'$\log(\mathrm{SFR}/\mathrm{M_\odot \; yr^{-1}})$',

     'Starrho_In_Rhpkpc_entry_minus200dex':  r'$\log(\rho_{\star}/\mathrm{M_\odot \; kpc-3})$',
     'Starrho_In_Rhpkpc_entry_minus100dex': r'$\log(M_{\star}/\mathrm{M_\odot})$',
     'Starrho_In_Rhpkpc_entry_minus150dex': r'$\log(M_{\star}/\mathrm{M_\odot})$',
     'Starrho_In_Rhpkpc_entry_plus100dex': r'$\log(M_{\star}/\mathrm{M_\odot})$',
     'Starrho_In_Rhpkpc_entry_minus050dex': r'$\log(M_{\star}/\mathrm{M_\odot})$',
     'Starrho_In_Rhpkpc_entry_minus025dex': r'$\log(M_{\star}/\mathrm{M_\odot})$',
     'Starrho_In_Rhpkpc_entry': r'$\log(M_{\star}/\mathrm{M_\odot})$',
     'Starrho_In_Rhpkpc_entry_plus025dex': r'$\log(M_{\star}/\mathrm{M_\odot})$',
     'Starrho_In_Rhpkpc_entry_plus050dex': r'$\log(M_{\star}/\mathrm{M_\odot})$',
     
     'Gasrho_In_Rhpkpc_entry_minus200dex':  r'$\log(\rho_{\mathrm{gas}}/\mathrm{M_\odot \; kpc-3})$',
     'Gasrho_In_Rhpkpc_entry_minus100dex': r'$\log(M_{\mathrm{gas}}/\mathrm{M_\odot})$',
     'Gasrho_In_Rhpkpc_entry_minus150dex': r'$\log(M_{\mathrm{gas}}/\mathrm{M_\odot})$',
     'Gasrho_In_Rhpkpc_entry_plus100dex': r'$\log(M_{\mathrm{gas}}/\mathrm{M_\odot})$',
     'Gasrho_In_Rhpkpc_entry_minus050dex': r'$\log(M_{\mathrm{gas}}/\mathrm{M_\odot})$',
     'Gasrho_In_Rhpkpc_entry_minus025dex': r'$\log(M_{\mathrm{gas}}/\mathrm{M_\odot})$',
     'Gasrho_In_Rhpkpc_entry': r'$\log(M_{\mathrm{gas}}/\mathrm{M_\odot})$',
     'Gasrho_In_Rhpkpc_entry_plus025dex': r'$\log(M_{\mathrm{gas}}/\mathrm{M_\odot})$',
     'Gasrho_In_Rhpkpc_entry_plus050dex': r'$\log(M_{\mathrm{gas}}/\mathrm{M_\odot})$',
 
    'StarMass_In_Rhpkpc_entry_minus050dex': r'$\log(M_{\star}/\mathrm{M_\odot})$',
    'StarMass_In_Rhpkpc_entry_minus025dex': r'$\log(M_{\star}/\mathrm{M_\odot})$',
    'StarMass_In_Rhpkpc_entry': r'$\log(M_{\star}/\mathrm{M_\odot})$',
    'StarMass_In_Rhpkpc_entry_plus025dex': r'$\log(M_{\star}/\mathrm{M_\odot})$',
    'StarMass_In_Rhpkpc_entry_plus050dex': r'$\log(M_{\star}/\mathrm{M_\odot})$', 
    
    'Starvrad_In_Rhpkpc_entry_minus050dex':  r'$v_{\mathrm{rad,\; \star}} [\mathrm{km \, s^{-1}}]$',
    'Starvrad_In_Rhpkpc_entry_minus025dex': r'$v_{\mathrm{rad,\; \star}} [\mathrm{km \, s^{-1}}]$',
    'Starvrad_In_Rhpkpc_entry': r'$v_{\mathrm{rad,\; \star}} [\mathrm{km \, s^{-1}}]$',
    'Starvrad_In_Rhpkpc_entry_plus025dex': r'$v_{\mathrm{rad,\; \star}} [\mathrm{km \, s^{-1}}]$',
    'Starvrad_In_Rhpkpc_entry_plus050dex': r'$v_{\mathrm{rad,\; \star}} [\mathrm{km \, s^{-1}}]$',
    
    'GasMass_In_Rhpkpc_entry_minus050dex':  r'$\log(M_{\mathrm{gas}}/\mathrm{M_\odot})$',
    'GasMass_In_Rhpkpc_entry_minus025dex':  r'$\log(M_{\mathrm{gas}}/\mathrm{M_\odot})$',
    'GasMass_In_Rhpkpc_entry':  r'$\log(M_{\mathrm{gas}}/\mathrm{M_\odot})$',
    'GasMass_In_Rhpkpc_entry_plus025dex':  r'$\log(M_{\mathrm{gas}}/\mathrm{M_\odot})$',
    'GasMass_In_Rhpkpc_entry_plus050dex':  r'$\log(M_{\mathrm{gas}}/\mathrm{M_\odot})$',
    
    'Gasvrad_In_Rhpkpc_entry_minus050dex': r'$v_{\mathrm{rad,\; gas}} [\mathrm{km \, s^{-1}}]$',
    'Gasvrad_In_Rhpkpc_entry_minus025dex': r'$v_{\mathrm{rad,\; gas}} [\mathrm{km \, s^{-1}}]$',
    'Gasvrad_In_Rhpkpc_entry': r'$v_{\mathrm{rad,\; gas}} [\mathrm{km \, s^{-1}}]$',
    'Gasvrad_In_Rhpkpc_entry_plus025dex': r'$v_{\mathrm{rad,\; gas}} [\mathrm{km \, s^{-1}}]$',
    'Gasvrad_In_Rhpkpc_entry_plus050dex': r'$v_{\mathrm{rad,\; gas}} [\mathrm{km \, s^{-1}}]$',
    
    'GasMassInflow_In_Rhpkpc_entry_minus050dex':   r'$\log(M_{\mathrm{gas,\; inflow}}/\mathrm{M_\odot})$',
    'GasMassInflow_In_Rhpkpc_entry_minus025dex': r'$\log(M_{\mathrm{gas,\; inflow}}/\mathrm{M_\odot})$',
    'GasMassInflow_In_Rhpkpc_entry': r'$\log(M_{\mathrm{gas,\; inflow}}/\mathrm{M_\odot})$',
    'GasMassInflow_In_Rhpkpc_entry_plus025dex': r'$\log(M_{\mathrm{gas,\; inflow}}/\mathrm{M_\odot})$',
    'GasMassInflow_In_Rhpkpc_entry_plus050dex': r'$\log(M_{\mathrm{gas,\; inflow}}/\mathrm{M_\odot})$',
    
    'sSFR_In_Rhpkpc_entry_minus050dex':  r'$\log(\mathrm{sSFR}/\mathrm{yr}^{-1})$',
    'sSFR_In_Rhpkpc_entry_minus025dex':r'$\log(\mathrm{sSFR}/\mathrm{yr}^{-1})$',
    'sSFR_In_Rhpkpc_entry':r'$\log(\mathrm{sSFR}/\mathrm{yr}^{-1})$',
    'sSFR_In_Rhpkpc_entry_plus025dex': r'$\log(\mathrm{sSFR}/\mathrm{yr}^{-1})$',
    'sSFR_In_Rhpkpc_entry_plus050dex': r'$\log(\mathrm{sSFR}/\mathrm{yr}^{-1})$',
    
    'SFR_In_Rhpkpc_entry_minus050dex':  r'$\log(\mathrm{SFR}/\mathrm{M_\odot \; yr^{-1}})$',
    'SFR_In_Rhpkpc_entry_minus025dex': r'$\log(\mathrm{SFR}/\mathrm{M_\odot \; yr^{-1}})$',
    'SFR_In_Rhpkpc_entry': r'$\log(\mathrm{SFR}/\mathrm{M_\odot \; yr^{-1}})$',
    'SFR_In_Rhpkpc_entry_plus025dex': r'$\log(\mathrm{SFR}/\mathrm{M_\odot \; yr^{-1}})$',
    'SFR_In_Rhpkpc_entry_plus050dex': r'$\log(\mathrm{SFR}/\mathrm{M_\odot \; yr^{-1}})$',
    
    
    'StarMass_minus050dex_r_Rhpkpc_entry': r'$\log(M_{\star}/\mathrm{M_\odot})$',
    'StarMass_Rhpkpc_entry_r_plus050dex': r'$\log(M_{\star}/\mathrm{M_\odot})$', 
    'Starvrad_minus050dex_r_Rhpkpc_entry': r'$v_{\mathrm{rad,\; \star}} [\mathrm{km \, s^{-1}}]$',
    'Starvrad_Rhpkpc_entry_r_plus050dexex': r'$v_{\mathrm{rad,\; \star}} [\mathrm{km \, s^{-1}}]$',
    
    'GasMass_minus050dex_r_Rhpkpc_entry':  r'$\log(M_{\mathrm{gas}}/\mathrm{M_\odot})$',
    'GasMass_Rhpkpc_entry_r_plus050dexex':  r'$\log(M_{\mathrm{gas}}/\mathrm{M_\odot})$',
    'Gasvrad_minus050dex_r_Rhpkpc_entry': r'$v_{\mathrm{rad,\; gas}} [\mathrm{km \, s^{-1}}]$',
    'Gasvrad_Rhpkpc_entry_r_plus050dex': r'$v_{\mathrm{rad,\; gas}} [\mathrm{km \, s^{-1}}]$',
    'GasMassInflow_minus050dex_r_Rhpkpc_entry': r'$\log(M_{\mathrm{gas,\; inflow}}/\mathrm{M_\odot})$',
    'GasMassInflow_Rhpkpc_entry_r_plus050dex': r'$\log(M_{\mathrm{gas,\; inflow}}/\mathrm{M_\odot})$',
    
    'sSFR_minus050dex_r_Rhpkpc_entry': r'$\log(\mathrm{sSFR}/\mathrm{yr}^{-1})$',
    'sSFR_Rhpkpc_entry_r_plus050dex': r'$\log(\mathrm{sSFR}/\mathrm{yr}^{-1})$',
    'SFR_minus050dex_r_Rhpkpc_entry': r'$\log(\mathrm{SFR}/\mathrm{M_\odot \; yr^{-1}})$',
    'SFR_Rhpkpc_entry_r_plus050dex': r'$\log(\mathrm{SFR}/\mathrm{M_\odot \; yr^{-1}})$',
    
    #Fracs
    'GasFrac_99': r'$(M_\mathrm{gas}/ M)_{z = 0}$',
    'StarFrac_99': r'$(M_\star/ M)_{z = 0}$',
    'DMFrac_99': r'$(M_\mathrm{DM}/ M)_{z = 0}$',

    'Mgas_Norm_Max':  r'$(M/ M_\mathrm{max})$',
    'MDM_Norm_Max':  r'$(M/ M_\mathrm{max})$', 
    'Mstar_Norm_Max':r'$(M/ M_\mathrm{max})$', 
    
    # Group
    'Group_M_Crit200': r'$\log(M_{200}/\mathrm{M}_\odot)$',
    'Group_M_Crit200FirstGroup': r'$\log(M_{200}/\mathrm{M}_\odot)$',
    'Group_M_Crit200FinalGroup': r'$\log(M_{200}/\mathrm{M}_\odot)$',
    
    'Group_R_Crit200': r'$\log(R/\mathrm{kpc})$',
    'Group_R_Crit200FirstGroup': r'$\log(R/\mathrm{kpc})$',
    'Group_R_Crit200FinalGroup': r'$\log(R/\mathrm{kpc})$',

    'Group_M_Crit500': r'$\log(M/\mathrm{M}_\odot)$',
    'Group_R_Crit500': r'$\log(R/\mathrm{kpc})$',

    'M200': r'$\log(M_{200}/\mathrm{M}_\odot)$',
    'R200': r'$\log(R/\mathrm{kpc})$',
    'l200': r'$\lambda_{200}$',
    'lamMeanAfter1Gyr': r'$\overline{\lambda}_{200,\; \mathrm{1 \; Gyr\; After-Birth}}$',
    'lamMeanAfter2Gyr': r'$\overline{\lambda}_{200,\; \mathrm{2 \; Gyr\; After-Birth}}$',
    'lamAtFirst': r'$\lambda_{200,\; \mathrm{Birth}}$',
    'Lambda_at_99': r'$\lambda_{200,\; z = 0}$', 
    'l200_at_Birth': r'$\lambda_{200,\; \mathrm{Birth}}$',
    'fEx_at_99': r'$(\mathrm{M}_{\star, \mathrm{ex-situ}} / \mathrm{M}_{\star, \mathrm{in-situ}})_{z = 0}$',
    'Ex_at_99': r'$\log(\mathrm{M}_{\star, \mathrm{ex-situ}} / M_\odot)_{z = 0}$',

    'GroupNsubs': r'Number',
    'GroupNsubsFirstGroup': r'Number of satellites',
    'GroupNsubsFinalGroup': r'Number of satellites',
    'GroupNsubsPriorGroup': r'Number of subhalos',

    #ExSitu Contribution
    'MassExNormalize': r'Normalized $M_\mathrm{ex-situ}$',
    'MassInNormalize': r'$(M_{\mathrm{in-situ}} / M_{\mathrm{ex-situ},\; z = 0})$',

    'MassExNormalizeAll': r'Normalized $M_\mathrm{ex-situ}$',
    'StellarMassExSituMinor': r'$\log(M_\mathrm{ex-situ}/\mathrm{M}_\odot)$',
    'StellarMassExSituIntermediate': r'$\log(M_\mathrm{ex-situ}/\mathrm{M}_\odot)$',
    'StellarMassExSituMajor': r'$\log(M_\mathrm{ex-situ}/\mathrm{M}_\odot)$',
    
    'logStellarMassExSituMinor_99': r'$\log(M_\mathrm{ex-situ}/\mathrm{M}_\odot)_{z = 0}$',
    'logStellarMassExSituIntermediate_99': r'$\log(M_\mathrm{ex-situ}/\mathrm{M}_\odot)_{z = 0}$',
    'logStellarMassExSituMajor_99': r'$\log(M_\mathrm{ex-situ}/\mathrm{M}_\odot)_{z = 0}$',



    'LBTimeMajorMerger': 'Lookback time \n [Gyr]',
    'LBTimeMinorMerger': 'Lookback time \n [Gyr]',
    'LBTimeIntermediateMerger': 'Lookback time \n [Gyr]',

    'ExMassType0Evolution': r'$\log(M_{\mathrm{ex-situ}}/\mathrm{M}_\odot)$',
    'ExMassType1Evolution': r'$\log(M_{\mathrm{ex-situ}}/\mathrm{M}_\odot)$',
    'ExMassType4Evolution': r'$\log(M_{\mathrm{ex-situ}}/\mathrm{M}_\odot)$',

    'deltaStarInAbove':  r'$(M_\mathrm{\star,\; r > 5 \; kpc} - M_\mathrm{\star,\; r < 2 \; kpc}) / M_\star$',
    #Orbits
    'r_over_R_Crit200': r'$R/R_{200}$',
    'tsincebirth': r'$t - t_\mathrm{birth}$',
    'rOverR200Min': r'$(R/R_{200})_\mathrm{min}$',
    'zInfall': r'$z$ of first infall',
    'GroupM200_99': r'$\log(M_{200}/\mathrm{M}_\odot)_{z = 0}$',
    'zInfall_New': r'$z$ of first infall',

    #Others:
    'MassTensorEigenVals': r'$\mu_1 / \sqrt{\mu_2 \mu_3}$',
    'logjProfile': r'$\log (j_{\mathrm{gas}} / \, \, [\mathrm{kpc \; km  \; s^{-1}}])$',
    'rToRNearYoung': r'$d_{\mathrm{NNB}}$ [kpc]',
    
    'sSFRCoreRatioAfterz5': r'$\overline{(\mathrm{sSFR}_{r < r_{1/2}} / \mathrm{sSFR}_{r > r_{1/2}})}_{z < 5}$',
    'MassIn_Infall_to_GasLost': r'$(\Delta M_\star)_{\mathrm{inner}}^\mathrm{entry-to-gas-loss} / M_\star^\mathrm{entry}$', #'Relative inner stellar mass \n change during period', #
    'MassAboveAfterInfall_Lost': r'$(\Delta M_\star)_{\mathrm{outer}} ^\mathrm{no-gas} / M_{\star}^{\mathrm{gas-loss}}$', #r'$(\Delta M_\star)_{r > r_{1/2, z = 0},  M_\mathrm{gas, \, min} \mathrm{\, to \,} z = 0} / M_{\star, M_\mathrm{gas, \, min} }$',
    'MassAboveAfter_Infall_to_GasLost': r'$(\Delta M_\star)_{\mathrm{outer}}^\mathrm{entry-to-gas-loss} / M_\star^{\mathrm{entry}}$', #'Relative inner stellar mass \n change during period', #
    'DecreaseBetweenGasStar_Over_starFinal': r'$(\Delta \log r_{1/2})^{\mathrm{no-gas}} - (\Delta \log r_{1/2})^\mathrm{entry-to-gas-loss}$',
    'SnapLostGas': 'Gas loss lookback time  [Gyr]', #$M_\mathrm{gas} = 0$',
    'Rhalf_MaxProfile_Minus_HalfRadstar_99': r'$r_{1/2,\; \mathrm{sf}} - r_{1/2, z = 0}$',
    'Rhalf_MinProfile_Minus_HalfRadstar_99': r'$r_{1/2,\; \mathrm{ts}} - r_{1/2, z = 0}$',

    'Rhalf_MaxProfile_Minus_HalfRadstar_Entry': r'$r_{1/2,\; \mathrm{sf}} - r_{1/2, z_\mathrm{entry}}$',
    'Rhalf_MinProfile_Minus_HalfRadstar_Entry': r'$r_{1/2,\; \mathrm{ts}} - r_{1/2, z_\mathrm{entry}}$',

    'Rhalf_MaxProfile_Over_HalfRadstar_99': r'$r_{1/2,\; \mathrm{sf}} / r_{1/2, z = 0}$',  
    'Rhalf_MinProfile_Over_HalfRadstar_99': r'$r_{1/2,\; \mathrm{ts}} / r_{1/2, z = 0}$',
    
    'StarMass_Above_Normalize_99': r'$(M_\star / M_{\star, \mathrm{max}})_{\mathrm{outer}, z = 0}$', #'Relative inner stellar mass \n change during period', #
    'StarMass_In_Normalize_99': r'$(M_\star / M_{\star, \mathrm{max}})_{\mathrm{inner}, z = 0}$', #'Relative inner stellar mass \n change during period', #
    
     
    'DeltaStarMass_Above_Normalize_99': r'$(M_{\star, \mathrm{max}} - M_{\star, \mathrm{z \;=\; 0}} )_{\mathrm{outer}} /  M_{\star, \mathrm{z \;=\; 0}}$', #'Relative inner stellar mass \n change during period', #
    'DeltaStarMass_In_Normalize_99':  r'$(M_{\star, \mathrm{max}} - M_{\star, \mathrm{z \;=\; 0}} )_{\mathrm{inner}} /  M_{\star, \mathrm{z \;=\; 0}}$', #'Relative inner stellar mass \n change during period', #
    'DeltaMassIn_Infall_In_Normalize_99':r'$(\Delta M_\star)_{\mathrm{inner}}^\mathrm{entry-to-gas-loss} /  M_{\star, \mathrm{z \;=\; 0}}$', #'Relative inner stellar mass \n change during period', #

    'deltaInnersSFR_afterEntry': r'$(\mathrm{sSFR}_{\mathrm{max,\;entry-to-gas-loss}}/\mathrm{sSFR}_{\mathrm{entry}})_{r < r_{1/2, z = 0}}$',
    'deltaInnersSFR_afterEntry_all': r'$(\overline{\mathrm{sSFR}_{\mathrm{entry-to-gas-loss}}}/\overline{\mathrm{sSFR}_{\mathrm{entry}}})_{r < 2 r_{1/2, z = 0}}$',
    'BeforesSFR_Entry': r'$\overline{\log{(\mathrm{sSFR}}/\mathrm{yr^{-1}})}_{r < 2r_{1/2, z = 0}}^{\mathrm{entry}}$',
    'InnersSFR_Entry_to_quench': r'$\overline{\log{(\mathrm{sSFR}}/\mathrm{yr^{-1}})}_{r < 2r_{1/2, z = 0}}^{\mathrm{entry-to-gas-loss}}$',
    
    'MassStarIn_Over_Above_absolutevalue':r'$((\Delta M_\star)_{\mathrm{inner}} / (\Delta M_\star)_{\mathrm{outer}})^\mathrm{entry-to-gas-loss} $',

    'sSFRCoreRatioAfterz5': r'$\overline{(\mathrm{sSFR}_{r < r_{1/2}} / \mathrm{sSFR}_{r > r_{1/2}})}_{z < 5}$',
    'sSFRinHalfRadAfterz5': r'$\overline{\log{(\mathrm{sSFR}_{r < r_{1/2}}/\mathrm{yr^{-1}})}}_{z < 5}$',

    'Decrease_Entry_To_NoGas': r'$(\Delta r_{1/2})^\mathrm{entry-to-gas-loss}$',
    'Decrease_NoGas_To_Final': r'$(\Delta r_{1/2})^\mathrm{no-gas}$',
    
    'Decrease_Entry_To_NoGas_Norm': r'$(\Delta r_{1/2})^\mathrm{entry-to-gas-loss}$',
    'Decrease_NoGas_To_Final_Norm': r'$(\Delta r_{1/2})^\mathrm{no-gas}$',

    'Decrease_Entry_To_NoGas_Norm': r'$(\Delta r_{1/2})^\mathrm{entry-to-gas-loss}$',
    'Decrease_NoGas_To_Final_Norm': r'$(\Delta r_{1/2})^\mathrm{no-gas}$',

    'Norbit_Entry_To_NoGas': r'$\mathrm{N_{orb}}^\mathrm{entry-to-gas-loss}$',

    #Profile
    'RadVelocity':  r'$v_\mathrm{r} (r) \, \, [\mathrm{km \, s}^{-1}]$',
    'j': r'$j_{\mathrm{gas}} (r)  \; \, \, [\mathrm{km \; kpc \; s^{-1}}]$',
    'RadVelocityPartType0':  r'$v_\mathrm{r, \; gas} (r) \, \, [\mathrm{km \, s}^{-1}]$',
    'RadVelocityPartType4':  r'$v_\mathrm{r, \; \star} (r) \, \, [\mathrm{km \, s}^{-1}]$',
    'jPartType0': r'$j_{\mathrm{gas}} (r)  \; \, \, [\mathrm{km \; kpc \; s^{-1}}]$',
    'jPartType4': r'$j_{\star} (r)  \; \, \, [\mathrm{km \; kpc \; s^{-1}}]$',
    'DensityGas': r'$\rho_{\mathrm{gas}} (r)  \; \, \, [\mathrm{M_\odot  \; kpc^{-3}}]$',
    'SFR': r'$\mathrm{SFR} \, \, [\mathrm{M_\odot \; yr^{-1}}]$',
    'DensityStar': r'$\rho_{\star} (r)  \; \, \, [\mathrm{M_\odot  \; kpc^{-3}}]$',

    'joverR': r'$j (r) / r  \; \, \, [\mathrm{km  \; s^{-1}}]$',
    'joverRPartType0': r'$j_{\mathrm{gas}} (r) / r  \; \, \, [\mathrm{km  \; s^{-1}}]$',
    'joverRPartType4': r'$j_{\star} (r) / r  \; \, \, [\mathrm{km  \; s^{-1}}]$',
    'DensityGasOverR2':  r'$\rho_{\mathrm{gas}} (r)  r^2 \; \, \, [\mathrm{M_\odot  \; kpc^{-1}}]$',
    'sSFR': r'$\mathrm{sSFR} \, \, [\mathrm{M_\odot \; yr^{-1}}]$',
    'DensityStarOverR2':  r'$\rho_{\star} (r)  r^2 \; \, \, [\mathrm{M_\odot  \; kpc^{-1}}]$',

    'rad': r'$r \, \, [\mathrm{kpc}]$',

    #Scatter
    'z_Birth': r'$z_\mathrm{birth}$',
    'DMFrac_Birth': r'$(M_\mathrm{DM}/M)_\mathrm{birth}$',
    
    'nn2_distance': '$r_{nn} [\mathrm{kpc}]$',
    'nn5_distance': '$r_{nn} [\mathrm{kpc}]$',
    'nn10_distance': '$r_{nn} [\mathrm{kpc}]$',
    
    'nn2_distance_massive': '$r_{nn, \mathrm{massive}} [\mathrm{kpc}]$',
    'nn5_distance_massive': '$r_{nn, \mathrm{massive}} [\mathrm{kpc}]$',
    'nn10_distance_massive': '$r_{nn, \mathrm{massive}} [\mathrm{kpc}]$',
    
    
    'N_aper_1_Mpc':  '$N \mathrm{neighbours}$',
    'N_aper_2_Mpc': '$N \mathrm{neighbours}$',
    'N_aper_5_Mpc':  '$N \mathrm{neighbours}$',
    
    'N_aper_1_Mpc_massive':  '$N \mathrm{massive \, neighbours}$',
    'N_aper_2_Mpc_massive':  '$N \mathrm{massive \,neighbours}$',
    'N_aper_5_Mpc_massive': '$N \mathrm{massive \, neighbours}$',
    
    # PAPERII
    'M200Mean': r'$\overline{\log(M_{200} / \mathrm{M}_\odot)}$',
    'rOverR200Mean': r'$\overline{(\mathrm{R} / \mathrm{R}_{200})}$',
    'rOverR200Mean_New': r'$\overline{(\mathrm{R} / \mathrm{R}_{200})}$',

    'LBTime_Loss_Gass': 'Gas loss lookback time  [Gyr]',
    'z_At_FinalEntry':  r'$z_{\mathrm{infall}}$ in final host',
    'z_At_FirstEntry':  r'$z_{\mathrm{infall}}$ in first host',
    'deltaFirst_to_Final':  r'$\Delta \mathrm{t}_{\mathrm{first-to-final}}$ [Gyr]',
    



    None: 'Any'
}

component = { 
        '0': 'gas',
        '1': 'dm',
        '4': 'stars',
        '5': 'bhs',
}

def format_func_loglog(value, tick_number):
    '''
    change label in log plots
    Parameters
    ----------
    value : label values.
    tick_number : number of tickers
    Returns
    -------
    Requested label
    -------
    Author: Abhner P. de Almeida (abhner.almeida AAT usp.br)
    '''
    
    if value == 0:
        return str(0)
    sign = value/abs(value)
    N = int(np.round(np.log10(abs(value))))

    if abs(N) < 2:
        string = 10**N
        if sign*string >= 1:
            return str(int(sign*string))
        else:
            return str(sign*string)
    elif abs(N) >= 2:
        N = N*sign
        label = ('$10^{%4.0f}$ ' % N)
        return label
    
def Legend(names, mult = 2, msizeMult= 1.2, linewidth = 1.5):
    '''
    make the legend
    Parameters
    ----------
    names : name for the legend. 
    Returns
    -------
    lines, labels, number of columns and fontsize multiplicative factor for the legend
    -------
    Author: Abhner P. de Almeida (abhner.almeida AAT usp.br)
    '''
    
    custom_lines = []
    label = []
    for name in names: 
        if 'Scatter' in name or name == 'Bian et al. (2025)':
            name = name.replace('Scatter', '')
            BlackLine = False

            if 'Legend' in name:
                name = name.replace('Legend', '')
                
                lw = 2
                lwe = lw
            elif 'Empty' in name:
                if 'BadFlag' in name:
                    name = name.replace('Empty', '')

                lw = 0
                lwe = 1.5
            elif 'BlackLine' in name:
                name = name.replace('BlackLine', '')
                BlackLine = True
                lwe = 0.8
                lw = 0
            else:
                lw = 0
                lwe = lw
                BlackLine = False
            if name == 'Bian et al. (2025)':
                custom_lines.append(Patch(facecolor='tab:red', alpha = 0.4))
            else:
                if 'Normal' in name:
                    msizeMult = 1.8
                    
                if 'SBC' in name or 'MBC' in name  or 'Diffuse' in name:
                    msizeMult = 1.2
                    
                if BlackLine:
                    custom_lines.append(
                    Line2D([0], [0], color=colors.get(name, 'black'), lw=lw, marker=markers.get(name, None),  markeredgewidth = lwe,
                           markersize = msizeMult*msize.get(name, 1.5), markeredgecolor = 'k'))
                else:
                    custom_lines.append(
                    Line2D([0], [0], color=colors.get(name, 'black'), lw=lw, marker=markers.get(name, None),  markeredgewidth = lwe,
                           markersize = msizeMult*msize.get(name, 1.5), markeredgecolor = edgecolors.get(name, 'k')))
            label.append(titles.get(name, name))

        elif name == 'None':
            custom_lines.append(Line2D([0], [0], lw=0))
            label.append('')
            
        elif 'IDsColumn' in name and 'RadType' in name:
            name = name.replace('IDsColumn', '')
            custom_lines.append(Line2D([0], [0], color=colors.get(
                name, 'black'), ls= 'solid', lw=mult * 0.5* linesthicker.get(name, 1.8), dash_capstyle = capstyles.get(name, 'projecting')))
            label.append(titles.get(name, name))
        
        else:
            name = name.replace('IDsColumn', '')
            custom_lines.append(Line2D([0], [0], color=colors.get(
                name, 'black'), ls=lines.get(name, 'solid'), lw=mult * 0.5* linesthicker.get(name, 1.3), dash_capstyle = capstyles.get(name, 'projecting')))
            label.append(titles.get(name, name))

    if len(names) < 4:
        ncol = 1
        mult = 0.6
    elif len(names) <= 8:
        ncol = 2
        mult = 0.6
    else:
        ncol = 3
        mult = 0.5
    return custom_lines, label, ncol, mult

def savefig(savepath, savefigname, TRANSPARENT = True, SIM = SIMTNG):
    '''
    save figures
    Parameters
    ----------
    savepath : save path. 
    savefigname : fig name.
    -------
    Author: Abhner P. de Almeida (abhner.almeida AAT usp.br)
    '''
    
    pathBase =  os.getenv("HOME")+'/TNG_Analyzes/Figs/' + SIM + '/'
    try:
        plt.savefig(pathBase + savepath+'/'+savefigname +
                    '.pdf', bbox_inches='tight')
        
        plt.savefig(pathBase + savepath+'/'+'PNG'+'/'+savefigname +
                        '.png', bbox_inches='tight',  transparent=False, dpi=200)
        
        if TRANSPARENT:
            plt.savefig(pathBase + savepath+'/'+'PNG'+'/'+savefigname +
                        '.png', bbox_inches='tight', facecolor='white', transparent=True, dpi=400)

    except:
        directories = savepath.split('/')
        directories.append('PNG')
        path = pathBase
        for name in directories:
            path = os.path.join(path, name)
            if not os.path.isdir(path):
                os.mkdir(path)
        plt.savefig(pathBase + savepath +  '/'+savefigname +
                    '.pdf', bbox_inches='tight')

        plt.savefig(path +'/'+savefigname +
                        '.png', bbox_inches='tight', facecolor='white',  transparent=False, dpi=200)
        
        if TRANSPARENT:
        
            plt.savefig(path + '/'+savefigname +
                        '.png', bbox_inches='tight',  transparent=True, dpi=400)
            
def PlotMedianEvolution(names, columns, rows, Type='Evolution', Xparam=['Time'], title=False, 
                        PhasingPlot = False, NormalizedExSitu = False,
                        GasLim=False, CompareToNormal_Name = False, XScaleSymlog = False,
                        xlabelintext=False, lineparams=False, legendColumn = False, LookBackTime = True, SmallerScale = False,
                        ColumnPlot=True, limaxis=False,  legend=False,  Transparent = True, Text = None, Pericenter = False, Supertitle = False,
                        savepath='PlotMedianEvolution',  savefigname='fig', dfName='Sample', SampleName='SubfindID_99', 
                        Supertitle_Name = 'DM-rich', LegendNames='None',   loc=['best'], loctext = ['best'],
                        Softening = False, xPhaseLim = 8, EntryMedian = False,
                        yscale = 'linear', ylimmin = None, ylimmax = None, xlimmin = None, xlimmax = None, legpositions = None,
                        lNum = 6, cNum = 6, GridMake = False, NormalRatio = False, CompareToNormal = False, CompareToNormalLog = True,
                        alphaShade=0.3,  linewidth=1.1, framealpha = 0.95,  fontlabel=26, nboots=100,  JustOneXlabel = False,
                        Supertitle_y = 0.99,  multtick = 0.99, columnspacing = 0.5, handlelength = 2, handletextpad = 0.4, labelspacing = 0.3, 
                        bins=10, seed=16040105):
    '''
    Plot Median Evolution or Co-Evolution
    Parameters
    ----------
    names : sample names. array with str
    columns : specific set in the sample / or different param to plot in each column. array with str
    rows : specific set in the sample / or different param to plot in each row. array with str
    Type : plot type. Default: 'Evolution'. str 'Evolution' or 'CoEvolution.
    Xparam : x param for Co-evolution. Default: ['Time']. array with str
    title : to make the title. Default: False. False or array with str
    limaxis : if you use lineparams to make the labels. Default:False.bool
    lineparams : if we want different params as different lines. Default: False. bool
    legendColumn: Legend if not column plot. Default: False
    LookBackTime: LookBack time for the x axis. Default: True
    loctext : loc for text. Default:['best']. array with str
    The rest is the same as the previous functions
    Returns
    -------
    Requested Evolution or Co-Evolution plot
    -------
    Author: Abhner P. de Almeida (abhner.almeida AAT usp.br)
    '''

    np.random.seed(seed)

    dfTime = TNG.extractDF('SNAPS_TIME')
    #snapsTimeII = np.array([88, 76, 63, 50, 37, 24])
    snapsTime = np.array([88, 81, 64, 51, 37, 24])
    # Verify NameParameters
    if type(columns) is not list and type(columns) is not np.ndarray:
        columns = [columns]

    if type(rows) is not list and type(rows) is not np.ndarray:
        rows = [rows]
        
    if Pericenter:
        dataROverR200, errROverR200 = TNG.makedataevolution( names, columns, ['r_over_R_Crit200'], SampleName=SampleName, dfName = dfName, nboots=nboots)

        
    if Type == 'Evolution':
        if ColumnPlot:
            if lineparams:
                datasAll = []
                dataserrAll = []
                if PhasingPlot:
                    datasPhaseAll = []
                    datasTimeAll = []
                for row in rows:
                    
                    if PhasingPlot:
                        datas, dataserr, dataPhase, datasTime = TNG.makedataevolution(
                            names, columns, row, PhasingPlot = PhasingPlot , SampleName=SampleName, dfName = dfName, nboots=nboots)
                        
                        datasPhaseAll.append(dataPhase)
                        datasTimeAll.append(datasTime)
                    else:
                        datas, dataserr = TNG.makedataevolution(
                            names, columns, row, PhasingPlot = PhasingPlot , SampleName=SampleName, dfName = dfName, nboots=nboots)
                    datasAll.append(datas)
                    dataserrAll.append(dataserr)
            else:
                if PhasingPlot:
                        datas, dataserr, datasPhase, datasTime = TNG.makedataevolution(
                            names, columns, rows, PhasingPlot = PhasingPlot , SampleName=SampleName, dfName = dfName, nboots=nboots)
                else:
                    datas, dataserr = TNG.makedataevolution(
                        names, columns, rows,  SampleName=SampleName, dfName = dfName, nboots=nboots)
            time = dfTime.Age.values

        else:
            if lineparams:
                datasAll = []
                dataserrAll = []
                if PhasingPlot:
                    datasPhaseAll = []
                    datasTimeAll = []


                for column in columns:
                    if PhasingPlot:
                        datas, dataserr, datasPhase, datasTime = TNG.makedataevolution(
                            names,  rows,column,  PhasingPlot = PhasingPlot , SampleName=SampleName, dfName = dfName, nboots=nboots)
                        
                        datasPhaseAll.append(dataPhase)
                        datasTimeAll.append(datasTime)

                    else:
                        datas, dataserr = TNG.makedataevolution(
                        names,  rows,column,  SampleName=SampleName, dfName = dfName, nboots=nboots)
                    datasAll.append(datas)
                    dataserrAll.append(dataserr)
                    
            
            else:
                if PhasingPlot:
                    datas, dataserr, datasPhase, datasTime= TNG.makedataevolution(
                        names,  rows,column,  PhasingPlot = PhasingPlot , SampleName=SampleName, dfName = dfName, nboots=nboots)
                else:
                    datas, dataserr = TNG.makedataevolution(
                        names, rows, columns,  SampleName=SampleName, dfName = dfName, nboots=nboots)
            time = dfTime.Age.values


    elif Type == 'CoEvolution':
        if ColumnPlot:
            if lineparams:
                datasX, datasXerr = TNG.makedataevolution(
                    names, columns, Xparam,  SampleName=SampleName, dfName = dfName, nboots=nboots)
                datasY, datasYerr = TNG.makedataevolution(
                    names, columns, rows[0],  SampleName=SampleName, dfName = dfName, nboots=nboots)

            else:
                datasX, datasXerr = TNG.makedataevolution(
                    names, columns, Xparam, SampleName=SampleName, dfName = dfName, nboots=nboots)
                datasY, datasYerr = TNG.makedataevolution(
                    names, columns, rows,  SampleName=SampleName, dfName = dfName, nboots=nboots)

        else:
            datasY, datasYerr = TNG.makedataevolution(
                names, columns, rows,  SampleName=SampleName, dfName = dfName, nboots=nboots)
            datasX, datasXerr = TNG.makedataevolution(
                names, columns, Xparam,  SampleName=SampleName, dfName = dfName, nboots=nboots)
            columns = Xparam
            
        time = dfTime.Age.values

    # Define axes
    plt.rcParams.update({'figure.figsize': (cNum*len(columns), lNum*len(rows))})
    fig = plt.figure()
    gs = fig.add_gridspec(len(rows), len(columns), hspace=0, wspace=0)
    axs = gs.subplots(sharex='col', sharey='row')

    # Verify axs shape
    if type(axs) is not list and type(axs) is not np.ndarray:
        axs = [axs]
    if type(axs[0]) is not np.ndarray:
        axs = np.array([axs])
        if len(columns) == 1:
            axs = axs.T

    for i, row in enumerate(rows):
        
        
        for j, column in enumerate(columns):
            if Softening and 'Type4' in row:
                rSoftening = ETNG.Softening()
                axs[i][j].fill_between(np.flip(time), -1, np.log10(rSoftening), alpha=0.1, color='tab:red')  # yellow


            if not lineparams:

                if Type == 'Evolution':
                    if ColumnPlot:
                        param = row
                        data = datas[i][j]
                        dataerr = dataserr[i][j]
                        if PhasingPlot:
                            dataphase = datasPhase[i][j]
                            datatime = datasTime[i][j]
                            
                    else:
                        param = column
                        data = datas[j][i]
                        dataerr = dataserr[j][i]
                        if PhasingPlot:
                            dataphase = datasPhase[j][i]
                            datatime = datasTime[j][i]

                elif Type == 'CoEvolution':
                    param = row
                    if ColumnPlot:
                        xparam = Xparam[i]
                        dataX = datasX[0][j]
                        data = datasY[i][j]
                        if Pericenter:
                            dataROverR200Now = dataROverR200[0][j]

                    else:
                        xparam = Xparam[i]
                        dataX = datasX[j][0]
                        data = datasY[i][0]
                        if Pericenter:
                            dataROverR200Now = dataROverR200[j][0]
                            
                                            
                
                for l, values in enumerate(data):
                    if CompareToNormal:
                        if CompareToNormal_Name:
                            Y, Yerr = TNG.makedataevolution(['Normal'], [names[l]], [row], SampleName=SampleName, dfName = dfName, nboots=nboots)

                        else:
                            Y, Yerr = TNG.makedataevolution(['Normal'], [column], [row], SampleName=SampleName, dfName = dfName, nboots=nboots)
                        Yerr = np.array([value for value in Yerr[0][0][0]])
                        Y = np.array([value for value in Y[0][0][0]])


                    
                    if NormalizedExSitu:
                        Mass4, Mass4err = TNG.makedataevolution([names[l]], [column], [row], SampleName=SampleName, dfName = dfName, nboots=nboots)
                        Mass4err = np.array([value for value in Mass4err[0][0][0]])
                        Mass4 = np.array([value for value in Mass4[0][0][0]])



                    if PhasingPlot:
                        xParam = dataphase[l]
                        timeParam = datatime[l]
                    else:
                        xParam = time
                        timeParam = time
                    values = np.array([value for value in values])
                    
                    


                    if Type == 'Evolution':
                        err = np.array([value for value in dataerr[l]])

                    #if PhasingPlot:
                    #    xParam = xParam[(~np.isnan(values))]
                    #    err = err[(~np.isnan(values))]
                    #    values = values[(~np.isnan(values))]
                    #    X_Y_Spline = interp1d(xParam, values,kind=‘linear’,fill_value=‘extrapolate’)
                    #    X_ = np.linspace(xParam.min(), xParam.max(), 500)
                    #    values = X_Y_Spline(X_)
                    #    X_Y_Spline = interp1d(xParam, err,kind=‘linear’,fill_value=‘extrapolate’)
                    #    err = X_Y_Spline(X_)
                    #    xParam = X_

                    if Pericenter:
                        ROverR200 = np.array([value for value in dataROverR200Now[l]])
                        argInfall1 = np.argwhere(ROverR200 < 1).T[0]
                        argInfall2 = np.argwhere(ROverR200 < 2).T[0]

    
                    if Type == 'Evolution':
                        
                        if param in ['sSFRCoreRatio']:
                            values[values == 0] = np.nan
                        elif 'sSFR' in param:
                            values[values < -13.5] = np.nan
                        elif 'SFR' in param:
                            values[values < -4] = np.nan
                            
                        if PhasingPlot:
                            if 'GasMass' in row[l] and 'LoseTheir' in column:
                                argWhereNan = np.argwhere(np.isnan(values)).T[0]
                                
                                values[argWhereNan[0]:] = np.nan
                                
                        
                        if CompareToNormal:
                            if not CompareToNormalLog:
                                err = np.sqrt((Yerr / Y)**2.+(err*values/ Y**2)**2.)
                                values = values/ Y
                            elif CompareToNormalLog:
                                err = np.sqrt((10**Yerr/ 10**Y)**2.+(err*10**values/ (10**Y)**2.)**2.)
                                values = 10**values/ 10**Y
                                
                        if NormalizedExSitu:
                            Mass4, Mass4err = TNG.makedataevolution([names[l]], [column], ['SubhaloMassType4'], SampleName=SampleName, dfName = dfName, nboots=nboots)
                            Frac4, Frac4err = TNG.makedataevolution([names[l]], [column], ['MassExNormalizeAll'], SampleName=SampleName, dfName = dfName, nboots=nboots)

                            Mass4err = np.array([value for value in Mass4err[0][0][0]])
                            Mass4 = np.array([value for value in Mass4[0][0][0]])

                            Frac4err = np.array([value for value in Frac4err[0][0][0]])
                            Frac4 = np.array([value for value in Frac4[0][0][0]])

                            values = 10**values / 10**Mass4[0]
                            err = Frac4err
                            
                        
                            
                        axs[i][j].plot(xParam[~np.isnan(values)], values[~np.isnan(
                            values)], color=colors.get(names[l], 'black'), ls=lines.get(names[l], 'solid'), 
                            lw=1.5*linesthicker.get(names[l], linewidth), dash_capstyle = capstyles.get(names[l], 'projecting'))
            

                        axs[i][j].fill_between(
                            xParam[~np.isnan(values)], values[~np.isnan(values)] - err[~np.isnan(values)], 
                            values[~np.isnan(values)] + err[~np.isnan(values)], color=colors.get(names[l] +'Error', 'black'), ls=lines.get(names[l], 'solid'), alpha=alphaShade)
                         
                        if EntryMedian:
                            dfPop = TNG.extractPopulation(names[l]+column, dfName = dfName)
                            axs[i][j].axvline(dfTime.Age.loc[dfTime.Snap == int(np.nanmedian(dfPop.Snap_At_FirstEntry))].values[0],  ymax=0.15-l*0.015,
                                              ls='solid', color=colors.get(names[l], 'black'), lw=1.5*linesthicker.get(names[l], linewidth))
                            axs[i][j].axvline(dfTime.Age.loc[dfTime.Snap == int(np.nanmedian(dfPop.Snap_At_FinalEntry))].values[0],  ymax=0.15-l*0.015,
                                              ls='--', color=colors.get(names[l], 'black'),lw=1.5*linesthicker.get(names[l], linewidth))
                            
                        if Pericenter:
                            if len(argInfall1) == 0:
                                continue
                            else:
                                argInfall1 = argInfall1[-1]
    
                                axs[i][j].scatter(xParam[argInfall1], values[argInfall1], color=colors.get(names[l] , 'black'),
                                              lw= 3 *linewidth, marker='x',  edgecolors=colors.get(names[l], 'black'), s=120, alpha=0.9)

                    elif Type == 'CoEvolution':

                        x = np.array([value for value in dataX[l]])
                    
                        colorSnap = np.array(
                                    ['darkblue', 'tab:blue', 'cyan', 'darkgreen', 'tab:orange', 'red'])
                        
                        f_linear = interp1d(np.arange(100)[~np.isnan(values)], values[~np.isnan(values)], fill_value='extrapolate')
                        f_linearX = interp1d(np.arange(100)[~np.isnan(x)], x[~np.isnan(x)], fill_value='extrapolate')
                        
                        plotLine = colored_line(f_linearX(np.arange(100)), f_linear(np.arange(100)), time, 
                                             axs[i][j], linewidth= 2, cmap="bwr_r")
                        
                        axs[i][j].plot(x[~np.isnan(values)], values[~np.isnan(values)], color=colors.get(names[l]  , 'black'), ls=lines.get(
                                names[l], 'dashed'), lw= 1.5* linesthicker.get(names[l], linewidth), dash_capstyle = capstyles.get(names[l], 'projecting'), zorder=1)                        
                        
                        
                        
                        #axs[i][j].scatter(f_linearX(99-snapsTime), f_linear(99-snapsTime), color=colorSnap,
                        #                  lw= 3 *linewidth, marker='d',  edgecolors=colors.get(names[l]), s=190, alpha=1., zorder=2)
                        axs[i][j].scatter(x[0], f_linear(0), color='black', lw= 2 * linewidth, marker='o',  edgecolors=colors.get(
                            names[l], 'black'), s=50 , alpha=0.9, zorder=2)
                        
                        
                        if Pericenter:
                            
                            if len(argInfall2) == 0:
                                continue
                            else:
                                argInfall2 = argInfall2[-1]
    
                                axs[i][j].scatter(x[argInfall2], values[argInfall2], color=colors.get(names[l] + 'Error', 'black'),
                                              lw= 3 *linewidth, marker='x',  edgecolors=colors.get(names[l]+'Error', 'black'), s=190, alpha=1.0, zorder=3)
                            
                            if len(argInfall1) == 0:
                                continue
                            else:
                                argInfall1 = argInfall1[-1]
    
                                axs[i][j].scatter(x[argInfall1], values[argInfall1], color=colors.get(names[l]  + 'Error'),
                                              lw= 3 *linewidth, marker='x',  edgecolors=colors.get(names[l]+'Error'), s=190, alpha=1.0, zorder=3)
                                
 

            if lineparams:
                for m, xparam in enumerate(Xparam):
                    if ColumnPlot:
                        varParam  = row
                    else:
                        varParam  = column
                    for k, paramname in enumerate(varParam):
                        if Type == 'Evolution':
                            if ColumnPlot:
                                param = paramname
                                data = datasAll[i][k][j]
                                dataerr = dataserrAll[i][k][j]
                                if PhasingPlot:
                                    dataphase = datasPhaseAll[i][k][j]
                                    datatime = datasTimeAll[i][k][j]
                            else:
                                param = column[-1]
                                data = datasAll[i][j][k]
                                dataerr = dataserrAll[i][j][k]
                                if PhasingPlot:
                                    dataphase = datasPhaseAll[i][j][k]
                                    datatime = datasTimeAll[i][j][k]
                                if Pericenter:
                                    dataROverR200Now = dataROverR200[j][0]

                        elif Type == 'CoEvolution':
    
                            if ColumnPlot:
                                xparam = Xparam[i]
                                param = paramname
                                dataX = datasX[m][j]
                                data = datasY[k][j]
    
                            else:
                                
                                param = column[-1]
                                dataX = datasX[i][j]
                                data = datasY[k][j]
                                if Pericenter:
                                    dataROverR200Now = dataROverR200[j][0]
                                    
                        if CompareToNormal:
                            Y, Yerr = TNG.makedataevolution(['Normal'], [column], [paramname], SampleName=SampleName, dfName = dfName, nboots=nboots)
                            Yerr = np.array([value for value in Yerr[0][0][0]])
                            Y = np.array([value for value in Y[0][0][0]])

    
                        for l, values in enumerate(data):
                            if Pericenter:
                                ROverR200 = np.array([value for value in dataROverR200Now[l]])
                                argInfall = np.argwhere(ROverR200 < 1).T[0]
    
                            values = np.array([value for value in values])


                            

                            if PhasingPlot:
                                xParam = dataphase[l]
                                timeParam = datatime[l]
                            else:
                                xParam = time
                                timeParam = time
                            values = np.array([value for value in values])


                            if Type == 'Evolution':
                                if 'sSFR' in paramname:
                                    values[values < -13.5] = np.nan
                                    
                                elif 'SFR' in paramname:
                                    values[values < -4] = np.nan
                                    
                                err = np.array([value for value in dataerr[l]])
                                
                                if CompareToNormal:
                                    err = np.sqrt((Yerr)**2.+(err)**2.)
                                    values = values - Y
                                
                                
                                if GasLim and ('Gas' in paramname or 'SFR' in paramname or 'Type0' in paramname):
                                    print(len(timeParam), len(values))
                                    dfPop = TNG.extractPopulation(names[l]+column, dfName = dfName)
                                    if ~np.isnan(np.nanmedian(dfPop.SnapLostGas)) and np.nanmedian(dfPop.SnapLostGas) > 0:
                                        values[timeParam > dfTime.Age.loc[dfTime.Snap == int(np.nanmedian(dfPop.SnapLostGas))].values[0]] = np.nan
                             
                                
                                axs[i][j].plot(
                                    xParam[~np.isnan(values)], values[~np.isnan(values)], color=colors.get(names[l], 'k'), ls=lines.get(paramname, 'solid'), 
                                    lw=1.7*linesthicker.get(paramname, linewidth),
                                    dash_capstyle = capstyles.get(paramname, 'projecting'))
                                axs[i][j].fill_between(
                                    xParam[~np.isnan(values)], values[~np.isnan(values)] - err[~np.isnan(values)], values[~np.isnan(values)] + err[~np.isnan(values)], 
                                    color=colors.get(names[l]+'Error', 'k'), ls=lines.get(names[l], 'solid'), alpha=alphaShade)
    
                                if EntryMedian:
                                    dfPop = TNG.extractPopulation(names[l]+column, dfName = dfName)
                                    dfPop = dfPop.Snap_At_FirstEntry.values
                                    dfPop = dfPop[~np.isnan(dfPop)]
                                    if len(dfPop) > 5:
                                        if dfTime.Age.loc[dfTime.Snap == int(np.nanmedian(dfPop))].values[0] > dfTime.Age.loc[dfTime.Snap == int(np.nanmedian(dfPop))].values[0]:
                                            axs[i][j].axvline(dfTime.Age.loc[dfTime.Snap == int(np.nanmedian(dfPop))].values[0],  ymax=0.15-l*0.015,
                                                              ls='solid', color=colors.get(names[l], 'black'), lw=1.5*linesthicker.get(names[l], linewidth))
                                            axs[i][j].axvline(dfTime.Age.loc[dfTime.Snap == int(np.nanmedian(dfPop))].values[0],  ymax=0.15-l*0.015,
                                                              ls='--', color=colors.get(names[l], 'black'),lw=1.5*linesthicker.get(names[l], linewidth))
                                        else:
                                            axs[i][j].axvline(dfTime.Age.loc[dfTime.Snap == int(np.nanmedian(dfPop))].values[0],  ymax=0.15-l*0.015,
                                                              ls='solid', color=colors.get(names[l], 'black'), lw=1.5*linesthicker.get(names[l], linewidth))
                                            axs[i][j].axvline(dfTime.Age.loc[dfTime.Snap == int(np.nanmedian(dfPop))].values[0],  ymax=0.15-l*0.015,
                                                              ls='--', color=colors.get(names[l], 'black'),lw=1.5*linesthicker.get(names[l], linewidth))
                                        
                                    
                                if Pericenter:
                                    if len(argInfall) == 0:
                                        continue
                                    else:
                                        argInfall = argInfall[-1]
            
                                        axs[i][j].scatter(xParam[argInfall], values[argInfall], color=colors.get(names[l] , 'black'),
                                                      lw= 3 *linewidth, marker='x',  edgecolors=colors.get(names[l]), s=120, alpha=0.9)
                                        
                            elif Type == 'CoEvolution':
                                x = np.array([value for value in dataX[l]])
                                colorSnap = np.array(
                                    ['darkblue', 'tab:blue', 'cyan', 'darkgreen', 'tab:orange', 'red'])
                                if len(Xparam) > 1:
                                    axs[i][j].plot(x[~np.isnan(values)], values[~np.isnan(values)], color=colors.get(names[l]   , 'black'), ls=lines.get(
                                            paramname, 'dashed'), lw= 1.7* linesthicker.get(paramname, linewidth), dash_capstyle = capstyles.get(paramname, 'projecting'))

                                    axs[i][j].scatter(x[99-snapsTime], values[99-snapsTime], color=colorSnap,
                                                      lw= 2 *linewidth, marker='d',  edgecolors=colors.get(names[l] + Xparam[m]), s=50, alpha=0.9)
                                    axs[i][j].scatter(x[0], values[0], color='black', lw= 2 * linewidth, marker='o',  edgecolors=colors.get(
                                        paramname), s=40 , alpha=0.9)
                                else:
                                    axs[i][j].plot(x[~np.isnan(values)], values[~np.isnan(values)], color=colors.get(names[l]  , 'black'), ls=lines.get(
                                            paramname, 'dashed'), lw= 1.7* linesthicker.get(paramname, linewidth), dash_capstyle = capstyles.get(paramname, 'projecting'))
            
                                    axs[i][j].scatter(x[99-snapsTime], values[99-snapsTime], color=colorSnap,
                                                      lw= 2 *linewidth, marker='d',  edgecolors=colors.get(paramname ), s=50, alpha=1.)
                                    axs[i][j].scatter(x[0], values[0], color='black', lw= 2 * linewidth, marker='o',  edgecolors=colors.get(
                                        paramname), s=40 , alpha=0.9)
                                    
                            

            if CompareToNormal:
                axs[i][j].axhline(y = 0, color= 'gray', lw= 1.5*linewidth)
                
            # Plot details
            if GridMake:
                axs[i][j].grid(GridMake, color='#9e9e9e',  which="major", linewidth= 0.6,alpha= 0.3 , linestyle=':')            
    
            if ylimmin != None and ylimmax != None:
                axs[i][j].set_ylim(ylimmin[i], ylimmax[i])
            if not NormalizedExSitu:
                axs[i][j].set_yscale(scales.get(param, yscale))
            if scales.get(param, yscale) == 'log':
                
                axs[i][j].yaxis.set_major_formatter(
                    FuncFormatter(format_func_loglog))
            elif NormalizedExSitu:
                axs[i][j].set_yscale('log')
                axs[i][j].yaxis.set_major_formatter(
                    FuncFormatter(format_func_loglog))
            
            if param == 'MassExNormalizeAll' and lineparams:
                axs[i][j].set_yticks([0.001, 0.005, 0.01, 0.02, 0.05])
                axs[i][j].set_yticklabels(['0.001','0.005', '0.01', '0.02', '0.05'])
                
            elif param == 'MassExNormalizeAll' or NormalizedExSitu:
                axs[i][j].set_yticks([0.005, 0.01, 0.05])
                axs[i][j].set_yticklabels(['0.005', '0.01', '0.05'])
                
            if param == 'MassExNormalize':
                
                axs[i][j].set_yticks([0.01, 0.02, 0.05, 0.1, 0.5, 1])
                axs[i][j].set_yticklabels(['0.01', '0.02', '0.05', '0.1', '0.5', '1'])
                
            if param == 'MassInNormalize':
                
                axs[i][j].set_yticks([0.01, 0.02, 0.05, 0.1, 0.5, 1])
                axs[i][j].set_yticklabels(['0.01', '0.02', '0.05', '0.1', '0.5', '1'])
                
                
            if param == 'GroupNsubsFinalGroup':
                
                axs[i][j].set_yticks([20, 30, 40, 60])
                axs[i][j].set_yticklabels(['20', '30', '40', '60'])
                

            if param == 'StarMassNormalized':
                axs[i][j].set_yticks([0.1, 0.2, 0.5, 1])
                axs[i][j].set_yticklabels(['0.1','0.2', '0.5', '1'])

            #if 'Nsub' in param:
            #    axs[i][j].set_yticks([20, 30, 40, 60])
            #    axs[i][j].set_yticklabels(['20', '30', '40', '60'])

            if legend:
                for legpos, LegendName in enumerate(LegendNames):
                    if j == legpositions[legpos][0] and i ==legpositions[legpos][1]:
                        custom_lines, label, ncol, mult = Legend(
                            LegendName)
                        axs[i][j].legend(
                            custom_lines, label, ncol=ncol, loc=loc[legpos], fontsize=0.88*fontlabel, framealpha = framealpha, 
                            columnspacing = columnspacing, handlelength = handlelength, handletextpad = handletextpad, labelspacing = labelspacing)

            if j == 0:

                if xlabelintext:
                    if not CompareToNormal:
                        axs[i][j].set_ylabel(
                            labelsequal.get(param, param), fontsize=fontlabel)
                    elif CompareToNormal:
                        axs[i][j].set_ylabel(
                            labelsequal.get(param, '') + '$-$'+ labelsequal.get(param, '') + '$_\mathrm{Normals}$', fontsize=fontlabel)
                    axs[i][j].tick_params(axis='y', labelsize=multtick*fontlabel)

                else:
                    if param in  ['SubhalosSFRInHalfRad', 'SubhalosSFRwithinHalfandRad', 'SubhalosSFRwithinRadandAll'] and Text != None and Text != ['DM-poor', 'DM-rich']:

                        if not CompareToNormal:
                            axs[i][j].set_ylabel(
                                labelsequal.get(param, param), fontsize=fontlabel)
                        elif CompareToNormal:
                            axs[i][j].set_ylabel(
                                labelsequal.get(param) + '$-$'+ labelsequal.get(param) + '$_\mathrm{Normals}$', fontsize=fontlabel)
                        axs[i][j].tick_params(axis='y', labelsize=0.99*fontlabel)
                        Afont = {'color':  'black',
                                 'size': fontlabel,
                                 }
                        anchored_text = AnchoredText(
                            Text[i], loc='upper right', prop=Afont)
                        axs[i][1].add_artist(anchored_text)
                        
    
                    else:
                        if not CompareToNormal and not NormalizedExSitu:
                            
                            if lineparams and len(varParam) > 1:
                                axs[i][j].set_ylabel(labelsequal.get(param, param), fontsize=fontlabel)
                            else:
                                axs[i][j].set_ylabel(labels.get(param, param), fontsize=fontlabel)
                        if not CompareToNormal and NormalizedExSitu:
                            axs[i][j].set_yscale('log')
                            axs[i][j].set_ylabel(
                            labels.get('MassExNormalizeAll'), fontsize=fontlabel)
                        elif CompareToNormal:
                            axs[i][j].set_ylabel(
                                labels.get(param, '') + '$-$'+ labels.get(param, '') + '$_\mathrm{Normals}$', fontsize=fontlabel)
                        axs[i][j].tick_params(axis='y', labelsize=multtick*fontlabel)
                        
                if param in ['LastMerger', 'LastMajorMerger', 'LastMinorMerger']:
                    axs[i][j].tick_params(
                        left=True, right=False, labelsize=0.99*fontlabel)
                    axs[i][j].set_yticks([2, 4, 6, 8, 10, 11])
                    axs[i][j].set_yticklabels(
                        ['2', '4', '6', '8', '10', ''])
                    axs[i][j].tick_params(axis='y', labelsize=multtick*fontlabel)
                                        
                    lim = axs[i][j].get_ylim()
                    ax2label = axs[i][j].twinx() #secondary_xaxis('top', which='major')
                    ax2label.grid(False)
                    ax2label.set_xlim(lim)

                    zticks = np.array([0.2, 0.5, 1., 2., 5.])
                    zlabels = np.array(
                        ['', '0.5', '1', '2', '5'])
                    zticks_Age = np.array(
                        [11, 8.587, 5.878, 3.285, 1.2])

                    zticks = zticks.tolist()
                    zticks_Age = zticks_Age.tolist()

                    x_locator = FixedLocator(zticks_Age)
                    x_formatter = FixedFormatter(zlabels)
                    ax2label.yaxis.set_major_locator(x_locator)
                    ax2label.yaxis.set_major_formatter(x_formatter)
                    ax2label.set_ylabel(r"$z$", fontsize=fontlabel)
                    ax2label.tick_params(labelsize=multtick*fontlabel)
                    ax2label.tick_params(axis='x',  which='minor', top=False)
                    ax2label.minorticks_off()

            if j == len(columns) - 1:
                if Text != None and not param in  ['SubhalosSFRInHalfRad', 'SubhalosSFRwithinHalfandRad', 'SubhalosSFRwithinRadandAll']:

                    Afont = {'color':  'black',
                             'size': fontlabel,
                             }
                    anchored_text = AnchoredText(
                        Text[i], loc='lower left', prop=Afont)
                    axs[i][j].add_artist(anchored_text)
                    
                if xlabelintext and not limaxis and len(rows) > 1:
                    Afont = {'color':  'black',
                             'size': fontlabel,
                             }
                    anchored_text = AnchoredText(
                        texts.get(param), loc='upper right', prop=Afont)
                    axs[i][j].add_artist(anchored_text)

            if xlabelintext and limaxis and len(rows) > 1 and len(texts) == 1:
                Afont = {'color':  'black',
                         'size': fontlabel,
                         }
                anchored_text = AnchoredText(
                    texts[param], loc=loctext[i], prop=Afont)
                axs[i][j].add_artist(anchored_text)

            if i == 0:

                if title:
                    axs[i][j].set_title(titles.get(
                        title[j], title[j]), fontsize=1.1*fontlabel)

                if Type == 'Evolution' and not  PhasingPlot:

                    axs[i][j].tick_params(
                        bottom=True, top=False, labelsize=multtick*fontlabel)
                    lim = axs[i][j].get_xlim()
                    ax2label = axs[i][j].twiny() #secondary_xaxis('top', which='major')
                    ax2label.grid(False)
                    if  XScaleSymlog:
                        ax2label.set_xlim(-0.5, 14.)
                        ax2label.set_xscale('symlog')
                        zticks = np.array([0., 0.5, 1., 2., 5., 20])
                        zlabels = np.array(['0', '0.5', '1', '2', '5', '20'])
                        
                        zticks_Age = np.array([13.803, 8.587, 5.878, 3.285, 1.2, 0])
                        
                        
                    else:
                        ax2label.set_xlim(-0.5, 14.5)

                        zticks = np.array([0., 0.2, 0.5, 1., 2., 5., 20.])
                        if not (JustOneXlabel) and not (SmallerScale):
                            zlabels = np.array(
                                ['0', '0.2', '0.5', '1', '2', '5', '20'])
                        else:
                            if j == 0:
                                zlabels = np.array(
                                    ['0', '0.2', '0.5', '1', '2', '5', '20'])
                            else:
                                zlabels = np.array(
                                    ['0', '0.2', '0.5', '1', '2', '5', ''])
                        
                        zticks_Age = np.array(
                            [13.803, 11.323, 8.587, 5.878, 3.285, 1.2, 0])
                        
                    zticks = zticks.tolist()
                    zticks_Age = zticks_Age.tolist()

                    x_locator = FixedLocator(zticks_Age)
                    x_formatter = FixedFormatter(zlabels)
                    ax2label.xaxis.set_major_locator(x_locator)
                    ax2label.xaxis.set_major_formatter(x_formatter)
                    ax2label.set_xlabel(r"$z$", fontsize=fontlabel)
                    ax2label.tick_params(labelsize=multtick*fontlabel)
                    ax2label.tick_params(axis='x',  which='minor', top=False)
                    #zlabelsMinor = np.array(['', '', '', '', '', '', ''])
                    #ax2label.xaxis.set_minor_locator(FixedLocator(zticks_Age))
                    #ax2label.xaxis.set_minor_formatter(FixedFormatter(zlabelsMinor))

                else:
                    axs[i][j].tick_params(axis='x',  which='minor', top=False)

            if i == len(rows) - 1:
                
                if Type == 'Evolution':
                    if i == 0:
                        axs[i][j].tick_params( 
                            bottom=True, top=False, labelsize=multtick*fontlabel)
                        

                    if LookBackTime and not PhasingPlot:
                        if ( JustOneXlabel and j == 1):
                            axs[i][j].set_xlabel(r'$\mathrm{Lookback \; Time} \, \, [\mathrm{Gyr}]$', fontsize=fontlabel)
                        if not JustOneXlabel:
                            axs[i][j].set_xlabel(r'$\mathrm{Lookback \; Time} \, \, [\mathrm{Gyr}]$', fontsize=fontlabel)

                        if  XScaleSymlog:
                            axs[i][j].set_xscale('symlog')
                            axs[i][j].set_xlim(-0.5, 14.5)
                            axs[i][j].set_xticks([0, 1.97185714, 3.94371429,  5.91557143,  7.88742857, 9.85928571, 13.803 ])

                            axs[i][j].set_xticklabels(['14', '12', '10', '8', '6', '4', '0'])
                          

                        else:
                            axs[i][j].set_xlim(-0.5, 14.5)
                            axs[i][j].set_xticks([0.  ,  1.97185714,  3.94371429,  5.91557143,  7.88742857, 9.85928571, 11.83114286, 13.803  ])
                            if not (JustOneXlabel) and not (SmallerScale):
                                axs[i][j].set_xticklabels(
                                ['14', '12', '10', '8', '6', '4', '2', '0'])
                                
                            else:
                                if j == 0:
                                    axs[i][j].set_xticklabels(
                                    ['14', '12', '10', '8', '6', '4', '2', '0'])
                                else:
                                    axs[i][j].set_xticklabels(
                                    [ '', '12', '10', '8', '6', '4', '2', '0'])
                    elif not PhasingPlot:
                        axs[i][j].set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
                        if ( JustOneXlabel and i == 1):
                            axs[i][j].set_xlabel(r'$t \, \, [\mathrm{Gyr}]$', fontsize=fontlabel)
                        if not JustOneXlabel:
                            axs[i][j].set_xlabel(r'$t \, \, [\mathrm{Gyr}]$', fontsize=fontlabel)
                        if not (JustOneXlabel) :
                            axs[i][j].set_xticklabels(
                                ['0', '2', '4', '6', '8', '10', '12', '14'])
                        else:
                            if j == 0:
                                axs[i][j].set_xticklabels(
                                    ['0', '2', '4', '6', '8', '10', '12', '14'])
                            else:
                                axs[i][j].set_xticklabels(
                                    ['', '2', '4', '6', '8', '10', '12', '14'])
                    elif PhasingPlot:
                        limXparam = int(xPhaseLim + 1)
                        postiveXticks = np.arange(limXparam)
                        postiveXLabels = np.array([str(int(i)) for i in postiveXticks])

                        postiveXticks = np.append([-1, -0.5], postiveXticks)
                        postiveXLabels = np.append(['', 'E'], postiveXLabels)
                        axs[i][j].set_xlabel(r'$\phi_\mathrm{Orbital}$', fontsize=fontlabel)
                        axs[i][j].set_xticks(postiveXticks)
                        axs[i][j].set_xticklabels(postiveXLabels)
                        axs[i][j].set_xlim(-1, xPhaseLim+0.5)
                    axs[i][j].tick_params(axis='x', labelsize=multtick*fontlabel)

                elif Type == 'CoEvolution':
                    axs[i][j].set_xscale(scales.get(xparam, 'linear'))
                    if scales.get(xparam) == 'log':
                        axs[i][j].xaxis.set_major_formatter(
                            FuncFormatter(format_func_loglog))
                    if xlimmin != None and xlimmax != None:
                        axs[i][j].set_xlim(xlimmin[i], xlimmax[i])
                    axs[i][j].set_xlabel(labels.get(xparam, 'None'), fontsize=fontlabel)
                    axs[i][j].tick_params(axis='x', labelsize=multtick*fontlabel)
                    
    
    
        


    if Type == 'CoEvolution':
        cb =  fig.colorbar(plotLine,  ax=axs.ravel().tolist(), ticks=[0.  ,  1.97185714,  3.94371429,  5.91557143,  7.88742857, 9.85928571, 11.83114286, 13.803  ], pad=0.02, aspect = 50)
        cb.ax.set_yticklabels(['14', '12', '10', '8', '6', '4', '2', '0'])
      
        cb.set_label('Lookback Time [Gyr]', fontsize=1*fontlabel)
        cb.ax.tick_params(labelsize=multtick*fontlabel)
                    
    if Supertitle:
        plt.suptitle(Supertitle_Name, fontsize = 1.3*fontlabel, y=Supertitle_y)

    savefig(savepath, savefigname, Transparent)

    return


def PlotID(columns, rows, IDs, IDColumn = False, dataMarker=None, dataLine=None, Type='Evolution', Xparam='Time', 
           title=False, xlabelintext=False, lineparams=False, sSFRMedian = False,  TRANSPARENT = False, Softening = False,
           QuantileError=True, ColumnPlot=True, limaxis=False, LookBackTime = False, Pericenter = False,   legend=False, 
           postext = ['best'],  loc='best', SIM = SIMTNG, fmt = 'csv',
           savepath='PlotID', savefigname='fig', dfName='Sample', SampleName='Samples',LegendNames='None', 
           ylimmin = None, ylimmax = None, xlimmin = None, xlimmax = None, legpositions = None,
           lNum = 6, cNum = 6, GridMake = False, yscale ='linear',
           alphaShade=0.3,  linewidth=0.5, fontlabel=24,  nboots=100,    framealpha = 0.95,
           columnspacing = 0.5,handlelength = 2,  handletextpad = 0.4, labelspacing = 0.3, bins=10, seed=16010504):
    '''
    Plot teh evolution for random sample
    Parameters
    ----------
    columns : specific set in the sample / or different param to plot in each column. array with str
    rows : specific set in the sample / or different param to plot in each row. array with str
    IDs: IDs for selected subhalos. 
    The rest is the same as the previous functions
    Returns
    -------
    Requested Evolution or Co-Evolution plot
    -------
    Author: Abhner P. de Almeida (abhner.almeida AAT usp.br)
    '''

    np.random.seed(seed)

    dfTime = pd.read_csv(os.getenv("HOME")+"/TNG_Analyzes/SubhaloHistory/SNAPS_TIME.csv")
    snapsTime = np.array([88, 81, 64, 51, 37, 24])
    # Verify NameParameters
    if type(columns) is not list and type(columns) is not np.ndarray:
        columns = [columns]

    if type(rows) is not list and type(rows) is not np.ndarray:
        rows = [rows]

    # Define axes
    plt.rcParams.update({'figure.figsize': (cNum*len(columns), lNum*len(rows))})
    fig = plt.figure()
    gs = fig.add_gridspec(len(rows), len(columns), hspace=0, wspace=0)
    axs = gs.subplots(sharex='col', sharey='row')
    
    if Pericenter:
        r_over_R_Crit200 = TNG.extractDF('r_over_R_Crit200', SIM = SIM, fmt = fmt)

    # Verify axs shape
    if type(axs) is not list and type(axs) is not np.ndarray:
        axs = [axs]
    if type(axs[0]) is not np.ndarray:
        axs = np.array([axs])
        if len(columns) == 1:
            axs = axs.T

    time = dfTime.Age.values


    for i, row in enumerate(rows):

        for j, column in enumerate(columns):
            if ColumnPlot:
    
                argIDs = i  
                data = TNG.makeDF(row, column,  dfName = dfName, IDs=IDs[argIDs], SIM = SIM)
                if Type == 'CoEvolution':
                    dataX = TNG.makeDF(Xparam[j], dfName = dfName, IDs=IDs[argIDs], SIM = SIM)
            else:
                argIDs = j
                data = TNG.makeDF(column,  row,  dfName = dfName, IDs=IDs[argIDs], SIM = SIM)
                if Type == 'CoEvolution':
                    dataX = TNG.makeDF(column, Xparam[i],  dfName = dfName, IDs=IDs[argIDs], SIM = SIM)

            if dataLine is not None:
                datalinevalues = TNG.makeDF(column, dataLine,  dfName = dfName, IDs=IDs[argIDs], SIM = SIM)
            if dataMarker is not None:
                if 'Merger' in dataMarker:
                    datamarkerTotvalues = TNG.makeDF(column,  'NumMergersTotal',dfName = dfName, IDs=IDs[argIDs], SIM = SIM)
                    dataMarkervalues = TNG.makeDF(column, 'NumMajorMergersTotal',  dfName = dfName, IDs=IDs[argIDs], SIM = SIM)
                    datamarkervalues = TNG.makeDF(column, 'NumMinorMergersTotal', dfName = dfName, IDs=IDs[argIDs], SIM = SIM)
                else:
                    datamarkervalues = TNG.makeDF(column, dataMarker, dfName = dfName, IDs=IDs[argIDs], SIM = SIM)

            if Softening and row == 'SubhaloHalfmassRadType4':
                rSoftening = ETNG.Softening()
                rSoftening = np.flip(rSoftening)
                axs[i][j].plot(time[(~np.isinf(rSoftening))], np.log10(rSoftening[(~np.isinf(rSoftening))]), 
                               color='black', ls='solid', lw=2*linewidth)

            if sSFRMedian and row == 'SubhalosSFRInHalfRad':
                Y, Yerr = TNG.makedataevolution([''], ['Central'], ['SubhalosSFRInHalfRad'], SampleName=SampleName, dfName = dfName, nboots=nboots)
                Y = np.array([value for value in Y[0][0][0]])
                Yerr = np.array([value for value in Yerr[0][0][0]])
                axs[i][j].plot(time, Y, 
                               color='grey', ls='solid', lw=2*linewidth)    
                axs[i][j].fill_between(time, Y - 4*Yerr, 
                               Y+ 4*Yerr,
                               color='grey', alpha = 0.5)    
                    
            for l, IDvalue in enumerate(IDs[argIDs]):
                values = np.array(
                    [value for value in data[str(IDvalue)].values])
                if len(values.shape) > 1:
                    values = values.T[0]

                if Type == 'Evolution':
                    if row == 'r_over_R_Crit200_WithoutCorrection':
                        values[values == 0] = np.nan
                    argnotnan = ~np.isnan(values)
                    
                    if Xparam[i] == 'tsincebirth':
                        TimeBirth = time[argnotnan] - time[argnotnan][-1]
                        axs[i][j].plot(TimeBirth, values[argnotnan], color=colors.get(
                                str(l), 'black'),  ls=lines.get(str(l), 'solid'), lw=linewidth)
                    else:
                        axs[i][j].plot(time[argnotnan], values[argnotnan], color=colors.get(
                                str(l), 'black'),  ls=lines.get(str(l), 'solid'), lw=linewidth)
                            
                    if Pericenter : #and not row == 'r_over_R_Crit200':
                        rOveR200 = np.array([value for value in r_over_R_Crit200[str(IDvalue)].values])
                        rOveR200[rOveR200 > 1] = np.nan
                        args = argrelextrema(rOveR200, np.less)[0]
                        for arg in args:
                            #axs[i][j].arrow(time[arg], limmax.get(row+'Pericenter'), 0, -limin.get(row+'Pericenter'), color=colors.get(
                            #    str(l), 'black'),  ls=lines[column], lw=1.7*linewidth, head_width = 0)
                            axs[i][j].scatter(time[arg], values[arg],color=colors.get(str(l), 'black'), marker = 'X', s = 30, edgecolor = 'black' )
                        
                    if dataLine is not None:
                        linevalues = np.array(
                            [value for value in datalinevalues[str(IDvalue)].values])
                        if len(linevalues.shape) > 1:
                            linevalues = linevalues.T[0]
                            linevalues = np.array(
                                [value for value in linevalues])
                        axs[i][j].plot(time[(~np.isinf(linevalues)) & (~np.isnan(linevalues))], values[(~np.isinf(
                            linevalues)) & (~np.isnan(linevalues))], color=colors.get(str(l), 'black'), ls=lines.get(str(l), 'solid'), lw=2*linewidth)

                    if dataMarker is not None:
                        markervalues = np.array(
                            [value for value in datamarkervalues[str(IDvalue)].values])
                        if len(markervalues.shape) > 1:
                            markervalues = markervalues.T[0]
                            markervalues = np.array(
                                [value for value in markervalues])

                        if 'Merger' in dataMarker:
                            mergerTot = np.array(
                                [value for value in datamarkerTotvalues[str(IDvalue)].values])
                            if len(mergerTot.shape) > 1:
                                mergerTot = mergerTot.T[0]
                                mergerTot = np.array(
                                    [value for value in mergerTot])

                            MarkerTotvalues = np.array(
                                [value for value in datamarkerTotvalues[str(IDvalue)].values])
                            if len(MarkerTotvalues.shape) > 1:
                                MarkerTotvalues = MarkerTotvalues.T[0]
                                MarkerTotvalues = np.array(
                                    [value for value in MarkerTotvalues])

                            mergernumber = np.array(
                                [value for value in datamarkervalues[str(IDvalue)].values])
                            if len(mergernumber.shape) > 1:
                                mergernumber = mergernumber.T[0]
                                mergernumber = np.array(
                                    [value for value in mergernumber])

                            Mergernumber = np.array(
                                [value for value in dataMarkervalues[str(IDvalue)].values])
                            if len(Mergernumber.shape) > 1:
                                Mergernumber = Mergernumber.T[0]
                                Mergernumber = np.array(
                                    [value for value in Mergernumber])

                            Markervalues = np.array(
                                [value for value in datamarkervalues[str(IDvalue)].values])
                            if len(Markervalues.shape) > 1:
                                Markervalues = Markervalues.T[0]
                                Markervalues = np.array(
                                    [value for value in Markervalues])

                            mergernumber = np.flip(mergernumber)
                            Mergernumber = np.flip(Mergernumber)
                            mergerTot = np.flip(mergerTot)

                            for nmergerindex, nmerger in enumerate(mergernumber):

                                if nmergerindex == 0:
                                    markervalues[nmergerindex] = 0
                                    continue
                                else:
                                    if np.isnan(nmerger):
                                        markervalues[nmergerindex] = 0
                                    else:
                                        if np.isnan(mergernumber[nmergerindex - 1]):
                                            markervalues[nmergerindex] = int(
                                                nmerger)
                                        else:
                                            markervalues[nmergerindex] = int(
                                                nmerger) - int(mergernumber[nmergerindex - 1])

                            for nmergerindex, nmerger in enumerate(Mergernumber):

                                if nmergerindex == 0:
                                    Markervalues[nmergerindex] = 0
                                    continue
                                else:
                                    if np.isnan(nmerger):
                                        Markervalues[nmergerindex] = 0
                                    else:
                                        if np.isnan(mergernumber[nmergerindex - 1]):
                                            Markervalues[nmergerindex] = int(
                                                nmerger)
                                        else:
                                            Markervalues[nmergerindex] = int(
                                                nmerger) - int(Mergernumber[nmergerindex - 1])

                            for nmergerindex, nmerger in enumerate(mergerTot):

                                if nmergerindex == 0:
                                    MarkerTotvalues[nmergerindex] = 0
                                    continue
                                else:
                                    if np.isnan(nmerger):
                                        MarkerTotvalues[nmergerindex] = 0
                                    else:
                                        if np.isnan(mergernumber[nmergerindex - 1]):
                                            MarkerTotvalues[nmergerindex] = int(
                                                nmerger)
                                        else:
                                            MarkerTotvalues[nmergerindex] = int(
                                                nmerger) - int(mergerTot[nmergerindex - 1])
                            MarkerTotvalues = MarkerTotvalues - Markervalues - markervalues
                            Markervalues = np.flip(Markervalues)
                            markervalues = np.flip(markervalues)
                            MarkerTotvalues = np.flip(MarkerTotvalues)

                        axs[i][j].scatter(time[(Markervalues > 0)], values[(Markervalues > 0)], color=colors.get(
                            str(l), 'black'), lw=1., marker='o',  edgecolors='black', s=250, alpha=0.7)
                        axs[i][j].scatter(time[(markervalues > 0)], values[(markervalues > 0)], color=colors.get(
                            str(l), 'black'), lw=1., marker='s',  edgecolors='black', s=100, alpha=0.7)
                        axs[i][j].scatter(time[(MarkerTotvalues > 0)], values[(MarkerTotvalues > 0)], color=colors.get(
                            str(l), 'black'), lw=1., marker='s',  edgecolors='black', s=100, alpha=0.7)

                elif Type == 'CoEvolution':
                    x = dataX[str(IDvalue)].values
                    if len(x.shape) > 1:
                        x = np.array([value for value in x.T[0]])
                    else:
                        x = np.array([value for value in x])
                    colorSnap = np.array(
                        ['magenta', 'blue', 'cyan', 'lime', 'darkorange', 'red'])
                    if Xparam[i] != 'tsincebirth':
                        axs[i][j].scatter(x[99-snapsTime], values[99-snapsTime], color=colorSnap,
                                          lw=1., marker='d',  edgecolors=colors.get(column), s=100, alpha=0.9)
                        axs[i][j].scatter(x[0], values[0], color='black', lw=1.,
                                          marker='o',  edgecolors=colors.get(column, 'black'), s=70, alpha=0.9)
                    argnotnan = ~np.isnan(values)
                    axs[i][j].plot(x[argnotnan], values[argnotnan], color=colors.get(
                        str(l), 'black'), ls=lines.get(column, 'solid'))

                    if dataLine is not None:
                        linevalues = np.array(
                            [value for value in datalinevalues[str(IDvalue)].values])
                        if len(linevalues.shape) > 1:
                            linevalues = linevalues.T[0]
                            linevalues = np.array(
                                [value for value in linevalues])
                        axs[i][j].plot(x[(~np.isinf(linevalues)) & (~np.isnan(linevalues))], values[(~np.isinf(linevalues)) & (
                            ~np.isnan(linevalues))], color=colors.get(str(l), 'black'), ls=lines.get(str(l), 'solid'), lw=3.)

                    if dataMarker is not None:
                        markervalues = np.array(
                            [value for value in datamarkervalues[str(IDvalue)].values])
                        if len(markervalues.shape) > 1:
                            markervalues = markervalues.T[0]
                            markervalues = np.array(
                                [value for value in markervalues])

                        if 'Merger' in dataMarker:
                            mergerTot = np.array(
                                [value for value in datamarkerTotvalues[str(IDvalue)].values])
                            if len(mergerTot.shape) > 1:
                                mergerTot = mergerTot.T[0]
                                mergerTot = np.array(
                                    [value for value in mergerTot])

                            MarkerTotvalues = np.array(
                                [value for value in datamarkerTotvalues[str(IDvalue)].values])
                            if len(MarkerTotvalues.shape) > 1:
                                MarkerTotvalues = MarkerTotvalues.T[0]
                                MarkerTotvalues = np.array(
                                    [value for value in MarkerTotvalues])

                            mergernumber = np.array(
                                [value for value in datamarkervalues[str(IDvalue)].values])
                            if len(mergernumber.shape) > 1:
                                mergernumber = mergernumber.T[0]
                                mergernumber = np.array(
                                    [value for value in mergernumber])

                            Mergernumber = np.array(
                                [value for value in dataMarkervalues[str(IDvalue)].values])
                            if len(Mergernumber.shape) > 1:
                                Mergernumber = Mergernumber.T[0]
                                Mergernumber = np.array(
                                    [value for value in Mergernumber])

                            Markervalues = np.array(
                                [value for value in datamarkervalues[str(IDvalue)].values])
                            if len(Markervalues.shape) > 1:
                                Markervalues = Markervalues.T[0]
                                Markervalues = np.array(
                                    [value for value in Markervalues])

                            mergernumber = np.flip(mergernumber)
                            Mergernumber = np.flip(Mergernumber)
                            mergerTot = np.flip(mergerTot)

                            for nmergerindex, nmerger in enumerate(mergernumber):

                                if nmergerindex == 0:
                                    markervalues[nmergerindex] = 0
                                    continue
                                else:
                                    if np.isnan(nmerger):
                                        markervalues[nmergerindex] = 0
                                    else:
                                        if np.isnan(mergernumber[nmergerindex - 1]):
                                            markervalues[nmergerindex] = int(
                                                nmerger)
                                        else:
                                            markervalues[nmergerindex] = int(
                                                nmerger) - int(mergernumber[nmergerindex - 1])

                            for nmergerindex, nmerger in enumerate(Mergernumber):

                                if nmergerindex == 0:
                                    Markervalues[nmergerindex] = 0
                                    continue
                                else:
                                    if np.isnan(nmerger):
                                        Markervalues[nmergerindex] = 0
                                    else:
                                        if np.isnan(mergernumber[nmergerindex - 1]):
                                            Markervalues[nmergerindex] = int(
                                                nmerger)
                                        else:
                                            Markervalues[nmergerindex] = int(
                                                nmerger) - int(Mergernumber[nmergerindex - 1])

                            for nmergerindex, nmerger in enumerate(mergerTot):

                                if nmergerindex == 0:
                                    MarkerTotvalues[nmergerindex] = 0
                                    continue
                                else:
                                    if np.isnan(nmerger):
                                        MarkerTotvalues[nmergerindex] = 0
                                    else:
                                        if np.isnan(mergernumber[nmergerindex - 1]):
                                            MarkerTotvalues[nmergerindex] = int(
                                                nmerger)
                                        else:
                                            MarkerTotvalues[nmergerindex] = int(
                                                nmerger) - int(mergerTot[nmergerindex - 1])
                            MarkerTotvalues = MarkerTotvalues - Markervalues - markervalues
                            Markervalues = np.flip(Markervalues)
                            markervalues = np.flip(markervalues)
                            MarkerTotvalues = np.flip(MarkerTotvalues)

                        axs[i][j].scatter(x[(Markervalues > 0)], values[(Markervalues > 0)], color=colors.get(
                            str(l), 'black'), lw=1., marker='o',  edgecolors='black', s=130, alpha=0.5)
                        axs[i][j].scatter(x[(markervalues > 0)], values[(markervalues > 0)], color=colors.get(
                            str(l), 'black'), lw=1., marker='o',  edgecolors='black', s=110, alpha=0.5)
                        #axs[i][j].scatter(x[(MarkerTotvalues > 0)], values[(MarkerTotvalues > 0)], color=colors.get(
                            #str(l), 'black'), lw=1., marker='o',  edgecolors='black', s=15, alpha=0.5)

            # Plot details
            if GridMake:
                axs[i][j].grid(GridMake, color='#9e9e9e',  which="major", linewidth= 0.6,alpha= 0.3 , linestyle=':')
                
            axs[i][j].tick_params(axis='y', labelsize=0.99*fontlabel)
            axs[i][j].tick_params(axis='x', labelsize=0.99*fontlabel)

            if ylimmin != None and ylimmax != None:
                    axs[i][j].set_ylim(ylimmin[i], ylimmax[i])
	    
            if ColumnPlot:
                axs[i][j].set_yscale(scales.get(column, yscale))
                if scales.get(column, yscale) == 'log':
                    axs[i][j].yaxis.set_major_formatter(
                        FuncFormatter(format_func_loglog))
            else:
                axs[i][j].set_yscale(scales.get(row, yscale))
                if scales.get(row, yscale) == 'log':
                    axs[i][j].yaxis.set_major_formatter(
                        FuncFormatter(format_func_loglog))

            if legend:
                for legpos, LegendName in enumerate(LegendNames):
                    if j == legpositions[legpos][0] and i ==legpositions[legpos][1]:
                        custom_lines, label, ncol, mult = Legend(
                            LegendName)
                        axs[i][j].legend(
                            custom_lines, label, ncol=ncol, loc=loc[legpos], fontsize=0.88*fontlabel, framealpha = framealpha, 
                            columnspacing = columnspacing, handlelength = handlelength, handletextpad = handletextpad, labelspacing = labelspacing)

            if j == 0:

                if xlabelintext:
                    axs[i][j].set_ylabel(
                        labelsequal.get(row), fontsize=fontlabel)

                else:
                    if ColumnPlot:
                        axs[i][j].set_ylabel(labels.get(column), fontsize=fontlabel)
                    else:
                        axs[i][j].set_ylabel(labels.get(row), fontsize=fontlabel)

            if j == len(columns) - 1:
                if xlabelintext and not limaxis and len(rows) > 1:
                    Afont = {'color':  'black',
                             'size': fontlabel,
                             }
                    anchored_text = AnchoredText(
                        texts.get(row), loc='upper right', prop=Afont)
                    axs[i][j].add_artist(anchored_text)

            if xlabelintext and limaxis and len(rows) > 1:
                Afont = {'color':  'black',
                         'size': fontlabel,
                         }
                anchored_text = AnchoredText(
                    texts[row], loc='upper left', prop=Afont)
                axs[i][j].add_artist(anchored_text)
                
            if j == 0 and len(rows) > 1:
                if title and ColumnPlot:
                    Afont = {'color':  'black',
                             'size': fontlabel,
                             }
                    anchored_text = AnchoredText(
                        titles.get(
                            title[i], title[i]), loc=postext[i], prop=Afont)
                    axs[i][j].add_artist(anchored_text)


            if i == 0:

                if title and not ColumnPlot:
                    axs[i][j].set_title(titles.get(
                        title[j], title[j]), fontsize=1.1*fontlabel)
                

                if Type == 'Evolution' and Xparam[i] != 'tsincebirth':

                    axs[i][j].tick_params(bottom=True, top=False)
                    lim = axs[i][j].get_xlim()
                    ax2label = axs[i][j].twiny() #secondary_xaxis('top', which='major')
                    ax2label.grid(False)
                    ax2label.set_xlim(lim)

                    if row == ('rToRNearYoung' or savefigname == 'Young') :
                        zticks = np.array([0., 0.2])
                        zlabels = np.array(
                            ['0', '0.2'])
                        zticks_Age = np.array(
                            [13.803, 11.323])
                    else:
                        zticks = np.array([0., 0.2, 0.5, 1., 2., 5., 20.])
                        zlabels = np.array(
                            ['0', '0.2', '0.5', '1', '2', '5', '20'])
                        zticks_Age = np.array(
                            [13.803, 11.323, 8.587, 5.878, 3.285, 1.2, 0])


                    zticks = zticks.tolist()
                    zticks_Age = zticks_Age.tolist()

                    x_locator = FixedLocator(zticks_Age)
                    x_formatter = FixedFormatter(zlabels)
                    ax2label.xaxis.set_major_locator(x_locator)
                    ax2label.xaxis.set_major_formatter(x_formatter)
                    ax2label.set_xlabel(r"$z$", fontsize=fontlabel)
                    ax2label.tick_params(labelsize=0.99*fontlabel)

            if i == len(rows) - 1:
                if Type == 'Evolution':

                    axs[i][j].set_xlabel(r'$t \, \,  [\mathrm{Gyr}]$', fontsize=fontlabel)
                    if row == 'rToRNearYoung' or savefigname == 'Young':
                        axs[i][j].set_xticks([10, 12, 14])
                        axs[i][j].set_xticklabels(
                            ['10', '12', '14'])
                        if Xparam[i] == 'tsincebirth':
                            axs[i][j].set_xticks([0, 1, 2, 3, 4])
                            axs[i][j].set_xticklabels(
                                ['0', '1', '2', '3', '4'])
                            axs[i][j].set_xlabel(r'$t - t_\mathrm{birth} [\mathrm{Gyr}]$', fontsize=fontlabel)
                            axs[i][j].set_xlim(-0.09, 4.2)

                    else:
                        if LookBackTime:
                            axs[i][j].set_xlabel(r'$\mathrm{Lookback \; Time} \, \, [\mathrm{Gyr}]$', fontsize=fontlabel)
                            axs[i][j].set_xticks([0.  ,  1.97185714,  3.94371429,  5.91557143,  7.88742857, 9.85928571, 11.83114286, 13.803  ])

                            axs[i][j].set_xticklabels(
                                ['14', '12', '10', '8', '6', '4', '2', '0'])
                        else:
                            axs[i][j].set_xticks([0, 2, 4, 6, 8, 10, 12, 14])

                            axs[i][j].set_xlabel(r'$t \, \, [\mathrm{Gyr}]$', fontsize=fontlabel)
        
                            axs[i][j].set_xticklabels(
                                ['0', '2', '4', '6', '8', '10', '12', '14'])

                elif Type == 'CoEvolution':
                    axs[i][j].set_xscale(scales.get(Xparam[i], 'linear'))
                    if scales.get(Xparam[i]) == 'log':
                        axs[i][j].xaxis.set_major_formatter(
                            FuncFormatter(format_func_loglog))
                    axs[i][j].set_xlabel(labels.get(
                        Xparam[i], 'None'), fontsize=fontlabel)

    savefig(savepath, savefigname, TRANSPARENT = TRANSPARENT, SIM = SIM)

    return

def PlotIDsColumns(IDs, rows, dataMarker=None, dataLine=None, SatelliteTime = False, 
                   PhasingPlot = False, ShowPop = False, ShowPopName = 'Normal', SnapTransition = False, 
                   SnapTransitionName = '',
                   title=False, xlabelintext=False, lineparams=False,  QuantileError=True, 
           alphaShade=0.3,  linewidth=0.5, fontlabel=24, nboots=100,  ColumnPlot=False, limaxis=False, 
           columnspacing = 0.5, handlelength = 2, handletextpad = 0.4, labelspacing = 0.3, LookBackTime = False, Pericenter = False, postext = ['best'],
           ylimmax = None, ylimmin = None, GridMake = False, CompareToNormal = False,
           lNum = 6, cNum = 6, InfallTime = False, NoGas = False, SmallerScale = False,
           Type='Evolution', Xparam='Time', savepath='fig/PlotIDColumns', savefigname='fig', dfName='Sample', SampleName='Samples', legend=False, LegendNames='None',  loc='best',
           bins=10, seed=16010504, TRANSPARENT = False, Softening = False, MaxSizeType = False):
    
    
    '''
    Plot teh evolution for random sample
    Parameters
    ----------
    columns : specific set in the sample / or different param to plot in each column. array with str
    rows : specific set in the sample / or different param to plot in each row. array with str
    IDs: IDs for selected subhalos. 
    The rest is the same as the previous functions
    Returns
    -------
    Requested Evolution or Co-Evolution plot
    -------
    Author: Abhner P. de Almeida (abhner.almeida AAT usp.br)
    '''

    np.random.seed(seed)

    dfTime = pd.read_csv(os.getenv("HOME")+"/TNG_Analyzes/SubhaloHistory/SNAPS_TIME.csv")
    Sample = TNG.extractPopulation(dfName, dfName = dfName)

    snapsTime = np.array([88, 81, 64, 51, 37, 24])
    # Verify NameParameters
    if type(IDs) is not list and type(IDs) is not np.ndarray:
        IDs = [IDs]

    if type(rows) is not list and type(rows) is not np.ndarray:
        rows = [rows]

    # Define axes(cNum*len(columns), lNum*len(rows))})
    plt.rcParams.update({'figure.figsize': (lNum*len(IDs), cNum*len(rows))})
    fig = plt.figure()
    gs = fig.add_gridspec(len(rows), len(IDs), hspace=0, wspace=0)
    axs = gs.subplots(sharex='col', sharey='row')
    
    if Pericenter:
        r_over_R_Crit200 = TNG.extractDF('r_over_R_Crit200')

   
    # Verify axs shape
    if type(axs) is not list and type(axs) is not np.ndarray:
        axs = [axs]
    if type(axs[0]) is not np.ndarray:
        axs = np.array([axs])
        if len(IDs) == 1:
            axs = axs.T

    time = dfTime.Age.values

    for i, row in enumerate(rows):
        if type(row) is not list and type(row) is not np.ndarray:
            row = [row]
        
        dfs = []
        Ys = []
        Yerrs = []
        for param in row:
            dfs.append(TNG.extractDF(param))
            if CompareToNormal:
                Y, Yerr = TNG.makedataevolution(['Normal'], [''], [param], SampleName=SampleName, dfName = dfName, nboots=nboots)
                Yerr = np.array([value for value in Yerr[0][0][0]])
                Y = np.array([value for value in Y[0][0][0]])
                Ys.append(Y)
                Yerrs.append(Yerr)
                
       
            
        if Type == 'CoEvolution':
            dfX = TNG.extractDF(Xparam[i]) 
        
        if dataLine is not None:
            datalinevalues = TNG.extractDF(dataLine) 

        if dataMarker is not None:
            if 'Merger' in dataMarker:
                datamarkerTotvalues = TNG.extractDF('NumMergersTotal') 
                dataMarkervalues =TNG.extractDF('NumMajorMergersTotal') 
                datamarkervalues = TNG.extractDF('NumMinorMergersTotal')               
            else:
                datamarkervalues = TNG.extractDF(dataMarker) 

        
        for j, ID in enumerate(IDs):
            
            if j == 0:
               
                if i > 0 and ('SubhaloHalfmassRadType0' in rows[i - 1][0] or  'StarMass_In_Rhpkpc' in rows[i - 1][0] ) and 'Mgas_Norm_Max' in row[0]:
                    None
                elif 'StarMass_In_Rhpkpc' in rows[i - 1][0] :
                    None
                elif legend and LegendNames !='None':
                        if len(LegendNames) <= i  :
                            None
                        else:
                            custom_lines, label, ncol, mult = Legend(LegendNames[i])
    
                            axs[i][j].legend(
                                   custom_lines, label, ncol=ncol, loc=loc, fontsize=0.88*fontlabel, 
                                  columnspacing = columnspacing, handlelength = handlelength, handletextpad = handletextpad, labelspacing = labelspacing)
                            loc = 'best'
                    
                else:
                    
                    loc = 'best'   
                    if row == ['SubhaloStellarMass_in_Rhpkpc', 'SubhaloStellarMass_Above_Rhpkpc', 'SubhaloGasMass_in_Rhpkpc', 'SubhaloGasMass_Above_Rhpkpc']:
                        custom_lines, label, ncol, mult = Legend(['in_Rhpkpc', 'Above_'])

                    elif row == ['SubhalosSFRInHalfRad', 'SubhalosSFRwithinHalfandRad']:

                        custom_lines, label, ncol, mult = Legend(['SubhalosSFRInHalfRad', 'SubhalosSFRwithinHalfandRad'])
                        loc = 'best'
                    elif len(row) > 1:
                        namesrow = [namerow for namerow in row]
                        for index, namerow in enumerate(namesrow):
                            namesrow[index] = namerow+'IDsColumn'
                        custom_lines, label, ncol, mult = Legend(namesrow)
                    
                    
                    if legend and not (row == ['r_over_R_Crit200_WithoutCorrection', 'r_over_R_Crit200'] or row == ['sSFR_In_TrueRhpkpc', 'sSFR_Above_TrueRhpkpc'] or row == ['SFR_In_Rhpkpc', 'SFR_Above_Rhpkpc'] or row == ['logStar_GFM_Metallicity_In_Rhpkpc', 'logStar_GFM_Metallicity_Above_Rhpkpc'] or row == ['sSFR_In_Rhpkpc', 'sSFR_Above_Rhpkpc'] ) and len(row) > 1: # or row == ['sSFR_In_Rhpkpc', 'sSFR_Above_Rhpkpc']
                        
                        axs[i][j].legend(
                               custom_lines, label, ncol=ncol, loc=loc, fontsize=0.88*fontlabel, 
                              columnspacing = columnspacing, handlelength = handlelength, handletextpad = handletextpad, labelspacing = labelspacing)
                        loc = 'best'

            if Softening and 'SubhaloHalfmassRadType4' in row:
                rSoftening = ETNG.Softening()
                rSoftening = np.flip(rSoftening)
                axs[i][j].plot(time[(~np.isinf(rSoftening))], np.log10(rSoftening[(~np.isinf(rSoftening))]), 
                               color='black', ls='solid', lw=2*linewidth)
            
            for l, df in enumerate(dfs):
                #Y = Ys[l]
                #Yerr = Yerrs[l]
                values = np.array([value for value in df[str(ID)].values])
                if Type == 'Evolution':
                    if row[l] == 'r_over_R_Crit200_FirstGroup':
                        values[values == 0] = np.nan
                        arg = np.argwhere(np.isnan(values)).T[0]
                        values[arg[0]:] = np.nan
                    
                    if 'Type4' in row[l] or 'star' in row[l] and not 'HalfRad' in row[l]:
                        color = 'blue'
                        ls = 'solid'
                    elif 'Type0' in row[l] or ('gas' in row[l] and (not '_in_' in row[l] and not '_Above_' in row[l]) ):
                        color = 'green'
                        ls = 'solid'
                    elif 'Type1' in row[l] or 'DM' in row[l]:
                        color = 'purple'
                        ls = 'solid'
                    elif 'SubhalosSFRInHalfRad' in row[l]:
                        color = 'darkblue'
                        ls = 'solid'
                    elif 'SubhalosSFRwithinHalfandRad' in row[l]:
                        color = 'darkred'
                        ls = (0, (10, 8))
                    elif ('r_over_R_Crit200_FirstGroup' in row[l] ) or ('Group_M_Crit200' in row[l]):
                        color = 'red'
                        ls = 'dashed'
                    elif 'r_over_R_Crit200' in row[l]:
                        color = 'darkorange'
                        ls = 'solid'
                       
                    elif ('in_Rhpkpc' in row[l] or 'In_TrueRhpkpc' in row[l] or   'In_Rhpkpc' in row[l]) and not ('Inflow'  in row[l] or 'Outflow' in row[l] or 'Rhpkpc_entry'  in row[l]):
                        color = 'darkblue'
                        ls = 'solid'
                    elif ('Above_Rhpkpc' in row[l] or 'Above_TrueRhpkpc' in row[l]) and not ('Inflow'  in row[l] or 'Outflow' in row[l] ):
                        color = 'tab:blue'
                        ls =  (0, (10, 6))

                    else:
                        color = colors.get(row[l], 'black')
                        ls = lines.get(row[l], 'solid')
        
                    if CompareToNormal:
                        values[~np.isnan(values)] = (values[~np.isnan(values)] - Y[~np.isnan(values)]) / Yerr[~np.isnan(values)]
        
                    if PhasingPlot :
                        xparam = np.arange(-1, 9)
                        xparam = np.append(xparam, xparam+0.5)
                        xparam = np.append(xparam, np.linspace(-1, 9, 1000))
                        xparam = np.unique(xparam)
                        values = np.flip(values)
                        dfPopulation = TNG.extractPopulation(dfName, dfName = dfName)
                        
                        phases = TNG.PhasingData(ID, dfPopulation)
                        
                        if type(phases) != np.ndarray:
                            continue
                        phases = phases[(~np.isnan(values)) & (~np.isinf(values))]
                        values = values[(~np.isnan(values)) & (~np.isinf(values))]
                        if len(values) == 0:
                            continue
                        X_Y_Spline = interp1d(phases, values,kind="linear",fill_value="extrapolate")
                        values = X_Y_Spline(xparam)
                        if phases.max() < 8:
                            values[xparam > phases.max()] = np.nan
                        else:
                            values[xparam > 4] = np.nan

                    else:
                        xparam = time

                    if ShowPop:
                        Y, Yerr = TNG.makedataevolution([ShowPopName], [''], [row[l]], SampleName=SampleName, dfName = dfName, nboots=nboots)
                        Yerr = np.array([value for value in Yerr[0][0][0]])
                        Y = np.array([value for value in Y[0][0][0]])
                       
                        if ('Gas' in row[l] or 'SFR' in row[l] or 'Type0' in row[l]):
                            
                            dfPop = TNG.extractPopulation(ShowPopName, dfName = dfName)
                            if ~np.isnan(np.nanmedian(dfPop.SnapLostGas)) and np.nanmedian(dfPop.SnapLostGas) > 0:
                                Y[xparam > dfTime.Age.loc[dfTime.Snap == int(np.nanmedian(dfPop.SnapLostGas))].values[0]] = np.nan
                     
                        
                        axs[i][j].plot(xparam[~np.isnan(Y)], Y[~np.isnan(
                            Y)], color=colors.get(ShowPopName, 'black'), ls=ls, 
                            lw=1.*linesthicker.get(ShowPopName, linewidth), dash_capstyle = capstyles.get(ShowPopName, 'projecting'))
            

                        axs[i][j].fill_between(
                            xparam[~np.isnan(Y)], Y[~np.isnan(Y)] - Yerr[~np.isnan(Y)], 
                            Y[~np.isnan(Y)] + Yerr[~np.isnan(Y)], color=colors.get(ShowPopName+'Error', 'black'), ls=ls, alpha=alphaShade)
                         
                    axs[i][j].plot(xparam[~np.isnan(values)], values[~np.isnan(values)], color=color,  ls=ls, lw=linewidth)

                    if Pericenter :#and not row == 'r_over_R_Crit200':
                        snapFirstPeri = Sample['SnapFirstPeri'].loc[Sample.SubfindID == ID].values[0]
                        SnapSecondPeri = Sample['SnapSecondPeri'].loc[Sample.SubfindID == ID].values[0]
                        SnapThirdPeri = Sample['SnapThirdPeri'].loc[Sample.SubfindID == ID].values[0]
                        SnapFirstApo = Sample['SnapFirstApo'].loc[Sample.SubfindID == ID].values[0]
                        SnapSecondApo = Sample['SnapSecondApo'].loc[Sample.SubfindID == ID].values[0]
                        
                        if ~np.isnan(SnapThirdPeri):
                            Peris = np.array([99-int(snapFirstPeri), 99-int(SnapSecondPeri), 99-int(SnapThirdPeri)])
                        elif ~np.isnan(SnapSecondPeri):
                            Peris = np.array([99-int(snapFirstPeri), 99-int(SnapSecondPeri)])
                        elif ~np.isnan(snapFirstPeri):
                            Peris = np.array([99-int(snapFirstPeri)])
                            
                        if ~np.isnan(snapFirstPeri):
                            axs[i][j].scatter(time[Peris], values[Peris],color='red', marker = 'x', s = 30, edgecolor = 'black' )
                        
                        if ~np.isnan(SnapSecondApo):
                            Apos = np.array([99-int(SnapFirstApo), 99-int(SnapSecondApo)])
                        elif ~np.isnan(SnapFirstApo):
                            Apos = np.array([99-int(SnapFirstApo)])
                            
                        if ~np.isnan(SnapFirstApo):
                            axs[i][j].scatter(xparam[Apos], values[Apos],color='black', marker = 'x', s = 30, edgecolor = 'black' )

                    if InfallTime:
                        
                        infallsnap = Sample.loc[Sample.SubfindID_99 == ID, 'Snap_At_FirstEntry'].values[0]
                        infallsnap = float(infallsnap)
                        if ~np.isnan(infallsnap) and infallsnap > 0:
                            infallsnap = int(99-infallsnap)
                            axs[i][j].axvline(xparam[infallsnap], color='black', ls = (0, (10, 8)))
                            
                    if SnapTransition:
                        
                        infallsnap = Sample.loc[Sample.SubfindID_99 == ID, SnapTransitionName].values[0]
                        if ~np.isnan(infallsnap) and infallsnap > 0:
                            infallsnap = int(99-infallsnap)
                            axs[i][j].axvline(xparam[infallsnap], color='red', ls = (0, (10, 8)))

                    if SatelliteTime and 'Group_M_Crit200' in param:
                        
                        infallsnap = Sample.loc[Sample.SubfindID_99 == ID, 'SnapBecomeSatellite'].values[0]
                        if ~np.isnan(infallsnap) and infallsnap > 0:
                            axs[i][j].scatter(xparam[int(99-infallsnap)], values[int(99-infallsnap)], marker = '*', s = 220, color = 'red')

                    if NoGas:
                       infallsnap =  Sample.loc[Sample.SubfindID_99 == ID, 'SnapLostGas'].values[0]
                        
                       if  ~np.isnan(infallsnap) and infallsnap > 0:
                            infallsnap = int(99-infallsnap)
                            axs[i][j].axvspan(xparam[infallsnap], time[0], color='pink', alpha=0.5, lw=0)
                        
                    if MaxSizeType :
                         MaxSize = Sample['MaxSizeType4'].loc[Sample.SubfindID == ID].values[0]
                         axs[i][j].axhline(MaxSize)
                         
                    if dataLine is not None:
                        linevalues = np.array(
                            [value for value in datalinevalues[str(ID)].values])
                        if len(linevalues.shape) > 1:
                            linevalues = linevalues.T[0]
                            linevalues = np.array(
                                [value for value in linevalues])
                        axs[i][j].plot(xparam[(~np.isinf(linevalues)) & (~np.isnan(linevalues))], values[(~np.isinf(
                            linevalues)) & (~np.isnan(linevalues))], color=color, ls='solid', lw=2*linewidth)

                    if dataMarker is not None:
                        markervalues = np.array(
                            [value for value in datamarkervalues[str(ID)].values])
                        if len(markervalues.shape) > 1:
                            markervalues = markervalues.T[0]
                            markervalues = np.array(
                                [value for value in markervalues])

                        if 'Merger' in dataMarker:
                            SnapCorotateMerger = Sample.loc[Sample.SubfindID_99 == ID, 'SnapCorotateMergers'].values[0]
                            
                            mergerTot = np.array(
                                [value for value in datamarkerTotvalues[str(ID)].values])
                            if len(mergerTot.shape) > 1:
                                mergerTot = mergerTot.T[0]
                                mergerTot = np.array(
                                    [value for value in mergerTot])

                            MarkerTotvalues = np.array(
                                [value for value in datamarkerTotvalues[str(ID)].values])
                            if len(MarkerTotvalues.shape) > 1:
                                MarkerTotvalues = MarkerTotvalues.T[0]
                                MarkerTotvalues = np.array(
                                    [value for value in MarkerTotvalues])

                            mergernumber = np.array(
                                [value for value in datamarkervalues[str(ID)].values])
                            if len(mergernumber.shape) > 1:
                                mergernumber = mergernumber.T[0]
                                mergernumber = np.array(
                                    [value for value in mergernumber])

                            Mergernumber = np.array(
                                [value for value in dataMarkervalues[str(ID)].values])
                            if len(Mergernumber.shape) > 1:
                                Mergernumber = Mergernumber.T[0]
                                Mergernumber = np.array(
                                    [value for value in Mergernumber])

                            Markervalues = np.array(
                                [value for value in datamarkervalues[str(ID)].values])
                            if len(Markervalues.shape) > 1:
                                Markervalues = Markervalues.T[0]
                                Markervalues = np.array(
                                    [value for value in Markervalues])

                            mergernumber = np.flip(mergernumber)
                            Mergernumber = np.flip(Mergernumber)
                            mergerTot = np.flip(mergerTot)

                            for nmergerindex, nmerger in enumerate(mergernumber):

                                if nmergerindex == 0:
                                    markervalues[nmergerindex] = 0
                                    continue
                                else:
                                    if np.isnan(nmerger):
                                        markervalues[nmergerindex] = 0
                                    else:
                                        if np.isnan(mergernumber[nmergerindex - 1]):
                                            markervalues[nmergerindex] = int(
                                                nmerger)
                                        else:
                                            markervalues[nmergerindex] = int(
                                                nmerger) - int(mergernumber[nmergerindex - 1])

                            for nmergerindex, nmerger in enumerate(Mergernumber):

                                if nmergerindex == 0:
                                    Markervalues[nmergerindex] = 0
                                    continue
                                else:
                                    if np.isnan(nmerger):
                                        Markervalues[nmergerindex] = 0
                                    else:
                                        if np.isnan(mergernumber[nmergerindex - 1]):
                                            Markervalues[nmergerindex] = int(
                                                nmerger)
                                        else:
                                            Markervalues[nmergerindex] = int(
                                                nmerger) - int(Mergernumber[nmergerindex - 1])

                            for nmergerindex, nmerger in enumerate(mergerTot):

                                if nmergerindex == 0:
                                    MarkerTotvalues[nmergerindex] = 0
                                    continue
                                else:
                                    if np.isnan(nmerger):
                                        MarkerTotvalues[nmergerindex] = 0
                                    else:
                                        if np.isnan(mergernumber[nmergerindex - 1]):
                                            MarkerTotvalues[nmergerindex] = int(
                                                nmerger)
                                        else:
                                            MarkerTotvalues[nmergerindex] = int(
                                                nmerger) - int(mergerTot[nmergerindex - 1])
                            MarkerTotvalues = MarkerTotvalues - Markervalues - markervalues
                            Markervalues = np.flip(Markervalues)
                            markervalues = np.flip(markervalues)
                            MarkerTotvalues = np.flip(MarkerTotvalues)

                        axs[i][j].scatter(xparam[(Markervalues > 0)], values[(Markervalues > 0)], color=colors.get(
                            str(l), 'black'), lw=1., marker='o',  edgecolors='black', s=250, alpha=0.7)
                        axs[i][j].scatter(xparam[(markervalues > 0)], values[(markervalues > 0)], color=colors.get(
                            str(l), 'black'), lw=1., marker='s',  edgecolors='black', s=100, alpha=0.7)
                        
                        if ~np.isnan(SnapCorotateMerger):
                            axs[i][j].scatter(time[(MarkerTotvalues > 0)], values[(MarkerTotvalues > 0)], 
                                              color=colors.get(str(l), 'black'), lw=1., marker='*',  edgecolors='black', s=300, alpha=0.7)

                elif Type == 'CoEvolution':
                    x = dfX[str(ID)].values
                    
                    if 'Type4' in row[l] or 'StarMass' in row[l]:
                        color = 'blue'
                    elif 'Type0' in row[l] or 'GasMass' in row[l]:
                        color = 'green'
                    elif 'Type1' in row[l] or 'DMMass' in row[l]:
                        color = 'purple'
                    else:
                        color = 'black'
                        
                    if len(x.shape) > 1:
                        x = np.array([value for value in x.T[0]])
                    else:
                        x = np.array([value for value in x])
                    colorSnap = np.array(
                        ['magenta', 'blue', 'cyan', 'lime', 'darkorange', 'red'])
                    if Xparam[i] != 'tsincebirth':
                        axs[i][j].scatter(x[99-snapsTime], values[99-snapsTime], color=colorSnap,
                                          lw=1., marker='d',  edgecolors=color, s=100, alpha=0.9)
                        axs[i][j].scatter(x[0], values[0], color='black', lw=1.,
                                          marker='o',  edgecolors=color, s=70, alpha=0.9)
                    argnotnan = ~np.isnan(values)
                    axs[i][j].plot(x[argnotnan], values[argnotnan], color=color, ls= 'solid')

                    if dataLine is not None:
                        linevalues = np.array(
                            [value for value in datalinevalues[str(ID)].values])
                        if len(linevalues.shape) > 1:
                            linevalues = linevalues.T[0]
                            linevalues = np.array(
                                [value for value in linevalues])
                        axs[i][j].plot(x[(~np.isinf(linevalues)) & (~np.isnan(linevalues))], values[(~np.isinf(linevalues)) & (
                            ~np.isnan(linevalues))], color=color, ls='solid', lw=3.)

                    if dataMarker is not None:
                        markervalues = np.array(
                            [value for value in datamarkervalues[str(ID)].values])
                        if len(markervalues.shape) > 1:
                            markervalues = markervalues.T[0]
                            markervalues = np.array(
                                [value for value in markervalues])

                        if 'Merger' in dataMarker:
                            mergerTot = np.array(
                                [value for value in datamarkerTotvalues[str(ID)].values])
                            if len(mergerTot.shape) > 1:
                                mergerTot = mergerTot.T[0]
                                mergerTot = np.array(
                                    [value for value in mergerTot])

                            MarkerTotvalues = np.array(
                                [value for value in datamarkerTotvalues[str(ID)].values])
                            if len(MarkerTotvalues.shape) > 1:
                                MarkerTotvalues = MarkerTotvalues.T[0]
                                MarkerTotvalues = np.array(
                                    [value for value in MarkerTotvalues])

                            mergernumber = np.array(
                                [value for value in datamarkervalues[str(ID)].values])
                            if len(mergernumber.shape) > 1:
                                mergernumber = mergernumber.T[0]
                                mergernumber = np.array(
                                    [value for value in mergernumber])

                            Mergernumber = np.array(
                                [value for value in dataMarkervalues[str(ID)].values])
                            if len(Mergernumber.shape) > 1:
                                Mergernumber = Mergernumber.T[0]
                                Mergernumber = np.array(
                                    [value for value in Mergernumber])

                            Markervalues = np.array(
                                [value for value in datamarkervalues[str(ID)].values])
                            if len(Markervalues.shape) > 1:
                                Markervalues = Markervalues.T[0]
                                Markervalues = np.array(
                                    [value for value in Markervalues])

                            mergernumber = np.flip(mergernumber)
                            Mergernumber = np.flip(Mergernumber)
                            mergerTot = np.flip(mergerTot)

                            for nmergerindex, nmerger in enumerate(mergernumber):

                                if nmergerindex == 0:
                                    markervalues[nmergerindex] = 0
                                    continue
                                else:
                                    if np.isnan(nmerger):
                                        markervalues[nmergerindex] = 0
                                    else:
                                        if np.isnan(mergernumber[nmergerindex - 1]):
                                            markervalues[nmergerindex] = int(
                                                nmerger)
                                        else:
                                            markervalues[nmergerindex] = int(
                                                nmerger) - int(mergernumber[nmergerindex - 1])

                            for nmergerindex, nmerger in enumerate(Mergernumber):

                                if nmergerindex == 0:
                                    Markervalues[nmergerindex] = 0
                                    continue
                                else:
                                    if np.isnan(nmerger):
                                        Markervalues[nmergerindex] = 0
                                    else:
                                        if np.isnan(mergernumber[nmergerindex - 1]):
                                            Markervalues[nmergerindex] = int(
                                                nmerger)
                                        else:
                                            Markervalues[nmergerindex] = int(
                                                nmerger) - int(Mergernumber[nmergerindex - 1])

                            for nmergerindex, nmerger in enumerate(mergerTot):

                                if nmergerindex == 0:
                                    MarkerTotvalues[nmergerindex] = 0
                                    continue
                                else:
                                    if np.isnan(nmerger):
                                        MarkerTotvalues[nmergerindex] = 0
                                    else:
                                        if np.isnan(mergernumber[nmergerindex - 1]):
                                            MarkerTotvalues[nmergerindex] = int(
                                                nmerger)
                                        else:
                                            MarkerTotvalues[nmergerindex] = int(
                                                nmerger) - int(mergerTot[nmergerindex - 1])
                            MarkerTotvalues = MarkerTotvalues - Markervalues - markervalues
                            Markervalues = np.flip(Markervalues)
                            markervalues = np.flip(markervalues)
                            MarkerTotvalues = np.flip(MarkerTotvalues)

                        axs[i][j].scatter(x[(Markervalues > 0)], values[(Markervalues > 0)], color=colors.get(
                            str(l), 'black'), lw=1., marker='o',  edgecolors='black', s=130, alpha=0.5)
                        axs[i][j].scatter(x[(markervalues > 0)], values[(markervalues > 0)], color=colors.get(
                            str(l), 'black'), lw=1., marker='o',  edgecolors='black', s=110, alpha=0.5)
                        #axs[i][j].scatter(x[(MarkerTotvalues > 0)], values[(MarkerTotvalues > 0)], color=colors.get(
                            #str(l), 'black'), lw=1., marker='o',  edgecolors='black', s=15, alpha=0.5)

            # Plot details

            if row[-1] == 'StarMassNormalized':
                axs[i][j].set_yticks([0.1, 0.2, 0.5, 1])
                axs[i][j].set_yticklabels(['0.1','0.2', '0.5', '1'])
                
            

            if GridMake:
                axs[i][j].grid(GridMake, color='#9e9e9e',  which="major", linewidth= 0.6,alpha= 0.3 , linestyle=':')
               
            axs[i][j].tick_params(axis='y', labelsize=0.99*fontlabel)
            axs[i][j].tick_params(axis='x', labelsize=0.99*fontlabel)
	    
            
            if ylimmin != None and ylimmax != None:
                axs[i][j].set_ylim(ylimmin[i], ylimmax[i])
            if scales.get(row[0]) != None :
                axs[i][j].set_yscale(scales.get(row[0]))
            if scales.get(row[0]) == 'log' :
                axs[i][j].yaxis.set_major_formatter(
                    FuncFormatter(format_func_loglog))
                
            if row[-1] == 'FracType1':
                axs[i][j].set_yticks([0.2, 0.3, 0.5, 1])
                axs[i][j].set_yticklabels(['0.2','0.3', '0.5', '1'])

            
            if j == 0:

                if len(row) > 1:
                    axs[i][j].set_ylabel(
                        labelsequal.get(row[0]), fontsize=fontlabel)

                else:
                    axs[i][j].set_ylabel(labels.get(row[0]), fontsize=fontlabel)

            if j == len(IDs) - 1:
                if xlabelintext and not limaxis and len(rows) > 1:
                    Afont = {'color':  'black',
                             'size': fontlabel,
                             }
                    anchored_text = AnchoredText(
                        texts.get(row), loc='upper right', prop=Afont)
                    axs[i][j].add_artist(anchored_text)

            if xlabelintext and limaxis and len(rows) > 1:
                Afont = {'color':  'black',
                         'size': fontlabel,
                         }
                anchored_text = AnchoredText(
                    texts[row], loc='upper left', prop=Afont)
                axs[i][j].add_artist(anchored_text)
                
            if j == 0:
                
                
                if title != None and ColumnPlot:
                    Afont = {'color':  'black',
                             'size': fontlabel,
                             }
                    anchored_text = AnchoredText(
                        titles.get(
                            title[i], title[i]), loc=postext[i], prop=Afont)
                    axs[i][j].add_artist(anchored_text)


            if i == 0:

                if title and not ColumnPlot:
                    axs[i][j].set_title(titles.get(
                        title[j], title[j]), fontsize=1*fontlabel)
                

                if Type == 'Evolution' and not PhasingPlot:

                    axs[i][j].tick_params(bottom=True, top=False)
                    lim = axs[i][j].get_xlim()
                    ax2label = axs[i][j].twiny() #secondary_xaxis('top', which='major')
                    ax2label.grid(False)
                    ax2label.set_xlim(lim)

                    if row == 'rToRNearYoung' or savefigname == 'Young':
                        zticks = np.array([0., 0.2])
                        zlabels = np.array(
                            ['0', '0.2'])
                        zticks_Age = np.array(
                            [13.803, 11.323])
                    elif not PhasingPlot:
                        zticks = np.array([0., 0.2, 0.5, 1., 2., 5., 20.])
                        if SmallerScale:

                            if j == 0:
                                zlabels = np.array(
                                    ['0', '0.2', '0.5', '1', '2', '5', '20'])
                            if j != 0:
                                zlabels = np.array(
                                    ['0', '0.2', '0.5', '1', '2', '5', ''])
                        else:
                            zlabels = np.array(
                                ['0', '0.2', '0.5', '1', '2', '5', '20'])
                        zticks_Age = np.array(
                            [13.803, 11.323, 8.587, 5.878, 3.285, 1.2, 0])


                    zticks = zticks.tolist()
                    zticks_Age = zticks_Age.tolist()

                    x_locator = FixedLocator(zticks_Age)
                    x_formatter = FixedFormatter(zlabels)
                    ax2label.xaxis.set_major_locator(x_locator)
                    ax2label.xaxis.set_major_formatter(x_formatter)
                    ax2label.set_xlabel(r"$z$", fontsize=fontlabel)
                    ax2label.tick_params(labelsize=0.85*fontlabel)
                    ax2label.tick_params(axis='x',  which='minor', top=False)


            if i == len(rows) - 1:
                
                if Type == 'Evolution':
                    
                    
                    if row == 'rToRNearYoung' or savefigname == 'Young':
                        axs[i][j].set_xlabel(r'$t \, \,  [\mathrm{Gyr}]$', fontsize=fontlabel)
                        axs[i][j].set_xticks([10, 12, 14])
                        axs[i][j].set_xticklabels(
                            ['10', '12', '14'])
                    elif not PhasingPlot:
                        if LookBackTime:
                            axs[i][j].set_xticks([0.  ,  1.97185714,  3.94371429,  5.91557143,  7.88742857, 9.85928571, 11.83114286, 13.803  ])
                            if SmallerScale:
                                fig.supxlabel(r'$\mathrm{Lookback \; Time} \, \, [\mathrm{Gyr}]$', fontsize=fontlabel, y = 0.07)

                                if j == 0:
                                    axs[i][j].set_xticklabels(
                                    ['14', '12', '10', '8', '6', '4', '2', '0'])
                                if j != 0:
                                    axs[i][j].set_xticklabels(
                                    ['', '12', '10', '8', '6', '4', '2', '0'])
                            else:
                                axs[i][j].set_xlabel(r'$\mathrm{Lookback \; Time} \, \, [\mathrm{Gyr}]$', fontsize=fontlabel)

                                axs[i][j].set_xticklabels(
                                    ['14', '12', '10', '8', '6', '4', '2', '0'])
                        else:
                            axs[i][j].set_xticks([0, 2, 4, 6, 8, 10, 12, 14])

                            axs[i][j].set_xlabel(r'$t \, \, [\mathrm{Gyr}]$', fontsize=fontlabel)
        
                            axs[i][j].set_xticklabels(
                                ['0', '2', '4', '6', '8', '10', '12', '14'])

                    elif PhasingPlot:
                        axs[i][j].set_xlabel(r'$\phi_\mathrm{Orbital}$', fontsize=fontlabel)
                        axs[i][j].set_xticks([-1, -0.5, 0, 1, 2, 3, 4, 5] )
                        axs[i][j].set_xticklabels(['', 'E', '0', '1', '2', '3', '4', '5'])
                        axs[i][j].set_xlim(-1, 5.5)
                        
                elif Type == 'CoEvolution':
                    axs[i][j].set_xscale(scales.get(Xparam[i], 'linear'))
                    if scales.get(Xparam[i]) == 'log':
                        axs[i][j].xaxis.set_major_formatter(
                            FuncFormatter(format_func_loglog))
                    axs[i][j].set_xlabel(labels.get(
                        Xparam[i], 'None'), fontsize=fontlabel)

    savefig(savepath, savefigname, TRANSPARENT)

    return

def PlotScatter(names, columns, ParamX, ParamsY,  Type='z0', snap=[99], title=False, medianBins=False, medianAll=False, xlabelintext=False, All=None,
                legend=False, SpearManTestAll = False, SpearManTest = False,NoneEdgeColor = False, LegendNames=None,  TRANSPARENT = False, COLORBAR = None, medianDot = False, MarkerSizes = None,
                alphaScater=1.,  alphaShade=0.3,  linewidth=1.2, fontlabel=26, 
                m='o', msizet=30, quantile=0.95,framealpha = 0.95, q = 0.95,
                InvertPlot = False, ylimmin = None,  ylimmax = None, xlimmin = None, xlimmax = None,  legpositions = None,
                lNum = 6, cNum = 6, msizeMult = 1, GridMake = False, EqualLine = False, EqualLineMin = None, EqualLineMax = None,
                columnspacing = 0.5, handlelength = 2, handletextpad = -0.5, labelspacing = 0.3, loc = 'best',
                savepath='fig/PlotScatter',  savefigname='fig', dfName='Sample', SampleName='Samples', cmap = 'inferno',
                bins=10, seed=16010504, mult = 4.1):

    '''
    Plot teh evolution for random sample
    Parameters
    ----------
    columns : specific set in the sample / or different param to plot in each column. array with str
    ParamX : specific the x param. str
    ParamsY : specific the y param. array with str
    medianBins : to plot the median of the x - y relation. Default: False. bool
    medianSample : to plot the median of all subhalos excepet 'Compacts'. Default: False. bool
    medianAll : to plot the median for each name in names of the x - y relation. Default: False. bool
    All : to plot all subhalos. Default: None. None or str name for dataframe with all subhalos. 
    alphaScater : alpha for scatter plot. Default: 0.5. float
    alphaAll : alpha for scatter plot of all subhalos. Default: 0.2. float
    m : marker. Default: 'o'. str
    msize : marker size. Default: 30. float
    quantile : median quantile to plot. Default: 0.95. float
    The rest is the same as the previous functions
    Returns
    -------
    Requested Evolution or Co-Evolution plot
    -------
    Author: Abhner P. de Almeida (abhner.almeida AAT usp.br)
    '''

    np.random.seed(seed)

    # Verify NameParameters
    if type(columns) is not list and type(columns) is not np.ndarray:
        columns = [columns]

    if type(ParamsY) is not list and type(ParamsY) is not np.ndarray:
        ParamsY = [ParamsY]

    if type(ParamX) is not list and type(ParamX) is not np.ndarray:
        ParamsX = [ParamX]
        LabelGeral = False
    else:
        LabelGeral = True
        ParamsX = ParamX

    while len(ParamsX) != len(ParamsY):
        ParamsX.append(ParamX)
        
        
    if columns == ['Snap']:
        columns = snap
        dataX = TNG.makedata(names, columns, ParamsX, 'Snap',
                         snap=snap, SampleName=SampleName, dfName = dfName)
        dataY = TNG.makedata(names, columns, ParamsY, 'Snap',
                         snap=snap, SampleName=SampleName, dfName = dfName)
        if COLORBAR != None:
            
            dataColorbar = TNG.makedata(names, columns, COLORBAR, 'Snap',
                             snap=snap, SampleName=SampleName, dfName = dfName)
            
             
    else:
        dataX = TNG.makedata(names, columns, ParamsX, Type,
                         snap=snap, SampleName=SampleName, dfName = dfName)
        dataY = TNG.makedata(names, columns, ParamsY, Type,
                         snap=snap, SampleName=SampleName, dfName = dfName)
        if MarkerSizes != None:
            dataMarker = TNG.makedata(names, columns, MarkerSizes, Type,
                             snap=snap, SampleName=SampleName, dfName = dfName)
            
        if COLORBAR != None:
            
            dataColorbar = TNG.makedata(names, columns, COLORBAR, Type,
                             snap=snap, SampleName=SampleName, dfName = dfName)
            

    # Define axes
    if len(snap) > 1:
        columns = np.full(len(snap), 'Snap')
    plt.rcParams.update({'figure.figsize': (cNum*len(columns), lNum*len(ParamsY))})
    fig = plt.figure()
    gs = fig.add_gridspec(len(ParamsY), len(columns), hspace=0, wspace=0)
    axs = gs.subplots(sharex='col', sharey='row')

    # Verify axs shape
    if type(axs) is not list and type(axs) is not np.ndarray:
        axs = [axs]
    if type(axs[0]) is not np.ndarray:
        axs = np.array([axs])
        if len(columns) == 1:
            axs = axs.T
            
    
     
    for i, param in enumerate(ParamsY):

        for j, titlename in enumerate(columns):
            print('\n Type: ', titlename)
            
            
            xMedianDot =  np.array([])
            yMedianDot =  np.array([])
            NamesMedianDot = np.array([])

            if ParamX == 'MassInAfterInfall' or (ParamX == 'MassIn_Infall_to_GasLost' and  (ParamsY[0] == 'MassAboveAfter_over_In_Infall_to_GasLost')) :


                axs[i][j].axvline(0,color = 'black',linestyle='dashed',lw=2)
                axs[i][j].axhline(0,color = 'black',linestyle='dashed',lw=2)
                
                #ax.fill_between([-0.2, roc_t],-0.2,roc_v,alpha=0.3, color='#1F98D0')  # blue
                axs[i][j].fill_between([0, 500], -500, 0, alpha=0.2, color='tab:green')  # yellow
                axs[i][j].fill_between([-500, 0], 0, 500, alpha=0.2, color='tab:red')  # orange
                axs[i][j].fill_between([0, 500], 0, 500, alpha=0.2, color='tab:blue')  # red
                axs[i][j].text(-.17, .5, 'TS', fontsize = 0.98*fontlabel)
                axs[i][j].text(0.12, .5, 'SF', fontsize = 0.98*fontlabel)
                axs[i][j].text(0.55,-1.5,  'Interplay', fontsize = 0.98*fontlabel)
                
       
            elif (ParamX == 'Decrease_Entry_To_NoGas_Norm_Delta' and  (ParamsY[0] == 'Decrease_NoGas_To_Final_Norm_Delta')) :



                #ax.fill_between([-0.2, roc_t],-0.2,roc_v,alpha=0.3, color='#1F98D0')  # blue
                #axs[i][j].fill_between([0, 500], -500, 0, alpha=0.2, color='tab:green')  # yellow
                xfitline  = np.linspace(-2 ,0.4, 100)
                axs[i][j].fill_between(xfitline, -1, xfitline, alpha=0.2, color='tab:red')  # orange
                axs[i][j].fill_between(xfitline, xfitline, 0.25, alpha=0.2, color='tab:blue')  # red
                axs[i][j].text(-.6,-0.8, "Faster compaction \n after gas loss", fontsize = 0.99*fontlabel)
                axs[i][j].text(-0.80, -0.22, "Faster compaction  \n with gas ", fontsize = 0.99*fontlabel)
                axs[i][j].axvline(0,color = 'black',linestyle='dashed',lw=linewidth)
                axs[i][j].axhline(0,color = 'black',linestyle='dashed',lw=linewidth)
                
                axs[i][j].set_xticks([-0.8, -0.6, -0.4, -0.2, 0.0, 0.2])
                axs[i][j].set_xticklabels(['-0.8', '-0.6', '-0.4', '-0.2', '0.0', '0.2'])
                
                axs[i][j].set_yticks([-0.8, -0.6, -0.4, -0.2, 0.0, 0.2])
                axs[i][j].set_yticklabels(['-0.8', '-0.6', '-0.4', '-0.2', '0.0', '0.2'])
                
                
            elif (ParamX == 'Rhalf_MaxProfile_Minus_HalfRadstar_Entry' and  (ParamsY[0] == 'Rhalf_MinProfile_Minus_HalfRadstar_Entry')) :



                #ax.fill_between([-0.2, roc_t],-0.2,roc_v,alpha=0.3, color='#1F98D0')  # blue
                #axs[i][j].fill_between([0, 500], -500, 0, alpha=0.2, color='tab:green')  # yellow
                xfitline  = np.linspace(-6 ,2, 100)
                axs[i][j].plot( xfitline, xfitline, ls='--', color='tab:blue', linewidth=linewidth)
                axs[i][j].fill_between(xfitline, -7, xfitline, alpha=0.2, color='tab:blue')  # orange
                axs[i][j].text(-2,-5, "TS", fontsize = 0.99*fontlabel)
                axs[i][j].fill_between(xfitline, xfitline, 1, alpha=0.2, color='tab:red')  # orange
                axs[i][j].text(-4.,-1, "SF", fontsize = 0.99*fontlabel)
              
            elif (ParamX == 'Relative_Rhalf_MaxProfile_Minus_HalfRadstar_Entry' and  (ParamsY[0] == 'Relative_Rhalf_MinProfile_Minus_HalfRadstar_Entry')) :



                #ax.fill_between([-0.2, roc_t],-0.2,roc_v,alpha=0.3, color='#1F98D0')  # blue
                #axs[i][j].fill_between([0, 500], -500, 0, alpha=0.2, color='tab:green')  # yellow
                xfitline  = np.linspace(-6 ,2, 100)
                axs[i][j].plot( xfitline, xfitline, ls='--', color='tab:blue', linewidth=linewidth)
                axs[i][j].fill_between(xfitline, -7, xfitline, alpha=0.2, color='tab:red')  # orange
                axs[i][j].text(0.2, -1.25, "TS", fontsize = 0.99*fontlabel)
                axs[i][j].fill_between(xfitline, xfitline, 1, alpha=0.2, color='tab:blue')  # orange
                axs[i][j].text(-1.8,-1., "SF", fontsize = 0.99*fontlabel)
                
            elif (ParamX == 'Relative_logInnerZ_At_Entry' and  (ParamsY[0] == 'Relative_logZ_At_Entry')) :
                xfitline  = np.linspace(0 ,1, 100)
                axs[i][j].plot( xfitline, xfitline, ls='--', color='tab:blue', linewidth=linewidth)

                
            elif (ParamX == 'Relative_Rhalf_MaxProfile_Minus_HalfRadstar_Entry' and  (ParamsY[0] == 'Relative_Rhalf_MinProfile_Minus_HalfRadstar_Entry')) :



                #ax.fill_between([-0.2, roc_t],-0.2,roc_v,alpha=0.3, color='#1F98D0')  # blue
                #axs[i][j].fill_between([0, 500], -500, 0, alpha=0.2, color='tab:green')  # yellow
                xfitline  = np.linspace(-6 ,2, 100)
                axs[i][j].plot( xfitline, xfitline, ls='--', color='tab:blue', linewidth=linewidth)
                axs[i][j].fill_between(xfitline, -7, xfitline, alpha=0.2, color='tab:red')  # orange
                axs[i][j].text(0.2, -1.25, "TS", fontsize = 0.99*fontlabel)
                axs[i][j].fill_between(xfitline, xfitline, 1, alpha=0.2, color='tab:blue')  # orange
                axs[i][j].text(-1.8,-1., "SF", fontsize = 0.99*fontlabel)
                
                
            elif (ParamX == 'FracStarLoss' and  (ParamsY[0] == 'FracStarAfterEntry')) :

                

                #ax.fill_between([-0.2, roc_t],-0.2,roc_v,alpha=0.3, color='#1F98D0')  # blue
                #axs[i][j].fill_between([0, 500], -500, 0, alpha=0.2, color='tab:green')  # yellow
                xfitline  = np.linspace(0.1 ,15, 1000)
                axs[i][j].plot( xfitline, xfitline, ls='--', color='tab:blue', linewidth=linewidth)
                axs[i][j].fill_between(xfitline, -7, xfitline, alpha=0.2, color='tab:red')  # orange
                axs[i][j].text(0.135, 0.85, "Greater star \n formation", fontsize = 0.99*fontlabel)
                axs[i][j].fill_between(xfitline, xfitline, 1, alpha=0.2, color='tab:blue')  # orange
                axs[i][j].text(0.135,0.05, "Greater stellar \n mass loss", fontsize = 0.99*fontlabel)
                
            
            elif (ParamX == 'GasInner_Entry_to_Nogas' or  ParamX == 'GasTrueInner_Entry_to_Nogas' ) and  (ParamsY[0] == 'GasAbove_Entry_to_Nogas' or  ParamsY[0]  == 'GasTrueAbove_Entry_to_Nogas') :



                #ax.fill_between([-0.2, roc_t],-0.2,roc_v,alpha=0.3, color='#1F98D0')  # blue
                #axs[i][j].fill_between([0, 500], -500, 0, alpha=0.2, color='tab:green')  # yellow
                xfitline  = np.linspace(-0.5 ,0.2, 100)
                axs[i][j].plot( xfitline, xfitline, ls='--', color='tab:blue', linewidth=linewidth)
                axs[i][j].fill_between(xfitline, -0.7, xfitline, alpha=0.2, color='tab:green')  # orange
                axs[i][j].text(-.15,-0.47, "Faster outer \n  gas loss", fontsize = 0.99*fontlabel)
                axs[i][j].text(-.15,-0.07, "Slower outer \n  gas loss", fontsize = 0.99*fontlabel)
                axs[i][j].axvline(0,ls='--', color='black', linewidth=linewidth)
            elif (ParamX == 'sSFRInner_BeforeEntry' and  (ParamsY[0] == 'sSFRInner_Entry_to_Nogas')) :



                #ax.fill_between([-0.2, roc_t],-0.2,roc_v,alpha=0.3, color='#1F98D0')  # blue
                #axs[i][j].fill_between([0, 500], -500, 0, alpha=0.2, color='tab:green')  # yellow
                xfitline  = np.linspace(-13 ,-7, 100)
                #axs[i][j].plot( xfitline, xfitline, ls='--', color='tab:blue', linewidth=linewidth)
                axs[i][j].fill_between(xfitline, -12, xfitline, alpha=0.2, color='tab:blue')  # orange
                axs[i][j].text(-10,-10.55, "Inner sSFR \n decrease", fontsize = 0.99*fontlabel)
                axs[i][j].fill_between(xfitline, xfitline,-8, alpha=0.2, color='tab:red')  # orange
                axs[i][j].text(-10.9,-9.5, "Inner sSFR \n increase", fontsize = 0.99*fontlabel)
                
            elif (ParamX == 'sSFRTrueInner_BeforeEntry' and  (ParamsY[0] == 'sSFRTrueInner_Entry_to_Nogas')) :



               #ax.fill_between([-0.2, roc_t],-0.2,roc_v,alpha=0.3, color='#1F98D0')  # blue
               #axs[i][j].fill_between([0, 500], -500, 0, alpha=0.2, color='tab:green')  # yellow
               xfitline  = np.linspace(-13 ,-7, 100)
               #axs[i][j].plot( xfitline, xfitline, ls='--', color='tab:blue', linewidth=linewidth)
               axs[i][j].fill_between(xfitline, -12, xfitline, alpha=0.2, color='tab:red')  # orange
               axs[i][j].text(-10,-10.55, "Inner $\overline{\mathrm{sSFR}}$ \n decrease", fontsize = 0.99*fontlabel)
               axs[i][j].fill_between(xfitline, xfitline,-8, alpha=0.2, color='tab:blue')  # orange
               axs[i][j].text(-10.9,-9.5, "Inner $\overline{\mathrm{sSFR}}$  \n increase", fontsize = 0.99*fontlabel)

            elif (ParamX == 'sSFRTrueInner_BeforeEntry' and  (ParamsY[0] == 'sSFRAfterPericenter')) :



               #ax.fill_between([-0.2, roc_t],-0.2,roc_v,alpha=0.3, color='#1F98D0')  # blue
               #axs[i][j].fill_between([0, 500], -500, 0, alpha=0.2, color='tab:green')  # yellow
               xfitline  = np.linspace(-13 ,-7, 100)
               #axs[i][j].plot( xfitline, xfitline, ls='--', color='tab:blue', linewidth=linewidth)
               axs[i][j].fill_between(xfitline, -12, xfitline, alpha=0.2, color='tab:red')  # orange
               axs[i][j].text(-10,-10.55, "Inner $\overline{\mathrm{sSFR}}$ \n decrease", fontsize = 0.99*fontlabel)
               axs[i][j].fill_between(xfitline, xfitline,-8, alpha=0.2, color='tab:blue')  # orange
               axs[i][j].text(-10.9,-9.5, "Inner $\overline{\mathrm{sSFR}}$  \n increase", fontsize = 0.99*fontlabel)

            elif ParamX == 'MassInAfterInfall' or (ParamX == 'MassIn_Infall_to_GasLost' and  ( ParamsY[0] == 'MassAboveAfter_Infall_to_GasLost')) :

                axs[i][j].axvline(0,color = 'black',linestyle='dashed',lw=2)
                axs[i][j].axhline(0,color = 'black',linestyle='dashed',lw=2)
                
                #ax.fill_between([-0.2, roc_t],-0.2,roc_v,alpha=0.3, color='#1F98D0')  # blue
                axs[i][j].fill_between([0, 500], -500, 0, alpha=0.2, color='tab:green')  # yellow
                axs[i][j].fill_between([-500, 0], -500, 0, alpha=0.2, color='tab:red')  # orange
                axs[i][j].fill_between([0, 500], 0, 500, alpha=0.2, color='tab:blue')  # red
                axs[i][j].text(-.145, -0.95, 'TS', fontsize = 0.98*fontlabel)
                axs[i][j].text(0.1, 0.05, 'SF', fontsize = 0.98*fontlabel)
                axs[i][j].text(0.3,-0.95,  'Interplay', fontsize = 0.98*fontlabel)

            elif ParamX == 'MassIn_Infall_to_GasLost' and ParamsY[0] == 'DecreaseBetweenGasStar_Over_starFinal' :
                

                axs[i][j].axvline(0,color = 'black',linestyle='dashed',lw=2)
                axs[i][j].axhline(0,color = 'black',linestyle='dashed',lw=2)
             
            if ParamX == 'AgeBorn':
                
                x = np.arange(14)
                axs[i][j].plot(x, x,color = 'black',linestyle='dashed',lw=2)
             
            if All is not None:
                xAll = All[ParamX]
                yAll = All[param]

                axs[i][j].scatter(xAll, yAll, color=colors['All'],
                                  edgecolor=colors['All'], alpha=1., marker='.', s=10)
                #axs[i][j].scatter(dataAllx[i][j], dataAlly[i][j], color=colors['All'],
                #                  edgecolor=colors['All'], alpha=1., marker='.', s=10)
                #axs[i][j].hexbin(dataAllx[i][j], dataAlly[i][j], cmap = 'Greys')
            
            if SpearManTestAll:
                XAllSMT = np.array([])
                YAllSMT = np.array([])
                ColorALLSMT = np.array([])

            for l, values in enumerate(dataY[i][j]):
                
                if InvertPlot:
                    if j == 1:
                        l =  len(dataY[i][j]) - l - 1
                        values = dataY[i][j][l]
                                
                if SpearManTest and not SpearManTestAll:
                    # Perform Spearman rank correlation test
                    print('Name: ', names[l])
                    XValuesCompare = dataX[i][j][l][(~np.isnan(values)) & (~np.isinf(values))]
                    ColorbarValuesCompare = dataColorbar[i][j][l][(~np.isnan(values)) & (~np.isinf(values))]
                    YValuesCompare = values[(~np.isnan(values)) & (~np.isinf(values))]


                    correlation, p_value = spearmanr(XValuesCompare[(~np.isnan(XValuesCompare)) & (~np.isinf(XValuesCompare))], YValuesCompare[(~np.isnan(XValuesCompare)) & (~np.isinf(XValuesCompare))])
                    
                   
                
               
                if MarkerSizes != None:
                    Markers = dataMarker[0][0][l]
                    
                    
                    axs[i][j].scatter(dataX[i][j][l][Markers <= 1], values[Markers <= 1], color=colors.get(
                            names[l], 'black'), edgecolor=edgecolors.get(names[l], None), alpha=alphaScater, lw = linesthicker.get(names[l], linewidth), 
                            marker=markers.get(names[l], 'o'), s=20)
                    axs[i][j].scatter(dataX[i][j][l][Markers == 2], values[Markers == 2], color=colors.get(
                            names[l] ), edgecolor=edgecolors.get(names[l], None), alpha=alphaScater, lw = linesthicker.get(names[l]), 
                            marker=markers.get(names[l], 'o'), s=45)
                    axs[i][j].scatter(dataX[i][j][l][Markers >= 3], values[Markers >= 3], color=colors.get(
                            names[l]), edgecolor=edgecolors.get(names[l], None), alpha=alphaScater, lw = linesthicker.get(names[l]), 
                            marker=markers.get(names[l], 'o'), s=120)

                elif COLORBAR != None:
                    Colorbar = dataColorbar[i][j][l]
                    cmap = plt.cm.get_cmap(cmap)
                    mult = 4.1
                    
                    if COLORBAR[0] in ['DecreaseBetweenGasStar_Over_starFinal', 'SubhaloMassType4', 'MassAboveAfter_over_In_Lost',  'MassInAfterInfall_Lost']:
                       sc = axs[i][j].scatter(dataX[i][j][l], values, c=Colorbar, 
                                              edgecolor=edgecolors.get(names[l] + 'Colorbar', 'black'), alpha=alphaScater, 
                                              lw = 1.5*linewidth,
                                              marker=markers.get(names[l] + 'Colorbar', 'o'), 
                                              s=7*msize.get(names[l]  + 'Colorbar', msizet),
                                              cmap = cmap,  #norm=matplotlib.colors.LogNorm()
                                              )
                       
                    elif COLORBAR[0] == 'sSFRRatio_ETOGAS_Before':
                        sc = axs[i][j].scatter(dataX[i][j][l], values, c=Colorbar, 
                                               edgecolor=edgecolors.get(names[l],'k'), alpha=alphaScater, 
                                               lw = 1.5*linewidth,
                                               marker=markers.get(names[l], 'o'), 
                                               s=7*msize.get(names[l], msizet),
                                               cmap = cmap, norm=mpl.colors.LogNorm(vmin = 0.005, vmax = 1.8),
                                               )
                       
                    elif COLORBAR[0] == 'TimeInnerRegion':
                        sc = axs[i][j].scatter(dataX[i][j][l], values, c=Colorbar, 
                                               edgecolor=edgecolors.get(names[l],'k'), alpha=alphaScater, 
                                               lw = 1.5*linewidth,
                                               marker=markers.get(names[l], 'o'), 
                                               s=7*msize.get(names[l], msizet),
                                               cmap = cmap, norm=mpl.colors.LogNorm(vmin = 0.005, vmax = 13),
                                               )
                    elif COLORBAR[0] == 'RatioSFR':
                        sc = axs[i][j].scatter(dataX[i][j][l], values, c=Colorbar, 
                                               edgecolor=edgecolors.get(names[l],'k'), alpha=alphaScater, 
                                               lw = 1.5*linewidth,
                                               marker=markers.get(names[l], 'o'), 
                                               s=7*msize.get(names[l], msizet),
                                               cmap = cmap, norm=mpl.colors.LogNorm(vmin = 0.005, vmax = 1.1),
                                               )
                        
                    elif COLORBAR[0] == 'logMstar_Entry':
                        sc = axs[i][j].scatter(dataX[i][j][l], values, c=Colorbar, 
                                               edgecolor=edgecolors.get(names[l],'k'), alpha=alphaScater, 
                                               lw = 1.5*linewidth,
                                               marker=markers.get(names[l], 'o'), 
                                               s=7*msize.get(names[l], msizet),
                                               cmap = cmap, 
                                               )
                       
                    elif COLORBAR[0] == 'MassInAfterInfall':
                       cmap = plt.cm.get_cmap(cmap, 2)

                       sc = axs[i][j].scatter(dataX[i][j][l], values, c=Colorbar, 
                                              edgecolor=edgecolors.get(names[l]+ 'Colorbar','k'), alpha=alphaScater, 
                                              lw = 1.5*linewidth,
                                              marker=markers.get(names[l] + 'Colorbar', 'o'), 
                                              s=7*msize.get(names[l]  + 'Colorbar', msizet),
                                              cmap = cmap
                                              )
                       
                    elif COLORBAR[0] == 'SigmaIn':

                       sc = axs[i][j].scatter(dataX[i][j][l], values, c=Colorbar, 
                                              edgecolor=edgecolors.get(names[l]+ 'Colorbar','k'), alpha=alphaScater, 
                                              lw = 1.5*linewidth,
                                              marker=markers.get(names[l] + 'Colorbar', 'o'), 
                                              s=7*msize.get(names[l]  + 'Colorbar', msizet),
                                              cmap = cmap, vmin = 40, vmax = 85,
                                              )
                       
                    elif COLORBAR[0] == 'U-r':
                       cmap = plt.cm.get_cmap(cmap, 2)

                       sc = axs[i][j].scatter(dataX[i][j][l], values, c=Colorbar, 
                                              edgecolor=edgecolors.get(names[l]+ 'Colorbar','k'), alpha=alphaScater, 
                                              lw = 1.5*linewidth,
                                              marker=markers.get(names[l] + 'Colorbar', 'o'), 
                                              s=7*msize.get(names[l]  + 'Colorbar', msizet),
                                              cmap = cmap, vmin = -0.3, vmax = 2.1,
                                              )
                       
                    elif COLORBAR[0] == 'rOrbMean_Entry_to_Gas':
                        sc = axs[i][j].scatter(dataX[i][j][l], values, c=Colorbar, 
                                              edgecolor=edgecolors.get(names[l] + 'Colorbar', 'black'), alpha=alphaScater, 
                                              lw = 1.5*linewidth,
                                              marker=markers.get(names[l] + 'Colorbar', 'o'), 
                                              s=7*msize.get(names[l]  + 'Colorbar', msizet),
                                              cmap = cmap,  norm=mpl.colors.LogNorm(vmin = 0.05, vmax = 0.7),
                                           )

                    elif COLORBAR[0] == 'sSFRTrueRatio_Entry_to_Nogas':
                        sc = axs[i][j].scatter(dataX[i][j][l], values, c=Colorbar, 
                                              edgecolor=edgecolors.get(names[l] + 'Colorbar', 'black'), alpha=alphaScater, 
                                              lw = 1.5*linewidth,
                                              marker=markers.get(names[l] + 'Colorbar', 'o'), 
                                              s=7*msize.get(names[l]  + 'Colorbar', msizet),
                                              cmap = cmap, vmin = 0.8, vmax = 1.7,
                                           )
                       
                    elif COLORBAR[0] == 'SnapLastMerger_Before_Entry':
                        sc = axs[i][j].scatter(dataX[i][j][l], values, c=Colorbar, 
                                              edgecolor=colors.get(names[l]+ 'Colorbar', 'k'), alpha=alphaScater, 
                                              lw = 1.5*linewidth,
                                              marker=markers.get(names[l] + 'Colorbar', 'o'), 
                                              s=7*msize.get(names[l]  + 'Colorbar', msizet),
                                              vmin = 0, vmax = 14,
                                              cmap = cmap
                                              )
                        
                    elif COLORBAR[0] == 'TimeSinceQuenching':

                       sc = axs[i][j].scatter(dataX[i][j][l], values, c=Colorbar, 
                                              edgecolor=colors.get(names[l]+ 'Colorbar', 'k'), alpha=alphaScater, 
                                              lw = 1.5*linewidth,
                                              marker=markers.get(names[l] + 'Colorbar', 'o'), 
                                              s=7*msize.get(names[l]  + 'Colorbar', msizet),
                                              vmin = 0, vmax = 6,
                                              cmap = cmap
                                              )
                       
                       
                    elif COLORBAR[0] == 'DeltasSFRRatio_Entry_to_Nogas':

                       sc = axs[i][j].scatter(dataX[i][j][l], values, c=Colorbar, 
                                              edgecolor=colors.get(names[l]+ 'Colorbar', 'k'), alpha=alphaScater, 
                                              lw = 1.5*linewidth,
                                              marker=markers.get(names[l] + 'Colorbar', 'o'), 
                                              s=7*msize.get(names[l]  + 'Colorbar', msizet),
                                              vmin = -2, vmax = 3,
                                              cmap = cmap
                                              )


                    
                       
                      
                    elif COLORBAR[0] == 'deltaFirst_to_Final':

                       sc = axs[i][j].scatter(dataX[i][j][l], values, c=Colorbar, 
                                              edgecolor=colors.get(names[l]+ 'Colorbar', 'k'), alpha=alphaScater, 
                                              lw = 1.5*linewidth,
                                              marker=markers.get(names[l] + 'Colorbar', 'o'), 
                                              s=7*msize.get(names[l]  + 'Colorbar', msizet),
                                              vmin = 0, vmax = 10,
                                              cmap = cmap
                                              )
                       
                    elif COLORBAR[0] == 'MassStarIn_Over_Above_absolutevalue':

                       sc = axs[i][j].scatter(dataX[i][j][l], values, c=Colorbar, 
                                          edgecolor=colors.get(names[l]+ 'Colorbar', 'k'), alpha=alphaScater, 
                                          lw = 1.5*linewidth,
                                          marker=markers.get(names[l] + 'Colorbar', 'o'), 
                                          s=7*msize.get(names[l]  + 'Colorbar', msizet),
                                          cmap = cmap,  vmin = 0, vmax = 2,
                                          )
                       
                    elif COLORBAR[0] == 'deltaInnersSFR_afterEntry':
                       cmap = plt.cm.get_cmap(cmap)
                       bounds = [0.9, 1., 1.3]
                       norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

                       sc = axs[i][j].scatter(dataX[i][j][l], values, c=Colorbar, 
                                              edgecolor=edgecolors.get(names[l]+ 'Colorbar','k'), alpha=alphaScater, 
                                              lw = 1.5*linewidth,
                                              marker=markers.get(names[l] + 'Colorbar', 'o'), 
                                              s=7*msize.get(names[l]  + 'Colorbar', msizet),
                                              cmap = cmap,  
                                              )
                       
                       

                    elif COLORBAR[0] == 'GasjInflow_BeforeEntry':
                       sc = axs[i][j].scatter(dataX[i][j][l], values, c=Colorbar, 
                                              edgecolor=edgecolors.get(names[l] + 'Colorbar', 'black'), alpha=alphaScater, 
                                              lw = 1.5*linewidth,
                                              marker=markers.get(names[l] + 'Colorbar', 'o'), 
                                              s=7*msize.get(names[l]  + 'Colorbar', msizet),
                                              cmap = cmap,  vmin = 2, vmax = 4,
                                           )
                       
                    elif COLORBAR[0] == 'deltaInnersSFR_afterEntry_all':
                       cmap = plt.cm.get_cmap(cmap)
                       sc = axs[i][j].scatter(dataX[i][j][l], values, c=Colorbar, 
                                              edgecolor=edgecolors.get(names[l]+ 'Colorbar','k'), alpha=alphaScater, 
                                              lw = 1.5*linewidth,
                                              marker=markers.get(names[l] + 'Colorbar', 'o'), 
                                              s=7*msize.get(names[l]  + 'Colorbar', msizet),
                                              cmap = cmap, norm=mpl.colors.LogNorm(vmin = 0.05, vmax = 1.2),
                                              )
                       
                    elif COLORBAR[0] == 'z_At_FirstEntry':
                       cmap = plt.cm.get_cmap(cmap)
                       sc = axs[i][j].scatter(dataX[i][j][l], values, c=Colorbar, 
                                              edgecolor=edgecolors.get(names[l],None), alpha=alphaScater, 
                                              lw = 1.5*linewidth,
                                              marker=markers.get(names[l] + 'Colorbar', 'o'), 
                                              s=msizet*msize.get(names[l]  + 'Colorbar', msizet),
                                              cmap = cmap, norm=mpl.colors.LogNorm(vmin = 0.2, vmax = 3),
                                              )
                    elif COLORBAR[0] == 'FracStarAfterEntry_Inner':
                       cmap = plt.cm.get_cmap(cmap)
                       sc = axs[i][j].scatter(dataX[i][j][l], values, c=Colorbar, 
                                              edgecolor=edgecolors.get(names[l],None), alpha=alphaScater, 
                                              lw = 1.5*linewidth,
                                              marker=markers.get(names[l] + 'Colorbar', 'o'), 
                                              s=msizet*msize.get(names[l]  + 'Colorbar', msizet),
                                              cmap = cmap, vmin = 0.1, vmax = 0.55,
                                              )
                       
                    elif COLORBAR[0] == 'FracNew_Loss':
                       cmap = plt.cm.get_cmap(cmap)
                       sc = axs[i][j].scatter(dataX[i][j][l], values, c=Colorbar, 
                                              edgecolor=edgecolors.get(names[l],None), alpha=alphaScater, 
                                              lw = 1.5*linewidth,
                                              marker=markers.get(names[l] + 'Colorbar', 'o'), 
                                              s=msizet*msize.get(names[l]  + 'Colorbar', msizet),
                                              cmap = cmap, vmin = 0, vmax = 0.8,
                                              )
                    elif COLORBAR[0] == 'deltaTrueInnersSFR_afterEntry_all':
                       cmap = plt.cm.get_cmap(cmap)
                       sc = axs[i][j].scatter(dataX[i][j][l], values, c=Colorbar, 
                                              edgecolor=edgecolors.get(names[l]+ 'Colorbar','k'), alpha=alphaScater, 
                                              lw = 1.5*linewidth,
                                              marker=markers.get(names[l] + 'Colorbar', 'o'), 
                                              s=7*msize.get(names[l]  + 'Colorbar', msizet),
                                              cmap = cmap, norm=mpl.colors.LogNorm(vmin = 0.05, vmax = 1.2),
                                              )
                    elif COLORBAR[0] == 'MDM_Norm_Max_99':
                       cmap = plt.cm.get_cmap(cmap)
                       sc = axs[i][j].scatter(dataX[i][j][l], values, c=Colorbar, 
                                              edgecolor=edgecolors.get(names[l]+ 'Colorbar','k'), alpha=alphaScater, 
                                              lw = 1.5*linewidth,
                                              marker=markers.get(names[l] + 'Colorbar', 'o'), 
                                              s=7*msize.get(names[l]  + 'Colorbar', msizet),
                                              cmap = cmap, norm=mpl.colors.LogNorm(vmin = 0.005, vmax = 0.2),
                                              )

                    elif COLORBAR[0] == 'sSFRRatioPericenter':
                        class ThresholdNormalize(mcolors.Normalize):
                            def __init__(self, vmin=None, vmax=None, threshold=1.0, **kwargs):
                                super().__init__(vmin, vmax, **kwargs)
                                self.threshold = threshold
                        
                            def __call__(self, value, clip=None):
                                value = np.asarray(value)
                                norm_value = super().__call__(value, clip)
                                norm_value[value > self.threshold] = 1.0  # Assign max color to values above threshold
                                return norm_value
                        
                        # Create colormap (e.g., viridis) but force values above the threshold to be red
                        new_cmap = mcolors.ListedColormap(cmap(np.linspace(0, 1, 256)))  # Copy viridis
                        new_cmap.colors[-1] = [0, 0, 1, 1]   # Change last color to red (RGBA)
                        
                        # Normalize with threshold
                        norm = ThresholdNormalize(vmin=0, vmax=2, threshold=1)

                        sc = axs[i][j].scatter(dataX[i][j][l], values, c=Colorbar, 
                                              edgecolor=edgecolors.get(names[l]+ 'Colorbar','k'), alpha=alphaScater, 
                                              lw = 1.5*linewidth,
                                              marker=markers.get(names[l] + 'Colorbar', 'o'), 
                                              s=7*msize.get(names[l]  + 'Colorbar', msizet),
                                              cmap = cmap, norm= norm,#mpl.colors.AsinhNorm(vmin = -1.5, vmax = 5),
                                              )

                    elif COLORBAR[0] == 'logStarZ_99':
                        class ThresholdNormalize(mcolors.Normalize):
                            def __init__(self, vmin=None, vmax=None, threshold=1.0, **kwargs):
                                super().__init__(vmin, vmax, **kwargs)
                                self.threshold = threshold
                        
                            def __call__(self, value, clip=None):
                                value = np.asarray(value)
                                norm_value = super().__call__(value, clip)
                                norm_value[value > self.threshold] = 1.0  # Assign max color to values above threshold
                                return norm_value
                        
                        # Create colormap (e.g., viridis) but force values above the threshold to be red
                        new_cmap = mcolors.ListedColormap(cmap(np.linspace(0, 1, 256)))  # Copy viridis
                        new_cmap.colors[-1] = [0, 0, 1, 1]   # Change last color to red (RGBA)
                        
                        # Normalize with threshold
                        norm = ThresholdNormalize(vmin=0, vmax=0.6, threshold=0.3)

                        sc = axs[i][j].scatter(dataX[i][j][l], values, c=Colorbar, 
                                              edgecolor=edgecolors.get(names[l],'k'), alpha=alphaScater, 
                                              lw = 1.5*linewidth,
                                              marker=markers.get(names[l] + 'Colorbar', 'o'), 
                                              s=7*msize.get(names[l]  + 'Colorbar', msizet),
                                              cmap = cmap, norm= norm,#mpl.colors.AsinhNorm(vmin = -1.5, vmax = 5),
                                              )
                        
                    elif COLORBAR[0] == 'DeltasSFR_Ratio':

                        
                       cmap = plt.cm.get_cmap(cmap)

                       norm = mpl.colors.BoundaryNorm([-1.2, -1, 0, 1, 2, 4], cmap.N)

                       sc = axs[i][j].scatter(dataX[i][j][l], values, c=Colorbar, 
                                              edgecolor=edgecolors.get(names[l]+ 'Colorbar','k'), alpha=alphaScater, 
                                              lw = 1.5*linewidth,
                                              marker=markers.get(names[l] + 'Colorbar', 'o'), 
                                              s=7*msize.get(names[l]  + 'Colorbar', msizet),
                                              cmap = cmap, norm= norm,#mpl.colors.AsinhNorm(vmin = -1.5, vmax = 5),
                                              )
                        
                    elif COLORBAR[0] == 'deltaInnersSFR_afterEntry_all_EntryRh':
                       cmap = plt.cm.get_cmap(cmap)
                       sc = axs[i][j].scatter(dataX[i][j][l], values, c=Colorbar, 
                                              edgecolor=edgecolors.get(names[l]+ 'Colorbar','k'), alpha=alphaScater, 
                                              lw = 1.5*linewidth,
                                              marker=markers.get(names[l] + 'Colorbar', 'o'), 
                                              s=7*msize.get(names[l]  + 'Colorbar', msizet),
                                              cmap = cmap, norm=mpl.colors.LogNorm(vmin = 0.5, vmax = 3),
                                              )


                    elif COLORBAR[0] == 'dSize_NoGas_Final':
                       sc = axs[i][j].scatter(dataX[i][j][l], values, c=Colorbar, 
                                              edgecolor=edgecolors.get(names[l] + 'Colorbar', 'black'), alpha=alphaScater, 
                                              lw = 1.5*linewidth,
                                              marker=markers.get(names[l] + 'Colorbar', 'o'), 
                                              s=7*msize.get(names[l]  + 'Colorbar', msizet),
                                              cmap = cmap,  vmin = -0.006, vmax = 0.01,
                                           )
                       
                    elif COLORBAR[0] == 'DMFrac_99':
                       sc = axs[i][j].scatter(dataX[i][j][l], values, c=Colorbar, 
                                              edgecolor=edgecolors.get(names[l] + 'Colorbar', 'black'), alpha=alphaScater, 
                                              lw = 1.5*linewidth,
                                              marker=markers.get(names[l] + 'Colorbar', 'o'), 
                                              s=7*msize.get(names[l]  + 'Colorbar', msizet),
                                              cmap = cmap,  vmin = 0.3, vmax = 0.9,
                                           )
                       
                    elif COLORBAR[0] == 'sSFRRatio_Entry_to_Nogas':
                       sc = axs[i][j].scatter(dataX[i][j][l], values, c=Colorbar, 
                                              edgecolor=edgecolors.get(names[l] + 'Colorbar', 'black'), alpha=alphaScater, 
                                              lw = 1.5*linewidth,
                                              marker=markers.get(names[l] + 'Colorbar', 'o'), 
                                              s=7*msize.get(names[l]  + 'Colorbar', msizet),
                                              cmap = cmap,  vmin = 0.3, vmax = 3.4,
                                           )
                       
                    elif COLORBAR[0] == 'GasAbove_Entry_to_Nogas':
                       sc = axs[i][j].scatter(dataX[i][j][l], values, c=Colorbar, 
                                              edgecolor=edgecolors.get(names[l] + 'Colorbar', 'black'), alpha=alphaScater, 
                                              lw = 1.5*linewidth,
                                              marker=markers.get(names[l] + 'Colorbar', 'o'), 
                                              s=7*msize.get(names[l]  + 'Colorbar', msizet),
                                              cmap = cmap,  vmin = -0.3, vmax = -0.05,
                                           )
                       
                    elif COLORBAR[0] == 'sSFRInner_Entry_to_Nogas':
                       sc = axs[i][j].scatter(dataX[i][j][l], values, c=Colorbar, 
                                              edgecolor=edgecolors.get(names[l] + 'Colorbar', 'black'), alpha=alphaScater, 
                                              lw = 1.5*linewidth,
                                              marker=markers.get(names[l] + 'Colorbar', 'o'), 
                                              s=7*msize.get(names[l]  + 'Colorbar', msizet),
                                              cmap = cmap,  vmin = -10.8, vmax = -9.1,
                                           )
                    
                    elif COLORBAR[0] == 'dSize_Max_to_Nogas':

                       sc = axs[i][j].scatter(dataX[i][j][l], values, c=Colorbar, 
                                              edgecolor=edgecolors.get(names[l] + 'Colorbar', 'black'), alpha=alphaScater, 
                                              lw = 1.5*linewidth,
                                              marker=markers.get(names[l] + 'Colorbar', 'o'), 
                                              s=7*msize.get(names[l]  + 'Colorbar', msizet),
                                              cmap = cmap,  vmin = -1, vmax = 0.05,
                                           )
                    
                    elif COLORBAR[0] == 'dSize_Entry_to_Max':

                       sc = axs[i][j].scatter(dataX[i][j][l], values, c=Colorbar, 
                                              edgecolor=edgecolors.get(names[l] + 'Colorbar', 'black'), alpha=alphaScater, 
                                              lw = 1.5*linewidth,
                                              marker=markers.get(names[l] + 'Colorbar', 'o'), 
                                              s=7*msize.get(names[l]  + 'Colorbar', msizet),
                                              cmap = cmap,  vmin = -1.2, vmax = 0.3,
                                           )
                       
                    elif COLORBAR[0] == 'InnersSFR_Entry_to_quench':
                       cmap = plt.cm.get_cmap(cmap)
                       bounds = [0.9, 1., 1.3]
                       norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

                       sc = axs[i][j].scatter(dataX[i][j][l], values, c=Colorbar, 
                                              edgecolor=edgecolors.get(names[l]+ 'Colorbar','k'), alpha=alphaScater, 
                                              lw = 1.5*linewidth,
                                              marker=markers.get(names[l] + 'Colorbar', 'o'), 
                                              s=7*msize.get(names[l]  + 'Colorbar', msizet),
                                              cmap = cmap,  vmin = -11.5, vmax = -9,
                                              )
                       
                    elif COLORBAR[0] == 'MassAboveAfterInfall_Lost':
                       sc = axs[i][j].scatter(dataX[i][j][l], values, c=Colorbar, 
                                              edgecolor=edgecolors.get(names[l] , 'black'), alpha=alphaScater, 
                                              lw = 1.5*linewidth,
                                              marker=markers.get(names[l] + 'Colorbar', 'o'), 
                                              s=7*msize.get(names[l]  + 'Colorbar', msizet),
                                              cmap = cmap,  vmin = -0.66, vmax = 0.0,
                                           )
                       
                    elif COLORBAR[0] == 'TimeLossGass':
                       sc = axs[i][j].scatter(dataX[i][j][l], values, c=Colorbar, 
                                          edgecolor=edgecolors.get(names[l]+ 'Colorbar', 'k'), alpha=alphaScater, 
                                          lw = 1.5*linewidth,
                                          marker=markers.get(names[l] + 'Colorbar', 'o'), 
                                          s=7*msize.get(names[l]  + 'Colorbar', msizet),
                                          cmap = cmap,  vmin = 0, vmax =12,
                                          )
                  
                    elif COLORBAR[0] == 'StarMass_GasLoss_Over_EntryToGas':
                       sc = axs[i][j].scatter(dataX[i][j][l], values, c=Colorbar, 
                                          edgecolor=edgecolors.get(names[l]+ 'Colorbar', 'k'), alpha=alphaScater, 
                                          lw = 1.5*linewidth,
                                          marker=markers.get(names[l] + 'Colorbar', 'o'), 
                                          s=7*msize.get(names[l]  + 'Colorbar', msizet),
                                          cmap = cmap,  vmin = 0, vmax =1,
                                          )
                       
                    elif COLORBAR[0] == 'DeltaStarMass_Above_Normalize_99':
                        cmap = plt.cm.get_cmap(cmap)
                        bounds = [0.5, 1., 2,3]
                        sc = axs[i][j].scatter(dataX[i][j][l], values, c=Colorbar, 
                                          edgecolor=colors.get(names[l]+ 'Colorbar', 'k'), alpha=alphaScater, 
                                          lw = 1.5*linewidth,
                                          marker=markers.get(names[l] + 'Colorbar', 'o'), 
                                          s=7*msize.get(names[l]  + 'Colorbar', msizet),
                                          cmap = cmap,  vmin = 0.02, vmax =10.2, 
                                          )
                       
                    elif COLORBAR[0] == 'logHalfRadstar_99':
                        sc = axs[i][j].scatter(dataX[i][j][l], values, c=Colorbar, 
                                               edgecolor=edgecolors.get(names[l]+'Colorbar','k'), alpha=alphaScater, 
                                               lw = 1.5*linewidth,
                                               marker=markers.get(names[l]+'Colorbar', 'o'), 
                                               s=msizet*msize.get(names[l]+'Colorbar', msizet),
                                               cmap = cmap, vmin = 0.035, vmax = 0.85,
                                               )
                        
                    elif COLORBAR[0] == 'logSUM_Mstar_merger_Corotate':
                        sc = axs[i][j].scatter(dataX[i][j][l], values, c=Colorbar, 
                                               edgecolor=edgecolors.get(names[l]+'Colorbar','k'), alpha=alphaScater, 
                                               lw = 1.5*linewidth,
                                               marker=markers.get(names[l]+'Colorbar', 'o'), 
                                               s=msizet*msize.get(names[l]+'Colorbar', msizet),
                                               cmap = cmap, vmin = 5, vmax = 8.5,
                                               )
  
                    elif COLORBAR[0] == 'fEx_at_99':
                       sc = axs[i][j].scatter(dataX[i][j][l], values, c=Colorbar, 
                                          edgecolor=colors.get(names[l]+ 'Colorbar', 'k'), alpha=alphaScater, 
                                          lw = 1.5*linewidth,
                                          marker=markers.get(names[l] + 'Colorbar', 'o'), 
                                          s=msizet*msize.get(names[l]  + 'Colorbar', msizet), #norm=mpl.colors.LogNorm(),
                                          cmap = cmap,  vmin = 0.002, vmax =0.1, 
                                          )
                       
                    elif COLORBAR[0] == 'M200Mean':
                       sc = axs[i][j].scatter(dataX[i][j][l], values, c=Colorbar, 
                                          edgecolor=colors.get(names[l]+ 'Colorbar', 'k'), alpha=alphaScater, 
                                          lw = 1.5*linewidth,
                                          marker=markers.get(names[l] + 'Colorbar', 'o'), 
                                          s=msizet*msize.get(names[l]  + 'Colorbar', msizet), #norm=mpl.colors.LogNorm(),
                                          cmap = cmap,  vmin = 11.7, vmax =13.3, 
                                          )
                    elif COLORBAR[0] == 'rOverR200Min':
                       sc = axs[i][j].scatter(dataX[i][j][l], values, c=Colorbar, 
                                          edgecolor=colors.get(names[l]+ 'Colorbar', 'k'), alpha=alphaScater, 
                                          lw = 1.5*linewidth,
                                          marker=markers.get(names[l] + 'Colorbar', 'o'), 
                                          s=msizet*msize.get(names[l]  + 'Colorbar', msizet), #norm=mpl.colors.LogNorm(),
                                          cmap = cmap,  vmin = 0.002, vmax =0.2, 
                                          )
                       
                    elif COLORBAR[0] == 'Norbit_Entry_To_NoGas':
                       cmap = plt.cm.get_cmap(cmap)

                       #norm = mpl.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

                       sc = axs[i][j].scatter(dataX[i][j][l], values, c=Colorbar, 
                                          edgecolor=colors.get(names[l]+ 'Colorbar', 'k'), alpha=alphaScater, 
                                          lw = 1.5*linewidth,
                                          marker=markers.get(names[l] + 'Colorbar', 'o'), 
                                          s=msizet*msize.get(names[l]  + 'Colorbar', msizet), #norm=norm,
                                          cmap = cmap,
                                          )
                       
                    elif COLORBAR[0] == 'SnapLostGas':
                       cmap = plt.cm.get_cmap(cmap)

                       #norm = mpl.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

                       sc = axs[i][j].scatter(dataX[i][j][l], values, c=Colorbar, 
                                          edgecolor=colors.get(names[l]+ 'Colorbar', 'k'), alpha=alphaScater, 
                                          lw = 1.5*linewidth,
                                          marker=markers.get(names[l] + 'Colorbar', 'o'), 
                                          s=msizet*msize.get(names[l]  + 'Colorbar', msizet),  vmin = 0, vmax = 14,
                                          cmap = cmap,
                                          )
                       
                    elif COLORBAR[0] == 'NCorotateMergers':
                        cmap = plt.cm.get_cmap("Blues", 4)  
                        Colorbar[Colorbar ==0] = np.nan
                        # define os limites para 0,1,2,3
                        bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
                        norm = mcolors.BoundaryNorm(bounds, cmap.N)


                        sc = axs[i][j].scatter(dataX[i][j][l], values, c=Colorbar, 
                                          edgecolor=colors.get(names[l]+ 'Colorbar', 'k'), alpha=alphaScater, 
                                          lw = 1.5*linewidth,
                                          marker=markers.get(names[l] + 'Colorbar', 'o'), 
                                          s=msizet*msize.get(names[l]  + 'Colorbar', msizet),  norm=norm,
                                          cmap = cmap,
                                          )
                       
                    elif COLORBAR[0] == 'deltaRatio_gastoend_entrytogas':
                       sc = axs[i][j].scatter(dataX[i][j][l], values, c=Colorbar, 
                                          edgecolor=colors.get(names[l]+ 'Colorbar', 'k'), alpha=alphaScater, 
                                          lw = 1.5*linewidth,
                                          marker=markers.get(names[l] + 'Colorbar', 'o'), 
                                          s=msizet*msize.get(names[l]  + 'Colorbar', msizet), #norm=mpl.colors.LogNorm(),
                                          cmap = cmap,  vmin = 0, vmax =5, 
                                          )
                       
                    else:
                        
                        sc = axs[i][j].scatter(dataX[i][j][l], values, c=Colorbar, 
                                           edgecolor=colors.get(names[l]+ 'Colorbar', None), alpha=alphaScater, 
                                           lw = 1.5*linewidth,
                                           marker=markers.get(names[l] + 'Colorbar', 'o'), 
                                           s=msizet*msize.get(names[l]  + 'Colorbar', msizet),
                                           cmap = cmap,  vmin = min(Colorbar), vmax = max(Colorbar),
                                           )
                        
                    
                    
                else:
                    if 'BadFlag' in names[l]:
                        edcolor = 'red'
                    elif NoneEdgeColor:
                        edcolor = None
                    else:
                        edcolor = 'black'
                    axs[i][j].scatter( dataX[i][j][l], values, color=colors.get(
                            names[l], 'k'), edgecolor= edcolor, alpha=alphaScater, lw = linesthicker.get(names[l], linewidth), marker=markers.get(names[l], 'o'), 
                            s=msizet*msize.get(names[l], msizet))

                if medianBins:
                    xmeanfinal, ymedian, yquantile95, yquantile5 = MATH.split_quantiles(
                        dataX[i][j][l], values, total_bins=bins, quantile=quantile)

                    axs[i][j].errorbar(xmeanfinal, ymedian, yerr=(ymedian - yquantile5, yquantile95 - ymedian),
                                       ls='None', markeredgecolor='black', elinewidth=2, ms=10, fmt='s', c=colors[names[l]])
                elif medianDot:
                    xMedianDot =  np.append(xMedianDot, np.nanmedian( dataX[i][j][l]))
                    yMedianDot =  np.append(yMedianDot, np.nanmedian( values))
                    NamesMedianDot = np.append(NamesMedianDot, names[l])
                    if not (SpearManTest or SpearManTestAll):
                        print(names[l], ' X = ', np.nanmedian( dataX[i][j][l]), 'Y = ', np.nanmedian( values))
                    if COLORBAR != None and not (SpearManTest or SpearManTestAll):
                        print('Colorbar = ', np.nanmedian( Colorbar))

                    axs[i][j].scatter(np.nanmedian( dataX[i][j][l]), np.nanmedian( values),
                                       marker='*', edgecolor='black', c=colors.get(names[l], 'black'), s = 450, lw = 1.5)

                elif medianAll:
                    xmeanfinal, ymedian, yquantile95, yquantile5 = MATH.split_quantiles(
                        dataX[i][j][l], values, total_bins=bins)
                    axs[i][j].plot(xmeanfinal, ymedian, color=colors.get(
                        names[l]), ls=lines.get(names[l]), linewidth=linewidth)
                    axs[i][j].fill_between(xmeanfinal, yquantile5,  yquantile95, color=colors.get(
                        names[l]), alpha=alphaShade)
                    
                if SpearManTestAll:
                    
                    XAllSMT = np.append(XAllSMT, dataX[i][j][l])
                    YAllSMT  = np.append(YAllSMT, values)
                    try:
                        ColorALLSMT =  np.append(ColorALLSMT, Colorbar)
                    except:
                        None
            

            # Plot details
            
            if SpearManTestAll:
                # Perform Spearman rank correlation test
                try:
                    print('Name: ', title[j])
                except:
                    print('Name: ', texts.get(param))
                condX = ((~np.isnan(XAllSMT)) & (~np.isinf(XAllSMT))) 
                condY = ((~np.isnan(YAllSMT)) & (~np.isinf(YAllSMT))) 
                try:
                    condC = ((~np.isnan(ColorALLSMT)) & (~np.isinf(ColorALLSMT))) 
                except:
                    None       
                
                correlation, p_value = spearmanr(XAllSMT[condX & condY], YAllSMT[condX & condY])
                
                # Results
                print('X and Y')
                print(f"Spearman rank correlation coefficient: {correlation:.3f}")
                print(f"P-value: {p_value:.2e}")
                
                try:
                    correlation, p_value = spearmanr( XAllSMT[condX & condC],  ColorALLSMT[condX & condC])
                
                    # Results
                    print('X and ColorBAR')
                    print(f"Spearman rank correlation coefficient: {correlation:.3f}")
                    print(f"P-value: {p_value:.2e}")
                    
                    correlation, p_value = spearmanr(YAllSMT[condC & condY],  ColorALLSMT[condC & condY])
                    
                    # Results
                    print('Y and ColorBAR')
                    print(f"Spearman rank correlation coefficient: {correlation:.3f}")
                    print(f"P-value: {p_value:.2e}")
                except:
                    None   
                
                
            if medianDot:
                for ind, NameMedianDot in enumerate(NamesMedianDot):
                    axs[i][j].scatter(xMedianDot[ind], yMedianDot[ind],
                                       marker='*', edgecolor='black', c=colors.get(NameMedianDot, 'k'), s = 350, lw = 1.25)
                

            if GridMake:
                axs[i][j].grid(GridMake, color='#9e9e9e',  which="major", linewidth= 0.6,alpha= 0.3 , linestyle=':')

            
            if  ('StarFrac' in ParamX  and (('GasFrac' in param) or ('DMFrac' in param)) )  :

                x = np.linspace(0, 1)
                axs[i][j].plot( x, x, ls='--', color='tab:blue', linewidth=linewidth)
                
            if  ( ParamX == 'StarMass_In_Normalize_99' and (param == 'StarMass_Above_Normalize_99')) :
                x = np.linspace(0, 1)
                axs[i][j].plot( x, x, ls='--', color='tab:red', linewidth=linewidth)
            if  ( ParamX == 'DeltaStarMass_In_Normalize_99' or ParamX == 'MassIn_Infall_to_GasLost') and (param == 'DeltaStarMass_Above_Normalize_99') :
                x = np.linspace(0, 1.4)
                axs[i][j].plot( x, x, ls='--', color='tab:red', linewidth=linewidth)
       
            if   ('BeforesSFR_Entry' in ParamX  and ('MaxInnersSFR_afterEntry' in param or 'InnersSFR_Entry_to_quench' in param)):
                x = np.linspace(-12,-8)
                axs[i][j].plot( x, x, ls='--', color='tab:blue', linewidth=linewidth)

    
                axs[i][j].set_yscale(scales.get(param, 'linear'))
                
            if   ('sSFRInner_BeforeEntry' in ParamX  and ('sSFRInner_Entry_to_Nogas' in param)):
                x = np.linspace(-12,-8)
                axs[i][j].plot( x, x, ls='--', color='tab:blue', linewidth=linewidth)

    
                axs[i][j].set_yscale(scales.get(param, 'linear'))
            if   ('sSFRTrueInner_BeforeEntry' in ParamX  and ('sSFRTrueInner_Entry_to_Nogas' in param)):
                x = np.linspace(-12,-8)
                axs[i][j].plot( x, x, ls='--', color='tab:blue', linewidth=linewidth)

    
                axs[i][j].set_yscale(scales.get(param, 'linear'))   
            if   ('sSFRTrueInner_BeforeEntry' in ParamX  and ('sSFRAfterPericenter' in param)):
                x = np.linspace(-12,-8)
                axs[i][j].plot( x, x, ls='--', color='tab:blue', linewidth=linewidth)

    
                axs[i][j].set_yscale(scales.get(param, 'linear')) 
            if   ('IN_sSFR_Entry' in ParamX  and ('IN_sSFR_Entry_No_gas' in param or 'IN_sSFR_1sfPert' in param)):
                x = np.linspace(-12,-8)
                axs[i][j].plot( x, x, ls='--', color='tab:blue', linewidth=linewidth)

            axs[i][j].set_yscale(scales.get(param, 'linear'))
            if   ('IN_GasMass_Entry' in ParamX  and ('IN_GasMass_1sfPert' in param or 'InnersSFR_Entry_to_quench' in param)):
                x = np.linspace(7,9)
                axs[i][j].plot( x, x, ls='--', color='tab:blue', linewidth=linewidth)

            axs[i][j].set_yscale(scales.get(param, 'linear'))
            if   ('IN_StarMass_Entry' in ParamX  and ('IN_StarMass_1sfPert' in param or 'InnersSFR_Entry_to_quench' in param)):
                x = np.linspace(0,2)
                axs[i][j].plot( x, x, ls='--', color='tab:blue', linewidth=linewidth)

            axs[i][j].set_yscale(scales.get(param, 'linear'))
            if   ('IN_SFR_Entry' in ParamX  and ('IN_SFR_1sfPert' in param or 'InnersSFR_Entry_to_quench' in param)):
                x = np.linspace(-2,0)
                axs[i][j].plot( x, x, ls='--', color='tab:blue', linewidth=linewidth)

            if EqualLine:
                x = np.linspace(EqualLineMin,EqualLineMax)

                axs[i][j].plot( x, x, ls='--', color='tab:blue', linewidth=linewidth)

            axs[i][j].set_yscale(scales.get(param, 'linear'))
            
            if scales.get(param)  ==  'log' or scales.get(param)  ==  'symlog' :
                axs[i][j].yaxis.set_major_formatter(
                    FuncFormatter(format_func_loglog))

            if  ParamX == 'Decrease_Entry_To_NoGas_Norm_Delta':
                axs[i][j].set_xticks([-0.8, -0.6, -0.4, -0.2, 0.0, 0.2])
                axs[i][j].set_xticklabels(['-0.8', '-0.6', '-0.4', '-0.2', '0.0', '0.2'])

                axs[i][j].set_yticks([-0.8, -0.6, -0.4, -0.2, 0.0, 0.2])
                axs[i][j].set_yticklabels(['-0.8', '-0.6', '-0.4', '-0.2', '0.0', '0.2'])
                
            if legend:
                for legpos, LegendName in enumerate(LegendNames):
                    if j == legpositions[legpos][0] and i ==legpositions[legpos][1]:
                        custom_lines, label, ncol, mult = Legend(
                            LegendName, msizeMult = msizeMult, linewidth = linewidth)
                        axs[i][j].legend(
                            custom_lines, label,ncol=ncol, loc=loc[legpos], fontsize=0.88*fontlabel, framealpha = framealpha, 
                            columnspacing = columnspacing, handlelength = handlelength, handletextpad = handletextpad, labelspacing = labelspacing)


            if j == 0:
                if LabelGeral:
                    axs[i][j].set_ylabel(labelsequal.get(param, param), fontsize=fontlabel)
                else:
                    axs[i][j].set_ylabel(labels.get(param, param), fontsize=fontlabel)
                axs[i][j].tick_params(axis='y', labelsize=0.99*fontlabel)

                if ylimmin != None and ylimmax != None:
                    axs[i][j].set_ylim(ylimmin[i], ylimmax[i])
                
                if 'Snap' in param and ylimmin == None:
                    axs[i][j].set_ylim(-0.2, 14.2)
                    axs[i][j].set_yticks([0, 2, 4, 6, 8, 10, 12, 14])
                    axs[i][j].set_yticklabels(
                        ['0', '2', '4', '6', '8', '10', '12', '14'])
                    
                if 'DMFrac_Birth' in param:
                    axs[i][j].set_yticks([0.001, 0.01, 0.1, 0.5, 0.9, 0.99])
                    axs[i][j].set_yticklabels(
                        ['$10^{-3}$', '$10^{-2}$', '0.1', '0.5', '0.9', '0.99'])
                
                if 'DecreaseBetweenGasStar_Over_starFinal' in param:
                    axs[i][j].set_yticks([-0.5, 0, 0.5, 1])
                    axs[i][j].set_yticklabels(
                        ['-0.5', '0', '0.5', '1'])

              
                if 'MassAboveAfter_over_In_Infall_to_GasLost' in param:
                    axs[i][j].set_yticks([-5, -2, -1, -0.5, 0,  0.5, 1])
                    axs[i][j].set_yticklabels(
                        ['-5', '-2', '-1', '-0.5',  '0',  '0.5',  '1'])
                    
                    x = np.linspace(0, 0.7)
                    #axs[i][j].plot(x, -2*x)
            if j == len(columns) - 1:
                if xlabelintext:
                    Afont = {'color':  'black',
                             'size': fontlabel,
                             }
                    anchored_text = AnchoredText(
                        texts.get(param), loc='upper right', prop=Afont)
                    axs[i][j].add_artist(anchored_text)

            if i == 0:
                if columns[j] == 'Snap':
                    axs[i][j].set_title(
                        r'$z = %.1f$' % dfTime.z.loc[dfTime.Snap == snap[j]].values[0], fontsize=1.1*fontlabel)
                if title:
                    axs[i][j].set_title(titles.get(
                        title[j], title[j]), fontsize=1.1*fontlabel)

            if i == len(ParamsY) - 1:
                if LabelGeral:
                    axs[i][j].set_xlabel(labelsequal.get(ParamsX[j], ParamsX[j]), fontsize=fontlabel)
                    axs[i][j].set_xscale(scales.get(ParamsX[j], 'linear'))
                    if scales.get(ParamsX[j]) == 'log' or scales.get(ParamsX[j]) == 'symlog':
                        axs[i][j].xaxis.set_major_formatter(
                            FuncFormatter(format_func_loglog))
                          

                else:
                    axs[i][j].set_xlabel(labels.get(ParamX, ParamX), fontsize=fontlabel)

                    axs[i][j].set_xscale(scales.get(ParamX, 'linear'))
                    if scales.get(ParamX) == 'log' or scales.get(ParamX) == 'symlog':
                        axs[i][j].xaxis.set_major_formatter(
                            FuncFormatter(format_func_loglog))
                      
                if xlimmin != None and xlimmax != None:
                    axs[i][j].set_xlim(xlimmin[i], xlimmax[i])

                axs[i][j].tick_params(axis='x', labelsize=0.99*fontlabel)
                
                if ParamX == 'DecreaseBeforeGas':
                    axs[i][j].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.])
                    axs[i][j].set_xticklabels(
                        ['', '0.2', '0.4', '0.6', '0.8', '1.0'])
                    
                if  ParamX == 'Decrease_Entry_To_NoGas_Norm_Delta':
                    axs[i][j].set_xticks([-0.8, -0.6, -0.4, -0.2, 0.0, 0.2])
                    axs[i][j].set_xticklabels(['-0.8', '-0.6', '-0.4', '-0.2', '0.0', '0.2'])

                if 'Snap' in ParamX:
                    axs[i][j].set_xlim(-0.2,  14.2)
                    axs[i][j].set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
                    axs[i][j].set_xticklabels(
                        ['0', '2', '4', '6', '8', '10', '12', '14'])
                if ParamX == 'MassIn_Infall_to_GasLost':
                    axs[i][j].set_xticks([-0.15, 0, 0.25, 0.5, 0.75])
                    axs[i][j].set_xticklabels(
                        ['-0.15', '0', '0.25', '0.50', '0.75'])
                
                if ParamX == 'MassIn_Infall_to_GasLost' and  ParamsY[0] == 'MassAboveAfter_Infall_to_GasLost':
                    x = np.linspace(0, 1)
                    y = -x
                    axs[i][j].plot(x, y,color = 'darkorange',linestyle='dashed',lw=2)
                    
                if  'StarFrac' in ParamX   and 'GasFrac' in param and not ylimmin == [0.001]:
                    axs[i][j].tick_params(axis='y', labelsize=0.88*fontlabel)
                    axs[i][j].tick_params(axis='x', labelsize=0.88*fontlabel)

                    axs[i][j].set_yticks([ 0.02, 0.03, 0.04, 0.06, 0.08, 0.1])
                    axs[i][j].set_yticklabels(
                        ['0.02', '0.03', '0.04', '0.06', '0.08', '0.1'])
                    axs[i][j].set_xticks([0.004, 0.006, 0.01, 0.02, 0.03])
                    axs[i][j].set_xticklabels(
                        ['0.004', '0.006', '0.01', '0.02', '0.03'])
       

            
    if COLORBAR != None:
        if 'Snap' in COLORBAR[0]:
                        cb =  fig.colorbar(sc,  ax=axs.ravel().tolist(), ticks=[0.  ,  1.97185714,  3.94371429,  5.91557143,  7.88742857, 9.85928571, 11.83114286, 13.803  ], pad=0.02, aspect = 30)
                        cb.ax.set_yticklabels(['14', '12', '10', '8', '6', '4', '2', '0'])
        else:
            if COLORBAR[0] == 'deltaInnersSFR_afterEntry':
                cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),  ax=axs.ravel().tolist(), pad=0.02, aspect = 30)
            elif COLORBAR[0] == 'Norbit_Entry_To_NoGas':
                cb = fig.colorbar(sc,  ax=axs.ravel().tolist(), ticks=[0  ,  1,  2,  3, 4], pad=0.02, aspect = 30)
                cb.ax.set_yticklabels(['0', '1', '2', '3', '4'])
            elif COLORBAR[0] == 'RatioSFR':
                cb = fig.colorbar(sc,  ax=axs.ravel().tolist(), ticks=[0.01 , 0.1, 1], pad=0.02, aspect = 30)
                cb.ax.set_yticklabels(['0.01', '0.1', '1'])
            elif COLORBAR[0] == 'deltaInnersSFR_afterEntry_all':
                cb = fig.colorbar(sc,  ax=axs.ravel().tolist(), ticks=[0.01, 0.1, 1], pad=0.02, aspect = 30)
                #cb.ax.set_ylim(0.009, 0.07)
                cb.ax.set_yticklabels(['0.01',  '0.1', '1'])
            elif COLORBAR[0] == 'sSFRRatio_ETOGAS_Before':
                cb = fig.colorbar(sc,  ax=axs.ravel().tolist(), ticks=[0.1, 1], pad=0.02, aspect = 30)
                #cb.ax.set_ylim(0.009, 0.07)
                cb.ax.set_yticklabels(['0.1', '1'])
            elif COLORBAR[0] == 'deltaTrueInnersSFR_afterEntry_all':
                cb = fig.colorbar(sc,  ax=axs.ravel().tolist(), ticks=[ 0.1, 1], pad=0.02, aspect = 30)
                #cb.ax.set_ylim(0.009, 0.07)
                cb.ax.set_yticklabels(['0.1', '1'])
            elif COLORBAR[0] == 'sSFRRatioPericenter':
                cb = fig.colorbar(sc,  ax=axs.ravel().tolist(), ticks=[0, 0.25, 0.5, 0.75, 1,  2], pad=0.02, aspect = 30)
                #cb.ax.set_ylim(0.009, 0.07)
                cb.ax.set_yticklabels(['0', '0.25', '0.5', '0.75', '1', '2'])
            elif COLORBAR[0] == 'logStarZ_99':
                cb = fig.colorbar(sc,  ax=axs.ravel().tolist(), ticks=[0, 0.1, 0.2, 0.3, 0.7], pad=0.02, aspect = 30)
                #cb.ax.set_ylim(0.009, 0.07)
                cb.ax.set_yticklabels(['0', '0.1', '0.2', '0.3', '0.7'])
            elif COLORBAR[0] == 'DeltasSFR_Ratio':
                cb = fig.colorbar(sc,  ax=axs.ravel().tolist(), ticks= [-1,  0, 1, 2, 4], pad=0.02, aspect = 30)
                #cb.ax.set_ylim(0.009, 0.07)
                cb.ax.set_yticklabels(['-1', '0', '1', '2', '4'])

            elif COLORBAR[0] == 'MDM_Norm_Max_99':
                cb = fig.colorbar(sc,  ax=axs.ravel().tolist(), ticks=[ 0.01, 0.1], pad=0.02, aspect = 30)
                #cb.ax.set_ylim(0.009, 0.07)
                cb.ax.set_yticklabels(['0.01', '0.1'])

            elif COLORBAR[0] == 'deltaInnersSFR_afterEntry_all_EntryRh':
                cb = fig.colorbar(sc,  ax=axs.ravel().tolist(), ticks=[0.6, 0.8, 1, 2, 3], pad=0.02, aspect = 30)
                #cb.ax.set_ylim(0.009, 0.07)
                cb.ax.set_yticklabels(['', '0.8',  '1', '2', ''])
            else:
                cb = fig.colorbar(sc,  ax=axs.ravel().tolist(), pad=0.02, aspect = 30)
        cb.set_label(labels.get(COLORBAR[0]), fontsize=1*fontlabel)
        cb.ax.tick_params(labelsize=0.99*fontlabel)
    savefig(savepath, savefigname, TRANSPARENT)

    return

def PlotHist(names, columns, rows, Type='z0', snap=[99], density=False,  mean=False, legend=False, ColumnPlot=True, NormCount=False,
             LegendNames=None, title=False, median=False, medianPlot=False, xlabelintext=False, legendColumn = False,
             Supertitle = False, LookBackTime = False,  TRANSPARENT = False,
             alphaShade=0.3,  linewidth=1.8, fontlabel=24,toplim = 1e3,
             nboots=100, framealpha = 0.95,  savepath='fig/PlotHist', 
             ylimmin = None, ylimmax = None, xlimmin = None, xlimmax = None, legpositions = None, 
             lNum = 6, cNum = 6, GridMake = False, JustOneXlabel = False,
             savefigname='fig', yscale='linear', xscale = 'linear', dfName='Sample', SampleName='Samples', loc='best', 
             columnspacing = 0.5, handlelength = 2 , handletextpad = 0.4, labelspacing = 0.3,  SupertitleName = '',
             limaixsy=False, liminvalue=[0], limax=[1], bins='rice', seed=16010504, Supertitle_y = 1.22):


    '''
    Plot Hist
    Parameters
    ----------
    names : sample names. array with str
    columns : specific set in the sample / or different param to plot in each column. array with str
    rows : specific set in the sample / or different param to plot in each row. array with str
    Type : plot type. Default: 'z0' since the plot is for a specific snapshot. str
    snap : specific snapshot. Default: 99. int in [0, 99]
    density : to plot a PDF. Default: False. bool
    legend : to make the legend. Default: False. bool
    LegendNames : name in the legend. Default: None. None or array with str
    title : to make the title. Default: False. array with str
    mean : to show the mean line. Default: False. bool
    median : to show the median line. Default: False. bool
    medianPlot : to show the median error region. Default: False. bool
    xlabelintext : to show the x param as a text. Default: False. bool
    legendColumn : if the legend is in the column 0 in each row. Default: False. bool
    alphaShade : alpha value for the median error region. Default: 0.3. float
    linewidth : linewidth for histogram. Default: 0.3. float
    fontlabel : fontsize for labels. Default: 24. float
    nboots  : fboots for compute median error. Default: 100. int
    framealpha  : alpha for legend. Default: 0.95. float
    ColumnPlot : to make a column plot. Default: True. bool
    NormCount : normalized histogram. Default: False. bool
    savepath : save path. Default: 'fig/PlotHist'. str
    savefigname : fig name. Default: 'fig'. str
    yscale  : scale for y axis. Default: 'linear'. str
    dfName : for the complete dataframe sample to select the subhalos. Default: None. str
    SampleName : the sample name if using a specific dataframe sample. Default: Sample. str
    loc : loc legend. Default: 'best'. str
    limaixsy : limit y axis. Default: False. bool
    liminvalue : min limit y axis. Default:[0]. array with float
    limax : max limit y axis. Default:[1]. array with float
    bins : bins for the histogram. Default:10. int
    seed : seed if you use some random choice. Default: 16010504.
    
    Returns
    -------
    Requested Histogram
    -------
    Author: Abhner P. de Almeida (abhner.almeida AAT usp.br)
    '''
    np.random.seed(seed)

    # Verify NameParameters
    if type(columns) is not list and type(columns) is not np.ndarray:
        columns = [columns]

    if type(rows) is not list and type(rows) is not np.ndarray:
        rows = [rows]

    if ColumnPlot:
        if columns == 'Snap':
            columns = snap
            datas = TNG.makedata(names, columns, rows, 'Snap',
                             snap=snap, dfName=dfName, SampleName=SampleName)
        else:
            datas = TNG.makedata(names, columns, rows, Type,
                             snap=snap, dfName=dfName, SampleName=SampleName)
    else:
        if columns == 'Snap':
            columns = snap
            datas = TNG.makedata(names, rows, columns, 'Snap',
                             snap=snap, dfName=dfName, SampleName=SampleName)
        else:
            datas = TNG.makedata(names, rows, columns, Type,
                             snap=snap, dfName=dfName, SampleName=SampleName)


    dfTime = TNG.extractDF('SNAPS_TIME')

    # Define axes
    plt.rcParams.update({'figure.figsize': (cNum*len(columns),lNum*len(rows))})
    fig = plt.figure()
    gs = fig.add_gridspec(len(rows), len(columns), hspace=0, wspace=0)
    axs = gs.subplots(sharex='col', sharey='row')

    # Verify axs shape
    if type(axs) is not list and type(axs) is not np.ndarray:
        axs = [axs]
    if type(axs[0]) is not np.ndarray:
        axs = np.array([axs])
        if len(columns) == 1:
            axs = axs.T

    for i, row in enumerate(rows):

        for j, column in enumerate(columns):

            if ColumnPlot:
                titlename = column
                param = row
                data = datas[i][j]
            else:
                titlename = row
                param = column
                data = datas[j][i]

            for l, values in enumerate(data):

                values = np.array([value for value in values])

                if 'Snap' in param:
                    values = np.array([dfTime.loc[dfTime.Snap == int(v), 'Age'].values[0] for v in values ])
                
                valuesOriginal = values
                values = values[~np.isinf(values)]
                
               
                if len(values) == 0:
                    continue

                if density:

                    kde = stats.gaussian_kde(values)
                    xx = np.linspace(min(values), max(values), 1000)
                    factor = 1
                    axs[i][j].plot(xx, factor*kde(xx), color=colors.get(names[l], 'black'),
                                   ls=lines.get(names[l], 'solid'), linewidth=linewidth, dash_capstyle = capstyles.get(names[l], 'projecting'))

                else:
                    if NormCount:
                        hist, bin_edges = np.histogram(
                            values, bins=bins, density=density)
                        axs[i][j].step(bin_edges[:-1], hist/sum(hist), color=colors.get(
                            names[l], 'black'), ls=lines.get(names[l], 'solid'))
                    else:
                        if type(bins) == list:
                            binnumber = bins[i][l]
                        else:
                            binnumber = bins
                        if param == 'rOverR200Min'  or param == 'rOverR200_99':
                            axs[i][j].hist(values, bins=np.geomspace(0.03,5,binnumber), color=colors.get(names[l], 'black'), log=True, alpha=1, histtype='step', ls=lines.get(
                                names[l], 'solid'), density=density,  linewidth=linewidth)
                        elif param == 'r_over_R_Crit200':
                            axs[i][j].hist(values, bins=np.logspace(np.log10(0.1),np.log10(10), 20), color=colors.get(names[l], 'black'), alpha=1, histtype='step', ls=lines.get(
                                names[l], 'solid'), density=density,  linewidth=linewidth)
                        elif param == 'GasFrac':
                            axs[i][j].hist(values, bins=np.logspace(np.log10(0.001),np.log10(0.5), 10), color=colors.get(names[l], 'black'), alpha=1, histtype='step', ls=lines.get(
                                names[l], 'solid'), density=density,  linewidth=linewidth)
                        elif param == 'rho_50':
                            axs[i][j].hist(values, bins=np.logspace(np.log10(0.001),np.log10(1e5), 10), color=colors.get(names[l], 'black'), alpha=1, histtype='step', ls=lines.get(
                                names[l], 'solid'), density=density,  linewidth=linewidth)
                        elif param == 'DMFrac':
                            axs[i][j].hist(values, bins=np.logspace(np.log10(0.15),np.log10(1), 10), color=colors.get(names[l], 'black'), alpha=1, histtype='step', ls=lines.get(
                                names[l], 'solid'), density=density,  linewidth=linewidth)
                        
                        elif param == 'StarFrac':
                            axs[i][j].hist(values, bins=np.logspace(np.log10(0.005),np.log10(1), 10), color=colors.get(names[l], 'black'), alpha=1, histtype='step', ls=lines.get(
                                names[l], 'solid'), density=density,  linewidth=linewidth)
                        
                        elif 'Above1' in param:
                            if names[l] == 'MBC':
                                values[values == 0] = np.nan
                            axs[i][j].hist(values, bins=binnumber, color=colors.get(names[l], 'black'), alpha=1, histtype='step', ls=lines.get(
                                names[l], 'solid'), density=density,  linewidth=linewidth)

                        else:
                            axs[i][j].hist(values[~np.isnan(values)], bins=binnumber, color=colors.get(names[l], 'black'), alpha=1, histtype='step', ls=lines.get(
                            names[l], 'solid'), density=density,  linewidth=linewidth)

                if mean:
                     print(names[l] + ': '+str(np.nanmean(valuesOriginal)))

                        
                     axs[i][j].axvline(np.nanmean(valuesOriginal),  ymax=0.15-l*0.015, color =  colors.get(names[l], 'black'), ls =  lines.get(names[l], 'solid'), linewidth = 2.3*linewidth)

                     if medianPlot:
                         xerr = MATH.boostrap_func(values, num_boots=nboots)
                         xerr = np.std(values)
                         axs[i][j].axvspan(np.nanmean(values) - xerr, np.nanmean(values) + xerr, color=colors.get(names[l], 'black'),
                                           ls=lines.get(names[l], 'solid'), linewidth=linewidth, alpha=alphaShade)


                if median or medianPlot:
                    print(names[l] + ': '+str(np.nanmedian(valuesOriginal)))

                    axs[i][j].axvline(np.nanmedian(valuesOriginal),  ymax=0.15-l*0.03, color =  colors.get(names[l], 'black'), ls =  lines.get(names[l], 'solid'), linewidth = 2.3*linewidth)
                  
                    if medianPlot:
                        xerr = MATH.boostrap_func(values, num_boots=nboots)
                        xerr = np.std(values)
                        axs[i][j].axvspan(np.nanmedian(values) - xerr, np.nanmedian(values) + xerr, color=colors.get(names[l] + 'Error', 'black'),
                                          ls=lines.get(names[l], 'solid'), linewidth=linewidth, alpha=alphaShade)

            # Plot details

            if GridMake:
                axs[i][j].grid(GridMake, color='#9e9e9e',  which="major", linewidth= 0.6,alpha= 0.3 , linestyle=':')

            axs[i][j].set_yscale(yscale)
            axs[i][j].tick_params(labelsize=0.99*fontlabel)
            axs[i][j].set_ylim(bottom=0.5, top = toplim)

            if param == 'rOverR200Min' or param == 'rOverR200_99':
                
                axs[i][j].set_xscale('log')

                axs[i][j].xaxis.set_major_formatter(
                    FuncFormatter(format_func_loglog))   

            elif xscale == 'log' or param == 'GasFrac':
                
                axs[i][j].set_xscale(xscale)

                axs[i][j].xaxis.set_major_formatter(
                    FuncFormatter(format_func_loglog))   

            if yscale == 'log':
                axs[i][j].yaxis.set_major_formatter(
                    FuncFormatter(format_func_loglog))
                
             

            if legend:
                for legpos, LegendName in enumerate(LegendNames):
                    if j == legpositions[legpos][0] and i ==legpositions[legpos][1]:
                        custom_lines, label, ncol, mult = Legend(
                            LegendName)
                        axs[i][j].legend(
                            custom_lines, label, ncol=ncol, loc=loc[legpos], fontsize=0.88*fontlabel, framealpha = framealpha, 
                            columnspacing = columnspacing, handlelength = handlelength, handletextpad = handletextpad, labelspacing = labelspacing)

            if limaixsy:
                axs[i][j].set_ylim(liminvalue[i], limax[i])
                
            if ylimmin != None and ylimmax != None:
                axs[i][j].set_ylim(ylimmin[i], ylimmax[i])
                
            if j == 0:
                if density:
                    axs[i][j].set_ylabel('Density', fontsize=fontlabel)
                else:
                    if NormCount:
                        axs[i][j].set_ylabel(
                            'Normalized Counts', fontsize=fontlabel)
                    else:
                        axs[i][j].set_ylabel('Counts', fontsize=fontlabel)
                axs[i][j].tick_params(axis='y', labelsize=0.99*fontlabel)

            if j == 0:
                if xlabelintext:

                    Afont = {'color':  'black',
                             'size': fontlabel,
                             }
                    if type(xlabelintext) is not bool:
                        anchored_text = AnchoredText(
                            titles.get(xlabelintext[i], xlabelintext[i]), loc='upper left', prop=Afont,
                                        pad = 0.3)
                        anchored_text.patch.set_facecolor('linen')
                        anchored_text.patch.set_edgecolor('black')
                        anchored_text.patch.set_alpha(0.5)
                        anchored_text.patch.set_boxstyle('round')


                    else:
                        anchored_text = AnchoredText(texts.get(param, param)
                            , loc='upper right', prop=Afont)
                    axs[i][j].add_artist(anchored_text)

            if i == 0:
                if columns == 'Snap':
                    axs[i][j].set_title(
                        r'$z = %.1f$' % dfTime.z.loc[dfTime.Snap == titlename].values[0], fontsize=1.1*fontlabel)
                if title:
                    axs[i][j].set_title(titles.get(
                        title[j], title[j]), fontsize=1.1*fontlabel)
                if 'Gyr' in labels.get(param, 'None') and not 'Gyr^' in labels.get(param, 'None') and not '_after_' in param  and not 'Delta' in labels.get(param, 'None'):
                    axs[i][j].tick_params(
                        left=True, right=False, labelsize=0.99*fontlabel)
                    axs[i][j].set_xticks([2, 4, 6, 8, 10, 11])
                    axs[i][j].set_xticklabels(
                        ['2', '4', '6', '8', '10', ''])
                    axs[i][j].tick_params(axis='x', labelsize=0.99*fontlabel)
                                        
                    #lim = axs[i][j].get_xlim()
                    ax2label = axs[i][j].twiny() #secondary_xaxis('top', which='major')
                    ax2label.grid(False)
                    ax2label.set_xlim(-0.5, 14.5)

                    if len(columns) == 3:
                        zticks = np.array([0., 0.2, 0.5, 1., 2., 5.])

                        zlabels = np.array(
                            ['0', '0.2', '0.5', '1', '2', '5'])
                        
                        if j == 2:
                            zticks = np.array([0., 0.2, 0.5, 1., 2., 5.])

                            zlabels = np.array(
                                ['0', '0.2', '0.5', '1', '2', '5'])
                    else:
                        zticks = np.array([0., 0.2, 0.5, 1., 2., 5., 20.])

                        zlabels = np.array(
                            ['0', '0.2', '0.5', '1', '2', '5', '20'])
                        
                    zticks_Age = np.array(
                        [13.803, 11.323, 8.587, 5.878, 3.285, 1.2, 0])

                    zticks = zticks.tolist()
                    zticks_Age = zticks_Age.tolist()

                    x_locator = FixedLocator(zticks_Age)
                    x_formatter = FixedFormatter(zlabels)
                    ax2label.xaxis.set_major_locator(x_locator)
                    ax2label.xaxis.set_major_formatter(x_formatter)
                    ax2label.set_xlabel(r"$z$", fontsize=fontlabel)
                    ax2label.tick_params(labelsize=0.99*fontlabel)
                    ax2label.minorticks_off()
                    

            if i == len(rows) - 1:
                if param == 'StarMass_GasLoss_Over_EntryToGas':
                    if j == 0:
                        axs[i][j].set_xticks([0, 0.5, 1, 1.5, 2])
    
                        axs[i][j].set_xticklabels(['0','0.5', '1', '1.5', '2'])
                    else:
                        axs[i][j].set_xticks([0, 1, 2, 3])
    
                        axs[i][j].set_xticklabels(['0','1', '2', '3'])
                if JustOneXlabel and i != int(len(rows)/2):
                    continue
                if xlimmin != None and xlimmax != None:
                    axs[i][j].set_xlim(xlimmin[j], xlimmax[j])
                    if xlimmin[j] == -0.05 and xlimmax[j] == 1.05:
                        axs[i][j].set_xticks([0, 0.5, 1])
                        axs[i][j].set_xticklabels(
                            ['0', '0.5', '1'])
                        
                #axs[i][j].set_xscale(xscale)
                    
                if xscale == 'log':
                    axs[i][j].xaxis.set_major_formatter(
                        FuncFormatter(format_func_loglog))   

                if JustOneXlabel and j != 1:
                    
                    continue
                if JustOneXlabel and j == 1:
                    fig.supxlabel(
                        labels.get(param), fontsize=fontlabel, y = -0.05)

                    continue
                if len(row) > 1:
                   
                    
                    if 'Gyr' in labels.get(param, 'None') and (not 'Gyr^' in labels.get(param, 'None') or not '_after_' in param) and not 'Delta' in labels.get(param, 'None'):
                        axs[i][j].set_xlabel(r'Gyr', fontsize=fontlabel)

                    elif 'Stellar' in labels.get(param, 'None'):
                        axs[i][j].set_xlabel(
                            r'$\log M [\mathrm{M_\odot}]$', fontsize=fontlabel)
                    else:
                        axs[i][j].set_xlabel(
                            labels.get(param), fontsize=fontlabel)
                    axs[i][j].tick_params(labelsize=0.99*fontlabel)

                else:

                    axs[i][j].set_xlabel(labels.get(param), fontsize=fontlabel)
                    axs[i][j].tick_params(axis='x', labelsize=0.99*fontlabel)

                if ('Gyr' in labels.get(param, 'None') and not '_after_' in param and not 'Delta' in labels.get(param, 'None')) or 'Snap' in param: #and (not 'Gyr^' in labels.get(param, 'None'): #and not '\n' in labels.get(param, 'None')):
                    if LookBackTime:
                        axs[i][j].set_xlabel('Lookback  Time \n  [Gyr]', fontsize=fontlabel)
                        axs[i][j].set_xticks([0.  ,  1.97185714,  3.94371429,  5.91557143,  7.88742857, 9.85928571, 11.83114286, 13.803  ])

                        if len(columns) == 3:
                            axs[i][j].set_xticks([ 1.97185714,    5.91557143,   9.85928571, 13.803  ])

                            axs[i][j].set_xticklabels(
                                ['12',  '8', '4', '0'])
                            
                            
                        else:
                            axs[i][j].set_xticklabels(
                                ['14', '12', '10', '8', '6', '4', '2', '0'])
                        
                            
                    else:
                        axs[i][j].set_xlim(-0.9, 14.5)
                        axs[i][j].set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
                        if len(columns) == 3:
                            axs[i][j].set_xticks([0,  4, 8, 12])

                            axs[i][j].set_xticklabels(
                            ['0', '4', '8',  '12'])
                        else:
                            
                            axs[i][j].set_xticklabels(
                            ['', '2', '4', '6', '8', '10', '12', '14'])
                        
    if Supertitle:
        plt.suptitle(SupertitleName, fontsize = 1.3*fontlabel, y=Supertitle_y)

    savefig(savepath, savefigname, TRANSPARENT)

    return

def PlotIDsAllTogether(Names, rows, IDsNotNames = False, title=False, xlabelintext=False, lineparams=False,  QuantileError=True, 
           alphaShade=0.3,  linewidth=0.5, fontlabel=24, nboots=100,  ColumnPlot=False, limaxis=False, 
           columnspacing = 0.5, xPhaseLim = 7, handlelength = 2, handletextpad = 0.4, labelspacing = 0.3,
           MedianPlot = False, LookBackTime = False, Pericenter = False, postext = ['best'],
           ylimmax = None, ylimmin = None, GridMake = False, 
           ColorMaps = [plt.get_cmap('Reds')] , PhasePlot = False,
           lNum = 6, cNum = 6, InfallTime = False, NoGas = False, SmallerScale = False,
           Type='Evolution', Xparam='Time', savepath='fig/PlotIDsAllTogether', savefigname='fig',
           dfName='Sample', SampleName='Samples', legend=False, LegendNames='None',  loc='best',
           bins=10, seed=16010504, TRANSPARENT = False, Softening = False, MaxSizeType = False):
    
    
    '''
    Plot teh evolution for random sample
    Parameters
    ----------
    columns : specific set in the sample / or different param to plot in each column. array with str
    rows : specific set in the sample / or different param to plot in each row. array with str
    IDs: IDs for selected subhalos. 
    The rest is the same as the previous functions
    Returns
    -------
    Requested Evolution or Co-Evolution plot
    -------
    Author: Abhner P. de Almeida (abhner.almeida AAT usp.br)
    '''

    np.random.seed(seed)

    dfTime = pd.read_csv(os.getenv("HOME")+"/TNG_Analyzes/SubhaloHistory/SNAPS_TIME.csv")
    
    # Verify NameParameters
    if type(Names) is not list and type(Names) is not np.ndarray:
        Names = [Names]

    if type(rows) is not list and type(rows) is not np.ndarray:
        rows = [rows]

    # Define axes
    plt.rcParams.update({'figure.figsize': (cNum*len(Names), lNum*len(rows))})
    fig = plt.figure()
    gs = fig.add_gridspec(len(rows), len(Names), hspace=0, wspace=0)
    axs = gs.subplots(sharex='col', sharey='row')
    

    # Verify axs shape
    if type(axs) is not list and type(axs) is not np.ndarray:
        axs = [axs]
    if type(axs[0]) is not np.ndarray:
        axs = np.array([axs])
        if len(Names) == 1:
            axs = axs.T

    time = dfTime.Age.values

    for i, row in enumerate(rows):
        
        df = TNG.extractDF(row)
        
        
        for j, Name in enumerate(Names):
            
            if not IDsNotNames:
                
                dfPopulation = TNG.extractPopulation(Name, dfName = dfName)

                IDs = dfPopulation['SubfindID_99'].values
            else:
                dfPopulation = TNG.extractPopulation('PaperII', dfName = 'PaperII')

                IDs = Name
            
            colorsMap = ColorMaps[j](np.linspace(0,0.9,len(IDs)))

            
            if Softening and 'SubhaloHalfmassRadType4' in row:
                rSoftening = ETNG.Softening()
                rSoftening = np.flip(rSoftening)
                axs[i][j].plot(time[(~np.isinf(rSoftening))], np.log10(rSoftening[(~np.isinf(rSoftening))]), 
                               color='black', ls='solid', lw=2*linewidth)
            FinalValues = None
    
            
            for idindex, ID in enumerate(IDs):
                try:
                    values = np.array([value for value in df[str(ID)].values])
                except:
                    continue
                xparam = time
                if PhasePlot :
                    xparam = np.arange(-1, 9)
                    xparam = np.append(xparam, xparam+0.5)
                    xparam = np.append(xparam, np.linspace(-1, 9, 1000))
                    xparam = np.unique(xparam)
                    
                    phases = TNG.PhasingData(ID, dfPopulation)
                    
                    if type(phases) != np.ndarray:
                        continue
                    phases = phases[(~np.isnan(values))]
                    values = values[(~np.isnan(values))]
                    if len(values) == 0:
                        continue
                    elif len(values[values < 0]) > 0:
                        X_Y_Spline = interp1d(phases, values,kind="linear",fill_value="extrapolate")
                        values = X_Y_Spline(xparam)

                    else:
                        X_Y_Spline = interp1d(phases, np.log10(values),kind="linear",fill_value="extrapolate")
                        values = 10**X_Y_Spline(xparam)
                    values[xparam > phases.max()] = np.nan
                    phases = xparam
               
                if row == 'r_over_R_Crit200_FirstGroup':
                    values[values == 0] = np.nan
                    arg = np.argwhere(np.isnan(values)).T[0]
                    values[arg[0]:] = np.nan
                if not MedianPlot:
                    axs[i][j].plot(xparam[~np.isnan(values)], values[~np.isnan(values)], color = colorsMap[idindex], ls =  'solid', lw=0.25*linewidth)

                try:
                    if idindex == 0:
                        FinalValues = values
                    else:
                        FinalValues = np.vstack((FinalValues, values))
                except:
                     continue
            
            try:
                if not MedianPlot:
                    FinalValues = FinalValues.T
            except:
                continue
            
            if not MedianPlot:
                y = np.array([])
                if len(FinalValues) > 0:
                    if len(FinalValues.shape) > 1:
                        for k, value in enumerate(FinalValues):
                            valueNotNan = value[~np.isnan(value)]
                            valueNotInf = valueNotNan[~np.isinf(valueNotNan)]

                            if len(valueNotInf) > 5:
                                y = np.append(y, np.nanmedian(value))
                            else:
                                y = np.append(y, np.nan)
                    else:
                        y = FinalValues
            
                else:
                    y = np.nan
                    
                colorsMap = ColorMaps[j]([0.1, 0.999999999])
                try:
                   
                    
                    axs[i][j].plot(xparam[~np.isnan(y)], y[~np.isnan(y)], color = colorsMap[1], ls =  'solid', lw=1.5*linewidth)
            
                except:
                    None
            elif MedianPlot:
                if PhasePlot:
                    Y, Yerr, xPhase, xTime = TNG.makedataevolution([Name], [''], [row],  SampleName=SampleName, PhasingPlot=PhasePlot, dfName = dfName, nboots=nboots)
                    Yerr = np.array([value for value in Yerr[0][0][0]])
                    Y = np.array([value for value in Y[0][0][0]])
                    xparamMedian = np.array([value for value in xPhase[0][0][0]])
                else:
                    Y, Yerr = TNG.makedataevolution([Name], [''], [row],  SampleName=SampleName, PhasingPlot=PhasePlot, dfName = dfName, nboots=nboots)
                    Yerr = np.array([value for value in Yerr[0][0][0]])
                    Y = np.array([value for value in Y[0][0][0]])
                    xparamMedian = xparam
                #colorsMap = ColorMaps[j]([0.1, 0.999999999])
                
                
                
                # Compute absolute deviation from median
                if PhasePlot:
                    #xparam = np.flip(xparam)

                    for arrayValues in FinalValues:
                        #if 'sSFR' in row:
                        #    arrayValues[arrayValues < -13.5] = np.nan
                        if 'SBC' in Name:
                            print(xparam[~np.isnan(arrayValues)], arrayValues[~np.isnan(arrayValues)])
                        try:
                            COND = (~np.isnan(arrayValues)) & (~np.isinf(arrayValues))
                            axs[i][j].plot(xparam[COND], np.flip(arrayValues[COND]), color = 'gray', ls =  'solid', lw=0.25*linewidth)
                        except:
                            None

                    #FinalValuesNew = np.array([])
                    #for arrayValues in FinalValues:
                    #    if len(FinalValuesNew) == 0:
                    #        indices = np.isin(xparam, xparamMedian)
                    #        FinalValuesNew = arrayValues[indices]
                    #    else:
                    #        indices = np.isin(xparam, xparamMedian)

                    #        FinalValuesNew = np.vstack((FinalValuesNew, arrayValues[indices] ))
                    #FinalValues = FinalValuesNew
                    #xparam = xparamMedian
                
                else:
                    deviation = np.abs(FinalValues - Y)
    
                    # Normalize deviation (used for transparency)
                    #Cond = (~np.isnan(deviation)) & (~np.isinf(deviation))
                    max_dev = np.nanpercentile(deviation, 90)  # Use the 90th percentile to avoid extreme effects
                    normalized_dev = np.clip(deviation / max_dev, 0, 1)  # Scale between 0 and 1
                    
                    # Compute alpha (inverted deviation)
                    alpha_values = 1 - normalized_dev  # Closer to median -> higher opacity
                    alpha_values[np.isnan(alpha_values)] = 0
    
                    if 'Normal' in Name:
                        alpha = 0.05
                    else:
                        alpha = 0.3
                    for idindex, ID in enumerate(IDs):
                        try:
                            values = FinalValues[idindex, :]
                        except:
                            continue
                        xTime = xparamMedian[~np.isnan(values)]
                        values = values[~np.isnan(values)]
                        if 'SBC' in Name:
    
                            print(xTime, values)
                        if 'sSFR' in row:
                            values[values < -13.5] = np.nan
                        for idtime in range(len(xTime) -1):
                            axs[i][j].plot(xTime[idtime:idtime+2], values[idtime:idtime+2], color = 'gray', alpha=alpha_values[idindex, idtime]*alpha,  ls =  'solid', lw=0.35*linewidth)

                #try:
                if 'sSFR' in row:
                    Yerr[Y < -3.5] = np.nan

                    Y[Y < -13.5] = np.nan
                axs[i][j].plot(xparamMedian[~np.isnan(Y)], Y[~np.isnan(Y)], color = colors.get(Name, 'black'), ls =  'solid', lw=1.5*linewidth)
                
                if 'Normal' in Name:
                    Yerr = Yerr * 2
                    alpha = 1.3
                else:
                    alpha = 1
                axs[i][j].fill_between(
                    xparamMedian[(~np.isnan(Y)) & (~np.isnan(Yerr))], Y[(~np.isnan(Y)) & (~np.isnan(Yerr))] - Yerr[(~np.isnan(Y)) & (~np.isnan(Yerr))], Y[(~np.isnan(Y)) & (~np.isnan(Yerr))] + Yerr[(~np.isnan(Y)) & (~np.isnan(Yerr))], 
                    color=colors.get(Name, 'black'), ls= 'solid', alpha=0.7*alpha)
                axs[i][j].fill_between(
                    xparamMedian[(~np.isnan(Y)) & (~np.isnan(Yerr))], Y[(~np.isnan(Y)) & (~np.isnan(Yerr))] - 3*Yerr[(~np.isnan(Y)) & (~np.isnan(Yerr))], Y[(~np.isnan(Y)) & (~np.isnan(Yerr))] + 3*Yerr[(~np.isnan(Y)) & (~np.isnan(Yerr))], 
                    color=colors.get(Name, 'black'), ls= 'solid', alpha=0.4*alpha)
                
                #except:
                #    None
            # Plot details

            if GridMake:
                axs[i][j].grid(GridMake, color='#9e9e9e',  which="major", linewidth= 0.6,alpha= 0.3 , linestyle=':')
               
            axs[i][j].tick_params(axis='y', labelsize=0.99*fontlabel)
            axs[i][j].tick_params(axis='x', labelsize=0.99*fontlabel)
	    
            
            if ylimmin != None and ylimmax != None:
                axs[i][j].set_ylim(ylimmin[i], ylimmax[i])
            if scales.get(row) != None :
                axs[i][j].set_yscale(scales.get(row))
            if scales.get(row) == 'log' :
                axs[i][j].yaxis.set_major_formatter(
                    FuncFormatter(format_func_loglog))

            
            if j == 0:

                if len(row) > 1:
                    axs[i][j].set_ylabel(
                        labelsequal.get(row), fontsize=fontlabel)

                else:
                    axs[i][j].set_ylabel(labels.get(row), fontsize=fontlabel)

            
            if i == 0:

                if title and not ColumnPlot:
                    axs[i][j].set_title(titles.get(
                        title[j], title[j]), fontsize=1*fontlabel)
                

                if not PhasePlot:

                    axs[i][j].tick_params(bottom=True, top=False)
                    lim = axs[i][j].get_xlim()
                    ax2label = axs[i][j].twiny() #secondary_xaxis('top', which='major')
                    ax2label.grid(False)
                    ax2label.set_xlim(lim)

                    if row == 'rToRNearYoung' or savefigname == 'Young':
                        zticks = np.array([0., 0.2])
                        zlabels = np.array(
                            ['0', '0.2'])
                        zticks_Age = np.array(
                            [13.803, 11.323])
                    else:
                        zticks = np.array([0., 0.2, 0.5, 1., 2., 5., 20.])
                        if SmallerScale:

                            if j == 0:
                                zlabels = np.array(
                                    ['0', '0.2', '0.5', '1', '2', '5', '20'])
                            if j != 0:
                                zlabels = np.array(
                                    ['0', '0.2', '0.5', '1', '2', '5', ''])
                        else:
                            zlabels = np.array(
                                ['0', '0.2', '0.5', '1', '2', '5', '20'])
                        zticks_Age = np.array(
                            [13.803, 11.323, 8.587, 5.878, 3.285, 1.2, 0])


                    zticks = zticks.tolist()
                    zticks_Age = zticks_Age.tolist()

                    x_locator = FixedLocator(zticks_Age)
                    x_formatter = FixedFormatter(zlabels)
                    ax2label.xaxis.set_major_locator(x_locator)
                    ax2label.xaxis.set_major_formatter(x_formatter)
                    ax2label.set_xlabel(r"$z$", fontsize=fontlabel)
                    ax2label.tick_params(labelsize=0.85*fontlabel)

            if i == len(rows) - 1:
                
                if Type == 'Evolution':
                    
                    
                    if row == 'rToRNearYoung' or savefigname == 'Young':
                        axs[i][j].set_xlabel(r'$t \, \,  [\mathrm{Gyr}]$', fontsize=fontlabel)
                        axs[i][j].set_xticks([10, 12, 14])
                        axs[i][j].set_xticklabels(
                            ['10', '12', '14'])
                    else:
                        if LookBackTime and not PhasePlot:
                            axs[i][j].set_xticks([0.  ,  1.97185714,  3.94371429,  5.91557143,  7.88742857, 9.85928571, 11.83114286, 13.803  ])
                            if SmallerScale:
                                if j == 1:
                                    axs[i][j].set_xlabel(r'$\mathrm{Lookback \; Time} \, \, [\mathrm{Gyr}]$', fontsize=fontlabel)

                                if j == 0:
                                    axs[i][j].set_xticklabels(
                                    ['14', '12', '10', '8', '6', '4', '2', '0'])
                                if j != 0:
                                    axs[i][j].set_xticklabels(
                                    ['', '12', '10', '8', '6', '4', '2', '0'])
                            else:
                                axs[i][j].set_xlabel(r'$\mathrm{Lookback \; Time} \, \, [\mathrm{Gyr}]$', fontsize=fontlabel)

                                if j == 0:
                                    axs[i][j].set_xticklabels(
                                    ['14', '12', '10', '8', '6', '4', '2', '0'])
                                if j != 0:
                                    axs[i][j].set_xticklabels(
                                    ['', '12', '10', '8', '6', '4', '2', '0'])
                                
                        elif PhasePlot:
                            limXparam = int(xPhaseLim + 1)
                            postiveXticks = np.arange(limXparam)
                            postiveXLabels = np.array([str(int(i)) for i in postiveXticks])

                            postiveXticks = np.append([-1, -0.5], postiveXticks)
                            postiveXLabels = np.append(['', 'E'], postiveXLabels)
                            axs[i][j].set_xlabel(r'$\phi_\mathrm{Orbital}$', fontsize=fontlabel)
                            axs[i][j].set_xticks(postiveXticks)
                            axs[i][j].set_xticklabels(postiveXLabels)
                            axs[i][j].set_xlim(-1, xPhaseLim+0.5)
                        else:
                            axs[i][j].set_xticks([0, 2, 4, 6, 8, 10, 12, 14])

                            axs[i][j].set_xlabel(r'$t \, \, [\mathrm{Gyr}]$', fontsize=fontlabel)
        
                            axs[i][j].set_xticklabels(
                                ['0', '2', '4', '6', '8', '10', '12', '14'])

               
    savefig(savepath, savefigname, TRANSPARENT)

    return


def PlotProfile(IDs, names, columns, rows, PartTypes,  ParamX='rad', Condition='All', cumulative=False, title=False, xlabelintext=False, 
                framealpha = 0.95, linewidth=1.2, fontlabel=24, nboots=100, Nlim=100,    Entry = False,
                quantile=0.95, rmaxlim=50, norm=False, legend=False, LegendNames=None, line=False, 
                columnspacing = 0.7, handletextpad = 0.4, labelspacing = 0.3, handlelength = 2.0, Supertitle = False,
                ylimmin = None, ylimmax = None, xlimmin = None, xlimmax = None, legpositions = None,
                lNum = 6, cNum = 6, GridMake = False, dfSample = None,
                savepath='fig/PlotProfile', savefigname='fig', dfName='Sample', SampleName='Samples',  loc='best',
                PATH = os.getenv("HOME")+'/TNG_Analyzes/SubhaloHistory', SIMTNG = 'TNG50',
                nbins=25, seed=16010504, TRANSPARENT = False, Softening = False ):
    '''
    Plot radial profiles
    Parameters
    ----------
    IDs: IDs for selected subhalos. 
    names : sample names. array with str
    rows : specific set in the sample / or different param to plot in each row. array with str
    PartTpes : particle types for each param. array with str
    ParamX : specific the x param. Default: 'rad'. str
    Condition : specific the particle condition. Default: 'Original'. str
    cumulative : make cumulative plot. Default: False. bool
    rmaxlim : max rad. Default: 120. float
    norm : to normalized with the max value. Default: False. bool
    line: line for half rads. Default: False. bool
    The rest is the same as the previous functions
    Returns
    -------
    Requested Evolution or Co-Evolution plot''
    -------
    Author: Abhner P. de Almeida (abhner.almeida AAT usp.br)
    '''

    np.random.seed(seed)

    # Verify NameParameters
    if type(columns) is not list and type(columns) is not np.ndarray:
        columns = [columns]

    if type(rows) is not list and type(rows) is not np.ndarray:
        rows = [rows]

    dfTime = TNG.extractDF('SNAPS_TIME')

    # Define axes
    plt.rcParams.update({'figure.figsize': (cNum*len(columns), lNum*len(rows))})
    fig = plt.figure()
    gs = fig.add_gridspec(len(rows), len(columns), hspace=0, wspace=0)
    axs = gs.subplots(sharex='col', sharey='row')

    # Verify axs shape
    if type(axs) is not list and type(axs) is not np.ndarray:
        axs = [axs]
    if type(axs[0]) is not np.ndarray:
        axs = np.array([axs])
        if len(columns) == 1:
            axs = axs.T

    dFHalfStar = TNG.extractDF(
        'SubhaloHalfmassRadType4', PATH=PATH)
    dFHalfGasRad = TNG.extractDF(
        'SubhaloHalfmassRadType0', PATH=PATH)
    dfGasMass = TNG.extractDF(
        'SubhaloMassType0', PATH=PATH)

    for i, row in enumerate(rows):
        if PartTypes[i] == 'PartType4':
            dFHalfRad = TNG.extractDF(
                'SubhaloHalfmassRadType4', PATH=PATH)
        if PartTypes[i] == 'PartType0' or PartTypes[i] == 'gas':
            dFHalfRad = TNG.extractDF(
                'SubhaloHalfmassRadType0', PATH=PATH)
        if PartTypes[i] == 'PartType1' or PartTypes[i] == 'DM':
            dFHalfRad = TNG.extractDF(
                'SubhaloHalfmassRadType1', PATH=PATH)
            
        if 'Galactic' in Condition:
            dFHalfRadGas = TNG.extractDF(
                'SubhaloHalfmassRadType0', PATH=PATH)

        for l, ID in enumerate(IDs):
            Rads = np.array([])
           

            for j, snap in enumerate(columns):
                if 'Entry' in names[l]:
                    if j == 0:
                        axs[i][j].set_title(
                            r'$z_\mathrm{entry}$', fontsize=1.1*fontlabel)
                    elif j == 1:
                        snap = 99
                        axs[i][j].set_title(
                            r'$z = %.1f$' % dfTime.z.loc[dfTime.Snap == snap].values[0], fontsize=1.1*fontlabel)

                else:
                    if i == 0:
                        axs[i][j].set_title(
                            r'$z = %.1f$' % dfTime.z.loc[dfTime.Snap == snap].values[0], fontsize=1.1*fontlabel)


                RadStars = np.array([])
                RadGas = np.array([])
                GasMass = np.array([])

                for idValue in ID:
                    if 'Entry' in names[l] and j == 0:
                        snap = int(dfSample.Snap_At_FirstEntry.loc[dfSample.SubfindID_99 == idValue].values[0])
                    try:
                        HalfRad = dFHalfStar[str(
                            idValue)].loc[dFHalfStar.Snap == snap].values[0]
                        HalfGasRad = dFHalfGasRad[str(
                            idValue)].loc[dFHalfGasRad.Snap == snap].values[0]
                        GasMassType = dfGasMass[str(
                            idValue)].loc[dfGasMass.Snap == snap].values[0]
                    except:
                        RadStars = np.append(RadStars, np.nan)
                        RadGas = np.append(RadGas, np.nan)
                        GasMass = np.append(GasMass, np.nan)
                        continue
                    RadStars = np.append(RadStars, HalfRad)
                    RadGas = np.append(RadGas, HalfGasRad)
                    GasMass = np.append(GasMass, GasMassType)

                Rads = np.array([])

                for idValue in ID:
                    if 'Entry' in names[l] and j == 0:
                        snap = int(dfSample.Snap_At_FirstEntry.loc[dfSample.SubfindID_99 == idValue].values[0])

                    if 'Galactic' in Condition :
                        HalfRadGas = dFHalfRadGas[str(idValue)].loc[dFHalfRadGas.Snap == snap].values[0]
                        Rads = np.append(Rads, 10**HalfRadGas)
                        
                    else:
                        try:
                            HalfRad = dFHalfRad[str(
                            idValue)].loc[dFHalfRad.Snap == snap].values[0]
                        except:
                            HalfRad = 1.2
                        Rads = np.append(Rads, 10**HalfRad)

                if PartTypes[i] == 'PartType4':
                    rmin = np.nanmedian(Rads)/5
                    rmax = np.nanmedian(Rads)*150
                    
                elif (PartTypes[i] == 'PartType0' or PartTypes[i] == 'gas') :
                    rmin = np.nanmedian(Rads)/300
                    rmax = np.nanmedian(Rads)*7

                elif (PartTypes[i] == 'PartType1' or PartTypes[i] == 'DM') :
                    rmin = np.nanmedian(Rads)/300
                    rmax = np.nanmedian(Rads)*7
                    
                if rmax > rmaxlim or rmax == 0.:
                    rmax = rmaxlim

                elif PartTypes[i] == 'PartType0' and rmin < 0.07:
                    rmin = 0.07

                elif PartTypes[i] == 'PartType1' and rmin < 0.3:
                    rmin = 0.3

                elif PartTypes[i] == 'PartType4' and rmin < 0.1:
                    rmin = 0.1
            

                if np.median(Rads) == 0:
                    x = [np.nan]
                    axs[i][j].plot([np.nan], [np.nan], color=colors.get(names[l] ), ls=lines.get(
                        names[l] ), lw= 2.5*linesthicker.get(names[l], linewidth), dash_capstyle = capstyles.get(names[l], 'projecting'))
                   
                    continue
                
                
                if row != 'sSFR' and row != 'joverR' and row != 'DensityGasOverR2' and row != 'DensityStarOverR2' and row != 'GFM_Metallicity_Zodot' and row != 'GradsSFR':
                    
                    try:
                        df = pd.read_csv(PATH + '/'+SIMTNG+ '/Profiles/'+ Condition + '/'+row+'/'+PartTypes[i]+'/'+str(columns[j])+'/'+names[l] + Condition  +'.csv')
                        rad = df.Rads.values
                        ymedian = df.ymedians.values
                        yerr = df.yerrs.values
                    except:
                        rad, ymedian, yerr = TNG.make_profile(
                            ID, snap, row, PartTypes[i], rmin=rmin, rmax=rmax, nbins=nbins, 
                            nboot=nboots, Condition=Condition, dfSample = dfSample, Entry = Entry)
                        
                        if type(rad) == float:
                            continue
                        dic = {'Rads': rad, 'ymedians': ymedian, 'yerrs': yerr}
                        df = pd.DataFrame(data=dic)
                        
                        try:
                            df.to_csv(PATH + '/'+SIMTNG+'/Profiles/'+ Condition + '/'+row+'/'+PartTypes[i]+'/'+str(columns[j])+'/'+names[l]+ Condition  +'.csv')
                            
                        except:
                            path = PATH + '/'+SIMTNG+'/'
                                                        
                            for name in ['Profiles', Condition, row, PartTypes[i], str(columns[j])]:
                                path = os.path.join(path, name)
                                if not os.path.isdir(path):
                                    os.mkdir(path)
                            df.to_csv(PATH + '/'+SIMTNG+'/Profiles/'+ Condition + '/'+row +
                                         '/'+PartTypes[i]+'/'+str(columns[j])+'/'+names[l]+ Condition + '.csv')

                        rad = rad
                        ymedian = ymedian
                        yerr = yerr
                        
                elif row == 'sSFR':
                                           
                    try:
                        df = pd.read_csv(PATH + '/'+SIMTNG+ '/Profiles/'+ Condition + '/'+'SFR'+'/'+PartTypes[i]+'/'+str(columns[j])+'/'+names[l]+ Condition + '.csv')
                    except:
                        continue
                    radSFR = df.Rads.values
                    ymedianSFR = df.ymedians.values
                    yerrSFR = df.yerrs.values

                    df = pd.read_csv(PATH + '/'+SIMTNG+ '/Profiles/'+ Condition + '/'+'Mstellar'+'/PartType4/'
                                        +str(columns[j])+'/'+names[l]+ Condition + '.csv')
                    radMstellar = df.Rads.values
                    ymedianMstellar = df.ymedians.values

                    new_y = interp1d(
                        radMstellar, ymedianMstellar,  kind='cubic',  fill_value='extrapolate')(radSFR)
                    rad = radSFR
                    ymedian = ymedianSFR / new_y
                    yerr = yerrSFR / new_y
                    
                elif row == 'GFM_Metallicity_Zodot':
                                            
                    df = pd.read_csv(PATH + '/'+SIMTNG+ '/Profiles/'+ Condition + '/'+'GFM_Metallicity'+'/'+PartTypes[i]+'/'+str(columns[j])+'/'+names[l]+ Condition + '.csv')
                    rad = df.Rads.values
                    ymedian = df.ymedians.values / 0.0127
                    yerr = df.yerrs.values  / 0.0127

                elif row == 'GradsSFR':
                    df = pd.read_csv(PATH + '/'+SIMTNG+ '/Profiles/'+ Condition + '/'+'SFR'+'/'+PartTypes[i]+'/'+str(columns[j])+'/'+names[l]+ Condition + '.csv')
                    radSFR = df.Rads.values
                    ymedianSFR = df.ymedians.values
                    yerrSFR = df.yerrs.values

                    df = pd.read_csv(PATH + '/'+SIMTNG+ '/Profiles/'+ Condition + '/'+'Mstellar'+'/PartType4/'
                                        +str(columns[j])+'/'+names[l]+ Condition + '.csv')
                    radMstellar = df.Rads.values
                    ymedianMstellar = df.ymedians.values

                    new_y = interp1d(
                        radMstellar, ymedianMstellar, kind='cubic', fill_value='extrapolate')(radSFR)
                    rad = radSFR
                    ymedian = ymedianSFR / new_y
                    yerr = yerrSFR / new_y
                    ymedian = np.gradient(ymedian, rad)

                elif row == 'joverR':
                    
                   
                    try:
                        df = pd.read_csv(PATH + '/'+SIMTNG+ '/Profiles/'+ Condition + '/'+'j'+
                                            '/'+PartTypes[i]+'/'+str(columns[j])+'/'+names[l]+ Condition + '.csv')
                        
                    except:
                        continue
                    rad = df.Rads.values
                    ymedian = df.ymedians.values
                    yerr = df.yerrs.values

                    rad = rad
                    ymedian = ymedian / rad
                    yerr = yerr / rad

                elif row == 'DensityGasOverR2':
                    
                    
                    try: 
                        df = pd.read_csv(PATH + '/'+SIMTNG+ '/Profiles/'+ Condition + '/'+'DensityGas'+
                                       '/'+PartTypes[i]+'/'+str(columns[j])+'/'+names[l]+ Condition + '.csv')
                    except:
                        continue
                    rad = df.Rads.values
                    ymedian = df.ymedians.values
                    yerr = df.yerrs.values

                    rad = rad
                    ymedian = ymedian * rad**2
                    yerr = yerr * rad**2
                    
                elif row == 'DensityStarOverR2':
                    
                    try:
                        df = pd.read_csv(PATH + '/'+SIMTNG+ '/Profiles/'+ Condition + '/'+'DensityStar'+
                                        '/'+PartTypes[i]+'/'+str(columns[j])+'/'+names[l]+ Condition + '.csv')
                    except:
                        continue
                    rad = df.Rads.values
                    ymedian = df.ymedians.values
                    yerr = df.yerrs.values

                    rad = rad
                    ymedian = ymedian * rad**2
                    yerr = yerr * rad**2
                                        
                argnan = np.argwhere(~np.isnan(rad)).T[0]
                rad = rad[argnan]
                ymedian = ymedian[argnan]
                yerr = yerr[argnan]

                argnan = np.argwhere(~np.isnan(ymedian)).T[0]
                rad = rad[argnan]
                ymedian = ymedian[argnan]
                yerr = yerr[argnan]


                
                if cumulative and row in ['Mstellar', 'Mgas']:
                    ymedian = np.cumsum(ymedian)
                    if len(ymedian) == 0:
                        yerr = np.ones(len(rad)) * np.nan
                        ymedian = np.ones(len(rad)) * np.nan
                    else:
                        if norm:

                            yerr = yerr / np.max(ymedian)
                            ymedian = ymedian / np.max(ymedian)
                            
                if PartTypes[i] == 'PartType0':
                    dfSample = TNG.extractDF(dfName)
                    try:
                        SnapCheck = dfSample.loc[dfSample.SubfindID_99.isin(ID) , 'SnapLostGas'].values
                        SnapCheck[SnapCheck < 0] = 99
                        
                        if not len(SnapCheck[SnapCheck >= snap]) > int(len(SnapCheck) / 2):   
                            continue
                    except:
                        continue
                    
                if len(rad) <= 2:
                    axs[i][j].plot(rad, ymedian, color=colors.get(names[l] ), ls=lines.get(
                        names[l] ), lw= 3.5*linesthicker.get(names[l], linewidth), dash_capstyle = capstyles.get(names[l], 'projecting'))
                   
                else:
                    
                    Func = interp1d(rad, ymedian, kind='cubic', fill_value = 'extrapolate')
                 
                    x = np.geomspace(min(rad), max(rad), 25)
                    y = Func(x)
                    if not 'RadVelocity' in row:
                        axs[i][j].plot(x[y > 0], y[y > 0], color=colors.get(names[l] ), ls=lines.get(
                        names[l] ), lw= 3.5*linesthicker.get(names[l], linewidth), dash_capstyle = capstyles.get(names[l], 'projecting'))
                    else:
                       axs[i][j].plot(x, y, color=colors.get(names[l] ), ls=lines.get(
                       names[l] ), lw= 3.5*linesthicker.get(names[l], linewidth), dash_capstyle = capstyles.get(names[l], 'projecting'))
                  
         
                    if line and i == len(rows) - 1:
                        #axs[i][j].arrow(2*10**np.nanmedian(RadStars), 1e2, 0, -1e2, color=colors.get(
                            #names[l] , 'black'), ls=lines.get(names[l] + 'Profile', 'solid'), linewidth= 3 * linewidth, head_width=0.)
                        axs[i][j].axvline(10**np.nanmedian(RadStars),
                                          ls='--', color=colors.get(names[l], 'black'), lw=1.1)
                        
                        #axs[i][j].axvline(10**np.nanmedian(RadGas),
                                          #ls='--', color=colors.get(names[l]))
                                          
                if Softening and 'DensityStar' in row:
                    rSoftening = ETNG.Softening()
                    axs[i][j].axvspan(0, rSoftening[snap],  facecolor='tab:red', alpha=.1)

                # Plot details
                if GridMake:
                    axs[i][j].grid(GridMake, color='#9e9e9e',  which="major", linewidth= 0.6,alpha= 0.3 , linestyle=':')
                             
                if ylimmin != None and ylimmax != None:
                    axs[i][j].set_ylim( ylimmin[i], ylimmax[i])
                if xlimmin != None and xlimmax != None:
                    if len(xlimmin) > 1:
                        axs[i][j].set_xlim(xlimmin[j], xlimmax[j])
                    else:
                        axs[i][j].set_xlim(xlimmin[0], xlimmax[0])

                
                if legend:
                    for legpos, LegendName in enumerate(LegendNames):
                        if j == legpositions[legpos][0] and i ==legpositions[legpos][1]:
                            custom_lines, label, ncol, mult = Legend(
                                LegendName, mult = 5)
                            axs[i][j].legend(
                                custom_lines, label, ncol=ncol, loc=loc[legpos], fontsize=0.88*fontlabel, framealpha = framealpha, 
                                columnspacing = columnspacing, handlelength = handlelength, handletextpad = handletextpad, labelspacing = labelspacing)

                if j == 0:
                    if cumulative and row in ['Mstellar', 'Mgas']:
                        if norm:
                            axs[i][j].set_yscale(
                                scales.get(row + 'Norm', 'linear'))

                            if scales.get(row + 'Norm') == 'log':
                                axs[i][j].yaxis.set_major_formatter(
                                    FuncFormatter(format_func_loglog))
                            axs[i][j].set_ylabel(labels.get(
                                row + 'Norm'), fontsize=fontlabel)
                        else:

                            axs[i][j].set_yscale(
                                scales.get(row + 'Cum', 'linear'))
                            if scales.get(row + 'Cum') == 'log':
                                axs[i][j].yaxis.set_major_formatter(
                                    FuncFormatter(format_func_loglog))
                            axs[i][j].set_ylabel(labels.get(
                                row + 'Cum'), fontsize=fontlabel)

                    else:
                        axs[i][j].set_yscale(scales.get(row, 'linear'))
                        if scales.get(row) == 'log':
                            axs[i][j].yaxis.set_major_formatter(
                                FuncFormatter(format_func_loglog))
                        if row in ['j', 'RadVelocity', 'joverR']:
                            axs[i][j].set_ylabel(
                                labels.get(row+PartTypes[i]), fontsize=fontlabel)
                        else:
                            axs[i][j].set_ylabel(
                                labels.get(row), fontsize=fontlabel)

                if j == len(columns) - 1:
                    if xlabelintext:
                        Afont = {'color':  'black',
                                 'size': fontlabel,
                                 }
                        anchored_text = AnchoredText(
                            texts.get(row), loc='upper right', prop=Afont)
                        axs[i][j].add_artist(anchored_text)

                
                if i == len(rows) - 1:
                    axs[i][j].set_xscale(scales.get(ParamX, 'linear'))
                    if scales.get(ParamX) == 'log':
                        axs[i][j].xaxis.set_major_formatter(
                            FuncFormatter(format_func_loglog))
                    axs[i][j].set_xlabel(labels.get(
                        ParamX), fontsize=fontlabel)

                axs[i][j].tick_params(labelsize=0.99*fontlabel)

    if Supertitle:
        plt.suptitle('Satellites', fontsize = 1.3*fontlabel, y=1.1)
    savefig(savepath, savefigname, TRANSPARENT)

    return


def Ternary(ftop,fleft,fright, SAMPLES, labeltop=None,labelleft=None,labelright=None,
            points='scatter',gridsize=10,func=len,norm=None, alpha = 0.9, TRANSPARENT = True,
            ms=5,ls='-',lw=3,color='r',marker='o', scale = 'linear', columnspacing = 0.5,  handletextpad = 0.4, 
            labelspacing = 0.3, handlelength = 2.0, Names = ['NormalScatter',  'MBCScatter', 'SBCScatter', 'DiffuseScatter', 'SatelliteScatter', 'CentralScatter', 'SBCBornYoungScatter',  'BadFlagScatter'],
            edgecolor='face',fontsize=10,title=None,saveprefix=None):
    """ternary (triangle) plots (for 3 variables in [0,1] that sum to unity)
    
    arguments:
        ftop,fleft,fright: 3 variables in np.arrays in [0,1] that sum to unity
        labeltop,labelleft,labelright: labels [without 'fraction', which is added automatically]
        points: one of 'scatter', 'plot', 'tribin', 'hexbin'
                    (the latter two for density plots)
        func: function for density plots (default len for counts)
        norm: None or 'log' for density plots in log counts
        other arguments are graphics related
        
    author: Gary Mamon (gam AAT iap.fr)
    """
    plt.rcParams.update({'figure.figsize' : (4, 4)})
    ax = plt.subplot(projection="ternary")
    if norm == 'log':
        norm = mplcol.LogNorm()
    for l, sample in enumerate(SAMPLES):
        if points == 'scatter':
            if 'Central' in sample:
                ax.scatter(ftop[l],fleft[l],fright[l],s=2*msize.get(sample.replace('Central', ''), ms),color=colors.get(sample.replace('Central', '') , 'black'),marker=markers.get(sample.replace('Central', ''), 'o'),   edgecolor = 'none', alpha = alpha)
            else:
                if sample == 'BadFlag':
                    ecolor = 'red'
                else:
                    ecolor = 'black'
                ax.scatter(ftop[l],fleft[l],fright[l],s=2.2*msize.get(sample, ms),color=colors.get(sample, 'black'),marker=markers.get(sample, 'o'), lw = 0.9*linesthicker.get(sample, 1.1),  edgecolor = ecolor, alpha = alpha)
        elif points == 'plot':
            ax.plot(ftop[l],fleft[l],fright[l],color=colors.get(sample, 'black'),ls=lines.get(sample, '-'),lw=lw)
        elif points == 'tribin':
            ax.tribin(ftop[l],fleft[l],fright[l],gridsize=gridsize,
                      color=colors.get(sample, 'black'), cmap='Greys',reduce_C_function=func,
                      edgecolors=edgecolor)
        elif points == 'hexbin':
            ax.hexbin(ftop,fleft,fright,gridsize=gridsize,norm=norm,
                      color=color,cmap='Greys',reduce_C_function=func,
                      edgecolors=edgecolor)
    if labeltop is not None:
        ax.set_tlabel('$\leftarrow$ ' + labeltop + ' fraction' ,fontsize=1.*fontsize)
    if labelleft is not None:
        ax.set_llabel('$\leftarrow$ ' + labelleft + ' fraction',fontsize=1.*fontsize)
    if labelright is not None:
        ax.set_rlabel(labelright + ' fraction $\\rightarrow$',fontsize=1.*fontsize)
    position = 'tick1'
    ax.taxis.set_label_position(position)
    ax.laxis.set_label_position(position)
    ax.raxis.set_label_position(position)
    
    custom_lines, label, ncol, mult = Legend(Names)
    plt.legend(
        custom_lines, label, ncol=3, fontsize=0.8*fontsize, framealpha=0.9,  loc = 'upper right',
        columnspacing = columnspacing, handlelength = handlelength, handletextpad = handletextpad, labelspacing = labelspacing,
        bbox_to_anchor=(1.1, 1.48))

    if title is not None:
        ax.set_title(title, fontsize = 1.1*fontsize, x=0.05, y=0.9)
    if saveprefix is not None:
        plt.savefig(saveprefix + '.pdf',  bbox_inches='tight')

        if TRANSPARENT:
            plt.savefig(saveprefix + '.png',  bbox_inches='tight',  transparent=True, dpi=700)



def PlotMap(IDs, PartType, snaps, Names = [], Type = 'Mosaic', 
            width = 25, cNum = 1, lNum = 1, xtext = 0, ytext = 15,
            xtextZ = 10, ytextZ = -15,
            r='infinity',  axis = 'x', cmap_dm = 'bone', dfName = 'Sample',
            SIMS = 'TNG50-1', SIMTNG = 'TNG50', vmin = None, vmax = None, VelPlot = False,
            Param = None, savepath='fig/PlotMap', savefigname='fig', OrbitalEvolution = False,
            fontlabel = 20):
    
    if Type == 'Evolution':
        cNum = len(snaps)
        if OrbitalEvolution:
            cNum = 5
        lNum = len(IDs)
    # Define axes
    plt.rcParams.update({'figure.figsize': (cNum*4, lNum*4)})
    fig = plt.figure()
    #gs = fig.add_gridspec(lNum, cNum, hspace=0, wspace=0)
    #axs = gs.subplots(sharex='col', sharey='row')
    grid = AxesGrid( fig,
                (0.075, 0.075, 0.85, 0.85),
                nrows_ncols = (lNum, cNum),
                axes_pad=0.,
                label_mode="L",
                share_all=True,
                cbar_location="right",
                cbar_mode="single",
                cbar_size="5%",
                cbar_pad=0.15) 

    
    
    if Type == 'Mosaic':
        snap = snaps[0]
        
        while len(IDs) != cNum*lNum:
            IDs = np.append(IDs, np.nan)
            
        while len(Names) != cNum*lNum:
            Names = np.append(Names, Names[0])
            
        if cNum == lNum == 1:
            IDsReshape = np.array([IDs])
        else:
            IDsReshape = IDs.reshape((lNum, cNum))
        
     
        
    for i in range(lNum):
        for j in range(cNum):
            if Type == 'Mosaic':
                ID = IDsReshape[i, j]
                if len(Names) > 0:
                    Name = Names[i*cNum+ j]
                    
            elif Type == 'Evolution':
                
                ID = IDs[i]
                Name = Names[i]
                
                if OrbitalEvolution:
                    r_over_R_Crit200 = TNG.extractDF('r_over_R_Crit200_FirstGroup')

                   
                    
                    dfSample = TNG.extractDF(dfName)
                    SnapAtEntry = dfSample.loc[dfSample.SubfindID_99 == ID, 'Snap_At_FirstEntry'].values[0]
                    SnapLostGas = dfSample.loc[dfSample.SubfindID_99 == ID, 'SnapLostGas'].values[0]

                    rOverR200 = np.flip(np.array([v for v in r_over_R_Crit200[str(ID)].values]))
                    rOverR200[:int(SnapAtEntry)] = np.nan
                    pericenters, _ = TNG.find_peaks(-rOverR200)
                    
                    if j == 0:
                        SnapBefore05 = np.nanmin(dfTime.loc[abs(dfTime.Age.values - dfTime.Age.loc[dfTime.Snap == SnapAtEntry].values[0]) < 0.5, 'Snap'].values)
                        snap = int(SnapBefore05)
                    elif j == 1:
                        SnapFirstPeri = np.nanmin(pericenters)
                        SnapBefore05 = np.nanmin(dfTime.loc[abs(dfTime.Age.values - dfTime.Age.loc[dfTime.Snap == SnapFirstPeri].values[0]) < 0.5, 'Snap'].values)

                        snap = int(SnapBefore05)
                        
                    elif j == 2:
                        SnapFirstPeri = np.nanmin(pericenters)
                        snap = int(SnapFirstPeri)

                    elif j == 3:
                        SnapFirstPeri = np.nanmin(pericenters)

                        SnapAfter05 = np.nanmax(dfTime.loc[abs(dfTime.Age.values - dfTime.Age.loc[dfTime.Snap == SnapFirstPeri].values[0]) < 0.5, 'Snap'].values)
                        snap = int(SnapAfter05)

                    elif j == 4:
                        if np.isnan(SnapLostGas) or SnapLostGas < 0:
                            SnapLostGas = 99
                        SnapLostGas = int(SnapLostGas)
                        SnapBefore05 = np.nanmin(dfTime.loc[abs(dfTime.Age.values - dfTime.Age.loc[dfTime.Snap == SnapLostGas].values[0]) < 0.5, 'Snap'].values)
                        snap = int(SnapBefore05)
                    
                else:
                    snap = snaps[j]
                
            
            if np.isnan(ID):
                grid[i*cNum+ j].axis('off')
                
               
            else:
            
                if 'Mass' in Param and PartType == 'PartType0':
                    cmap_dm = 'magma'

                if 'Mass' in Param and PartType == 'PartType4':
                    cmap_dm = 'bone'
                    
                if VelPlot:
                    img, extent,  VelArrayX, VelArrayY, PosArrayX, PosArrayY =  MakeMap(ID, snap, PartType = PartType, VelPlot = VelPlot, 
                                                                                        axis = axis, Param = Param,
                                                                                        r=r, width = width, cmap_dm = cmap_dm,  
                                                                                        SIMS = SIMS , SIMTNG = SIMTNG )
                    
                else:
                    
                    img, extent =  MakeMap(ID, snap, PartType = PartType, axis = axis, Param = Param,
                                                                 r=r, width = width, cmap_dm = cmap_dm,  
                                                                 SIMS = SIMS , SIMTNG = SIMTNG )
                if type(img) == np.ndarray:
                    
                    
                    if Param != None:
                        sc = grid[i*cNum+ j].imshow( np.log10(img), extent=extent, cmap=cmap_dm,origin='lower',
                                                    vmin = vmin, vmax = vmax)
    
                    else:
                        sc = grid[i*cNum+ j].imshow(img, extent=extent, cmap=cmap_dm, origin='lower')
                    
                    if VelPlot:
                        # Subsample arrows (only every nth arrow)
                        step = 100 # Adjust to control arrow density
                        grid_size = 30  # Controls smoothness of streamlines
                        PosArrayX = PosArrayX[::step]
                        PosArrayY = PosArrayY[::step]
                        VelArrayX = VelArrayX[::step]
                        VelArrayY = VelArrayY[::step]
                        
                        x_bins = np.linspace(PosArrayX.min(), PosArrayX.max(), grid_size)
                        y_bins = np.linspace(PosArrayY.min(), PosArrayY.max(), grid_size)
                        
                        x_grid, y_grid = np.meshgrid(x_bins, y_bins)

                        # Compute binned velocity field
                        vx_grid = np.zeros_like(x_grid)
                        vy_grid = np.zeros_like(y_grid)
                        counts = np.zeros_like(x_grid)
                        
                        # Bin velocities by averaging in each grid cell
                        for kgrid in range(len(VelArrayX)):
                            xi = np.digitize(PosArrayX[kgrid], x_bins) - 1
                            yi = np.digitize(PosArrayY[kgrid], y_bins) - 1
                            if 0 <= xi < grid_size - 1 and 0 <= yi < grid_size - 1:
                                vx_grid[yi, xi] += VelArrayX[kgrid]
                                vy_grid[yi, xi] += VelArrayY[kgrid]
                                counts[yi, xi] += 1
                        
                        # Normalize by the number of particles in each cell
                        mask = counts > 0
                        vx_grid[mask] /= counts[mask]
                        vy_grid[mask] /= counts[mask]

                        grid[i*cNum + j].streamplot(x_grid, y_grid, vx_grid, vy_grid, 
                                                    color='cyan', linewidth=1, density=1.2, arrowstyle='->')

                        #grid[i*cNum + j].quiver(PosArrayX, PosArrayY, VelArrayX, VelArrayY, color='red', scale=800, width=0.002)

                        
                    grid[i*cNum+ j].set_xlim(extent[0], extent[1])
                    grid[i*cNum+ j].set_ylim(extent[2], extent[3])
                
        
            if j == 0:
                if axis == 'z':
                    grid[i*cNum+ j].set_ylabel(r'$y/\mathrm{kpc}$', fontsize=fontlabel)
                    grid[i*cNum+ j].tick_params(axis='y', labelsize=0.98*fontlabel)
                
                elif axis == 'x':
                    grid[i*cNum+ j].set_ylabel(r'$z/\mathrm{kpc}$', fontsize=fontlabel)
                    grid[i*cNum+ j].tick_params(axis='y', labelsize=0.98*fontlabel)
                    
            if i == 0:
                ax2label = grid[i*cNum+ j].secondary_xaxis('top')
                ax2label.set_xlabel(r'$x/\mathrm{kpc}$', fontsize=fontlabel)
                ax2label.tick_params(labelsize=0.99*fontlabel)
               
                    
            if i == lNum - 1:
                grid[i*cNum+ j].set_xlabel(r'$x/\mathrm{kpc}$', fontsize=fontlabel)
                grid[i*cNum+ j].tick_params(axis='x', labelsize=0.98*fontlabel)
                
            if Type == 'Evolution' and i == 0 and not OrbitalEvolution:
                z = dfTime.z[99-int(snap)]
                b = z - int(z)
                if b < 0.005 :
                    zlabel = 'z = %.0f' % z
                else:
                    zlabel = 'z = %.1f' % z
                    
                grid[i*cNum+ j].set_title(zlabel,  fontsize=1.02*fontlabel)
                
            elif OrbitalEvolution:
                z = dfTime.z[99-int(snap)]
                b = z - int(z)
                if b < 0.005 :
                    zlabel = 'z = %.0f' % z
                else:
                    zlabel = 'z = %.1f' % z
                grid[i*cNum+ j].text(xtextZ, ytextZ, zlabel, color = 'blue',  fontsize=fontlabel)
                
            if  Type == 'Evolution' and j == cNum - 1 and len(Names) > 0 and Name != 'None':
                grid[i*cNum + j].text(xtext, ytext, 'ID: '+str(int(ID))+'\n'+Name, color = 'red',  fontsize=fontlabel)
                
            if Type == 'Mosaic' and len(Names) > 0 and Name != 'None':
                grid[i*cNum + j].text(xtext, ytext, 'ID: '+str(int(ID))+'\n'+Name, color = 'red',  fontsize=fontlabel)
            
    
    if Param != None:
        cb = fig.colorbar(sc, cax=grid.cbar_axes[0])
        if Param == 'Mass':
            cb.set_label(r'$\log(M/\mathrm{M}_\odot)$', fontsize=fontlabel)
        if Param == 'T':
            cb.set_label(r'$\log(T/\mathrm{K})$', fontsize=fontlabel)
        if Param == 'MassSF':
            cb.set_label(r'$\log(M_{\mathrm{sf-gas}}/\mathrm{M}_\odot)$', fontsize=fontlabel)

        cb.ax.tick_params(labelsize=0.98*fontlabel)
                    
                
    savefig(savepath, savefigname, False)

    return
                        
                
def MakeMap(ID, snap, PartType = 'PartType0', axis = 'x',  r='infinity', 
            Param = None, VelPlot = False,
            seed = 160401, lenLim = 2500000,
            width = 25, cmap_dm = 'GnBu', PATH = os.getenv("HOME")+'/TNG_Analyzes/SubhaloHistory',
            SIMS = 'TNG50-1', SIMTNG = 'TNG50', HOME = os.getenv("HOME")+'/'
            ):
    
    ID = int(ID)
    SubhaloHalfmassRadType4 = TNG.extractDF('SubhaloHalfmassRadType4')
    SubhaloHalfmassRadType0 = TNG.extractDF('SubhaloHalfmassRadType0')
    dfSubfindID = TNG.extractDF('SubfindID')

    IDatSnap = dfSubfindID[str(ID)].iloc[99 - snap]
    
    try:
        file = h5py.File(PATH+ '/' + SIMTNG +'/Particles/'+str(int(ID))+'/'+str(snap)+'Rotate.hdf5', 'r')
    except:
        
        PathOrigin = HOME+'SIMS/TNG/'+SIMS+'/snapshots/'+str(snap)+'/subhalos/'+str(int(IDatSnap))+'/cutout_'+SIMS+'_'+str(snap)+'_'+str(int(IDatSnap))+'.hdf5'
        try:
            shutil.copyfile(PathOrigin,  PATH+ '/' + SIMTNG +'/Particles/'+str(int(ID))+'/'+str(snap)+'Rotate.hdf5')
        except:
            f = TNG.extractParticles(ID, snaps = [snap])[0]
            f.close()
            shutil.copyfile(PathOrigin,  PATH+ '/' + SIMTNG +'/Particles/'+str(int(ID))+'/'+str(snap)+'Rotate.hdf5')

        file = h5py.File(PATH+ '/' + SIMTNG +'/Particles/'+str(int(ID))+'/'+str(snap)+'Rotate.hdf5', 'r+')
                         

        if 'PartType0' in file.keys():
            file = MATH.Rotate(file, 10**SubhaloHalfmassRadType0[str(ID)].iloc[99 - snap], z = dfTime.z.loc[dfTime.Snap == snap].values[0])
        elif 'PartType4' in file.keys():
            file = MATH.Rotate(file, 10**SubhaloHalfmassRadType4[str(ID)].iloc[99 - snap], z = dfTime.z.loc[dfTime.Snap == snap].values[0])
            

    np.random.seed(seed)
       
    try:
        posGas = file['PartType0']['Coordinates'][:] / (1+dfTime.z.values[dfTime.Snap == snap]) / h #kpc
        massGas = file['PartType0']['Masses'][:] * 1e10 / h
        velGas = file['PartType0']['Velocities'][:] * np.sqrt(1. / (1+dfTime.z[int(99-snap)]))
    except:
        posGas = np.array([[0,0,0]])
        massGas = np.array([0])
        velGas = np.array([[0,0,0]])
    try:
        posStar = file['PartType4']['Coordinates'][:] / (1+dfTime.z.values[dfTime.Snap == snap]) / h #kpc
        massStar = file['PartType4']['Masses'][:] * 1e10 / h
        velStar = file['PartType4']['Velocities'][:] * np.sqrt(1. / (1+dfTime.z[int(99-snap)]))
    except:
        posStar = np.array([[0,0,0]])
        massStar = np.array([0])
        velStar = np.array([[0,0,0]])
       
    
    if len(massStar) > 1 and massStar[0] != 0:
        Cen = np.array([MATH.weighted_median(posStar[:, 0], massStar), MATH.weighted_median(posStar[:, 1], massStar), MATH.weighted_median(posStar[:, 2], massStar)])
        VelMean = np.array([MATH.weighted_median(velStar[:, 0], massStar), MATH.weighted_median(velStar[:, 1], massStar), MATH.weighted_median(velStar[:, 2], massStar)])    
    else:
        Cen = np.array([MATH.weighted_median(posGas[:, 0], massGas), MATH.weighted_median(posGas[:, 1], massGas), MATH.weighted_median(posGas[:, 2], massGas)])
        VelMean = np.array([MATH.weighted_median(velGas[:, 0], massGas), MATH.weighted_median(velGas[:, 1], massGas), MATH.weighted_median(velGas[:, 2], massGas)])    
    
    if not PartType in file.keys():
        img = None
        extent = None
        VelArrayX = None
        VelArrayY = None
        PosArrayX = None
        PosArrayY = None
        if VelPlot:
            return img, extent, VelArrayX, VelArrayY, PosArrayX, PosArrayY
        else:
            return img, extent
    
    pos = file[PartType]['Coordinates'][:] / (1+dfTime.z.values[dfTime.Snap == snap]) / h #kpc
    mass = file[PartType]['Masses'][:] * 1e10 / h
    vel = file[PartType]['Velocities'][:] * np.sqrt(1. / (1+dfTime.z[int(99-snap)]))
    
    pos = ETNG.FixPeriodic(pos - Cen)
    vel = vel - VelMean


    if len(file[PartType]['Coordinates']) > lenLim:
        args = np.random.choice(np.arange(len(file[PartType]['Coordinates'])), size = lenLim, replace = False)
        pos = pos[args]
        mass = mass[args]
        vel = vel[args]
    
    if axis == 'x':  
        prad = 0
        t = 90
        
        if VelPlot:
        
            VelArrayX = vel[:, 0]
            VelArrayY = vel[:, 2]
            PosArrayX = pos[:, 0]
            PosArrayY = pos[:, 2]

    elif axis == 'z':
        prad = 0
        t = 0
        
        if VelPlot:
            VelArrayX = vel[:, 0]
            VelArrayY = vel[:, 1]
            PosArrayX = pos[:, 0]
            PosArrayY = pos[:, 1]
        
    
    if Param == 'Mass':
        qv = QuickView(pos,  r=r,  t=t, x = 0, y= 0 , z = 0, p = prad, plot=False, 
                       extent=[-width,width,-width,width], logscale=False)
        density_field = qv.get_image()
        
        
        qv   = QuickView(pos, mass = mass, plot=False,  r=r,  t=t, x = 0, y= 0 , z = 0, p = prad, 
                         extent=[-width,width,-width,width], logscale=False)

        img  =  qv.get_image() #/density_field
        extent = qv.get_extent()

    elif Param == 'T':
    
        
        u = file['PartType0']['InternalEnergy'][:]
        xe = file['PartType0']['ElectronAbundance'][:]
        Xh = 0.76
        gamma = 5./3.
        
        mp = 1.673e-24 #g
        
        kb = 1.380658e-16 # erg / K
        
        mu = 4./ (1. + 3. *Xh + 4. *Xh * xe) * mp
        T = (gamma - 1.) * (u / kb) * mu 
        T = T * 1e10

        qv   = QuickView(pos, T, plot=False,  r=r,  t=t, x = 0, y= 0 , z = 0, p = prad, 
                         extent=[-width,width,-width,width], logscale=False)

        img  =  qv.get_image()
        extent = qv.get_extent()
        
    elif Param == 'SFR':
        SFR = file['PartType0']['StarFormationRate'][:]

        qv = QuickView(pos, mass,  r=r,  t=t, x = 0, y= 0 , z = 0, p = prad, plot=False, 
                       extent=[-width,width,-width,width], logscale=False)
        density_field = qv.get_image()
        
        qv   = QuickView(pos, SFR, plot=False,  r=r,  t=t, x = 0, y= 0 , z = 0, p = prad, 
                         extent=[-width,width,-width,width], logscale=False)

        img  =  qv.get_image()/density_field
        extent = qv.get_extent()

    elif Param == 'MassSF':
        SFR = file['PartType0']['StarFormationRate'][:]

        qv   = QuickView(pos[SFR > 0], mass = mass[SFR > 0], plot=False,  r=r,  t=t, x = 0, y= 0 , z = 0, p = prad, 
                         extent=[-width,width,-width,width], logscale=False)

        img  =  qv.get_image()
        extent = qv.get_extent()
        
    elif 'Situ' in Param:
         IDs = file[PartType]['ParticleIDs'][:]
        
         df = pd.read_csv(PATH+'/'+SIMTNG+'/DFs/'+str(ID)+'_StarParticles.csv')
        
         Origin = df.Origin.values
         ID_df = df.Star_ID.values
         if 'Ex' in Param:
            IDSelect = ID_df[Origin == 'E']
            Cond = np.isin(IDs, IDSelect)
            
         elif 'In' in Param:
            IDSelect = ID_df[Origin == 'I']
            Cond = np.isin(IDs, IDSelect)

         qv = QuickView(pos[Cond],  r=r,  t=t, x = 0, y= 0 , z = 0, p = prad, plot=False, 
                        extent=[-width,width,-width,width], logscale=False)
         density_field = qv.get_image()
         
         
         qv   = QuickView(pos[Cond], mass = mass[Cond], plot=False,  r=r,  t=t, x = 0, y= 0 , z = 0, p = prad, 
                          extent=[-width,width,-width,width], logscale=False)

         img  =  qv.get_image()/density_field
         extent = qv.get_extent()
         
    elif len(Param) > 0:
         param = file['PartType0'][Param][:]

         qv = QuickView(pos, param,  r=r,  t=t, x = 0, y= 0 , z = 0, p = prad, plot=False, 
                        extent=[-width,width,-width,width], logscale=False)

         img  =  qv.get_image()
         extent = qv.get_extent()
        
    else:
        qv   = QuickView(pos,  plot=False,  r=r,  t=t, x = 0, y= 0 , z = 0, p = prad, extent=[-width,width,-width,width])
        img  = qv.get_image()
        extent = qv.get_extent()
    
    file.close()
    
    if VelPlot:
        return img, extent, VelArrayX, VelArrayY, PosArrayX, PosArrayY
    else:
        return img, extent




def HistExIn(dfs, IDs, Names, cNum = 6, lNum = 6, numBins = 16, fontlabel=18,
             columnspacing = 0.5, handlelength = 2, handletextpad = 0.4, labelspacing = 0.3, framealpha = 0.4,
             GridMake = True):
    
    # Define axes
    plt.rcParams.update({'figure.figsize': (cNum*2,lNum*len(dfs))})
    fig = plt.figure()
    gs = fig.add_gridspec(len(dfs), 2, hspace=0, wspace=0)
    axs = gs.subplots(sharex='col', sharey='row')

    # Verify axs shape
    if type(axs) is not list and type(axs) is not np.ndarray:
        axs = [axs]
    if type(axs[0]) is not np.ndarray:
        axs = np.array([axs])
        if len(dfs) == 1:
            axs = axs.T
    
    for i, df in enumerate(dfs):

        for j, row in enumerate(['r', 'vrad']):
            weightsInSitu = np.ones_like(df.loc[(df.BirthTag == 'InSitu'), 'mass'].values) / len(df.loc[(df.BirthTag == 'InSitu'), 'mass'].values)
            weightsExSitu = np.ones_like(df.loc[(df.BirthTag == 'ExSitu'), 'mass'].values) / len(df.loc[(df.BirthTag == 'ExSitu'), 'mass'].values)
        
            f = np.round(np.nansum(df.loc[(df.BirthTag == 'ExSitu'), 'mass'].values) / np.nansum(df.loc[(df.BirthTag == 'InSitu'), 'mass'].values), decimals = 2)
            if row == 'r':
                B = np.logspace(np.log10(min(df.r.values)), np.log10(max(df.r.values)), num = numBins)
            elif row == 'vrad':
                B = np.linspace(min(df.vrad.values), max(df.vrad.values), num = numBins)
            axs[i][j].hist(df.loc[(df.BirthTag == 'InSitu'), row].values, bins =  B, alpha = 0.4, 
                     color = 'blue', histtype='stepfilled',  weights= weightsInSitu, label = 'In-Situ')
            axs[i][j].hist(df.loc[(df.BirthTag == 'ExSitu'), row].values, bins =  B,alpha = 0.4, 
                     color = 'red', histtype='stepfilled', weights= weightsExSitu, label = 'Ex-Situ')
            
            axs[i][j].tick_params(labelsize=0.99*fontlabel)
            
            
            if GridMake:
                axs[i][j].grid(GridMake, color='#9e9e9e',  which="major", linewidth= 0.6,alpha= 0.3 , linestyle=':')

            if i == 0 and j == 0:
                axs[i][j].legend(fontsize=0.98*fontlabel, framealpha = framealpha, loc = 'best',
                    columnspacing = columnspacing, handlelength = handlelength, handletextpad = handletextpad, labelspacing = labelspacing)
                
            if j == 0:
                axs[i][j].set_ylabel('Normalized Count', fontsize = fontlabel)

            if j == 1:
                axs[i][j].text(75, 0.25, 'ID: '+str(IDs[i]), fontsize = 0.98*fontlabel)
                axs[i][j].text(75, 0.23, str(Names[i]), fontsize = 0.98*fontlabel)
                axs[i][j].text(75, 0.21, '$f_\mathrm{ex/in}:$ '+str(f), fontsize = 0.98*fontlabel)

                
            if row == 'r' and i == len(dfs) - 1:
                
                axs[i][j].set_xscale('log')
                axs[i][j].set_xlabel('$r [\mathrm{kpc}]$', fontsize = fontlabel)
            elif row == 'vrad'  and i == len(dfs) - 1:
                axs[i][j].set_xlabel('$v_\mathrm{rad} [\mathrm{km \; s}^{-1}]$', fontsize = fontlabel)
            
  
    
def colored_line(x, y, c, ax, **lc_kwargs):
    """
    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    if "array" in lc_kwargs:
        warnings.warn('The provided "array" keyword argument will be overridden')

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment

    return ax.add_collection(lc)


def MakeMedianAndIDs(Snaps, IDs, rmin, rmax, nbins, dfSample, PartType = 'PartType4', velPlot= False):
    yIDs  = np.array([])
    massIDs = np.array([])
    xIDs = np.array([])
    notIndex = np.array([])
    
    for l, ID in enumerate(IDs):
        if ID == 603556 or ID == 602133:
            continue
        snap = Snaps[l]
        if np.isnan(snap):
            continue
        snap = int(snap)
        #print('snap: ', snap)
        yrad, rad, mass = TNG.MakeDensityProfileMean(snap, ID, rmin, rmax, nbins, PartType = PartType, velPlot= velPlot)
        
        if len(yrad) == 1 or (ID == 603556 or ID == 602133):
            notIndex = np.append(notIndex, l)
            continue
        if l == 0 or len(yIDs ) == 0:
            yIDs  = np.append(yIDs , yrad)
            xIDs = np.append(xIDs, rad)
            massIDs = np.append(massIDs, mass)
    
        else:
            yIDs  = np.vstack((yIDs , yrad))
            massIDs = np.vstack((massIDs, mass))
            xIDs = np.vstack((xIDs, rad))
           
    
        Rvalues = xIDs.T
        Values = yIDs .T
        Masses = massIDs.T
    
    x = np.array([])
    y = np.array([])
    yerr = np.array([])
    mass = np.array([])

    if len(Values) > 0:
        if len(Values.shape) > 1:
            for k, value in enumerate(Values):
                x = np.append(x, np.nanmedian(Rvalues[k]))
                y = np.append(y, np.nanmedian(value))
                yerr = np.append(yerr, MATH.boostrap_func(value, func=np.nanmedian, num_boots=1000))
                mass = np.append(mass, np.nanmedian(Masses[k]))
        else:
            x = Rvalues
            y = Values
            yerr = np.zeros(len(y))
            mass = Masses
    
    else:
        x = np.nan
        y = np.nan
        yerr = np.nan
        mass = np.nan
            
    return x, y,yerr, mass, xIDs, yIDs, massIDs, notIndex


def MakeLines(j, ax,  yIDs, xIDs, IDs, notIndex, colors):
    k = 0
    for l, ID in enumerate(IDs):
        if l in notIndex:
            continue
        
        yvalues = yIDs[k]
        xvalues = xIDs[k]
        alpha = 0.1
        if j == 1 or j == 2:
            xvalues = xvalues[yvalues > 0] 
            yvalues = yvalues[yvalues > 0]*xvalues**2.
            if j == 2:
                if yvalues[xvalues == 1.01871524] > 5e7:
                    alpha = 0
            ax.plot(xvalues, yvalues , 
                                 lw = 0.15,  alpha = alpha,  color = colors[k])
        else:
            ax.plot(xvalues , yvalues, 
                                     lw = 0.15,  alpha = alpha,  color = colors[k])

        k = k+ 1
        
        #y_p2 = np.percentile(yIDs, 25, axis=0)     # 2.5th percentile
    #y_p97 = np.percentile(yIDs, 75, axis=0)   # 97.5th percentile
    #if j == 0:
    #    axs[j][linplot].fill_between(xvalues[~np.isnan(y_p2)], y_p2[~np.isnan(y_p2)], y_p97[~np.isnan(y_p97)], color=ColorFill, alpha=0.2)  # 2σ equivalent
    #else:
    #    axs[j][linplot].fill_between(xvalues[~np.isnan(y_p2)] , y_p2[~np.isnan(y_p2)]  * xvalues[~np.isnan(y_p2)]**2., y_p97[~np.isnan(y_p97)]  * xvalues[~np.isnan(y_p97)]**2., color=ColorFill, alpha=0.2)  # 2σ equivalent



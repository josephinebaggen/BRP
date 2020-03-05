'''
    Simple catalog
'''

import numpy as np
from sample_clone import HVSsample
from astropy.table import Table
from ejection_clone import Contigiani2018
from utils.dustmap import DustMap
from utils.mwpotential import MWPotential
from astropy import units as u

'''
    Create ejection catalog
'''

# Initialize an ejection model, i.e. how the spatial and velocity distribution of the stars will be sampled
# In this case the default is Contigiani2018
#ejectionmodel = Contigiani2018(name_modifier='TEST')

# Eject a sample of n stars from Sgr A*
# The n argument is how many stars the class /tries/ to eject, not necessarily how many are returned
# Any cuts on ejection parameters (see 'sampler' in ejection_example.py) will cut down number of returned stars
#mysample = HVSsample(ejectionmodel, name='My test sample', n=500,verbose=True)

# Save ejection sample
#mysample.save('./cat_ejection.fits')


'''
    Propagate ejection catalogue through the galaxy
'''

# Load ejection sample
#mysample = HVSsample('./cat_ejection.fits')
#mysample.subsample(cut = [1])

# Take the default MW galactic potential - a BH + Bulge + Disk + DM halo
m = np.arange(-0.5192, 0.2809, 0.1)
r = np.arange(1.19445, 1.6, 0.05)
vej = np.zeros([9,9]) * u.km/u.s
for i in range(len(m)):
    for j in range(len(r)):
        mysample = HVSsample('./cat_ejection.fits')
        mysample.subsample(cut = np.array([1]))
        default_potential = MWPotential(Ms = 10**m[i], rs=10**r[j])
        mysample.propagate(potential = default_potential, dt=0.1*u.Myr, threshold = 1e-7) # See documentation
        mysample.GetFinal()
        mysample.save('./cat_propagated'+str(i) + str(j)+'.fits')

        print(i,j)
        vej[i,j] = mysample.GCv[0]
        
print(vej)                    
#Propagate sample. Change timestep as needed
#mysample.propagate(potential = default_potential, dt=0.1*u.Myr, threshold = 1e-7) # See documentation


#Get final galactocentric distance and velocity for sample
#mysample.GetFinal()

#Save propagated sample
#mysample.save('./cat_propagated.fits')
#mysample = HVSsample('./cat_propagated.fits')
#mysample.GetVesc(default_potential)

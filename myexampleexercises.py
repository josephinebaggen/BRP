'''
    Simple catalog
'''

import numpy as np
import matplotlib.pyplot as plt
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

# Take the default MW galactic potential - a BH + Bulge + Disk + DM halo
#default_potential = MWPotential(Ms = 0.76)


#Propagate sample. Change timestep as needed
#mysample.propagate(potential = default_potential, dt=0.1*u.Myr, threshold = 1e-7) # See documentation


#Get final galactocentric distance and velocity for sample
#mysample.GetFinal()

#Save propagated sample
#mysample.save('./cat_propagated.fits')
#mysample = HVSsample('./cat_propagated.fits')
#mysample.GetVesc(default_potential)


#mysample.subsample(np.array(np.argwhere(mysample.v0>900*u.km/u.s)[26]))
#print(mysample.v0)
#mysample.propagate(potential = default_potential, dt=0.1*u.Myr, threshold = 1e-7)


#EXCERSISE 2
'''
m = np.arange(-0.5192, 0.2809, 0.1)
r = np.arange(1.19445, 1.6, 0.05)
vfinal = np.zeros([9,9]) * u.km/u.s
for i in range(len(m)):
    for j in range(len(r)):
        mysample = HVSsample('./cat_ejection.fits')
        mysample.subsample(cut = np.array(np.argwhere(mysample.v0>1600*u.km/u.s)[0]))
        default_potential = MWPotential(Ms = 10**m[i], rs=10**r[j])
        mysample.propagate(potential = default_potential, dt=0.1*u.Myr, threshold = 1e-7) # See documentation
        mysample.GetFinal()
        mysample.save('./cat_propagated'+str(i) + str(j)+'.fits')

        print(i,j)
        vfinal[i,j] = mysample.GCv[0]
        
print(vfinal)

for i in range(len(m)):
    plt.plot(r, vfinal[i], label = 'm = '+ str(m[i]))
plt.xlabel('r')
plt.ylabel('final velocity (km/s)')
plt.title('r dependence on velocity')
plt.legend(bbox_to_anchor = [1,1])
plt.show()
    
for i in range(len(r)):
    plt.plot(m, vfinal[:,i], label = 'r = '+ str(r[i]))
plt.xlabel('m')
plt.ylabel('final velocity (km/s)')
plt.title('m dependence on velocity')
plt.legend(bbox_to_anchor = [1,1])
plt.show()
'''
#EXCERSISE 3
default_potential = MWPotential(Ms = 0.76)
'''
#ejectionmodel = Contigiani2018(name_modifier='TEST')
#mysample = HVSsample(ejectionmodel, name='My test sample', n=500,verbose=True)
#mysample.save('./cat_ejection.fits')

mysample = HVSsample('./cat_ejection.fits')
mysample.subsample(np.array(np.argwhere(mysample.v0>900*u.km/u.s )))
print(mysample.size)
mysample.propagate(potential = default_potential, dt=0.1*u.Myr, threshold = 1e-7, plot = False) 
mysample.GetFinal()
mysample.save('./cat_propagated.fits')
mysample = HVSsample('./cat_propagated.fits')

delta = mysample.theta - mysample.theta0
plt.scatter(mysample.theta0/np.pi, delta/np.pi)
plt.xlabel('theta0 / $\pi$')
plt.ylabel('delta theta / $\pi$')
plt.show()

plt.scatter(mysample.v0, delta/np.pi)
plt.xlabel('Ejection velocity (km/s)')
plt.ylabel('delta theta / $\pi$')
plt.show()

from mpl_toolkits import mplot3d

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.scatter3D(mysample.theta0, mysample.v0, delta, c =delta,  cmap = 'viridis');

ax.set_xlabel('theta0 / $\pi$')
ax.set_zlabel('delta theta / $\pi$')
ax.set_ylabel('Ejection velocity (km/s)')
plt.show()
'''

#Get 1 star from the sample with velocity between 700 and 900 km/s
mysample = HVSsample('./cat_ejection.fits')
mysample.subsample(np.array(np.argwhere(((mysample.v0>800*u.km/u.s) * (mysample.v0<900*u.km/u.s )))[0]))
#Let the 1 star propagate and show the plot of all R, z
mysample.propagate(potential = default_potential, dt=0.1*u.Myr, threshold = 1e-7, plot = True) 
mysample.GetFinal()
mysample.save('./cat_propagated.fits')
print(mysample.v0)










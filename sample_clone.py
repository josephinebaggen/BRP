from astropy import units as u
import numpy  as np
from ejection_clone import EjectionModel
import time
import astropy.coordinates as coord

class HVSsample:
    '''
        HVS sample class. Main features:

        - Generate a sample of HVS at ejection according to a specified ejection model
        - Propagate the ejection sample in the Galaxy
        - Perform the Gaia selection cut and computes the expected errorbars in the phase space observables
        - Computes the sample likelihood of a given ejection model and galactic potential
        - Save/Load resulting catalog as FITS file (astropy.table)

        Attributes
        ---------
            self.size : int
                Size of the sample
            self.name : str
                Catalog name, 'Unknown'  by default
            self.ejmodel_name : str
                String identifier of the ejection model used to generate the sample, 'Unknown' by default
            self.cattype : int
                0 if ejection sample, 1 is galactic sample, 2 if Gaia sample
            self.dt : Quantity
                Timestep used for orbit integration, 0.01 Myr by default
            self.T_MW : Quantity
                Milky Way maximum lifetime

            self.r0, self.phi0, self.theta0, self.v0, self.phiv0, self.thetav0
                Initial phase space coordinates at ejection in galactocentrict spherical coordinates
            self.ra, self.dec, self.dist, self.pmra, self.pmdec, self.vlos : Quantity
                Equatorial coordinates of the stars. Right ascension and declination (ra, dec), heliocentric
                distance (dist), proper motion in ra and dec directions (pmra is declination-corrected),
                line of sight velocity (vlos)
            self.e_ra, self.e_dec, self.e_par, self.e_pmra, self.e_pmdec, self.e_vlos : Quantity
                Errors on the coordinates, photometry() computes e_par and e_vlos (parallax and vlos)

            self.GRVS : ndarray
                Apparent magnitude in the G_RVS Gaia band

            self.m
                Stellar masses of the sample
            self.tage, self.tflight : Quantity
                Age and flight time of the stars

            solarmotion : Quantity
                Solar motion

        Methods
        -------
            __init__():
                Initializes the class: loads catalog if one is provided, otherwise creates one based on a given
                ejection model

            _eject(): 
                Initializes a sample at t=0 ejected from Sgr A*
            backprop():
                Propagates a sample backwards in time for a given max integration time
            propagate():
                Propagates the sample in the Galaxy, changes cattype from 0 to 1
            photometry():
                Calculates the Gaia photometric properties, changes cattype from 1 to 2
            GetFinal():
                For a propagated sample, calculate galactocentric v as well as r, phi and theta in galactocentric spherical coordinates
            subsample():
                For a propagated sample, returns a subsample - either specific indices, a random selection of stars or ones that meet given velocity or G_RVS cutoffs
            save():
                Saves the sample in a FITS file
            _load():
                Load existing HVSsample, either ejected or propagated
            loadExt():
                For given .fits file of observations NOT created here, reads it in as an HVSsample
            likelihood():
                Checks the likelihood of the sample for a given potential&ejection model combination
    '''

    # Solar U, V, W peculiar motion in km/s in galactocentric coordinates. Galpy notation requires U to have a minus sign.

    #solarmotion = [-14., 12.24, 7.25] #Schonrich 2012
    solarmotion = [-11.1, 12.24, 7.25] #Schonrich 2010
    dt = 0.01*u.Myr
    T_MW = 13.8*u.Gyr # MW maximum lifetime from Planck2015

    #Sun-to-galactic-centre distance and Galactic rotation speed at the Solar position
    #DO NOT CHANGE THESE WITHOUT CONSULTING ME FIRST - it's not that simple
    vrot = [0., 220., 0.] * u.km / u.s #Canonical
    RSun = 8. * u.kpc #Canonical

    #vrot = [0., 232.76, 0.] * u.km / u.s #McMillan 2017
    #RSun = 8.178 * u.kpc #Gravity Collab 2019

    #Height of the Sun above the Galactic Disk
    zSun = 0.025 * u.kpc

    #@init
    def __init__(self, inputdata=None, name=None, isExternal=False,**kwargs):

        '''
        Parameters
        ----------
            inputdata : EjectionModel or str
                Instance of an ejection model or string to the catalog path
            name : str
                Name of the catalog
            isExternal : Bool
                Flag if the loaded catalog has been downloaded by another source, i.e. has not been generated here
            **kwargs : dict
                Arguments to be passed to the ejection model sampler if inputdata is an EjectionModel instance
        '''
        if(inputdata is None):
            raise ValueError('Initialize the class by either providing an \
                                ejection model or an input HVS catalog.')

        #Name catalog
        if(name is None):
            self.name = 'HVS catalog '+str(time.time())
        else:
            self.name = name
        #If inputdata is ejection model, create new ejection sample
        if isinstance(inputdata, EjectionModel):
            self._eject(inputdata, **kwargs)
        #If inputdata is a filename and isExternal=True, loads existing sample of star observations    
        if(isinstance(inputdata, str) and (isExternal)):
            self._loadExt(inputdata)
        #If inputdata is a filename and isExternal=False, loads existing already-propagated sample of stars
        if (isinstance(inputdata, str) and (not isExternal)):
            self._load(inputdata)

    #@eject
    def _eject(self, ejmodel, **kwargs):
        '''
            Initializes the sample as an ejection sample
        '''
        print('snark')
        self.ejmodel_name = ejmodel._name

        self.cattype = 0

        self.r0, self.phi0, self.theta0, self.v0, self.phiv0, self.thetav0, \
            self.m, self.tage, self.tflight, self.size = ejmodel.sampler(**kwargs)

    #@backprop
    def backprop(self, mh, rs, e, potential, dt=0.01*u.Myr, threshold=None):
        '''
            Propagates the sample in the Galaxy backwards in time.

            Parameters
            ----------
                potential : galpy potential instance
                    Potential instance of the galpy library used to integrate the orbits
                dt : Quantity
                    Integration timestep. Defaults to 0.01 Myr
                threshold : float
                    Maximum relative energy difference between the initial energy and the energy at any point needed
                    to consider an integration step an energy outliar. E.g. for threshold=0.01, any excess or
                    deficit of 1% (or more) of the initial energy is enough to be registered as outliar.
                    A table E_data.fits is created in the working directory containing for every orbit the percentage
                    of outliar points (pol)

        '''

        from galpy.orbit import Orbit
        from galpy.util.bovy_coords import pmllpmbb_to_pmrapmdec, lb_to_radec, vrpmllpmbb_to_vxvyvz, lbd_to_XYZ
        import os
        from astropy.table import Table
        import astropy.coordinates as coord

        if(threshold is None):
            check = False
        else:
            check = True

        # Integration time step
        self.dt = dt
        # Maximum integration time
        tint_max = 1000*u.Myr
        #Number of integration steps
        nsteps = np.ceil((tint_max/self.dt).to('1').value)

        self.orbits = [None] * self.size

        print(self.name)

        #Integration loop for the n=self.size orbits
        for i in range(self.size):
            ts = np.linspace(0, 1, nsteps)*tint_max

            #Initialize orbit instance using astrometry and motion of the Sun,
            #.flip() method reverses the orbit so we integrate backwards in time
            self.orbits[i] = Orbit(vxvv = [self.ra[i], self.dec[i], self.dist[i], \
                                    self.pmra[i], self.pmdec[i], self.vlos[i]], \
                                    solarmotion=self.solarmotion, radec=True).flip()

            self.orbits[i].integrate(ts, potential, method='dopr54_c')

            # Uncomment these and comment the rest of the lines in the for loop to return only final positions
            #self.dist[i], self.ll[i], self.bb[i], self.pmll[i], self.pmbb[i], self.vlos[i] = \
            #                                    self.orbits[i].dist(self.tflight[i], use_physical=True), \
            #                                    self.orbits[i].ll(self.tflight[i], use_physical=True), \
            #                                    self.orbits[i].bb(self.tflight[i], use_physical=True), \
            #                                    self.orbits[i].pmll(self.tflight[i], use_physical=True) , \
            #                                    self.orbits[i].pmbb(self.tflight[i], use_physical=True)  , \
            #                                    self.orbits[i].vlos(self.tflight[i], use_physical=True)

            self.testra, self.testdec, self.testdist, self.testpmra, self.testpmdec, self.testvlos = \
                                                self.orbits[i].ra(ts, use_physical=True)*u.deg, \
                                                self.orbits[i].dec(ts, use_physical=True)*u.deg, \
                                                self.orbits[i].dist(ts, use_physical=True)*u.kpc, \
                                                self.orbits[i].pmra(ts, use_physical=True)*u.mas/u.yr, \
                                                self.orbits[i].pmdec(ts, use_physical=True)*u.mas/u.yr, \
                                                self.orbits[i].vlos(ts, use_physical=True)*u.km/u.s

            #Path to write orbits to
            path='./My_Path_here'

            #Creates path if it doesn't already exist
            if not os.path.exists(path):
                os.mkdir(path)

            #Assembles table of equatorial coordinates for the star in each timestep
            datalist=[ts, self.testra, self.testdec, self.testdist, self.testpmra, self.testpmdec, self.testvlos]
            namelist = ['t', 'ra', 'dec', 'dist', 'pm_ra', 'pm_dec', 'vlos']
            data_table = Table(data=datalist, names=namelist)

            #Writes equatorial orbits to file. Each star gets its own file
            data_table.write(path+'/flight'+str(i)+'.fits', overwrite=True)

            vSun = [-self.solarmotion[0], self.solarmotion[1], self.solarmotion[2]] * u.km / u.s # (U, V, W)

            #Initializes galactocentric Cartesian reference frame, accounting for solar motion and rotation
            v_sun = coord.CartesianDifferential(self.vrot+vSun)
            gc = coord.Galactocentric(galcen_distance=self.RSun, z_sun=self.zSun, galcen_v_sun=v_sun)

            #Transforms equatorial coordinates of orbits to Galactic Cartesian
            ICRS = coord.ICRS(ra=self.testra, dec=self.testdec, distance=self.testdist, pm_ra_cosdec=self.testpmra, pm_dec=self.testpmdec, radial_velocity=self.testvlos)
            gal = ICRS.transform_to(gc)

            #Grabs the phase space info in Cartesian
            v_x, v_y, v_z = gal.v_x, gal.v_y, gal.v_z
            xpos, ypos, zpos = gal.x, gal.y, gal.z
            r = np.sqrt(xpos**2 + ypos**2 + zpos**2)

            datalist=[ts, xpos, ypos, zpos, v_x, v_y, v_z]
            namelist = ['t', 'x', 'y', 'z', 'v_x', 'v_y', 'v_z']
            data_table = Table(data=datalist, names=namelist)
            data_table.write(path+'/flight'+str(i)+'_Cart.fits', overwrite=True)

        # Uncomment these to write final positions only
        # Radial velocity and distance + distance modulus
        #self.vlos, self.dist = self.vlos * u.km/u.s, self.dist * u.kpc

        # Sky coordinates and proper motion
        #data = pmllpmbb_to_pmrapmdec(self.pmll, self.pmbb, self.ll, self.bb, degree=True)*u.mas / u.year
        #self.pmra, self.pmdec = data[:, 0], data[:, 1]
        #data = lb_to_radec(self.ll, self.bb, degree=True)* u.deg
        #self.ra, self.dec = data[:, 0], data[:, 1]

        #datalist=[ts, self.ra, self.dec, self.dist, self.pmra, self.pmdec, self.vlos]
        #namelist = ['t', 'ra', 'dec', 'dist', 'pm_ra', 'pm_dec', 'vlos']
        #data_table = Table(data=datalist, names=namelist)
        #data_table.write('/path/to/where/you/want.fits', overwrite=True)

    #@propagate
    def propagate(self, potential, dt=0.01*u.Myr, threshold=None):
        '''
            Propagates the sample in the Galaxy forwards in time, changes cattype from 0 to 1.

            Parameters
            ----------
                potential : galpy potential instance
                    Potential instance of the galpy library used to integrate the orbits
                dt : Quantity
                    Integration timestep. Defaults to 0.01 Myr
                threshold : float
                    Maximum relative energy difference between the initial energy and the energy at any point needed
                    to consider an integration step an energy outliar. E.g. for threshold=0.01, any excess or
                    deficit of 1% (or more) of the initial energy is enough to be registered as outliar.
                    A table E_data.fits is created in the working directory containing for every orbit the percentage
                    of outliar points (pol)

        '''
        from galpy.orbit import Orbit
        from galpy.util.bovy_coords import pmllpmbb_to_pmrapmdec, lb_to_radec, vrpmllpmbb_to_vxvyvz, lbd_to_XYZ
        from astropy.table import Table
        import astropy.coordinates as coord

        if(threshold is None):
            check = False
        else:
            check = True

        # Integration time step
        self.dt = dt
        tint_max = 100*u.Myr
        nsteps = np.ceil((self.tflight/self.dt).to('1').value)
        #nsteps = np.ceil((tint_max/self.dt).to('1').value)
        nsteps[nsteps<100] = 100

        # Initialize position in cylindrical coords
        rho = self.r0 * np.sin(self.theta0)
        z = self.r0 * np.cos(self.theta0)
        phi = self.phi0

        #... and velocity
        vR = self.v0 * np.sin(self.thetav0) * np.cos(self.phiv0)
        vT = self.v0 * np.sin(self.thetav0) * np.sin(self.phiv0)
        vz = self.v0 * np.cos(self.thetav0)

        # Initialize empty arrays to save orbit data and integration steps
        self.pmll, self.pmbb, self.ll, self.bb, self.vlos, self.dist, self.energy_var = \
                                                                            (np.zeros(self.size) for i in range(7))

        self.orbits = [None] * self.size

        #Integration loop for the self.size orbits
        for i in range(self.size):
            #print(i,self.tflight[i])
            ts = np.linspace(0, 1, nsteps[i])*self.tflight[i]

            #Initialize orbit using galactocentric cylindrical phase space info of stars
            self.orbits[i] = Orbit(vxvv = [rho[i], vR[i], vT[i], z[i], vz[i], phi[i]], solarmotion=self.solarmotion)
            self.orbits[i].integrate(ts, potential, method='dopr54_c')

            # Export the Final position
            self.dist[i], self.ll[i], self.bb[i], self.pmll[i], self.pmbb[i], self.vlos[i] = \
                                                self.orbits[i].dist(np.ones(1)*self.tflight[i], use_physical=True), \
                                                self.orbits[i].ll(np.ones(1)*self.tflight[i], use_physical=True), \
                                                self.orbits[i].bb(np.ones(1)*self.tflight[i], use_physical=True), \
                                                self.orbits[i].pmll(np.ones(1)*self.tflight[i], use_physical=True) , \
                                                self.orbits[i].pmbb(np.ones(1)*self.tflight[i], use_physical=True)  , \
                                                self.orbits[i].vlos(np.ones(1)*self.tflight[i], use_physical=True)

            #Print ALL positions
            #self.dist, self.ll, self.bb, self.pmll, self.pmbb, self.vlos, self.testra, self.testdec, self.testpmra, self.testpmdec = \
            #                                    self.orbits[i].dist(ts, use_physical=True), \
            #                                    self.orbits[i].ll(ts, use_physical=True), \
            #                                    self.orbits[i].bb(ts, use_physical=True), \
            #                                    self.orbits[i].pmll(ts, use_physical=True) , \
            #                                    self.orbits[i].pmbb(ts, use_physical=True)  , \
            #                                    self.orbits[i].vlos(ts, use_physical=True) , \
            #                                    self.orbits[i].ra(ts, use_physical=True) , \
            #                                    self.orbits[i].dec(ts, use_physical=True) , \
            #                                    self.orbits[i].pmra(ts, use_physical=True) , \
            #                                    self.orbits[i].pmdec(ts, use_physical=True)



            #self.testra, self.testdec, self.testdist, self.testpmra, self.testpmdec, self.testvlos = \
            #                                    self.orbits[i].ra(ts, use_physical=True)*u.deg, \
            #                                    self.orbits[i].dec(ts, use_physical=True)*u.deg, \
            #                                    self.orbits[i].dist(ts, use_physical=True)*u.kpc, \
            #                                    self.orbits[i].pmra(ts, use_physical=True)*u.mas/u.yr, \
            #                                    self.orbits[i].pmdec(ts, use_physical=True)*u.mas/u.yr, \
            #                                    self.orbits[i].vlos(ts, use_physical=True)*u.km/u.s

            #vSun = [-self.solarmotion[0], self.solarmotion[1], self.solarmotion[2]] * u.km / u.s # (U, V, W)
            #v_sun = coord.CartesianDifferential(self.vrot+vSun)

            #gc = coord.Galactocentric(galcen_distance=self.RSun, z_sun=self.zSun, galcen_v_sun=v_sun)
            #galactic = coord.Galactic(l=self.ll, b=self.bb, distance=self.dist, pm_l_cosb=self.pmll, pm_b=self.pmbb, radial_velocity=self.vlos)
            #gal = galactic.transform_to(gc)

            #v_x, v_y, v_z = gal.v_x, gal.v_y, gal.v_z
            #xpos, ypos, zpos = gal.x, gal.y, gal.z
            #r = np.sqrt(xpos**2 + ypos**2 + zpos**2)

        # Radial velocity and distance
        self.vlos, self.dist = self.vlos * u.km/u.s, self.dist * u.kpc

        # Sky coordinates and proper motion
        data = pmllpmbb_to_pmrapmdec(self.pmll, self.pmbb, self.ll, self.bb, degree=True)*u.mas / u.year
        self.pmra, self.pmdec = data[:, 0], data[:, 1]
        data = lb_to_radec(self.ll, self.bb, degree=True)* u.deg
        self.ra, self.dec = data[:, 0], data[:, 1]

        # Done propagating
        self.cattype = 1

        #Uncomment these to write final coordinates to file (in Galactic frame)
        #datalist=[ts, self.ll, self.bb, self.dist, self.pmll, self.pmbb, self.vlos]
        #namelist = ['t', 'll', 'bb', 'dist', 'pmll', 'pmbb', 'vlos']
        #data_table = Table(data=datalist, names=namelist)
        #data_table.write('/path/to/where/you/want.fits', overwrite=True)

        #Uncomment these to write final coordinates to file(in equatorial frame)
        #datalist=[ts, self.ra, self.dec, self.dist, self.pmra, self.pmdec, self.vlos]
        #namelist = ['t', 'ra', 'dec', 'dist', 'pm_ra', 'pm_dec', 'vlos']
        #data_table = Table(data=datalist, names=namelist)
        #data_table.write('/path/to/where/you/want.fits', overwrite=True)

        #Uncomment these to write final coordinates to file (in Cartesian frame)
        #vSun = [-self.solarmotion[0], self.solarmotion[1], self.solarmotion[2]] * u.km / u.s # (U, V, W)
        #v_sun = coord.CartesianDifferential(self.vrot+vSun)

        #gc = coord.Galactocentric(galcen_distance=self.RSun, z_sun=self.zSun, galcen_v_sun=v_sun)
        #galactic = coord.Galactic(l=self.ll, b=self.bb, distance=self.dist, pm_l_cosb=self.pmll, pm_b=self.pmbb, radial_velocity=self.vlos)
        #gal = galactic.transform_to(gc)

        #v_x, v_y, v_z = gal.v_x, gal.v_y, gal.v_z
        #xpos, ypos, zpos = gal.x, gal.y, gal.z
        #r = np.sqrt(xpos**2 + ypos**2 + zpos**2)

        #datalist=[xpos, ypos, zpos, v_x, v_y, v_z,ts, np.arccos(zpos/r), np.arctan(ypos/xpos), r]
        #print([len(xpos),len(v_x),len(ts),len(r)])
        #namelist = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'ts', 'theta', 'phi', 'r']
        #data_table = Table(data=datalist, names=namelist)
        #data_table.write('/path/to/where/you/want.fits', overwrite=True)
        
    def GetVesc(self, potential):
            '''
            Returns the escape speed of a given potential for each star in a propagated sample
            '''
           
            from galpy.potential import evaluatePotentials

            #find galactocentric x,y,z coordinates of propagated sample
            
            #c = SkyCoord(self.ra*u.degree, self.dec *u.degree, distance=self.GCdist)
            vSun = [-self.solarmotion[0], self.solarmotion[1], self.solarmotion[2]] * u.km / u.s # (U, V, W)

            #Initializes galactocentric Cartesian reference frame, accounting for solar motion and rotation
            v_sun = coord.CartesianDifferential(self.vrot+vSun)
            gc = coord.Galactocentric(galcen_distance=self.RSun, z_sun=self.zSun, galcen_v_sun=v_sun) 
            
            #Transforms equatorial coordinates of orbits to Galactic Cartesian
            ICRS = coord.ICRS(ra=self.ra, dec=self.dec, distance=self.dist)
            gal = ICRS.transform_to(gc)
            
           
            galactocentric_coords_x = gal.x
            galactocentric_coords_y = gal.y
            galactocentric_coords_z = gal.z
            
            #galactocentric_coords.y = self.GCdist * np.sin(self.theta) *np.sin(self.phi)
            #galactocentric_coords.z = self.GCdist * np.cos(self.theta)
            
            R = np.sqrt(galactocentric_coords_x**2. + galactocentric_coords_y**2)
            
            z = galactocentric_coords_z
            #print(R,z)
            #print("hoi")
            #Initialize Vesc array
            self.Vesc = np.zeros(self.size)#*u.km/u.s
            escaped = np.zeros(self.size, dtype = bool)
            
            for i in range(self.size):
                Potvalue = evaluatePotentials(potential,R[i],z[i])
                self.Vesc[i] = np.sqrt(2* (- Potvalue))
                if self.Vesc[i]*u.km/u.s <= self.GCv[i]:
                    escaped[i] = True
                    
                else:
                    escaped[i] = False
                    
            
     
            print('Fraction escaped = ' ,np.sum(escaped)/len(escaped) * 100, '%' )
            
            self.subsample(cut=np.argwhere(escaped))
            print(self.size)
            #print(self.Vesc[escaped])
            #print(len(self.Vesc[escaped]))
            
    #@photometry
    def photometry(self, dustmap=None, v=True):
        '''
        Computes the Grvs-magnitudes and total velocity in Galactocentric restframe.

        Parameters
        ----------
            dustmap : DustMap
                Dustmap object to be used
        '''

        from utils.dustmap import DustMap
        from utils.gaia import get_GRVS
        from galpy.util.bovy_coords import radec_to_lb

        if(self.cattype < 1):
            raise RuntimeError('The catalog needs to be propagated!')

        if(not isinstance(dustmap, DustMap)):
            raise ValueError('You must provide an instance of the class DustMap.')

        #Needs galactic lat/lon to get G_RVS - converts to it if only equatorial are available
        if(hasattr(self,'ll')):
            self.GRVS, self.V, self.G, self.e_par, self.e_pmra, self.e_pmdec = get_GRVS(self.dist.to('kpc').value, self.ll, self.bb, self.m.to('Msun').value, self.tage.to('Myr').value, dustmap)
        else:
            data = radec_to_lb(self.ra.to('deg').value, self.dec.to('deg').value, degree=True)
            l, b = data[:, 0], data[:, 1]

            #Calls get_GRVS to get G_RVS magnitude accounting for Galactic dust extinction
            self.GRVS, self.V, self.G, self.e_par, self.e_pmra, self.e_pmdec = get_GRVS(self.dist.to('kpc').value, l, b, self.m.to('Msun').value, self.tage.to('Myr').value, dustmap)
    
        self.e_par = self.e_par * u.uas
        self.e_pmra = self.e_pmra * u.uas / u.yr
        self.e_pmdec = self.e_pmdec * u.uas /u.yr

        #If v=TRUE, calculates total velocity and distance in the galactocentric frame
        if(v):
            import astropy.coordinates as coord
            from galpy.util.bovy_coords import radec_to_lb, pmrapmdec_to_pmllpmbb

            vSun = [-self.solarmotion[0], self.solarmotion[1], self.solarmotion[2]] * u.km / u.s # (U, V, W)
            v_sun = coord.CartesianDifferential(vSun + self.vrot)
            GCCS = coord.Galactocentric(galcen_distance=self.RSun, z_sun=self.zSun, galcen_v_sun=v_sun)

            data = radec_to_lb(self.ra.to('deg').value, self.dec.to('deg').value, degree=True)
            ll, bb = data[:, 0], data[:, 1]
            data = pmrapmdec_to_pmllpmbb(self.pmra, self.pmdec, self.ra.to('deg').value, self.dec.to('deg').value, degree=True)
            pmll, pmbb = data[:, 0], data[:, 1]

            galactic_coords = coord.Galactic(l=ll*u.deg, b=bb*u.deg, distance=self.dist, pm_l_cosb=pmll*u.mas/u.yr, pm_b=pmbb*u.mas/u.yr, radial_velocity=self.vlos)
            galactocentric_coords = galactic_coords.transform_to(GCCS)

            self.vtot = np.sqrt(galactocentric_coords.v_x**2. + galactocentric_coords.v_y**2. + galactocentric_coords.v_z**2.).to(u.km/u.s)
            self.GCv = np.sqrt(galactocentric_coords.v_x**2. + galactocentric_coords.v_y**2. + galactocentric_coords.v_z**2.).to(u.km/u.s)
            self.GCdist = np.sqrt(galactocentric_coords.x**2. + galactocentric_coords.y**2. + galactocentric_coords.z**2.).to(u.kpc)

        self.cattype = 2

    #@GetVtot
    def GetFinal(self, v=True):
        '''
        Returns galactocentric distance, velocity, theta and phi of propagated sample

        '''

        #from utils import DustMap
        #from utils.gaia import get_GRVS
        from galpy.util.bovy_coords import radec_to_lb
        import astropy.coordinates as coord
        from galpy.util.bovy_coords import radec_to_lb, pmrapmdec_to_pmllpmbb

        #if(self.cattype < 1):
        #    raise RuntimeError('The catalog needs to be propagated!')

        vSun = [-self.solarmotion[0], self.solarmotion[1], self.solarmotion[2]] * u.km / u.s # (U, V, W)
        v_sun = coord.CartesianDifferential(vSun + self.vrot)

        GCCS = coord.Galactocentric(galcen_distance=self.RSun, z_sun=self.zSun, galcen_v_sun=v_sun)

        data = radec_to_lb(self.ra.to('deg').value, self.dec.to('deg').value, degree=True)
        ll, bb = data[:, 0], data[:, 1]
        data = pmrapmdec_to_pmllpmbb(self.pmra, self.pmdec, self.ra.to('deg').value, self.dec.to('deg').value, degree=True)
        pmll, pmbb = data[:, 0], data[:, 1]

        galactic_coords = coord.Galactic(l=ll*u.deg, b=bb*u.deg, distance=self.dist, pm_l_cosb=pmll*u.mas/u.yr, pm_b=pmbb*u.mas/u.yr, radial_velocity=self.vlos)
        galactocentric_coords = galactic_coords.transform_to(GCCS)

        self.GCdist = np.sqrt(galactocentric_coords.x**2. + galactocentric_coords.y**2. + galactocentric_coords.z**2.).to(u.kpc)
        self.GCv = np.sqrt(galactocentric_coords.v_x**2. + galactocentric_coords.v_y**2. + galactocentric_coords.v_z**2.).to(u.km/u.s)

        self.thetaf = np.arccos(galactocentric_coords.z/self.GCdist)
        self.phif = np.arctan2(galactocentric_coords.y,galactocentric_coords.x)

        self.cattype = 1

    #@subsample
    def subsample(self, cut=None, vtotcut=450*u.km/u.s, GRVScut=16):
        '''
        Restrict the sample based on the values of cut.

        Parameters
        ----------
            cut : str or np.array or int
                If 'Gaia' the sample is cut according to GRVScut and vtot arguments above,
                If np.array, the indices corresponding to cut are selected,
                If int, N=cut number of points are selected randomly.
            vtotcut : Quantity
                Total galactocentric velocity cut to satisfy if cut=='Gaia'
            GRVScut : float
                Cut in GRVS apparent magnitude to satisfy if cut=='Gaia'
        '''

        namelist = ['r0', 'phi0', 'theta0', 'v0', 'phiv0', 'thetav0', 'm', 'tage', 'tflight', 'ra', 'dec', 'pmra',
                    'pmdec', 'dist', 'vlos', 'GRVS', 'V', 'G', 'e_par', 'e_pmra', 'e_pmdec', 'vtot', 'GCv', 'GCdist', 'thetaf','phif']

        if(cut == 'Gaia'):
                idx = (self.vtot > vtotcut) & (self.GRVS<GRVScut)
                for varname in namelist:
                    try:
                        setattr(self, varname, getattr(self, varname)[idx])
                    except:
                        pass
                self.size = idx.sum()

        if(type(cut) is int):
                idx_e = np.random.choice(np.arange(int(self.size)), cut, replace=False)
                for varname in namelist:
                    try:
                        setattr(self, varname, getattr(self, varname)[idx_e])
                    except:
                        pass
                self.size = cut
        if(type(cut) is np.ndarray):
                for varname in namelist:
                    try:
                        setattr(self, varname, getattr(self, varname)[cut])
                    except:
                        pass
                self.size = cut.size
       
    #@likelihood
    #Ignore this for now
    def likelihood(self, potential, ejmodel,sigmar=10,sigmaL=10, dt=0.001*u.Myr, xi = 0, individual=False, weights=None, tint_max=150*u.Myr):
        '''
        Computes the non-normalized ln-likelihood of a given potential and ejection model for a given potential.
        When comparing different ejection models or biased samples, make sure you renormalize the likelihood
        accordingly. See Contigiani+ 2018.

        Can return the ln-likelihoods of individual stars if individual is set to True.

        Parameters
        ----------
        potential : galpy potential
            Potential to be tested and to integrate the orbits with.
        ejmodel : EjectionModel
            Ejectionmodel to be tested.
        individual : bool
            If True the method returns individual likelihoods. The default value is False.
        weights : iterable
            List or array containing the weights for the ln-likelihoods of the different stars.
        xi : float or array
            Assumed metallicity for stellar lifetime
        tint_max : Quantity
            Integrate back all stars only for a time tint_max.

        Returns
        -------

        log likelihood values : numpy.array or float
            Returns the ln-likelihood of the entire sample or the log-likelihood for every single star if individual
            is True.

        '''
        from galpy.orbit import Orbit
        import astropy.coordinates as coord
        from astropy.table import Table
        from utils import t_MS

        if(self.cattype == 0):
            raise ValueError("The likelihood can be computed only for a propagated sample.")

        if(self.size > 1e3):
            print("You are computing the likelihood of a large sample. This might take a while.")

        weights = np.array(weights)
        if((weights != None) and (weights.size != self.size)):
            raise ValueError('The length of weights must be equal to the number of HVS in the sample.')

      #  self.size = 151
        self.backwards_orbits = [None] * self.size
        self.xfinal = [None] * self.size
        self.yfinal = [None] * self.size
        self.zfinal = [None] * self.size
        self.vfinal = [None] * self.size
        self.dfinal = [None] * self.size
        self.Lfinal = [None] * self.size
        self.L0 = [None] * self.size
        self.minr = [None] * self.size
        self.minL = [None] * self.size
        self.back_dt = dt
        self.lnlike = np.ones(self.size) * (-np.inf)

        #if(tint_max is None):
        lifetime = t_MS(self.m, xi)
        lifetime[lifetime>self.T_MW] = self.T_MW
        #else:
         #lifetime = tint_max*np.ones(self.size)

        nsteps = np.ceil((lifetime/self.back_dt).to('1').value)
        nsteps = np.ceil((tint_max*np.ones(self.size)/self.back_dt).to('1').value)

        #print(nsteps)
        #self.tflight = self.tflight + 1*u.Myr
        #nsteps = np.ceil((self.tflight/self.back_dt).to('1').value)
        #print(nsteps)

        nsteps[nsteps<100] = 100
        
        vSun = [-self.solarmotion[0], self.solarmotion[1], self.solarmotion[2]] * u.km / u.s # (U, V, W)
        vrot = [0., 220., 0.] * u.km / u.s

        RSun = 8. * u.kpc
        zSun = 0.025 * u.kpc

        v_sun = coord.CartesianDifferential(vrot+vSun)
        gc = coord.Galactocentric(galcen_distance=RSun, z_sun=zSun, galcen_v_sun=v_sun)

        #print(self.size)
        for i in range(self.size):
        #for i in range(star,star+1):
            #ts = np.linspace(0, 1, nsteps[i])*lifetime[i]
            ts = np.linspace(0, 1, nsteps[i])*tint_max

            #ts = np.linspace(0, 1, 2)*lifetime[i]
            #ts = np.linspace(0, 1, nsteps[i])*(self.tflight[i])
            #ts = np.linspace(0, 1, nsteps[i])*lifetime[i]*(self.tflight[i]/lifetime[i])

            #idx = (ts<self.tflight[i])
            #ts = ts[idx]

            self.backwards_orbits[i] = Orbit(vxvv = [self.ra[i], self.dec[i], self.dist[i], \
                                    self.pmra[i], self.pmdec[i], self.vlos[i]], \
                                    solarmotion=self.solarmotion, radec=True).flip()
            #self.backwards_orbits[i].integrate(ts, potential, method='dopr54_c')
            #self.backwards_orbits[i].integrate(ts, potential)
            #self.backwards_orbits[i].integrate(ts, potential,method='rk4_c')
            self.backwards_orbits[i].integrate(ts, potential, method='leapfrog_c')

            dist, ll, bb, pmll, pmbb, vlos, testL2 = self.backwards_orbits[i].dist(ts, use_physical=True) * u.kpc, \
                                                self.backwards_orbits[i].ll(ts, use_physical=True) * u.deg, \
                                                self.backwards_orbits[i].bb(ts, use_physical=True) * u.deg, \
                                                self.backwards_orbits[i].pmll(ts, use_physical=True) *u.mas/u.year, \
                                                self.backwards_orbits[i].pmbb(ts, use_physical=True) * u.mas/u.year, \
                                                self.backwards_orbits[i].vlos(ts, use_physical=True) * u.km/u.s , \
                                                self.backwards_orbits[i].L(ts, use_physical=True) * u.kpc*u.km/u.s
                                        
            galactic = coord.Galactic(l=ll, b=bb, distance=dist, pm_l_cosb=pmll, pm_b=pmbb, radial_velocity=vlos)
            gal = galactic.transform_to(gc)
            vx, vy, vz = gal.v_x, gal.v_y, gal.v_z
            x, y, z = gal.x, gal.y, gal.z

            #print(testL2)
            #testL = np.sqrt((y*vz - z*vy)**2. + (x*vz-z*vx)**2. + (x*vy-y*vx)**2.)
            r = np.sqrt(x**2+y**2+z**2)
            v = np.sqrt(vx**2+vy**2+vz**2)

            #print('end transform, start L')
            #testr = ((testd - 0.003*u.kpc)/(0.01*u.kpc)).to(1).value # normalized r
            #testL = (testL/(10*u.pc*u.km/u.s)).to(1).value # normalized L

            L = map(np.linalg.norm,testL2)*u.kpc*u.km/u.s
            #L = np.sqrt((y*vz - z*vy)**2. + (x*vz-z*vx)**2. + (x*vy-y*vx)**2.)

            #L = testL2*u.kpc*u.km/u.s
            #print(L)

            #print([self.m[i],lifetime[i],testd[0],min(testd),min(testr),testL[np.argmin(testd)]])
            
            #self.sigmar = 50000*u.pc
            #self.sigmaL = 30000000*u.pc*u.km/u.s

            setattr(self,'sigmar',sigmar*u.pc)
            setattr(self,'sigmaL',sigmaL*u.pc*u.km/u.s)

            #print(self.sigmar)
            #print('end L,start like')
            #self.lnlike[i] = np.log( ( ejmodel.R2(self.m[i], v, r, L,self.sigmar,self.sigmaL) * ejmodel.g( np.linspace(0, 1, nsteps[i]) ) ).sum() )
            #self.lnlike[i] = np.log( ( ejmodel.R2(self.m[i], v, r, L,self.sigmar,self.sigmaL)  ).sum() )
            self.lnlike[i] = np.log( ( ejmodel.R2(self.m[i], v, r, L,self.sigmar,self.sigmaL) * ejmodel.g( np.linspace(0, 1, nsteps[i])*(self.tflight[i]/lifetime[i]) ) ).sum() )
            #self.lnlike[i] = np.log( ( ejmodel.R2(self.m[i], v, r, L,self.sigmar,self.sigmaL) * ejmodel.g( np.linspace(0, 1, len(v))*(self.tflight[i]/lifetime[i]) ) ).sum() )
            #like = np.log( ( ejmodel.R2(self.m[i], v, r, L,self.sigmar,self.sigmaL) * ejmodel.g( np.linspace(0, 1, len(v))*(self.tflight[i]/lifetime[i]) ) ))
            like = np.log( ( ejmodel.R2(self.m[i], v, r, L,self.sigmar,self.sigmaL) * ejmodel.g( np.linspace(0, 1, nsteps[i])) ) )
            #print('end like, start minr/minL')
            #self.lnlike[i] = np.log( ejmodel.R(self.m[i], vx, vy, vz, x, y, z,self.sigmar,self.sigmaL) * ejmodel.g( np.linspace(0, 1, nsteps[i]) ) ).sum() 
    
            self.minr[i] = min(r).value

            rnorm = ((r-3*u.pc)/(sigmar*u.pc)).to(1).value
            rdiff = [item - rnorm[j-1] for j,item in enumerate(rnorm)][1:]
            rdiff = np.array([-100] + rdiff)
            idx = (rnorm < 4) #& (rnorm > 0) & (rdiff<0)
            print(r[idx][np.argmin(L[idx])]) 
            print(r[idx][np.argmin(L[idx])-1]) 
            print(r[idx][np.argmin(L[idx])+1]) 
            self.minL[i] = min(L[idx]).value if len(idx[idx])>0 else np.inf
            #self.minL[i] = L[idx][-1].value if len(idx[idx])>0 else np.inf

            #idx = ((1000.*r.value-3)/sigmar) < 4
            #self.minL[i] = L[np.argmin(np.array(r.value))].value
            #self.minL[i] = min(L[idx]).value if len(idx[idx])>0 else np.inf

            theta = np.arccos(np.array(z.value)/np.array(r.value))
            phi = np.arctan(np.array(y.value)/np.array(x.value))

            #print([i,self.lnlike[i]])
            #if((self.lnlike[i] == -np.inf) and (not individual)):
            #    print('snark')
            #    break

            #self.xfinal[i] = x[0].value
            #self.yfinal[i] = y[0].value
            #self.zfinal[i] = z[0].value
            #self.vfinal[i] = v[0].value
            #self.dfinal[i] = np.sqrt(self.xfinal[i]**2 + self.yfinal[i]**2 + self.zfinal[i]**2)
            #self.Lfinal[i] = L[0].value
            #self.L0[i] = L[nsteps[i]].value

            #print([i,self.minr[i],v[np.argmin(np.array(r.value))],self.minL[i],self.lnlike[i],self.Lfinal[i]])
            print([i,self.minr[i],self.minL[i],self.lnlike[i],self.pmra[i].value,self.pmdec[i].value,self.dist[i].value])

  #          datalist=[ts,x,y,z,r,vx,vy,vz,v,L,[self.lnlike[i]]*len(y)]
            #datalist=[ts,x,y,z,r,vx,vy,vz,v,L,like,theta,phi]
            #namelist = ['t','x','y','z','d','vx','vy','vz','v','L','like','theta','phi']
            #data_table = Table(data=datalist,names=namelist)

            datalist=[ts,x,y,z,r,v]
            namelist = ['t','x','y','z','d','v']
            data_table = Table(data=datalist,names=namelist)
            data_table.write('/data1/Backflights/HVS867/testtflightthree_onestep.fits', overwrite=True)

            #data_table.write('/data1/Backflights/HVS867/MockHVS867_Backflight_TubeCenter_'+str(self.pmra[i].value)+'_0.001Myr.fits', overwrite=True)
            #data_table.write('/data1/Backflights/HVS529/MockHVS529_Backflight_CloseFiducial_'+str(i)+'_0.01Myr.fits', overwrite=True)
            #data_table.write('/data1/Backflights/HVS867/testtflight_Fid_0.001Myr.fits', overwrite=True)
            #data_table.write('/data1/Backflights/HVS867/REALTest_Zoom_NoRedo.fits', overwrite=True)
        #print(self.vfinal)

        #datalist=[self.xfinal, self.yfinal, self.zfinal,self.dfinal,self.Lfinal,self.vfinal,self.m,self.tflight]
        #print(type(self.vfinal))
        #data_table = Table(data=datalist,names=namelist)
        #data_table.write('/data1/test.fits', overwrite=True)

            #namelist = ['t', 'x','y','z','d', 'L', 'v']
            #datalist=[ts,x,y,z,r,L,v]
            #data_table = Table(data=datalist, names=namelist)
            #data_table.write('/data1/Backflight13.fits', overwrite=True)

        #np.savetxt('/data1/Distance_Final.txt',np.asscalar(self.dfinal))
        if(individual):
            return np.array([self.minr,self.minL,self.lnlike])
            #print(self.minr)
            #print(self.minL)
            #print(np.array(self.minr)*np.array(self.minL))
            #return np.array(self.minL)
            #return np.array(self.minr)
            #return np.array(self.lnlike)

        #return self.lnlike.sum()

    #@save
    def save(self, path):
        '''
            Saves the sample in a FITS file to be grabbed later

            Parameters
            ----------
                path : str
                    Path to the ourput fits file
        '''
        from astropy.table import Table

        meta_var = {'name' : self.name, 'ejmodel' : self.ejmodel_name, 'cattype' : self.cattype, \
                    'dt' : self.dt.to('Myr').value}
        if(self.cattype == 0):
            # Ejection catalog
            datalist = [self.r0, self.phi0, self.theta0, self.v0, self.phiv0, self.thetav0, \
                        self.m, self.tage, self.tflight]
            namelist = ['r0', 'phi0', 'theta0', 'v0', 'phiv0', 'thetav0', 'm', 'tage', 'tflight']

        if(self.cattype == 1):
            # Propagated catalog
            datalist = [self.r0, self.phi0, self.theta0, self.v0, self.phiv0, self.thetav0, \
                        self.m, self.tage, self.tflight, self.ra, self.dec, self.pmra, self.pmdec, \
                        self.dist, self.vlos, self.GCdist, self.GCv, self.thetaf, self.phif]
            namelist = ['r0', 'phi0', 'theta0', 'v0', 'phiv0', 'thetav0', 'm', 'tage', 'tflight', 'ra', \
                        'dec', 'pmra', 'pmdec', 'dist', 'vlos', 'GCdist', 'GCv','theta','phi']

        if(self.cattype == 2):
            # Sample cut with a given G_RVS and total velocity cut
            datalist = [self.r0, self.phi0, self.theta0, self.v0, self.phiv0, self.thetav0, \
                        self.m, self.tage, self.tflight, self.ra, self.dec, self.pmra, self.pmdec, \
                        self.dist, self.vlos, self.GRVS, self.V, self.G, self.e_par, self.e_pmra, self.e_pmdec, self.GCdist, self.GCv]
            namelist = ['r0', 'phi0', 'theta0', 'v0', 'phiv0', 'thetav0', 'm', 'tage', 'tflight', 'ra', \
                        'dec', 'pmra', 'pmdec', 'dist', 'vlos', 'GRVS', 'V', 'G', 'e_par', 'e_pmra', 'e_pmdec', 'GCdist', 'GCv']

        data_table = Table(data=datalist, names=namelist, meta=meta_var)
        data_table.write(path, overwrite=True)

    #@load
    def _load(self, path):
        '''
            Loads a HVS sample from a fits table
        '''
        from astropy.table import Table

        namelist = ['r0', 'phi0', 'theta0', 'v0', 'phiv0', 'thetav0', 'm', 'tage', 'tflight', 'ra', 'dec', 'pmra',
                    'pmdec', 'dist', 'vlos', 'GRVS', 'V', 'G', 'par', 'e_par', 'e_pmra', 'e_pmdec', 'GCdist', 'GCv']

   
        default_units = {'r0': u.pc, 'phi0': u.rad, 'theta0':u.rad, 'v0':u.km/u.s, 'phiv0':u.rad, 'thetav0':u.rad, 'm':u.solMass, 'tage':u.Myr,'tflight':u.Myr,'ra':u.deg,'dec':u.deg,'pmra':u.mas/u.yr, 'pmdec':u.mas/u.yr,'dist':u.kpc,'vlos':u.km/u.s,'GRVS':None,'V':None,'G':None,'par':1e-6*u.arcsec,'e_par':1e-6*u.arcsec,'e_pmra':u.mas/u.yr,'e_pmdec':u.mas/u.yr,'GCdist':u.kpc,'GCv':u.km/u.s}

        data_table = Table.read(path)

        #METADATA
        data_table.meta =  {k.lower(): v for k, v in data_table.meta.items()}
        #self.name = 'Unkown'
        self.ejmodel_name = 'Unknown'
        self.dt = 0*u.Myr

        uflag = False

        if ('name' in data_table.meta):
            self.name = data_table.meta['name']

        if('ejmodel' in data_table.meta):
            self.ejmodel_name = data_table.meta['ejmodel']

        if('dt' in data_table.meta):
            self.dt = data_table.meta['dt']*u.Myr

        if('cattype' not in data_table.meta):
            #raise ValueError('Loaded fits table must contain the cattype metavariable!')
            print('Sample assumed to be already propagated. If this is an ejection sample, set a metavariable called "cattype" to 0 in the FITS header.')
            #return False
        else:
            self.cattype = data_table.meta['cattype']

        self.size = len(data_table)

        #DATA
        i=0
        for colname in data_table.colnames:
            try:
                i = namelist.index(colname)
                if(data_table[colname].unit==None and default_units[colname] is not None):
                    print(colname)
                    setattr(self, colname, data_table[colname].quantity*default_units[colname])       
                    uflag = True
                else:
                    setattr(self, colname, data_table[colname].quantity)

                i+=1
            except ValueError:
                print('Column not recognized: ' + str(colname))
                i+=1
                continue

        if(uflag):
            print('One or more units were not specified - set to default values')

    #@loadExt
    #Ignore this for now too, not using real data yet
    def _loadExt(self, path, ejmodel='Contigiani2018',dt=0.01*u.Myr):
        from astropy.coordinates import SkyCoord
        '''
            Loads an external HVS sample from external source (e.g., from literature)

        Parameters
        ----------
            path: str
                Path to catalog
            ejmodel = str
                Suspected ejection model generating the sample. Not sure if this would do anything right now if only the likelihood method is being used
            dt = float
                Timestep to be used for the back-propagation

            See self.likelihood() for other parameters

        '''

        from astropy.table import Table

        #namelist = ['r0', 'phi0', 'theta0', 'v0', 'phiv0', 'thetav0', 'm', 'tage', 'tflight', 'ra', 'dec', 'pmra',
        #            'pmdec', 'dist', 'vlos', 'GRVS', 'V', 'G', 'e_par', 'e_pmra', 'e_pmdec', 'GCdist', 'GCv']

        data_table = Table.read(path)

        #data_table['pmra_Gaia'][1] = -0.175
        #data_table['pmdec_Gaia'][1] = -0.719
        #data_table['err_pmra_Gaia'][1] = 0.316
        #data_table['err_pmdec_Gaia'][1] = 0.287

        #Manually set variables that would normally be in metadata
        self.ejmodel_name = ejmodel
        self.dt = dt
        self.cattype = 2
        self.size = len(data_table)

        setattr(self,'m',data_table['M']*u.solMass)
        setattr(self,'pmra',data_table['pmra_Gaia']*u.mas/u.yr)
        setattr(self,'pmdec',data_table['pmdec_Gaia']*u.mas/u.yr)
        setattr(self,'vlos',data_table['vrad']*u.km/u.second)
        setattr(self,'dist',data_table['d']*u.kpc)
        setattr(self,'tage',data_table['tage']*u.Myr)
        setattr(self,'ID',data_table['ID'])
        setattr(self,'e_pmra',data_table['err_pmra_Gaia']*u.mas/u.yr)
        setattr(self,'e_pmdec',data_table['err_pmdec_Gaia']*u.mas/u.yr)
        setattr(self,'e_dist',data_table['d_errhi']*u.kpc)
        setattr(self,'e_vlos',data_table['vrad_errhi']*u.km/u.second)
        setattr(self,'ra',data_table['ra']*u.degree)
        setattr(self,'dec',data_table['dec']*u.degree)


        #self.pmra[1] = -0.175*u.mas/u.yr
        #setattr(self,'pmra[1]',-0.175*u.mas/u.yr)
        #setattr(self,'pmdec[1]',-0.719*u.mas/u.yr)
        #setattr(self,'err_pmra[1]',0.316*u.mas/u.yr)
        #setattr(self,'err_pmdec[1]',0.287*u.mas/u.yr)

        #Read in ra, dec in hhmmss.ss/DDmmss.ss, convert to degrees
        #ratmp = data_table['RA']
        #dectmp = data_table['Dec']
        #c = SkyCoord(ratmp,dectmp)
        #setattr(self,'ra',c.ra.value*u.degree)
        #setattr(self,'dec',c.dec.value*u.degree)

        #l = c.galactic.l.value*np.pi/180.0
        #b = c.galactic.b.value*np.pi/180.0

        #p1 = (np.cos(b)**2)*(np.cos(l)**2) + (np.cos(b)**2)*(np.sin(l)**2) + np.sin(b)**2
        #p2 = -16.0*np.cos(b)*np.cos(l)
        #p3 = 64 - data_table['RGC']**2

        #dist = np.zeros(self.size)

        #for i in range(self.size):
        #    dist[i] = max(np.roots([p1[i],p2[i],p3[i]]))

        #DATA
        #i=0
        #for colname in data_table.colnames:
        #    try:
        #        i = namelist.index(colname)
        #        setattr(self, colname, data_table[colname].quantity)
        #        i+=1
        #    except ValueError:
        #        print('Column not recognized: ' + str(colname))
        #        i+=1
        #        continue

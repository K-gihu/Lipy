# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 09:37:37 2018

@author: FAREWELL
"""

import numpy as np
import LightPipes as LP
import special_beam
import Turbulence_Lib
import beam_process

import Chinese_Lib
Chinese_Lib.set_ch()

# =============================================================================
# vortex beam experiment class.
# =============================================================================
class vortex_experiment(special_beam.special_beam,Turbulence_Lib.Turbulence_Lib,beam_process.beam_process):
    
    """ 
    This class is a derivative of the vortex_beam class
    """
    
    def fork_function(self, gamma, n, r, phi, m=1, fork_type='y'):
    
        Cn = np.sin(n*np.pi*0.25)/(n*np.pi)
        if fork_type == 'x':
            Tn = Cn*np.exp(1j*n*m*phi)*np.exp(-1j*n*gamma*r*np.sin(phi))
        elif fork_type == 'y':
            Tn = Cn*np.exp(1j*n*m*phi)*np.exp(-1j*n*gamma*r*np.cos(phi))
        
        return Tn
    
    def fork_grating(self, r, phi, d=140*LP.um, upper_lim=1, m=1, Cn=1, fork_type='y'):
        
        """
        Generate a fork grating.
        Parameters:
            r,phi: polar coordinate.
            d: grating period.
        Returns:
            
        """
        
        (N,N) = r.shape
        gamma = 2*np.pi/d
        T = np.zeros((N,N),dtype='complex')
        for i in range(1,upper_lim+1):
            T += self.fork_function(gamma,i,r,phi,m,fork_type)
        for i in range(1,upper_lim+1):
            T += self.fork_function(gamma,-i,r,phi,m,fork_type)
        
        C0 = 0.25
        T += C0
        return T
    
    def Aperture(self, Amp, size):
        
        """
        Add an aperture to the beam (through multiplying a circ function).
        Parameters:
            
            Amp: beam complex amplitude.
            size: relative size of aperture (0~1, 0 represents no light will pass).
            
        Returns:
            ndarray: 2-D complex amplitude.
        """
        
        (N,N) = Amp.shape
        for i in range(N):
            for j in range(N):
                if (i-256)**2+(j-256)**2 > (N*size)**2:
                    Amp[i,j] = 0
                else:
                    pass
        
        return Amp
    
    def Circ_obstacle(self, Amp, size):
        
        """
        Add an circular obstacle right in the center of beam.
        Parameters:
            
            Amp: beam complex amplitude.
            size: relative size of obstacle (0~1, 0 represents no light will pass).
        """
        (N,N) = Amp.shape
        for i in range(N):
            for j in range(N):
                if (i-256)**2+(j-256)**2 < (N*size)**2:
                    Amp[i,j] = 0
                else:
                    pass
        
        return Amp
    
# In[]: Generating Methods.
     
    def Fork_Grating():
        
        return
    
    def Phase_Hologram(self, TC=1, z=50*LP.m, plot=True):
        
        x = np.linspace(-self.WindowSize/2, self.WindowSize/2, self.N)
        y = np.linspace(-self.WindowSize/2, self.WindowSize/2, self.N)
        x,y= np.meshgrid(x, y)
        phi = np.arctan2(y, x)
        Phase = np.exp(1j*TC*phi) 
        Phase = self.Aperture(Phase, size=0.5)      # add an aperture to the phase hologram.
        
        Amp = self.Gaussian_beam()
        Amp *= Phase                               # transmit the light beam through the phase hologram.
        
        Amp = self.Free_space_propagation(Amp, self.wvl, self.WindowSize/self.N, self.WindowSize/self.N, z)
        if plot:
            self.show_intensity_image(Amp)
            return True
        else:
            return Amp 
    
    def Optical_Fiber():
        
        return
    
# In[]: Measurement Methods.
    
    def Spherical_Wave(self,z=500*LP.m):
        
        x = np.linspace(-self.WindowSize/2,self.WindowSize/2,self.N)
        y = np.linspace(-self.WindowSize/2,self.WindowSize/2,self.N)
        x,y= np.meshgrid(x,y)
        r = np.sqrt(x**2+y**2)
        wvn = 2*np.pi/self.wvl
        Amp = np.exp(-1j*wvn*z*(1+1/2*r**2/z**2))  
        
        return Amp
        
    def Spherical_Interferometry(self, z=500*LP.m, TC=3, plot=True):
        
        x = np.linspace(-self.WindowSize/2,self.WindowSize/2,self.N)
        y = np.linspace(-self.WindowSize/2,self.WindowSize/2,self.N)
        x,y= np.meshgrid(x,y)
        phi = np.arctan2(y,x)
        Amp = np.exp(1j*TC*phi)     # ideal vortex beam amplitude
        Amp_spherical = self.Spherical_Wave(z)
        Amp += Amp_spherical
        if plot:
            self.show_intensity_image(Amp)
            return True
        else:
            return Amp 
    
    def Planar_Wave(self,direction=True):
        
        x = np.linspace(-self.WindowSize/2,self.WindowSize/2,self.N)
        y = np.linspace(-self.WindowSize/2,self.WindowSize/2,self.N)
        x,y= np.meshgrid(x,y)
        if direction:    # Horizontal
            Amp_planar = np.exp(1j*2*np.pi*x/self.wvl)
        else:            # Vertical
            Amp_planar = np.exp(1j*2*np.pi*y/self.wvl)
        
        return Amp_planar
    
    def Planar_Interferometry(self, TC=3, plot=True, direction=True):
        
        x = np.linspace(-self.WindowSize/2,self.WindowSize/2,self.N)
        y = np.linspace(-self.WindowSize/2,self.WindowSize/2,self.N)
        x,y= np.meshgrid(x,y)
        phi = np.arctan2(y,x)
        Amp = np.exp(1j*TC*phi)     # vortex beam
        Amp_planar = self.Planar_Wave(direction)
        Amp += Amp_planar
        if plot:
            self.show_intensity_image(Amp)
            return True
        else:
            return Amp 
    
    def Double_slit_Interferometry():
        
        return
    
    def Cylindrical_Lens(self,Amp,f=0.1*LP.m):
        
        wvn = 2*np.pi/self.wvl
        x = np.linspace(-self.WindowSize/2,self.WindowSize/2,self.N)
        y = np.linspace(-self.WindowSize/2,self.WindowSize/2,self.N)
        x,y = np.meshgrid(x,y)
        
        Amp *= np.exp(-1j*wvn*y**2/2/f)
        
        return Amp
    
    def Optical_Lens(self,Amp,f=0.1*LP.m):
        
        wvn = 2*np.pi/self.wvl
        x = np.linspace(-self.WindowSize/2,self.WindowSize/2,self.N)
        y = np.linspace(-self.WindowSize/2,self.WindowSize/2,self.N)
        x,y = np.meshgrid(x,y)
        
        Amp *= np.exp(-1j*wvn/2/f*(x**2+y**2))
        
        return Amp
# In[]: Multiplex and de-Multiplex Methods.
        
    def Multiplex():
        
        return
    
    def Dammann_Grating():
        
        return
    
    def de_Multiplex():
        
        return
    
    
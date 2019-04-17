# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 15:45:02 2017

@author: Farewell
"""

import numpy as np
from matplotlib import pyplot as plt
import LightPipes as LP 
import time
import basic_math
import basic_tools
import random
from scipy import special

class Turbulence_Lib(basic_tools.basic_tools,basic_math.basic_math):
    
    """
    Free space and turbulence propagation realtive functions.
    """
    # Don't define __init__() function here, python will automatically call the initialization-function 
    # of superclass when constructing object.
    
# In[]:Interconversion for Cn2 and r0；
        
    def Cn2_r0(self,Cn2=1e-14,deltz=50*LP.m):
        
        """
        convert Cn2 to r0.
        Cn2: refractive index structure constant(1e-14~1e-17).
        deltz: the length of transmission path.
        """
        
        wvn = 2*np.pi/self.wvl
        r0 = (1.46*wvn**2*Cn2*deltz)**(-3/5)*2.1
        return r0
  
    def r0_Cn2(self,r0=0.16,deltz=50*LP.m):
        
        """
        convert r0 to Cn2.
        r0: atmospheric coherent length.
        deltz: the length of transmission path.
        """
        
        wvn = 2*np.pi/self.wvl
        Cn2 = 1/(1.46*wvn**2*deltz)*(r0/2.1)**(-5/3)
        return Cn2
    
    def Free_space_propagation(self, Amp, wvl, inputSpacing, outputSpacing, distance=100*LP.m):
        
        """
        Dealing with beam propagation in vacuum(free space)
        
        Amp: Complex amplitude
        wvl: Beam wavelength.
        inputspacing: Grid size of input plane.
        outputspacing: Grid size of output plane.
        distance: propagation distance.
        
        Returns: Complex Amplitude in the outputplane.
        """
        
#        if outputSpacing != None:
        Amp = self.angularSpectrum(Amp, wvl, inputSpacing, outputSpacing, distance)
        return Amp
        
#        else:
#            
#            Extension_Ratio = distance/3000+1
#            outputSpacing = inputSpacing*Extension_Ratio
#            
#            Amp = self.angularSpectrum(Amp, wvl, inputSpacing, outputSpacing,distance)
#            return Amp,Extension_Ratio        
        
    def Generate_screen(self, Cn2=1e-16, N=512, delta=0.001, L0=5*LP.m, l0=1e-2, von_karman=False,
                        alpha=11./3, im=True, plot=False):
        
        """
        Calculate a simple phase screen generator,or just plot 
        a simple gray-scale image of phase screen.  
        """
        
        r0 = self.Cn2_r0(Cn2)
        screen = self.ft_sh_phase_screen(r0, N, delta, L0, l0, alpha, von_karman)
        
        if plot:
            
            fig = plt.figure()
            if im:
                im = plt.imshow(next(screen),cmap='gray',interpolation='bicubic')
                fig.colorbar(im)
            else: 
                pl = plt.pcolormesh(next(screen))
                plt.xlim(0,N/2)
                plt.ylim(0,N/2)
                fig.colorbar(pl)
                
            return True
        else:
            return screen
    
    def ft_sh_phase_screen(self, r0, N, delta, L0, l0, alpha, von_karman=False, FFT=None, seed=None):
        
        '''
        Creates a random phase screen with Von Karman or Non-Kolmogorov statistics 
        with added sub-harmonics to augment tip-tilt modes.
        (Schmidt 2010)
        
        Args:
            r0 (float): r0 parameter of scrn in metres
            N (int): Size of phase scrn in pxls
            delta (float): size in Metres of each pxl
            L0 (float): Size of outer-scale in metres
            l0 (float): inner scale in metres
    
        Returns:
            ndarray: numpy array representing phase screen
        '''
        
        R = random.SystemRandom(time.time())
        if seed is None:
            seed = int(R.random()*100000)
        np.random.seed(seed)
    
        D = N*delta
        # spatial grid [m]
        coords = np.arange(-N/2,N/2)*delta
        x, y = np.meshgrid(coords,coords)
        # high-frequency screen from FFT method
        phs_hi = self.ft_phase_screen(r0, N, delta, L0, l0, alpha, von_karman, FFT, seed=seed)
        # initialize low-freq screen
        phs_lo = np.zeros([N,N])
        # loop over frequency grids with spacing 1/(3^p*L)
        while 1:
            for p in range(1,4):
                # setup the PSD
                
                del_f = 1 / (3**p*D) #frequency grid spacing [1/m]
                fx = np.arange(-1,2) * del_f
        
                # frequency grid [1/m]
                fx, fy = np.meshgrid(fx,fx)
                
                # modified von Karman atmospheric phase PSD
                
                if von_karman:
                    
                    PSD_phi  = self.von_karman_PowerSpectrum(r0, N, del_f, L0, l0)
                    
                else:
                    
                    PSD_phi = self.non_Kolmogorov_PowerSpectrum(r0, N, del_f, L0, l0, alpha)
                                    
                PSD_phi[N//2+1, N//2+1] = 0
                # random draws of Fourier coefficients
                cn = ((np.random.normal(size=(N,N)) + 1j*np.random.normal(size=(N,N)) )
                                * np.sqrt(PSD_phi[p-1]) * del_f)
                SH = np.zeros((N,N),dtype="complex")
            
                # loop over frequencies on this grid
                for i in range(0,2):
                    for j in range(0,2):
        
                        SH += cn[i,j] * np.exp(1j*2*np.pi*(fx[i,j]*x+fy[i,j]*y))        
    
            phs_lo = phs_lo + SH
            
            # accumulate subharmonics
            phs_lo = phs_lo.real - phs_lo.real.mean()
            phs = phs_lo + next(phs_hi)    
        
            yield phs
    
    def ft_phase_screen(self,r0, N, delta, L0, l0, alpha=11./3, von_karman=False, FFT=None, seed=None):
        
        '''
        Creates a random phase screen with Von Karman statistics.
        (Modified : non-Kolmogorov statistics)
        
        (Schmidt 2010)
        
        Parameters:
            r0 (float): r0 parameter of scrn in metres
            N (int): Size of phase scrn in pxls
            delta (float): size in Metres of each pxl
            L0 (float): Size of outer-scale in metres
            l0 (float): inner scale in metres
            alpha (float,3~4): general index
            
        Returns:
            ndarray: numpy array representing phase screen
        '''
        
        delta = float(delta)
        r0 = float(r0)
        L0 = float(L0)
        l0 = float(l0)
    
        R = random.SystemRandom(time.time())
        if seed is None:
            seed = int(R.random()*100000)
        np.random.seed(seed)
        del_f = 1./(N*delta)
        if von_karman:
            
            PSD_phi  = self.von_karman_PowerSpectrum(r0, N, del_f, L0, l0)
            
        else:
            
            PSD_phi = self.non_Kolmogorov_PowerSpectrum(r0, N, del_f, L0, l0, alpha)
    
        PSD_phi[N//2+1, N//2+1] = 0
        
        while 1:
            
            cn = ( (np.random.normal(size=(N,N)) + 1j* np.random.normal(size=(N,N)) )
                        * np.sqrt(PSD_phi)*del_f )
    
            phs = self.ift2(cn,1, FFT).real
    
            yield phs
            
    def ift2(self, G, delta_f ,FFT=None):
        
        """
        Wrapper for inverse fourier transform
    
        Parameters:
            G: data to transform
            delta_f: pixel seperation
            FFT (FFT object, optional): An accelerated FFT object
        """
            
        N = G.shape[0]
    
        if FFT:
            g = np.fft.fftshift( FFT( np.fft.fftshift(G) ) ) * (N * delta_f)**2
        else:
            g = np.fft.ifftshift( np.fft.ifft2( np.fft.fftshift(G) ) ) * (N * delta_f)**2
    
        return g
    
    def von_karman_PowerSpectrum(self, r0, N, del_f, L0, l0):
        
        """
        Calculate a von-Karman power spectrum matrix.
        
        r0 : atmospheric coherent length.
        N: Sample points.
        del_f: Sample interval.
        L0: turbulence outer scale.
        l0: turbulence inner scale.
        
        Returns: 
            ndarray: atmospheric turbulence power spectrum(von-Karman)
        """
	        	
        fx = np.arange(-N/2.,N/2.) * del_f
        (fx,fy) = np.meshgrid(fx,fx)
        f = np.sqrt(fx**2 + fy**2)           # spatial frquency Kappa
        	
        fm = 5.92/l0/(2*np.pi) 
        f0 = 1./L0
        	
        PSD_phi  = (0.023*r0**(-5./3.) * np.exp(-1*((f/fm)**2)) /
                    ( ( (f**2) + (f0**2) )**(11./6) ) )
    					
        return PSD_phi

    def non_Kolmogorov_PowerSpectrum(self, r0, N, del_f, L0, l0, alpha):
        
        """
        Get a non Kolmogorov power spectrum of atmospheric turbulence
        
        r0 : atmospheric coherent length
        N: Sample points.
        del_f: Sample interval.
        L0: turbulence outer scale.
        l0: turbulence inner scale.
        alpha: generalized index.
        
        Returns: 
            ndarray: atmospheric turbulence power spectrum(non-Kolmogorov)
        """
                
        fx = np.arange(-N/2.,N/2.) * del_f
        (fx,fy) = np.meshgrid(fx,fx)    
        f = np.sqrt(fx**2 + fy**2)           # spatial frquency Kappa
        A_alpha = special.gamma(alpha-1)*np.cos(alpha*np.pi/2)/(4*np.pi**2)
        C_alpha = (special.gamma((5-alpha)/2)*A_alpha*2*np.pi/3)**(1/(alpha-5))
        fm = C_alpha/l0/(2*np.pi)       
        f0 = 1./L0
        PSD_phi = (A_alpha * 0.033/0.023 *r0**(-5./3.) * np.exp(-1*((f/fm)**2)) /
                        ( ( (f**2) + (f0**2) )**(alpha/2) ) )
        
        return PSD_phi
    
    def Oceanic_Power_Spectrum(self, N, del_f, epsilon=1e-5, Kolmogorov_Scale=1e-3, w=4, Tvdr=1e-7):
        
        """
        Get a Spatial power spectrum of oceanic turbulence
        
        N: Sample points.
        del_f: Sample interval.
        epsilon: Turbulence kinetic energy dissipation rate, unit:m^2/s^3.
        Kolmogorov_Scale: Kolmogorov Scale.
        w: Ratio of power spectrum contribution of temperature and salinity.
        Tvdr: Temperature variance dissipation rate.
        
        Returns: 
            Oceanic power spectrum.
            (Reference:《拉盖尔-高斯波束在弱湍流海洋中轨道角动量传输特性变化》)
        """
        
        # some constants.
        A_T = 1.863e-2
        A_S = 1.9e-4
        A_Ts = 9.41e-3
        
        fx = np.arange(-N/2.,N/2.) * del_f
        (fx,fy) = np.meshgrid(fx,fx)    
        f = np.sqrt(fx**2 + fy**2)           # spatial frquency Kappa
        
        delta = 8.284*(f*Kolmogorov_Scale)**(4/3)+12.978*(f*Kolmogorov_Scale)**2
        
        PSD_phi = (0.388e-8 * epsilon**(-1/3) * f**(-11/3)*Tvdr/w**2 * (1+2.35*(f*Kolmogorov_Scale)**(2/3))
            *(w**2*np.exp(-A_T*delta) + np.exp(-A_S*delta) - 2*w*np.exp(-A_Ts*delta)))
        
        return PSD_phi
        
    def angularSpectrum(self, inputComplexAmp, wvl, inputSpacing, outputSpacing, z):
        
        """
        Propogates light complex amplitude using an angular spectrum algorithm
    
        Parameters:
            inputComplexAmp (ndarray): Complex array of input complex amplitude
            wvl (float): Wavelength of light to propagate
            inputSpacing (float): The spacing between points on the input array in metres
            outputSpacing (float): The desired spacing between points on the output array in metres
            z (float): Distance to propagate in metres
    
        Returns:
            ndarray: propagated complex amplitude
        """
        
        # If propagation distance is 0, don't bother 
        if z==0:
            return inputComplexAmp
    
        N = inputComplexAmp.shape[0] #Assumes Uin is square.
        k = 2*np.pi/wvl     #optical wavevector
    
        (x1,y1) = np.meshgrid(inputSpacing*np.arange(-N/2,N/2),
                                 inputSpacing*np.arange(-N/2,N/2))
        r1sq = (x1**2 + y1**2) + 1e-10
    
        #Spatial Frequencies (of source plane)
        df1 = 1. / (N*inputSpacing)
        fX,fY = np.meshgrid(df1*np.arange(-N/2,N/2),
                               df1*np.arange(-N/2,N/2))
        fsq = fX**2 + fY**2
    
        #Scaling Param
        mag = float(outputSpacing)/inputSpacing
    
        #Observation Plane Co-ords
        x2,y2 = np.meshgrid( outputSpacing*np.arange(-N/2,N/2),
                                outputSpacing*np.arange(-N/2,N/2) )
        r2sq = x2**2 + y2**2
    
        #Quadratic phase factors
        Q1 = np.exp( 1j * k/2. * (1-mag)/z * r1sq)
    
        Q2 = np.exp(-1j * np.pi**2 * 2 * z/mag/k*fsq)
    
        Q3 = np.exp(1j * k/2. * (mag-1)/(mag*z) * r2sq)
    
        #Compute propagated field
        outputComplexAmp = Q3 * self.ift2(
                        Q2 * self.ft2(Q1 * inputComplexAmp/mag,inputSpacing), df1)
        return outputComplexAmp
    
    
    def oneStepFresnel(self, Uin, wvl, d1, z):
        """
        Fresnel propagation using a one step Fresnel propagation method.
    
        Parameters:
            Uin (ndarray): A 2-d, complex, input array of complex amplitude
            wvl (float): Wavelength of propagated light in metres
            d1 (float): spacing of input plane
            z (float): metres to propagate along optical axis
    
        Returns:
            ndarray: Complex ampltitude after propagation
        """
        N = Uin.shape[0]    #Assume square grid
        k = 2*np.pi/wvl     #optical wavevector
     
        #Source plane coordinates
        x1,y1 = np.meshgrid( np.arange(-N/2.,N/2.) * d1,
                                np.arange(-N/2.,N/2.) * d1)
        #observation plane coordinates
        d2 = wvl*z/(N*d1)
        x2,y2 = np.meshgrid( np.arange(-N/2.,N/2.) * d2,
                                np.arange(-N/2.,N/2.) * d2 )
    
        #evaluate Fresnel-Kirchoff integral
        A = 1/(1j*wvl*z)
        B = np.exp( 1j * k/(2*z) * (x2**2 + y2**2))
        C = basic_math.ft2(Uin *np.exp(1j * k/(2*z) * (x1**2+y1**2)), d1)
    
        Uout = A*B*C
    
        return Uout
    
    def twoStepFresnel(self, Uin, wvl, d1, d2, z):
        """
        Fresnel propagation using a two step Fresnel propagation method.
    
        Parameters:
            Uin (ndarray): A 2-d, complex, input array of complex amplitude
            wvl (float): Wavelength of propagated light in metres
            d1 (float): spacing of input plane
            d2 (float): desired output array spacing
            z (float): metres to propagate along optical axis
    
        Returns:
            ndarray: Complex ampltitude after propagation
        """
    
        N = Uin.shape[0] #Number of grid points
        k = 2*np.pi/wvl #optical wavevector
    
        #source plane coordinates
        x1, y1 = np.meshgrid( np.arange(-N/2,N/2) * d1,
                                np.arange(-N/2.,N/2.) * d1 )
    
        #magnification
        m = float(d2)/d1
    
        #intermediate plane
        try:
            Dz1  = z / (1-m) #propagation distance
        except ZeroDivisionError:
            Dz1 = z / (1+m)
        d1a = wvl * abs(Dz1) / (N*d1) #coordinates
        x1a, y1a = np.meshgrid( np.arange( -N/2.,N/2.) * d1a,
                                  np.arange( -N/2.,N/2.) * d1a )
    
        #Evaluate Fresnel-Kirchhoff integral
        A = 1./(1j * wvl * Dz1)
        B = np.exp(1j * k/(2*Dz1) * (x1a**2 + y1a**2) )
        C = basic_math.ft2(Uin * np.exp(1j * k/(2*Dz1) * (x1**2 + y1**2)), d1)
        Uitm = A*B*C
        #Observation plane
        Dz2 = z - Dz1
    
        #coordinates
        x2,y2 = np.meshgrid( np.arange(-N/2., N/2.) * d2,
                                np.arange(-N/2., N/2.) * d2 )
    
        #Evaluate the Fresnel diffraction integral
        A = 1. / (1j * wvl * Dz2)
        B = np.exp( 1j * k/(2 * Dz2) * (x2**2 + y2**2) )
        C = basic_math.ft2(Uitm * np.exp( 1j * k/(2*Dz2) * (x1a**2 + y1a**2)), d1a)
        Uout = A*B*C
    
        return Uout
    
    def lensAgainst(self, Uin, wvl, d1, f):
        '''
        Propagates from the pupil plane to the focal
        plane for an object placed against (and just before)
        a lens.
    
        Parameters:
            Uin (ndarray): Input complex amplitude
            wvl (float): Wavelength of light in metres
            d1 (float): spacing of input plane
            f (float): Focal length of lens
    
        Returns:
            ndarray: Output complex amplitude
        '''
    
        N = Uin.shape[0] #Assume square grid
        k = 2*np.pi/wvl  #Optical Wavevector
    
        #Observation plane coordinates
        fX = np.arange( -N/2.,N/2.)/(N*d1)
    
        #Observation plane coordinates
        x2,y2 = np.meshgrid(wvl * f * fX, wvl * f * fX)
        del(fX)
    
        #Evaluate the Fresnel-Kirchoff integral but with the quadratic
        #phase factor inside cancelled by the phase of the lens
        Uout = np.exp( 1j*k/(2*f) * (x2**2 + y2**2) )/ (1j*wvl*f) * basic_math.ft2( Uin, d1)
    
        return Uout
    
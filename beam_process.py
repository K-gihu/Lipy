# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 08:59:59 2018

@author: Farewell
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import special
import LightPipes as LP
from scipy import integrate
import basic_tools
import basic_math
import math
from math import factorial as Fac
import os 
import special_beam
import Turbulence_Lib 
from scipy.special import comb,gamma,gammainc

class beam_process(basic_tools.basic_tools,basic_math.basic_math):
    
    # Don't define __init__() function here, python will automatically call the initialization-function 
    # of superclass when constructing object.
    
# In[]:
    def mode_decomposition(self,Amp,TC=range(-5,5)):
        
        """
        Decomposite any beam into sum of spiral spectrum in polar coordinate
		(usually used with show_oam_spectram() function);
        Parameters:
            Amp: the amplitude of light field to be processed(2-D complex ndarray).
            TC: decomposite range, default from -5 to 5.
        Returns:
            ndarray: the amplitude of sprial spectrum.
        """
        
        Pl_total = np.zeros((len(TC)),dtype='float')
        r = np.linspace(0,np.sqrt(2)*self.WindowSize,self.N)
        phi = np.linspace(0,2*np.pi,self.N)
        r,phi = np.meshgrid(r,phi)
        
        for m in TC:
            res = np.trapz(Amp*np.exp(-1j*m*phi),axis=0)/np.sqrt(2.0*np.pi)
            Pl = (np.trapz(res*np.conjugate(res)*r,axis=-1)).real
            Pl_total[m] = Pl[0]
            
        return Pl_total

## In[]: 此函数对离轴涡旋光束进行螺旋谱分解操作；
#        
#    def mode_decomposition_2(self, Amp, r, phi, phi0, TC):
#        
#        Pl_total = np.zeros((len(TC)),dtype='float')
#        
#        for m in TC:
#            res = np.trapz(Amp*np.exp(-1j*m*(phi-phi0)),axis=0)/np.sqrt(2.0*np.pi)
#            Pl = (np.trapz(res*np.conjugate(res)*r,axis=-1)).real
#            Pl_total[m] = Pl[0]
#        
#        return 
    
# =============================================================================
# mode decomposition in cartesian coordinate(Test failed)    
# =============================================================================
#    def mode_decomposition(self,Amp,TC = range(-5,5),Rho = 4,w0 = 0.003):
#    
#        Pl_total = np.zeros((len(TC)),dtype='float')
#        for i in TC:
#           
#            res = np.trapz(-Amp*np.exp(-1j*i*phi)*y/(x**2+y**2),axis=1)+np.trapz(Amp*np.exp(-1j*i*phi)*x/(x**2+y**2),axis=0)
#            res /= np.sqrt(2.0*np.pi)
#            Pl = (np.trapz(res*np.conjugate(res)*r,axis=-1)).real
#            Pl_total[i] = Pl[0]
#            
#        return Pl_total
    
    def show_oam_spectram(self, Pl_total, TC=range(-5,5), get_percent=False, title='OAM spectrum', save=False,
                          path=os.getcwd()):
        """
        Show the oam spectrum map with data calculated above.
        Parameters:
            Pl_total: spiral spectrum amplitude array.
            TC: decomposite range, default from -5 to 5.
            get_percent: default set as False, if set True function will return the normalized spiral spectrum.
            title: map title.
            save: default set as False, if set True the map will be saved in the folder given in path.
            path: save path, default set as current working directory.
                
        Returns:
            True if displayed successfully.
            The normalized spiral spectrum if 'get_percent' set True.
        """
        
        (N,) = Pl_total.shape
        Pl_total = abs(Pl_total/np.sum(Pl_total))
        
        # Adjust the relative position of the data  in Pl_total.
        
        for i in range(0,N//2):
            temp = Pl_total[i]
            Pl_total[i] = Pl_total[i+N//2]
            Pl_total[i+N//2] = temp
        
        if not get_percent:
            plt.bar(x=TC,height=Pl_total,width=0.5,align='center',yerr=0.000001)
            plt.xlim(np.min(TC),np.max(TC))
            plt.ylim(0,1.2)
            plt.xlabel("Topological Charge")
            plt.ylabel("Normalized amplitude")
            plt.title(title)
            if save:     # save image；
                plt.savefig(path+"\\"+title+'.png')
            else:
                pass
        else:
            return Pl_total
        
        return True
    
    def Pm_Statistics(self, Amp, TC=range(-5,5)):
        
        """
        Analyze the statistical properties of OAM spiral spetrum.
        Parameters:
            Amp: amplitude of light field.
            TC: decomposite range, default from -5 to 5.
        Returns:
            Mean value and variance of OAM spiral spectrum.
        """
        
        r = np.linspace(0,np.sqrt(2)*self.WindowSize,self.N)
        phi = np.linspace(0,2*np.pi,self.N)
        r,phi = np.meshgrid(r,phi)
        Pl_total = np.zeros((len(TC)),dtype='float')
        for m in TC:
            res = np.trapz(Amp*np.exp(-1j*m*phi),axis=0)/np.sqrt(2.0*np.pi)
            Pl = (np.trapz(res*np.conjugate(res)*r,axis=-1)).real
            Pl_total[m] = Pl[0]
            
        (M,) = Pl_total.shape
        Pl_total = abs(Pl_total/np.sum(Pl_total))  # Normalization.
        
        # Adjust the relative position of the data  in Pl_total.
        
        for i in range(0,M//2):
            temp = Pl_total[i]
            Pl_total[i] = Pl_total[i+M//2]
            Pl_total[i+M//2] = temp
        
        Mean = 0  # Mean value of Topological charge    
        for i in range(0,M):
            Mean += Pl_total[i]*TC[i]
        
        Var = 0  # Variance of Pm(topological charge detection probability)
        for i in range(0,M):
            Var += Pl_total[i]*TC[i]**2       
        Var -= Mean**2
        
        return Mean,Var
                
    def BER_Func(self, Pm):
        
        """
        Calculate bit error rate in OOK modulation.
        Parameters:
            Pm: OAM spiral spectrum.
        Returns:
            bit error rate.
        """
        
        SNR = Pm/(1-Pm)
        BER = special.erfc(SNR/np.sqrt(2))/2
        
        return BER
    
    def beam_radius(self, x, Amp, beam_type='vortex', Amp_Flag=True):
        
        """
        Calculate beam radius using numerical method.
        Parameters:
            x: coordinate axis.
            Amp: amplitude of light field.
            beam_type: 'vortex' or 'gauss', 'vortex' calculate the radius of the brightest ring(first order),
            'gauss' calculate the beam waist width.
                
        Returns:
            bit error rate.
        """
        
#        dx = x[[0],[1]]-x[[0],[0]]
#        
#        Intensity = (Amp*Amp.conjugate()).real
#        N,N = Amp.shape
#        
#        if beam_type == 'vortex':
#            
#            
#            m,n = matrix_Lib.getPositon(Intensity)
#            
#        elif beam_type == 'gauss':
#               
#            m,n = matrix_Lib.getPositon(Intensity,value=np.max(Intensity)/np.e**2)
#        
#        # cartesian coordinate only；
#        radius = np.sqrt(((m-N/2)*dx)**2+((n-N/2)*dx)**2)
#        
#        return radius
       
        dx = x[[0],[1]]-x[[0],[0]]
        
        if Amp_Flag:
            Intensity = (Amp*Amp.conjugate()).real
        else:
            Intensity = Amp
            
        N,N = Amp.shape
        
        if beam_type == 'vortex':
            
            radius = 0
            Max = np.max(Intensity)
            
            NumofDots = 0
            
            for i in range(N):
                for j in range(N):
                    if Intensity[i,j] > math.floor(Max*1e8)/1e8:
                        radius += np.sqrt(((i-N/2)*dx)**2+((j-N/2)*dx)**2)
                        NumofDots += 1
                                  
            radius = radius/NumofDots
            
        elif beam_type == 'gauss':
               
            m,n = self.getPositon(Intensity, value = np.max(Intensity)/np.e**2)
            # appropriate for cartesian coordinate only；
            radius = np.sqrt(((m-N/2)*dx)**2+((n-N/2)*dx)**2)
        
        return radius*2
    
    def Light_Spot_Centroid(self,Amp,x,y,Amp_flag=True):
    
        """
        Calculate the centroid of light spot. 
        Parameters:
            Amp: amplitude of light field.
            x,y: cartesian coordinate.
            Amp_flag: default set as True,if set false, deal Amp as intensity.
        Returns:
            centroid coordinate.
        """
        
        if Amp_flag:
            I = (Amp*np.conjugate(Amp)).real
        else:
            I = Amp
        dx = x[0,1]-x[0,0]
        Nominator_x = self.double_trapz(I*x,dx=dx,dy=dx)
        Nominator_y = self.double_trapz(I*y,dx=dx,dy=dx)
        Denominator = self.double_trapz(I,dx=dx,dy=dx)
    
        x_c = Nominator_x/Denominator
        y_c = Nominator_y/Denominator
        
        return x_c,y_c
    
    def Mean_square_beam_radius(self,Amp,x,y,Amp_flag=True):
    
        """
        Calculate the mean square beam width using numerical methods.  
        Parameters:
            Amp: amplitude of light field.
            x,y: cartesian coordinate.
            Amp_flag: if the parameter Amp is amplitude, set True, else if the parameter Amp is intensity, set False.
        Returns:
            Mean square beam width.
        """
        
        if Amp_flag:
            I = (Amp*np.conjugate(Amp)).real
        else:
            I = Amp
        dx = x[0,1]-x[0,0]
        x_c,y_c = self.Light_Spot_Centroid(Amp,x,y,Amp_flag)
        Nominator_x = self.double_trapz(I*(x-x_c)**2,dx=dx,dy=dx)
        Nominator_y = self.double_trapz(I*(y-y_c)**2,dx=dx,dy=dx)
        Denominator = self.double_trapz(I,dx=dx,dy=dx)
        Res = Nominator_x/Denominator+Nominator_y/Denominator
        
        return np.sqrt(Res)
    
    def Channel_capacity(self,L,Pm0_m):
        
        """
        Calculate the channel capacity in discrete memoryless channels using OAM communication.
        Parameters:
            
            L:  range of topological charge (-L,L)，thus formed N=2*L+1 dimensional Hilbert space.
            Pm: 2-D probability distribution matrix
        
        Returns：
            channel capacity.
        
        !! Usage of this function and the subfunction below:
        Pm0_m = VB.Pm_m0(m0_Max,m_Max)       # transmit the computed result of subfunction to Channel_capacity().
        C = VB.Channel_capacity(m0_Max,Pm0_m)
        
        """
        
        N = 2*L+1
        C = np.log2(N)
        
        A = 0
        B = 0
        
        for m in range(-L,L):
            for m0 in range(-L,L):
                
                A += Pm0_m[m,m0]*np.log2(Pm0_m[m,m0])     
                for m1 in range(-L,L):
                
                    B += Pm0_m[m,m0]*np.log2(Pm0_m[m,m1])
                
                C += A-B
        
        return C/N
    
    def Pm0_m(self,m0_Max=3,m_Max=5,N=512,Distance=1000*LP.m,Cn2=1e-14):
    
        """
        Calculate the topological charge probability distribution in received light field while the topological charge
        of initial light field varies in (-m0_Max,m0_Max).

        Parameters:
            m0_Max：initial light field topological charge range is(-m0_Max,m0_Max);
            m_Max: received (analytical) light field range is (-m_Max,m_Max);
            N: light field sampling number;
            Distance: light field propagation distance;
            Cn2: atmospheric turbulence intensity;
        
        Returns：
            2-D probability distribution matrix (horizontal axis stands for input topological charge variation, vertical axis 
            stands for output topological charge variation)
        """
        
        Pm0_m = np.zeros((2*m0_Max+1,2*m_Max+1),dtype='float')    # pretreatment of the probability distribution matrix;
        
    # In[]: propagate in turbulece and calculate the spiral spectrum coefficients.
    
        for m0 in range(-m0_Max,m0_Max+1):
            
            Amp = special_beam.Laguerre_GaussianBeam(l=m0,polar=True)     # generate orginal light field amplitude;
        
            Phase_screen = Turbulence_Lib.Generate_screen(Cn2,N=N,delta=0.002,L0=5*LP.m,l0=5*LP.mm,
                                      von_karman=True,alpha=11./3,im=True,plot=False) 
            
            Amp = Turbulence_Lib.Free_space_propagation(Amp, self.wvl, self.WindowSize/N, self.WindowSize/N, distance = 50*LP.m)
            for j in range(int(Distance//50)-1):
                screen = next(Phase_screen)
                Amp *= np.exp(1j*screen)                          # add turbulence-caused phase disturbance;
                Amp = Turbulence_Lib.Free_space_propagation(Amp, self.wvl, self.WindowSize/N, self.WindowSize/N, distance = 50*LP.m)
            
            Pm = self.mode_decomposition(Amp,TC=range(-m_Max,m_Max+1))
            Pm = abs(Pm/np.sum(Pm))
            
# In[]: Adjust the relative position of the data  in Pl_total.
# The original order is[0 1 2 -2 -1], final order is [-2 -1 0 1 2];
            
            (M,) = Pm.shape
            Pm = list(Pm)                # turn array into list to facilitate the following operations;
            Temp = Pm[M//2+1:]
            del Pm[M//2+1:]
            Temp.extend(Pm)
            
            Pm0_m[m0 + m0_Max] = Temp    # add the result of mode decomposition into an 2-D matrix longitudinally.
    
        return Pm0_m
    
    def Power_in_the_Bucket(self,PIB_Func,a,b):

        """
        Calculate the PIB using numerical methods.
            PIB, power in the bucket, defined as the ratio of the far field laser power in given 
            size 'bucket'  and the total power(percentage), it can be used to describe the laser power 
            concentration, and reflect the focusing ability in far field of practical laser.

        Parameters:    
            Func: integrand given below.
            r,theta: polar coordinate.
            a: half width of light spot.
            b: half width of the bucket.
        Returns:
            PIB.
        """
        
        Temp1 = integrate.dblquad(PIB_Func,0,2*np.pi,lambda x:0,lambda x:b)[0]
        Temp2 = integrate.dblquad(PIB_Func,0,2*np.pi,lambda x:0,lambda x:a)[0]
        PIB = Temp1/Temp2
            
        return PIB
        
    def PIB_Func(self, Amp, r):
           
        """
        Offer an integrand func in Power_in_the_Bucket().
        (Attention！this funciton is a template func, this function should be modified based on 
        actual situation each time)
        Parameters:
            r: polar coordinate.
        
        Returns: 
            Integrand.
        """
        
        Integrand = (Amp*Amp.conjugate()).real*r
        
        return Integrand

    def FGB_H(self, theta, kappa, z, L=1000*LP.m, order=5):
        
        """
        计算FGB光束的H，参考卢芳论文(3-16)
        """
        
        wvn = 2*np.pi/self.wvl
        D_L = 0
        for n in range(1,order+1):
            
            D_L += ((-1)**(n-1)/(2j*n*L + wvn*self.w0**2)*special.comb(order,n))
        H = wvn/D_L
        Sum = 0
        for n in range(1,order+1):
            
            Temp = 2j*n*L+wvn*self.w0**2
            Sum += (1j*(-1)**(n-1)/Temp*special.comb(order,n)
            *np.exp((L-z)*(2*n*z-1j*wvn*self.w0**2)/2/wvn/Temp*kappa**2))
            
        H *= Sum
        return H
    
#    def vortex_H(self, theta, kappa, z, L=1000*LP.m, TC=1):
#        
#        x = np.linspace(-self.WindowSize/2,self.WindowSize/2,self.N)
#        y = np.linspace(-self.WindowSize/2,self.WindowSize/2,self.N)
#        x,y = np.meshgrid(x,y)
#        wvn = 2*np.pi/self.wvl
#        p = 1/self.w0**2-1j*wvn/2/z
#        p_L = 1/self.w0**2-1j*wvn/2/L
#        p1 = wvn**2/4/z**2/p-1j*wvn/2/z-1j*wvn/2/(L-z)
#        q1_L = -1j*wvn/2/z*x
#        q2_L = -1j*wvn/2/z*y
#        q3 = 1j*kappa*np.cos(theta)/2-1j*wvn/2/(L-z)*x
#        q4 = 1j*kappa*np.sin(theta)/2-1j*wvn/2/(L-z)*y
#        Sum = np.array((self.N,self.N),dtype='complex')
#        
#        for r in range(abs(TC)+1):
#            for t1 in range((abs(TC)-r)//2+1):
#                for t2 in range(r//2+1):
#                    for f1 in range((abs(TC)-r-2*t1)//2+1):
#                        for f2 in range((r-2*t2)//2+1):
#                            
#                            Gamma0 = (4**(-t1-t2)*comb(abs(TC),r)*(1j*np.sign(TC))**r*Fac(abs(TC)-r)*Fac(r)
#                            /Fac(abs(TC)-r-2*t1)/Fac(t1)/Fac(r-2*t2)/Fac(t2))
#                            Gamma = Gamma0*(4**(-f1-f2)*Fac(abs(TC)-r-2*t1)*Fac(r-2*t2)/Fac(abs(TC)-r-2*t1-2*f1)
#                            /Fac(f1)/Fac(r-2*t2-2*f2)/Fac(f2))
#                            # !!!未完成
#        return 
    
    # In[]: 对于vortex_H表达式中的分母能否提到求和中约掉，进行验证.
    
#    def Verification(self, theta=np.pi/4, kappa=1, z=2000*LP.m, L=1000*LP.m, TC=1):
#        
#        x = np.linspace(-self.WindowSize/2,self.WindowSize/2,self.N)
#        y = np.linspace(-self.WindowSize/2,self.WindowSize/2,self.N)
#        x,y = np.meshgrid(x,y)
#        wvn = 2*np.pi/self.wvl
#        p = 1/self.w0**2-1j*wvn/2/z
#        p_L = 1/self.w0**2-1j*wvn/2/L
#        p1 = wvn**2/4/z**2/p-1j*wvn/2/z-1j*wvn/2/(L-z)
#        q1_L = -1j*wvn/2/z*x
#        q2_L = -1j*wvn/2/z*y
#        q3 = 1j*kappa*np.cos(theta)/2-1j*wvn/2/(L-z)*x
#        q4 = 1j*kappa*np.sin(theta)/2-1j*wvn/2/(L-z)*y
#        Sum1 = np.array((self.N,self.N),dtype='complex')
#        Sum2 = np.array((self.N,self.N),dtype='complex')
#        
#        for r in range(abs(TC)+1):
#            for t1 in range((abs(TC)-r)//2+1):
#                for t2 in range(r//2+1):
#                    for f1 in range((abs(TC)-r-2*t1)//2+1):
#                        for f2 in range((r-2*t2)//2+1):
#                            
#                            Gamma0 = (4**(-t1-t2)*comb(abs(TC),r)*(1j*np.sign(TC))**r*Fac(abs(TC)-r)*Fac(r)
#                            /Fac(abs(TC)-r-2*t1)/Fac(t1)/Fac(r-2*t2)/Fac(t2))
#                            Gamma = Gamma0*(4**(-f1-f2)*Fac(abs(TC)-r-2*t1)*Fac(r-2*t2)/Fac(abs(TC)-r-2*t1-2*f1)
#                            /Fac(f1)/Fac(r-2*t2-2*f2)/Fac(f2))
#                            
#                            Sum1 += (Gamma*(-1j*wvn/2/z)**(abs(TC)-2*t1-2*t2)*p**(t1+t2-abs(TC))*p1**(f1+f2+2*t1+2*t2-abs(TC))*q3**(abs(TC)-r-2*t1-2*f1)
#                            *q4**(r-2*t2-f2)*np.exp((q3**2+q4**2)/p1))
#                            
#        for r in range(abs(TC)+1):
#            for t1 in range((abs(TC)-r)//2+1):
#                for t2 in range(r//2+1):
#                    Gamma0 = (4**(-t1-t2)*comb(abs(TC),r)*(1j*np.sign(TC))**r*Fac(abs(TC)-r)*Fac(r)
#                            /Fac(abs(TC)-r-2*t1)/Fac(t1)/Fac(r-2*t2)/Fac(t2))
#                    Sum2 += (Gamma0*p_L**(t1+t2-abs(TC))*q1_L**(abs(TC)-r-2*t1)*q2_L**(r-2*t2))
#                    
#        Res = Sum1/Sum2
#        
#        return Res
#                    
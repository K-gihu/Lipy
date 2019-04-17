# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 12:07:11 2017

@author: Farewell
"""
from math import factorial as Fac
import numpy as np
from matplotlib import pyplot as plt
from scipy import special,integrate
import LightPipes as LP 
import vortex_array
import vortex_experiment
import vortex_analysis

#==============================================================================
# This is a Vortex beam class:
#==============================================================================
class vortex_beam(vortex_array.vortex_array, vortex_experiment.vortex_experiment, vortex_analysis.vortex_analysis):
    
    """
    A class designed for sovling vortex beam related problems,including:
    models,propagation,quality measurement etc.
    """
   
#    def __init__(self,wvl=1550*LP.nm,w0=0.01*LP.m,
#                 WindowSize=0.1*LP.m,N=512):
#        
#        self.wvl = wvl   # wavelength;
#        self.w0 = w0     # beam waist;
#        self.WindowSize = WindowSize    # plane size of light field;
#        self.N = N    # grid sampling points;
        
# Private Functions:
    
# In[]: Reset the member variables.s
    
    def set_windowsize(self, D):
        
        self.WindowSize = D
        
    def set_w0(self, w0):
        
        self.w0 = w0
        
    def set_wvl(self, wvl):
        
        self.wvl = wvl
        
    def set_N(self, N):
        
        self.N = N
        
    def set_Cn2(self, Cn2):
        
        self.Cn2 = Cn2
        
    def set_L0(self, L0):
        
        self.L0 = L0
        
    def set_l0(self, l0):
    
        self.l0 = l0
    
    def set_alpha(self, alpha):
    
        self.alpha = alpha
    
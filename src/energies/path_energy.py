from .inner_dissipation_path_energy import *
from .outer_dissipation_path_energy import *
import time 


def pathEnergy(pos, lines, normal, bending_weight,membrane_weight,K, xA, xB, energy_weight= 1., a_weight = 1. , 
               b_weight = -.95, mass_factor=None, timedict = None):
    """
    path energy of path with K time steps, pos is (K-1)*3n vector of free vertices, xA and xB are 1*3n vectors of start and end shape
    or None if no fixed start and end shape, energy_weight is weight of outer dissipation energy,  
    a_weight and b_weight are parameters of anisotropy tensor for outer dissipation
    """
    start_time = time.time()
    E = innerDissipationPathEnergy(pos,lines, normal, bending_weight,membrane_weight, K, xA, xB)
    E += energy_weight*outerDissipationPathEnergy(pos, K, xA, xB, a_weight=a_weight, b_weight=b_weight, 
                                                  mass_factor=mass_factor, timedict = timedict)
    if timedict is not None:
        timedict["fullpathEnergy"] += time.time() - start_time  
    return E

def gradientPathEnergy(pos, lines, normal, bending_weight,membrane_weight, K , xA, xB, energy_weight = 1.,a_weight = 1. , 
                       b_weight = -.95, mass_factor=None, timedict = None) :
    """
        return grad in (K-1)*3n
    """
    start_time = time.time()

    grad = gradientInnerDissipationPathEnergy(pos, lines, normal, bending_weight, membrane_weight, K , xA, xB)
    grad += energy_weight*gradientOuterDissipationPathEnergy(pos, K, xA, xB, a_weight=a_weight, b_weight=b_weight, 
                                                             mass_factor=mass_factor, timedict = timedict)
    if timedict is not None:
        timedict["fullgradpathEnergy"] += time.time() - start_time      
    
    return grad

def hessianPathEnergy(pos, lines, normal, bending_weight,membrane_weight, K , xA, xB, energy_weight = 1.,a_weight = 1. ,
                       b_weight = -.95, mass_factor=None, timedict = None) : 

    start_time = time.time()
    hess = hessianInnerDissipationPathEnergy(pos, lines, normal, bending_weight, membrane_weight, K , xA, xB)
    if timedict is not None:
        timedict["elasthess"] += time.time() - start_time   
    hess += energy_weight* hessianOuterDissipationEnergy(pos, K,xA,xB,a_weight = a_weight, b_weight = b_weight, 
                                                         mass_factor=mass_factor, timedict=timedict)             
    if timedict is not None:
        timedict["fullhess"] += time.time() - start_time                      
    return hess
import numpy as np
import matplotlib.pyplot as plt
import os

#load CSV file Jonas
path = open(os.path.expanduser("~/Git_Repos/EvaluationMICADAS/RCD_data.csv"))

#general statistical tools
def mean(x):
    return sum(x)/np.size(x)

def weightedmean(x, w):
    return sum(x*w)/sum(w)

#3.3.2.1 background correction
def weighting(x, C12, t):
    p = C12*t
    return weightedmean(x, p)

def backgroundcorrect(C12, C14, k, C13mol):
    return (C14 - k*C13mol)/C12

def dbackgroundcorrect(dC14, dk):
    return np.sqrt(dC14**2 + dk**2)

#3.3.2.2 blank substraction

def R_molbl(R_mol, Rbl_mol):
    return R_mol - mean(Rbl_mol)

def dR_molbl(d_bl, d_mol):
    return np.sqrt(d_bl**2 + d_mol**2)

#3.3.2.3 Mass Fractionation correction

def dC13_sampleVPDB(C13_sample, C12_sample, C13VPDB, C12VPDB): #VPDB limestrone standard Friedman 1982
    return ((C13_sample/C12_sample)/(C13VPDB/C12VPDB) - 1)*1000

def dC13_sampleVPDB(C13_sample, C12_sample, dC13_std, wmeanallratio_std):
    return (((C13_sample/C12_sample)*(1 + dC13_std/1000))/wmeanallratio_std - 1)*1000

#Fraction correction

def R_molblf(R_molbl, dC13_sample):
    return R_molbl*(0.975/(1+ dC13_sample/1000))**2

def dR_molblf(dR_molbl, dC13_sample):
    return dR_molbl*(0.975/(1+dC13_sample/1000))**2

#3.3.2.4 Standard normalisation

def xred2(d_std, d_stdmolblf):
    d_ext2 = 0.002
    return mean(d_std)**2/np.sqrt(d_ext2 + mean(d_stdmolblf)**2) #goal xred2 close to 1. if larger than 2 ->additional external error. Therefor we have d_ext

def FC14(R_molblf, FC14OXIInom, Rstd_molblf):
    return mean(R_molblf)*(FC14OXIInom/mean(Rstd_molblf))

def dFC14(FC14, dR_molblf, R_molblf, dstdR_molblf, Rstd_molblf):
    return FC14*np.sqrt((dR_molblf/R_molblf)**2 + (mean( dstdR_molblf)/mean(Rstd_molblf))**2)


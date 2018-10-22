import numpy as np
import matplotlib.pyplot as plt
import scipy.stats 
import os #used for data path

# #load CSV file Jonas
"""path_unicode = u'~/Git_Repos/EvaluationMICADAS/RCD_data2csv.csv'
path_unicode.encode('utf-8')"""
path_jonas = open(os.path.expanduser('~/Git_Repos/EvaluationMICADAS/RCD_data2csv.csv'), encoding='utf-8')
data_file = np.genfromtxt(path_jonas, delimiter=',')
# #format data file to only have the relevant number; this should be a 28 by 7 matrix
DF = np.delete(np.delete(data_file, 0,0), np.s_[:4] ,1)
# print(DF.shape)
# # format of DF: 14C counts | 12C (HE) muA | 13C (HE) nA | 13 CH nA (molecular current) |r-time | cyc | sample weight

# Joël's file reader - Jonas file reader does not work at my computer... but as long as main is
# in the same directory as RDC_data2csv.csv this version should work everywhere.

"""
data_file = np.genfromtxt('RCD_data2csv.csv', delimiter=',')
DF = np.delete(np.delete(data_file, 0,0), np.s_[:4] ,1)
"""

#Splitting values into seperate arrays:

print(DF.shape)
C14_counts = DF[:, 0]
C12_microA = DF[:, 1]
C13_nanoA = DF[:, 2]
C13molecularCurrent_nanoA = DF[:, 3]
rtime_s = DF[:, 4]
cycles = DF[:, 5]
sampleweight_microg = DF[:, 6]


#conversion into desired unit

C13_microA = C13_nanoA/1000
C13molecularCurrent_microA = C13molecularCurrent_nanoA/1000



#defining canstants used in the calculation 
def dk(t): 
    return 10*t

def k(t): 
    return 50*t
F14COXII = 1.34066 # Nominal F14C of OxII standard
dF14COXII = -0.0178*F14COXII
_d13COXIInom = -17.8/1000 

#general statistical tools
def mean(x):
    return sum(x)/np.size(x)

def weightedmean(x, w):
    return sum(x*w)/sum(w)

def var(x):
    return np.sqrt(mean(x*x) - mean(x)**2)

#3.3.2.1 background correction

def weighting(x, C12, t):
    p = C12*t
    return weightedmean(x, p)

def backgroundcorrect(C12, C14, t, C13mol):
    return (C14 - k(t)*C13mol)/C12

def dbackgroundcorrect(dC14, dk):
    return np.sqrt(dC14*dC14 + dk*dk)

#3.3.2.2 blank substraction

def R_molbl(R_mol, Rbl_mol):
    return R_mol - mean(Rbl_mol)*np.ones_like(R_mol) #Problem here

def dR_molbl(d_bl, d_mol):
    return np.sqrt(d_bl**2 + d_mol**2)

#3.3.2.3 Mass Fractionation correction

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
    return mean(d_std)**2/np.sqrt(d_ext2 + mean(d_stdmolblf)**2)  # goal xred2 close to 1. if larger than 2
    # ->additional external error. Therefor we have d_ext

def mychisquare(sample):
    _chisquared = 0
    for i in range(sample.shape[0]):
        _chisquared += (mean(sample) - sample[i])**2/np.sqrt(sample[i])
    return _chisquared

def FC14(R_molblf, FC14OXIInom, Rstd_molblf): #ERROR
    return mean(R_molblf)*(FC14OXIInom/mean(Rstd_molblf))

def dFC14(FC14, dR_molblf, R_molblf, dstdR_molblf, Rstd_molblf): #ERROR
    return FC14*np.sqrt((dR_molblf/R_molblf)**2 + (mean( dstdR_molblf)/mean(Rstd_molblf))**2)

# calculations

#age calculations
_R_mol = backgroundcorrect(C12_microA, C14_counts, rtime_s, C13molecularCurrent_microA)


_R_molbl = R_molbl(_R_mol, backgroundcorrect(C12_microA[0:4], C14_counts[0:4], rtime_s[0:4], C13molecularCurrent_microA[0:4]))
dC13_std = np.sqrt(mean(C13_microA[6:12]))
wmeanallratio_std = weightedmean(C13_microA[6:12]/C12_microA[6:12], rtime_s[6:12])
_dC13_sampleVPDB = dC13_sampleVPDB(C13_microA, C12_microA, dC13_std, wmeanallratio_std)
_dC13_sample = 0 # for single sample
_R_molblf = R_molblf(_R_molbl, _dC13_sample)

# 1 calculate with mean of all samples
F14C = (weightedmean(_R_molblf[13:28], rtime_s[13:28])/weightedmean(_R_molblf[6:12], rtime_s[6:12]))*F14COXII
print('weighted mean of all samples F14C ratio: ', F14C)

T_14Cyears = -8033*np.log(F14C) #conventional radiocarbon age
print('weighted mean of all samples T_14C: ', T_14Cyears) #Years BP meaning years before 1950


# 2 date all samples individually:
F14C2 = (_R_molblf[13:28]/weightedmean(_R_molblf[6:12], rtime_s[6:12]))*F14COXII
print('individual F14C ratios of all samples: ', '\n' , F14C2)
T_14Cyears2 = -8033*np.log(F14C2) #conventional radiocarbon age
print('Individual T14C for all samples: ', '\n', T_14Cyears2) #Years BP meaning years before 1950


#error calculation @TODO: use weighted mean instead of mean!!
print('\n', 'Error calculations:', '\n')

_d14C = np.sqrt(C14_counts)  #estimated as sqrt of counts of 14C
_dmol = 3e-16 #given value
_dR_mol = dbackgroundcorrect(_d14C, dk(rtime_s)) 
_dR_molbl = dR_molbl(_dR_mol, _dmol)
_wmeanratio_standard = weightedmean((C13_microA[6:12]/C12_microA[6:12]), rtime_s[6:12])
d13C_sample = dC13_sampleVPDB(C13_microA, C12_microA, _d13COXIInom, _wmeanratio_standard) 
_dR_molblf = dR_molblf(_dR_molbl, d13C_sample)

#standard normalisation

chisquare = scipy.stats.chisquare(C14_counts[13:]) #do the results make sense?

print('mychisquare: ', mychisquare(C14_counts[13:]))
print('sigma(std): ', weightedmean(np.sqrt(C14_counts[6:12]), rtime_s[6:12]))
print('d_std_molblf: ', weightedmean(_dR_molblf[6:12], rtime_s[6:12]))
chisquared_red = weightedmean(np.sqrt(C14_counts[6:12]), rtime_s[6:12])**2 / weightedmean(_dR_molblf[6:12], rtime_s[6:12])**2
print('chisquare_red =', chisquared_red) #nonsense

#ratio correction

dF14C = dFC14(F14C2, _dR_molblf[13:], _R_molblf[13:],  _dR_molblf[6:12], _R_molblf[6:12]) #EROOR

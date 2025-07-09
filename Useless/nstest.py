import matplotlib.pyplot as plt
import numpy as np
import camb
import camb.correlations
from getdist import loadMCSamples

xvals = np.linspace(0.9999,-0.9999,1800) # For easy camb comparations 
theta= np.arccos(xvals)*180/np.pi


def compute_cl_cor_ns_test(parss,ns,lmax,xvals):
        pars = camb.set_params(ombh2=parss['omegabh2'], omch2=parss['omegach2'], H0 = parss['H0'],omk=parss['omegak'],
                                    YHe=parss['yheused'], nnu=parss['nnu'], nrun=parss['nrun'], Alens=parss['Alens'], ns=ns, As=np.exp(parss['logA'])*1e-10,w=-1,wa=parss['wa'], mnu=parss['mnu'], tau=parss['tau'])
        resu = camb.get_results(pars)
        cls = resu.get_cmb_power_spectra(pars, CMB_unit='muK',lmax=lmax)
        totCL=cls['total']
        TTCl=totCL[2:,0]
        totCorr= camb.correlations.cl2corr(totCL,xvals,lmax)
        TTcor=totCorr[:,0]
        return TTCl,  TTcor, TTcor[-1]









roots = ['COM_CosmoParams_fullGrid_R3.01/base/CamSpecHM_TT/base_CamSpecHM_TT']
data_dict={}
for root in roots:
    try:
        samples = loadMCSamples(file_root=root)
        
        params = samples.getParamBestFitDict()
        
        fixed = samples.ranges.fixedValueDict()
        
        dic = params | fixed
        ns_values = np.linspace(0.9,1.1,15)
        for ns in ns_values:
            try:
                TTCl,  TTcor, _ = compute_cl_cor_ns_test(dic,ns,40,xvals)
            except Exception as e:
                print(e)
                continue
            plt.plot(theta,TTcor,label=rf'$n_s={round(ns,3)}$')
        plt.title(f'{root}'+'\n'+r'$n_s $' + f'{min(ns_values)}-{max(ns_values)}')
        plt.legend(ncol=3)
        plt.savefig('Testns4.png',dpi=300)
    except Exception as e:
        print(e)
        continue
---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.2
  kernelspec:
    display_name: .venv
    language: python
    name: python3
---

# Libraries

```python
# Math
import numpy as np
from math import factorial
from decimal import getcontext, Decimal
from scipy.integrate import simpson

import pandas as pd
# Cosmology
from astropy.io import fits
import healpy as hp
import camb
import camb.correlations
from getdist import loadMCSamples

# Plotting
import matplotlib.pyplot as plt
import matplotlib.colors as colors
# import scienceplots 
#plt.style.use(['science','ieee'])
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 12,
    'text.usetex': False  # Set True if LaTeX installed
})
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


from matplotlib.patches import Rectangle
%matplotlib inline
# System
import pickle
import os

# Decorator to time execution time 

import time
import functools

def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} executed in {end - start:.2f} seconds")
        return result
    return wrapper

```

# Functions


```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2 * np.pi, 200)
y = np.sin(x) * np.exp(-0.2 * x)

plt.figure(figsize=(4, 3))
plt.plot(x, y, label="damped sine")
plt.title("Test plot")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.tight_layout()

plt.show()

```
## Legendre polynomials

```python

def legendre(lmax, x): 
  if lmax == 0: 
    return 1.0 
  elif lmax == 1: 
    return x 
  elif lmax == -1: 
    return 0 
  else: 
    p0 = 1.0 # P_0 
    p1 = x # P_1 
    for l in range(2, lmax + 1): 
      p_next = ((2 * l - 1) / l) * x * p1 - ((l - 1) / l) * p0 
      p0, p1 = p1, p_next 
  return p1

P=np.vectorize(legendre)
```

## Correlation Function

```python

@timeit
def correlation_func(D_ell, xvals):
    l = np.arange(len(D_ell)) + 2
    fac2 = (2*l + 1) / (2 * l * (l + 1)) * D_ell
    cor = np.sum(fac2[:, None] * P(l[:, None], xvals),axis=0)
    return cor



def correlation_func_err(error,xvals):
    cor_err=0

    fac_err= [(2*(l+2)+1)/(2*(l+2)*((l+2)+1))*c for l,c in enumerate(error)]
    for l,f in enumerate(fac_err):
        cor_err+=(f*P(l+2,xvals))**2
    return cor_err**0.5

def correlation_func_err2(error,xvals):
    cor_err=0

    fac_err= [(2*(l+2)+1)/(2*(l+2)*((l+2)+1))*c for l,c in enumerate(error)]
    for l,f in enumerate(fac_err):
        cor_err+=(f*P(l+2,xvals))
    return (cor_err**2)**0.5
```

```python
#print(P(200,np.linspace(0,1,2)))
print(correlation_func(np.linspace(1000,2000,200),np.linspace(0,1,5)))
```
## Estimators

<!-- #region -->
### $\bar{\langle\xi\rangle}_a^b$


$$\bar{\langle\xi\rangle}_a^b=\frac{\int_a^b C(\theta)\sin(\theta)d\theta}{\cos(b)-\cos(a)} =\int_a^b \sum_l \frac{2l+1}{2l(l+1)}D_lP_l(\cos(\theta)) d\cos(\theta)=\sum_l \frac{2l+1}{2l(l+1)}D_l \int_a^b P_l(\cos(\theta)) d\cos(\theta)=\sum_l \frac{2l+1}{2l(l+1)}D_l\left[ \frac{P_{l+1}(\cos(\theta))-P_{l-1}(\cos(\theta))}{2l+1}\right]^b_a=$$

$$=\sum_l \frac{D_l}{2l(l+1)}\left[P_{l+1}(\cos(\theta))-P_{l-1}(\cos(\theta))\right]^b_a=\sum_l \frac{D_l}{2l(l+1)}\left[P_{l+1}(b)-P_{l-1}(b)-P_{l+1}(a)+P_{l-1}(a)\right]$$

$$\Delta^2\bar{\langle\xi\rangle}_a^b=\left(\frac{\partial \bar{\langle\xi\rangle}_a^b}{\partial D_l}\Delta D_l\right)^2$$
<!-- #endregion -->

```python


@timeit
def xivar2(D_ell, a,b):
    l = np.arange(len(D_ell)) + 2
    fac = D_ell / (2 * l * (l + 1))
    term = (P(l + 1, b) - P(l - 1, b)-P(l + 1, a) + P(l - 1, a))
    s=np.sum(fac*(term))
        
    return s/(b-a)
@timeit 
def xivar_num(cor,a,b):
    return simpson(cor,np.linspace(a,b,1800))

def xivar_err(D_ell_err, a,b):
    s=0
    for i,d in enumerate(D_ell_err):
        l=i+2
        fac=d/(2*l*(l+1))
        
        
        integral = Decimal(legendre(l + 1, b) - legendre(l - 1, b)-legendre(l + 1, a) + legendre(l - 1, a))
    
        s+=(fac*float(integral)/(b-a))**2
    
    return (s)**0.5

@timeit
def xivar(D_ell, a,b): 
   s=0 
   for i,d in enumerate(D_ell): 
       l=i+2 
       fac= d / (2*l*(l+1)) 
       term = Decimal(legendre(l + 1, b) - legendre(l - 1, b)-legendre(l + 1, a) + legendre(l - 1, a)) 
       s+=fac*float(term) 
   return s/(b-a)

def xivar_err2(D_ell_err, a,b):
    s=0
    for i,d in enumerate(D_ell_err):
        l=i+2
        fac=d/(2*l*(l+1))
        
        
        integral = Decimal(legendre(l + 1, b) - legendre(l - 1, b)-legendre(l + 1, a) + legendre(l - 1, a))
    
        s+=(fac*float(integral)/(b-a))
    
    return (s**2)**0.5
```
### $S_a^b$


$$S_a^b=\int_a^b C(\theta)^2\sin(\theta)d\theta =\int_a^b \left[\sum_l \frac{2l+1}{2l(l+1)}D_lP_l(\cos(\theta))\right]^2 d\cos(\theta)= \sum_n\sum_m \frac{2n+1}{2n(n+1)}D_n \frac{2m+1}{2m(m+1)}D_m\underbrace{\int_a^b P_n(\cos(\theta))P_m(\cos(\theta)) d\cos(\theta)}_{T_{nm}}$$

$$\Delta^2 S_a^b=\left(\frac{\partial S_a^b}{\partial D_n}\Delta D_n\right)^2+\left(\frac{\partial S_a^b}{\partial D_m}\Delta D_m\right)^2=$$

$$=\left(\sum_n\sum_m \frac{2n+1}{2n(n+1)} \frac{2m+1}{2m(m+1)}D_m T_{nm}\Delta D_n\right)^2+\left(\sum_n\sum_m \frac{2n+1}{2n(n+1)}D_n \frac{2m+1}{2m(m+1)} T_{nm}\Delta D_m\right)^2=$$
$$=\sum_n\sum_m\left( \frac{2n+1}{2n(n+1)} \frac{2m+1}{2m(m+1)}D_m T_{nm}\Delta D_n\right)^2+\left( \frac{2n+1}{2n(n+1)}D_n \frac{2m+1}{2m(m+1)} T_{nm}\Delta D_m\right)^2$$

$$=\sum_n\sum_m\left( \frac{2n+1}{2n(n+1)} \frac{2m+1}{2m(m+1)}T_{nm}\right)^2 \left(D_m^2 \Delta D_n^2+D_n^2 \Delta D_m^2\right)$$

```python
def A_r(r):
    """Compute A_r using Decimal for high precision."""
    numerator = Decimal(1)
    for i in range(1, r+1):
        numerator *= Decimal(2*i-1)
    denominator = factorial(r)
    return numerator / denominator


def Tmn(l,l1,l2,a=-1,b=1/2):
    # Set the precision high enough to handle large calculations
    getcontext().prec = 1000
    
    matrix = np.zeros((l, l), dtype=float)
    for i in range(l):
        n=i+2
        
        for j in range(l):
            m=j+2
            
            for r in range(min(m,n)+1):
                integral=A_r(r)*A_r(m-r)*A_r(n-r)/A_r(m+n-r) /(2*m+2*n-2*r+1) * Decimal((legendre(m+n-2*r + 1, b) - legendre(m+n-2*r - 1, b))-(legendre(m+n-2*r + 1, a) - legendre(m+n-2*r - 1, a)))
                matrix[i,j]+=np.float64(integral)       
        
    np.save(f"Tmn_{l1}_{l2}.npy", matrix)


def S12(D_ell,M):
    # Set the precision high enough to handle large calculations
    getcontext().prec = 1000
    s=Decimal(0)
    for i,xn in enumerate(D_ell):
        n=i+2
        fac1=(((2*n+1)*xn)/(2*n*(n+1)))
        for j,xm in enumerate(D_ell):
            m=j+2
            fac2=(((2*m+1)*xm)/(2*m*(m+1)))
        
            integral=M[i,j]
    
            s += Decimal(fac1)*Decimal(fac2)*Decimal(integral)
        
    return float(s)
    
def S12_vec(D_ell, M):
    getcontext().prec = 1000
    D_ell = np.array(D_ell, dtype=float)
    M = np.array(M, dtype=float)
    n = np.arange(2, len(D_ell) + 2)
    f = ((2 * n + 1) * D_ell) / (2 * n * (n + 1))
    return float(f @ M @ f)

def S12_err(D_ell,D_ell_err,M):
    # Set the precision high enough to handle large calculations
    getcontext().prec = 1000
    s=Decimal(0)
    for i,xn in enumerate(D_ell_err):
        n=i+2
        fac1=(((2*n+1))/(2*n*(n+1))) 
        for j,xm in enumerate(D_ell):
            m=j+2
            fac2=(((2*m+1))/(2*m*(m+1))) 
            Amn=fac1*fac2
            integral=M[i,j]

            s += Decimal(Amn**2)*Decimal(integral**2)*Decimal(D_ell[i]**2*xm**2 + D_ell[j]**2*xn**2)
        
    return float(s)**0.5

def S12_err2(D_ell,D_ell_err,M):
    # Set the precision high enough to handle large calculations
    getcontext().prec = 1000
    s1=Decimal(0)
    s2=Decimal(0)
    for i,xn in enumerate(D_ell_err):
        n=i+2
        fac1=(((2*n+1))/(2*n*(n+1))) 
        for j,xm in enumerate(D_ell):
            m=j+2
            fac2=(((2*m+1))/(2*m*(m+1))) 
            Amn=fac1*fac2
            integral=M[i,j]

            s1 += Decimal(Amn)*Decimal(integral)*Decimal(D_ell[i]*xm)
            s2 += Decimal(Amn)*Decimal(integral)*Decimal(D_ell[j]*xn)
        
    return float(s1**2+s2**2)**0.5
```

## Map Functions

```python
def map_rot_refl(map):
    opposite_map = np.zeros_like(map)
    nside = hp.get_nside(map)
    l, b = hp.pix2ang(nside,np.arange(len(map)),lonlat=True)

    b_opposite = -b
    l_opposite = l + 180
    
    opposite_index = hp.ang2pix(nside,l_opposite, b_opposite,lonlat=True)
    opposite_map = map[opposite_index]
    return opposite_map

def estimate_coef(x, y):
    # number of observations/points
    n = np.size(x)
  
    # mean of x and y vector
    m_x = np.mean(x)
    m_y = np.mean(y)
  
    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x
  
    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x
  
    return (b_0, b_1)

def plot180(map,opposite_map,map_name,filename,lower=False):
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'text.usetex': False  # Set True if LaTeX installed
    })
    if lower:
        low_nside = hp.get_nside(map)//2
        map=hp.ud_grade(map,low_nside)

    mult = map*opposite_map
    
    a,b= estimate_coef(x=map,y=opposite_map)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(map,opposite_map,marker='o',c='blue', label= rf'Impainting {map_name} $C(180)={np.mean(mult):3f}$', s=2, alpha=0.05) #map, cmap='viridis'
    ax.plot(map,b*map+a,color='red', label=rf"$y={b:3f}x {a:3f}$")
    ax.set_xlabel(r'T [$K_{CMB}$]')
    ax.set_ylabel(r'Inverted map T [$K_{CMB}$]')
    #ax.set_xscale('log')
    ax.grid(True, which='both', ls='--', alpha=0.6)
    ax.legend(frameon=True, loc='best')

    plt.tight_layout()
    plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.close() 

def map_contours(map,opposite_map,filename):
    fig= plt.figure(figsize=(8,8))

    h,xx,yy,_=plt.hist2d(map, opposite_map,bins=np.linspace(-200,200,100))
    x = (xx[1:]+xx[:-1])/2
    y = (yy[1:]+yy[:-1])/2
    plt.colorbar()
    plt.grid(True, which='both', ls='--', alpha=0.6, color='red')
    plt.axis('equal')
    plt.contour(x,y,h, levels=[4000,6000,10000],colors='k')
    plt.contour(x,-y,h, levels=[4000,6000,10000],colors='w')
    plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.close() 
```

## Chains and CAMB calculations

```python
def compute_cl_cor_pl(parss,lmax,xvals):
        pars = camb.set_params(ombh2=parss['omegabh2'], omch2=parss['omegach2'], H0 = parss['H0'],omk=parss['omegak'],
                                    YHe=parss['yheused'], nnu=parss['nnu'], nrun=parss['nrun'], Alens=parss['Alens'], ns=parss['ns'], As=np.exp(parss['logA'])*1e-10,w=-1,wa=parss['wa'], mnu=parss['mnu'], tau=parss['tau'])
        resu = camb.get_results(pars)
        cls = resu.get_cmb_power_spectra(pars, CMB_unit='muK',lmax=lmax)
        totCL=cls['total']
        TTCl=totCL[2:,0]
        totCorr= camb.correlations.cl2corr(totCL,xvals,lmax)
        TTcor=totCorr[:,0]
        return TTCl,  TTcor, TTcor[-1]

def compute_cl_cor_dv(parss,lmax,xvals):
        pars = camb.set_params(ombh2=parss['ombh2'], omch2=parss['omch2'], H0 = parss['H0'],omk=0,
                                    YHe=parss['YHe'], ns=parss['ns'], As=np.exp(parss['logA'])*1e-10,w=parss['w'], mnu=parss['mnu'], tau=parss['tau'])
        resu = camb.get_results(pars)
        cls = resu.get_cmb_power_spectra(pars, CMB_unit='muK',lmax=lmax)
        totCL=cls['total']
        TTCl=totCL[2:,0]
        totCorr= camb.correlations.cl2corr(totCL,xvals,lmax)
        TTcor=totCorr[:,0]
        return TTCl,  TTcor, TTcor[-1]

def expand_dict_values(dict1, dict2):
    # Get the length of arrays in the first dictionary
    first_key = next(iter(dict1))  # Get any key from dict1
    target_length = len(dict1[first_key])  # Get the array length

    # Expand the values in dict2 to match the target length
    expanded_dict2 = {key: [value] * target_length for key, value in dict2.items()}
    z = dict1 | expanded_dict2
    return z

def chain_calculations(parss,lmax,xvals,intervals,c):
    if c<=4:
        TTCl,  TTcor, C180 = compute_cl_cor_pl(parss,lmax,xvals)
    else:
        TTCl,  TTcor, C180 = compute_cl_cor_dv(parss,lmax,xvals)
    th_values= []
    for a, b in intervals:
        M = np.load(f"Tmn__{round(np.arccos(a)*180/np.pi)}__{round(np.arccos(b)*180/np.pi)}.npy")
        s12 = S12(TTCl,M)
        th_values.append(s12)
        xiv = xivar(TTCl, a, b)
        th_values.append(xiv)

    return (TTCl, TTcor, C180, *th_values)


def chain_results(intervals,xvals, roots,name,n=1000):
    chain_cols = []
    chain_cols += ['D_ell', 'Cor']
    for a, b in intervals:
        chain_cols += [f's12_{round(np.arccos(a)*180/np.pi)}_{round(np.arccos(b)*180/np.pi)}',
                       f'xiv_{round(np.arccos(a)*180/np.pi)}_{round(np.arccos(b)*180/np.pi)}']

    chain_cols += ['C180']
    

    data_dict = {}

    for i,root in enumerate(roots):
        try:
            samples = loadMCSamples(file_root=root)
            print(f'Procesando: {root}')
            print('_' * 40)

            params = samples.getParams()
            fixed = samples.ranges.fixedValueDict()

            chain_data = {name.name: getattr(params, name.name, np.nan) for name in samples.paramNames.names}
            data = expand_dict_values(chain_data,fixed)
            df = pd.DataFrame.from_dict(data).tail(n)

            chain_result = df.apply(chain_calculations, axis=1, args=(200, xvals, intervals, i))
            df_chain = pd.DataFrame(chain_result.tolist(), columns=chain_cols, index=df.index)
            df = pd.concat([df, df_chain], axis=1)

            stacked_Cl = np.vstack(df['D_ell'].values)
            stacked_Cor = np.vstack(df['Cor'].values)

            mean_Cl = stacked_Cl.mean(axis=0)
            std_Cl = stacked_Cl.std(axis=0,ddof=0)
            mean_Cor = stacked_Cor.mean(axis=0)
            std_Cor = stacked_Cor.std(axis=0,ddof=0)

            short_name = root.strip().split('/')[-1]
            data_dict[short_name] = (df[chain_cols], mean_Cl, std_Cl, mean_Cor, std_Cor)

        except Exception as e:
            print(f'Error en {root}: {e}')
            continue

        finally:
            
            # Guardar todo el diccionario completo en un solo archivo
            with open(f'{name}.pkl', 'wb') as f:
                pickle.dump(data_dict, f)



```

## Plot functions


### $\bar{\langle\xi\rangle}_a^b$ plots

```python
def plot_corr_with_xivar(xvals, unbin_cl,corr_th, est_df, intervals,name):
    """
    Grafica la correlación experimental y teórica, junto con las integrales
    y medias por intervalos, y bandas de error usando fill_between.

    Parámetros:
    - xvals: valores de x (eje horizontal)
    - unbin_cl: diccionario con 'D_ell' y 'Error'
    - D_ell_th: no se usa en esta versión pero puedes añadirlo si quieres comparar con teoría
    - intervals: lista de tuplas con los intervalos [(a1, b1), (a2, b2), ...]
    """
    
    corr = correlation_func(unbin_cl['D_ell'][:200], xvals)

    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    # Plot datos experimentales con banda de error
    ax.scatter(np.arccos(xvals)*180/np.pi, corr, s=2,marker='o',c='r', label='Correlation function data')
    ax.scatter(np.arccos(xvals)*180/np.pi, corr_th, s=2,marker='o',c='b', label='Correlation function model')

    # Para cada intervalo, calcular media y error y dibujar con fill_between
    for (a, b), i in zip(intervals,est_df.columns):
        mean_sq = xivar(unbin_cl['D_ell'][:200], a, b)
        mean_sq_err = xivar_err2(unbin_cl['Error'][:200], a, b)

        
        
        ax.fill_between([np.arccos(a)*180/np.pi, np.arccos(b)*180/np.pi],
                        [mean_sq - mean_sq_err] ,
                        [mean_sq + mean_sq_err] , alpha=0.4,color='r')
        ax.plot([np.arccos(a)*180/np.pi, np.arccos(b)*180/np.pi],
                        [mean_sq,mean_sq] ,ls='-.', alpha=0.8,color='r',
                        label=rf'$\xi_{{{round(np.arccos(a)*180/np.pi)}}}^{{{round(np.arccos(b)*180/np.pi)}}}$: {mean_sq:.2f} ± {mean_sq_err:.2f}')
    
        

        ax.fill_between([np.arccos(a)*180/np.pi, np.arccos(b)*180/np.pi],
                        [np.mean(est_df[i]) - np.std(est_df[i])] ,
                        [np.mean(est_df[i]) + np.std(est_df[i])] , alpha=0.4,color='b')
        ax.plot([np.arccos(a)*180/np.pi, np.arccos(b)*180/np.pi],
                        [np.mean(est_df[i]) , np.mean(est_df[i]) ] ,ls='-.', alpha=0.8,color='b',
                        label=f'{i}: {np.mean(est_df[i]):.2f} ± {np.std(est_df[i]):.2f}')

    #plt.yscale('log')
    plt.ylim(-400,400)
    plt.xlabel(r"$\theta$ [º]")
    plt.ylabel(r"$C(\theta)$")
    plt.title(r"$\xi_a^b$ analysis for "+ str(name))
    plt.legend(fontsize=7)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
```

### $S_a^b$ plots

```python
def plot_corr_with_S12(xvals, unbin_cl, cor_th,est_df, intervals,name):
    """
    Grafica la correlación experimental y teórica, junto con las integrales
    y medias por intervalos, y bandas de error usando fill_between.

    Parámetros:
    - xvals: valores de x (eje horizontal)
    - unbin_cl: diccionario con 'D_ell' y 'Error'
    - D_ell_th: no se usa en esta versión pero puedes añadirlo si quieres comparar con teoría
    - intervals: lista de tuplas con los intervalos [(a1, b1), (a2, b2), ...]
    """
    
    corr = correlation_func(unbin_cl['D_ell'][:200], xvals)
    
    plt.figure(figsize=(15, 8))
    ax = plt.gca()

    # Plot datos experimentales con banda de error
    ax.scatter(np.arccos(xvals)*180/np.pi, corr**2, s=2,marker='o',c='r', label='Correlation function data')
    ax.scatter(np.arccos(xvals)*180/np.pi, cor_th**2, s=2,marker='o',c='b', label='Correlation function model')

    # Para cada intervalo, calcular media y error y dibujar con fill_between
    for (a, b), i in zip(intervals,est_df.columns):
        M = np.load(f"Tmn__{round(np.arccos(a)*180/np.pi)}__{round(np.arccos(b)*180/np.pi)}.npy")
        mean_sq = S12(unbin_cl['D_ell'][:200],M)
        mean_sq_err = S12_err2(unbin_cl['D_ell'][:200],unbin_cl['Error'][:200], M)

        
        ax.fill_between([np.arccos(a)*180/np.pi, np.arccos(b)*180/np.pi],
                        [mean_sq - mean_sq_err] ,
                        [mean_sq + mean_sq_err] , alpha=0.4,color='r')
        ax.plot([np.arccos(a)*180/np.pi, np.arccos(b)*180/np.pi],
                        [mean_sq,mean_sq] ,ls='-.', alpha=0.8,color='r',
                        label=rf'$S_{{{round(np.arccos(a)*180/np.pi)}}}^{{{round(np.arccos(b)*180/np.pi)}}}$: {mean_sq:.2f} ± {mean_sq_err:.2f}')
        

        ax.fill_between([np.arccos(a)*180/np.pi, np.arccos(b)*180/np.pi],
                        [np.mean(est_df[i]) - np.std(est_df[i])] ,
                        [np.mean(est_df[i]) + np.std(est_df[i])] , alpha=0.4,color='b')
        ax.plot([np.arccos(a)*180/np.pi, np.arccos(b)*180/np.pi],
                        [np.mean(est_df[i]) , np.mean(est_df[i]) ] ,ls='-.', alpha=0.4,color='b',
                        label=f'{i}: {np.mean(est_df[i]):.2f} ± {np.std(est_df[i]):.2f}')
    
    plt.yscale('symlog', linthresh=10)#plt.yscale('log')
    #plt.ylim(0,3e4)
    plt.xlabel(r"$\theta$ [º]")
    plt.ylabel(r"$C(\theta)^2$")
    plt.title(r"$S_a^b$ analysis for "+ str(name))
    plt.legend(fontsize=7)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
```

### General plots

```python
def create_histogram_grid(df, labels, exp_vals,title, figsize=(15, 15), bins='auto'):
    """
    Creates a grid of histograms from DataFrame columns, overlaying experimental value ± error.

    Parameters:
    - df: Pandas DataFrame with numerical columns.
    - labels: List of strings for subplot titles (must match df.columns).
    - exp_vals: List [val1, err1, val2, err2, ...] (length = 2 * len(labels)).
    - figsize: Size of the full figure.
    - bins: Bin specification for histograms.
    """
    n_cols = len(df.columns)
    if len(labels) != n_cols or len(exp_vals) != 2 * n_cols:
        raise ValueError("Mismatch in number of columns, labels, or experimental values.")
    
    n_rows = round(np.ceil(n_cols / 2))
    fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
    axes = axes.flatten()
    fig.suptitle(title, fontsize=16)

    for i, (col, label) in enumerate(zip(df.columns, labels)):
        ax = axes[i]
        data = df[col].dropna()
        if len(data) == 0:
            continue

        exp_value = exp_vals[2*i]
        exp_err = exp_vals[2*i + 1]
        
        mean_val = np.mean(data)
        std_val = np.std(data)
        sigmas = np.sqrt((exp_value - mean_val)**2 / (exp_err**2 + std_val**2))

        # Histograma
        n, bins_hist, _ = ax.hist(data, bins=bins, edgecolor='b', alpha=0.7)

        # Experimental error
        rect = Rectangle((exp_value - exp_err , 0),
                         2*exp_err,
                         np.max(n),
                         color='r', alpha=0.3, label=f'± {exp_err:2f}')
        ax.add_patch(rect)

        # Experimental value
        ax.axvline(exp_value, color='r', linestyle='--', label=f'{exp_value:2f}')
        # Theoretical value and deviation
        rect2 = Rectangle((mean_val - std_val , 0),
                         2*std_val,
                         np.max(n),
                         color='b', alpha=0.3, label=f'± {std_val:2f}')
        ax.add_patch(rect2)

        # Mean of the theoretical model
        ax.axvline(mean_val, color='b', linestyle='--', label=f'{mean_val:2f}')

        ax.set_title(f"{label}\nΔ = {sigmas:.2f} σ", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=8)

    for j in range(n_cols, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def plot_power_and_correlation(unbin_cl, mean_Cl, std_Cl,
                                xvals, mean_Cor, std_Cor,
                                root="Title", figsize=(18, 7)):
    """
    Plots power spectrum and correlation function side-by-side with uncertainty bands and error bars.
    
    Parameters:
    - ell, D_ell, unbin_cl: Data for power spectrum
    - mean_Cl, std_Cl: Mean and std for Cl
    - theta, corr, corr_err: Data for correlation function
    - mean_Cor, std_Cor: Mean and std for correlation
    - root: Title string
    - figsize: Size of the full figure (tuple)
    """
    theta= np.arccos(xvals) * 180 / np.pi  # Convert xvals to degrees
    corr = correlation_func(unbin_cl['D_ell'][:200], xvals)
    corr_err = correlation_func_err2(unbin_cl['Error'][:200], xvals)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # ----------- Power Spectrum ------------
    ax1.set_title(f"{root} - Power Spectrum")
    ax1.errorbar(
        unbin_cl['ell'][:200], unbin_cl['D_ell'][:200],
        yerr=(unbin_cl['-dD_ell'][:200], unbin_cl['+dD_ell'][:200]),
        fmt='o', color='red',
        ecolor=colors.to_rgba('blue', 0.2),
        elinewidth=1, capsize=2, markersize=2,
        label="Power spectrum data"
    )
    ax1.plot(unbin_cl['ell'][:200], mean_Cl, label="Power spectrum mean", color='k')
    ax1.fill_between(unbin_cl['ell'][:200], mean_Cl - 5*std_Cl, mean_Cl + 5*std_Cl,
                     color='k', alpha=0.2, label=r'$5\sigma$')
    ax1.fill_between(unbin_cl['ell'][:200], mean_Cl - std_Cl, mean_Cl + std_Cl,
                     color='k', alpha=0.4, label=r'$1\sigma$')
    ax1.legend()
    ax1.grid(True)

    # ----------- Correlation Function ------------
    ax2.set_title(f"{root} - Correlation Function")
    ax2.errorbar(
        theta, corr, yerr=abs(corr_err),
        fmt='o', color='red',
        ecolor=colors.to_rgba('blue', 0.2),
        elinewidth=1, capsize=2, markersize=2,
        label='Correlation function data'
    )
    ax2.plot(theta, mean_Cor, label="Correlation function mean", color='k')
    ax2.fill_between(theta, mean_Cor - 5*std_Cor, mean_Cor + 5*std_Cor,
                     color='k', alpha=0.2, label=r'$5\sigma$')
    ax2.fill_between(theta, mean_Cor - std_Cor, mean_Cor + std_Cor,
                     color='k', alpha=0.4, label=r'$1\sigma$')
    ax2.axhline(0, color='k')
    ax2.legend()
    ax2.grid(True)

    # Inset plot
    axins = inset_axes(ax2, width="30%", height="30%", loc='center', borderpad=1)
    axins.plot(theta, mean_Cor, color='k')
    axins.errorbar(theta, corr, yerr=corr_err, fmt='o', color='red',
                   ecolor=colors.to_rgba('blue', 0.2),
                   elinewidth=1, capsize=2, markersize=2)
    axins.fill_between(theta, mean_Cor - 5*std_Cor, mean_Cor + 5*std_Cor, color='k', alpha=0.2)
    axins.fill_between(theta, mean_Cor - std_Cor, mean_Cor + std_Cor, color='k', alpha=0.4)
    axins.set_ylim(-600, 500)
    axins.set_xlim(20, 180)
    axins.grid(True)
    mark_inset(ax2, axins, loc1=2, loc2=4, fc="none", ec="k", lw=1)

    plt.tight_layout()
    plt.show()

def create_histogram_grid2(df, labels, exp_df, title, figsize=(15, 15), bins='auto'):
    """
    Creates a grid of histograms from DataFrame columns, overlaying experimental value ± error.

    Parameters:
    - df: Pandas DataFrame with numerical columns.
    - labels: List of strings for subplot titles (must match df.columns).
    - exp_df: Pandas DataFrame with 2 rows: first row = experimental values, second row = errors.
              Columns must match df.columns in name and order.
    - title: Title of the whole figure.
    - figsize: Size of the full figure.
    - bins: Bin specification for histograms.
    """
    n_cols = len(df.columns)
    
    if len(labels) != n_cols or list(exp_df.columns) != list(df.columns):
        raise ValueError("Mismatch in number of columns, labels, or experimental DataFrame format.")
    
    n_rows = round(np.ceil(n_cols / 2))
    fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
    axes = axes.flatten()
    fig.suptitle(title, fontsize=16)

    for i, (col, label) in enumerate(zip(df.columns, labels)):
        ax = axes[i]
        data = df[col].dropna()
        data_exp= exp_df[col].dropna()
        if len(data) == 0 or len(data_exp) == 0:
            print(f"Warning: No data for column '{col}'. Skipping.")
            continue

        

        mean_val = np.mean(data)
        std_val = np.std(data)
        exp_value = np.mean(data_exp)
        exp_err = np.std(data_exp)
        sigmas = np.sqrt((exp_value - mean_val)**2 / (exp_err**2 + std_val**2))

        # Histograma
        n, bins_hist, _ = ax.hist(data, bins=bins, edgecolor='b', alpha=0.7)
        n2, bins_hist, _ = ax.hist(data_exp, bins=bins, edgecolor='r', alpha=0.7)

        # Experimental error
        rect = Rectangle((exp_value - exp_err, 0),
                         2 * exp_err,
                         np.max(n2),
                         color='r', alpha=0.3, label=f'± {exp_err:.2f}')
        ax.add_patch(rect)

        # Línea valor experimental
        ax.axvline(exp_value, color='r', linestyle='--', label=f'{exp_value:.2f}')

        # Rango del modelo teórico
        rect2 = Rectangle((mean_val - std_val, 0),
                         2 * std_val,
                         np.max(n),
                         color='b', alpha=0.3, label=f'± {std_val:.2f}')
        ax.add_patch(rect2)

        # Línea media modelo
        ax.axvline(mean_val, color='b', linestyle='--', label=f'{mean_val:.2f}')

        ax.set_title(f"{label}\nΔ = {sigmas:.2f} σ", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=8)

    for j in range(n_cols, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
```

## Data and variables


### Plancks Data

```python
unbin_cl = pd.read_csv('maps/COM_PowerSpect_CMB-TT-full_R3.01.txt', sep=',')

unbin_cl.head()

unbin_cl.columns = ['ell', 'D_ell', '-dD_ell', '+dD_ell']



ell = np.array(unbin_cl['ell'])
D_ell = np.array(unbin_cl['D_ell'])
errors = (np.array(unbin_cl['-dD_ell']),np.array(unbin_cl['+dD_ell']))

unbin_cl['Error']= (errors[0]+errors[1])/2

xvals = np.linspace(0.9999999999999999,-0.9999999999999999,1800) # For easy camb comparations 
theta= np.arccos(xvals)*180/np.pi

intervals = [(-0.9999999999999999, 0.5), (0.866, 0.9999999999999999), (0.5, 0.866), (0, 0.5), (-0.5, 0), (-0.866, -0.5), (-0.9999999999999999, -0.866)]

roots_planck = ['COM_CosmoParams_fullGrid_R3.01/base_omegak/CamSpecHM_TT_lowl_lowE/base_omegak_CamSpecHM_TT_lowl_lowE',
        'COM_CosmoParams_fullGrid_R3.01/base_omegak/plikHM_TTTEEE_lowl_lowE/base_omegak_plikHM_TTTEEE_lowl_lowE',
         'COM_CosmoParams_fullGrid_R3.01/base/CamSpecHM_TT_lowl_lowE/base_CamSpecHM_TT_lowl_lowE',
         'COM_CosmoParams_fullGrid_R3.01/base/plikHM_TT_lowl_lowE/base_plikHM_TT_lowl_lowE']



```

```python
print(xivar(D_ell[:200],0.9999999999999999,0.5))
print(xivar2(D_ell[:200],0.9999999999999999,0.5))
print(xivar_num(corr,0.9999999999999999,0.5))
```

Ejecutar solo 1 vez para crear los archivos Tmn

for i,j in intervals:
    Tmn(200,f"_{round(np.arccos(i)*180/np.pi)}",f"_{round(np.arccos(j)*180/np.pi)}",i,j)


### Drived and estimators for plancks data


#### Derived

```python
corr=correlation_func(D_ell[:200],xvals)
corr_err=correlation_func_err2(unbin_cl['Error'][:200],xvals)
```


#### Estimators

```python

exp_values=[corr[-1],corr_err[-1]]

for a, b in intervals:
    M = np.load(f"Tmn__{round(np.arccos(a)*180/np.pi)}__{round(np.arccos(b)*180/np.pi)}.npy") # Load the matrix
    s12 = S12(unbin_cl['D_ell'][:200],M) # Compute S12
    s12_err = S12_err2(unbin_cl['D_ell'][:200],unbin_cl['Error'][:200], M) # Compute S12 error
    exp_values.append(s12)
    exp_values.append(s12_err)
    
    xiv = xivar(unbin_cl['D_ell'][:200], a, b)
    xiv_err = xivar_err2(unbin_cl['Error'][:200], a, b)
    exp_values.append(xiv)
    exp_values.append(xiv_err)

print(exp_values)
```

#### Montecarlo simulation

```python



def MC_calculations(data,lmax,xvals,intervals):
    
    """
    Realiza cálculos Monte Carlo para las distribuciones de Cl y Cor, y calcula S12 y xiv.
    Parámetros:
    - data: DataFrame con las columnas 'D_ell' 
    - lmax: máximo valor de l para los cálculos
    - xvals: valores de x para la función de correlación
    - intervals: lista de tuplas con los intervalos [(a1, b1), (a2, b2), ...]
    """
    TTCl,  TTcor = data.iloc[0][:lmax], correlation_func(data.iloc[0][:lmax], xvals)
    C180 = TTcor[-1]
    th_values= []
    for a, b in intervals:
        M = np.load(f"Tmn__{round(np.arccos(a)*180/np.pi)}__{round(np.arccos(b)*180/np.pi)}.npy")
        s12 = S12(TTCl,M)
        th_values.append(s12)
        xiv = xivar(TTCl, a, b)
        th_values.append(xiv)

    return (TTCl, TTcor, C180, *th_values)

def MC_results(intervals, data,xvals,n=1000):
    """
    Realiza cálculos Monte Carlo para las distribuciones de Cl y Cor, y calcula S12 y xiv.
    
    Parámetros:
    - intervals: lista de tuplas con los intervalos [(a1, b1), (a2, b2), ...]
    - data: DataFrame con las columnas 'D_ell' y 'Error'
    - n: número de muestras para la simulación
    """
    

# Crear una nueva columna con las distribuciones

    data['dist_per_cl'] = data.apply(lambda row: np.random.normal(loc=row['D_ell'], scale=row['Error'], size=n), axis=1)
    distribuciones = data['dist_per_cl'].to_list()

# Transponer: cada lista interna tendrá los i-ésimos elementos de cada distribución
    trasp = list(map(list, zip(*distribuciones)))
    df_arrays = pd.DataFrame({'valores': [np.array(row) for row in trasp]})
    
    chain_cols = []
    chain_cols += ['D_ell', 'Cor']
    for a, b in intervals:
        chain_cols += [f's12_{round(np.arccos(a)*180/np.pi)}_{round(np.arccos(b)*180/np.pi)}',
                       f'xiv_{round(np.arccos(a)*180/np.pi)}_{round(np.arccos(b)*180/np.pi)}']
    chain_cols += ['C180']
    data_dict = {}

    
    try:
        
        chain_result = df_arrays.apply(MC_calculations, axis=1, args=(200, xvals, intervals))
        df_chain = pd.DataFrame(chain_result.tolist(), columns=chain_cols, index=df_arrays.index)
        df = pd.concat([df_chain, df_arrays], axis=1)

        stacked_Cl = np.vstack(df['D_ell'].values)
        stacked_Cor = np.vstack(df['Cor'].values)
        mean_Cl = stacked_Cl.mean(axis=0)
        std_Cl = stacked_Cl.std(axis=0,ddof=0)
        mean_Cor = stacked_Cor.mean(axis=0)
        std_Cor = stacked_Cor.std(axis=0,ddof=0)

        data_dict['Simulation'] = (df[chain_cols], mean_Cl, std_Cl, mean_Cor, std_Cor)

    except Exception as e:
        print(f'Error: {e}')
        

    finally:
        
        # Guardar todo el diccionario completo en un solo archivo
        with open(f'Simulation_1000.pkl', 'wb') as f:
            pickle.dump(data_dict, f)




```

```python
# MC_results(intervals, unbin_cl,xvals,1000)
```

```python
with open('Simulation_1000.pkl', 'rb') as f:
    data_dict = pickle.load(f)

# Recorrer cada entrada del diccionario

for root_name, (df, mean_Cl, std_Cl, mean_Cor, std_Cor) in data_dict.items():
    xiv_df = df.filter(regex=r'xiv_')
    s12_df = df.filter(regex=r's12_')
    est_df = df.copy()
    est_df.drop(columns=['D_ell', 'Cor'], inplace=True)
       
    #create_histogram_grid(est_df, est_df.columns, exp_values, f"Histograms for {root_name}", figsize=(15, 15), bins=100)
    #plot_corr_with_xivar(xvals, unbin_cl,mean_Cor, xiv_df, intervals,root_name)
    #plot_corr_with_S12(xvals, unbin_cl,mean_Cor, s12_df, intervals,root_name)


    #plot_power_and_correlation(unbin_cl, mean_Cl, std_Cl,
    #                            xvals, mean_Cor, std_Cor,
    #                            root=root_name, figsize=(18, 7))
```

### Chain calculations

```python
#chain_results(intervals, roots_planck,'chain_results_planck',1000) # No need to execute this if you have chain_results_planck.pkl (4 chains) or chain_results.pkl (1 chain)
```

```python
with open('chain_results_planck.pkl', 'rb') as f1:
    data_dict = pickle.load(f1)
with open('Simulation_1000.pkl', 'rb') as f2:
    simu_dict = pickle.load(f2)

df_simu, cl_simu, _, _, _ = simu_dict['Simulation']
df2 = df_simu.copy()

xiv_simu_df = df_simu.filter(regex=r'xiv_')
s12_simu_df = df_simu.filter(regex=r's12_')
df_simu.drop(columns=['D_ell', 'Cor'], inplace=True)
# Recorrer cadacentrada del diccionario
for root_name, (df, mean_Cl, std_Cl, mean_Cor, std_Cor) in data_dict.items():
    xiv_df = df.filter(regex=r'xiv_')
    s12_df = df.filter(regex=r's12_')
    est_df = df.copy()
    est_df.drop(columns=['D_ell', 'Cor'], inplace=True)
    
    
    create_histogram_grid(est_df, est_df.columns, exp_values, f"Histograms for {root_name} Experimental VS Theoretical", figsize=(15, 15), bins=100)
    create_histogram_grid(df_simu, df_simu.columns, exp_values, f"Histograms for {root_name} Experimental VS Gaussian Simulation", figsize=(15, 15), bins=100)    

    
    create_histogram_grid2(df_simu, df_simu.columns, est_df, f"Histograms for {root_name} Theoretical VS Gaussian Simulation", figsize=(15, 15), bins=100)

    #plot_corr_with_xivar(xvals, df2, mean_Cor, xiv_simu_df, intervals, root_name)
    #plot_corr_with_S12(xvals, df2, mean_Cor, s12_simu_df, intervals, root_name)
#
    #plot_power_and_correlation(df2, mean_Cl, std_Cl,
    #                           xvals, mean_Cor, std_Cor,
    #                            root=root_name, figsize=(18, 7))#
```




import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import matplotlib.colors as mcolors

# 定义画图相关的函数

def color_inv_alpha(color,bgcolor,alpha):
    '''
    Input:
        color: foregroud color, hex/rgb
        bgcolor: background color, hex/rgb
        alpha
    Output:
        color'
    Calculation:
        color'(rgb) = (C_fg - (1-alpha)*C_bg)/alpha
    '''
    if isinstance(color,str):
        color = mcolors.hex2color(color)
    elif np.max(color)>1:
        color/=255
    
    if isinstance(bgcolor,str):
        color = mcolors.hex2color(bgcolor)
    elif np.max(bgcolor)>1:
        bgcolor/=255
                
    color1=[]
    for i in range(3):
        c=(color[i]-(1-alpha)*bgcolor[i])/alpha
        color1.append(np.clip(c,0,1))

    return color1


def stat_m_e(data,mtype,etype):
    '''
    Input:
        data: [nsamples, ngroup]
        mtype: mean/median
        etype: std/sem/68%CI
    Output:
        data statistics
    '''

    if mtype == 'mean':
        mdata = np.nanmean(data,axis=0)
    elif mtype == 'median':
        mdata = np.nanmedian(data,axis=0)

    if etype == 'std':
        edata = np.nanstd(data,axis=0)
    elif etype == 'sem':
        edata = stats.sem(data,axis=0,nan_policy='omit')
    elif etype == 'ci':
        if mtype == 'mean':
            ci = stats.bootstrap((data,),np.nanmean,axis=0,confidence_level=0.68,method='percentile').confidence_interval
        elif mtype == 'median':
            ci = stats.bootstrap((data,),np.nanmedian,axis=0,confidence_level=0.68,method='percentile').confidence_interval
        edata = np.array([mdata-ci[0],ci[1]-mdata])
        
    return [mdata,edata]

def fisherztrans(r):
    '''
    Input:
        r: Correlation coefficient
    Output:
        Fisher z transformation
    '''
    return 0.5*np.log((1+r)/(1-r))

def pair_test(data,method='wilcoxon',correction='none'):
    '''
    Input:
        data: sample x group x condition[2]
        method: 
            small sample / non-normal distribution --- 'wilcoxon'
            normal distribution --- 'ttest_rel'
        correction:
            'bonferroni': p/ntask
            'fdr_bh': large sample, less strict
            'none'
    Output:
        p value
    '''
    ng = data.shape[1]
    ps = []
    for g in range(ng):
        if method == 'wilcoxon':
            [s,p] = stats.wilcoxon(data[:,g,0],data[:,g,1],nan_policy='omit')
        elif method == 'ttest_rel':
            [s,p] = stats.ttest_rel(data[:,g,0],data[:,g,1],nan_policy='omit')
        elif method =='sign':
            differences = data[:,g,0] - data[:,g,1]
            pos_diffs = np.sum(differences>0)
            neg_diffs = np.sum(differences<0)
            total_pairs = pos_diffs+neg_diffs
            p = stats.binomtest(min(pos_diffs,neg_diffs),n=total_pairs,alternative='two-sided').pvalue            
        ps.append(p)
    ps = np.array(ps)
    if correction == 'none':
        return ps
    else:
        ps_corrected = multipletests(ps,method = correction)[1]
        return ps_corrected
    

def sig(p,sig_level=[0.05,0.01,0.001]):
    '''
    Input: 
        p: p value
        sig_level: significant threshold
    Output:
        significence: ***/**/*/n.s.
    '''
    if p < sig_level[2]:
        star = 3*'*'
    elif p < sig_level[1]:
        star = 2*'*'
    elif p < sig_level[0]:
        star = 1*'*'
    else:
        star = 'n.s.'
    return star
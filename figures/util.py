import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import pingouin
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

    if etype == 'ci':
        if mtype == 'mean':
            res = stats.bootstrap((data,),np.nanmean,n_resamples=10000,axis=0,confidence_level=0.68,method='percentile')
            ci = res.confidence_interval
            bsamples = res.bootstrap_distribution.transpose(1,0)
        elif mtype == 'median':
            res = stats.bootstrap((data,),np.nanmedian,n_resamples=10000,axis=0,confidence_level=0.68,method='percentile')
            ci = res.confidence_interval
            bsamples = res.bootstrap_distribution.transpose(1,0)
        edata = np.array([mdata-ci[0],ci[1]-mdata])
    else:
        bsamples = []
        if etype == 'std':
            edata = np.nanstd(data,axis=0)
        elif etype == 'sem':
            edata = stats.sem(data,axis=0,nan_policy='omit')
    
    return [mdata,edata,bsamples]
    
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
        stats: full stats (pd.DataFrame or object)
    '''
    ng = data.shape[1]
    ps = []
    for g in range(ng):
        if method == 'wilcoxon':
            res = pingouin.wilcoxon(data[:,g,0],data[:,g,1])
            p = res['p-val'].iloc[0]
        elif method == 'ttest_rel':
            res = pingouin.ttest(data[:,g,0],data[:,g,1],paired=True)
            p = res['p-val'].iloc[0]
        elif method =='sign':
            differences = data[:,g,0] - data[:,g,1]
            pos_diffs = np.sum(differences>0)
            neg_diffs = np.sum(differences<0)
            total_pairs = pos_diffs+neg_diffs
            res = stats.binomtest(min(pos_diffs,neg_diffs),n=total_pairs,alternative='two-sided')
            res.ci = res.proportion_ci(confidence_level=0.95)
            p = res.pvalue            
        ps.append(p)
    ps = np.array(ps)
    
    if correction == 'none':
        ps_f = ps
    else:
        ps_f = multipletests(ps,method = correction)[1]
    
    return ps_f,res
    

def sig(p,fmt='text'):
    '''
    Input: 
        p: p value
        sig_level: significant threshold
        fmt: "text"--text; "star"--star
    Output:
        significence: ***/**/*/n.s.
    '''
    if fmt == 'star':
        sig_level=[0.05,0.01,0.001]
        if p < sig_level[2]:
            s = 3*'*'
        elif p < sig_level[1]:
            s = 2*'*'
        elif p < sig_level[0]:
            s = 1*'*'
        else:
            s = 'n.s.'
    else:
        sig_level=[0.05,0.001]
        if p < sig_level[1]:
            s = "p < .001"
            s = "***"
        elif p < sig_level[0]:
            s = "p = "+"{:.3f}".format(p)[1:]
        else:
            s = "n.s."

    return s
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import pingouin
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import matplotlib.colors as mcolors
import pingouin as pg

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
    

def paired_ttest_full(x, y, method='wilcoxon'):
    """
    paired test FULL STATISTICS
    
    output: DataFrame
        
        paired-ttest:
            - t, df, p, cohen's d, 95% CI of cohen's d
            - mean & SE
            - mean_diff = mean_cond1 - mean_cond2

        wilcoxon-test:
            - 
            - mean &  SE
            - mean_diff = mean_cond1 - mean_cond2
            
        
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    
    df = pd.DataFrame({'cond1': x, 'cond2': y})
    
    # mean &  SE
    mean1 = df['cond1'].mean()
    mean2 = df['cond2'].mean()
    se1 = df['cond1'].sem()
    se2 = df['cond2'].sem()
    
    mean_diff = mean1 - mean2
    
    if method == 'wilcoxon':
        wilcoxon_res = pg.wilcoxon(x, y)

        # matched pairs rank-biserial correlation (effect size)
        W_val = wilcoxon_res['W-val'].iloc[0]
        p_val = wilcoxon_res['p-val'].iloc[0]
        RBC = wilcoxon_res['RBC'].iloc[0]

        report = pd.DataFrame({
            'mean_diff': [mean_diff],
            'W': [W_val],
            'p': [p_val],
            'RBC': [RBC],
            'mean1': [mean1],
            'SE1': [se1],
            'mean2': [mean2],
            'SE2': [se2],        
        })
        
        return report


    elif method == 'ttest_rel':

        ttest_res = pg.ttest(df['cond1'], df['cond2'], paired=True)
        
        t_val = ttest_res['T'].iloc[0]
        df_val = ttest_res['dof'].iloc[0]
        p_val = ttest_res['p-val'].iloc[0]
        cohen_d = ttest_res['cohen-d'].iloc[0]
        

        report = pd.DataFrame({
            'mean_diff': [mean_diff],
            't': [t_val],
            'df': [df_val],
            'p': [p_val],
            'cohen_d': [cohen_d],
            'mean1': [mean1],
            'SE1': [se1],
            'mean2': [mean2],
            'SE2': [se2],        
        })
        
        return report
        

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



def corr_batch_nan(X,ctype='corr',ddof=1):

    """
    Compute pairwise covariance or Pearson correlation with NaN handling.
    
    Parameters
    ----------
    X : np.ndarray, shape (..., N, T)
        Input data, last two dims are variables and time points
    ctype : str, 'cov' or 'corr'
        Return covariance or correlation
    ddof : int
        Delta degrees of freedom
    
    Returns
    -------
    out : np.ndarray, shape (..., N, N)
        Pairwise covariance or correlation
    """

    mask = (~np.isnan(X)).astype(np.int64)
    X0 = np.where(mask,X,0.0)
    
    # pairwise valid count
    n = np.einsum('...iT,...jT->...ij',mask,mask)
    
    # pairwise sum
    sum_x = np.einsum('...iT,...jT->...ij',X0,mask)
    sum_y = np.einsum('...iT,...jT->...ij',mask,X0)

    # pairwise mean
    mean_x = np.divide(sum_x, n, out=np.full_like(sum_x, np.nan), where=n>0)
    mean_y = np.divide(sum_y, n, out=np.full_like(sum_y, np.nan), where=n>0)

    # pairwise covariance numerator
    sum_xy = np.einsum('...iT,...jT->...ij', X0, X0)
    cov_num  = sum_xy - mean_x * sum_y - mean_y * sum_x + n * mean_x * mean_y

    cov = np.divide(cov_num, n - ddof, out=np.full_like(cov_num, np.nan), where=n>ddof)

    
    if ctype == 'cov':
        
        return cov
    

    elif ctype == 'corr':

        sum_x2 = np.einsum('...iT,...jT->...ij', X0**2, mask)
        sum_y2 = np.einsum('...iT,...jT->...ij', mask, X0**2) 

        # pairwise variance
        var_x = (sum_x2 - 2 * mean_x * sum_x + n * mean_x**2) / (n - ddof)
        var_y = (sum_y2 - 2 * mean_y * sum_y + n * mean_y**2) / (n - ddof)
        denom = np.sqrt(var_x * var_y)
        corr = np.divide(cov, denom, out=np.full_like(cov, np.nan), where=denom>0)
        
        return corr

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from matplotlib_inline.backend_inline import set_matplotlib_formats
set_matplotlib_formats('retina')


def myvar(X,m):
    # X: (varaible, observation)
    # m: (varaible,)
    X = np.mat(X)
    m = np.mat(m)
    X_dm = X-m.T
    c = np.var(X_dm,axis=1)
    return c

def mycov(X,m):
    # X: (varaible, observation)
    # m: (varaible,)
    X = np.mat(X)
    m = np.mat(m)
    X_dm = X-m.T
    T = X.shape[1]
    c = np.dot(X_dm,X_dm.T)/(T-1)
    return c

def myinv(q):
    qinv = np.zeros(q.shape)*np.nan
    nanind = np.max(np.where(~np.isnan(q[:,0])))+1
    q1 = q[0:nanind,0:nanind]
    q1inv = np.linalg.inv(q1)
    qinv[0:nanind,0:nanind] = q1inv
    return qinv,nanind

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


def fisherztrans(r):
    '''
    z_r = np.zeros((r_matrix.shape[0],r_matrix.shape[1]))
    for i in range(r_matrix.shape[0]):
        for j in range(r_matrix.shape[1]):
            r = r_matrix[i,j]
            z_r[i,j] = 0.5*np.log((1+r)/(1-r))
    return z_r
    '''    
    return 0.5*np.log((1+r)/(1-r))

def stat_m_e(data,mtype,etype):
    if mtype == 'mean':
        mdata = np.nanmean(data,axis=0)
    elif mtype == 'median':
        mdata = np.nanmedian(data,axis=0)

    if etype == 'std':
        edata = np.nanstd(data,axis=0)
    elif etype == 'sem':
        edata = stats.sem(data,axis=0,nan_policy='omit')
    elif etype == 'ci':
        ci = stats.bootstrap((data,),np.nanmedian,axis=0,confidence_level=0.68,method='percentile').confidence_interval
        edata = np.array([mdata-ci[0],ci[1]-mdata])
        
    return [mdata,edata]

def paired_mean_diff(x, y):
    return np.mean(x - y)

def pair_test(data,method='wilcoxon',correction='none',pround=5):
    '''
    data: sample x group x condition[2]
    method: 
        small sample / non-normal distribution --- 'wilcoxon'
        normal distribution --- 'ttest_rel'
    correction:
        'bonferroni': p/ngroup
        'fdr_bh': large sample, less strict
        'none'
    pround: int
        Round to a certain number of decimal places.
        'none': no
    '''
    ng = data.shape[1]
    ps = []
    for g in range(ng):
        if method == 'wilcoxon':
            if (data[:,g,0]-data[:,g,1]==0).all():
                p = 1
            else:
                [s,p] = stats.wilcoxon(data[:,g,0],data[:,g,1],nan_policy='omit')
        elif method == 'ttest_rel':
            [s,p] = stats.ttest_rel(data[:,g,0],data[:,g,1],nan_policy='omit')
        elif method == 'permutation':
            p = stats.permutation_test((data[:,g,0],data[:,g,1]),statistic=paired_mean_diff,permutation_type='samples',n_resamples = 10000).pvalue
        if np.isnan(p):
            p = 1
        ps.append(p)
    ps = np.array(ps)
    if correction == 'none':
        ps_report = ps
        
    else:
        ps_corrected = multipletests(ps,method = correction)[1]
        ps_report = ps_corrected

    if pround != 'none':
        ps_report = [round(p,pround) for p in ps_report]

    return ps_report
     
def one_sample_test(data, popmean=0, method='ttest_1samp', correction='none',pround=5):
    '''
    对多组数据进行单样本检验。

    参数：
    ----------
    data : array_like
        shape = (sample, group)
        每列代表一组（例如不同条件或脑区），每行是被试。
    popmean : float, optional
        假设总体均值（默认 0）。
    method : str, optional
        'ttest_1samp' — 正态分布假设下的单样本 t 检验
        'wilcoxon' — 非参数单样本检验（检验中位数是否等于 popmean）
        'permutation' — 置换检验（需要较多样本）
    correction : str, optional
        'bonferroni' — Bonferroni 校正
        'fdr_bh' — FDR (Benjamini-Hochberg)
        'none' — 不进行校正
    pround: int, optional
        Round to a certain number of decimal places.
        'none' - 不进行取整
    返回：
    ----------
    ps : ndarray
        每组对应的 p 值（若有 correction 则返回校正后的 p 值）
    '''
    ng = data.shape[1]
    ps = []

    for g in range(ng):
        d = data[:, g]
        d = d[~np.isnan(d)]  # 去掉 NaN

        if len(d) == 0:
            p = 1
        elif method == 'ttest_1samp':
            _, p = stats.ttest_1samp(d, popmean)
        elif method == 'wilcoxon':
            if np.all(d - popmean == 0):
                p = 1
            else:
                _, p = stats.wilcoxon(d - popmean)
        elif method == 'permutation':
            def mean_diff(x):
                return np.mean(x)
            p = stats.permutation_test(
                (d,), statistic=mean_diff,
                permutation_type='one-sample',
                alternative='two-sided',
                n_resamples=10000
            ).pvalue
        else:
            raise ValueError(f"Unknown method: {method}")

        if np.isnan(p):
            p = 1
        ps.append(p)

    ps = np.array(ps)

    if correction == 'none':
        ps_report = ps
        
    else:
        ps_corrected = multipletests(ps,method = correction)[1]
        ps_report = ps_corrected

    if pround != 'none':
        ps_report = [round(p,pround) for p in ps_report]

    return ps_report

def test_norm(data,sampletype=1):
    '''
    sampletype: 1-large sample,0-small sample
    return: 1-norm, 2-not norm
    '''
    # 正态性检验
    if sampletype==0:
        stat, p = stats.shapiro(data)
    else:
        stat, p = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data)))
    
    if p > 0.05: # 正态
        return 1
    else:
        return -1

def sig(p,sig_level=[0.05,0.01,0.001]):
    if p < sig_level[2]:
        star = 3
    elif p < sig_level[1]:
        star = 2
    elif p < sig_level[0]:
        star = 1
    else:
        star = 0
    return star

def mybarplot(data,mtype,etype,cm,xlbl,ylbl,xtk,lg,tlt,ifscatter,ifpairedline,xgap=1,bartype='paired'):
    '''
    bartype: 'paired'
        data: nsample x nxl x ngroup
        color: ngroup x 3
    bartype: 'ind'
        data: nsample x nxl
    '''
    if bartype == 'paired':
        [nsample,nxl,ngroup] = data.shape
        w = 0.8/ngroup*xgap
        if mtype == 'mean': mdata = np.nanmean(data,axis=0)
        elif mtype == 'median': mdata = np.nanmedian(data,axis=0)
        if etype == 'std': edata = np.nanstd(data,axis=0).reshape((1,nxl,ngroup))
        elif etype == 'sem': edata = stats.sem(data,axis=0,nan_policy='omit').reshape((1,nxl,ngroup))
        elif etype == 'ci': ci = stats.bootstrap((data,),np.nanmedian,axis=0,confidence_level=0.68,method='percentile').confidence_interval; edata = np.array([mdata-ci[0],ci[1]-mdata])
        x = np.linspace(1,nxl*xgap,nxl)
        for ng in range(ngroup):
            plt.bar(x+(-0.4*xgap+w*(ng+0.5)),mdata[:,ng],yerr=edata[:,:,ng],width=w,color=cm[ng,:],alpha=0.5)
            if ifscatter: plt.scatter(nsample*[xx+(-0.4*xgap+w*(ng+0.5)) for xx in x],data[:,:,ng],color=cm[ng,:],s=1/xgap)
        if ifpairedline:
            for nx in range(nxl): plt.plot([nsample*[1+nx+(-0.4*xgap+w*(ng+0.5))] for ng in range(ngroup)],data[:,nx,:].T,color='grey',linewidth=1/xgap)
        plt.xlabel(xlbl);    plt.ylabel(ylbl);    plt.xticks(x,xtk);    plt.legend(lg);    plt.title(tlt)
    elif bartype == 'ind':
        [nsample,nxl] = data.shape
        w = 0.8*xgap
        if mtype == 'mean': mdata = np.nanmean(data,axis=0)
        elif mtype == 'median': mdata = np.nanmedian(data,axis=0)
        if etype == 'std': edata = np.nanstd(data,axis=0)
        elif etype == 'sem': edata = stats.sem(data,axis=0,nan_policy='omit')
        elif etype == 'ci': ci = stats.bootstrap((data,),np.nanmedian,axis=0,confidence_level=0.68,method='percentile').confidence_interval; edata = np.array([mdata-ci[0],ci[1]-mdata])
        x = np.linspace(1,nxl*xgap,nxl)
        plt.bar(x,mdata,yerr=edata,width=w,color=cm,alpha=0.5)
        if ifscatter: plt.scatter(nsample*[xx for xx in x],data,color=cm,s=1/xgap)
        if ifpairedline: 
            #plt.plot(np.array([np.arange(nsample)+1 for _ in range(nxl)]).T,data,color='grey',linewidth=1/xgap)
            for sample_i in range(nsample):
                plt.plot(np.arange(nxl)+1,data[sample_i,:],color='k',linewidth=1/xgap,alpha=(sample_i+1)/(nsample+1))
        plt.xlabel(xlbl);    plt.ylabel(ylbl);    plt.xticks(x,xtk);   plt.title(tlt)


def myheatmap(ax,data,cmap,vm,xlbl,ylbl,xtk,ytk,tlt,cblbl,iftxt):
    # data: nrow x ncol
    if cmap=='-+':
        cm = 'PuOr_r'
    else:
        cm = 'YlGn'
    if vm==None:
        if cm == 'YlGn':
            vm = [np.nanmin(data),np.nanmax(data)]
        else:
            vm = max(abs(np.nanmin(data)),abs(np.nanmax(data)))
            vm=[-vm,vm]
    im = plt.imshow(data,cmap=cm,vmin=vm[0],vmax=vm[1])
    plt.xticks(np.arange(data.shape[1]),xtk)
    plt.yticks(np.arange(data.shape[1]),ytk)
    plt.tick_params(top=False,bottom=False,left=False,right=False,labeltop=True,labelbottom=False)
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    plt.title(tlt)
    if iftxt:
        for i in range(data.shape[1]):
            for j in range(data.shape[0]):
                if data[j,i]!=0:
                    plt.text(i,j,round(data[j,i],2),ha='center',va='center',color='grey')
    cbar = ax.figure.colorbar(im,ax=ax)
    cbar.ax.set_ylabel(cblbl,rotation=-90,va='bottom')

def myviolinplot(data,cm,xg,hg,xlbl,ylbl,lg,tlt):
    # data: nobv x nxl x ngroup
    [nobv,nxl,ngroup] = data.shape
    d_x = np.array([])
    d_y = np.array([])
    d_g = np.array([])
    for ng in range(ngroup):
        for nx in range(nxl):
            d_x = np.append(d_x,[xg[nx]]*nobv)
            d_y = np.append(d_y,data[:,nx,ng])
            d_g = np.append(d_g,[hg[ng]]*nobv)
    dataset = pd.DataFrame({ylbl:d_y,xlbl:pd.Series(d_x,dtype='category'),lg:pd.Series(d_g,dtype='category')})
    sns.violinplot(data=dataset, x=xlbl, y=ylbl, hue=lg, split=True, inner="quart")
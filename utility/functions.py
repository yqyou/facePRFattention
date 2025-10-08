import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
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

def pair_test(data,method='wilcoxon',correction='none'):
    '''
    data: sample x group x condition[2]
    method: 
        small sample / non-normal distribution --- 'wilcoxon'
        normal distribution --- 'ttest_rel'
    correction:
        'bonferroni': p/ngroup
        'fdr_bh': large sample, less strict
        'none'
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
        return ps
    else:
        ps_corrected = multipletests(ps,method = correction)[1]
        return ps_corrected
     
def test_norm(data,sampletype=1):
    # sampletype: 1-large sample,0-small sample
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

def mybarplot(data,mtype,etype,cm,xlbl,ylbl,xtk,lg,tlt,ifscatter,ifpairedline,xgap=1):
    '''
    data: nsample x nxl x ngroup
    color: ngroup x 3
    '''
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
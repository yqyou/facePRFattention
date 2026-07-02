import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import scipy.stats as stats
import pingouin
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import matplotlib.colors as mcolors
from util import stat_m_e,sig
#from RZutilpy.figure import default_img_set

cms=np.array([[255, 195, 73],[145, 115, 185]])/255 # 黄、紫
cms4=['#ed9f98','#8dcc8c','#5ea6db',cms[1]]
cms3=['#e15f41','#c44569','#5ea6db']

def set_figure():
    #default_img_set()
    # figure
    mpl.rcParams['font.family'] = 'Arial' #字体
    mpl.rcParams['font.size'] = 13 #字号
    mpl.rcParams['font.weight'] = 'regular' #粗体
    mpl.rcParams['axes.labelweight'] = 'normal'
    mpl.rcParams['legend.frameon'] = True
    mpl.rcParams['mathtext.fontset'] = "custom" # supported values are ['dejavusans', 'dejavuserif', 'cm', 'stix', 'stixsans', 'custom']

def set_ax(ax):
    ax.tick_params(axis='both', width=1, direction='out') # , bottom=False
    ax.tick_params(axis='x', rotation=0, width=1, direction='out') # , bottom=False
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)


def mybar_pair(ax, data_all, mtype, etype, ps, py):

    nconditions = data_all.shape[2]
    xgap = 1; nxl = data_all.shape[1]; w = 0.8/nconditions*xgap
    x = np.linspace(xgap,xgap*nxl,nxl)
    data_m,data_e = np.zeros([data_all.shape[1],data_all.shape[2]]),np.zeros([2,data_all.shape[1],data_all.shape[2]])

    for ng in range(nconditions):
        data = data_all[:,:,ng]
        [data_m[:,ng],data_e[:,:,ng],bsample] = stat_m_e(data,mtype=mtype,etype=etype)
        ax.bar(x+(-0.4*xgap+w*(ng+0.5)),data_m[:,ng],yerr=data_e[:,:,ng],
                     width=w,color=cms[ng],alpha=0.5,
                     error_kw=dict(lw=2,ecolor=np.array([1,1,1])*0))
    ax.set_xticks(x)

    # 加显著性
    sig_plot(ax,x,w*2,ps,py,data_m,pair=True)

def myboxplot_pair(ax,data_all,mtype,etype,ps,py,defaultci=True):
    '''
    ax
    data_all: nsample x ngroup x ncondition
    default_ci: 95%CI, 10000 bootstrap
    '''
    nconditions = data_all.shape[2]
    nxl = data_all.shape[1]; w = 0.3
    x = np.linspace(1,nxl,nxl)
    data_m,data_e = np.zeros([data_all.shape[1],data_all.shape[2]]),np.zeros([2,data_all.shape[1],data_all.shape[2]])
    for ng in range(nconditions):
        data = data_all[:,:,ng]
        [data_m[:,ng],data_e[:,:,ng],bsamples] = stat_m_e(data,mtype=mtype,etype=etype)
        ci = np.array([data_m[:,ng]-data_e[0,:,ng],data_m[:,ng]+data_e[1,:,ng]]).T
        if defaultci:
            ax.boxplot(data, positions = x-w*0.55+w*1.1*ng, notch = True, bootstrap=10000,
                    widths = w, showcaps = True, showbox = True, showfliers = False,
                    boxprops = dict(facecolor=cms[ng],edgecolor=cms[ng],alpha=0.5), 
                    whiskerprops = dict(color=cms[ng]), capprops = dict(color=cms[ng]),
                    medianprops = dict(color='k',linewidth=1.5), patch_artist=True)
        else:
            ax.boxplot(data, positions = x-w*0.55+w*1.1*ng, notch = True, conf_intervals = ci,
                    widths = w, showcaps = True, showbox = True, showfliers = False,
                    boxprops = dict(facecolor=cms[ng],edgecolor=cms[ng],alpha=0.5), 
                    whiskerprops = dict(color=cms[ng]), capprops = dict(color=cms[ng]),
                    medianprops = dict(color='k',linewidth=1.5), patch_artist=True)
    ax.set_xticks(x)

    # 加显著性
    sig_plot(ax,x,w*2,ps,py,data_m,pair=True,data_all = data_all)

         

def myviolinplot_pair(ax, data_all, mtype, etype, ps, py, cm=None):
    '''
    ax
    data_all: nsample x ngroup x ncondition
    '''
    if not cm:
        if data_all.shape[2] == 2:
            cm = cms
        else:
            cm = cms4
    nconditions = data_all.shape[2]
    sides = ['low','high']
    xgap = 1; nxl = data_all.shape[1]; w = 1.8/data_all.shape[2]*xgap
    x = np.linspace(xgap,xgap*nxl,nxl)
    data_m,data_e = np.zeros([data_all.shape[1],data_all.shape[2]]),np.zeros([2,data_all.shape[1],data_all.shape[2]])
    for ng in range(nconditions):

        if etype == "ci":
            data_orig = data_all[:,:,ng]
            [data_m[:,ng],data_e[:,:,ng],bsamples] = stat_m_e(data_orig,mtype=mtype,etype=etype)
            data = bsamples.copy()
        else:
            data = data_all[:,:,ng]
            [data_m[:,ng],data_e[:,:,ng],bsamples] = stat_m_e(data,mtype=mtype,etype=etype)

        
        # 填充部分
        vp1 = ax.violinplot(data, positions = x-(1-ng*2)*0.04, vert = True, widths = w*0.85, 
                        points = 100, bw_method = 'scott', side = sides[ng],
                    showmeans = False, showextrema = False, showmedians = False, quantiles = None)
        for b1 in vp1['bodies']:
            b1.set_edgecolor("none")
            b1.set_facecolor(cm[ng])
            b1.set_alpha(0.2)

        # 加外轮廓
        vp1 = ax.violinplot(data, positions = x-(1-ng*2)*0.04, vert = True, widths = w*0.85, 
                            points = 100, bw_method = 'scott', side = sides[ng],
                        showmeans = False, showextrema = False, showmedians = False, quantiles = None)
        for b1 in vp1['bodies']:
            b1.set_edgecolor(cm[ng])
            b1.set_facecolor("none")
            b1.set_alpha(0.7)

        # 加errorbar
        ax.errorbar(x-(1-ng*2)*0.05,data_m[:,ng],data_e[:,:,ng],
                            color='k',marker='',linestyle='',elinewidth=1.2)
        ax.plot([x-(1-ng*2)*0.12,x-(1-ng*2)*0.05],[data_m[:,ng],data_m[:,ng]],
                            color='k',linewidth=1.2)
        
        # 加散点
        if etype != "ci":
            jitter = np.random.uniform(0,w/8,data.shape)
            ax.scatter(np.zeros_like(data)+x-(1-ng*2)*(0.15+jitter), data, 
                    color = cm[ng], alpha = 1, s = 15, edgecolors = 'white',linewidths=0.5)
    ax.set_xticks(x)

    # 加显著性
    sig_plot(ax,x,w,ps,py,data_m,pair=True,data_all = data_all)


def myviolinplot_ind(ax, data, mtype, etype, ps, py,color='grey'):
    '''
    ax
    data_all: nsample x ngroup
    ps: list of p-value
    py: [a,b]; ycoor of p-value text and line.
    '''
    xgap = 1; nxl = data.shape[1]; w = 0.8*xgap
    x = np.linspace(xgap,xgap*nxl,nxl)
    [data_m,data_e,bsample] = stat_m_e(data,mtype=mtype,etype=etype)
    
    # 填充部分
    vp1 = ax.violinplot(data, positions = x, vert = True, widths = w*0.85, 
                    points = 100, bw_method = 'scott', side = "both",
                showmeans = False, showextrema = False, showmedians = False, quantiles = None)
    for b1 in vp1['bodies']:
        b1.set_edgecolor("none")
        b1.set_facecolor(color)
        b1.set_alpha(0.2)

    # 加外轮廓
    vp1 = ax.violinplot(data, positions = x, vert = True, widths = w*0.85, 
                        points = 100, bw_method = 'scott', side = "both",
                    showmeans = False, showextrema = False, showmedians = False, quantiles = None)
    for b1 in vp1['bodies']:
        b1.set_edgecolor(color)
        b1.set_facecolor("none")
        b1.set_alpha(0.7)

    # 加errorbar
    ax.errorbar(x,data_m,data_e,color='k',marker='o',linestyle='',elinewidth=1.8,markersize=5)

    # 加散点
    jitter = np.random.uniform(-w/4,w/4,data.shape)
    ax.scatter(np.zeros_like(data)+x+jitter, data, 
                color = color, alpha = 1, s = 15, edgecolors = 'white',linewidths=0.5)

    ax.set_xticks(x)

    # 加显著性
    if len(ps) > 0:
        sig_plot(ax,x,w,ps,py,data_m,pair=False,data_all = data)


def myviolinplot_multi(ax, data, mtype, etype, ps, py):
    '''
    ax
    data_all: nsample x ngroup
    ps: list of p-value
    py: [a,b]; ycoor of p-value text and line
    '''
    xgap = 1; nxl = data.shape[1]; w = 0.85*xgap
    x = np.linspace(xgap,xgap*nxl,nxl)

    for ti in range(nxl):

        [data_m,data_e,bsample] = stat_m_e(data[:,ti],mtype=mtype,etype=etype)
        
        if nxl == 4:
            c = cms4[ti]
        elif nxl == 3:
            c = cms3[ti]
        else:
            c = "grey"

        # 填充部分
        vp1 = ax.violinplot(data[~np.isnan(data[:,ti]),ti], positions = [x[ti]], vert = True, widths = w*0.85, 
                        points = 100, bw_method = 'scott', side = "both",
                    showmeans = False, showextrema = False, showmedians = False, quantiles = None)
        for b1 in vp1['bodies']:
            b1.set_edgecolor("none")
            b1.set_facecolor(c)
            b1.set_alpha(0.2)

        # 加外轮廓
        vp1 = ax.violinplot(data[~np.isnan(data[:,ti]),ti], positions = [x[ti]], vert = True, widths = w*0.85, 
                            points = 100, bw_method = 'scott', side = "both",
                        showmeans = False, showextrema = False, showmedians = False, quantiles = None)
        for b1 in vp1['bodies']:
            b1.set_edgecolor(c)
            b1.set_facecolor("none")
            b1.set_alpha(0.7)

        # 加errorbar
        ax.errorbar(x[ti],data_m,data_e,color='k',marker='o',linestyle='',elinewidth=1.8,markersize=5)

        # 加散点
        jitter = np.random.uniform(-w/4,w/4,data.shape[0])
        ax.scatter(np.zeros_like(data[:,ti])+x[ti]+jitter, data[:,ti], 
                    color = c, alpha = 0.7, s = 20, edgecolors = 'white',linewidths=0.5)

    ax.set_xticks(x)

    # 加显著性
    sig_plot(ax,x,w,ps,py,data_m,pair=False,data_all = data)


def myviolinplot_group(ax, data_all, mtype, etype, group_labels, cond_labels,colors,legendtitle=None):
    """
    data_all: nsample x ngroup x ncondition
    (ps: ngroup>2)
    """

    _, ngroup, ncondition = data_all.shape

    group_gap = 1
    cond_gap = 1.1
    width = 0.8

    alphas = np.linspace(0.3, 0.9, ngroup)

    data_m,data_e = np.zeros([data_all.shape[1],data_all.shape[2]]),np.zeros([2,data_all.shape[1],data_all.shape[2]])
    for cond_i in range(ncondition):
        base_x = cond_i * (ngroup*group_gap + cond_gap)

        if etype == "ci":
            [data_m[:,cond_i],data_e[:,:,cond_i],bsamples] = stat_m_e(data_all[:,:,cond_i],mtype=mtype,etype=etype)
            data = bsamples.copy()
        else:
            [data_m[:,cond_i],data_e[:,:,cond_i],bsamples] = stat_m_e(data_all[:,:,cond_i],mtype=mtype,etype=etype)
            data = data_all[:,:,cond_i]


        for group_i in range(ngroup):
            x = base_x + group_i * group_gap + 1
            y = data[:, group_i]
            
            y = y[np.isfinite(y)]
            if len(y) == 0:
                continue

            # violin
            vp = ax.violinplot(y, positions=[x],
                vert=True, widths=width, points=100, bw_method='scott', 
                showmeans=False, showextrema=False, showmedians=False,
            )

            for body in vp['bodies']:
                body.set_facecolor(colors[cond_i])
                body.set_edgecolor("none")
                body.set_alpha(alphas[group_i]/2)
            
            if cond_i == 0: vp["bodies"][0].set_label(group_labels[group_i])

            vp = ax.violinplot(y, positions=[x],
                vert=True, widths=width, points=100, bw_method='scott', 
                showmeans=False, showextrema=False, showmedians=False
            )

            for body in vp['bodies']:
                body.set_facecolor("none")
                body.set_edgecolor(colors[cond_i])
                body.set_alpha(alphas[group_i])

            # errorbar
            ax.errorbar(x, data_m[group_i,cond_i],data_e[:,group_i,cond_i][:,None],
                color='k', marker='o', linestyle='', 
                elinewidth=1.8, markersize=5)

            # scatter
            jitter = np.random.uniform(-width / 4, width / 4, size=len(y))
            ax.scatter(np.full(len(y), x) + jitter,y,
                color=colors[cond_i], alpha=alphas[group_i],
                s=15,edgecolors='white',linewidths=0.5)

    # condition label
    cond_center = np.arange(ncondition) * (ngroup*group_gap + cond_gap) + (ngroup + 1) / 2
    ax.set_xticks(cond_center)
    ax.set_xticklabels(cond_labels)
    if legendtitle: ax.legend(frameon=True,title=legendtitle,loc='center left',bbox_to_anchor=(1.02,0.5))


def sig_plot(ax,x,w,ps,pys,data_m,pair=True,data_all=[]):
    nxl = len(x)
    
    if len(pys) == 0:
        for nxl_i in range(nxl):
            if len(data_all) > 0:
                pys.append([np.max(data_all[:,nxl_i,:])+np.max(data_m)*0.16,np.max(data_all[:,nxl_i,:])+np.max(data_m)*0.14])
            else:
                pys.append([np.max(data_m[nxl_i,:])+np.max(data_m)*0.16,np.max(data_m[nxl_i,:])+np.max(data_m)*0.14])
    else:
        pys = [pys]*nxl
    
    for nxl_i in range(nxl):
        star = sig(ps[nxl_i])
        if star == 'n.s.':
            continue
        elif star == "***":
            ax.text(x[nxl_i],pys[nxl_i][0],star,color='k',fontsize=16,fontfamily='Arial',horizontalalignment='center', verticalalignment='center')    
        else:
            ax.text(x[nxl_i],pys[nxl_i][0],star,color='k',fontsize=11,fontfamily='Arial',horizontalalignment='center', verticalalignment='bottom')    
        
        if pair:
            ax.plot([x[nxl_i]-w*0.3,x[nxl_i]+w*0.3],[pys[nxl_i][1],pys[nxl_i][1]],'k',linewidth=1,clip_on=False)

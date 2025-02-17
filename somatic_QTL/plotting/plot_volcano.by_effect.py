from __future__ import division
import sys, os
import numpy as np
import matplotlib
matplotlib.use('pdf')
from matplotlib import pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
#matplotlib.rcParams['figure.autolayout'] = True

args = sys.argv[1:]
inf = args[0]
outpre = args[1]

# set index for data
dind = None
if 'OLS' in inf: dind = 8
elif 'Ridge' in inf or 'Lasso' in inf: dind = 9
elif 'ElasticNet' in inf: dind = 10

# get data
mbeta, mpval, mfdr, nsamps, loci = [], [], [], [], []
cn = {}
for line in open(inf).readlines()[1:]:
    l = line.strip().split('\t')
    g = l[0]
    b, p, f = map(float, l[dind:dind+3])
    cb,cp,cf= map(float, l[dind+3:dind+6])
    ns = int(l[dind-1])
    loc = l[-1]

    mbeta.append(b)
    mpval.append(p)
    mfdr.append(f)
    loci.append(loc)
    nsamps.append(ns)

    if g not in cn:
        cn[g] = [cb, cp, cf]

# create mutations plot
beta = np.array(mbeta)
pval = np.array(mpval)
fdr = np.array(mfdr)
loci = np.array(loci)
nsamps = np.array(nsamps)
sigInd = (fdr < 0.1)
nsInd = (fdr >= 0.1)

nlog10pv = -np.log10(pval + 1e-322)
MAX = 12
nlog10pv[nlog10pv > MAX] = MAX

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(beta[nsInd], nlog10pv[nsInd], 'ko', alpha = 0.5, markersize = 6)
ax1.plot(beta[sigInd], nlog10pv[sigInd], 'ro', alpha = 0.5, markersize = 6)
for ind in np.argwhere(sigInd == True):
    name = loci[ind][0] + ' (%d)'%(nsamps[ind])
    ax1.text(beta[ind], nlog10pv[ind], name)
ax1.set_xlabel(r'Mutation class $\beta$', fontsize = 10)
ax1.set_ylabel(r'$-log_{10}$ (p-value)', fontsize = 10)
bmax = np.ceil(10 * max(np.abs(beta))) / 10
ax1.set_ylim([-0.2, MAX + .2])
ax1.set_xlim([-1.2, 1.2])
fig.savefig(outpre + '_volcano_mutations.pdf')

# create CNA plot
cbeta, cpval, cfdr = [], [], []
for g in cn:
    b, p, f = cn[g]
    cbeta.append(b)
    cpval.append(p)
    cfdr.append(f)
beta = np.array(cbeta)
pval = np.array(cpval)
fdr = np.array(cfdr)
sigInd = (fdr < 0.1)
nsInd = (fdr >= 0.1)

nlog10pv = -np.log10(pval + 1e-322)
nlog10pv[nlog10pv > 10] = 10

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(beta[nsInd], nlog10pv[nsInd], 'ko', alpha = 0.5, markersize = 6)
ax.plot(beta[sigInd], nlog10pv[sigInd], 'ro', alpha = 0.5, markersize = 6)
ax.set_xlabel(r'$\beta_{CNA}$', fontsize = 10)
ax.set_ylabel(r'$-log_{10} (p-value)$', fontsize = 10)
cmax = np.ceil(10 * max(np.abs(beta))) / 10
ax.set_xlim([-bmax, bmax])
ax.set_ylim([-0.2, 10.2])
ax.set_xlim([-1.2, 1.2])

fig.savefig(outpre + '_volcano_CNA.pdf', dpi = 200)


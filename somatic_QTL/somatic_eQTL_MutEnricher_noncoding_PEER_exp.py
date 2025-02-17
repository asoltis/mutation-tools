import sys, os, cPickle
import gzip
import numpy as np
import glob
from interlap import InterLap
from collections import defaultdict
from pysam import TabixFile

sys.path.insert(0, '') # Include path to MutEnricher code
sys.path.insert(0, '') # Include path to MutEnricher math_funcs directory
from noncoding_enrichment import Region
from scipy import stats
import statsmodels.api as SM
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import ShuffleSplit, cross_val_score
import warnings
import multiprocessing as mp

def load_gtf(gtf,genefield,gene_list,genes_to_use):
    '''
    Load and process genes in GTF file.
    '''
    genes = {}
    bad_genes = set() # list for problematic genes
    
    if gtf.endswith('.gz'): FH = gzip.open(gtf,'rb')
    else: FH = open(gtf)
    
    for line in FH:
        if line.startswith('#'): continue # some GTFs have header lines
        l = line.strip().split('\t')
        chrom,typ,start,stop,strand,data = l[0],l[2],int(l[3]),int(l[4]),l[6],l[-1]

        if len(chrom.split('_'))>1: continue # skip complex chromosomes (e.g. randoms)
        
        # parse data string
        gene_id,tx_id = '',''
        for d in data.split(';'):
            if d=='': continue
            elif d[0] == ' ': d = d[1:]
            if d.startswith(genefield):
                d1 = d.split(' ')[1]
                gene_id = d1.replace('"','')
        
        # Check if restriction list supplied
        if gene_list != None:
            if gene_id not in genes_to_use: continue

        # get gene data
        if gene_id not in genes:
            genes[gene_id] = {}
            genes[gene_id]['chrom'] = chrom
            genes[gene_id]['strand'] = strand
            genes[gene_id]['exons'] = []
            genes[gene_id]['CDS'] = []
        if chrom != genes[gene_id]['chrom']: bad_genes.add(gene_id)

        reg = (start,stop)
        if typ == 'exon':
            if reg in genes[gene_id]['exons']: continue
            genes[gene_id]['exons'].append(reg)
        elif typ == 'CDS':
            if reg in genes[gene_id]['CDS']: continue
            genes[gene_id]['CDS'].append(reg)
            
    # delete problematic genes
    print '  Deleting %d genes annotated to multiple chromosomes.'%(len(bad_genes))
    for g in bad_genes:
        del(genes[g])

    # return
    for gene_id in genes:
        genes[gene_id]['exons'].sort()
        genes[gene_id]['CDS'].sort()
    FH.close()
    return genes

def get_sig_regions(regions, rfdr = 0.1, hfdr = 0.01, atype = 'region_and_hotspot'):
    '''
    Get significant regions. 

    Default behavior is "region_and_hotspot", which finds significant hotspots first and captures those as individual test units. 
    Then, if the overall region is significant, include the additional mutated loci as "remainder" elements. 
    If the overall region is significant but no significant hotspot exists, then the entire region is used as a single unit. 

    IN DEVELOPMENT:
    Alternatively, "by_loci" as the analysis type treats each individual variant locus (by position) as a single unit. Here, each position
    in a significant hotspot and/or overall region is treated as its own term in the regression model. 
    '''
   
    # loop over regions, find significant overall regions and hotspots
    ME_sig = []
    for r in regions:
        msig, hs_positions = [], []
        n_hs = len(r.clusters)
        if n_hs > 0:
            for hs in r.cluster_enrichments:
                hs_reg, hs_fdr, hs_pos, hs_samps = hs[0], hs[9], hs[11], hs[-1]
                if hs_fdr < hfdr:
                    if atype == 'region_and_hotspot':
                        msig.append(['HS_'+hs_reg, hs_samps.split(';')])
                        for pos in hs_pos.split(';'):
                            hs_positions.append(int(pos.split('_')[0]))
                    elif atype == 'by_loci':
                        for p in hs_pos.split(';'):
                            pos = int(p.split('_')[0])
                            mut_samps = r.samples_by_positions[pos]
                            msig.append(['%s_%d'%(r.chrom, pos), mut_samps])
                            hs_positions.append(pos)

        #if r.fisher_qval < rfdr:
        if r.fisher_rw_qval < rfdr: # updated region attribute - this is the combined region + WAP stat (consistent with earlier)
            if len(msig) == 0: 
                if atype == 'region_and_hotspot':
                    mut_samps = r.mutations_by_sample.keys()
                    msig.append([r.region_string, mut_samps])
                elif atype == 'by_loci':
                    for pos in r.samples_by_positions:
                        mut_samps = r.samples_by_positions[pos]
                        msig.append(['%s_%d'%(r.chrom, pos), mut_samps])
            else: 
                rem_pos, rem_samps = set(), set()
                for pos in r.samples_by_positions:
                    if pos in hs_positions: continue
                    else:
                        rem_pos.add(pos)
                        for s in r.samples_by_positions[pos]:
                            rem_samps.add(s)
        
                if len(rem_samps)>0:
                    if atype == 'region_and_hotspot':
                        msig.append([r.region_string, list(rem_samps)])
                    elif atype == 'by_loci':
                        for pos in rem_pos:
                            mut_samps = r.samples_by_positions[pos]
                            msig.append(['%s_%d'%(r.chrom, pos), mut_samps])

        if len(msig)>0:
            if atype == 'region_and_hotspot':
                ME_sig.append([r, msig])
            elif atype == 'by_loci':
                # need to correct cases where multiple loci have same exact patients mutated
                # create small sample by positions matrix
                p2s, s2p = {}, {}
                for ms in msig:
                    mpos, msamps = ms
                    if mpos not in p2s: p2s[mpos] = []
                    for s in msamps:
                        if s not in s2p: s2p[s] = []
                        s2p[s].append(mpos)
                        p2s[mpos].append(s)
                mat = np.zeros((len(s2p), len(p2s)))
                samps = np.array([s for s in s2p])
                posits = np.array([p for p in p2s])
                for i,s in enumerate(samps):
                    for j,p in enumerate(posits):
                        if p in s2p[s]:
                            mat[i,j] = 1
                # check if matrix rank-defficient
                # if not, return msig
                # if so, need to re-work
                matrank = np.linalg.matrix_rank(mat)
                if matrank == len(posits):
                    #print mat[:, np.argsort(np.sum(mat, axis=0))[::-1]] 
                    ME_sig.append([r, msig])
                else:
                    # sort matrix by recurrency (i.e. by column), then lexsort
                    si = np.argsort(np.sum(mat, axis=0))[::-1]
                    mat = mat[:, si]
                    posits = posits[si]
                    si2 = np.lexsort(mat, axis=0)
                    mat = mat[:, si2]
                    posits = posits[si2]
                    # find linearly dependent columns with Cauch-Schwarz inequality
                    dep_cols = []
                    for i in range(0, mat.shape[1]-1):
                        for j in range(i+1, mat.shape[1]):
                            ip = np.inner(mat[:,i], mat[:,j])
                            ni = np.linalg.norm(mat[:,i])
                            nj = np.linalg.norm(mat[:,j])
                            if abs(ip - ni*nj) < 1e-10:
                                if len(dep_cols)>0:
                                    if i in dep_cols[-1]:
                                        if j not in dep_cols[-1]: dep_cols[-1].append(j)
                                    else:
                                        dep_cols.append([i,j])
                                else:
                                    dep_cols.append([i,j])
                    # reduce dependent columns and create new matrix
                    cols_used = set()
                    new_mat, new_pos = [], []
                    for dc in dep_cols:
                        for c in dc: cols_used.add(c)
                        pi = sorted(posits[dc])
                        m = list(mat[:, dc[0]])
                        ps = ''
                        for i,p in enumerate(pi):
                            if i == 0:
                                ps += p
                            else:
                                ps += '_%s'%(p.split('_')[1])
                        new_mat.append(m)
                        new_pos.append(ps)
                    for i in range(mat.shape[1]):
                        if i in cols_used: continue
                        m = list(mat[:, i])
                        ps = posits[i]
                        new_mat.append(m)
                        new_pos.append(ps)
                    new_mat = np.array(new_mat).T
                    # Write data
                    msig_new = []
                    for i in range(0, new_mat.shape[1]):
                        pname = new_pos[i]
                        mut_samps = samps[(new_mat[:, i]==1)]
                        msig_new.append([pname, mut_samps])
                    ME_sig.append([r, msig_new])
 
    # return
    return ME_sig

def map_regions(ME_sig, genes_inter, WINL = 10000, mapType = 'all_proximity', reg2gene = None):
    '''
    Map regions to potential regulated genes. 

    If mapType = 'all_proximity', all genes in genes_inter object mapping within +/- WINL from region coordinates 
    are reported.

    If mapType = 'reg_to_gene', behavior of 'all_proximity' is supplemented with region-to-gene mappings in supplied
    reg2gene, which is an open pysam TabixFile object. This file is a BED file, with the mapped gene reported in the
    6th column.
    '''

    # initialize output
    genes2reg = {}
    
    # Map significant regions/hotspots to genes
    for ME in ME_sig:
        r, msig = ME
        # extract region info
        chrom, start, stop = r.chrom, r.start, r.stop
        start -= WINL
        stop += WINL
        rname = r.name
        rstring = r.region_string

        # get overlapping genes near region
        olap_genes = set()
        for olap in list(genes_inter[chrom].find((start, stop))):
            gene = olap[2]
            olap_genes.add(gene)
        
        # assign peak-to-gene links if mapType is "reg_to_gene"
        if mapType == 'reg_to_gene':
            # query peak to gene file for mapping
            tbq = [t.split('\t') for t in reg2gene.fetch(rstring)] 
            if len(tbq) > 0:
                for t in tbq:
                    gene = t[5]
                    olap_genes.add(gene)
        
        # Assign regions to genes
        for g in olap_genes:
            if g not in genes2reg: 
                genes2reg[g] = {}
                genes2reg[g]['region_names'] = []
                genes2reg[g]['mdata'] = []
            for ms in msig:
                genes2reg[g]['mdata'].append(ms)
                genes2reg[g]['region_names'].append(rname)

    # Return data
    return genes2reg

def run_regression(gene, GINFO, regType, nrep = 20):
    '''
    
    '''
    # return variable
    REG_RESULTS = {}
    
    # extract X, Y data
    gene_X = GINFO['X']
    GX_red = GINFO['Xred']
    gene_Y = GINFO['Y']
    loci_inds = GINFO['loci_inds']

    # other params
    #alphas = [x * 1e-4 for x in range(0, 10001, 50)]
    #alphas = alphas[1:]
    
    #alphas = ([pow(10, x) for x in [x*1e-2 for x in range(-400, 50, 5)]])
    alphas = ([pow(10, x) for x in [x*1e-2 for x in range(-400, 200, 5)]])
    l1_ratios = [x * 0.01 for x in range(0, 101, 5)]
    
    # OLS
    if regType == 'OLS':
        # run linear regression 
        clf = LinearRegression(fit_intercept = False) 
        clf.fit(gene_X, gene_Y) 
        SSE = np.sum((clf.predict(gene_X) - gene_Y) ** 2, axis = 0) / float(gene_X.shape[0] - gene_X.shape[1])
        SE = np.array([np.sqrt(np.diagonal(SSE[i] * np.linalg.inv(np.dot(gene_X.T, gene_X)))) for i in range(SSE.shape[0])])
        tstat = clf.coef_ / SE
        pvals = 2 * (1-stats.t.cdf(np.abs(tstat), gene_Y.shape[0] - gene_X.shape[1]))
        beta = clf.coef_[0]
        pvals = pvals[0]
        Rsquare = clf.score(gene_X, gene_Y)
   
        clf_red = LinearRegression(fit_intercept = False)
        clf_red.fit(GX_red, gene_Y)
        
        # F-test
        RSS_full = np.sum((clf.predict(gene_X) - gene_Y) ** 2)
        RSS_red  = np.sum((clf_red.predict(GX_red) - gene_Y) ** 2)
        np_full = gene_X.shape[1]
        df_full = gene_X.shape[0] - np_full
        np_red = GX_red.shape[1]
        df_red = GX_red.shape[0] - np_red
        Fstat = ( (RSS_red - RSS_full) / (np_full - np_red) ) / (RSS_full / df_full)
        if Fstat < 0: Fstat = 0.0
        Fpv = 1 - stats.f.cdf(Fstat, np_full - np_red, df_full)
       
        # report
        REG_RESULTS['gene'] = gene
        REG_RESULTS['beta'] = beta
        REG_RESULTS['pvals'] = pvals
        REG_RESULTS['Rsquare'] = Rsquare
        REG_RESULTS['Fstat'] = Fstat
        REG_RESULTS['Fpval'] = Fpv

    # Ridge - L2 penalty
    elif regType == 'Ridge':
        train_res = [] # tracker for CV results
        # use ordinary linear regression for alpha = 0 case
        clf_l = LinearRegression(fit_intercept = False)
        cv = ShuffleSplit(n_splits = nrep, test_size = 0.3, random_state = 0)
        scores = cross_val_score(clf_l, gene_X, gene_Y, cv = cv, scoring = 'neg_mean_squared_error')
        train_res.append([0, np.mean(-scores)])
        # perform regression cross validation - Ridge
        for alpha in alphas:
            clf_cv = Ridge(alpha = alpha, fit_intercept = False)
            cv = ShuffleSplit(n_splits = nrep, test_size = 0.3, random_state = 0)
            scores = cross_val_score(clf_cv, gene_X, gene_Y, cv = cv, scoring = 'neg_mean_squared_error')
            train_res.append([alpha, np.mean(-scores)])
        train_res.sort(key = lambda x: x[1], reverse=False)
         
        # opt fit
        alpha = train_res[0][0]
        if alpha > 0:
            clf = Ridge(alpha = alpha, fit_intercept = False)
            clf_red = Ridge(alpha = alpha, fit_intercept = False)
            
            clf.fit(gene_X, gene_Y)
            beta = clf.coef_[0]
            Rsquare = clf.score(gene_X, gene_Y)
            Ypred = clf.predict(gene_X)
            RSSf = np.sum((Ypred - gene_Y) ** 2)
            np_full = gene_X.shape[1]
            df_full = gene_X.shape[0] - np_full

            clf_red.fit(GX_red, gene_Y)
            Ypred_red = clf_red.predict(GX_red)
           
            # compute F-test p-values for each coefficient
            pvals = []
            clf_for_pv = Ridge(alpha = alpha, fit_intercept = False)
            for i, b in enumerate(beta):
                inds = [j for j in range(0, len(beta)) if j != i]
                gx = gene_X[:, inds]
                
                clf_for_pv.fit(gx, gene_Y)
                Ypred_pv = clf_for_pv.predict(gx)
                RSSr = np.sum((Ypred_pv - gene_Y) ** 2)
                npr = gx.shape[1]
                dpr = gx.shape[0] - npr
                
                fs = ((RSSr - RSSf) / (np_full - npr)) / (RSSf / df_full)
                if fs < 0: fpv = 1.0
                else: fpv = 1 - stats.f.cdf(fs, np_full - npr, df_full)
                pvals.append(fpv)
        
        else: 
            clf = LinearRegression(fit_intercept = False)
            clf_red = LinearRegression(fit_intercept = False)
        
            clf.fit(gene_X, gene_Y)
            clf_red.fit(GX_red, gene_Y)
        
            Ypred = clf.predict(gene_X)
            Ypred_red = clf_red.predict(GX_red)

            SSE = np.sum((Ypred - gene_Y) ** 2, axis = 0) / float(gene_X.shape[0] - gene_X.shape[1])
            SE = np.array([np.sqrt(np.diagonal(SSE[i] * np.linalg.inv(np.dot(gene_X.T, gene_X)))) for i in range(SSE.shape[0])])
            tstat = clf.coef_ / SE
            pvals = 2 * (1-stats.t.cdf(np.abs(tstat), gene_Y.shape[0] - gene_X.shape[1]))
            beta = clf.coef_[0]
            pvals = pvals[0]
            Rsquare = clf.score(gene_X, gene_Y)
   
        # F-test for all loci
        RSS_full = np.sum((Ypred - gene_Y) ** 2)
        RSS_red  = np.sum((Ypred_red - gene_Y) ** 2)
        np_full = gene_X.shape[1]
        df_full = gene_X.shape[0] - np_full
        np_red = GX_red.shape[1]
        df_red = GX_red.shape[0] - np_red
        Fstat = ( (RSS_red - RSS_full) / (np_full - np_red) ) / (RSS_full / df_full)
        if Fstat < 0: Fpv = 1.0
        else: Fpv = 1 - stats.f.cdf(Fstat, np_full - np_red, df_full)
        
        # report
        REG_RESULTS['gene'] = gene
        REG_RESULTS['beta'] = beta
        REG_RESULTS['pvals'] = pvals
        REG_RESULTS['Rsquare'] = Rsquare
        REG_RESULTS['alpha'] = alpha
        REG_RESULTS['Fstat'] = Fstat
        REG_RESULTS['Fpval'] = Fpv

    # Lasso - L1 penalty
    elif regType == 'Lasso':
        train_res = [] # tracker for CV results
        # use ordinary linear regression for alpha = 0 case
        clf_l = LinearRegression(fit_intercept = False)
        cv = ShuffleSplit(n_splits = nrep, test_size = 0.3, random_state = 0)
        scores = cross_val_score(clf_l, gene_X, gene_Y, cv = cv, scoring = 'neg_mean_squared_error')
        train_res.append([0, np.mean(-scores)])
        # perform regression cross validation - LASSO
        for alpha in alphas:
            clf_cv = Lasso(alpha = alpha, fit_intercept = False)
            cv = ShuffleSplit(n_splits = nrep, test_size = 0.3, random_state = 0)
            scores = cross_val_score(clf_cv, gene_X, gene_Y, cv = cv, scoring = 'neg_mean_squared_error')
            train_res.append([alpha, np.mean(-scores)])
        train_res.sort(key = lambda x: x[1], reverse=False)
        
        # opt fit
        alpha = train_res[0][0]
        if alpha > 0:
            clf = Lasso(alpha = alpha, fit_intercept = False)
            clf.fit(gene_X, gene_Y)
            beta = clf.coef_
            Rsquare = clf.score(gene_X, gene_Y)
            
            # If loci coefficients all zero, progressively lower alpha to get positive value for one
            sumzero = sum(np.abs([beta[i] for i in loci_inds]))
            while sumzero == 0:
                alpha_ind = np.argwhere(np.array(alphas) == alpha)[0][0]
                if (alpha_ind - 1) < 0: break
                alpha = alphas[alpha_ind - 1] # lower alpha
                clf = Lasso(alpha = alpha, fit_intercept = False)
                clf.fit(gene_X, gene_Y)
                sumzero = sum(np.abs([clf.coef_[i] for i in loci_inds]))
                
            if sumzero == 0: # If still zero, fall back to OLS
                alpha = 0 # re-set alpha to zero
                clf = LinearRegression(fit_intercept = False)
                clf.fit(gene_X, gene_Y)
                beta = clf.coef_[0]
                Rsquare = clf.score(gene_X, gene_Y)
                
                clf_red = LinearRegression(fit_intercept = False)
                clf_red.fit(GX_red, gene_Y)
                beta_red = clf_red.coef_[0]
             
                Ypred = clf.predict(gene_X)
                SSE = np.sum((Ypred - gene_Y) ** 2, axis = 0) / float(gene_X.shape[0] - gene_X.shape[1])
                SE = np.array([np.sqrt(np.diagonal(SSE[i] * np.linalg.inv(np.dot(gene_X.T, gene_X)))) for i in range(SSE.shape[0])])
                tstat = clf.coef_ / SE
                pvals = 2 * (1-stats.t.cdf(np.abs(tstat), gene_Y.shape[0] - gene_X.shape[1]))
                pvals = pvals[0]
                
            else: # else, use new alpha
                beta = clf.coef_
                Rsquare = clf.score(gene_X, gene_Y)
                
                clf_red = Lasso(alpha = alpha, fit_intercept = False) 
                clf_red.fit(GX_red, gene_Y)
                beta_red = clf_red.coef_
            
                Rsquare = clf.score(gene_X, gene_Y)
                Ypred = clf.predict(gene_X).reshape(-1,1) # Lasso produces predicted output not matching shape of original Y
                RSSf = np.sum((Ypred - gene_Y) ** 2)
                np_full = gene_X.shape[1]
                df_full = gene_X.shape[0] - np_full

                # compute F-test p-values for each coefficient
                pvals = []
                clf_for_pv = Lasso(alpha = alpha, fit_intercept = False)
                for i, b in enumerate(beta):
                    inds = [j for j in range(0, len(beta)) if j != i]
                    gx = gene_X[:, inds]
                    
                    clf_for_pv.fit(gx, gene_Y)
                    Ypred_pv = clf_for_pv.predict(gx).reshape(-1,1)
                    RSSr = np.sum((Ypred_pv - gene_Y) ** 2)
                    npr = gx.shape[1]
                    dpr = gx.shape[0] - npr
                    
                    fs = ((RSSr - RSSf) / (np_full - npr)) / (RSSf / df_full)
                    if fs < 0: fpv = 1.0
                    else: fpv = 1 - stats.f.cdf(fs, np_full - npr, df_full)
                    pvals.append(fpv)
        
        else: # if alpha = 0, use standard OLS
            clf = LinearRegression(fit_intercept = False)
            clf.fit(gene_X, gene_Y)
            beta = clf.coef_[0]
            Rsquare = clf.score(gene_X, gene_Y)
            
            clf_red = LinearRegression(fit_intercept = False)
            clf_red.fit(GX_red, gene_Y)
            beta_red = clf_red.coef_[0]
            
            Ypred = clf.predict(gene_X)
            SSE = np.sum((Ypred - gene_Y) ** 2, axis = 0) / float(gene_X.shape[0] - gene_X.shape[1])
            SE = np.array([np.sqrt(np.diagonal(SSE[i] * np.linalg.inv(np.dot(gene_X.T, gene_X)))) for i in range(SSE.shape[0])])
            tstat = clf.coef_ / SE
            pvals = 2 * (1-stats.t.cdf(np.abs(tstat), gene_Y.shape[0] - gene_X.shape[1]))
            pvals = pvals[0]

        # get predicted Y values from regressions
        if alpha > 0: # Lasso produces predict output not matching shape of original Y data
            Ypred = clf.predict(gene_X).reshape(-1,1) 
            Ypred_red = clf_red.predict(GX_red).reshape(-1,1)
        else:
            Ypred = clf.predict(gene_X)
            Ypred_red = clf_red.predict(GX_red)

        # F-test
        RSS_full = np.sum((Ypred - gene_Y) ** 2)
        RSS_red  = np.sum((Ypred_red - gene_Y) ** 2) 
        np_full = gene_X.shape[1] #sum(np.abs(np.array(beta)) > 0)
        df_full = gene_X.shape[0] - np_full
        np_red =  GX_red.shape[1] #sum(np.abs(np.array(beta_red)) > 0)
        df_red = GX_red.shape[0] - np_red
        Fstat = ( (RSS_red - RSS_full) / (np_full - np_red) ) / (RSS_full / df_full)
        if Fstat < 0: Fpv = 1.0
        else: Fpv = 1 - stats.f.cdf(Fstat, np_full - np_red, df_full) 
        
        # report
        REG_RESULTS['gene'] = gene
        REG_RESULTS['beta'] = beta
        REG_RESULTS['pvals'] = pvals
        REG_RESULTS['Rsquare'] = Rsquare
        REG_RESULTS['alpha'] = alpha
        REG_RESULTS['Fstat'] = Fstat
        REG_RESULTS['Fpval'] = Fpv

    # Elastic Net - combined L1/L2 penalties
    elif regType == 'ElasticNet': 
        train_res = [] # tracker for CV results
         # use ordinary linear regression for alpha = 0 case
        clf_l = LinearRegression(fit_intercept = False)
        cv = ShuffleSplit(n_splits = nrep, test_size = 0.3, random_state = 0)
        scores = cross_val_score(clf_l, gene_X, gene_Y, cv = cv, scoring = 'neg_mean_squared_error')
        train_res.append([0, 0, np.mean(-scores)])
        # perform regression cross validation - Elastic Net 
        for alpha in alphas:
            for l1r in l1_ratios:
                clf_cv = ElasticNet(alpha = alpha, l1_ratio = l1r, fit_intercept = False, max_iter = 5000)
                cv = ShuffleSplit(n_splits = nrep, test_size = 0.3, random_state = 0)
                scores = cross_val_score(clf_cv, gene_X, gene_Y, cv = cv, scoring = 'neg_mean_squared_error')
                train_res.append([alpha, l1r, np.mean(-scores)])
        train_res.sort(key = lambda x: x[2], reverse=False)
        
        # opt fit
        alpha, l1_ratio = train_res[0][0:2]
        if alpha > 0: 
            clf = ElasticNet(alpha = alpha, l1_ratio = l1_ratio, fit_intercept = False, max_iter = 10000)
            clf_red = ElasticNet(alpha = alpha, l1_ratio = l1_ratio, fit_intercept = False, max_iter = 10000)
       
            clf.fit(gene_X, gene_Y)
            beta = clf.coef_
            Rsquare = clf.score(gene_X, gene_Y)
            Ypred = clf.predict(gene_X).reshape(-1,1)
            RSSf = np.sum((Ypred - gene_Y) ** 2)
            np_full = gene_X.shape[1]
            df_full = gene_X.shape[0] - np_full

            clf_red.fit(GX_red, gene_Y)
            Ypred_red = clf_red.predict(GX_red).reshape(-1,1)

            # compute F-test p-values for each coefficient
            pvals = []
            clf_for_pv = ElasticNet(alpha = alpha, l1_ratio = l1_ratio, fit_intercept = False, max_iter = 10000)
            for i, b in enumerate(beta):
                inds = [j for j in range(0, len(beta)) if j != i]
                gx = gene_X[:, inds]
                
                clf_for_pv.fit(gx, gene_Y)
                Ypred_pv = clf_for_pv.predict(gx).reshape(-1,1)
                RSSr = np.sum((Ypred_pv - gene_Y) ** 2)
                npr = gx.shape[1]
                dpr = gx.shape[0] - npr
                
                fs = ((RSSr - RSSf) / (np_full - npr)) / (RSSf / df_full)
                if fs < 0: fpv = 1.0
                else: fpv = 1 - stats.f.cdf(fs, np_full - npr, df_full)
                pvals.append(fpv)
        
        else: 
            clf = LinearRegression(fit_intercept = False)
            clf_red = LinearRegression(fit_intercept = False)
        
            clf.fit(gene_X, gene_Y)
            clf_red.fit(GX_red, gene_Y)
        
            Ypred = clf.predict(gene_X)
            Ypred_red = clf_red.predict(GX_red)

            SSE = np.sum((Ypred - gene_Y) ** 2, axis = 0) / float(gene_X.shape[0] - gene_X.shape[1])
            SE = np.array([np.sqrt(np.diagonal(SSE[i] * np.linalg.inv(np.dot(gene_X.T, gene_X)))) for i in range(SSE.shape[0])])
            tstat = clf.coef_ / SE
            pvals = 2 * (1-stats.t.cdf(np.abs(tstat), gene_Y.shape[0] - gene_X.shape[1]))
            beta = clf.coef_[0]
            pvals = pvals[0]
            Rsquare = clf.score(gene_X, gene_Y)
   
        # get predicted Y values from regressions
        if alpha > 0: # ElasticNet produces predict output not matching shape of original Y data
            Ypred = clf.predict(gene_X).reshape(-1,1) 
            Ypred_red = clf_red.predict(GX_red).reshape(-1,1)
        else:
            Ypred = clf.predict(gene_X)
            Ypred_red = clf_red.predict(GX_red)
 
        # F-test
        RSS_full = np.sum((Ypred - gene_Y) ** 2)
        RSS_red  = np.sum((Ypred_red - gene_Y) ** 2) 
        np_full = gene_X.shape[1]
        df_full = gene_X.shape[0] - np_full
        np_red = GX_red.shape[1]
        df_red = GX_red.shape[0] - np_red
        Fstat = ( (RSS_red - RSS_full) / (np_full - np_red) ) / (RSS_full / df_full)
        if Fstat < 0: Fpv = 1.0
        else: Fpv = 1 - stats.f.cdf(Fstat, np_full - np_red, df_full)
        
        # report
        REG_RESULTS['gene'] = gene
        REG_RESULTS['beta'] = beta
        REG_RESULTS['pvals'] = pvals
        REG_RESULTS['Rsquare'] = Rsquare
        REG_RESULTS['alpha'] = alpha
        REG_RESULTS['l1_ratio'] = l1_ratio
        REG_RESULTS['Fstat'] = Fstat
        REG_RESULTS['Fpval'] = Fpv

    # RETURN
    return REG_RESULTS

def fdr_BH(pvals):
    '''
    Compute Benjamini-Hochberg FDR q-values on sorted array of p-values.
    '''
    total = pvals.size
    si = np.argsort(pvals) 
    pvals_sort = pvals[si]
    
    fdrs0 = total*pvals_sort/range(1,total+1)
    fdrs = []
    # Preserve monotonicity
    for i in range(0,total):
        fdrs.append(min(fdrs0[i:]))
    
    # preserve original sort order 
    pairs = [(i,j) for i, j in enumerate(si)]
    fdrs_resort = np.zeros(pvals.shape)
    for pair in pairs: fdrs_resort[pair[1]] = fdrs[pair[0]]
    return fdrs_resort

def warn(*args, **kwargs):
    pass

def main():

    warnings.warn = warn 

    ##########
    # INPUTS #
    ##########
    ## PEER expression
    peer_fn = '' # python pickle object with PEER expression data and factors

    ## MutEnricher   
    MEDir = '' # Main MutEnricher non-coding analysis directory
    MEvars = '' # Text string for somatic variants used in ME analysis
    MEparams = '' # Text string for folder describing ME run parameters
    MEfile = '' # MutEnricher noncoding analysis ouptut *_region_data.pkl'
    MEfn_prom = '%s/promoters/%s/%s/%s' % (MEDir, MEvars, MEparam, MEfile)
    MEfn_3p = '%s/3pUTR/%s/%s/%s' % (MEDir, MEvars, MEparam, MEfile)
    MEfn_5p = '%s/5pUTR/%s/%s/%s' % (MEDir, MEvars, MEparam, MEfile)
    MEfn_dist = '%s/distal/%s/%s/%s' % (MEDir, MEvars, MEparam, MEfile)
    MEfns = [MEfn_prom, MEfn_3p, MEfn_5p, MEfn_dist] # List of MutEnricher files for different noncoding region types
    
    ## CNA  
    CNA_fn = '' # Text file with gene-wise copy number levels (expressed as log2(CN) - 1)

    ## Tumor purity 
    pur_fn = '' # Per-sample tumor purity estimates from WGS data

    # GTF
    gtf = 'gencode.v28.basic.annotation.gtf' # Gene tranfser format file
    genefield = 'gene_name' 
    goi_fn = '' # Simple text file listing genes of interest for analysis
    goi = [x.strip() for x in open(goi_fn).readlines()]

    # peak-to-gene mapping file for ATAC-seq peaks
    p2g_fn = 'TCGA/ATACseq/supp_data_files/TCGA-ATAC_DataS7_PeakToGeneLinks_v2.all_links.hg38_coords_cut.sort.bed.gz'

    ##########
    # PARAMS #
    ##########
    # processors
    nproc = 20

    # mapping
    WINL = 1000000 # mapping window for regions to genes
    mapType = 'all_proximity'
    #mapType = 'reg_to_gene'

    # analysis type
    atype = 'region_and_hotspot' #
    #atype = 'by_loci' # Still in DEVELOPMENT

    # region sig
    rsig = 0.2
    hs_sig = 0.05

    # regression
    standardize = True
    regType = 'ElasticNet'
    nrep = 50

    # OUTPUT
    prefix = 'noncoding_eQTL'
    adate = 'date' 
    ODIR = 'outdir'
    os.system('mkdir -p %s'%(ODIR))
    
    # save input file info to file
    infn = open(ODIR + 'input_files.txt', 'w')
    for fns in [peer_fn, CNA_fn, gtf, goi_fn, p2g_fn, pur_fn] + MEfns:
        infn.writelines('%s\n'%(fns))
    infn.close()

    # initialize output file name
    if mapType == 'all_proximity':
        ONAME = ODIR + '%s_somatic_eQTL_%s_regression_all_prox_TSS_winLen_%d_regFDR_%s_hsFDR_%s.txt'%(prefix, regType, WINL, str(rsig), str(hs_sig))
    elif mapType == 'reg_to_gene':
        ONAME = ODIR + '%s_somatic_eQTL_%s_regression_TCGA_links_map_TSS_winLen_%d_regFDR_%s_hsFDR_%s.txt'%(prefix, regType, WINL, str(rsig), str(hs_sig))

    # create header for specific regression type
    if regType == 'OLS':
        head = ['Gene', 'gene_nsamps', 'gene_nloci', 'R^2', 'gene_Fstat', 'gene_pval', 'gene_FDR', 
                'locus_nsamps', 'locus_beta', 'locus_pval', 'locus_FDR', 'CNA_beta', 'CNA_pval', 'CNA_FDR', 'locus', 'source_region']
    
    elif regType == 'Ridge' or regType == 'Lasso': 
        head = ['Gene', 'gene_nsamps', 'gene_nloci', 'R^2', 'alpha', 'gene_Fstat', 'gene_pval', 'gene_FDR', 
                'locus_nsamps', 'locus_beta', 'locus_pval', 'locus_FDR', 'CNA_beta', 'CNA_pval', 'CNA_FDR', 'locus', 'source_region']
       
    elif regType == 'ElasticNet': 
        head = ['Gene', 'gene_nsamps', 'gene_nloci', 'R^2', 'alpha', 'l1_ratio', 'gene_Fstat', 'gene_pval', 'gene_FDR', 
                'locus_nsamps', 'locus_beta', 'locus_pval', 'locus_FDR', 'CNA_beta', 'CNA_pval', 'CNA_FDR', 'locus', 'source_region']
    
    # initialize output, write header
    OF = open(ONAME, 'w')
    OF.writelines('\t'.join(head) + '\n')
    
    ########
    # PREP #
    ########
    ## Load GTF
    print 'Loading GTF...'
    #GTF = load_gtf(gtf, genefield, goi_fn, goi)
    GTF = cPickle.load(open('genes_gtf.pkl'))
    print 'GTF loaded.\n'

    ## Create genes interlap object - take first annotated exon as TSS (strand-aware)
    genes_inter = defaultdict(InterLap)
    for g in GTF:
        chrom = GTF[g]['chrom']
        exons = GTF[g]['exons']
        strand = GTF[g]['strand']
        if strand == '+':
            e1, e2 = exons[0] 
        elif strand == '-':
            e1, e2, = exons[-1]
        genes_inter[chrom].add((e1, e2, g, strand))
        
    ## Load CNA data
    print 'Loading CNA data...'
    cna_genes, cna_samps, cna_mat = [], [], []
    for line in open(CNA_fn):
        if line.startswith('Gene'):
            cna_samps = [x for x in line.strip().split('\t')[1:]]
            continue
        else:
            l = line.strip().split('\t')
            cna_genes.append(l[0])
            vals = map(float, l[1:])
            cna_mat.append(vals)
    cna_mat = np.array(cna_mat)
    cna_genes = np.array(cna_genes)
    cna_samps = np.array(cna_samps)
    print 'CNA data loaded.\n'

    ## Load MutEnricher data
    print 'Loading MutEnricher region data...'
    ME_regions = []
    for MEfn in MEfns:
        reg = cPickle.load(open(MEfn))
        ME_regions += reg
    # get significant regions
    ME_sig = get_sig_regions(ME_regions, rsig, hs_sig, atype)
    # map to genes
    if mapType == 'all_proximity':
        genes2reg = map_regions(ME_sig, genes_inter, WINL)
    elif mapType == 'reg_to_gene':
        if not os.path.isfile(p2g_fn):
            print 'Region to gene mapping file does not exist. Exiting.'
            sys.exit()
        r2gTBF = TabixFile(p2g_fn)
        genes2reg = map_regions(ME_sig, genes_inter, WINL, mapType, r2gTBF)
    print len(genes2reg)
    print 'MutEnricher data loaded.\n'

    ## Load PEER data
    print 'Loading PEER expression data...'
    peer = cPickle.load(open(peer_fn))
    esamps = np.array(peer['samples'])
    egenes = np.array(peer['genes'])
    exp_mat = peer['input']
    nfact = peer['factors'].shape[1]
    #e_factors = peer['factors'][:, [0, 1] + range(3, nfact)] # remove intercept if standardizing data
    #e_factors = peer['factors'][:, 0:2] # take only sex and ancestry
    e_factors = peer['factors'][:, [0, 1] + range(3, 7)] # use sex, ancestry, and first X hidden PEER factors
    exp_WF = np.matmul(peer['weights'][:, [0, 1, 2]], peer['factors'][:, [0, 1, 2]].T).T
    print 'PEER data loaded.\n'
    
    ## Tumor purity info
    puritiesD = {}
    for line in open(pur_fn).readlines()[1:]:
        s, p = line.strip().split('\t')
        puritiesD[s] = float(p)
    purities = []
    for samp in esamps:
        purities.append(puritiesD[samp])

    ## Set up pool
    pool = mp.Pool(processes = nproc)

    #############
    # EXECUTION #
    ############# 
    print 'Extracting gene CNA data and creating regression input data...'
    # Loop over mapped genes and compute regressions
    GENE_INFO = {}
    for g in genes2reg: 
        # check that gene is a part of current data
        if g not in GTF: 
            print ' %s gene is not part of positive set of genes to test; skipping.'%(g)
            continue
        #if g not in ['ENY2']: continue
        
        # initialize gene info
        GENE_INFO[g] = {}

        # get gene index
        gind = (egenes == g)
 
        # get mapped regions
        regsmap = genes2reg[g]['mdata']
        regnames = genes2reg[g]['region_names']

        # Get mutated sample indices in expression data for region
        #samp_inds = [i for i,x in enumerate(esamps) if x in mut_samps]
        #non_inds = [i for i,x in enumerate(esamps) if x not in mut_samps]
        samp_set, samps_per_reg = set(), []
        Xnames, Xsamps = [], []
        for rdata in regsmap:
            xn, xs = rdata
            for samp in xs: samp_set.add(samp)
            xsamp = []
            for samp in esamps:
                if samp in xs: xsamp.append(1)
                else: xsamp.append(0)
            Xsamps.append(xsamp)
            Xnames.append(xn)
            samps_per_reg.append(xs)
        GENE_INFO[g]['rnames'] = Xnames
        GENE_INFO[g]['region_names'] = regnames
        GENE_INFO[g]['samps_per_region'] = samps_per_reg

        # Get gene's expression data, subtract off covariate, PEER factors
        gene_Y = exp_mat[:, gind] #- exp_WF[:, gind]
        
        # Get copy number alteration data for gene
        CNA = []
        gcn_ind = (cna_genes == g)
        for samp in esamps:
            scn_ind = (cna_samps == samp)
            cn = cna_mat[gcn_ind, scn_ind]
            CNA.append(cn[0])
   
        # Create X matrix for regression
        gene_X, GX_red = [], []
        gene_X.append(CNA)
        GX_red.append(CNA)
        loci_inds = [] 
        l_ind_track = len(gene_X)
        for xs in Xsamps: # don't add to reduced
            gene_X.append(xs)
            loci_inds.append(l_ind_track)
            l_ind_track += 1
        for fi in range(0, e_factors.shape[1]):
            gene_X.append(list(e_factors[:, fi]))
            GX_red.append(list(e_factors[:, fi]))
        #gene_X.append(purities)
        #GX_red.append(purities)
        gene_X = np.array(gene_X).T
        GX_red = np.array(GX_red).T
       
        # Standardize
        if standardize:
            yscaler = StandardScaler(with_mean = True, with_std = False) # PEER used VST data, which is variance stabilized
            yscaler.fit(gene_Y)
            gene_Y = yscaler.transform(gene_Y)
            xscaler = StandardScaler(with_mean = True, with_std = True)
            xscaler.fit(gene_X)
            gene_X = xscaler.transform(gene_X)
            xscaler.fit(GX_red)
            GX_red = xscaler.transform(GX_red)

        # Store data for regression runs
        GENE_INFO[g]['X'] = gene_X
        GENE_INFO[g]['Y'] = gene_Y
        GENE_INFO[g]['Xred'] = GX_red
        GENE_INFO[g]['nsamps'] = len(samp_set)
        GENE_INFO[g]['loci_inds'] = loci_inds
     
    # Set up list for output lines
    OLINES = []
    
    # Run regression 
    print '\nRunning %s regressions...'%(regType) 
    #for g in GENE_INFO:
        #try: rr = run_regression(g, GENE_INFO[g], regType, nrep) 
        #except:
            #print g
            #print GENE_INFO[g]['X'][:, GENE_INFO[g]['loci_inds']]
            #print GENE_INFO[g]['rnames']
            #print GENE_INFO[g]['region_names']
    #sys.exit()
    regres = [pool.apply_async(run_regression, args = (g, GENE_INFO[g], regType, nrep)) for g in GENE_INFO]
    g_pvals, gr_pvals, cna_pvals = [], [], []
    GOUT = {}
    for i, res in enumerate(regres):
        rget = res.get()
        g = rget['gene']
        Rsquare = rget['Rsquare']
        beta = rget['beta']
        pvals = rget['pvals']
        Fstat = rget['Fstat']
        Fpv = rget['Fpval']
        nsamps = GENE_INFO[g]['nsamps']
        rnames = GENE_INFO[g]['rnames']
        regnames=GENE_INFO[g]['region_names']
        loci_inds = GENE_INFO[g]['loci_inds']
        sPerReg = GENE_INFO[g]['samps_per_region']

        # get p-value information for later FDR correction
        g_pvals.append(Fpv)
        for li in loci_inds:
            gr_pvals.append([i, pvals[li]])
        cna_pvals.append(pvals[0])

        if regType == 'OLS':           
            OL = [g, nsamps, len(loci_inds), Rsquare, Fstat, Fpv, 1.0,
                  sPerReg, [beta[x] for x in loci_inds], [pvals[x] for x in loci_inds],
                  beta[0], pvals[0], rnames, regnames]
            OLINES.append(OL)
   
        elif regType == 'Ridge' or regType == 'Lasso':
            OL = [g, nsamps, len(loci_inds), Rsquare, rget['alpha'], Fstat, Fpv, 1.0,
                  sPerReg, [beta[x] for x in loci_inds], [pvals[x] for x in loci_inds],
                  beta[0], pvals[0], rnames, regnames]
            OLINES.append(OL)

        elif regType == 'ElasticNet':
            OL = [g, nsamps, len(loci_inds), Rsquare, rget['alpha'], rget['l1_ratio'], Fstat, Fpv, 1.0,
                  sPerReg, [beta[x] for x in loci_inds], [pvals[x] for x in loci_inds],
                  beta[0], pvals[0], rnames, regnames]
            OLINES.append(OL)
    
        #GOUT['regdata'] = rget
        #GOUT['gene_info'] = GENE_INFO[g]

    #cPickle.dump(GOUT, open('ENY2_ElasticNet_data.pkl','w'))
    #sys.exit()

    print 'Regression complete. Writing output...'
     
    # FDR correct
    g_FDR = fdr_BH(np.array(g_pvals))
    cna_FDR = fdr_BH(np.array(cna_pvals))
    gr_FDR = fdr_BH(np.array([x[1] for x in gr_pvals]))
    gr_FDR_dict = {}
    for a,b in zip(gr_pvals, gr_FDR):
        ind = a[0]
        fdr = b
        if ind not in gr_FDR_dict:
            gr_FDR_dict[ind] = []
        gr_FDR_dict[ind].append(fdr)
    
    # Create new output lines
    OLINES_NEW = []
    for i, OL in enumerate(OLINES):
        g_fdr = g_FDR[i]
        cna_fdr = cna_FDR[i]
        gr_fdr = gr_FDR_dict[i]
        gene = OL[0]
        
        for j in range(0, len(gr_fdr)):

            # Set last index of fixed values
            if regType == 'OLS': fixN = 5
            elif regType == 'Ridge' or regType == 'Lasso': fixN = 6
            elif regType == 'ElasticNet': fixN = 7

            OLN = OL[0:fixN+1] # fixed values
            OLN.append(g_fdr) # overall gene F-test p-value
            OLN.append(len(OL[fixN+2][j])) # sample count for loci-gene pair
            OLN += [OL[fixN+3][j], OL[fixN+4][j], gr_fdr[j]] # locus-gene beta, p-value, and FDR    
            OLN += [OL[fixN+5], OL[fixN+6], cna_fdr] # CNA beta, p-value, and FDR
            OLN.append(OL[fixN+7][j]) # loci
            OLN.append(OL[fixN+8][j]) # region names

            OLINES_NEW.append(OLN)

    # Sort output lines by locus-gene p-value and write
    if regType == 'OLS': 
        OLINES_NEW.sort(key = lambda x: x[9]) # sort on locus-gene F-test p-value

        # write output lines, adding FDR p-value
        fmtstr = '%s\t%d\t%d\t%0.3f\t%0.3f\t%0.3g\t%0.3g\t%d\t%0.4f\t%0.3g\t%0.3g\t%0.4f\t%0.3g\t%0.3g\t%s\t%s\n'
        for i, OL in enumerate(OLINES_NEW):
            OF.writelines(fmtstr % tuple(OL))
    
    elif regType == 'Ridge' or regType == 'Lasso':
        OLINES_NEW.sort(key = lambda x: x[10]) # sort on locus-gene F-test p-value

        # write output lines, adding FDR p-value
        fmtstr = '%s\t%d\t%d\t%0.3f\t%0.4f\t%0.3f\t%0.3g\t%0.3g\t%d\t%0.4f\t%0.3g\t%0.3g\t%0.4f\t%0.3g\t%0.3g\t%s\t%s\n'
        for i, OL in enumerate(OLINES_NEW):
            OF.writelines(fmtstr % tuple(OL))
    
    elif regType == 'ElasticNet':
        OLINES_NEW.sort(key = lambda x: x[11]) # sort on locus-gene F-test p-value

        # write output lines, adding FDR p-value
        fmtstr = '%s\t%d\t%d\t%0.3f\t%0.4f\t%0.3f\t%0.3f\t%0.3g\t%0.3g\t%d\t%0.4f\t%0.3g\t%0.3g\t%0.4f\t%0.3g\t%0.3g\t%s\t%s\n'
        for i, OL in enumerate(OLINES_NEW):
            OF.writelines(fmtstr % tuple(OL))
    
    # Close output file and pooll
    OF.close()
    pool.close()
    print '\nDone.'

if __name__ == '__main__': main()


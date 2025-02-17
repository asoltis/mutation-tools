import sys, os, cPickle
import numpy as np
import glob
from pysam import TabixFile

sys.path.insert(0, '') # Include path to MutEnricher code
sys.path.insert(0, '') # Include path to MutEnricher math_funcs directory
from coding_enrichment import Gene
from scipy import stats
import statsmodels.api as SM
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import ShuffleSplit, cross_val_score
import warnings
import multiprocessing as mp

def get_gene_data(genes, atype = 'full_gene', gfdr = 0.1, hfdr = 0.01):
    '''
    
    '''
    # initialize output
    genes_info = {}
    
    # loop over regions, find significant overall regions and hotspots
    ME_sig = []
    for g in genes:
        msig, hs_positions = [], []
        n_hs = len(g.clusters)
        if n_hs > 0 and atype == 'gene_and_hotspot':
            for hs in g.cluster_enrichments:
                hs_reg, hs_fdr, hs_pos, hs_samps = hs[1], hs[8], hs[10], hs[-1]
                if hs_fdr < hfdr:
                    msig.append(['HS_'+hs_reg, hs_samps.split(';')])
                    for pos in hs_pos.split(';'):
                        hs_positions.append(int(pos.split('_')[0]))

        if g.fisher_qval < gfdr:
            if len(msig) == 0:
                mbys = g.mutations_by_sample
                mut_samps = [s for s in mbys if len(mbys[s]['nonsilent']) > 0]
                msig.append([g.name+'_full', mut_samps])
            else:
                rem_pos, rem_samps = set(), set()
                for pos in g.samples_by_positions['nonsilent']:
                    if pos in hs_positions: continue
                    else:
                        rem_pos.add(pos)
                        for s in g.samples_by_positions['nonsilent'][pos]:
                            rem_samps.add(s)
                
                if len(rem_samps)>0:
                    msig.append([g.name+'_remainder', list(rem_samps)])

        if len(msig)>0:
            ME_sig.append([g, msig])

    # fill return variable
    for ME in ME_sig:
        g, msig = ME
        gn = g.name
        genes_info[gn] = {}
        genes_info[gn]['gene_data'] = g
        genes_info[gn]['mdata'] = []
        for ms in msig:
            genes_info[gn]['mdata'].append(ms)

    # return
    return genes_info

def get_gene_data_by_effect(genes, gfdr = 0.1, vcfs_fn = None):
    '''
    Get gene data for significantly mutated genes and extract mutation effect (e.g. SNV, frameshift, stopgain, etc.).
    Return groups based on mutation type categories.
    '''
    # import cyvcf2
    from cyvcf2 import VCF

    # initialize output
    genes_info = {}
    
    # effect terms, ordered by precedence
    elist = ['frameshift_insertion', 'frameshift_deletion', 'frameshift_substitution', 
             'stopgain', 'stoploss', 'nonframeshift_insertion', 'nonframeshift_deletion', 'nonframeshift_substitution',
             'nonsynonymous_SNV', 'splicing']
    terms = ['ExonicFunc.wgEncodeGencodeBasicV28', 'Func.wgEncodeGencodeBasicV28']

    # Load VCF info
    VCFS = {}
    for line in open(vcfs_fn):
        v, s = line.strip().split('\t')
        VCFS[s] = v

    # loop over regions, find significant overall regions, and get effects
    ME_sig = []
    for g in genes:
        msig = []
        if g.fisher_qval < gfdr:
            mbys = g.mutations_by_sample
            mut_samps = [s for s in mbys if len(mbys[s]['nonsilent']) > 0]
            rstr = '%s:%d-%d' % (g.chrom, g.start, g.stop)
            effect_samples = {}
            for samp in mut_samps:
                vfn = VCFS[samp]
                mutstrings = mbys[samp]['nonsilent']
                vcf = VCF(vfn)
                effects = []
                for v in vcf(rstr):
                    if v.FILTER != None: continue
                    POS, REF, ALT = str(v.POS), v.REF.encode('ascii'), v.ALT
                    for a in ALT:
                        alt = a.encode('ascii')
                        mstr = '%s_%s_%s'%(POS, REF, alt)
                        if mstr in mutstrings:
                            for term in terms:
                                eff = v.INFO[term].encode('ascii')
                                if term.startswith('Func'):
                                    if 'splicing' in eff: effects.append('splicing')
                                else: 
                                    if eff != '.': 
                                        if eff in elist: effects.append(eff)
                if len(effects) > 1:
                    if len(set(effects)) == 1:
                        eff = effects[0]
                        if eff not in effect_samples: effect_samples[eff] = []
                        effect_samples[eff].append(samp)
                    else:
                        e_prec = []
                        for eff in set(effects):
                            prec = np.argwhere(np.array(elist)==eff)[0][0]
                            e_prec.append([eff, prec])
                        e_prec.sort(key = lambda x: x[1])
                        eff = e_prec[0][0]
                        if eff not in effect_samples: effect_samples[eff] = []
                        effect_samples[eff].append(samp)
                else:
                    eff = effects[0]
                    if eff not in effect_samples: effect_samples[eff] = []
                    effect_samples[eff].append(samp)

            # add to msig list
            for eff in effect_samples:
                msamps = effect_samples[eff]
                msig.append(['%s_%s' % (g.name, eff), msamps])
         
        # add to ME_sig
        if len(msig)>0:
            ME_sig.append([g, msig])

    # fill return variable
    for ME in ME_sig:
        g, msig = ME
        gn = g.name
        genes_info[gn] = {}
        genes_info[gn]['gene_data'] = g
        genes_info[gn]['mdata'] = []
        for ms in msig:
            genes_info[gn]['mdata'].append(ms)
        
    # return
    return genes_info

def run_regression(gene, GINFO, regType, nrep = 20):
    '''
    Compute various regression procedures on gene data.    
    '''
    # return variable
    REG_RESULTS = {}
    
    # extract X, Y data
    gene_X = GINFO['X']
    GX_red = GINFO['Xred']
    gene_Y = GINFO['Y']
    loci_inds = GINFO['loci_inds']
 
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
    MEfn = '' # MutEnricher coding analysis output *_gene_data.pkl file  
    vcfs_fn = '' # Text file with paths to sample VCF files (used in MutEnricher analysis)

    ## CNA  
    CNA_fn = '' # Text file with gene-wise copy number levels (expressed as log2(CN) - 1)

    ## Tumor purity 
    pur_fn = '' # Per-sample tumor purity estimates from WGS data

    # GTF
    gtf = 'gencode.v28.basic.annotation.gtf' # Gene tranfser format file
    genefield = 'gene_name' 
    goi_fn = '' # Simple text file listing genes of interest for analysis
    goi = [x.strip() for x in open(goi_fn).readlines()]

    ##########
    # PARAMS #
    ##########
    # processors
    nproc = 20

    # analysis type (full_gene or gene_and_hotspot)
    #atype = 'full_gene'
    #atype = 'gene_and_hotspot'
    atype = 'by_effect'

    # gene significance
    gsig = 0.2
    hsfdr = 0.001

    # regression
    standardize = True
    regType = 'OLS'
    nrep = 50

    ##########
    # OUTPUT #
    ##########
    prefix = 'coding_eQTL'
    adate = 'date' 
    ODIR = 'outdir'
    os.system('mkdir -p %s'%(ODIR))
    
    # save input file info to file
    infn = open(ODIR + 'input_files.txt', 'w')
    for fns in [peer_fn, MEfn, vcfs_fn, CNA_fn, gtf, goi_fn]:
        infn.writelines('%s\n'%(fns))
    infn.close()

    # initialize output
    if regType == 'OLS':
        ONAME = ODIR + '%s_somatic_eQTL_%s_regression_geneFisherFDR_%s.txt'%(prefix, regType, str(gsig))
        head = ['Gene', 'gene_nsamps', 'gene_nloci', 'R^2', 'gene_Fstat', 'gene_pval', 'gene_FDR', 
                'locus_nsamps', 'locus_beta', 'locus_pval', 'locus_FDR', 'CNA_beta', 'CNA_pval', 'CNA_FDR', 'locus']
    
    elif regType == 'Ridge' or regType == 'Lasso':
        #pass
        ONAME = ODIR + '%s_somatic_eQTL_%s_regression_geneFisherFDR_%s.txt'%(prefix, regType, str(gsig))
        head = ['Gene', 'gene_nsamps', 'gene_nloci', 'R^2', 'alpha', 'gene_Fstat', 'gene_pval', 'gene_FDR', 
                'locus_nsamps', 'locus_beta', 'locus_pval', 'locus_FDR', 'CNA_beta', 'CNA_pval', 'CNA_FDR', 'locus']
        
    elif regType == 'ElasticNet':
        ONAME = ODIR + '%s_somatic_eQTL_%s_regression_geneFisherFDR_%s.txt'%(prefix, regType, str(gsig))
        head = ['Gene', 'gene_nsamps', 'gene_nloci', 'R^2', 'alpha', 'l1_ratio', 'gene_Fstat', 'gene_pval', 'gene_FDR', 
                'locus_nsamps', 'locus_beta', 'locus_pval', 'locus_FDR', 'CNA_beta', 'CNA_pval', 'CNA_FDR', 'locus']
        
    if atype == 'gene_and_hotspot':
        ONAME.replace('.txt', '_hsFDR_%s.txt'%(hsfdr))
    OF = open(ONAME, 'w')
    OF.writelines('\t'.join(head) + '\n')

    ########
    # PREP #
    ########    
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
    print 'Loading MutEnricher gene data...'
    ME_genes = cPickle.load(open(MEfn))
    if atype in ['full_gene', 'gene_and_hotspot']:
        ME_sig = get_gene_data(ME_genes, atype, gsig, hsfdr) 
    elif atype == 'by_effect':
        ME_sig = get_gene_data_by_effect(ME_genes, gsig, vcfs_fn)
    print 'MutEnricher data loaded.\n'
    
    ## PEER data
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

    ## set up pool
    pool = mp.Pool(processes = nproc)

    #############
    # EXECUTION #
    ############# 
    # Loop over genes and compute regressions
    print 'Extracting gene CNA data and creating regression input data...'
    # Loop over mapped genes and compute regressions
    GENE_INFO = {}
    for gn in ME_sig:
        ginfo = ME_sig[gn]
        gene = ginfo['gene_data']
        # get gene name
        g = gene.name
        #if g not in ['EGFR']: continue
        
        gstart, gend = gene.start, gene.stop
        gstr = '%s:%d-%d'%(gene.chrom, gstart, gend)
        
        # initialize gene info
        GENE_INFO[g] = {}

        # get region data
        mdata = ginfo['mdata']

        # get gene index
        gind = (egenes == g)
        
        # Get mutated sample indices in expression data for gene
        samp_set, samps_per_locus = set(), []
        Xnames, Xsamps = [], []
        for md in mdata:
            xn, xs = md
            for samp in xs: samp_set.add(samp)
            xsamp = []
            for samp in esamps:
                if samp in xs: xsamp.append(1)
                else: xsamp.append(0)
            Xsamps.append(xsamp)
            Xnames.append(xn)
            samps_per_locus.append(xs)
        GENE_INFO[g]['gnames'] = Xnames
        GENE_INFO[g]['samps_per_locus'] = samps_per_locus

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
    
    print 'CNA and regression data gathered.'
    
    # Set up list for output lines
    OLINES = []
    
    # Run regression 
    print '\nRunning %s regressions...'%(regType) 
    #for g in GENE_INFO:
    #    rr = run_regression(g, GENE_INFO[g], regType, nrep)
    #    print rr['gene']
    #    print rr['Fstat'], rr['Fpval']
    #sys.exit()
    gres = [pool.apply_async(run_regression, args = (g, GENE_INFO[g], regType, nrep)) for g in GENE_INFO]
    g_pvals, gr_pvals, cna_pvals = [], [], []
    GOUT = {}
    for i, res in enumerate(gres):
        rget = res.get()
        g = rget['gene']
        Rsquare = rget['Rsquare']
        beta = rget['beta']
        pvals = rget['pvals']
        Fstat = rget['Fstat']
        Fpv = rget['Fpval']
        nsamps = GENE_INFO[g]['nsamps']
        loci_inds = GENE_INFO[g]['loci_inds']
        nloci = len(loci_inds)
        loci = GENE_INFO[g]['gnames']
        sPerLoc = GENE_INFO[g]['samps_per_locus']
        
        # get p-value information for later FDR correction
        g_pvals.append(Fpv)
        for li in loci_inds:
            gr_pvals.append([i, pvals[li]])
        cna_pvals.append(pvals[0])

        if regType == 'OLS':           
            OL = [g, nsamps, nloci, Rsquare, Fstat, Fpv, 1.0,
                  sPerLoc, [beta[x] for x in loci_inds], [pvals[x] for x in loci_inds],
                  beta[0], pvals[0], loci]
            OLINES.append(OL)
  
        elif regType == 'Ridge' or regType == 'Lasso':
            OL = [g, nsamps, len(loci_inds), Rsquare, rget['alpha'], Fstat, Fpv, 1.0,
                  sPerLoc, [beta[x] for x in loci_inds], [pvals[x] for x in loci_inds],
                  beta[0], pvals[0], loci]
            OLINES.append(OL)

        elif regType == 'ElasticNet':
            OL = [g, nsamps, nloci, Rsquare, rget['alpha'], rget['l1_ratio'], Fstat, Fpv, 1.0,
                  sPerLoc, [beta[x] for x in loci_inds], [pvals[x] for x in loci_inds],
                  beta[0], pvals[0], loci]
            OLINES.append(OL)
        
        #GOUT['regdata'] = rget
        #GOUT['gene_info'] = GENE_INFO[g]

    ## below lines used to create data for plotting
    #cPickle.dump(GOUT, open('EGFR_ElasticNet_data_by_effect.pkl','w'))
    #sys.exit()

    print 'Regressions complete. Writing output...'

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
        
            OLINES_NEW.append(OLN)

    # Sort output lines by locus-gene p-value and write
    if regType == 'OLS': 
        OLINES_NEW.sort(key = lambda x: x[9]) # sort on locus-gene F-test p-value

        # write output lines, adding FDR p-value
        fmtstr = '%s\t%d\t%d\t%0.3f\t%0.3f\t%0.3g\t%0.3g\t%d\t%0.4f\t%0.3g\t%0.3g\t%0.4f\t%0.3g\t%0.3g\t%s\n'
        for i, OL in enumerate(OLINES_NEW):
            OF.writelines(fmtstr % tuple(OL))
    
    elif regType == 'Ridge' or regType == 'Lasso':
        OLINES_NEW.sort(key = lambda x: x[10]) # sort on locus-gene F-test p-value

        # write output lines, adding FDR p-value
        fmtstr = '%s\t%d\t%d\t%0.3f\t%0.4f\t%0.3f\t%0.3g\t%0.3g\t%d\t%0.4f\t%0.3g\t%0.3g\t%0.4f\t%0.3g\t%0.3g\t%s\n'
        for i, OL in enumerate(OLINES_NEW):
            OF.writelines(fmtstr % tuple(OL))
    
    elif regType == 'ElasticNet':
        OLINES_NEW.sort(key = lambda x: x[11]) # sort on locus-gene F-test p-value

        # write output lines, adding FDR p-value
        fmtstr = '%s\t%d\t%d\t%0.3f\t%0.4f\t%0.3f\t%0.3f\t%0.3g\t%0.3g\t%d\t%0.4f\t%0.3g\t%0.3g\t%0.4f\t%0.3g\t%0.3g\t%s\n'
        for i, OL in enumerate(OLINES_NEW):
            OF.writelines(fmtstr % tuple(OL))
    
    # Close output file and pooll
    OF.close()
    pool.close()
    print '\nDone.'

if __name__ == '__main__': main()


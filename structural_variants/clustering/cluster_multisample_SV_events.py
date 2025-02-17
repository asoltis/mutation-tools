from __future__ import division
import sys, os, re
import argparse
import networkx as nx
import numpy as np
from cyvcf2 import VCF
import interlap
from collections import defaultdict, Counter
from pysam import TabixFile
import scipy.cluster.hierarchy as sch
import multiprocessing as mp
from scipy.stats import poisson

'''
Updates:

10-23-2018:
    - Included ability to read in RNA fusion calls for samples and integrate with SVs for clustering.

10-30-2018:
    - Included handling for recipricol inversion calls (i.e. those with both INV3 and INV5 calls).
    - This handling merges the two calls into one, taking the average start/stop coordinates

09-11-2020:
    - Update issue of ID_map object containing old IDs merged from INV35 calls. 
    - Fix removes old maps and updates to new INV35 intid.
'''

def dist_fun(a,b):
    '''

    '''
    d1=np.abs(a[0]-b[0])/3e9
    d2=np.abs(a[1]-b[1])/3e9
    d3=np.abs(a[0]-b[1])/3e9
    d4=np.abs(a[1]-b[0])/3e9
    dist1 = d1*d2
    dist2 = d3*d4
    dist = min(dist1,dist2) 
    
    return dist

def cluster_vars(olap_inter, clust_thresh):
    '''
    '''
    # calculate overlap member distances
    nolap = len(olap_inter)
    labels = []
    for i in range(0,nolap): labels.append(olap_inter[i][2])
    dists= []
    for i in range(0,nolap-1):
        for j in range(i+1,nolap):
            a,b = olap_inter[i],olap_inter[j] 
            dist = dist_fun(a,b)
            dists.append(dist)
                        
    # Agglomerative clustering
    Y1 = sch.linkage(np.array(dists),method='single') 
    clusters = {}
    for i in range(0,Y1.shape[0]):
        clust_n = i + nolap
        x1, x2, dist = Y1[i,0], Y1[i,1], Y1[i,2]
        
        if dist > clust_thresh: continue
        
        this_clust = []
        for x in [x1,x2]:
            if x > (nolap-1): 
                this_clust += clusters[x]
                del(clusters[x])
            else: this_clust.append(x)
        clusters[clust_n] = this_clust
    
    clust_members = {}
    for clust_n in clusters:
        members = [labels[int(i)] for i in clusters[clust_n]]
        clust_members[clust_n] = members

    return clust_members

def entropy(positions):
    ''' 
    Calculate SV-type-weighted entropy of start and stop positions of clustered SVs.
    '''

    startE, stopE = {}, {}
    for invtype in positions:
        # get start position entropy for inversion type
        startC = Counter(positions[invtype]['starts'])
        nstarts = positions[invtype]['count']
        if nstarts == 1:
            sE = 1.0 # definition
        elif nstarts > 1:
            sE = 0
            for s in startC:
                prob = startC[s] / nstarts
                sE += (prob * np.log(prob) / np.log(nstarts))
            sE = -sE
        startE[invtype] = sE
        
        # get stop position entropy for inversion type
        stopC = Counter(positions[invtype]['stops'])
        nstops = positions[invtype]['count']
        if nstops == 1:
            sE = 1.0 # definition
        elif nstops > 1:
            sE = 0
            for s in stopC:
                prob = stopC[s] / nstops
                sE += (prob * np.log(prob) / np.log(nstops))
            sE = -sE
        stopE[invtype] = sE

    # get weighted entropy
    startWeightEnt, startTot = 0, 0
    for invtype in startE:
        count = positions[invtype]['count']
        startTot += count
        startWeightEnt += (count * startE[invtype])
    startWeightEnt /= startTot
    
    stopWeightEnt, stopTot = 0, 0
    for invtype in stopE:
        count = positions[invtype]['count']
        stopTot += count
        stopWeightEnt += (count * stopE[invtype])
    stopWeightEnt /= stopTot

    return startWeightEnt, stopWeightEnt

def read_manta_contiguous(SV_vcfs,sample_names,fusions,distT,rcT,outdir,prefix,mapr,nproc,skip_stats=False):
    '''
    
    '''
    # set up for storing intervals (as InterLap objects)
    inter = defaultdict(interlap.InterLap)
    
    # mapping dictionary for reduced IDs back to Mant IDs
    ID_map = {}

    # other info
    valid_chroms = ['chr%s'%(str(x)) for x in range(1,23)] + ['chrX']
  
    # storage dictionary for tracking reciprocal inversions
    INV_events = {}

    # Loop over samples, perform analysis
    nsamps = len(SV_vcfs)
    sampnum = 0
    probs = {}
    print ' Reading VCFs...'
    TOT_VARS=0
    for (vfn, name) in zip(SV_vcfs,sample_names):
        count = 1
        ID_track = set()
        sampnum += 1 
        if sampnum % 10 == 0: print '  reading VCF %d of %d...'%(sampnum, nsamps)
        for v in VCF(vfn):
            if v.FILTER != None: continue # take only PASS
            ID = v.ID
            if 'Canvas' in ID: continue
            CHROM, POS, SVTYPE = v.CHROM.encode('ascii'), v.POS, v.INFO['SVTYPE'].encode('ascii')
            if CHROM not in valid_chroms: continue
             
            if SVTYPE == 'BND': continue
            elif SVTYPE == 'INV':
                inv3,inv5 = False,False
                try: inv5 = v.INFO['INV5']
                except KeyError: pass
                try: inv3 = v.INFO['INV3']
                except KeyError: pass

                if inv5: SVTYPE = 'INV5'
                elif inv3: SVTYPE = 'INV3'
            END = v.INFO['END']
            
            # other info
            LEN = END-POS + 1
            SCORE = v.INFO['SOMATICSCORE']
            PR = v.format('PR')[1,1] # want second sample (index 1) and second value for alt counts
            if 'SR' in v.FORMAT: SR = v.format('SR')[1,1]
            else: SR = 0
            
            ### FILTERING ###
            #if SVTYPE == 'INV': continue ### TESTING ###
            #if LEN > 10e6: continue ### TESTING ###
            #if SCORE < 60: continue ### TESTING ###
            if LEN <= 30: continue
            if SVTYPE == 'DEL' and LEN < 1000: continue
            if (PR + SR) < rcT: continue
            ### END FILTERING ###

            # check for position and end confidence intervals
            # if found, adjust start and end coordinates
            cpos, cend = False, False
            s1, e1, s2, e2 = 0, 0, 0, 0
            try:
                CIPOS = v.INFO['CIPOS']
                cpos = True
            except KeyError: pass
            try:
                CIEND = v.INFO['CIEND']
                cend = True
            except KeyError: pass
            if cpos:
                span = v.INFO['CIPOS']
                s1 = POS + span[0]
                e1 = POS + span[1]
                POS += span[0] # lower bound, taking left-most
            else: 
                s1, e1 = POS, POS
            if cend:
                span = v.INFO['CIEND']
                s2 = END + span[0]
                e2 = END + span[1]
                END += span[1] # upper bound, taking right-most
            else: 
                s2, e2 = END, END

            # if breakpoint inspector calls used, overwrite positions with these
            bpi_start, bpi_end = False, False
            try:
                BPI_start = v.INFO['BPI_START']
                bpi_start = True
            except KeyError: pass
            try:
                BPI_end = v.INFO['BPI_END']
                bpi_end = True
            except KeyError: pass
            if bpi_start:
                s1, e1, POS = BPI_start, BPI_start, BPI_start
            if bpi_end:
                s2, e2, END = BPI_end, BPI_end, BPI_end

            # Check mappability of intervals if data provided
            if mapr != None:
                mappable = True
                for r in [(s1,e1),(s2,e2)]:
                    rlen = r[1]-r[0]+1
                    mlen = 0
                    rstr = '%s:%d-%d'%(CHROM,r[0],r[1])
                    rq = [t.split('\t') for t in mapr.fetch(rstr)]
                    if len(rq) == 0:
                        mappable = False
                        break
                    else:
                        for t in rq:
                            qs, qe = int(t[1])+1,int(t[2])
                            if qs < r[0]: ostart = r[0]
                            else: os = qs
                            if qe > r[1]: ostop = r[1]
                            else: ostop = qe
                            mlen += (ostop-ostart+1)

                if mlen < rlen: 
                    mappable = False

                if mappable == False: continue

            # name interval
            intid = '%s_%d_%s:%d-%d_%s'%(name,count,CHROM,POS,END,SVTYPE)
            count += 1
            ID_map[intid] = ID

            # add to interlap dictionary
            inv_event = None
            try: inv_event = v.INFO['EVENT']
            except KeyError: pass
        
            # UPDATE 10-30-2018
            if (SVTYPE == 'INV3' or SVTYPE == 'INV5') and inv_event != None:
                if inv_event not in INV_events: INV_events[inv_event] = []
                INV_events[inv_event].append(intid)

            else:
                inter[CHROM].add((POS,END,intid))
                TOT_VARS += 1
    
    # UPDATE 10-30-2018
    # handle inversion events
    for inv_event in INV_events:
        inversions = INV_events[inv_event]
        if len(inversions) == 1:
            inv_split = inversions[0].split('_')
            region = inv_split[2]
            svt = inv_split[3]
            chrom = region.split(':')[0]
            start, stop = map(int, region.split(':')[1].split('-'))

            inter[chrom].add((start, stop, inversions[0]))
            TOT_VARS += 1
        else:
            iname, ichrom = None, None
            starts, stops, icounts, types = [], [], [], []
            for i, inv in enumerate(inversions):
                inv_split = inv.split('_')
                region = inv_split[2]
                if i == 0:
                    iname = inv_split[0]
                    ichrom = region.split(':')[0]
                start, stop = map(int, region.split(':')[1].split('-'))
                ic = inv_split[1]
                typ = inv_split[-1]

                starts.append(start)
                stops.append(stop)
                icounts.append(ic)
                types.append(typ)

            # re-create as new SV
            start = int(round(np.mean(starts)))
            stop = int(round(np.mean(stops)))
            count = '-'.join(icounts)
            if 'INV3' in types and 'INV5' in types: typ = 'INV35'
            else: typ = list(set(types))[0]

            intid = '%s_%s_%s:%d-%d_%s'%(iname, count, ichrom, start, stop, typ)
            inter[ichrom].add((start, stop, intid))
            TOT_VARS += 1
           
            # Update ID map - added 2020-09-11
            mapids = []
            for inv in inversions:
                idm = ID_map[inv]
                mapids.append(idm)
                del(ID_map[inv])
            ID_map[intid] = mapids
    
    print ' Loaded %d total intrachromosomal SVs.\n'%(TOT_VARS)
    
    # UPDATE: 10-23-2018 for fusions
    if fusions != None:
        TOT_FUS = 0
        for name in fusions:
            fusfn = fusions[name]
            for line in open(fusfn):
                l = line.strip('\n').split('\t')
                chroms, dend, astart, fid = tuple(l[0:4])
                dchrom, achrom = chroms.split('~')
                if dchrom != achrom: continue # only consider intrachromosomal
                if dchrom not in valid_chroms or achrom not in valid_chroms: continue
                dend, astart = map(int,[dend, astart])
                if dend > astart:
                    pos = [dend, astart]
                    dend, astart = pos[1], pos[0] # Flip 

                if mapr != None:
                    mappable = True
                    for c,p in zip([dchrom,achrom],[dend,astart]):
                        rstr = '%s:%d-%d'%(c,p,p)
                        try: rq = [t.split('\t') for t in mapr.fetch(rstr)]
                        except: continue
                        if len(rq) == 0:
                            mappable = False
                            break
                    
                    if mappable == False: continue

                # Add passing fusion
                fusid = '%s_%s_%s:%d-%d_%s'%(name,fid.replace('_',''),dchrom,dend,astart,'FUS')
                ID_map[fusid] = fid
                inter[dchrom].add((dend,astart,fusid))
                TOT_FUS += 1

        print ' Loaded %d total intrachromosomal fusions.\n'%(TOT_FUS)
    
    # find overlap intervals and merge close ones (e.g. within 1 kb)
    olap_intervals = {}
    for chrom in inter:
        prior = None
        was_olap = False
        intervals = []
        olap_inters = interlap.reduce([(x[0],x[1]) for x in inter[chrom]])
        nolap = len(olap_inters)
        for i,oi in enumerate(olap_inters):
            if prior == None: 
                prior = [oi[0],oi[1]]
                if nolap == 1: intervals.append(tuple(prior))
                continue
            else:
                if (oi[0] - prior[1]) < distT:
                    was_olap = True
                    prior[1] = oi[1]
                    if (i+1) == nolap: # for last case
                        intervals.append(tuple(prior))
                else:
                    intervals.append(tuple(prior))
                    prior = [oi[0],oi[1]]
                    if (i+1) == nolap:
                        intervals.append(tuple(prior))
        olap_intervals[chrom] = intervals
   
    # Cluster regions by proximity similarity
    print ' Performing SV clustering analysis...'
    OLAP_INTERS = []
    SINGLETONS = []
    for chrom in inter:
        olap_inters = olap_intervals[chrom]
        for oi in olap_inters:
            olap_inter = list(inter[chrom].find((oi[0],oi[1])))
            if len(olap_inter)>1: OLAP_INTERS.append(olap_inter)
            elif len(olap_inter)==1: SINGLETONS.append(olap_inter[0])

    pool = mp.Pool(processes=nproc)
    dones,was_done = [],0
    clust_thresh = pow((distT/3e9),2)            
    res = [pool.apply_async(cluster_vars,args=(IO,clust_thresh),callback=dones.append) for IO in OLAP_INTERS]
    while len(dones) != len(OLAP_INTERS):
        if len(dones)%10==0 and was_done != len(dones):
            was_done = len(dones)
            print '  %d of %d clustering tasks complete.'%(len(dones),len(OLAP_INTERS))
    
    # Get cluster info
    print ' Clustering analysis complete.\n'
    print ' Computing statistics...'
    OLINES0 = []
    s_in_clusters, tot_clusters = 0, 0
    vars_used = set()
    for r in res:
        cluster = r.get()
        for cn in cluster:
            members = cluster[cn]
            for m in members: vars_used.add(m)
            chrom = members[0].split('_')[-2].split(':')[0]
            member_range = interlap.reduce([tuple(map(int,m.split('_')[-2].split(':')[1].split('-'))) for m in members])
            rstart, rend = member_range[0][0], member_range[-1][1]
            member_samples = sorted(list(set(['_'.join(m.split('_')[0:-3]) for m in members])))
            nmemsamps = len(member_samples)
            nmembers = len(members)
            SV_types = ['DEL','DUP','INS','INV3','INV5','INV35']
            if fusions != None: SV_types += ['FUS']
            SV_counts = Counter([x.split('_')[-1] for x in members])
            SV_cnt_str = []
            for svt in SV_types:
                try: count = SV_counts[svt]
                except KeyError: count = 0
                SV_cnt_str.append('%s=%d'%(svt,count))
            SV_cnt_str = ':'.join(SV_cnt_str)

            # get entropy
            POS = {}
            for m in members:
                typ = m.split('_')[-1]
                if typ not in POS:
                    POS[typ] = {}
                    POS[typ]['starts'] = []
                    POS[typ]['stops'] = []
                    POS[typ]['count'] = 0
                start, stop = map(int,m.split('_')[-2].split(':')[1].split('-'))
                POS[typ]['starts'].append(start)
                POS[typ]['stops'].append(stop)
                POS[typ]['count'] += 1
            startEnt, stopEnt = entropy(POS)

            ol = [chrom,str(rstart),str(rend),str(rend-rstart+1),startEnt,stopEnt,
                  nmemsamps,nmembers,None,SV_cnt_str,';'.join(member_samples),';'.join(members)]
            OLINES0.append(ol)

            s_in_clusters += nmemsamps
            tot_clusters += 1
    
    # Get singleton info
    for sng in SINGLETONS:
        member = sng[2]
        vars_used.add(member)
        chrom = member.split('_')[-2].split(':')[0]
        rstart, rend = sng[0], sng[1]
        member_samples = '_'.join(member.split('_')[0:-3])
        nmemsamps = 1
        nmembers = 1
        SV_types = ['DEL','DUP','INS','INV3','INV5','INV35']
        if fusions != None: SV_types += ['FUS']
        SV_counts = Counter([member.split('_')[-1]])
        SV_cnt_str = []
        for svt in SV_types:
            try: count = SV_counts[svt]
            except KeyError: count = 0
            SV_cnt_str.append('%s=%d'%(svt,SV_counts[svt]))
        SV_cnt_str = ':'.join(SV_cnt_str)

        ol = [chrom,str(rstart),str(rend),str(rend-rstart+1),1.0,1.0,
              nmemsamps,nmembers,None,SV_cnt_str,member_samples,member]
        OLINES0.append(ol)

        #s_in_clusters += nmemsamps
        #tot_clusters += 1

    # Get unaccounted for variants
    for member in ID_map:
        if member not in vars_used:
            chrom = member.split('_')[-2].split(':')[0]
            rstart, rend = map(int,member.split('_')[-2].split(':')[1].split('-'))
            member_samples = '_'.join(member.split('_')[0:-3])
            nmemsamps = 1
            nmembers = 1
                   
            SV_types = ['DEL','DUP','INS','INV3','INV5','INV35']
            if fusions != None: SV_types += ['FUS']
            SV_counts = Counter([member.split('_')[-1]])
            SV_cnt_str = []
            for svt in SV_types:
                try: count = SV_counts[svt]
                except KeyError: count = 0
                SV_cnt_str.append('%s=%d'%(svt,SV_counts[svt]))
            SV_cnt_str = ':'.join(SV_cnt_str)

            ol = [chrom,str(rstart),str(rend),str(rend-rstart+1),1.0,1.0,
                  nmemsamps,nmembers,None,SV_cnt_str,member_samples,member]
            OLINES0.append(ol)
 
            #s_in_clusters += nmemsamps
            #tot_clusters += 1
    
    # Calculate stats and positional entropy
    OLINES = []
    if skip_stats:
        for ol in OLINES0:
            ol[8] = np.nan
            OLINES.append(ol)
        del(OLINES0)
    else:
        mu = s_in_clusters / tot_clusters # Poisson rate parameter
        print ' Mu parameter for Poisson tests: %0.3f\n'%(mu)
        for ol in OLINES0:
            K = ol[6] # number of member samples
            pval = 1
            for k in range(0,K): # Loop over 0 to K-1
                pval -= poisson.pmf(k,mu)
            pval = max(0,pval)

            ol[8] = pval
            OLINES.append(ol)
        del(OLINES0)

    # Sort and write output
    print ' Writing output...'
    OLINES.sort(key=lambda x: x[8]) 
    # set up output
    of = open(outdir+prefix+'SVs.txt','w')
    
    of.writelines('\t'.join(['chrom','start','end','length','entropy_starts','entropy_ends',
                             'nsamps','nvars','poisson_pval','SV_type_counts','samples','vars'])+'\n')
 
    for ol in OLINES:
        ol[4] = '%0.3f'%(ol[4])
        ol[5] = '%0.3f'%(ol[5])
        ol[6] = str(ol[6])
        ol[7] = str(ol[7])
        ol[8] = '%0.5g'%(ol[8])
        of.writelines('\t'.join(ol)+'\n')

    # close output file
    of.close()
    print ' Done.'
    
def read_manta_breakend(SV_vcfs,sample_names,fusions,distT,rcT,outdir,prefix,mapr,nproc,skip_stats=False):
    '''
    
    '''  
    # mapping dictionary for reduced IDs back to Mant IDs
    ID_map = {}

    # other info
    valid_chroms = ['chr%s'%(str(x)) for x in range(1,23)] + ['chrX']

    # set up graph
    G = nx.Graph()

    # set up breakend storage
    breakends = []

    # Loop over samples, perform analysis
    nsamps = len(SV_vcfs)
    sampnum = 0
    print ' Reading VCFs...'
    for (vfn, name) in zip(SV_vcfs,sample_names):
        count = 1
        ID_track = set()
        sampnum += 1
        if sampnum % 10 == 0: print '  reading VCF %d of %d...'%(sampnum, nsamps)
        for v in VCF(vfn):
            if v.FILTER != None: continue # take only PASS
            ID = v.ID
            if 'Canvas' in ID: continue
            CHROM1, POS1, SVTYPE, ALT = v.CHROM.encode('ascii'), v.POS, v.INFO['SVTYPE'].encode('ascii'), v.ALT[0].encode('ascii') 
            if SVTYPE != 'BND': continue
            
            # check for mate info - only keep one
            if ID in ID_track: continue
            else: 
                mateid = v.INFO['MATEID']
                ID_track.add(mateid)

            # split ALT string to get second breakend position
            bndchar = '['
            if bndchar not in ALT: bndchar = ']'
            r = re.compile(r'\%s(.+?)\%s'%(bndchar, bndchar))
            CHROM2, POS2 = r.findall(ALT)[0].split(':')
            POS2 = int(POS2)

            if CHROM1 not in valid_chroms or CHROM2 not in valid_chroms: continue

            # other info
            SCORE = v.INFO['SOMATICSCORE']
            PR = v.format('PR')[1,1] # want second sample (index 1) and second value for alt counts
            if 'SR' in v.FORMAT: SR = v.format('SR')[1,1]
            else: SR = 0
            
            ### FILTERING ###
            #if SVTYPE == 'INV': continue ### TESTING ###
            #if LEN > 10e6: continue ### TESTING ###
            #if SCORE < 60: continue ### TESTING ###
            if (PR + SR) < rcT: continue
            ### END FILTERING ###

            # check for position and end confidence intervals
            # if found, adjust start and end coordinates
            cpos, cend = False, False
            s1, e1, s2, e2 = 0, 0, 0, 0
            try:
                CIPOS = v.INFO['CIPOS']
                cpos = True
            except KeyError: pass
            try:
                CIEND = v.INFO['CIEND']
                cend = True
            except KeyError: pass
            if cpos:
                span = v.INFO['CIPOS']
                #POS += span[0] # lower bound, taking left-most
                s1 = POS1 + span[0]
                e1 = POS1 + span[1]
            else: 
                s1, e1 = POS1, POS1
            if cend:
                span = v.INFO['CIEND']
                #END += span[1] # upper bound, taking right-most
                s2 = POS2 + span[0]
                e2 = POS2 + span[1]
            else: 
                s2, e2 = POS2, POS2
         
            # if breakpoint inspector calls used, overwrite positions with these
            bpi_start, bpi_end = False, False
            try:
                BPI_start = v.INFO['BPI_START']
                bpi_start = True
            except KeyError: pass
            try:
                BPI_end = v.INFO['BPI_END']
                bpi_end = True
            except KeyError: pass
            if bpi_start:
                s1, e1, POS1 = BPI_start, BPI_start, BPI_start
            if bpi_end:
                s2, e2, POS2 = BPI_end, BPI_end, BPI_end

            # Check mappability of intervals if data provided
            if mapr != None:
                mappable = True
                for r in [(CHROM1,s1,e1),(CHROM2,s2,e2)]:
                    rlen = r[2]-r[1]+1
                    mlen = 0
                    rstr = '%s:%d-%d'%(r[0],r[1],r[2])
                    rq = [t.split('\t') for t in mapr.fetch(rstr)]
                    if len(rq) == 0:
                        mappable = False
                        break
                    else:
                        for t in rq:
                            qs, qe = int(t[1])+1,int(t[2])
                            if qs < r[1]: ostart = r[1]
                            else: os = qs
                            if qe > r[2]: ostop = r[2]
                            else: ostop = qe
                            mlen += (ostop-ostart+1)

                if mlen < rlen: 
                    mappable = False
                if mappable == False: continue

            # name interval
            intid = '%s_%d_%s:%d--%s:%d_%s'%(name,count,CHROM1,POS1,CHROM2,POS2,'BND')
            count += 1
            ID_map[intid] = ID_map
            
            breakend = [(CHROM1,POS1,s1,e1),(CHROM2,POS2,s2,e2),intid]
            breakends.append(breakend)

    # UPDATE: 10-23-2018 for fusions
    if fusions != None:
        TOT_FUS = 0
        for name in fusions:
            fusfn = fusions[name]
            for line in open(fusfn):
                l = line.strip('\n').split('\t')
                chroms, dend, astart, fid = tuple(l[0:4])
                dchrom, achrom = chroms.split('~')
                if dchrom == achrom: continue # only consider interchromosomal
                if dchrom not in valid_chroms or achrom not in valid_chroms: continue
                dend, astart = map(int,[dend, astart])
                
                if mapr != None:
                    mappable = True
                    for c,p in zip([dchrom,achrom],[dend,astart]):
                        rstr = '%s:%d-%d'%(c,p,p)
                        try: rq = [t.split('\t') for t in mapr.fetch(rstr)]
                        except: continue
                        if len(rq) == 0:
                            mappable = False
                            break
                    
                    if mappable == False: continue

                # Add passing fusion
                fusid = '%s_%s_%s:%d--%s:%d_%s'%(name,fid.replace('_',''),dchrom,dend,achrom,astart,'FUS')
                ID_map[fusid] = fid
                
                breakend = [(dchrom,dend,dend,dend),(achrom,astart,astart,astart),fusid]
                breakends.append(breakend)
                TOT_FUS += 1

        print ' Loaded %d total intrachromosomal fusions.\n'%(TOT_FUS)
 
    # Loop over breakends, compute distances, and add to graph
    print ' Creating breakend graph...'
    dist_thresh = pow((distT/3e9),2)            
    for i in range(0,len(breakends)-1):
        for j in range(i+1,len(breakends)):
            be1, be2 = breakends[i], breakends[j]
            c11, c12 = be1[0][0], be1[1][0] # first breakend chroms
            c21, c22 = be2[0][0], be2[1][0] # second breakend chroms

            if c11 != c21 or c12 != c22: continue 
            else:
                a = [be1[0][1],be1[1][1]]
                b = [be2[0][1],be2[1][1]]
                dist = dist_fun(a,b)
                
                if dist < dist_thresh:
                    edge = (be1[-1],be2[-1])
                    G.add_edge(*edge)

    # determine connected components of graph and write to output 
    print ' Finding connected components and writing output...'
    ConnComp = sorted(nx.connected_components(G), key = len, reverse = True)
    s_in_clusters, tot_clusters = 0, 0
    OLINES0 = []
    for cc in ConnComp:
        IDs = set([x.split('_')[0] for x in cc])
        numIDs = len(IDs)
        p1 = [int(x.split('_')[-2].split('--')[0].split(':')[1]) for x in cc]
        p2 = [int(x.split('_')[-2].split('--')[1].split(':')[1]) for x in cc]
        c1 = [x.split('_')[-2].split('--')[0].split(':')[0] for x in cc][0]
        c2 = [x.split('_')[-2].split('--')[1].split(':')[0] for x in cc][0]
        
        s1, e1 = np.min(p1), np.max(p1)
        s2, e2 = np.min(p2), np.max(p2)

        members = sorted(list(cc))
        member_samples = sorted(list(set(['_'.join(m.split('_')[0:-3]) for m in members])))
        
        SV_types = ['BND']
        if fusions != None: SV_types += ['FUS']
        SV_counts = Counter([x.split('_')[-1] for x in members])
        SV_cnt_str = []
        for svt in SV_types:
            try: count = SV_counts[svt]
            except KeyError: count = 0
            SV_cnt_str.append('%s=%d'%(svt,count))
        SV_cnt_str = ':'.join(SV_cnt_str)


        ol = [c1,str(s1),str(e1),c2,str(s2),str(e2),len(member_samples),len(members),None,SV_cnt_str,';'.join(member_samples),';'.join(members)]
        OLINES0.append(ol)

        s_in_clusters += len(member_samples)
        tot_clusters += 1

    # get singletons - breakends not in graph
    nodes = G.nodes()
    for be in breakends:
        if be[-1] not in nodes:
            sample = '_'.join(be[2].split('_')[0:-3])
            members = [be[2]]

            SV_types = ['BND']
            if fusions != None: SV_types += ['FUS']
            SV_counts = Counter([x.split('_')[-1] for x in members])
            SV_cnt_str = []
            for svt in SV_types:
                try: count = SV_counts[svt]
                except KeyError: count = 0
                SV_cnt_str.append('%s=%d'%(svt,count))
            SV_cnt_str = ':'.join(SV_cnt_str)

            ol = [be[0][0],str(be[0][2]),str(be[0][3]),str(be[1][0]),str(be[1][2]),str(be[1][3]),
                  1,1,None,SV_cnt_str,sample,be[2]]
            OLINES0.append(ol)

    # Calculate stats
    OLINES = []
    if skip_stats:
        for ol in OLINES0:
            ol[8] = np.nan
            OLINES.append(ol)
        del(OLINES0)
    else:
        mu = s_in_clusters / tot_clusters # Poisson rate parameter
        print ' Mu parameter for Poisson tests: %0.3f\n'%(mu)
         
        for ol in OLINES0:
            K = ol[6] # number of member samples
            pval = 1
            for k in range(0,K): # Loop over 0 to K-1
                pval -= poisson.pmf(k,mu)
            pval = max(0,pval)

            ol[8] = pval
            OLINES.append(ol)
        del(OLINES0)

    # set up output
    of = open(outdir+prefix+'breakends.txt','w')
    of.writelines('\t'.join(['chrom1','start1','end1','chrom2','start2','end2','nsamps','nvars','Poisson_pval','SV_type_counts','samples','vars'])+'\n')

    OLINES.sort(key=lambda x: x[8])
    for ol in OLINES:
        ol[6] = str(ol[6])
        ol[7] = str(ol[7])
        ol[8] = '%0.5g'%(ol[8])
        of.writelines('\t'.join(ol)+'\n')
    
    of.close()
    print ' Done.'

def main():

    usage = ''
    description = ''
    parser = argparse.ArgumentParser(usage=usage,description=description,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('SV_vcfs',metavar='SV_vcfs',type=str,
                        help='List of Manta (or HAS SV) vcf files and IDs (two columns, no header) (Required).')
    parser.add_argument('-f','--fusions',type=str,default=None,dest='fusions_fn',
                        help='Provide file list of MapSplice fusions results (same format as input SV_vcfs).')
    parser.add_argument('-o','--outdir',type=str,default='./',dest='outdir',
                        help='Provide output directory for analysis.')
    parser.add_argument('--prefix',type=str,default='SV',dest='prefix',
                        help='Provide analysis prefix.')
    parser.add_argument('-p','--processors',dest='nprocessors',type=int,default=1,
                        help='Select number of processors.')
    parser.add_argument('-t','--threshold',dest='distT',type=float,default=1000.0,
                        help='Select distance threshold (in bp) for cluster identification.')
    parser.add_argument('--rc-thresh',dest='rcT',type=int,default=5,
                        help='Set somatic read count threshold for inclusion in analysis. This threshold is compared against the sum of somatic\
                        paired-read and split-read evidences (i.e. PR+SR) for the variant.')
    parser.add_argument('-m','--mappable-regions',type=str,default=None,dest='mappable',
                        help='Provide mappable regions a bgzipped, tabix-indexed BED file.')
    parser.add_argument('--skip-stats', action='store_true',dest='skip_stats',
                        help='Set this flag to skip calculation of Poisson stats; NaN reported in stats output columns.')

    args = parser.parse_args()
    vargs = vars(args)

    # Parse required
    SV_vcfs_fn = vargs['SV_vcfs']
    if not os.path.isfile(SV_vcfs_fn): parser.error('SV vcfs list file not found!')
    SV_vcfs, sample_names = [],[]
    for line in open(SV_vcfs_fn):
        vcf, sample = tuple(line.strip().split('\t'))
        if not os.path.isfile(vcf): parser.error('VCF file for sample %s does not exist: %s'%(sample,vcf))
        SV_vcfs.append(vcf)
        sample_names.append(sample)
    nsamps = len(sample_names)
    if len(set(sample_names)) < nsamps: parser.error('Non-unique sample IDs provided. Exiting.')

    # Options
    fusions_fn = vargs['fusions_fn']
    fusions = None
    if fusions_fn != None: 
        if not os.path.isfile(fusions_fn): parser.error('Fusions file list not found!')
        fusions = {}
        for line in open(fusions_fn):
            fusfn, sample = tuple(line.strip().split('\t'))
            if not os.path.isfile(fusfn): parser.error('Fusions file for sample %s does not exist: %s'%(sample, fusfn))
            if sample not in sample_names:
                print ' no SV vcf provided for fusion sample %s; skipping.'%(sample)
                continue
            fusions[sample] = fusfn
    outdir = vargs['outdir']
    os.system('mkdir -p %s'%(outdir))
    if not outdir.endswith('/'): outdir += '/'
    prefix = vargs['prefix'] + '_'
    nprocessors = min(mp.cpu_count(),vargs['nprocessors'])
    mappable = vargs['mappable']
    distT = vargs['distT']
    rcT = vargs['rcT']
    if mappable != None:
        if not os.path.isfile(mappable) or not os.path.isfile(mappable+'.tbi'): parser.error('Mappability file and/or its index not found!')
        mapr = TabixFile(mappable)
    else: mapr = None
    skip_stats = vargs['skip_stats']

    # RUN FUNCTIONS #
    print '---Performing intrachromosomal SV clustering analysis---'
    read_manta_contiguous(SV_vcfs, sample_names, fusions, distT, rcT, outdir, prefix, mapr, nprocessors, skip_stats)
    
    print '\n---Performing interchromosomal breakend analysis---'
    read_manta_breakend(SV_vcfs, sample_names, fusions, distT, rcT, outdir, prefix, mapr, nprocessors, skip_stats)

if __name__ == '__main__': main()


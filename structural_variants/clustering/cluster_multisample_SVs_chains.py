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

class StrucVar:
    '''
    '''
    def __init__(self,var_info):
        self.chrom1 = var_info[0] 
        self.pos1 = var_info[1]
        self.chrom2 = var_info[2]
        self.pos2 = var_info[3]
        self.svtype = var_info[4]
        self.name = var_info[5]

    def overlap(self, other, distT):
        olap = False
        s1_chr1, s1_pos1, s1_chr2, s1_pos2 = self.chrom1, self.pos1, self.chrom2, self.pos2
        s2_chr1, s2_pos1, s2_chr2, s2_pos2 = other.chrom1, other.pos1, other.chrom2, other.pos2
        
        if s1_chr1 == s2_chr1:
            if abs(s1_pos1 - s2_pos1) < distT:
                olap = True
        if s1_chr1 == s2_chr2:
            if abs(s1_pos1 - s2_pos2) < distT:
                olap = True
        if s1_chr2 == s2_chr1:
            if abs(s1_pos2 - s2_pos1) < distT:
                olap = True
        if s1_chr2 == s2_chr2:
            if abs(s1_pos2 - s2_pos2) < distT:
                olap = True
        
        return olap

def read_manta_contiguous(SV_vcfs,sample_names,fusions,rcT,mapr):
    '''
    '''
    # master output variable
    SV_list = {}

    # other info
    valid_chroms = ['chr%s'%(str(x)) for x in range(1,23)] + ['chrX']
  
    # storage dictionary for tracking reciprocal inversions
    INV_events = {}

    # 
    nsamps = len(SV_vcfs)
    sampnum = 0
    probs = {}
    print 'Reading VCFs for intrachromosomal SVs...'
    TOT_VARS=0
    for (vfn, name) in zip(SV_vcfs,sample_names):
        count = 1
        ID_track = set()
        sampnum += 1 
        SV_list[name] = []
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
            
            # check for event
            inv_event = None
            try: inv_event = v.INFO['EVENT']
            except KeyError: pass
        
            # UPDATE 10-30-2018
            if (SVTYPE == 'INV3' or SVTYPE == 'INV5') and inv_event != None:
                if inv_event not in INV_events: INV_events[inv_event] = []
                INV_events[inv_event].append(intid)
            else:
                svinfo = [CHROM,POS,CHROM,END,SVTYPE,intid]
                strucVar = StrucVar(svinfo)
                SV_list[name].append(strucVar)
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

            # add
            svinfo = [chrom,start,chrom,stop,svt,inversions[0]]
            strucVar = StrucVar(svinfo)
            SV_list[name].append(strucVar)
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
            svinfo = [ichrom, start, ichrom, stop, typ, intid]
            strucVar = StrucVar(svinfo)
            SV_list[name].append(strucVar)
            TOT_VARS += 1

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
                svinfo = [dchrom,dend,dchrom,astart,'FUS',fusid]
                strucVar = StrucVar(svinfo)
                SV_list[name].append(strucVar)
                TOT_FUS += 1

        print ' Loaded %d total intrachromosomal fusions.\n'%(TOT_FUS)
    
    # return
    return SV_list
    
def read_manta_breakend(SV_vcfs,sample_names,fusions,rcT,mapr):
    '''
    
    '''  
    # other info
    valid_chroms = ['chr%s'%(str(x)) for x in range(1,23)] + ['chrX']

    # set up breakend storage
    breakends = {}

    # Loop over samples, perform analysis
    nsamps = len(SV_vcfs)
    sampnum = 0
    print 'Reading VCFs for interchromosomal SVs...'
    for (vfn, name) in zip(SV_vcfs,sample_names):
        count = 1
        ID_track = set()
        sampnum += 1
        breakends[name] = []
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
            svinfo = [CHROM1,POS1,CHROM2,POS2,'BND',intid]
            strucVar = StrucVar(svinfo)
            breakends[name].append(strucVar)
            count += 1
            
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
                svinfo = [dchrom,dend,achrom,astart,'FUS',fusid]
                strucVar = StrucVar(svinfo)
                breakends[name].append(strucVar)
                
                TOT_FUS += 1

        print ' Loaded %d total intrachromosomal fusions.\n'%(TOT_FUS)
 
    # return
    return breakends

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
    #parser.add_argument('-p','--processors',dest='nprocessors',type=int,default=1,
    #                    help='Select number of processors.')
    parser.add_argument('-t','--threshold',dest='distT',type=float,default=10000.0,
                        help='Select distance threshold (in bp) for cluster identification.')
    parser.add_argument('--rc-thresh',dest='rcT',type=int,default=5,
                        help='Set somatic read count threshold for inclusion in analysis. This threshold is compared against the sum of somatic\
                        paired-read and split-read evidences (i.e. PR+SR) for the variant.')
    parser.add_argument('-m','--mappable-regions',type=str,default=None,dest='mappable',
                        help='Provide mappable regions a bgzipped, tabix-indexed BED file.')

    args = parser.parse_args()
    vargs = vars(args)

    # Parse required
    SV_vcfs_fn = vargs['SV_vcfs']
    if not os.path.isfile(SV_vcfs_fn): parser.error('SV vcfs list file not found!')
    SV_vcfs, sample_names = [],[]
    for line in open(SV_vcfs_fn).readlines():
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
    #nprocessors = min(mp.cpu_count(),vargs['nprocessors'])
    mappable = vargs['mappable']
    distT = vargs['distT']
    rcT = vargs['rcT']
    if mappable != None:
        if not os.path.isfile(mappable) or not os.path.isfile(mappable+'.tbi'): parser.error('Mappability file and/or its index not found!')
        mapr = TabixFile(mappable)
    else: mapr = None
    
    ## RUN
    # Read SVs
    SV_cont = read_manta_contiguous(SV_vcfs,sample_names,fusions,rcT,mapr)
    SV_BND  = read_manta_breakend(SV_vcfs,sample_names,fusions,rcT,mapr)

    # Join together
    SV_list = []
    for sample in sample_names:
        SV_list += SV_cont[sample] + SV_BND[sample]

    # Get overlaps and create graph
    G = nx.Graph()
    for i in range(0, len(SV_list)-1):
        sv1 = SV_list[i]
        for j in range(i+1, len(SV_list)):
            sv2 = SV_list[j]
            olap = sv1.overlap(sv2, distT)
            if olap:
                edge = (sv1, sv2)
                G.add_edge(*edge)

    # Initialize output list
    OLINES = []
    of = open('%sSV_clusters.txt'%(outdir),'w')
    of.writelines('\t'.join(['SAMPLE','CHROM1','POS1','CHROM2','POS2','SV_type','SV_name','cluster'])+'\n')

    # find connected components
    ConnComp = sorted(nx.connected_components(G), key = len, reverse = True)
    curr_clust = -1
    in_cluster = set()
    CLUSTERS = {}
    for i,cc in enumerate(ConnComp):
        curr_clust = i+1
        cc = list(cc)
        cc.sort(key = lambda x: (x.chrom1,x.pos1))
        SAMPS = set()
        for c in cc:
            in_cluster.add(c.name)
            samp = c.name.split('_')[0]
            OLINES.append([samp, c.chrom1, c.pos1, c.chrom2, c.pos2, c.svtype, c.name, curr_clust])
            SAMPS.add(samp)
        CLUSTERS[curr_clust] = [len(cc), len(SAMPS)]

    # get singletons
    nodes = G.nodes()
    singletons = []
    for SV in SV_list:
        svname = SV.name
        if svname not in in_cluster:
            singletons.append(SV)        
    
    curr_clust += 1
    singletons.sort(key=lambda x: (x.chrom1,x.pos1))
    for SV in singletons:
        samp = SV.name.split('_')[0]
        OLINES.append([samp, SV.chrom1, SV.pos1, SV.chrom2, SV.pos2, SV.svtype, SV.name, curr_clust])
        CLUSTERS[curr_clust] = [1, 1]
        curr_clust += 1
        
    print '%d total SVs - %d clusters with %d variants and %d singletons.'%(len(SV_list), len(ConnComp), len(in_cluster), len(singletons))

    # write clusters output
    for OL in OLINES:
        of.writelines('%s\t%s\t%d\t%s\t%d\t%s\t%s\t%d\n'%(tuple(OL)))
    of.close()

    # compute stats
    mu = sum([CLUSTERS[x][1] for x in CLUSTERS]) / len(CLUSTERS)
    print ' Mu parameter for Poissons tests: %0.3f\n'%(mu)
    COUT = []
    for c in CLUSTERS:
        K = CLUSTERS[c][1]
        pval = 1
        for k in range(0,K):
            pval -= poisson.pmf(k, mu)
        pval = max(0, pval)

        cout = [c] + CLUSTERS[c] + [pval]
        COUT.append(cout)
    COUT.sort(key = lambda x: x[-1])

    # Write stat output
    ofs = open('%sSV_cluster_stats.txt'%(outdir),'w')
    ofs.writelines('\t'.join(['CLUSTER','NVARS','NSAMPS','Poisson_pval'])+'\n')
    for cdata in COUT:
        ofs.writelines('%d\t%d\t%d\t%0.3g\n'%(tuple(cdata)))
    ofs.close()

if __name__ == '__main__': main()

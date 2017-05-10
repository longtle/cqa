import scipy.stats
import scipy
import scipy.spatial
import random
import networkx
import time
import csv

start_time=time.time()
DIR = "./data/"
for name in ['us']:
    edges = {}
    #edge file
    IN = open(DIR + name + '-friendships.csv', 'r')
    #OUT = open(DIR + name + '_vals', 'w')
    OUT = csv.writer(open(DIR + name + '-netsimile.csv', 'wb'), delimiter = ',')
    OUTID = open(DIR + name + '-ids', 'w')
    
    read_line = IN.readline()
    to_add_edges = []
    while(read_line):
        t = read_line.rstrip().split(',')
        if len(t) > 0 and t[0] != t[1]:
            if t[0] not in edges:
                edges[t[0]] = set([])
            if t[1] not in edges:
                edges[t[1]] = set([])
            if t[0] not in edges[t[1]] and t[1] not in edges[t[0]]:
                to_add_edges.append([t[0], t[1]])
                edges[t[0]].add(t[1])
                edges[t[1]].add(t[0])
        read_line = IN.readline()
    IN.close()
    print 'done getting edges'

    G = networkx.Graph()
    G.add_edges_from(to_add_edges)
    print 'done making graph'
    features = {}
    features['degs'] = {}
    features['deg_adj'] = {}
    features['ego'] = {}
    features['CC'] = {}
    features['CC_adj'] = {}
    features['ego_out'] = {}
    features['ego_adj'] = {}
    samp = set([v for v in edges])
    samp_adj = set([v for v in edges])
 
    #get node degrees
    n_count = 0
    for n in samp_adj:
        #print 'degs', n_count, len(edges)
        n_count += 1
        features['degs'][n] = G.degree(n)

    #get node neighbor degrees
    n_count = 0
    for n in samp:
        features['deg_adj'][n] = 0
        #print 'deg_adj', n_count, len(edges)
        n_count += 1
        for a in G.neighbors(n):
            features['deg_adj'][n] += features['degs'][a]
        features['deg_adj'][n] = float(features['deg_adj'][n])/float(features['degs'][n])
        
    #get size of egonet
    n_count = 0
    for n in samp:
        #print 'ego', n_count, len(samp)
        n_count += 1
        adj = edges[n].copy()
        count = 0
        for a in adj:
            for b in adj:
                if a != b:
                    if a in edges[b]:
                        count += 1
        count = count/2
        features['ego'][n] = count

    
    #get clustering coefficient
    n_count = 0
    for n in samp_adj:

        CC = networkx.clustering(G, [n])
        features['CC'][n] = CC[n]
    #get CC of neighbors
    n_count = 0
    for n in samp:
        #print 'avg CC', n_count, len(edges)
        n_count += 1
        avg_CC = 0
        adj = edges[n].copy()
        for a in adj:
            avg_CC += features['CC'][a]
        avg_CC = float(avg_CC)/float(features['degs'][n])
        features['CC_adj'][n] = avg_CC
    #get edges out of egonet
    n_count = 0
    for n in samp:
        #print 'ego_out', n_count, len(edges)
        n_count += 1
        adj = G.neighbors(n)
        out_edges = 0
        for a in adj:
            for b in G.neighbors(a):
                if b not in adj:
                    out_edges += 1
        features['ego_out'][n] = out_edges
    
    #get number of egonet neighbors
    n_count = 0
    for n in samp:
        #print 'ego_adj', n_count, len(edges)
        n_count += 1
        adj = edges[n].copy()
        out_nbrs = set([])
        for a in adj:
            out_nbrs = out_nbrs.union(edges[a])
        out_nbrs = out_nbrs.difference(adj)
        features['ego_adj'][n] = len(out_nbrs)
    OUT.writerow(['user_id', 'degs', 'deg_adj', 'ego', 'CC','CC_adj', 'ego_out', 'ego_adj'])
    for n in samp:
        OUTID.write(n + '\n')
        print 'n: ', n, n[1:-1]
        #OUT.write(str(features['degs'][n]) + '\t' + str(features['deg_adj'][n]) + '\t' + str(features['ego'][n]) + '\t' + str(features['CC'][n]) + '\t' + str(features['CC_adj'][n]) + '\t' + str(features['ego_out'][n]) + '\t' + str(features['ego_adj'][n]) + '\n')
        OUT.writerow([str(n)[1:-1], str(features['degs'][n]), str(features['deg_adj'][n]), str(features['ego'][n]),str(features['CC'][n]),str(features['CC_adj'][n]),str(features['ego_out'][n]),str(features['ego_adj'][n])])
    #OUT.close()
    OUTID.close()
    
    del G
    print name
    
print time.time() - start_time, "seconds"
    
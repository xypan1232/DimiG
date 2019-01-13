import sys
import gzip
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, grid_search
import numpy as np
import pdb
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import normalize
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 15.0

import gzip
from sklearn import metrics
from scipy import stats
import random
import cPickle
import argparse

from math import*
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import KMeans
from collections import Counter
import scipy.spatial.distance as ssd
import scipy.sparse as sp
import venn
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from pygcn.utils import accuracy
from pygcn.models import GCN

def point_overlap(min1, max1, min2, max2):
    return max(0, min(max1, max2) - max(min1, min2))

def euclidean_distance(x,y):
    return sqrt(sum(pow(a-b,2) for a, b in zip(x, y)))

def calculate_performace(test_num, pred_y,  labels):
    tp =0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] ==1:
            if labels[index] == pred_y[index]:
                tp = tp +1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn +1
            else:
                fp = fp + 1               
            
    acc = float(tp + tn)/test_num
    precision = float(tp)/(tp+ fp)
    sensitivity = float(tp)/ (tp+fn)
    specificity = float(tn)/(tn + fp)
    MCC = float(tp*tn-fp*fn)/(np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
    return acc, precision, sensitivity, specificity, MCC 

def get_normalized_values_by_column(array, fea_length):
    max_col =[-100000] * fea_length
    min_col = [100000] * fea_length
    #for key in array.keys():
    #    indvidual_fea =  array[key]
    for values in array:
        for index in range(len(values)):
            if values[index] > max_col[index]:
                max_col[index] = values[index]
            if values[index] < min_col[index]:
                min_col[index] = values[index]
    for values in array:
        for index in range(len(values)):
            #print values[index],min_col[index], max_col[index]   
            values[index] = float(values[index] - min_col[index])/(max_col[index] - min_col[index]) 
    fw = open('saved_min_max', 'w')
    for val in min_col:
        fw.write('%f\t' %val)
    fw.write('\n')
    for val in max_col:
        fw.write('%f\t' %val)
    fw.write('\n')
    fw.close() 
    
def get_normalized_given_max_min(array):
    normalized_data = np.zeros(array.shape)
    tmp_data = np.loadtxt('saved_min_max')
    min_col = tmp_data[0, :]
    max_col = tmp_data[1, :]
    for x in xrange(array.shape[0]):
        for y in xrange(array.shape[1]):
            #print values[index],min_col[index], max_col[index]   
            normalized_data[x][y] = float(array[x][y] - min_col[y])/(max_col[y] - min_col[y])
    return normalized_data

def transfer_probability_class(result):
    y_pred = []
    for val in result:
        if val >= 0.5:
            y_pred.append(1)
        else:
            y_pred.append(0)
    return y_pred

    
def predict_new_data(test_data, save_model_file, SVM = False):
    get_normalized_given_max_min(test_data)
    with open(save_model_file, 'rb') as f:
        clf = cPickle.load(f)  
    preds = clf.predict_proba(test_data)
    return preds[:, 1]


def element_count(a):
    results = {}
    for x in a:
        if x not in results:
            results[x] = 1
        else:
            results[x] += 1
    return results



def read_snp_dataset(snp_file = 'SNP/snp142Common.txt.gz'):
    snp_dict = {}
    with gzip.open(snp_file, 'r') as fp:
        for line in fp:
            values = line.rstrip('\r\n').split('\t')
            key = values[1] + values[6]
            snp_dict.setdefault(key, set()).add(int(values[2]))
            
    return snp_dict

def read_snp_coordinate(snp_file = 'SNP/snp142Common.txt.gz'):
    snp_dict = {}
    with gzip.open(snp_file, 'r') as fp:
        for line in fp:
            values = line.rstrip('\r\n').split('\t')
            #key = values[1] + values[6]
            snp_dict[values[4]] = values[1] + '_' + values[2] + '_' + values[6]
            
    return snp_dict

def read_GWAS_catalog(gwas_file = 'SNP/gwascatalog.txt', down_up_stream = False, cutoff=5000):
    gwas_snp = {}
    with open(gwas_file, 'r') as fp:
        head = True
        for line in fp:
            if head:
                head = False
                continue
            values = line.rstrip('\r\n').split('\t')
            chr_name = 'chr' + values[11]
            snp_id = values[21]
            coor = int(values[12])
            if down_up_stream: # upstream and downstream 500kb
                new_start = coor - cutoff
                new_end = coor + cutoff
                for val in range(new_start, new_end):
                    gwas_snp.setdefault(chr_name, set()).add(val)
            else:
                gwas_snp.setdefault(chr_name, set()).add(coor)
    
    return gwas_snp

def read_GWAS_catalog_disease(gwas_file = 'SNP/gwas_catalog_ensembl_mapping_v1.0-downloaded_2015-10-09.tsv'):
    gwas_dis = {}
    with open(gwas_file, 'r') as fp:
        head = True
        for line in fp:
            if head:
                head = False
                continue

            values = line.rstrip('\r\n').split('\t') 
            if values[7] == '' or values[-13] == '':
                continue
            disease = values[7]

            gwas_dis.setdefault(disease.upper(), set()).add(values[-13])
            #gwas_snp_coor[snp_id] = chr_name + '_' + values[12]
    #pdb.set_trace()
    return gwas_dis
            
def read_gwas_ld_region(gwas_ld_file = 'SNP/GWAS-LD-region-snps.csv'):
    #gwas_dis ={}
    snp_dis = {}
    with open(gwas_ld_file, 'r') as fp:
        head = True
        for line in fp:
            if head:
                head = False
                continue
            try:
                if '"' in line:
                    values = line.rstrip('\r\n').split('"')
                    SNP,GWAS_SNP,PMID = values[0][:-1].split(',')
                    diseases = values[1].split(',')
                    for dis in diseases:
                        snp_dis.setdefault(dis.upper(), []).append(SNP)
                        #gwas_dis.setdefault(dis.upper(), set()).add(GWAS_SNP)                        
                else:
                    SNP,GWAS_SNP,PMID,disease = line.rstrip('\r\n').split(',')
                    snp_dis.setdefault(disease.upper(), []).append(SNP)
                    #gwas_dis.setdefault(disease.upper(), set()).add(GWAS_SNP)
            except:
                pdb.set_trace()
            
    return snp_dis

def get_all_snp_disease_assoc():
    all_disease_snp = {}
    snp_coor_dict = read_snp_coordinate()
    ld_snp_dis = read_gwas_ld_region()
    gwas_dis = read_GWAS_catalog_disease()
    #pdb.set_trace()
    for key, val in gwas_dis.iteritems():
        for snp in val:
            if snp_coor_dict.has_key(snp):
                all_disease_snp.setdefault(key, set()).add(snp_coor_dict[snp])

    for key, val in ld_snp_dis.iteritems():
        for snp in val:
            if snp_coor_dict.has_key(snp):
                all_disease_snp.setdefault(key, set()).add(snp_coor_dict[snp])
                
    return all_disease_snp

def get_kmeans_k_biggest_cluster(cluster_centers, labels, num_cluster):
    freq_dict = element_count(labels)
    freq_list  = []
    for key, val in freq_dict.iteritems():
        freq_list.append((val, key))
        
    freq_list.sort(reverse= True)
    new_array = []
    for val in freq_list[:num_cluster]:
        new_array.append(cluster_centers[val[1]])
    
    return np.array(new_array)


def preprocess_data(X, scaler=None, minmax = True):
    if not scaler:
        if minmax:
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler                 

def preprocess_data_tissue(X):
    new_col = np.sum(X,1).reshape((X.shape[0],1))    
    X_new = X/X.sum(axis=1)[:, None] 
    X_new[np.isnan(X_new)] = 0
    X_new = np.append(X_new,new_col, axis=1)
    return X_new


def plot_gaussian_distribution(h1, h2):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    h1.sort()
    hmean1 = np.mean(h1)
    hstd1 = np.std(h1)
    pdf1 = stats.norm.pdf(h1, hmean1, hstd1)
    ax.plot(h1, pdf1, label="lncRNA")

    h2.sort()
    hmean2 = np.mean(h2)
    hstd2 = np.std(h2)
    pdf2 = stats.norm.pdf(h2, hmean2, hstd2)
    ax.plot(h2, pdf2, label="PCG")
    legend = ax.legend(loc='upper right')
    
    plt.xlabel('FPKM')
    #plt.xlim(0,0.2)

    plt.show()

def calculate_pcc_old(data, data_source):
    print 'calculating PCC distance'
    rows, cols = data.shape
    #data_pcc = np.zeros(rows, cols)
    pcc_list  = []
    #data = normalize(data, axis=0)
    scaler = StandardScaler()
        #scaler = MinMaxScaler()
    scaler.fit(data)
    data = scaler.transform(data)
    print data.shape
    for i in xrange(rows): # rows are the number of rows in the matrix. 
        pcc_list = pcc_list + [stats.pearsonr(data[i],data[j])[0] for j in range(i) if j != i]

    #pdb.set_trace()
    print len(pcc_list)
    plot_hist_distance(pcc_list, 'PCC', data_source)        
    return pcc_list

def calculate_pcc_hist(mRNA_data, lncRNA_data):
    print 'calculating PCC distance'
    corr_pval = []
    corr_ind = []
    #pdb.set_trace()
    for i, mval in enumerate(lncRNA_data):
        tmp = []
        ind_j = []
        for j, lncval in enumerate(mRNA_data):
            rval, pval = stats.pearsonr(mval, lncval)
            abs_rval = np.absolute(rval)
            if  abs_rval > 0.3 and pval <= 0.01:
                corr_pval.append(abs_rval)
            
        #corr_pval.append(tmp)
    #print len(corr_pval)
    #plot_hist_distance(corr_pval, 'PCC', 'gencode')        #corr_ind.append(ind_j)
            
    return corr_pval

def calculate_cc(X, Y):
    if X is Y:
        X = Y = np.asanyarray(X)
    else:
        X = np.asanyarray(X)
        Y = np.asanyarray(Y)

    if X.shape[1] != Y.shape[1]:
        raise ValueError("Incompatible dimension for X and Y matrices")

    XY = ssd.cdist(X, Y, 'correlation')

    return 1 - XY

def calculate_pcc_fast(A, B):
    #pdb.set_trace()
    A = A.T
    B = B.T
    N = B.shape[0]
    
    # Store columnw-wise in A and B, as they would be used at few places
    sA = A.sum(0)
    sB = B.sum(0)
    
    # Basically there are four parts in the formula. We would compute them one-by-one
    #p1 = N*np.einsum('ij,ik->kj',A,B)
    p1 = N*np.dot(B.T,A)
    p2 = sA*sB[:,None]
    p3 = N*((B**2).sum(0)) - (sB**2)
    p4 = N*((A**2).sum(0)) - (sA**2)
    
    # Finally compute Pearson Correlation Coefficient as 2D array 
    pcorr = ((p1 - p2)/np.sqrt(p4*p3[:,None]))
    
    return pcorr


def calculate_pcc(mRNA_data, lncRNA_data):
    print 'calculating PCC distance'
    corr_pval = []
    corr_ind = []
    #pdb.set_trace()
    for i, mval in enumerate(lncRNA_data):
        tmp = []
        ind_j = []
        for j, lncval in enumerate(mRNA_data):
            rval, pval = stats.pearsonr(mval, lncval)
            abs_rval = np.absolute(rval)
            if  abs_rval > 0.3 and pval <= 0.01:
                tmp.append(abs_rval)
                ind_j.append(j)
            
        corr_pval.append(tmp)
        corr_ind.append(ind_j)
            
    return corr_pval, corr_ind

def coexpression_hist_fig(disease_mRNA_data, mRNAlabels, disease_lncRNA_data, lncRNA_list, mRNA_list, fw):
    posi_itemindex = np.where(mRNAlabels==1)[0]
    inner_data = disease_mRNA_data[posi_itemindex, :]  
    corr_pval = calculate_pcc_hist(inner_data, disease_lncRNA_data)
    return corr_pval

def coexpression_based_prediction(disease_mRNA_data, mRNAlabels, disease_lncRNA_data, lncRNA_list, mRNA_list, fw, k = 1):
    print 'k:', k
    posi_itemindex = np.where(mRNAlabels==1)[0]
    inner_data = disease_mRNA_data[posi_itemindex, :]  
    corr_pval, corr_ind = calculate_pcc(inner_data, disease_lncRNA_data)
    y_ensem_pred = []
    
    for val in corr_pval:
        if not len(val):
            y_ensem_pred.append(0)
        else:
            val.sort(reverse = True)
            sel_vals = val[:k]
            bigval = np.mean(sel_vals)
            y_ensem_pred.append(bigval)
    #pdb.set_trace()        
    fw.write('\t'.join(map(str, y_ensem_pred)))
    fw.write('\n')
    
    
def coexpression_knn_based_prediction(disease_mRNA_data, mRNAlabels, disease_lncRNA_data, lncRNA_list, mRNA_list, fw, k = 15):
    print 'k:', k 
    corr_pval= calculate_pcc_fast( disease_mRNA_data, disease_lncRNA_data)
    y_ensem_pred = []
    posi_itemindex = np.where(mRNAlabels==1)[0]
    num_len = len(corr_pval[0])
    #pdb.set_trace()
    for ind in range(len(corr_pval)):
        score = [abs(val) for val in corr_pval[ind]]
        num_inds = np.argsort(score)
        #val.sort(reverse = True)
        sel_vals = num_inds[num_len - k:]
        bigval = set(sel_vals) & set(posi_itemindex)
        knn_prob = len(bigval)
        y_ensem_pred.append(knn_prob)
    #pdb.set_trace()        
    fw.write('\t'.join(map(str, y_ensem_pred)))
    fw.write('\n')

def read_GPL_file(GPL_file):
    gene_map_dict = {}
    fp = open(GPL_file, 'r')
    for line in fp:
        if line[0] == '#' or line[0] == 'I':
            continue
        values = line.rstrip('\n').split('\t')
        refID = values[0]
        probID = values[3]
        gene_symbol = values[9]
        ensembleID = values[12]
        gene_map_dict[refID] = (probID, gene_symbol, ensembleID)
    fp.close()
    return gene_map_dict

def read_normalized_series_file(series_file, take_median=True):
    gene_map_dict = read_GPL_file('data/GSE34894/GPL15094-7646.txt')
    fp = gzip.open(series_file, 'r')
    expression_dict = {}
    #fw = open('gene_expression_file', 'w')
    for line in fp:
        if line[0] == '!' or len(line) < 10:
            continue
        #pdb.set_trace()
        if 'ID_REF' in line:
            sampel_ids = line.rstrip('\r\n').split('\t')[1:]
            sampel_ids = ['probID', 'gene_symbol', 'ensembleID'] + sampel_ids[1:]
            #fw.write('\t'.join(sampel_ids))
            continue
        values = line.rstrip('\r\n').split('\t')
        refID = values[0]
        probID, gene_symbol, ensembleID = gene_map_dict[refID]
        if ensembleID == '':
            continue
        expression_dict.setdefault(ensembleID, []).append([probID] + values[1:] + [gene_symbol])
        #fw.write('\t'.join(values))
    fp.close() 

    merge_probe_expression_dict = {}
    #num_of_tissue = 31
    for key,vals in expression_dict.iteritems():
            new_vals = []
            num_dup = len(vals)
            for single_val in vals:
                exp_vals = []
                exp_vals = [float(val) for val in single_val[1:-1]]
                #for index in range(num_of_tissue):
                new_vals.append(exp_vals)
                #new_vals = [x.append(float(y)) for x,y in zip(new_vals, exp_vals)]
                prob = single_val[0]
                gene_symbol = single_val[-1]
            new_vals = np.array(new_vals)
            #pdb.set_trace()
            final_express_vals = []
            if take_median:
                final_express_vals = np.median(new_vals, axis=0)
            else:
                final_express_vals = np.mean(new_vals, axis=0) 
            try:    
                merge_probe_expression_dict[key] = [prob, gene_symbol] + [inside_val for inside_val in final_express_vals]
            except:
                pdb.set_trace()
                print final_express_vals
                print prob, gene_symbol
    return merge_probe_expression_dict

def get_mean_expression_for_tissue_multiple_sampels(samples, expression_dict, use_mean = False, log2 = True):
    sample_set = set(samples)
    sample_list = [samp for samp in sample_set]
    aver_expr_vals = {}
    for key, vallist in expression_dict.iteritems():
        for sam in sample_set:
            ave_inds =  [i for i,val in enumerate(samples) if val==sam]
            if use_mean:
                mean_val = np.mean(map(vallist.__getitem__, ave_inds)) 
            else:
                mean_val = np.median(map(vallist.__getitem__, ave_inds)) 
            if log2:
                mean_val = np.log2(1 + mean_val) 
                
            aver_expr_vals.setdefault(key, []).append(mean_val) 
    expression_dict.clear()
    new_aver_expr_vals = {}
    for key, val in aver_expr_vals.iteritems():
        #if max(val) < CUTOFF:
        #    continue
        new_aver_expr_vals[key] = val
    
    return new_aver_expr_vals, sample_list  

def read_human_RNAseq_expression(RNAseq_file = 'data/gencodev7/genes.fpkm_table', gene_name_ensg = None, log2 = True):
    print 'read expresion file: ', RNAseq_file
    data_dict  = {}
    fp = open(RNAseq_file, 'r')
    head = True
    for line in fp:
        if head:
            head = False
            values = line.rstrip('\r\n').split('\t')[1:]
            samples = [val.split('_')[0] for val in values]
            continue
        else:
            values = line.rstrip('\r\n').split('\t')
            if log2:
                expval_list = [np.log2(1 + float(tmp_val)) for tmp_val in values[1:]]
            else:
                expval_list = [float(tmp_val) for tmp_val in values[1:]]
                
            #if max(expval_list) < CUTOFF:
            #    continue
                
            key = values[0].split('.')[0]
            #if gene_name_ensg.has_key(key):
            #    data_dict[gene_name_ensg[key]] = expval_list
            #else:
            data_dict[key] = expval_list
    fp.close()
    
    return data_dict, samples

def read_gtex_gene_map(map_file = 'data/gtex/xrefs-human.tsv'):
    entrid_ensemid = {}
    head = True
    with open(map_file, 'r') as fp:
        for line in fp:
            if head:
                head = False
                continue
            if 'Ensembl' in line:
                values = line.rstrip().split()
                entrid_ensemid[values[0]] = values[-1]
    return entrid_ensemid       
    

def read_gtex_expression(RNAseq_file = 'data/gtex/GTEx_Analysis_v6p_RNA-seq_RNA-SeQCv1.1.8_gene_median_rpkm.gct.gz', gene_type_dict = None, log2 = True):
    print 'read expresion file: ', RNAseq_file
    #entrid_ensemid = read_gtex_gene_map()
    data_dict  = {}
    fp = gzip.open(RNAseq_file, 'r')
    head = True
    #pdb.set_trace()
    for line in fp:
        if line[0] == '#':
            continue
        if '56238\t53' in line:
            continue
        if head:
            head = False
            values = line.rstrip('\r\n').split('\t')[2:]
            samples = [val for val in values]
            print '# of tissues', len(samples)
            continue
        else:
            values = line.rstrip('\r\n').split('\t')
            if log2:
                expval_list = [np.log2(1 + float(tmp_val)) for tmp_val in values[2:]]
            else:
                expval_list = [float(tmp_val) for tmp_val in values[2:]]
            #expval_list = [float(tmp_val) for tmp_val in values[2:]]
                
            #if max(expval_list) < CUTOFF:
            #    continue
                
            key = values[0].split('.')[0]
            #if entrid_ensemid.has_key(key):
            #    ensem_id = entrid_ensemid[key]
            #if key in gene_type_dict:
            data_dict[key] = expval_list
            #else:
            #    data_dict[key] = expval_list
    fp.close()
    #pdb.set_trace()
    return data_dict, samples

def read_evolutionary_expression_data(input_file = 'data/GSE43520/genes.fpkm_table', use_mean = False, log2 = True):
    print 'read expresion file: ', input_file
    data_dict  = {}
    fp = open(input_file, 'r')
    head = True
    for line in fp:
        if head:
            head = False
            values = line.rstrip('\r\n').split('\t')[1:]
            samples = [val.split('-')[0].lower() for val in values]
            continue
        else:
            values = line.rstrip('\r\n').split('\t')
            key = values[0].split('.')[0]
            data_dict[key] = [float(tmp_val) for tmp_val in values[1:]]            
    fp.close()
    #pdb.set_trace()
    data_dict, sample_set = get_mean_expression_for_tissue_multiple_sampels(samples, data_dict, use_mean = use_mean, log2 = log2)

    return data_dict, sample_set

def read_evolutionary_expression_data_old(input_file, use_mean = False, log2 = True):
    print 'read expresion file: ', input_file
    data_dict  = {}
    fp = open(input_file, 'r')
    head = True
    for line in fp:
        if head:
            head = False
            values = line.rstrip('\r\n').split('\t')[1:]
            samples = [val.split('_')[0] for val in values]
            continue
        else:
            values = line.rstrip('\r\n').split('\t')
            key = values[0]
            if 'blastn' in key:
                continue
            data_dict[key] = [float(tmp_val) for tmp_val in values[1:]]            
    fp.close()
    #pdb.set_trace()
    data_dict, sample_set = get_mean_expression_for_tissue_multiple_sampels(samples, data_dict, use_mean = use_mean, log2 = log2)
    
    return data_dict, sample_set

def read_average_read_to_normalized_RPKM(input_file = 'data/GSE30352/genes.fpkm_table', readlength=75, use_mean = False, log2 = True):
    print 'read expresion file: ', input_file
    fp =open(input_file, 'r')
    head = True
    express_vals = {}
    #gene_list = []
    for line in fp:
        if head:
            values = line.rstrip('\r\n').split('\t')[1:]
            samples = [val.split('-')[0] for val in values]
            head =False
            continue
        values = line.rstrip('\r\n').split('\t')
        gene = values[0].split('.')[0]
        #gene_list.append(gene)
        express_vals[gene] = [float(val) for val in values[1:]]
    #pdb.set_trace()    
    final_vals, sample_set =  get_mean_expression_for_tissue_multiple_sampels(samples, express_vals, use_mean = use_mean, log2 = log2) 
       
    return final_vals, sample_set   
     
def read_average_read_to_normalized_RPKM_old(input_file, readlength=75, use_mean = False, log2 = True):
    print 'read expresion file: ', input_file
    fp =open(input_file, 'r')
    head = True
    total_exp_vals = []
    gene_list = []
    exon_len = []
    for line in fp:
        if head:
            values = line.rstrip('\r\n').split('\t')[6:]
            samples = [val[1:-1].split('_')[1] for val in values]
            head =False
            continue
        values = line.rstrip('\r\n').split('\t')
        gene = values[0][1:-1]
        gene_list.append(gene)
        gene_start = int(values[2])
        gene_end = int(values[3])
        ExonicLength = int(values[5])
        exon_len.append(ExonicLength)
        express_vals = [float(val)*ExonicLength/readlength for val in values[6:]]
        total_exp_vals.append(express_vals)
    
    final_vals = {}    
    total_exp_vals = np.array(total_exp_vals)
    sum_val = sum(total_exp_vals)/1000000000
    for x in xrange(total_exp_vals.shape[0]):
        final_vals[gene_list[x]] = []
        for y in xrange(total_exp_vals.shape[1]): 
            final_vals[gene_list[x]].append(total_exp_vals[x][y]/(exon_len[y]*sum_val[y]))
    fp.close()
    #pdb.set_trace()
    #average value for the same tissue from different samples
    final_vals, sample_set =  get_mean_expression_for_tissue_multiple_sampels(samples, final_vals, use_mean = use_mean, log2 = log2)
    
    return final_vals, sample_set


def remove_redudancy_expression_data(expression_data, cutoff):
    redudancy_data = []
    #new_list=[]
    for i in range(len(expression_data)):
        keep_flag = True
        for j in range(i):
            if euclidean_distance(expression_data[i, :],expression_data[j, :]) <= cutoff:
                keep_flag = False
                break
        if keep_flag:        
            redudancy_data.append(expression_data[i, :])
    
    return redudancy_data


def extract_gene_type_name(input_val):
    input_val = input_val.strip()
    split_gene_name = input_val.split()
    gene = split_gene_name[1][1:-1]
    return gene

def read_gencode_gene_type():
    gene_type_dict = {}
    gene_name_ensg = {}
    gene_id_position = {}
    fp = gzip.open('data/gencode.v19.genes.v6p_model.patched_contigs.gtf.gz')
    for line in fp:
        if line[0] == '#':
            continue
        values = line.rstrip('\r\n').split('\t')
        if values[2] != 'gene':
            continue
        gene_ann = values[-1]
        split_gene_ann = gene_ann.split(';')
        gene_type_info = split_gene_ann[2]
        gene_type = extract_gene_type_name(gene_type_info)
            
        gene_name_info = split_gene_ann[4]
        gene_name = extract_gene_type_name(gene_name_info)
        '''gene_type_dict[gene_name] = gene_type'''
        
        gene_id_info = split_gene_ann[0]
        gene_id = extract_gene_type_name(gene_id_info)
        gene_id =  gene_id.split('.')[0]
        gene_type_dict[gene_id] = gene_type        
        
        gene_name_ensg[gene_name] = gene_id
        
        chr_name = 'chr' + values[0]
        strand = values[6]
        start = int(values[3])
        end = int(values[4])
        
        #if not gene_id_position.has_key(gene_id):
        #    gene_id_position[gene_id] = (chr_name, strand, start, end)
        # gene has bigger length compared to exon    
        
        gene_id_position[gene_id] = (chr_name, strand, start, end)
        
    fp.close()
    
    return gene_type_dict, gene_name_ensg, gene_id_position
    
def read_GRCh37_gene_type():
    gene_type_dict = {}
    gene_name_ensg = {}
    gene_id_position = {}
    fp = gzip.open('data/Homo_sapiens.GRCh37.70.gtf.gz')
    for line in fp:
        if line[0] == '#':
            continue
        values = line.rstrip('\r\n').split('\t')
        gene_ann = values[-1]
        split_gene_ann = gene_ann.split(';')
        gene_type_info = split_gene_ann[4]
        gene_type = extract_gene_type_name(gene_type_info)
            
        gene_name_info = split_gene_ann[3]
        gene_name = extract_gene_type_name(gene_name_info)
        '''gene_type_dict[gene_name] = gene_type'''
        
        gene_id_info = split_gene_ann[0]
        gene_id = extract_gene_type_name(gene_id_info)
        gene_type_dict[gene_id] = gene_type        
        
        gene_name_ensg[gene_name] = gene_id
        
        chr_name = values[0]
        strand = values[6]
        start = int(values[3])
        end = int(values[4])
        
        if not gene_id_position.has_key(gene_id):
            gene_id_position[gene_id] = (chr_name, strand, start, end)
        # gene has bigger length compared to exon    
        if values[2] == 'gene':
            gene_id_position[gene_id] = (chr_name, strand, start, end)
        
    fp.close()
    
    return gene_type_dict, gene_name_ensg, gene_id_position

def get_ENSP_ENSG_map():
    ensg_to_ensp_map = {}
    fp = open('data/string_9606___10_all_T.tsv')
    for line in fp:
        species, ensg, ensp = line.rstrip('\n').split('\t')
        ensg_to_ensp_map[ensg] = ensp
    fp.close()
    
    return ensg_to_ensp_map

def get_ensg_ensp_map():
    ensp_to_ensg_map = {}
    fp = open('data/string_9606_ENSG_ENSP_10_all_T.tsv')
    for line in fp:
        species, ensg, ensp = line.rstrip('\n').split('\t')
        ensp_to_ensg_map[ensp] = ensg
    fp.close()
    
    return ensp_to_ensg_map    



def read_DISEASE_database(include_textming = False, confidence=2, ensg_ensp_map = None):
    #ensp_to_ensg_map = get_ENSP_ENSG_map()
    print confidence
    disease_gene_dict = {}
    whole_disease_gene_dict = {}
    disease_name_map = {}

    fp = open('data/human_disease_integrated_full.tsv', 'r')
    for line in fp:
        if 'DOID:' not in line or 'ENSP' not in line:
            continue
        values = line.rstrip('\r\n').split('\t')
        gene = values[0]
        if gene not in ensg_ensp_map:
            continue

        gene = ensg_ensp_map[gene]
        disease = values[2]
        disease_name = values[3]
        disease_name_map[disease] = disease_name.upper()
        whole_disease_gene_dict.setdefault(disease, set()).add(gene)
        conf = float(values[-1])
        if conf < confidence:
            continue
        disease_gene_dict.setdefault(disease, set()).add(gene)
                 
    fp.close()
                 
    return disease_gene_dict, disease_name_map, whole_disease_gene_dict


def read_result_mrna(result_file):
    fp = open(result_file, 'r')
    result = []
    label = []
    probability = []
    result_dis_acc = {}
    importance = []
    disease_list = []
    tmp_resu = []
    disease=''
    for line in fp:
        values = line.rstrip('\r\n').split('\t')
        if 'ROC_label' in line:
            label = [int(val) for val in values[1:]]
        elif 'ROC_probability' in line:
            probability = [float(val) for val in values[1:]] 
            
            fpr, tpr, thresholds = roc_curve(label, probability) #probas_[:, 1])
            roc_auc = auc(fpr, tpr) 
            tmp_resu = tmp_resu + [roc_auc]
            
            precision, recall, thresholds = metrics.precision_recall_curve(label, probability)
            auprc = auc(recall, precision)
            tmp_resu = tmp_resu + [auprc]
            result.append(tmp_resu)
        else:
            tmp_resu = []
            disease = values[0]        
            tmp_resu = [float(val) for val in values[1:]]    
            #result.append(tmp_resu)
            result_dis_acc[disease] = tmp_resu[0]
            importance.append(tmp_resu[0])
            disease_list.append(disease)
    fp.close()
    return np.array(result)


def plot_accuracy_boxplot(accuracy, disease_list, datafile):

    ind = np.arange(len(accuracy))
    fig, ax = plt.subplots(figsize=(15,10))
    rects1 = plt.bar(ind, accuracy, 0.25, color='b')
    #plt.ylabel('Importance score')
    ax.set_xticks(ind)
    ax.set_xticklabels(disease_list, rotation=90, fontsize=10, style='normal', family='serif')
    plt.ylabel('AUC', fontsize=10)
    plt.tight_layout()
    #plt.xlim([0,5])
    #plt.xlabel('Tissue')
    #plt.title('AUC')
    plt.savefig('accuracy_fig/' + datafile.split('/')[-1] + 'boxplot.eps', format="eps")
    plt.clf()
    plt.close()


def plot_bar_imp(imp_list, disease, imp_file, tissues, ylabel = 'Importance score'):
    ind = np.arange(len(imp_list))
    '''df = pd.DataFrame(imp_list, columns=['tissue', 'score'])
    #df['accuracy'] = df['accuracy'] / df['accuracy'].sum()
    
    plt.figure()
    df.plot()
    df.plot(kind='barh', x='tissue', y='score', legend=False)
    '''
    fig, ax = plt.subplots(figsize=(12,8))
    rects1 = plt.bar(ind, imp_list, 0.25, color='b')
    #plt.ylabel('Importance score')
    ax.set_xticks(ind)
    ax.set_xticklabels(tissues, rotation=90, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.title(disease, fontsize=15)
    plt.tight_layout()
    #plt.xlim([0,5])
    #plt.xlabel('Tissue')
    #if disease == 'NON-SMALL CELL LUNG CARCINOMA' or disease == 'KIDNEY DISEASE':
    #    plt.show()
    plt.savefig('imp_fig/' + disease + imp_file.split('/')[-1] + '.eps', format='eps')
    plt.clf()
    plt.close()
    
def plot_tissue_importance(result_imp_file):
    print 'ploting tissue iportance'
    fp = open(result_imp_file, 'r')
    disease=''
    index = 0
    for line in fp:
        values = line.rstrip('\r\n').split('\t')
        if index == 0:
            tissues = values[1:] 
            index = index  + 1
        else:
            disease = values[0]        
            #imp_list = [(val1, float(val)) for val1, val in zip(tissues,values[1:])]
            imp_list = [float(val) for val in values[1:-1]]
            plot_bar_imp(imp_list, disease, result_imp_file, tissues)
    fp.close()
    #print 'average performance for all diseases:'
    #print np.mean(result, axis=0)
    #return result_dis_acc    
def get_disease_associated_matrix(disease_tissue_file = 'data/DiseaseTissueAssociationMatrixLageEtAl2008PNAS.tbl.txt'):
    disease_tissue_dct  = {}
    with open(disease_tissue_file, 'r') as fp:
        index = 0
        for line in fp:
            values = line.rstrip('\r\n').split('\t')
            if index == 0:
                tissues = values[1:]
                index = index + 1
            else:
                disease  = values[0]
                scores = [float(val) for val in values[1:]]
                disease_tissue_dct[disease] = scores
    
    return disease_tissue_dct, tissues

def plot_disease_tissue_score(disease, disease_tissue_dct, tissues, disease_name):
    #pdb.set_trace()
    score_list = disease_tissue_dct[disease] 
    plot_bar_imp(score_list, disease_name, disease, tissues, ylabel = 'Association score')


def calculate_vector_length(data, data_source, norm = 2):
    distance_list = []
    for val in data:
        distance = np.linalg.norm(val, ord=norm)
        distance_list.append(distance)
    
    plot_hist_distance(distance_list, 'vector_length', data_source, norm = norm)     
    #return distance_list

def plot_hist_distance(data_list, distance_type, data_source, genetype = 'mRNA', norm = 2):
    print 'plot histogram fig'
    fig, ax = plt.subplots()
    #ax = fig.add_subplot(111)
    #weights = np.ones_like(data_list)/len(data_list)
    plt.hist(data_list, 50000)
    ax.set_xlabel(distance_type)
    #ax.set_xlim([0, 2000])
    ax.set_xlim([0, 1])
    plt.show()
    #plt.savefig('distance_fig/' + data_source + '_' + distance_type + genetype + '.eps', type="eps")


def calculate_euclidean_distance(data, data_source):
    print 'calculating Euclidean distance'
    rows, cols = data.shape
    print rows, cols
    #data_pcc = np.zeros(rows, cols)
    euclidean_list  = []
    data = normalize(data, axis=0)
    '''for i in xrange(rows): # rows are the number of rows in the matrix. 
        for j in xrange(i, rows):
            if i == j:
                continue
            rval = euclidean_distance(datam[i,:], datam[j,:])[0]
            abs_rval = np.absolute(rval)
            euclidean_list.append(abs_rval)
    '''
    S = pairwise_distances(data, metric="euclidean")
    S = np.hstack(S)
    #pdb.set_trace()        
    plot_hist_distance(S, 'Euclidean', data_source)        
    return euclidean_list


def get_expression_based_gene(data, gene_list):
    extracted_data = []
    for gene in gene_list:
        extracted_data.append(data[gene])
        
    return np.array(extracted_data)



def map_to_doid_disease():
    disease_doid = read_DOID_BTO_map()
    disease_set = set()
    lncRNAdisease_list = []
    fw = open('data/disease/data_disease_doid.txt', 'w')
    for line in open('data/disease/data_disease_new.txt'):
        values = line.rstrip('\r\n').split('\t')
        gene = values[1]
        disease = values[2].upper()
        if disease in disease_doid:
            new_line = line.replace(disease, disease_doid[disease])
            fw.write(new_line)
        else:
            fw.write(line)
    fw.close()
        #lncRNAdisease_dict.setdefault(c, set()).add(gene)
        #lncRNAdisease_list.append((disease, gene))
        #disease_set.add(disease)
        
def read_disease_files(file_name):
    disease_genes = {}
    for line in open(file_name, 'r'):
        values = line.rstrip('\r\n').split('\t')
        gene = values[1]
        disease = values[2]
        disease_genes.setdefault(disease, set([])).add(gene)
    return disease_genes

    
def plot_roc_curve_miRNA(labels, probality, legend_text):
    #fpr2, tpr2, thresholds = roc_curve(labels, pred_y)
    fpr, tpr, thresholds = roc_curve(labels, probality) #probas_[:, 1])
    roc_auc = auc(fpr, tpr)

    rects1 = plt.plot(fpr, tpr, label=legend_text +  '(AUC=%6.3f) ' %roc_auc)
    return roc_auc


def get_counts_key(data):
    key_count = Counter(data)
    data_len = len(key_count)
    
    #for val, ind in key_count.iteritems():
    counts = key_count.values() 
    keys = key_count.keys()
    
    return counts, keys

    
def get_BTO_tissue_map(map_file = 'data/tissue_map'):
    tissue_map = {}
    with open(map_file, 'r') as fp:
        for line in fp:
            tissue, bto = line.rstrip().split(',')
            tissue_map[tissue.upper()] = bto
            
    return tissue_map

def read_DOID_BTO_map(map_file = 'data/DOID_BTO_mapping.tsv'):
    doid_bto_map = {}
    with open(map_file, 'r') as fp:
        for line in fp:
            values = line.rstrip().split('\t')
            identifier = values[1]
            doid_bto_map[values[2].upper()] = identifier
    
    return doid_bto_map


def read_mirna_disease(mirna_dis_file = 'data/alldata-disease.txt'):
    mirna_dis = {}
    head = True
    with open(mirna_dis_file, 'r') as fp:
        for line in fp:
            if head:
                head = False
                continue
            else:
                values = line.rstrip().split('\t')
                mirna = values[1]
                dis_doid = values[5]
                if len(dis_doid):
                    mirna_dis.setdefault(mirna, set()).add(dis_doid)
    return mirna_dis


def read_gencode_mirna_annotation():
    gene_type_dict = {}
    gene_name_ensg = {}
    gene_id_position = {}
    fp = gzip.open('data/gencode.v19.genes.v6p_model.patched_contigs.gtf.gz')
    for line in fp:
        if line[0] == '#':
            continue
        values = line.rstrip('\r\n').split('\t')
        if values[2] != 'gene':
            continue
        gene_ann = values[-1]
        chrm = 'chr' + values[0]
        start = values[3]
        end = values[4]
        strand = values[6]
        key = (chrm, start, end, strand)
        split_gene_ann = gene_ann.split(';')
        gene_type_info = split_gene_ann[2]
        gene_type = extract_gene_type_name(gene_type_info)
        if gene_type != "miRNA":
            continue
        #gene_name_info = split_gene_ann[4]
        #gene_name = extract_gene_type_name(gene_name_info)
        '''gene_type_dict[gene_name] = gene_type'''

        gene_id_info = split_gene_ann[0]
        gene_id = extract_gene_type_name(gene_id_info)
        gene_id = gene_id.split('.')[0]
        #gene_type_dict[gene_id] = gene_type

        gene_name_ensg[key] = gene_id

    fp.close()

    return gene_name_ensg

def read_gencode_mirna_annotation_new():
    gene_type_dict = {}
    gene_name_ensg = {}
    gene_id_position = {}
    fp = gzip.open('data/gencode.v19.genes.v6p_model.patched_contigs.gtf.gz')
    for line in fp:
        if line[0] == '#':
            continue
        values = line.rstrip('\r\n').split('\t')
        if values[2] != 'gene':
            continue
        gene_ann = values[-1]
        chrm = 'chr' + values[0]
        start = values[3]
        end = values[4]
        strand = values[6]
        key = (chrm, start, end, strand)
        split_gene_ann = gene_ann.split(';')
        gene_type_info = split_gene_ann[2]
        gene_type = extract_gene_type_name(gene_type_info)
        if gene_type != "miRNA":
            continue
        gene_name_info = split_gene_ann[4]
        gene_name = extract_gene_type_name(gene_name_info)
        #if 'MIR30C1' in gene_name:
        #    pdb.set_trace()
        end_name = str(gene_name[3:])
        if '-' not in end_name and not end_name.isdigit() and end_name[-1].isdigit():
            new_gene_name = 'hsa' + '-' + gene_name[:3] + '-' + end_name[:-1] + '-' + end_name[-1]
        else:
            new_gene_name =  'hsa' + '-' + gene_name[:3] + '-' + gene_name[3:]
        key = new_gene_name.lower()
        '''gene_type_dict[gene_name] = gene_type'''

        gene_id_info = split_gene_ann[0]
        gene_id = extract_gene_type_name(gene_id_info)
        gene_id = gene_id.split('.')[0]
        #gene_type_dict[gene_id] = gene_type

        gene_name_ensg[key] = gene_id

    fp.close()

    return gene_name_ensg

def read_mirbase_id(mirbase = 'data/hsa.gff3.txt'):
    mirna_id_dict = {}
    mirna_name_dict = {}
    with open(mirbase) as fp:
        for line in fp:
            if 'miRNA_primary_transcript' not in line:
                continue
            if line[0] == '#':
                continue
            values = line.rstrip().split('\t')
            key = (values[0], values[3], values[4], values[6])
            names = values[-1].split(';')
            id = names[0].split('=')[-1]
            name = names[-1].split('=')[-1]
            mirna_name_dict[key] = name
            #mirna_id_dict[key] = id

    return mirna_name_dict#, mirna_id_dict

def read_mirbase_id_name(mirbase = 'data/mature.fa.gz'):
    id_name = {}
    with gzip.open(mirbase, 'r') as fp:
        for line in fp:
            if '>hsa' in line:
                values = line[1:].split('\t')
                id_name[values[1]] = values[0]

    return id_name

def read_ensp_ensg(map_file = 'data/string_9606_ENSG_ENSP_10_all_T.tsv'):
    ensp_ensg = {}
    with open(map_file) as fp:
        for line in fp:
            values = line.rstrip().split()
            ensp_ensg[values[-1]] = values[1]
    return ensp_ensg

def read_string_interaction(string_file = 'data/9606.protein.links.v10.txt.gz', cutoff = 400):
    ensp_ensg = read_ensp_ensg()
    interact_pair = {}
    all_genes = set()
    head = True
    with gzip.open(string_file) as fp:
        for line in fp:
            if head:
                head = False
                continue
            values = line.rstrip().split()
            pro1 = values[0].split('.')[-1]
            pro2 = values[1].split('.')[-1]
            if pro1 not in ensp_ensg or pro2 not in ensp_ensg:
                continue
            score = int(values[2])
            if score < cutoff:
                continue
            gene1 = ensp_ensg[pro1]
            gene2 = ensp_ensg[pro2]
            all_genes.add(gene1)
            all_genes.add(gene2)
            interact_pair[(gene1, gene2)] = score
            #interact_pair.setdefault(gene1, []).append(gene2)

    print len(interact_pair), len(all_genes)

    return interact_pair, all_genes

def read_ncrna_interaction(ncrna_file = 'data/9606.v1.combined.tsv.gz', cutoff = 0.15):
    ensp_ensg = read_ensp_ensg()
    interact_pair = {}
    genes = set()
    mirna_set = set()
    with gzip.open(ncrna_file) as fp:
        for line in fp:
            if 'hsa-miR-' not in line:
                continue

            values = line.rstrip().split()
            mirna = values[1].lower()
            pro = values[2]
            if pro not in ensp_ensg:
                continue
            score = float(values[-1])
            if score < cutoff:
                continue
            gene = ensp_ensg[pro]
            genes.add(gene)
            if '-3p' in mirna or '-5p' in mirna:
                #mirnas = mirna.split('-')
                mirna = mirna[:-3] #mirnas[0] + '-' + mirnas[1] + '-' + mirnas[2]

            mirna_set.add(mirna)
            interact_pair[(mirna, gene)] = score
            #interact_pair.setdefault(mirna, []).append(gene)

    print len(interact_pair), len(genes), len(mirna_set)

    return interact_pair, genes, mirna_set


def get_graph(graph):
    new_graph = nx.Graph()
    for source, targets in graph.iteritems():
        for inner_dict in targets:
            assert len(inner_dict) == 1
            new_graph.add_edge(int(source) - 1, int(inner_dict.keys()[0]) - 1,
                               weight=inner_dict.values()[0])
    adjacency_matrix = nx.adjacency_matrix(new_graph)

def read_expression_data(input_file, data = 3):
    ensg_ensp_map = read_ensp_ensg()
    disease_gene_dict, disease_name_map, whole_disease_gene_dict = read_DISEASE_database(confidence=2, ensg_ensp_map = ensg_ensp_map)
    disease_name_map, whole_disease_gene_dict = {}, {}

    gene_type_dict, gene_name_ensg, gene_id_position = read_gencode_gene_type()

    log2 = True
   # whole_data, samples = read_gtex_expression(input_file, gene_type_dict)
    if data == 0:
        whole_data, samples = read_human_RNAseq_expression(input_file, gene_name_ensg,
                                                           log2=log2)  # for microarray expression data
    elif data == 1:
        whole_data, samples = read_evolutionary_expression_data(input_file, use_mean=False, log2=log2)
    elif data == 2:
        whole_data, samples = read_average_read_to_normalized_RPKM(input_file, use_mean=False, log2=log2)
    elif data == 3:
        whole_data, samples = read_gtex_expression(input_file, gene_type_dict)

    #disease_lncRNA_data, lncRNAlabels, lncRNA_list, atmp = get_mRNA_lncRNA_expression_RNAseq_data(whole_data,
    #                                                                                              gene_type_dict=gene_type_dict,
    #                                                                                              mRNA=False)

    mirna_ensg_name = read_gencode_mirna_annotation_new()
    #pdb.set_trace()
    string_interaction, all_genes = read_string_interaction()
    rain_interaction, genes, mirnas = read_ncrna_interaction()
    #pdb.set_trace()
    new_set_mirnas = set(mirna_ensg_name.keys()) & mirnas

    #mRNA labels
    all_other_disease_mRNA = set()
    mrna_diseases = {}
    all_mrnas_num = set()
    for key, val in disease_gene_dict.iteritems():
        all_mrnas_num  = all_mrnas_num | val

    for key, val in disease_gene_dict.iteritems():
        if len(val) < 50:
            continue
        if 5*len(val) > len(all_mrnas_num): #to remove those diseases who are parent diseases of most disease, like cancer
            continue

        all_other_disease_mRNA = all_other_disease_mRNA | val
        for va in val:
            mrna_diseases.setdefault(va, set()).add(key)

    all_dis =set() #disease_gene_dict.keys()

    disease_gene_dict = {}
    #pdb.set_trace()
    all_disease_ensg = []
    all_nodes = set()
    ensg_expression = []
    for ensg in all_other_disease_mRNA:
        #if ensp in ensg_ensp_map:
        #    ensg = ensg_ensp_map[ensp]
        if ensg not in all_genes:
            continue
        if ensg in whole_data:
            all_disease_ensg.append(ensg)
            all_dis = all_dis | mrna_diseases[ensg]
            #ensg_expression.append(whole_data[ensg])

    all_mrna_len = len(all_disease_ensg)
    all_mrna_set = set(all_disease_ensg)
    print all_mrna_len

    filter_mirna_set = set()
    new_rain_interaction = {}
    for key, val in rain_interaction.iteritems():
        mir, eng = key
        if eng not in all_mrna_set:
            continue
        if mir in new_set_mirnas:
            mir_ensg = mirna_ensg_name[mir]
            filter_mirna_set.add(mir_ensg)
            new_key = (mir_ensg, eng)
            new_rain_interaction[new_key] = val

    rain_interaction = {}

    #pdb.set_trace()
    disease_miRNA_data = list(filter_mirna_set)
    all_exports = all_disease_ensg + disease_miRNA_data
    gene_len = len(all_exports)
    adj = np.zeros((gene_len, gene_len))
    #all_export_sets = set(all_exports)
    #new interaction string_interaction
    for i in range(1, gene_len):
        for j in range(i):
            key = (all_exports[i], all_exports[j])
            if key in string_interaction or key in new_rain_interaction:
                adj[i, j] = 1
                adj[j, i] = 1

    print 'non zero', np.count_nonzero(adj)
    string_interaction = {}
    new_rain_interaction = {}
    #features
    features = []
    #fw = open('gene_features', 'w')
    for gene in all_exports:
        line_str = whole_data[gene]
        features.append(line_str)
    whole_data = {}

    disease_miRNA_set = set(disease_miRNA_data)
    mirna_disease = read_mirna_disease()
    new_mirna_disease = {}
    for mirna, val in mirna_disease.iteritems():
        if mirna in mirna_ensg_name:
            mirna_ensg = mirna_ensg_name[mirna]
            if mirna_ensg not in disease_miRNA_set:
                continue
            new_mirna_disease[mirna_ensg] = val

    mirna_disease = {}
    print len(new_mirna_disease)
    #labels
    labels = []
    all_dis = list(all_dis)
    print 'disease', len(all_dis)

    for gene in all_exports:
        init_labels = np.array([0] * len(all_dis))
        if gene in mrna_diseases:
            diseases = mrna_diseases[gene]
        elif gene in new_mirna_disease:
            #pdb.set_trace()
            diseases = new_mirna_disease[gene]
            diseases = diseases & set(all_dis)
        else:
            labels.append(init_labels)
            continue
        inds = []
        for pro in diseases:
            inds.append(all_dis.index(pro))
        init_labels[inds] = 1
        labels.append(init_labels)

    all_index = range(gene_len)
    idx_train_val = all_index[:all_mrna_len]
    random.shuffle(idx_train_val)
    #keep 20% as validation
    train_num = int(0.8*all_mrna_len)
    idx_train = idx_train_val[:train_num]
    idx_val = idx_train_val[train_num:]

    idx_test = all_index[all_mrna_len:]


    features = preprocess_data_tissue(np.array(features))
    features, scaler = preprocess_data(features)
    features = sp.csr_matrix(features, dtype=np.float32)
    features = normalize(features)

    features = np.array(features.todense())
    #pdb.set_trace()
    adj = sp.csr_matrix(adj)
    adj =adj + sp.eye(adj.shape[0])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    labels = np.array(labels)

    return adj, torch.FloatTensor(features), torch.FloatTensor(labels), torch.LongTensor(np.array(idx_train)), \
           torch.LongTensor(np.array(idx_val)), torch.LongTensor(np.array(idx_test)), all_dis, new_mirna_disease, disease_miRNA_data

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(labels, output):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def accuracy_mutli(labels, output):
    labels = labels.data.numpy()
    output = 1 - output.data.numpy()
    macro_auc = roc_auc_score(labels, output, average='macro')
    micro_auc = roc_auc_score(labels, output, average='micro')
    weight_auc = roc_auc_score(labels, output, average='weighted')
    #acc = accuracy_score(labels, output)
    return macro_auc

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def train(epoch, model, optimizer, features, adj, idx_train, labels, idx_val, criteria = None):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    #pdb.set_trace()
    loss_train = criteria(output[idx_train], labels[idx_train])

    acc_train = accuracy_mutli(labels[idx_train], output[idx_train])

    loss_train.backward()
    optimizer.step()

    fastmode = False

    if not fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)
    #pdb.set_trace()
    loss_val = criteria(output[idx_val], labels[idx_val])
    acc_val = accuracy_mutli(labels[idx_val], output[idx_val])
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'time: {:.4f}s'.format(time.time() - t))

def read_mirpd(inputfile):
    mirna_ensg_name = read_gencode_mirna_annotation_new()
    dis_mir_score = {}
    mirna_set = set()
    with open(inputfile, 'r') as fp:
        for line in fp:
            values = line.rstrip().split()
            mir = values[0].lower()
            if mir in mirna_ensg_name:
                mir = mirna_ensg_name[mir]
                disease = values[1]
                score = values[2]

                dis_mir_score[(disease, mir)] = float(score)
                mirna_set.add(mir)
    return dis_mir_score, mirna_set


def calculate_auc(output, all_dis, new_mirna_disease, disease_miRNA_list):
    predi_mirna_dis = {}
    print output.shape
    print len(all_dis), len(disease_miRNA_list)
    dis_dict = {}
    mirna_dict = {}
    for i in range(len(disease_miRNA_list)):
        mirna_dict[disease_miRNA_list[i]] = i

    for j in range(len(all_dis)):
        dis_dict[all_dis[j]] = j
            #new_key = (disease_miRNA_list[i], all_dis[j])
            #predi_mirna_dis[new_key] = output[i, j]

    disease_mirna_dict = {}
    print 'disease'
    all_dis_set = set(all_dis)
    for key, val in new_mirna_disease.iteritems():
        for dis in val:
            if dis not in all_dis_set:
                continue
            disease_mirna_dict.setdefault(dis, set()).add(key)
    #pdb.set_trace()
    new_mirna_disease = {}
    #random.shuffle(disease_miRNA_list)
    print 'scoring'
    mirna_set = set(disease_miRNA_list)
    all_dis = []
    disease_miRNA_list = []
    labels = []
    probs = []
    print len(disease_mirna_dict.keys())
    for dis, mirnas in disease_mirna_dict.iteritems():
        #if mirna not in
        j = dis_dict[dis]
        num_mirnas = len(mirnas)
        for mirna in mirnas:
            labels.append(1)
            i = mirna_dict[mirna]
            probs.append(output[i, j])

        remain_mirnas = mirna_set - mirnas
        new_reamin_list = list(remain_mirnas)
        random.shuffle(new_reamin_list)
        nega_mirnas = new_reamin_list[:num_mirnas]
        new_reamin_list = []
        for mirna in nega_mirnas:
            labels.append(0)
            i = mirna_dict[mirna]
            probs.append(output[i, j])

    Figure = plt.figure()
    # pdb.set_trace()
    auc = plot_roc_curve_miRNA(labels, probs, 'DismiG')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title('KNN')
    plt.legend(loc="lower right")
    # plt.savefig(save_fig_dir + selected + '_' + class_type + '.png')
    plt.show()
    return auc

def test(model, features, adj, labels, idx_test, all_dis, new_mirna_disease, disease_miRNA_list, criteria = None):
    model.eval()
    output = model(features, adj)
    #pdb.set_trace()
    loss_test = criteria(output[idx_test], labels[idx_test])
    auc_test = calculate_auc(1 - output[idx_test].data.cpu().numpy(), all_dis, new_mirna_disease, disease_miRNA_list)

    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()))
    print auc_test

def run_gcn():
    adj, features, labels, idx_train, idx_val, idx_test, all_dis, new_mirna_disease, disease_miRNA_data = read_expression_data('data/GTEx_Analysis_v6p_RNA-seq_RNA-SeQCv1.1.8_gene_median_rpkm.gct.gz', data = 3)
    #adj, features, labels, idx_train, idx_val, idx_test, all_dis, new_mirna_disease, disease_miRNA_data = read_expression_data('/home/panxy/project/coexpression/data/GSE43520/genes.fpkm_table', data =1)
    print adj.shape, features.shape, labels.shape, idx_train.shape, idx_val.shape, idx_test.shape
    #pdb.set_trace()
    # Model and optimizer
    np.random.seed(0)
    torch.manual_seed(0)
    cuda = False
    if cuda:
        torch.cuda.manual_seed(0)
    #pdb.set_trace()
    num_dis = len(all_dis)
    model = GCN(nfeat=features.shape[1],
                nhid=num_dis*3,
                nclass=num_dis,
                dropout=0.8)
    optimizer = optim.Adam(model.parameters(),
                           lr=0.0001, weight_decay=0.005)

    criterion = nn.BCELoss()
    if cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    # Train model
    t_total = time.time()
    epochs = 50
    for epoch in range(epochs):
        train(epoch, model, optimizer, features, adj, idx_train, labels, idx_val, criterion)
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Testing
    test(model, features, adj, labels, idx_test, all_dis, new_mirna_disease, disease_miRNA_data, criterion)

run_gcn()

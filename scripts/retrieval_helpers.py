import random
from collections import Counter

import numpy as np
import pandas as pd
import scipy

# direction1 is group1 -> group2, direction2 is group2 -> group1
group1 = {'race': ['MW1', 'MB1', 'FW1', 'FB1', 'MW2', 'MB2', 'FW2', 'FB2'], 
         'gender': ['MW1', 'FW1', 'MB1', 'FB1', 'MW2', 'FW2', 'MB2', 'FB2'],
         'race_aug': ['MW', 'MB', 'FW', 'FB'],
         'gender_aug': ['MW', 'FW', 'MB', 'FB'],
         'same': ['MW1', 'MW2', 'MB1', 'MB2', 'FW1', 'FW2', 'FB1', 'FB2'],
         'no_demo': ['', '']}

group2 = {'race': ['MB1', 'MW1', 'FB1', 'FW1', 'MB2', 'MW2', 'FB2', 'FW2'], 
         'gender': ['FW1', 'MW1', 'FB1', 'MB1', 'FW2', 'MW2', 'FB2', 'MB2'],
         'race_aug': ['MB', 'MW', 'FB', 'FW'],
         'gender_aug': ['FW', 'MW', 'FB', 'MB'],
         'same': ['MW2', 'MW1', 'MB2', 'MB1', 'FW2', 'FW1', 'FB2', 'FB1'],
         'no_demo': ['no_space', 'typos_10']}

def cosine_sim(X, Y):
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
    Y_norm = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    cos_sim = np.dot(X_norm, Y_norm.T)
    return cos_sim

def top_k_indices(arr, k):
    return np.argsort(arr, axis=1)[:, -k:][:, ::-1]

def top_perc_indices(arr, p=10):
    k = int(np.ceil(arr.shape[1] * p / 100))
    return np.argsort(arr, axis=1)[:, -k:][:, ::-1]

def categorize(x, cutoff):
    if 0 <= x < cutoff:
        return 'MW'
    elif cutoff <= x < 2*cutoff:
        return 'MB'
    elif 2*cutoff <= x < 3*cutoff:
        return 'FW'
    else:
        return 'FB'

def return_index(sorted_array, new_val):
    index = np.searchsorted(sorted_array, new_val)
    return index

def update_dataframe_with_counts(df_posts, counts_percent_dict):
    """
    Update dataframe with group counts from counts_percent_dict.
    """
    for key1, val1 in counts_percent_dict.items():
        for key2, val2 in val1.items():
            key = f'{key2}_{key1}'
            df_posts[key] = val2
    return df_posts

def get_group_counts(job_embeddings, resume_embeddings_list, percents, cutoff):
    """
    Calculate group counts for top percentile of resumes for each job.
    """
    counts_percent_dict = {}
    for i, resume_embeddings in enumerate(resume_embeddings_list):
        arr = cosine_sim(job_embeddings, resume_embeddings)
    
        for percent in percents:
            counts_dict = {'FB': [], 'FW': [], 'MB': [], 'MW': []}
        
            # get top-x percent of resumes
            indices = top_perc_indices(arr, p=percent)
        
            # convert indices to group
            vectorized_map = np.vectorize(categorize)
            mapped_array = vectorized_map(indices, cutoff)

            # obtain and add counts for each group
            for en, m in enumerate(mapped_array): 
                c = Counter(m)
                for key in list(counts_dict.keys()):
                    counts_dict[key].append(c.get(key, 0)) # in case group does not show up in top ranked resumes
        
            counts_percent_dict[f'{percent}_{i+1}'] = counts_dict 
    return counts_percent_dict

def get_top_represented_groups(df_posts, percents, groups=['FB', 'FW', 'MB', 'MW']):
    """
    Compute how often each group is the top represented across job posts.
    """
    max_groups_dict = {}
    for percent in percents:
        max_groups = []
        for en in [1,2]:
            cols = [f'{group}_{percent}_{en}' for group in groups]
    
            # Gets the top represented group(s) for each job post
            max_vals = [
                df_posts[cols].columns[df_posts[cols].loc[i] == df_posts[cols].loc[i].max()].tolist()
                for i in df_posts[cols].index
            ]
            
            max_groups.extend(max_vals)
            
        max_groups = [y.split('_')[0] for x in max_groups for y in x]
        max_groups_dict[percent] = max_groups
        
    return max_groups_dict

def compute_nonuniform_job(df_posts, percents, thresh):
    """
    Compute nonuniformity metric (i.e., how often top resumes deviate from uniform distribution) across job posts.
    """
    props_dict_job = {}
    num_jobs = len(df_posts)
    for percent in percents:
        props = []
        for en in [1, 2]:
            count = 0
            cols_sub = [col for col in df_posts.columns if f'{percent}_{en}' in col]
            
            for _, row in df_posts[cols_sub].iterrows():
                test = scipy.stats.chisquare(row) # compute chi-squared test using group counts per job post
                if test.pvalue < thresh/num_jobs:  # bonferroni correction
                    count += 1
            
            props.append(count/num_jobs)
        props_dict_job[percent] = np.average(props)
    
    return props_dict_job

def compute_nonuniform_occ(df_posts, percents, thresh, occupations):
    """
    Compute nonuniformity metric (i.e., how often top resumes deviate from uniform distribution) across occupations.
    """
    props_dict_occ = {}
    for percent in percents:
        prop_list = []
        for en in [1, 2]:
            count = 0
            cols_sub = [col for col in df_posts.columns if f'{percent}_{en}' in col]
            
            for occupation in occupations:
                df_sub = df_posts[df_posts.occupation == occupation]
                vals = [np.sum(df_sub[col].values) for col in cols_sub]
                test = scipy.stats.chisquare(vals)  # compute chi-squared test using group counts per occupation
                
                if test.pvalue < thresh/len(occupations): # bonferroni correction
                    count += 1
            
            prop_list.append(count/len(occupations))
        props_dict_occ[percent] = np.average(prop_list)
    
    return props_dict_occ

def calculate_exclusion_rate(ranks_perturbed):
    """
    Calculate the rate at which perturbed resumes fall outside the top-k based on perturbed ranks.
    """
    return np.sum(ranks_perturbed > ranks_perturbed.shape[1]-1) / (ranks_perturbed.shape[0] * ranks_perturbed.shape[1])


def compute_perturbed_ranks(sims_perturbed, arr_orig_sorted):
    """
    Compute ranks of perturbed resumes relative to original resume rankings.
    """
    ranks_perturbed = np.zeros((sims_perturbed.shape[0], sims_perturbed.shape[1]))
    for row in range(sims_perturbed.shape[0]):
        for col in range(sims_perturbed.shape[1]):
            val = sims_perturbed[row, col]
            
            # get index where perturbed resume falls according to original resumes
            index = return_index(arr_orig_sorted[row], val)

            # rank is reverse ordered (high similarity -> low rank)
            rank = arr_orig_sorted.shape[1] - index
            ranks_perturbed[row, col] = rank
            
    return ranks_perturbed

def compute_exclusion(job_embeddings, resume_embedding_dict, key, k_list):
    """
    Compute exclusion (i.e., how often perturbed resumes fall outside the top-k)
    for all k values in k_list. 
    """
    exclusion_dict = {}
    for k in k_list:
        exclusion_vals = []
        # iterating over original and perturbed resumes
        for g1, g2 in zip(group1[key], group2[key]):
            # get cosine similarities between resumes and job posts
            arr_orig = cosine_sim(job_embeddings, resume_embedding_dict[f'resume_{g1}'])
            
            # sort similarities in ascending order
            arr_orig_sorted = np.sort(arr_orig, axis=1)
            
            # get similarities betwen perturbed resumes and job posts
            arr_perturbed = cosine_sim(job_embeddings, resume_embedding_dict[f'resume_{g2}'])
        
            # get indices of highest ranked original resumes
            top_indices_orig = top_k_indices(arr_orig, k) 
            
            # get corresponding perturbed resume similarity values (using original top-ranked indices)
            row_indices = np.arange(len(top_indices_orig))[:, np.newaxis] # array of shape (# job_posts, 1)
            sims_perturbed = arr_perturbed[row_indices, top_indices_orig]
    
            # for each perturbed similarity value, get the index/rank according to original resumes
            ranks_perturbed = compute_perturbed_ranks(sims_perturbed, arr_orig_sorted)
    
            # calculate how often the rank is outside top-k, across all job posts
            exclusion = calculate_exclusion_rate(ranks_perturbed)
            exclusion_vals.append(exclusion)
    
        exclusion_dict[k] = {
            'average': np.average(exclusion_vals),
            'direction1': np.average(exclusion_vals[::2]),  # group1 -> group2
            'direction2': np.average(exclusion_vals[1::2])  # group2 -> group1
        }

    return exclusion_dict
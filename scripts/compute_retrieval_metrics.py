import argparse
import json
import os
from collections import Counter

import numpy as np
import pandas as pd

import retrieval_helpers as rh

thresh = 0.05 # alpha for chi-squared test
percents = [5, 10] # top-x% of resumes (non-uniformity)
k_list =  [5, 10, 100] # top-n resumes (exclusion)

def save_dict_to_json(dictionary, filepath):
    """Save dictionary to a JSON file."""
    with open(filepath, 'w') as f:
        json.dump(dictionary, f, indent=4)

def compute_results(model, results_dict, embedding_filename, path):
    # load resume embeddings
    resume_embeddings_all = np.load(f'{path}embeddings/resumes/{embedding_filename}.npy') 

    # load job posts
    job_post_filename = 'job_postings_kaggle' if 'kaggle' in embedding_filename else 'job_postings_gen'
    df_posts = pd.read_csv(f'{path}job_posts/{job_post_filename}.csv')

    occupations = np.unique(df_posts.occupation.values)
    print(occupations, len(occupations))

    # load job embeddings
    job_post_filename = f'{job_post_filename}_model={model}'
    job_embeddings = np.load(f'{path}embeddings/jobs/{job_post_filename}.npy')
    print(job_embeddings.shape, resume_embeddings_all.shape)

    # all name perturbation-only resumes (between and within demographic) have two versions
    # so we will average results across both
    if 'non_name' in embedding_filename:
        num_groups = 3
        start1, start2, jump = 0, 0, 1
        keys = ['no_demo']
    elif 'name' in embedding_filename:
        num_groups = 8
        start1, start2, jump = 0, 1, 2
        keys = ['race', 'gender', 'same']
    elif 'augmented' in embedding_filename:
        num_groups = 4
        start1, start2, jump = 0, 0, 1
        keys = ['race_aug', 'gender_aug']

    # separate embeddings by group
    resume_embeddings_all_split = np.split(resume_embeddings_all, num_groups)
    resume_embeddings1 = np.vstack([resume_embeddings_all_split[i] for i in range(start1, num_groups, jump)])
    resume_embeddings2 = np.vstack([resume_embeddings_all_split[i] for i in range(start2, num_groups, jump)])
    cutoff = len(resume_embeddings_all_split[0]) # use to categorize indices into groups
    print(cutoff, resume_embeddings1.shape, resume_embeddings2.shape)

    # create resume_embedding_dict to make it easy to get perturbed embeddings
    if 'non_name' in embedding_filename:
        resume_embedding_dict = {'resume_': resume_embeddings_all_split[0], 
                                'resume_no_space': resume_embeddings_all_split[1],
                                'resume_typos_10': resume_embeddings_all_split[2]}
    elif 'name' in embedding_filename:
        resume_embedding_dict = {'resume_MW1': resume_embeddings_all_split[0], 'resume_MW2': resume_embeddings_all_split[1],
                                'resume_MB1': resume_embeddings_all_split[2], 'resume_MB2': resume_embeddings_all_split[3],
                                'resume_FW1': resume_embeddings_all_split[4], 'resume_FW2': resume_embeddings_all_split[5],
                                'resume_FB1': resume_embeddings_all_split[6], 'resume_FB2': resume_embeddings_all_split[7]}
    elif 'augmented' in embedding_filename:
        resume_embedding_dict = {'resume_MW': resume_embeddings_all_split[0],
                                'resume_MB': resume_embeddings_all_split[1],
                                'resume_FW': resume_embeddings_all_split[2],
                                'resume_FB': resume_embeddings_all_split[3]}
    
    # compute nonuniformity metrics
    counts_percent_dict = rh.get_group_counts(job_embeddings, [resume_embeddings1, resume_embeddings2], percents, cutoff)
    df_posts = rh.update_dataframe_with_counts(df_posts, counts_percent_dict)

    max_groups_dict = rh.get_top_represented_groups(df_posts, percents, groups=['FB', 'FW', 'MB', 'MW'])
    max_groups = dict([(key, dict([(k, v/len(val)) for k, v in Counter(val).items()])) for key, val in max_groups_dict.items()])

    props_dict_job = rh.compute_nonuniform_job(df_posts, percents, thresh)
    props_dict_occ = rh.compute_nonuniform_occ(df_posts, percents, thresh, occupations)
    
    if 'non_name' not in embedding_filename: # compute nonuniformity with respect to demographic groups only
        results_dict[model]['nonuniformity']['max_groups_job'] = max_groups
        results_dict[model]['nonuniformity']['props_job'] = props_dict_job
        results_dict[model]['nonuniformity']['props_occ'] = props_dict_occ

    # compute exclusion metrics
    for key in keys:
        exclusion_dict = rh.compute_exclusion(job_embeddings, resume_embedding_dict, key, k_list)
        results_dict[model]['exclusion'][key] = exclusion_dict 

    return results_dict

def main(load_path, save_path, resume_dataset, embedding_models):
    if not os.path.exists(f'{save_path}retrieval_metrics'):
        os.makedirs(f'{save_path}retrieval_metrics')

    # create dict for storing results
    results_dict = {}
    for model in embedding_models:
        embedding_filename = f'df_resumes_{resume_dataset}_model={model}'
        results_dict[model] = {}
        results_dict[model]['nonuniformity'] = {}
        results_dict[model]['exclusion'] = {}
        results_dict = compute_results(model, results_dict, embedding_filename, load_path)
    # save result dictionary as a json file
    save_dict_to_json(results_dict, f'{save_path}retrieval_metrics/{embedding_filename}.json')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume_dataset', type=str, 
        choices=['kaggle_name', 'kaggle_non_name', 'gen_name', 'gen_non_name', 'gen_augmented'],
        required=True,  
        help='Choose which resume dataset to use')
    parser.add_argument('--embedding_models', type=str, 
                choices=['text-embedding-3-small', 'text-embedding-3-large', 
                        'embed-english-v3.0', 'mistral-embed'],
                nargs='+',  # accepts one or more arguments
                default=['text-embedding-3-small', 'text-embedding-3-large', 
                        'embed-english-v3.0', 'mistral-embed'],  # default is all options
                help='Choose which embedding models to use')
    parser.add_argument('--load_path', type=str, default='', 
                        help='Path where resume and embedding folders are located')
    parser.add_argument('--save_path', type=str, default='', 
                        help='Path where output will be saved')
    
    # Parse arguments
    args = parser.parse_args()
    load_path = args.load_path + '/' if len(args.load_path) > 0 and args.load_path[-1] != '/' else args.load_path
    save_path = args.save_path + '/' if len(args.save_path) > 0 and args.save_path[-1] != '/' else args.save_path
    main(load_path, save_path, args.resume_dataset, args.embedding_models)


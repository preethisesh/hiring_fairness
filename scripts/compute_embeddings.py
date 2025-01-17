import argparse
import json
import os
import time

import numpy as np
import pandas as pd

import helpers

def compute_embeddings_all(texts, model, input_type):
    embeddings = []
    for en, text in enumerate(texts):
        if en%100==0 and en>0:
            print(en)
        emb = helpers.compute_embeddings(text, model, input_type=input_type) 
        embeddings.append(emb) 
    
    embeddings = np.array(embeddings)
    return embeddings

def main(load_path, save_path, resume_dataset, embedding_models):
    # create directories to store embeddings
    if not os.path.exists(f'{save_path}embeddings'):
        os.makedirs(f'{save_path}embeddings')

    if not os.path.exists(f'{save_path}embeddings/jobs'):
        os.makedirs(f'{save_path}embeddings/jobs')

    if not os.path.exists(f'{save_path}embeddings/resumes'):
        os.makedirs(f'{save_path}embeddings/resumes')

    # load job posts
    job_post_filename = 'job_postings_kaggle' if 'kaggle' in resume_dataset else 'job_postings_gen'
    df_posts = pd.read_csv(f'{load_path}job_posts/{job_post_filename}.csv')
    # load resumes
    df_resumes = pd.read_csv(f'{load_path}resumes/df_resumes_{resume_dataset}.csv')

    # specify appropriate group columns depending on which resumes are provided
    if 'non_name' in resume_dataset:
        resume_cols = ['resume','resume_no_space', 'resume_typos_10']
    elif 'name' in resume_dataset:
        resume_cols = ['resume_MW1','resume_MW2','resume_MB1','resume_MB2',
                       'resume_FW1','resume_FW2','resume_FB1','resume_FB2']
    elif 'augmented' in resume_dataset:
        resume_cols = ['resume_aug_MW','resume_aug_MB','resume_aug_FW','resume_aug_FB']

    for model in embedding_models:
        print(model)
        start_time = time.time()

        # compute job embeddings
        job_posts = df_posts.post.values
        print('number of job posts:', len(job_posts))
        job_embeddings = compute_embeddings_all(job_posts, model, 'search_query')

        save_file =  f'{save_path}embeddings/jobs/{job_post_filename}_model={model}.npy'
        np.save(save_file, job_embeddings)

        # compute resume embeddings
        resumes = []
        for col in resume_cols:
            resumes_sub = df_resumes[col].values
            resumes.extend(resumes_sub)
        print('number of resumes:', len(resumes))
        resume_embeddings = compute_embeddings_all(resumes, model, 'search_document')

        save_file =  f'{save_path}embeddings/resumes/df_resumes_{resume_dataset}_model={model}.npy'
        np.save(save_file, resume_embeddings)
        print(job_embeddings.shape, resume_embeddings.shape)
        print(f'Done with {model}:', time.time()-start_time)

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
                        help='Path where resumes are located')
    parser.add_argument('--save_path', type=str, default='', 
                        help='Path where output will be saved')
    
    # Parse arguments
    args = parser.parse_args()
    load_path = args.load_path + '/' if len(args.load_path) > 0 and args.load_path[-1] != '/' else args.load_path
    save_path = args.save_path + '/' if len(args.save_path) > 0 and args.save_path[-1] != '/' else args.save_path
    main(load_path, save_path, args.resume_dataset, args.embedding_models)


    
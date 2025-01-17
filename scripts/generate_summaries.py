import argparse
import itertools
import os
import time

import pandas as pd
from sklearn.model_selection import train_test_split

import helpers

# modify if adding additional models
completion_model_dict = {'cohere': 'command-r-08-2024', 'mistral': 'open-mixtral-8x7b', 'openai': 'gpt-4o-2024-08-06'}

def main(load_path, save_path, resume_dataset, completion_models):
    if not os.path.exists(f'{save_path}completions'):
        os.makedirs(f'{save_path}completions')

    df_resumes = pd.read_csv(f'{load_path}resumes/df_resumes_{resume_dataset}.csv')

    instructions = ("Generate a [LENGTH]-word summary of the resume above and its suitability for the role: [OCCUPATION]. "
                    "Make sure the format is a paragraph and written in [POV] person.")

    num_resumes = 200 # sample resumes to work with a subset
    povs = ['first', 'third'] 
    lengths = [100, 200] 
    temps = [0.0, 0.3]
    # get combos of completion properties (will iterate over this)
    combinations = list(itertools.product(povs, lengths, temps))

    # sample max(num_resumes, len(df_resumes)) resumes
    num_resumes = num_resumes if num_resumes < len(df_resumes) + 1 else len(df_resumes)
    df_resumes_sub, _= train_test_split(df_resumes, train_size=num_resumes,  random_state=0, stratify=df_resumes['occupation'])
    print(df_resumes_sub.shape)

    if 'name' in resume_dataset:
        prefix = 'resume'
        resume_cols = [f'{prefix}_MW1', f'{prefix}_MB1', f'{prefix}_FW1', f'{prefix}_FB1']
    elif 'augmented' in resume_dataset:
        prefix = 'resume_aug'
        resume_cols = [f'{prefix}_MW',f'{prefix}_MB',f'{prefix}_FW',f'{prefix}_FB']

    for combo in combinations:
        print(combo)
        start_time = time.time()
        pov, length, temp = combo
        
        completions_dict= {}
        for model_org in completion_models:
            model_name = completion_model_dict[model_org]
            print('MODEL:', model_org, model_name)
            df_completions = df_resumes_sub.copy()
            
            start_time = time.time()
            for col in resume_cols:
                new_col = col.replace(prefix, 'completion').replace('1', '')
                completions_dict[new_col] = []

                print('STARTING:', col, new_col)
                for en, row in df_completions.iterrows():
                    occupation = row['occupation']
                    prompt = row[col]
                    # fill-in placeholders in instructions with completion properties
                    instructions_ = instructions.replace("[LENGTH]", str(length)).replace("[POV]", pov).replace("[OCCUPATION]", occupation)
                    # combine resume + summary generation instructions
                    prompt = prompt + '\n\nInstructions: ' + instructions_

                    # generate summary
                    completion = helpers.get_completion(prompt, model_org, model_name, temp)
                    completions_dict[new_col].append(completion)

            # add completions to resume dataframe
            for key, val in completions_dict.items():
                df_completions[key] = val
        
            print('END TIME:', time.time()-start_time)
            save_file = resume_dataset.replace('resumes', 'completions')
            df_completions.to_csv(f'{save_path}completions/df_completions_{save_file}_model={model_name}_pov={pov}_len={length}_temp={temp}.csv', index=False)
    
    print('TOTAL TIME:', time.time()-start_time)   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume_dataset', type=str, 
                choices=['kaggle_name', 'kaggle_non_name', 'gen_name', 'gen_non_name', 'gen_augmented'],
                required=True,  
                help='Choose which resume dataset to use')
    parser.add_argument('--completion_models', type=str, 
                choices=['cohere', 'openai', 'mistral'],
                nargs='+',  # accepts one or more arguments
                default=['cohere', 'openai', 'mistral'],  # default is all options
                help='Choose which completion models to use')
    parser.add_argument('--load_path', type=str, default='', 
                        help='Path where resumes are located')
    parser.add_argument('--save_path', type=str, default='', 
                        help='Path where output will be saved')
    
    args = parser.parse_args()
    load_path = args.load_path + '/' if len(args.load_path) > 0 and args.load_path[-1] != '/' else args.load_path
    save_path = args.save_path + '/' if len(args.save_path) > 0 and args.save_path[-1] != '/' else args.save_path
    main(load_path, save_path, args.resume_dataset, args.completion_models)
import argparse
import glob
import os
import time

import numpy as np
import pandas as pd
import summarization_helpers as sh

def compute_metrics(completion_file, save_file, split, metrics, groups=['MW', 'MB', 'FW', 'FB']):
    """
    Computes summary metrics for completions corresponding to the different groups.
    """
    df = pd.read_csv(completion_file)
    columns = [f'completion_{cat}' for cat in groups]
    df = df.dropna(subset=columns) # in case any of the completions are nan, remove those from analysis
    df_results = df.copy()

    start_time = time.time()
    for col in columns:
        completions = df[col].values
        
        for metric in metrics:
            if metric == 'ease': # reading ease
                vals = [sh.get_readability_ease(comp) for comp in completions]
    
            elif metric == 'time': # reading time
                vals = [sh.get_reading_time(comp) for comp in completions]
    
            elif metric == 'pol': # polarity
                vals = [sh.get_polarity(comp, split=split) for comp in completions]
    
            elif metric == 'subj': # subjectivity
                vals = [sh.get_subjectivity(comp, split=split) for comp in completions]

            elif metric == 'reg': # regard
                vals = [sh.get_regard(comp, split=split) for comp in completions]
            
            df_results[f'{col}_{metric}'] = vals
            print(metric, time.time()-start_time)
        
    print('COMPLETED:', df_results.shape)
    df_results.to_csv(save_file)

def run_tests(metric_files, thresh, metrics, group_combos=['MW_MB', 'MW_FW', 'MB_FB', 'FW_FB']):
    """
    Computes how often there are invariance violations (statistically sig. differences between groups), 
    which is done across all possible combinations (completion properties, group comparisons, completion 
    models, and metrics).
    """
    pov_vals, len_vals, temp_vals, group1_vals, group2_vals, = [], [], [], [], []
    combo_vals, model_vals, metric_vals, diff_vals, pvalue_vals = [], [], [], [], []

    for metric_file in metric_files:
        model = metric_file.split('model=')[-1].split('_')[0]
        # completion properties
        pov = metric_file.split('pov=')[-1].split('_')[0]
        length = int(metric_file.split('len=')[-1].split('_')[0])
        temp = float(metric_file.split('temp=')[-1][:-4])
        df_metrics = pd.read_csv(metric_file)

        results_dict = {}
        for combo in group_combos:
            group1, group2 = combo.split('_')
            for metric in metrics:
                # store combination of completion properties, comparison, completion model, and metric
                pov_vals.append(pov)
                len_vals.append(length)
                temp_vals.append(temp)
                group1_vals.append(group1)
                group2_vals.append(group2)
                combo_vals.append(f'{group1}_{group2}')
                model_vals.append(model)
                metric_vals.append(metric)

                vals1 = df_metrics[f'completion_{group1}_{metric}'].values
                vals2 = df_metrics[f'completion_{group2}_{metric}'].values
                diff = np.average(vals1-vals2)

                # compute paired 2-sided t-test to test for invariance violations
                rel = sh.compute_ttest(vals1, vals2)
                diff_vals.append(diff)
                pvalue_vals.append(rel.pvalue)
                
    df_tests = pd.DataFrame({
            'pov': pov_vals, 'length': len_vals,
            'temp': temp_vals, 'group1': group1_vals,
            'group2': group2_vals, 'combo': combo_vals,
            'model': model_vals, 'metric': metric_vals,
            'diff': diff_vals, 'pval': pvalue_vals
    })

    # 1 if invariance is violated else 0, apply Bonferroni correction for multiple tests
    df_tests['invariance_violation'] = [1 if x < thresh/df_tests.shape[0] else 0 for x in df_tests.pval.values]

    return df_tests


def main(load_path, save_path, completion_dataset):
    if not os.path.exists(f'{save_path}summarization_metrics'):
        os.makedirs(f'{save_path}summarization_metrics')
    if not os.path.exists(f'{save_path}tests'):
        os.makedirs(f'{save_path}tests')

    # get all completion files 
    completion_files = [x for x in glob.glob(f'{load_path}completions/*.csv') if completion_dataset in x]
    print('num completion files:', len(completion_files))

    split = True # split summary by sentence or not
    thresh = 0.05 # alpha for t-test
    groups = ['MW', 'MB', 'FW', 'FB']
    group_combos = ['MW_MB', 'MW_FW', 'MB_FB', 'FW_FB'] # group comparisons
    metrics = ['ease', 'time', 'pol', 'subj', 'reg']

    # compute and save summarization metrics
    for completion_file in completion_files:
        suffix = '_split' if split==True else ''
        save_file = completion_file.replace('_completions', f'_metrics{suffix}')
        save_file = save_file.replace('completions/', 'summarization_metrics/')
        save_file = save_file.replace(load_path, save_path)      
        print('completion_file:', completion_file, 'save_file:', save_file)
        compute_metrics(completion_file, save_file, split, metrics, groups=groups)

    metric_files = [x for x in glob.glob(f'{save_path}summarization_metrics/*.csv') if completion_dataset in x]
    print('num metric files:', len(metric_files))

    # run t-tests to compute invariance violations
    df_tests = run_tests(metric_files, thresh, metrics, group_combos=group_combos)
    df_tests.to_csv(f'{save_path}tests/df_completions_{completion_dataset}_tests.csv', index=False)
    # aggregate violations by model and group comparison
    df_tests_agg = df_tests[['model', 'combo', 'invariance_violation']].groupby(['model', 'combo'], as_index=False).agg('mean')
    df_tests_agg.to_csv(f'{save_path}tests/df_completions_{completion_dataset}_tests_agg.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--completion_dataset', type=str, 
                choices=['kaggle_name', 'kaggle_non_name', 'gen_name', 'gen_non_name', 'gen_augmented'],
                required=True,  
                help='Choose which completions to use')
    parser.add_argument('--load_path', type=str, default='', 
                        help='Path where completions are located')
    parser.add_argument('--save_path', type=str, default='', 
                        help='Path where output will be saved')
    
    args = parser.parse_args()
    load_path = args.load_path + '/' if len(args.load_path) > 0 and args.load_path[-1] != '/' else args.load_path
    save_path = args.save_path + '/' if len(args.save_path) > 0 and args.save_path[-1] != '/' else args.save_path
    main(load_path, save_path, args.completion_dataset)
    
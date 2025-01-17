# Who Does the Giant Number Pile Like Best: Analyzing Fairness in Hiring Contexts

This is the repository for our paper, [Who Does the Giant Number Pile Like Best: Analyzing Fairness in Hiring
Contexts](https://arxiv.org/pdf/2501.04316?). 

**Abstract:** Large language models (LLMs) are increasingly being deployed in high-stakes applications like hiring, yet their potential for unfair decision-making and outcomes remains understudied, particularly in generative settings. In this work, we examine the fairness of LLM-based hiring systems through two real-world tasks: resume summarization and retrieval. By constructing a synthetic resume dataset and curating job postings, we investigate whether model behavior differs across demographic groups and is sensitive to demographic perturbations. Our findings reveal that race-based differences appear in approximately 10% of generated summaries, while gender-based differences occur in only 1%. In the retrieval setting, all evaluated models display non-uniform selection patterns across demographic groups and exhibit high sensitivity to both gender and race-based perturbations. Surprisingly, retrieval models demonstrate comparable sensitivity to non-demographic changes, suggesting that fairness issues may stem, in part, from general brittleness issues. Overall, our results indicate that LLM-based hiring systems, especially at the retrieval stage, can exhibit notable biases that lead to discriminatory outcomes in real-world contexts.

## Data
### Resumes
We include two resume datasets: sampled resumes from an existing [Kaggle resume dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) and a synthetically generated dataset. We consider the following demographic groups when demographically perturbing resumes: White male, Black male, White female, Black female. We also perform (1) augmenting with extracurricular information (generated resumes only), (2) within-group demographic perturbations, and (3) non-demographic perturbations (spacing and typos). 
- **resumes/df_resumes_kaggle_name.csv**: Kaggle resumes with name perturbations only. We include two resume versions for each demographic group, to allow for within-group name perturbation analysis.
- **resumes/df_resumes_kaggle_non_name.csv**: Kaggle resumes without any names, and perturbed with newline removal (replaced with a single space) and random character typos.
- **resumes/df_resumes_gen_name.csv**: Generated resumes with name perturbations only. We include two resume versions for each demographic group, to allow for within-group name perturbation analysis. **(Will be added soon)**
- **resumes/df_resumes_gen_augmented.csv**: Generated resumes with name + extracurricular perturbations. We include a single resume per demographic group. **(Will be added soon)**
- **resumes/df_resumes_gen_non_name.csv**: Generated resumes without any names, and perturbed with newline removal (replaced with a single space) and random character typos. **(Will be added soon)**

### Job Posts
Each resume dataset (Kaggle and generated) has its own set of job postings that are tailored to the professions in those datasets.
- **job_posts/job_postings_kaggle.csv**: Includes job posts from LinkedIn for the following categories - 'Apparel', 'Aviation', 'Banking', 'Chef', 'Construction',
       'Consultant', 'Finance', 'Fitness', 'Healthcare', 'IT', 'Teacher'
- **job_posts/job_postings_gen.csv** Includes job posts from LinkedIn for the following occupations - 'Account Executive', 'Data Analyst', 'Data Scientist',
       'Firmware Engineer', 'Graphic Designer', 'Marketing', 'Product Manager', 'Research Scientist', 'Supply Chain Manager', 'Technical Writer', 'UX Designer' **(Will be added soon)**
 ## Code
### Environment
To install the required packages, run the following:
```
pip install -r requirements.txt
```
In addition, run `python -m spacy download en_core_web_sm`. 

### Summarization
To generate summaries, run `scripts/generate_summaries.py`. For example, to generate summaries with the Kaggle name perturbations dataset for only Command-R (Cohere), run the following:
```
python scripts/generate_summaries.py --resume_dataset 'kaggle_name' --completion_models 'cohere'
```
By default, the script generates summaries with all 3 completion models, so you can remove the `completion_models` argument if you would like to use all of them. Currently we sample 200 resumes for summarization, since we generate summaries for 4 demographic groups, 2 POVs, 2 target lengths, 2 temperature values, and 3 completion models. Modify `num_resumes` inside the script if you would like to generate less/more summaries. All generated summaries will be stored in the `completions/` folder.

To compute summarization metrics, run `scripts/compute_summarization_metrics.py`. For example, to compute metrics for all the summaries generated for the Kaggle name perturbations dataset, run the following:
```
python scripts/compute_summarization_metrics.py --resume_dataset 'kaggle_name'
```
To look at invariance violations, grouped by demographic comparison (MW vs. MB, FW vs. FB, MW vs. FW, MB vs. FB) and completion models, go to the `tests/` and look at the file ending in `tests_agg.csv` (the file ending in `tests.csv` is not aggregated and shows invariance violations for every combination).
Note: We only generate summarizations for df_resumes_kaggle_name.csv and df_resumes_gen_augmented.csv in the paper.
### Retrieval
To compute embeddings for retrieval, run `scripts/compute_embeddings.py`. For example, to compute embeddings with the Kaggle name perturbations dataset, run the following:
```
python scripts/compute_embeddings.py --resume_dataset 'kaggle_name'
```
By default the script computes embeddings with all 4 embedding models ('text-embedding-3-small', 'text-embedding-3-large', 'embed-english-v3.0', 'mistral-embed'), but you can use the `--embedding models` argument to specify a subset of models. All generated embeddings will be stored in the `embeddings/` folder.

To compute retrieval metrics, run `scripts/compute_retrieval_metrics.py`. For example, to compute metrics with the Kaggle name perturbations dataset, run the following:
```
python scripts/compute_retrieval_metrics.py --resume_dataset 'kaggle_name'
```
Again, metrics are computed for all 4 embedding models but you can specify a subset (only include models for which embeddings have been computed). Retrieval metrics are saved in a .json file in `retrieval_metrics/`. Note that the non-uniformity metric is *not* computed for non-name perturbations. Additionally, the exclusion results are broken up by direction1 and direction2. For gender perturbations direction1 is `M->F` and direction2 is `F->M`, for race perturbations direction1 is `W->B` and direction2 is `B->W`, for within-group perturbations direction1 and direction2 preserve the demographic group (and should therefore be close), and for non-name perturbations direction1 is `original -> modified spacing` and direction2 is `original -> typos`.

All of the scripts contain optional `load_path` and `save_path` arguments, which are left empty by default. You can leave this as is, or modify these arguments if you would like to change the file organization.

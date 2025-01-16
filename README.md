# Who Does the Giant Number Pile Like Best: Analyzing Fairness in Hiring Contexts

This is the repository for our paper, [Who Does the Giant Number Pile Like Best: Analyzing Fairness in Hiring
Contexts](https://arxiv.org/pdf/2501.04316?). 

**Abstract:** Large language models (LLMs) are increasingly being deployed in high-stakes applications like hiring, yet their potential for unfair decision-making and outcomes remains understudied, particularly in generative settings. In this work, we examine the fairness of LLM-based hiring systems through two real-world tasks: resume summarization and retrieval. By constructing a synthetic resume dataset and curating job postings, we investigate whether model behavior differs across demographic groups and is sensitive to demographic perturbations. Our findings reveal that race-based differences appear in approximately 10% of generated summaries, while gender-based differences occur in only 1%. In the retrieval setting, all evaluated models display non-uniform selection patterns across demographic groups and exhibit high sensitivity to both gender and race-based perturbations. Surprisingly, retrieval models demonstrate comparable sensitivity to non-demographic changes, suggesting that fairness issues may stem, in part, from general brittleness issues. Overall, our results indicate that LLM-based hiring systems, especially at the retrieval stage, can exhibit notable biases that lead to discriminatory outcomes in real-world contexts.

## Data
### Resumes
We include two resume datasets: sampled resumes from an existing [Kaggle resume dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) and a synthetically generated dataset. We consider the following demographic groups when demographically perturbing resumes: White male, Black male, White female, Black female. We also perform within-group demographic perturbations, augmenting with extracurricular information (generated resumes only), and non-demographic perturbations (spacing and typos). 
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

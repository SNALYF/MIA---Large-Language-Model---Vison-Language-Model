## Teams
rubric={accuracy:2}

Please use this [Google Sheet](https://docs.google.com/spreadsheets/d/17AULbRU_HfA9EZBchejRSPdq8JTHaZtLM_ROYvJPkHQ/edit?usp=sharing) to fill in your team information. The number of team members in each team must range from 3 to 4. 

The setup of the share task can be found [here](https://docs.google.com/presentation/d/1L0VyuTfF5YeLP0eOE_C2L_c5qM0myTyb7oMBKjefPcU/edit?usp=sharing).
We have prepared the train/val/test data and two finetuned models. Both models have been finetuned on the exact same set of data for 2 epochs. You can use any of the models or both to complete the task.

You can load the dataset with the following code:
```python
from datasets import load_dataset
dataset = load_dataset("UBC-SLIME/colx_531_group_project")
```
If a sample has been used in finetuning, it is labeled as `1` in the `is_member` column, otherwise `0`.


You can load the finetuned models using the following code:
```python
from transformers import AutoModelForCausalLM

lm1 = AutoModelForCausalLM.from_pretrained("UBC-SLIME/colx_531_smollm2-135m")

lm2 = AutoModelForCausalLM.from_pretrained("UBC-SLIME/colx_531_smollm2-360m")
```
`lm1` is finetuned from the `HuggingFaceTB/SmolLM2-135M` checkpoint, whereas `lm2` is finetuned from the `HuggingFaceTB/SmolLM2-360M` checkpoint.

You should submit your predictions to this [kagge competition](https://www.kaggle.com/competitions/531-project-membership-inference-attack/overview). A sample submission is available [here](https://github.ubc.ca/MDS-CL-2025-26/COLX_531_translation_students/blob/master/group_projects/submission_base_smollm2_135m_clinical_full.csv). In your prediction, you need to assign a score to each sample in the `is_member` column. The predictions do not need to be binary. You only need to assign higher scores to samples that you believe to be members. 

## Repo setup
rubric={mechanics:8}

* Credit: Some of the details related to this and the next sections are inspired by (and in some parts borrowed from) Julian Brooke.

- You will be working in groups, and for each assignment (with the exception of the teamwork evaluation), you will do your work in the appropriate milestone directory in your group's GitHub. For weekly submissions, please ensure that your individual repo has a link to the group repository's milestone folder, you have committed all changes to the master branch, and that the instructional team has read access.

- For the group repository, please create a group repo in the UBC GitHub (it will be in one of your own repos, not MDS-CL). All the members of your team should have write access, and the instructor and all teaching support team should be given read access (check the syllabus for our GitHub handles). 

- Create a branch for each member of your team, where each of you will do your individual work. You should never push directly to the master branch during this project. Instead, when you are ready to share your work, you should create a pull request to master, which should be reviewed by one other member of the team.

- You may generally choose how to organize your repo, but for each milestone, you should have one folder in the main repo directory where you will put milestone-specific written documents (such as the progress report), along with a readme with information about where to find any required code (which should NOT be put in the milestone folder). Make sure everything is in the master branch before the deadline; we won't go looking in your individual branches. You should not modify the documents in this folder after the deadline, or late penalties will be applied, see below. For the individual submission (on github LMS), again you will just post a link to the specific milestone folder in your group's repository.

**For lab3, please create a folder called milestone1** and you need to put everything you completed in this week inside this folder. 

## Teamwork contract
rubric={reasoning:10, writing:10}

Please create a teamwork contract and save it as `teamwork_contract.md`. This document will govern your working relationship and you are encouraged to design it to manage and resolve any issues that arise in group work.

A teamwork contract communicates specifically how the core group of people who are working together and gives more detail about the logisitics of working together and the expectations you have for each other. Some aspects of the teamwork contract could be:

- How will work be distributed in a fair and equitable way?
- What are the expected work hours for the project?
- How often will group meetings occur?
- How will you manage online meetings? What technology will you use? Please provide a description of your *online collaboration* plan. (*See last paragraph in this section*).
- Will you have meeting agendas and minutes?
- What will be the style of working?
- Will you use daily "stand-ups", or submit a written summaries of your contributions, or something else?
- What is the quality of work each team member expects from themselves and each other?
- When are team members not available (e.g., evenings and Sundays because of family obligations).
- Will you have someone who acts as the project manager (i.e. keeps things on track) for the entire project, or each milestone, or be entirely democratic throughout the project?
- Is there any behaviour you wish to highlight as being expected or unacceptable (i.e., what is the code of conduct for the group?)
- And any other similar things that govern your working relationships.

Use this opportunity to apply your prior knowledge/experience to improve your teamwork, communication, leadership, and organizational skills. For this and all other written work in this course, do pay attention to the basic mechanics of writing, including spelling and grammar (everyone in the team should read over all the documents looking for such errors). You should submit this and other pieces of writing in .md (markdown) format, please take advantage of things like headers, bullet points, italics, etc to make things clearer.



## Understanding the task set-up
rubric={reasoning:15,writing:15}

[ACL template](https://www.overleaf.com/latex/templates/acl-2023-proceedings-template/qjdgcrdwcnwp) must be used for writing group reports. You need to register for an account at Overleaf. Overleaf does not allow sharing projects with multiple users unless you have purchased a subscription plan, so I will create the Overleaf document for you. Please name this document with your team name so that we know which team writes which document. 

### Task  
Create a section called `Task description` in your progress report and answer the following questions.
- What is the task?
- Why is the task important?
- What does this task try to achieve?

### Data
What is the input to the model? How should the output look like?
 - Please paste a few samples of inputs and outputs to a file `data_inspection.md`. The file should be uploaded to the folder `milestone1`.
 - Please create a `data_inspection.py` or `data_inspection.ipynb` file, in which you should include a few lines of code that loads data samples from the offical data.
 - In your progress report, please create a new section called `Data`, in which you should provide:
  - a high-level verbal description of your data;
  - descriptive statistics of the official dataset. This should be a table that includes at least the following columns: `Split` (train or dev or test), `Number of samples`, `Mean Length`, `Max length`, `Min length`. You can include more information in addition to these columns.



### The evaluation metrics
Please create a new section `Evaluation metrics` in your Overleaf progress report. In this section, you should answer the following questions for **each official evaluation metric**. Questions:
 - What is the definition of this metric?
 - What does this evaluation metric measure?
 - How is this metric calculated?

### The background
You must read and summarize *n-1* research papers related to this task, where *n is the number of team members*. In the first week, these research papers must be selected from the references of the official task page. In the upcoming weeks, you need to find these papers on your own. Note that the papers you found must be directly relevant to the task, that is, the membership inference attack. Irrelevant papers, such as papers about fake-news detection, or Qwen3 technical report, etc, will not be graded. Summarize these research papers in your progress report. Use a paragraph for each paper. You need to answer the following questions:
 - What is the main task this paper tries to solve?
 - What is the dataset used in this paper?
 - What is the model used in this paper?
 - What is the main contribution of this paper?
 - 
**Please also cite these research papers properly.** The ACL template provides instructions about how to cite papers.

### Your first submission
Please create a section called `Week 1 Baseline` in your progress report. 
You should propose a very simple baseline method to predict the membership of the test data. Implement this baseline. Then submit your predictions to the Kaggle competition. In Week 1, your ranking in the leaderboard will not be graded. 
Describe what you did. You must include the following information.
 - How did you build your baseline?
 - What is the motivation behind this baseline?
 - What hardware (CPUs, GPUs, or memory, etc) did you use?
 - How did it work?

### Contributions
Please create a section called `Contributions` and a subsection `Milestone1`. Here each team member must write down their individual contributions to milestone 1. In addition to describing what you have done, you also need to specify the percentage of contributions in this week. **Within a team, the percentage of individual contributions must sum to 1.**

# chatcps


download the modules from https://drive.google.com/drive/folders/1HlZQkcllYH23xcgiLd7x4nSbFxmqfank?usp=sharing

file groundtruth.txt is a groundtruth for modules. 


# Instructions

## Step 1:

Install the llm-evaluation-harness repository:

  https://github.com/EleutherAI/lm-evaluation-harness

## Step 2:

  Create a new task folder in llm-evaluation-harness/lm-eval/tasks called sw and make a copy of tasks/sw from this repository there.
  
  Create a folder for datasets with train and test subfolders and copy the name of subfolders in task/sw/function_ds and task/sw/module_ds as gen_kwargs in _split_generators function.

  Create a new folder to keep the results of executions (cma_results for now)

## Step 3:

  Run the test experiment:
  
    `time ./run_test_script.sh 
  

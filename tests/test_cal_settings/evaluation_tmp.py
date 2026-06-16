import os
import pandas as pd

results_f1 = []

# Get path to the results of experiments
current_dir = os.path.dirname(os.path.abspath(__file__))
all_exp_dir = os.path.join(current_dir, 'experiments_new')

for exp_dir in sorted(os.listdir(all_exp_dir)):
    if not exp_dir.startswith("."):
        exp_path = os.path.join(all_exp_dir, exp_dir, 'output')
        for run in range(1, 5):
            # Get path to the current run directory
            run_path = os.path.join(exp_path, f'run_{run}')

            # Save experiment name and run number for further use
            if "logits_nomerge" in exp_dir:
                row_f1 = ['sensation_300_lora_uncertainty_logitsnomerge', run]
            elif "logits_merge" in exp_dir:
                row_f1 = ['sensation_300_lora_uncertainty_logitsmerge', run]
            elif "embs_nomerge_smallbatch" in exp_dir:
                row_f1 = ['sensation_300_lora_uncertainty_embsnomerge4batch', run]
            elif "embs_nomerge_lessepochs" in exp_dir:
                row_f1 = ['sensation_300_lora_uncertainty_embsnomerge10epochs', run]
            elif "diversity" in exp_dir:
                row_f1 = ['sensation_300_lora_diversity_embsnomergediversity', run]
            elif "embs_nomerge" in exp_dir:
                row_f1 = ['sensation_300_lora_uncertainty_embsnomerge', run]

            # Loop for each directory myadapter_i for continual active learning iteration i
            for i in range(11):
                adapter_path = os.path.join(run_path, f'peft_module_{i}', 'classification_report.csv')
                classification_report = pd.read_csv(adapter_path)

                # Get metrics from the classification report
                f1 = classification_report.iloc[3, 3]

                # Round the values and combine with previously stored values
                row_f1.append(round(float(f1), 3))

            # Add results to corresponding arrays
            results_f1.append(row_f1)

df_results_f1 = pd.DataFrame(results_f1, columns=['experiment', 'run'] + [f'f1_{i}' for i in range(11)])

df_results_f1.to_csv('results_new.csv', index=False)
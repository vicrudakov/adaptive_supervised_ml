import os
import pandas as pd

# Empty arrays for baselines
baseline_acc = []
baseline_prec = []
baseline_rec = []
baseline_f1 = []

# Empty arrays for continual active learning results
results_acc = []
results_prec = []
results_rec = []
results_f1 = []

# Get path to the results of experiments
current_dir = os.path.dirname(os.path.abspath(__file__))
all_exp_dir = os.path.join(current_dir, '..', 'training', 'experiments')

# Loop for all directories in experiments directory
for exp_dir in sorted(os.listdir(all_exp_dir)):
    # Apply the following code only for directories with results (name starting with 1000 or 2000)
    if (exp_dir.startswith("1000") or exp_dir.startswith("2000")):
        exp_path = os.path.join(all_exp_dir, exp_dir, 'output')

        # Directories with baselines (name ending with _baseline)
        if exp_dir.endswith('_baseline'):
            # Directories with baselines have only directory myadapter_0 with classification_report.csv
            adapter_path = os.path.join(exp_path, 'myadapter_0', 'classification_report.csv')
            classification_report = pd.read_csv(adapter_path)

            # Get metrics from the classification report
            acc = classification_report.iloc[2, 1]
            prec = classification_report.iloc[3, 1]
            rec = classification_report.iloc[3, 2]
            f1 = classification_report.iloc[3, 3]

            # Round the values and combine with experiment name
            row_acc = [exp_dir, round(float(acc), 3)]
            row_prec = [exp_dir, round(float(prec), 3)]
            row_rec = [exp_dir, round(float(rec), 3)]
            row_f1 = [exp_dir, round(float(f1), 3)]

            # Add results to corresponding arrays
            baseline_acc.append(row_acc)
            baseline_prec.append(row_prec)
            baseline_rec.append(row_rec)
            baseline_f1.append(row_f1)

        # Directories with continual active learning results
        else:
            # Save experiment name for further use
            row_acc = [exp_dir]
            row_prec = [exp_dir]
            row_rec = [exp_dir]
            row_f1 = [exp_dir]

            # Loop for each directory myadapter_i for continual active learning iteration i
            for i in range(11):
                adapter_path = os.path.join(exp_path, f'myadapter_{i}', 'classification_report.csv')
                classification_report = pd.read_csv(adapter_path)

                # Get metrics from the classification report
                acc = classification_report.iloc[2, 1]
                prec = classification_report.iloc[3, 1]
                rec = classification_report.iloc[3, 2]
                f1 = classification_report.iloc[3, 3]

                # Round the values and combine with previously stored values
                row_acc.append(round(float(acc), 3))
                row_prec.append(round(float(prec), 3))
                row_rec.append(round(float(rec), 3))
                row_f1.append(round(float(f1), 3))

            # Add results to corresponding arrays
            results_acc.append(row_acc)
            results_prec.append(row_prec)
            results_rec.append(row_rec)
            results_f1.append(row_f1)

# Dataframes for baselines
df_baseline_acc = pd.DataFrame(baseline_acc, columns=['experiment', 'acc'])
df_baseline_prec = pd.DataFrame(baseline_prec, columns=['experiment', 'prec'])
df_baseline_rec = pd.DataFrame(baseline_rec, columns=['experiment', 'rec'])
df_baseline_f1 = pd.DataFrame(baseline_f1, columns=['experiment', 'f1'])

# Dataframes for continual active learning results
df_results_acc = pd.DataFrame(results_acc, columns=['experiment'] + [f'acc_{i}' for i in range(11)])
df_results_prec = pd.DataFrame(results_prec, columns=['experiment'] + [f'prec_{i}' for i in range(11)])
df_results_rec = pd.DataFrame(results_rec, columns=['experiment'] + [f'rec_{i}' for i in range(11)])
df_results_f1 = pd.DataFrame(results_f1, columns=['experiment'] + [f'f1_{i}' for i in range(11)])

# Saving dataframes for baselines
df_baseline_acc.to_csv('results/baseline/baseline_acc.csv', index=False)
df_baseline_prec.to_csv('results/baseline/baseline_prec.csv', index=False)
df_baseline_rec.to_csv('results/baseline/baseline_rec.csv', index=False)
df_baseline_f1.to_csv('results/baseline/baseline_f1.csv', index=False)

# Saving dataframes for continual active learning results
df_results_acc.to_csv('results/continual_active_learning/results_acc.csv', index=False)
df_results_prec.to_csv('results/continual_active_learning/results_prec.csv', index=False)
df_results_rec.to_csv('results/continual_active_learning/results_rec.csv', index=False)
df_results_f1.to_csv('results/continual_active_learning/results_f1.csv', index=False)


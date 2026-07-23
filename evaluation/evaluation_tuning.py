import os
import pandas as pd
import json

def summarise_results(dataset_name, tuning_step):
    # Empty arrays for continual active learning results
    results_acc = []
    results_prec = []
    results_rec = []
    results_f1 = []

    # Get path to the results of experiments
    current_dir = os.path.dirname(os.path.abspath(__file__))
    all_exp_dir = os.path.join(current_dir, '..', 'training', f'tuning_{tuning_step}')

    # Loop for all directories in experiments directory
    for exp_dir in sorted(os.listdir(all_exp_dir)):
        # Apply the following code only for directories with results
        if (exp_dir.startswith(f'tuning_{dataset_name}')):
            # Get path to the current run directory
            run_path = os.path.join(all_exp_dir, exp_dir, 'output', 'run_1')

            # Save experiment name for further use
            row_acc = [exp_dir]
            row_prec = [exp_dir]
            row_rec = [exp_dir]
            row_f1 = [exp_dir]

            # Loop for each directory peft_module_i for continual active learning iteration i
            for i in range(10):
                peft_module_classification_report_path = os.path.join(run_path, f'peft_module_{i}', 'classification_report.csv')
                classification_report = pd.read_csv(peft_module_classification_report_path, index_col=0)

                # Get metrics from the classification report
                acc = classification_report.loc['accuracy', 'f1-score']
                prec = classification_report.loc['macro avg', 'precision']
                rec = classification_report.loc['macro avg', 'recall']
                f1 = classification_report.loc['macro avg', 'f1-score']

                # Round the values and combine with previously stored values
                row_acc.append(round(float(acc), 5))
                row_prec.append(round(float(prec), 5))
                row_rec.append(round(float(rec), 5))
                row_f1.append(round(float(f1), 5))

            # Add results to corresponding arrays
            results_acc.append(row_acc)
            results_prec.append(row_prec)
            results_rec.append(row_rec)
            results_f1.append(row_f1)

    # Dataframes for continual active learning results
    df_results_acc = pd.DataFrame(results_acc, columns=['experiment'] + [f'acc_{i}' for i in range(10)])
    df_results_prec = pd.DataFrame(results_prec, columns=['experiment'] + [f'prec_{i}' for i in range(10)])
    df_results_rec = pd.DataFrame(results_rec, columns=['experiment'] + [f'rec_{i}' for i in range(10)])
    df_results_f1 = pd.DataFrame(results_f1, columns=['experiment'] + [f'f1_{i}' for i in range(10)])

    # Saving dataframes for continual active learning results
    df_results_acc.to_csv(f'results_tuning/tuning_{tuning_step}/{dataset_name}/results_acc.csv', index=False)
    df_results_prec.to_csv(f'results_tuning/tuning_{tuning_step}/{dataset_name}/results_prec.csv', index=False)
    df_results_rec.to_csv(f'results_tuning/tuning_{tuning_step}/{dataset_name}/results_rec.csv', index=False)
    df_results_f1.to_csv(f'results_tuning/tuning_{tuning_step}/{dataset_name}/results_f1.csv', index=False)

def summarise_time(dataset_name, tuning_step):
    # Empty arrays for time
    time_selection = []
    time_training = []
    time_test = []

    # Get path to the results of experiments
    current_dir = os.path.dirname(os.path.abspath(__file__))
    all_exp_dir = os.path.join(current_dir, '..', 'training', f'tuning_{tuning_step}')

    # Loop for all directories in experiments directory
    for exp_dir in sorted(os.listdir(all_exp_dir)):
        # Apply the following code only for directories with results
        if (exp_dir.startswith(f'tuning_{dataset_name}')):
            # Get path to the current run directory
            run_path = os.path.join(all_exp_dir, exp_dir, 'output', 'run_1')

            # Save experiment name for further use
            row_selection = [exp_dir]
            row_training = [exp_dir]
            row_test = [exp_dir]

            # Loop for each directory peft_module_i for continual active learning iteration i
            for i in range(10):
                peft_module_time_path = os.path.join(run_path, f'peft_module_{i}', 'time.json')
                with open(peft_module_time_path, 'r') as file:
                    time = json.load(file)

                # Get time information
                selection = time['time_selection']
                training = time['time_training']
                test = time['time_test']

                # Round the values and combine with previously stored values
                row_selection.append(round(float(selection)))
                row_training.append(round(float(training)))
                row_test.append(round(float(test)))

            # Add times to corresponding arrays
            time_selection.append(row_selection)
            time_training.append(row_training)
            time_test.append(row_test)

    # Dataframes for time
    df_time_selection = pd.DataFrame(time_selection, columns=['experiment'] + [f'time_{i}' for i in range(10)])
    df_time_training = pd.DataFrame(time_training, columns=['experiment'] + [f'time_{i}' for i in range(10)])
    df_time_test = pd.DataFrame(time_test, columns=['experiment'] + [f'time_{i}' for i in range(10)])

    # Saving dataframes for time
    df_time_selection.to_csv(f'time_tuning/tuning_{tuning_step}/{dataset_name}/time_selection.csv', index=False)
    df_time_training.to_csv(f'time_tuning/tuning_{tuning_step}/{dataset_name}/time_training.csv', index=False)
    df_time_test.to_csv(f'time_tuning/tuning_{tuning_step}/{dataset_name}/time_test.csv', index=False)

# summarise_results("agnews", tuning_step=1)
# summarise_results("sensation", tuning_step=1)
# summarise_results("trec", tuning_step=1)
#
# summarise_time("agnews", tuning_step=1)
# summarise_time("sensation", tuning_step=1)
# summarise_time("trec", tuning_step=1)

# summarise_results("agnews", tuning_step=2)
# summarise_results("sensation", tuning_step=2)
# summarise_results("trec", tuning_step=2)
#
# summarise_time("agnews", tuning_step=2)
# summarise_time("sensation", tuning_step=2)
# summarise_time("trec", tuning_step=2)
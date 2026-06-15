import pandas as pd
from loguru import logger

# Load dataset (source: https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label; https://huggingface.co/datasets/CogComp/trec)
dataset = pd.DataFrame(
    [[line.strip().split(' ', 1)[0].split(':')[0].lower(), line.strip().split(' ', 1)[1]]
     for line in open('train_5500.label.txt', encoding='latin-1')],
    columns=['label', 'text']
)[['text', 'label']]
# abbr - Abbreviation
# enty - Entity
# desc - Description and abstract concept
# hum - Human being
# loc - Location
# num - Numeric value

# Select random observations from data
test = dataset.groupby('label').sample(n=76, random_state=42).sample(n=452, random_state=42)
train = dataset.drop(test.index)

# Save training and test datasets
train.to_csv("train.csv", index=False, header=False)
test.to_csv("test.csv", index=False, header=False)
logger.debug(f'Prepared data; train size: {len(train)}, test size: {len(test)}')

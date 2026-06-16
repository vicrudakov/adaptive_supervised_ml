import pandas as pd
from loguru import logger

# Load datasets (source: https://huggingface.co/datasets/fancyzhx/ag_news)
test = pd.read_parquet('test-00000-of-00001.parquet', engine='fastparquet')
train = pd.read_parquet('train-00000-of-00001.parquet', engine='fastparquet')

# Select random observations from data
test = test.groupby('label').sample(n=150, random_state=42).reset_index()
train = train.groupby('label').sample(n=1500, random_state=42).reset_index()

# Select variables
test = test[['text', 'label']]
train = train[['text', 'label']]

# Map labels to new values
mapping = {
    0: "world", # World
    1: "sports", # Sports
    2: "business", # Business
    3: "sci_tech" # Science and Technologies
}
test['label'] = [mapping[value] for value in test['label'].tolist()]
train['label'] = [mapping[value] for value in train['label'].tolist()]

# Save training and test datasets
train.to_csv("train.csv", index=False, header=False)
test.to_csv("test.csv", index=False, header=False)
logger.debug(f'Prepared data; train size: {len(train)}, test size: {len(test)}')
import pandas as pd
from loguru import logger

# Load datasets (source: https://huggingface.co/datasets/fancyzhx/ag_news)
test = pd.read_parquet('test-00000-of-00001.parquet', engine='fastparquet')
train = pd.read_parquet('train-00000-of-00001.parquet', engine='fastparquet')

# Select the classes World and Sci/Tech
test = test[test['label'].isin([0, 3])]
train = train[train['label'].isin([0, 3])]

# Select random observations from data
test = test.groupby('label').sample(n=209, random_state=42).reset_index()
train = train.groupby('label').sample(n=2000, random_state=42).reset_index()

# Map labels to new values
mapping = {
    0: "world",
    3: "sci_tech"
}
test['label'] = [mapping[value] for value in test['label'].tolist()]
train['label'] = [mapping[value] for value in train['label'].tolist()]

# Save training and test datasets
train.to_csv("train.csv", index=False, header=False)
test.to_csv("test.csv", index=False, header=False)
logger.debug(f'Prepared data; train size: {len(train)}, test size: {len(test)}')
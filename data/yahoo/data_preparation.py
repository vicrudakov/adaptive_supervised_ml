import pandas as pd
from loguru import logger

# Load datasets (source: https://huggingface.co/datasets/community-datasets/yahoo_answers_topics)
test = pd.read_parquet('test-00000-of-00001.parquet', engine='fastparquet')
train_1 = pd.read_parquet('train-00000-of-00002.parquet', engine='fastparquet')
train_2 = pd.read_parquet('train-00001-of-00002.parquet', engine='fastparquet')
train = pd.concat([train_1, train_2])

# Select the classes Society & Culture and Science & Mathematics
test = test[test['topic'].isin([0, 1])]
train = train[train['topic'].isin([0, 1])]

# Select random observations from data
test = test.groupby('topic').sample(n=209, random_state=42).reset_index()
train = train.groupby('topic').sample(n=2000, random_state=42).reset_index()

# Combine text variables into one
test['text'] = (
    test['question_title'].fillna('') + ' ' +
    test['question_content'].fillna('') + ' ' +
    test['best_answer'].fillna('')
)
train['text'] = (
    train['question_title'].fillna('') + ' ' +
    train['question_content'].fillna('') + ' ' +
    train['best_answer'].fillna('')
)

# Select variables
test = test[['text', 'topic']]
train = train[['text', 'topic']]

# Map labels to new values
mapping = {
    0: "society_culture",
    1: "science_mathematics"
}
test['topic'] = [mapping[value] for value in test['topic'].tolist()]
train['topic'] = [mapping[value] for value in train['topic'].tolist()]

# Save training and test datasets
train.to_csv("train.csv", index=False, header=False)
test.to_csv("test.csv", index=False, header=False)
logger.debug(f'Prepared data; train size: {len(train)}, test size: {len(test)}')

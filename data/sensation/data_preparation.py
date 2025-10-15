import pandas as pd
from loguru import logger

# Load dataset and map labels to new values
data = pd.read_csv("sensation.csv")
mapping = {
    0: "neutral",
    1: "sensationalistisch",
    2: "sensationalistisch",
    3: "neutral"
}
data['label'] = [mapping[value] for value in data['label'].tolist()]

# Get unique documents-year pairs and assign them to 3 year bins
documents = data[['document', 'year']].drop_duplicates()
documents['year_bin'] = pd.qcut(documents['year'], q=3, labels=False)

# Calculate document lengths and assign them to 3 length bins
document_lengths = data['document'].value_counts().rename('document_length')
documents = documents.merge(document_lengths, left_on='document', right_index=True)
documents['document_length_bin'] = pd.qcut(documents['document_length'], q=3, labels=False)

# Sample test documents with stratification by year and length bins
test_documents = (documents.groupby(['year_bin', 'document_length_bin'], group_keys=False)
                  .apply(lambda group: group.sample(n=5, random_state=54) if len(group) >= 5 else group,
                         include_groups=False))

# Extract training and test datasets from the original dataset
test_full = data[data['document'].isin(test_documents['document'])]
test = test_full[['text', 'label']]
train_full = data.drop(test.index)
train = train_full[['text', 'label']]

# Save training and test datasets
train.to_csv("train.csv", index=False, header=False)
test.to_csv("test.csv", index=False, header=False)
logger.debug(f'Prepared data; train size: {len(train)}, test size: {len(test)}')
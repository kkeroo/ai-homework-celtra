import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
from openai import OpenAI
import tiktoken
import re

api_key = "API_KEY"
client = OpenAI(api_key=api_key)

# Load the dataset
df = pd.read_csv('resources/dataset_with_descriptions.csv', sep=';')

def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )

    embs = [d.embedding for d in response.data]

    return embs

# Keep only first 3 sentences in each description
def get_first_three_sentences(text):
    if not isinstance(text, str):
        return ""
    # Split by sentence endings (., !, ?) followed by space or end of string
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Take first 3 sentences or all if less than 3
    first_three = sentences[:3]
    # Join back together with spaces
    return " ".join(first_three)

# Calculate tokens and truncate long descriptions
def num_tokens_from_string(string, encoding_name="cl100k_base"):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


# Apply the function to keep only first 3 sentences
print("Truncating descriptions to first 3 sentences...")
df['DESCRIPTION'] = df['DESCRIPTION'].fillna("").apply(get_first_three_sentences)


# Calculate tokens for each description
print("Analyzing token counts...")
df['token_count'] = df['DESCRIPTION'].fillna("").apply(num_tokens_from_string)

# Report token statistics
print(f"Max tokens: {df['token_count'].max()}")
print(f"Mean tokens: {df['token_count'].mean():.2f}")

# After loading the dataset
# Count error descriptions
no_wiki_count = df[df['DESCRIPTION'] == "No Wikipedia page found"].shape[0]
disambiguation_count = df[df['DESCRIPTION'] == "Disambiguation page encountered, description not retrieved"].shape[0]
error_count = df[df['DESCRIPTION'].str.startswith("Error: ")].shape[0]

# Print results
print(f"No Wikipedia page found: {no_wiki_count}")
print(f"Disambiguation page encountered: {disambiguation_count}")
print(f"Error messages: {error_count}")
print(f"Total error descriptions: {no_wiki_count + disambiguation_count + error_count}")
print(f"Total records: {df.shape[0]}")
print(f"Percentage with errors: {((no_wiki_count + disambiguation_count + error_count) / df.shape[0]) * 100:.2f}%")

# Drop rows with missing descriptions
df = df.dropna(subset=['DESCRIPTION'])

# Features and labels
X = df['DESCRIPTION']
y = df['CATEGORY']

# Split dataset (train 80%, test 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_vec = []
X_test_vec = []

batch_size = 500

for i in tqdm(range(0, len(X_train), batch_size)):
    X_train_vec.extend(get_embedding(X_train[i:i+batch_size]))

for i in tqdm(range(0, len(X_test), batch_size)):
    X_test_vec.extend(get_embedding(X_test[i:i+batch_size]))

# Train classifier (Logistic Regression)
classifier = LogisticRegression(max_iter=500, random_state=42)
classifier.fit(X_train_vec, y_train)

# Predictions on test set
y_pred = classifier.predict(X_test_vec)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Classification Accuracy: {accuracy:.4f}")

# Evaluate the model
print("Classification Report:\n", classification_report(y_test, y_pred))

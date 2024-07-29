from datasets import load_dataset

# Load the dataset
dataset = load_dataset('projecte-aina/catalan_general_crawling', trust_remote_code=True)

# Extract the 'train' split and preprocess
corpus_text = " ".join(dataset['train']['text'])

# Save the corpus to a file in data folder

with open("data/corpus.txt", "w") as f:
    f.write(corpus_text)
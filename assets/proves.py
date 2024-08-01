from datasets import load_dataset

# Load the C4 dataset in Catalan
dataset = load_dataset("allenai/c4", "multilingual", data_files="multilingual/c4-ca.*.json.gz", streaming=True)

# Function to show fragments of the dataset
def show_fragments(dataset, num_fragments=15):
    iterator = iter(dataset['train'])
    for _ in range(num_fragments):
        try:
            fragment = next(iterator)
            print(fragment['text'][:500])  # Show the first 500 characters of the fragment
            print('-' * 80)
        except StopIteration:
            print("No hay más fragmentos disponibles.")
            break

show_fragments(dataset)

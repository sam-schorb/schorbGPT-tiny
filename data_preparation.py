# data_preparation.py

import os
import glob
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
import logging
from tqdm import tqdm

def inspect_and_log_dataset(dataset, tokenizer, filename="dataset_inspection.txt"):
    """Inspect and log detailed information about the dataset using sampling for better performance"""
    with open(filename, 'w') as f:
        # 1. Basic Dataset Information
        f.write("=== Dataset Inspection Report ===\n\n")
        f.write(f"Dataset size: {len(dataset):,} samples\n")
        f.write(f"Dataset format: {type(dataset)}\n")
        f.write(f"Features: {dataset.features}\n\n")

        # 2. Sample Data Inspection (unchanged, already fast)
        f.write("=== Sample Data (First 3 Examples) ===\n\n")
        for i in range(min(3, len(dataset))):
            example = dataset[i]
            f.write(f"\nExample {i+1}:\n")
            f.write(f"Input IDs shape: {len(example['input_ids'])}\n")
            f.write(f"Attention Mask shape: {len(example['attention_mask'])}\n")
            f.write(f"Input IDs type: {type(example['input_ids'])}\n")
            
            decoded_text = tokenizer.decode(example['input_ids'][:50])
            f.write(f"First 50 tokens decoded: {decoded_text}...\n")
            
            f.write("\nDetailed view of first 10 tokens:\n")
            for j, token_id in enumerate(example['input_ids'][:10]):
                token = tokenizer.decode([token_id])
                f.write(f"Position {j}: ID={token_id}, Token='{token}'\n")

        # 3. Statistical Information - Now using sampling
        f.write("\n=== Statistical Information (Based on Sample) ===\n")
        # Take a random sample of 10000 examples for statistics
        sample_size = min(10000, len(dataset))
        sample_indices = np.random.choice(len(dataset), sample_size, replace=False)
        
        lengths = [len(dataset[i]['input_ids']) for i in sample_indices]
        f.write(f"Statistics based on random sample of {sample_size:,} sequences\n")
        f.write(f"Estimated average sequence length: {sum(lengths)/len(lengths):.2f}\n")
        f.write(f"Sample max sequence length: {max(lengths)}\n")
        f.write(f"Sample min sequence length: {min(lengths)}\n")

        # Add length distribution information
        percentiles = np.percentile(lengths, [25, 50, 75, 90, 95, 99])
        f.write("\nSequence Length Distribution:\n")
        f.write(f"25th percentile: {percentiles[0]:.0f}\n")
        f.write(f"Median: {percentiles[1]:.0f}\n")
        f.write(f"75th percentile: {percentiles[2]:.0f}\n")
        f.write(f"90th percentile: {percentiles[3]:.0f}\n")
        f.write(f"95th percentile: {percentiles[4]:.0f}\n")
        f.write(f"99th percentile: {percentiles[5]:.0f}\n")

        # 4. Tensor Information (unchanged, already fast)
        f.write("\n=== Tensor Information ===\n")
        sample_batch = dataset[:4]
        f.write(f"Batch structure: {type(sample_batch)}\n")
        for key in sample_batch:
            f.write(f"Key '{key}' shape: {len(sample_batch[key])} examples\n")

def main():
    # Set up logging to console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    logging.info('Starting data preparation...')

    # Path to your Parquet files
    parquet_dir = os.path.expanduser('~/fineweb-edu-10B')  # Use absolute path
    if not os.path.exists(parquet_dir):
        raise FileNotFoundError(f"Directory not found: {parquet_dir}")
    
    parquet_files = glob.glob(os.path.join(parquet_dir, '*.parquet'))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {parquet_dir}")
    
    parquet_files.sort()  # Ensure consistent order

    # Only use the first 10% of the files
    num_files_to_use = max(1, int(len(parquet_files) * 0.1))
    parquet_files = parquet_files[:num_files_to_use]
    
    logging.info(f'Found {len(parquet_files)} parquet files total')
    logging.info(f'Using first {num_files_to_use} Parquet files for initial training')
    for file in parquet_files:
        logging.info(f'  - {os.path.basename(file)}')

    # Columns to read
    columns_to_read = ['text', 'language', 'language_score', 'score']
    logging.info(f'Reading columns: {", ".join(columns_to_read)}')

    # Read Parquet files and concatenate into a single dataset
    dataframes = []
    total_rows = 0
    logging.info('Reading Parquet files...')
    for file in tqdm(parquet_files, desc='Reading Parquet files'):
        df = pd.read_parquet(file, columns=columns_to_read)
        total_rows += len(df)
        logging.info(f'Read {len(df):,} rows from {os.path.basename(file)}')
        dataframes.append(df)
    full_df = pd.concat(dataframes, ignore_index=True)
    logging.info(f'Finished reading {total_rows:,} total rows from Parquet files')

    # Filter for English texts with high language scores
    logging.info('Filtering dataset...')
    initial_size = len(full_df)
    filtered_df = full_df[
        (full_df['language'] == 'en') &
        (full_df['language_score'] > 0.9) &
        (full_df['score'] >= 3)  # Adjust the threshold as needed
    ]
    retained_percentage = (len(filtered_df) / initial_size) * 100
    logging.info(f'Filtered dataset from {initial_size:,} to {len(filtered_df):,} samples ({retained_percentage:.2f}% retained)')
    logging.info(f'Filtering criteria applied:')
    logging.info(f'  - Language: English')
    logging.info(f'  - Language score > 0.9')
    logging.info(f'  - Quality score >= 3')

    # Drop unnecessary columns
    texts = filtered_df['text'].tolist()

    # Initialize the tokenizer
    logging.info('Initializing tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    # Set padding token to be the EOS token
    tokenizer.pad_token = tokenizer.eos_token
    logging.info(f'Vocabulary size: {len(tokenizer)}')
    logging.info(f'Padding token set to: {tokenizer.pad_token}')

    # Tokenize the texts
    logging.info('Tokenizing texts...')
    def batch_tokenize(batch):
        return tokenizer(
            batch['text'],
            truncation=True,
            max_length=512,
            padding='max_length'
        )

    # Create a Hugging Face Dataset
    dataset = Dataset.from_pandas(filtered_df[['text']])

    # Tokenize the dataset with a progress bar
    tokenized_dataset = dataset.map(
        batch_tokenize,
        batched=True,
        remove_columns=['text'],
        desc='Tokenizing',
    )
    logging.info('Tokenization complete.')
    logging.info(f'Dataset contains {len(tokenized_dataset):,} sequences')

    # Split the dataset into training and validation sets
    logging.info('Splitting dataset into training and validation sets...')
    train_test = tokenized_dataset.train_test_split(test_size=0.1)
    train_dataset = train_test['train']
    valid_dataset = train_test['test']
    logging.info(f'Training set size: {len(train_dataset):,} samples')
    logging.info(f'Validation set size: {len(valid_dataset):,} samples')


    # Add inspection before saving
    logging.info('Generating dataset inspection report...')
    inspect_and_log_dataset(train_dataset, tokenizer, "dataset_inspection.txt")
    
    # Continue with saving datasets
    logging.info('Saving tokenized datasets...')
    train_dataset.save_to_disk('train_dataset')
    valid_dataset.save_to_disk('valid_dataset')
    logging.info('Data preparation complete.')

    # Print summary statistics
    logging.info('\nFinal Summary:')
    logging.info(f'Initial number of files processed: {num_files_to_use}')
    logging.info(f'Total rows read: {total_rows:,}')
    logging.info(f'Rows after filtering: {len(filtered_df):,}')
    logging.info(f'Final training samples: {len(train_dataset):,}')
    logging.info(f'Final validation samples: {len(valid_dataset):,}')
    logging.info(f'Data saved to: {os.path.abspath("train_dataset")} and {os.path.abspath("valid_dataset")}')

if __name__ == '__main__':
    main()
# generate_text.py
import torch
from transformers import GPT2LMHeadModel, AutoTokenizer
import logging
import os

def main():
    # Set up logging to console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    logging.info('Starting text generation...')

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    # Path to the trained model
    model_dir = 'model_epoch_1'
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # Load the tokenizer and model from the saved directory
    logging.info('Loading tokenizer and model...')
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = GPT2LMHeadModel.from_pretrained(model_dir)  # Load directly from saved directory
        
        # Set the padding token
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        
        logging.info(f'Model loaded from: {model_dir}')
    except Exception as e:
        logging.error(f'Error loading model: {str(e)}')
        raise

    model.to(device)
    model.eval()
    logging.info('Model loaded successfully.')

    # Input loop
    while True:
        try:
            # Prompt for text generation
            prompt = input("\nEnter a prompt (or 'quit' to exit): ")
            if prompt.lower() == 'quit':
                break
                
            logging.info(f'User prompt: {prompt}')

            # Encode the prompt
            inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            # Generate text
            max_new_tokens = 200  # Generate 200 new tokens
            logging.info('Starting text generation...')
            
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            logging.info('Text generation complete.')

            # Decode the generated text
            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            print("\nGenerated Text:")
            print("-" * 80)
            print(generated_text)
            print("-" * 80)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            logging.error(f'Error during generation: {str(e)}')
            continue

if __name__ == '__main__':
    main()
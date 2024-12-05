# train.py
import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import GPT2LMHeadModel, AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_from_disk
import logging
from tqdm import tqdm
import argparse
import psutil
import GPUtil
from datetime import datetime
from typing import Dict, List

def setup_logging(local_rank):
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Set up logging to both file and console
    handlers = [
        logging.FileHandler(f'logs/train_rank_{local_rank}.log'),
        logging.StreamHandler()
    ]
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [Rank {local_rank}] %(message)s',
        handlers=handlers
    )

def parse_args():
    parser = argparse.ArgumentParser(description='Distributed Training')
    parser.add_argument('--local_rank', type=int, help='Local process rank.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size per GPU')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of data loading workers per GPU')
    return parser.parse_args()

def log_gpu_memory_usage(local_rank, device):
    gpu = GPUtil.getGPUs()[device.index]
    logging.info(f'GPU {local_rank} Memory: {gpu.memoryUsed:.2f}MB / {gpu.memoryTotal:.2f}MB ({gpu.memoryUtil*100:.1f}%)')

def collate_fn(batch: List[Dict]) -> Dict:
    """Convert batch of samples to tensors."""
    return {
        'input_ids': torch.tensor([sample['input_ids'] for sample in batch], dtype=torch.long),
        'attention_mask': torch.tensor([sample['attention_mask'] for sample in batch], dtype=torch.long)
    }

def main():
    args = parse_args()
    start_time = datetime.now()
    
    # Initialize the process group first
    try:
        dist.init_process_group(backend='nccl')
        world_size = dist.get_world_size()
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        setup_logging(local_rank)
    except Exception as e:
        print(f"Failed to initialize distributed training: {str(e)}")
        raise

    logging.info(f'Process {local_rank}/{world_size} initialized on {device}')
    if local_rank == 0:
        logging.info(f'Total GPUs: {world_size}')
        logging.info(f'Per-GPU batch size: {args.batch_size}')
        logging.info(f'Global batch size: {args.batch_size * world_size}')
        logging.info(f'CPU cores available: {psutil.cpu_count()}')
        logging.info(f'RAM available: {psutil.virtual_memory().total / (1024**3):.1f}GB')

    # Load the tokenized datasets
    try:
        logging.info('Loading tokenized datasets...')
        train_dataset = load_from_disk('train_dataset')
        valid_dataset = load_from_disk('valid_dataset')
        logging.info(f'Training samples: {len(train_dataset):,}')
        logging.info(f'Validation samples: {len(valid_dataset):,}')
    except Exception as e:
        logging.error(f'Failed to load datasets: {str(e)}')
        raise

    # Prepare data loaders
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank)
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=local_rank)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2,
        collate_fn=collate_fn
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        sampler=valid_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2,
        collate_fn=collate_fn
    )

    # Model initialization
    logging.info('Initializing model...')
    try:
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        model.resize_token_embeddings(len(tokenizer))
        model.to(device)
        
        # Log model size
        if local_rank == 0:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logging.info(f'Model parameters: {total_params:,} total, {trainable_params:,} trainable')
    except Exception as e:
        logging.error(f'Failed to initialize model: {str(e)}')
        raise

    # Wrap model with DDP
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    # Optimizer and scheduler setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    num_epochs = 1
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=500, num_training_steps=total_steps
    )

    if local_rank == 0:
        logging.info(f'Training config:')
        logging.info(f'- Epochs: {num_epochs}')
        logging.info(f'- Steps per epoch: {len(train_loader):,}')
        logging.info(f'- Total steps: {total_steps:,}')
        logging.info(f'- Learning rate: 5e-5')
        logging.info(f'- Warmup steps: 500')

    # Training loop
    best_loss = float('inf')
    log_interval = 100  # Log every 100 steps
    
    for epoch in range(num_epochs):
        epoch_start_time = datetime.now()
        logging.info(f'Starting epoch {epoch + 1}/{num_epochs}')
        model.train()
        epoch_loss = 0.0
        train_sampler.set_epoch(epoch)

        progress_bar = tqdm(
            train_loader,
            desc=f'Epoch {epoch + 1}/{num_epochs}',
            disable=local_rank != 0,
            position=0,
            leave=True
        )

        for step, batch in enumerate(progress_bar):
            try:
                inputs = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                # Debug logging for first batch
                if step == 0 and local_rank == 0:
                    logging.info(f'First batch shapes:')
                    logging.info(f'input_ids: {inputs.shape}')
                    logging.info(f'attention_mask: {attention_mask.shape}')
                    logging.info(f'input_ids device: {inputs.device}')
                    logging.info(f'First sequence tokens: {tokenizer.decode(inputs[0][:50])}...')

                outputs = model(
                    input_ids=inputs,
                    attention_mask=attention_mask,
                    labels=inputs
                )
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                current_lr = optimizer.param_groups[0]['lr']

                # Update progress bar
                if local_rank == 0:
                    progress_bar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'lr': f'{current_lr:.2e}',
                        'step': f'{step}/{len(train_loader)}'
                    })

                # Periodic logging
                if step % log_interval == 0 and local_rank == 0:
                    avg_loss = epoch_loss / (step + 1)
                    speed = (step + 1) * args.batch_size * world_size / (datetime.now() - epoch_start_time).total_seconds()
                    log_gpu_memory_usage(local_rank, device)
                    logging.info(
                        f'Step: {step}/{len(train_loader)} | '
                        f'Loss: {avg_loss:.4f} | '
                        f'LR: {current_lr:.2e} | '
                        f'Speed: {speed:.1f} samples/sec'
                    )

            except Exception as e:
                logging.error(f'Error during training step {step}: {str(e)}')
                raise

        # End of epoch logging
        avg_epoch_loss = epoch_loss / len(train_loader)
        epoch_time = datetime.now() - epoch_start_time

        # Validation
        model.eval()
        valid_loss = 0.0
        valid_steps = 0
        with torch.no_grad():
            for valid_batch in valid_loader:
                try:
                    inputs = valid_batch['input_ids'].to(device)
                    attention_mask = valid_batch['attention_mask'].to(device)
                    
                    outputs = model(
                        input_ids=inputs,
                        attention_mask=attention_mask,
                        labels=inputs
                    )
                    valid_loss += outputs.loss.item()
                    valid_steps += 1
                except Exception as e:
                    logging.error(f'Error during validation step: {str(e)}')
                    raise

        avg_valid_loss = valid_loss / valid_steps

        if local_rank == 0:
            logging.info(
                f'Epoch {epoch + 1} completed in {epoch_time}. '
                f'Train loss: {avg_epoch_loss:.4f} | '
                f'Valid loss: {avg_valid_loss:.4f}'
            )

        # Save model (only master process)
        if local_rank == 0 and avg_valid_loss < best_loss:
            best_loss = avg_valid_loss
            output_dir = f'model_epoch_{epoch + 1}'
            os.makedirs(output_dir, exist_ok=True)
            model.module.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logging.info(f'Model saved to {output_dir}')

    # Training complete
    if local_rank == 0:
        total_time = datetime.now() - start_time
        logging.info(f'Training completed in {total_time}')
        logging.info(f'Best loss achieved: {best_loss:.4f}')

    dist.destroy_process_group()

if __name__ == '__main__':
    main()
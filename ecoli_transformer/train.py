import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import argparse

from ecoli_transformer.model import CodonEncoder

def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        input_ids, pair_ids, attention_mask, mlm_labels, cai_target, dg_target = [t.to(device) for t in batch]

        optimizer.zero_grad()
        _, loss = model(input_ids=input_ids, pair_ids=pair_ids, attention_mask=attention_mask,
                        mlm_labels=mlm_labels, cai_target=cai_target, dg_target=dg_target)
        
        if loss is not None:
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    return total_loss / len(dataloader)

def main():
    parser = argparse.ArgumentParser(description="Train CodonEncoder model.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the training data .pt file.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    args = parser.parse_args()

    # --- M2 Mac Optimization ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS).")
    else:
        device = torch.device("cpu")
        print("MPS not available. Using CPU.")
    # -------------------------

    data = torch.load(args.data_path)
    
    # Dummy tensors for demonstration if not present in your .pt file
    # In a real scenario, you would have your actual data here
    num_samples = len(data['token_ids'])
    max_len = max(len(t) for t in data['token_ids'])
    
    # Pad sequences to the max length in the batch
    padded_token_ids = torch.nn.utils.rnn.pad_sequence(data['token_ids'], batch_first=True, padding_value=0)
    padded_pair_ids = torch.nn.utils.rnn.pad_sequence(data['pair_ids'], batch_first=True, padding_value=0)
    
    # Create attention masks
    attention_mask = (padded_token_ids != 0)

    dataset = TensorDataset(
        padded_token_ids,
        padded_pair_ids,
        attention_mask,
        padded_token_ids.clone(),  # Using token_ids as mlm_labels for this example
        data['cai'],
        data['mfe']
    )
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = CodonEncoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        avg_loss = train(model, dataloader, optimizer, device)
        print(f"Epoch {epoch+1}/{args.epochs}, Average Loss: {avg_loss:.4f}")

    print("Training complete.")

if __name__ == "__main__":
    main()

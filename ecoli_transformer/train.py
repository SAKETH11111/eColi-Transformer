import argparse
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import r2_score
import numpy as np
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from ecoli_transformer.model import CodonEncoder
from ecoli_transformer.tokenizer import CodonTokenizer

PAD_ID = CodonTokenizer().pad_id

class GeneDataset(Dataset):
    def __init__(self, pt_file, tiny=False):
        data = torch.load(pt_file)
        
        if tiny:
            print("Using --tiny flag: loading only first 500 sequences.")
            data = {k: v[:500] for k, v in data.items()}

        self.input_ids = data['input_ids']
        self.labels = data['labels']
        self.organism_ids = data['organism_id']
        self.cai = data['cai']
        self.mfe = data['mfe']

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx],
            "organism_id": self.organism_ids[idx],
            "cai": self.cai[idx],
            "dg": self.mfe[idx]
        }

def collate_fn(batch):
    input_ids = pad_sequence([item['input_ids'] for item in batch], batch_first=True, padding_value=PAD_ID)
    labels = pad_sequence([item['labels'] for item in batch], batch_first=True, padding_value=-100)
    
    attention_mask = (input_ids != PAD_ID).float()
    
    organism_ids = torch.tensor([item['organism_id'] for item in batch], dtype=torch.long)
    cai = torch.tensor([item['cai'] for item in batch], dtype=torch.float)
    dg = torch.tensor([item['dg'] for item in batch], dtype=torch.float)

    return input_ids, organism_ids, attention_mask, labels, cai, dg

def main():
    parser = argparse.ArgumentParser(description="Train eColi Transformer Model")
    parser.add_argument("--train_pt", type=str, required=True, help="Path to training .pt file")
    parser.add_argument("--val_pt", type=str, required=True, help="Path to validation .pt file")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--save", type=str, default="checkpoints/baseline.pt", help="Path to save best checkpoint")
    parser.add_argument("--cai_weight", type=float, default=0.2, help="Weight for CAI loss")
    parser.add_argument("--dg_weight", type=float, default=0.2, help="Weight for dG loss")
    parser.add_argument("--tiny", action="store_true", help="Train on a tiny subset for testing")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Path to load a checkpoint from, without resuming optimizer state")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--freeze_encoder", action="store_true", help="Freeze the transformer encoder layers")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Number of warmup steps for the scheduler")
    parser.add_argument("--lr_scheduler", type=str, default="linear", choices=["linear", "cosine"], help="LR scheduler type")
    args = parser.parse_args()

    if args.tiny:
        args.epochs = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)

    tokenizer = CodonTokenizer()
    train_dataset = GeneDataset(args.train_pt, tiny=args.tiny)
    val_dataset = GeneDataset(args.val_pt, tiny=args.tiny)

    cai_mean, cai_std = train_dataset.cai.mean(), train_dataset.cai.std()
    mfe_mean, mfe_std = train_dataset.mfe.mean(), train_dataset.mfe.std()
    
    train_dataset.cai = (train_dataset.cai - cai_mean) / cai_std
    train_dataset.mfe = (train_dataset.mfe - mfe_mean) / mfe_std
    val_dataset.cai = (val_dataset.cai - cai_mean) / cai_std
    val_dataset.mfe = (val_dataset.mfe - mfe_mean) / mfe_std

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    tokenizer = CodonTokenizer()
    model = CodonEncoder(vocab_size=tokenizer.vocab_size).to(device)

    if args.load_checkpoint and Path(args.load_checkpoint).exists():
        print(f"Loading model weights from checkpoint: {args.load_checkpoint}")
        checkpoint = torch.load(args.load_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])

    if args.freeze_encoder:
        print("Freezing encoder layers. Only training regression heads.")
        for param in model.token_embedding.parameters():
            param.requires_grad = False
        for param in model.transformer_encoder.parameters():
            param.requires_grad = False
        
        optimizer = AdamW(list(model.cai_head.parameters()) + list(model.dg_head.parameters()), lr=args.lr)
    else:
        optimizer = AdamW(model.parameters(), lr=args.lr)
    
    num_training_steps = args.epochs * len(train_loader)
    
    if args.lr_scheduler == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps - args.warmup_steps)
    else: # linear
        def lr_lambda(current_step):
            if current_step < args.warmup_steps:
                return float(current_step) / float(max(1, args.warmup_steps))
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - args.warmup_steps))
            )
        scheduler = LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler('cuda')

    best_composite_score = -float('inf')
    epochs_no_improve = 0
    start_epoch = 0

    if args.resume and Path(args.resume).exists():
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        if not args.freeze_encoder:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_mlm_acc = checkpoint.get('best_val_mlm_acc', -1)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_train_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for i, batch in enumerate(progress_bar):
            input_ids, organism_ids, attention_mask, labels, cai_target, dg_target = [t.to(device) for t in batch]

            with torch.amp.autocast('cuda'):
                mlm_logits, loss, _, _ = model(
                    input_ids=input_ids,
                    organism_ids=organism_ids,
                    attention_mask=attention_mask,
                    mlm_labels=labels,
                    cai_target=cai_target,
                    dg_target=dg_target,
                    cai_weight=args.cai_weight,
                    dg_weight=args.dg_weight
                )
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.data.clamp_(-1.0, 1.0)

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            if torch.isnan(loss):
                print(f"NaN loss detected at step {i}. Stopping training.")
                break
            total_train_loss += loss.item()

            if (i + 1) % 100 == 0:
                progress_bar.set_postfix({
                    'train_loss': total_train_loss / (i + 1),
                    'grad_norm': grad_norm.item()
                })

        model.eval()
        total_val_loss = 0
        all_mlm_preds = []
        all_mlm_labels = []
        all_cai_preds = []
        all_cai_targets = []
        all_dg_preds = []
        all_dg_targets = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids, organism_ids, attention_mask, labels, cai_target, dg_target = [t.to(device) for t in batch]
                
                with torch.amp.autocast('cuda'):
                    mlm_logits, _, cai_pred, dg_pred = model(input_ids, organism_ids=organism_ids, attention_mask=attention_mask)

                masked_tokens = labels != -100
                all_mlm_preds.append(mlm_logits[masked_tokens].argmax(dim=-1))
                all_mlm_labels.append(labels[masked_tokens])

                all_cai_preds.append(cai_pred.squeeze())
                all_cai_targets.append(cai_target)
                all_dg_preds.append(dg_pred.squeeze())
                all_dg_targets.append(dg_target)

        val_mlm_acc = (torch.cat(all_mlm_preds) == torch.cat(all_mlm_labels)).float().mean().item()
        
        cai_preds_np = torch.cat(all_cai_preds).cpu().numpy()
        cai_targets_np = torch.cat(all_cai_targets).cpu().numpy()
        val_cai_r2 = r2_score(cai_targets_np[~np.isnan(cai_targets_np)], cai_preds_np[~np.isnan(cai_targets_np)])
        
        dg_preds_np = torch.cat(all_dg_preds).cpu().numpy()
        dg_targets_np = torch.cat(all_dg_targets).cpu().numpy()
        val_dg_rmse = np.sqrt(np.mean((dg_preds_np[~np.isnan(dg_targets_np)] - dg_targets_np[~np.isnan(dg_targets_np)])**2))

        max_dg_rmse = 50.0 
        composite_score = val_mlm_acc + val_cai_r2 - (val_dg_rmse / max_dg_rmse)

        print(f"Epoch {epoch}: train_loss {total_train_loss / len(train_loader):.2f} | val_MLM_acc {val_mlm_acc:.2f} | val_CAI_R2 {val_cai_r2:.2f} | val_dG_RMSE {val_dg_rmse:.2f} | Composite_Score {composite_score:.2f}")

        if composite_score > best_composite_score:
            best_composite_score = composite_score
            epochs_no_improve = 0
            print(f"Saved checkpoint: {args.save} (Score: {best_composite_score:.2f})")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_composite_score': best_composite_score,
            }, args.save)
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= args.patience:
            print(f"Early stopping triggered after {args.patience} epochs with no improvement.")
            break

if __name__ == "__main__":
    main()

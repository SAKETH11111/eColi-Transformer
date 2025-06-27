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
        # Remove samples with NaN CAI or MFE to prevent NaN loss
        mask = torch.isfinite(data['cai']) & torch.isfinite(data['mfe'])
        idxs = mask.nonzero(as_tuple=True)[0]
        mask_bool = mask.tolist()
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                data[key] = val[idxs]
            else:
                # filter Python lists (e.g., input_ids, labels, gene_id)
                data[key] = [item for item, m in zip(val, mask_bool) if m]

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
    parser.add_argument("--warmup_ratio", type=float, default=None, help="Warmup ratio of total training steps")
    parser.add_argument("--lr_scheduler", type=str, default="linear", choices=["linear", "cosine"], help="LR scheduler type")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", "mps"], help="Device to use (cpu, cuda, or mps)")
    parser.add_argument("--accum_steps", type=int, default=1, help="Gradient accumulation steps")
    args = parser.parse_args()

    if args.tiny:
        args.epochs = 1

    # Select device: override if specified, else auto-detect
    if args.device:
        # Verify requested device support
        if args.device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available; falling back to CPU.")
            device = torch.device("cpu")
        elif args.device == "mps" and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and torch.backends.mps.is_built()):
            print("Warning: MPS requested but not supported by this PyTorch build; falling back to CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device(args.device)
    else:
        # Auto-detect: CUDA > MPS > CPU
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    print(f"Using device: {device}")
    # Reduce MPS memory fraction to prevent OOM on Mac GPU
    if device.type == "mps":
        try:
            # Use at most 60% of MPS pool
            torch.mps.set_per_process_memory_fraction(0.6, device=device)
            print("Set MPS per-process memory fraction to 0.6")
        except Exception:
            pass

    # Initialize datasets, loaders, etc.
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
    
    warmup_steps = args.warmup_steps
    if args.warmup_ratio is not None:
        warmup_steps = int(num_training_steps * args.warmup_ratio)
        print(f"Using warmup ratio of {args.warmup_ratio}, which corresponds to {warmup_steps} steps.")

    if args.lr_scheduler == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps - warmup_steps)
    else: # linear
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - warmup_steps))
            )
        scheduler = LambdaLR(optimizer, lr_lambda)
    # Use a GradScaler for mixed-precision training if on CUDA
    use_amp = device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

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
        optimizer.zero_grad()
        total_train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for i, batch in enumerate(progress_bar):
            input_ids, organism_ids, attention_mask, labels, cai_target, dg_target = [t.to(device) for t in batch]

            with torch.autocast(device_type=amp_device_type, dtype=torch.float16, enabled=use_amp):
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
            
            # scale loss by accum_steps
            loss = loss / args.accum_steps
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            total_train_loss += loss.item() * args.accum_steps

            # step optimizer after accumulation
            if (i + 1) % args.accum_steps == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            if torch.isnan(loss):
                print(f"NaN loss detected at step {i}. Stopping training.")
                break
            total_train_loss += loss.item()

            if (i + 1) % 100 == 0:
                progress_bar.set_postfix({
                    'train_loss': total_train_loss / (i + 1)
                })

        # handle remaining gradients if not divisible
        if args.accum_steps > 1 and len(train_loader) % args.accum_steps != 0:
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

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
                with torch.autocast(device_type=amp_device_type, dtype=torch.float16, enabled=use_amp):
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

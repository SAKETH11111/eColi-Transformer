import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class CodonEncoder(nn.Module):
    def __init__(self, vocab_size=152, num_organisms=1, pair_vocab_size=4096, hidden_dim=512, nhead=8, num_layers=6, dim_feedforward=2048, max_seq_len=4096):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.organism_embedding = nn.Embedding(num_organisms, hidden_dim)
        self.pair_embedding = nn.Embedding(pair_vocab_size, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))

        encoder_layers = TransformerEncoderLayer(hidden_dim, nhead, dim_feedforward, batch_first=True, norm_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

        self.mlm_head = nn.Linear(hidden_dim, vocab_size)
        self.cai_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.dg_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def resize_token_embeddings(self, new_num_tokens: int):
        """
        Resizes the token embeddings and the MLM head to accommodate a new vocabulary size.
        """
        old_num_tokens, hidden_dim = self.token_embedding.weight.shape
        
        # Create new embedding and MLM head layers
        new_embedding = nn.Embedding(new_num_tokens, hidden_dim)
        new_mlm_head = nn.Linear(hidden_dim, new_num_tokens)
        
        # Copy old weights
        new_embedding.weight.data[:old_num_tokens, :] = self.token_embedding.weight.data
        new_mlm_head.weight.data[:old_num_tokens, :] = self.mlm_head.weight.data
        new_mlm_head.bias.data[:old_num_tokens] = self.mlm_head.bias.data
        
        # Initialize new weights
        new_embedding.weight.data[old_num_tokens:].normal_(mean=0.0, std=0.02)
        new_mlm_head.weight.data[old_num_tokens:].normal_(mean=0.0, std=0.02)
        new_mlm_head.bias.data[old_num_tokens:].zero_()
        
        self.token_embedding = new_embedding
        self.mlm_head = new_mlm_head

    def forward(self, input_ids, organism_ids=None, pair_ids=None, attention_mask=None, mlm_labels=None, cai_target=None, dg_target=None, cai_weight=0.2, dg_weight=0.2):
        token_emb = self.token_embedding(input_ids)
        
        if organism_ids is not None:
            org_emb = self.organism_embedding(organism_ids)
            x = token_emb + org_emb.unsqueeze(1)
        else:
            x = token_emb

        if pair_ids is not None:
            padded_pair_ids = torch.full_like(input_ids, 0)
            padded_pair_ids[:, :pair_ids.size(1)] = pair_ids
            x += self.pair_embedding(padded_pair_ids)
        
        x += self.positional_encoding[:, :input_ids.size(1), :]
        
        x = self.transformer_encoder(x, src_key_padding_mask=attention_mask)

        mlm_logits = self.mlm_head(x)
        cai_pred = self.cai_head(x.mean(dim=1))
        dg_pred = self.dg_head(x.mean(dim=1))

        loss = None
        if mlm_labels is not None or cai_target is not None or dg_target is not None:
            loss = 0
            if mlm_labels is not None:
                mlm_loss = nn.CrossEntropyLoss(ignore_index=-100)(mlm_logits.view(-1, mlm_logits.size(-1)), mlm_labels.view(-1))
                loss += mlm_loss
            if cai_target is not None and cai_weight > 0:
                cai_loss = nn.MSELoss()(cai_pred.squeeze(), cai_target)
                loss += cai_weight * cai_loss
            if dg_target is not None and dg_weight > 0:
                dg_loss = nn.MSELoss()(dg_pred.squeeze(), dg_target)
                loss += dg_weight * dg_loss

        return mlm_logits, loss, cai_pred, dg_pred

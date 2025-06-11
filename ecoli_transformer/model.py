import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class CodonEncoder(nn.Module):
    def __init__(self, vocab_size, pair_vocab_size=4096, hidden_dim=512, nhead=8, num_layers=6, dim_feedforward=2048, max_seq_len=4096):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
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

    def forward(self, input_ids, pair_ids=None, attention_mask=None, mlm_labels=None, cai_target=None, dg_target=None, cai_weight=0.2, dg_weight=0.2):
        x = self.token_embedding(input_ids)
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

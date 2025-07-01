# Fix the resize_token_embeddings method to handle both expansion and shrinking

with open('ecoli_transformer/model.py', 'r') as f:
    content = f.read()

# Replace the buggy resize method
old_method = '''    def resize_token_embeddings(self, new_num_tokens: int):
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
        self.mlm_head = new_mlm_head'''

new_method = '''    def resize_token_embeddings(self, new_num_tokens: int):
        """
        Resizes the token embeddings and the MLM head to accommodate a new vocabulary size.
        Handles both expansion (new > old) and shrinking (new < old).
        """
        old_num_tokens, hidden_dim = self.token_embedding.weight.shape
        
        if old_num_tokens == new_num_tokens:
            return  # No change needed
        
        # Create new embedding and MLM head layers
        new_embedding = nn.Embedding(new_num_tokens, hidden_dim)
        new_mlm_head = nn.Linear(hidden_dim, new_num_tokens)
        
        # Copy weights for overlapping tokens
        num_to_copy = min(old_num_tokens, new_num_tokens)
        
        new_embedding.weight.data[:num_to_copy, :] = self.token_embedding.weight.data[:num_to_copy, :]
        new_mlm_head.weight.data[:num_to_copy, :] = self.mlm_head.weight.data[:num_to_copy, :]
        new_mlm_head.bias.data[:num_to_copy] = self.mlm_head.bias.data[:num_to_copy]
        
        # Initialize new weights if expanding
        if new_num_tokens > old_num_tokens:
            new_embedding.weight.data[old_num_tokens:].normal_(mean=0.0, std=0.02)
            new_mlm_head.weight.data[old_num_tokens:].normal_(mean=0.0, std=0.02)
            new_mlm_head.bias.data[old_num_tokens:].zero_()
        
        self.token_embedding = new_embedding
        self.mlm_head = new_mlm_head'''

content = content.replace(old_method, new_method)

with open('ecoli_transformer/model.py', 'w') as f:
    f.write(content)

print("Fixed resize_token_embeddings method")

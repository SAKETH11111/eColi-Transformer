# Fix the retokenization script to handle variable-length sequences

with open('tools/retokenize_dataset.py', 'r') as f:
    content = f.read()

# Replace the problematic save function
old_save = '''def save_dataset_split(sequences: List[Dict[str, Any]], output_path: Path, split_name: str):
    """Save a dataset split as a .pt file."""
    if not sequences:
        print(f"Warning: No sequences to save for {split_name}")
        return
        
    # Organize data by keys
    dataset = {}
    for key in sequences[0].keys():
        if key in ['gene_id', 'sequence']:
            # Keep as lists for string data
            dataset[key] = [seq[key] for seq in sequences]
        else:
            # Stack tensors
            dataset[key] = torch.stack([seq[key] for seq in sequences])
    
    output_file = output_path / f"{split_name}.pt"
    torch.save(dataset, output_file)
    print(f"Saved {len(sequences)} sequences to {output_file}")'''

new_save = '''def save_dataset_split(sequences: List[Dict[str, Any]], output_path: Path, split_name: str):
    """Save a dataset split as a .pt file."""
    if not sequences:
        print(f"Warning: No sequences to save for {split_name}")
        return
        
    # Organize data by keys
    dataset = {}
    for key in sequences[0].keys():
        if key in ['gene_id', 'sequence']:
            # Keep as lists for string data
            dataset[key] = [seq[key] for seq in sequences]
        elif key in ['input_ids', 'labels']:
            # Handle variable-length sequences - store as list of tensors
            dataset[key] = [seq[key] for seq in sequences]
        else:
            # Stack scalar tensors
            dataset[key] = torch.stack([seq[key] for seq in sequences])
    
    output_file = output_path / f"{split_name}.pt"
    torch.save(dataset, output_file)
    print(f"Saved {len(sequences)} sequences to {output_file}")'''

content = content.replace(old_save, new_save)

with open('tools/retokenize_dataset.py', 'w') as f:
    f.write(content)

print("Fixed retokenization script to handle variable-length sequences")

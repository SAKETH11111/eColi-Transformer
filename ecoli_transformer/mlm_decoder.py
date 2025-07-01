import torch
import random
from typing import List, Tuple

from ecoli_transformer.model import CodonEncoder
from ecoli_transformer.tokenizer_v2 import CodingTokenizerV2 as CodonTokenizer, RESTRICTION_SITES, AA_TO_CODONS

def has_restriction_site(sequence: str, sites: dict = RESTRICTION_SITES) -> bool:
    """Checks if a sequence contains any of the given restriction sites."""
    for site in sites.values():
        if site in sequence:
            return True
    return False

def protein_to_optimal_codons(protein_seq: str) -> str:
    """Initialize with E. coli-preferred codons based on usage frequency."""
    # Most frequent codon per amino acid in E. coli (based on codon usage tables)
    ECOLI_PREFERRED = {
        'A': 'GCG',  # Ala: 36%
        'C': 'TGC',  # Cys: 55% 
        'D': 'GAT',  # Asp: 63%
        'E': 'GAA',  # Glu: 69%
        'F': 'TTT',  # Phe: 57%
        'G': 'GGC',  # Gly: 40%
        'H': 'CAC',  # His: 57%
        'I': 'ATC',  # Ile: 42%
        'K': 'AAA',  # Lys: 74%
        'L': 'CTG',  # Leu: 52%
        'M': 'ATG',  # Met: 100%
        'N': 'AAC',  # Asn: 55%
        'P': 'CCG',  # Pro: 52%
        'Q': 'CAG',  # Gln: 65%
        'R': 'CGC',  # Arg: 40%
        'S': 'AGC',  # Ser: 28%
        'T': 'ACC',  # Thr: 43%
        'V': 'GTG',  # Val: 37%
        'W': 'TGG',  # Trp: 100%
        'Y': 'TAT'   # Tyr: 57%
    }
    
    result = []
    for aa in protein_seq:
        if aa in ECOLI_PREFERRED:
            result.append(ECOLI_PREFERRED[aa])
        elif aa in AA_TO_CODONS:
            # Fallback to random choice if not in preferred list
            result.append(random.choice(AA_TO_CODONS[aa]))
        else:
            # Invalid amino acid
            result.append("NNN")
    
    return "".join(result)

def apply_random_masking(dna_sequence: str, tokenizer, mask_prob: float = 0.15) -> Tuple[torch.Tensor, List[int]]:
    """Masks a random subset of codons in a DNA sequence."""
    codons = [dna_sequence[i:i+3] for i in range(0, len(dna_sequence), 3)]
    
    # Use the tokenizer's encoding method
    encoded = tokenizer(codons, add_special_tokens=True, return_tensors="pt")
    
    num_codons = len(codons)
    num_to_mask = int(num_codons * mask_prob)
    
    if num_to_mask > 0:
        mask_positions = sorted(random.sample(range(num_codons), min(num_to_mask, num_codons)))
    else:
        mask_positions = []
    
    for pos in mask_positions:
        if pos + 1 < encoded['input_ids'].shape[1]:
            encoded['input_ids'][0, pos + 1] = tokenizer.mask_token_id # +1 to account for [CLS] token
        
    return encoded['input_ids'], mask_positions

def update_with_predictions(dna_sequence: str, logits: torch.Tensor, mask_positions: List[int], tokenizer) -> str:
    """Updates the DNA sequence with the model's predictions at the masked positions."""
    codons = [dna_sequence[i:i+3] for i in range(0, len(dna_sequence), 3)]
    
    # Get predictions - handle different tensor shapes
    predicted_ids = torch.argmax(logits, dim=-1)
    
    # Ensure we have a 1D tensor by flattening if needed
    if predicted_ids.dim() > 1:
        predicted_ids = predicted_ids.flatten()
    
    # Convert to list for safer indexing
    predicted_ids_list = predicted_ids.cpu().tolist()
    
    for pos in mask_positions:
        idx = pos + 1  # Account for [CLS] token
        if idx < len(predicted_ids_list):
            # Get the predicted token ID as a regular integer
            predicted_id = predicted_ids_list[idx]
            
            try:
                predicted_token = tokenizer.convert_ids_to_tokens([predicted_id])[0]
                
                # Check if it's a valid DNA codon (3 nucleotides)
                if (len(predicted_token) == 3 and 
                    all(c in 'ATCG' for c in predicted_token.upper())):
                    codons[pos] = predicted_token.upper()
                elif predicted_token.startswith('[') and predicted_token.endswith(']'):
                    # Handle special tokens - keep original codon
                    continue
                else:
                    # For invalid tokens, try to find valid alternatives
                    # Get top-k predictions and find the first valid codon
                    position_logits = logits.flatten()[idx:idx+1] if logits.dim() > 1 else logits[idx:idx+1]
                    if logits.dim() > 1:
                        # Extract logits for this specific position
                        batch_idx = idx // logits.shape[1]
                        seq_idx = idx % logits.shape[1]
                        if batch_idx < logits.shape[0] and seq_idx < logits.shape[1]:
                            position_logits = logits[batch_idx, seq_idx, :]
                            top_k_ids = torch.topk(position_logits, k=min(10, len(position_logits))).indices
                            
                            for alt_id in top_k_ids:
                                alt_token = tokenizer.convert_ids_to_tokens([alt_id.item()])[0]
                                if (len(alt_token) == 3 and 
                                    all(c in 'ATCG' for c in alt_token.upper())):
                                    codons[pos] = alt_token.upper()
                                    break
                            
            except Exception as e:
                # If token conversion fails, keep the original codon
                continue
        
    return "".join(codons)

def repair_sequence(dna_sequence: str, protein_seq: str) -> str:
    """Repairs a DNA sequence by re-sampling problematic codons."""
    codons = [dna_sequence[i:i+3] for i in range(0, len(dna_sequence), 3)]
    
    # Find positions that might need repair
    repair_positions = []
    
    # Check for restriction sites
    for site_name, site_seq in RESTRICTION_SITES.items():
        start = 0
        while True:
            pos = dna_sequence.find(site_seq, start)
            if pos == -1:
                break
            # Add affected codon positions
            start_codon = pos // 3
            end_codon = (pos + len(site_seq) - 1) // 3
            repair_positions.extend(range(start_codon, end_codon + 1))
            start = pos + 1
    
    # Check for homopolymer runs (>6 consecutive identical nucleotides)
    for i in range(len(dna_sequence) - 6):
        if len(set(dna_sequence[i:i+7])) == 1:
            # Add affected codon positions
            start_codon = i // 3
            end_codon = (i + 6) // 3
            repair_positions.extend(range(start_codon, end_codon + 1))
    
    # Remove duplicates and ensure valid positions
    repair_positions = list(set(repair_positions))
    repair_positions = [pos for pos in repair_positions if pos < len(protein_seq)]
    
    # If no specific problems found, randomly sample some positions for GC adjustment
    if not repair_positions:
        num_to_change = min(3, len(protein_seq))
        repair_positions = random.sample(range(len(protein_seq)), num_to_change)
    
    # Re-sample codons at problematic positions
    for pos in repair_positions:
        if pos < len(protein_seq):
            aa = protein_seq[pos]
            if aa in AA_TO_CODONS:
                # Choose a different codon for this amino acid
                available_codons = AA_TO_CODONS[aa]
                if len(available_codons) > 1:
                    # Avoid the current codon if possible
                    current_codon = codons[pos] if pos < len(codons) else None
                    choices = [c for c in available_codons if c != current_codon]
                    if choices:
                        codons[pos] = random.choice(choices)
                    else:
                        codons[pos] = random.choice(available_codons)
    
    return "".join(codons)

def apply_constraints(dna_sequence: str, protein_seq: str, max_attempts: int = 10) -> str:
    """Apply biological constraints with repair attempts."""
    for attempt in range(max_attempts):
        # Check all constraints
        gc_content = (dna_sequence.count('G') + dna_sequence.count('C')) / len(dna_sequence)
        has_restriction = has_restriction_site(dna_sequence)
        
        # Check for homopolymer runs (>6 consecutive identical nucleotides)
        has_homopolymer = False
        for i in range(len(dna_sequence) - 6):
            if len(set(dna_sequence[i:i+7])) == 1:
                has_homopolymer = True
                break
        
        # Check if all constraints are satisfied
        if (not has_restriction and 
            not has_homopolymer and 
            0.45 <= gc_content <= 0.60):
            return dna_sequence
        
        # If constraints violated, repair the sequence
        dna_sequence = repair_sequence(dna_sequence, protein_seq)
    
    return dna_sequence  # Return best attempt after max_attempts

def validate_inputs(protein_seq: str) -> bool:
    """Validate protein sequence contains only valid amino acids."""
    valid_aas = set(AA_TO_CODONS.keys()) - {'*'}
    return all(aa in valid_aas for aa in protein_seq)

def verify_translation(dna_seq: str, expected_protein: str) -> bool:
    """Verify DNA translates to expected protein."""
    # Standard genetic code for translation
    GENETIC_CODE = {
        'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
        'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
        'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
        'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
        'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
        'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
        'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
        'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
        'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
        'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
        'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
        'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
        'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
        'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
        'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
        'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
    }
    
    # Translate DNA sequence
    if len(dna_seq) % 3 != 0:
        return False
    
    translated = []
    for i in range(0, len(dna_seq), 3):
        codon = dna_seq[i:i+3]
        if codon in GENETIC_CODE:
            aa = GENETIC_CODE[codon]
            if aa == '*':  # Stop codon
                break
            translated.append(aa)
        else:
            # Invalid codon (contains N or other invalid nucleotides)
            return False
    
    translated_protein = ''.join(translated)
    return translated_protein == expected_protein

def iterative_optimization(protein_seq: str, model: CodonEncoder, tokenizer: CodonTokenizer, device: torch.device, iterations: int = 5) -> str:
    """
    Optimizes a DNA sequence for a given protein sequence using iterative masking and prediction.
    """
    # Input validation
    if not validate_inputs(protein_seq):
        raise ValueError(f"Invalid characters in protein sequence: {protein_seq}")
    
    if not protein_seq:
        raise ValueError("Protein sequence cannot be empty")
    
    if len(protein_seq) > 1000:  # Reasonable length limit
        raise ValueError(f"Protein sequence too long: {len(protein_seq)} amino acids (max 1000)")

    model.eval()
    
    # Initialize with optimal codons
    current_dna = protein_to_optimal_codons(protein_seq)
    
    # Apply initial constraints
    current_dna = apply_constraints(current_dna, protein_seq)
    
    print(f"Starting optimization for {len(protein_seq)} amino acids...")
    print(f"Initial GC content: {(current_dna.count('G') + current_dna.count('C')) / len(current_dna):.3f}")
    
    best_dna = current_dna
    best_score = float('-inf')  # Could implement a scoring function later
    
    for i in range(iterations):
        print(f"Iteration {i+1}/{iterations}")
        
        # Apply random masking
        input_ids, mask_positions = apply_random_masking(current_dna, tokenizer)
        input_ids = input_ids.to(device)
        
        # Get model predictions
        with torch.no_grad():
            logits, _, _, _ = model(input_ids)
            
        # Update with predictions
        current_dna = update_with_predictions(current_dna, logits.cpu(), mask_positions, tokenizer)
        
        # Apply biological constraints
        current_dna = apply_constraints(current_dna, protein_seq)
        
        # Track progress
        gc_content = (current_dna.count('G') + current_dna.count('C')) / len(current_dna)
        has_restrictions = has_restriction_site(current_dna)
        print(f"  GC content: {gc_content:.3f}, Restriction sites: {has_restrictions}")
        
        # Keep track of best sequence (could implement proper scoring)
        if not has_restrictions and 0.45 <= gc_content <= 0.60:
            best_dna = current_dna
    
    # Final validation
    if not verify_translation(best_dna, protein_seq):
        print("ERROR: Final sequence does not translate to the expected protein!")
        print(f"Expected: {protein_seq}")
        print(f"Sequence: {best_dna}")
        # Fallback to optimal codons if translation fails
        best_dna = protein_to_optimal_codons(protein_seq)
        print("Returning fallback sequence with optimal codons")
    else:
        print("✓ Translation verified successfully")

    return best_dna

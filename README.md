# eColi-Transformer

Project to optimize E. coli codons using a Transformer model.

# E. coli Transformer Data Processing Pipeline Setup & Execution

These instructions guide you through setting up the environment and running the scripts to process the raw E. coli data into encoded tensors for model training.

**1. Clone the Repository:**

   Get the project code. Replace `<repository_url>` with the actual URL of your Git repository.

   ```bash
   git clone https://github.com/SAKETH11111/eColi-Transformer
   cd eColi-Transformer
   ```

**2. Create and Activate Conda Environment:**

   Create a dedicated conda environment to manage dependencies.

   ```bash
   conda create --name ecoli_env python -y
   conda activate ecoli_env
   ```
   *Note: You'll need to activate this environment (`conda activate ecoli_env`) in every new terminal session before running the scripts.*

**3. Install Dependencies:**

   Install the necessary packages, including ViennaRNA for `RNAfold`.

   ```bash
   conda install -c bioconda viennarna -y
   pip install -r requirements.txt
   ```

   c.  **Build Encoded Tensors (Train Set):**
       This script tokenizes the cleaned sequences, extracts metadata (CAI), calculates MFE using `RNAfold`, and saves the data as PyTorch tensors. This step can take a while depending on the number of sequences and CPU cores.

       ```bash
       python scripts/build_encoded_tensors.py \
           --fasta data/processed/train.fasta \
           --out data/processed/train.pt
       ```
       *Expected output: Creates `data/processed/train.pt` and prints a final summary message (`✔️ Encoded ...`).*
       *Troubleshooting: If this step fails mentioning `RNAfold not found`, double-check that ViennaRNA installed correctly in step 3 and that the `ecoli_env` conda environment is active.*

   d.  **(Optional) Build Encoded Tensors (Validation & Test Sets):**
       Repeat the process for the validation and test sets.

       ```bash
       # Encode validation set
       python scripts/build_encoded_tensors.py \
           --fasta data/processed/val.fasta \
           --out data/processed/val.pt

       # Encode test set
       python scripts/build_encoded_tensors.py \
           --fasta data/processed/test.fasta \
           --out data/processed/test.pt
       ```

**6. Verification:**

*   Check that the expected `.fasta`, `.csv`, and `.pt` files exist in the `data/processed/` directory.
*   Review the summary statistics printed by the `data_clean.py` and `build_encoded_tensors.py` scripts upon completion (Send them to me).
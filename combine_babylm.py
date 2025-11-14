"""
Merge all BabyLM training files from train_10M directory and split into training and validation sets
"""
from pathlib import Path
import random

def combine_babylm_files(input_dir="train_10M", 
                        train_output="corpus_split/train_babylm.txt",
                        val_output="corpus_split/val_babylm.txt",
                        val_ratio=0.05,
                        seed=42,
                        shuffle_files=True,
                        shuffle_lines=False):
    """
    Merge all .train files from train_10M directory and split into training and validation sets.
    Maintains sentence order within each file to preserve dialogue and text coherence.
    
    Args:
        input_dir: Input directory containing all .train files
        train_output: Training set output file path
        val_output: Validation set output file path
        val_ratio: Validation set ratio (default 5%)
        seed: Random seed
        shuffle_files: Whether to shuffle file order (default True, maintains coherence within files)
        shuffle_lines: Whether to shuffle all lines (default False, maintains sentence coherence)
    """
    input_path = Path(input_dir)
    train_path = Path(train_output)
    val_path = Path(val_output)
    
    # Ensure output directory exists
    train_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get all .train files
    train_files = sorted(input_path.glob("*.train"))
    
    if not train_files:
        raise ValueError(f"No .train files found in {input_dir}")
    
    print(f"Found {len(train_files)} training files:")
    for f in train_files:
        print(f"  - {f.name}")
    
    # Read files one by one, maintaining order within each file
    file_data = []
    for train_file in train_files:
        print(f"\nReading {train_file.name}...")
        file_lines = []
        with open(train_file, 'r', encoding='utf-8') as infile:
            for line in infile:
                line = line.strip()
                if line:  # Skip empty lines
                    file_lines.append(line)
        file_data.append((train_file.name, file_lines))
        print(f"  Read {len(file_lines):,} lines from {train_file.name}")
    
    total_lines = sum(len(lines) for _, lines in file_data)
    print(f"\nTotal lines: {total_lines:,}")
    
    # If shuffling file order, only shuffle at file level, maintain order within files
    if shuffle_files:
        random.seed(seed)
        random.shuffle(file_data)
        print("  ✓ Shuffled file order (maintaining sentence coherence within each file)")
    
    # Merge data from all files, maintaining order within each file
    all_lines = []
    for filename, lines in file_data:
        all_lines.extend(lines)
    
    # If shuffling all lines (may break coherence, not recommended for dialogue data)
    if shuffle_lines:
        random.seed(seed)
        random.shuffle(all_lines)
        print("  Shuffled all lines (may break dialogue coherence)")
    else:
        print("  Kept sentence order (maintaining dialogue and text coherence)")
    
    # Split into training and validation sets
    val_size = int(len(all_lines) * val_ratio)
    val_lines = all_lines[:val_size]
    train_lines = all_lines[val_size:]
    
    print(f"\nSplitting data:")
    print(f"  Training set: {len(train_lines):,} lines ({100*(1-val_ratio):.1f}%)")
    print(f"  Validation set: {len(val_lines):,} lines ({100*val_ratio:.1f}%)")
    
    # Write training set
    print(f"\nWriting training set to {train_path}...")
    with open(train_path, 'w', encoding='utf-8') as outfile:
        for line in train_lines:
            outfile.write(line + '\n')
    print(f"  ✓ Training set size: {train_path.stat().st_size / (1024*1024):.2f} MB")
    
    # Write validation set
    print(f"\nWriting validation set to {val_path}...")
    with open(val_path, 'w', encoding='utf-8') as outfile:
        for line in val_lines:
            outfile.write(line + '\n')
    print(f"  ✓ Validation set size: {val_path.stat().st_size / (1024*1024):.2f} MB")
    
    return train_path, val_path

if __name__ == "__main__":
    combine_babylm_files()


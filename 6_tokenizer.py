from pathlib import Path
from tokenizers import (Tokenizer, decoders, models, pre_tokenizers,
                       processors, trainers)
from tokenizers.normalizers import NFKC

def train_tokenizer(corpus_path="corpus_split/train_babylm.txt", 
                    output_name="tokenizer-clean.json"):
    """
    Train tokenizer
    
    Args:
        corpus_path: Training corpus path (default uses BabyLM data)
        output_name: Output filename
    """
    # Define paths
    corpus_path = Path(corpus_path)
    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)

    # Initialize tokenizer with BPE (Byte-Pair Encoding) model
    tokenizer = Tokenizer(models.BPE())

    # Set up the tokenizer components
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    
    # NFKC normalization ensures consistent handling of unicode characters
    tokenizer.normalizer = NFKC()

    # Set up the trainer
    # We include <pad> as a special token even though it's not in the corpus
    trainer = trainers.BpeTrainer(
        vocab_size=16000,          # Total size of vocabulary
        min_frequency=2,           # Token must appear at least twice to be included
        special_tokens=["<pad>", "<s>", "</s>"]  # Special tokens the model should know about
    )

    # Train the tokenizer on the corpus
    print("Training tokenizer...")
    tokenizer.train([str(corpus_path)], trainer)

    # Save the trained tokenizer
    tokenizer_path = output_dir / output_name
    tokenizer.save(str(tokenizer_path), pretty=True)
    print(f"Tokenizer saved to {tokenizer_path}")

    # Let's test the tokenizer to make sure it works correctly
    print("\nTesting tokenizer with a sample from the corpus:")
    # Read first 100 characters of your corpus for testing
    sample_text = corpus_path.read_text(encoding="utf-8")[:100]
    encoded = tokenizer.encode(sample_text)
    print(f"Sample encoded tokens: {encoded.tokens[:10]}...")  # Show first 10 tokens
    decoded = tokenizer.decode(encoded.ids)

    # Normalize newline differences for clearer comparison
    sample_clean = sample_text.replace("\r\n", "\n")
    decoded_clean = decoded.replace("\r\n", "\n")

    match = sample_clean == decoded_clean
    print(f"Decoded correctly? {match}")
    print("Sample snippet :", repr(sample_clean[:80]))
    print("Decoded snippet:", repr(decoded_clean[:80]))

if __name__ == "__main__":
    # Train tokenizer using BabyLM data
    train_tokenizer(
        corpus_path="corpus_split/train_babylm.txt",
        output_name="tokenizer-clean.json"
    )
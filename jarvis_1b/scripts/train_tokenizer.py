import argparse
from pathlib import Path
from jarvis_scientific.tokenizer import train_tokenizer


def main():
    p = argparse.ArgumentParser(description='Train v4 ONLY on document-split training JSONL. No downloads.')
    p.add_argument('--train-jsonl', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--vocab-size', type=int, default=32768)
    p.add_argument('--min-frequency', type=int, default=2)
    p.add_argument('--max-documents', type=int)
    args = p.parse_args()
    print(train_tokenizer(args.train_jsonl, args.output, args.vocab_size,
                          args.min_frequency, args.max_documents, verify_repeat=True))


if __name__ == '__main__':
    main()

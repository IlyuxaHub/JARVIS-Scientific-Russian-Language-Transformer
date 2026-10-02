"""Offline frozen byte-level BPE. No tokenizer dependency in the training hot path."""
import hashlib
import json
import os
from pathlib import Path
import re

SPECIAL_TOKENS = ['<|pad|>', '<|bos|>', '<|eos|>', '<|unk|>']


DOCUMENT_METADATA_FIELDS = ('source', 'source_document_id', 'provenance', 'license', 'language', 'domain')


def exact_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def normalized_hash(text):
    # Candidate detection only. We never discard solely on this hash because
    # whitespace and Unicode details can be meaningful in formulas and code.
    normalized = text.replace('\r\n', '\n').replace('\r', '\n')
    normalized = '\n'.join(line.rstrip(' \t') for line in normalized.split('\n')).strip()
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def canonical_hash(text):
    """Compatibility name: exact content hash after the safety audit."""
    return exact_hash(text)


def validate_document_metadata(obj, source, line, required=False):
    if required:
        missing = [name for name in DOCUMENT_METADATA_FIELDS
                   if name not in obj or obj[name] in (None, '', [])]
        if missing:
            raise ValueError(f'{source}:{line}: missing metadata {missing}')
    language = obj.get('language')
    if language is not None and language not in ('en', 'ru', 'mixed'):
        raise ValueError(f'{source}:{line}: language must be en, ru, or mixed')


def documents(path):
    with open(path, encoding='utf-8', newline='') as f:
        for number, line in enumerate(f, 1):
            obj = json.loads(line)
            if not isinstance(obj.get('text'), str) or not obj['text']:
                raise ValueError(f'{path}:{number}: nonempty text field required')
            yield obj


class FrozenTokenizer:
    def __init__(self, path):
        from tokenizers import Tokenizer
        self.path = Path(path)
        self.backend = Tokenizer.from_file(str(path))
        # Special-looking source text remains ordinary text; BOS/EOS are appended by ID.
        self.backend.encode_special_tokens = True
        self.pad_id, self.bos_id, self.eos_id, self.unk_id = range(4)
        if [self.backend.token_to_id(t) for t in SPECIAL_TOKENS] != list(range(4)):
            raise ValueError('Special-token IDs differ from frozen v4 contract')
        self.vocab_size = self.backend.get_vocab_size()
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()

    def encode(self, text):
        return self.backend.encode(text, add_special_tokens=False).ids

    def decode(self, ids):
        return self.backend.decode(ids, skip_special_tokens=False)


def train_tokenizer(paths, output, vocab_size=32768, min_frequency=2, max_documents=None, verify_repeat=True):
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders, __version__
    from .data import sha256_file
    paths = sorted(Path(p).resolve() for p in paths)
    if vocab_size < 260:
        raise ValueError('Need four special tokens plus all 256 byte values')
    output = Path(output)
    if output.exists():
        raise FileExistsError('Tokenizer artifacts are immutable: choose a fresh output directory')
    output.mkdir(parents=True)
    corpus = [{'path': str(p), 'sha256': sha256_file(p), 'bytes': p.stat().st_size} for p in paths]
    count = 0

    def iterator():
        nonlocal count
        count = 0
        for path in paths:
            for obj in documents(path):
                if max_documents is not None and count >= max_documents:
                    return
                count += 1
                yield obj['text']

    def fit():
        tok = Tokenizer(models.BPE(unk_token=SPECIAL_TOKENS[3]))
        tok.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Digits(individual_digits=True),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)])
        tok.decoder = decoders.ByteLevel()
        tok.train_from_iterator(iterator(), trainers.BpeTrainer(
            vocab_size=vocab_size, min_frequency=min_frequency, show_progress=False,
            initial_alphabet=sorted(pre_tokenizers.ByteLevel.alphabet()),
            special_tokens=SPECIAL_TOKENS, max_token_length=64))
        tok.encode_special_tokens = True
        return json.dumps(json.loads(tok.to_str()), ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n'

    serialized = fit()
    if verify_repeat and serialized != fit():
        raise RuntimeError('BPE retraining differed: no frozen artifact was produced')
    if count == 0:
        raise ValueError('Empty tokenizer training corpus')
    target = output / 'tokenizer.json'
    target.write_text(serialized, encoding='utf-8')
    frozen = FrozenTokenizer(target)
    if frozen.vocab_size != vocab_size:
        target.unlink()
        raise ValueError(f'Corpus only supports {frozen.vocab_size} tokens, requested {vocab_size}; enlarge corpus, do not pad the vocabulary')
    meta = dict(version=4, tokenizer_sha256=frozen.sha256, tokenizers_version=__version__,
                special_tokens=dict(zip(SPECIAL_TOKENS, range(4))), vocab_size=vocab_size,
                min_frequency=min_frequency, max_documents=max_documents, documents=count,
                repeat_verified=verify_repeat, corpus=corpus, normalization=None,
                digits='individual', byte_alphabet=256, max_token_length=64)
    (output / 'training_manifest.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n')
    # Full tokenizer.json already contains exact vocab, merges and pipeline config.
    return meta

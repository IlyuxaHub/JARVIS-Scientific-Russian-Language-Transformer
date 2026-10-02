"""Stream tokenization into immutable binaries while preserving document metadata."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sqlite3
import numpy as np
from jarvis_scientific.data import sha256_file
from jarvis_scientific.tokenizer import (FrozenTokenizer, documents, exact_hash,
                                         normalized_hash, validate_document_metadata)

DEFAULT_SLICES = ('english_stem', 'russian_stem', 'mixed_stem', 'mathematics')


def validation_memberships(obj):
    language = obj.get('language')
    domain = str(obj.get('domain', '')).lower()
    tags = {str(x).lower() for x in obj.get('validation_tags', [])}
    result = []
    if language == 'en':
        result.append('english_stem')
    elif language == 'ru':
        result.append('russian_stem')
    elif language == 'mixed':
        result.append('mixed_stem')
    if domain in ('math', 'mathematics') or tags & {'mathematics', 'latex_heavy'}:
        result.append('mathematics')
    return result


def prepare(train, validation, tokenizer, output, require_metadata=False,
            validation_slices=()):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Choose a new dataset directory; existing data are immutable')
    tok = FrozenTokenizer(tokenizer)
    tokenizer_manifest = Path(tokenizer).with_name('training_manifest.json')
    tok_meta = json.loads(tokenizer_manifest.read_text())
    if tok_meta['tokenizer_sha256'] != tok.sha256 or not tok_meta['repeat_verified']:
        raise ValueError('Train and freeze a reproducible tokenizer first')
    train_sha, val_sha = sha256_file(train), sha256_file(validation)
    corpus_hashes = {x['sha256'] for x in tok_meta['corpus']}
    if val_sha in corpus_hashes:
        raise ValueError('Validation was used to train this tokenizer')
    if {train_sha} != corpus_hashes:
        raise ValueError('Tokenizer corpus must be exactly this training JSONL; combine approved shards first')
    output.mkdir(parents=True)
    shutil.copyfile(tokenizer, output / 'tokenizer.json')
    shutil.copyfile(tokenizer_manifest, output / 'tokenizer_training_manifest.json')
    db = sqlite3.connect(output / 'document_index.sqlite')
    db.execute('create table docs(hash text primary key, normalized_hash text, split text)')
    db.execute('create table groups(name text primary key, split text)')
    manifest = {'format': 'jarvis-tokens-v1', 'synthetic': False,
                'vocab_size': tok.vocab_size, 'disjoint_documents': True,
                'metadata_schema': 'jarvis-document-metadata-v2',
                'packing': 'BOS + document + EOS, concatenated; cross-document attention allowed',
                'tokenizer': {'path': 'tokenizer.json', 'sha256': tok.sha256},
                'validation_sets': {}}
    dtype = '<u2' if tok.vocab_size <= 65536 else '<u4'
    requested = tuple(validation_slices)
    if len(set(requested)) != len(requested) or any(x not in DEFAULT_SLICES for x in requested):
        raise ValueError(f'validation_slices must be selected from {DEFAULT_SLICES}')

    def register(obj, split, source, line):
        validate_document_metadata(obj, source, line, require_metadata)
        digest = exact_hash(obj['text'])
        prior = db.execute('select split from docs where hash=?', (digest,)).fetchone()
        if prior:
            raise ValueError(f'Duplicate document in {split}, previously in {prior[0]}; split/deduplicate first')
        normalized = normalized_hash(obj['text'])
        db.execute('insert into docs values (?,?,?)', (digest, normalized, split))
        group = str(obj.get('group') or obj.get('source_document_id') or digest)
        prior_group = db.execute('select split from groups where name=?', (group,)).fetchone()
        if prior_group and prior_group[0] != split:
            raise ValueError('Source group overlaps training and validation')
        db.execute('insert or ignore into groups values (?,?)', (group, split))
        ids = tok.encode(obj['text'])
        if tok.unk_id in ids or any(x < 4 for x in ids):
            raise ValueError('Unexpected special/unknown token in ordinary source text')
        if tok.decode(ids) != obj['text']:
            raise ValueError('Tokenizer encode/decode failed')
        packed = [tok.bos_id, *ids, tok.eos_id]
        meta = dict(obj)
        meta.pop('text', None)
        meta.update(document_hash=digest, normalized_hash=normalized, split=split,
                    character_count=len(obj['text']), byte_count=len(obj['text'].encode('utf-8')),
                    token_count=len(packed), validation_slices=validation_memberships(obj) if split == 'validation' else [])
        return packed, meta

    try:
        for split, source in [('train', Path(train)), ('validation', Path(validation))]:
            final = output / f'{split}.bin'
            temporary = final.with_suffix('.bin.tmp')
            metadata_path = output / f'{split}.metadata.jsonl'
            token_count = document_count = max_id = 0
            language_tokens, domain_tokens = {}, {}
            slice_files = {}
            slice_stats = {name: {'tokens': 0, 'documents': 0, 'max_token_id': 0} for name in requested}
            if split == 'validation':
                for name in requested:
                    path = output / f'validation_{name}.bin.tmp'
                    slice_files[name] = path.open('wb')
            try:
                with temporary.open('wb') as binary, metadata_path.open('w', encoding='utf-8', newline='\n') as metadata:
                    for line, obj in enumerate(documents(source), 1):
                        packed, meta = register(obj, split, source, line)
                        values = np.asarray(packed, dtype=dtype)
                        values.tofile(binary)
                        metadata.write(json.dumps(meta, ensure_ascii=False) + '\n')
                        token_count += len(packed)
                        document_count += 1
                        max_id = max(max_id, max(packed))
                        language = str(obj.get('language', 'unknown'))
                        domain = str(obj.get('domain', 'unknown'))
                        language_tokens[language] = language_tokens.get(language, 0) + len(packed)
                        domain_tokens[domain] = domain_tokens.get(domain, 0) + len(packed)
                        for name in set(meta['validation_slices']) & set(requested):
                            values.tofile(slice_files[name])
                            item = slice_stats[name]
                            item['tokens'] += len(packed)
                            item['documents'] += 1
                            item['max_token_id'] = max(item['max_token_id'], max(packed))
                        if document_count % 10000 == 0:
                            db.commit()
                    binary.flush(); os.fsync(binary.fileno())
                if document_count == 0:
                    raise ValueError(f'Empty {split} dataset')
            finally:
                for f in slice_files.values():
                    f.flush(); os.fsync(f.fileno()); f.close()
            os.replace(temporary, final)
            manifest[split] = {'path': final.name, 'dtype': dtype, 'tokens': token_count,
                               'max_token_id': max_id, 'documents': document_count,
                               'sha256': sha256_file(final), 'source_sha256': sha256_file(source),
                               'metadata_path': metadata_path.name,
                               'metadata_sha256': sha256_file(metadata_path),
                               'language_tokens': language_tokens, 'domain_tokens': domain_tokens}
            if split == 'validation':
                for name, stats in slice_stats.items():
                    if stats['documents'] == 0:
                        raise ValueError(f'Required validation slice {name!r} is empty')
                    temporary_slice = output / f'validation_{name}.bin.tmp'
                    final_slice = output / f'validation_{name}.bin'
                    os.replace(temporary_slice, final_slice)
                    manifest['validation_sets'][name] = {
                        'path': final_slice.name, 'dtype': dtype,
                        'tokens': stats['tokens'], 'documents': stats['documents'],
                        'max_token_id': stats['max_token_id'], 'sha256': sha256_file(final_slice)}
        db.commit()
        (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    finally:
        db.close()
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train', type=Path, required=True)
    p.add_argument('--validation', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--validation-slices', nargs='*', default=list(DEFAULT_SLICES))
    p.add_argument('--allow-missing-metadata', action='store_true',
                   help='Compatibility only; production corpora should not use this')
    args = p.parse_args()
    print(prepare(args.train, args.validation, args.tokenizer, args.output,
                  require_metadata=not args.allow_missing_metadata,
                  validation_slices=args.validation_slices))


if __name__ == '__main__':
    main()

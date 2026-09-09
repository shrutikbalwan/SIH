import csv
from collections import Counter
from pathlib import Path

manifest_path = r"d:\SIH\ira-wakeword\dataset\split_manifest.csv"

pos_train, pos_val, pos_test = 0, 0, 0
real_train, real_val, real_test = 0, 0, 0
tts_train, tts_val, tts_test = 0, 0, 0

amb_train, amb_val, amb_test = 0, 0, 0
sp_train, sp_val, sp_test = set(), set(), set()

with open(manifest_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for r in reader:
        split = r['split']
        label = int(r.get('label', -1))
        group = r.get('group', '').lower()
        path = r['path'].lower()
        
        # Positive
        is_pos = ('positive' in path) or ('real' in group) or ('piper' in group)
        if label == 0: is_pos = False
        
        if is_pos:
            is_real = ('real' in group) or ('real' in path)
            is_tts = ('piper' in group) or ('piper' in path)
            if not is_real and not is_tts:
                if 'tts' in path: is_tts = True
                else: is_real = True
                
            if split == 'train':
                pos_train += 1
                if is_real: real_train += 1
                else: tts_train += 1
            elif split == 'validation':
                pos_val += 1
                if is_real: real_val += 1
                else: tts_val += 1
            elif split == 'test':
                pos_test += 1
                if is_real: real_test += 1
                else: tts_test += 1
                
        # Negative Ambient
        if label == 0 and ('background' in group or 'ambient' in path or 'background' in path):
            if split == 'train': amb_train += 1
            elif split == 'validation': amb_val += 1
            elif split == 'test': amb_test += 1
            
        # Negative Speech
        if label == 0 and ('speech' in group or 'librispeech' in group or 'librispeech' in path):
            # Extract speaker ID
            parts = Path(path).name.split('_')
            spk = parts[0]
            if spk.isdigit():
                if split == 'train': sp_train.add(spk)
                elif split == 'validation': sp_val.add(spk)
                elif split == 'test': sp_test.add(spk)

print(f"Pos Train: {pos_train} (Real: {real_train}, TTS: {tts_train})")
print(f"Pos Val: {pos_val} (Real: {real_val}, TTS: {tts_val})")
print(f"Pos Test: {pos_test} (Real: {real_test}, TTS: {tts_test})")

print(f"\nAmb Train files: {amb_train}")
print(f"Amb Val files: {amb_val}")
print(f"Amb Test files: {amb_test}")

print(f"\nSpeech Train speakers ({len(sp_train)}): {sorted(list(sp_train))}")
print(f"Speech Val speakers ({len(sp_val)}): {sorted(list(sp_val))}")
print(f"Speech Test speakers ({len(sp_test)}): {sorted(list(sp_test))}")

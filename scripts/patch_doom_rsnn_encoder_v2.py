"""Apply a deliberately small experiment to the ViZDoom convolutional LIF encoder.

Run in the project root: python scripts/patch_doom_rsnn_encoder_v2.py
The script is intentionally strict: it only patches the known baseline implementation.
"""
from pathlib import Path

source = Path('src/doomrsnn/model.py')
if not source.is_file():
    candidate = Path(__file__).resolve().parent.parent / 'src/doomrsnn/model.py'
    if candidate.is_file():
        source = candidate
    else:
        raise SystemExit('Run this script from your doom_rsnn_ampere project root')

code = source.read_text()
changes = [
    ('from dataclasses import dataclass',
     'from dataclasses import dataclass\nimport math'),
    ("self.spatial_pool = tuple(config['pool_hw'])",
     "# Normalize convolutional LIF currents to reduce saturation.\n"
     "        self.conv_norms = nn.ModuleList([nn.GroupNorm(math.gcd(4, c), c) for c in channels])\n"
     "        self.spatial_pool = tuple(config['pool_hw'])"),
    ('self.recurrent = nn.Linear(self.hidden, self.hidden, bias=False)',
     'self.recurrent = nn.Linear(self.hidden, self.hidden, bias=False)\n'
     '        self.recurrent_norm = nn.LayerNorm(self.hidden)'),
    ('membrane=list(state.membranes)\n        spike=list(state.spikes)',
     '# Reinitialize visual LIF neurons for each new observation; preserve recurrent state.\n'
     '        membrane=[torch.zeros_like(m) for m in state.membranes[:3]] + [state.membranes[3]]\n'
     '        spike=[torch.zeros_like(s) for s in state.spikes[:3]] + [state.spikes[3]]'),
    ('current=conv(signal)',
     'current=self.conv_norms[i](conv(signal))'),
    ('current=self.input_projection(pooled) + self.recurrent(spike[3])',
     'current=self.recurrent_norm(self.input_projection(pooled) + self.recurrent(spike[3]))'),
]
for old, new in changes:
    n = code.count(old)
    if n != 1:
        raise SystemExit(f'Expected one occurrence of {old!r}, found {n}. No files were changed.')
    code = code.replace(old, new, 1)

repo_root = source.resolve().parent.parent.parent
backup = repo_root / 'runs/backups/model_before_norm_v2.py'
backup.parent.mkdir(parents=True, exist_ok=True)
if backup.exists():
    raise SystemExit(f'Backup already exists: {backup}. Refusing to overwrite.')
backup.write_text(source.read_text())
source.write_text(code)
print(f'Patched {source}')
print(f'Old implementation backed up to {backup}')
print('Model architecture changed: OLD checkpoints are not compatible with this new model.')

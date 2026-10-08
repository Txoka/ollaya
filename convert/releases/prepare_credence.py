"""Recreate small Credence package inputs from its pinned public release.

Run from the repository root, then:
PYTHONPATH=convert python -m ollaya_convert.package credencev1-gemma4
"""
import json
import urllib.request
from pathlib import Path

release = json.loads(Path(__file__).with_name('credencev1-gemma4-e4b.json').read_text())
for variant, metadata in [('accuracy', release), ('calibrated', release['calibrated_variant'])]:
    pin = metadata['source_publication']
    base = f"https://huggingface.co/{pin['repo']}/resolve/{pin['revision']}"
    out = Path(__file__).parents[1] / 'out' / f'credencev1-gemma4-{variant}'
    out.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f'{base}/{variant}/decision.json') as response:
        decision = json.load(response)
    assert decision['layout'] == 'winnow-v1' and decision['thought'] is False
    assert decision['gguf']['sha256'] == metadata['gguf_sha256']
    # Rebind provenance to the published, byte-identical file; tokens are unchanged.
    decision['gguf'].update(repo=pin['repo'], revision=pin['revision'],
                            path=pin['path'], url=f"{base}/{pin['path']}")
    (out / 'decision.json').write_text(json.dumps(decision, indent=2))
    with urllib.request.urlopen(f'{base}/{variant}/calibration.json') as response:
        calibration = json.load(response)
    expected = metadata['external_temperature']
    expected = expected if isinstance(expected, list) else [expected] * 3
    assert calibration['temperature'] == expected
    (out / 'calibration.json').write_text(json.dumps(calibration, indent=2))
    print('Prepared', out)

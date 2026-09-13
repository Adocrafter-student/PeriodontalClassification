"""Reproduce held-out predictions from all saved checkpoints; never train."""
from pathlib import Path
import json

import joblib
import numpy as np
import torch

from .common import device_for, load_config, read_prepared, write_json
from .evaluate import collect, PROB_COLS
from .features import read_features
from .models import EfficientNet
from .train import cnn_probabilities, loader_for


@torch.inference_mode()
def main():
    cfg = load_config()
    samples, manifest = read_prepared(cfg)
    samples = samples.sort_values('image_id').reset_index(drop=True)
    report = []
    for method in ('efficientnet', 'dino_global', 'dino_global_regions'):
        saved = collect(cfg, samples, manifest, method)
        if method == 'efficientnet':
            model = EfficientNet(cfg, pretrained=False).to(device_for(cfg)).eval()
        else:
            features = read_features(cfg, samples, manifest, method.removeprefix('dino_'))
        for fold in range(cfg['outer_folds']):
            folder = Path(cfg['output']) / 'runs' / method / f'fold_{fold}'
            selection = json.loads((folder / 'selection.json').read_text())
            indices = np.flatnonzero(samples.fold.to_numpy() == fold)
            test = samples.iloc[indices].reset_index(drop=True)
            development_ids = set(samples.loc[samples.fold != fold, 'image_id'])
            assert set(selection['test_ids']) == set(test.image_id)
            if method == 'efficientnet':
                assert set(selection['refit_ids']) == development_ids
                a, b = set(selection['selection_train_ids']), set(selection['selection_val_ids'])
                assert not a & b and a | b == development_ids
                history = json.loads((folder / 'selection_history.json').read_text())
                expected_epoch = max(history, key=lambda row: row['val_macro_f1'])['epoch']
                assert selection['best_epoch'] == expected_epoch
                refit = json.loads((folder / 'refit_history.json').read_text())
                assert len(refit) == expected_epoch
                assert all(np.isfinite(row['train_loss']) for row in history + refit)
                model.load_state_dict(torch.load(folder / 'model.pt', map_location=device_for(cfg), weights_only=True))
                probabilities = cnn_probabilities(model, loader_for(test, cfg, model, False, cfg['seed'] + fold), device_for(cfg))
            else:
                assert set(selection['train_ids']) == development_ids
                classifier = joblib.load(folder / 'classifier.joblib')
                assert np.array_equal(classifier.classes_, [1, 2, 3, 4])
                probabilities = classifier.predict_proba(features[indices])
            expected = saved.loc[saved.fold == fold, PROB_COLS].to_numpy()
            np.testing.assert_allclose(probabilities, expected, rtol=1e-4, atol=1e-6)
            assert np.array_equal(probabilities.argmax(1), expected.argmax(1))
            record = dict(method=method, fold=fold, samples=len(test), predictions_reproduced=True,
                          max_absolute_probability_difference=float(np.abs(probabilities - expected).max()))
            report.append(record)
            print(record, flush=True)
        if method == 'efficientnet':
            del model
            torch.cuda.empty_cache()
    write_json(Path(cfg['output']) / 'checks' / 'trained_checkpoints.json', report)
    print('All 15 checkpoints reproduced their held-out predictions; no training performed.')


if __name__ == '__main__':
    main()

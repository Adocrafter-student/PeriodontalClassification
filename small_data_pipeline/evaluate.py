from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import experiment_signature, metrics, read_prepared, write_json

METHODS = ('efficientnet', 'dino_global', 'dino_global_regions')
PROB_COLS = [f'prob_stage_{i}' for i in range(1, 5)]


def collect(cfg, samples, manifest, method):
    parts = []
    signature = experiment_signature(cfg, manifest, method)
    for fold in range(cfg['outer_folds']):
        folder = Path(cfg['output']) / 'runs' / method / f'fold_{fold}'
        info = json.loads((folder / 'complete.json').read_text())
        if info['signature'] != signature:
            raise ValueError(f'{method} fold {fold} has different experiment settings.')
        predictions = pd.read_csv(folder / 'predictions.csv', dtype={'group_id': str})
        expected = samples[samples.fold == fold].sort_values('image_id').reset_index(drop=True)
        predictions = predictions.sort_values('image_id').reset_index(drop=True)
        columns = ['image_id', 'group_id', 'stage_2017', 'fold']
        if not predictions[columns].equals(expected[columns]):
            raise ValueError(f'{method} fold {fold} predictions do not match its held-out patients.')
        metrics(predictions.stage_2017, predictions[PROB_COLS].to_numpy())
        parts.append(predictions)
    result = pd.concat(parts).sort_values('image_id').reset_index(drop=True)
    if result.image_id.duplicated().any() or len(result) != len(samples):
        raise ValueError('Each patient image must have exactly one out-of-fold prediction.')
    return result


def bootstrap_indices(samples, rng):
    """Resample whole patient groups within stage, conditional on the observed class mix."""
    pieces = []
    for _, stage in samples.groupby('stage_2017'):
        groups = stage.group_id.unique()
        for group in rng.choice(groups, size=len(groups), replace=True):
            pieces.extend(stage.index[stage.group_id == group].tolist())
    return np.asarray(pieces)


def evaluate(cfg, methods, bootstraps=1000):
    samples, manifest = read_prepared(cfg)
    samples = samples.sort_values('image_id').reset_index(drop=True)
    results, reports, summary = {}, {}, []
    for method in methods:
        df = collect(cfg, samples, manifest, method)
        results[method] = df
        report = metrics(df.stage_2017, df[PROB_COLS].to_numpy())
        report['fold_metrics'] = {str(fold): metrics(part.stage_2017, part[PROB_COLS].to_numpy())
                                  for fold, part in df.groupby('fold')}
        reports[method] = report
        summary.append(dict(method=method, **{k: report[k] for k in
                       ('accuracy', 'balanced_accuracy', 'macro_f1', 'ordinal_mae', 'quadratic_weighted_kappa')}))
    rng = np.random.default_rng(cfg['seed'])
    scores = {method: [] for method in methods}
    for _ in range(bootstraps):
        index = bootstrap_indices(samples, rng)
        for method, df in results.items():
            part = df.iloc[index]
            scores[method].append(metrics(part.stage_2017, part[PROB_COLS].to_numpy())['macro_f1'])
    output = Path(cfg['output']) / 'evaluation'
    output.mkdir(exist_ok=True)
    for method, report in reports.items():
        if bootstraps:
            report['macro_f1_conditional_95pct_interval'] = np.percentile(scores[method], [2.5, 97.5]).tolist()
        report['interval_note'] = ('Stratified patient bootstrap of fixed out-of-fold predictions; '
                                   'does not include retraining variability or external generalization.')
        write_json(output / f'{method}.json', report)
        results[method].to_csv(output / f'{method}_oof.csv', index=False)
    if 'efficientnet' in scores and bootstraps:
        write_json(output / 'paired_comparisons.json', {
            method: dict(macro_f1_difference=reports[method]['macro_f1'] - reports['efficientnet']['macro_f1'],
                         conditional_95pct_interval=np.percentile(
                             np.asarray(values) - np.asarray(scores['efficientnet']), [2.5, 97.5]).tolist())
            for method, values in scores.items() if method != 'efficientnet'})
    table = pd.DataFrame(summary)
    table.to_csv(output / 'comparison.csv', index=False)
    print(table.to_string(index=False))

"""Training entry points. Never imported/executed by the non-training check command."""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .common import (STAGES, device_for, experiment_signature, metrics, prediction_frame,
                     read_prepared, runtime, seed_all, write_json)
from .data import Images, inner_splits, validation_split
from .features import read_features
from .models import EfficientNet, transform_for


def claim_fold(cfg, manifest, method, fold):
    out = Path(cfg['output']) / 'runs' / method / f'fold_{fold}'
    signature = experiment_signature(cfg, manifest, method)
    if (out / 'complete.json').exists():
        saved = json.loads((out / 'complete.json').read_text())
        if saved['signature'] != signature:
            raise ValueError(f'{out} belongs to different settings. Choose a new output directory.')
        print(f'Skipping completed {method} fold {fold}')
        return None, signature
    # Interrupted folds restart from external initialization, not a partial checkpoint.
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'started.json', dict(signature=signature, config=cfg, runtime=runtime()))
    return out, signature


def finish_fold(out, signature, test, probabilities, details):
    prediction_frame(test, probabilities).to_csv(out / 'predictions.csv', index=False)
    write_json(out / 'complete.json', dict(signature=signature, **details))


def train_dino(cfg, fold_arg=None, mode='global'):
    samples, manifest = read_prepared(cfg)
    x = read_features(cfg, samples, manifest, mode)
    method = f'dino_{mode}'
    for fold in range(cfg['outer_folds']) if fold_arg is None else [fold_arg]:
        out, signature = claim_fold(cfg, manifest, method, fold)
        if out is None:
            continue
        train_idx = np.flatnonzero(samples.fold.to_numpy() != fold)
        test_idx = np.flatnonzero(samples.fold.to_numpy() == fold)
        dev = samples.iloc[train_idx].reset_index(drop=True)
        splits = inner_splits(dev, cfg['inner_folds'], cfg['seed'] + fold)
        # Scaling is fitted inside each inner split, never on held-out rows.
        estimator = Pipeline([('scale', StandardScaler()), ('classifier', LogisticRegression(
            class_weight='balanced', max_iter=5000, solver='lbfgs', random_state=cfg['seed'] + fold))])
        search = GridSearchCV(estimator, {'classifier__C': cfg['dino']['c_values']}, cv=splits,
                              scoring=make_scorer(f1_score, labels=STAGES, average='macro', zero_division=0),
                              n_jobs=1, refit=True, error_score='raise')
        print(f'{method} fold {fold}: classifier fitting on CPU; features were extracted on CUDA.')
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            search.fit(x[train_idx], dev.stage_2017.to_numpy())
        if not np.array_equal(search.best_estimator_.classes_, STAGES):
            raise ValueError('Classifier class order is not 1,2,3,4.')
        probabilities = search.predict_proba(x[test_idx])
        joblib.dump(search.best_estimator_, out / 'classifier.joblib')
        write_json(out / 'selection.json', dict(best_C=search.best_params_['classifier__C'],
                   inner_macro_f1=float(search.best_score_), candidates=[
                       dict(C=p['classifier__C'], mean_macro_f1=float(score))
                       for p, score in zip(search.cv_results_['params'], search.cv_results_['mean_test_score'])],
                   train_ids=dev.image_id.tolist(), test_ids=samples.iloc[test_idx].image_id.tolist()))
        finish_fold(out, signature, samples.iloc[test_idx], probabilities, dict(method=method, mode=mode))


def loader_for(df, cfg, model, train, seed):
    c = cfg['efficientnet']
    dataset = Images(df, cfg['dataset']['root'], transform_for(model, c['image_size'], train))
    return DataLoader(dataset, batch_size=c['batch_size'], shuffle=train,
                      generator=torch.Generator().manual_seed(seed), num_workers=cfg['num_workers'],
                      pin_memory=str(cfg['device']).startswith('cuda'))


@torch.inference_mode()
def cnn_probabilities(model, loader, device):
    model.eval()
    return np.concatenate([torch.softmax(model(x[:, 0].to(device)), 1).float().cpu().numpy()
                           for x, _ in loader])


def fit_cnn(cfg, train, val, seed, epoch_limit, out, phase, initial_state):
    c = cfg['efficientnet']
    device = device_for(cfg)
    seed_all(seed)
    # Same externally initialized encoder AND same random head in selection/refit.
    model = EfficientNet(cfg, pretrained=False)
    model.load_state_dict(initial_state)
    model = model.to(device)
    loader = loader_for(train, cfg, model, True, seed)
    val_loader = loader_for(val, cfg, model, False, seed) if val is not None else None
    counts = np.bincount(train.stage_2017.to_numpy() - 1, minlength=4)
    if (counts == 0).any():
        raise ValueError('Training partition lacks a stage.')
    weights = torch.tensor(len(train) / (4 * counts), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=c['label_smoothing'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=c['lr'], weight_decay=c['weight_decay'])
    amp = c['amp'] and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    history, best_score, best_epoch, stale = [], -1.0, 1, 0
    for epoch in range(1, epoch_limit + 1):
        # Use the original planned schedule during refit, not a compressed schedule.
        warmup = min(3, c['epochs'])
        factor = epoch / warmup if epoch <= warmup else .5 * (
            1 + math.cos(math.pi * (epoch - warmup) / max(1, c['epochs'] - warmup)))
        for group in optimizer.param_groups:
            group['lr'] = c['lr'] * factor
        model.train()
        total_loss, count = 0.0, 0
        for x, y in tqdm(loader, desc=f'{phase} epoch {epoch}/{epoch_limit}', leave=False):
            x, y = x[:, 0].to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                loss = loss_fn(model(x), y)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss; fold is not marked complete.')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(y)
            count += len(y)
        row = dict(epoch=epoch, train_loss=total_loss / count, lr=optimizer.param_groups[0]['lr'])
        if val_loader is not None:
            probabilities = cnn_probabilities(model, val_loader, device)
            score = metrics(val.stage_2017, probabilities)['macro_f1']
            row['val_macro_f1'] = score
            if score > best_score:
                best_score, best_epoch, stale = score, epoch, 0
            else:
                stale += 1
        history.append(row)
        write_json(out / f'{phase}_history.json', history)
        print(f'{phase}: {row}', flush=True)
        if val_loader is not None and stale >= c['patience']:
            break
    return model, best_epoch


def train_efficientnet(cfg, fold_arg=None):
    samples, manifest = read_prepared(cfg)
    for fold in range(cfg['outer_folds']) if fold_arg is None else [fold_arg]:
        out, signature = claim_fold(cfg, manifest, 'efficientnet', fold)
        if out is None:
            continue
        seed = cfg['seed'] + fold
        dev = samples[samples.fold != fold].reset_index(drop=True)
        test = samples[samples.fold == fold].reset_index(drop=True)
        a, b = validation_split(dev, cfg['efficientnet']['validation_fraction'], seed)
        seed_all(seed)
        initial_model = EfficientNet(cfg, pretrained=True)
        initial_state = {k: v.detach().cpu().clone() for k, v in initial_model.state_dict().items()}
        del initial_model
        selected, best_epoch = fit_cnn(cfg, dev.iloc[a], dev.iloc[b], seed,
                                      cfg['efficientnet']['epochs'], out, 'selection', initial_state)
        del selected
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Fit on ALL outer-development patients for selected epochs, then evaluate once.
        model, _ = fit_cnn(cfg, dev, None, seed, best_epoch, out, 'refit', initial_state)
        probabilities = cnn_probabilities(model, loader_for(test, cfg, model, False, seed), device_for(cfg))
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out / 'model.pt')
        write_json(out / 'selection.json', dict(best_epoch=best_epoch, selection_train_ids=dev.iloc[a].image_id.tolist(),
                   selection_val_ids=dev.iloc[b].image_id.tolist(), refit_ids=dev.image_id.tolist(),
                   test_ids=test.image_id.tolist(), initialization=cfg['efficientnet']['model']))
        finish_fold(out, signature, test, probabilities, dict(method='efficientnet', best_epoch=best_epoch))
        del model, initial_state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

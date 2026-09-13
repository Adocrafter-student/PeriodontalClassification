from __future__ import annotations

import copy
import json
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from small_data_pipeline.common import (experiment_signature, file_hash, load_config, metrics,
                                         prediction_frame, read_prepared, write_json)
from small_data_pipeline.data import Letterbox, assert_partition, inner_splits, validation_split, views
from small_data_pipeline.evaluate import bootstrap_indices, collect
from small_data_pipeline.features import feature_signature, read_features
from small_data_pipeline.models import pool_features


class PipelineTests(unittest.TestCase):
    def test_prepared_folds_cover_each_image_once_and_keep_all_stages(self):
        cfg = load_config()
        df, _ = read_prepared(cfg)
        self.assertFalse(df.image_id.duplicated().any())
        self.assertEqual(set(df.fold), set(range(cfg['outer_folds'])))
        for fold in range(cfg['outer_folds']):
            a = np.flatnonzero(df.fold.to_numpy() != fold)
            b = np.flatnonzero(df.fold.to_numpy() == fold)
            assert_partition(df, a, b)
            dev = df.iloc[a].reset_index(drop=True)
            inner_splits(dev, cfg['inner_folds'], cfg['seed'] + fold)
            validation_split(dev, .15, cfg['seed'] + fold)

    def test_repeated_patients_cannot_cross_boundaries(self):
        df = pd.DataFrame([dict(group_id=f'{stage}-{patient}', stage_2017=stage)
                           for stage in range(1, 5) for patient in range(15) for _ in range(2)])
        splits = inner_splits(df, 3, 42)
        for a, b in splits:
            assert_partition(df, a, b)
        a, b = validation_split(df, .2, 42)
        assert_partition(df, a, b)
        with self.assertRaisesRegex(ValueError, 'crosses'):
            assert_partition(df, np.array([0]), np.array([1]))

    def test_letterboxing_preserves_full_wide_image(self):
        source = Image.new('RGB', (200, 100), (255, 255, 255))
        result = np.asarray(Letterbox(100)(source))
        self.assertEqual(result.shape, (100, 100, 3))
        self.assertTrue((result[25:75] == 255).all())
        self.assertTrue((result[:25] == 0).all())
        self.assertTrue((result[75:] == 0).all())

    def test_regions_and_pooling(self):
        image = Image.new('RGB', (200, 100))
        self.assertEqual([x.size for x in views(image, 'global_regions')],
                         [(200, 100), (100, 100), (100, 100), (100, 100)])
        features = torch.tensor([[[1., 2.], [3., 4.], [5., 6.], [7., 8.]]])
        torch.testing.assert_close(pool_features(features, 'global_regions'), torch.tensor([[1., 2., 5., 6.]]))
        with self.assertRaises(ValueError):
            pool_features(features[:, :1], 'global_regions')

    def test_metrics_use_stage_order_and_equal_class_weighting(self):
        y = np.array([1, 2, 3, 3, 3, 4])
        p = np.eye(4)[np.array([2, 2, 3, 3, 3, 4]) - 1]
        report = metrics(y, p)
        self.assertAlmostEqual(report['accuracy'], 5 / 6)
        self.assertAlmostEqual(report['balanced_accuracy'], .75)
        self.assertAlmostEqual(report['ordinal_mae'], 1 / 6)
        self.assertEqual(report['per_stage']['1']['recall'], 0)
        with self.assertRaises(ValueError):
            metrics(y, p * .5)

    def test_float32_probability_rounding_does_not_change_classification(self):
        p = np.full((4, 4), .05)
        np.fill_diagonal(p, .85)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            report = metrics([1, 2, 3, 4], p * (1 + 1e-7))
        self.assertEqual(report['accuracy'], 1.0)
        self.assertAlmostEqual(report['log_loss'], -np.log(.85))

    def test_cache_rejects_changed_recipe_and_shuffled_ids(self):
        cfg = copy.deepcopy(load_config())
        samples = pd.DataFrame({'image_id': [1, 2]})
        manifest = {'samples_sha256': 'fixture'}
        with tempfile.TemporaryDirectory() as temp:
            cfg['output'] = temp
            folder = Path(temp) / 'features'
            folder.mkdir()
            np.savez(folder / 'dino.npz', image_id=[2, 1], features=np.zeros((2, 4, 384), dtype=np.float32))
            (folder / 'encoder.pt').write_bytes(b'checksum fixture; never deserialized')
            info = dict(signature=feature_signature(cfg, manifest),
                        feature_sha256=file_hash(folder / 'dino.npz'), encoder_sha256=file_hash(folder / 'encoder.pt'))
            write_json(folder / 'manifest.json', info)
            with self.assertRaisesRegex(ValueError, 'ordering'):
                read_features(cfg, samples, manifest, 'global')
            cfg['dino']['image_size'] = 224
            with self.assertRaisesRegex(ValueError, 'preprocessing'):
                read_features(cfg, samples, manifest, 'global')

    def test_oof_collection_rejects_wrong_patients(self):
        cfg = copy.deepcopy(load_config())
        cfg['outer_folds'] = 2
        manifest = {'samples_sha256': 'fixture'}
        samples = pd.DataFrame([dict(image_id=i, group_id=str(i), stage_2017=i % 4 + 1, fold=i // 4)
                                for i in range(8)])
        with tempfile.TemporaryDirectory() as temp:
            cfg['output'] = temp
            for fold in range(2):
                folder = Path(temp) / 'runs' / 'dino_global' / f'fold_{fold}'
                write_json(folder / 'complete.json', dict(signature=experiment_signature(cfg, manifest, 'dino_global')))
                part = samples[samples.fold == fold]
                prediction_frame(part, np.eye(4)).to_csv(folder / 'predictions.csv', index=False)
            actual = collect(cfg, samples, manifest, 'dino_global')
            self.assertEqual(len(actual), 8)
            target = Path(temp) / 'runs' / 'dino_global' / 'fold_0' / 'predictions.csv'
            bad = pd.read_csv(target)
            bad.loc[0, 'image_id'] = 999
            bad.to_csv(target, index=False)
            with self.assertRaisesRegex(ValueError, 'held-out'):
                collect(cfg, samples, manifest, 'dino_global')

    def test_bootstrap_keeps_repeated_patient_images_together(self):
        samples = pd.DataFrame([dict(group_id=f'{stage}-{i}', stage_2017=stage)
                                for stage in range(1, 5) for i in range(3) for _ in range(2)])
        indices = bootstrap_indices(samples, np.random.default_rng(42))
        counts = np.bincount(indices, minlength=len(samples))
        np.testing.assert_array_equal(counts[::2], counts[1::2])


if __name__ == '__main__':
    unittest.main()

from __future__ import annotations

import argparse

from .common import load_config


def main():
    parser = argparse.ArgumentParser(description='BRAR: clean EfficientNet vs frozen DINOv2')
    parser.add_argument('--config', help='YAML config; relative paths resolve beside that file')
    parser.add_argument('--device', help='Override device (cuda:0 by default)')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare', help='Audit and create immutable shared folds; no training')
    check = sub.add_parser('check', help='Forward-only CUDA and data checks; no training')
    check.add_argument('--offline', action='store_true', help='Random weights for architecture checks only')
    sub.add_parser('extract', help='Frozen DINO feature extraction only; no classifier fitting')
    train = sub.add_parser('train', help='Explicitly start training')
    train.add_argument('--method', required=True, choices=['efficientnet', 'dino_global', 'dino_global_regions'])
    train.add_argument('--fold', type=int, help='One outer fold, numbered 0..4; otherwise all')
    evaluate = sub.add_parser('evaluate', help='Aggregate completed out-of-fold predictions')
    evaluate.add_argument('--methods', nargs='+', default=['efficientnet', 'dino_global', 'dino_global_regions'],
                          choices=['efficientnet', 'dino_global', 'dino_global_regions'])
    evaluate.add_argument('--bootstraps', type=int, default=1000)
    predict = sub.add_parser('predict', help='Predict one image using completed fold models')
    predict.add_argument('--method', required=True, choices=['efficientnet', 'dino_global', 'dino_global_regions'])
    predict.add_argument('--input', required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.device:
        cfg['device'] = args.device
    if args.command == 'prepare':
        from .data import prepare
        prepare(cfg)
    elif args.command == 'check':
        from .check import check
        check(cfg, args.offline)
    elif args.command == 'extract':
        from .features import extract
        extract(cfg)
    elif args.command == 'train':
        if args.fold is not None and not 0 <= args.fold < cfg['outer_folds']:
            parser.error('--fold is outside the configured outer folds')
        from .train import train_dino, train_efficientnet
        if args.method == 'efficientnet':
            train_efficientnet(cfg, args.fold)
        else:
            train_dino(cfg, args.fold, args.method.removeprefix('dino_'))
    elif args.command == 'evaluate':
        if args.bootstraps < 0:
            parser.error('--bootstraps must be nonnegative')
        from .evaluate import evaluate
        evaluate(cfg, args.methods, args.bootstraps)
    elif args.command == 'predict':
        from .predict import predict
        predict(cfg, args.method, args.input)


if __name__ == '__main__':
    main()

"""Bounded development comparison. Run --preflight-only before market training.

No automatic final holdout, official promotion, or live trading. Fixed folds:
2021-2022 -> 2023; 2021-2023 -> 2024. 2025+ is not read by this runner.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import threading
import time
from datetime import datetime, timezone

import duckdb
import numpy as np
import pandas as pd
from filelock import FileLock

from deep_alpha.dataset import QuantTensorDataset, build_streaming_dataset
from evaluation.research_contract import purged_split, daily_rank_ic, summarize_ic
from evaluation.portfolio_replay import replay_next_open
from research.data import (FEATURES, inspect_data, inspect_staging_data,
                           load_panel, load_staging_panel, build_features)

ROOT = Path(__file__).resolve().parents[1]
FOLDS = [("2023", "20221231", "20230101", "20231231"),
         ("2024", "20231231", "20240101", "20241231")]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def save_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(path)


def select_dataset(ds, mask):
    indices = np.flatnonzero(np.asarray(mask))
    return QuantTensorDataset(ds.features,ds.labels,[ds.indices_map[i] for i in indices],ds.seq_len,
                              ds.sample_metadata.iloc[indices].reset_index(drop=True))


def matrix(ds):
    return np.stack([ds.features[s:e].reshape(-1) for s,e in ds.indices_map])


def source_hashes():
    paths = sorted([*ROOT.joinpath('research').glob('*.py'),
                    ROOT/'evaluation/research_contract.py',ROOT/'evaluation/portfolio_replay.py',
                    ROOT/'deep_alpha/dataset.py',ROOT/'deep_alpha/daily_sampler.py',
                    ROOT/'deep_alpha/train_pipeline.py',ROOT/'deep_alpha/models/itransformer.py'])
    return {str(p.relative_to(ROOT)):digest(p) for p in paths}


def cost_sensitivity(panel, signals, calendar, *, top_n, horizon=5,
                     bps_per_side=(20, 50, 100)):
    """Replay the same signals under a fixed per-side cost grid.

    This is a descriptive development diagnostic. It never changes model
    selection or promotion state, and it is only callable when the panel has
    the full execution columns required by ``replay_next_open``.
    """
    results = {}
    for bps in bps_per_side:
        rate = float(bps) / 10_000.0
        summary, _, _, _ = replay_next_open(
            panel, signals, calendar, horizon=horizon, top_n=top_n,
            buy_cost=rate, sell_cost=rate)
        results[str(int(bps))] = {
            key: summary.get(key)
            for key in ("total_return", "annual_return", "sharpe", "max_drawdown",
                        "cost_total", "fills", "rejected_orders")
        }
    return {"status": "UNVERIFIED", "bps_per_side": [int(x) for x in bps_per_side],
            "results": results}


def capacity_sensitivity(panel, signals, calendar, *, top_n, horizon=5,
                         participation=(0.0025, 0.005, 0.01, 0.02)):
    """Replay identical signals at explicit prior-day amount participation caps."""
    results = {}
    for rate in participation:
        if not np.isfinite(rate) or rate <= 0 or rate > 1:
            raise ValueError("participation caps must be finite and in (0, 1]")
        summary, _, _, _ = replay_next_open(
            panel, signals, calendar, horizon=horizon, top_n=top_n,
            max_participation=float(rate))
        key = f"{rate:g}"
        results[key] = {
            "participation": float(rate),
            "total_return": summary.get("total_return"),
            "annual_return": summary.get("annual_return"),
            "sharpe": summary.get("sharpe"),
            "cost_total": summary.get("cost_total"),
            "fills": summary.get("fills"),
            "rejected_orders": summary.get("rejected_orders"),
            "mean_gross_traded_fraction": summary.get("mean_gross_traded_fraction"),
        }
    return {"status": "UNVERIFIED", "participation_caps": [float(x) for x in participation],
            "results": results}


def run(args):
    args.data_mode = getattr(args, 'data_mode', 'api')
    args.start = getattr(args, 'start', '20210101')
    args.end = getattr(args, 'end', '20241231')
    if (args.start, args.end) != ('20210101', '20241231'):
        raise ValueError('fixed development folds require --start 20210101 --end 20241231')
    out = Path(args.run_dir).resolve()
    out.mkdir(parents=True,exist_ok=True)
    lock = FileLock(str(ROOT/'research'/'experiment.lock'),timeout=0)
    with lock:
        state = {"status":"PREFLIGHT","pid":os.getpid(),"updated_at":None,"stage":"environment"}
        stop = threading.Event()
        def beat(**updates):
            state.update(updates)
        def writer():
            while not stop.is_set():
                payload = dict(state)
                payload['updated_at'] = datetime.now(timezone.utc).isoformat()
                save_json(out/'heartbeat.json',payload)
                stop.wait(5)
        thread = threading.Thread(target=writer,daemon=True)
        thread.start()
        try:
            return experiment(args,out,beat)
        except Exception as e:
            state.update(status='BLOCKED',reason=f'{type(e).__name__}: {e}')
            save_json(out/'status.json',dict(state))
            print(state['reason'],flush=True)
            return 2
        finally:
            stop.set(); thread.join()
            state['updated_at'] = datetime.now(timezone.utc).isoformat()
            save_json(out/'heartbeat.json',state)


def experiment(args,out,beat):
    environment = {"python":platform.python_version(),"packages":{p:importlib.metadata.version(p)
                   for p in ['numpy','pandas','scipy','scikit-learn','lightgbm','torch','duckdb','filelock']}}
    save_json(out/'environment.json',environment)
    beat(stage='real_data_preflight')
    try:
        conn = duckdb.connect(str(Path(args.db).resolve()),read_only=True)
    except Exception as e:
        result = {'status':'BLOCKED','reason':'DATABASE_UNAVAILABLE','detail':str(e),'data_mode':'real','training_started':False}
        save_json(out/'preflight.json',result); save_json(out/'status.json',result)
        beat(status='BLOCKED',stage='database_unavailable')
        print(json.dumps(result,ensure_ascii=False)); return 2
    try:
        start = args.start
        end = args.end
        preflight = (inspect_staging_data(conn, start, end)
                     if args.data_mode == 'staging'
                     else inspect_data(conn, start, end))
        save_json(out/'preflight.json',preflight)
        if not preflight['ready']:
            result = {'status':'BLOCKED','reason':'DATA_INTEGRITY','errors':preflight['errors'],'training_started':False}
            save_json(out/'status.json',result); beat(status='BLOCKED'); print(json.dumps(result)); return 2
        if args.preflight_only:
            result = {'status':'PREFLIGHT_PASSED','training_started':False,'promotion':'BLOCKED',
                      'data_mode':args.data_mode,'execution_ready':bool(preflight.get('execution_ready', args.data_mode == 'api'))}
            save_json(out/'status.json',result); beat(status='PREFLIGHT_PASSED'); print(result); return 0
        beat(stage='load_real_panel')
        panel,codes = (load_staging_panel(conn, start, end, preflight['calendar'], args.max_stocks, 5)
                       if args.data_mode == 'staging'
                       else load_panel(conn, start, end, preflight['calendar'], args.max_stocks, 5))
    finally:
        conn.close()
    if len(codes) < 50: raise ValueError('need at least 50 stocks for daily IC')
    panel_hash = hashlib.sha256(pd.util.hash_pandas_object(panel,index=False).values.tobytes()).hexdigest()
    config = {'contract':'next_open_five_sessions_v1','data_mode':args.data_mode,
              'label_adjustment':'raw_unadjusted' if args.data_mode == 'staging' else 'adj_factor',
              'execution_ready':args.data_mode == 'api','folds':FOLDS,'sequence_length':args.seq_len,
              'features':FEATURES,'seed':42,'max_stocks':args.max_stocks,'codes':codes,'epochs':args.epochs,
              'buy_cost':.002,'sell_cost':.002,'horizon':5,'device':args.device,'transformer':args.transformer,
              'scope':'development_validation_only','panel_hash':panel_hash,'sources':source_hashes(),
              'environment':environment,'trial_budget':'one fixed factor, Ridge, LightGBM and optional iTransformer per fold',
              'holdout':'2025 onward not read; prior exposure unknown; no untouched claim',
              'universe':('initial-date staging metadata listings, first N sorted codes; bounded nonrepresentative sample'
                          if args.data_mode == 'staging' else
                          'initial-date SH/SZ listings, first N sorted codes; bounded nonrepresentative sample')}
    config_file = out/'config.json'
    if config_file.exists():
        if not args.resume: raise ValueError('run directory already used; choose new --run-dir or --resume')
        previous = json.loads(config_file.read_text(encoding='utf-8'))
        if previous != json.loads(json.dumps(config)): raise ValueError('resume rejected: data, code, environment or configuration changed')
    else:
        save_json(config_file,config)
    save_json(out/'data_snapshot.json',{'sha256':panel_hash,'rows':len(panel),'stocks':len(codes),
              'min_date':str(panel.trade_date.min()),'max_date':str(panel.trade_date.max()),'label_nonmissing':int(panel.target_ret.notna().sum())})
    frame = build_features(panel)
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from lightgbm import LGBMRegressor
    import joblib
    import torch
    torch.set_num_threads(4)
    np.random.seed(42); torch.manual_seed(42)
    reports = []
    for fold,train_end,valid_start,valid_end in FOLDS:
        fold_dir = out/fold; fold_dir.mkdir(exist_ok=True)
        beat(status='RUNNING',stage=f'{fold}:prepare')
        train_mask,_ = purged_split(frame,train_end,valid_start,valid_end)
        fit_rows = train_mask & np.isfinite(frame[FEATURES]).all(axis=1)
        if not fit_rows.any(): raise ValueError('no training features')
        processor = StandardScaler().fit(frame.loc[fit_rows,FEATURES])
        work = frame.copy()
        work[FEATURES] = processor.transform(frame[FEATURES])
        # Inference must not select stocks using availability of future labels.
        # Unknown labels are excluded only from training and metric computation.
        ds = build_streaming_dataset(work,FEATURES,seq_len=args.seq_len,require_label=False)
        train_ids,valid_ids = purged_split(ds.sample_metadata,train_end,valid_start,valid_end)
        known = np.asarray([np.isfinite(ds.labels[e-1]) for _,e in ds.indices_map])
        train_ids &= known; valid_ids &= known
        train_ds,valid_ds = select_dataset(ds,train_ids),select_dataset(ds,valid_ids)
        inference_ds = select_dataset(ds,ds.sample_metadata.trade_date.between(valid_start,valid_end))
        if not len(train_ds) or not len(valid_ds): raise ValueError('empty purged split')
        x_train,x_valid = matrix(train_ds),matrix(inference_ds)
        y_train = np.asarray([train_ds.labels[e-1] for _,e in train_ds.indices_map])
        truth = inference_ds.sample_metadata.copy()
        truth['target_ret'] = [inference_ds.labels[e-1] for _,e in inference_ds.indices_map]
        truth.loc[truth.label_end_date.isna() | truth.label_end_date.gt(valid_end),'target_ret'] = np.nan
        joblib.dump(processor,fold_dir/'processor.joblib')
        save_json(fold_dir/'split.json',{'train_samples':len(train_ds),'valid_samples':len(valid_ds),
                  'train_last_signal':str(train_ds.sample_metadata.trade_date.max()),
                  'train_last_label_end':str(train_ds.sample_metadata.label_end_date.max()),
                  'inference_samples_including_unknown_targets':len(inference_ds),
                  'valid_first_signal':str(truth.trade_date.min()),'valid_last_signal':str(truth.trade_date.max())})
        models = ['reversal_5','ridge','lightgbm'] + (['itransformer_adapted'] if args.transformer else [])
        prior_predictions = {}
        for name in models:
            beat(stage=f'{fold}:{name}')
            artifact = fold_dir/name; artifact.mkdir(exist_ok=True)
            completed = artifact/'complete.json'
            if args.resume and completed.exists():
                old = json.loads(completed.read_text(encoding='utf-8'))
                if not all((artifact/p).exists() and digest(artifact/p)==sha for p,sha in old['hashes'].items()):
                    raise ValueError(f'corrupt checkpoint: {fold}/{name}')
                pred = pd.read_csv(artifact/'predictions.csv',dtype={'trade_date':str,'label_end_date':str,'ts_code':str})
                report = old['report']
            else:
                # Fixed order: Transformer cannot run before all three baselines.
                if name=='reversal_5': scores=x_valid[:,-len(FEATURES)]
                elif name in ['ridge','lightgbm']:
                    model = Ridge(alpha=100.) if name=='ridge' else LGBMRegressor(
                        n_estimators=120,num_leaves=15,max_depth=5,learning_rate=.03,
                        min_child_samples=100,reg_lambda=10.,random_state=42,n_jobs=4,verbosity=-1)
                    model.fit(x_train,y_train); scores=model.predict(x_valid)
                    joblib.dump(model,artifact/'model.joblib')
                else:
                    from deep_alpha.models.itransformer import iTransformer
                    from deep_alpha.train_pipeline import train_deep_model
                    torch.manual_seed(42)
                    model=iTransformer(feat_dim=len(FEATURES),seq_len=args.seq_len,d_model=32,n_heads=2,num_layers=1,dropout=.1)
                    trained=train_deep_model(model,train_ds,valid_ds,job_id='itransformer',epochs=args.epochs,
                              device=args.device,output_dir=artifact,resume=args.resume,heartbeat=lambda **kw:beat(**kw))
                    save_json(artifact/'training.json',trained)
                    model.eval(); device=next(model.parameters()).device
                    with torch.no_grad():
                        scores=np.concatenate([model(x.to(device)).flatten().cpu().numpy()
                              for x,_ in torch.utils.data.DataLoader(inference_ds,batch_size=256,shuffle=False)])
                pred=truth.copy(); pred['score']=np.asarray(scores)
                if not np.isfinite(pred.score).all(): raise ValueError('nonfinite predictions')
                pred.to_csv(artifact/'predictions.csv',index=False)
                daily=daily_rank_ic(pred); daily.to_csv(artifact/'daily_ic.csv',index=False)
                ic=summarize_ic(daily)
                if ic['days']==0: raise ValueError('no valid daily cross-sections')
                if args.data_mode == 'staging':
                    summary = {'status': 'BLOCKED', 'reason': 'staging lacks authoritative adjustment factors, opening limits, PIT membership and ST/delist execution state'}
                    report={'fold':fold,'model':name,'ic':ic,'portfolio':summary,
                            'status':'UNVERIFIED','execution':'BLOCKED',
                            'cost_sensitivity': {'status': 'BLOCKED', 'bps_per_side': [20, 50, 100],
                                                'reason': summary['reason']},
                            'capacity_sensitivity': {'status': 'BLOCKED',
                                                    'participation_caps': [0.0025, 0.005, 0.01, 0.02],
                                                    'reason': summary['reason']},
                            'data_mode':'staging'}
                else:
                    days=[d for d in preflight['calendar'] if valid_start<=d<=valid_end]
                    summary,equity,fills,rejects=replay_next_open(panel,pred,days,horizon=5,top_n=min(30,len(codes)//5))
                    equity.to_csv(artifact/'equity.csv',index=False)
                    fills.to_csv(artifact/'fills.csv',index=False); rejects.to_csv(artifact/'rejected_orders.csv',index=False)
                    sensitivity = cost_sensitivity(
                        panel, pred, days, horizon=5, top_n=min(30, len(codes)//5))
                    capacity = capacity_sensitivity(
                        panel, pred, days, horizon=5, top_n=min(30, len(codes)//5))
                    report={'fold':fold,'model':name,'ic':ic,'portfolio':summary,'status':'UNVERIFIED',
                            'cost_sensitivity': sensitivity, 'capacity_sensitivity': capacity,
                            'data_mode':'api'}
                save_json(artifact/'report.json',report)
                hashes={p.name:digest(p) for p in artifact.iterdir() if p.is_file() and p.name!='complete.json'}
                save_json(completed,{'hashes':hashes,'report':report})
            prior_predictions[name]=pred
            reports.append(report)
        # Paired, keyed validation diagnostics. Blending is a registered analysis,
        # not a newly selected strategy and never evaluated on a final test here.
        if 'itransformer_adapted' in prior_predictions:
            left=prior_predictions['lightgbm']; right=prior_predictions['itransformer_adapted']
            pair=left.merge(right[['trade_date','ts_code','score']],on=['trade_date','ts_code'],suffixes=('_lgbm','_transformer'),validate='one_to_one')
            pair['score']=pair.groupby('trade_date').score_lgbm.rank(pct=True)*.5 + pair.groupby('trade_date').score_transformer.rank(pct=True)*.5
            blend=summarize_ic(daily_rank_ic(pair))
            correlations=pair.groupby('trade_date').apply(lambda g:g.score_lgbm.corr(g.score_transformer,method='spearman'))
            save_json(fold_dir/'incremental_diagnostic.json',{'equal_rank_blend_ic':blend,
                      'mean_prediction_correlation':float(correlations.mean()),'status':'development_diagnostic_only'})
    result={'status':'UNVERIFIED','experiment_complete':True,'promotion':'BLOCKED',
            'data_mode':args.data_mode,'execution_ready':args.data_mode == 'api','reports':reports,
            'next_gate':'full data/PIT audit, cost sensitivity, multiple seeds and untouched final holdout before paper trading'}
    save_json(out/'report.json',result); save_json(out/'status.json',result)
    beat(status='COMPLETED_UNVERIFIED',stage='done')
    print(f'Development experiment complete: {out / "report.json"}',flush=True)
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--db')
    p.add_argument('--data-mode',choices=['api','staging'],default='api')
    p.add_argument('--start',default='20210101')
    p.add_argument('--end',default='20241231')
    p.add_argument('--run-dir',default=str(ROOT/'research_runs/factor_baseline_v1'))
    p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--transformer',action='store_true')
    p.add_argument('--max-stocks',type=int,default=128)
    p.add_argument('--seq-len',type=int,default=20)
    p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--device',default='cpu')
    args=p.parse_args()
    if args.db is None:
        args.db = ('daily_staging.duckdb' if args.data_mode == 'staging'
                   else str(ROOT/'tushare_data.duckdb'))
    if not 50<=args.max_stocks<=512 or args.seq_len<2 or not 1<=args.epochs<=20:
        p.error('bounded run requires 50..512 stocks, sequence >=2, epochs 1..20')
    return run(args)


if __name__=='__main__': raise SystemExit(main())

import os
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import mean_squared_error
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from matplotlib import pyplot as plt

import jpholiday
import datetime

# sales の足し算を取る
def _sales_sum(row):
    return sum(v for i, v in row.items() if 'Sales' in i) 

def _log(x):
    return np.log(x + 0.00000001)

def _is_holiday(x):
    return 0 if jpholiday.is_holiday_name(x.name) is None else 1

def _is_weekend(x):
    return 1 if x.name.weekday_name in ['Saturday', 'Sunday'] else 0

def _get_weekday(x):
    m = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}
    return m[x.name.weekday_name]


# データを読込む
def read_data(in_dir, target):
    df_train_X = pd.read_excel(os.path.join(in_dir, 'train_X.xlsx'), header=0, index_col=0)
    df_train_Y = pd.read_excel(os.path.join(in_dir, 'train_Y.xlsx'), header=0, index_col=0)
    
    df_val_X = pd.read_excel(os.path.join(in_dir, 'val_X.xlsx'), header=0, index_col=0)
    df_val_Y = pd.read_excel(os.path.join(in_dir, 'val_Y.xlsx'), header=0, index_col=0)
    
    #print(df_train_X.shape, df_train_Y.shape, df_val_X.shape, df_val_Y.shape)
    
    # 週末、祝日情報を追加
    df_train_Y['weekday_Y'] = df_train_Y.apply(_get_weekday, axis=1)
    df_train_Y['isHoliday_Y'] = df_train_Y.apply(_is_holiday, axis=1)
    df_train_Y['isWeekend_Y'] = df_train_Y.apply(_is_weekend, axis=1)
    
    df_val_Y['weekday_Y'] = df_val_Y.apply(_get_weekday, axis=1)
    df_val_Y['isHoliday_Y'] = df_val_Y.apply(_is_holiday, axis=1)
    df_val_Y['isWeekend_Y'] = df_val_Y.apply(_is_weekend, axis=1)

    df_train_X['weekday_Y'] = pd.Series(df_train_Y['weekday_Y'].values, index=df_train_X.index)
    df_train_X['isHoliday_Y'] = pd.Series(df_train_Y['isHoliday_Y'].values, index=df_train_X.index)
    df_train_X['isWeekend_Y'] = pd.Series(df_train_Y['isWeekend_Y'].values, index=df_train_X.index)
    
    df_val_X['weekday_Y'] = pd.Series(df_val_Y['weekday_Y'].values, index=df_val_X.index)
    df_val_X['isHoliday_Y'] = pd.Series(df_val_Y['isHoliday_Y'].values, index=df_val_X.index)
    df_val_X['isWeekend_Y'] = pd.Series(df_val_Y['isWeekend_Y'].values, index=df_val_X.index)

    
    # sales の場合は合計を取る
    '''
    if target == 'Sales':
        df_train_Y['Sales'] = df_train_Y.apply(lambda x: _sales_sum(x), axis=1)
        df_val_Y['Sales'] = df_val_Y.apply(lambda x: _sales_sum(x), axis=1)
    '''
    df_train_Y[f'{target}_log'] = df_train_Y[target].apply(_log)
    df_val_Y[f'{target}_log'] = df_val_Y[target].apply(_log)
    
    return df_train_X, df_train_Y, df_val_X, df_val_Y

# xgboost のデータ構造へ転換
def mtx_trans(target, df_train_X, df_train_Y, df_val_X, df_val_Y):
    d_train_sales = xgb.DMatrix(df_train_X, label=df_train_Y[f'{target}_log'])
    d_val_sales = xgb.DMatrix(df_val_X, label=df_val_Y[f'{target}_log'])
    d_val = xgb.DMatrix(df_val_X)
    
    return d_train_sales, d_val_sales, d_val
    
# パラメータ最適化
def _score(params):
    print("Training with params: ")
    print(params)

    evals = [(d_train_sales, 'train'), (d_val_sales, 'eval')]
    evals_result = {}

    model = xgb.train(params, 
              d_train_sales, 
              num_boost_round=1000, 
              evals=evals,
              early_stopping_rounds=20,
              evals_result=evals_result)
    
    d_pred = np.exp(model.predict(d_val))
    loss = mean_squared_error(d_pred, df_val_Y['Sales'].values)
    print(f'loss: {loss}')
    return {'loss': loss, 'status': STATUS_OK}

def optimize(random_state=71):
    
    space = {
        'n_estimators': hp.quniform('n_estimators', 100, 1000, 1),
        'learning_rate': hp.quniform('eta', 0.025, 0.5, 0.025),
        'max_depth':  hp.choice('max_depth', np.arange(3, 10, dtype=int)),
        'min_child_weight': hp.loguniform('min_child_weight', np.log(0.1), np.log(10)),
        'subsample': hp.quniform('subsample', 0.6, 0.95, 0.05),
        'gamma': hp.loguniform('gamma', np.log(1e-8), np.log(1.0)),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.6, 0.95, 0.05),
        'alpha': hp.loguniform('alpha', np.log(1e-8), np.log(1.0)),
        'lambda': hp.loguniform('lambda', np.log(1e-6), np.log(10.0)),
        'nthread': 4,
        'seed': random_state
    }
    best = fmin(_score, space, 
                algo=tpe.suggest, 
                max_evals=250)
    return best

# モデルを学習
def train2(d_train_sales, d_val_sales, params, best_iter=1000, esr=20):
    evals = [(d_train_sales, 'train'), (d_val_sales, 'eval')]
    evals_result = {}
    model = xgb.train(params, 
                      d_train_sales, 
                      num_boost_round=best_iter,
                      early_stopping_rounds=esr,
                      evals=evals,
                      evals_result=evals_result)
    return model, evals_result

# 損失関数を描画
def draw_loss(evals_result):
    train_metric = evals_result['train']['rmse']
    plt.plot(train_metric, label='train rmse')
    eval_metric = evals_result['eval']['rmse']
    plt.plot(eval_metric, label='eval rmse')
    plt.grid()
    plt.legend()
    plt.title('loss curves')
    plt.xlabel('rounds')
    plt.ylabel('rmse')
    plt.show()
    
# 予測結果と実測値を描画
def draw_preds(model, d_val, df_val_Y, target):
    d_pred_log = model.predict(d_val)
    d_pred = np.exp(d_pred_log)

    plt.figure(figsize=(15,5))
    plt.plot(df_val_Y[target].values, label='real')
    plt.plot(d_pred, label='pred')
    plt.legend()
    plt.grid()
    plt.show()
    
# 特徴量の寄与度を描画
def draw_feat_importances(model):
    _, ax = plt.subplots(figsize=(15,30))
    xgb.plot_importance(model, ax=ax, importance_type='gain')
    plt.show()
    
# 特徴量の寄与度を大きい順番に取得
def get_feat_importances(model):
    m = model.get_score(importance_type='gain')
    return sorted(m, key=lambda x: m[x], reverse=True)
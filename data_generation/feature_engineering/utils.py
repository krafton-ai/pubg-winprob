import os
import pandas as pd
import numpy as np
import glob
import json
import re
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import gzip
import math
from collections import Counter
import time
import functools


def save_dict(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)

# Dictionary 불러오기
def load_dict(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def top_k_frequent_items(lst, k):
    if not lst:
        return None
    # 아이템 빈도 계산
    counter = Counter(lst)
    # 상위 k개의 가장 흔한 아이템과 그 빈도 반환
    return counter.most_common(k)



def calculate_normal_probability(N, K, alpha, beta, p):
    """
    N: 총 게임 수
    K: 치터로 판정된 게임 수
    alpha: False Positive Rate (일반 유저를 치터로 잘못 판정할 확률)
    beta: True Positive Rate (치터를 올바르게 치터로 판정할 확률)
    p: 치터의 사전 확률 (전체 유저 중 치터의 비율)
    """

    # 이항 계수 계산
    binom_coeff = math.comb(N, K)

    # 일반 유저가 K번 치터로 판정될 확률
    P_K_given_Normal = binom_coeff * (alpha ** K) * ((1 - alpha) ** (N - K))

    # 치터가 K번 치터로 판정될 확률
    P_K_given_Cheater = binom_coeff * (beta ** K) * ((1 - beta) ** (N - K))

    # 전체 확률 P(K번 치터)
    P_K = (P_K_given_Normal * (1 - p)) + (P_K_given_Cheater * p)

    # 베이즈 정리를 이용하여 일반 유저일 확률 계산
    P_Normal_given_K = (P_K_given_Normal * (1 - p)) / P_K

    return P_Normal_given_K


def get_confidence_values(mean, std_dev, confidence_level=99):
    z_score_99 = 2.576
    z_score_95 = 1.96
    
    if confidence_level == 99:
        confidence_lower = mean - z_score_99 * std_dev
        confidence_upper = mean + z_score_99 * std_dev
    elif confidence_level == 95:
        confidence_lower = mean - z_score_95 * std_dev
        confidence_upper = mean + z_score_95 * std_dev
    
    return confidence_lower, confidence_upper

def safe_literal_eval(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return val

    
def load_match_df_from_ded_jsonl(path, use_log_types, use_cols, return_df=False):
    data = []
    with open(path, 'r', encoding='utf-8-sig') as file:
        for i, line in enumerate(file):
            try:
                json_obj = json.loads(line.strip())
                data.append(json_obj)
            except json.JSONDecodeError as e:
                print(f"{i} JSON 디코드 오류: {e}, 해당 라인 건너뜀: {line}")
    
    log_data = []
    for json_d in data:
        if json_d['_T'].lower() in use_log_types:
            log_data.append(json_d)
            
    ded_df = pd.DataFrame(log_data)
    ded_df.columns = [col.lower() for col in ded_df.columns] 
    valid_columns = ['_t', '_d'] + [col for col in use_cols if col in ded_df.columns]
    ded_df['_t'] = ded_df['_t'].apply(lambda x:x.lower())
    
    df = ded_df[valid_columns]
    
    N, C = df.shape
    LogMatchDefinition = df[df['_t'] == 'logmatchdefinition']
    LogMatchDefinition = LogMatchDefinition.dropna(axis=1, how='all')
    MatchId = LogMatchDefinition.iloc[0]['matchid']
    if return_df:
        return df.copy(), MatchId, N
    else:
        return MatchId, N    


def s3filecopy(path, temp_path='/path/to/match_sample/'):
    local_path = os.path.join(temp_path, path.split('/')[-1])
    dbutils.fs.cp(path, f"{local_path}")
    return local_path


def read_gzip(path):
    data = []
    with gzip.open(path, "rt", encoding="utf-8-sig") as gz_file:
        for i, line in enumerate(gz_file):
            try:
                json_obj = json.loads(line.strip())
                data.append(json_obj)
            except json.JSONDecodeError as e:
                print(f"{i} JSON 디코드 오류: {e}, 해당 라인 건너뜀: {line}")

    return data


def load_match_df_from_s3(path, use_log_types, use_cols, return_df=False):
    data = read_gzip(path)
    log_data = []
    for json_d in data:
        if json_d['_T'].lower() in use_log_types:
            log_data.append(json_d)
            
    ded_df = pd.DataFrame(log_data)
    ded_df.columns = [col.lower() for col in ded_df.columns] 
    valid_columns = ['_t', '_d'] + [col for col in use_cols if col in ded_df.columns]
    ded_df['_t'] = ded_df['_t'].apply(lambda x:x.lower())
    df = ded_df[valid_columns]
    
    N, C = df.shape
    LogMatchDefinition = df[df['_t'] == 'logmatchdefinition']
    LogMatchDefinition = LogMatchDefinition.dropna(axis=1, how='all')
    MatchId = LogMatchDefinition.iloc[0]['matchid']
    if return_df:
        return df.copy(), MatchId, N
    else:
        return MatchId, N    


def extract_MatchId_from(json_file, return_df=False):
    file_path = json_file
    with open(file_path, 'r') as file:
        data = json.load(file)

    df = pd.DataFrame(data)
    df.columns = [col.lower() for col in df.columns]
    df['_t'] = df['_t'].apply(lambda x:x.lower()) 
    
    N, C = df.shape
    LogMatchDefinition = df[df['_t'] == 'logmatchdefinition']
    LogMatchDefinition = LogMatchDefinition.dropna(axis=1, how='all')
    MatchId = LogMatchDefinition.iloc[0]['matchid']
    if return_df:
        return df.copy(), MatchId, N
    else:
        return MatchId, N

def get_ingame_time_from(df):
    if ('event_at' in df.columns) & (df['_d'].apply(len).iloc[0] < 12):
        df['_d'] = df['event_at']
    
    df['_d'] = pd.to_datetime(df['_d'])
    df = df.sort_values('_d').reset_index(drop=True)
    timestamp_elapsed0 = df[df['elapsedtime'] == 0.0]._d.max()
    df['ingame_time'] = (df._d - timestamp_elapsed0).apply(lambda x:x.total_seconds() if x.total_seconds() > 0 else 0)
    # df[df.elapsedTime.notnull()][['ingame_time', 'elapsedTime']].plot()
    return df

def dict2cols(df, col_name):
    """
    DataFrame의 딕셔너리 컬럼을 개별 컬럼으로 확장합니다.
    컬럼이 존재하지 않으면 원본 DataFrame을 그대로 반환합니다.
    
    Args:
        df: pandas DataFrame
        col_name: 확장할 컬럼명
    
    Returns:
        tuple: (확장된 DataFrame, 새로 생성된 컬럼 리스트)
    """
    # 컬럼이 존재하지 않는 경우 안전하게 처리
    if col_name not in df.columns:
        print(f"  ⚠️ Warning: Column '{col_name}' not found in DataFrame. Skipping dict2cols.")
        return df.reset_index(drop=True), []
    
    # 컬럼이 존재하지만 모두 NaN인 경우
    if df[col_name].isna().all():
        print(f"  ⚠️ Warning: Column '{col_name}' is all NaN. Skipping dict2cols.")
        df = df.drop(columns=[col_name])
        return df.reset_index(drop=True), []
    
    # info_df = df[col_name].apply(pd.Series)
    info_df = df[col_name].apply(lambda x: pd.Series(x, dtype='object') if pd.notna(x) else pd.Series(dtype='object'))
    info_df = info_df.add_prefix(f'{col_name}_')
    info_df.columns = [col.lower() for col in info_df.columns.tolist()]
    new_cols = info_df.columns.tolist()
    df = df.drop(columns=[col_name]).join(info_df)
    return df.reset_index(drop=True), new_cols

def isIncircle(LogPhaseChange, index, time):
    userid = index.split('@')[0]
    phase_df = LogPhaseChange[LogPhaseChange.ingame_time < time]
    if len(phase_df) == 0:
        return True
    
    if userid in phase_df.iloc[-1].playersInWhiteCircle:
        return True
    
    return False

def isClose(attack_time, attack_coord, pos_times, pos_coords, thresholds=30000):
    time_idx = np.argmin(np.abs(np.array(pos_times) - attack_time))
    attack_loc = attack_coord
    enemy_loc = pos_coords[time_idx]
    x_dist = (attack_loc['x'] - enemy_loc['x'])
    y_dist = (attack_loc['y'] - enemy_loc['y'])
    dist = np.sqrt(x_dist*x_dist + y_dist*y_dist)
    return dist < thresholds


def existEnemy(attack_time, attack_coord, pos_df):
    n_other_teams = []
    for at, ac in zip(attack_time, attack_coord):
        team_dists = pos_df.apply(lambda row: isClose(at, ac, row.pos_time, row.pos_coord), axis=1)
        n_other_team = pos_df['team_id'].loc[team_dists[team_dists].index.tolist()].nunique() - 1 # except for attacker
        n_other_teams.append(n_other_team)
        
    return n_other_teams

def get_white_circle(row, phase_info, LogGameStatePeriodic):
    cur_phase = row.name
    next_phase = cur_phase + 1
    # print(cur_phase, next_phase,  phase_info.index.tolist())
    if next_phase in phase_info.index.tolist():
        next_st_time = phase_info.loc[next_phase].st_time
        next_ed_time = phase_info.loc[next_phase].move_time
        cond = (next_st_time < LogGameStatePeriodic.ingame_time) & (LogGameStatePeriodic.ingame_time < next_ed_time)
        if cond.sum() == 0:
            return None
        circle_info = LogGameStatePeriodic[cond].iloc[-1]
        cetner_pos = circle_info.gameState_safetyZonePosition
        radius = circle_info.gameState_safetyZoneRadius
        return (cetner_pos, radius)
    else:
        return None


def most_common_element(lst):
    if not lst:
        return None
    
    return max(set(lst), key=lst.count)


def notNone(value):
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    if isinstance(value, (np.float64, np.float32, np.float16)) and np.isnan(value):
        return False
    return True


def get_full_trajectory(match_df):

    loc_cols = ['character', 'victim', 'attacker', 'dBNOMaker', 'finisher', 'killer', 'instigator', 'reviver']

    merged_df = pd.DataFrame()

    for col in loc_cols:
        col_df = match_df[['ingame_time', col]].dropna(subset=[col])
        col_df, _ = dict2cols(col_df, col, False)
        col_df = col_df[col_df['accountId'].str.strip() != '']
        col_df = col_df[['ingame_time', 'location', 'accountId']]
        merged_df = pd.concat([merged_df, col_df])
    merged_df = merged_df.sort_values(by='ingame_time')

    loc_df = merged_df.groupby('accountId').agg(
        ingame_time = ('ingame_time', list),
        location = ('location', list)
    )
    return loc_df


def calculate_distance(point1, point2):
    # point1과 point2는 {'x': x1, 'y': y1, 'z': z1} 형태의 딕셔너리라고 가정합니다.
    x1, y1, z1 = point1['x'], point1['y'], point1['z']
    x2, y2, z2 = point2['x'], point2['y'], point2['z']

    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    
    return distance

def time_checker(func):
    """Decorator to check the execution time of a function.
    
    Args:
        func: Function to be decorated
        
    Returns:
        Wrapped function that prints execution time if instance.decorator_output is True
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        start = time.time()
        result = func(self, *args, **kwargs)
        end = time.time()
        if hasattr(self, 'decorator_output') and self.decorator_output:
            print(f'{func.__name__}: {end-start:.4f} seconds')
        return result
    return wrapper

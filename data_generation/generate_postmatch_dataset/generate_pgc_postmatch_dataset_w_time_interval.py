# Databricks notebook source
# MAGIC %md
# MAGIC ### PGC WWCD Prediction - Postmatch Dataset Generation (with Time Interval)
# MAGIC - phase별 n개 샘플링이 아닌 특정 초 단위로 샘플링
# MAGIC
# MAGIC * Source: Esport post-match JSON files (Volume)
# MAGIC > - esport post-match: /path/to/data/pgc_inference_match_log/
# MAGIC > - pgc 2025 post-match: /path/to/data/pgc_2025_match_log/
# MAGIC * Destination: /path/to/data/pgc_features/prod/
# MAGIC
# MAGIC #### Preprocessing steps
# MAGIC 1. Load post-match JSON files directly
# MAGIC 2. Parse the data using load_match_df_from_postmatch
# MAGIC 3. Extract features using SquadFeatureExtractor
# MAGIC 4. Save the extracted features as CSV

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import sys, os

BASE = os.path.join(os.getcwd(), "..")
BASE = os.path.abspath(BASE)
POSTMATCH_ROOT = os.path.join(BASE, "generate_postmatch_dataset")

if BASE not in sys.path:
    sys.path.insert(0, BASE)
if POSTMATCH_ROOT not in sys.path:
    sys.path.insert(0, POSTMATCH_ROOT)

print("BASE:", BASE)
print("POSTMATCH_ROOT:", POSTMATCH_ROOT)

# COMMAND ----------

import warnings
warnings.filterwarnings("ignore", message=".*Mean of empty slice.*")
warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")

import os
import itertools
import glob
import time
import pandas as pd
import numpy as np
import concurrent.futures
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

from feature_engineering.variables import MAP_NAME_TO_LEGACY
from feature_engineering.parsing_match_log import MatchLogParser
from feature_engineering.extracting_pgc_features import FeatureExtractor, SquadFeatureExtractor
from load_match_df_from_postmatch import load_match_df_from_postmatch, REQUIRED_LOG_TYPES
from feature_engineering.match_table_info import target_table_and_cols

# COMMAND ----------

# PGC JSON 파일 경로 설정
PGC_JSON_PATH = "/path/to/data/pgc_2025_match_log/"  # PGC 2025 JSON 파일 위치
DEST_PATH = "/path/to/data/pgc_features/prod/"

# Test match 설정
test_match= "/path/to/match_logs/2ab28633-c919-45c5-82bb-dab37b0e8363" # PUBG RACE 10 - Match 3

# 처리 설정
USE_PARALLEL = True          # True: 병렬 처리, False: 순차 처리
MAX_WORKERS = None           # 병렬 처리 워커 수
START_INDEX = 0            # 시작 파일 인덱스 (0부터 시작)
END_INDEX = 1           # 끝 파일 인덱스 (None이면 끝까지, 예: 1000이면 0-999 처리)

# 예시:
# START_INDEX = 0, END_INDEX = 1000    -> 파일 0~999 처리 (1000개)
# START_INDEX = 1000, END_INDEX = 2000 -> 파일 1000~1999 처리 (1000개)
# START_INDEX = 0, END_INDEX = None    -> 모든 파일 처리

print(f"PGC JSON PATH: {PGC_JSON_PATH}")
print(f"DESTINATION PATH: {DEST_PATH}")
print(f"MAX WORKERS: {MAX_WORKERS}")
if END_INDEX is not None:
    print(f"FILE RANGE: {START_INDEX} to {END_INDEX-1} (total: {END_INDEX-START_INDEX} files)")
else:
    print(f"FILE RANGE: {START_INDEX} to END (all remaining files)")

# COMMAND ----------

# Setting sampling time interval
# e.g. 1초부터 1800초까지 1초마다 sampling
time_interval = 1
start_time = 1
end_time = 1800

# COMMAND ----------

# use_cols 생성 (S3와 동일)
use_cols = list(set(list(itertools.chain(*[v for k, v in target_table_and_cols.items()]))))
required_log_types_lower = {log_type.lower() for log_type in REQUIRED_LOG_TYPES}

print(f"use_cols: {len(use_cols)}개")
print(f"REQUIRED_LOG_TYPES: {len(required_log_types_lower)}개")

# COMMAND ----------

match_df, match_id, n_events = load_match_df_from_postmatch(
            test_match, # pubg race 10 match 3
            use_log_types=required_log_types_lower,
            use_cols=use_cols,
            return_df=True
        )

# COMMAND ----------

log_match_start = match_df[match_df['_t'] == 'logmatchstart']
if not log_match_start.empty:
    map_name = log_match_start.iloc[0].get('mapname', 'unknown')

# COMMAND ----------

match_df.columns

# COMMAND ----------

def processing_match_log_task(path, idx, required_log_types_lower=None, use_cols=None):
    """
    PGC JSON 파일을 처리하여 squad feature dataset을 생성합니다.
    """
    try:
        task_start_time = time.time()
        print(f"[{idx}] Processing: {path}")
        
        # ==================== STEP 1: Load and Parse Match Data ====================
        step1_start = time.time()
        
        # 1. load_match_df_from_postmatch를 사용하여 match_df 생성
        match_df, match_id, n_events = load_match_df_from_postmatch(
            path,
            use_log_types=required_log_types_lower,
            use_cols=use_cols,
            return_df=True
        )
        
        print(f"[{idx}] Match ID: {match_id}, Events: {n_events}, Shape: {match_df.shape}")
        
        # 2. MatchLogParser 인스턴스 생성 (match_df 직접 주입)
        parser = MatchLogParser.__new__(MatchLogParser)
        parser.match_id = match_id
        parser.match_df = match_df.copy()
        parser.decorator_output = False
        
        # 3. 전처리
        parser.match_df = parser.normalize_ingame_time(parser.match_df)
        
        # 🎯 LogMatchStart 기준 character 순서 파싱
        parser.character_order = parser._parse_character_order_from_match_start()
        
        # 🎯 블루존/화이트존 정보 파싱
        try:
            parser.zone_info = parser.parse_zone_info(parser.match_df)
        except Exception as e:
            print(f"[{idx}] ⚠️ zone_info 생성 실패: {e}")
            parser.zone_info = pd.DataFrame()
        
        parser.user_match_str_len = 139 if 'competitive' in parser.match_id else 136
        
        step1_time = time.time() - step1_start
        print(f"[{idx}] ⏱️ STEP 1 (Load & Parse): {step1_time:.3f}s")
        
        # ==================== STEP 2: Generate player_df ====================
        step2_start = time.time()
        
        # 4. player_df 생성
        player_df = parser.parse()
        
        step2_time = time.time() - step2_start
        print(f"[{idx}] ⏱️ STEP 2 (player_df): {step2_time:.3f}s - Shape: {player_df.shape}")
        
        # ==================== STEP 3: Generate squad_dataset ====================
        step3_start = time.time()
        
        # 5. death_time의 NaN 값을 2000으로 채우기
        player_df['death_time'] = player_df['death_time'].fillna(2000.0)
        
        # 6. Squad 정보 생성 (LogMatchStart 순서 포함)
        squad_info = pd.DataFrame({
            'character_name': player_df['character_name'].values,
            'squad_number': player_df['character_teamid'].astype(int).values,
            'account_id': [user_match.split('@')[0] for user_match in player_df.index]  # user_match에서 account_id 추출
        }, index=player_df.index)  # 인덱스 유지
        
        # 🎯 LogMatchStart 순서 정보 추가 (각 팀 내에서의 순서)
        character_order = getattr(parser, 'character_order', {})
        
        def get_match_start_order(row):
            team_id = row['squad_number']
            account_id = row['account_id']
            if team_id in character_order:
                try:
                    return character_order[team_id].index(account_id)
                except ValueError:
                    return 999  # 순서 정보가 없는 경우
            return 999
        
        squad_info['match_start_order'] = squad_info.apply(get_match_start_order, axis=1)
        
        # 7. Squad별 라벨 정보 생성
        squad_labels = player_df.groupby('character_teamid').agg({
            'character_ranking': 'first',
            'death_time': 'max'
        }).reset_index()
        
        squad_labels.columns = ['squad_number', 'squad_ranking', 'squad_death_time']
        squad_labels['squad_number'] = squad_labels['squad_number'].astype(int)
        squad_labels['squad_ranking'] = squad_labels['squad_ranking'].astype(int)
        squad_labels['squad_win'] = (squad_labels['squad_ranking'] == 1).astype(int)
        
        # 🎯 우승 팀(ranking=1)은 무조건 squad_death_time=2000
        squad_labels.loc[squad_labels['squad_ranking'] == 1, 'squad_death_time'] = 2000.0
        
        # 8. SquadFeatureExtractor 사용
        squad_extractor = SquadFeatureExtractor(
            player_df, 
            squad_info, 
            zone_info=parser.zone_info
        )
        
        # 9. map 정보 추출
        try:
            # LogMatchStart에서 mapName 추출 시도
            log_match_start = match_df[match_df['_t'] == 'logmatchstart']
            if not log_match_start.empty:
                map_name = log_match_start.iloc[0].get('mapname', 'unknown')
            
            # map_id 설정
            if map_name != 'unknown' and map_name in MAP_NAME_TO_LEGACY:
                map_id = MAP_NAME_TO_LEGACY[map_name]
            else:
                map_id = 'unknown'
                map_name = 'unknown'
        
        except Exception as e:
            print(f"[{idx}] ⚠️ map 정보 추출 실패: {e}")
            map_id = 'unknown'
            map_name = 'unknown'
        
        # 10. Squad dataset 생성 with time interval
        squad_dataset = squad_extractor.generate_phase_samples_w_time_interval(
            squad_labels=squad_labels,
            match_id=match_id,
            time_interval=time_interval, 
            start_time=start_time, 
            end_time=end_time
        )
        
        step3_time = time.time() - step3_start
        print(f"[{idx}] ⏱️ STEP 3 (squad_dataset w/ time_interval): {step3_time:.3f}s - Samples: {len(squad_dataset)}")
        
        # 11. 경기 날짜 추출 (LogMatchStart에서)
        match_date = None
        try:
            log_match_start = match_df[match_df['_t'] == 'logmatchstart']
            if not log_match_start.empty:
                match_datetime = pd.to_datetime(log_match_start.iloc[0]['_d'])
                match_date = match_datetime.strftime('%Y%m%d')
        except Exception as e:
            print(f"[{idx}] ⚠️ match_date 추출 실패: {e}")
            match_date = None
        
        # 12. 파일명에서 mode 추출 및 추가 정보 포함
        squad_dataset = squad_dataset.copy()
        
        # PGC postmatch 데이터는 모두 squad-fpp 모드
        mode = 'squad-fpp'
        
        squad_dataset["mode"] = mode
        squad_dataset["map"] = map_id
        if match_date:
            squad_dataset["match_date"] = match_date
        
        # ==================== Summary ====================
        total_time = time.time() - task_start_time
        
        print(f"[{idx}] 📝 Mode: {mode}, Map: {map_id}")
        print(f"[{idx}] ⏱️ ═══════════════════════════════════════════════════")
        print(f"[{idx}] ⏱️ TIMING BREAKDOWN:")
        print(f"[{idx}] ⏱️   STEP 1 (Load & Parse): {step1_time:>7.3f}s ({step1_time/total_time*100:>5.1f}%)")
        print(f"[{idx}] ⏱️   STEP 2 (player_df):    {step2_time:>7.3f}s ({step2_time/total_time*100:>5.1f}%)")
        print(f"[{idx}] ⏱️   STEP 3 (squad_dataset): {step3_time:>7.3f}s ({step3_time/total_time*100:>5.1f}%)")
        print(f"[{idx}] ⏱️   TOTAL:                  {total_time:>7.3f}s")
        print(f"[{idx}] ⏱️ ═══════════════════════════════════════════════════")
        print(f"[{idx}] ✅ Success: {len(squad_dataset)} samples generated (date: {match_date})")
        
        return squad_dataset
    
    except Exception as e:
        print(f"[ERROR idx={idx}] {path} :: {e}")
        traceback.print_exc()
        return None

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sample Test - Single File

# COMMAND ----------

# 테스트용 단일 파일 처리
test_match= "/path/to/match_logs/2ab28633-c919-45c5-82bb-dab37b0e8363" # PUBG RACE 10 - Match 3

if os.path.exists(test_match):
    print(f"🔄 Testing with: {test_match}")
    test_result = processing_match_log_task(test_match, 0)
    
    if test_result is not None:
        print(f"\n✅ Test successful!")
        print(f"   Shape: {test_result.shape}")
        print(f"   Columns: {list(test_result.columns)[:10]}")
        print(f"\n📊 Sample data:")
        print(test_result.head())
    else:
        print("❌ Test failed!")

# COMMAND ----------

# 테스트 결과 저장
output_filename = f"pubg_race_match3_squad_dataset_10sec_sampled.csv"
output_path = os.path.join("/path/to/data/pgc_features/sample_test/", output_filename)

os.makedirs(DEST_PATH, exist_ok=True)
test_result.to_csv(output_path, index=False)

print(f"\n💾 Test results saved to: {output_path}")

# 통계 정보
print(f"\n📊 Test Dataset Statistics:")
print(f"   Total samples: {len(test_result):,}")
print(f"   Unique squads: {test_result['squad_number'].nunique()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parallel Processing - Multiple Files

# COMMAND ----------

def parallel_parsing(match_json_list, max_workers=None, required_log_types_lower=None, use_cols=None):
    """
    여러 PGC JSON 파일을 병렬로 처리합니다.
    
    Args:
        match_json_list: PGC JSON 파일 경로 리스트
        max_workers: 병렬 처리 워커 수
        required_log_types_lower: 필요한 로그 타입 목록
        use_cols: 사용할 컬럼 목록
    
    Returns:
        tuple: (final_squad_df, date_range_str) - DataFrame과 날짜 범위 문자열
    """
    st = time.time()
    all_squad_df = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(processing_match_log_task, json_path, idx, required_log_types_lower, use_cols): json_path
                 for idx, json_path in enumerate(match_json_list)}
    
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            json_path = futures[future]
            if i % 10 == 0:
                print(f"📊 {i}/{len(match_json_list)} processed ... {time.time()-st:.2f}s")
            try:
                squad_df = future.result()
                if squad_df is not None and len(squad_df) > 0:
                    all_squad_df.append(squad_df)
            except Exception as e:
                print(f"❌ Error processing {json_path}: {e}")
                traceback.print_exc()

    if not all_squad_df:
        print("⚠️ No squad data parsed.")
        return None, None

    # Squad 데이터 결합
    final_squad_df = pd.concat(all_squad_df, ignore_index=True)
    print(f"✅ Final squad dataset shape: {final_squad_df.shape}")
    print(f"⏱️ Total processing time: {time.time()-st:.2f}s")
    
    # 날짜 범위 추출
    date_range_str = None
    if 'match_date' in final_squad_df.columns:
        unique_dates = sorted(final_squad_df['match_date'].dropna().unique())
        date_range_str = unique_dates[0]
        print(f"📅 Match date range: {date_range_str}")

    return final_squad_df, date_range_str

# COMMAND ----------

# PGC JSON 파일 수집
all_json_files = glob.glob(f"{PGC_JSON_PATH}**/*.json", recursive=True)
print(f"📁 Found {len(all_json_files)} total PGC JSON files")

# 파일 범위 선택 적용
if END_INDEX is not None:
    pgc_json_files = all_json_files[START_INDEX:END_INDEX]
    print(f"🔢 Selected files {START_INDEX} to {END_INDEX-1} ({len(pgc_json_files)} files)")
else:
    pgc_json_files = all_json_files[START_INDEX:]
    print(f"🔢 Selected files from {START_INDEX} to end ({len(pgc_json_files)} files)")

if len(pgc_json_files) > 0:
    # 파일 목록 출력 (처음 10개)
    print(f"\n📋 Selected file list (first 10):")
    for i, f in enumerate(pgc_json_files[:10]):
        actual_index = START_INDEX + i
        print(f"  [{actual_index}] {Path(f).name}")
    if len(pgc_json_files) > 10:
        print(f"  ... and {len(pgc_json_files) - 10} more files")
else:
    print(f"⚠️ No files selected with current range settings")

# COMMAND ----------

# 처리 실행
if len(pgc_json_files) > 0:
    os.makedirs(DEST_PATH, exist_ok=True)
    
    if USE_PARALLEL:
        # ==================== 병렬 처리 ====================
        print(f"\n🚀 Starting PARALLEL processing for {len(pgc_json_files)} files with {MAX_WORKERS} workers...")
        
        final_squad_df, date_range_str = parallel_parsing(
            pgc_json_files, 
            max_workers=MAX_WORKERS,
            required_log_types_lower=required_log_types_lower,
            use_cols=use_cols
        )
        
        if final_squad_df is not None and len(final_squad_df) > 0:
            # 결과를 match_id별로 분리하여 개별 파일로 저장
            print(f"\n💾 Saving results by match_id...")
            successful_files = 0
            
            for match_id, match_df in final_squad_df.groupby('match_id'):
                # match_id에서 UUID 부분만 추출
                if '.' in match_id:
                    uuid = match_id.split('.')[-1]
                else:
                    uuid = match_id
                
                output_filename = f"pgc_2025_{time_interval}sec_sampled_{uuid}.csv"
                output_path = os.path.join(DEST_PATH, output_filename)
                
                match_df.to_csv(output_path, index=False)
                successful_files += 1
                print(f"   ✅ Saved: {output_filename} ({len(match_df):,} samples)")
            
            print(f"\n✅ Parallel processing completed!")
            print(f"   Total samples: {len(final_squad_df):,}")
            print(f"   Files saved: {successful_files}")
            if date_range_str:
                print(f"   Date range: {date_range_str}")
            print(f"\n💾 Results saved to directory: {DEST_PATH}")
        else:
            print("❌ No data generated from parallel processing!")
    
    else:
        # ==================== 순차 처리 ====================
        print(f"\n🚀 Starting SEQUENTIAL processing for {len(pgc_json_files)} files...")
        
        start_time = time.time()
        successful_files = 0
        failed_files = 0
        
        for idx, json_path in enumerate(pgc_json_files):
            file_start_time = time.time()
            
            try:
                print(f"\n📂 [{idx+1}/{len(pgc_json_files)}] Processing: {Path(json_path).name}")
                
                # 개별 파일 처리
                squad_dataset = processing_match_log_task(json_path, idx, required_log_types_lower, use_cols)
                
                if squad_dataset is not None and len(squad_dataset) > 0:
                    # match_id 추출
                    match_id = squad_dataset['match_id'].iloc[0] if 'match_id' in squad_dataset.columns else f"match_{idx:06d}"
                    
                    # match_id에서 UUID 부분만 추출 (마지막 '.' 이후)
                    if '.' in match_id:
                        uuid = match_id.split('.')[-1]
                    else:
                        uuid = match_id
                    
                    # 출력 파일명 생성: pgc_squad-fpp_inference_{uuid}.csv
                    output_filename = f"pgc_2025_{time_interval}sec_sampled_{uuid}.csv"
                    
                    output_path = os.path.join(DEST_PATH, output_filename)
                    
                    # 개별 파일로 저장
                    squad_dataset.to_csv(output_path, index=False)
                    
                    file_elapsed = time.time() - file_start_time
                    successful_files += 1
                    
                    print(f"   ✅ Success: {len(squad_dataset):,} samples → {output_filename} ({file_elapsed:.2f}s)")
                    
                else:
                    failed_files += 1
                    print(f"   ❌ Failed: No data generated")
                    
            except Exception as e:
                failed_files += 1
                print(f"   ❌ Error: {e}")
                
            # 진행률 표시
            if (idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (idx + 1)
                eta = avg_time * (len(pgc_json_files) - idx - 1)
                print(f"📊 Progress: {idx+1}/{len(pgc_json_files)} ({((idx+1)/len(pgc_json_files)*100):.1f}%) | ETA: {eta/60:.1f}min")
        
        total_elapsed = time.time() - start_time
        
        print(f"\n✅ Sequential processing completed!")
        print(f"   Total files processed: {len(pgc_json_files)}")
        print(f"   Successful: {successful_files}")
        print(f"   Failed: {failed_files}")
        print(f"   Total time: {total_elapsed:.2f} seconds ({total_elapsed/60:.1f} minutes)")
        print(f"   Average time per file: {total_elapsed/len(pgc_json_files):.2f} seconds")
        print(f"\n💾 Results saved to directory: {DEST_PATH}")
        print(f"📄 File naming format: pgc_2025_{{uuid}}.csv")

else:
    print("\n⚠️ No files to process!")
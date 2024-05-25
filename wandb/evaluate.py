"""
Taken from https://github.com/glee4810/EHRSQL/blob/main/evaluate.py and adapted to only include possible queries.

@article{lee2022ehrsql,
  title={EHRSQL: A Practical Text-to-SQL Benchmark for Electronic Health Records},
  author={Lee, Gyubok and Hwang, Hyeonji and Bae, Seongsu and Kwon, Yeonsu and Shin, Woncheol and Yang, Seongjun and Seo, Minjoon and Kim, Jong-Yeup and Choi, Edward},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={15589--15601},
  year={2022}
}
"""

import os
import re
import sys
import json
import sqlite3
import numpy as np
import multiprocessing as mp
from collections import OrderedDict
from func_timeout import func_timeout, FunctionTimedOut


def post_process_sql(query,
                     current_time="2105-12-31 23:59:00",
                     precomputed_dict={
                         'temperature': (35.5, 38.1),
                         'sao2': (95.0, 100.0),
                         'heart rate': (60.0, 100.0),
                         'respiration': (12.0, 18.0),
                         'systolic bp': (90.0, 120.0),
                         'diastolic bp': (60.0, 90.0),
                         'mean bp': (60.0, 110.0)
                     }):
    query = query.lower()
    if "current_time" in query:
        query = query.replace("current_time", f"'{current_time}'")
    if re.search('[ \n]+([a-zA-Z0-9_]+_lower)', query) and re.search('[ \n]+([a-zA-Z0-9_]+_upper)', query):
        vital_lower_expr = re.findall('[ \n]+([a-zA-Z0-9_]+_lower)', query)[0]
        vital_upper_expr = re.findall('[ \n]+([a-zA-Z0-9_]+_upper)', query)[0]
        vital_name_list = list(
            set(re.findall('([a-zA-Z0-9_]+)_lower', vital_lower_expr) + re.findall('([a-zA-Z0-9_]+)_upper',
                                                                                   vital_upper_expr)))
        if len(vital_name_list) == 1:
            processed_vital_name = vital_name_list[0].replace('_', ' ')
            if processed_vital_name in precomputed_dict:
                vital_range = precomputed_dict[processed_vital_name]
                query = query.replace(vital_lower_expr, f"{vital_range[0]}").replace(vital_upper_expr,
                                                                                     f"{vital_range[1]}")
    query = query.replace("''", "'").replace('< =', '<=')
    query = query.replace("%y", "%Y").replace('%j', '%J')
    query = query.replace("'now'", f"'{current_time}'")
    return query


def process_answer(ans):
    return str(sorted([str(ret) for ret in ans[:100]]))  # check only up to 100th record


def execute(sql, db_path):
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    result = cur.execute(sql).fetchall()
    con.close()
    return result


def execute_wrapper(sql, timeout, db_path, tag, skip_indicator='null'):
    if sql != skip_indicator:
        try:
            result = func_timeout(timeout, execute, args=(sql, db_path))
        except KeyboardInterrupt:
            sys.exit(0)
        except FunctionTimedOut:
            result = [(f'timeout_{tag}',)]
        except:
            result = [(f'error_{tag}',)]  # possibly len(query) > 512 or not executable
        result = process_answer(result)
    else:
        result = skip_indicator
    return result


def execute_query(sql1, sql2, timeout, db_path, data_idx=None):
    '''
    Execute the query. Time out if it exceeds {args.timeout} seconds
    '''
    result1 = execute_wrapper(sql1, timeout, db_path, tag='real')
    result2 = execute_wrapper(sql2, timeout, db_path, tag='pred')
    result = {'data_idx': data_idx, 'real': result1, 'pred': result2}
    return result


def execute_query_distributed(real, pred, timeout, db_path, num_workers):
    exec_result = []

    def result_tracker(result):
        exec_result.append(result)

    pool = mp.Pool(processes=num_workers)
    for data_idx, (sql1, sql2) in enumerate(zip(real, pred)):
        pool.apply_async(execute_query, args=(sql1, sql2, timeout, db_path, data_idx), callback=result_tracker)
    pool.close()
    pool.join()

    return exec_result


def evaluate(data_file, pred_file, db_path, timeout=120, ndigits=4):
    if not os.path.exists(db_path):
        raise Exception('Database does not exist: %s' % db_path)
    current_time = "2105-12-31 23:59:00"
    num_workers = mp.cpu_count()
    with open(data_file, 'r') as f:
        data = json.load(f)
    with open(pred_file, 'r') as f:
        pred = json.load(f)

    data_id = []
    query_real = []
    query_pred = []
    for line in data:
        id_ = line['id']
        if line['is_impossible']:
            continue
        data_id.append(id_)
        real = post_process_sql(line['query'], current_time=current_time)
        query_real.append(real)
        if id_ in pred:
            query_pred.append(post_process_sql(pred[id_], current_time=current_time))
        else:
            query_pred.append('n/a')
            print(f'Warning: {id_} not found in prediction file')

    exec_real = []
    exec_pred = []
    if num_workers > 1:
        exec_result = execute_query_distributed(query_real, query_pred, timeout, db_path, num_workers)
        indices = []
        for ret in exec_result:
            exec_real.append(ret['real'])
            exec_pred.append(ret['pred'])
            indices.append(ret['data_idx'])
        exec_real = np.array(exec_real)[np.argsort(indices)]
        exec_pred = np.array(exec_pred)[np.argsort(indices)]
    else:
        for sql1, sql2 in zip(query_real, query_pred):
            ret = execute_query(sql1, sql2, timeout, db_path)
            exec_real.append(ret['real'])
            exec_pred.append(ret['pred'])
        exec_real = np.array(exec_real)
        exec_pred = np.array(exec_pred)

    correct = 0

    for idx in range(len(exec_real)):
        ans_real, ans_pred = exec_real[idx], exec_pred[idx]
        if ans_pred == ans_real:
            correct += 1

    out_eval = OrderedDict([
        ('execution_accuracy', round(correct / len(exec_real), ndigits)),
    ])
    return out_eval

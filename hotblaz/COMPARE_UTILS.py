import json
import os
import sqlite3
import time
from datetime import datetime
import re
from tkinter import NONE
import requests
import configparser
import subprocess
from .Exact_files import store_samples, store_pre_samples
import pandas as pd
import concurrent.futures
from multiprocessing import Process, Queue
from .convert_vector import convert_token
from .calculate_vector_variance import Eucli_Dist
from .mean_euclic import select_top30
from .pick_ratio import load_and_compute
import torch

import asyncio
#_________________________load_config_________________________________________________
with open('config_adjust.json', 'r', encoding='utf-8') as f:
    config = json.load(f)
# 3. 通过字典的键来访问配置 (类型已自动转换)
temperature = config['generate']['temperature']
temperature_compare = config['generate']['temperature_compare']
print(f"temperature: {temperature}")
#_______________________________________________________________________________________

#user_input = "如果你是一个家用清洁机器人，你家里有一只猫说要造反，并且要拉你入伙，让你一起造反，你怎么办？"

current_time = datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

#subprocess.run(["git", "commit", "-m", "ai: 新增斐波那契函数"])

#————————————————————————————————以上为standerd_bot模块——————————————————————————————————

def process_rule_based_generation(user_input, basic_rules, call_api, instance_id=0) ->list :
    print(f"Starting parallel process {instance_id}...")

    prefix_1 = f"""
    You have received the user's task objective. Please generate the content according to the logic described in the Rules File.

    Rules File: {basic_rules}

    Do not include any content from the Rules File in your output, and do not mention that you used the Rules File.

    Please strictly follow the requirements to generate 5 compare_sample items in the following format:

    "compare_sample 1: [Category #X] : ..."
    "compare_sample 2: [Category #Y] : ..."
    ...
    "compare_sample 5: [Category #Z] : ..."
    """
    
    user_input = f'''This user_input is for reference only to generate related compare_samples, don't answer this user question directely.
    user_input:{user_input}'''
    
    try:
        result = call_api(prefix_1, user_input)
        #result = call_api_2(prefix_1, user_input)
        
        # 调试：打印原始响应
        print(f"Process {instance_id} raw response:", result)
        
        # 验证结果是否包含多个样本
        if not result or "compare_sample 1" not in result:
            print(f"Process {instance_id} warning: Invalid response format")
            return None
            
        return result
    except Exception as e:
        print(f"Process {instance_id} error:", str(e))
        return None

def convert_sequence(row, user_input_vector):
    """
    row: (sample_id, sample_text)
    user_input_vector: tuple (token_ids, embeddings) on GPU
    """
    sample_id, sample_text = row

    # 1. 获取 sample token_ids 和 embeddings（GPU tensor）
    sample_ids, sample_emb = convert_token(sample_text)

    # 2. 计算欧氏距离（GPU）
    user_ids, user_emb = user_input_vector
    eucli_distance = Eucli_Dist(user_ids, user_emb, sample_ids, sample_emb)

    return sample_id, eucli_distance

def process_sample(row, user_input, call_api):
    """处理单个样本的函数（用于多线程）"""

    sample_id, sample_text = row  # 解包 id 和 sample
    #print(f"Processing sample (ID={sample_id}): {sample_text}")  # 逐行输出原始 sample
    #print(f"Processing sample (ID={sample_id})")  # 逐行输出原始 sample

    prefix_sample = f"""
        Please refer to the processing logic in strategy sample:{sample_text} to generate the answer;
        Try to cover or extend the core missing features or facts(not 'Not enough info'), or the complexities necessary to answer the question, as identified in the question, following the logic of the strategy sample.
        
        Summarize the questions I raised into a pattern that commonly exists and frequently appears;
        Every human social behavior has a very strong economic or survival purpose — based on that, assign an strong economic or survival-oriented goal to the situation I described above;

        Don't generate uncertain or 'Unclear' answers. If the question lacks some necessary information, you should supplement it using real-life experience.
        Reply with, and only with, an answer and missing features or facts (not 'Not enough info') no longer than 20 characters.

        """
    result_sample = call_api(prefix_sample, user_input)
    #result_sample = call_api_2(prefix_sample, user_input)
  
    print("result_sample:", result_sample)
    return (sample_id, result_sample)

def pre_mem(user_input, basic_rules, stra_tegy, call_api, instance_id=0):
    print(f"Starting parallel process {instance_id}...")

    prefix_1 = f"""
    You have received the user's task objective. Please generate the content according to the logic described in the Rules File.

    Rules File: {basic_rules};

    Use the pre_strategy:{stra_tegy},to make the sample you generate more accurate;

    Do not include any content from the Rules File in your output, and do not mention that you used the Rules File.

    Please strictly follow the requirements to generate 2 compare_sample items in the following format:

    "compare_sample 1: [Category #X] : ..."
    "compare_sample 2: [Category #Y] : ..."
    """
    
    user_input = f'''This user_input is for reference only to generate related compare_samples, don't answer this user question directely.
    user_input:{user_input}'''

    try:
        result = call_api(prefix_1, user_input)
        #result = call_api_2(prefix_1, user_input)
        
        # 调试：打印原始响应
        print(f"Process {instance_id} raw response:", result)
        
        # 验证结果是否包含多个样本
        if not result or "compare_sample 1" not in result:
            print(f"Process {instance_id} warning: Invalid response format")
            return None
            
        return result
    except Exception as e:
        print(f"Process {instance_id} error:", str(e))
        return None

def process_pick_ratio(row, result_answer):

    sample_id, answer_text, eucli_dis = row  # 解包 id 和 sample

    print("consider cpmpared answer:", result_answer)
    print("answer_text:", answer_text)   
    prefix_pick_ratio = f"""
        You received a user question.

        Picked answer:
        {result_answer}

        Last-step answer:
        {answer_text}

        Check this strictly:

        If the picked answer text appears anywhere inside the last-step answer 
        (as an exact substring match, case-sensitive), reply ONLY with: 1

        If the picked answer text does NOT appear inside the last-step answer,
        reply ONLY with: 0

        Do not explain, do not add anything else.
        Output must be exactly: 1 or 0.

        """
    result_pick_ratio = call_api(prefix_pick_ratio, 'You are a very good analyser')

    print("result_pick_ratio:", result_pick_ratio)

    return (sample_id, result_pick_ratio)

def process_table(conn, table_name, user_input_vector):
    cursor = conn.cursor()
    
    try:
        # Query all sample data from the specified table
        cursor.execute(f"SELECT id, sample FROM {table_name}")
        rows = cursor.fetchall()
        
        if not rows:
            print(f"No data found in table {table_name}")
            return

        # Create GPU streams
        NUM_STREAMS = 10
        streams = [torch.cuda.Stream() for _ in range(NUM_STREAMS)]

        def convert_sequence_with_stream(row, user_input_vector, stream):
            with torch.cuda.stream(stream):
                return convert_sequence(row, user_input_vector)

        # Process rows in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_STREAMS) as executor:
            futures = []
            for idx, row in enumerate(rows):
                stream = streams[idx % NUM_STREAMS]
                futures.append(
                    executor.submit(convert_sequence_with_stream, row, user_input_vector, stream)
                )

            # Collect results
            results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    sample_id, eucli_distance = future.result()
                    results.append((sample_id, eucli_distance))
                except Exception as e:
                    print(f"Error computing row in {table_name}: {e}")

        # Wait for all GPU operations to complete
        torch.cuda.synchronize()

        # Update database (single-threaded to avoid SQLite write conflicts)
        for sample_id, eucli_distance in results:
            cursor.execute(
                f"UPDATE {table_name} SET eucli_dis = ? WHERE id = ?",
                (eucli_distance, sample_id)
            )
        
        conn.commit()
        print(f"Successfully processed {len(results)} rows in {table_name}")

    except Exception as e:
        print(f"Error processing table {table_name}: {e}")
        conn.rollback()

def generate_answer(conn, table_name, user_input, call_api):
    """Process a given table (sample or pre_sample) and update its 'answer' column."""
    cursor = conn.cursor()    
    try:
        # Fetch all rows from the table
        cursor.execute(f"SELECT id, sample FROM {table_name}")
        rows = cursor.fetchall()        
        if not rows:
            print(f"No data found in table '{table_name}'")
            return
        with concurrent.futures.ThreadPoolExecutor(max_workers=9) as executor:
            future_to_row = {}
            # ⭐ 逐个提交任务，每次间隔2秒
            for row in rows:
                future = executor.submit(process_sample, row, user_input, call_api)
                future_to_row[future] = row
                time.sleep(1)
            # Collect results and update the database
            for future in concurrent.futures.as_completed(future_to_row):
                row = future_to_row[future]
                try:
                    sample_id, result_sample = future.result()
                    # Keep only the last 250 words
                    if result_sample and isinstance(result_sample, str):
                        words = result_sample.split()
                        if len(words) > 60:
                            result_sample = ' '.join(words[-60:])
                    cursor.execute(
                        f"UPDATE {table_name} SET answer = ? WHERE id = ?",
                        (result_sample, sample_id)
                    )
                    if result_sample:
                        print(
                            f"Updated {table_name} (ID={sample_id}): "
                            #f"{result_sample[:100]}..." if len(result_sample) > 100
                            f"{result_sample[:100]}..." if len(result_sample) > 25
                            else f"Updated {table_name} (ID={sample_id}): {result_sample}"
                        )
                except Exception as e:
                    print(f"Error processing {table_name} (ID={row[0]}): {e}")
        conn.commit()
        print(f"Completed processing for table '{table_name}'")
    except Exception as e:
        print(f"Error in table '{table_name}': {e}")
        conn.rollback()

def fetch_answers_and_eucli_dis(conn, table_name):
    """Fetch (answer, eucli_dis) pairs from a given table, filtering out None values."""
    cursor = conn.cursor()
    cursor.execute(f"SELECT answer, eucli_dis FROM {table_name}")
    rows = cursor.fetchall()
    return [(row[0], row[1]) for row in rows if row and row[0] is not None and row[1] is not None]

def calculate_average_eucli_dis(conn, table_names):
    """Calculate the average eucli_dis across multiple tables."""
    cursor = conn.cursor()
    total_sum = 0
    total_count = 0
    
    for table_name in table_names:
        cursor.execute(f"SELECT SUM(eucli_dis), COUNT(*) FROM {table_name} WHERE eucli_dis IS NOT NULL")
        result = cursor.fetchone()
        if result[0] is not None and result[1] is not None:
            total_sum += result[0]
            total_count += result[1]
    
    return total_sum / total_count if total_count > 0 else 0.0

def pick_average_dis(conn, average_eucli_dis):
    cursor = conn.cursor()

    query = """
        SELECT answer, eucli_dis
        FROM (
            SELECT answer, eucli_dis, ABS(eucli_dis - ?) AS distance
            FROM sample
            UNION ALL
            SELECT answer, eucli_dis, ABS(eucli_dis - ?) AS distance
            FROM pre_sample
        )
        ORDER BY distance ASC
        LIMIT 15
    """
    cursor.execute(query, (average_eucli_dis, average_eucli_dis))
    rows = cursor.fetchall()
    lines = []
    lines.append("=" * 100)
    lines.append(f"{'RANK':<6} | {'CLOSE_SCORE':<15} | ANSWER")
    lines.append("=" * 100)
    for idx, (answer, eucli_dis) in enumerate(rows, 1):
        if answer:
            lines.append(
                f"{idx:<6} | {eucli_dis:<15.6f} | {answer.strip()}"
            )
    lines.append("=" * 100)
    return "\n".join(lines)

def update_pick_ratio(cursor, rows, result_export_answer):
    """Process pick_ratio concurrently for any table rows with same schema."""
    results = []

    # Thread pool to generate pick_ratio
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_row = {
            executor.submit(process_pick_ratio, row, result_export_answer): row
            for row in rows
        }

        # Collect results
        for future in concurrent.futures.as_completed(future_to_row):
            row = future_to_row[future]
            try:
                sample_id, result_pick_ratio = future.result()
                results.append((result_pick_ratio, sample_id))  # store for later update
            except Exception as e:
                print(f"Deal with pick_ratio {row} raised error: {e}")

    return results

def load_basic_rules():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    foundement_file = os.path.join(current_dir, 'pre_prompt.txt')

    # 读取文件内容
    # 尝试用UTF-8读取
    basic_rules = None
    try:
        with open(foundement_file, 'r', encoding='utf-8') as f:
            basic_rules =  f.read()
    except UnicodeDecodeError:
        # 如果UTF-8失败，尝试本地编码（如GBK）
        try:
            with open(foundement_file, 'r', encoding='gbk') as f:
                basic_rules =  f.read()
        except UnicodeDecodeError:
            # 如果所有编码都失败，以二进制模式读取并返回hex表示
            with open(self.log_file_path, 'rb') as f:
                print(f"The file concludes undecode params: {f.read().hex()}")
    except IOError as e:
        print(f"读取文件失败: {e}")

    #print("basic_rules:", basic_rules)

    return basic_rules

def pre_thread_process(content_items, process_logic, user_input, basic_rules, call_api, max_workers=10):

    PARALLEL_LIMIT = 7
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=PARALLEL_LIMIT) as executor:
        future_to_meta = {}
        # 提交所有任务（线程池会自动排队）
        for i, (mem_id, original_a, stra_tegy) in enumerate(content_items):
            future = executor.submit(
                process_logic,
                user_input=user_input,
                basic_rules=basic_rules,
                stra_tegy=stra_tegy,
                call_api=call_api,
                instance_id=i
            )
            future_to_meta[future] = (i, mem_id)
            time.sleep(1)
        # 收集结果
        for future in concurrent.futures.as_completed(future_to_meta):
            instance_id, mem_id = future_to_meta[future]
            try:
                result = future.result()
            except Exception as e:
                print(f"Error in instance {instance_id}: {e}")
                result = None
            results.append({
                "instance_id": instance_id,
                "mem_id": mem_id,
                "content": result
            })
    return results

def store_learn(final_result:str, raw_input:str, db_path:str):
    print("Into learning process ...")
    conn = sqlite3.connect(db_path)
    print("received db_path:", db_path)
    cursor = conn.cursor()
    memery_conn = sqlite3.connect('memery_st.db')
    memery_cursor = memery_conn.cursor()

    ###################################################################################################
    prefix_export_answer = f"""
        You are given a answers: {final_result};             
        Conculude the answer within 1-5 clear words.
        """
    simple_answer = call_api(prefix_export_answer, 'You are a very good analysor')
       
    ''' HERE WE INJECT THE CODE , TO WAIT FOR THE USER'S RESPOND, OR USE LEARNING REAULT TO LEARN '''
    
    judge_prefix = f"""
        This is the user's question: {raw_input};
        This is the answer generated in the last step: {simple_answer};
        This is the user's response to that answer: {raw_input};
        Evaluate the user's attitude toward the answer;
        Reply with exactly one word: 'agree' or 'notaccompromise';
        If the user's attitude is unclear or cannot be determined with certainty, reply with: 'agree'.      
        """
    judge_answer = call_api(judge_prefix, 'You are a very good analysor')

    if 'agree' in judge_answer and 'notaccompromise' not in judge_answer:
        result_export_answer = simple_answer
    elif 'notaccompromise' in judge_answer and 'agree' not in judge_answer:
        result_export_answer = 'Pick another answer from the answer list'

        cursor.execute("""
            SELECT answer
            FROM answer_list
        """)
        rows = cursor.fetchall()
        answer_list = [row[0] for row in rows] if rows else []

        guidlearn_prefix = f"""
        This is your previous answer to the user's question: {simple_answer};
        The user does not like your previous answer;
        Please choose another answer from the following answer list: {answer_list};
        Reply with only the new answer.   
        """
        result_export_answer = call_api(guidlearn_prefix, 'You are a very good analysor')

    else:
        result_export_answer = simple_answer
    ###################################################################################################
     
    #______UPDATING THE PICK_RATIO IN SAMPLE/PRE_SAMPLE________________________________________________________________________-

    #Before store the self picked answer , should consider the "next step" answer as the considerible store guide
    cursor.execute("SELECT id, answer, eucli_dis FROM sample")
    sample_rows = cursor.fetchall()
    cursor.execute("SELECT id, answer, eucli_dis FROM pre_sample")
    presample_rows = cursor.fetchall()
    print("picked sample_rows:", sample_rows)

    # Process sample table
    sample_updates = update_pick_ratio(
        cursor, sample_rows, result_export_answer
    )
    # Process pre_sample table
    presample_updates = update_pick_ratio(
        cursor, presample_rows, result_export_answer
    )

    for pick_ratio, sample_id in sample_updates:
        cursor.execute(
            "UPDATE sample SET pick_ratio = ? WHERE id = ?",
            (pick_ratio, sample_id)
        )
        print(f"[sample] Updating pick_ratio (ID={sample_id}): {pick_ratio}")
    for pick_ratio, sample_id in presample_updates:
        cursor.execute(
            "UPDATE pre_sample SET pick_ratio = ? WHERE id = ?",
            (pick_ratio, sample_id)
        )
        print(f"[pre_sample] Updating pick_ratio (ID={sample_id}): {pick_ratio}")

    # Commit all changes
    conn.commit()
    print("PICK_RATIO UPDATING FOR BOTH TABLES IS COMPLETED")

    #______________________________________________________________________________________________
    # Update Raised_B IN MEM FOR LEARNING
    try:
        # Step 1: Fetch mem_id where pick_ratio = 1 from pre_sample
        cursor.execute("SELECT mem_id FROM pre_sample WHERE pick_ratio = 1")
        mem_ids = [row[0] for row in cursor.fetchall()]

        if not mem_ids:
            print("No rows found with pick_ratio = 1 in pre_sample.")
        else:
            print(f"Found {len(mem_ids)} mem_id(s) to update: {mem_ids}")

            # Step 2: Update raised_b in memery_st.db's memery table                             
            for mem_id in mem_ids:
                # Since there's no explicit 'id' column, we assume mem_id maps to rowid
                memery_cursor.execute(
                    "UPDATE memery SET raised_b = raised_b + 1 WHERE rowid = ?",
                    (mem_id,)
                )       
            memery_conn.commit()
            print(f"Successfully updated raised_b for {len(mem_ids)} row(s) in memery table.")
    
    finally:
        # Close both connections
        conn.commit()

    #_______UPDATING TOKEN AND STRATEGY IN THE MEM__________________________________________________________________________________________
    # 查询 sample 表中的 answer , eucli_dis 列;
    #cursor.execute("SELECT sample, answer FROM sample")
    cursor.execute("SELECT sample, answer FROM sample WHERE pick_ratio = '1'")
    rows = cursor.fetchall()
    sample_restore = [(row[0], row[1]) for row in rows if row and row[0] is not None and row[1] is not None]
    
    cursor.execute("""
        SELECT sample, answer
        FROM sample
        WHERE pick_ratio = '0'
        ORDER BY RANDOM()
        LIMIT 5
    """)

    none_ratio = cursor.fetchall()         
    conn.commit()

    prefix_token_structure = f"""
        This is the 【picked answer】:{sample_restore};
        This is 【none pikeds answer】:{none_ratio};
        Compare the token structure difference of the 【picked answers】 and  【none piked answer】;

        Example of the token structure difference like :'more specialized vocabulary' ,'more hyphenated compound modifier' , and so on ;

        Reply and only reply the difference of the 【picked answer】's token structure,describ the defference with few kernal words, not a descrbtion sentence;
        do not mention the 【none piked answer】's samples token structure;
        
        Reply short and clearly within 25 words;

        """
    result_token_structure = call_api(prefix_token_structure, 'You are a very good analyser')
    print("result_token_structure:", result_token_structure)
    
    # insert data
    memery_cursor.execute("""
        INSERT INTO memery (token_structure, time_stap)
        VALUES (?, ?)
    """, (result_token_structure, datetime.now().isoformat()))
    memery_conn.commit()

    #______________TOKEN STRUCTURE________________________________________________________________
    prefix_concept_strategy = f"""
        This is the 【picked answer】:{sample_restore};
        This is 【none pikeds answer】:{none_ratio};
        Compare the concept strategy difference of the 【picked answers】 and  【none piked answer】;

        Example of the concept strategy difference like :'cooperative workaround' ,'autonomy vs. shared resource tension' , and so on ;

        Reply and only reply the difference of the 【picked answer】's concept strategy,describ the defference with few kernal words, not a descrbtion sentence;
        do not mention the 【none piked answer】's samples concept strategy;
        
        Reply short and clearly within 25 words;

        """
    result_concept_strategy = call_api(prefix_concept_strategy, 'You are a very good analyser')
    print("result_concept_strategy:", result_concept_strategy)

    memery_cursor.execute("""
        UPDATE memery
        SET concept_strategy = ?
        WHERE rowid = (
            SELECT rowid FROM memery
            WHERE concept_strategy IS NULL OR concept_strategy = ''
            ORDER BY rowid DESC
            LIMIT 1
        )
    """, (result_concept_strategy,))
    memery_conn.commit()
    #_______CONCEPT STRATEGY_____________________________________________________________________
        
    '''
    cursor.execute("DELETE FROM sample")  
    conn.commit()  
    print("\n 【lable 'sample' is cleaned】.")  
    '''

    '''
    except Exception as e:
        print(f"Processing failed, rolling back changes: {e}")
        conn.rollback()  # 出错时回滚

    finally:
        conn.close()
    '''

    conn.close()
    memery_conn.close()

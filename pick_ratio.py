import sqlite3
import math
from datetime import datetime

# ------------------------------
# 1. Time-varying decay function
# ------------------------------
def r_eff(t):
    if t < 8:
        return 0.01 + 0.000625 * t
    elif t <= 18:
        return 0.015
    else:
        return 0.015 + 0.0005 * (t - 18)

# ------------------------------
# 2. Y_mod(t) function
# ------------------------------
def Y_mod(t, a, b, A=10, C=0.5):
    '''
    if a == 0:     # avoid division by zero
        accuracy = 0
    else:
    '''
    accuracy = (b + 0.5) / (a + 1)
    
    return A * math.log(t + 1) * accuracy * math.exp(-r_eff(t) * t) + C


# ------------------------------
# 3. Load data from sqlite3 and get top 10 related columns
# ------------------------------
def load_and_compute(db_path="memery_st.db"):
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Use implicit rowid
    cursor.execute("""
        SELECT rowid, pick_a, raised_b, time_stap, token_structure, concept_strategy
        FROM memery
    """)
    rows = cursor.fetchall()

    results = []
    now = datetime.now()

    for (row_id, a, b, time_str, token_struct, concept_strat) in rows:
        time_dt = datetime.fromisoformat(time_str)
        t_days = (now - time_dt).total_seconds() / 86400
        y_value = Y_mod(t_days, a, b)

        results.append({
            "id": row_id,
            "a": a,
            "b": b,
            "time_stamp": time_str,
            "t_days": t_days,
            "Y_mod": y_value,
            "token_structure": token_struct,
            "concept_strategy": concept_strat
        })

    # Pick top 10 using current order
    top_token_data = [(x["id"], x["a"], x["token_structure"]) for x in results[:9]]
    top_concept_data = [(x["id"], x["a"], x["concept_strategy"]) for x in results[:9]]

    # Update DB
    for row_id, _, _ in top_token_data:
        cursor.execute("UPDATE memery SET pick_a = pick_a + 1 WHERE rowid = ?", (row_id,))

    for row_id, _, _ in top_concept_data:
        cursor.execute("UPDATE memery SET pick_a = pick_a + 1 WHERE rowid = ?", (row_id,))

    conn.commit()
    conn.close()

    return {
        "top_9_token_structure_with_id_and_original_a": top_token_data,
        "top_9_concept_strategy_with_id_and_original_a": top_concept_data
    }

# ------------------------------
# 4. Example usage
# ------------------------------
if __name__ == "__main__":
    output = load_and_compute()
    print(output)
    
    '''
    # 打印计算结果
    print("=== 完整计算结果 ===")
    for item in output["computed_results"]:
        print(item)
    
    # 打印top10的token_structure数据及其原始pick_a值
    print("\n=== Top 10 token_structure 数据 (原始pick_a值) ===")
    for idx, (original_a, data) in enumerate(output["top_10_token_structure_with_original_a"], 1):
        print(f"{idx}. 原始pick_a: {original_a}, token_structure: {data}")
    
    # 打印top10的concept_strategy数据及其原始pick_a值
    print("\n=== Top 10 concept_strategy 数据 (原始pick_a值) ===")
    for idx, (original_a, data) in enumerate(output["top_10_concept_strategy_with_original_a"], 1):
        print(f"{idx}. 原始pick_a: {original_a}, concept_strategy: {data}")
    '''
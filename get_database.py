# 你包内部的代码（db_tool.py）
import os
import sqlite3

def get_db_path(db_name="memery_st.db"):
    # 🔥 核心：获取【当前用户运行 main.py 的目录】
    current_dir = os.getcwd()  # 这就是用户的主目录
    db_path = os.path.join(current_dir, db_name)
    return db_path

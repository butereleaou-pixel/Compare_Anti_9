# your_package/__init__.py

# 从内部模块 导入 函数，让外部可以直接访问
from .convert_vector import convert_token
from .calculate_vector_variance import Eucli_Dist
from .mean_euclic import select_top30
from .pick_ratio import load_and_compute
from .get_database import get_db_path
from .COMPARE_UTILS import ( process_rule_based_generation, pre_mem, process_table, 
            generate_answer, fetch_answers_and_eucli_dis, calculate_average_eucli_dis, 
            load_basic_rules, pre_thread_process, store_learn, pick_average_dis)

# 可选：写版本号
__version__ = "0.1.7"
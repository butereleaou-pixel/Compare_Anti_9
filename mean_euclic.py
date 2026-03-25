import sqlite3
import statistics

def select_top30():
    # Path to your database
    db_path = "compare_50.db"

    # Connect to DB
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. Read all eucli_dis values
    cursor.execute("SELECT eucli_dis FROM sample")
    eucli_values = [row[0] for row in cursor.fetchall()]

    # Compute mean
    mean_eucli = statistics.mean(eucli_values)
    print("Mean eucli_dis:", mean_eucli)

    # 2. Define the range [mean - 2, mean + 2]
    lower_bound = mean_eucli - 1
    upper_bound = mean_eucli + 1

    # 3. Read rows where eucli_dis is within the range
    cursor.execute("""
        SELECT answer
        FROM sample
        WHERE eucli_dis BETWEEN ? AND ?
    """, (lower_bound, upper_bound))

    # 4. Extract answers
    selected_answers = [row[0] for row in cursor.fetchall()]

    conn.close()
    return selected_answers

'''
# Example usage
answers = select_top30()
print("\nSelected answers:")
for a in answers:
    print(a)
'''






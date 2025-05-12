import json
import sqlite3
import pandas as pd
import ast

def load_and_clean(path):
    with open(path, 'r') as f:
        data = json.load(f)

    cleaned_commit = []
    cleaned_code = []
    cleaned_stack = []
    for item in data:
        if 'context' in item:
            context = ast.literal_eval(item['context']) if isinstance(item['context'], str) else item['context']
        if item['source'] == 'git_commit':
            cleaned_commit.append({
                'question': item['question'],
                'answer': item['answer'],
                'author': context.get('author'),
                'commit_hash': context.get('commit_hash'),
                'date': context.get('date'),
                'message': context.get('message'),
                "context": item['context'],
                'source': item['source']
            })
        elif item['source'] == 'source_code':
            cleaned_code.append({
                'question': item['question'],
                'answer': item['answer'],
                'name': context.get('name'),
                'type': context.get('type'),
                'docstring': context.get('docstring'),
                'file_docstring': context.get('file_docstring'),
                'source': item['source'],
                "context": item['context']
            })
        elif item['source'] == 'stackoverflow':
            cleaned_stack.append({
                'question': item['question'],
                'answer': item['answer'],
                'source': item['source'],
                'question_id': item['metadata']['question_id'],
                'link': item['metadata']['link'],
                "context": ""
            })
    return pd.DataFrame(cleaned_commit), pd.DataFrame(cleaned_code), pd.DataFrame(cleaned_stack)

df_train_commit, df_train_code, _ = load_and_clean("/data/train_data.json")
_, _, df_eval_stack = load_and_clean("/data/evaluation_data.json")

conn = sqlite3.connect("/data/qa_data.db")
df_train_commit.to_sql("train_commit", conn, if_exists="replace", index=False)
df_train_code.to_sql("train_code", conn, if_exists="replace", index=False)
df_eval_stack.to_sql("eval_stack", conn, if_exists="replace", index=False)
conn.close()

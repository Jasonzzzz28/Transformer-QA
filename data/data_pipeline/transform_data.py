from openai import OpenAI
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import json
import os
from tqdm import tqdm
# api_key = input("Enter your openai api key: ")
# api_base = input("Enter your openai api base: ")
os.environ["OPENAI_API_KEY"] = "EMPTY"
os.environ["OPENAI_API_BASE"] = "http://localhost:6006/v1"

def summarize_docstring(name, type, docstring):
    # 使用 GPT 模型生成摘要
    while True:
        try:
            prompt = """
            Summarize the following docstring, tell me what does the {type} `{name}` do.
            Docstring:
            {docstring}
            """
            question_pool = [
                "What does the {type} {name} do?"
                "What is the function of the {type} {name}?",
                "How does the {type} {name} work?",
                "What role does the {type} {name} play?",
                "What is the purpose of the {type} {name}?",
                "What does the {type} {name} accomplish?",
                "Can you explain what the {type} {name} is used for?",
                "Why do we need the {type} {name}?",
                "What is the {type} {name} responsible for?",
                "What task does the {type} {name} perform?",
                "What kind of behavior does the {type} {name} define?"
            ]
            question = random.choice(question_pool)
            question = question.format(name=name, type=type)
            client = OpenAI(
                api_key="EMPTY",
                base_url="http://localhost:6006/v1"
            )
            prompt = prompt.format(name=name, type=type, docstring=docstring)
            response = client.chat.completions.create(
                model="mistralai/Ministral-8B-Instruct-2410",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            # print(response.choices[0].message.content)
            return question, response.choices[0].message.content
        except Exception as e:
            print(e)
            time.sleep(10)
            continue

def format_source_code(data):
    formatted_data = []
    for item in data:
        formatted_item = {
            "question": item["question"],
            "answer": item["answer"],
            "source": item["source"],
            "context": "docstring: "+item['docstring']+"\nfile_docstring: "+item['file_docstring']
        }
        formatted_data.append(formatted_item)
    return formatted_data

if __name__ == "__main__":
    # Transform source code data
    with open("/data/offline_data/source_code_qa.json", "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    qa_data_with_summary = []
    with ThreadPoolExecutor(max_workers=100) as executor:
        futures = [executor.submit(summarize_docstring, qa["name"], qa["type"], qa["docstring"]) for qa in qa_data]
        for future in tqdm(as_completed(futures)):
            qa = qa_data[futures.index(future)]
            qa["question"], qa["answer"] = future.result()
            qa_data_with_summary.append(qa)
    formatted_data = format_source_code(qa_data_with_summary)
    with open("/data/offline_data/source_code_qa_formatted.json", "w", encoding="utf-8") as f:
        json.dump(formatted_data, f, indent=4, ensure_ascii=False)
    
    # Transform commit data
    with open("/data/offline_data/qa_from_commits.json", "r") as f:
        qa_from_commits = json.load(f)
    qa_from_commits_formatted = []
    for qa in qa_from_commits:
        context = qa["metadata"]
        context = str(context)
        qa_from_commits_formatted.append({
            "question": qa["question"],
            "answer": qa["answer"],
            "context": context,
            "source": "git_commit"
        })
    with open("/data/offline_data/qa_from_commits_formatted.json", "w", encoding="utf-8") as f:
        json.dump(qa_from_commits_formatted, f, indent=4, ensure_ascii=False)

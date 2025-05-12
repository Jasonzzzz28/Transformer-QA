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

def data_split(source_code_qa, huggingface_qa, qa_from_commits):
    train_data = source_code_qa + qa_from_commits
    random.shuffle(train_data)
    random.shuffle(huggingface_qa)
    train_data_count =len(train_data)
    evaluation_data = train_data[:int(train_data_count*0.1)]
    train_data = train_data[int(train_data_count*0.1):]
    production_data = huggingface_qa[:20]
    online_data = huggingface_qa[20:]
    with open("/data/offline_data/train_data.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=4, ensure_ascii=False)
    with open("/data/offline_data/evaluation_data.json", "w", encoding="utf-8") as f:
        json.dump(evaluation_data, f, indent=4, ensure_ascii=False)
    with open("/data/offline_data/production_data.json", "w", encoding="utf-8") as f:
        json.dump(production_data, f, indent=4, ensure_ascii=False)
    with open("/data/offline_data/online_data.json", "w", encoding="utf-8") as f:
        json.dump(online_data, f, indent=4, ensure_ascii=False)

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
    
    # Transform stackoverflow data
    with open('/data/offline_data/huggingface_qa.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    formatted_data = []
    for item in data:
        for i in range(len(item['answers'])):
            formatted_data.append({
                "question": item['title'] + "\n" + item['body'],
                "answer": item['answers'][i],
                "source": "stackoverflow",
                "metadata": {
                "question_id": item['question_id'],
                "link": item['link']
            }
        })

    with open('/data/offline_data/huggingface_qa_formatted.json', 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, indent=2, ensure_ascii=False)

    print("✅ Done! Saved 100 Q&A items to huggingface_qa_formatted.json")

    # Data split
    data_split(formatted_data, formatted_data, formatted_data)
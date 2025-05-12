import ast
import os
import json
import tokenize
from io import BytesIO
from tqdm import tqdm
from git import Repo
import random
import requests
import time

def get_questions_with_answers(tag='huggingface', target_count=100):
    collected = []
    page = 1
    page_size = 100
    while len(collected) < target_count:
        print(f"Fetching page {page}...")
        url = "https://api.stackexchange.com/2.3/questions"
        params = {
            'order': 'desc',
            'sort': 'activity',
            'tagged': tag,
            'site': 'stackoverflow',
            'filter': 'withbody',
            'pagesize': page_size,
            'page': page
        }
        res = requests.get(url, params=params)
        data = res.json()
        items = data.get('items', [])

        for item in items:
            if item.get('answer_count', 0) > 0:
                collected.append({
                    'question_id': item['question_id'],
                    'title': item['title'],
                    'body': item['body'],
                    'link': item['link']
                })
                if len(collected) >= target_count:
                    break

        if not data.get('has_more', False):
            break  # 没有更多数据
        page += 1
        time.sleep(1)  # 避免触发速率限制
    return collected

def get_answers_for_question(qid):
    url = f"https://api.stackexchange.com/2.3/questions/{qid}/answers"
    params = {
        'order': 'desc',
        'sort': 'votes',
        'site': 'stackoverflow',
        'filter': 'withbody'
    }
    res = requests.get(url, params=params)
    data = res.json()
    return [ans['body'] for ans in data.get('items', [])]

def extract_comments(source_code):
    comments = []
    tokens = tokenize.tokenize(BytesIO(source_code.encode("utf-8")).readline)
    for toknum, tokval, _, _, _ in tokens:
        if toknum == tokenize.COMMENT:
            comments.append(tokval.strip("# ").strip())
    return comments

def extract_docstrings_and_defs(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source)
    results = []
    module_docstring = ast.get_docstring(tree)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            name = node.name
            docstring = ast.get_docstring(node)
            node_type = "function" if isinstance(node, ast.FunctionDef) else "class"
            source_lines = source.splitlines()
            start_line = node.lineno - 1  # ast 行号从1开始，列表索引从0开始
            end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line
            source_code = '\n'.join(source_lines[start_line:end_line])
            results.append({
                "type": node_type,
                "name": name,
                "docstring": docstring or "",
                "source_code": source_code,
                "file_docstring": module_docstring
            })

    comments = extract_comments(source)
    return results, comments

def generate_qa_from_entry(entry):
    name = entry["name"]
    doc = entry["docstring"]
    if not doc:
        return None

    # question = f"What does the {entry['type']} `{name}` do?"
    # answer = doc.strip()
    source_code = entry.get("source_code", "")
    file_docstring = entry.get("file_docstring", "")

    return {
        "name": name,
        "docstring": doc.strip(),
        "file_docstring": file_docstring,
        "source": "source_code",
        "type": entry["type"],
        "code": source_code
    }

def process_directory(dir_path):
    qa_pairs = []
    for root, _, files in tqdm(os.walk(dir_path)):
        for file in tqdm(files):
            if file.endswith(".py"):
                full_path = os.path.join(root, file)
                try:
                    entries, comments = extract_docstrings_and_defs(full_path)
                    for entry in entries:
                        qa = generate_qa_from_entry(entry)
                        if qa:
                            qa["file"] = full_path
                            qa_pairs.append(qa)
                except Exception as e:
                    print(f"Failed to parse {full_path}: {e}")
    return qa_pairs


def extract_commit_data(repo_path):
    REPO_PATH = repo_path
    repo = Repo(REPO_PATH)

    output = []

    # 遍历最近 N 个 commit（可调整）
    for commit in repo.iter_commits('main', max_count=10):
        commit_data = {
            "commit_hash": commit.hexsha,
            "author": commit.author.name,
            "date": commit.committed_datetime.isoformat(),
            "message": commit.message.strip()
        }

        # 获取 diff 的简要变化（可设置为 full_diff=True 看更多上下文）
        diffs = commit.diff(commit.parents[0] if commit.parents else None, create_patch=True)

        diff_texts = []
        for diff in diffs:
            try:
                diff_texts.append(diff.diff.decode("utf-8", errors="ignore"))
            except Exception as e:
                continue

        diff_summary = "\n".join(diff_texts)
        commit_data["diff_summary"] = diff_summary

        # 构造 QA 对
        # qa_item = {
        #     "question": f"What changed in commit {commit.hexsha[:7]}?",
        #     "answer": f"{commit.message.strip()}\n\nSummary of changes:\n{diff_summary[:1000]}...",
        #     "source": "git_commit",
        #     "metadata": commit_data
        # }
        question_pool = [
            "What changed in commit {hash}?",
            "What modifications were introduced in commit {hash}?",
            "Can you summarize the changes made in commit {hash}?",
            "What updates does commit {hash} contain?",
            "What's new in commit {hash}?",
            "Describe the differences introduced by commit {hash}.",
            "What was added, removed, or modified in commit {hash}?",
            "What does commit {hash} change in the codebase?",
            "Which files or functions were affected by commit {hash}?",
            "What's the purpose of commit {hash}?",
            "How does commit {hash} alter the existing implementation?"
        ]
        question = random.choice(question_pool)
        question = question.format(hash=commit.hexsha[:7])
        qa_item = {
            "question": question,
            "answer": f"{commit.message.strip()}",
            "source": "git_commit",
            "metadata": commit_data
        }

        output.append(qa_item)

        question_pool = [
            "Who is the author of the commit {hash}?",
            "Who made the commit {hash}?",
            "Who is responsible for commit {hash}?",
            "Can you tell me who authored commit {hash}?",
            "Who's the person behind commit {hash}?",
            "Who committed {hash}?",
            "Which developer authored commit {hash}?",
            "Who was the contributor for commit {hash}?",
            "Do you know who wrote commit {hash}?",
            "Who pushed commit {hash} to the repository?",
            "Whose work is represented by commit {hash}?"
        ]
        question = random.choice(question_pool)
        question = question.format(hash=commit.hexsha[:7])
        qa_item = {
            "question": question,
            "answer": f"{commit.author.name}",
            "source": "git_commit",
            "metadata": commit_data
        }
        output.append(qa_item)

        question_pool = [
            "When was the commit {hash} made?",
            "What is the timestamp of commit {hash}?",
            "When exactly did commit {hash} occur?",
            "At what time was commit {hash} created?",
            "Can you tell me the date of commit {hash}?",
            "On what date was commit {hash} made?",
            "Do you know when commit {hash} was pushed?",
            "Any idea when commit {hash} happened?",
            "When did commit {hash} go through?",
            "What's the date on commit {hash}?",
            "When did they make commit {hash}?"
        ]
        question = random.choice(question_pool)
        question = question.format(hash=commit.hexsha[:7])
        qa_item = {
            "question": question,
            "answer": f"{commit.committed_datetime.isoformat()}",
            "source": "git_commit",
            "metadata": commit_data
        }
        output.append(qa_item)
    # 保存为 JSON
    with open("/data/offline_data/qa_from_commits.json", "w") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    # Extract source code data
    directory = "transformers"
    if not os.path.exists("/data/offline_data"):
        os.makedirs("/data/offline_data")
    qa_data = process_directory(directory)

    with open("/data/offline_data/source_code_qa.json", "w", encoding="utf-8") as f:
        json.dump(qa_data, f, indent=4, ensure_ascii=False)

    print(f"Extracted {len(qa_data)} QA pairs.")

    # Extract commit data
    extract_commit_data(directory)

    # Extract stackoverflow data for online data
    questions = get_questions_with_answers()

    for q in questions:
        q['answers'] = get_answers_for_question(q['question_id'])
        print(f"Fetched {len(q['answers'])} answers for question: {q['title'][:50]}...")
        time.sleep(0.5)

    with open('/data/offline_data/huggingface_qa.json', 'w', encoding='utf-8') as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)

    print("✅ Done! Saved 100 Q&A items to huggingface_qa.json")
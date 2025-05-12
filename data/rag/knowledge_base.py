import json
from retriever import Retriever
from memory import Base_Memory_3

def test_knowledge_base():
    retrieval_model = Retriever(model_name="/root/autodl-tmp/BAAI/bge-m3")
    memory_processor = Base_Memory_3(retrieval_model=retrieval_model)

    memory_processor.load_from_disk("/root/autodl-tmp/QA_memory")
    query = [
        "What does the function update_metrics accomplish?"
        ]
    contexts = memory_processor.rag_preprocess(query, 1)
    print(contexts)

def process_knowledge_base(knowledge_base, save_path):
    retrieval_model = Retriever(model_name="/root/autodl-tmp/BAAI/bge-m3")
    memory_processor = Base_Memory_3(retrieval_model=retrieval_model)
    memory_processor.process_knowledge_base(knowledge_base, save_path)

test_knowledge_base()
# if __name__ == "__main__":
#     with open('/root/autodl-tmp/Transformer-QA/data/offline_data/source_code_qa_with_summary_formatted.json', 'r') as f:
#         data = json.load(f)
#     knowledge_base = []
#     for item in data[:10]:
#         text = item['question']
#         knowledge_base.append((text, item['context']))
#     # with open('test_knowledge_base.json', 'w') as f:
#     #     json.dump(knowledge_base, f, indent=4)
#     save_path = "/root/autodl-tmp/QA_memory"
#     process_knowledge_base(knowledge_base, save_path)


    

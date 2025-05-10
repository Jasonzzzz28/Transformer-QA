import json
from retriever import Retriever
from memory import Base_Memory_3

# def test_knowledge_base():
#     model, tokenizer, retrieval_model = load_model(model_path="/root/autodl-tmp/meta-llama/Llama-3.2-1B-Instruct", retrieval_model_path="/root/autodl-tmp/Explicit-Memory/model/BAAI/bge-m3")
#     cache = M3_cache()
#     memory_processor = Base_Memory_3(model=model, tokenizer=tokenizer, retrieval_model=retrieval_model, config=model.config)
#     # print(model)
#     # print(tokenizer)
#     # print(model.config)
#     # knowledge_base = [
#     #     "1+1=2",
#     #     "2+2=4",
#     #     "3+3=6",
#     #     "4+4=8",
#     #     "5+5=10"
#     # ]
#     # save_path = "/root/autodl-tmp/memory"

#     # memory_processor.process_knowledge_base(knowledge_base, save_path)

#     # query = "1+1="

#     memory_processor.load_from_disk("/root/memory")
#     query = [
#         "The cubic polynomial $p(x)$ satisfies $p(2) = X $p(7) = 19,$ $p(15) = 11,$ and $p(20) = 29.$  Find",
#         "Carla is downloading a 200 GB file. Normally she can download 2 GB/minute, but 40% of the way through the download, Windows forces a restart to install updates, which takes 20 minutes. Then Carla has to restart the download from the beginning. How long does it take to download the file?"
#         ]
#     contexts = memory_processor.rag_preprocess(query)
#     print(contexts)
#     # _, indices = memory_processor.retrieve_memory_batch(query, 2)
#     # print(indices)
#     # memory_processor._load_memory_chunk_from_disk("/root/memory", indices[0])
#     # print([memory_processor.memory_chunks[i].text for i in range(len(memory_processor.memory_chunks))])
#     # print(memory_processor.memory_chunks[0].key_states[0].shape)
#     # memory_processor._load_memory_chunk_from_disk("/root/memory", indices[1])
#     # print([memory_processor.memory_chunks[i].text for i in range(len(memory_processor.memory_chunks))])
#     # print(memory_processor.memory_chunks[0].key_states[0].shape)

def process_knowledge_base(knowledge_base, save_path):
    retrieval_model = Retriever(model_name="BAAI/bge-m3")
    memory_processor = Base_Memory_3(retrieval_model=retrieval_model)
    memory_processor.process_knowledge_base(knowledge_base, save_path)

# test_knowledge_base()
if __name__ == "__main__":
    with open('source_code_qa_with_summary_formatted.json', 'r') as f:
        data = json.load(f)
    knowledge_base = []
    for item in data:
        text = item['question']
        knowledge_base.append((text, item['context']))
    # with open('test_knowledge_base.json', 'w') as f:
    #     json.dump(knowledge_base, f, indent=4)
    save_path = "/memory"
    process_knowledge_base(knowledge_base, save_path)


    

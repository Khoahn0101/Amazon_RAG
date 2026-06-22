import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from rag_pipeline import RAGPipeline

def test_conversational_flow():
    pipeline = RAGPipeline(data_path='data/chunks.jsonl')
    
    # 1st query
    q1 = "Tôi muốn mua đồ cho con gái tôi, nó sắp đi du lịch vậy bạn có thể cho tôi các loại vali dưới 100 đô và ít nhất 4 sao trở lên không"
    print(f"\n--- User Query 1: {q1} ---")
    answer1, docs1 = pipeline.answer_question(q1)
    print("\n--- Assistant Answer 1 ---")
    print(answer1)
    print("\n--- Retrieved Products 1 ---")
    for doc in docs1:
        print(f"- {doc.payload.get('text').splitlines()[0]} [Price: ${doc.payload.get('price')}, Stars: {doc.payload.get('stars')}]")
        
    # Build history
    history = [
        {"role": "user", "content": q1},
        {"role": "assistant", "content": answer1}
    ]
    
    # 2nd query (follow-up)
    q2 = "Vậy bạn có thể cho tôi thêm thông tin về 3 món đồ ở trên không"
    print(f"\n--- User Query 2: {q2} ---")
    answer2, docs2 = pipeline.answer_question(q2, chat_history=history, previous_docs=docs1)
    print("\n--- Assistant Answer 2 ---")
    print(answer2)
    print("\n--- Retrieved Products 2 ---")
    for doc in docs2:
        print(f"- {doc.payload.get('text').splitlines()[0]} [Price: ${doc.payload.get('price')}, Stars: {doc.payload.get('stars')}]")

if __name__ == "__main__":
    test_conversational_flow()

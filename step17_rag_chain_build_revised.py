# step17_rag_chain_build_revised.py
import os
import sys
from dotenv import load_dotenv

# --- 必要な LangChain モジュール ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
# RunnablePassthrough と RunnableLambda をインポート
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# Vector Store (FAISS)
try:
    from langchain_community.vectorstores import FAISS
    import faiss
    print("FAISS をインポートしました。")
except ImportError:
    print("エラー: FAISS または langchain-community が見つかりません。")
    print("   'pip install faiss-cpu langchain-community' 等を実行してください。")
    sys.exit(1)

print("--- 必要なモジュールのインポート完了 ---")

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    print("エラー: OpenAI API キーが設定されていません。")
    sys.exit(1)
else:
    print("OpenAI API キーを読み込みました。")

# --- Embedding モデルと LLM の準備 ---
try:
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=openai_api_key)
    print(f"--- Embedding モデル ({embeddings_model.model}) と LLM ({llm.model_name}) 準備完了 ---")
except Exception as e:
    print(f"エラー: モデルの初期化に失敗しました: {e}")
    sys.exit(1)

# --- Vector Store と Retriever の準備 (ステップ 15, 16 の内容) ---
# (簡略化のため、インメモリで再作成)
sample_documents = [
    Document(page_content="LangGraph は状態を持つ複雑なグラフを構築できます。", metadata={"source": "doc-a", "page": 1}),
    Document(page_content="LCEL は LangChain Expression Language の略です。", metadata={"source": "doc-b", "page": 1}),
    Document(page_content="LangGraph ではノード間で状態が共有されます。", metadata={"source": "doc-a", "page": 2}),
]
try:
    vector_store = FAISS.from_documents(sample_documents, embeddings_model)
    retriever = vector_store.as_retriever(search_kwargs={'k': 2}) # 上位2件を取得
    print("--- Vector Store (FAISS) と Retriever 準備完了 (k=2) ---")
except Exception as e:
    print(f"エラー: Vector Store または Retriever の準備中にエラーが発生しました: {e}")
    sys.exit(1)

# --- Context フォーマット関数と RunnableLambda ---
def format_docs(docs: list[Document]) -> str:
    """Document リストを結合して単一の文字列にする"""
    return "\n\n".join(doc.page_content for doc in docs)

# 関数を RunnableLambda でラップ
format_docs_runnable = RunnableLambda(format_docs)

print("--- Context フォーマット関数と RunnableLambda 準備完了 ---")
# step17_rag_chain_build_revised.py (続き)

# RAG 用のプロンプトテンプレート
template = """
以下の Context のみを基にして、最後の Question に答えてください。

Context:
{context}

Question: {question}
"""

rag_prompt = ChatPromptTemplate.from_template(template)
print("--- RAG プロンプトテンプレート準備完了 ---")
# step17_rag_chain_build_revised.py (続き)

print("\n--- RAG チェーンの組み立て ---")

# RAG チェーン本体
rag_chain = (
    # チェーンへの入力は {"question": "..."} という辞書を想定します
    # 1. RunnablePassthrough.assign を使って、入力辞書に Context を追加
    RunnablePassthrough.assign(
        # 新しく "context" というキーを作る
        # 値は、lambda 関数を使って計算する
        # lambda 関数の入力 x は、この時点でのデータ（{"question": ...}）
        context=lambda x: format_docs_runnable.invoke(retriever.invoke(x["question"]))
    )
    # これで、データは {"question": "...", "context": "フォーマット済み文書文字列"} になった！
    | rag_prompt          # 2. この辞書をプロンプトテンプレートに渡す
    | llm                 # 3. 生成されたプロンプトを LLM に渡す
    | StrOutputParser()   # 4. LLM の答え(AIMessage)を普通の文字列にする
)

print("--- RAG チェーン組み立て完了 ---")
print("処理の流れ: 入力(question) -> Retriever検索 & Contextフォーマット -> プロンプト注入 -> LLM生成 -> 文字列出力")
# step17_rag_chain_build_revised.py (続き)

# --- RAG チェーンの実行 ---
question = "LangGraph の状態共有について説明して"
print(f"\n--- RAG チェーン実行 (質問: '{question}') ---")

try:
    # チェーンの invoke には、最初の入力形式に合わせた辞書を渡す
    final_answer = rag_chain.invoke({"question": question})

    print("\n--- 最終的な回答 ---")
    print(final_answer)

except Exception as e:
    print(f"エラー: RAG チェーンの実行中にエラーが発生しました: {e}")

print("\n--- 処理終了 ---")
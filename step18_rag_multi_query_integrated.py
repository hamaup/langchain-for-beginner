# step18_rag_multi_query_integrated.py
import os
import sys
import logging # ★ 生成されたクエリを見るために追加
from dotenv import load_dotenv

# --- 必要な LangChain モジュール ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# MultiQueryRetriever をインポート (パスは環境により変わる可能性あり)
# langchain 本体からのインポートが一般的
try:
    from langchain.retrievers.multi_query import MultiQueryRetriever
    print("MultiQueryRetriever をインポートしました (from langchain.retrievers.multi_query)。")
except ImportError:
    # もし上記でエラーが出る場合は community を試す
    try:
        from langchain_community.retrievers.multi_query import MultiQueryRetriever
        print("MultiQueryRetriever をインポートしました (from langchain_community.retrievers.multi_query)。")
        print("   'pip install langchain-community' が必要かもしれません。")
    except ImportError:
        print("エラー: MultiQueryRetriever が見つかりません。")
        print("   'pip install langchain' を確認してください。")
        sys.exit(1)


# Vector Store (FAISS)
try:
    from langchain_community.vectorstores import FAISS
    import faiss
    print("FAISS をインポートしました。")
except ImportError:
    print("エラー: FAISS または langchain-community が見つかりません。")
    sys.exit(1)

print("--- 必要なモジュールのインポート完了 ---")

# --- ★ ロギング設定 (INFO レベルで生成クエリを表示) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
print("--- ロギング設定完了 ---")

# --- ステップ 17 と同様の準備 ---
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    print("エラー: OpenAI API キーが設定されていません。")
    sys.exit(1)
else:
    print("OpenAI API キーを読み込みました。")

try:
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=openai_api_key) # クエリ生成と最終回答生成の両方に使う
    print(f"--- Embedding モデルと LLM 準備完了 ---")
except Exception as e:
    print(f"エラー: モデルの初期化に失敗しました: {e}")
    sys.exit(1)

# Vector Store とベース Retriever の準備
sample_documents = [
    Document(page_content="LangChain は LLM アプリケーション開発を支援するフレームワークです。主要な機能に LCEL があります。", metadata={"source": "doc-b", "page": 1, "category": "core"}),
    Document(page_content="エージェントはツールを使って外部システムと対話し、タスクを実行します。", metadata={"source": "doc-c", "page": 1, "category": "agent"}),
    Document(page_content="LangGraph は状態を持つ循環可能なグラフを構築し、複雑なエージェントランタイムを作成できます。", metadata={"source": "doc-a", "page": 1, "category": "graph"}),
    Document(page_content="LangChain のメモリ機能を使うと、チャットボットが会話履歴を記憶できます。", metadata={"source": "doc-g", "page": 1, "category": "memory"}),
    Document(page_content="プロンプトテンプレートは、LLM への入力を動的に生成するための仕組みです。", metadata={"source": "doc-f", "page": 1, "category": "core"}),
    Document(page_content="ベクトルストア (FAISS など) は、テキスト埋め込みの高速な類似検索を可能にします。", metadata={"source": "doc-d", "page": 1, "category": "vectorstore"}),
]
try:
    vector_store = FAISS.from_documents(sample_documents, embeddings_model)
    base_retriever = vector_store.as_retriever(search_kwargs={'k': 2}) # ベースは上位2件
    print("--- Vector Store (FAISS) とベース Retriever 準備完了 (k=2) ---")
except Exception as e:
    print(f"エラー: Vector Store またはベース Retriever の準備中にエラーが発生しました: {e}")
    sys.exit(1)

# Context フォーマット関数 (ステップ 17 と同じ)
def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)
format_docs_runnable = RunnableLambda(format_docs)
print("--- Context フォーマット関数準備完了 ---")
# step18_rag_multi_query_integrated.py (続き)

print("\n--- MultiQueryRetriever の作成 ---")
try:
    # MultiQueryRetriever を作成
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm
    )
    print("MultiQueryRetriever 作成完了！")

except Exception as e:
    print(f"エラー: MultiQueryRetriever の作成中にエラーが発生しました: {e}")
    sys.exit(1)
# step18_rag_multi_query_integrated.py (続き)

print("\n--- MultiQueryRetriever を使った RAG チェーンの組み立て ---")

# RAG プロンプト (ステップ 17 と同じ)
template = """
以下の Context のみを基にして、最後の Question に答えてください。

Context:
{context}

Question: {question}
"""
rag_prompt = ChatPromptTemplate.from_template(template)

# チェーンを組み立てる (Retriever 部分が変わる！)
rag_chain_multi_query = (
    # 入力は {"question": "..."}
    RunnablePassthrough.assign(
        # context の取得に multi_query_retriever を使う！
        context=lambda x: format_docs_runnable.invoke(multi_query_retriever.invoke(x["question"]))
    )
    | rag_prompt
    | llm
    | StrOutputParser()
)

print("--- MultiQueryRetriever を使った RAG チェーン組み立て完了 ---")
# step18_rag_multi_query_integrated.py (続き)

# --- チェーンの実行 ---
# 質問 (複数の要素を含む可能性のある質問)
question = "LangChain のエージェントとメモリについて教えて"
print(f"\n--- MultiQuery RAG チェーン実行 (質問: '{question}') ---")

try:
    # チェーンを実行！ 内部でクエリ生成のログが出るはず
    final_answer = rag_chain_multi_query.invoke({"question": question})

    print("\n--- 最終的な回答 (MultiQuery) ---")
    print(final_answer)

    # (比較用：もしベースRetrieverのチェーンも実行したい場合)
    # print("\n--- (比較) ベース RAG チェーン実行 ---")
    # rag_chain_base = (
    #     RunnablePassthrough.assign(
    #         context=lambda x: format_docs_runnable.invoke(base_retriever.invoke(x["question"]))
    #     ) | rag_prompt | llm | StrOutputParser()
    # )
    # base_answer = rag_chain_base.invoke({"question": question})
    # print("\n--- 最終的な回答 (ベース) ---")
    # print(base_answer)

except Exception as e:
    print(f"エラー: RAG チェーンの実行中にエラーが発生しました: {e}")

print("\n--- 処理終了 ---")
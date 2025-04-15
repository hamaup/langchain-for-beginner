# step15_vectorstore_basic_faiss.py
import os
import sys
from dotenv import load_dotenv
# from typing import List # List は直接使用しないため削除

# --- 必要な LangChain モジュール ---
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
# FAISS Vector Store をインポート
try:
    from langchain_community.vectorstores import FAISS
    print("FAISS をインポートしました (from langchain_community.vectorstores)。")
    # 必要ならインストール: pip install langchain-community
except ImportError:
    print("エラー: langchain-community が見つかりません。")
    print("   'pip install langchain-community' を実行してください。")
    sys.exit(1)

# FAISS 本体も必要
try:
    import faiss
    print(f"faiss バージョン: {faiss.__version__}")
except ImportError:
    print("エラー: faiss がインストールされていません。")
    print("   'pip install faiss-cpu' または 'pip install faiss-gpu' を実行してください。")
    sys.exit(1)


print("--- 必要なモジュールのインポート完了 ---")

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    print("エラー: OpenAI API キーが設定されていません。")
    sys.exit(1)
else:
    print("OpenAI API キーを読み込みました。")

# --- Embedding モデルの準備 ---
try:
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)
    print(f"--- Embedding モデル準備完了 ({embeddings_model.model}) ---")
except Exception as e:
    print(f"エラー: Embedding モデルの初期化に失敗: {e}")
    sys.exit(1)

# --- Vector Store に格納するデータの準備 ---
sample_documents = [
    Document(page_content="LangGraph は状態を持つ複雑なグラフを構築できます。", metadata={"source": "doc-a", "page": 1, "category": "graph"}),
    Document(page_content="LCEL は LangChain Expression Language の略です。", metadata={"source": "doc-b", "page": 1, "category": "core"}),
    Document(page_content="エージェントはツールを使ってタスクを実行します。", metadata={"source": "doc-c", "page": 1, "category": "agent"}),
    Document(page_content="ベクトルストアは埋め込みベクトルを高速に検索します。", metadata={"source": "doc-d", "page": 1, "category": "vectorstore"}),
    Document(page_content="LangGraph ではノード間で状態が共有されます。", metadata={"source": "doc-a", "page": 2, "category": "graph"}),
    Document(page_content="RAG は検索拡張生成の略で、外部知識を利用します。", metadata={"source": "doc-e", "page": 1, "category": "rag"}),
    Document(page_content="プロンプトテンプレートを使うと入力を動的にできます。", metadata={"source": "doc-f", "page": 1, "category": "core"}),
]
print(f"--- サンプル Document (チャンク) {len(sample_documents)} 件準備完了 ---")

# step15_vectorstore_basic_faiss.py (続き)

print("\n--- FAISS Vector Store の作成 (インメモリ) ---")
vector_store = None # エラー時に備えて初期化
try:
    # .from_documents() で Document リストからインメモリ FAISS Vector Store を作成
    # 内部で embedding モデルを使ってベクトル化とインデックス構築が行われる
    vector_store = FAISS.from_documents(
        documents=sample_documents,
        embedding=embeddings_model
    )
    print("FAISS Vector Store (インメモリ) が作成されました。")

    # --- (参考) FAISS Index の保存と読み込み ---
    # db_directory = "./faiss_index"
    # print(f"\n--- FAISS Index の保存 (場所: {db_directory}) ---")
    # vector_store.save_local(db_directory)
    # print("FAISS Index が保存されました。")
    # # 次回読み込む場合:
    # # vector_store_loaded = FAISS.load_local(db_directory, embeddings_model, allow_dangerous_deserialization=True) # セキュリティリスクを理解した上で True に
    # # print("FAISS Index を読み込みました。")

except Exception as e:
    print(f"Vector Store の作成中にエラーが発生しました: {e}")
    sys.exit(1)

if vector_store is None:
     print("エラー: Vector Store の作成に失敗しました。")
     sys.exit(1)

# step15_vectorstore_basic_faiss.py (続き)

# --- 検索クエリ ---
query1 = "グラフの状態管理について教えて"
query2 = "LangChain のコア機能は？" # フィルタリングはコメントアウト

print(f"\n--- 検索クエリ1: '{query1}' ---")
try:
    # スコア付きで類似度検索 (k=上位3件)
    # FAISS の similarity_search_with_score は (Document, score) のタプルリストを返す
    # score は通常 L2 距離 (小さいほど類似)
    search_results_score = vector_store.similarity_search_with_score(query1, k=3)
    print(f"\nスコア付き類似度検索の結果 ({len(search_results_score)} 件):")
    for doc, score in search_results_score:
        # スコアは L2 距離。小さいほど類似度が高い
        print(f"  Score (L2 Distance): {score:.4f} (小さいほど類似)")
        print(f"  Content: {doc.page_content}")
        print(f"  Metadata: {doc.metadata}")

except Exception as e:
    print(f"類似度検索中にエラーが発生しました: {e}")

# --- FAISSでのメタデータフィルタリングについて ---
# FAISSのLangChain実装では、similarity_search に直接的な filter 引数がないことが多いです。
# そのため、以下のフィルタリング検索はコメントアウトします。
# フィルタリングが必要な場合は、多めに取得してからPythonで絞り込むなどの対応が必要です。
# print(f"\n--- 検索クエリ2: '{query2}' (メタデータフィルタリングは FAISS では非推奨) ---")
# try:
#     # 多めに検索 (例: k=5)
#     search_results_all = vector_store.similarity_search(query2, k=5)
#     # Python でフィルタリング
#     filtered_docs = [doc for doc in search_results_all if doc.metadata.get("category") == "core"]
#     print(f"\n手動フィルタリング検索の結果 ({len(filtered_docs)} 件):")
#     for i, doc in enumerate(filtered_docs[:2]): # 上位2件表示
#         print(f"--- 結果 {i+1} ---")
#         print(f"  Content: {doc.page_content}")
#         print(f"  Metadata: {doc.metadata}")
# except Exception as e:
#     print(f"フィルタリング検索中にエラーが発生しました: {e}")
# step15_vectorstore_basic_faiss.py (続き)

print("\n--- Retriever として利用 (.as_retriever) ---")

try:
    # Retriever を作成。k=2 を指定。filter は指定しない。
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={'k': 2} # filter は FAISS では直接指定できないことが多い
    )
    print("Retriever を作成しました (k=2)。")

    # Retriever を使って関連文書を取得
    query_graph = "LangGraph の状態について" # query1 と似たクエリ
    retrieved_docs = retriever.invoke(query_graph)
    print(f"\nRetriever による検索結果 ({len(retrieved_docs)} 件) for query: '{query_graph}'")

    for i, doc in enumerate(retrieved_docs):
        print(f"--- 結果 {i+1} ---")
        print(f"  Content: {doc.page_content}")
        print(f"  Metadata: {doc.metadata}") # メタデータ自体は取得できる

except Exception as e:
    print(f"Retriever の作成または使用中にエラーが発生しました: {e}")

print("\n--- 処理終了 ---")
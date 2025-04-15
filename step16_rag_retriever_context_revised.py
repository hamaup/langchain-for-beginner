# step16_rag_retriever_context_revised.py
import os
import sys
from dotenv import load_dotenv

# --- 必要な LangChain モジュール ---
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
# FAISS Vector Store をインポート
try:
    from langchain_community.vectorstores import FAISS
    print("FAISS をインポートしました (from langchain_community.vectorstores)。")
except ImportError:
    print("エラー: langchain-community が見つかりません。")
    print("   'pip install langchain-community' を実行してください。")
    sys.exit(1)

# FAISS 本体も確認
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
    # 注意: モデル名は最新のものを確認してください
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)
    print(f"--- Embedding モデル準備完了 ({embeddings_model.model}) ---")
except Exception as e:
    print(f"エラー: Embedding モデルの初期化に失敗しました: {e}")
    sys.exit(1)

# --- FAISS Vector Store の準備 (インメモリで作成) ---
sample_documents = [
    Document(page_content="LangGraph は状態を持つ複雑なグラフを構築できます。", metadata={"source": "doc-a", "page": 1, "category": "graph"}),
    Document(page_content="LCEL は LangChain Expression Language の略です。", metadata={"source": "doc-b", "page": 1, "category": "core"}),
    Document(page_content="エージェントはツールを使ってタスクを実行します。", metadata={"source": "doc-c", "page": 1, "category": "agent"}),
    Document(page_content="ベクトルストアは埋め込みベクトルを高速に検索します。", metadata={"source": "doc-d", "page": 1, "category": "vectorstore"}),
    Document(page_content="LangGraph ではノード間で状態が共有されます。", metadata={"source": "doc-a", "page": 2, "category": "graph"}),
    Document(page_content="RAG は検索拡張生成の略で、外部知識を利用します。", metadata={"source": "doc-e", "page": 1, "category": "rag"}),
    Document(page_content="プロンプトテンプレートを使うと入力を動的にできます。", metadata={"source": "doc-f", "page": 1, "category": "core"}),
]
try:
    vector_store = FAISS.from_documents(
        documents=sample_documents,
        embedding=embeddings_model
    )
    print("--- Vector Store (FAISS インメモリ) 準備完了 ---")
except Exception as e:
    print(f"エラー: Vector Store (FAISS) の作成中にエラーが発生しました: {e}")
    sys.exit(1)
# step16_rag_retriever_context_revised.py (続き)

print("\n--- Retriever の作成 ---")
try:
    # FAISS Vector Store から .as_retriever() で Retriever を作成
    # search_kwargs で検索の挙動を指定
    retriever = vector_store.as_retriever(
        search_type="similarity", # 類似度で検索 (これが基本)
        search_kwargs={'k': 2}    # 上位2件を取得する設定
    )
    print("Retriever 作成完了。")
    print(f"  検索タイプ: {retriever.search_type}")
    print(f"  取得件数 (k): {retriever.search_kwargs.get('k')}")

except Exception as e:
    print(f"エラー: Retriever の作成中にエラーが発生しました: {e}")
    sys.exit(1)
# step16_rag_retriever_context_revised.py (続き)

# --- Context を取得するための質問 ---
query = "LangGraph の状態管理について教えて" # 聞きたいことを入力
print(f"\n--- Context の取得 (検索クエリ: '{query}') ---")

try:
    # .invoke() メソッドで Retriever に検索を依頼
    context_documents = retriever.invoke(query)

    print(f"取得した Context (Document リスト): {len(context_documents)} 件")

    # 取得した Context の内容を確認します
    print("\n--- 取得した Context の詳細 ---")
    if context_documents:
        for i, doc in enumerate(context_documents):
            print(f"--- Document {i+1} ---")
            print(f"  Content: {doc.page_content}")
            print(f"  Metadata: {doc.metadata}")
    else:
        print("関連する Context が見つかりませんでした。")

except Exception as e:
    # 必要に応じてエラーハンドリングを実装
    print(f"エラー: Context の取得中にエラーが発生しました: {e}")

print("\n--- 処理終了 ---")
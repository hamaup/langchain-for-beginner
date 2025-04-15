# step19_rag_compression.py
import os
import sys
from dotenv import load_dotenv

# --- 必要な LangChain モジュール ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
# Contextual Compression Retriever と EmbeddingsFilter をインポート
try:
    from langchain.retrievers import ContextualCompressionRetriever
    # Document Compressors は通常 langchain 本体から
    from langchain.retrievers.document_compressors import EmbeddingsFilter
    print("ContextualCompressionRetriever と EmbeddingsFilter をインポートしました。")
    # 必要なら: pip install langchain
except ImportError:
    print("エラー: ContextualCompressionRetriever または EmbeddingsFilter が見つかりません。")
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

# --- ステップ 17/18 と同様の準備 ---
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
# (APIキーチェック) ...
print("OpenAI API キーを読み込みました。")
# (Embedding モデルと LLM の準備) ...
try:
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=openai_api_key) # Extractorで使う可能性も考慮
    print(f"--- Embedding モデルと LLM 準備完了 ---")
except Exception as e:
    print(f"エラー: モデルの初期化に失敗しました: {e}")
    sys.exit(1)
# (Vector Store とベース Retriever の準備) ...
sample_documents = [
    Document(page_content="LangChain は LLM アプリケーション開発を支援するフレームワークです。", metadata={"source": "doc-b"}),
    Document(page_content="LCEL は LangChain Expression Language の略で、チェーン構築を容易にします。", metadata={"source": "doc-b"}),
    Document(page_content="エージェントはツールを使って外部システムと対話し、タスクを実行します。", metadata={"source": "doc-c"}),
    Document(page_content="LangGraph は状態を持つ複雑なグラフを構築できます。", metadata={"source": "doc-a"}),
    Document(page_content="ベクトルストア (FAISSなど) は埋め込みの高速検索を可能にします。", metadata={"source": "doc-d"}),
    # わざと関連性の低そうな文書を追加
    Document(page_content="一般的な料理レシピ：カレーライスの作り方。", metadata={"source": "doc-recipe"}),
]
try:
    vector_store = FAISS.from_documents(sample_documents, embeddings_model)
    # ベースの Retriever は少し多めに取得するように k=4 に設定
    base_retriever = vector_store.as_retriever(search_kwargs={'k': 4})
    print("--- Vector Store (FAISS) とベース Retriever 準備完了 (k=4) ---")
except Exception as e:
    print(f"エラー: Vector Store またはベース Retriever の準備中にエラーが発生しました: {e}")
    sys.exit(1)
# step19_rag_compression.py (続き)

print("\n--- Document Compressor (EmbeddingsFilter) の作成 ---")

# EmbeddingsFilter を作成
# embeddings 引数には類似度計算に使う Embedding モデルを渡す
# similarity_threshold 引数で類似度の閾値を指定 (0~1、高いほど厳しい。要調整)
embeddings_filter = EmbeddingsFilter(
    embeddings=embeddings_model,
    similarity_threshold=0.76 # この値は実験して調整する必要がある
)
print("EmbeddingsFilter 作成完了！")
print(f"  使用する Embedding: {embeddings_filter.embeddings.model}")
print(f"  類似度閾値: {embeddings_filter.similarity_threshold}")
# step19_rag_compression.py (続き)

print("\n--- ContextualCompressionRetriever の作成 ---")

# ContextualCompressionRetriever を作成！
# base_compressor に使う Compressor を、base_retriever に元の Retriever を指定
compression_retriever = ContextualCompressionRetriever(
    base_compressor=embeddings_filter,
    base_retriever=base_retriever
)
print("ContextualCompressionRetriever 作成完了！")
print(f"  ベース Retriever: {type(compression_retriever.base_retriever)}")
print(f"  ベース Compressor: {type(compression_retriever.base_compressor)}")
# step19_rag_compression.py (続き)

# --- 検索の実行と比較 ---
query = "LangChain のコアコンセプトは？"
print(f"\n--- 検索実行 (質問: '{query}') ---")

print("\n--- 1. ベース Retriever (k=4) での検索結果 ---")
try:
    base_results = base_retriever.invoke(query)
    print(f"取得した文書数: {len(base_results)}")
    for i, doc in enumerate(base_results):
        # 類似度スコアも見てみる (FAISS は距離なので小さい方が類似)
        # score = vector_store.similarity_search_with_score(query, k=4)[i][1] # 正確ではないが目安
        print(f"  {i+1}. {doc.page_content[:50]}... (Source: {doc.metadata.get('source')})")
except Exception as e:
    print(f"エラー: ベース Retriever の検索中にエラーが発生しました: {e}")


print("\n--- 2. ContextualCompressionRetriever (EmbeddingsFilter) での検索結果 ---")
try:
    # Compression Retriever を実行！
    compressed_results = compression_retriever.invoke(query)
    print(f"圧縮/フィルタリング後の文書数: {len(compressed_results)}")
    for i, doc in enumerate(compressed_results):
        print(f"  {i+1}. {doc.page_content[:50]}... (Source: {doc.metadata.get('source')})")
    print(f"\n-> ベース Retriever の結果 ({len(base_results)}件) からフィルタリングされました。")

except Exception as e:
    print(f"エラー: Compression Retriever の検索中にエラーが発生しました: {e}")


print("\n--- 処理終了 ---")

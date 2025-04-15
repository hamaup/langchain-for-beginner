# step20_rag_parent_document_inmemory_retry_full.py
import os
import sys
from dotenv import load_dotenv
import uuid # ID 生成用

# --- 必要な LangChain モジュール ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
# ParentDocumentRetriever
try:
    from langchain.retrievers import ParentDocumentRetriever
    print("ParentDocumentRetriever をインポートしました (from langchain.retrievers)。")
except ImportError:
    print("エラー: ParentDocumentRetriever が見つかりません。")
    print("   'pip install langchain' を確認してください。")
    sys.exit(1)

# Document Store (今回は InMemoryStore を再挑戦)
try:
    # InMemoryStore を langchain_core からインポート (推奨)
    from langchain_core.stores import InMemoryStore
    print("InMemoryStore をインポートしました (from langchain_core.stores)。")
except ImportError:
    # 古いバージョンや別の場所にある可能性も考慮
    try:
        from langchain.storage import InMemoryStore
        print("InMemoryStore をインポートしました (from langchain.storage)。")
    except ImportError:
        print("エラー: InMemoryStore が見つかりません。")
        print("   'pip install langchain langchain-core' を確認してください。")
        sys.exit(1)

# Vector Store (FAISS)
try:
    from langchain_community.vectorstores import FAISS
    # FAISS 本体も必要
    try:
        import faiss
        print("FAISS をインポートしました。")
    except ImportError:
        print("エラー: faiss ライブラリが見つかりません。")
        print("   'pip install faiss-cpu' または 'pip install faiss-gpu' を実行してください。")
        sys.exit(1)
except ImportError:
    print("エラー: FAISS または langchain-community が見つかりません。")
    print("   'pip install faiss-cpu langchain-community' を実行してください。")
    sys.exit(1)

# Text Splitter
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    print("RecursiveCharacterTextSplitter をインポートしました。")
except ImportError:
    print("エラー: RecursiveCharacterTextSplitter が見つかりません。")
    print("   'pip install langchain-text-splitters' を実行してください。")
    sys.exit(1)

# トークン計算用
try:
    import tiktoken
    print("tiktoken をインポートしました。")
except ImportError:
    print("エラー: tiktoken が見つかりません。")
    print("   'pip install tiktoken' を実行してください。")
    sys.exit(1)

print("--- 必要なモジュールのインポート完了 ---")

# --- 基本設定 (LLM, Embedding) ---
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    print("エラー: OpenAI API キーが設定されていません。")
    sys.exit(1)
else:
    print("OpenAI API キーを読み込みました。")

try:
    # Embedding モデルの次元数を定義 (text-embedding-3-small は 1536次元)
    embedding_dim = 1536
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=openai_api_key) # 将来 RAG チェーンで使う
    print(f"--- Embedding モデル (次元数: {embedding_dim}) と LLM 準備完了 ---")
except Exception as e:
    print(f"エラー: モデルの初期化に失敗しました: {e}")
    sys.exit(1)

# --- トークン計算関数の準備 ---
try:
    tokenizer = tiktoken.get_encoding("cl100k_base")
    print(f"--- tiktoken トークナイザ準備完了 ---")
except Exception as e:
    print(f"エラー: tiktoken の準備に失敗しました: {e}")
    sys.exit(1)
def tiktoken_len(text):
    # 注意: tiktoken は OpenAI モデル固有のトークナイザです
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)
print("--- トークン計算関数準備完了 ---")


# --- ストアの準備 ---
# Parent Document を格納する Docstore (メモリ上に作成)
docstore = InMemoryStore() # ★ InMemoryStore を使用
print("--- Docstore (InMemoryStore) 準備完了 ---")

# Child Document のベクトルを格納する Vectorstore (FAISS, インメモリ)
try:
    index = faiss.IndexFlatL2(embedding_dim)
    # FAISS の初期化時には docstore (InMemoryStoreインスタンス) を渡す必要がある
    vectorstore = FAISS(embedding_function=embeddings_model, index=index, docstore=docstore, index_to_docstore_id={})
    print("--- Vectorstore (FAISS - 空, InMemoryStore連携) 準備完了 ---")
    print("   (これから Retriever が文書を追加します)")
except Exception as e:
    print(f"エラー: 空の FAISS Vectorstore の作成に失敗しました: {e}")
    sys.exit(1)


# --- Splitter の準備 ---
parent_chunk_size = 800
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=50, length_function=tiktoken_len)
child_chunk_size = 200
child_splitter = RecursiveCharacterTextSplitter(chunk_size=child_chunk_size, chunk_overlap=20, length_function=tiktoken_len)
print(f"--- Parent/Child Splitter 準備完了 ---")
# step20_rag_parent_document_inmemory_retry.py (続き)

print("\n--- ParentDocumentRetriever の作成 ---")
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,       # ← InMemoryStore のインスタンス
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
print("ParentDocumentRetriever 作成完了！")
# step20_rag_parent_document_inmemory_retry.py (続き)

# --- 元となる文書 (複数) ---
original_docs = [
    Document(
        page_content="LangChain は、開発者が大規模言語モデル（LLM）やその他の計算ソースを活用して、文脈に応じた推論アプリケーションを構築できるように設計されたフレームワークです。主な価値提案は、コンポーネント性とユースケース中心性です。多様なコンポーネントを組み合わせて特定のユースケースに対応したチェーンやエージェントを構築できます。",
        metadata={"source": "lc_overview.txt"}
    ),
    Document(
        page_content="LCEL (LangChain Expression Language) は、チェーンを簡単に構築・実行するための宣言的な方法を提供します。パイプ演算子 | を使ってコンポーネントを連結でき、ストリーミング、バッチ処理、非同期処理などの機能をすぐに利用できます。入力スキーマと出力スキーマの検証も可能です。",
        metadata={"source": "lcel_intro.txt"}
    ),
    Document(
        page_content="LangChain のエージェントは、LLM を推論エンジンとして使用し、利用可能なツール群からどのツールを、どの順序で、どの入力で呼び出すかを決定します。ReAct や OpenAI Functions Agent など、様々なタイプのエージェントが存在し、タスクに応じて選択できます。ツールには電卓や検索エンジンなどがあります。",
        metadata={"source": "agents_concepts.txt"}
    ),
]
print(f"\n--- 元文書 ({len(original_docs)} 件) を Retriever に追加 ---")
print("   内部で Parent/Child 分割、ベクトル化、ストア格納が行われます...")
print("   ★★★ 注意: この組み合わせでは以前エラーが発生しました ★★★")

try:
    # .add_documents() を呼び出し → ここでエラーが再発する可能性が高い
    retriever.add_documents(original_docs, ids=None)
    print("文書の追加完了！(InMemoryStore でも成功しました！)") # もし成功した場合

except Exception as e:
    print(f"エラー: 文書の追加中にエラーが発生しました: {e}")
    print("   やはり InMemoryStore との組み合わせでは問題が発生するようです。")
    print("   前の手順で試した SQLiteStore を使うか、手動追加の方法を検討してください。")
    # この後の検索は実行できないので、ここで終了するのが自然
    # sys.exit(1) # 必要ならコメント解除
    retrieved_docs = [] # エラーの場合は空リストにしておく

# --- 検索の実行 (エラーが発生しなかった場合のみ意味がある) ---
if retrieved_docs is not None and len(retrieved_docs) > 0: # retrieved_docsがNoneでない、かつ空でないことを確認
    query = "LCEL のストリーミング機能について"
    print(f"\n--- 検索実行 (質問: '{query}') ---")
    try:
        retrieved_docs = retriever.invoke(query)
        print(f"\n--- ParentDocumentRetriever の検索結果 ({len(retrieved_docs)} 件) ---")
        if retrieved_docs:
            for i, doc in enumerate(retrieved_docs):
                print(f"--- 結果 {i+1} ---")
                print(f"Content (Parent - {tiktoken_len(doc.page_content)}トークン): {doc.page_content}")
                print(f"Metadata: {doc.metadata}")
        else:
            print("関連する文書が見つかりませんでした。")
    except Exception as e:
        print(f"エラー: 検索中にエラーが発生しました: {e}")
else:
     print("\n文書追加でエラーが発生したため、検索はスキップします。")


print("\n--- 処理終了 ---")

# step21_rag_citation_final.py
import os
import sys
from dotenv import load_dotenv
from operator import itemgetter # ★ これが必要！

# --- 必要な LangChain モジュール ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
# RunnablePassthrough, RunnableLambda, RunnableParallel をインポート
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel

# Vector Store (FAISS)
try:
    from langchain_community.vectorstores import FAISS
    # FAISS 本体も必要
    try:
        import faiss
        print("FAISS をインポートしました。")
    except ImportError:
        print("エラー: faiss ライブラリが見つかりません。\n   'pip install faiss-cpu' または 'pip install faiss-gpu' を実行してください。")
        sys.exit(1)
except ImportError:
    print("エラー: FAISS または langchain-community が見つかりません。\n   'pip install faiss-cpu langchain-community' を実行してください。")
    sys.exit(1)

# Text Splitter (サンプル文書用)
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    print("RecursiveCharacterTextSplitter をインポートしました。")
except ImportError:
    print("エラー: RecursiveCharacterTextSplitter が見つかりません。\n   'pip install langchain-text-splitters' を実行してください。")
    sys.exit(1)

# トークン計算用
try:
    import tiktoken
    print("tiktoken をインポートしました。")
except ImportError:
    print("エラー: tiktoken が見つかりません。\n   'pip install tiktoken' を実行してください。")
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
    embedding_dim = 1536
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=openai_api_key)
    print(f"--- Embedding モデルと LLM 準備完了 ---")
except Exception as e:
    print(f"エラー: モデルの初期化に失敗しました: {e}")
    sys.exit(1)

# --- Vector Store と Retriever の準備 ---
sample_documents = [
    Document(page_content="LangChain は LLM アプリケーション開発を支援するフレームワークです。", metadata={"source": "doc-b", "page": 1}),
    Document(page_content="LCEL は LangChain Expression Language の略で、チェーン構築を容易にします。", metadata={"source": "doc-b", "page": 2}), # ★ page メタデータあり！
    Document(page_content="エージェントはツールを使って外部システムと対話し、タスクを実行します。", metadata={"source": "doc-c", "page": 1}),
    Document(page_content="LangGraph は状態を持つ複雑なグラフを構築できます。", metadata={"source": "doc-a", "page": 1}),
]
try:
    vector_store = FAISS.from_documents(sample_documents, embeddings_model)
    retriever = vector_store.as_retriever(search_kwargs={'k': 2})
    print("--- Vector Store (FAISS) と Retriever 準備完了 (k=2) ---")
except Exception as e:
    print(f"エラー: Vector Store または Retriever の準備中にエラー: {e}")
    sys.exit(1)

# --- Context フォーマット関数 ---
def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

format_docs_runnable = RunnableLambda(format_docs)
print("--- Context フォーマット関数準備完了 ---")
# step21_rag_citation_final.py (続き)

# RAG 用のプロンプトテンプレート (ステップ 17 と同じ)
template = """
以下の Context のみを基にして、最後の Question に答えてください。

Context:
{context}

Question: {question}
"""
rag_prompt = ChatPromptTemplate.from_template(template)
print("--- RAG プロンプトテンプレート準備完了 ---")
# step21_rag_citation_final.py (続き)

print("\n--- 出典付き RAG チェーンの組み立て ---")

# 1. Context を取得し、質問文字列と一緒に保持する部分 (★ itemgetter を使用 ★)
#    入力: {"question": "..."}
#    出力: {"context": [Document...], "question": "..."} (questionは文字列)
setup_and_retrieval = RunnableParallel(
    context=itemgetter("question") | retriever,
    question=itemgetter("question")
)

# 2. 回答を生成する部分のチェーン (変更なし)
answer_generation_chain = (
    rag_prompt
    | llm
    | StrOutputParser()
)

# 3. 全体を組み立て: 回答生成と Context 保持を並行処理
#    入力は {"question": "..."}
#    出力は {"answer": "...", "context": [Document...]}
rag_chain_with_source = (
    setup_and_retrieval # ① Context と Question を準備
    | RunnableParallel( # ② answer と context を並行して準備
          # "answer" の準備:
          # ①の出力辞書から context ([Docs..]) を取り出し文字列化し、
          # question と一緒に answer_generation_chain へ渡す
          answer=RunnablePassthrough.assign(
              context=itemgetter("context") | format_docs_runnable
          ) | answer_generation_chain,
          # "context" の準備:
          # ①の出力辞書から context ([Docs..]) をそのまま取り出す
          context=itemgetter("context")
      )
)

print("--- 出典付き RAG チェーン組み立て完了 ---")
print("処理の流れ: 入力(question) -> Retriever & 質問保持 -> (並列データ処理: 回答生成 / Context保持) -> 最終辞書出力")
# step21_rag_citation_final.py (続き)

# --- 出典付き RAG チェーンの実行 ---
question = "LCEL について教えて"
print(f"\n--- 出典付き RAG チェーン実行 (質問: '{question}') ---")

try:
    # チェーンを実行！ 出力は answer と context を含む辞書
    rag_output = rag_chain_with_source.invoke({"question": question})

    print("\n--- [AIの回答] ---")
    print(rag_output.get("answer", "回答がありません"))

    print("\n--- [出典情報 (Context)] ---")
    context_docs = rag_output.get("context")
    if context_docs:
        print(f"({len(context_docs)} 件の Context が参照されました)")
        for doc in context_docs:
            source = doc.metadata.get("source", "不明なソース")
            page = doc.metadata.get("page", "-") # page メタデータを表示！
            print(f"- 出典: {source}, ページ: {page}")
    else:
        print("出典情報が見つかりませんでした。")

except Exception as e:
    print(f"エラー: RAG チェーンの実行中にエラーが発生しました: {e}")

print("\n--- 処理終了 ---")
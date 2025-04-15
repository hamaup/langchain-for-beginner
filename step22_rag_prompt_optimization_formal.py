# step22_rag_prompt_optimization_formal.py
import os
import sys
from dotenv import load_dotenv
from operator import itemgetter # ステップ21の修正で使用

# --- 必要な LangChain モジュール ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate # ← これを改良
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel


# Vector Store (FAISS)
try:
    from langchain_community.vectorstores import FAISS
    import faiss
    print("FAISS をインポートしました。")
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
    Document(page_content="LCEL は LangChain Expression Language の略で、チェーン構築を容易にします。", metadata={"source": "doc-b", "page": 2}),
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
# step22_rag_prompt_optimization_formal.py (続き)

print("\n--- ★ 改良版 RAG プロンプトテンプレートの作成 ★ ---")

# 改良版プロンプトテンプレート
new_template = """
以下の指示に従って、与えられた Context の情報のみを使用して質問に回答してください。
- 回答は Context の内容に厳密に基づいて生成してください。
- Context に質問に対する回答が明確に含まれていない場合は、追加情報を提供しようとせず、「提供された情報の中には、ご質問に該当する内容が見つかりませんでした。」と正確に回答してください。
- Context 以外の知識や外部の情報を使用しないでください。

<context>
{context}
</context>

<question>
{question}
</question>

回答:"""

# 新しいテンプレートから ChatPromptTemplate を作成
new_rag_prompt = ChatPromptTemplate.from_template(new_template)

print("--- 改良版 RAG プロンプトテンプレート準備完了 ---")
print("--- 新しい指示内容 ---")
print(new_template)
# step22_rag_prompt_optimization_formal.py (続き)

print("\n--- 改良版プロンプトを使った RAG チェーンの組み立て ---")

# 1. Context 取得と質問保持 (ステップ 21 と同じ)
setup_and_retrieval = RunnableParallel(
    context=itemgetter("question") | retriever,
    question=itemgetter("question")
)

# 2. 回答生成チェーン (★プロンプトを new_rag_prompt に変更★)
new_answer_generation_chain = (
    new_rag_prompt # ← 差し替え
    | llm
    | StrOutputParser()
)

# 3. 全体の組み立て (★回答生成部分を new_answer_generation_chain に変更★)
new_rag_chain_with_source = (
    setup_and_retrieval
    | RunnableParallel(
          answer=RunnablePassthrough.assign(
              context=itemgetter("context") | format_docs_runnable
          ) | new_answer_generation_chain, # ← 差し替え
          context=itemgetter("context")
      )
)

print("--- 改良版 RAG チェーン組み立て完了 ---")
# step22_rag_prompt_optimization_formal.py (続き)

# --- 出典表示用関数 ---
def print_sources(context_docs):
    if context_docs:
        print(f"({len(context_docs)} 件の Context が参照されました)")
        for doc in context_docs:
            source = doc.metadata.get("source", "不明")
            page = doc.metadata.get("page", "-")
            print(f"- 出典: {source}, ページ: {page}")
    else:
        print("出典情報が見つかりませんでした。")

# --- 改良版 RAG チェーンの実行 ---

# --- ケース 1: Context に答えがある質問 ---
question_1 = "LCEL は何の略称ですか？"
print(f"\n--- 実行ケース 1 (質問: '{question_1}') ---")
try:
    output_1 = new_rag_chain_with_source.invoke({"question": question_1})
    print("\n[AIの回答 1]")
    print(output_1.get("answer", "回答なし"))
    print("\n[出典情報 1]")
    print_sources(output_1.get("context"))


except Exception as e: print(f"エラー1: {e}")

# --- ケース 2: Context に答えがない質問 ---
question_2 = "Python の async/await について詳しく教えて" # サンプル文書にはない内容
print(f"\n--- 実行ケース 2 (質問: '{question_2}') ---")
try:
    output_2 = new_rag_chain_with_source.invoke({"question": question_2})
    print("\n[AIの回答 2]")
    print(output_2.get("answer", "回答なし")) # ← 指示通りの応答を期待
    print("\n[出典情報 2]")
    print_sources(output_2.get("context")) # Context自体は取得される場合がある

except Exception as e: print(f"エラー2: {e}")


print("\n--- 処理終了 ---")

# step10_lcel_with_memory_final.py
import os
import sys
from dotenv import load_dotenv

# 基本的な LCEL コンポーネント
from langchain_openai import ChatOpenAI
# MessagesPlaceholder を langchain_core からインポート
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

# メモリ関連コンポーネント
# ChatMessageHistory は langchain_community から
try:
    from langchain_community.chat_message_histories import ChatMessageHistory
    print("ChatMessageHistory をインポートしました。")
except ImportError:
    print("エラー: langchain-community がインストールされていません。")
    print("`pip install langchain-community` を実行してください。")
    sys.exit(1)
# BaseChatMessageHistory は型ヒントのため (必須ではない)
from langchain_core.chat_history import BaseChatMessageHistory
# RunnableWithMessageHistory を直接インポート
from langchain_core.runnables.history import RunnableWithMessageHistory

print("--- 必要なモジュールのインポート完了 ---")

# APIキーなどをロード
load_dotenv()

# LLM の準備
try:
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    print("--- LLM準備完了 ---")
except Exception as e:
    print(f"エラー: LLMの初期化に失敗しました: {e}")
    sys.exit(1)

# プロンプトテンプレートの定義
prompt = ChatPromptTemplate.from_messages([
    ("system", "あなたはユーザーの質問に簡潔かつ親切に答えるアシスタントです。"),
    MessagesPlaceholder(variable_name="history"), # 履歴用のプレースホルダ
    ("human", "{input}") # ユーザーの現在の入力
])
print("--- プロンプトテンプレート定義完了 ---")

# 出力パーサーの準備
output_parser = StrOutputParser()
print("--- 出力パーサー準備完了 ---")

# メモリ機能を含まない基本チェーン
chain_base = prompt | llm | output_parser
print("--- 基本チェーン定義完了 (prompt | llm | output_parser) ---")


# --- 履歴管理機能の準備 ---
# セッションID と ChatMessageHistory オブジェクトのマッピングを保持する辞書
# 注意: この store はインメモリであり、プログラムを再起動すると内容は失われます。
store = {}

# 型ヒントには BaseChatMessageHistory を使うのがより一般的
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    セッションIDに対応する BaseChatMessageHistory オブジェクトを取得または新規作成する関数。
    """
    if session_id not in store:
        print(f"  (新規セッション開始: {session_id})")
        store[session_id] = ChatMessageHistory()
    else:
        print(f"  (既存セッション再開: {session_id})")
    return store[session_id]

print("--- 履歴管理関数 (get_session_history) 定義完了 ---")
print(f"初期の履歴ストア: {store}")


# --- メモリ機能付きチェーンの作成 ---
# RunnableWithMessageHistory クラスを直接使ってインスタンス化する
chain_with_memory = RunnableWithMessageHistory(
    runnable=chain_base,             # ラップする基本チェーンを指定
    get_session_history=get_session_history, # 上で定義した履歴管理関数
    input_messages_key="input",     # プロンプト内のユーザー入力変数名
    history_messages_key="history"  # プロンプト内の履歴プレースホルダ名
)
print("--- メモリ機能付きチェーン作成完了 (RunnableWithMessageHistoryを使用) ---")

# (補足) LangChainには .with_message_history() という便利なメソッドもありますが、
# バージョンによっては動作しない場合があるため、今回はクラスを直接使う方法を採用しています。


# --- チェーンの実行と確認 (一般的な質問を使用) ---
# --- セッションAでの対話 ---
session_a_id = "user_alice"
print(f"\n--- セッション '{session_a_id}' での対話開始 ---")

print("\n[アリス] 1回目の入力:")
config_a = {"configurable": {"session_id": session_a_id}}
response_a1 = chain_with_memory.invoke(
    {"input": "こんにちは、私の名前はアリスです。"},
    config=config_a
)
print(f"AI: {response_a1}")

print("\n[アリス] 2回目の入力:")
response_a2 = chain_with_memory.invoke(
    {"input": "私の名前、覚えていますか？"},
    config=config_a
)
print(f"AI: {response_a2}")

# --- セッションBでの対話 ---
session_b_id = "user_bob"
print(f"\n--- セッション '{session_b_id}' での対話開始 ---")

print("\n[ボブ] 1回目の入力:")
config_b = {"configurable": {"session_id": session_b_id}}
response_b1 = chain_with_memory.invoke(
    {"input": "日本の首都はどこですか？"}, # 一般的な質問に変更
    config=config_b
)
print(f"AI: {response_b1}")

# --- 再度セッションAでの対話 ---
print(f"\n--- セッション '{session_a_id}' に戻って対話 ---")

print("\n[アリス] 3回目の入力:")
response_a3 = chain_with_memory.invoke(
    {"input": "面白いジョークを一つ教えてください。"}, # 一般的な質問に変更
    config=config_a
)
print(f"AI: {response_a3}")

# --- 履歴ストアの中身を確認 ---
print("\n--- 最終的な履歴ストアの中身 ---")
for session_id, history_obj in store.items():
    print(f"セッションID: {session_id}")
    for msg in history_obj.messages:
        print(f"  - {type(msg).__name__}: {msg.content}")

print("\n--- 処理終了 ---")
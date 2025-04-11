# step8_memory_bot_refined.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# --- LangChain Core Components ---
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
)
from langchain_core.messages import SystemMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# --- Chat History Implementation (from Community) ---
# Requires: pip install langchain-community
try:
    from langchain_community.chat_message_histories import ChatMessageHistory
except ImportError:
    print("エラー: 'langchain-community' パッケージが必要です。")
    print("コマンドラインで 'pip install langchain-community' を実行してください。")
    exit()

# --- Initial Setup ---
load_dotenv()
print("環境変数を読み込みました。")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
print(f"LLM ({llm.model_name}) の準備完了。")

# --- In-memory Chat History Store ---
session_memory_store = {}
def get_chat_history_for_session(session_id: str) -> BaseChatMessageHistory:
    if session_id not in session_memory_store:
        print(f"システム: 新しいセッション (ID: {session_id}) の履歴を作成します。")
        session_memory_store[session_id] = ChatMessageHistory()
    else:
        print(f"システム: 既存のセッション (ID: {session_id}) の履歴を使用します。")
    return session_memory_store[session_id]
print("インメモリ会話履歴ストアの準備完了。")

# --- Define the prompt template ---
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="あなたはフレンドリーなアシスタントです。以前の会話を考慮して応答してください。"),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{user_input}")
])
print("プロンプトテンプレートを作成しました。")

# --- Build the base LCEL chain ---
base_chain = prompt | llm
print("基本LCELチェーンを構築しました。")

# --- Wrap the base chain with memory management ---
chain_with_memory = RunnableWithMessageHistory(
    base_chain,
    get_chat_history_for_session,
    input_messages_key="user_input",
    history_messages_key="chat_history",
)
print("メモリ管理機能付きチェーンを構築しました。")

# --- Start Conversation ---
session_id = "user123_session_A"
print(f"\n会話を開始します (セッションID: {session_id})")
chat_config = {"configurable": {"session_id": session_id}}

# --- Interaction 1: Introduction ---
print("\n--- 1回目の対話 ---")
input1 = {"user_input": "こんにちは。私の名前は佐藤です。"}
print(f"あなた: {input1['user_input']}")
try:
    response1 = chain_with_memory.invoke(input1, config=chat_config)
    print(f"AI: {response1.content}")
except Exception as e:
    print(f"エラーが発生しました: {e}")

# --- Interaction 2: Casual Talk ---
print("\n--- 2回目の対話 ---")
input2 = {"user_input": "おすすめの映画はありますか？"}
print(f"あなた: {input2['user_input']}")
try:
    response2 = chain_with_memory.invoke(input2, config=chat_config)
    print(f"AI: {response2.content}")
except Exception as e:
    print(f"エラーが発生しました: {e}")

# --- Interaction 3: Memory Check ---
print("\n--- 3回目の対話 ---")
input3 = {"user_input": "私の名前を覚えていますか？"}
print(f"あなた: {input3['user_input']}")
try:
    response3 = chain_with_memory.invoke(input3, config=chat_config)
    print(f"AI: {response3.content}") # Expect the AI to respond with "佐藤さん"
except Exception as e:
    print(f"エラーが発生しました: {e}")

# --- Inspect the Chat History (Optional) ---
print("\n--- 会話履歴の確認 ---")
try:
    final_history = get_chat_history_for_session(session_id)
    if final_history.messages:
        for message in final_history.messages:
            print(f"[{message.type.upper()}] {message.content}")
    else:
        print("(履歴は空です)")
except Exception as e:
    print(f"履歴の取得中にエラーが発生しました: {e}")

print("\n会話を終了します。")
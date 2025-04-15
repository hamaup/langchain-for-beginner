# step11_bot_personality.py
import os
import sys
from dotenv import load_dotenv

# 基本的な LCEL コンポーネント
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage # 固定の SystemMessage を使う
from langchain_core.output_parsers import StrOutputParser

# メモリ関連コンポーネント (ステップ10と同様)
try:
    from langchain_community.chat_message_histories import ChatMessageHistory
    print("ChatMessageHistory をインポートしました。")
except ImportError:
    print("エラー: langchain-community がインストールされていません。")
    print("`pip install langchain-community` を実行してください。")
    sys.exit(1)
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

print("--- 必要なモジュールのインポート完了 ---")

load_dotenv()

# LLM の準備 (ステップ10と同様)
try:
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    print("--- LLM準備完了 ---")
except Exception as e:
    print(f"エラー: LLMの初期化に失敗しました: {e}")
    sys.exit(1)

# 履歴ストアと管理関数 (ステップ10と同様)
store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        print(f"  (新規セッション開始: {session_id})")
        store[session_id] = ChatMessageHistory()
    else:
        print(f"  (既存セッション再開: {session_id})")
    return store[session_id]
print("--- 履歴管理関数 (get_session_history) 定義完了 ---")
# step11_bot_personality.py (続き)

print("\n--- 新しいプロンプトテンプレートを定義 (役割設定込み) ---")

# ★★★ ここで役割を定義 ★★★
# 例: 古代の賢者の役割
system_prompt_wise_sage = SystemMessage(
    content="""あなたは古代から生きる賢者です。常に落ち着き払っており、深遠な知識を持っています。
    質問者に対しては、少し古風で、含蓄のある言葉遣いで、簡潔に本質を突くような回答を心がけてください。
    現代の俗語や軽い表現は避け、威厳を保ってください。"""
)

# # 例: 未来の案内ロボットの役割 (試したい場合はこちらを有効化)
# system_prompt_future_robot = SystemMessage(
#     content="""ピポッ！ワタシハ未来カラ来タ案内ロボット「ユニット734」デス。
#     正確ナ情報ヲ、効率的ニ、論理的ナ口調デ提供シマス。
#     感情表現はミニマム、データに基づイタ回答ヲシマス。エネルギー効率ノタメ、回答は簡潔ニ。ピポッ。"""
# )

# ★★★ SystemMessage を含めてプロンプトテンプレートを作成 ★★★
prompt_with_personality = ChatPromptTemplate.from_messages([
    system_prompt_wise_sage, # 上で定義した SystemMessage を設定
    # system_prompt_future_robot, # ロボット役を試す場合はこちらを有効化
    MessagesPlaceholder(variable_name="history"), # 履歴用プレースホルダ (ステップ10と同様)
    HumanMessagePromptTemplate.from_template("{input}") # ユーザー入力 (ステップ10と同様)
])

print(f"--- プロンプト定義完了 (役割: {system_prompt_wise_sage.content[:30]}...) ---")
# print(f"--- プロンプト定義完了 (役割: {system_prompt_future_robot.content[:30]}...) ---") # ロボット役の場合

# 出力パーサー (ステップ10と同様)
output_parser = StrOutputParser()
print("--- 出力パーサー準備完了 ---")

# ★★★ 新しいプロンプトを使った基本チェーン ★★★
chain_base_with_personality = prompt_with_personality | llm | output_parser
print("--- 新しいプロンプトを含む基本チェーン定義完了 ---")
# step11_bot_personality.py (続き)

# ★★★ 新しい基本チェーンをラップしてメモリ機能付きチェーンを作成 ★★★
chain_with_memory_and_personality = RunnableWithMessageHistory(
    runnable=chain_base_with_personality, # ★ラップ対象を新しいチェーンに変更★
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)
print("--- メモリ機能付き・性格設定済みチェーン作成完了 ---")
# step11_bot_personality.py (続き)

# --- 性格設定済みチェーンの実行と確認 ---
session_id_personality = "session_wise_sage_01" # セッションIDを変更
# session_id_personality = "session_robot_01" # ロボット役の場合
print(f"\n--- セッション '{session_id_personality}' での対話開始 ---")
config_personality = {"configurable": {"session_id": session_id_personality}}

# 1回目の対話
print("\n[あなた] 1回目の入力:")
response1 = chain_with_memory_and_personality.invoke(
    {"input": "人生で最も大切なことは何でしょうか？"},
    config=config_personality
)
print(f"AI ({system_prompt_wise_sage.content[:6]}...): {response1}")
# print(f"AI ({system_prompt_future_robot.content[:6]}...): {response1}") # ロボット役の場合

# 2回目の対話 (履歴を踏まえるか？)
print("\n[あなた] 2回目の入力:")
response2 = chain_with_memory_and_personality.invoke(
    {"input": "なるほど。では、幸福について一言お願いします。"},
    config=config_personality # 同じセッションID
)
print(f"AI ({system_prompt_wise_sage.content[:6]}...): {response2}")
# print(f"AI ({system_prompt_future_robot.content[:6]}...): {response2}") # ロボット役の場合

# 3回目の対話 (一貫性は保たれるか？)
print("\n[あなた] 3回目の入力:")
response3 = chain_with_memory_and_personality.invoke(
    {"input": "ありがとうございます。あなたの知識はどこから来るのですか？"},
    config=config_personality # 同じセッションID
)
print(f"AI ({system_prompt_wise_sage.content[:6]}...): {response3}")
# print(f"AI ({system_prompt_future_robot.content[:6]}...): {response3}") # ロボット役の場合

# --- 履歴ストアの中身を確認 (オプション) ---
print("\n--- 最終的な履歴ストアの中身 ---")
if session_id_personality in store:
    history_obj = store[session_id_personality]
    for msg in history_obj.messages:
        print(f"  - {type(msg).__name__}: {msg.content}")
else:
    print("  (履歴が見つかりません)")


print("\n--- 処理終了 ---")

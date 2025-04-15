# step28_agent_memory.py
import os
import sys
from dotenv import load_dotenv
from operator import itemgetter

# --- 必要な LangChain モジュール ---
try:
    from langchain_openai import ChatOpenAI
    print("ChatOpenAI をインポートしました。")
except ImportError:
    print("エラー: langchain_openai が見つかりません。\n   'pip install langchain-openai' を確認してください。")
    sys.exit(1)

try:
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # ★ MessagesPlaceholder をインポート！
    print("ChatPromptTemplate, MessagesPlaceholder をインポートしました。")
except ImportError:
    print("エラー: langchain_core.prompts が見つかりません。\n   'pip install langchain-core' を確認してください。")
    sys.exit(1)

try:
    from langchain.agents import create_openai_tools_agent, AgentExecutor
    print("create_openai_tools_agent と AgentExecutor をインポートしました。")
except ImportError:
    print("エラー: エージェント関連モジュールが見つかりません。\n   'pip install langchain' を確認してください。")
    sys.exit(1)

try:
    from langchain import hub
    print("langchain hub をインポートしました。")
except ImportError:
    print("エラー: langchain hub が見つかりません。\n   'pip install langchainhub' を実行してください。")
    sys.exit(1)

# ★ メモリクラスをインポート！
try:
    # メモリ関連クラスは langchain パッケージにあることが多い
    from langchain.memory import ConversationBufferMemory
    print("ConversationBufferMemory をインポートしました (from langchain.memory)。")
except ImportError:
    print("エラー: ConversationBufferMemory が見つかりません。")
    print("   'pip install langchain' を確認してください。")
    sys.exit(1)

# ツール (DuckDuckGo)
try:
    from langchain_community.tools import DuckDuckGoSearchRun
    print("DuckDuckGoSearchRun をインポートしました。")
    # 必要なら: pip install langchain-community duckduckgo-search
except ImportError:
    print("エラー: DuckDuckGoSearchRun が見つかりません。\n   'pip install langchain-community duckduckgo-search' を確認してください。")
    sys.exit(1)

print("--- 必要なモジュールのインポート完了 ---")

# --- 基本設定 (LLM) ---
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    print("エラー: OpenAI API キーが設定されていません。")
    sys.exit(1)
else:
    print("OpenAI API キーを読み込みました。")

try:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_api_key)
    print(f"--- LLM ({llm.model_name}) 準備完了 ---")
except Exception as e:
    print(f"エラー: モデルの初期化に失敗しました: {e}")
    sys.exit(1)

# --- ツールの準備 ---
try:
    search_tool = DuckDuckGoSearchRun()
    tools = [search_tool]
    print("--- 検索ツール (DuckDuckGoSearchRun) 準備完了 ---")
    print(f"--- エージェント用ツールリスト準備完了 (計 {len(tools)} 個) ---")
except Exception as e:
    print(f"エラー: ツールの準備中にエラー: {e}")
    sys.exit(1)
# step28_agent_memory.py (続き)

print("\n--- メモリとプロンプトの準備 ---")

# 1. メモリの作成 (キー名を 'chat_history' に設定)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
print(f"メモリ ({type(memory).__name__}) を作成しました (memory_key='chat_history')。")

# 2. プロンプトの取得と確認
try:
    prompt = hub.pull("hwchase17/openai-tools-agent")
    print("LangChain Hub からプロンプト 'hwchase17/openai-tools-agent' を取得しました。")

    # ★★★ プロンプトに必要な変数を確認！ 'chat_history' があるはず ★★★
    print("プロンプトが期待する入力変数:", prompt.input_variables)
    if "chat_history" not in prompt.input_variables:
        print("【重要】警告: 取得したプロンプトに 'chat_history' が含まれていません！メモリが機能しません。")
        print("         プロンプトを修正するか、MessagesPlaceholder('chat_history') を含む別のプロンプトを使用してください。")
        # sys.exit(1) # 止める場合
    elif "input" not in prompt.input_variables or "agent_scratchpad" not in prompt.input_variables:
        print("【重要】警告: 取得したプロンプトに 'input' または 'agent_scratchpad' が不足しています。")
        # sys.exit(1) # 止める場合
    else:
        print("プロンプトに必要なプレースホルダー ('input', 'agent_scratchpad', 'chat_history') が含まれています。OK！")

except Exception as e:
    print(f"エラー: LangChain Hub からのプロンプト取得に失敗: {e}")
    sys.exit(1)
# step28_agent_memory.py (続き)

print("\n--- エージェント (意思決定ロジック) の作成 ---")
try:
    agent = create_openai_tools_agent(llm, tools, prompt)
    print("OpenAI Tools agent を作成しました！")
except Exception as e:
    print(f"エラー: エージェントの作成中にエラー: {e}")
    sys.exit(1)
# step28_agent_memory_final.py (続き)

print("\n--- AgentExecutor の作成 (メモリ付き！) ---")
try:
    # ★★★ memory 引数に作成した memory オブジェクトを渡す！ ★★★
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory, # ← コレ！ これで記憶機能 ON！
        verbose=True   # 動作確認のために True に
    )
    print("メモリ付き AgentExecutor を作成しました (verbose=True)。")

except Exception as e:
    print(f"エラー: AgentExecutor の作成中にエラーが発生しました: {e}")
    sys.exit(1)
# step28_agent_memory.py (続き)

# --- 会話形式での実行 ---
print("\n--- 1回目の会話 ---")
input1 = "僕の名前は AI 見習い です。よろしく！"
print(f"あなた: {input1}")
try:
    response1 = agent_executor.invoke({"input": input1})
    print(f"エージェント: {response1.get('output', '応答なし')}")
except Exception as e:
    print(f"エラー1: {e}")

print("\n--- 2回目の会話 ---")
input2 = "僕の名前、覚えてる？" # 1回目の発言を踏まえた質問！
print(f"あなた: {input2}")
try:
    response2 = agent_executor.invoke({"input": input2})
    print(f"エージェント: {response2.get('output', '応答なし')}") # ←「AI 見習いさんですね」と答えるはず！
except Exception as e:
    print(f"エラー2: {e}")

print("\n--- 3回目の会話 (記憶を使ってツール実行) ---")
input3 = "僕の名前（AI 見習い）に関する最近のニュースを Web で検索してくれない？"
print(f"あなた: {input3}")
try:
    response3 = agent_executor.invoke({"input": input3})
    print(f"エージェント: {response3.get('output', '応答なし')}") # ← 名前を使って検索するはず
except Exception as e:
    print(f"エラー3: {e}")

# (おまけ) メモリの中身を確認
print("\n--- (おまけ) 現在のメモリの中身 ---")
try:
    # memory.load_memory_variables({}) で現在のメモリ内容を取得
    # 'chat_history' キーで Message オブジェクトのリストが返る
    current_memory = memory.load_memory_variables({})
    print(current_memory.get("chat_history"))
except Exception as e:
    print(f"メモリ内容の取得エラー: {e}")

print("\n--- 処理終了 ---")
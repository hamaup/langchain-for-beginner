# step27_agent_execution.py
import os
import sys
from dotenv import load_dotenv

# --- 必要な LangChain モジュール ---
try:
    from langchain_openai import ChatOpenAI
    print("ChatOpenAI をインポートしました (from langchain_openai)。")
    # 必要なら: pip install langchain-openai
except ImportError:
    print("エラー: langchain_openai が見つかりません。\n   'pip install langchain-openai' を確認してください。")
    sys.exit(1)

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document # これは直接は使わないが参考として

# エージェント作成と実行に必要なもの
try:
    from langchain.agents import create_openai_tools_agent, AgentExecutor
    print("create_openai_tools_agent と AgentExecutor をインポートしました。")
    # 必要なら: pip install langchain
except ImportError:
    print("エラー: エージェント関連モジュールが見つかりません。\n   'pip install langchain' を確認してください。")
    sys.exit(1)

# LangChain Hub からプロンプトを読み込むため
try:
    from langchain import hub
    print("langchain hub をインポートしました。")
except ImportError:
    print("エラー: langchain hub が見つかりません。\n   'pip install langchainhub' を実行してください。")
    sys.exit(1)

# ステップ 25 で準備したツール (DuckDuckGo)
try:
    from langchain_community.tools import DuckDuckGoSearchRun
    print("DuckDuckGoSearchRun をインポートしました。")
    # 必要なら: pip install langchain-community duckduckgo-search
except ImportError:
    print("エラー: DuckDuckGoSearchRun が見つかりません。\n   'pip install langchain-community duckduckgo-search' を確認してください。")
    sys.exit(1)

# (ステップ 24 のカスタムツールも使うならインポート)
try:
    from step24_custom_tool_pydantic_final import search_user_info # ファイル名・変数名を修正
    print("カスタムツール (search_user_info) をインポートしました。")
except ImportError:
    print("警告: カスタムツール (search_user_info) が見つかりません。組み込みツールのみ使用します。")
    search_user_info = None

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
    # Tool Calling 対応モデルを選択 (gpt-4o-mini は比較的高速で安価)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_api_key)
    print(f"--- LLM ({llm.model_name}) 準備完了 ---")
except Exception as e:
    print(f"エラー: モデルの初期化に失敗しました: {e}")
    sys.exit(1)

# --- ツールの準備 ---
try:
    search_tool = DuckDuckGoSearchRun()
    print("--- 検索ツール (DuckDuckGoSearchRun) 準備完了 ---")
    # ★ エージェントに渡すツールリストを作成 ★
    tools = [search_tool]
    if search_user_info: # カスタムツールがあれば追加
        tools.append(search_user_info)
        print("カスタムツールもリストに追加しました。")
    print(f"--- エージェント用ツールリスト準備完了 (計 {len(tools)} 個) ---")
    print("ツールリスト:")
    for t in tools:
        print(f"  - {t.name}: {t.description}") # ツール名と説明を確認
except Exception as e:
    print(f"エラー: ツールの準備中にエラー: {e}")
    sys.exit(1)
# step27_agent_execution.py (続き)

print("\n--- エージェント用プロンプトの取得 ---")
try:
    # LangChain Hub から OpenAI Tools agent 用の推奨プロンプトを pull (取得)
    prompt = hub.pull("hwchase17/openai-tools-agent")
    print("LangChain Hub からプロンプト 'hwchase17/openai-tools-agent' を取得しました。")

    # ★ プロンプトに必要なプレースホルダーを確認 (重要) ★
    print("プロンプトが必要とする入力変数:", prompt.input_variables)
    # -> 通常、'input', 'agent_scratchpad' が含まれます。
    #    チャット履歴を扱う場合は 'chat_history' も必要になります。

except Exception as e:
    print(f"エラー: LangChain Hub からのプロンプト取得に失敗: {e}")
    print("   ネットワーク接続を確認するか、 'pip install langchainhub' を試してください。")
    sys.exit(1)
# step27_agent_execution.py (続き)

print("\n--- エージェント (意思決定ロジック) の作成 ---")
try:
    # OpenAI Tools agent ロジック (Runnable) を作成
    agent = create_openai_tools_agent(llm, tools, prompt)
    print("OpenAI Tools agent を作成しました！")
    # agent 自体も Runnable なので、より複雑なチェーンに組み込むことも可能
    # print(type(agent)) # -> Runnable

except Exception as e:
    print(f"エラー: エージェントの作成中にエラー: {e}")
    sys.exit(1)
# step27_agent_execution.py (続き)

print("\n--- AgentExecutor の作成と実行 ---")
try:
    # AgentExecutor を作成
    # verbose=True で途中の思考プロセスやツール呼び出しを表示
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    print("AgentExecutor を作成しました (verbose=True)。")

    # エージェントに仕事を依頼！ 質問を入力辞書で渡す
    # プロンプトが 'input' を期待しているので、キーを 'input' にする
    user_input = "LangChain の開発元と、今日の東京の天気を教えてください。"
    print(f"\n--- エージェント実行開始！ (入力: '{user_input}') ---")

    # .invoke() でエージェント実行！
    response = agent_executor.invoke({"input": user_input})

    print("\n--- エージェント実行完了！ ---")
    print("\n--- 最終的な応答 ---")
    # 応答の辞書から 'output' キーで最終回答を取得
    print(response.get('output', '応答がありませんでした'))

except Exception as e:
    print(f"エラー: AgentExecutor の作成または実行中にエラーが発生しました: {e}")

print("\n--- 処理終了 ---")

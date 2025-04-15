# step35_multi_agent_implementation_revised.py
import os
import sys
from dotenv import load_dotenv
from typing import TypedDict, Optional, Dict, Any, List, Literal

# LangGraph
try:
    from langgraph.graph import StateGraph, START, END
    print("LangGraph をインポートしました。")
except ImportError:
    print("エラー: langgraph が見つかりません。'pip install langgraph'")
    sys.exit(1)

# LLM, Prompt, Messages
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage # AIMessage は直接使わないかも

# Agent & Tools
try:
    from langchain.agents import AgentExecutor, create_openai_tools_agent
    from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
    print("Agent と Tool 関連モジュールをインポートしました。")
except ImportError:
    print("エラー: langchain または langchain_community が不足している可能性があります。")
    print("   'pip install langchain langchain-community' を実行してください。")
    sys.exit(1)

print("--- 必要なモジュールのインポート完了 ---")

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    print("エラー: OpenAI API キーが設定されていません。")
    sys.exit(1)
else:
    print("OpenAI API キーを読み込みました。")

# LLM の準備
# step35_multi_agent_implementation_revised.py (LLM 初期化部分の修正)

# LLM の準備
try:
    # gpt-4o や gpt-3.5-turbo など、利用可能なモデルを指定
    llm = ChatOpenAI(model="gpt-4o", temperature=0.1, api_key=openai_api_key)
    print(f"--- LLM準備完了 ---") # ひとまず成功したことを出力

    # ★★★ モデル名表示部分を修正 ★★★
    # 属性名 'model' ではなく 'model_name' を試す
    try:
        print(f"  (使用モデル: {llm.model_name})")
    except AttributeError:
        # もし 'model_name' もない場合は、指定したモデル名を直接表示する
        print(f"  (使用モデル: gpt-4o)") # 初期化時に指定したモデル名を書く
        print("  (注意: model_name 属性が見つかりませんでした)")
    # ★★★ 修正ここまで ★★★

except Exception as e:
    print(f"エラー: LLMの初期化に失敗しました: {e}")
    sys.exit(1)
# ツールの準備 (リサーチャー用)
try:
    search_tool = DuckDuckGoSearchRun()
    print("--- DuckDuckGo Search ツール準備完了 ---")
except ImportError:
    print("エラー: DuckDuckGo Search を使うには 'duckduckgo-search' が必要です。")
    print("   'pip install -U duckduckgo-search' を実行してください。")
    sys.exit(1)
# step35_multi_agent_implementation_revised.py (続き)

print("\n--- エージェントの作成 ---")

# --- リサーチャーエージェント ---
researcher_prompt = ChatPromptTemplate.from_messages([
    ("system", "あなたは優秀なリサーチャーです。与えられたトピックについて DuckDuckGo Search ツールを必ず使って Web 検索を行い、最新かつ信頼性の高い情報源から重要なポイントを収集し、結果を箇条書きで要約してください。"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
    ("human", "調査トピック: {input}")
])
researcher_tools = [search_tool]
researcher_agent_runnable = create_openai_tools_agent(llm, researcher_tools, researcher_prompt)
researcher_executor = AgentExecutor(agent=researcher_agent_runnable, tools=researcher_tools, verbose=True)
print("リサーチャーエージェントを作成しました。")

# --- ライターエージェント ---
writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "あなたはプロのライターです。提供された元のリクエストとリサーチ結果の要約に基づいて、読者にとって分かりやすく、簡潔で、客観的なレポートを作成してください。リサーチ結果を適切に引用・参照し、レポートのタイトルも付けてください。"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
    ("human", "以下の情報からレポートを作成してください:\n\n# 元のリクエスト:\n{request}\n\n# リサーチ結果要約:\n{research_summary}")
])
writer_tools = [] # ライターはツールを使わない
writer_agent_runnable = create_openai_tools_agent(llm, writer_tools, writer_prompt)
writer_executor = AgentExecutor(agent=writer_agent_runnable, tools=writer_tools, verbose=True)
print("ライターエージェントを作成しました。")
# step35_multi_agent_implementation_revised.py (続き)

# グラフの状態を定義
class ResearchWorkflowState(TypedDict):
    request: str
    research_summary: Optional[str]
    report: Optional[str]
    error: Optional[str] # エラーメッセージ用

print("\n--- グラフの状態 (ResearchWorkflowState) 定義完了 ---")
# step35_multi_agent_implementation_revised.py (続き)

# --- ノード関数 の定義 ---

def run_researcher(state: ResearchWorkflowState) -> Dict[str, Optional[str]]:
    """リサーチャーエージェントを実行し、結果またはエラーを返す"""
    print("\n--- ノード: Researcher 実行 ---")
    try:
        response = researcher_executor.invoke({"input": state['request']})
        summary = response.get("output")
        if summary:
             print(f"  -> リサーチ結果取得 (一部): {summary[:100]}...")
             # エラーがなければ error を None にしておく (重要)
             return {"research_summary": summary, "error": None}
        else:
             print("  -> リサーチ結果が空です。")
             return {"error": "Researcher returned empty summary."}
    except Exception as e:
        print(f"  -> リサーチャー実行エラー: {e}")
        return {"error": f"Researcher failed: {e}"}

def run_writer(state: ResearchWorkflowState) -> Dict[str, Optional[str]]:
    """ライターエージェントを実行し、結果またはエラーを返す"""
    print("\n--- ノード: Writer 実行 ---")
    # 前のステップでエラーが発生していないか確認 (エラーがあればスキップも可能だが今回はそのまま進む)
    if state.get("error"):
        print("  -> 前のステップでエラーが発生したため、ライターはスキップします。")
        # エラー状態を維持するか、ここで処理を決める
        return {} # 何も更新しない

    summary = state.get('research_summary')
    if not summary: # None または空文字列の場合
        print("  -> エラー: リサーチ結果がありません。")
        return {"error": "Writer error: Research summary is missing or empty."}

    try:
        writer_input = {
            "request": state['request'],
            "research_summary": summary
        }
        response = writer_executor.invoke(writer_input)
        report_text = response.get("output")
        if report_text:
            print(f"  -> レポート生成完了 (一部): {report_text[:100]}...")
            return {"report": report_text, "error": None} # 成功時は error を None に
        else:
            print("  -> レポートが空です。")
            return {"error": "Writer returned empty report."}
    except Exception as e:
        print(f"  -> ライター実行エラー: {e}")
        return {"error": f"Writer failed: {e}"}

# エラー処理用ノード (シンプルにエラーを出力するだけ)
def handle_error(state: ResearchWorkflowState) -> Dict[str, str]:
    """エラー発生時に呼び出されるノード"""
    print("\n--- ノード: Error Handler 実行 ---")
    error_message = state.get("error", "不明なエラーが発生しました。")
    print(f"  -> エラー発生: {error_message}")
    # 必要に応じて、ここでエラー通知などの処理を追加できる
    # 最終出力としてエラーメッセージを設定する例
    return {"output": f"処理中にエラーが発生しました: {error_message}"}

print("--- ノード関数 (run_researcher, run_writer, handle_error) 定義完了 ---")


# --- エラーチェック用 分岐関数 ---
def check_error(state: ResearchWorkflowState) -> Literal["writer", "error_handler"]:
    """状態にエラーがあるかチェックし、次のノードを決定"""
    print(f"\n--- 分岐関数: エラーチェック実行 (エラー状態: {state.get('error')}) ---")
    if state.get("error"):
        print("  -> エラー検出: handle_error へ")
        return "error_handler"
    else:
        print("  -> エラーなし: writer へ")
        return "writer"

print("--- 分岐関数 (check_error) 定義完了 ---")
# step35_multi_agent_implementation_revised.py (続き)

print("\n--- グラフの構築開始 (エラー処理分岐あり) ---")
workflow = StateGraph(ResearchWorkflowState)

# ノードを追加
workflow.add_node("researcher", run_researcher)
workflow.add_node("writer", run_writer)
workflow.add_node("error_handler", handle_error)
print("ノードを追加しました: researcher, writer, error_handler")

# エッジを設定
workflow.add_edge(START, "researcher") # まずリサーチャーを実行

# researcher の後にエラーチェック分岐を設定
workflow.add_conditional_edges(
    source="researcher",
    path=check_error, # check_error 関数の結果で分岐
    path_map={ # 分岐関数が返す値と実際のノード名を対応付ける
        "writer": "writer",
        "error_handler": "error_handler"
    }
)
print("条件分岐エッジを追加しました: researcher からエラーチェック経由で writer または error_handler へ")

# writer または error_handler が終わったら終了
workflow.add_edge("writer", END)
workflow.add_edge("error_handler", END)
print("終了へのエッジを追加しました。")

# エントリーポイントは START からのエッジで自動設定
print("--- グラフの構築完了 ---")
# step35_multi_agent_implementation_revised.py (続き)

print("\n--- グラフのコンパイル開始 ---")
app = workflow.compile()
print("--- グラフのコンパイル完了 ---")

print("\n--- グラフの実行 ---")
initial_state = {"request": "LangGraph の主要な機能とその利点について日本語でレポートしてください"}
print(f"初期状態 (リクエスト): {initial_state}")

# --- ストリーミング実行の例 ---
print("\n--- ストリーミング実行ログ (各ステップの結果) ---")
final_state = {} # 最終状態を保持する変数
try:
    # config で再帰制限を設定
    for event in app.stream(initial_state, config={'recursion_limit': 10}):
        # イベントの内容を表示 (キーはイベントが発生したノード名)
        # print(f"イベント: {list(event.keys())[0]}")
        # print(event[list(event.keys())[0]]) # イベントデータを表示
        print(".", end="", flush=True) # 進行状況をドットで表示

        # 最後のイベントの状態を final_state に保持する
        # stream() の最後は __end__ キーになることが多い
        if END not in event:
            latest_key = list(event.keys())[-1]
            final_state.update(event[latest_key]) # 状態をマージ

    print("\n--- ストリーミング実行完了 ---")

except Exception as e:
    print(f"\n--- グラフ実行中に予期せぬエラー ---")
    print(f"エラー: {e}")
    final_state = {"error": f"Unexpected error during execution: {e}"} # エラー情報を状態に記録


# --- 最終結果の表示 ---
print("\n\n--- 最終状態の確認 ---")
# print(f"最終状態全体: {final_state}") # デバッグ用に全状態を表示

print(f"\nリクエスト:\n{final_state.get('request')}")
print(f"\nリサーチ結果要約:\n{final_state.get('research_summary', '(リサーチ未実行または失敗)')}")
print(f"\n最終レポート:\n{final_state.get('report', '(レポート未作成または失敗)')}")
if final_state.get('error'):
    print(f"\nエラーメッセージ:\n{final_state.get('error')}")


# --- (オプション) グラフ構造を Mermaid で出力 ---
# print("\n--- グラフ構造 (Mermaid) ---")
# try:
#     mermaid_txt = app.get_graph().draw_mermaid()
#     print(mermaid_txt)
# except Exception as e:
#     print(f"Mermaid図の生成中にエラー: {e}")

print("\n--- 全体の処理終了 ---")

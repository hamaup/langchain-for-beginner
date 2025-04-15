# step36_human_in_the_loop_revised.py
import os
import sys
from dotenv import load_dotenv
from typing import TypedDict, Optional, Dict, Any, List, Literal
import uuid # スレッドID生成用

# LangGraph (StateGraph, START, END) とチェックポインタ
try:
    from langgraph.graph import StateGraph, START, END
    # チェックポインタ (今回はメモリを使用)
    from langgraph.checkpoint.memory import MemorySaver
    print("LangGraph と MemorySaver をインポートしました。")
except ImportError:
    print("エラー: langgraph が見つかりません。'pip install langgraph'")
    sys.exit(1)

# --- ステップ35から流用するインポート ---
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
# --- 流用ここまで ---

print("--- 必要なモジュールのインポート完了 ---")

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
# ...(APIキーチェック、LLM初期化、ツール準備、エージェント作成 はステップ35と同じ)...
# ※※※ ステップ35の該当コードをここにコピーしてください ※※※

# ...(状態定義 ResearchWorkflowState はステップ35と同じ)...
# ※※※ ステップ35の該当コードをここにコピーしてください ※※※

# ...(ノード関数 run_researcher, run_writer, handle_error はステップ35と同じ)...
# ※※※ ステップ35の該当コードをここにコピーしてください ※※※

# ...(分岐関数 check_error はステップ35と同じ)...
# ※※※ ステップ35の該当コードをここにコピーしてください ※※※

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


# step36_human_in_the_loop_revised.py (続き)

# --- グラフの構築 (ステップ35と同じ) ---
print("\n--- グラフの構築開始 ---")
workflow = StateGraph(ResearchWorkflowState)
workflow.add_node("researcher", run_researcher)
workflow.add_node("writer", run_writer)
workflow.add_node("error_handler", handle_error)
print("ノードを追加しました。")
workflow.add_edge(START, "researcher")
workflow.add_conditional_edges("researcher", check_error, {"writer": "writer", "error_handler": "error_handler"})
workflow.add_edge("writer", END)
workflow.add_edge("error_handler", END)
print("エッジを追加しました。")
print("--- グラフの構築完了 ---")

# step36_human_in_the_loop_revised.py (続き)

print("\n--- グラフのコンパイル開始 (中断設定あり) ---")

# チェックポインタの準備 (インメモリ)
# これでグラフの途中状態がメモリに保存される
memory = MemorySaver()

# グラフをコンパイル。researcher ノードの後で中断、チェックポインタを指定
app_interruptible = workflow.compile(
    checkpointer=memory,           # チェックポインタを指定
    interrupt_after=["researcher"] # "researcher" ノードの実行後に中断する
)
print("--- グラフのコンパイル完了 (中断設定 & チェックポインタ付き) ---")

# step36_human_in_the_loop_revised.py (続き)

print("\n--- グラフの実行 (Human-in-the-loop) ---")
# スレッドIDを持つ config を作成 (チェックポイント管理用)
thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": thread_id}}
print(f"実行コンフィグ (スレッドID: {thread_id}): {config}")

initial_state = {"request": "LangGraph の Human-in-the-loop 機能について使い方を説明してください"}
print(f"初期状態 (リクエスト): {initial_state}")

# 最終状態を格納する変数
final_state = None
# 人間からのフィードバックを格納する変数
human_feedback = None

try:
    print("\n--- 1回目の実行 (リサーチャー実行 -> 中断) ---")
    # stream で実行開始。中断ポイントで停止するはず
    current_state = None
    for event in app_interruptible.stream(initial_state, config, stream_mode="values"):
        print(".", end="", flush=True) # 進行表示
        # 最新の状態を保持
        current_state = event

    print("\n--- グラフが中断しました ---")

    if current_state:
        print("\n--- 中断時点の状態 ---")
        print(f"  リクエスト: {current_state.get('request')}")
        research_summary = current_state.get('research_summary')
        print(f"  リサーチ結果要約:\n    {research_summary if research_summary else '(まだありません)'}")
        if current_state.get('error'):
             print(f"  エラー: {current_state.get('error')}")
             # エラーがあればここで処理終了
             final_state = current_state
        else:
            # --- 人間の確認と指示 ---
            print("\n--- 人間の確認ステップ ---")
            print("リサーチ結果を確認してください。")
            action = input("アクションを選択してください [yes: 承認して続行, edit: 修正して続行, no: 中断]: ").strip().lower()

            if action == "yes":
                print("承認されました。グラフの実行を再開します...")
                human_feedback = {"feedback": "Approved"} # 承認したことを記録 (オプション)

            elif action == "edit":
                print("修正内容を入力してください (入力後 Enter 2回で確定):")
                edited_summary = sys.stdin.read() # 複数行入力対応
                human_feedback = {"feedback": "Edited", "edited_summary": edited_summary.strip()}
                print("修正を受け付けました。修正内容でグラフの実行を再開します...")

            else: # "no" またはそれ以外
                print("中断を選択しました。処理を終了します。")
                human_feedback = {"feedback": "Rejected"}
                final_state = current_state # 中断時点の状態を最終とする

            # --- グラフの再開 (承認または修正の場合) ---
            if action == "yes" or action == "edit":
                # 再開時の入力。承認のみなら None、修正があれば更新内容を渡す
                resume_input = None
                if action == "edit" and human_feedback and human_feedback.get("edited_summary"):
                    # 人間が修正した内容で research_summary を更新して再開する
                    resume_input = {"research_summary": human_feedback["edited_summary"]}
                    print(f"  -> 状態を更新して再開: research_summary を上書き")

                print("\n--- グラフ実行再開 ---")
                # 同じ config を使って stream を再開
                # 再開時の入力として resume_input (修正内容 or None) を渡す
                for event in app_interruptible.stream(resume_input, config, stream_mode="values"):
                     print(".", end="", flush=True) # 進行表示
                     # 最新の状態を final_state に保持
                     final_state = event

                print("\n--- グラフ実行再開完了 ---")

    else:
        print("エラー: 中断状態を取得できませんでした。")
        final_state = {"error": "Failed to get interrupted state."}


except Exception as e:
    print(f"\n--- グラフ実行中に予期せぬエラー ---")
    print(f"エラー: {e}")
    # エラーが発生した場合も、可能であればチェックポイントから状態を取得してみる (応用)
    try:
        final_state = app_interruptible.get_state(config)
    except Exception:
        final_state = {"error": f"Unexpected error during execution: {e}"}


# --- 最終結果の表示 ---
print("\n\n--- 最終状態の確認 ---")
# print(f"最終状態全体: {final_state}") # デバッグ用に全状態を表示

print(f"\nリクエスト:\n{final_state.get('request')}")
print(f"\nリサーチ結果要約:\n{final_state.get('research_summary', '(未実行 or 失敗)')}")
print(f"\n最終レポート:\n{final_state.get('report', '(未実行 or 失敗 or 未承認)')}")
if final_state.get('error'):
    print(f"\nエラーメッセージ:\n{final_state.get('error')}")
if human_feedback:
    print(f"\n人間のフィードバック: {human_feedback}")


# --- (オプション) グラフ構造を Mermaid で出力 ---
# ... (ステップ35と同じコード) ...

print("\n--- 全体の処理終了 ---")

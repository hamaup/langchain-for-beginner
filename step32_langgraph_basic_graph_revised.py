# step32_langgraph_basic_graph_revised.py
import os
import sys
from typing import TypedDict, Optional, Dict, Any # 状態定義と型ヒントのため

# LangGraph の StateGraph クラスと特別なノード名をインポート
try:
    from langgraph.graph import StateGraph, START, END
    print("LangGraph をインポートしました。")
    # 必要ならインストール: pip install langgraph
except ImportError:
    print("エラー: langgraph が見つかりません。")
    print("   'pip install langgraph' を実行してください。")
    sys.exit(1)

print("--- 必要なモジュールのインポート完了 ---")
# step32_langgraph_basic_graph_revised.py (続き)

# グラフ全体で管理する状態を TypedDict で定義
class BasicGraphState(TypedDict):
    input: str # グラフへの入力文字列
    output: Optional[str] # 処理結果の文字列 (初期値やエラー時は None になりうる)

print("--- グラフの状態 (BasicGraphState) 定義完了 ---")
# step32_langgraph_basic_graph_revised.py (続き)

# ノード1: 入力文字列に接尾辞を追加する関数
def process_input(state: BasicGraphState) -> Dict[str, Any]:
    """入力文字列に '(processed)' を追加して output に設定する"""
    print(f"--- ノード: process_input 実行 (入力: {state.get('input')}) ---")
    input_text = state['input'] # TypedDict なので state.get('input') も使える
    processed_text = input_text + " (processed)"
    # 更新する状態の部分だけを辞書で返す
    return {"output": processed_text}

# ノード2: 結果をフォーマットする関数
def format_output(state: BasicGraphState) -> Dict[str, Any]:
    """output 文字列を大文字に変換して output を更新する"""
    print(f"--- ノード: format_output 実行 (入力: {state.get('output')}) ---")
    current_output = state.get('output') # None の可能性があるので .get() を使う
    # 入力が None でないかチェック (防御的プログラミング)
    if current_output is None:
        print("警告: format_output に渡された output が None です。")
        # エラーを示す値を返すか、あるいは何も更新しないかを選択できる
        # 今回は更新しない例
        return {}
        # return {"output": "ERROR: No output to format"} # エラー文字列を返す例
    formatted_output = current_output.upper()
    return {"output": formatted_output}

print("--- ノード関数 (process_input, format_output) 定義完了 ---")
# step32_langgraph_basic_graph_revised.py (続き)

print("\n--- グラフの構築開始 ---")
# グラフのワークフローを作成 (状態の型を指定)
workflow = StateGraph(BasicGraphState)

# ノードをグラフに追加
workflow.add_node("processor", process_input)
workflow.add_node("formatter", format_output)
print("ノードを追加しました: processor, formatter")

# エッジをグラフに追加
# START から processor ノードへ (これがエントリーポイントになる)
workflow.add_edge(START, "processor")
workflow.add_edge("processor", "formatter") # processor から formatter ノードへ
workflow.add_edge("formatter", END) # formatter から 終了点へ
print("エッジを追加しました: START -> processor -> formatter -> END")

# set_entry_point() は add_edge(START, ...) を使えば通常は不要
print("エントリーポイントは START からのエッジで設定されました。")
print("--- グラフの構築完了 ---")
# step32_langgraph_basic_graph_revised.py (続き)

print("\n--- グラフのコンパイル開始 ---")
# グラフを実行可能な Runnable にコンパイル
app = workflow.compile()
print("--- グラフのコンパイル完了 ---")
# step32_langgraph_basic_graph_revised.py (続き)
print("\n--- グラフの実行 ---")

initial_state = {"input": "hello langgraph"}
print(f"初期状態 (入力): {initial_state}")

try:
    # .invoke() でグラフを実行
    final_state = app.invoke(initial_state)

    print("\n--- グラフの実行完了 ---")
    print(f"最終状態: {final_state}")
    print(f"最終的な出力 (output): {final_state.get('output')}")

except Exception as e:
    print(f"\n--- グラフ実行中にエラーが発生しました ---")
    print(f"エラー: {e}")
    # ここでより詳細なエラーハンドリングを行うことも可能

# --- 別の入力で試す ---
print("\n--- 別の入力で実行 ---")
initial_state_2 = {"input": "another test"}
print(f"初期状態 (入力): {initial_state_2}")
try:
    final_state_2 = app.invoke(initial_state_2)
    print("\n--- グラフの実行完了 ---")
    print(f"最終状態: {final_state_2}")
    print(f"最終的な出力 (output): {final_state_2.get('output')}")
except Exception as e:
    print(f"\n--- グラフ実行中にエラーが発生しました ---")
    print(f"エラー: {e}")

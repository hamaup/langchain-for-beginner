# step33_langgraph_conditional_edges_final_v2.py
import os
import sys
# 型ヒントに必要なものをインポート
from typing import TypedDict, Optional, Dict, Any, Literal

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

# グラフ全体で管理する状態を定義 (拡張版)
class ConditionalGraphState(TypedDict):
    input: str
    # ★分岐の判断に使う情報を入れる場所を追加★
    # Literal["short", "long"] は、'short' または 'long' という文字列だけが入ることを示す
    input_length_type: Optional[Literal["short", "long"]]
    # 最終的な出力
    output: Optional[str]

print("--- グラフの状態 (ConditionalGraphState) 定義完了 ---")
# step33_langgraph_conditional_edges_final_v2.py (続き)

# --- ノード関数 (処理役たち) の定義 ---

# 1. 入力チェック役
def check_input_length(state: ConditionalGraphState) -> Dict[str, Any]:
    """入力文字列の長さをチェックして、状態の 'input_length_type' を更新"""
    print(f"--- ノード: check_input_length 実行 (入力: '{state.get('input')}') ---")
    input_text = state['input']
    # 例: 10文字より長ければ 'long', それ以外は 'short' とする
    length_type = "long" if len(input_text) > 10 else "short"
    print(f"  -> 入力タイプ判定: {length_type}")
    # 状態の input_length_type を更新するための辞書を返す
    return {"input_length_type": length_type}

# 2a. 短い入力処理役
def process_short_input(state: ConditionalGraphState) -> Dict[str, Any]:
    """短い入力用の処理"""
    print(f"--- ノード: process_short_input 実行 ---")
    input_text = state['input']
    result = f"短い入力 ('{input_text}') を受け付けました。OK!"
    # 状態の output を更新
    return {"output": result}

# 2b. 長い入力処理役
def process_long_input(state: ConditionalGraphState) -> Dict[str, Any]:
    """長い入力用の処理"""
    print(f"--- ノード: process_long_input 実行 ---")
    input_text = state['input']
    result = f"長い入力 ('{input_text}') を処理しました。文字数: {len(input_text)}。"
    # 状態の output を更新
    return {"output": result}

print("--- ノード関数 (3つ) 定義完了 ---")

# --- 分岐関数 (案内人役) の定義 (エラー処理修正版) ---
# 戻り値の型ヒントから '__error__' を削除
def route_based_on_length(state: ConditionalGraphState) -> Literal["short_processor", "long_processor"]:
    """状態の 'input_length_type' を見て、次に進むノード名を返す"""
    print(f"--- 分岐関数 実行 (判定対象: '{state.get('input_length_type')}') ---")
    length_type = state.get('input_length_type')

    if length_type == "short":
        print("  -> 分岐先: short_processor")
        return "short_processor"
    elif length_type == "long":
        print("  -> 分岐先: long_processor")
        return "long_processor"
    else:
        # input_length_type が想定外の値 (None など) だった場合
        print(f"  -> エラー: 分岐条件が不明です。length_type='{length_type}'")
        # エラーを示す文字列を返す代わりに、例外を発生させて問題を知らせる
        raise ValueError(f"Invalid input_length_type for routing: {length_type}")

print("--- 分岐関数 (route_based_on_length) 定義完了 ---")
# step33_langgraph_conditional_edges_final_v2.py (続き)

print("\n--- グラフの構築開始 (条件分岐あり) ---")
# 状態の型を指定して StateGraph を作成
workflow_conditional = StateGraph(ConditionalGraphState)

# 作った関数をノードとしてグラフに追加
workflow_conditional.add_node("checker", check_input_length)
workflow_conditional.add_node("short_processor", process_short_input)
workflow_conditional.add_node("long_processor", process_long_input)
print("ノードを追加しました: checker, short_processor, long_processor")

# 通常のエッジ (まっすぐな道) を設定
workflow_conditional.add_edge(START, "checker")       # スタートからまず checker へ
workflow_conditional.add_edge("short_processor", END) # 短い処理が終わったら終了へ
workflow_conditional.add_edge("long_processor", END)  # 長い処理が終わったら終了へ
print("通常のエッジを追加しました。")

# ★条件分岐エッジ (分かれ道) を設定★
workflow_conditional.add_conditional_edges(
    source="checker",          # checker ノードの後で分岐する
    path=route_based_on_length, # route_based_on_length 関数に行き先を聞く
    # path_map は今回は不要 (関数が直接ノード名を返すため)
)
print("条件分岐エッジを追加しました (add_conditional_edges を使用): checker から分岐")

print("--- グラフの構築完了 ---")
# step33_langgraph_conditional_edges_final_v2.py (続き)

print("\n--- グラフのコンパイル開始 ---")
app_conditional = workflow_conditional.compile()
print("--- グラフのコンパイル完了 ---")

# Mermaid 形式でグラフ構造を出力
mermaid_text = app_conditional.get_graph().draw_mermaid()
print("\n--- グラフの構造 (Mermaid) ---")
print(mermaid_text)

# --- 短い入力で実行してみる ---
print("\n--- 短い入力 ('hello') で実行 ---")
# 初期状態として input を含む辞書を与える
initial_state_short = {"input": "hello"}
print(f"初期状態: {initial_state_short}")
try:
    # invoke で実行。config で念のため再帰制限を設定
    final_state_short = app_conditional.invoke(initial_state_short, config={'recursion_limit': 5})
    print("\n実行完了 (短い入力)")
    print(f"最終状態: {final_state_short}")
    print(f"--> 最終出力: {final_state_short.get('output')}") # 短い入力用のメッセージのはず
except Exception as e:
    print(f"エラーが発生しました: {e}")

# --- 長い入力で実行してみる ---
print("\n--- 長い入力 ('LangGraph is powerful!') で実行 ---")
initial_state_long = {"input": "LangGraph is powerful!"} # 10文字より長い
print(f"初期状態: {initial_state_long}")
try:
    final_state_long = app_conditional.invoke(initial_state_long, config={'recursion_limit': 5})
    print("\n実行完了 (長い入力)")
    print(f"最終状態: {final_state_long}")
    print(f"--> 最終出力: {final_state_long.get('output')}") # 長い入力用のメッセージのはず
except Exception as e:
    print(f"エラーが発生しました: {e}")

print("\n--- 処理終了 ---")
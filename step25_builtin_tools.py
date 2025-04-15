# step25_builtin_tools_final.py
import os
import sys
from dotenv import load_dotenv

# --- 組み込みツールをインポート ---
try:
    # DuckDuckGo は直接ツールクラスをインポートできることが多い
    from langchain_community.tools import DuckDuckGoSearchRun
    print("DuckDuckGoSearchRun をインポートしました。")
    # ※裏側では langchain_community.utilities.duckduckgo_search が使われています
    # ※別途 pip install duckduckgo-search が必要
except ImportError:
    print("エラー: DuckDuckGoSearchRun が見つかりません。")
    print("   'pip install langchain-community duckduckgo-search' を確認してください。")
    sys.exit(1)

try:
    # Wikipedia もツールクラスがあるが、カスタマイズのために Wrapper を使う例も示す
    from langchain_community.tools import WikipediaQueryRun
    from langchain_community.utilities.wikipedia import WikipediaAPIWrapper # Wrapper もインポート
    print("WikipediaQueryRun と WikipediaAPIWrapper をインポートしました。")
    # ※別途 pip install wikipedia が必要
except ImportError as e:
    print(f"エラー: Wikipedia 関連のインポートに失敗しました: {e}")
    print("   'pip install langchain-community wikipedia' を確認してください。")
    sys.exit(1)

# (参考) 前のステップで作ったカスタムツールもインポート
try:
    # ステップ 24 のファイル名とツールが定義された変数を指定
    from step24_custom_tool_pydantic_final import search_weather
    print("前のステップのカスタムツール (search_weather) をインポートしました。")
except ImportError:
    print("警告: 前のステップのカスタムツール (search_weather) が見つかりませんでした。")
    search_weather = None # スクリプトが止まらないように None を設定

print("--- 必要なモジュールのインポート完了 ---")

load_dotenv()
print("環境変数をロードしました (必要に応じて)")
# step25_builtin_tools_final.py (続き)

print("\n--- 組み込みツールのインスタンス化 ---")

try:
    # DuckDuckGo 検索ツールを作成 (APIキー不要！)
    duckduckgo_search = DuckDuckGoSearchRun()
    print("DuckDuckGoSearchRun インスタンスを作成しました。")

    # Wikipedia 検索ツールを作成 (シンプルな初期化)
    # wikipedia_search_default = WikipediaQueryRun() # 最もシンプルな方法
    # print("WikipediaQueryRun (デフォルト) インスタンスを作成しました。")

    # Wikipedia 検索ツール (日本語優先カスタマイズ版！)
    print("WikipediaQueryRun (日本語優先) インスタンスを作成中...")
    # Wrapper で言語などを設定
    wiki_api_wrapper_ja = WikipediaAPIWrapper(lang="ja", top_k_results=1, load_all_available_meta=False)
    # カスタマイズした Wrapper を使ってツールを作成
    wikipedia_search_ja = WikipediaQueryRun(api_wrapper=wiki_api_wrapper_ja)
    print("WikipediaQueryRun (日本語優先) インスタンスを作成しました。")


except Exception as e:
    print(f"エラー: ツールのインスタンス化中にエラー: {e}")
    sys.exit(1)
# step25_builtin_tools_final.py (続き)

print("\n--- 組み込みツールの情報を確認 ---")
import json # args_schema 表示用

# DuckDuckGo Search
print("\n[DuckDuckGoSearchRun]")
print(f"名前: {duckduckgo_search.name}")
print(f"説明: {duckduckgo_search.description}") # ← エージェントがこれを読みます！
print("引数スキーマ:")
# args_schema は Pydantic モデルの場合がある
if hasattr(duckduckgo_search.args_schema, 'model_json_schema'):
    print(json.dumps(duckduckgo_search.args_schema.model_json_schema(), indent=2, ensure_ascii=False))
else:
    print(duckduckgo_search.args) # 古い形式やシンプルな辞書の場合

# Wikipedia Search (日本語版)
print("\n[WikipediaQueryRun (日本語版)]")
print(f"名前: {wikipedia_search_ja.name}")
print(f"説明: {wikipedia_search_ja.description}") # ← この説明も重要！
print("引数スキーマ:")
if hasattr(wikipedia_search_ja.args_schema, 'model_json_schema'):
     print(json.dumps(wikipedia_search_ja.args_schema.model_json_schema(), indent=2, ensure_ascii=False))
else:
    print(wikipedia_search_ja.args)
# step25_builtin_tools_final.py (続き)

print("\n--- 組み込みツールを直接実行してみる ---")

try:
    print("\nDuckDuckGo で 'LangChain 最新バージョン' を検索:")
    # invoke は通常、引数を辞書で渡す
    ddg_result = duckduckgo_search.invoke({"query": "LangChain 最新バージョン"})
    # .run() メソッドがあれば、文字列だけで実行できることが多い
    # ddg_result_run = duckduckgo_search.run("LangChain 最新バージョン")
    print(f"結果 (抜粋): {ddg_result[:150]}...") # 長いので一部だけ表示

    print("\nWikipedia (日本語) で '大規模言語モデル' を検索:")
    # WikipediaQueryRun も invoke を使う
    wiki_result = wikipedia_search_ja.invoke({"query": "大規模言語モデル"})
    # wiki_result_run = wikipedia_search_ja.run("大規模言語モデル") # .run() も使える
    print(f"結果 (抜粋): {wiki_result[:150]}...")

except Exception as e:
    print(f"エラー: ツールの実行中にエラーが発生しました: {e}")
    # 外部 API に依存するため、ネットワークエラーなどが考えられます
# step25_builtin_tools_final.py (続き)

print("\n--- エージェント用のツールリストを作成 ---")

# 組み込みツールをリストに入れる
tools = [duckduckgo_search, wikipedia_search_ja]

# 前のステップのカスタムツールがインポートできていれば追加
if search_weather:
    tools.append(search_weather)
    print("カスタムツール (search_weather) もリストに追加しました。")
else:
    print("カスタムツール (search_weather) は見つからなかったので追加しませんでした。")


print(f"\n完成したツールリスト (計 {len(tools)} 個):")
for i, tool_obj in enumerate(tools):
    print(f"  {i+1}. {tool_obj.name}: {tool_obj.description[:50]}...") # 名前と説明の一部を表示

print("\nこの 'tools' リストを、次のステップ以降でエージェントに渡して使わせます！")
print("--- 処理終了 ---")
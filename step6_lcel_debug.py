# step6_lcel_debugging_intro.py (実践1部分)
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
# 構造化出力に必要な部品
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
# LCELのコア部品 (PassthroughとLambda)
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
# プロンプトやメッセージの型情報 (型ヒントや中身確認用)
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.messages import AIMessage
# 文字列パーサー (後で使う)
from langchain_core.output_parsers import StrOutputParser
# デバッグ用コールバックハンドラー
from langchain.callbacks.tracers import ConsoleCallbackHandler

# --- 初期設定 ---
# .envファイルからAPIキーなどを読み込む
load_dotenv()
print("--- 環境変数読み込み完了 ---")

# LLM (ChatGPT) を準備する
try:
    # temperature=0 で、毎回同じような応答を期待する
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    print(f"--- LLM準備完了: {llm.model_name} (temperature={llm.temperature}) ---")
    print("   (temperature=0 は応答の多様性を抑え、結果を安定させるための設定です)")
except Exception as e:
    print(f"❌ エラー: ChatOpenAI の初期化に失敗しました: {e}")
    print("   (APIキーが正しく設定されているか確認してください)")
    exit() # 続行不可

# --- 実践 1: StructuredOutputParser を使ったチェーン ---
print("\n--- 実践 1: StructuredOutputParser を使ったチェーン ---")

# サンプルの議事録
meeting_minutes_text = """
# 会議議事録 (2025/04/07)
参加者: 山田, 佐藤, 田中
決定事項:
- 新機能AのUI調整は田中が担当 (〆切: 今週末)
- プロモーション計画案は佐藤が修正 (〆切: 水曜)
次回アクション:
- 鈴木がテスト準備 (来週月曜開始)
"""

# 1. どんな情報を、どんな名前で抽出したいか定義 (ResponseSchema)
print("   1. 出力形式の設計図(Schema)を定義中...")
purpose_schema = ResponseSchema(name="purpose", description="会議の主な目的や議題 (string)")
decisions_schema = ResponseSchema(name="decisions", description="決定事項のリスト (list[str])")
actions_schema = ResponseSchema(name="next_actions", description="次回アクションのリスト (list[str])")
response_schemas = [purpose_schema, decisions_schema, actions_schema]

# 2. 設計図をもとに、AI応答を辞書に変換するパーサーを作成
print("   2. 構造化パーサー(StructuredOutputParser)を作成中...")
output_parser_structured = StructuredOutputParser.from_response_schemas(response_schemas)
# パーサーから、LLMに渡すための「出力形式の指示」を取り出す
format_instructions = output_parser_structured.get_format_instructions()
print("   StructuredOutputParser準備完了")


# 3. プロンプトテンプレートを作成 (入力: minutes, format_instructions)
print("   3. プロンプトテンプレートを作成中...")
prompt_template_structured = ChatPromptTemplate.from_template(
    """以下の会議議事録を分析し、指定された形式で情報を抽出してください。

議事録:
{minutes}

出力形式の指示:
{format_instructions}"""
)
print("   プロンプトテンプレート準備完了")

# 4. LCEL チェーンを構築！ (プロンプト -> LLM -> 構造化パーサー)
print("   4. LCEL チェーンを構築中...")
chain_structured = prompt_template_structured | llm | output_parser_structured
print("   チェーン構築完了 (プロンプト | LLM | StructuredOutputParser)")

# 5. チェーンを実行して結果を確認
try:
    # プロンプトに必要な入力データを辞書で渡す
    input_data_structured = {
        "minutes": meeting_minutes_text,
        "format_instructions": format_instructions
    }
    print(f"\n   実行中... 入力キー: {input_data_structured.keys()}")

    # チェーン実行！ output_parser_structured が最後なので辞書が返るはず
    output_dict = chain_structured.invoke(input_data_structured)
    print("✅ OK: 応答を受信しました。")

    # 結果の確認
    print("\n   【応答の型】:", type(output_dict)) # -> <class 'dict'>
    print("   【応答 (辞書)】:")
    print("   ", output_dict)
    # 辞書の中身も確認
    print("\n   抽出された決定事項:", output_dict.get('decisions')) # -> list

except Exception as e:
    print(f"❌ エラー: チェーン (Structured) の実行中にエラーが発生しました: {e}")
    print("   ヒント: LLMが指示通りの形式で出力しなかった場合、ここでパースエラーが起こることがあります。")
# step6_lcel_debugging_intro.py (実践2部分)
print("\n--- 実践 2: RunnablePassthrough と辞書で入力と出力を組み合わせる ---")

# 質問応答用のプロンプトと、シンプルな文字列パーサー
prompt_qa = ChatPromptTemplate.from_template("{question} について簡潔に教えてください。")
parser_str = StrOutputParser()

# チェーンを定義する
print("   チェーン構築中...")
chain_combined = (
    # 1. まず RunnablePassthrough を置き、入力辞書 {"question": ...} をそのまま次に渡す
    RunnablePassthrough()
    # 2. 次のステップは辞書。前のステップからの入力が、この辞書内の各キーの処理に渡される
    | {
        # 3. 'question' キーの処理:
        #    渡された入力辞書から 'question' の値を取り出す関数(lambda)を指定
        #    -> これで元の質問が保持される
        "question": lambda input_dict: input_dict["question"],

        # 4. 'answer' キーの処理:
        #    渡された入力辞書をそのまま回答生成チェーンに渡す
        #    (prompt_qaが入力辞書から'question'を使い、LLMが回答し、parser_strが文字列にする)
        "answer": prompt_qa | llm | parser_str
      }
)
# 結果として、{"question": 元の質問, "answer": AIの回答} という辞書が出力される
print("   チェーン構築完了 (RunnablePassthrough | 辞書を使用)")

try:
    # 最初の入力データ（質問）
    input_qa = {"question": "大規模言語モデル"}
    print(f"\n   実行中... 入力: {input_qa}")

    # チェーンを実行
    result_dict = chain_combined.invoke(input_qa)
    print("✅ OK: 応答を受信しました。")

    # 結果を確認
    print("\n   【応答の型】:", type(result_dict)) # -> <class 'dict'>
    print("   【応答 (辞書)】:")
    print("   ", result_dict)
    # 辞書の中身を確認
    print("\n   元の質問:", result_dict.get("question"))
    print("   AIの回答:", result_dict.get("answer"))

except Exception as e:
    print(f"❌ エラー: チェーン (Combined) の実行中にエラーが発生しました: {e}")
    # step6_lcel_debugging_intro.py (実践3部分)
print("\n--- 実践 3: ConsoleCallbackHandler で実行過程を追跡する ---")
print("   ▼ chain_combined を ConsoleCallbackHandler 付きで実行:")

try:
    # 別の質問で試してみる
    input_qa_callback = {"question": "Reactフレームワーク"}
    print(f"\n   実行中... 入力: {input_qa_callback}")

    # 1. ConsoleCallbackHandler のインスタンス（実物）を用意する
    console_callback = ConsoleCallbackHandler()

    # 2. invoke() メソッドの config 引数に、'callbacks' キーで指定する
    #    値はリスト形式で、中に用意したインスタンスを入れる
    #    (複数の Callback Handler を同時に使うことも可能)
    result_callback = chain_combined.invoke(
        input_qa_callback,
        config={'callbacks': [console_callback]} # Callback 有効化！
    )

    # --- ここに注目！ ---
    # 上記 invoke() の実行中に、コンソールに [chain/start], [llm/start]
    # などの色付きのログが自動で出力されるはずです。
    # 処理の開始・終了や入出力データがわかります。
    # --------------------

    print("\n✅ OK: Callback Handler 付きでの実行が完了しました。(ログは上記に出力されているはずです)")
    # 最終的な結果自体は result_callback に入っている (今回はログ確認が目的なので表示は省略)
    # print("\n   最終結果:", result_callback)

except Exception as e:
    print(f"❌ エラー: チェーン (Callback) の実行中にエラーが発生しました: {e}")
    # step6_lcel_debugging_intro.py (実践4部分)
print("\n--- 実践 4: RunnableLambda で特定の中間データを出力する ---")

# デバッグ用の関数を定義:
#   - 引数名 (ここでは prompt_value) は、何を受け取るか分かりやすい名前にする
#   - 中身を表示 (print) する
#   - ★★★ 必ず受け取った引数をそのまま return する ★★★
def print_prompt_info(prompt_value: ChatPromptValue):
    """プロンプト生成後の ChatPromptValue を受け取り、コンソールに表示してからそのまま返す関数"""
    print("\n---- [RunnableLambda が受け取ったプロンプト情報 START] ----")
    print(f"型: {type(prompt_value)}") # 型を表示
    print("内容:")
    # 中のメッセージリストを分かりやすく表示
    for message in prompt_value.to_messages():
        print(f"- {type(message).__name__}: {message.content}")
    print("---- [RunnableLambda が受け取ったプロンプト情報 END] ----")

    # 重要！ 次のステップにデータを渡すために、必ず return する
    return prompt_value

# チェーンの定義: プロンプトとLLMの間に RunnableLambda を挿入！
print("   チェーン構築中...")
chain_with_lambda = (
    prompt_qa                          # 1. 入力からプロンプト情報を生成
    | RunnableLambda(print_prompt_info) # 2. ★生成されたプロンプト情報をここで表示★
    | llm                              # 3. プロンプト情報をLLMに渡す
    | parser_str                       # 4. LLMの応答(AIMessage)を文字列に
)
print("   チェーン構築完了 (RunnableLambda を使用)")

try:
    # 別の質問で試す
    input_qa_lambda = {"question": "Pythonのジェネレータ"}
    print(f"\n   実行中... 入力: {input_qa_lambda}")

    # チェーンを実行！ 途中で print_prompt_info が呼び出されるはず
    result_lambda = chain_with_lambda.invoke(input_qa_lambda)
    print("\n✅ OK: Lambda 付きチェーンの実行が完了しました。")
    print("   最終結果:", result_lambda)

except Exception as e:
    print(f"❌ エラー: チェーン (Lambda) の実行中にエラーが発生しました: {e}")

print("\n--- 全ての処理が終了しました ---")

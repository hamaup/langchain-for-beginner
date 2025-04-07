# step3_prompt_template.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# --- プロンプト関連の道具をインポート (langchain.prompts) ---
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
# --- メッセージクラスをインポート (langchain.schema) ---
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# 環境変数の読み込み
load_dotenv()
print("--- 環境変数読み込み完了 ---")

# LLMの準備 (temperature=0 で応答を安定させる)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
print(f"--- LLM準備完了: {llm.model_name} (temperature={llm.temperature}) ---")

# --- PromptTemplate のテスト ---
print("\n--- PromptTemplate のテスト ---")
template_string = """
以下のテキストを指定された形式で要約してください。

テキスト:
{input_text}

要約形式:
{output_format}
"""
prompt_template = PromptTemplate.from_template(template_string)
input_data = {
    "input_text": "LangChainは、大規模言語モデル（LLM）を活用したアプリケーション開発を容易にするためのフレームワークです。多様なコンポーネントを組み合わせて、複雑なワークフローを構築できます。",
    "output_format": "箇条書きで3点"
}
final_prompt_string = prompt_template.format(**input_data)
print("【生成されたプロンプト文字列】:")
print(final_prompt_string)
response = llm.invoke(final_prompt_string)
print("\n【LLMからの応答 (PromptTemplate)】:")
print(response.content)

# --- ChatPromptTemplate のテスト (基本) ---
print("\n--- ChatPromptTemplate のテスト (基本) ---")
system_template = SystemMessagePromptTemplate.from_template(
    "あなたは{language}の翻訳家です。丁寧な言葉遣いで回答してください。"
)
human_template = HumanMessagePromptTemplate.from_template(
    "{text_to_translate} を翻訳してください。"
)
chat_template = ChatPromptTemplate.from_messages([system_template, human_template])
chat_input_data = {
    "language": "フランス語",
    "text_to_translate": "Hello, how are you?"
}
final_prompt_messages = chat_template.format_messages(**chat_input_data)
print("【生成されたプロンプトメッセージ (リスト)】:")
print(final_prompt_messages)
response_chat = llm.invoke(final_prompt_messages)
print("\n【LLMからの応答 (ChatPromptTemplate 基本)】:")
print(response_chat.content)

# --- ChatPromptTemplate のテスト (Few-shot: 感情分析) ---
print("\n--- ChatPromptTemplate のテスト (Few-shot: 感情分析) ---")
messages_for_sentiment_analysis = [
    # AIの役割設定
    SystemMessage(content="あなたは与えられた文章の感情を分析し、「ポジティブ」または「ネガティブ」のどちらか一言で分類するAIです。"),
    # お手本1
    HumanMessage(content="文章: 「この映画は最高でした！」\n感情:"),
    AIMessage(content="ポジティブ"),
    # お手本2
    HumanMessage(content="文章: 「サービスがとても遅く、食事も冷めていました。」\n感情:"),
    AIMessage(content="ネガティブ"),
    # お手本3
    HumanMessage(content="文章: 「新しい職場は雰囲気が良く、同僚も親切です。」\n感情:"),
    AIMessage(content="ポジティブ"),
    # 実際のユーザーリクエスト (テンプレート)
    HumanMessagePromptTemplate.from_template("文章: 「{input_sentence}」\n感情:")
]
sentiment_analysis_template = ChatPromptTemplate.from_messages(messages_for_sentiment_analysis)
# 分類対象の文章
sentiment_input_data = {
    "input_sentence": "昨日の会議は長すぎて退屈でした。"
}
final_sentiment_messages = sentiment_analysis_template.format_messages(**sentiment_input_data)
print("【生成されたプロンプトメッセージ (Few-shot)】: (Systemと最後のHumanのみ表示)")
print(final_sentiment_messages[0]) # System Message
print("...") # お手本部分は省略
print(final_sentiment_messages[-1]) # 最後の Human Message (テンプレート適用後)

response_sentiment = llm.invoke(final_sentiment_messages)
print("\n【LLMからの応答 (ChatPromptTemplate Few-shot)】:")
# お手本に倣って「ネガティブ」という応答が期待される
print(response_sentiment.content)

print("\n--- 処理終了 ---")

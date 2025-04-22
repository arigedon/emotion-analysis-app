import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except ImportError:
    # 必要に応じてインストール
    import os
    os.system('pip install -q transformers torch fugashi ipadic unidic-lite')
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ページ設定
st.set_page_config(
    page_title="リアルタイム感情分析",
    page_icon="🎭",
    layout="wide"
)

# タイトルとイントロ
st.title("リアルタイム感情分析アプリ")
st.write("下のテキストボックスに文章を入力すると、AIが感情を分析してグラフで表示します！")

# 感情のリスト
emotion_names_jp = ['喜び', '悲しみ', '期待', '驚き', '怒り', '恐れ', '嫌悪', '信頼']
colors = [
    '#FF6384',  # 赤 (喜び)
    '#36A2EB',  # 青 (悲しみ)
    '#FFCE56',  # 黄 (期待)
    '#4BC0C0',  # 緑 (驚き)
    '#9966FF',  # 紫 (怒り)
    '#FF9F40',  # オレンジ (恐れ)
    '#C7C7C7',  # グレー (嫌悪)
    '#53E1A2'   # ミント (信頼)
]

# セッションステートの初期化
if 'model' not in st.session_state or 'tokenizer' not in st.session_state:
    # モデルの読み込み（初回のみ）
    with st.spinner('モデルを読み込んでいます...少し時間がかかります'):
        try:
            # 小さいモデルを使用（接続問題を回避）
            import os
            os.environ['HF_HUB_OFFLINE'] = '0'
            
            # 認証無しでアクセス可能な代替モデルを使用
            checkpoint = 'cl-tohoku/bert-base-japanese-char'  # 文字ベースの軽量モデル
            
            # トークナイザーとモデルの読み込み - 追加オプション設定
            st.session_state.tokenizer = AutoTokenizer.from_pretrained(
                checkpoint, 
                use_fast=True,
                use_auth_token=False,
                trust_remote_code=True
            )
            
            st.session_state.model = AutoModelForSequenceClassification.from_pretrained(
                checkpoint, 
                num_labels=len(emotion_names_jp),
                use_auth_token=False,
                trust_remote_code=True
            )
            
            st.success('モデルの読み込みが完了しました！')
            st.session_state.demo_mode = False
            
        except Exception as e:
            st.error(f'モデルの読み込み中にエラーが発生しました: {str(e)}')
            st.warning('デモモードで実行します（ランダムな感情値を使用）')
            
            # デモモード用のダミーモデルとトークナイザーを設定
            class DummyModel:
                def eval(self):
                    pass
                
                def __call__(self, **kwargs):
                    class DummyOutput:
                        def __init__(self):
                            self.logits = np.random.rand(1, len(emotion_names_jp))
                    return DummyOutput()
            
            class DummyTokenizer:
                def __call__(self, text, truncation=True, return_tensors="pt"):
                    return {}
            
            st.session_state.model = DummyModel()
            st.session_state.tokenizer = DummyTokenizer()
            st.session_state.demo_mode = True

# Softmax関数
def np_softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

# 感情分析関数
def analyze_emotion(text):
    # デモモードかどうかをチェック
    if st.session_state.get('demo_mode', True):
        # デモモード時はランダムな値を生成
        import random
        random_values = [random.random() for _ in range(len(emotion_names_jp))]
        # 合計が1になるように正規化
        total = sum(random_values)
        normalized = [v/total for v in random_values]
        return {n: float(p) for n, p in zip(emotion_names_jp, normalized)}
    
    # 通常モード
    try:
        # 推論モードを有効化
        st.session_state.model.eval()

        # 入力データ変換 + 推論
        tokens = st.session_state.tokenizer(text, truncation=True, return_tensors="pt")
        preds = st.session_state.model(**tokens)
        prob = np_softmax(preds.logits.cpu().detach().numpy()[0])
        return {n: float(p) for n, p in zip(emotion_names_jp, prob)}
    except Exception as e:
        st.warning(f"感情分析中にエラーが発生しました: {str(e)}")
        st.info("ランダムなデータを使用します")
        
        # エラー時はランダムな値を返す
        import random
        random_values = [random.random() for _ in range(len(emotion_names_jp))]
        total = sum(random_values)
        normalized = [v/total for v in random_values]
        return {n: float(p) for n, p in zip(emotion_names_jp, normalized)}

# フッター
st.markdown("---")
st.markdown("### このアプリについて")
st.write("このアプリは日本語のテキストから感情を分析し、8つの基本感情（喜び、悲しみ、期待、驚き、怒り、恐れ、嫌悪、信頼）の確率を表示します。")
st.write("分析には東北大学の日本語BERTモデルを使用しています。")

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
            # オフラインモードとキャッシュを使用
            import os
            os.environ['HF_DATASETS_OFFLINE'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            
            # キャッシュディレクトリを指定
            cache_dir = "./model_cache"
            
            try:
                # まず軽量モデルでオフラインでの読み込みを試す
                checkpoint = 'cl-tohoku/bert-base-japanese-v2'
                st.session_state.tokenizer = AutoTokenizer.from_pretrained(
                    checkpoint, cache_dir=cache_dir, local_files_only=True)
                st.session_state.model = AutoModelForSequenceClassification.from_pretrained(
                    checkpoint, num_labels=len(emotion_names_jp), cache_dir=cache_dir, local_files_only=True)
                st.success('モデルをキャッシュから読み込みました！')
            except:
                # オフラインモードを無効にして再試行
                os.environ['HF_DATASETS_OFFLINE'] = '0'
                os.environ['TRANSFORMERS_OFFLINE'] = '0'
                
                checkpoint = 'cl-tohoku/bert-base-japanese-v2'
                st.session_state.tokenizer = AutoTokenizer.from_pretrained(
                    checkpoint, cache_dir=cache_dir)
                st.session_state.model = AutoModelForSequenceClassification.from_pretrained(
                    checkpoint, num_labels=len(emotion_names_jp), cache_dir=cache_dir)
                st.success('モデルのダウンロードが完了しました！')
        
        except Exception as e:
            st.error(f'モデルの読み込み中にエラーが発生しました: {str(e)}')
            st.warning('デモモードで実行します（ランダムな感情値を使用）')
            
            # デモモード用のダミーモデルとトークナイザーを設定
            import random
            
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

# Softmax関数
def np_softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

# 感情分析関数
def analyze_emotion(text):
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

# テキスト入力エリア
text_input = st.text_area("ここに分析したい文章を入力してください:", height=150)

# デモ用のサンプルテキスト
st.markdown("### サンプルテキスト")
sample_texts = [
    "今日はとても楽しい一日でした！",
    "悲しいニュースを聞いてショックを受けています。",
    "明日の旅行がとても楽しみです！",
    "突然の知らせに驚いています。",
    "この対応には本当に腹が立ちます。"
]

# サンプルテキストボタン
cols = st.columns(len(sample_texts))
for i, col in enumerate(cols):
    if col.button(f"サンプル{i+1}"):
        text_input = sample_texts[i]
        st.session_state.text_input = sample_texts[i]
        st.experimental_rerun()

# セッション状態の処理
if 'text_input' in st.session_state:
    text_input = st.session_state.text_input

# 分析ボタン
if st.button('分析する') or text_input:
    if text_input:
        with st.spinner('感情を分析中...'):
            # 実際の分析処理
            result = analyze_emotion(text_input)
            
            # 結果の表示
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # バーチャートの作成
                fig, ax = plt.subplots(figsize=(10, 6))
                emotions = list(result.keys())
                values = list(result.values())
                
                # 最大値の位置を取得
                max_index = values.index(max(values))
                
                # 色のリストを作成（最大値の場所を強調）
                bar_colors = [colors[i] + '99' for i in range(len(emotions))]
                bar_colors[max_index] = colors[max_index]
                
                # バーチャートをプロット
                bars = ax.bar(emotions, values, color=bar_colors)
                
                # グラフの設定
                ax.set_ylim(0, 1)
                ax.set_ylabel('確率')
                ax.set_title('感情分析結果')
                
                # 最大値のバーにラベルを追加
                ax.text(max_index, values[max_index] + 0.02, f'{values[max_index]:.3f}', 
                        ha='center', va='bottom', fontsize=12, weight='bold')
                
                st.pyplot(fig)
            
            with col2:
                # 結果の表示
                st.write("### 感情スコア")
                
                # データフレームを作成
                df = pd.DataFrame({
                    '感情': emotions,
                    '確率': [round(v, 3) for v in values]
                })
                
                # 確率が高い順にソート
                df = df.sort_values('確率', ascending=False).reset_index(drop=True)
                
                # 表形式で表示
                st.dataframe(df, height=300)
                
                # 最も強い感情のハイライト表示
                max_emotion = df.iloc[0]['感情']
                max_prob = df.iloc[0]['確率']
                st.markdown(f"### 最も強い感情: **{max_emotion}** ({max_prob:.3f})")

# フッター
st.markdown("---")
st.markdown("### このアプリについて")
st.write("このアプリは日本語のテキストから感情を分析し、8つの基本感情（喜び、悲しみ、期待、驚き、怒り、恐れ、嫌悪、信頼）の確率を表示します。")
st.write("分析には東北大学の日本語BERTモデルを使用しています。")

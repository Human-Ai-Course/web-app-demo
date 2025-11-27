%%writefile app.py
from flask import Flask, render_template, request, jsonify
import boto3
import cv2
import numpy as np
import base64
import io
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__)

# --- ユーザー設定 ---
REGION_NAME = "ap-northeast-1"
# --------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        img_data_b64 = data['image'].split(',')[1]
        
        img_bytes = base64.b64decode(img_data_b64)
        image = Image.open(io.BytesIO(img_bytes))
        
        # PIL形式(RGB)のまま処理開始
        img_w, img_h = image.size

        # AWSクライアント作成
        rekognition = boto3.client('rekognition', region_name=REGION_NAME)
        polly = boto3.client('polly', region_name=REGION_NAME)
        # 【追加】翻訳クライアント
        translate = boto3.client('translate', region_name=REGION_NAME)

        # 認識実行
        response = rekognition.detect_labels(
            Image={'Bytes': img_bytes}, MaxLabels=20, MinConfidence=50
        )
        labels = response['Labels']
        
        # --- 診断レポート用 ---
        debug_lines = ["📸 解析完了", "--- トップ5 (日本語変換) ---"]

        # 描画の準備（日本語対応のためPillowを使用）
        draw = ImageDraw.Draw(image)
        try:
            # ダウンロードしたフォントを読み込む (サイズ30)
            font = ImageFont.truetype("static/font.ttf", 30)
        except:
            font = ImageFont.load_default() # 失敗したらデフォルト（日本語出ない）

        speech_text = ""
        found_main_object = False

        # 上位5つをループ
        for i, label in enumerate(labels[:5]):
            en_name = label['Name']
            conf = label['Confidence']
            
            # 【追加】英語 -> 日本語へ翻訳
            trans_res = translate.translate_text(
                Text=en_name, SourceLanguageCode='en', TargetLanguageCode='ja'
            )
            ja_name = trans_res['TranslatedText']

            instances = label.get('Instances', [])
            status = "枠なし"

            if len(instances) > 0:
                status = "✅ 枠あり"
                
                # 1位の枠付き物体をメインの音声にする
                if not found_main_object:
                    speech_text = f"{ja_name}を見つけました"
                    found_main_object = True

                for instance in instances:
                    box = instance['BoundingBox']
                    x1 = box['Left'] * img_w
                    y1 = box['Top'] * img_h
                    x2 = (box['Left'] + box['Width']) * img_w
                    y2 = (box['Top'] + box['Height']) * img_h
                    
                    # 枠を描く (緑色)
                    draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=5)
                    
                    # 日本語テキストを描くための背景（文字が読みやすいように黒背景）
                    text_w = draw.textlength(ja_name, font=font)
                    text_bg = [x1, y1 - 35, x1 + text_w + 10, y1]
                    draw.rectangle(text_bg, fill=(0, 255, 0))
                    
                    # 日本語テキストを描画 (白文字)
                    draw.text((x1 + 5, y1 - 35), ja_name, font=font, fill=(255, 255, 255))
            
            debug_lines.append(f"{i+1}. {en_name} -> 「{ja_name}」 ({status})")

        result_text = "\n".join(debug_lines)

        # 音声テキストがない場合（枠なし）
        if not speech_text:
            if labels:
                # 1位のラベルを翻訳して喋る
                top_en = labels[0]['Name']
                top_trans = translate.translate_text(Text=top_en, SourceLanguageCode='en', TargetLanguageCode='ja')
                top_ja = top_trans['TranslatedText']
                speech_text = f"たぶん、{top_ja}だと思います"
            else:
                speech_text = "何もわかりませんでした"

        # 音声合成
        polly_res = polly.synthesize_speech(
            Text=speech_text, OutputFormat='mp3', VoiceId='Kazuha', Engine='neural'
        )
        audio_stream = polly_res['AudioStream'].read()
        audio_b64 = base64.b64encode(audio_stream).decode()

        # 画像をBase64に変換 (PIL -> Bytes)
        buf = io.BytesIO()
        image.save(buf, format='JPEG')
        processed_img_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

        return jsonify({
            'image': processed_img_b64,
            'text': result_text,
            'audio': audio_b64
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000)

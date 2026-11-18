# 顔モザイクツール

写真・動画に写っている人の顔を自動で見つけてモザイクをかけます。処理はすべてこのMacの中で完結し、画像を外部に送信することはありません。

## セットアップ（初回のみ）

```
git clone https://github.com/KZ6435868/face-mosaic.git ~/face-mosaic
cd ~/face-mosaic
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## 使い方

ターミナルで `~/face-mosaic/mosaic ` と入力してから、写真・動画・フォルダをターミナルにドラッグ＆ドロップして Enter。

```
~/face-mosaic/mosaic 写真.jpg
~/face-mosaic/mosaic 動画.mov
~/face-mosaic/mosaic ~/Desktop/投稿用/      # フォルダ内をまとめて処理
```

元ファイルと同じ場所に `元の名前_mosaic.jpg` / `元の名前_mosaic.mp4` が保存されます（元ファイルは変更しません）。

## オプション

| オプション | 内容 |
|---|---|
| `--style blur` | モザイクの代わりにぼかし（`fill` で黒塗り） |
| `--strength 5` | モザイクを粗くする（既定 8。小さいほど粗い） |
| `--margin 0.4` | 隠す範囲を広げる（既定 0.25） |
| `--threshold 0.5` | 見逃しを減らす（既定 0.6。下げると顔以外にもかかりやすくなる） |
| `--hold 0.5` | 動画で検出が途切れても前後何秒モザイクを残すか（既定 0.3） |
| `--out ~/Desktop/出力` | 保存先フォルダを指定 |

## 対応形式

- 写真: jpg / png / webp / heic（iPhone）。heic は jpg で保存
- 動画: mp4 / mov / m4v / avi / mkv / webm。H.264 の mp4 で保存（音声はそのまま残る）

## 注意

- 自動検出なので、横顔・後ろ向き・とても小さい顔・手やマスクで大きく隠れた顔は見逃すことがあります。投稿前に必ず目で確認してください。
- 写真の保存時に位置情報などのメタデータ（EXIF）は消えます。

## ライセンス表記

顔検出モデル `models/face_detection_yunet_2023mar.onnx` は [OpenCV Zoo](https://github.com/opencv/opencv_zoo) の YuNet（MIT License）です。

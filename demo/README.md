# GaussianAvatars × VHAP End-to-End Demo

このディレクトリは **VHAP (cuda128 fork)** を使った前処理から
**GaussianAvatars** の学習・レンダリングまでを 3 ステップで実行するための
デモスクリプトを提供します。

VHAP は `submodules/VHAP` に git submodule として組み込まれており、
[MTamon/VHAP @ cuda128](https://github.com/MTamon/VHAP/tree/cuda128) を参照します。

```
demo/
├── _common.sh                          # shared helpers (env activation, paths, unbuffered py)
├── setup_env.sh                        # 統合 conda env を構築
├── 01_vhap_preprocess_monocular.sh     # 単眼動画 → NeRF 形式
├── 01_vhap_preprocess_nersemble.sh     # NeRSemble マルチビュー → NeRF 形式
├── 02_train.sh                         # GaussianAvatars 学習
└── 03_render.sh                        # 学習済みアバターのレンダリング
```

`PYTHONUNBUFFERED=1` + `python -u` をデモ側で指定しているので、
**前処理・学習・レンダリングそれぞれの段階で tqdm のプログレスバーが
リアルタイムで表示されます**（FPS / iter / ETA が見えます）。

| 段階 | プログレスバー実装 |
| --- | --- |
| 前処理 (matting / extraction) | `submodules/VHAP/vhap/preprocess_video.py:66,120,141` |
| FLAME tracking | `submodules/VHAP/vhap/track*.py` 内の global epoch ループ |
| NeRF export | `submodules/VHAP/vhap/export_as_nerf_dataset.py:68,241,380` |
| 学習 | `train.py:60` `Training progress` bar |
| レンダリング | `render.py:70` `Rendering progress` bar |

---

## 0. 前提条件

### 0.1 リポジトリと submodule の取得

```shell
git clone <this repo>
cd GaussianAvatars
git submodule update --init --recursive
```

### 0.2 統合 conda 環境を 1 つ作成

GaussianAvatars と VHAP は CUDA 12.8 / PyTorch 2.9.1 / Python 3.11 に揃って
おり、両方の `setup.sh` が `--no-deps` で慎重にピンを管理しているため、
**同じ conda env に両方を流し込めます**。`demo/setup_env.sh` がそれを行います。

```shell
bash demo/setup_env.sh
```

内部的には:

1. `bash setup.sh` — conda env `gaussian-avatars` を作成し GA スタックを導入
2. `conda activate gaussian-avatars`
3. `bash submodules/VHAP/setup.sh --pip-only --no-assets` — 同じ env に VHAP スタックを追加

衝突する 3 パッケージは GA → VHAP の順インストールにより VHAP の pin が
後勝ちで採用されます (詳細は `demo/setup_env.sh` 冒頭コメント参照):

| パッケージ | GA pin | VHAP pin | 結果 (env内) | 影響 |
| --- | --- | --- | --- | --- |
| tyro | 0.9.13 | 0.8.14 | **0.8.14** | 両者の `tyro.cli(dataclass)` は 0.8/0.9 で互換、影響なし |
| dearpygui | 2.1.4 | 1.11.1 | **1.11.1** | デモ非依存。viewer を使う場合のみ要注意 |
| chumpy | mattloper master (git) | (VHAPの実装による) | VHAPに依存 | `import chumpy` + `chumpy.Ch` が機能すればOK |

特定パッケージだけ別バージョンにしたい場合は env を活性化したあとに
`pip install --no-deps <pkg>==<ver>` で上書きしてください。

env 名を変えたい場合は `GA_ENV` を export してください。

```shell
GA_ENV=my-env bash demo/setup_env.sh
```

### 0.3 FLAME アセット

VHAP と GaussianAvatars がそれぞれ独立に FLAME 2023 を必要とします。
[FLAME 公式サイト](https://flame.is.tue.mpg.de/download.php) からダウンロードし、
以下の 2 箇所に配置してください（シンボリックリンクで重複を避けても OK）。

| ファイル | GaussianAvatars 側 | VHAP 側 |
| --- | --- | --- |
| FLAME 2023 | `flame_model/assets/flame/flame2023.pkl` | `submodules/VHAP/asset/flame/flame2023.pkl` |
| FLAME masks | `flame_model/assets/flame/FLAME_masks.pkl` | `submodules/VHAP/asset/flame/FLAME_masks.pkl` |

---

## 1. VHAP による前処理デモ

### 1a. 単眼動画 (monocular)

VHAP は単眼動画にも対応しているため、これを通すことで GaussianAvatars を
**1 カメラの動画でも学習可能**になります（情報量が少ないぶん再構成品質は
マルチビューには及びませんが、本来の用途を超えて適用できます）。

```shell
# 1. 入力動画を配置
mkdir -p submodules/VHAP/data/monocular
cp /path/to/obama.mp4 submodules/VHAP/data/monocular/

# 2. 前処理 → トラッキング → NeRF 形式エクスポート
SEQUENCE_FILE=obama.mp4 bash demo/01_vhap_preprocess_monocular.sh
```

出力:

```
submodules/VHAP/output/monocular/obama_whiteBg_staticOffset/      # 中間結果
submodules/VHAP/export/monocular/obama_whiteBg_staticOffset_maskBelowLine/
    ├── transforms_train.json
    ├── transforms_val.json
    ├── transforms_test.json
    ├── images/
    └── flame_param/
```

### 1b. NeRSemble マルチビュー

```shell
# 1. NeRSemble の素材を配置 (公式の data/nersemble/<subject>/<sequence>* レイアウト)
#    submodules/VHAP/data/nersemble/074/EMO-1/...

# 2. 16 ビュー一括前処理 → トラッキング → エクスポート
SUBJECT=074 SEQUENCE=EMO-1 bash demo/01_vhap_preprocess_nersemble.sh
```

複数シーケンスを 1 つのデータセットに結合したい場合は、VHAP の
`vhap/combine_nerf_datasets.py` を直接実行してください
（`submodules/VHAP/doc/nersemble.md` Step 4 参照）。

### 主な環境変数

| 変数 | デフォルト | 用途 |
| --- | --- | --- |
| `SEQUENCE_FILE` | `obama.mp4` | 単眼: 入力動画ファイル名 |
| `SEQUENCE` | `${SEQUENCE_FILE%.*}` (mono) / `EMO-1` (nersemble) | シーケンス名 |
| `SUBJECT` | `074` | NeRSemble の subject ID |
| `DOWNSAMPLE` | `4` | NeRSemble: 画像ダウンサンプル倍率 |
| `SUFFIX` / `EXPORT_SUFFIX` | (派生名) | 出力フォルダ名のサフィックス |

---

## 2. GaussianAvatars 学習デモ

`SOURCE_PATH` に Step 1 の export フォルダの **絶対パス** を渡します。

```shell
# 単眼の場合
SOURCE_PATH=$PWD/submodules/VHAP/export/monocular/obama_whiteBg_staticOffset_maskBelowLine \
  bash demo/02_train.sh

# NeRSemble の場合
SOURCE_PATH=$PWD/submodules/VHAP/export/nersemble/074_EMO-1_v16_DS4_whiteBg_staticOffset_maskBelowLine \
  bash demo/02_train.sh
```

学習中は `Training progress: NN%|#####| iter/total [elapsed<eta, it/s, loss=...]`
の tqdm バーが表示されます。学習終了後、
`output/<RUN_NAME>/point_cloud/iteration_*/point_cloud.ply` などが生成されます。

### 主な環境変数

| 変数 | デフォルト | 用途 |
| --- | --- | --- |
| `SOURCE_PATH` | (必須) | VHAP export ディレクトリ |
| `MODEL_PATH` | `output/<basename of SOURCE_PATH>` | 学習成果物の保存先 |
| `RUN_NAME` | `<basename of SOURCE_PATH>` | `MODEL_PATH` 名の自動生成に使用 |
| `ITERATIONS` | `600000` | 学習イテレーション数。動作確認は `30000` 程度でも可 |
| `PORT` | `60000` | リモートビューア用 GUI ポート |

学習中に `python remote_viewer.py --port 60000` で進捗を可視化することも可能です
（リモートビューアは tqdm とは別系統）。

---

## 3. レンダリングデモ

```shell
MODEL_PATH=$PWD/output/obama_whiteBg_staticOffset_maskBelowLine \
  bash demo/03_render.sh
```

レンダリング中は `Rendering progress: NN%|#####| view/total [elapsed<eta, it/s]`
が表示され、完了後は train / val / test それぞれの PNG シーケンスと MP4 が
出力されます。

### 主な環境変数

| 変数 | デフォルト | 用途 |
| --- | --- | --- |
| `MODEL_PATH` | (必須) | 学習済みモデルディレクトリ |
| `ITERATION` | `-1` (最新) | ロードするイテレーション |
| `SELECT_CAMERA_ID` | (なし) | 単一カメラのみレンダリング (例: NeRSemble の正面 = `8`) |
| `TARGET_PATH` | (なし) | クロスアイデンティティ再現用に別シーケンスを駆動モーションとして使用 |

### 例: クロスアイデンティティ再現

```shell
MODEL_PATH=$PWD/output/074_EMO-1_v16_DS4_whiteBg_staticOffset_maskBelowLine \
TARGET_PATH=$PWD/submodules/VHAP/export/nersemble/218_FREE_v16_DS4_whiteBg_staticOffset_maskBelowLine \
SELECT_CAMERA_ID=8 \
  bash demo/03_render.sh
```

---

## エンドツーエンドの最短コマンド (単眼)

```shell
# 0. submodule + 統合 env (初回のみ)
git submodule update --init --recursive
bash demo/setup_env.sh

# 1. 動画を置いて前処理
mkdir -p submodules/VHAP/data/monocular
cp /path/to/obama.mp4 submodules/VHAP/data/monocular/
SEQUENCE_FILE=obama.mp4 bash demo/01_vhap_preprocess_monocular.sh

# 2. 学習 (動作確認なら 30k iter で十分)
SOURCE_PATH=$PWD/submodules/VHAP/export/monocular/obama_whiteBg_staticOffset_maskBelowLine \
ITERATIONS=30000 \
  bash demo/02_train.sh

# 3. レンダリング
MODEL_PATH=$PWD/output/obama_whiteBg_staticOffset_maskBelowLine \
  bash demo/03_render.sh
```

---

## トラブルシューティング

- **`conda env 'gaussian-avatars' not found`**: `bash demo/setup_env.sh` を未実行。
- **tqdm バーが表示されない**: 出力をパイプ/ファイルにリダイレクトしている場合、
  tqdm はコンパクトモードに切り替わります。素のターミナルで実行するか、
  `script -q -c "bash demo/02_train.sh" /tmp/log.txt` のように pty を介すと
  通常モードのバーが残ります。
- **`input video not found`**: VHAP 側の `data/monocular/` に動画を置く必要があります。
  パスは VHAP のルート (`submodules/VHAP/`) からの相対パスで解決されます。
- **`transforms_train.json` が見つからない**: VHAP の export ステップが未完了です。
  `submodules/VHAP/export/...` の下に各 transforms\_\*.json があるか確認してください。
- **FLAME pickle 読み込みエラー**: `flame2023.pkl` を VHAP 側 / GaussianAvatars 側
  の両方に配置しているか確認してください。
- **conda env を分けたい場合**: 旧バージョン (env 別) の構成が必要なら、
  `bash setup.sh` と `bash submodules/VHAP/setup.sh` を個別に実行し、
  デモ側で `GA_ENV` / 直接 `conda activate` を書き換えてください。

# GaussianAvatars × VHAP End-to-End Demo

このディレクトリは **VHAP (cuda128 fork)** を使った前処理から
**GaussianAvatars** の学習・レンダリングまでを 3 ステップで実行するための
デモスクリプトを提供します。

VHAP は `submodules/VHAP` に git submodule として組み込まれており、
[MTamon/VHAP @ cuda128](https://github.com/MTamon/VHAP/tree/cuda128) を参照します。

```
demo/
├── _common.sh                              # shared helpers (env activation, paths, unbuffered py)
├── setup_env.sh                            # 統合 conda env を構築
├── 01_vhap_preprocess_monocular.sh         # 単眼動画 1 本 → NeRF 形式
├── 01_vhap_preprocess_nersemble.sh         # NeRSemble マルチビュー → NeRF 形式
├── 01c_vhap_preprocess_monocular_multi.sh  # 同条件で複数撮りした単眼動画 → 1 つの NeRF 形式
├── 02_train.sh                             # GaussianAvatars 学習
└── 03_render.sh                            # 学習済みアバターのレンダリング
```

`PYTHONUNBUFFERED=1` + `python -u` をデモ側で指定しているので、
**前処理・学習・レンダリングそれぞれの段階で tqdm のプログレスバーが
リアルタイムで表示されます**（FPS / iter / ETA が見えます）。

デモスクリプトの実行パラメータは `--source-path` や `--iterations` のような
CLI オプションで渡します。環境変数はシェルセッションや上位のジョブランナーから
意図せず継承されることがあるため、データパスやイテレーション数の指定には
使わない方針です。

| 段階 | プログレスバー実装 |
| --- | --- |
| 前処理 (matting / extraction) | `submodules/VHAP/vhap/preprocess_video.py:66,120,141` |
| FLAME tracking | `submodules/VHAP/vhap/track*.py` 内の global epoch ループ |
| NeRF export | `submodules/VHAP/vhap/export_as_nerf_dataset.py:68,241,380` |
| 学習 | `train.py:60` `Training progress` bar |
| レンダリング | `render.py:70` `Rendering progress` bar |

---

## 0. 前提条件

### 0.1 リポジトリの取得

```shell
git clone <this repo>
cd GaussianAvatars
```

> submodule (`submodules/{simple-knn, diff-gaussian-rasterization, VHAP}`)
> は `bash setup.sh` の Step [0/5] で**自動初期化**されます。明示的に
> `git submodule update --init --recursive` を実行しても OK ですが、
> 必須ではありません。

### 0.2 統合 conda 環境を 1 つ作成

GaussianAvatars と VHAP は CUDA 12.8 / PyTorch 2.9.1 / Python 3.11 に揃って
おり、両方の `setup.sh` が `--no-deps` で慎重にピンを管理しているため、
**同じ conda env に両方を流し込めます**。`demo/setup_env.sh` がそれを行います。

```shell
bash demo/setup_env.sh
```

内部的には:

1. `bash setup.sh` — submodule auto-init → conda env `gaussian-avatars` 作成 → GA スタックを導入
2. `conda activate gaussian-avatars`
3. `bash submodules/VHAP/setup.sh --pip-only --no-assets` — 同じ env に VHAP スタックを追加

#### 衝突パッケージの解決方針

GA → VHAP の順インストールで VHAP の pin が後勝ちになります。**これは
デモ用途では正しい挙動**です（`demo/setup_env.sh` 冒頭コメントに根拠あり）:

| パッケージ | GA pin | VHAP pin | 結果 | デモ・本体への影響 |
| --- | --- | --- | --- | --- |
| tyro | 0.9.13 | 0.8.14 | **0.8.14** | デモはVHAP CLIスクリプト (track*.py 等) を呼ぶので0.8.14が正解。GA train.py/render.py は argparse 使用で tyro 非依存 |
| dearpygui | 2.1.4 | 1.11.1 | **1.11.1** | デモは viewer 非依存。VHAP の flame_viewer/flame_editor が動くメリットあり |
| chumpy | mattloper@`580566ea` (0.71) | (VHAP実装に依存) | バージョン報告は **0.71 で一致** | FLAME pickle deserialize 用途で同等 |

個別 override は env を活性化後に:
```shell
conda activate gaussian-avatars
pip install --no-deps tyro==0.9.13       # 例: GAのviewerで0.9機能を使いたい
pip install --no-deps dearpygui==2.1.4   # 例: GAのviewerで2.x機能を使いたい
```

env 名を変えたい場合:
```shell
bash demo/setup_env.sh --env my-env
```

### 0.3 FLAME アセット (デフォルトで自動ダウンロード)

VHAP と GaussianAvatars はそれぞれ FLAME 2023 を必要とします:

| ファイル | GaussianAvatars 側 | VHAP 側 |
| --- | --- | --- |
| FLAME 2023 | `flame_model/assets/flame/flame2023.pkl` | `submodules/VHAP/asset/flame/flame2023.pkl` |
| FLAME masks | `flame_model/assets/flame/FLAME_masks.pkl` | `submodules/VHAP/asset/flame/FLAME_masks.pkl` |

**`bash demo/setup_env.sh` を実行すると、FLAME のユーザー名/パスワードを
1度だけ対話で聞かれ、両方の場所に配置されます** (GA 側にダウンロード →
VHAP 側はそこへの symlink)。FLAME サーバーへの問い合わせは1回だけです。

非対話で実行したい場合は CLI オプションで渡せます:

```shell
bash demo/setup_env.sh --flame-user 'you@example.com' --flame-pass '...'
```

#### スキップしたい場合

すでにアセットを持っている、または FLAME アカウントが無い場合:

```shell
bash demo/setup_env.sh --skip-download-assets
```

その場合は手動で上の表の2箇所に flame2023.pkl / FLAME_masks.pkl を配置してください。
[FLAME 公式サイト](https://flame.is.tue.mpg.de/download.php) からダウンロードできます。

#### スタンドアロン実行

env 構築とは独立にアセットだけ取得することもできます:

```shell
# GA 側
bash download_assets.sh                                 # 対話
bash download_assets.sh --flame_user U --flame_pass P   # 非対話

# VHAP 側
bash submodules/VHAP/download_assets.sh                 # 同じ I/F
```

GA の `download_assets.sh` は、VHAP 側に既にファイルがある場合は **symlink で
再利用** し、ダウンロードを省略します（その逆方向は VHAP の `download_assets.sh`
は対応していないため、`demo/setup_env.sh` 内で明示的に symlink を作っています）。

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
bash demo/01_vhap_preprocess_monocular.sh --sequence-file obama.mp4
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
bash demo/01_vhap_preprocess_nersemble.sh --subject 074 --sequence EMO-1
```

複数シーケンスを 1 つのデータセットに結合したい場合は、VHAP の
`vhap/combine_nerf_datasets.py` を直接実行してください
（`submodules/VHAP/doc/nersemble.md` Step 4 参照）。

### 主な CLI オプション

| オプション | デフォルト | 用途 |
| --- | --- | --- |
| `--sequence-file` | `obama.mp4` | 単眼: 入力動画ファイル名 |
| `--sequence` | `<sequence-file の stem>` (mono) / `EMO-1` (nersemble) | シーケンス名 |
| `--subject` | `074` | NeRSemble の subject ID (1c でも必須) |
| `--downsample` | `4` | NeRSemble: 画像ダウンサンプル倍率 |
| `--suffix` / `--export-suffix` | (派生名) | 出力フォルダ名のサフィックス |
| `--batch-size` | `16` | 単眼 VHAP tracking のフレーム batch size。保守的にしたい場合は `1` |
| `--no-skip-existing` | off | 既存成果物と VHAP landmark cache を再利用せず単眼 VHAP 前処理を再実行 |

### 1c. 同条件で撮った複数単眼テイクを 1 つのデータセットへ

同一日・同一髪型・同一服装・同一照明・同一カメラ・同一背景で
**異なるタイミングに録った正面向き単眼動画が複数本** ある場合、
それらを束ねて 1 つの GaussianAvatars 学習データセットに変換できます。

`01_vhap_preprocess_monocular.sh` は 1 動画専用ですが、`01c_*_multi.sh` は
**FLAME のグローバルパラメータ** (shape / static_offset / lights / tex_* /
focal_length) を全テイクで**共有**することで identity の整合を取ります。

#### パイプライン全体像

```
Phase A:  shape source clip を通常 track → tracked_flame_params.npz を確定
Phase B:  残り全テイクを `--load-globals-only --freeze-globals-from-init`
          で再 track (Phase A の npz から globals をロードし最適化対象から除外)
Phase C:  vhap/combine_nerf_datasets.py で全テイクの export を結合
```

#### 共有/非共有パラメータ

| パラメータ | 性質 | 全テイクで共有 | 根拠 |
| --- | --- | --- | --- |
| `shape` | global | ✓ | 同一被験者 |
| `static_offset` | global | ✓ | 同一髪型・服装 |
| `tex_extra` / `tex_pca` | global | ✓ | 同一照明・同一肌反射 |
| `lights` (SH) | global | ✓ | 同一撮影環境 |
| `focal_length` | global | ✓ | 同一カメラ・同一画角 |
| `rotation`, `translation` | per-frame | テイクごと最適化 | テイクごとの頭部運動 |
| `neck_pose`, `jaw_pose`, `eyes_pose` | per-frame | テイクごと最適化 | テイクごとの首・顎・視線 |
| `expr` | per-frame | テイクごと最適化 | テイクごとの表情 |
| `dynamic_offset` (使う場合) | per-frame | テイクごと最適化 | テイクごとの一過性変形 |

このフラグを使うには **VHAP submodule が `cuda128alpha` 以降** である必要
があります (`load-globals-only` / `freeze-globals-from-init` フラグ)。本リポジトリの
`.gitmodules` は既に `cuda128alpha` を指しているため、`git submodule update --remote`
で同期してください。

#### 撮影条件 (前提)

- 同一被験者・同一日に撮影
- **髪型・服装・照明・カメラ位置・背景がテイク間で変わっていない**
- 全テイクで概ね正面向き (アバターのカメラ視点が正面に集中する点で同じ)
- カメラ機材・レンズ・画角がテイク間で同じ (focal_length 共有のため)

これらが崩れている場合は、各テイクを `01_vhap_preprocess_monocular.sh` で
独立に track し、最後だけ `vhap/combine_nerf_datasets.py` で結合してください
(その場合、shape は先頭テイクからコピーされ、他テイクは独立フィット結果が
混在することになります。品質は劣る可能性があります)。

#### shape source clip の選び方

Phase A で全パラメータを最適化する 1 本を `--shape-source-clip` で**手動指定**します。
以下の条件を満たすものが望ましいです:

- 最も**長い**テイク(最適化に使えるフレームが多いほど shape が安定)
- 最も**正面**にカメラを向けているテイク(顔の幾何精度が上がる)
- **照明が安定**していて影が出ていないテイク
- 髪が顔にかかっていない、眼鏡や帽子が外れていないテイク

#### 実行例

```shell
# 6 本セット (約 12 分相当)、take1.mp4 を shape source に指定
bash demo/01c_vhap_preprocess_monocular_multi.sh \
  --subject subj01 \
  --shape-source-clip take1 \
  --sequence-file take1.mp4 \
  --sequence-file take2.mp4 \
  --sequence-file take3.mp4 \
  --sequence-file take4.mp4 \
  --sequence-file take5.mp4 \
  --sequence-file take6.mp4
```

入力配置:

```
submodules/VHAP/data/monocular/take1.mp4
submodules/VHAP/data/monocular/take2.mp4
...
submodules/VHAP/data/monocular/take6.mp4
```

出力 (例):

```
submodules/VHAP/output/monocular/subj01_take1_whiteBg_staticOffset/<TS>/...   # Phase A の生 track
submodules/VHAP/output/monocular/subj01_take2_whiteBg_staticOffset/<TS>/...   # Phase B (global 凍結)
...
submodules/VHAP/export/monocular/subj01_take1_whiteBg_staticOffset_maskBelowLine/   # 各テイク export
submodules/VHAP/export/monocular/subj01_take2_whiteBg_staticOffset_maskBelowLine/
...
submodules/VHAP/export/monocular/subj01_UNION6_whiteBg_staticOffset_maskBelowLine/  # 結合後 (これを学習へ)
    ├── transforms_train.json
    ├── transforms_val.json   # 単眼なので空 (novel-view 評価不可)
    ├── transforms_test.json  # division-mode で 1 テイクが test に hold-out
    ├── canonical_flame_param.npz   # Phase A 由来の共有 canonical
    └── sequences_*.txt
```

#### 主な CLI オプション

| オプション | デフォルト | 用途 |
| --- | --- | --- |
| `--subject` | (必須) | combine 時の prefix。**`_` を含めない**こと |
| `--shape-source-clip` | (必須) | Phase A に使うテイクの stem (拡張子有無どちらでも可) |
| `--sequence-file` | (必須・複数指定) | 入力動画パス |
| `--division-mode` | `last` | `random_single` / `random_group` / `last` から 1 つを test に hold-out |
| `--suffix` / `--export-suffix` | (派生名) | 既存スクリプトと同じ命名規則 |
| `--batch-size` | `16` | VHAP tracking のフレーム batch size。保守的にしたい場合は `1` |
| `--no-skip-existing` | off | 既存成果物と VHAP landmark cache を再利用せず全ステージを再実行 |

> **NOTE:** `--subject` の値が `_` を含むと、`vhap/combine_nerf_datasets.py` の
> 被験者一致 assert (`split('_')[0]` で先頭トークンを比較) を満たせなくなります。
> スクリプトは事前にこれを検出してエラー終了します。

---

## 2. GaussianAvatars 学習デモ

`--source-path` に Step 1 の export フォルダの **絶対パス** を渡します。

```shell
# 単眼の場合
bash demo/02_train.sh \
  --source-path "$PWD/submodules/VHAP/export/monocular/obama_whiteBg_staticOffset_maskBelowLine"

# NeRSemble の場合
bash demo/02_train.sh \
  --source-path "$PWD/submodules/VHAP/export/nersemble/074_EMO-1_v16_DS4_whiteBg_staticOffset_maskBelowLine"
```

学習中は `Training progress: NN%|#####| iter/total [elapsed<eta, it/s, loss=...]`
の tqdm バーが表示されます。学習終了後、
`output/<RUN_NAME>/point_cloud/iteration_*/point_cloud.ply` などが生成されます。

### 主な CLI オプション

| オプション | デフォルト | 用途 |
| --- | --- | --- |
| `--source-path` | (必須) | VHAP export ディレクトリ |
| `--model-path` | `output/<basename of source path>` | 学習成果物の保存先 |
| `--run-name` | `<basename of source path>` | `--model-path` 名の自動生成に使用 |
| `--iterations` | `600000` | 学習イテレーション数。動作確認は `30000` 程度でも可 |
| `--port` | `60000` | リモートビューア用 GUI ポート |

学習中に `python remote_viewer.py --port 60000` で進捗を可視化することも可能です
（リモートビューアは tqdm とは別系統）。

---

## 3. レンダリングデモ

```shell
bash demo/03_render.sh \
  --model-path "$PWD/output/obama_whiteBg_staticOffset_maskBelowLine"
```

レンダリング中は `Rendering progress: NN%|#####| view/total [elapsed<eta, it/s]`
が表示され、完了後は train / val / test それぞれの PNG シーケンスと MP4 が
出力されます。

### 主な CLI オプション

| オプション | デフォルト | 用途 |
| --- | --- | --- |
| `--model-path` | (必須) | 学習済みモデルディレクトリ |
| `--iteration` | `-1` (最新) | ロードするイテレーション |
| `--select-camera-id` | (なし) | 単一カメラのみレンダリング (例: NeRSemble の正面 = `8`) |
| `--target-path` | (なし) | クロスアイデンティティ再現用に別シーケンスを駆動モーションとして使用 |

### 例: クロスアイデンティティ再現

```shell
bash demo/03_render.sh \
  --model-path "$PWD/output/074_EMO-1_v16_DS4_whiteBg_staticOffset_maskBelowLine" \
  --target-path "$PWD/submodules/VHAP/export/nersemble/218_FREE_v16_DS4_whiteBg_staticOffset_maskBelowLine" \
  --select-camera-id 8
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
bash demo/01_vhap_preprocess_monocular.sh --sequence-file obama.mp4

# 2. 学習 (動作確認なら 30k iter で十分)
bash demo/02_train.sh \
  --source-path "$PWD/submodules/VHAP/export/monocular/obama_whiteBg_staticOffset_maskBelowLine" \
  --iterations 30000

# 3. レンダリング
bash demo/03_render.sh \
  --model-path "$PWD/output/obama_whiteBg_staticOffset_maskBelowLine"
```

---

## トラブルシューティング

- **`conda env 'gaussian-avatars' not found`**: `bash demo/setup_env.sh` を未実行。
- **キャッシュがホームディレクトリに出る**: demo スクリプトは `XDG_CACHE_HOME`,
  `TORCH_HOME`, `MPLCONFIGDIR`, `TORCH_EXTENSIONS_DIR`, `TMPDIR` などを
  リポジトリ内 `.cache/demo/` に固定します。既に別のシェルで起動済みの処理には
  反映されないため、再実行してください。
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
  デモ側では `--env` で使用する env 名を指定してください。

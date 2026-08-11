# FACTR2 NEXT 外力推定 — 利用ガイド (OMX)

このパッケージは [FACTR2](https://arxiv.org/abs/2606.12406) の **NEXT (Neural External Torque
Estimation)** を、専用の力覚/トルクセンサを持たない ROBOTIS OpenManipulator-X
(`omx_follower`) 上で再現するためのものです。関節の位置・速度・追従誤差の履歴から
「接触が無いときに測定されるはずのモータトルク」を学習し、実測値との残差として外力
(接触力) を推定します。

対象は **アーム5関節のみ** (`shoulder_pan, shoulder_lift, elbow_flex, wrist_flex,
wrist_roll`)。グリッパは Current-based Position 制御で独自の把持力制限を既に持つため対象外です。

## 全体の流れ

```
1. 自由運動データ収集         examples/omx/force_sensing/collect_free_motion.py
   (接触なし・約10分以上・複数関節スイープ)
              │  .npz ログ (position / velocity / goal position / current の時系列)
              ▼
2. NEXT モデル学習            examples/omx/force_sensing/train_next.py
   (LSTM+MLP, L2回帰, 早期終了)
              │  checkpoint (.pt)
              ▼
3. リアルタイム外力推定デモ    examples/omx/force_sensing/demo_force_sensing.py
   (アームを一定姿勢に保持しつつ τ_ext をコンソール表示)

   (検証用)                    examples/omx/force_sensing/evaluate_free_motion.py
   (収集済みログをそのまま推論に流し、接触なし区間で τ_ext ≈ 0 になるか確認)

   (検証用)                    examples/omx/force_sensing/evaluate_dataset_force.py
   (record_bilateral.py で記録済みのデータセットから force.* の分布を集計し、
    上記のノイズフロアと比較する)
```

コアの再利用可能ロジック (`NextTorqueEstimator`, `NextWindowDataset`, `train_next`,
`OnlineExternalTorqueEstimator`) は `src/lerobot/force_estimation/` に置かれており、
`examples/omx/force_sensing/` のスクリプトはこれらを呼び出す薄いラッパーです。

## 実行コマンド例

すべてリポジトリルートから実行します。

```bash
# 1. 自由運動データ収集 (ワークスペースに物を置かず、接触なしで実行)
uv run python -m examples.omx.force_sensing.collect_free_motion \
    --port /dev/ttyACM0 --robot_id omx_follower \
    --output data/omx_free_motion/run1.npz --duration_min 12

# 2. NEXT モデル学習 (複数回分のログをまとめて渡せる)
uv run python -m examples.omx.force_sensing.train_next \
    --data data/omx_free_motion/run1.npz data/omx_free_motion/run2.npz \
    --output checkpoints/omx_next.pt

# 3. リアルタイム外力推定デモ (各関節を素手で押して反応を確認)
uv run python -m examples.omx.force_sensing.demo_force_sensing \
    --port /dev/ttyACM0 --checkpoint checkpoints/omx_next.pt

# (検証用) 収集済みログをそのまま推論に流し、joint別に mean/std/max|.| を表示
uv run python -m examples.omx.force_sensing.evaluate_free_motion \
    --checkpoint checkpoints/omx_next.pt \
    --data data/omx_free_motion/run1.npz data/omx_free_motion/run2.npz

# (検証用) record_bilateral.py で記録済みのデータセットの force.* 分布を集計
uv run python -m examples.omx.force_sensing.evaluate_dataset_force \
    --repo_id <hf_username>/omx_bilateral_force --root data/omx_bilateral_force
```

## 既知の制約・注意点

- **`shoulder_lift` と `elbow_flex` は独立した軸ではありません。** この2軸を単純な矩形範囲で
  独立にスイープすると、手首/グリッパがベースプレート方向に衝突し、モータの過負荷保護が
  作動して停止することがあります (実際に発生した不具合)。`collect_free_motion.py` は
  `examples/omx/record_grab.py` の `_random_stuck_pose()` で検証済みの区分線形の連動範囲
  (`safe_elbow_flex_range(shoulder_lift)`) を再利用し、`elbow_flex` を常に現在の
  `shoulder_lift` に応じた安全範囲へクランプし、`wrist_flex` も
  `horizontal_wrist_flex(shoulder_lift, elbow_flex)` から連動して計算しています。また、
  `Hardware_Error_Status`/`Torque_Enable` を毎セグメント後に監視し、過負荷保護が作動したら
  即座に収集を中断してその時点までのログを保存するようにしています。
  この連動範囲はあくまで既存コードで検証済みの1パターンであり、お使いの設置環境
  (取り付け高さ・配線・周囲の障害物) によっては安全でない可能性があります。初回は
  `--duration_min` を短く設定し、電源スイッチにすぐ手が届く状態で挙動を確認してください。
- **物理単位 (Nm) には較正していません。** Follower の `shoulder_pan/shoulder_lift/elbow_flex`
  は XL430-W250 で、`Present_Current` として読める値は実際には ROBOTIS e-manual 上
  "Present Load" (内部PWMデューティ比から推定した負荷率, 単位0.1%) です。一方
  `wrist_flex/wrist_roll` は XL330-M288 で、こちらは本当の電流センサ (入力電源側電流,
  単位1mA) を持ちますが、モータ相電流そのものではありません。関節ごとに物理的な意味が
  異なるため、トルク定数 K による Nm 換算はせず、各関節の生レジスタ値をそのまま
  「その関節固有のトルク代理信号」として学習・推定に使っています。データ駆動の LSTM が
  関節ごとのスケール差を暗黙的に吸収する設計です。物理単位への較正は、将来のバイラテラル
  力フィードバックのゲイン調整フェーズで必要になった時点で対応します。
- **100Hz サンプリングは実機で保証されていません。** `DynamixelMotorsBus` は1レジスタにつき
  1回の `sync_read` が必要なため、Position・Velocity・Current の3回の読み出しが1制御ステップ
  ごとに発生します。`collect_free_motion.py` は実測タイムスタンプを記録し、
  `dataset.resample_uniform()` で学習前にオフラインで一定100Hzへリサンプリングするため、
  多少のループジッタは許容されますが、極端に遅い場合は収集時間を伸ばすか
  `--duration_min` を増やしてください。
- **特徴量・ターゲットは標準化 (zero-mean, unit-std) して学習しています。** 位置・速度・電流が
  全く異なるスケールの生レジスタ値であるため、論文には明記されていませんが実務上必要な
  前処理です。標準化統計量は checkpoint (`x_mean`, `x_std`, `y_mean`, `y_std`) に保存され、
  `OnlineExternalTorqueEstimator` が推論時に自動的に同じ変換を適用します。
- **`τ_ext` はどちらも生レジスタ単位の相対量です。** 関節間の大小比較や、同一関節内での経時変化の
  比較には使えますが、関節をまたいだ絶対的な力の大きさの比較 (Nm換算なしでの比較) には
  注意してください。

## チューニング: ノイズと接触時の差が小さいとき

`demo_force_sensing.py`/バイラテラル制御でのフォースフィードバックは動くものの、
`evaluate_free_motion.py` のノイズフロアと実際に接触した際の `τ_ext` の差が小さいと感じる場合、
以下を疑ってください (可能性が高い順)。

1. **学習時と推論時のサンプリングレート不一致。** `train_next.py` は既定で `--resample-hz 100.0`
   (=`history_length=50` が学習時は約0.5秒分の履歴) を前提にしていますが、実際の制御ループ
   (`bilateral_teleop_demo.py`/`record_bilateral.py`) はシリアル通信の直列実行 (複数バスへの
   `sync_read`/`sync_write` を毎ステップ何度も発行) のため、実測レートが100Hzを大きく下回ることが
   あります (起動時ログや、`bilateral_teleop_demo.py` が毎秒表示する実測Hzで確認できます)。この場合
   推論時の履歴ウィンドウが学習時より長い実時間 (例: 48Hzなら約1.04秒) をカバーしてしまい、
   学習時の分布とずれます。**対処**: 既存の `.npz` 自由運動ログに対して、実測レートに合わせた
   `--resample-hz` で再学習するだけで直せます (新規データ収集は不要)。
   ```bash
   uv run python -m examples.omx.force_sensing.train_next \
       --data data/omx_free_motion/run1.npz data/omx_free_motion/run2.npz \
       --output checkpoints/omx_next_48hz.pt --resample-hz 48.0
   ```
   `bilateral_teleop_demo.py`/`record_bilateral.py` は起動時に `checkpoint` の `resample_hz` と
   `--hz` が20%以上ずれていると警告を出すので、目安として活用してください (実測レートは
   `--hz` 未満になりがちな点に注意)。
2. **チューニング効果を数値で確認する。** `evaluate_dataset_force.py` (上記) を使い、
   `record_bilateral.py` で録ったデータセットの `force.*` の分布 (全体・エピソードごとの
   `max|.|` など) を、`evaluate_free_motion.py` で見たノイズフロアと並べて比較してください。
   再学習やチューニングの前後でこの数値を比較することで、体感だけに頼らず効果を判断できます。
3. **推定出力の平滑化 (オプション)。** `OnlineExternalTorqueEstimator(checkpoint,
   smoothing_alpha=0.3)` のように渡す (または `bilateral_teleop_demo.py`/`record_bilateral.py`
   の `--force_smoothing_alpha 0.3`) と、関節ごとにEMA (`smoothed = alpha*raw +
   (1-alpha)*smoothed_prev`) がかかりノイズが減ります。既定は `None` (無効、生の `τ_ext` を返す)
   なので既存の挙動は変わりません。ただし値を小さくしすぎると接触の立ち上がりが鈍るため、
   再学習だけで十分な場合はこの段は不要です。効果は同じく `evaluate_dataset_force.py` で確認して
   ください。

## 検証方法

- **`evaluate_free_motion.py` で接触なしデータを流す**: 学習に使った (または学習に使って
  いない、より厳密な検証なら別途収集した) 自由運動ログをそのまま推論に流し、関節ごとに
  `mean` / `std` / `max|.|` を表示します。接触が無いはずのデータなので、これらの値が
  0付近の小さな値に収まっていれば「自由空間モデルが安定してフィットしている」ことの
  裏付けになります。目安として、収集中に動いていた区間で観測される `Present_Current` の
  変動幅 (収集ログの `current` 列を見て確認できます) と比べて、ここでの `std`/`max|.|` が
  十分小さければ (数% 程度以下) 良好、動作時の変動幅と同程度かそれより大きい場合は
  データ不足や過学習/未学習を疑ってください。
- **`demo_force_sensing.py` で接触ありの反応を確認**: 各関節を個別に軽く/強く押して、
  `τ_ext` の符号・大小が力の向き・強さと定性的に一致するかを確認してください。真に意味の
  ある結果かどうかは、この接触時の値が `evaluate_free_motion.py` で見た「接触なしのノイズ
  水準」よりも明確に大きいかどうかで判断してください — ノイズ水準と同程度の値しか出ない
  場合は、モデルが接触を検出できていない可能性があります。特に XL430 (load%ベース) と
  XL330 (電流ベース) で応答性に差が出ないかも比較すると良いです。

## 将来のバイラテラル制御・重力補償フェーズへの拡張ポイント

このパッケージは今回 (NEXT による外力推定) のみを実装しており、バイラテラル力フィードバック
と重力補償は未実装です。今後の実装では以下のフックが使えます:

- `OnlineExternalTorqueEstimator` を `omx_leader.OmxLeader.send_feedback()`
  (`src/lerobot/teleoperators/omx_leader/omx_leader.py`, 現状 `NotImplementedError`) から
  呼び出し、`τ_feedback = K_fp · τ_ext` (原著 FACTR と同一則) をリーダーの `Goal_Current` に
  書き込む形で統合できます。ただしリーダー側モータを Position 制御から Current Control
  Mode に切り替える改修が別途必要です。
- `src/lerobot/scripts/lerobot_teleoperate.py` の
  `if robot.name == "unitree_g1": teleop.send_feedback(obs)` というハードコードされた
  特例分岐を、能力ベースの汎用フックに置き換える必要があります。
- 重力補償にはリーダーアーム用の URDF (ROBOTIS の `open_manipulator` リポジトリ等から用意)
  と、`lerobot[kinematics]` extra 経由で利用可能な Pinocchio の RNEA
  (`τ_g = modifier × rnea(model, q, q̇, 0)`) が必要です。NEXT 自体は自由空間トルクを
  データから直接学習するため、この重力補償モデルとは独立しています。

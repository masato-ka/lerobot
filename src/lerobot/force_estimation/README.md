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
```

## 既知の制約・注意点

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

## 検証方法

- 学習後、収集データ自体を推論にかけ、接触の無い区間で `τ_ext ≈ 0` (ノイズレベル程度) に
  なることを確認してください。
- `demo_force_sensing.py` を実行し、各関節を個別に軽く/強く押して `τ_ext` の符号・大小が
  力の向き・強さと定性的に一致するかを確認してください。特に XL430 (load%ベース) と XL330
  (電流ベース) で応答性に差が出ないか比較すると良いです。

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

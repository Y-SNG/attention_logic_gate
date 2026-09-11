# attention_logic_gate

微分可能論理ゲートネットワーク(difflogic; Petersen et al.)で **Attention(入力依存ルーティング)を学習できるか** の検証。

**結論(Phase 1): できる。** 完全一致型ゲートAttention(設計A・QK共有エンコーダ)は、residual初期化 + 12bitコードの下で、連想想起タスクを **3シード全てで100%** 解き、学習系列長 N=8 から **N=16/32 へ完全汎化**する。しかも学習後にゲートを固定した**純粋なブール回路(soft→hardの精度劣化ゼロ)**として。

**Phase 2**: 閾値型のハード化ギャップはθ整数校正で全面解決(6/6ラン100%)。induction head(カウント+argmax=多数決読み出し)も**オラクル上限で全長100%**。2段ルーティングの合成は段階学習でなら100%に到達するが、end-to-end学習は未解決 — これが現在のフロンティア。

## セットアップ

```bash
pip install -r requirements.txt   # CPUのみで全実験が回る(Phase1は1ラン約30-60秒)
python experiments/phase0_sanity.py --skip-mnist
python experiments/phase1_recall.py --residual-init --code-bits 12 --hidden 96 --steps 8000
```

`gatelogic/` は difflogic の純PyTorch再実装(真理値表ベースの16ゲート緩和 + `forward_hard` によるビット演算推論)。API形状を本家に合わせてあるので、GPU環境では本家CUDA実装に差し替え可能。

## Phase 0: パイプライン検証(`results/phase0.json`)

| タスク | soft精度 | hard精度(ゲート固定・ビット演算) |
|---|---|---|
| XOR-2 | 100% | 100%(soft/hard完全一致) |
| parity-4 | 100% | 100%(同上) |
| 2:1 MUX | 88.7% | 88.3% |
| MNIST 14×14二値化(5600ゲート) | 87.4% | 87.2% |

「学習 → ゲート固定 → ビット演算推論」のループは正しく動く。MUX(=最小の入力依存ルーティング)が1シードで解け残るのは、後のPhase 1初期結果と同根の前兆。

## Phase 1: 連想想起でゲートAttentionを学習

タスク: `k1 v1 ... kN vN` + クエリ `kq`(文脈中のキー)→ 対応する `vq` を出力。キー6bit(語彙64、系列内は重複なし)、値4bit。毎ステップ新規サンプル(暗記不可能、ルーティング規則のみが汎化する)。N=8で学習、N=8/16/32で評価(4bit完全一致率)。

設計(集約部はどちらも固定構造・位置パラメータなし → 任意長を受理):

- **A(ハッシュ一致型)**: `match_i = AND_j XNOR(codeK_ij, codeQ_j)`、`out = OR_i (match_i AND v_i)`。エンコーダのみ学習
- **B(popcount閾値型)**: `match_i = [popcount(XNOR) ≥ θ]`(θ学習、sigmoidで緩和)

### 対照実験(タスクの妥当性)

| モデル | N=8 | N=16 | N=32 |
|---|---|---|---|
| MLP(Attentionなし) | 19% | — | — |
| softmax Attention(1ヘッド) | 100% | 100% | 100% |

MLPでは解けず、Attentionがあれば解ける = タスクは「入力依存ルーティング」を正しく単離している。

### 主結果(`results/phase1b.json`: residual初期化 + 12bitコード + 混合長、8000step、3シード)

| モデル | N=8 soft/hard | N=16 soft/hard | N=32 soft/hard |
|---|---|---|---|
| **gateA-shared** | **100 / 100(3/3シード)** | **100 / 100** | **100 / 100** |
| gateA-sep | 79–100 / 同左 | 57–100 | 32–100(1/3シードで全長100%) |
| gateB-shared | ~99 / 85–100 | 94–98 / 70–100 | 76–91 / 47–100 |
| gateB-sep | 98–100 / 79–100 | 94–97 / 60–100 | 77–91 / 36–100 |

- **gateA-shared が第1図の主張**: 学習されたブール回路による入力依存ルーティング、学習長の4倍への完全汎化、ハード化劣化ゼロ
- Q/K別エンコーダ(gateA-sep)は「2つの回路のコード空間アライメント」が本質的難所で、シード依存
- B(閾値型)はsoftでは最適化しやすいが、ハード化の成否がθの収束位置に依存(ハード化で改善するシードもある)

### アブレーション(gateA-shared、N=8 soft平均、3シード)

| 設定 | N=8 | 全長100%達成 |
|---|---|---|
| ベースライン(6bitコード) | 72% | ✗ |
| + residual初期化のみ | 79% | ✗ |
| + 12bitコードのみ | 73% | ✗ |
| + 混合長カリキュラムのみ | 55% | ✗(単独では有害) |
| **residual初期化 + 12bitコード** | **100%** | **✓ 3/3シード** |
| residual初期化 + 混合長 | 79% | ✗ |
| 12bitコード + 混合長 | 61% | ✗ |

**効くのは「residual初期化 × コード幅の余裕」の組**。解釈: residual初期化(パススルーゲート寄りの初期化)が深さ方向の勾配と恒等写像への到達可能性を確保し、12bitコード(6bitキーに対し2倍)が「厳密な全単射」を「余裕のある単射」に緩めて衝突回避を容易にする。片方だけでは局所解(コード衝突)から抜けられない。カリキュラムは不要。

## Phase 2: ハード化ギャップ・多段合成・induction head

### 2a: B(閾値型)のハード化ギャップ解消 — 解決(`results/phase2a.json`)

処方: 学習中にβ(sigmoid閾値の鋭さ)を2→10へアニール + 学習後にθを整数グリッドで検証校正。
**6ラン(sep/shared × 3シード)全てで hard 100%(N=8/16/32)**。校正θは11〜12(12bit中)に収束 = ほぼ完全一致判定をpopcountで実装した形。soft 5.9%なのにhard 100%というシードもあり(連続緩和より離散回路が正しい)、θ校正が本質的に効いている。

### 2c: induction head(LMへの一歩)— 解決(`results/phase2c.json`)

タスク: 相異な64語彙トークンのランダムパターン(長さT/2)+その複写。後半の次トークン予測は「現トークンの過去出現を探し、その次を出す」inductionが必須(サンプル毎に写像が変わるため暗記不能)。

モデル `CountingGateAttention`: prev-token配線は固定、学習するのは一致判定回路のみ。**読み出しはsoftmaxではなく「一致カウント+argmax」= 多値多数決**(元計画の「正規化のゲート化」をこの形で投入)。ハード推論は整数投票。

| モデル | T=32(学習長) | T=64 |
|---|---|---|
| **gate-counting** | **100 / 100(soft/hard、3/3シード)** | **100 / 100** |
| オラクル(完全ルーティング上限) | 100 | 100 |
| Transformer 2層(対照) | 100 | —(学習位置埋め込みのため実行不能) |
| static(現トークンのみ) | 1.6% | 1.5% |

ゲート版は**位置パラメータを持たないため2倍長へも無条件で完全汎化**。

### 2b: 2段ルーティングの合成(V[V[q]])— 部分的成功・研究フロンティア(`results/phase2b*.json`)

| 学習方法(gateA 2層) | N=8 | N=16 | N=32 |
|---|---|---|---|
| end-to-end(QK非共有) | 11–15% | ~3% | ~2%(=1層対照と同等、チャンス付近) |
| end-to-end(QK共有) | 50–78% | 23–55% | 6–23% |
| + 中間目標V[q]の補助損失 | 78–85% | 54–66% | 23–41% |
| **段階学習(hop1→凍結→hop2)** | **100 / 92 / 100** | **100 / 82 / 100** | **100 / 60 / 100**(シード別) |
| softmax 2層(対照) | 100% | 100% | 100% |
| MLP(対照) | 17% | — | — |

全ゲート実験で**soft=hardが完全一致** → ボトルネックは離散化ではなく**合成回路のend-to-end最適化**。段階学習で2/3シードが全長100%になることから、2-hopブール回路の表現は勾配で到達可能であり、「教師信号が最終出力のみのとき、2つのルーティング回路を同時に整列させる」ことだけが未解決。これが論文の第2の主張(限界と処方)になる。

## リポジトリ構成

```
gatelogic/layers.py      LogicLayer(16ゲート緩和・forward_hard)/ GroupSum / GateEncoder
gatelogic/attention.py   GateAttentionA(XNOR+AND)/ GateAttentionB(popcount≥θ)/ MajorityNorm(Phase2用)
gatelogic/baselines.py   softmax Attention / MLP対照
experiments/phase0_sanity.py   ブール関数 + 二値化MNIST(soft vs hard)
experiments/phase1_recall.py   連想想起(--code-bits --hidden --residual-init --mix-lengths --models --out)
experiments/export_circuit.py  勝ち構成をハード化して回路をJSONへ(→ viz/)
experiments/phase2a_theta.py   βアニール+θ整数校正
experiments/phase2b_twohop.py  2-hop連想想起(--models で共有/補助損失/段階学習を選択)
experiments/phase2c_induction.py  induction head(カウント+argmax読み出し)
viz/circuit_viewer.html  学習回路のインタラクティブ可視化
results/                 全実験のJSON
```

## 次のステップ(Phase 3〜)

1. **2-hopのend-to-end最適化**(本命の未解決問題): 段階学習→joint fine-tune、コード幅アニール、Gumbelノイズ、hop間のカリキュラム
2. **CountingGateAttention + ゲートMLPブロックの多層化**で実テキストの文字レベルLM(2-gram一致など文脈幅の拡大)
3. GPU環境で本家difflogic CUDA実装に差し替え、スケール検証 → ModernBERT蒸留

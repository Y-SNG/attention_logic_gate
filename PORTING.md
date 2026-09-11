# GPU環境への移行手順(Phase 4-4)

このリポジトリはCPUのみで全実験が再現できるが、スケール実験(大型LM、ModernBERT蒸留)には
GPUと本家 difflogic(CUDA)への差し替えが必要。手順と対応表を以下にまとめる。

## セットアップ

```bash
git clone <this repo> && cd attention_logic_gate
pip install torch --index-url https://download.pytorch.org/whl/cu121   # 環境に合わせて
pip install difflogic   # Petersen et al. CUDA拡張をビルド(nvcc必須)
```

## APIの対応表

`gatelogic` は本家とAPI形状を揃えてある。差し替えポイントは2箇所のみ:

| gatelogic(このリポジトリ) | difflogic(本家CUDA) | 備考 |
|---|---|---|
| `LogicLayer(in_dim, out_dim, generator)` | `difflogic.LogicLayer(in_dim, out_dim, device='cuda', implementation='cuda', connections='random')` | ゲート順序は本家と同一(truth table = インデックスの2進展開) |
| `GroupSum(k, tau)` | `difflogic.GroupSum(k, tau)` | 同一 |
| `layer.forward_hard(bool_tensor)` | 本家は `model.eval()` + PackBitsTensor / CompiledLogicNet | 本家の方が高速(ビットパック推論) |
| `residual_init=True` | 本家にはない → 差し替え後も**必ず移植する**(`weights.data[:, 3] += 5.0`) | Phase 1の結論: これがないとルーティングが学習できない |
| `layer.temp`(温度アニール) | 本家にはない → softmax温度をパッチ | Phase 4の結論: LMのsoft-hardギャップ解消に必要 |

注意: 本家 `LogicLayer` の実装によってはゲート列挙の順序・初期化分布が異なる版がある。
差し替え後は必ず `experiments/phase0_sanity.py` 相当(XOR/parity → soft・hard一致)から回すこと。

## 学習レシピ(このリポジトリで確立した既定値)

- **residual初期化**(パススルーゲート寄り): 必須。ないと単層ルーティングすら不安定
- **コード幅はキー幅の2倍**(6bitキー→12bitコード): 「厳密な全単射」を「余裕のある単射」に緩める
- **QK共有エンコーダ**: 同一トークン空間なら共有が安定
- **多段ルーティングは後段の学習率ウォームアップ**(0→通常値を40%→70%で線形): 中間教師は不要。
  同時学習は共適応で必ず準最適解に落ちる(補助損失の重みづけでは回避不可)
- **閾値型(popcount≥θ)はβアニール+学習後のθ整数校正**
- **分類/LM読み出しは GroupSum(+スケール学習)、Attention集約はカウント+argmax(多数決)**
- **ゲートsoftmaxの温度アニール**(1.0→0.2、終盤40%): ハード化ギャップが出る場合に

## スケール実験の優先順位

1. GateLM をコーパス全体・文脈256+・多層(Attention→静的回路のブロック積層)で
2. 語彙をバイトBPE(~256)に拡大、コード幅32-64
3. ModernBERT蒸留: 教師のattentionパターンを一致コードの教師信号に使う
   (Phase 3の結論より、多段の中間信号があると学習は容易になる — 蒸留はまさにそれを与える)

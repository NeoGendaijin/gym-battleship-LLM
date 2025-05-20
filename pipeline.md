# gym-battleship-LLM: パイプライン詳細ドキュメント

本ドキュメントは、gym-battleship-LLMリポジトリの全体パイプライン・設計思想・拡張方法・実験/分析フローを論文レベルで詳細に解説します。  
新規エージェントの追加や論文用実験の再現も容易に行えるよう、各構成要素の役割と連携を明示します。

---

## 1. 概要

gym-battleship-LLMは、戦艦ゲーム（Battleship）を題材に、強化学習・模倣学習・LLM（大規模言語モデル）・知識蒸留など多様なAIエージェントの設計・学習・評価・分析を一貫して行う研究基盤です。

---

## 2. ディレクトリ・ファイル構成

```
gym-battleship-LLM/
├── gym_battleship/                # コア環境・アルゴリズム実装
│   ├── environments/              # ゲーム環境（OpenAI Gym形式）
│   ├── distill/                   # 知識蒸留関連
│   └── lpml/                      # LLM用パーサ・注釈
├── expert/                        # エキスパート軌跡収集
├── examples/                      # 実験・分析スクリプト
│   ├── figures/                   # 解析グラフ・レポート
├── eval/                          # 評価用スクリプト
├── distill/                       # 蒸留用スクリプト
├── trajectories/                  # 軌跡データ
├── tests/                         # テスト
├── run_experiment.sh              # 実験一括実行シェル
├── pyproject.toml/requirements.txt # 依存管理
└── pipeline.md                    # ★本ドキュメント
```

---

## 3. パイプライン全体図

```mermaid
flowchart TD
    subgraph 環境
        E1[environments/battleship.py]
        E2[environments/adversarial_battleship.py]
    end
    subgraph エージェント
        A1[examples/train_student_policy.py]
        A2[examples/evaluate_model.py]
        A3[expert/collect_trajectories.py]
        A4[distill/]
    end
    subgraph データ
        D1[trajectories/]
        D2[examples/rag_vs_student_results.json]
    end
    subgraph 分析・可視化
        V1[examples/rag_vs_student_analysis.py]
        V2[examples/figures/repot.md]
    end

    E1 -->|Gym API| A1
    E2 -->|Gym API| A1
    A1 -->|学習済みモデル| D1
    A3 -->|エキスパート軌跡| D1
    D1 -->|訓練/評価| A2
    A2 -->|評価結果| D2
    D2 -->|可視化| V1
    V1 -->|グラフ/レポート| V2
```

---

## 4. 各モジュールの詳細

### 4.1 環境（gym_battleship/environments/）

- `battleship.py`  
  - OpenAI Gym APIに準拠した戦艦ゲーム環境
  - 盤面状態・行動空間・報酬設計・リセット/ステップ関数
- `adversarial_battleship.py`  
  - 難易度調整や対戦型拡張

### 4.2 エージェント・学習・評価

- `examples/train_student_policy.py`  
  - Studentエージェントの強化学習/模倣学習
- `examples/evaluate_model.py`  
  - 学習済みモデルの性能評価
- `distill/`  
  - 知識蒸留（KL, 戦略蒸留など）
- `expert/collect_trajectories.py`  
  - 人間や強いAIによるエキスパート軌跡収集

### 4.3 データ収集・保存

- `trajectories/`  
  - 各種エージェントの行動履歴（軌跡）を保存
- `examples/rag_vs_student_results.json`  
  - RAG型とstudent型エージェントの対戦全記録

### 4.4 分析・可視化

- `examples/rag_vs_student_analysis.py`  
  - 対戦結果をpandas/Seabornで解析し、報酬推移・ヒートマップ・ヒット率・ターン数分布などをグラフ化
- `examples/figures/repot.md`  
  - 生成グラフの解説付きレポート（論文用図表に直結）

---

## 5. エージェントの追加・カスタマイズ方法

1. `examples/`や`distill/`に新規エージェントクラス/スクリプトを追加
2. Gym API（reset, step, action_space, observation_space）に準拠
3. 学習/評価/分析スクリプトでエージェントを指定して実験可能
4. 例: 新規エージェント`my_agent.py`を追加し、`train_student_policy.py`や`evaluate_model.py`でimportして利用

---

## 6. 代表的な実験フロー例

### 6.1 強化学習エージェントの訓練

```bash
poetry run python examples/train_student_policy.py --agent student --episodes 10000
```

### 6.2 エキスパート軌跡の収集

```bash
poetry run python expert/collect_trajectories.py --agent expert
```

### 6.3 モデル評価・対戦

```bash
poetry run python examples/evaluate_model.py --agent1 rag --agent2 student
```

### 6.4 分析・可視化

```bash
poetry run python examples/rag_vs_student_analysis.py
```
- 生成グラフ・解説: `examples/figures/repot.md`

---

## 7. 生成アウトプット例

- [examples/figures/repot.md](examples/figures/repot.md)  
  - 各種グラフと詳細解説を掲載。論文図表にそのまま利用可能。

---

## 8. 再現手順・依存関係

- Python 3.11, Poetry
- 依存: numpy, pandas, matplotlib, seaborn, gym など
- セットアップ例
```bash
cd gym-battleship-LLM
poetry install
```

---

## 9. 拡張性・今後の展望

- LLMエージェントの追加・プロンプト設計の自動化
- マルチエージェント・自己対戦・カリキュラム学習
- 盤面サイズ・ルールの拡張
- 分析モジュールの自動レポート生成

---

## 付録: 参考図表

- [examples/figures/repot.md](examples/figures/repot.md) を参照

---

本ドキュメントを参照することで、gym-battleship-LLMの全体像・拡張方法・実験再現性・論文用図表作成まで一貫して理解・活用できます。

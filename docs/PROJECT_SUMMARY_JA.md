# Time Adapter 研究サマリ（公開範囲）

## 結論

音声対話モデルへ外部タイマー由来の小さな残差ベクトルを注入し、高頻度経路では `/W`（待機）、`/B`（相槌）、`/S`（応答）の短い制御トークンだけを判定する構成を検証した研究スナップショットです。

主要な観測は次の通りです。

- 明示的な沈黙秒数はTransformer hidden stateから強く線形回収でき、Qwen3-4BではR² 0.999だった。
- Qwen2.5-Omniのhidden stateへTime Adapterを注入し、専用decision/proxy headで読む構成はaccuracy 0.990、macro F1 0.989だった。
- 同じhidden注入を未学習のLM headへ直接読ませるだけではmacro F1 0.428に留まった。
- `/W`, `/B`, `/S` を短い制御トークンとしてLoRA学習したDirectLM方式では、audio-only評価でaccuracy 0.998、macro F1 0.998だった。
- 制御トークンと短文応答を同時学習した実験はaccuracy 0.998、macro F1 0.998だった。
- 疑似リアルタイムの制御判定はp95 309.1ms、500ms以内率99.3%だった。本文生成と音声合成は別経路に分ける必要がある。

## 推奨構成

```text
microphone audio prefix
  + explicit silence/timing features
  -> Time Adapter hidden injection
  -> lightweight action-token scoring
  -> /W: silence
  -> /B or /S: asynchronous text/speech generation
```

重要なのは、0.5秒ごとのtickで常に本文生成やTalkerを実行しないことです。高頻度に回すのは制御判定だけにし、応答が必要なイベント時に生成処理を起動します。

## 主な結果

| 系列 | 条件 | accuracy | macro F1 | 補足 |
|---|---|---:|---:|---|
| Omni sequential v2 | correct Time Adapter + proxy head | 0.990 | 0.989 | exact sequence 0.920 |
| Generation hook v3 | correct Time Adapter + base LM head | 0.578 | 0.428 | 目的関数の不一致を確認 |
| DirectLM single-token | audio-only 1,000 timepoints | 0.998 | 0.998 | exact sequence 0.980 |
| Control + response | correct Time Adapter, 500 timepoints | 0.998 | 0.998 | 短文応答も同時学習 |

## 解釈

Time Adapterはhidden stateへ時間情報を運べますが、その情報を既存LM headが望ましい会話行動へ変換できるとは限りません。専用headを使うか、実際に使用する制御トークンのlogitを直接学習する必要があります。モデルサイズを増やすだけでは、この目的関数の不一致は解消しません。

## 公開上の境界

公開版には、テンプレートと決定的ラベル規則から再生成可能なコードだけを残しています。学習データ、生成音声、実録音、モデル重み、およびテキストモデルで意味内容を生成したデータ系列とその派生成果は含めていません。数値は限定された研究評価であり、自然会話・多言語・安全性・バイアス・本番レイテンシを保証しません。

# 05 · Transformer

## Qué es
Arquitectura basada **únicamente en atención** (sin recurrencia ni convolución). Cada token puede relacionarse directamente con todos los demás, y todo el cómputo se paraleliza en GPU. Es la base de los LLM, ViT, Whisper y la mayoría de modelos actuales.

## Para qué sirve
Texto, código, visión, audio, proteínas (AlphaFold), robótica... Si hay una secuencia (o conjunto) de elementos, probablemente hay un Transformer encima.

## Cómo funciona

```mermaid
flowchart TB
    T[Tokens] --> E[Embedding + codificación posicional]
    E --> B1
    subgraph Bloque ×N
      B1[LayerNorm] --> MHA[Multi-Head Self-Attention] --> R1((+))
      R1 --> B2[LayerNorm] --> FFN[Feed-Forward / MLP] --> R2((+))
    end
    R2 --> OUT[LayerNorm + cabeza lineal] --> L[Logits]
```

### Self-attention (escalada por producto punto)
Cada token se proyecta a consulta, clave y valor: `Q = XW_Q`, `K = XW_K`, `V = XW_V`.

```
Attention(Q, K, V) = softmax( Q Kᵀ / √d_k ) V
```

- `QKᵀ` mide similitud entre todos los pares de tokens → matriz `n × n`.
- `√d_k` evita que el softmax se sature.
- **Multi-head**: `h` cabezas en paralelo, cada una con su subespacio; se concatenan y proyectan.
- **Máscara causal**: en modelos generativos, cada token solo ve los anteriores.

### Posición
La atención no conoce el orden, así que se inyecta: senoidal, aprendida, **RoPE** (rotatoria, estándar en LLaMA) o **ALiBi**.

### Costos
- Cómputo y memoria **O(n²)** en la longitud de secuencia.
- **KV-cache**: en inferencia se guardan K y V pasados para no recalcular; domina el uso de memoria en contextos largos.

## Optimizaciones modernas
| Técnica | Qué resuelve |
|---|---|
| **FlashAttention** | Atención exacta con menos acceso a memoria (IO-aware) |
| **GQA / MQA** | Comparte K,V entre cabezas → KV-cache más pequeño |
| **RoPE + escalado** | Extender el contexto |
| **Cuantización** (INT8/4) | Menor memoria y latencia |
| **Speculative decoding** | Acelera la generación con un modelo borrador |
| **LoRA / QLoRA** | Ajuste fino eficiente en parámetros |

## Leyes de escala
La pérdida decrece de forma predecible con parámetros, datos y cómputo (Kaplan 2020; Chinchilla 2022: ≈20 tokens por parámetro para entrenar óptimo en cómputo).

## Pros y contras
| Ventajas | Limitaciones |
|---|---|
| Paralelo, escala excelente | Atención cuadrática en longitud |
| Contexto global directo | Muy hambriento de datos y cómputo |
| Transfer learning potente (preentrenar + afinar) | Inferencia costosa (memoria KV) |

## Cuándo usarlo
Casi siempre que haya suficientes datos o un modelo preentrenado disponible. Para contextos extremos, mira [SSM](12-ssm-mamba.md); para escalar barato, [MoE](11-moe.md).

## Ejemplo
[`examples/transformer_block.py`](../examples/transformer_block.py)

## Notas de práctica
- En producción el cuello de botella es la memoria del KV-cache y la latencia, no la calidad. Mide tokens por segundo y memoria antes de elegir el modelo.
- Cuantiza (INT8/INT4), usa batching continuo y atención eficiente (FlashAttention) antes de pensar en hardware más caro.
- Prefiere ajustar un modelo preentrenado (LoRA/QLoRA) a entrenar desde cero. Entrenar desde cero rara vez se justifica.
- Un buen conjunto de evaluación propio vale más que cualquier cambio de arquitectura.

## Referencias
- Vaswani et al. (2017). *Attention Is All You Need*.
- Dao et al. (2022). *FlashAttention*.
- Hoffmann et al. (2022). *Training Compute-Optimal LLMs* (Chinchilla).

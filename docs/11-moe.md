# 11 · Mixture of Experts (MoE)

## Qué es
Sustituye el FFN denso de cada bloque Transformer por **varios "expertos"** (FFNs) y un **router** que decide a qué expertos enviar cada token. Solo se activan `k` de `E` expertos por token: **muchos parámetros, poco cómputo por token**.

## Para qué sirve
Aumentar la capacidad (conocimiento) de un modelo sin multiplicar el costo de inferencia por token. Ej.: Mixtral 8×7B, DeepSeek-V3, Switch Transformer, y varios modelos de frontera.

## Cómo funciona

```mermaid
flowchart LR
    t[Token] --> R[Router<br/>softmax + top-k]
    R -->|peso g₁| E1[Experto 1]
    R -->|peso g₂| E3[Experto 3]
    R -. no activos .-> E2[Experto 2] & E4[Experto 4]
    E1 & E3 --> S((Σ gᵢ·Eᵢ(x))) --> out[Salida]
```

```
y = Σ_{i ∈ TopK(g(x))}  g_i(x) · E_i(x),     g(x) = softmax(W_r x)
```

### Desafíos
- **Balanceo de carga**: sin control, el router colapsa en unos pocos expertos. Se añade una *auxiliary loss* (o balanceo sin pérdida auxiliar en DeepSeek-V3) y una *capacity factor*.
- **Comunicación**: expertos repartidos en GPUs → *all-to-all*; es el cuello de botella de entrenamiento.
- **Memoria**: aunque se calculen pocos, **todos los expertos deben estar cargados**.

## Variantes
- **Switch (top-1)**, **GShard / Mixtral (top-2)**.
- **Expertos compartidos + finos** (DeepSeekMoE): algunos siempre activos, muchos pequeños especializados.
- **Expert choice**: los expertos eligen tokens, no al revés.

## Pros y contras
| Ventajas | Limitaciones |
|---|---|
| Más capacidad al mismo FLOPs/token | Mucha VRAM (todos los pesos residentes) |
| Entrenamiento eficiente por calidad | Complejidad de ingeniería, balanceo |
| Rendimiento superior a densos del mismo costo | Ajuste fino más delicado |

## Cuándo usarlo
Cuando tienes presupuesto de memoria pero quieres **latencia/FLOPs de un modelo pequeño con calidad de uno grande**. Si la VRAM es el límite (edge), un modelo denso más pequeño suele convenir.

## Ejemplo
[`examples/moe.py`](../examples/moe.py)

## Notas de práctica
- Pregunta primero si la memoria alcanza: aunque solo se activen algunos expertos, todos los pesos deben estar cargados.
- Vigila el balanceo de carga durante el entrenamiento; un router que colapsa en pocos expertos desperdicia capacidad.

## Referencias
- Shazeer et al. (2017). *Outrageously Large Neural Networks: Sparsely-Gated MoE*.
- Fedus et al. (2021). *Switch Transformers*.
- Jiang et al. (2024). *Mixtral of Experts*.

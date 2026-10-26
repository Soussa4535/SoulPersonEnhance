# Soul Person Enhance

**Melhore cada pessoa de uma foto separadamente, e devolva todas para o lugar.**

Um retrato em grupo tem um problema simples: quando você amplia a imagem inteira,
cada rosto recebe uma fatia minúscula da atenção do modelo. Quem está no fundo
continua borrado, quem está na frente ganha textura de plástico.

Este pacote inverte a ordem. Cada pessoa é **recortada**, enquadrada e melhorada
**sozinha**, no tamanho cheio que o modelo sabe tratar. Depois volta para a
imagem original, no pixel exato de onde saiu, com a borda derretida na cena.

```
foto ─► SAM3 detecta pessoas ─► SoulPersonTiles ─► [melhora cada recorte] ─► SoulPersonTilesMerge ─► foto
```

---

## Os nodes

### `SoulPersonTiles` — cada pessoa vira um recorte

Recebe a imagem e as máscaras do SAM3. Devolve uma lista de recortes, um por
pessoa, todos no mesmo tamanho.

| entrada | o que faz |
|---|---|
| `largura` / `altura` | tamanho do recorte. 496 ou 512 funcionam bem |
| `padding` | quanto de folga em volta da pessoa (0.3 = 30%) |
| `modo_caixa` | como a caixa é escolhida; `travada por trechos` mantém estável |
| `min_area` | descarta detecções menores que isso (0.02 = 2% da imagem) |
| `max_pessoas` | **0 = todas.** Ponha 3 e ele pega as três maiores |
| `margem` | margem extra de segurança nas bordas |

Saídas: `crops` (a lista), `crop_masks`, `tracks` (as coordenadas, que o Merge
precisa), `preview` (visualização das caixas) e `quantidade`.

### `SoulPersonTilesMerge` — devolve para o lugar

Recebe a imagem original, os recortes **já melhorados** e os `tracks`.

| entrada | o que faz |
|---|---|
| `crescer` | dilata a máscara antes de colar, em pixels |
| `feather` | suaviza a borda. 20 costuma sumir com a emenda |
| `forca` | 1 = cola inteiro; abaixo disso mistura com o original |

### Três utilitários

`SoulImageSizeCheck`, `SoulDenoise` e `SoulSwitch` — medir, limpar e desviar.
Existem aqui só para o workflow de exemplo rodar sem depender de mais nada.

---

## Instalação

```bash
cd ComfyUI/custom_nodes
git clone <url-deste-repo> SoulPersonEnhance
```

Reinicie o ComfyUI. Não há dependência além do que ele já traz (`torch`,
`numpy`, `Pillow`, `opencv`).

---

## Para rodar o workflow de exemplo

`workflows/Enhance_Pessoas.json` é um pipeline completo. Ele usa outros pacotes
além deste — todos públicos:

| pacote | para quê |
|---|---|
| [comfyui-sam3](https://github.com/PozzettiAndrea/ComfyUI-SAM3) | detectar as pessoas |
| [ComfyUI-Easy-Use](https://github.com/yolain/ComfyUI-Easy-Use) | desvio condicional e utilidades |
| [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) | Set/Get de variáveis |
| [ComfyUI_TTP_Toolset](https://github.com/TTPlanetPig/Comfyui_TTP_Toolset) | fatiar e remontar em tiles |
| [ComfyUI-nunchaku](https://github.com/nunchaku-tech/ComfyUI-nunchaku) | rodar o Flux quantizado |
| [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) | carregar o encoder em GGUF |
| [ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack) | lista ↔ lote de imagens |
| [SeedVR2](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) | pré-ampliar foto pequena |
| [WAS Node Suite](https://github.com/WASasquatch/was-node-suite-comfyui) | salvar com DPI |

### Modelos

| arquivo | pasta |
|---|---|
| `svdq-fp4_r32-fluxmania-legacy.safetensors` | `models/diffusion_models` |
| `t5-v1_1-xxl-encoder-Q8_0.gguf` | `models/text_encoders` |
| `clip_l.safetensors` | `models/text_encoders` |
| `ae.safetensors` | `models/vae` |
| `4x-UltraSharp.pth` | `models/upscale_models` |
| `seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors` | `models/SEEDVR2` |
| `ema_vae_fp16.safetensors` | `models/SEEDVR2` |

O SAM3 baixa sozinho na primeira execução.

> O enhance de cada recorte é só um subgraph trocável. Os dois nodes principais
> não sabem nem se importam com qual modelo melhora o recorte — troque por
> Qwen, SDXL, o que preferir. O que este pacote faz é o recorte e a recolagem.

### Memória

Testado numa RTX 5070 Ti de 16 GB. O recorte e a recolagem custam quase nada;
quem pesa é o modelo que você escolher para melhorar.

---

## Licença

AGPL-3.0-or-later. Veja [LICENSE](LICENSE).

Em resumo: use, estude e modifique à vontade. Se distribuir uma versão
modificada — **inclusive rodando como serviço na internet** — você precisa
publicar o código dela sob a mesma licença.

Os modelos citados acima têm licenças próprias, cada um com as suas regras.
Verifique antes de usar comercialmente.

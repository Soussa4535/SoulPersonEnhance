# Soul Person Enhance

**Melhore cada pessoa de uma foto separadamente — ou só as que você escolher.**

Um retrato em grupo tem um problema simples: quando você amplia a imagem
inteira, cada rosto recebe uma fatia minúscula da atenção do modelo. Quem está
no fundo continua borrado, quem está na frente ganha pele de plástico.

Este pacote inverte a ordem. Cada pessoa é **recortada**, enquadrada e
melhorada **sozinha**, no tamanho cheio que o modelo sabe tratar. Depois volta
para a imagem original, no pixel exato de onde saiu, com a borda derretida na
cena.

E se você não quer todo mundo — **clique nas pessoas que quer.**

```
foto ─► SAM3 acha todas ─► você clica (opcional) ─► recorta ─► melhora ─► cola de volta
```

---

## Os nodes

### `SoulPersonTiles` — cada pessoa vira um recorte

| entrada | |
|---|---|
| `images` | a foto |
| `masks` | as pessoas que o SAM3 achou |
| `pontos` *(opcional)* | os cliques: **só essas** viram recorte |
| `pontos_fora` *(opcional)* | os cliques de exclusão: **essas saem** |

| ajuste | padrão | |
|---|---|---|
| `tamanho` | 496 | lado do recorte, em pixel. É o que o modelo de enhance recebe |
| `padding` | 0.30 | folga em volta da pessoa, pra não cortar cabelo e pé |
| `min_area` | 0.02 | ignora mancha menor que esta % do quadro |
| `max_pessoas` | 0 | 0 = todas. Outro número fica com as N maiores |
| `usar_pontos` | ligado | desligado, ignora os cliques mesmo ligados |

Saídas: `crops` (a lista), `crop_masks`, `tracks` (as coordenadas, que o Merge
precisa), `preview` (as caixas desenhadas) e `quantidade`.

### `SoulPersonTilesMerge` — devolve para o lugar

Recebe a foto original, os recortes **já melhorados** e os `tracks`.

| ajuste | padrão | |
|---|---|---|
| `crescer` | 8 | dilata a máscara antes de colar, em pixels |
| `feather` | 20 | suaviza a borda. Some com a emenda |
| `forca` | 1 | 1 cola inteiro; abaixo disso mistura com o original |

### `SoulPersonTilesAvancado`

O mesmo recorte, com os parâmetros de **vídeo** expostos (`modo_caixa`,
`suavizacao`, `min_presenca`, largura e altura separadas). Use só se for tratar
sequência de quadros; para foto, o `SoulPersonTiles` já faz tudo.

---

## A escolha por clique

O `SoulPersonTiles` aceita pontos no formato do
[comfyui-sam3](https://github.com/PozzettiAndrea/ComfyUI-SAM3), então o seletor
visual já existe — não é preciso instalar mais nada:

```
LoadImage ─┬─► SAM3Grounding ("person") ──► masks ──┐
           │                                         ├─► SoulPersonTiles
           └─► SAM3PointCollector (você clica) ──────┘
```

- **Clique esquerdo** → `positive_points` → entrada `pontos` → essa pessoa entra
- **Shift + clique** → `negative_points` → entrada `pontos_fora` → essa sai

Sem nenhum clique, ele recorta **todas** — o comportamento automático de sempre.

O teste do clique é feito no **rótulo do pixel**, não na caixa: duas pessoas
lado a lado têm caixas que se cruzam, mas rótulos distintos, então o clique
nunca pega a errada.

---

## Instalação

```bash
cd ComfyUI/custom_nodes
git clone <url-deste-repo> SoulPersonEnhance
```

Reinicie o ComfyUI. Sem dependência além do que ele já traz (`torch`, `numpy`,
`Pillow`).

---

## Para rodar o workflow de exemplo

`workflows/Enhance_Pessoas.json` é o pipeline completo. Ele usa outros pacotes,
todos públicos:

| pacote | para quê |
|---|---|
| [comfyui-sam3](https://github.com/PozzettiAndrea/ComfyUI-SAM3) | achar as pessoas e coletar os cliques |
| [ComfyUI-Easy-Use](https://github.com/yolain/ComfyUI-Easy-Use) | listas de imagem e limpeza de VRAM |
| [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) | Set/Get de variáveis |
| [ComfyUI-nunchaku](https://github.com/nunchaku-tech/ComfyUI-nunchaku) | rodar o Flux quantizado |
| [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) | carregar o encoder em GGUF |
| [ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack) | lista ↔ lote de imagens |

### Modelos

| arquivo | pasta |
|---|---|
| `svdq-fp4_r32-fluxmania-legacy.safetensors` | `models/diffusion_models` |
| `t5-v1_1-xxl-encoder-Q8_0.gguf` | `models/text_encoders` |
| `clip_l.safetensors` | `models/text_encoders` |
| `ae.safetensors` | `models/vae` |
| `4x-UltraSharp.pth` | `models/upscale_models` |

O SAM3 baixa sozinho na primeira execução.

> **O enhance é trocável.** Os dois nodes principais não sabem nem se importam
> com qual modelo melhora o recorte — ele vive num subgraph à parte. Troque por
> Qwen, SDXL, o que preferir: o recorte e a recolagem continuam iguais.

### Memória

Testado numa placa de 16 GB. O recorte e a recolagem custam quase nada; quem
pesa é o modelo que você escolher para melhorar, e ele roda **uma pessoa por
vez** — então o custo é por pessoa, não pela imagem toda.

---

## Licença

AGPL-3.0-or-later. Veja [LICENSE](LICENSE).

Em resumo: use, estude e modifique à vontade. Se distribuir uma versão
modificada — **inclusive rodando como serviço na internet** — você precisa
publicar o código dela sob a mesma licença.

Os modelos citados acima têm licenças próprias, cada um com as suas regras.
Verifique antes de usar comercialmente.

Autor: Felipe Soussa.

# Soul Person Enhance

**Enhance each person in a photo on her own — or only the ones you pick.**

A group portrait has a simple problem: upscale the whole image and every face
gets a tiny slice of the model's attention. Whoever is in the back stays
blurry, whoever is in front gets plastic skin.

This pack flips the order. Each person is **cropped**, framed and enhanced
**alone**, at the full size the model knows how to handle. Then she goes back
into the original image, at the exact pixel she came from, with the edge melted
into the scene.

And if you don't want everyone — **click the people you want.**

```
photo ─► SAM3 finds everyone ─► you click (optional) ─► crop ─► enhance ─► paste back
```

![before and after](examples/after.jpg)

At 1:1, the same face before and after. The ear, the hair strands and the
elastic were a smear; the pavement behind her is untouched.

![face detail, before and after](examples/compare-face2.jpg)

---

## The nodes

### `SoulPersonTiles` — each person becomes a crop

| input | |
|---|---|
| `images` | the photo |
| `masks` | the people SAM3 found |
| `points` *(optional)* | the clicks: **only these** become crops |
| `points_exclude` *(optional)* | the exclusion clicks: **these are dropped** |

| setting | default | |
|---|---|---|
| `size` | 496 | side of the crop, in pixels. This is what the enhance model receives |
| `padding` | 0.30 | room around the person, so hair and feet are not clipped |
| `min_area` | 0.02 | ignores any blob smaller than this % of the frame |
| `max_people` | 0 | 0 = everyone. Any other number keeps the N largest |
| `use_points` | on | off, the clicks are ignored even while connected |

Outputs: `crops` (the list), `crop_masks`, `tracks` (the coordinates the Merge
needs), `preview` (the boxes drawn over the photo) and `count`.

### `SoulPersonTilesMerge` — puts them back

Takes the original photo, the **already enhanced** crops, and the `tracks`.

| setting | default | |
|---|---|---|
| `grow` | 8 | dilates the mask before pasting, in pixels |
| `feather` | 20 | softens the edge. This is what makes the seam disappear |
| `strength` | 1 | 1 pastes it whole; below that it blends with the original |

### `SoulPersonTilesAdvanced`

The same cropping, with the **video** parameters exposed (`box_mode`,
`smoothing`, `min_presence`, separate width and height). Use it only to treat a
frame sequence; for a photo, `SoulPersonTiles` already does everything.

---

## Picking by click

`SoulPersonTiles` takes points in the
[comfyui-sam3](https://github.com/PozzettiAndrea/ComfyUI-SAM3) format, so the
visual picker already exists — nothing else to install:

```
LoadImage ─┬─► SAM3Grounding ("person") ──► masks ──┐
           │                                         ├─► SoulPersonTiles
           └─► SAM3PointCollector (you click) ───────┘
```

- **Left click** → `positive_points` → `points` input → this person is in
- **Shift + click** → `negative_points` → `points_exclude` input → this one is out

With no clicks at all it crops **everyone** — the automatic behavior it always had.

The click is tested against the **pixel label**, not the box: two people side
by side have overlapping boxes but distinct labels, so a click never catches
the wrong one.

**The picker shows the photo before you run.** On its own, SAM3PointCollector
only paints its canvas once the graph has executed, so you would have to run
blind before you could see what to click. This pack ships a small frontend
extension (`web/point_preview.js`) that fills that same canvas from the image
Load Image already points at, as soon as you pick it. Load, click, run — once.

---

## Install

```bash
cd ComfyUI/custom_nodes
git clone <url-of-this-repo> SoulPersonEnhance
```

Restart ComfyUI. No dependency beyond what it already ships (`torch`, `numpy`,
`Pillow`).

---

## Running the example workflow

`workflows/Enhance_Pessoas.json` is the complete pipeline. It uses other packs,
all public:

| pack | what for |
|---|---|
| [comfyui-sam3](https://github.com/PozzettiAndrea/ComfyUI-SAM3) | finding the people and collecting the clicks |
| [ComfyUI-Easy-Use](https://github.com/yolain/ComfyUI-Easy-Use) | image lists and VRAM cleanup |
| [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) | Set/Get variables |
| [ComfyUI-nunchaku](https://github.com/nunchaku-tech/ComfyUI-nunchaku) | running quantized Flux |
| [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) | loading the encoder in GGUF |
| [ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack) | list ↔ image batch |

### Models

| file | folder |
|---|---|
| `svdq-fp4_r32-fluxmania-legacy.safetensors` | `models/diffusion_models` |
| `t5-v1_1-xxl-encoder-Q8_0.gguf` | `models/text_encoders` |
| `clip_l.safetensors` | `models/text_encoders` |
| `ae.safetensors` | `models/vae` |
| `4x-UltraSharp.pth` | `models/upscale_models` |

SAM3 downloads itself on the first run.

> **The enhance is swappable.** The two main nodes neither know nor care which
> model improves the crop — it lives in its own subgraph. Swap in Qwen, SDXL,
> whatever you prefer: the cropping and the merging stay the same.

### Memory

Tested on a 16 GB card. Cropping and merging cost almost nothing; what weighs
is the model you choose to enhance with, and it runs **one person at a time**
— so the cost is per person, not for the whole image.

---

## License

AGPL-3.0-or-later. See [LICENSE](LICENSE).

In short: use, study and modify freely. If you distribute a modified version —
**including running it as a service over a network** — you have to publish its
source under the same license.

The models listed above have licenses of their own, each with its own rules.
Check them before using commercially.

Author: Felipe Soussa.

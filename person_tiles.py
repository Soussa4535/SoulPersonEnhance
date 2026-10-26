# -*- coding: utf-8 -*-
"""Soul Person Enhance — recorte e recolagem por pessoa.

Dois nodes para ComfyUI:

    SoulPersonTiles       uma mascara de pessoa (SAM3) vira um recorte proprio,
                          enquadrado e padronizado, pronto para ser melhorado
    SoulPersonTilesMerge   devolve os recortes melhorados para a imagem original,
                          cada um no seu lugar, com borda suavizada

Extraido do pacote pessoal do autor; aqui ficam so os dois nodes de imagem.

Copyright (C) 2026 Felipe Soussa
Licenca AGPL-3.0-or-later — veja LICENSE.
"""
import numpy as np
import torch
from PIL import Image as PILImage, ImageFilter


def _img_np(t):
    """IMAGE tensor -> np [N,H,W,3] float 0-1."""
    a = t.detach().cpu().numpy().astype(np.float32)
    if a.ndim == 3:
        a = a[None, ...]
    if a.shape[-1] == 4:
        a = a[..., :3]
    return np.clip(a, 0.0, 1.0)


def _mask_np(t):
    """MASK tensor -> np [N,H,W] float 0-1, aceitando [H,W] e [N,H,W,1]."""
    a = t.detach().cpu().numpy().astype(np.float32)
    if a.ndim == 2:
        a = a[None, ...]
    if a.ndim == 4:
        a = a[..., 0] if a.shape[-1] == 1 else a.mean(-1)
    return np.clip(a, 0.0, 1.0)


def _resize_rgb(a, w, h):
    p = PILImage.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    return np.asarray(p.resize((w, h), PILImage.LANCZOS), dtype=np.float32) / 255.0


def _resize_gray(a, w, h):
    p = PILImage.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8), mode="L")
    return np.asarray(p.resize((w, h), PILImage.BILINEAR), dtype=np.float32) / 255.0


def _bbox(m, thr=0.5):
    ys, xs = np.where(m > thr)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _det_simples(m):
    b = _bbox(m)
    if b is None:
        return None
    return {"m": m, "lab": None, "idx": 0, "box": b, "area": float((m > 0.5).sum())}


def _blobs(m, thr=0.5):
    """Quebra UMA máscara nas manchas separadas dela (componentes conectados).

    É isto que faz o node achar as pessoas por conta própria: se o detector
    entregar uma máscara só com a turma inteira pintada, cada pessoa que não
    encosta na outra vira uma detecção aqui dentro. Sem isso o node fica
    refém de quantas máscaras o SAM3 resolveu separar.

    Guarda o mapa de rótulos (`lab`) em vez de materializar uma máscara
    [H,W] por pessoa: numa imagem 6000x3348 cada máscara dessas custa 80 MB,
    e 20 pessoas seriam 1,6 GB à toa. A máscara de cada uma é recortada só
    dentro da caixa dela, na hora de usar.
    """
    b = (m > thr).astype(np.uint8)
    if not b.any():
        return []
    try:
        import cv2
    except ImportError:
        d = _det_simples(m)
        return [d] if d else []
    n, lab, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    out = []
    for k in range(1, n):
        x, y, w, h, a = stats[k]
        out.append({"m": m, "lab": lab, "idx": int(k),
                    "box": (int(x), int(y), int(x + w), int(y + h)), "area": float(a)})
    return out


def _pontos_px(p, W, H):
    """SAM3_POINTS_PROMPT -> lista de (x, y) em pixel.

    O SAM3 entrega {"points": [[x, y], ...], "labels": [...]} com x e y
    NORMALIZADOS de 0 a 1. Aceito tambem ja em pixel (quem passa de 1.5 nao
    pode ser normalizado) e uma lista crua de pares, pra nao travar se vier
    de outro node.
    """
    if not p:
        return []
    pares = p.get("points", []) if isinstance(p, dict) else p
    out = []
    for q in pares:
        try:
            x, y = (q["x"], q["y"]) if isinstance(q, dict) else (q[0], q[1])
        except (TypeError, KeyError, IndexError):
            continue
        x, y = float(x), float(y)
        if x <= 1.5 and y <= 1.5:          # normalizado
            x, y = x * W, y * H
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < W and 0 <= yi < H:
            out.append((xi, yi))
    return out


def _tem_ponto(d, pts):
    """O ponto caiu DENTRO desta pessoa? Testa no rotulo, nao na caixa — duas
    pessoas lado a lado tem caixas que se cruzam, mas rotulos distintos."""
    if not pts:
        return False
    lab, idx = d.get("lab"), d.get("idx")
    x0, y0, x1, y1 = d["box"]
    for x, y in pts:
        if lab is not None and idx is not None:
            if lab[y, x] == idx:
                return True
        elif x0 <= x < x1 and y0 <= y < y1:
            return True
    return False


def _filtra_pontos(vivos, sim, nao, onde=""):
    """Deixa passar so quem tem ponto em cima; tira quem tem ponto negativo.

    Serve para as DUAS formas de `vivos` que o node monta: a do lote (uma
    pessoa por imagem) e a das trilhas (uma pessoa ao longo dos frames). Nas
    duas, `v["mask"]` e um dicionario frame -> blob, entao basta olhar todos
    os frames dessa pessoa.
    """
    if not sim and not nao:
        return vivos
    def bate(v, pts):
        return any(_tem_ponto(d, pts) for d in v["mask"].values())
    antes = len(vivos)
    if sim:
        vivos = [v for v in vivos if bate(v, sim)]
    if nao:
        vivos = [v for v in vivos if not bate(v, nao)]
    print("[Soussa_Neo_PersonTiles] pontos%s: %d marcado(s), %d excluido(s) "
          "-> %d de %d pessoa(s)" % (onde, len(sim), len(nao), len(vivos), antes))
    if not vivos:
        raise ValueError(
            "[Soussa_Neo_PersonTiles] nenhum clique caiu em cima de uma pessoa. "
            "Clique dentro do corpo (nao no fundo), ou desligue a entrada "
            "`pontos` para recortar todas.")
    return vivos


def _ajusta_caixa(box, ar, pad, W, H):
    """Infla a caixa pelo padding, força a proporção do tile e tenta empurrar
    pra dentro da imagem. Se mesmo assim não couber, deixa estourar — quem
    resolve isso é o preenchimento por borda no _extrai()."""
    x0, y0, x1, y1 = map(float, box)
    w, h = x1 - x0, y1 - y0

    # padding proporcional ao MAIOR lado: a máscara de uma pessoa é alta e
    # estreita, e um padding por eixo daria quase nada de folga lateral —
    # o recorte cortaria os braços rente ao corpo.
    p = pad * max(w, h)
    x0 -= p; x1 += p; y0 -= p; y1 += p
    w, h = x1 - x0, y1 - y0

    if w / h < ar:
        nw = h * ar
        cx = (x0 + x1) / 2
        x0, x1 = cx - nw / 2, cx + nw / 2
    else:
        nh = w / ar
        cy = (y0 + y1) / 2
        y0, y1 = cy - nh / 2, cy + nh / 2

    w, h = x1 - x0, y1 - y0
    if w <= W:
        if x0 < 0:
            x1 -= x0; x0 = 0.0
        if x1 > W:
            x0 -= (x1 - W); x1 = float(W)
    if h <= H:
        if y0 < 0:
            y1 -= y0; y0 = 0.0
        if y1 > H:
            y0 -= (y1 - H); y1 = float(H)

    return int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))


def _extrai(arr, box):
    """Recorta arr ([H,W] ou [H,W,C]) na caixa, replicando a borda no que
    estiver fora da imagem — assim o recorte sempre sai no tamanho pedido."""
    x0, y0, x1, y1 = box
    H, W = arr.shape[:2]
    ax0, ay0 = max(0, x0), max(0, y0)
    ax1, ay1 = min(W, x1), min(H, y1)
    sub = arr[ay0:ay1, ax0:ax1]
    pt, pb, pl, pr = ay0 - y0, y1 - ay1, ax0 - x0, x1 - ax1
    if pt or pb or pl or pr:
        padw = [(pt, pb), (pl, pr)] + [(0, 0)] * (arr.ndim - 2)
        sub = np.pad(sub, padw, mode="edge")
    return sub


_CORES = np.array([
    [1.0, 0.30, 0.20], [0.20, 0.80, 1.0], [0.40, 1.0, 0.45],
    [1.0, 0.85, 0.20], [0.90, 0.40, 1.0], [0.35, 0.60, 1.0],
], dtype=np.float32)


_KERN = {}


def _borda(m, crescer, feather):
    """Engorda + desfoca a máscara antes de colar.

    Usa cv2 quando existe. O caminho PIL antigo fazia `MaxFilter(5)` em laço —
    uma dilatação de 8px vira 4 passadas de rank filter por máscara, e num vídeo
    são pessoas x frames máscaras. O `cv2.dilate` faz o mesmo numa chamada.
    O PIL fica de reserva pra não depender do cv2 estar instalado.
    """
    if crescer <= 0 and feather <= 0:
        return m
    try:
        import cv2
    except ImportError:
        p = PILImage.fromarray((np.clip(m, 0, 1) * 255).astype(np.uint8), mode="L")
        for _ in range(max(0, int(round(crescer / 2.0)))):
            p = p.filter(ImageFilter.MaxFilter(5))
        if feather > 0:
            p = p.filter(ImageFilter.GaussianBlur(radius=feather / 2.0))
        return np.asarray(p, dtype=np.float32) / 255.0
    x = np.ascontiguousarray(m, dtype=np.float32)
    if crescer > 0:
        k = _KERN.get(crescer)
        if k is None:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                          (crescer * 2 + 1, crescer * 2 + 1))
            _KERN[crescer] = k
        x = cv2.dilate(x, k)
    if feather > 0:
        ks = int(feather) | 1
        x = cv2.GaussianBlur(x, (ks, ks), feather / 2.0)
    return x


def _desenha_cores(img, caixas_com_id, esp=3):
    """Igual ao _desenha, mas a cor vem do ID da pessoa — assim dá pra ver no
    preview se a caixa trocou de pessoa no meio do vídeo."""
    o = img.copy()
    H, W = o.shape[:2]
    for (x0, y0, x1, y1), pid in caixas_com_id:
        c = _CORES[pid % len(_CORES)]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W, x1), min(H, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        e = min(esp, (y1 - y0) // 2, (x1 - x0) // 2) or 1
        o[y0:y0 + e, x0:x1] = c
        o[y1 - e:y1, x0:x1] = c
        o[y0:y1, x0:x0 + e] = c
        o[y0:y1, x1 - e:x1] = c
    return o


def _caixa_movel(bbs, ok, F, ar, padding, modo, suav, margem, W, H):
    """Dada a bbox por frame, devolve (bw, bh, cx[], cy[]) — TAMANHO constante,
    centro suavizado. Extraído do PersonTrack pra o node de vídeo usar o mesmo
    comportamento já testado, em vez de uma segunda implementação parecida."""
    if modo == "fixa (uniao)":
        ux0 = min(bbs[i][0] for i in ok); uy0 = min(bbs[i][1] for i in ok)
        ux1 = max(bbs[i][2] for i in ok); uy1 = max(bbs[i][3] for i in ok)
        bw0, bh0 = ux1 - ux0, uy1 - uy0
        fixo = ((ux0 + ux1) / 2.0, (uy0 + uy1) / 2.0)
    else:
        bw0 = max(bbs[i][2] - bbs[i][0] for i in ok)
        bh0 = max(bbs[i][3] - bbs[i][1] for i in ok)
        fixo = None

    p = padding * max(bw0, bh0)
    cw, ch = bw0 + 2 * p, bh0 + 2 * p
    if cw / ch < ar:
        cw = ch * ar
    else:
        ch = cw / ar
    enc = min(1.0, W / cw, H / ch)
    cw, ch = cw * enc, ch * enc
    bw, bh = max(8, int(round(cw))), max(8, int(round(ch)))

    idx = np.array(ok, dtype=np.float64)
    cxv = np.array([(bbs[i][0] + bbs[i][2]) / 2.0 for i in ok])
    cyv = np.array([(bbs[i][1] + bbs[i][3]) / 2.0 for i in ok])
    t = np.arange(F, dtype=np.float64)
    cx = np.interp(t, idx, cxv)
    cy = np.interp(t, idx, cyv)
    repos = -1

    if modo == "fixa (uniao)":
        cx[:] = fixo[0]; cy[:] = fixo[1]
        repos = 0
    elif modo == "travada por trechos":
        mgx, mgy = margem * bw, margem * bh
        lx, ly = cx[0], cy[0]
        nx, ny = np.empty(F), np.empty(F)
        repos = 0
        for i in range(F):
            b = bbs[i]
            if b is not None:
                if (b[0] < lx - bw / 2 + mgx or b[2] > lx + bw / 2 - mgx or
                        b[1] < ly - bh / 2 + mgy or b[3] > ly + bh / 2 - mgy):
                    lx, ly = cx[i], cy[i]
                    repos += 1
            nx[i], ny[i] = lx, ly
        cx, cy = nx, ny

    if modo != "fixa (uniao)":
        cx = _suaviza(cx, suav)
        cy = _suaviza(cy, suav)
    if bw <= W:
        cx = np.clip(cx, bw / 2.0, W - bw / 2.0)
    if bh <= H:
        cy = np.clip(cy, bh / 2.0, H - bh / 2.0)
    return bw, bh, cx, cy, repos


def _suaviza(v, janela):
    """Média móvel centrada, com as pontas seguradas (sem atraso)."""
    if janela <= 1 or len(v) < 3:
        return v
    j = int(min(janela, len(v)))
    if j % 2 == 0:
        j += 1
    pad = j // 2
    vp = np.pad(v, (pad, pad), mode="edge")
    k = np.ones(j, dtype=np.float64) / j
    return np.convolve(vp, k, mode="valid")


def _trilhas(ms, F, limite, dist_max, sumir_max):
    """Segue cada pessoa pelos frames: por frame, quebra a máscara em manchas e
    liga cada mancha à trilha cujo último centro está mais perto.

    Por que dá pra fazer assim, sem um obj_id do SAM3 por pessoa: dentro de um
    trecho curto (~1s) ninguém troca de lugar com ninguém. A suposição quebra
    exatamente quando duas pessoas se cruzam coladas na frente da câmera — aí a
    caixa pode pular de uma pra outra. Em vídeo longo, use um SAM3VideoOutput
    por obj_id, que é exato.
    """
    trilhas = []
    for f in range(F):
        cands = [d for d in _blobs(ms[f]) if d["area"] >= limite]
        centros = [((d["box"][0] + d["box"][2]) / 2.0,
                    (d["box"][1] + d["box"][3]) / 2.0) for d in cands]
        pares = []
        for ci, c in enumerate(centros):
            for ti, t in enumerate(trilhas):
                if f - t["ultimo"] > sumir_max:
                    continue
                dd = np.hypot(c[0] - t["cx"], c[1] - t["cy"])
                if dd <= dist_max:
                    pares.append((dd, ci, ti))
        pares.sort()
        usados_c, usados_t = set(), set()
        for dd, ci, ti in pares:
            if ci in usados_c or ti in usados_t:
                continue
            usados_c.add(ci); usados_t.add(ti)
            t = trilhas[ti]
            t["bbs"][f] = cands[ci]["box"]
            t["mask"][f] = cands[ci]
            t["cx"], t["cy"] = centros[ci]
            t["ultimo"] = f
        for ci, c in enumerate(centros):
            if ci in usados_c:
                continue
            trilhas.append({"bbs": {f: cands[ci]["box"]}, "mask": {f: cands[ci]},
                            "cx": c[0], "cy": c[1], "ultimo": f, "nasceu": f})
    return trilhas


class SoussaNeoPersonTilesVideo:
    """O PersonTiles, mas lendo VÍDEO: separa as pessoas como no frame parado e
    faz a caixa de cada uma ACOMPANHAR ela pelos frames.

    Sai como LISTA — uma sequência de recortes por pessoa. O ComfyUI executa o
    que vier depois (WAN) uma vez por pessoa sozinho, sem duplicar o grafo.

    Uma máscara só do SAM3 (obj_id = -1, todas as pessoas) basta: a separação
    por pessoa é feita aqui, por manchas + proximidade entre frames.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Os frames do vídeo (ou de um segmento)."}),
                "masks": ("MASK", {"tooltip": "A máscara do vídeo inteiro, um frame por frame. "
                                              "SAM3VideoOutput com obj_id = -1 pega TODAS as "
                                              "pessoas de uma vez — a separação é feita aqui."}),
                "largura": ("INT", {"default": 864, "min": 64, "max": 2048, "step": 16,
                                    "tooltip": "Tamanho do recorte por pessoa. Para VÍDEO o teto "
                                               "prático é 864 (limite do WAN) — acima disso o node "
                                               "avisa. Para IMAGEM (1 frame) pode subir: 1024 é o "
                                               "ponto doce do Flux. Passo 16."}),
                "altura": ("INT", {"default": 864, "min": 64, "max": 2048, "step": 16}),
                "padding": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.5, "step": 0.01,
                                      "tooltip": "Folga em volta da pessoa. Em vídeo convém mais "
                                                 "(0.25-0.4): ela se mexe dentro da caixa."}),
                "modo_caixa": (["travada por trechos", "segue suave", "fixa (uniao)"],
                               {"default": "travada por trechos"}),
                "suavizacao": ("INT", {"default": 9, "min": 1, "max": 121,
                                       "tooltip": "Janela da média móvel do centro, em frames."}),
                "min_area": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 50.0, "step": 0.01,
                                       "tooltip": "Ignora mancha menor que esta % do quadro. "
                                                  "Segura ruído da máscara e figurante minúsculo, "
                                                  "que só gastaria uma passada inteira do WAN."}),
                "min_presenca": ("FLOAT", {"default": 25.0, "min": 0.0, "max": 100.0, "step": 1.0,
                                           "tooltip": "Descarta a pessoa que aparece em menos de "
                                                      "X% dos frames. É o filtro que mata trilha "
                                                      "falsa criada por piscada da máscara."}),
                "max_pessoas": ("INT", {"default": 0, "min": 0, "max": 64,
                                        "tooltip": "0 = todas. Se puser número, fica com as que "
                                                   "aparecem em mais frames (as mais confiáveis)."}),
            },
            "optional": {
                "pontos": ("SAM3_POINTS_PROMPT",
                           {"tooltip": "OPCIONAL. Ligue o SAM3PointCollector aqui e clique "
                                       "nas pessoas que voce quer.\n\n"
                                       "Desligado: recorta TODAS (como sempre fez).\n"
                                       "Ligado: recorta so quem tem um clique em cima.\n\n"
                                       "O clique vale pelo PIXEL: pode cair em qualquer parte "
                                       "da pessoa."}),
                "pontos_fora": ("SAM3_POINTS_PROMPT",
                                {"tooltip": "OPCIONAL. Quem receber um clique aqui sai fora, "
                                            "mesmo que tenha sido escolhido acima. Use o "
                                            "clique NEGATIVO do SAM3PointCollector."}),
                "margem": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 0.45, "step": 0.01}),
                "usar_pontos": ("BOOLEAN", {"default": True,
                                "label_on": "respeita os pontos",
                                "label_off": "ignora os pontos (pega todas)",
                                "tooltip": 'Desligado, as entradas `pontos` e `pontos_fora` sao ignoradas mesmo ligadas, e o node recorta TODAS as pessoas.\n\nServe pra alternar entre automatico e escolhido sem desfazer a fiacao.'}),
                "tipo": (["imagens separadas", "video (rastreia entre frames)"],
                         {"default": "imagens separadas",
                          "tooltip": "O QUE entrou. Não dá pra deduzir: um lote de 3 imagens e um "
                                     "vídeo de 3 frames têm exatamente a mesma forma.\n\n"
                                     "'imagens separadas' — cada imagem tem as SUAS pessoas, sem "
                                     "relação entre elas. Entram 3 imagens, saem os recortes de "
                                     "cada uma, e o merge devolve as 3 tratadas.\n\n"
                                     "'video' — os frames são do mesmo plano e a caixa ACOMPANHA "
                                     "cada pessoa ao longo deles.\n\n"
                                     "Com 1 frame só os dois dão no mesmo."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "SOUSSA_PERSON_TRACKS", "IMAGE", "INT")
    RETURN_NAMES = ("crops", "crop_masks", "tracks", "preview", "quantidade")
    OUTPUT_IS_LIST = (True, True, False, False, False)
    FUNCTION = "run"
    CATEGORY = "Soussa/Pessoa"

    def run(self, images, masks, largura, altura, padding, modo_caixa, suavizacao,
            min_area, min_presenca, max_pessoas, margem=0.12,
            tipo="imagens separadas", pontos=None, pontos_fora=None,
            usar_pontos=True):
        # a chave vem antes de tudo: desligada, os pontos nem sao lidos
        if not usar_pontos:
            pontos = pontos_fora = None
        largura = min(2048, max(16, int(largura) // 16 * 16))
        altura = min(2048, max(16, int(altura) // 16 * 16))
        imgs = _img_np(images)
        F, H, W = imgs.shape[0], imgs.shape[1], imgs.shape[2]
        ms = _mask_np(masks)
        if ms.shape[1:] != (H, W):
            ms = np.stack([_resize_gray(m, W, H) for m in ms])

        # UM frame na entrada = modo IMAGEM: aí a MASK não são frames, são
        # PESSOAS (é o que o SAM3Grounding devolve). Com F frames, a MASK é um
        # frame por frame (SAM3VideoOutput). A regra desfaz a ambiguidade sem
        # precisar de um seletor pro usuário errar.
        # UM frame = imagem única (a MASK são pessoas). VÁRIOS frames podem ser
        # um LOTE de imagens independentes ou um VÍDEO — isso o `tipo` decide,
        # porque as duas coisas têm a mesma forma e adivinhar erra em silêncio:
        # em modo vídeo, um lote de imagens faz a caixa "rastrear" uma pessoa
        # entre fotos que não têm nada a ver uma com a outra.
        modo_img = F == 1
        lote = (not modo_img) and tipo == "imagens separadas"
        if modo_img:
            print(f"[Soussa_Neo_PersonTiles] 1 frame -> modo IMAGEM "
                  f"({len(ms)} mascara(s) = pessoas)")
        elif lote:
            if len(ms) == 1:
                ms = np.repeat(ms, F, axis=0)
            if len(ms) != F:
                raise ValueError(
                    f"[Soussa_Neo_PersonTiles] chegaram {F} frames e apenas {len(ms)} mascara(s).\n"
                    f"  {len(ms)} mascara(s) e a cara do 'SAM3 Text Segmentation' (SAM3Grounding): "
                    f"ele le SO O PRIMEIRO frame e devolve UMA MASCARA POR PESSOA. Em video "
                    f"precisa ser UMA POR FRAME.\n"
                    f"  COMO ARRUMAR: troque por SAM3VideoSegmentation -> SAM3Propagate -> "
                    f"SAM3VideoOutput (obj_id = -1).\n"
                    f"  Se voce corta o video antes com o Soussa_Neo_VideoChunks, ligue a mascara "
                    f"na entrada 'masks' DELE e traga a saida 'masks_blocos' pra ca — assim cada "
                    f"bloco vem com a mascara do trecho certo.")
            print(f"[Soussa_Neo_PersonTiles] {F} IMAGENS SEPARADAS — cada uma com as suas "
                  f"pessoas (sem rastreio entre elas)")
        else:
            if len(ms) == 1:
                ms = np.repeat(ms, F, axis=0)
            if len(ms) != F:
                raise ValueError(
                    f"[Soussa_Neo_PersonTiles] {F} frame(s) de video mas {len(ms)} mascara(s). "
                    f"Em video a mascara tem que ser uma por frame — ligue a saida 'masks' do "
                    f"SAM3VideoOutput, nao a do SAM3Grounding.")
            if max(largura, altura) > 864:
                print(f"[Soussa_Neo_PersonTiles] AVISO: recorte {largura}x{altura} em VIDEO. "
                      f"O WAN nao passa de 864 — se o destino for ele, baixe.")

        ar = largura / float(altura)
        limite = (min_area / 100.0) * (W * H)
        if lote:
            # cada imagem e um mundo: as pessoas dela nascem e morrem nela.
            # `frame` amarra a pessoa a sua imagem — e o merge cola so ali.
            vivos = []
            for i in range(F):
                for d in _blobs(ms[i]):
                    if d["area"] >= limite:
                        vivos.append({"bbs": {i: d["box"]}, "mask": {i: d}, "frame": i})
            vivos = _filtra_pontos(vivos, _pontos_px(pontos, W, H),
                                   _pontos_px(pontos_fora, W, H), " (lote)")

            porim = {}
            for v in vivos:
                porim[v["frame"]] = porim.get(v["frame"], 0) + 1
            if max_pessoas > 0:
                sel = []
                for i in range(F):
                    da = [v for v in vivos if v["frame"] == i]
                    da.sort(key=lambda v: -(v["mask"][i]["area"]))
                    sel += da[:max_pessoas]
                vivos = sel
            if not vivos:
                raise ValueError(
                    f"[Soussa_Neo_PersonTiles] nenhuma pessoa em nenhuma das {F} imagens "
                    f"(min_area={min_area}%). Baixe o min_area ou confira a mascara.")
            custo = len(vivos) * altura * largura * 3 * 4 / 1024**3
            if custo > 4.0:
                print(f"[Soussa_Neo_PersonTiles] AVISO: {len(vivos)} recortes a "
                      f"{largura}x{altura} = ~{custo:.1f} GB. Reduza o lote ou use max_pessoas.")
            crops, cmasks, pessoas, caixas = [], [], [], [[] for _ in range(F)]
            for p, v in enumerate(vivos):
                i = v["frame"]
                b = v["bbs"][i]
                box = _ajusta_caixa(b, ar, padding, W, H)
                d = v["mask"][i]
                mc = _extrai(d["m"], box)
                if d["lab"] is not None:
                    mc = mc * (_extrai(d["lab"], box) == d["idx"])
                crops.append(torch.from_numpy(np.ascontiguousarray(
                    _resize_rgb(_extrai(imgs[i], box), largura, altura)[None])))
                cmasks.append(torch.from_numpy(np.ascontiguousarray(
                    _resize_gray(mc, largura, altura)[None])))
                pessoas.append({"bw": box[2] - box[0], "bh": box[3] - box[1], "frame": i,
                                "quadros": [{"box": box, "valido": True,
                                             "mask": (np.clip(mc, 0, 1) * 255).astype(np.uint8)}]})
                caixas[i].append((box, len(caixas[i])))
            prev = np.stack([_desenha_cores(imgs[i], caixas[i],
                                            esp=max(2, min(W, H) // 300)) for i in range(F)])
            print(f"[Soussa_Neo_PersonTiles] {F} imagem(ns) {W}x{H} -> {len(vivos)} recorte(s) | "
                  f"por imagem: {[porim.get(i, 0) for i in range(F)]}")
            return (crops, cmasks,
                    {"largura": W, "altura": H, "total": F, "lote": True, "pessoas": pessoas},
                    torch.from_numpy(np.ascontiguousarray(prev)), len(vivos))

        if modo_img:
            # cada mascara vira uma trilha de 1 frame, ainda quebrada em manchas:
            # e o que faz uma mascara com a turma inteira virar N pessoas.
            trilhas = []
            for m in ms:
                for d in _blobs(m):
                    if d["area"] >= limite:
                        trilhas.append({"bbs": {0: d["box"]}, "mask": {0: d},
                                        "cx": 0, "cy": 0, "ultimo": 0, "nasceu": 0})
        else:
            # limiar de salto: o quanto uma pessoa pode andar entre 2 frames e ainda
            # ser a mesma. 12% da diagonal cobre passo rapido sem trocar de pessoa.
            dist_max = 0.12 * float(np.hypot(W, H))
            # quantos frames uma pessoa pode ficar sem aparecer e ainda ser a MESMA
            # trilha. Curto demais (2) fazia oclusao de 5 frames virar pessoa nova —
            # e duas trilhas pra mesma pessoa viram duas passadas do WAN na mesma
            # regiao, uma apagando a outra na colagem.
            trilhas = _trilhas(ms, F, limite, dist_max, sumir_max=max(8, F // 5))

        vivos = [t for t in trilhas if len(t["bbs"]) >= (min_presenca / 100.0) * F]
        vivos = _filtra_pontos(vivos, _pontos_px(pontos, W, H),
                               _pontos_px(pontos_fora, W, H))
        vivos.sort(key=lambda t: -len(t["bbs"]))
        if max_pessoas > 0:
            vivos = vivos[:max_pessoas]
        if not vivos:
            raise ValueError(
                f"[Soussa_Neo_PersonTiles] nenhuma pessoa sobrou. {len(trilhas)} trilha(s) "
                f"encontrada(s), todas abaixo de min_presenca={min_presenca}% ou de "
                f"min_area={min_area}%. Baixe os dois e rode de novo.")

        custo = len(vivos) * F * altura * largura * 3 * 4 / 1024**3
        if custo > 4.0:
            print(f"[Soussa_Neo_PersonTiles] AVISO: {len(vivos)} pessoa(s) x {F} frame(s) a "
                  f"{largura}x{altura} = ~{custo:.1f} GB so nos recortes. Rode por SEGMENTO em vez "
                  f"do video inteiro, ou baixe max_pessoas.")

        crops, cmasks, pessoas, caixas_por_frame = [], [], [], [[] for _ in range(F)]
        for p, t in enumerate(vivos):
            bbs = [t["bbs"].get(f) for f in range(F)]
            ok = [f for f in range(F) if bbs[f] is not None]
            bw, bh, cx, cy, _ = _caixa_movel(bbs, ok, F, ar, padding, modo_caixa,
                                             suavizacao, margem, W, H)
            cs, cm, quadros = [], [], []
            for f in range(F):
                x0 = int(round(cx[f] - bw / 2.0)); y0 = int(round(cy[f] - bh / 2.0))
                box = (x0, y0, x0 + bw, y0 + bh)
                d = t["mask"].get(f)
                if d is None:
                    mc = np.zeros((bh, bw), np.float32)
                else:
                    mc = _extrai(d["m"], box)
                    if d["lab"] is not None:
                        mc = mc * (_extrai(d["lab"], box) == d["idx"])
                cs.append(_resize_rgb(_extrai(imgs[f], box), largura, altura))
                cm.append(_resize_gray(mc, largura, altura))
                quadros.append({"box": box, "valido": bbs[f] is not None,
                                "mask": (np.clip(mc, 0, 1) * 255).astype(np.uint8)})
                caixas_por_frame[f].append((box, p))
            crops.append(torch.from_numpy(np.ascontiguousarray(np.stack(cs))))
            cmasks.append(torch.from_numpy(np.ascontiguousarray(np.stack(cm))))
            pessoas.append({"bw": bw, "bh": bh, "quadros": quadros})
            print(f"[Soussa_Neo_PersonTiles]   pessoa {p + 1}: caixa {bw}x{bh} -> "
                  f"{largura}x{altura} | presente em {len(ok)}/{F} frame(s)")

        prev = np.stack([_desenha_cores(imgs[f], caixas_por_frame[f],
                                        esp=max(2, min(W, H) // 300)) for f in range(F)])
        print(f"[Soussa_Neo_PersonTiles] {F} frame(s) {W}x{H} | {len(trilhas)} trilha(s) "
              f"-> {len(vivos)} pessoa(s) (min_presenca={min_presenca}% min_area={min_area}%) | "
              f"cada uma vira uma passada do que vier depois")
        return (crops, cmasks,
                {"largura": W, "altura": H, "total": F, "pessoas": pessoas},
                torch.from_numpy(np.ascontiguousarray(prev)), len(vivos))


class SoussaNeoPersonTilesVideoMerge:
    """Cola as pessoas tratadas de volta no vídeo, cada uma pela própria máscara."""

    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "O vídeo ORIGINAL inteiro."}),
                "crops": ("IMAGE", {"tooltip": "Os recortes tratados, um por pessoa "
                                               "(chega como lista automaticamente)."}),
                "tracks": ("SOUSSA_PERSON_TRACKS",),
                "crescer": ("INT", {"default": 8, "min": 0, "max": 128}),
                "feather": ("INT", {"default": 20, "min": 0, "max": 256}),
                "forca": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "mascara_colada")
    FUNCTION = "run"
    CATEGORY = "Soussa/Pessoa"

    def run(self, images, crops, tracks, crescer, feather, forca):
        um = lambda v: v[0] if isinstance(v, list) else v
        tk = um(tracks)
        crescer, feather, forca = um(crescer), um(feather), um(forca)
        lista = crops if isinstance(crops, list) else [crops]

        # `images` pode chegar como a LISTA DE BLOCOS do VideoChunks em vez do
        # video montado. Juntar aqui e melhor do que mandar o Soussa trocar o
        # cabo: `blocos` e a saida obvia pra ligar, e pegar so o primeiro (33 de
        # 99 frames) e um erro silencioso esperando pra acontecer. So junta se a
        # conta FECHA com o `tracks` — senao seria adivinhacao.
        ims = [x for x in (images if isinstance(images, list) else [images]) if x is not None]
        base = _img_np(ims[0]) if ims else np.zeros((1, 8, 8, 3), np.float32)
        if len(ims) > 1:
            partes = [_img_np(x) for x in ims]
            soma = sum(len(p) for p in partes)
            mesmo = len({p.shape[1:] for p in partes}) == 1
            if mesmo and soma == tk.get("total") and partes[0].shape[1:3] == (tk.get("altura"),
                                                                             tk.get("largura")):
                base = np.concatenate(partes, axis=0)
                print(f"[Soussa_Neo_PersonTilesMerge] chegaram {len(partes)} blocos "
                      f"({[len(p) for p in partes]} frames) em vez do video — juntei nos {soma} "
                      f"frames que o `tracks` espera. (Ligar o `video_base` do TilesToChunks da no "
                      f"mesmo e e mais direto.)")
        F, H, W = base.shape[0], base.shape[1], base.shape[2]
        pessoas = tk.get("pessoas", [])

        tW, tH, tF = tk.get("largura"), tk.get("altura"), tk.get("total")
        if (tW, tH, tF) != (W, H, F):
            # Diagnostico em vez de "ligue o video certo": o que chega errado aqui
            # e quase sempre uma de DUAS coisas, e da pra saber qual pelos numeros.
            if (tW, tH) == (W, H) and tF and F and tF % F == 0 and tF != F:
                dica = (f"  {F} frame(s) contra {tF} e exatamente {tF // F}x — isso e UM BLOCO do "
                        f"Soussa_Neo_VideoChunks, nao o video todo. `blocos` e LISTA: ligado aqui, "
                        f"o merge pega so o primeiro.\n"
                        f"  LIGUE a saida `video_base` do Soussa_Neo_TilesToChunks em `images`. "
                        f"Ela e a concatenacao dos {tF // F} blocos, que e contra o que o `tracks` "
                        f"foi medido.")
            elif (tW, tH) != (W, H) and W == H:
                dica = (f"  {W}x{H} quadrado e do tamanho do RECORTE, nao do video: parece que "
                        f"chegou aqui um crop em vez do video.\n"
                        f"  Em `images` vai o video INTEIRO (`video_base` do TilesToChunks, ou o "
                        f"mesmo video que entrou no PersonTiles). Os recortes tratados vao em "
                        f"`crops`.")
            else:
                dica = ("  Em `images` tem que entrar o MESMO video (mesma resolucao e mesmo "
                        "numero de frames) contra o qual o `tracks` foi gerado.")
            raise ValueError(
                f"[Soussa_Neo_PersonTilesMerge] o video aqui e {W}x{H} com {F} frame(s), mas "
                f"os tracks sao de {tW}x{tH} com {tF}.\n" + dica)
        # Em IMAGEM (1 frame) as K pessoas cabem tanto numa LISTA de K itens
        # quanto num BATCH de K. Aceitar os dois evita quebrar quem tem um
        # ImageListToImageBatch no caminho — que é o arranjo dos workflows
        # antigos, de quando a saída deste node ainda era batch.
        if len(lista) == 1 and len(pessoas) > 1 and F == 1:
            unico = _img_np(lista[0])
            if len(unico) == len(pessoas):
                lista = [torch.from_numpy(np.ascontiguousarray(unico[i:i + 1]))
                         for i in range(len(pessoas))]
                print(f"[Soussa_Neo_PersonTilesMerge] recebi 1 batch de {len(pessoas)} em vez de "
                      f"lista — desmontei (tem um ImageListToImageBatch no caminho; pode tirar).")
        # Contagem diferente NAO derruba o job: o numero de pessoas muda de
        # imagem pra imagem, e parar no meio de um lote e pior que colar o que
        # deu. Cola o que da, avisa alto, e as pessoas sem recorte ficam com o
        # original (que e o pior caso aceitavel: nao piora nada).
        n = min(len(lista), len(pessoas))
        if len(lista) != len(pessoas):
            print(f"[Soussa_Neo_PersonTilesMerge] AVISO: {len(lista)} recorte(s) para "
                  f"{len(pessoas)} pessoa(s) — colando {n} e deixando o resto original.")
            if len(lista) < len(pessoas):
                print(f"[Soussa_Neo_PersonTilesMerge] AVISO: faltou recorte. Se houver um "
                      f"ImageListToImageBatch entre o gerador e este merge, ele junta a lista "
                      f"num batch so — tire e ligue direto.")
        lista, pessoas = lista[:n], pessoas[:n]
        if n == 0:
            print("[Soussa_Neo_PersonTilesMerge] AVISO: nada a colar, video devolvido intacto.")
            return (torch.from_numpy(np.ascontiguousarray(base)),
                    torch.from_numpy(np.zeros((F, H, W), np.float32)))

        out = base.copy()
        acc = np.zeros((F, H, W), np.float32)
        for c_t, pess in zip(lista, pessoas):
            g = _img_np(c_t)
            if len(g) > F:
                g = g[:F]
            # em LOTE cada pessoa pertence a UMA imagem: cola so nela, senao a
            # pessoa da imagem 1 apareceria carimbada nas imagens 2 e 3
            fr = pess.get("frame")
            if fr is not None:
                pares = [(fr, pess["quadros"][0])]
            else:
                # `desloc` = em que frame do vídeo o recorte COMEÇA. Quem corta o
                # vídeo em blocos antes (TilesToChunks) devolve pessoas que só
                # existem no seu pedaço: sem isso o bloco 2 seria colado no
                # começo do vídeo. Sem o campo, 0 — que é o comportamento antigo.
                d0 = int(pess.get("desloc", 0))
                if len(g) < len(pess["quadros"]) - d0:
                    print(f"[Soussa_Neo_PersonTilesMerge] AVISO: uma pessoa voltou com {len(g)} "
                          f"frame(s) de {len(pess['quadros']) - d0} — colando so esses.")
                pares = [(d0 + k, q) for k, q in enumerate(pess["quadros"][d0:d0 + len(g)])]
            for k, (f, q) in enumerate(pares):
                gi = 0 if fr is not None else k
                if not q["valido"]:
                    continue
                x0, y0, x1, y1 = q["box"]
                bw, bh = x1 - x0, y1 - y0
                m = q["mask"].astype(np.float32) / 255.0
                if m.shape != (bh, bw):
                    m = _resize_gray(m, bw, bh)
                m = _borda(m, crescer, feather)
                if forca < 1.0:
                    m = m * forca
                # em lote o recorte tratado tem 1 frame (gi=0); em video, o frame f
                ci = g[gi] if gi < len(g) else g[-1]
                if ci.shape[:2] != (bh, bw):
                    ci = _resize_rgb(ci, bw, bh)
                ax0, ay0 = max(0, x0), max(0, y0)
                ax1, ay1 = min(W, x1), min(H, y1)
                if ax1 <= ax0 or ay1 <= ay0:
                    continue
                sx, sy = ax0 - x0, ay0 - y0
                sc = ci[sy:sy + (ay1 - ay0), sx:sx + (ax1 - ax0)]
                sm = m[sy:sy + (ay1 - ay0), sx:sx + (ax1 - ax0)][..., None]
                out[f, ay0:ay1, ax0:ax1] = out[f, ay0:ay1, ax0:ax1] * (1 - sm) + sc * sm
                acc[f, ay0:ay1, ax0:ax1] = np.maximum(acc[f, ay0:ay1, ax0:ax1], sm[..., 0])

        print(f"[Soussa_Neo_PersonTilesMerge] {len(lista)} pessoa(s) coladas em {F} frame(s) "
              f"{W}x{H} | crescer={crescer} feather={feather} forca={forca} | "
              f"{float(acc.mean()) * 100:.1f}% do video tocado")
        return (torch.from_numpy(np.ascontiguousarray(np.clip(out, 0, 1))),
                torch.from_numpy(np.ascontiguousarray(acc)))


class SoulPersonTiles:
    """Recorta cada pessoa. Cinco ajustes, nada mais.

    E a casca enxuta de SoussaNeoPersonTilesVideo: os parametros que so valem
    para VIDEO (modo_caixa, suavizacao, min_presenca) ficam no padrao e somem
    da tela. Quem for tratar video usa SoulPersonTilesAvancado, que expoe tudo.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "A foto."}),
                "masks": ("MASK", {"tooltip": "As pessoas que o SAM3 achou."}),
                "tamanho": ("INT", {"default": 496, "min": 64, "max": 2048, "step": 16,
                                    "tooltip": "Lado do recorte, em pixel. E o tamanho "
                                               "que o modelo de enhance vai receber: "
                                               "maior e melhor e mais lento."}),
                "padding": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 1.0, "step": 0.05,
                                      "tooltip": "Folga em volta da pessoa. 0.30 = 30%, "
                                                 "o bastante para nao cortar cabelo e pe."}),
                "min_area": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 50.0, "step": 0.01,
                                       "tooltip": "Ignora mancha menor que esta % do quadro. "
                                                  "Mata ruido de mascara e figurante de 10 "
                                                  "pixels, que gastaria uma passada inteira."}),
                "max_pessoas": ("INT", {"default": 0, "min": 0, "max": 64,
                                        "tooltip": "0 = todas. Qualquer outro numero fica "
                                                   "com as N maiores."}),
            },
            "optional": {
                "pontos": ("SAM3_POINTS_PROMPT",
                           {"tooltip": "OPCIONAL. Ligue o SAM3PointCollector e clique nas "
                                       "pessoas que voce quer. Vazio = todas."}),
                "pontos_fora": ("SAM3_POINTS_PROMPT",
                                {"tooltip": "OPCIONAL. Quem receber clique aqui sai fora."}),
                "usar_pontos": ("BOOLEAN", {"default": True,
                                            "label_on": "respeita os pontos",
                                            "label_off": "ignora (pega todas)"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "SOUSSA_PERSON_TRACKS", "IMAGE", "INT")
    RETURN_NAMES = ("crops", "crop_masks", "tracks", "preview", "quantidade")
    OUTPUT_IS_LIST = (True, True, False, False, False)
    FUNCTION = "run"
    CATEGORY = "Soul/Pessoa"

    def run(self, images, masks, tamanho, padding, min_area, max_pessoas,
            pontos=None, pontos_fora=None, usar_pontos=True):
        return SoussaNeoPersonTilesVideo().run(
            images=images, masks=masks, largura=tamanho, altura=tamanho,
            padding=padding, modo_caixa="travada por trechos", suavizacao=9,
            min_area=min_area, min_presenca=15, max_pessoas=max_pessoas,
            margem=0.12, tipo="imagens separadas",
            pontos=pontos, pontos_fora=pontos_fora, usar_pontos=usar_pontos)


NODE_CLASS_MAPPINGS = {
    "SoulPersonTiles":          SoulPersonTiles,
    "SoulPersonTilesAvancado":  SoussaNeoPersonTilesVideo,
    "SoulPersonTilesMerge":     SoussaNeoPersonTilesVideoMerge,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SoulPersonTiles":          "Soul Person Tiles (recorta cada pessoa)",
    "SoulPersonTilesAvancado":  "Soul Person Tiles - avancado (video)",
    "SoulPersonTilesMerge":     "Soul Person Tiles Merge (cola de volta)",
}

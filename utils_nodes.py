# -*- coding: utf-8 -*-
"""Tres utilitarios que o workflow de exemplo usa.

    SoulImageSizeCheck   mede o lado maior e responde sim/nao contra um limite
    SoulDenoise          reducao de ruido (NLM / bilateral / mediana)
    SoulSwitch           manda a entrada para uma de duas saidas

Nao ha nada de especial aqui — sao as pecas que ligam o ramo de pre-ampliacao
do workflow. Ficam no repo para que ele rode sem precisar de mais nada.

Copyright (C) 2026 Felipe Soussa
Licenca AGPL-3.0-or-later — veja LICENSE.
"""
import cv2
import numpy as np
import torch


class SoussaImageSizeCheck:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":      ("IMAGE",),
                "threshold":  ("INT", {
                    "default": 1024,
                    "min": 1,
                    "max": 99999,
                    "step": 1,
                    "display": "number",
                }),
                "comparar_por": (["lado_maior", "largura", "altura", "total_pixels"],),
            }
        }

    RETURN_TYPES  = ("BOOLEAN", "INT", "INT", "STRING")
    RETURN_NAMES  = ("maior_que_threshold", "valor_medido", "threshold", "resumo")
    FUNCTION      = "check"
    CATEGORY = "Soussa/Check"
    OUTPUT_NODE   = False

    def check(self, image, threshold, comparar_por):
        # image: [N, H, W, C]
        N, H, W = image.shape[0], image.shape[1], image.shape[2]

        if comparar_por == "lado_maior":
            valor = max(W, H)
        elif comparar_por == "largura":
            valor = W
        elif comparar_por == "altura":
            valor = H
        elif comparar_por == "total_pixels":
            valor = W * H
        else:
            valor = max(W, H)

        resultado = valor > threshold

        simbolo = ">" if resultado else "<="
        resumo = (
            f"{W}×{H} | {comparar_por}={valor} "
            f"{simbolo} {threshold} → {'True ✓' if resultado else 'False ✗'}"
        )

        print(f"[Soussa_Image_Size_Check] {resumo}")

        return (resultado, valor, threshold, resumo)


def _u8(img_t):
    a = img_t.detach().cpu().numpy().astype(np.float32)
    if a.ndim == 3:
        a = a[None, ...]
    if a.shape[-1] == 4:
        a = a[..., :3]
    return (np.clip(a, 0.0, 1.0) * 255.0).astype(np.uint8), a.shape

class SoussaNeoDenoise:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Uma imagem ou um batch de frames."}),
                "method": (["nlm", "bilateral", "median", "nlm+bilateral"],
                           {"default": "nlm"}),
                "strength": ("FLOAT", {
                    "default": 6.0, "min": 0.0, "max": 30.0, "step": 0.5,
                    "tooltip": "0 = DESLIGADO (a imagem passa intacta, sem custo). "
                               "nlm: 3=ruído leve, 6=médio, 10+=forte (aí começa a comer textura). "
                               "bilateral/median: vira o raio do filtro."}),
                "preserve_detail": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Mistura a imagem ORIGINAL de volta por cima da limpa. Cuidado: "
                               "isso devolve o detalhe E o ruído juntos — em teste, 0.25 custou "
                               "2 dB de qualidade. Deixe em 0; só suba (0.15-0.30) se o "
                               "resultado ficar plástico demais pro teu gosto."}),
            },
            "optional": {
                "color_strength": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 30.0, "step": 0.5,
                    "tooltip": "Só no nlm: limpeza extra das manchas de COR, separada do "
                               "brilho. 0 = usa o mesmo valor do strength. Suba se o ruído "
                               "for colorido (manchinhas verdes/roxas)."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "Soussa/Imagem"

    def run(self, images, method, strength, preserve_detail, color_strength=0.0):
        # Desligar tem que ser um curto-circuito explícito: passar 0 pros filtros
        # do OpenCV NÃO os torna neutros — o medianBlur, por exemplo, ainda usaria
        # raio 3 e borraria a imagem. Aqui em 0 nada roda e nem o OpenCV é tocado.
        if strength <= 0:
            print("[Soussa_Neo_Denoise] strength=0 — desligado, imagem passa intacta")
            return (images,)

        try:
            import cv2
        except ImportError:
            print("[Soussa_Neo_Denoise] OpenCV não encontrado — imagem devolvida sem alteração")
            return (images,)

        u8, _ = _u8(images)
        N, H, W, _ = u8.shape
        h = float(strength)
        hc = float(color_strength) if color_strength > 0 else h
        # raio ímpar pros filtros de janela
        k = max(3, int(round(h)) | 1)

        out = np.empty_like(u8)
        for i in range(N):
            f = u8[i]
            if method in ("nlm", "nlm+bilateral"):
                f = cv2.fastNlMeansDenoisingColored(f, None, h, hc, 7, 21)
            if method in ("bilateral", "nlm+bilateral"):
                # sigmaColor alto = mistura tons mais distintos; sigmaSpace segue o raio
                f = cv2.bilateralFilter(f, k, h * 12, h * 6)
            if method == "median":
                f = cv2.medianBlur(f, k)
            out[i] = f

        clean = out.astype(np.float32) / 255.0
        orig = u8.astype(np.float32) / 255.0
        if preserve_detail > 0:
            clean = clean * (1.0 - preserve_detail) + orig * preserve_detail

        # o quanto mudou, pra dar noção sem precisar comparar no olho
        delta = float(np.mean(np.abs(clean - orig))) * 255.0
        print(f"[Soussa_Neo_Denoise] {N} frame(s) {W}x{H} | {method} força={h:g} "
              f"| preserva={preserve_detail:.2f} | mudança média {delta:.1f} níveis (0-255)")

        return (torch.from_numpy(np.ascontiguousarray(np.clip(clean, 0.0, 1.0))),)


ANY = AnyType("*")

class SoussaNeoSwitch:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "entrada": (ANY, {"tooltip": "Qualquer tipo: imagem, latente, modelo, texto…"}),
                "condicao": ("BOOLEAN", {
                    "default": True, "label_on": "cima", "label_off": "baixo",
                    "tooltip": "True manda pra saída de cima, False pra de baixo. "
                               "Pode ser ligado num node de lógica."}),
            },
        }

    RETURN_TYPES = (ANY, ANY)
    RETURN_NAMES = ("se_verdadeiro", "se_falso")
    FUNCTION = "run"
    CATEGORY = "Soussa/Logica"
    DESCRIPTION = ("Manda a entrada pra saída de cima quando a condição é verdadeira, "
                   "ou pra de baixo quando é falsa. O ramo não escolhido nem executa.")

    def run(self, entrada, condicao):
        # message=None bloqueia em silêncio (sem erro na tela) — o ramo só não roda
        bloqueado = ExecutionBlocker(None)
        destino = "CIMA (se_verdadeiro)" if condicao else "BAIXO (se_falso)"
        print(f"[Soussa_Neo_Switch] condicao={condicao} -> {destino}; o outro ramo não executa")
        if condicao:
            return (entrada, bloqueado)
        return (bloqueado, entrada)


NODE_CLASS_MAPPINGS = {
    "SoulImageSizeCheck": SoussaImageSizeCheck,
    "SoulDenoise":        SoussaNeoDenoise,
    "SoulSwitch":         SoussaNeoSwitch,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SoulImageSizeCheck": "Soul Image Size Check",
    "SoulDenoise":        "Soul Denoise",
    "SoulSwitch":         "Soul Switch (1 entra, 2 saem)",
}

# -*- coding: utf-8 -*-
"""Soul Person Enhance — per-person cropping and merging back.

Two ComfyUI nodes:

    SoulPersonTiles       turns each person mask (from SAM3) into its own
                          framed, normalized crop, ready to be enhanced
    SoulPersonTilesMerge  puts the enhanced crops back into the original
                          image, each one where it came from, edge feathered

Extracted from the author's personal pack; only the two image nodes live here.

Copyright (C) 2026 Felipe Soussa
License AGPL-3.0-or-later — see LICENSE.
"""
import numpy as np
import torch
from PIL import Image as PILImage, ImageFilter

TAG = "[SoulPersonTiles]"
TAG_M = "[SoulPersonTilesMerge]"


def _img_np(t):
    """IMAGE tensor -> np [N,H,W,3] float 0-1."""
    a = t.detach().cpu().numpy().astype(np.float32)
    if a.ndim == 3:
        a = a[None, ...]
    if a.shape[-1] == 4:
        a = a[..., :3]
    return np.clip(a, 0.0, 1.0)


def _mask_np(t):
    """MASK tensor -> np [N,H,W] float 0-1, also accepting [H,W] and [N,H,W,1]."""
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


def _simple_det(m):
    b = _bbox(m)
    if b is None:
        return None
    return {"m": m, "lab": None, "idx": 0, "box": b, "area": float((m > 0.5).sum())}


def _blobs(m, thr=0.5):
    """Splits ONE mask into its separate blobs (connected components).

    This is what lets the node find people on its own: if the detector hands
    over a single mask with the whole crowd painted in it, every person who
    is not touching another becomes a detection right here. Without this the
    node would be at the mercy of however many masks SAM3 decided to split.

    Keeps the label map (`lab`) instead of materializing one [H,W] mask per
    person: on a 6000x3348 image each of those masks costs 80 MB, and 20
    people would be 1.6 GB for nothing. Each person's mask is cropped inside
    her own box only, at the moment it is used.
    """
    b = (m > thr).astype(np.uint8)
    if not b.any():
        return []
    try:
        import cv2
    except ImportError:
        d = _simple_det(m)
        return [d] if d else []
    n, lab, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    out = []
    for k in range(1, n):
        x, y, w, h, a = stats[k]
        out.append({"m": m, "lab": lab, "idx": int(k),
                    "box": (int(x), int(y), int(x + w), int(y + h)), "area": float(a)})
    return out


def _points_px(p, W, H):
    """SAM3_POINTS_PROMPT -> list of (x, y) in pixels.

    SAM3 hands over {"points": [[x, y], ...], "labels": [...]} with x and y
    NORMALIZED from 0 to 1. Pixel coordinates are accepted too (anything past
    1.5 cannot be normalized), as well as a raw list of pairs, so this does
    not break if the points come from another node.
    """
    if not p:
        return []
    pairs = p.get("points", []) if isinstance(p, dict) else p
    out = []
    for q in pairs:
        try:
            x, y = (q["x"], q["y"]) if isinstance(q, dict) else (q[0], q[1])
        except (TypeError, KeyError, IndexError):
            continue
        x, y = float(x), float(y)
        if x <= 1.5 and y <= 1.5:          # normalized
            x, y = x * W, y * H
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < W and 0 <= yi < H:
            out.append((xi, yi))
    return out


def _has_point(d, pts):
    """Did the click land INSIDE this person? Tested against the label map, not
    the box — two people side by side have overlapping boxes but distinct
    labels."""
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


def _filter_points(alive, include, exclude, where=""):
    """Keeps only whoever has a point on them; drops whoever has a negative one.

    Works for BOTH shapes of `alive` the node builds: the batch one (one person
    per image) and the track one (one person across frames). In both, `v["mask"]`
    is a frame -> blob dict, so looking at every frame of that person is enough.
    """
    if not include and not exclude:
        return alive
    def hits(v, pts):
        return any(_has_point(d, pts) for d in v["mask"].values())
    before = len(alive)
    if include:
        alive = [v for v in alive if hits(v, include)]
    if exclude:
        alive = [v for v in alive if not hits(v, exclude)]
    print("%s points%s: %d marked, %d excluded -> %d of %d person(s)"
          % (TAG, where, len(include), len(exclude), len(alive), before))
    if not alive:
        raise ValueError(
            TAG + " no click landed on a person. Click inside a body (not on the "
            "background), or disconnect the `points` input to crop everyone.")
    return alive


def _fit_box(box, ar, pad, W, H):
    """Inflates the box by the padding, forces the tile aspect ratio and tries
    to push it back inside the image. If it still does not fit, it is allowed to
    overflow — edge padding in _crop_pad() is what deals with that."""
    x0, y0, x1, y1 = map(float, box)
    w, h = x1 - x0, y1 - y0

    # padding proportional to the LONGEST side: a person mask is tall and
    # narrow, and per-axis padding would give almost no side room — the crop
    # would clip the arms right against the body.
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


def _crop_pad(arr, box):
    """Crops arr ([H,W] or [H,W,C]) to the box, replicating the edge for
    whatever falls outside the image — so the crop always comes out at the
    requested size."""
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


_COLORS = np.array([
    [1.0, 0.30, 0.20], [0.20, 0.80, 1.0], [0.40, 1.0, 0.45],
    [1.0, 0.85, 0.20], [0.90, 0.40, 1.0], [0.35, 0.60, 1.0],
], dtype=np.float32)


_KERN = {}


def _edge(m, grow, feather):
    """Dilates + blurs the mask before pasting.

    Uses cv2 when available. The old PIL path ran `MaxFilter(5)` in a loop — an
    8px dilation becomes 4 rank-filter passes per mask, and on video that is
    people x frames masks. `cv2.dilate` does the same in one call. PIL stays as
    a fallback so cv2 is not a hard dependency.
    """
    if grow <= 0 and feather <= 0:
        return m
    try:
        import cv2
    except ImportError:
        p = PILImage.fromarray((np.clip(m, 0, 1) * 255).astype(np.uint8), mode="L")
        for _ in range(max(0, int(round(grow / 2.0)))):
            p = p.filter(ImageFilter.MaxFilter(5))
        if feather > 0:
            p = p.filter(ImageFilter.GaussianBlur(radius=feather / 2.0))
        return np.asarray(p, dtype=np.float32) / 255.0
    x = np.ascontiguousarray(m, dtype=np.float32)
    if grow > 0:
        k = _KERN.get(grow)
        if k is None:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                          (grow * 2 + 1, grow * 2 + 1))
            _KERN[grow] = k
        x = cv2.dilate(x, k)
    if feather > 0:
        ks = int(feather) | 1
        x = cv2.GaussianBlur(x, (ks, ks), feather / 2.0)
    return x


def _draw_boxes(img, boxes_with_id, th=3):
    """Draws the boxes on the preview, colored by person ID — so you can see
    whether a box jumped to a different person mid-video."""
    o = img.copy()
    H, W = o.shape[:2]
    for (x0, y0, x1, y1), pid in boxes_with_id:
        c = _COLORS[pid % len(_COLORS)]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W, x1), min(H, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        e = min(th, (y1 - y0) // 2, (x1 - x0) // 2) or 1
        o[y0:y0 + e, x0:x1] = c
        o[y1 - e:y1, x0:x1] = c
        o[y0:y1, x0:x0 + e] = c
        o[y0:y1, x1 - e:x1] = c
    return o


def _moving_box(bbs, ok, F, ar, padding, mode, smooth, margin, W, H):
    """Given the per-frame bbox, returns (bw, bh, cx[], cy[]) — constant SIZE,
    smoothed center. Extracted from PersonTrack so the video node reuses the
    same tested behavior instead of a second, similar implementation."""
    if mode == "fixed (union)":
        ux0 = min(bbs[i][0] for i in ok); uy0 = min(bbs[i][1] for i in ok)
        ux1 = max(bbs[i][2] for i in ok); uy1 = max(bbs[i][3] for i in ok)
        bw0, bh0 = ux1 - ux0, uy1 - uy0
        fixed = ((ux0 + ux1) / 2.0, (uy0 + uy1) / 2.0)
    else:
        bw0 = max(bbs[i][2] - bbs[i][0] for i in ok)
        bh0 = max(bbs[i][3] - bbs[i][1] for i in ok)
        fixed = None

    p = padding * max(bw0, bh0)
    cw, ch = bw0 + 2 * p, bh0 + 2 * p
    if cw / ch < ar:
        cw = ch * ar
    else:
        ch = cw / ar
    fit = min(1.0, W / cw, H / ch)
    cw, ch = cw * fit, ch * fit
    bw, bh = max(8, int(round(cw))), max(8, int(round(ch)))

    idx = np.array(ok, dtype=np.float64)
    cxv = np.array([(bbs[i][0] + bbs[i][2]) / 2.0 for i in ok])
    cyv = np.array([(bbs[i][1] + bbs[i][3]) / 2.0 for i in ok])
    t = np.arange(F, dtype=np.float64)
    cx = np.interp(t, idx, cxv)
    cy = np.interp(t, idx, cyv)
    moves = -1

    if mode == "fixed (union)":
        cx[:] = fixed[0]; cy[:] = fixed[1]
        moves = 0
    elif mode == "locked in segments":
        mgx, mgy = margin * bw, margin * bh
        lx, ly = cx[0], cy[0]
        nx, ny = np.empty(F), np.empty(F)
        moves = 0
        for i in range(F):
            b = bbs[i]
            if b is not None:
                if (b[0] < lx - bw / 2 + mgx or b[2] > lx + bw / 2 - mgx or
                        b[1] < ly - bh / 2 + mgy or b[3] > ly + bh / 2 - mgy):
                    lx, ly = cx[i], cy[i]
                    moves += 1
            nx[i], ny[i] = lx, ly
        cx, cy = nx, ny

    if mode != "fixed (union)":
        cx = _smooth(cx, smooth)
        cy = _smooth(cy, smooth)
    if bw <= W:
        cx = np.clip(cx, bw / 2.0, W - bw / 2.0)
    if bh <= H:
        cy = np.clip(cy, bh / 2.0, H - bh / 2.0)
    return bw, bh, cx, cy, moves


def _smooth(v, window):
    """Centered moving average, with the ends held (no lag)."""
    if window <= 1 or len(v) < 3:
        return v
    j = int(min(window, len(v)))
    if j % 2 == 0:
        j += 1
    pad = j // 2
    vp = np.pad(v, (pad, pad), mode="edge")
    k = np.ones(j, dtype=np.float64) / j
    return np.convolve(vp, k, mode="valid")


def _track(ms, F, limit, dist_max, max_gap):
    """Follows each person across frames: per frame, splits the mask into blobs
    and links each blob to the track whose last center is closest.

    Why this works without a per-person obj_id from SAM3: within a short span
    (~1s) nobody swaps places with anybody. The assumption breaks exactly when
    two people cross while touching in front of the camera — then the box can
    jump from one to the other. On long video, use a SAM3VideoOutput per obj_id,
    which is exact.
    """
    tracks = []
    for f in range(F):
        cands = [d for d in _blobs(ms[f]) if d["area"] >= limit]
        centers = [((d["box"][0] + d["box"][2]) / 2.0,
                    (d["box"][1] + d["box"][3]) / 2.0) for d in cands]
        pairs = []
        for ci, c in enumerate(centers):
            for ti, t in enumerate(tracks):
                if f - t["last"] > max_gap:
                    continue
                dd = np.hypot(c[0] - t["cx"], c[1] - t["cy"])
                if dd <= dist_max:
                    pairs.append((dd, ci, ti))
        pairs.sort()
        used_c, used_t = set(), set()
        for dd, ci, ti in pairs:
            if ci in used_c or ti in used_t:
                continue
            used_c.add(ci); used_t.add(ti)
            t = tracks[ti]
            t["bbs"][f] = cands[ci]["box"]
            t["mask"][f] = cands[ci]
            t["cx"], t["cy"] = centers[ci]
            t["last"] = f
        for ci, c in enumerate(centers):
            if ci in used_c:
                continue
            tracks.append({"bbs": {f: cands[ci]["box"]}, "mask": {f: cands[ci]},
                           "cx": c[0], "cy": c[1], "last": f, "born": f})
    return tracks


class PersonTilesAdvanced:
    """PersonTiles, but reading VIDEO: it separates people the same way as on a
    still frame, and makes each box FOLLOW its person across the frames.

    Outputs a LIST — one sequence of crops per person. ComfyUI runs whatever
    comes next (WAN) once per person on its own, without duplicating the graph.

    A single SAM3 mask (obj_id = -1, everyone) is enough: the per-person split
    is done here, by blobs + proximity between frames.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The video frames (or those of one segment)."}),
                "masks": ("MASK", {"tooltip": "The mask of the whole video, one per frame. "
                                              "SAM3VideoOutput with obj_id = -1 grabs EVERY "
                                              "person at once — the split is done here."}),
                "width": ("INT", {"default": 864, "min": 64, "max": 2048, "step": 16,
                                  "tooltip": "Crop size per person. For VIDEO the practical "
                                             "ceiling is 864 (WAN's limit) — above that the node "
                                             "warns you. For IMAGE (1 frame) you can go higher: "
                                             "1024 is Flux's sweet spot. Step 16."}),
                "height": ("INT", {"default": 864, "min": 64, "max": 2048, "step": 16}),
                "padding": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.5, "step": 0.01,
                                      "tooltip": "Room around the person. Video wants more "
                                                 "(0.25-0.4): she moves inside the box."}),
                "box_mode": (["locked in segments", "smooth follow", "fixed (union)"],
                             {"default": "locked in segments"}),
                "smoothing": ("INT", {"default": 9, "min": 1, "max": 121,
                                      "tooltip": "Moving-average window for the center, in frames."}),
                "min_area": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 50.0, "step": 0.01,
                                       "tooltip": "Ignores any blob smaller than this % of the "
                                                  "frame. Holds back mask noise and tiny extras, "
                                                  "which would each burn a whole WAN pass."}),
                "min_presence": ("FLOAT", {"default": 25.0, "min": 0.0, "max": 100.0, "step": 1.0,
                                           "tooltip": "Drops a person who shows up in fewer than "
                                                      "X% of the frames. This is the filter that "
                                                      "kills false tracks from a flickering mask."}),
                "max_people": ("INT", {"default": 0, "min": 0, "max": 64,
                                       "tooltip": "0 = everyone. Give it a number and it keeps "
                                                  "those present in the most frames (the most "
                                                  "reliable ones)."}),
            },
            "optional": {
                "points": ("SAM3_POINTS_PROMPT",
                           {"tooltip": "OPTIONAL. Connect SAM3PointCollector here and click "
                                       "the people you want.\n\n"
                                       "Disconnected: crops EVERYONE (as it always did).\n"
                                       "Connected: crops only who has a click on them.\n\n"
                                       "The click counts by PIXEL: it can land anywhere on "
                                       "the person."}),
                "points_exclude": ("SAM3_POINTS_PROMPT",
                                   {"tooltip": "OPTIONAL. Anyone who gets a click here is left "
                                               "out, even if picked above. Use the NEGATIVE "
                                               "click of SAM3PointCollector."}),
                "margin": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 0.45, "step": 0.01}),
                "use_points": ("BOOLEAN", {"default": True,
                               "label_on": "honor the points",
                               "label_off": "ignore the points (take everyone)",
                               "tooltip": "Off, the `points` and `points_exclude` inputs are ignored even while connected, and the node crops EVERY person.\n\nLets you flip between automatic and hand-picked without unwiring anything."}),
                "input_type": (["separate images", "video (track across frames)"],
                               {"default": "separate images",
                                "tooltip": "WHAT came in. It cannot be inferred: a batch of 3 "
                                           "images and a 3-frame video have exactly the same "
                                           "shape.\n\n"
                                           "'separate images' — each image has ITS OWN people, "
                                           "unrelated to the others. 3 images in, the crops of "
                                           "each one out, and the merge returns the 3 treated.\n\n"
                                           "'video' — the frames belong to the same shot and the "
                                           "box FOLLOWS each person across them.\n\n"
                                           "With a single frame the two are the same."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "SOUL_PERSON_TRACKS", "IMAGE", "INT")
    RETURN_NAMES = ("crops", "crop_masks", "tracks", "preview", "count")
    OUTPUT_IS_LIST = (True, True, False, False, False)
    FUNCTION = "run"
    CATEGORY = "Soul/Person"

    def run(self, images, masks, width, height, padding, box_mode, smoothing,
            min_area, min_presence, max_people, margin=0.12,
            input_type="separate images", points=None, points_exclude=None,
            use_points=True):
        # the switch comes before everything: off, the points are not even read
        if not use_points:
            points = points_exclude = None
        width = min(2048, max(16, int(width) // 16 * 16))
        height = min(2048, max(16, int(height) // 16 * 16))
        imgs = _img_np(images)
        F, H, W = imgs.shape[0], imgs.shape[1], imgs.shape[2]
        ms = _mask_np(masks)
        if ms.shape[1:] != (H, W):
            ms = np.stack([_resize_gray(m, W, H) for m in ms])

        # ONE frame in = IMAGE mode: then the MASK is not frames, it is PEOPLE
        # (which is what SAM3Grounding returns). With F frames, the MASK is one
        # per frame (SAM3VideoOutput). This rule removes the ambiguity without
        # a selector the user could get wrong.
        # ONE frame = single image (the MASK is people). SEVERAL frames can be a
        # BATCH of independent images or a VIDEO — `input_type` decides, because
        # both have the same shape and guessing fails silently: in video mode, a
        # batch of images makes the box "track" a person across photos that have
        # nothing to do with each other.
        single_image = F == 1
        batch = (not single_image) and input_type == "separate images"
        if single_image:
            print(f"{TAG} 1 frame -> IMAGE mode ({len(ms)} mask(s) = people)")
        elif batch:
            if len(ms) == 1:
                ms = np.repeat(ms, F, axis=0)
            if len(ms) != F:
                raise ValueError(
                    f"{TAG} {F} frames arrived but only {len(ms)} mask(s).\n"
                    f"  {len(ms)} mask(s) is the signature of 'SAM3 Text Segmentation' "
                    f"(SAM3Grounding): it reads ONLY THE FIRST frame and returns ONE MASK PER "
                    f"PERSON. On video it has to be ONE PER FRAME.\n"
                    f"  HOW TO FIX: swap it for SAM3VideoSegmentation -> SAM3Propagate -> "
                    f"SAM3VideoOutput (obj_id = -1).")
            print(f"{TAG} {F} SEPARATE IMAGES — each with its own people "
                  f"(no tracking between them)")
        else:
            if len(ms) == 1:
                ms = np.repeat(ms, F, axis=0)
            if len(ms) != F:
                raise ValueError(
                    f"{TAG} {F} video frame(s) but {len(ms)} mask(s). On video the mask has to "
                    f"be one per frame — connect the 'masks' output of SAM3VideoOutput, not the "
                    f"one from SAM3Grounding.")
            if max(width, height) > 864:
                print(f"{TAG} WARNING: {width}x{height} crop on VIDEO. WAN does not go past 864 "
                      f"— if that is the destination, bring it down.")

        ar = width / float(height)
        limit = (min_area / 100.0) * (W * H)
        if batch:
            # each image is its own world: its people are born and die in it.
            # `frame` ties the person to her image — and the merge pastes only there.
            alive = []
            for i in range(F):
                for d in _blobs(ms[i]):
                    if d["area"] >= limit:
                        alive.append({"bbs": {i: d["box"]}, "mask": {i: d}, "frame": i})
            alive = _filter_points(alive, _points_px(points, W, H),
                                   _points_px(points_exclude, W, H), " (batch)")

            per_image = {}
            for v in alive:
                per_image[v["frame"]] = per_image.get(v["frame"], 0) + 1
            if max_people > 0:
                sel = []
                for i in range(F):
                    here = [v for v in alive if v["frame"] == i]
                    here.sort(key=lambda v: -(v["mask"][i]["area"]))
                    sel += here[:max_people]
                alive = sel
            if not alive:
                raise ValueError(
                    f"{TAG} no person found in any of the {F} image(s) "
                    f"(min_area={min_area}%). Lower min_area or check the mask.")
            cost = len(alive) * height * width * 3 * 4 / 1024**3
            if cost > 4.0:
                print(f"{TAG} WARNING: {len(alive)} crops at {width}x{height} = ~{cost:.1f} GB. "
                      f"Shrink the batch or use max_people.")
            crops, cmasks, people, boxes = [], [], [], [[] for _ in range(F)]
            for p, v in enumerate(alive):
                i = v["frame"]
                b = v["bbs"][i]
                box = _fit_box(b, ar, padding, W, H)
                d = v["mask"][i]
                mc = _crop_pad(d["m"], box)
                if d["lab"] is not None:
                    mc = mc * (_crop_pad(d["lab"], box) == d["idx"])
                crops.append(torch.from_numpy(np.ascontiguousarray(
                    _resize_rgb(_crop_pad(imgs[i], box), width, height)[None])))
                cmasks.append(torch.from_numpy(np.ascontiguousarray(
                    _resize_gray(mc, width, height)[None])))
                people.append({"bw": box[2] - box[0], "bh": box[3] - box[1], "frame": i,
                               "frames": [{"box": box, "valid": True,
                                           "mask": (np.clip(mc, 0, 1) * 255).astype(np.uint8)}]})
                boxes[i].append((box, len(boxes[i])))
            prev = np.stack([_draw_boxes(imgs[i], boxes[i],
                                         th=max(2, min(W, H) // 300)) for i in range(F)])
            print(f"{TAG} {F} image(s) {W}x{H} -> {len(alive)} crop(s) | "
                  f"per image: {[per_image.get(i, 0) for i in range(F)]}")
            return (crops, cmasks,
                    {"width": W, "height": H, "total": F, "batch": True, "people": people},
                    torch.from_numpy(np.ascontiguousarray(prev)), len(alive))

        if single_image:
            # each mask becomes a 1-frame track, still split into blobs: that is
            # what turns one mask holding the whole crowd into N people.
            tracks = []
            for m in ms:
                for d in _blobs(m):
                    if d["area"] >= limit:
                        tracks.append({"bbs": {0: d["box"]}, "mask": {0: d},
                                       "cx": 0, "cy": 0, "last": 0, "born": 0})
        else:
            # jump threshold: how far a person may travel between 2 frames and
            # still be the same one. 12% of the diagonal covers a fast step
            # without swapping people.
            dist_max = 0.12 * float(np.hypot(W, H))
            # how many frames a person may be missing and still be the SAME
            # track. Too short (2) turned a 5-frame occlusion into a new person
            # — and two tracks for one person mean two WAN passes over the same
            # region, one erasing the other at paste time.
            tracks = _track(ms, F, limit, dist_max, max_gap=max(8, F // 5))

        alive = [t for t in tracks if len(t["bbs"]) >= (min_presence / 100.0) * F]
        alive = _filter_points(alive, _points_px(points, W, H),
                               _points_px(points_exclude, W, H))
        alive.sort(key=lambda t: -len(t["bbs"]))
        if max_people > 0:
            alive = alive[:max_people]
        if not alive:
            raise ValueError(
                f"{TAG} nobody left. {len(tracks)} track(s) found, all below "
                f"min_presence={min_presence}% or min_area={min_area}%. Lower both and run again.")

        cost = len(alive) * F * height * width * 3 * 4 / 1024**3
        if cost > 4.0:
            print(f"{TAG} WARNING: {len(alive)} person(s) x {F} frame(s) at {width}x{height} = "
                  f"~{cost:.1f} GB in crops alone. Run it per SEGMENT instead of the whole video, "
                  f"or lower max_people.")

        crops, cmasks, people, boxes_per_frame = [], [], [], [[] for _ in range(F)]
        for p, t in enumerate(alive):
            bbs = [t["bbs"].get(f) for f in range(F)]
            ok = [f for f in range(F) if bbs[f] is not None]
            bw, bh, cx, cy, _ = _moving_box(bbs, ok, F, ar, padding, box_mode,
                                            smoothing, margin, W, H)
            cs, cm, frames = [], [], []
            for f in range(F):
                x0 = int(round(cx[f] - bw / 2.0)); y0 = int(round(cy[f] - bh / 2.0))
                box = (x0, y0, x0 + bw, y0 + bh)
                d = t["mask"].get(f)
                if d is None:
                    mc = np.zeros((bh, bw), np.float32)
                else:
                    mc = _crop_pad(d["m"], box)
                    if d["lab"] is not None:
                        mc = mc * (_crop_pad(d["lab"], box) == d["idx"])
                cs.append(_resize_rgb(_crop_pad(imgs[f], box), width, height))
                cm.append(_resize_gray(mc, width, height))
                frames.append({"box": box, "valid": bbs[f] is not None,
                               "mask": (np.clip(mc, 0, 1) * 255).astype(np.uint8)})
                boxes_per_frame[f].append((box, p))
            crops.append(torch.from_numpy(np.ascontiguousarray(np.stack(cs))))
            cmasks.append(torch.from_numpy(np.ascontiguousarray(np.stack(cm))))
            people.append({"bw": bw, "bh": bh, "frames": frames})
            print(f"{TAG}   person {p + 1}: box {bw}x{bh} -> {width}x{height} | "
                  f"present in {len(ok)}/{F} frame(s)")

        prev = np.stack([_draw_boxes(imgs[f], boxes_per_frame[f],
                                     th=max(2, min(W, H) // 300)) for f in range(F)])
        print(f"{TAG} {F} frame(s) {W}x{H} | {len(tracks)} track(s) -> {len(alive)} person(s) "
              f"(min_presence={min_presence}% min_area={min_area}%) | each one becomes one pass "
              f"of whatever comes next")
        return (crops, cmasks,
                {"width": W, "height": H, "total": F, "people": people},
                torch.from_numpy(np.ascontiguousarray(prev)), len(alive))


class PersonTilesMerge:
    """Pastes the treated people back into the video, each by her own mask."""

    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The WHOLE original video (or photo)."}),
                "crops": ("IMAGE", {"tooltip": "The treated crops, one per person "
                                               "(arrives as a list automatically)."}),
                "tracks": ("SOUL_PERSON_TRACKS",),
                "grow": ("INT", {"default": 8, "min": 0, "max": 128,
                                 "tooltip": "Dilates the mask before pasting, in pixels."}),
                "feather": ("INT", {"default": 20, "min": 0, "max": 256,
                                    "tooltip": "Softens the edge. This is what makes the seam "
                                               "disappear."}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                       "tooltip": "1 pastes it whole; below that it blends with "
                                                  "the original."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "pasted_mask")
    FUNCTION = "run"
    CATEGORY = "Soul/Person"

    def run(self, images, crops, tracks, grow, feather, strength):
        one = lambda v: v[0] if isinstance(v, list) else v
        tk = one(tracks)
        grow, feather, strength = one(grow), one(feather), one(strength)
        lst = crops if isinstance(crops, list) else [crops]

        # `images` may arrive as a LIST OF CHUNKS instead of the assembled
        # video. Joining them here beats asking the user to rewire: the chunk
        # output is the obvious one to connect, and taking only the first (33 of
        # 99 frames) is a silent error waiting to happen. It only joins when the
        # count MATCHES `tracks` — otherwise it would be guesswork.
        ims = [x for x in (images if isinstance(images, list) else [images]) if x is not None]
        base = _img_np(ims[0]) if ims else np.zeros((1, 8, 8, 3), np.float32)
        if len(ims) > 1:
            parts = [_img_np(x) for x in ims]
            total = sum(len(p) for p in parts)
            same = len({p.shape[1:] for p in parts}) == 1
            if same and total == tk.get("total") and parts[0].shape[1:3] == (tk.get("height"),
                                                                            tk.get("width")):
                base = np.concatenate(parts, axis=0)
                print(f"{TAG_M} got {len(parts)} chunks ({[len(p) for p in parts]} frames) "
                      f"instead of the video — joined them into the {total} frames `tracks` "
                      f"expects.")
        F, H, W = base.shape[0], base.shape[1], base.shape[2]
        people = tk.get("people", [])

        tW, tH, tF = tk.get("width"), tk.get("height"), tk.get("total")
        if (tW, tH, tF) != (W, H, F):
            # A diagnosis instead of "connect the right video": what arrives
            # wrong here is almost always one of TWO things, and the numbers
            # tell you which.
            if (tW, tH) == (W, H) and tF and F and tF % F == 0 and tF != F:
                hint = (f"  {F} frame(s) against {tF} is exactly {tF // F}x — that is ONE CHUNK "
                        f"of the video, not the whole thing. A chunk output is a LIST: connected "
                        f"here, the merge takes only the first one.\n"
                        f"  Connect the FULL video to `images`.")
            elif (tW, tH) != (W, H) and W == H:
                hint = (f"  {W}x{H} square is the size of a CROP, not of the video: it looks "
                        f"like a crop reached this input instead of the video.\n"
                        f"  `images` takes the WHOLE image or video — the same one that went "
                        f"into PersonTiles. The treated crops go to `crops`.")
            else:
                hint = ("  `images` has to receive the SAME video (same resolution and same "
                        "frame count) the `tracks` were generated against.")
            raise ValueError(
                f"{TAG_M} the video here is {W}x{H} with {F} frame(s), but the tracks are of "
                f"{tW}x{tH} with {tF}.\n" + hint)
        # On IMAGE (1 frame) the K people fit both in a LIST of K items and in a
        # BATCH of K. Accepting both avoids breaking anyone with an
        # ImageListToImageBatch in the way — which is how older workflows were
        # wired, back when this node's output was still a batch.
        if len(lst) == 1 and len(people) > 1 and F == 1:
            single = _img_np(lst[0])
            if len(single) == len(people):
                lst = [torch.from_numpy(np.ascontiguousarray(single[i:i + 1]))
                       for i in range(len(people))]
                print(f"{TAG_M} received 1 batch of {len(people)} instead of a list — unpacked "
                      f"it (there is an ImageListToImageBatch in the way; you can remove it).")
        # A count mismatch does NOT kill the job: the number of people changes
        # from image to image, and stopping mid-batch is worse than pasting what
        # worked. It pastes what it can, warns loudly, and the people without a
        # crop keep the original (the acceptable worst case: nothing gets worse).
        n = min(len(lst), len(people))
        if len(lst) != len(people):
            print(f"{TAG_M} WARNING: {len(lst)} crop(s) for {len(people)} person(s) — pasting "
                  f"{n} and leaving the rest original.")
            if len(lst) < len(people):
                print(f"{TAG_M} WARNING: a crop is missing. If there is an "
                      f"ImageListToImageBatch between the generator and this merge, it collapses "
                      f"the list into a single batch — remove it and connect directly.")
        lst, people = lst[:n], people[:n]
        if n == 0:
            print(f"{TAG_M} WARNING: nothing to paste, video returned untouched.")
            return (torch.from_numpy(np.ascontiguousarray(base)),
                    torch.from_numpy(np.zeros((F, H, W), np.float32)))

        out = base.copy()
        acc = np.zeros((F, H, W), np.float32)
        for c_t, person in zip(lst, people):
            g = _img_np(c_t)
            if len(g) > F:
                g = g[:F]
            # in BATCH mode each person belongs to ONE image: paste only there,
            # otherwise the person from image 1 would be stamped onto images 2 and 3
            fr = person.get("frame")
            if fr is not None:
                pairs = [(fr, person["frames"][0])]
            else:
                # `offset` = which video frame the crop STARTS at. Whoever cuts
                # the video into chunks beforehand returns people who only exist
                # in their own piece: without this, chunk 2 would be pasted at
                # the start of the video. Missing field means 0 — the old
                # behavior.
                d0 = int(person.get("offset", 0))
                if len(g) < len(person["frames"]) - d0:
                    print(f"{TAG_M} WARNING: one person came back with {len(g)} frame(s) out of "
                          f"{len(person['frames']) - d0} — pasting only those.")
                pairs = [(d0 + k, q) for k, q in enumerate(person["frames"][d0:d0 + len(g)])]
            for k, (f, q) in enumerate(pairs):
                gi = 0 if fr is not None else k
                if not q["valid"]:
                    continue
                x0, y0, x1, y1 = q["box"]
                bw, bh = x1 - x0, y1 - y0
                m = q["mask"].astype(np.float32) / 255.0
                if m.shape != (bh, bw):
                    m = _resize_gray(m, bw, bh)
                m = _edge(m, grow, feather)
                if strength < 1.0:
                    m = m * strength
                # in batch the treated crop has 1 frame (gi=0); on video, frame f
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

        print(f"{TAG_M} {len(lst)} person(s) pasted into {F} frame(s) {W}x{H} | "
              f"grow={grow} feather={feather} strength={strength} | "
              f"{float(acc.mean()) * 100:.1f}% of the image touched")
        return (torch.from_numpy(np.ascontiguousarray(np.clip(out, 0, 1))),
                torch.from_numpy(np.ascontiguousarray(acc)))


class PersonTiles:
    """Crops each person. Five settings, nothing more.

    This is the slim shell around PersonTilesAdvanced: the parameters that only
    matter for VIDEO (box_mode, smoothing, min_presence) stay at their defaults
    and disappear from the screen. For video, use SoulPersonTilesAdvanced, which
    exposes everything.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The photo."}),
                "masks": ("MASK", {"tooltip": "The people SAM3 found."}),
                "size": ("INT", {"default": 496, "min": 64, "max": 2048, "step": 16,
                                 "tooltip": "Side of the crop, in pixels. This is what the "
                                            "enhance model receives: bigger is better and "
                                            "slower."}),
                "padding": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 1.0, "step": 0.05,
                                      "tooltip": "Room around the person. 0.30 = 30%, enough "
                                                 "not to clip hair and feet."}),
                "min_area": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 50.0, "step": 0.01,
                                       "tooltip": "Ignores any blob smaller than this % of the "
                                                  "frame. Kills mask noise and the 10-pixel "
                                                  "extra, which would burn a whole pass."}),
                "max_people": ("INT", {"default": 0, "min": 0, "max": 64,
                                       "tooltip": "0 = everyone. Any other number keeps the N "
                                                  "largest."}),
            },
            "optional": {
                "points": ("SAM3_POINTS_PROMPT",
                           {"tooltip": "OPTIONAL. Connect SAM3PointCollector and click the "
                                       "people you want. Empty = everyone."}),
                "points_exclude": ("SAM3_POINTS_PROMPT",
                                   {"tooltip": "OPTIONAL. Anyone clicked here is left out."}),
                "use_points": ("BOOLEAN", {"default": True,
                                           "label_on": "honor the points",
                                           "label_off": "ignore (take everyone)"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "SOUL_PERSON_TRACKS", "IMAGE", "INT")
    RETURN_NAMES = ("crops", "crop_masks", "tracks", "preview", "count")
    OUTPUT_IS_LIST = (True, True, False, False, False)
    FUNCTION = "run"
    CATEGORY = "Soul/Person"

    def run(self, images, masks, size, padding, min_area, max_people,
            points=None, points_exclude=None, use_points=True):
        return PersonTilesAdvanced().run(
            images=images, masks=masks, width=size, height=size,
            padding=padding, box_mode="locked in segments", smoothing=9,
            min_area=min_area, min_presence=15, max_people=max_people,
            margin=0.12, input_type="separate images",
            points=points, points_exclude=points_exclude, use_points=use_points)


NODE_CLASS_MAPPINGS = {
    "SoulPersonTiles":          PersonTiles,
    "SoulPersonTilesAdvanced":  PersonTilesAdvanced,
    "SoulPersonTilesMerge":     PersonTilesMerge,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SoulPersonTiles":          "Soul Person Tiles (crop each person)",
    "SoulPersonTilesAdvanced":  "Soul Person Tiles - advanced (video)",
    "SoulPersonTilesMerge":     "Soul Person Tiles Merge (paste back)",
}

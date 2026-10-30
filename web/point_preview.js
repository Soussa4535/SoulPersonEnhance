/**
 * Soul Person Enhance — pre-run preview for the point picker.
 *
 * SAM3PointCollector only paints its canvas from `onExecuted`, when the Python
 * side sends the frame back as base64. That means you have to run the graph
 * once, blind, before you can see what you are clicking on — which defeats the
 * point of picking people by hand.
 *
 * This extension fills the same canvas from whatever Load Image or Load Video
 * node feeds it, as soon as the file is known. Nothing about the widget
 * changes: it sets `canvasWidget.image` and calls the node's own
 * `redrawCanvas()`, the exact path `onExecuted` takes. A real run still
 * overwrites it with the executed frame, so the two never fight.
 *
 * For video the browser does the decoding: the file is loaded into a detached
 * <video>, seeked to the first frame, and drawn once into an offscreen canvas.
 * Formats the browser cannot play (ProRes, 10-bit, some HEVC) simply fail and
 * leave the canvas as it was.
 *
 * Copyright (C) 2026 Felipe Soussa
 * License AGPL-3.0-or-later — see LICENSE.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const COLLECTOR = "SAM3PointCollector";
const MAX_HOPS = 8;

/**
 * Walks up from `node` through whatever is feeding its image input until it
 * finds a node that names a file: `image` on Load Image, `video` on the video
 * loaders. Nodes in between (ImageFromBatch, resizes, reroutes) are passed
 * through, so the picker still previews when it sits behind a few of them.
 */
function findSource(node) {
    let current = node;
    for (let hop = 0; hop < MAX_HOPS; hop++) {
        const input = current.inputs?.find(
            (x) => x.name === "image" || x.name === "images" || x.name === "video"
        );
        if (!input || input.link == null) return null;
        const link = app.graph.links[input.link];
        if (!link) return null;
        const source = app.graph.getNodeById(link.origin_id);
        if (!source) return null;

        const image = source.widgets?.find((w) => w.name === "image");
        if (image && typeof image.value === "string") return { widget: image, video: false };
        const video = source.widgets?.find((w) => w.name === "video");
        if (video && typeof video.value === "string") return { widget: video, video: true };

        current = source;
    }
    return null;
}

/**
 * ComfyUI writes a file reference as `name`, `sub/dir/name`, or `name [input]`
 * when it lives outside the input folder. Split it back into the three query
 * parameters /view expects.
 */
function viewURL(reference) {
    if (typeof reference !== "string" || !reference) return null;
    let type = "input";
    let path = reference;
    const tagged = path.match(/^(.*)\s\[(\w+)\]$/);
    if (tagged) {
        path = tagged[1];
        type = tagged[2];
    }
    const cut = path.lastIndexOf("/");
    const subfolder = cut === -1 ? "" : path.slice(0, cut);
    const filename = cut === -1 ? path : path.slice(cut + 1);
    if (!filename) return null;
    return api.apiURL(
        "/view?" + new URLSearchParams({ filename, subfolder, type }).toString()
    );
}

/** Resolves to an Image: the file itself, or the first frame decoded from it. */
function loadFrame(url, isVideo) {
    if (!isVideo) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => resolve(img);
            img.onerror = reject;
            img.src = url;
        });
    }
    return new Promise((resolve, reject) => {
        const video = document.createElement("video");
        video.muted = true;
        video.playsInline = true;
        video.preload = "auto";
        // Seeking to exactly 0 can land before the first decoded frame on some
        // containers; a few milliseconds in is reliably past it.
        const grab = () => {
            const off = document.createElement("canvas");
            off.width = video.videoWidth;
            off.height = video.videoHeight;
            if (!off.width || !off.height) return reject(new Error("no video size"));
            off.getContext("2d").drawImage(video, 0, 0);
            const img = new Image();
            img.onload = () => resolve(img);
            img.onerror = reject;
            img.src = off.toDataURL("image/jpeg", 0.92);
        };
        video.addEventListener("seeked", grab, { once: true });
        video.addEventListener("loadeddata", () => { video.currentTime = 0.04; }, { once: true });
        video.addEventListener("error", reject, { once: true });
        video.src = url;
    });
}

/** Paints the resolved frame into the collector's canvas, sized to it. */
function paint(node, url, isVideo) {
    const cw = node.canvasWidget;
    if (!cw?.canvas || !cw.container) {
        // The SAM3 extension builds the canvas in its own onNodeCreated. If we
        // got here first, forget this attempt so the retry is not skipped.
        node._soulPreviewURL = null;
        setTimeout(() => refresh(node), 100);
        return;
    }
    const canvas = cw.canvas;

    loadFrame(url, isVideo).then((img) => {
        // A run that finished while this was loading wins: it is the real frame.
        if (node._soulPreviewURL !== url) return;
        cw.image = img;
        canvas.width = img.width;
        canvas.height = img.height;

        const width = node.size[0] || 400;
        const height = Math.round((width - 20) * (img.height / img.width));
        node._isResizing = true;
        cw.widgetHeight = height;
        cw.container.style.height = height + "px";
        node.setSize([width, height + 80]);
        setTimeout(() => {
            node._isResizing = false;
        }, 50);

        node.redrawCanvas();
        app.graph.setDirtyCanvas(true, true);
    }).catch(() => {
        // Nothing uploaded yet, or a container the browser cannot decode.
        if (node._soulPreviewURL === url) node._soulPreviewURL = null;
    });
}

/** Resolves what the node should be showing and paints it if that changed. */
function refresh(node) {
    const source = findSource(node);
    if (!source) return;
    const { widget, video } = source;

    // Re-paint whenever the user picks another file, including uploads.
    if (widget._soulHooked !== node.id) {
        const original = widget.callback;
        widget.callback = function (...args) {
            const r = original?.apply(this, args);
            setTimeout(() => refresh(node), 0);
            return r;
        };
        widget._soulHooked = node.id;
    }

    const url = viewURL(widget.value);
    if (!url || node._soulPreviewURL === url) return;
    node._soulPreviewURL = url;
    paint(node, url, video);
}

app.registerExtension({
    name: "soul.person.enhance.point_preview",

    async nodeCreated(node) {
        if (node.comfyClass !== COLLECTOR) return;

        const onConnectionsChange = node.onConnectionsChange;
        node.onConnectionsChange = function (...args) {
            const r = onConnectionsChange?.apply(this, args);
            setTimeout(() => refresh(this), 0);
            return r;
        };

        // The graph is not wired yet at creation time; wait for it to settle.
        setTimeout(() => refresh(node), 0);
    },

    async afterConfigureGraph() {
        for (const node of app.graph.nodes) {
            if (node.comfyClass === COLLECTOR) refresh(node);
        }
    },
});

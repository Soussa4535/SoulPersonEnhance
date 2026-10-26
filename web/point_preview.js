/**
 * Soul Person Enhance — pre-run preview for the point picker.
 *
 * SAM3PointCollector only paints its canvas from `onExecuted`, when the Python
 * side sends the frame back as base64. That means you have to run the graph
 * once, blind, before you can see what you are clicking on — which defeats the
 * point of picking people by hand.
 *
 * This extension fills the same canvas from the image the Load Image node
 * already points at, as soon as it is known. Nothing about the widget changes:
 * it sets `canvasWidget.image` and calls the node's own `redrawCanvas()`, the
 * exact path `onExecuted` takes. A real run still overwrites it with the
 * executed frame, so the two never fight.
 *
 * Copyright (C) 2026 Felipe Soussa
 * License AGPL-3.0-or-later — see LICENSE.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const COLLECTOR = "SAM3PointCollector";

/** The filename widget of a Load Image node, if that is what feeds `node`. */
function upstreamImageWidget(node) {
    const input = node.inputs?.find((i) => i.name === "image");
    if (!input || input.link == null) return null;
    const link = app.graph.links[input.link];
    if (!link) return null;
    const source = app.graph.getNodeById(link.origin_id);
    if (!source) return null;
    return source.widgets?.find((w) => w.name === "image") ?? null;
}

/**
 * ComfyUI writes an image reference as `name`, `sub/dir/name`, or
 * `name [input]` when it lives outside the input folder. Split it back into
 * the three query parameters /view expects.
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

/** Paints `url` into the collector's canvas, sized to the image. */
function paint(node, url) {
    const cw = node.canvasWidget;
    if (!cw?.canvas || !cw.container) {
        // The SAM3 extension builds the canvas in its own onNodeCreated. If we
        // got here first, forget this attempt so the retry is not skipped.
        node._soulPreviewURL = null;
        setTimeout(() => refresh(node), 100);
        return;
    }
    const canvas = cw.canvas;

    const img = new Image();
    img.onload = () => {
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
    };
    img.onerror = () => {
        // The file may simply not exist yet (fresh workflow, nothing uploaded).
        if (node._soulPreviewURL === url) node._soulPreviewURL = null;
    };
    img.src = url;
}

/** Resolves what the node should be showing and paints it if that changed. */
function refresh(node) {
    const widget = upstreamImageWidget(node);
    if (!widget) return;

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
    paint(node, url);
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

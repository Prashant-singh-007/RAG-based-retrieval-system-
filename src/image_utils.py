"""Helpers for turning the image metadata produced by loader.py into
base64-encoded images that can be attached to a multimodal LLM message
(Section 3.2: "Text and Figures" jointly analyzed by the summarizing LLM;
Section 3.3.3: direct comparison mode retrieves "complete textual content and
associated visual data").
"""

import base64
import os

# Keep multimodal prompts bounded - too many images per call blows up context
# and latency for local vision models.
MAX_IMAGES_PER_CALL = 4


def parse_image_refs(image_refs_str):
    """Parse the 'tag::path;tag::path' string stored in chunk/page metadata
    (see loader.py) back into a list of {'tag', 'path'} dicts."""
    if not image_refs_str:
        return []
    refs = []
    for entry in image_refs_str.split(";"):
        if "::" not in entry:
            continue
        tag, path = entry.split("::", 1)
        refs.append({"tag": tag, "path": path})
    return refs


def encode_image_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def collect_image_paths(docs, max_images=MAX_IMAGES_PER_CALL):
    """Gather unique, existing image paths referenced across a list of
    Documents, bounded to max_images.

    Note: because chunking copies a page's metadata onto every chunk split
    from it, an image_refs entry may appear on chunks whose text doesn't
    actually contain that [IMAGE_REF_n] tag. This is a reasonable
    approximation (the image is still from the same page) rather than an
    exact per-chunk association.
    """
    paths = []
    seen = set()
    for doc in docs:
        for ref in parse_image_refs(doc.metadata.get("image_refs", "")):
            path = ref["path"]
            if path not in seen and os.path.exists(path):
                seen.add(path)
                paths.append(path)
            if len(paths) >= max_images:
                return paths
    return paths


def build_multimodal_content(text, image_paths):
    """Build the content list for a HumanMessage combining a text prompt with
    one or more images, in the format langchain_ollama's ChatOllama expects
    for vision-capable models (e.g. llama3.2)."""
    content = [{"type": "text", "text": text}]
    for path in image_paths:
        b64 = encode_image_base64(path)
        content.append({"type": "image_url", "image_url": f"data:image/png;base64,{b64}"})
    return content
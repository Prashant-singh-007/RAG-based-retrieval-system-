import os
import pdfplumber
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

# Paper (Section 3.1): "PDF documents are first processed using PyPDFLoader
# (from langchain v1.1.0) to extract textual content. Visual elements such as
# block diagrams, pinout schematics, and functional illustrations are
# identified and replaced with textual placeholders (e.g., [IMAGE_REF]), while
# the original image files are preserved separately and linked through
# metadata."
IMAGE_REF_TAG = "[IMAGE_REF_{n}]"


def _extract_images(pdf_path, images_dir):
    """Identify visual elements per page and save them as standalone image
    files. pdfplumber only gives image bounding boxes, so each region is
    rasterized (cropped + rendered) rather than extracted byte-for-byte;
    this is sufficient to preserve a linkable visual reference.
    Returns {page_num: [{"tag", "path", "bbox"}, ...]}.
    """
    os.makedirs(images_dir, exist_ok=True)
    source = os.path.splitext(os.path.basename(pdf_path))[0]
    page_images = {}

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            refs = []
            for idx, img in enumerate(page.images):
                try:
                    bbox = (
                        max(img["x0"], 0),
                        max(img["top"], 0),
                        min(img["x1"], page.width),
                        min(img["bottom"], page.height),
                    )
                    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                        continue
                    cropped = page.crop(bbox)
                    image_path = os.path.join(
                        images_dir, f"{source}_p{page_num + 1}_img{idx + 1}.png"
                    )
                    cropped.to_image(resolution=150).save(image_path)
                    refs.append(
                        {
                            "tag": IMAGE_REF_TAG.format(n=idx + 1),
                            "path": image_path,
                            "bbox": bbox,
                        }
                    )
                except Exception as e:
                    print(f"  Could not extract image {idx} on page {page_num + 1}: {e}")
            if refs:
                page_images[page_num] = refs

    return page_images


def load_pdf(pdf_path, images_dir=None, extract_images=True):
    """Load a single PDF: text via PyPDFLoader, visual elements replaced with
    [IMAGE_REF_n] placeholders whose originals are linked through metadata.
    """
    source = os.path.basename(pdf_path)
    loader = PyPDFLoader(pdf_path)
    raw_pages = loader.load()

    page_images = {}
    if extract_images:
        target_dir = images_dir or os.path.join(os.path.dirname(pdf_path), "_images")
        try:
            page_images = _extract_images(pdf_path, target_dir)
        except Exception as e:
            print(f"Failed to extract images from {pdf_path}: {e}")

    documents = []
    for page_num, page in enumerate(raw_pages):
        text = page.page_content
        if not text or not text.strip():
            continue

        refs = page_images.get(page_num, [])
        if refs:
            text = f"{text}\n\n" + " ".join(r["tag"] for r in refs)

        # Chroma metadata values must be scalars, so image links are flattened
        # to a single delimited string (tag::path pairs) rather than a list.
        image_refs_str = ";".join(f"{r['tag']}::{r['path']}" for r in refs)

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": source,
                    "page": page_num + 1,
                    "image_refs": image_refs_str,
                },
            )
        )

    return documents


def load_pdfs_from_dir(data_dir, extract_images=True):
    """Recursively load every PDF under data_dir into a single list of Documents."""
    documents = []

    for root, _, files in os.walk(data_dir):
        for filename in files:
            if filename.lower().endswith(".pdf"):
                pdf_path = os.path.join(root, filename)
                try:
                    documents.extend(load_pdf(pdf_path, extract_images=extract_images))
                except Exception as e:
                    print(f"Failed to load {pdf_path}: {e}")

    return documents

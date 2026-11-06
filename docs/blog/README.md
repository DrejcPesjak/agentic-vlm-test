# Blog package

Open `index.html` for the reading preview. Edit `post.md` for the article.

The article is about 1,200 words. The prose contains no em dashes, semicolons, or colons. It includes five original figures with PNG exports and editable SVG sources in `assets/`.

The two example figures embed exact image payloads from the completed evaluation. The lead example compares separate GPT-5.4 runs on the same image. The three-crop example shows actual GPT-4o actions in order. Neither figure invents model reasoning or changes experimental pixels. Display scaling and the green region outline are presentation annotations.

The architecture figure describes a simplified LLaVA-style design. It does not claim to show GPT's unpublished architecture. Research links and benchmark attribution are in the article. `assets/provenance.json` records the source episode IDs and crop counts.

Regenerate SVGs and the HTML preview from the repository root with

```bash
uv run python scripts/build_blog.py
```

To also regenerate PNGs, install Chromium and use

```bash
uv run playwright install chromium
uv run python scripts/build_blog.py --render
```

The generator needs the local evaluation artifacts. Reading or publishing the article only needs `index.html`, `post.md`, and `assets/`. The full-results link points to `../evaluation-results.md`, so keep that file with the package or update the link for the final blog location.

No API calls, hosting changes, or commits are needed to rebuild this package.

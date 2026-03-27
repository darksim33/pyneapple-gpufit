---
name: docs-style-guide
description: Writing conventions for documentation
---

# Documentation Style Guide

## When to use me

Use this skill when:
- Creating or editing `.md` files in `docs/`
- Writing README files
- Updating documentation structure

## Principles

1. **Brief over complete.** A reader who needs every detail will read the source.
2. **Show, don't just tell.** Every concept should have a concrete example.
3. **One idea per section.** If a section explains two things, split it.

## File structure

Every documentation file follows this template:

```markdown
# Title

> **TL;DR** — Two to four sentences summarizing the whole file.

---

## Section

Body text ...
```

- The TL;DR is a `> blockquote` immediately below the H1.
- Sections use `##`; subsections use `###`. Never go deeper than `####`.

## Language

- **American English** spelling (`minimize`, `color`, `behavior`).
- Avoid filler words: *simply*, *just*, *obviously*, *note that*, *please*.
- Capitalize proper names: Python, NumPy, NIfTI, NNLS, TOML, CUDA.

## Formatting

### Inline code

Use backticks for file names, paths, function/class/parameter names, CLI flags, TOML keys, and literal values.

### Code blocks

Always specify the language tag: `python`, `bash`, `toml`.

### Tables

Use tables for argument references, parameter lists, comparisons. Keep them scannable.

## Cross-references

- Link to other doc files with relative paths: `[API Reference](pyneapple-solver-api.md)`.
- Link to source only when a specific line is essential.

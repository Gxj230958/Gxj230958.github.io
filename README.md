# gxj230958.github.io

Personal homepage of Xingjian Guo, built with [Hugo](https://gohugo.io) (no
external theme — layouts and CSS in this repository, styled after the Minimal
Light academic theme). Deployed to GitHub Pages via GitHub Actions (build type:
workflow).

## Structure

- `content/_index.md` — About Me text
- `content/research/*.md` — the eight research projects (home shows title +
  summary; each file is a detail page with bullets and Code/Report/PR links)
- `content/internship/*.md` — internship entries (summary on home, details on
  the page)
- `layouts/` — baseof, home (`index.html`), `single.html`, `list.html`,
  `partials/sidebar.html`
- `static/css/` — fonts (Crimson Pro, Ubuntu Mono) and site styles
- `static/img/avatar.jpg`, `static/covers/*.svg`, `static/favicon.svg`

## Local preview

```
hugo serve
# then open http://localhost:1313
```

## Deploy

Push to `main`; the `Deploy Hugo site to Pages` workflow builds `./public`
and publishes <https://gxj230958.github.io/>.

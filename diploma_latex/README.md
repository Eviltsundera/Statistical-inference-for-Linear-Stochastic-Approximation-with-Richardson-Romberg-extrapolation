# LaTeX version of the thesis

This directory mirrors `diploma_typst`: every `.typ` source has a `.tex`
counterpart with the same relative path. The thesis entry point is `main.tex`;
the slide deck entry point is `presentation.tex`.

Build the thesis from this directory with:

```sh
latexmk -lualatex -interaction=nonstopmode -halt-on-error main.tex
```

Build the slides with:

```sh
latexmk -lualatex -interaction=nonstopmode -halt-on-error presentation.tex
```

The converted experiment graphics in `figures/` are rasterized copies of the
repository SVG files, included so that the build does not require Inkscape or
shell escape.

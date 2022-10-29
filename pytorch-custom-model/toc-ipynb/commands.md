## Commands

## Prerequisite 
The pandoc version requires [this nightly build](https://github.com/jgm/pandoc/actions/runs/3343773872) as of 29-Oct-2022
- It fixes [this issue](https://github.com/jgm/pandoc/issues/8402)
- The next pandoc release (>2.19.2) is expected to contain this fix.

### Commands use to generate TOC for notebook
- convert ipynb to markdown
    - `pandoc -f ipynb+raw_markdown -t markdown covidnet.ipynb -o covidnet.md`
- generate markdown toc
    - `pandoc -s --to markdown -f markdown --toc -o covidnet-toc.md covidnet.md`
- convert markdown with toc back to ipynb
    - `pandoc -f markdown -t ipynb covidnet-toc.md -o covidnet-toc.ipynb`

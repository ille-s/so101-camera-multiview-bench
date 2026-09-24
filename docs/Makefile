# Minimal makefile for Sphinx documentation.
#
# Build artefacts go to docs/_build/ (gitignored). Standalone: no workspace
# conventions, no directory walking.

SPHINXOPTS    ?= -j auto
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      ?= _build

help:
	@echo "Build artefacts → $(BUILDDIR)"
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile check clean livehtml

# The gate. -E discards the cached environment, which is the whole point: an
# incremental build happily reports zero warnings for pages it never re-read,
# and that false green hid two docutils errors in August 2026. -W turns the
# remaining warnings into a non-zero exit.
check:
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" -E -a -W $(SPHINXOPTS) $(O)

clean:
	rm -rf "$(BUILDDIR)" api/_generated

# Live-reload server for local development.
# Listens on 127.0.0.1:8765 by default; override with PORT=...
PORT ?= 8765
livehtml:
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O) \
		--host 127.0.0.1 --port $(PORT) \
		--re-ignore 'api/_generated/.*'

# Catch-all: route unknown targets to Sphinx via "make mode".
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

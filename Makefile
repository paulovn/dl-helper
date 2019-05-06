
NAME := dl-helper

# Package version: taken from the __init__.py file
VERSION_FILE := dl_helper/__init__.py
VERSION	     := $(shell grep VERSION $(VERSION_FILE) | sed -r "s/VERSION = '(.*)'/\1/")

PKG := dist/$(NAME)-$(VERSION).tar.gz


# -----------------------------------------------------------------------

all: $(PKG)


install: all
	pip install --upgrade $(PKG)

clean:
	rm -f $(PKG)

uninstall:
	pip uninstall $(NAME)


# -----------------------------------------------------------------------

$(PKG):  $(VERSION_FILE) setup.py
	python3 setup.py sdist


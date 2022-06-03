all: check coverage mutants

.PHONY: \
		all \
		check \
		clean \
		coverage \
		format \
		init \
		install \
		linter \
		mutants \
		mutants_bootstrapping \
		mutants_by_blocks \
		tests

module = bootstrapping_tools
codecov_token = ab86639f-449b-4365-a763-172f8c99214a

define lint
	pylint \
        --disable=bad-continuation \
        --disable=missing-class-docstring \
        --disable=missing-function-docstring \
        --disable=missing-module-docstring \
        ${1}
endef

check: install
	black --check --line-length 100 ${module}
	black --check --line-length 100 tests
	flake8 --max-line-length 100 ${module}
	flake8 --max-line-length 100 tests
	mypy ${module}
	mypy tests

clean:
	rm --force --recursive .*_cache
	rm --force --recursive ${module}.egg-info
	rm --force --recursive ${module}/__pycache__
	rm --force --recursive tests/__pycache__
	rm --force .mutmut-cache
	rm --force coverage.xml

coverage: install
	pytest --cov=${module} --cov-report=xml --verbose && \
	codecov --token=${codecov_token}

format:
	black --line-length 100 ${module}
	black --line-length 100 tests

init: install tests

install:
	pip install --editable .

linter:
	$(call lint, ${module})
	$(call lint, tests)

mutants: install
	mutmut run --paths-to-mutate ${module}

mutants_bootstrapping: install
	mutmut run --paths-to-mutate bootstrapping_tools/bootstrapping.py

mutants_by_blocks: install
	mutmut run --runner "pytest tests/test_resample_by_blocks.py" --paths-to-mutate bootstrapping_tools/resample_by_blocks.py 

tests: install
	pytest --verbose

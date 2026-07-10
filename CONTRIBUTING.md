# Contributing

Open an issue before making a large change. Bug reports should include the
operating system, Python version, command, complete error, and whether the data
identity gates passed. Pull requests must keep published evidence immutable,
add or update tests, and pass `python -m pytest -q` plus
`python scripts/reproduce_all.py --dry-run`.

Do not upload medical images, credentials, model weights, or dataset files.
Generated reproductions belong under the ignored `rebuild/` directory.

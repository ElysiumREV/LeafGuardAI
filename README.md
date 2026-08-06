## Development

This project uses **uv** to manage dependencies and the virtual environment.

### Installation

After cloning the repository, run:

```bash
uv sync
```

This command will install all required dependencies and set up the project's virtual environment automatically.

### Adding Dependencies

To add a new dependency to the project, use:

```bash
uv add <package-name>
```

The `uv` tool will automatically update both the `pyproject.toml` and `uv.lock` files.

### Running the Project

To start the application, run:

```bash
uv run python -m leafguardai
```

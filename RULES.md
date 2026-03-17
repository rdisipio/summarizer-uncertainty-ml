AI Code Assistant Rules
=======================

Purpose
-------
These rules define how code should be written and organized in this repository.
Any AI coding assistant must follow them by default.

Environment
-----------
1. Use Pipenv for environment and dependency management.
2. Target Python version must be **>= 3.13**.
3. Do not introduce tools that replace Pipenv unless explicitly requested.

Code Style and Quality
----------------------
1. Follow PEP 8.
2. Prefer clear, readable code over clever or condensed code.
3. Keep functions and classes focused on a single responsibility.
4. Avoid duplicated logic; extract shared behavior into reusable functions.
5. All imports appear at the beginning of each source file.

Typing and Documentation
------------------------
1. All functions must include type annotations for parameters and return values.
2. All public functions, methods, and classes must include a short docstring explaining purpose and behavior.
3. If a function has side effects, mention them in the docstring.

Project Structure
-----------------
1. Place importable Python code inside subfolders that contain an `__init__.py` file.
2. Use a package-oriented layout (avoid loose Python files at repository root, unless they are explicit entrypoints).
3. Keep related code grouped by feature or domain.

Testing and Reliability
-----------------------
1. New or changed behavior should include tests when a test suite exists.
2. Do not break existing tests.
3. Prefer deterministic code (avoid hidden global state and time-dependent behavior without explicit control).

AI Assistant Behavior Constraints
---------------------------------
1. Make minimal, focused changes that solve the requested task.
2. Do not refactor unrelated code unless explicitly asked.
3. Do not add new dependencies unless needed for the requested task.
4. If assumptions are required, state them clearly.
5. If a requirement is ambiguous, choose the simplest valid implementation.
6. Preserve backward compatibility unless a breaking change is explicitly requested.
7. Do not change this file unless clearly stated otherwise.

Security and Safety Baselines
-----------------------------
1. Never hardcode secrets, API keys, passwords, or tokens.
2. Validate external input at boundaries.
3. Prefer safe standard-library or well-maintained solutions over custom insecure implementations.

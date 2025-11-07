"""
Stub generator for @mergeable decorator.

Generates merge() method stubs that provide perfect IDE support for dataclasses
decorated with @mergeable.

Usage Examples:

# With syntax highlighting (if pygments installed)
    $ python -m c108.stubs mergeable c108/stubs/samples.py

# Disable highlighting
    $ python -m c108.stubs mergeable c108/stubs/samples.py --no-color

# Save to file (no highlighting needed)
    $ python -m c108.stubs mergeable c108/stubs/samples.py -o stubs.txt

"""

import ast
from pathlib import Path


def generate_merge_stub(
    class_name: str,
    fields_info: list[tuple[str, str]],
    decorator_config: dict,
    *,
    cli_sentinel: str = "UNSET",
    include_docs: bool = True,
) -> str:
    """
    Generate merge() method stub for IDE support.

    Args:
        class_name: Name of the dataclass
        fields_info: List of (field_name, type_hint) tuples
        decorator_config: Configuration extracted from @mergeable decorator
        cli_sentinel: Fallback sentinel from CLI (used if decorator doesn't specify)
        include_docs: Whether to include docstring in generated stub

    Returns:
        String containing the method definition ready to copy-paste
    """
    # Use decorator's sentinel, fallback to CLI sentinel
    sentinel_name = decorator_config.get("sentinel") or cli_sentinel
    include_fields = decorator_config.get("include")
    exclude_fields = decorator_config.get("exclude")

    # Filter fields based on decorator configuration
    filtered_fields = []
    for field_name, field_type in fields_info:
        # Skip private fields by default (unless explicitly included)
        if field_name.startswith("_") and (
            include_fields is None or field_name not in include_fields
        ):
            continue

        # Apply include/exclude filters
        if include_fields is not None:
            if field_name not in include_fields:
                continue
        elif exclude_fields is not None:
            if field_name in exclude_fields:
                continue

        filtered_fields.append((field_name, field_type))

    if not filtered_fields:
        return f"    # SKIPPING: No mergeable fields found for {class_name} (check include/exclude configuration)"

    # Build parameter list
    params = []
    doc_params = []
    for field_name, field_type in filtered_fields:
        params.append(f"{field_name}: {field_type} = {sentinel_name}")
        doc_params.append(f"            {field_name}: {_format_field_doc(field_name)}")

    param_str = ",\n              ".join(params)

    # Generate method with decorator-aware documentation
    if include_docs and doc_params:
        doc_param_str = "\n".join(doc_params)

        # Add configuration info to docstring
        config_info = []
        if include_fields:
            config_info.append(f"Only these fields can be merged: {', '.join(include_fields)}")
        if exclude_fields:
            config_info.append(f"These fields are excluded: {', '.join(exclude_fields)}")

        config_note = "\n        ".join(config_info) if config_info else ""
        if config_note:
            config_note = f"\n        \n        {config_note}"

        docstring = f'''        """
        Create a new {class_name} with selectively updated fields.
        
        If parameter value is {sentinel_name}, no update applied to the field.{config_note}
        
        Args:
{doc_param_str}
            
        Returns:
            New {class_name} instance with updated fields
        """'''
    else:
        docstring = f'        """Create new {class_name} with updated fields."""'

    stub_code = f"""    def merge(self, *,
              {param_str}) -> '{class_name}':
{docstring}
        # This method is a stub to provide IDE hinting support and documentation. 
        # The actual implementation is replaced by the @mergeable decorator.
        raise NotImplementedError("This method is replaced by @mergeable decorator")"""

    return stub_code


def extract_dataclass_info(file_path: Path) -> list[tuple[str, list[tuple[str, str]], dict]]:
    """
    Extract mergeable dataclass information from Python file using AST parsing.

    Returns:
        List of (class_name, fields_info, decorator_config) tuples where:
        - fields_info is list of (field_name, type_hint) tuples
        - decorator_config is dict with 'sentinel', 'include', 'exclude' keys
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        tree = ast.parse(content, filename=str(file_path))
    except (SyntaxError, UnicodeDecodeError) as e:
        raise ValueError(f"Could not parse {file_path}: {e}")

    dataclasses_info = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # Check for both @dataclass and @mergeable decorators
            has_dataclass = False
            mergeable_config = None

            for decorator in node.decorator_list:
                if _is_dataclass_decorator(decorator):
                    has_dataclass = True
                elif _is_mergeable_decorator(decorator):
                    mergeable_config = _extract_mergeable_config(decorator)

            # Only process classes that have BOTH decorators
            if has_dataclass and mergeable_config is not None:
                fields_info = _extract_fields_from_class(node)
                dataclasses_info.append((node.name, fields_info, mergeable_config))

    return dataclasses_info


def _is_dataclass_decorator(decorator: ast.expr) -> bool:
    """Check if decorator is @dataclass."""
    if isinstance(decorator, ast.Name):
        return decorator.id == "dataclass"
    elif isinstance(decorator, ast.Attribute):
        return decorator.attr == "dataclass"
    elif isinstance(decorator, ast.Call):
        # Handle @dataclass() with parentheses
        func = decorator.func
        if isinstance(func, ast.Name):
            return func.id == "dataclass"
        elif isinstance(func, ast.Attribute):
            return func.attr == "dataclass"
    return False


def _is_mergeable_decorator(decorator: ast.expr) -> bool:
    """Check if decorator is @mergeable."""
    if isinstance(decorator, ast.Name):
        return decorator.id == "mergeable"
    elif isinstance(decorator, ast.Attribute):
        return decorator.attr == "mergeable"
    elif isinstance(decorator, ast.Call):
        # Handle @mergeable() with parentheses
        func = decorator.func
        if isinstance(func, ast.Name):
            return func.id == "mergeable"
        elif isinstance(func, ast.Attribute):
            return func.attr == "mergeable"
    return False


def _extract_mergeable_config(decorator: ast.expr) -> dict:
    """Extract configuration from @mergeable decorator."""
    config = {
        "sentinel": None,  # Will use CLI default if not specified
        "include": None,
        "exclude": None,
    }

    if isinstance(decorator, ast.Call):
        # @mergeable(sentinel=UNSET, include=['field1'], exclude=['field2'])
        for keyword in decorator.keywords:
            if keyword.arg == "sentinel":
                config["sentinel"] = _ast_to_string(keyword.value)
            elif keyword.arg == "include":
                if isinstance(keyword.value, ast.List):
                    config["include"] = [_ast_to_string(elt) for elt in keyword.value.elts]
            elif keyword.arg == "exclude":
                if isinstance(keyword.value, ast.List):
                    config["exclude"] = [_ast_to_string(elt) for elt in keyword.value.elts]
    else:
        # @mergeable without parentheses - use defaults
        pass

    return config


def _extract_fields_from_class(class_node: ast.ClassDef) -> list[tuple[str, str]]:
    """Extract field information from dataclass AST node."""
    fields_info = []

    for node in class_node.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            # This is a type-annotated assignment: field_name: type = value
            field_name = node.target.id
            type_hint = _ast_to_string(node.annotation)
            fields_info.append((field_name, type_hint))
        elif isinstance(node, ast.Assign):
            # Handle assignments without type hints (less common in dataclasses)
            for target in node.targets:
                if isinstance(target, ast.Name):
                    field_name = target.id
                    fields_info.append((field_name, "Any"))

    return fields_info


def _ast_to_string(node: ast.expr) -> str:
    """Convert AST node to string representation."""
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Constant):
        # Fix: For string constants in lists, return the string value without extra quotes
        if isinstance(node.value, str):
            return node.value
        else:
            return repr(node.value)
    elif isinstance(node, ast.Attribute):
        return f"{_ast_to_string(node.value)}.{node.attr}"
    elif isinstance(node, ast.Subscript):
        value = _ast_to_string(node.value)
        slice_val = _ast_to_string(node.slice)
        return f"{value}[{slice_val}]"
    elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        # Handle Union types with | syntax (Python 3.10+)
        left = _ast_to_string(node.left)
        right = _ast_to_string(node.right)
        return f"{left} | {right}"
    elif isinstance(node, ast.Tuple):
        elements = [_ast_to_string(elt) for elt in node.elts]
        return f"({', '.join(elements)})"
    else:
        # Fallback: try to reconstruct from AST
        try:
            return ast.unparse(node)
        except AttributeError:
            # Python < 3.9 doesn't have ast.unparse
            return "Any"


def _format_field_doc(field_name: str) -> str:
    """Format field name for documentation."""
    return field_name.replace("_", " ").title()


def main(args):
    """Main function for mergeable stub generation CLI."""
    import sys
    from pathlib import Path

    # Try to import pygments for syntax highlighting
    has_pygments = False
    if not getattr(args, "no_color", False):
        try:
            from pygments import highlight
            from pygments.formatters import TerminalFormatter
            from pygments.lexers import PythonLexer

            lexer = PythonLexer()
            formatter = TerminalFormatter()
            has_pygments = True
        except ImportError:
            pass

    output_lines = []

    for file_path in args.files:
        path = Path(file_path)
        if not path.exists():
            print(f"Error: File {file_path} not found", file=sys.stderr)
            continue

        try:
            dataclasses_info = extract_dataclass_info(path)

            if dataclasses_info:
                output_lines.append(f"\n# Stubs for {file_path}")
                output_lines.append("# Copy these into your @mergeable dataclass definitions\n")

                for class_name, fields_info, decorator_config in dataclasses_info:
                    try:
                        stub = generate_merge_stub(
                            class_name,
                            fields_info,
                            decorator_config,
                            cli_sentinel=args.sentinel,
                            include_docs=not args.no_docs,
                        )
                        output_lines.append(f"# Stub for {class_name}:")
                        output_lines.append(stub)
                        output_lines.append("")
                    except Exception as e:
                        output_lines.append(f"# Error generating stub for {class_name}: {e}")

        except Exception as e:
            print(f"Error processing {file_path}: {e}", file=sys.stderr)

    # Output results
    if output_lines:
        result = "\n".join(output_lines)

        if args.output:
            with open(args.output, "w") as f:
                f.write(result)
            print(f"Stubs written to {args.output}")
        else:
            # Print to stdout with optional syntax highlighting
            if has_pygments and sys.stdout.isatty():
                highlighted = highlight(result, lexer, formatter)
                print(highlighted, end="")
            else:
                if not has_pygments and not getattr(args, "no_color", False):
                    print("# Tip: Install pygments for syntax highlighting: pip install pygments")
                print(result)
    else:
        print("No @mergeable dataclasses found in specified files")

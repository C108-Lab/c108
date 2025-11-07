"""
Full merge() implementation generator for dataclasses.

Generates complete merge() method implementations explicitly without decorators.
Uses sentinel-wrapper pairs to handle optional parameters.

Usage Examples:

# Generate for specific classes: User, Config
    $ python -m c108.stubs merge c108/stubs/samples.py --classes Merge

# Generate for all classes in models.py
    $ python -m c108.stubs merge c108/stubs/samples.py --all
"""

import ast
from pathlib import Path

# Sentinel to wrapper function mapping
SENTINEL_WRAPPERS = {
    "None": "ifnotnone",
    "DEFAULT": "ifnotdefault",
    "MISSING": "ifnotmissing",
    "NOT_FOUND": "iffound",
    "STOP": "ifnotstop",
    "UNSET": "ifnotunset",
}


def generate_merge_implementation(
    class_name: str,
    fields_info: list[tuple[str, str]],
    *,
    sentinel: str = "UNSET",
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    exclude_private: bool = True,
    include_docs: bool = True,
) -> str:
    """
    Generate complete merge() method implementation.

    Args:
        class_name: Name of the dataclass
        fields_info: List of (field_name, type_hint) tuples
        sentinel: Sentinel value to use for "not provided" parameters
        include: If provided, only these fields can be merged (whitelist)
        exclude: If provided, these fields are excluded from merging (blacklist)
        exclude_private: If True, exclude private fields (starting with '_')
        include_docs: Whether to include docstring

    Returns:
        String containing the complete method implementation
    """
    # Get wrapper function for sentinel
    wrapper_func = SENTINEL_WRAPPERS.get(sentinel)
    if not wrapper_func:
        raise ValueError(
            f"Unsupported sentinel '{sentinel}'. Supported: {list(SENTINEL_WRAPPERS.keys())}"
        )

    # Filter fields based on configuration
    filtered_fields = []
    for field_name, field_type in fields_info:
        # Skip private fields if exclude_private is True
        if exclude_private and field_name.startswith("_"):
            continue

        # Apply include/exclude filters
        if include is not None:
            if field_name not in include:
                continue
        elif exclude is not None:
            if field_name in exclude:
                continue

        filtered_fields.append((field_name, field_type))

    if not filtered_fields:
        return f"    # SKIPPING: No mergeable fields found for {class_name} (check include/exclude configuration)"

    # Build parameter list
    params = []
    assignments = []
    doc_params = []

    for field_name, field_type in filtered_fields:
        params.append(f"{field_name}: {field_type} = {sentinel}")
        assignments.append(
            f"        {field_name} = {wrapper_func}({field_name}, default=self.{field_name})"
        )
        doc_params.append(f"            {field_name}: {_format_field_doc(field_name)}")

    param_str = ",\n        ".join(params)
    assignment_str = "\n".join(assignments)
    constructor_args = ", ".join([f"{field}={field}" for field, _ in filtered_fields])

    # Generate method with full implementation
    if include_docs and doc_params:
        doc_param_str = "\n".join(doc_params)

        # Add configuration info to docstring
        config_info = []
        if include:
            config_info.append(f"Only these fields can be merged: {', '.join(include)}")
        if exclude:
            config_info.append(f"These fields are excluded: {', '.join(exclude)}")
        if exclude_private:
            config_info.append("Private fields (starting with '_') are excluded")

        config_note = "\n        ".join(config_info) if config_info else ""
        if config_note:
            config_note = f"\n        \n        {config_note}"

        docstring = f'''        """
        Create a new {class_name} instance with selectively updated fields.
        
        If parameter value is {sentinel}, no update applied to the field.{config_note}

        Args:
{doc_param_str}

        Returns:
            New {class_name} instance with merged configuration
        """'''
    else:
        docstring = f'        """Create new {class_name} with merged fields."""'

    implementation = f"""    def merge(
        self,
        *,
        {param_str},
    ) -> Self:
{docstring}
{assignment_str}

        return {class_name}({constructor_args})"""

    return implementation


def extract_dataclass_info(
    file_path: Path, *, target_classes: list[str] | None = None
) -> list[tuple[str, list[tuple[str, str]]]]:
    """
    Extract dataclass information from Python file using AST parsing.

    Args:
        file_path: Path to Python file to parse
        target_classes: If provided, only extract info for these specific classes.
                       If None, extract all dataclasses.

    Returns:
        List of (class_name, fields_info) tuples where fields_info is
        list of (field_name, type_hint) tuples

    Raises:
        ValueError: If target_classes specified but some classes not found
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        tree = ast.parse(content, filename=str(file_path))
    except (SyntaxError, UnicodeDecodeError) as e:
        raise ValueError(f"Could not parse {file_path}: {e}")

    dataclasses_info = []
    found_classes = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # Check for @dataclass decorator
            has_dataclass = False

            for decorator in node.decorator_list:
                if _is_dataclass_decorator(decorator):
                    has_dataclass = True
                    break

            if has_dataclass:
                # If target_classes specified, only include those classes
                if target_classes is None or node.name in target_classes:
                    fields_info = _extract_fields_from_class(node)
                    dataclasses_info.append((node.name, fields_info))
                    found_classes.add(node.name)

    # Validate that all target classes were found
    if target_classes is not None:
        missing_classes = set(target_classes) - found_classes
        if missing_classes:
            raise ValueError(
                f"Dataclasses not found in {file_path}: {', '.join(sorted(missing_classes))}"
            )

    return dataclasses_info


def list_dataclasses(file_path: Path) -> list[str]:
    """
    List all dataclass names in a Python file.

    Args:
        file_path: Path to Python file to parse

    Returns:
        List of dataclass names found in the file
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        tree = ast.parse(content, filename=str(file_path))
    except (SyntaxError, UnicodeDecodeError) as e:
        raise ValueError(f"Could not parse {file_path}: {e}")

    dataclass_names = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # Check for @dataclass decorator
            for decorator in node.decorator_list:
                if _is_dataclass_decorator(decorator):
                    dataclass_names.append(node.name)
                    break

    return sorted(dataclass_names)


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
    """Main function for merge implementation generation CLI."""
    import sys
    from pathlib import Path

    # Validate arguments
    if not args.classes and not args.all:
        print(
            "Error: Must specify either --classes CLASS_NAME [CLASS_NAME ...] or --all",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.classes and args.all:
        print("Error: Cannot use both --classes and --all", file=sys.stderr)
        sys.exit(1)

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
            # Handle --all vs --classes
            if args.all:
                # List available classes first
                available_classes = list_dataclasses(path)
                if not available_classes:
                    print(f"No dataclasses found in {file_path}", file=sys.stderr)
                    continue

                print(
                    f"Generating merge() for all dataclasses in {file_path}: {', '.join(available_classes)}"
                )
                dataclasses_info = extract_dataclass_info(path, target_classes=None)
            else:
                # Use specific classes
                dataclasses_info = extract_dataclass_info(path, target_classes=args.classes)

            if dataclasses_info:
                output_lines.append(f"\n# Merge implementations for {file_path}")
                if args.classes:
                    output_lines.append(f"# Generated for classes: {', '.join(args.classes)}")
                output_lines.append("# Copy these into your dataclass definitions\n")

                for class_name, fields_info in dataclasses_info:
                    try:
                        implementation = generate_merge_implementation(
                            class_name,
                            fields_info,
                            sentinel=args.sentinel,
                            include=getattr(args, "include", None),
                            exclude=getattr(args, "exclude", None),
                            exclude_private=args.exclude_private,
                            include_docs=not args.no_docs,
                        )
                        output_lines.append(f"# Implementation for {class_name}:")
                        output_lines.append(implementation)
                        output_lines.append("")
                    except Exception as e:
                        output_lines.append(
                            f"# Error generating implementation for {class_name}: {e}"
                        )

        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            # For missing classes, show available classes
            if "not found" in str(e):
                try:
                    available = list_dataclasses(path)
                    if available:
                        print(
                            f"Available dataclasses in {file_path}: {', '.join(available)}",
                            file=sys.stderr,
                        )
                except:
                    pass
            continue
        except Exception as e:
            print(f"Error processing {file_path}: {e}", file=sys.stderr)
            continue

    # Output results
    if output_lines:
        result = "\n".join(output_lines)

        if args.output:
            with open(args.output, "w") as f:
                f.write(result)
            print(f"Implementations written to {args.output}")
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
        print("No implementations generated")

from __future__ import annotations
"""
Tree Parser — parses AI response text into a hierarchical tree structure.

Purpose:
  AI responses often contain structured content: a sentence introducing a concept,
  followed by a list, code block, or table. This parser extracts that structure into
  a tree where parent nodes (sentences ending with ':') own their children
  (list items, code blocks, tables, continuation paragraphs).

  This solves a key problem with embedding-based retrieval: two related but different
  formulas (e.g. integration by parts vs fundamental theorem) get different parent nodes
  in the tree, so they're never confused during retrieval regardless of embedding similarity.

Tree structure:
  Root
  ├── Parent node: "Key properties of symmetric matrices:"        ← ends with ":"
  │   ├── Child: "1. All eigenvalues are real"                    ← list item
  │   ├── Child: "2. Eigenvectors are orthogonal"                 ← list item
  │   └── Child: "3. A = QLQ^T"                                  ← list item
  │
  ├── Parent node: "Implementation:"                              ← ends with ":"
  │   └── Child: ```python\ndef init_weights():...```             ← code block
  │
  ├── Flat node: "The weight matrix is initialized."              ← ends with ".", no children
  │
  └── Parent node: "Comparison of methods:"                       ← ends with ":"
      ├── Child: "| Method | Speed | Accuracy |"                  ← table row
      ├── Child: "|--------|-------|----------|"                  ← table row
      └── Child: "| QR     | Fast  | High     |"                  ← table row
"""

import re


# Shared regex for list item detection (used in both Node and TreeParser)
# Matches: "1. text", "1) text", "- text", "* text"
_LIST_ITEM_RE = re.compile(r'^\s*(?:\d+[\.\)]|[-*])\s')


class Node:
    """
    A single node in the response tree.

    Attributes:
        node_type: 'root' | 'parent' | 'child' | 'flat'
            - root: top-level container (the whole response)
            - parent: a sentence ending with ':' that introduces children
            - child: a list item, code block, table row under a parent
            - flat: a standalone sentence with no children
        text: the raw text content of this node
        children: list of child Node objects (only for root and parent)
        depth: how deep in the tree (0 = root, 1 = parent/flat, 2 = child)
        content_type: 'paragraph' | 'list_item' | 'code_block' | 'table'
        language: programming language if content_type == 'code_block' (e.g. 'python')
    """
    def __init__(self, node_type="flat", text="", depth=0):
        self.node_type = node_type    # 'root' | 'parent' | 'child' | 'flat'
        self.text = text.strip()      # the actual text content
        self.children = []            # child nodes
        self.depth = depth            # nesting level
        self.content_type = self._detect_content_type()
        self.language = self._detect_language()

    def _detect_content_type(self):
        """
        Detect whether this node is a code block, table, list item, or paragraph.
        Uses the same _LIST_ITEM_RE regex as the parser for consistency.
        """
        if self.text.startswith("```"):
            return "code_block"
        if self.text.startswith("|") and self.text.endswith("|"):
            return "table"
        if _LIST_ITEM_RE.match(self.text):
            return "list_item"
        return "paragraph"

    def _detect_language(self):
        """
        Extract language from code block opening fence (e.g. '```python').
        Handles extra attributes: '```python {.numberLines}' -> 'python'
        """
        if self.content_type == "code_block":
            first_line = self.text.split('\n')[0].strip()
            if first_line.startswith("```"):
                # Take only the first token after ``` to handle extra attributes
                return first_line[3:].strip().split()[0] or "unknown"
        return None

    def to_dict(self):
        """Convert the node and its children to a dictionary (for JSON/storage)."""
        result = {
            "type": self.node_type,
            "content_type": self.content_type,
            "text": self.text[:200],  # truncate for display
            "length": len(self.text),
        }
        if self.language:
            result["language"] = self.language
        if self.children:
            result["children"] = [c.to_dict() for c in self.children]
        return result

    def print_tree(self, indent=0):
        """Recursively print the tree structure for debugging."""
        prefix = "  " * indent
        type_tag = f"[{self.node_type}:{self.content_type}]"
        text_preview = self.text[:80].replace("\n", "\\n")
        print(f"{prefix}{type_tag} {text_preview}")
        for child in self.children:
            child.print_tree(indent + 1)

    def flatten(self, depth=0):
        """
        Flatten this node and its children into a list of
        (text, node_type, content_type, depth) tuples.
        Useful for embedding: each tuple can be independently evaluated.
        """
        results = []
        if self.node_type != "root":
            results.append((self.text, self.node_type, self.content_type, self.depth))
        for child in self.children:
            results.extend(child.flatten(depth + 1))
        return results

    def __repr__(self):
        prefix = "  " * self.depth
        type_tag = f"[{self.node_type}:{self.content_type}]"
        text_preview = self.text[:60].replace("\n", "\\n")
        return f"{prefix}{type_tag} {text_preview}"


class TreeParser:
    """
    Parses AI response text into a hierarchical tree of Nodes.

    Usage:
        parser = TreeParser()
        root = parser.parse(response_text)
        root.print_tree()
        root.flatten()  # get all texts for embedding
    """

    def parse(self, text):
        """
        Main entry point. Parse AI response text into a tree.

        Detection order (first match wins):
        1. Code blocks (lines starting with ```)
        2. List items (numbered/bulleted lines)
        3. Parent nodes (non-empty lines ending with ':')
        4. Tables (lines starting and ending with |)
        5. Regular paragraphs (everything else)

        After detection order is fixed (BUG 1): list items ending with ':'
        are correctly treated as list items, not parents.
        """
        root = Node("root", "", depth=0)

        # Guard against None or empty input
        if not text or not text.strip():
            return root

        lines = text.split('\n')

        i = 0
        current_parent = root  # tracks the current parent node collecting children

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Skip empty lines. They reset parent scope so that a paragraph
            # after a blank line starts as a flat node, not a child of the previous parent.
            if not stripped:
                current_parent = root  # blank line resets parent scope
                i += 1
                continue

            # ===== CODE BLOCK DETECTION =====
            # A code block starts with ``` and ends with ```
            if stripped.startswith("```"):
                code_lines = [line]
                i += 1
                while i < len(lines) and not lines[i].strip().startswith("```"):
                    code_lines.append(lines[i])
                    i += 1
                if i < len(lines):
                    code_lines.append(lines[i])  # include closing ```
                    i += 1
                code_text = '\n'.join(code_lines)

                # Code blocks become children under a parent or flat at root level.
                # After a code block, reset to root so any following paragraph
                # starts as a new flat node, not a child of the previous parent.
                code_node = Node(
                    "child" if current_parent != root else "flat",
                    code_text,
                    current_parent.depth + 1
                )
                current_parent.children.append(code_node)
                current_parent = root  # code block terminates parent scope
                continue

            # ===== LIST ITEM DETECTION (before parent check!) =====
            # Check list items BEFORE parent nodes so that a list item ending
            # with ":" (e.g. "1. Note: this is important") is correctly parsed
            # as a list item, not hijacked as a parent node.
            if _LIST_ITEM_RE.match(stripped):
                list_node = Node(
                    "child" if current_parent != root else "flat",
                    stripped,
                    current_parent.depth + 1
                )
                current_parent.children.append(list_node)
                i += 1
                continue

            # ===== PARENT NODE DETECTION =====
            # A line ending with ':' introduces children.
            # Exceptions: code fences, markdown headers (#), URLs
            if (stripped.endswith(":")
                    and not stripped.startswith("#")
                    and not stripped.startswith("http")):
                parent_node = Node(
                    "parent",
                    stripped.removesuffix(":"),  # removes only ONE trailing colon
                    depth=1
                )
                root.children.append(parent_node)
                current_parent = parent_node
                i += 1
                continue

            # ===== TABLE DETECTION =====
            # A table row starts and ends with |
            if stripped.startswith("|") and stripped.endswith("|"):
                table_lines = [line]
                i += 1
                while i < len(lines) and lines[i].strip().startswith("|") and lines[i].strip().endswith("|"):
                    table_lines.append(lines[i])
                    i += 1
                table_text = '\n'.join(table_lines)

                table_node = Node(
                    "child" if current_parent != root else "flat",
                    table_text,
                    current_parent.depth + 1
                )
                # _detect_content_type already catches tables, but the
                # content_type is explicit here for clarity
                table_node.content_type = "table"
                current_parent.children.append(table_node)
                current_parent = root  # table terminates parent scope
                continue

            # ===== REGULAR LINE (paragraph) =====
            # Under a parent: added as a child node (continuation text).
            # Under root: added as a flat (standalone) node.
            if current_parent != root:
                child_node = Node("child", stripped, current_parent.depth + 1)
                current_parent.children.append(child_node)
            else:
                flat_node = Node("flat", stripped, 1)
                root.children.append(flat_node)

            i += 1

        # Post-processing: convert parent nodes with no children back to flat nodes.
        # This handles the case where a line ends with ":" but nothing follows it.
        self._prune_empty_parents(root)

        return root

    def _prune_empty_parents(self, node):
        """
        Convert parent nodes that have no children back to flat nodes.
        A ":" doesn't guarantee children — the next line might be empty.
        Recurses through all children to handle nested structures.
        """
        new_children = []
        for child in node.children:
            if child.node_type == "parent" and not child.children:
                child.node_type = "flat"
            # Recurse into children that might have their own children
            self._prune_empty_parents(child)
            new_children.append(child)
        node.children = new_children


def format_as_markdown(node, indent=0):
    """
    Render the tree as markdown for readability/debugging.
    Shows full content for code blocks and tables.
    """
    prefix = "  " * indent
    marker = ""

    if node.node_type == "root":
        result = "# Response Tree\n\n"
        for child in node.children:
            result += format_as_markdown(child, indent)
        return result

    if node.node_type == "parent":
        marker = f"{prefix}- **{node.text}:**"
    elif node.node_type == "child":
        if node.content_type == "code_block":
            lang = node.language or ""
            marker = f"{prefix}  ```{lang}"
        elif node.content_type == "table":
            # Show actual table content
            rows = node.text.split('\n')
            marker = f"{prefix}  | {rows[0].strip('|').split('|')[0].strip()} ... |"
        elif node.content_type == "list_item":
            marker = f"{prefix}  - {node.text}"
        else:
            marker = f"{prefix}  - {node.text}"
    else:  # flat
        marker = f"{prefix}- {node.text}"

    result = marker + "\n"

    # For code blocks, show the full content without fences
    if node.content_type == "code_block":
        code_content = '\n'.join(node.text.split('\n')[1:-1])  # strip opening and closing fences
        for code_line in code_content.split('\n'):
            result += f"{prefix}  | {code_line}\n"

    # For tables, show all rows
    if node.content_type == "table":
        for row in node.text.split('\n'):
            result += f"{prefix}  | {row}\n"

    for child in node.children:
        result += format_as_markdown(child, indent + 1)

    return result


# ==================== TESTING ====================
# Runs a quick self-test when this file is executed directly

if __name__ == "__main__":
    print("=" * 70)
    print("Tree Parser Self-Test")
    print("=" * 70)

    parser = TreeParser()

    test_text = """A symmetric matrix has several important properties:
1. All eigenvalues are real numbers
2. Eigenvectors corresponding to distinct eigenvalues are orthogonal
3. The matrix can be diagonalized as A = QLQ^T

The implementation for weight initialization is:
```python
def init_weights(n, r):
    W = np.random.randn(r, n)
    W, _ = np.linalg.qr(W.T)
    return W.T
```

This ensures the weights are well-conditioned.

Comparison of optimization methods:
| Method | Speed | Memory | Accuracy |
|--------|-------|--------|----------|
| SGD    | Fast  | Low    | Medium   |
| Adam   | Slow  | High   | High     |

The loss function used is cross-entropy."""

    root = parser.parse(test_text)
    print("\nTree structure:")
    print("-" * 40)
    root.print_tree()

    print("\n\nFlattened (for embedding):")
    print("-" * 40)
    for text, ntype, ctype, depth in root.flatten():
        preview = text[:50].replace("\n", "\\n")
        print(f"  [{ntype}:{ctype}] (d={depth}) {preview}")

    print("\n\nMarkdown view:")
    print("-" * 40)
    print(format_as_markdown(root))

"""
Parser for Lean code that converts text input into structured command objects.
"""

import re
from dataclasses import dataclass
from typing import Optional, List


class LeanCommand:
    """Base class for all Lean commands."""

    def complete_str(self) -> str:
        """Return a string representation of the command if it is complete."""
        raise NotImplementedError("Subclasses must implement this method")


@dataclass
class LeanOpen(LeanCommand):
    """Represents an 'open' command in Lean."""

    opened: list[str] = None
    """The name of the module to open or None if the command is incomplete."""

    def complete_str(self) -> str:
        """Return a string representation of the command if it is complete."""
        if not self.opened:
            return "-- open"
        return f"open {' '.join(self.opened)}"


@dataclass
class LeanImport(LeanCommand):
    """Represents an 'import' command in Lean."""

    imported: Optional[str] = None
    """The name of the module to import or None if the command is incomplete."""

    def complete_str(self) -> str:
        """Return a string representation of the command if it is complete."""
        if self.imported is None:
            return "-- import"
        return f"import {self.imported}"


@dataclass
class LeanTheorem(LeanCommand):
    """Represents a theorem definition in Lean."""

    name: Optional[str] = None
    """The name of the theorem or None if the command is incomplete."""

    params: Optional[List[str]] = None
    """The parameters of the theorem or None if the command is incomplete."""

    typ: Optional[str] = None
    """The type of the theorem or None if the command is incomplete."""

    proof: Optional[str] = None
    """The proof of the theorem or None if the command is incomplete."""

    def complete_str(self) -> str:
        """Return a string representation of the command, filling in dummy values for missing parts."""
        if self.name is None:
            res = "theorem dummy "
        else:
            res = f"theorem {self.name} "
        if self.params is not None:
            res += " ".join(self.params)
        if self.typ is None:
            res += " : True"
        else:
            res += f" : {self.typ}"
        if self.proof is None:
            res += " := by sorry\n"
        else:
            res += f" := by {self.proof}"
        return res


@dataclass
class LeanUnknownCommand(LeanCommand):
    """Represents an unrecognized command in Lean."""

    command: Optional[str] = None

    def complete_str(self) -> str:
        """Return a string representation of the command if it is complete."""
        if self.command is None:
            return "-- unknown command"
        return f"-- {self.command}"


def complete_lean_str(commands: List[LeanCommand]) -> str:
    """Return a string representation of the commands."""
    return "\n".join(command.complete_str() for command in commands)


class LeanParser:
    """Parser for Lean code that converts text input into structured command objects."""

    input: str

    def __init__(self, input: str):
        """Initialize the parser with input text."""
        self.input = input

    def skip_ws(self):
        """Trim whitespace at the start of the input"""
        self.input = self.input.lstrip()

    def expect_ws(self):
        """Check if the input starts with whitespace, and advance the input if so"""
        stripped = self.input.lstrip()
        if stripped == self.input:
            return False
        self.input = stripped
        return True

    def expect_newline(self):
        """Check if the input starts with a newline, and advance the input if so"""
        if self.input.startswith("\n"):
            self.input = self.input[1:]
            return True
        return False

    def parse_rest_of_line(self) -> str:
        """Return the rest of the line, including the newline"""
        pos = self.input.find("\n")
        if pos == -1:
            rest = self.input
            self.input = ""
            return rest
        else:
            rest = self.input[: pos + 1]
            self.input = self.input[pos + 1 :]
            return rest

    def expect(self, s: str) -> bool:
        """Check if the input starts with s, and advance the input if so"""
        if self.input.startswith(s):
            self.input = self.input[len(s) :]
            return True
        return False

    def parse_ident(self) -> str:
        """Parse an identifier from the input"""
        self.skip_ws()
        pos = self.input.find(" ")
        if pos == -1:
            ident = self.input
            self.input = ""
            return ident
        else:
            ident = self.input[:pos]
            self.input = self.input[pos:]
            return ident

    def parse_param(self) -> Optional[str]:
        """Parse a parameter from the input."""
        self.skip_ws()
        return self.parse_brackets()

    def parse_type(self) -> Optional[str]:
        """Parse a type expression from the input."""
        self.skip_ws()
        typ = self.parse_well_bracketed(stop_on_regex=":=")
        if typ is not None:
            typ = typ.rstrip()
        return typ

    def parse_proof(self) -> str:
        """Parse the proof part of a theorem"""
        proof = ""
        while True:
            line = self.parse_well_bracketed(stop_on_regex=r"[;\n]")
            if line is None:
                break
            if self.expect(";"):
                proof += line + ";"
            elif self.expect_newline():
                proof += line + "\n"
                # If the next line is not indented, we're done
                m = re.match(r"\n*[^\s]", self.input)
                if m is not None:
                    break
            else:
                break
        return proof

    def parse_brackets(self) -> Optional[str]:
        """
        Parse a bracketed expression from the input.

        Supports parentheses (), square brackets [], and curly braces {}.

        Returns:
            The parsed bracketed expression as a string, or None if parsing fails
        """
        if self.expect("("):
            text = self.parse_well_bracketed()
            if text is None or not self.expect(")"):
                return None
            return "(" + text + ")"
        elif self.expect("["):
            text = self.parse_well_bracketed()
            if text is None or not self.expect("]"):
                return None
            return "[" + text + "]"
        elif self.expect("{"):
            text = self.parse_well_bracketed()
            if text is None or not self.expect("}"):
                return None
            return "{" + text + "}"
        return None

    def parse_well_bracketed(self, stop_on_regex: str = None) -> Optional[str]:
        """Parse a well-bracketed expression from the input.

        A well-bracketed expression is one where all brackets (), [], {} are properly matched.
        The parsing stops when either:
        1. An unmatched closing bracket is encountered
        2. The stop_on_regex pattern is matched (if provided)
        3. The end of input is reached

        Args:
            stop_on_regex: Optional regex pattern to stop parsing at. Common patterns are ":=" for types
                          and "[;\n]" for proofs.

        Returns:
            The parsed expression as a string, or None if parsing fails due to mismatched brackets
        """
        text = ""
        while True:
            regex = (
                r"[\(\[\{\)\]\}]|" + stop_on_regex
                if stop_on_regex
                else r"[\(\[\{\)\]\}]"
            )
            match = re.search(regex, self.input)
            if match:
                pos = match.start()
                text += self.input[:pos]
                self.input = self.input[pos:]
                match match.group(0):
                    case "(" | "[" | "{":
                        bracketed = self.parse_brackets()
                        if bracketed is None:
                            return None
                        text += bracketed
                    case ")" | "]" | "}":
                        return text
                    case _:
                        return text
            else:
                rest = self.input
                self.input = ""
                return rest

    def parse_open(self) -> LeanOpen:
        """Parse an open command from the input"""
        self.skip_ws()
        # read to the end of the line:
        pos = self.input.find("\n")
        if pos == -1:
            opened = self.input
            self.input = ""
        else:
            opened = self.input[:pos]
            self.input = self.input[pos + 1 :]
        opened = re.split(r"\s+", opened.lstrip())
        if pos == -1:
            # command is incomplete, so remove the last (incomplete) element
            opened = opened[:-1]
        return LeanOpen(opened)

    def parse_theorem(self) -> LeanTheorem:
        """Parse a theorem from the input"""
        theorem = LeanTheorem()

        # Parse name
        name = self.parse_ident()
        if not self.expect_ws():
            return theorem
        theorem.name = name

        # Parse parameters
        params = []
        while (param := self.parse_param()) is not None:
            params.append(param)
        theorem.params = params

        # Parse :
        if not self.expect(":"):
            return theorem

        # Parse type
        self.skip_ws()
        typ = self.parse_type()
        if typ is None:
            return theorem

        # Parse :=
        self.skip_ws()
        if not self.expect(":="):
            return theorem
        theorem.typ = typ

        # Parse by
        self.skip_ws()
        if not self.expect("by"):
            return theorem

        # Parse proof
        theorem.proof = self.parse_proof()
        return theorem

    def parse_command(self) -> LeanCommand:
        """Parse a command from the input"""
        while self.expect_newline():
            pass
        if self.expect("open "):
            return self.parse_open()
        elif self.expect("import "):
            imported = self.parse_rest_of_line()
            return LeanImport(imported.strip() if imported.endswith("\n") else None)
        elif self.expect("theorem "):
            return self.parse_theorem()
        else:
            return LeanUnknownCommand(self.parse_rest_of_line())

    def parse_lean(self) -> List[LeanCommand]:
        """Parse a list of commands from the input"""
        commands = []
        while self.input.strip() != "":
            command = self.parse_command()
            if command is not None:
                commands.append(command)
        return commands


def parse_lean(input: str) -> List[LeanCommand]:
    """
    Parse Lean code and return a list of command objects.

    This is a convenience function that creates a LeanParser instance
    and parses the input string.

    Args:
        input: The Lean code text to parse

    Returns:
        A list of LeanCommand objects representing the parsed commands
    """
    parser = LeanParser(input)
    return parser.parse_lean()

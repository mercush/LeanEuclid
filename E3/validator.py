"""Validate results."""
import os
import re
import signal
from subprocess import PIPE, Popen

from LeanEuclid.E3.utils import ROOT_DIR, format_test_file


class Validator:
    """Validate expression."""

    def __init__(
        self, tmp_path: str = os.path.join(ROOT_DIR, "tmp", "validate")
    ) -> None:
        """Initialize validator."""
        self.tmp_path = tmp_path
        os.makedirs(self.tmp_path, exist_ok=True)

    def validate(self, expression: str, instanceName: str) -> str:
        """Validate expression."""
        tmpFile = os.path.join(self.tmp_path, instanceName + ".lean")
        os.makedirs(os.path.dirname(tmpFile), exist_ok=True)

        leanFile = format_test_file(expression)
        with open(tmpFile, "w") as file:
            file.write(leanFile)

        command = ["lake", "env", "lean", "--run", tmpFile]
        process = Popen(
            command, stdin=PIPE, stdout=PIPE, cwd=ROOT_DIR, preexec_fn=os.setsid
        )
        try:
            stdout, _ = process.communicate()
            if stdout == b"":
                return ""
            else:
                error = stdout.decode()
                error = re.sub(r"/[^:]+:\d+:\d+: ", "", error)
                return error
        except:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            return "Unexpected error"

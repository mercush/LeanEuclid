
import asyncio
from lean_interact import LeanREPLConfig, AutoLeanServer, Command, LocalProject
from lean_interact.interface import LeanError, CommandResponse

async def main():
    """
    This test script initializes a Lean server, imports the SystemE package,
    and runs a test theorem to check for congruence of triangles.
    """
    lean_config = LeanREPLConfig(
        lean_version="v4.8.0-rc2",
        project=LocalProject(directory="/LeanEuclid"),
        memory_hard_limit_mb=4000
    )
    lean_server = AutoLeanServer(lean_config)

    import_command = Command(cmd="import SystemE")
    response = lean_server.run(import_command, timeout=60)
    print(f"Import Response: {response}")

    if isinstance(response, LeanError) or (isinstance(response, CommandResponse) and any(m.severity == "error" for m in response.messages)):
        print("Error importing SystemE. Aborting test.")
        return
    for _ in range(5):
        # 2. Run the congruence theorem
        theorem_command_str = """theorem congruence_sas (a b c d e f : Point) (AB BC AC DE EF DF : Line) (h1 : |(a─b)| = |(d─e)| ∧ |(a─c)| = |(d─f)|) : True := by sorry"""
        theorem_command = Command(cmd=theorem_command_str, env=0)
        response = lean_server.run(theorem_command, timeout=60)
        print(f"Theorem Response: {response}")
        
if __name__ == "__main__":
    asyncio.run(main())

import pytest
import os
import sys
import json
from unittest.mock import MagicMock, patch

# Ensure the 'src' directory is in the Python path for imports
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from LeanPotential.lean_parse import LeanParser, LeanTheorem
from LeanEuclid.reformat_theorems import reformat_theorem, main as reformat_main, process_file

# Test cases for the reformat_theorem function
@pytest.mark.parametrize("theorem_obj, expected_str", [
    (
        LeanTheorem(
            name="simple_theorem",
            params=["(A B : Point)", "(h1 : A != B)"],
            typ="exists_line (Line.new A B)"
        ),
        "∀ (A : Point) (B : Point), ((A != B)) → (exists_line (Line.new A B))"
    ),
    (
        LeanTheorem(
            name="multi_hypothesis_theorem",
            params=["(C : Circle)", "(p : Point)", "(h1 : C.center = p)", "(h2 : C.radius > 0)"],
            typ="p.is_on(C)"
        ),
        "∀ (C : Circle) (p : Point), ((C.center = p) ∧ (C.radius > 0)) → (p.is_on(C))"
    ),
    (
        LeanTheorem(
            name="no_hypothesis_theorem",
            params=["(L M : Line)"],
            typ="L.intersects(M) or not L.intersects(M)"
        ),
        "∀ (L : Line) (M : Line), (True) → (L.intersects(M) or not L.intersects(M))"
    ),
    (
        LeanTheorem(
            name="no_variable_theorem",
            params=["(h1 : 1 + 1 = 2)"],
            typ="True"
        ),
        "((1 + 1 = 2)) → (True)"
    ),
    (
        LeanTheorem(
            name="complex_vars_theorem",
            params=["(A B C : Point)", "(L : Line)", "(h1 : A.on_line L)", "(h2 : B.on_line L)", "(h3 : C.on_line L)"],
            typ="collinear A B C"
        ),
        "∀ (A : Point) (B : Point) (C : Point) (L : Line), ((A.on_line L) ∧ (B.on_line L) ∧ (C.on_line L)) → (collinear A B C)"
    ),
    (
        LeanTheorem(
            name="empty_theorem",
            params=[],
            typ="True"
        ),
        "(True)"
    ),
    (
        # This ensures non-standard parameters are ignored gracefully
        LeanTheorem(
            name="decidable_eq_theorem",
            params=["[DecidableEq Point]", "(A B : Point)"],
            typ="A = B or A != B"
        ),
        "∀ (A : Point) (B : Point), (True) → (A = B or A != B)"
    )
])
def test_reformat_theorem(theorem_obj, expected_str):
    """Tests the reformatting of a single LeanTheorem object."""
    assert reformat_theorem(theorem_obj) == expected_str

def test_process_file(tmp_path):
    """Tests processing a single JSON file."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    prediction_content = "theorem my_theorem (A : Point) : A = A := by sorry"
    # Add another key to ensure it's preserved in the output
    json_content = {"prediction": prediction_content, "source": "test"}
    
    input_file = input_dir / "test.json"
    with open(input_file, 'w') as f:
        json.dump(json_content, f)

    # The output file should now be a JSON file with the same name
    output_file = output_dir / "test.json"
    
    process_file(str(input_file), str(output_file))

    assert output_file.exists()
    with open(output_file, 'r') as f:
        output_data = json.load(f)
    
    assert output_data['prediction'] == "∀ (A : Point), (True) → (A = A)"
    assert output_data['source'] == "test" # Verify other data is preserved

def test_main_function(tmp_path):
    """Tests the main script logic for walking directories."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    # Create a nested directory structure
    sub_dir = input_dir / "subdir"
    sub_dir.mkdir()

    # File 1 at root
    json_content1 = {"prediction": "theorem theorem1 : True := by sorry", "id": 1}
    file1 = input_dir / "file1.json"
    with open(file1, 'w') as f:
        json.dump(json_content1, f)

    # File 2 in subdir
    json_content2 = {"prediction": "theorem theorem2 (A : Point) : A = A := by sorry", "id": 2}
    file2 = sub_dir / "file2.json"
    with open(file2, 'w') as f:
        json.dump(json_content2, f)

    reformat_main(str(input_dir), str(output_dir))

    # Check output file 1
    output_file1 = output_dir / "file1.json"
    assert output_file1.exists()
    with open(output_file1, 'r') as f:
        data1 = json.load(f)
    assert data1['prediction'] == "(True)"
    assert data1['id'] == 1

    # Check output file 2
    output_file2 = output_dir / "subdir" / "file2.json"
    assert output_file2.exists()
    with open(output_file2, 'r') as f:
        data2 = json.load(f)
    assert data2['prediction'] == "∀ (A : Point), (True) → (A = A)"
    assert data2['id'] == 2

@pytest.mark.parametrize("prediction_str, expected_reformatted_str", [
    (
        "theorem construct_equilateral_triangle (a b : Point) (AB : Line) (h : a ≠ b ∧ a.onLine AB ∧ b.onLine AB) : ∃ c : Point, |(a─c)| = |(b─c)| ∧ |(a─c)| = |(a─b)| ∧ |(b─c)| = |(a─b)| := by sorry.",
        "∀ (a : Point) (b : Point) (AB : Line), ((a ≠ b ∧ a.onLine AB ∧ b.onLine AB)) → (∃ c : Point, |(a─c)| = |(b─c)| ∧ |(a─c)| = |(a─b)| ∧ |(b─c)| = |(a─b)|)"
    ),
    (
        "theorem bisect_segment (a b : Point) (AB : Line) (h : distinctPointsOnLine a b AB) :\n  ∃ d : Point, d.onLine AB ∧ (|(a─d)| = |(b─d)|) := by sorry\n",
        "∀ (a : Point) (b : Point) (AB : Line), ((distinctPointsOnLine a b AB)) → (∃ d : Point, d.onLine AB ∧ (|(a─d)| = |(b─d)|))"
    ),
    (
        """theorem vertical_angle_theorem (a b c d e : Point) (AB CD : Line)
  (h1 : twoLinesIntersectAtPoint AB CD e) :\n  ∠ a:e:c = ∠ b:e:d ∧ ∠ c:e:b = ∠ a:e:d := by 
  sorry
""",
        "∀ (a : Point) (b : Point) (c : Point) (d : Point) (e : Point) (AB : Line) (CD : Line), ((twoLinesIntersectAtPoint AB CD e)) → (∠ a:e:c = ∠ b:e:d ∧ ∠ c:e:b = ∠ a:e:d)"
    ),
    (
        "theorem exists_point_on_line (L : Line) : ∃ (p : Point), p.on_line L := by sorry",
        "∀ (L : Line), (True) → (∃ (p : Point), p.on_line L)"
    ),
    (
        "theorem exists_midpoint (A B : Point) (h : A ≠ B) : ∃ (M : Point), is_midpoint M A B := by sorry",
        "∀ (A : Point) (B : Point), ((A ≠ B)) → (∃ (M : Point), is_midpoint M A B)"
    ),
    (
        "theorem parallel_line_exists (L : Line) (p : Point) (h : ¬ p.on_line L) : ∃ (M : Line), p.on_line M ∧ parallel L M := by sorry",
        "∀ (L : Line) (p : Point), ((¬ p.on_line L)) → (∃ (M : Line), p.on_line M ∧ parallel L M)"
    ),
    (
        "theorem exists_two_points : ∃ (A B : Point), A ≠ B := by sorry",
        "(∃ (A B : Point), A ≠ B)"
    ),
    (
        "theorem exists_two_points_on_line (L : Line) : ∃ (A B : Point), A.on_line L ∧ B.on_line L ∧ A ≠ B := by sorry",
        "∀ (L : Line), (True) → (∃ (A B : Point), A.on_line L ∧ B.on_line L ∧ A ≠ B)"
    )
])
def test_reformat_prediction(prediction_str, expected_reformatted_str):
    commands = LeanParser(prediction_str).parse_lean()
    output_lines = []
    for command in commands:
        if isinstance(command, LeanTheorem):
            reformatted_theorem = reformat_theorem(command)
            output_lines.append(reformatted_theorem)
    
    assert "\n".join(output_lines) == expected_reformatted_str

# To run these tests, use the following command from the 'src' directory:
# python -m pytest LeanEuclid/test_reformat_theorems.py

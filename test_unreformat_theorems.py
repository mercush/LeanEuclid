from LeanEuclid.AutoFormalization.unreformat_theorems import unreformat_theorem

def test_unreformat_with_variables_and_hypotheses():
    formula = "∀ (a b : Point) (AB : Line), distinctPointsOnLine a b AB → ∃ c : Point, |(c─a)| = |(a─b)| ∧ |(c─b)| = |(a─b)|"
    expected = "theorem example_thm (a b : Point) (AB : Line) (h1 : distinctPointsOnLine a b AB) : ∃ c : Point, |(c─a)| = |(a─b)| ∧ |(c─b)| = |(a─b)| := by sorry"
    assert unreformat_theorem(formula) == expected

def test_unreformat_with_multiple_hypotheses():
    formula = "∀ (a b c : Point), (hyp1) ∧ (hyp2) → (conclusion)"
    expected = "theorem example_thm (a b c : Point) (h1 : (hyp1)) (h2 : (hyp2)) : (conclusion) := by sorry"
    assert unreformat_theorem(formula) == expected

def test_unreformat_no_variables():
    formula = "(hyp1) ∧ (hyp2) → (conclusion)"
    expected = "theorem example_thm (h1 : (hyp1)) (h2 : (hyp2)) : (conclusion) := by sorry"
    assert unreformat_theorem(formula) == expected

def test_unreformat_no_hypotheses():
    formula = "∀ (a : Point), True → (conclusion)"
    expected = "theorem example_thm (a : Point) : (conclusion) := by sorry"
    assert unreformat_theorem(formula) == expected

def test_unreformat_no_premises():
    formula = "conclusion"
    expected = "theorem example_thm : conclusion := by sorry"
    assert unreformat_theorem(formula).strip() == expected.strip()

def test_unreformat_with_theorem_name():
    formula = "∀ (a : Point), True → (conclusion)"
    expected = "theorem my_awesome_thm (a : Point) : (conclusion) := by sorry"
    assert unreformat_theorem(formula, "my_awesome_thm") == expected

def test_unreformat_from_file_1():
    formula = "∀ (a b : Point) (AB : Line), distinctPointsOnLine a b AB → ∃ d : Point, (between a d b) ∧ (|(a─d)| = |(d─b)|)"
    expected = "theorem example_thm (a b : Point) (AB : Line) (h1 : distinctPointsOnLine a b AB) : ∃ d : Point, (between a d b) ∧ (|(a─d)| = |(d─b)|) := by sorry"
    assert unreformat_theorem(formula) == expected

def test_unreformat_from_file_2():
    formula = "∀ (a b c : Point) (AB : Line), distinctPointsOnLine a b AB ∧ between a c b → exists f : Point, ¬(f.onLine AB) ∧ (∠ a:c:f = ∟)"
    expected = "theorem example_thm (a b c : Point) (AB : Line) (h1 : distinctPointsOnLine a b AB) (h2 : between a c b) : exists f : Point, ¬(f.onLine AB) ∧ (∠ a:c:f = ∟) := by sorry"
    assert unreformat_theorem(formula) == expected

def test_unreformat_from_file_3():
    formula = "∀ (a b c d e : Point) (AB CD : Line), distinctPointsOnLine a b AB ∧ distinctPointsOnLine c d CD ∧ e.onLine AB ∧ e.onLine CD ∧ CD ≠ AB ∧ (between d e c) ∧ (between a e b) → (∠ a:e:c = ∠ d:e:b) ∧ (∠ c:e:b = ∠ a:e:d)"
    expected = "theorem example_thm (a b c d e : Point) (AB CD : Line) (h1 : distinctPointsOnLine a b AB) (h2 : distinctPointsOnLine c d CD) (h3 : e.onLine AB) (h4 : e.onLine CD) (h5 : CD ≠ AB) (h6 : (between d e c)) (h7 : (between a e b)) : (∠ a:e:c = ∠ d:e:b) ∧ (∠ c:e:b = ∠ a:e:d) := by sorry"
    assert unreformat_theorem(formula) == expected

def test_unreformat_with_tricky_hypothesis():
    formula = "∀ (a b c d e f g h : Point) (AB CD EF : Line), distinctPointsOnLine a b AB ∧ distinctPointsOnLine c d CD ∧ (∃ (x y : Z), x*x + y*y = c) ∧ (b.sameSide d EF) → ¬(AB.intersectsLine CD)"
    expected = "theorem example_thm (a b c d e f g h : Point) (AB CD EF : Line) (h1 : distinctPointsOnLine a b AB) (h2 : distinctPointsOnLine c d CD) (h3 : (∃ (x y : Z), x*x + y*y = c)) (h4 : (b.sameSide d EF)) : ¬(AB.intersectsLine CD) := by sorry"
    assert unreformat_theorem(formula) == expected

import SystemE

namespace Elements.Book1


-- theorem SAS_congruence (a b c d e f : Point) (AB AC DE DF : Line)
--   (h1 : |(a─b)| = |(d─e)|) (h2 : |(a─c)| = |(d─f)|)
--   (h3 : (∠ b:a:c : ℝ) = (∠ e:d:f : ℝ)) :
--   |(b─c)| = |(e─f)| ∧ Triangle.congruent (Triangle.ofPoints a b c) (Triangle.ofPoints d e f) ∧
--   (∠ a:b:c : ℝ) = (∠ d:e:f : ℝ) ∧ (∠ a:c:b : ℝ) = (∠ d:f:e : ℝ) := by

-- theorem SAS_congruence (a b c d e f : Point) (AB AC DE DF : Line) (h1 : |(a─b)| = |(d─e)|) (h2 : |(a─c)| = |(d─f)|) (h3 : (∠ b:a:c : ℝ) = (∠ e:d:f : ℝ)) : |(b─c)| = |(e─f)| ∧ Triangle.congruent (Triangle.ofPoints a b c) (Triangle.ofPoints d e f) ∧
--   (∠ a:b:c : ℝ) = (∠ d:e:f : ℝ) ∧ (∠ a:c:b : ℝ) = (∠ d:f:e : ℝ) := by


theorem proposition_1 : ∀ (a b : Point) (AB : Line),
  distinctPointsOnLine a b AB →
  ∃ c : Point, |(c─a)| = |(a─b)| ∧ |(c─b)| = |(a─b)| :=
by
  euclid_intros
  euclid_apply circle_from_points a b as BCD
  euclid_apply circle_from_points b a as ACE
  euclid_apply intersection_circles BCD ACE as c
  euclid_apply point_on_circle_onlyif a b c BCD
  euclid_apply point_on_circle_onlyif b a c ACE
  use c
  euclid_finish

theorem proposition_1' : ∀ (a b x : Point) (AB : Line),
  distinctPointsOnLine a b AB ∧ ¬(x.onLine AB) →
  ∃ c : Point, |(c─a)| = |(a─b)| ∧ |(c─b)| = |(a─b)| ∧ (c.opposingSides x AB) :=
by
  euclid_intros
  euclid_apply circle_from_points a b as BCD
  euclid_apply circle_from_points b a as ACE
  euclid_apply intersection_opposite_side BCD ACE x a b AB as c
  euclid_apply point_on_circle_onlyif a b c BCD
  euclid_apply point_on_circle_onlyif b a c ACE
  use c
  euclid_finish

end Elements.Book1

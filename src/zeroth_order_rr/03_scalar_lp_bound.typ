#import "../defs.typ": *

== Scalar $L^p$ Bound for the Zeroth-Order Term

The expression
$Delta J_n^((0, alpha))
  = -sum_(j=1)^(n-1) 2 alpha^2 overline(A) H_j^((n)) epsilon.alt(Z_j)$
is a weighted sum of values of the centered noise function $epsilon.alt$ along
the Markov chain. In the Berry--Esseen argument below only fixed scalar
projections are used, so we state the concentration input in scalar form.

#lemma[
  *(Weighted Markov concentration; Levin et al. (2025, Lemma 11).)*
  Assume *UGE 1*. Let ${g_i}_(i=1)^n$ be a family of measurable functions
  $g_i : Z -> bb(R)$ such that
  $c_i = ||g_i||_infinity < infinity$ and $pi(g_i) = 0$ for
  $i in {1, dots, n}$. Then, for any initial distribution $xi$ on
  $(Z, cal(Z))$, any $n in bb(N)$, and any $p >= 2$,
  $
  ||sum_(i=1)^n g_i(Z_i)||_(L_p(xi))
    <= C_("MC") sqrt(p thin t_"mix" thin sum_(i=1)^n c_i^2).
  $ <eq:zeroth-markov-conc>
] <lem:zeroth-weighted-markov-concentration>

// This is the scalar time-inhomogeneous form recorded as the external input
// @lem:external-markov-concentration; it is quoted here only to keep the
// preliminary calculation local in notation.

Fix a deterministic direction $u in bb(R)^d$. We apply the lemma with
$g_j^u(z) = -2 alpha^2 u^top overline(A) H_j^((n)) epsilon.alt(z)$ for $1 <= j <= n-1$ and $g_n^u = 0$. Each $g_j^u$ is centered under $pi$ since $pi(epsilon.alt) = 0$, so the centering hypothesis holds. Combining the bound $|| overline(A) || <= C_A$ (Assumption 2) with the previous subsection, define
the constant $K_(A,Q) := C_A K_Q = C_A kappa_Q$.
Then the per-summand bound is
$
||g_j^u||_infinity
// &<= 2 alpha^2 ||u|| K_(A,Q)
//     (1 - alpha a)^((n-j-1) slash 2)
//     frac(2, alpha a) ||epsilon.alt||_infinity
&<= frac(4 alpha ||u|| K_(A,Q) ||epsilon.alt||_infinity, a)
    (1 - alpha a)^((n-j-1) slash 2),
quad 1 <= j <= n-1.
$
Note the prefactor $alpha^2$ collapsing to $alpha slash a$: the $1 slash (alpha a)$ blow-up in $H_j^((n))$ is absorbed only by one factor of $alpha$. Squaring and summing over $j$:
$
sum_(j=1)^n ||g_j^u||_infinity^2
&= sum_(j=1)^(n-1) ||g_j^u||_infinity^2
&<= frac(16 alpha^2 ||u||^2 thin K_(A,Q)^2 ||epsilon.alt||_infinity^2, a^2) sum_(j=1)^(n-1) (1 - alpha a)^(n-j-1)
// &<= frac(16 alpha^2 ||u||^2 thin K_(A,Q)^2 ||epsilon.alt||_infinity^2, a^2) thin frac(1, alpha a) \
&<= frac(16 alpha ||u||^2 thin K_(A,Q)^2 ||epsilon.alt||_infinity^2, a^3).
$
Plugging this into the Markov concentration input
@eq:zeroth-markov-conc and defining
$
K_("RR",0) := 32 thin K_(A,Q) thin ||epsilon.alt||_infinity
  frac(sqrt(t_"mix"), a^(3 slash 2)),
$
gives
$
sqrt(p thin t_"mix" thin sum_(j=1)^n ||g_j^u||_infinity^2)
  <= sqrt(p) thin ||u|| thin K_("RR",0) thin sqrt(alpha).
$
Applying @eq:zeroth-markov-conc with
$X = u^top Delta J_n^((0, alpha))$ gives, for any $p >= 2$,
$
bb(E)^(1 slash p) [|u^top Delta J_n^((0, alpha))|^p]
  <= C_("MC") sqrt(p) thin ||u|| thin K_("RR",0) thin sqrt(alpha),
$
or equivalently
$
frac(1, sqrt(alpha)) thin ||u^top Delta J_n^((0, alpha))||_(L_p)
  <= C_("MC") sqrt(p) thin ||u|| thin K_("RR",0).
$
Thus every fixed scalar projection of the terminal-iterate zeroth-order RR
difference is $O(sqrt(alpha))$ in $L^p$, uniformly in $n$. A Euclidean-norm
bound can be recovered in fixed dimension by applying the scalar estimate to a
coordinate basis, but no dimension-free vector concentration is claimed here.

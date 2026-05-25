#import "defs.typ": *

== LSA Error Decomposition

We consider the recursion
$ theta_k = theta_(k-1) - alpha_k (A(Z_k) theta_(k-1) - b(Z_k)), quad alpha_k = alpha = "const". $

Define the transition products
$ Gamma_(m:k) = product_(l=m)^k (I - alpha A(Z_l)). $

Write $B_alpha := I - alpha overline(A)$ and introduce the deterministic-product
linearized term
$
J_k^((0, alpha)) = -alpha sum_(l=1)^k B_alpha^(k-l) thin epsilon.alt(Z_l),
quad J_0^((0, alpha)) = 0.
$
The difference between the exact random-product expansion and this
deterministic-product term is denoted by $R_k^((alpha))$:
$
theta_k^((alpha)) - theta^*
  = J_k^((0, alpha)) + B_alpha^k (theta_0 - theta^*) + R_k^((alpha)).
$
This convention matches the weight decomposition in Section 4.1.

A standard PR-averaged decomposition (cf. Chapter 4 for the full derivation) yields
$ sqrt(n) (overline(theta)_n^((alpha)) - theta^*) = W + D_1, $
where the leading martingale-like term is
$ W = -frac(1, sqrt(n)) sum_(l=1)^(n-1) Q_l thin epsilon.alt(Z_l),
quad
Q_l = alpha sum_(k=l)^(n-1) B_alpha^(k-l), $
and the residual term is
$ D_1 = frac(1, sqrt(n)) sum_(k=0)^(n-1) B_alpha^k (theta_0 - theta^*) + frac(1, sqrt(n)) sum_(k=1)^(n-1) R_k^((alpha)). $

== Last-Iterate RR Combination and the $tilde(J)_(n, "last")^((0, alpha))$ Term

Applying the deterministic-product decomposition of the previous subsection
separately to step sizes $alpha$ and $2 alpha$, and writing
$B_(2 alpha) := I - 2 alpha overline(A)$, define the chapter-local
last-iterate Richardson--Romberg object
$
theta_(n, "last")^(("RR", alpha)) := 2 theta_n^((alpha)) - theta_n^((2 alpha)).
$
It is not the PR-averaged RR estimator $overline(theta)_n^(("RR", alpha))$
studied in Chapters 4--5. Its deterministic-product decomposition has the form
$
theta_(n, "last")^(("RR", alpha)) - theta^*
&= [2 B_alpha^n - B_(2 alpha)^n](theta_0 - theta^*) \
&quad + [2 J_n^((0, alpha)) - J_n^((0, 2 alpha))]
  + [2 R_n^((alpha)) - R_n^((2 alpha))].
$
The first bracket is the deterministic transient in this convention. The
second bracket is the linearized stochastic RR difference, and the last one is
the higher-order random-product remainder. In this subsection we focus on the
zeroth-order RR difference, which we denote
$
tilde(J)_(n, "last")^((0, alpha)) = 2 J_n^((0, alpha)) - J_n^((0, 2 alpha)),
$
where, by definition,
$
J_n^((0, alpha)) = -alpha sum_(j=1)^n (I - alpha overline(A))^(n-j) epsilon.alt(Z_j).
$

Substituting and using the elementary identity $X^m - Y^m = (X - Y) sum_(i=1)^m X^(i-1) Y^(m-i)$ with $X = I - alpha overline(A)$, $Y = I - 2 alpha overline(A)$, $X - Y = alpha overline(A)$, we get
$
tilde(J)_(n, "last")^((0, alpha))
&= -2 alpha sum_(j=1)^n [(I - alpha overline(A))^(n-j) - (I - 2 alpha overline(A))^(n-j)] epsilon.alt(Z_j) \
&= -2 alpha^2 overline(A) sum_(j=1)^n
  underbrace(sum_(i=1)^(n-j) (I - alpha overline(A))^(i-1) (I - 2 alpha overline(A))^(n-j-i), =: H_j^((n)))
  epsilon.alt(Z_j) \
&= -2 alpha^2 overline(A) sum_(j=1)^(n-1) H_j^((n)) thin epsilon.alt(Z_j),
$
where the last equality uses $H_n^((n)) = 0$ because the inner sum is empty.
The left-factored matrix-power identity is legitimate here because
$I - alpha overline(A)$, $I - 2 alpha overline(A)$, and $overline(A)$ are
polynomials in the same matrix $overline(A)$ and therefore commute.
The extra factor $alpha overline(A)$ pulled out front is the source of the additional $alpha$-decay of the RR difference compared to a single LSA trajectory.

== Norm Estimate for $H_j^((n))$

To bound $||H_j^((n))||$ we use the following standard result.

#lemma[
  *(Lyapunov contraction.)*
  Let $-overline(A)$ be a Hurwitz matrix. Then for any $P = P^top succ 0$ there exists a unique
  $Q = Q^top succ 0$ satisfying
  $ overline(A)^top Q + Q overline(A) = P. $
  Moreover, letting
  $ a := frac(lambda_"min" (P), 2 ||Q||), quad
  alpha_infinity := min (frac(lambda_"min" (P), 2 kappa_Q ||overline(A)||_Q^2), frac(||Q||, lambda_"min" (P))), $
  with $kappa_Q := lambda_"max" (Q) slash lambda_"min" (Q)$, one has for all
  $alpha in [0, alpha_infinity]$:
  $ alpha a <= 1 slash 2, quad ||I - alpha overline(A)||_Q^2 <= 1 - alpha a. $
] <lem:lyapunov-contraction-local>

In the RR estimates below we assume explicitly that $0 < alpha$ and
$2 alpha <= alpha_infinity$. Hence the Lyapunov contraction applies at both
step sizes $alpha$ and $2 alpha$.

For $1 <= j <= n-1$, we estimate $||H_j^((n))||$ by combining the triangle inequality, submultiplicativity, the equivalence $|| X || <= kappa_Q^(1 slash 2) || X ||_Q$ (applied to *each* operator-norm factor, hence two factors of $kappa_Q^(1 slash 2)$ multiplying to $kappa_Q$), and the Lyapunov contraction at step sizes $alpha$ and $2 alpha$:
$
||H_j^((n))||
&<= sum_(i=1)^(n-j) ||I - alpha overline(A)||^(i-1) thin ||I - 2 alpha overline(A)||^(n-j-i)
  &&"(triangle + submult.)" \
&<= kappa_Q sum_(i=1)^(n-j) (1 - alpha a)^((i-1) slash 2) (1 - 2 alpha a)^((n-j-i) slash 2)
  &&"(equiv. + Lyapunov)" \
&= kappa_Q (1 - alpha a)^((n-j-1) slash 2) sum_(k=0)^(n-j-1) ((1 - 2 alpha a) / (1 - alpha a))^(k slash 2)
  &&"(reindex" k = n-j-i") " \
&<= kappa_Q (1 - alpha a)^((n-j-1) slash 2) frac(1, 1 - sqrt((1 - 2 alpha a) / (1 - alpha a)))
  &&"(geometric series)".
$
For $alpha a <= 1 slash 2$ the geometric rate is bounded below by an *elementary* inequality: combining $sqrt((1 - 2 alpha a) slash (1 - alpha a)) <= sqrt(1 - alpha a)$ (since $(1-2alpha a) <= (1-alpha a)^2$) with $1 - sqrt(1 - x) >= x slash 2$ on $[0, 1]$ gives
$
1 - sqrt(frac(1 - 2 alpha a, 1 - alpha a))
  >= 1 - sqrt(1 - alpha a)
  >= frac(alpha a, 2),
quad "i.e." quad
frac(1, 1 - sqrt((1 - 2 alpha a) / (1 - alpha a))) <= frac(2, alpha a).
$
Defining the chapter-local norm-equivalence constant
$ K_Q := kappa_Q $
(distinct from the assumption-2 sup-norm constant $C_A$, which enters separately via $|| overline(A) ||$ in the next subsection), we arrive at the final estimate
$
||H_j^((n))|| <= K_Q thin (1 - alpha a)^((n-j-1) slash 2) thin frac(2, alpha a).
$
The kernel decays geometrically in $n - j$ at rate $sqrt(1 - alpha a)$ but its summed weight is of order $1 slash (alpha a)$. The next subsection keeps the remaining powers of $a$ explicit when this kernel is multiplied by the prefactor $alpha^2$ in $tilde(J)_(n, "last")^((0, alpha))$.

== Scalar $L^p$ Bound for the Zeroth-Order Term

The expression
$
tilde(J)_(n, "last")^((0, alpha)) = -sum_(j=1)^(n-1) 2 alpha^2 overline(A) H_j^((n)) thin epsilon.alt(Z_j)
$
is a weighted sum of values of the centered noise function $epsilon.alt$ along the Markov chain. In the Berry--Esseen argument below only fixed scalar projections are used, so we state the concentration input in scalar form.

#lemma[
  *(Weighted Markov concentration.)*
  Assume *UGE 1*. Let ${g_i}_(i=1)^n$ be a family of measurable functions
  $g_i : Z -> bb(R)$ such that
  $ c_i = ||g_i||_infinity < infinity quad "for all" i >= 1, quad
  pi(g_i) = 0 quad "for all" i in {1, dots, n}. $
  Then, for any initial distribution $xi$ on $(Z, cal(Z))$, any
  $n in bb(N)$, and any $p >= 2$,
  $
  ||sum_(i=1)^n g_i(Z_i)||_(L_p(xi))
    <= C_("MC,0") sqrt(p thin t_"mix" thin sum_(i=1)^n c_i^2).
  $ <eq:zeroth-markov-conc>

  _Proof._ This is the local scalar specialization of the imported Markov
  concentration statement @lem:imported-markov-concentration. The only
  dependence on the coefficient sequence is
  through $sqrt(p thin t_"mix" thin sum_i c_i^2)$. Their result assumes the
  centering condition $pi(g_i)=0$ and is stated for arbitrary initial law
  $xi$, so no additional initial-bias term is introduced here.
] <lem:zeroth-weighted-markov-concentration>

Fix a deterministic direction $u in bb(R)^d$. We apply the lemma with
$g_j^u(z) = -2 alpha^2 u^top overline(A) H_j^((n)) epsilon.alt(z)$ for $1 <= j <= n-1$ and $g_n^u = 0$. Each $g_j^u$ is centered under $pi$ since $pi(epsilon.alt) = 0$, so the centering hypothesis holds. Combining the bound $|| overline(A) || <= C_A$ (Assumption 2) with the previous subsection, define
$
K_(A,Q) := C_A thin K_Q = C_A thin kappa_Q.
$
Then the per-summand bound is
$
||g_j^u||_infinity
&<= 2 alpha^2 ||u|| thin K_(A,Q) (1 - alpha a)^((n-j-1) slash 2) frac(2, alpha a) thin ||epsilon.alt||_infinity \
&= frac(4 alpha ||u|| thin K_(A,Q) ||epsilon.alt||_infinity, a) (1 - alpha a)^((n-j-1) slash 2),
quad 1 <= j <= n-1.
$
Note the prefactor $alpha^2$ collapsing to $alpha slash a$: the $1 slash (alpha a)$ blow-up in $H_j^((n))$ is absorbed only by one factor of $alpha$. Squaring and summing over $j$:
$
sum_(j=1)^n ||g_j^u||_infinity^2
&= sum_(j=1)^(n-1) ||g_j^u||_infinity^2 \
&<= frac(16 alpha^2 ||u||^2 thin K_(A,Q)^2 ||epsilon.alt||_infinity^2, a^2) sum_(j=1)^(n-1) (1 - alpha a)^(n-j-1) \
&<= frac(16 alpha^2 ||u||^2 thin K_(A,Q)^2 ||epsilon.alt||_infinity^2, a^2) thin frac(1, alpha a) \
&= frac(16 alpha ||u||^2 thin K_(A,Q)^2 ||epsilon.alt||_infinity^2, a^3).
$
Plugging this into the Markov concentration input
@eq:zeroth-markov-conc and defining
$
K_("last",0) := 32 thin K_(A,Q) thin ||epsilon.alt||_infinity
  frac(sqrt(t_"mix"), a^(3 slash 2)),
$
gives
$
sqrt(p thin t_"mix" thin sum_(j=1)^n ||g_j^u||_infinity^2)
  <= sqrt(p) thin ||u|| thin K_("last",0) thin sqrt(alpha).
$
Applying @eq:zeroth-markov-conc with
$X = u^top tilde(J)_(n, "last")^((0, alpha))$ gives, for any $p >= 2$,
$
bb(E)^(1 slash p) [|u^top tilde(J)_(n, "last")^((0, alpha))|^p]
  <= C_("MC,0") sqrt(p) thin ||u|| thin K_("last",0) thin sqrt(alpha),
$
or equivalently
$
frac(1, sqrt(alpha)) thin ||u^top tilde(J)_(n, "last")^((0, alpha))||_(L_p)
  <= C_("MC,0") sqrt(p) thin ||u|| thin K_("last",0).
$
Thus every fixed scalar projection of the last-iterate zeroth-order RR
difference is $O(sqrt(alpha))$ in $L^p$, uniformly in $n$. A Euclidean-norm
bound can be recovered in fixed dimension by applying the scalar estimate to a
coordinate basis, but no dimension-free vector concentration is claimed here.

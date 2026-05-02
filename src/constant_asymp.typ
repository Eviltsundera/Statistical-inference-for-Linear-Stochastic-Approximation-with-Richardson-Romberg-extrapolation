#import "defs.typ": *

== LSA Error Decomposition

We consider the recursion
$ theta_k = theta_(k-1) - alpha_k (A(Z_k) theta_(k-1) - b(Z_k)), quad alpha_k = alpha = "const". $

Define the transition products
$ Gamma_(m:k) = product_(l=m)^k (I - alpha A(Z_l)). $

We also introduce
$ J_k^((0)) = -sum_(l=1)^k alpha thin Gamma_(l+1:k) thin epsilon.alt(Z_l), quad
H_k^((0)) = -sum_(l=1)^k alpha thin Gamma_(l+1:k) (A(Z_l) - overline(A)) J_(l-1)^((0)), $
with initial conditions $J_0^((0)) = H_0^((0)) = 0$.

A standard decomposition yields
$ sqrt(n) (theta_n^((alpha)) - theta^*) = W + D_1, $
where
$ W = frac(1, sqrt(n)) sum_(l=1)^(n-1) Q_l thin epsilon.alt(Z_l), quad
D_1 = frac(1, sqrt(n)) sum_(k=0)^(n-1) Gamma_(1:k) (theta_0 - theta^*) + frac(1, sqrt(n)) sum_(k=1)^(n-1) J_k^((0)), $
and
$ Q_l = alpha sum_(k=l)^(n-1) (I - alpha overline(A))^(n-k-1). $

== RR Combination and the $tilde(J)_n^((0, alpha))$ Term

Applying the decomposition of the previous subsection separately to step sizes $alpha$ and $2 alpha$ and forming the Richardson--Romberg combination, we obtain
$
theta_n^(("RR", alpha)) - theta^*
&= [2 Gamma_(1:n)^((alpha)) - Gamma_(1:n)^((2 alpha))](theta_0 - theta^*) \
&quad + [2 J_n^((0, alpha)) - J_n^((0, 2 alpha))]
  + [2 J_n^((1, alpha)) - J_n^((1, 2 alpha))]
  + [2 H_n^((1, alpha)) - H_n^((1, 2 alpha))].
$
The first bracket is the deterministic transient and is exponentially small. The remaining three brackets are stochastic and need separate analysis. In this subsection we focus on the zeroth-order RR difference, which we denote
$
tilde(J)_n^((0, alpha)) = 2 J_n^((0, alpha)) - J_n^((0, 2 alpha)),
$
where, by definition,
$
J_n^((0, alpha)) = -alpha sum_(j=1)^n (I - alpha overline(A))^(n-j) epsilon.alt(Z_j).
$

Substituting and using the elementary identity $X^m - Y^m = (X - Y) sum_(i=1)^m X^(i-1) Y^(m-i)$ with $X = I - alpha overline(A)$, $Y = I - 2 alpha overline(A)$, $X - Y = alpha overline(A)$, we get
$
tilde(J)_n^((0, alpha))
&= -2 alpha sum_(j=1)^n [(I - alpha overline(A))^(n-j) - (I - 2 alpha overline(A))^(n-j)] epsilon.alt(Z_j) \
&= -2 alpha^2 overline(A) sum_(j=1)^n
  underbrace(sum_(i=1)^(n-j) (I - alpha overline(A))^(i-1) (I - 2 alpha overline(A))^(n-j-i), =: H_j^((n)))
  epsilon.alt(Z_j) \
&= -2 alpha^2 overline(A) sum_(j=1)^n H_j^((n)) thin epsilon.alt(Z_j).
$
The extra factor $alpha overline(A)$ pulled out front is the source of the additional $alpha$-decay of the RR difference compared to a single LSA trajectory.

== Norm Estimate for $H_j^((n))$

To bound $||H_j^((n))||$ we use the following standard result.

#lemma[
  Let $-overline(A)$ be a Hurwitz matrix. Then for any $P = P^top succ 0$ there exists a unique
  $Q = Q^top succ 0$ satisfying
  $ overline(A)^top Q + Q overline(A) = P. $
  Moreover, letting
  $ a = frac(lambda_"min" (P), 2 ||Q||), quad
  alpha_infinity = frac(lambda_"min" (P), 2 kappa_Q ||overline(A)||_Q^2) and frac(||Q||, lambda_"min" (P)), $
  with $kappa_Q = lambda_"max" (Q) slash lambda_"min" (Q)$, one has for all
  $alpha in [0, alpha_infinity]$:
  $ alpha a <= 1 slash 2, quad ||I - alpha overline(A)||_Q^2 <= 1 - alpha a. $
]

We now estimate $||H_j^((n))||$ by combining the triangle inequality, submultiplicativity, the Lyapunov contraction (applied at step sizes $alpha$ and $2 alpha$), and the equivalence of $||dot||$ and $||dot||_Q$ which absorbs $kappa_Q$ into a generic constant $C_A$:
$
||H_j^((n))||
&<= sum_(i=1)^(n-j) ||I - alpha overline(A)||^(i-1) thin ||I - 2 alpha overline(A)||^(n-j-i)
  quad &"(triangle + submultiplicativity)" \
&<= C_A sum_(i=1)^(n-j) (1 - alpha a)^((i-1) slash 2) (1 - 2 alpha a)^((n-j-i) slash 2)
  quad &"(Lyapunov contraction)" \
&= C_A (1 - alpha a)^((n-j-1) slash 2) sum_(k=0)^(n-j-1) ((1 - 2 alpha a) / (1 - alpha a))^(k slash 2)
  quad &"(reindex" k = n-j-i") " \
&<= C_A (1 - alpha a)^((n-j-1) slash 2) frac(1, 1 - sqrt((1 - 2 alpha a) / (1 - alpha a)))
  quad &"(geometric series)".
$
It remains to bound the geometric rate $1 - sqrt((1 - 2 alpha a) / (1 - alpha a))$ from below. Writing $(1 - 2 alpha a) / (1 - alpha a) = 1 - alpha a / (1 - alpha a)$ and Taylor-expanding $sqrt(1 - x)$ at $x = alpha a / (1 - alpha a)$ gives
$
sqrt(frac(1 - 2 alpha a, 1 - alpha a))
= 1 - frac(alpha a, 2 (1 - alpha a)) + O((alpha a)^2)
= 1 - frac(1, 2) alpha a + O(alpha^2),
$
where the second equality uses $alpha a <= 1 slash 2$. Consequently,
$
1 - sqrt(frac(1 - 2 alpha a, 1 - alpha a)) = frac(1, 2) alpha a + O(alpha^2),
quad
frac(1, 1 - sqrt((1 - 2 alpha a) / (1 - alpha a))) = frac(2, alpha a) + O(1).
$
Folding the $O(1)$ correction into a constant $overline(C)_A$, we arrive at the final estimate
$
||H_j^((n))|| <= overline(C)_A thin (1 - alpha a)^((n-j-1) slash 2) thin frac(2, alpha a).
$
The kernel decays geometrically in $n - j$ at rate $sqrt(1 - alpha a)$ but its summed weight is of order $1 slash (alpha a)$. The next subsection shows how the prefactor $alpha^2$ in $tilde(J)_n^((0, alpha))$ exactly compensates this divergence.

== $L^2$ Bound for the Zeroth-Order Term

The expression
$
tilde(J)_n^((0, alpha)) = -sum_(j=1)^n 2 alpha^2 overline(A) H_j^((n)) thin epsilon.alt(Z_j)
$
is a weighted sum of values of the centered noise function $epsilon.alt$ along the Markov chain. We control its tail via the following Markov concentration statement.

#lemma[
  Assume *UGE 1*. Let ${g_i}_(i=1)^n$ be a family of measurable functions
  $g_i : Z -> bb(R)^d$ such that
  $ c_i = ||g_i||_infinity < infinity quad "for all" i >= 1, quad
  pi(g_i) = 0 quad "for all" i in {1, dots, n}. $
  Then, for any initial distribution $xi$ on $(Z, cal(Z))$, any $n in bb(N)$ and any $t >= 0$,
  $ bb(P)_xi (lr(||sum_(i=1)^n g_i (Z_i)||) >= t) <= 2 exp(-frac(t^2, 2 u_n^2)), $
  where
  $ u_n = 8 (sum_(i=1)^n c_i^2)^(1 slash 2) sqrt(t_"mix"). $

  _Proof._ The proof follows the lines of Durmus et al. (2025, Lemma 9).
]

We apply the lemma with $g_j(z) = -2 alpha^2 overline(A) H_j^((n)) epsilon.alt(z)$. Each $g_j$ is centered under $pi$ since $pi(epsilon.alt) = 0$, so the centering hypothesis holds. Plugging the bound on $||H_j^((n))||$ from the previous subsection and writing $tilde(C)_A = ||A|| thin overline(C)_A$ yields the per-summand bound
$
||g_j||_infinity
&<= 2 alpha^2 tilde(C)_A (1 - alpha a)^((n-j-1) slash 2) frac(2, alpha a) thin ||epsilon.alt||_infinity
= 4 alpha tilde(C)_A ||epsilon.alt||_infinity (1 - alpha a)^((n-j-1) slash 2).
$
Note the prefactor $alpha^2$ collapsing to $alpha$: the $1 slash (alpha a)$ blow-up in $H_j^((n))$ is exactly absorbed by one factor of $alpha$. Squaring and summing over $j$, we get a geometric series whose closed form is bounded uniformly in $n$:
$
sum_(j=1)^n ||g_j||_infinity^2
&<= 16 alpha^2 tilde(C)_A^2 ||epsilon.alt||_infinity^2 sum_(j=1)^n (1 - alpha a)^(n-j-1) \
&<= 16 alpha^2 tilde(C)_A^2 ||epsilon.alt||_infinity^2 thin frac(1, (1 - alpha a) alpha a)
<= frac(C^2, a) thin alpha thin ||epsilon.alt||_infinity^2,
$
where in the last step we used $1 - alpha a >= 1 slash 2$. Plugging this into the variance proxy $u_n$ from the lemma absorbs the constants $64 dot 16$ and $sqrt(t_"mix")$ into a constant $hat(C)_A$ depending on $a$, $t_"mix"$, and $||epsilon.alt||_infinity$:
$
u_n^2 <= hat(C)_A^2 thin alpha.
$

To convert the sub-Gaussian tail into a moment bound, we use the following standard fact.

#lemma[
  Let $X$ be an $bb(R)^d$-valued random variable satisfying
  $ bb(P)(||X|| >= t) <= 2 exp(-frac(t^2, 2 sigma^2)) quad "for all" t >= 0, $
  for some $sigma^2 > 0$. Then, for any $p >= 2$, it holds that
  $ bb(E)[||X||^p] <= 2 thin p^(p slash 2) thin sigma^p. $
]

Applying the lemma with $sigma^2 = u_n^2 <= hat(C)_A^2 alpha$ and $p = 2$ gives
$
bb(E)^(1 slash 2)[||tilde(J)_n^((0, alpha))||^2]
<= 2 sqrt(alpha) thin hat(C)_A,
quad "i.e." quad
frac(1, sqrt(alpha)) thin bb(E)^(1 slash 2)[||tilde(J)_n^((0, alpha))||^2] <= 2 hat(C)_A.
$
The zeroth-order RR difference is therefore $O(sqrt(alpha))$ in $L^2$, uniformly in $n$ — an order-$sqrt(alpha)$ gain over a single LSA trajectory, which is $O(1)$ in $L^2$ in the stationary regime.

== Richardson--Romberg PR Weight Bounds

The Polyak--Ruppert version of the LSA expansion replaces the trajectory $theta_k - theta^*$ by its time average and rewrites the leading martingale contribution as a single sum
$
W = -frac(1, sqrt(n)) sum_(l = 1)^(n - 1) Q_l^((alpha)) thin epsilon.alt(Z_l),
quad
Q_l^((alpha))
  = alpha sum_(k = l)^(n - 1) (I - alpha overline(A))^(n - k - 1),
$
so that $Q_l^((alpha))$ collects the deterministic-product weight with which the noise $epsilon.alt(Z_l)$ enters the PR average. (For constant step this is the specialization of the time-dependent kernel
introduced in Samsonov et al. (2025).) Coupling two trajectories with steps $alpha$ and $2 alpha$
through the same noise realization $\{Z_k\}$ and forming the Richardson--Romberg combination,
$
cal(Q)_l^("RR") := 2 Q_l^((alpha)) - Q_l^((2 alpha)),
$
we obtain the leading martingale weight for $sqrt(n)(bar(theta)_n^(("RR", alpha)) - theta^*)$. The Berry--Esseen analysis of the RR-averaged iterate hinges on two contraction estimates for $cal(Q)_l^("RR")$: comparison with the asymptotic weight $overline(A)^(-1)$, and a bound on the total variation of the successive differences $cal(Q)_(l+1)^("RR") - cal(Q)_l^("RR")$. The latter controls the Abel-summation term in the Poisson-equation remainder.

Throughout this subsection write
$
B_alpha := I - alpha overline(A),
quad
B_(2 alpha) := I - 2 alpha overline(A),
quad
k := n - l.
$
We assume $alpha, 2 alpha in (0, alpha_infinity]$, so that the Lyapunov contraction
$|| B_alpha^m ||_Q^2 <= (1 - alpha a)^m$ and $|| B_(2 alpha)^m ||_Q^2 <= (1 - 2 alpha a)^m$ hold (and we use freely $1 - 2 alpha a <= 1 - alpha a$).

The geometric-series identity gives the closed form
$
Q_l^((alpha))
  = alpha sum_(j = 0)^(n - l - 1) B_alpha^j
  = alpha thin (alpha overline(A))^(-1) thin (I - B_alpha^(n - l))
  = overline(A)^(-1) (I - B_alpha^(n - l)).
$
Two immediate consequences are
$
Q_l^((alpha)) - overline(A)^(-1) = - overline(A)^(-1) B_alpha^(n - l),
quad
Q_(l + 1)^((alpha)) - Q_l^((alpha)) = - alpha B_alpha^(n - l - 1).
$
The first identity, applied to $alpha$ and $2 alpha$, yields the basic RR identity
$
cal(Q)_l^("RR") - overline(A)^(-1)
  = - overline(A)^(-1) thin (2 B_alpha^k - B_(2 alpha)^k),
$
and the second yields the discrete-difference identity
$
cal(Q)_(l + 1)^("RR") - cal(Q)_l^("RR")
  = - 2 alpha thin (B_alpha^(k - 1) - B_(2 alpha)^(k - 1)).
$

#lemma[
  Let $alpha, 2 alpha in (0, alpha_infinity]$, set $C_Q := kappa_Q^(1 slash 2) || overline(A)^(-1) ||$ and $tilde(C)_A := kappa_Q || overline(A) ||$, and write $k = n - l$.

  *(i)* For every $1 <= l <= n - 1$,
  $
  || cal(Q)_l^("RR") - overline(A)^(-1) || <= 3 C_Q (1 - alpha a)^(k slash 2).
  $

  *(ii)* For every $1 <= l <= n - 2$,
  $
  || cal(Q)_(l + 1)^("RR") - cal(Q)_l^("RR") ||
    <= 2 tilde(C)_A thin alpha^2 thin (k - 1) thin (1 - alpha a)^((k - 2) slash 2).
  $
]

_Proof of (i)._ Take norms in the basic RR identity above. Using the equivalence $|| dot || <= kappa_Q^(1 slash 2) || dot ||_Q$ and the Lyapunov contraction at both step sizes,
$
|| 2 B_alpha^k - B_(2 alpha)^k ||_Q
  <= 2 || B_alpha^k ||_Q + || B_(2 alpha)^k ||_Q
  <= 2 (1 - alpha a)^(k slash 2) + (1 - 2 alpha a)^(k slash 2)
  <= 3 (1 - alpha a)^(k slash 2).
$
Submultiplicativity in $|| dot ||_Q$ followed by norm equivalence then gives
$
|| cal(Q)_l^("RR") - overline(A)^(-1) ||
  <= kappa_Q^(1 slash 2) || overline(A)^(-1) || dot || 2 B_alpha^k - B_(2 alpha)^k ||_Q
  <= 3 C_Q (1 - alpha a)^(k slash 2).
$

_Proof of (ii)._ Apply the elementary identity
$X^m - Y^m = (X - Y) sum_(i = 1)^m X^(i - 1) Y^(m - i)$
with $X = B_alpha$, $Y = B_(2 alpha)$, $X - Y = alpha overline(A)$, and $m = k - 1$:
$
B_alpha^(k - 1) - B_(2 alpha)^(k - 1)
  = alpha overline(A) sum_(i = 1)^(k - 1)
    B_alpha^(i - 1) thin B_(2 alpha)^(k - 1 - i).
$
Each summand obeys, by submultiplicativity in $|| dot ||_Q$ and Lyapunov contraction,
$
|| B_alpha^(i - 1) thin B_(2 alpha)^(k - 1 - i) ||
  &<= kappa_Q^(1 slash 2) thin (1 - alpha a)^((i - 1) slash 2) (1 - 2 alpha a)^((k - 1 - i) slash 2) \
  &<= kappa_Q^(1 slash 2) thin (1 - alpha a)^((k - 2) slash 2),
$
where in the last step we used $1 - 2 alpha a <= 1 - alpha a$. Summing over $i in {1, dots, k - 1}$ and absorbing constants,
$
|| B_alpha^(k - 1) - B_(2 alpha)^(k - 1) ||
  <= alpha kappa_Q || overline(A) || (k - 1) (1 - alpha a)^((k - 2) slash 2).
$
Inserting this bound into the discrete-difference identity finishes the proof. $square$

#corollary[
  Under the assumptions of the previous lemma, uniformly in $n >= 2$,
  $
  sum_(l = 1)^(n - 1) || cal(Q)_l^("RR") - overline(A)^(-1) ||^2
    <= frac(C_1, alpha a),
  quad
  sum_(l = 1)^(n - 2) || cal(Q)_(l + 1)^("RR") - cal(Q)_l^("RR") ||
    <= frac(C_2, a^2),
  $
  with $C_1 = 9 C_Q^2$ and $C_2 = 32 tilde(C)_A$.
]

_Proof._ For the first sum apply (i) and the geometric series $sum_(k >= 1) (1 - alpha a)^k <= 1 / (alpha a)$. For the second, apply (ii) and
$
sum_(k >= 2) (k - 1) thin (1 - alpha a)^((k - 2) slash 2)
  = sum_(m >= 0) (m + 1) (1 - alpha a)^(m slash 2)
  <= frac(1, (1 - sqrt(1 - alpha a))^2)
  <= frac(4, (alpha a)^2),
$
where the last step uses $1 - sqrt(1 - alpha a) >= alpha a / 2$ for $alpha a <= 1 slash 2$. Multiplying by $2 tilde(C)_A alpha^2$ gives the claim, up to the stated universal constant. $square$

#remark[
  Comparison with the single-step case. The same identities for $Q_l^((alpha))$ give
  $|| Q_l^((alpha)) - overline(A)^(-1) || <= C_Q (1 - alpha a)^(k slash 2)$ and
  $|| Q_(l + 1)^((alpha)) - Q_l^((alpha)) || <= alpha kappa_Q^(1 slash 2) (1 - alpha a)^((k - 1) slash 2)$.
  Part (i) of the lemma above is therefore the same order as the single-step bound: the leading
  $- overline(A)^(-1)$ term of $cal(Q)_(n - 1)^("RR")$ is exactly $- overline(A)^(-1)$ (since
  $cal(Q)_(n - 1)^("RR") = 0$ identically), so no factor of $alpha$ is gained on the boundary.
  The Richardson--Romberg cancellation is visible in the successive difference of
  (ii): for fixed lag $k$ the discrete derivative gains an extra factor of $alpha$.
  This local gain comes with the factor $k - 1$, so after summing over the geometric
  tail it should not be read as a global extra factor of $alpha$ in the Abel term.
  What is needed for the Poisson-equation remainder is the uniform total-variation
  bound in the corollary. On the variance side the corollary gives
  $sum || cal(Q)_l^("RR") - overline(A)^(-1) ||^2 = O(1 / (alpha a))$, which yields
  the finite-horizon comparison error $O(1 / (n alpha a))$.
]

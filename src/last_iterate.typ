#import "defs.typ": *

== Centered Bound for the Shifted First-Order Perturbation

The purpose of this section is to isolate the last-iteration weighted term
which appears in the analysis of $J_n^((1, alpha))$, and to record a clean
$L_p$ bound for its centered part. The proof uses the deterministic-product
perturbation expansion from Samsonov et al. (2025, Proposition 9), specialized
to the constant-stepsize setting, and isolates the future-centered bilinear
concentration input used in that argument.

Let
$
B = I - alpha overline(A).
$
Throughout this section we assume $0 < alpha <= alpha_infinity$, so that
$
||B^m||_Q <= (1 - alpha a)^(m slash 2),
quad m >= 0.
$
When the shifted estimate is transferred back from $T_n^((1, alpha))$ to
$J_n^((1, alpha))$, we also use the elementary inverse-admissibility ceiling
$
alpha_("inv") := frac(1, 2 || overline(A) ||).
$
Indeed, if $w <= alpha_("inv")$, then the Neumann series gives
$|| (I - w overline(A))^(-1) || <= 2$.
The single-stepsize estimates below are stated for $alpha$. Whenever the
Richardson--Romberg combination is formed, the same estimates are applied at
both $w = alpha$ and $w = 2 alpha$; the local assumptions are then
$2 alpha <= alpha_infinity$ and $2 alpha <= alpha_("inv")$.
All constants denoted by $C$ may depend on $C_A$, $kappa_Q$, and norm
equivalence constants, but not on $alpha$, $n$, $p$, or $t_"mix"$.

The deterministic-product components are defined by
$
J_n^((0, alpha))
  = B J_(n-1)^((0, alpha)) - alpha epsilon.alt(Z_n),
quad
J_0^((0, alpha)) = 0,
$
and
$
J_n^((1, alpha))
  = B J_(n-1)^((1, alpha))
    - alpha tilde(A)(Z_n) J_(n-1)^((0, alpha)),
quad
J_0^((1, alpha)) = 0.
$
Hence
$
J_n^((0, alpha))
  = - alpha sum_(k=1)^n B^(n-k) epsilon.alt(Z_k),
$
and
$
J_n^((1, alpha))
&= - alpha sum_(j=1)^n
      B^(n-j) tilde(A)(Z_j) J_(j-1)^((0, alpha)) \
&= alpha^2 sum_(1 <= k < j <= n)
      B^(n-j) tilde(A)(Z_j) B^(j-1-k) epsilon.alt(Z_k) \
&= alpha^2 sum_(k=1)^(n-1) sum_(r=1)^(n-k)
      B^(n-k-r) tilde(A)(Z_(k+r)) B^(r-1) epsilon.alt(Z_k).
$
The sign in the second line is positive because
$J_(j-1)^((0, alpha))$ already contains the minus sign.

The last display is for $J_n^((1, alpha))$ itself. The last-iteration
estimate below operates on a *shifted* version, obtained by inserting one
additional left factor $B$ into the recursion:
$
S_n
  = sum_(t=0)^(n-1)
      B^(n-t) tilde(A)(Z_(t+1)) J_t^((0, alpha)).
$
Reindexing $j = t + 1$ rewrites $S_n$ in the same form as $J_n^((1, alpha))$
but with one extra power of $B$:
$
S_n
  = sum_(j=1)^n B^(n-j+1) tilde(A)(Z_j) J_(j-1)^((0, alpha))
  = - frac(1, alpha) thin B thin J_n^((1, alpha)).
$
Thus the corresponding shifted first-order contribution is
$
T_n^((1, alpha)) = - alpha S_n = B thin J_n^((1, alpha)),
$
i.e. $T_n^((1, alpha))$ is exactly $J_n^((1, alpha))$ pre-multiplied by one
additional $B$. Transferring a bound from $T_n^((1, alpha))$ back to
$J_n^((1, alpha))$ therefore requires a local inverse bound on
$B^(-1) = (I - alpha overline(A))^(-1)$; this inverse-bound step is used
explicitly when the shifted estimate is applied below.

#lemma[
  *(Imported future-centered bilinear estimate.)*
  Assume UGE 1 and stationarity. Let $cal(F)_k = sigma(Z_1, dots, Z_k)$.
  For $1 <= k <= n - 1$ and $1 <= l <= n - k$, let
  $g_(k,l) : cal(Z) -> bb(R)^d$ be deterministic functions satisfying
  $pi(g_(k,l)) = 0$ and $||g_(k,l)||_infinity <= beta_(k,l)$. Let
  $xi_k$ be $cal(F)_k$-measurable vectors with
  $sup_k ||xi_k||_infinity <= M_xi$. Then, for every $p >= 2$,
  $
  &|| sum_(k=1)^(n-1) {
      sum_(l=1)^(n-k) g_(k,l)(Z_(k+l))^top xi_k
      - bb(E) lr([
          sum_(l=1)^(n-k) g_(k,l)(Z_(k+l))^top xi_k
          thin | thin cal(F)_k
        ])
    } ||_(L_p) \
  &quad <= C p^(3 slash 2) t_"mix"^(1 slash 2) M_xi
     lr((sum_(k=1)^(n-1) sum_(l=1)^(n-k) beta_(k,l)^2))^(1 slash 2).
  $
  This is the scalar, constant-stepsize specialization of the
  block-decomposition and Berbee-coupling estimate used in Samsonov et al.
  (2025, Appendix D.2, Proposition 9; see in particular their Lemma 11 and the
  treatment of the coupled term $T_(21) + T_(22)$). The centering in the
  display is the conditional centering with respect to $cal(F)_k$; no
  stationary-centering inequality is applied to a future chain started from
  $Z_k$.
]

#lemma[
  Assume the Markov chain is started from stationarity, that is, the law of
  $Z_1$ is $pi$, and $pi(tilde(A)) = 0$. For every deterministic direction
  $u in bb(R)^d$ and every $p >= 2$,
  $
  ||u^top (S_n - bb(E)S_n)||_(L_p)
    <= C ||u|| thin ||epsilon.alt||_infinity
      (p^(3 slash 2) t_"mix"^(1 slash 2) frac(1, a)
        + p^(1 slash 2) t_"mix"^(3 slash 2) sqrt(frac(alpha, a))).
  $
  Consequently,
  $
  ||u^top (T_n^((1, alpha)) - bb(E)T_n^((1, alpha)))||_(L_p)
    <= C alpha ||u|| thin ||epsilon.alt||_infinity
      (p^(3 slash 2) t_"mix"^(1 slash 2) frac(1, a)
        + p^(1 slash 2) t_"mix"^(3 slash 2) sqrt(frac(alpha, a))).
  $
]

_Proof._ Since $J_0^((0, alpha)) = 0$, the term $t = 0$ in $S_n$ vanishes.
Substituting the explicit formula for $J_t^((0, alpha))$ gives
$
S_n
&= - alpha sum_(t=1)^(n-1) sum_(k=1)^t
    B^(n-t) tilde(A)(Z_(t+1)) B^(t-k) epsilon.alt(Z_k) \
&= - alpha sum_(k=1)^(n-1)
    H_(k+1)^((w)) epsilon.alt(Z_k),
$
where, after the change of summation index $l = t - k + 1$,
$
H_(k+1)^((w))
  = sum_(l=1)^(n-k)
      B^(n-k-l+1) tilde(A)(Z_(k+l)) B^(l-1).
$
The kernel $H_(k+1)^((w))$ acts on the past noise $epsilon.alt(Z_k)$ through future states $Z_(k+l)$, so $H_(k+1)^((w)) epsilon.alt(Z_k)$ is a future-weighted bilinear functional of the trajectory.

Define
$
mu_k^((w)) = bb(E)_pi lr([H_(k+1)^((w)) epsilon.alt(Z_k)]).
$
Then
$
S_n - bb(E)S_n
  = - alpha sum_(k=1)^(n-1)
      {H_(k+1)^((w)) epsilon.alt(Z_k) - mu_k^((w))}.
$

Let $cal(F)_k = sigma(Z_1, dots, Z_k)$. The Markov property gives
$
bb(E)[H_(k+1)^((w)) epsilon.alt(Z_k) | cal(F)_k]
  = v_k^((w, epsilon))(Z_k),
$
where
$
v_k^((w))(z)
  = sum_(l=1)^(n-k)
      B^(n-k-l+1) (sans(Q)^l tilde(A))(z) B^(l-1),
quad
v_k^((w, epsilon))(z)
  = v_k^((w))(z) epsilon.alt(z).
$
Here $sans(Q)$ denotes the one-step Markov transition kernel of $(Z_k)_(k>=1)$,
acting on bounded matrix-valued functions by integration against the
conditional law:
$
(sans(Q) tilde(A))(z)
  = integral tilde(A)(u) thin sans(Q)(z, thin d u)
  = bb(E) lr([tilde(A)(Z_(k+1)) | Z_k = z]),
$
and $sans(Q)^l$ is its $l$-fold iterate, the $l$-step kernel
$sans(Q)^l (z, thin d u) = bb(P)(Z_(k+l) in d u | Z_k = z)$, so that
$
(sans(Q)^l tilde(A))(z)
  = integral tilde(A)(u) thin sans(Q)^l (z, thin d u)
  = bb(E) lr([tilde(A)(Z_(k+l)) | Z_k = z]).
$
In particular $(sans(Q)^l tilde(A))(z) -> pi(tilde(A)) = 0$ at the geometric
rate dictated by UGE, which is the only fact about $sans(Q)^l$ used below.

Under stationarity,
$mu_k^((w)) = pi(v_k^((w, epsilon)))$. Therefore
$
S_n - bb(E)S_n = - alpha (U_M + U_R),
$
with
$
U_M
  = sum_(k=1)^(n-1)
      {H_(k+1)^((w)) epsilon.alt(Z_k)
       - v_k^((w, epsilon))(Z_k)}
$
and
$
U_R
  = sum_(k=1)^(n-1)
      {v_k^((w, epsilon))(Z_k)
       - pi(v_k^((w, epsilon)))}.
$
This decomposition splits the centered statistic into two structurally different parts: $U_R$ is an ordinary centered additive functional of the original Markov chain, while $U_M$ is a future-centered bilinear term — the conditional expectation given $cal(F)_k$ vanishes summand-wise, but the summands are not forward martingale differences. We bound each piece separately.

Fix a deterministic direction $u$. The case $u = 0$ is trivial. By homogeneity,
it is enough to prove the estimate for $||u|| = 1$; the general statement
follows by multiplying the right-hand side by $||u||$.

_Step 1: bound on $u^top U_R$._ Because $pi(tilde(A)) = 0$, applying $sans(Q)^l$ followed by integration against $pi$ replaces $tilde(A)$ by its $l$-step propagation away from stationarity. UGE then gives the geometric Dobrushin bound
$
||(sans(Q)^l tilde(A))(z)||
  = lr(|| integral tilde(A)(y) {sans(Q)^l (z, thin d y) - pi(d y)} ||)
  <= 2 C_A Delta(sans(Q)^l)
  <= 2 C_A (1 slash 4)^(floor(l slash t_"mix")).
$
Inserting this into $v_k^((w))$ and factoring out the slow rate $(1 - alpha a)^((n-k) slash 2)$ from the two $B$-powers, the inner sum becomes a fast-decaying $l$-series:
$
||v_k^((w))||_infinity
&<= C sum_(l=1)^(n-k)
    (1 - alpha a)^((n-k-l+1) slash 2)
    Delta(sans(Q)^l)
    (1 - alpha a)^((l-1) slash 2)
  &&"(triangle + Lyapunov)" \
&<= C (1 - alpha a)^((n-k) slash 2)
    sum_(l=1)^infinity (1 slash 4)^(floor(l slash t_"mix"))
  &&"(extend to" l = infinity") " \
&<= C t_"mix" (1 - alpha a)^((n-k) slash 2)
  &&"(geometric block sum)".
$
Multiplying by $||epsilon.alt||_infinity$ gives
$
||v_k^((w, epsilon))||_infinity
  <= C t_"mix" ||epsilon.alt||_infinity
      (1 - alpha a)^((n-k) slash 2).
$
The functions $v_k^((w, epsilon))(Z_k) - pi(v_k^((w, epsilon)))$ are centered under $pi$ and uniformly bounded by the previous display, so the weighted Markov concentration/Rosenthal bound for centered time-dependent functions yields
$
||u^top U_R||_(L_p)
&<= C p^(1 slash 2) t_"mix"^(1 slash 2)
    lr((sum_(k=1)^(n-1) ||v_k^((w, epsilon))||_infinity^2))^(1 slash 2)
  &&"(Rosenthal)" \
&<= C p^(1 slash 2) t_"mix"^(3 slash 2) ||epsilon.alt||_infinity
    lr((sum_(k=1)^(n-1) (1 - alpha a)^(n-k)))^(1 slash 2)
  &&"(plug previous bound)" \
&<= C p^(1 slash 2) t_"mix"^(3 slash 2)
    ||epsilon.alt||_infinity frac(1, sqrt(alpha a))
  &&"(geometric series)".
$

_Step 2: bound on $u^top U_M$._ Unfold the projected matrix kernel through
$(H_(k+1)^((w)))^top u$:
$
(H_(k+1)^((w)))^top u
  = sum_(l=1)^(n-k) g_(k,l)(Z_(k+l)),
quad
g_(k,l)(z)
  = (B^(l-1))^top thin tilde(A)(z)^top thin (B^(n-k-l+1))^top u.
$
Each $g_(k,l)$ is centered under $pi$ (since $pi(tilde(A)) = 0$), and the two $B$-powers give the uniform bound
$
||g_(k,l)||_infinity
  <= C (1 - alpha a)^((n-k) slash 2),
$
which is independent of $l$. Squaring and summing over $l$ produces only linear growth in $n - k$:
$
sum_(l=1)^(n-k) ||g_(k,l)||_infinity^2
  <= C (n-k)(1 - alpha a)^(n-k).
$
Note the contrast with the $U_R$ analysis: there, UGE folded the $l$-sum into a single $t_"mix"$ factor, whereas here the same $l$-independent bound is applied $(n-k)$ times — the price of conditional centering being weaker than $pi$-centering.

The conditional expectation in the imported future-centered estimate is exactly
$
bb(E) lr([
  sum_(l=1)^(n-k) g_(k,l)(Z_(k+l))^top epsilon.alt(Z_k)
  thin | thin cal(F)_k
])
  = u^top v_k^((w, epsilon))(Z_k).
$
Therefore the whole future-centered sum $u^top U_M$ can be bounded in one step
by applying the imported estimate with
$xi_k = epsilon.alt(Z_k)$ and
$beta_(k,l) = C (1 - alpha a)^((n-k) slash 2)$:
$
||u^top U_M||_(L_p)
&<= C p^(3 slash 2) t_"mix"^(1 slash 2) ||epsilon.alt||_infinity
    lr((sum_(k=1)^(n-1) sum_(l=1)^(n-k)
      (1 - alpha a)^(n-k)))^(1 slash 2) \
&= C p^(3 slash 2) t_"mix"^(1 slash 2) ||epsilon.alt||_infinity
    lr((sum_(k=1)^(n-1) (n-k)(1 - alpha a)^(n-k)))^(1 slash 2) \
&<= C p^(3 slash 2) t_"mix"^(1 slash 2)
    ||epsilon.alt||_infinity frac(1, alpha a)
  &&"(use" sum_m m thin r^m <= C (1-r)^(-2), thin r = 1 - alpha a ")".
$
The extra factor $1 slash sqrt(alpha a)$ relative to $U_R$ is precisely the cost of using conditional rather than stationary centering.

_Step 3: assembly._ Combining the two pieces via $S_n - bb(E)S_n = -alpha (U_M + U_R)$ and the triangle inequality,
$
||u^top (S_n - bb(E)S_n)||_(L_p)
&<= alpha thin (||u^top U_M||_(L_p) + ||u^top U_R||_(L_p)) \
&<= C ||epsilon.alt||_infinity
  lr((p^(3 slash 2) t_"mix"^(1 slash 2) frac(1, a)
    + p^(1 slash 2) t_"mix"^(3 slash 2) sqrt(frac(alpha, a)))).
$
The first term (from $U_M$) is the leading contribution for small $alpha$; the second (from $U_R$) carries the heavier $t_"mix"$ dependence but vanishes as $alpha -> 0$. Restoring the factor $||u||$ and multiplying by $alpha$ gives the asserted bound for $T_n^((1, alpha)) = -alpha S_n$. #h(1fr) $square$

== A Depth-One RR Misadjustment Bound and Its Limitation

This subsection records the natural depth-one attempt and explains why it is
not used in the final Berry--Esseen assembly. The actual stationary and
burned-in theorems use the depth-two Levin transfer developed in the next
chapter.

The PR-averaged Richardson--Romberg expansion produces, after Step (S8) of the
Samsonov scheme applied separately at step sizes $alpha$ and $2 alpha$, a
depth-one "misadjustment" remainder
$
D_1^("mis, RR")
  = frac(sqrt(n), n - n_0) sum_(k=n_0)^(n-1)
    (2 J_k^((1, alpha)) - J_k^((1, 2 alpha))),
$
whose centered part must be controlled to feed into a Berry--Esseen statement
by this route. We write
$
D_(1, "c")^("mis, RR")
  := D_1^("mis, RR") - bb(E) D_1^("mis, RR")
$
for this centered statistic.

In this subsection assume explicitly that
$2 alpha <= alpha_infinity$ and $2 alpha <= alpha_("inv")$. The first
condition makes the one-stepsize last-iterate lemma admissible at both RR
levels; the second makes the shifted-to-unshifted transfer
$T_k^((1,w)) = (I - w overline(A)) J_k^((1,w))$ uniformly invertible for
$w in {alpha, 2 alpha}$.

The stationary bias is smaller than the fluctuation term. By Levin et al.
(2025, Proposition 2),
$ bb(E)_pi lr([J_infinity^((1, alpha))]) = alpha Delta + O(alpha^2), $
so the linear term $alpha Delta$ cancels in the RR-combination and the
per-iterate stationary RR bias is $O(alpha^2)$. Therefore the PR-scaled
stationary bias satisfies
$
||bb(E) D_1^("mis, RR")|| <= C sqrt(n) thin alpha^2.
$
What remains is the centered fluctuation.

Define
$
Phi(p, alpha) := p^(3 slash 2) thin t_"mix"^(1 slash 2) / a
                + p^(1 slash 2) thin t_"mix"^(3 slash 2) sqrt(alpha slash a).
$
Fix a deterministic direction $u$. The lemma applied at $alpha$ and at
$2 alpha$ gives, separately,
$
||u^top (T_n^((1, alpha)) - bb(E) T_n^((1, alpha)))||_(L_p)
  <= C ||u|| thin alpha thin Phi(p, alpha),
quad
||u^top (T_n^((1, 2 alpha)) - bb(E) T_n^((1, 2 alpha)))||_(L_p)
  <= C ||u|| thin alpha thin Phi(p, 2 alpha)
  <= C' ||u|| thin alpha thin Phi(p, alpha),
$
where $C' = sqrt(2) C$ absorbs the $sqrt(2)$-factor coming from the $2 alpha$ scaling. Combining the two by the triangle inequality and using the index-shift identity $T_k^((1, w)) = (I - w overline(A)) thin J_k^((1, w))$ gives
$
u^top J_k^((1, w))
  = ((I - w overline(A))^(-top) u)^top T_k^((1, w)).
$
The local inverse bound $|| (I - w overline(A))^(-1) || <= 2$ therefore holds
for $w in {alpha, 2 alpha}$ and yields
$
||u^top lr((2 J_k^((1, alpha)) - J_k^((1, 2 alpha)))
     - bb(E) (2 J_k^((1, alpha)) - J_k^((1, 2 alpha))))||_(L_p)
  <= C ||u|| thin alpha thin Phi(p, alpha),
$
uniformly in $k$. PR-averaging through $sqrt(n) / (n - n_0)$ and absorbing the constant therefore yields
$
||u^top D_(1, "c")^("mis, RR")||_(L_p)
  <= C ||u|| thin sqrt(n) thin alpha thin Phi(p, alpha)
  = O(sqrt(n) thin alpha).
$
Together with the bias estimate,
$
||u^top D_1^("mis, RR")||_(L_p)
  <= C ||u|| thin sqrt(n) thin alpha thin Phi(p, alpha)
    + C ||u|| thin sqrt(n) thin alpha^2.
$
At the optimal scale $alpha asymp n^(-1 slash 2)$ the centered-fluctuation
term is $O(1)$, whereas the stationary bias is $O(n^(-1 slash 2))$. Hence this
depth-one route still does not yield a useful Berry--Esseen remainder of order
$n^(-1 slash 4)$: the centered misadjustment must be controlled more sharply
to be subleading.

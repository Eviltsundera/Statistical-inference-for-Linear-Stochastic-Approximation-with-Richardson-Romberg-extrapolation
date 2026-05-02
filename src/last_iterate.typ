#import "defs.typ": *

== Centered Bound for the Shifted First-Order Perturbation

The purpose of this section is to isolate the last-iteration weighted term
which appears in the analysis of $J_n^((1, alpha))$, and to record a clean
$L_p$ bound for its centered part. The proof uses the deterministic-product
perturbation expansion from Samsonov et al. (2025, Proposition 9), specialized
to the constant-stepsize setting.

Let
$
B = I - alpha overline(A).
$
Throughout this section we assume $0 < alpha <= alpha_infinity$, so that
$
||B^m||_Q <= (1 - alpha a)^(m slash 2),
quad m >= 0.
$
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

The last display is for $J_n^((1, alpha))$ itself. In the last-iteration
estimate below we use one additional left factor $B$:
$
S_n
  = sum_(t=0)^(n-1)
      B^(n-t) tilde(A)(Z_(t+1)) J_t^((0, alpha)).
$
Thus the corresponding shifted first-order contribution is
$
T_n^((1, alpha)) = - alpha S_n.
$

#lemma[
  Assume the Markov chain is started from stationarity, that is, the law of
  $Z_1$ is $pi$, and $pi(tilde(A)) = 0$. For every $p >= 2$,
  $
  ||S_n - bb(E)S_n||_(L_p)
    <= C ||epsilon.alt||_infinity
      (p^(3 slash 2) t_"mix"^(1 slash 2) frac(1, a)
        + p^(1 slash 2) t_"mix"^(3 slash 2) sqrt(frac(alpha, a))).
  $
  Consequently,
  $
  ||T_n^((1, alpha)) - bb(E)T_n^((1, alpha))||_(L_p)
    <= C alpha ||epsilon.alt||_infinity
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
mu_k^((w))
  = bb(E)_pi[H_(k+1)^((w)) epsilon.alt(Z_k)].
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

_Step 1: bound on $U_R$._ Because $pi(tilde(A)) = 0$, applying $sans(Q)^l$ followed by integration against $pi$ replaces $tilde(A)$ by its $l$-step propagation away from stationarity. UGE then gives the geometric Dobrushin bound
$
||(sans(Q)^l tilde(A))(z)||
  = lr(|| integral tilde(A)(u) {sans(Q)^l(z, d u) - pi(d u)} ||)
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
||U_R||_(L_p)
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

_Step 2: bound on $U_M$._ Project onto a deterministic unit vector $u$ and unfold $H_(k+1)^((w)) u$ as a sum over future states:
$
H_(k+1)^((w))u
  = sum_(l=1)^(n-k) g_(k,l)(Z_(k+l)),
quad
g_(k,l)(z)
  = B^(n-k-l+1) tilde(A)(z) B^(l-1) u.
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

Applying the Markov concentration lemma to the future chain conditionally on $cal(F)_k$,
$
||H_(k+1)^((w))u
    - bb(E)[H_(k+1)^((w))u thin | thin cal(F)_k]||_(L_p)
  <= C p^(1 slash 2) t_"mix"^(1 slash 2)
      sqrt((n-k)(1 - alpha a)^(n-k)),
$
and multiplying by $epsilon.alt(Z_k)$ (which is $cal(F)_k$-measurable and bounded by $||epsilon.alt||_infinity$) gives
$
||H_(k+1)^((w)) epsilon.alt(Z_k)
    - v_k^((w, epsilon))(Z_k)||_(L_p)
  <= C p^(1 slash 2) t_"mix"^(1 slash 2)
      ||epsilon.alt||_infinity
      sqrt((n-k)(1 - alpha a)^(n-k)).
$

The summands in $U_M$ are centered conditionally on $cal(F)_k$, but they are _not_ forward martingale differences: the kernel $H_(k+1)^((w))$ peeks into the future of $Z_k$. To assemble the per-$k$ bounds into a bound on the sum we therefore invoke the weighted Burkholder/block-coupling estimate for future-centered bilinear Markov sums of Samsonov et al. (2025, Proposition 9), specialized to the present constant-stepsize kernel:
$
||U_M||_(L_p)
  <= C p lr((sum_(k=1)^(n-1)
       ||H_(k+1)^((w)) epsilon.alt(Z_k)
          - v_k^((w, epsilon))(Z_k)||_(L_p)^2))^(1 slash 2).
$
Plugging in the previous bound and evaluating the resulting sum:
$
||U_M||_(L_p)
&<= C p^(3 slash 2) t_"mix"^(1 slash 2) ||epsilon.alt||_infinity
    lr((sum_(k=1)^(n-1) (n-k)(1 - alpha a)^(n-k)))^(1 slash 2)
  &&"(plug previous bound)" \
&<= C p^(3 slash 2) t_"mix"^(1 slash 2)
    ||epsilon.alt||_infinity frac(1, alpha a)
  &&"(use" sum_m m thin x^m <= C x^(-2) ")".
$
The extra factor $1 slash sqrt(alpha a)$ relative to $U_R$ is precisely the cost of using conditional rather than stationary centering.

_Step 3: assembly._ Combining the two pieces via $S_n - bb(E)S_n = -alpha (U_M + U_R)$ and the triangle inequality,
$
||S_n - bb(E)S_n||_(L_p)
&<= alpha thin (||U_M||_(L_p) + ||U_R||_(L_p)) \
&<= C ||epsilon.alt||_infinity
  lr((p^(3 slash 2) t_"mix"^(1 slash 2) frac(1, a)
    + p^(1 slash 2) t_"mix"^(3 slash 2) sqrt(frac(alpha, a)))).
$
The first term (from $U_M$) is the leading contribution for small $alpha$; the second (from $U_R$) carries the heavier $t_"mix"$ dependence but vanishes as $alpha -> 0$. Multiplying by $alpha$ gives the asserted bound for $T_n^((1, alpha)) = -alpha S_n$. #h(1fr) $square$

#remark[
  The stationarity assumption is used only to avoid boundary notation in this
  local calculation. For a non-stationary initial law, UGE gives the same bulk
  estimate plus the standard exponentially small mixing boundary terms. The
  bias of $T_n^((1, alpha))$ is a separate issue and is controlled through the
  stationary expansion of $bb(E)J_n^((1, alpha))$ in Levin et al. (2025).
]

== Application to the PR-averaged RR Misadjustment

The PR-averaged Richardson--Romberg expansion produces, after Step (S8) of the
Samsonov scheme applied separately at step sizes $alpha$ and $2 alpha$, a
"misadjustment" remainder
$
D_1^("mis, RR")
  = frac(1, sqrt(n)) sum_(k=n_0)^(n-1)
    (2 J_k^((1, alpha)) - J_k^((1, 2 alpha))),
$
whose centered part must be controlled to feed into a Berry--Esseen statement
for $sqrt(n) (overline(theta)_n^(("RR", alpha)) - theta^*)$. The bias is not
the obstruction: by Levin et al. (2025, Proposition 2),
$ bb(E)_pi[J_infinity^((1, alpha))] = alpha Delta + O(alpha^2), $
so the linear term $alpha Delta$ cancels in the RR-combination and the
stationary bias of $D_1^("mis, RR")$ is $O(alpha^2)$. What remains is the
centered fluctuation.

The lemma applied at $alpha$ and at $2 alpha$ gives, separately,
$ ||T_n^((1, alpha)) - bb(E) T_n^((1, alpha))||_(L_p) <= C alpha thin Phi(p, alpha), quad
||T_n^((1, 2 alpha)) - bb(E) T_n^((1, 2 alpha))||_(L_p) <= C alpha thin Phi(p, alpha), $
where $Phi(p, alpha) = p^(3 slash 2) t_"mix"^(1 slash 2) / a + p^(1 slash 2) t_"mix"^(3 slash 2) sqrt(alpha / a)$ collects the constants. Combining the two by the triangle inequality and accounting for the index-shift identification $J_k^((1, alpha)) = T_k^((1, alpha))$ up to a single $B$-factor, we get
$
||(2 J_k^((1, alpha)) - J_k^((1, 2 alpha))) - bb(E) (2 J_k^((1, alpha)) - J_k^((1, 2 alpha)))||_(L_p)
  <= C alpha thin Phi(p, alpha),
$
uniformly in $k$. PR-averaging through $sqrt(n) / (n - n_0)$ and absorbing the constant therefore yields
$
||D_1^("mis, RR")||_(L_p) = O(sqrt(n) thin alpha).
$
At the optimal scale $alpha asymp n^(-1 slash 2)$ this is $O(1)$, and as a
remainder in a Berry--Esseen statement of order $n^(-1 slash 4)$ it is
*useless*: the misadjustment must be at least $o(1)$ to be subleading.

== Conclusion: The Triangle Inequality Loses the RR Cancellation

Two structural reasons explain why the single-$alpha$ centered bound proved
above does not, by itself, deliver a usable RR-misadjustment estimate.

_The triangle inequality is blind to coupling._ Each $J_k^((1, alpha))$ is
genuinely $O(alpha)$ in $L_p$, and the same is true at step $2 alpha$. Their
RR-difference is therefore at most $O(alpha)$ when bounded termwise. The
cancellation that Richardson--Romberg actually provides --- one extra factor
of $alpha$ on the difference of two processes coupled through the same noise
realization ${Z_k}$ --- is invisible to a $|2 X - Y| <= 2 |X| + |Y|$ split.

_The cancellation lives in the kernel._ Reading the RR-difference as a single
weighted sum,
$
T_n^((1, alpha)) - T_n^((1, 2 alpha))
  = - sum_(k=1)^(n-1) cal(H)_(k+1)^("RR") thin epsilon.alt(Z_k),
$
with
$
cal(H)_(k+1)^("RR")
  = sum_(l=1)^(n-k) lr((
      2 alpha B_alpha^(n-k-l+1) tilde(A)(Z_(k+l)) B_alpha^(l-1)
      - 2 alpha B_(2 alpha)^(n-k-l+1) tilde(A)(Z_(k+l)) B_(2 alpha)^(l-1)
    )),
$
the elementary identity $X^m - Y^m = (X - Y) sum_(i=1)^m X^(i-1) Y^(m-i)$ with
$X = B_alpha$, $Y = B_(2 alpha)$, $X - Y = alpha overline(A)$ extracts an
additional factor of $alpha$ uniformly in the kernel --- the very same
mechanism that gave the $sqrt(alpha)$ gain for the zeroth-order RR-difference
$tilde(J)_n^((0, alpha))$ in the previous chapter. Repeating the conditional/
stationary centering argument of the present section with $cal(H)^("RR")$ in
place of $H^((w))$ would consequently gain one factor of $alpha$ globally,
lifting the misadjustment estimate to
$ ||D_1^("mis, RR")||_(L_p) = O(sqrt(n) thin alpha^2), $
i.e. $O(n^(-1 slash 2))$ at $alpha asymp n^(-1 slash 2)$ --- strictly below
the martingale Berry--Esseen scale $n^(-1 slash 4)$.

_Why we stop here._ For a first Berry--Esseen statement of rate
$n^(-1 slash 4)$, the RR-kernel refinement is not actually needed: the same
final rate already follows from the centered bound on the unweighted
statistic $sum tilde(A)(Z_(t+1)) J_t^((0, alpha))$ in Levin et al. (2025,
Proposition 3 / Corollary 6), combined with the depth-two perturbation
expansion $H^((1)) = J^((2)) + H^((2))$ and the high-order moment bounds on
$J^((2))$, $H^((2))$ (Levin et al., 2025, Propositions 8--9). Inserted into
the RR-decomposition of equation (26) of that paper, these tools give a
remainder of order $n^(-1 slash 4)$, which matches the leading martingale
Berry--Esseen contribution.

The lemma proved in this section is therefore a clean *intermediate object*:
it isolates the shifted last-iteration statistic $S_n$, decomposes its
centered part into an additive functional $U_R$ and a future-centered
bilinear part $U_M$, and exposes the conditional-centering split that any
finer RR-analysis must respect. It does not on its own improve the leading
order of the Berry--Esseen bound. A genuine improvement --- pushing the
misadjustment strictly below $n^(-1 slash 4)$ --- would require lifting the
shifted-kernel argument from a single $alpha$ to the RR-difference
$nabla_alpha S_n$, with the kernel $cal(H)^("RR")$ above. Because the
additional $alpha$-factor is purely deterministic (it lives entirely in
$B_alpha^m - B_(2 alpha)^m$), the conditional-centering, weighted Burkholder
and Markov-Rosenthal pieces of the proof carry over verbatim. We record this
RR-kernel version as the principal open thread of the present analysis
rather than expand on it here. // ПРОВЕРИТЬ ЕЩЕ РАЗ — стоит ли уточнять
// формулировку открытого вопроса.

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

_Step 1: bound on $U_R$._ Because $pi(tilde(A)) = 0$, applying $sans(Q)^l$ followed by integration against $pi$ replaces $tilde(A)$ by its $l$-step propagation away from stationarity. UGE then gives the geometric Dobrushin bound
$
||(sans(Q)^l tilde(A))(z)||
  = lr(|| integral tilde(A)(u) {sans(Q)^l (z, thin d u) - pi(d u)} ||)
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
  &&"(use" sum_m m thin r^m <= C (1-r)^(-2), thin r = 1 - alpha a ")".
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
$ bb(E)_pi lr([J_infinity^((1, alpha))]) = alpha Delta + O(alpha^2), $
so the linear term $alpha Delta$ cancels in the RR-combination and the
stationary bias of $D_1^("mis, RR")$ is $O(alpha^2)$. What remains is the
centered fluctuation.

Define
$
Phi(p, alpha) := p^(3 slash 2) thin t_"mix"^(1 slash 2) / a
                + p^(1 slash 2) thin t_"mix"^(3 slash 2) sqrt(alpha slash a).
$
The lemma applied at $alpha$ and at $2 alpha$ gives, separately,
$
||T_n^((1, alpha)) - bb(E) T_n^((1, alpha))||_(L_p) <= C alpha thin Phi(p, alpha),
quad
||T_n^((1, 2 alpha)) - bb(E) T_n^((1, 2 alpha))||_(L_p) <= C alpha thin Phi(p, 2 alpha) <= C' alpha thin Phi(p, alpha),
$
where $C' = sqrt(2) C$ absorbs the $sqrt(2)$-factor coming from the $2 alpha$ scaling. Combining the two by the triangle inequality and using the index-shift identity $T_k^((1, w)) = (I - w overline(A)) thin J_k^((1, w))$ together with the local inverse bound $|| (I - w overline(A))^(-1) || <= 2$ for $w in {alpha, 2 alpha}$, we get
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

== Richardson--Romberg Difference of $S_n$

The single-$alpha$ centered bound, applied separately at $alpha$ and
$2 alpha$ and combined termwise via $|| 2 X - Y || <= 2 || X || + || Y ||$,
produces a misadjustment estimate of order $O(sqrt(n) thin alpha) = O(1)$ at
$alpha asymp n^(-1 slash 2)$. This estimate is *blind to coupling*: it never
uses the fact that $S_n^((alpha))$ and $S_n^((2 alpha))$ share the same noise
realization ${Z_k}$, and is the one that holds for a generic centred
difference of two centred quantities. The Richardson--Romberg construction
nonetheless promises an additional factor of $alpha$ from the deterministic
prefactors, via the elementary identity
$X^m - Y^m = (X - Y) sum_(i = 1)^m X^(i - 1) Y^(m - i)$ at $X = B_alpha$,
$Y = B_(2 alpha)$, $X - Y = alpha overline(A)$.

We accordingly view the Richardson--Romberg difference of $S_n^((alpha))$ as
a single weighted sum on the common noise, with a deterministic RR-kernel in
place of the single-$alpha$ kernel $H^((w))$. The structure of the analysis
matches the previous subsection — single-kernel form, conditional/stationary
centering, Markov-Rosenthal and weighted Burkholder — with one new
ingredient, a pointwise bound on the RR-kernel that uses the identity above
to extract a factor of $alpha$ uniformly across the kernel.

=== Single-Kernel Form

For step sizes $alpha$ and $2 alpha$, write $B_alpha = I - alpha overline(A)$,
$B_(2 alpha) = I - 2 alpha overline(A)$, and consider the *Richardson--Romberg
difference* of the shifted statistic at fixed horizon $n$:
$
tilde(S)_n^("RR") := 2 thin S_n^((alpha)) - S_n^((2 alpha)).
$
Both $S_n^((alpha))$ and $S_n^((2 alpha))$ are driven by the *same* noise
realization ${Z_k}$. Substituting $S_n^((alpha)) = -alpha sum_(k=1)^(n-1)
H_(k+1)^((w), alpha) thin epsilon.alt(Z_k)$ from the previous subsection at
both step sizes and combining,
$
tilde(S)_n^("RR")
  = -2 alpha sum_(k=1)^(n-1)
    [H_(k+1)^((w), alpha) - H_(k+1)^((w), 2 alpha)] thin epsilon.alt(Z_k)
  = -sum_(k=1)^(n-1) cal(H)_(k+1)^("RR") thin epsilon.alt(Z_k),
$
with the *RR kernel*
$
cal(H)_(k+1)^("RR")
  := 2 alpha sum_(l = 1)^(n - k)
    [B_alpha^(a_l) tilde(A)(Z_(k + l)) B_alpha^(b_l)
     - B_(2 alpha)^(a_l) tilde(A)(Z_(k + l)) B_(2 alpha)^(b_l)],
quad
a_l := n - k - l + 1, quad
b_l := l - 1.
$
By construction $a_l + b_l = n - k$. The future-state factor
$tilde(A)(Z_(k+l))$ enters $cal(H)^("RR")$ in exactly the same way as in
$H^((w))$: only the deterministic prefactors $B_alpha^(a_l), B_(2 alpha)^(a_l)$
carry the RR cancellation.

=== Pointwise Kernel Bound

#lemma[
  Let $alpha, 2 alpha in (0, alpha_infinity]$, set $overline(C)_("RR") :=
  kappa_Q^(3 slash 2) thin || overline(A) ||$, and let $M in bb(R)^(d times d)$
  be any deterministic matrix. For all integers $j, m >= 0$ with $j + m >= 1$,
  $
  || B_alpha^j thin M thin B_alpha^m - B_(2 alpha)^j thin M thin B_(2 alpha)^m ||
    <= overline(C)_("RR") thin alpha thin (j + m) thin (1 - alpha a)^((j + m - 1) slash 2) thin || M ||.
  $
]

_Proof._ Apply the additive split
$
B_alpha^j M B_alpha^m - B_(2 alpha)^j M B_(2 alpha)^m
  = (B_alpha^j - B_(2 alpha)^j) M B_alpha^m
    + B_(2 alpha)^j M (B_alpha^m - B_(2 alpha)^m),
$
and the elementary telescope $X^r - Y^r = sum_(i=1)^r X^(i-1) (X - Y) Y^(r - i)$
with $X = B_alpha$, $Y = B_(2 alpha)$, $X - Y = alpha overline(A)$. For the
first piece, submultiplicativity in $|| dot ||_Q$ followed by Lyapunov
contraction at both step sizes (with $1 - 2 alpha a <= 1 - alpha a$) gives
$
|| B_alpha^j - B_(2 alpha)^j ||_Q
  <= alpha thin || overline(A) ||_Q thin j thin (1 - alpha a)^((j - 1) slash 2),
$
hence after one application of the equivalence $|| dot || <= kappa_Q^(1 slash 2) || dot ||_Q$,
$
|| (B_alpha^j - B_(2 alpha)^j) thin M thin B_alpha^m ||
  <= alpha thin kappa_Q^(3 slash 2) thin || overline(A) || thin j thin (1 - alpha a)^((j + m - 1) slash 2) thin || M ||.
$
The second piece is symmetric and gives the same bound with $j$ replaced by
$m$. Adding the two yields the claim. $square$

The lemma is the *only* place where Richardson--Romberg cancellation is used.
Two structural features deserve emphasis:

- The factor $alpha$ extracted by the telescope is the deterministic
  manifestation of RR coupling: it is the same identity that powered the
  zeroth-order bound on $tilde(J)_n^((0, alpha))$ in Chapter 2 and the
  discrete-derivative bound on $cal(Q)_(l + 1)^("RR") - cal(Q)_l^("RR")$ in
  Chapter 4.
- The factor $j + m$ is the count of telescope summands. It cannot be removed
  without strengthening the geometric decay, which is already optimal for
  $B_alpha$. In the present application $j + m = n - k$ is the *full lag*,
  and this growth is what spoils the gain in the variance proxies below.

=== Centered Decomposition

Following the single-$alpha$ scheme, fix $cal(F)_k = sigma(Z_1, dots, Z_k)$.
Under stationarity ($Z_1$ drawn from $pi$, $pi(tilde(A)) = 0$), the Markov
property gives
$
bb(E) [cal(H)_(k+1)^("RR") thin epsilon.alt(Z_k) thin | thin cal(F)_k]
  = v_k^((w, "RR", epsilon.alt))(Z_k),
$
with
$
v_k^((w, "RR"))(z)
  := 2 alpha sum_(l=1)^(n-k)
    [B_alpha^(a_l) (sans(Q)^l tilde(A))(z) B_alpha^(b_l)
     - B_(2 alpha)^(a_l) (sans(Q)^l tilde(A))(z) B_(2 alpha)^(b_l)],
quad
v_k^((w, "RR", epsilon.alt))(z) := v_k^((w, "RR"))(z) thin epsilon.alt(z),
$
and $bb(E)[v_k^((w, "RR", epsilon.alt))(Z_k)] = pi(v_k^((w, "RR", epsilon.alt)))$.
Therefore
$
tilde(S)_n^("RR") - bb(E) tilde(S)_n^("RR") = -(U_M^("RR") + U_R^("RR")),
$
$
U_M^("RR") = sum_(k=1)^(n-1) {cal(H)_(k+1)^("RR") epsilon.alt(Z_k) - v_k^((w, "RR", epsilon.alt))(Z_k)},
quad
U_R^("RR") = sum_(k=1)^(n-1) {v_k^((w, "RR", epsilon.alt))(Z_k) - pi(v_k^((w, "RR", epsilon.alt)))}.
$
The decomposition has the same conditional/stationary split as in the
single-$alpha$ proof; the RR-kernel bound feeds into the variance proxies of
each piece.

=== $L_p$ Bound

#lemma[
  Assume the Markov chain is started from stationarity and $pi(tilde(A)) = 0$.
  For every $p >= 2$,
  $
  || U_R^("RR") ||_(L_p)
    <= C thin || epsilon.alt ||_infinity
       thin p^(1 slash 2) thin t_"mix"^(3 slash 2)
       thin sqrt(alpha) slash a^(3 slash 2),
  $
  $
  || U_M^("RR") ||_(L_p)
    <= C thin || epsilon.alt ||_infinity
       thin p^(3 slash 2) thin t_"mix"^(1 slash 2)
       slash a^2,
  $
  with a constant $C$ depending only on $C_A, kappa_Q, ||overline(A)||$.
  Consequently,
  $
  || tilde(S)_n^("RR") - bb(E) tilde(S)_n^("RR") ||_(L_p)
    <= C thin || epsilon.alt ||_infinity
       lr((p^(3 slash 2) thin t_"mix"^(1 slash 2) slash a^2
            + p^(1 slash 2) thin t_"mix"^(3 slash 2) sqrt(alpha) slash a^(3 slash 2))).
  $
]

_Proof._ The structure mirrors the single-$alpha$ proof; only the variance
proxies change. We trace each piece.

*Step 1 ($U_R^("RR")$).* By UGE applied to the inner $sans(Q)^l tilde(A)$
factor, $|| (sans(Q)^l tilde(A))(z) || <= 2 C_A (1 slash 4)^(floor(l slash t_"mix"))$.
Combining with the kernel-difference Lemma (with $j + m = n - k$ uniform in
$l$, and $|| (sans(Q)^l tilde(A))(z) ||$ playing the role of $|| M ||$),
$
|| v_k^((w, "RR"))(z) ||
&<= 2 alpha sum_(l = 1)^(n - k) overline(C)_("RR") thin alpha thin (n - k) thin (1 - alpha a)^((n - k - 1) slash 2) thin 2 C_A thin (1 slash 4)^(floor(l slash t_"mix"))
  &&"(kernel + UGE)" \
&<= C thin t_"mix" thin alpha^2 thin (n - k) thin (1 - alpha a)^((n - k - 1) slash 2)
  &&"(geometric block sum)".
$
Multiplying by $|| epsilon.alt ||_infinity$,
$
|| v_k^((w, "RR", epsilon.alt)) ||_infinity
  <= C thin t_"mix" thin || epsilon.alt ||_infinity thin alpha^2 thin (n - k) thin (1 - alpha a)^((n - k - 1) slash 2).
$
Squaring and summing over $k$, with $sum_(m >= 1) m^2 (1 - alpha a)^(m - 1) <= 2 slash (alpha a)^3$,
$
sum_(k = 1)^(n - 1) || v_k^((w, "RR", epsilon.alt)) ||_infinity^2
  <= C^2 thin t_"mix"^2 thin || epsilon.alt ||_infinity^2 thin alpha^4
     thin frac(2, (alpha a)^3)
  = C^prime thin t_"mix"^2 thin || epsilon.alt ||_infinity^2 thin alpha slash a^3.
$
The weighted Markov concentration of Levin et al. (2025, Lemma 11) applied
to the centered functions $v_k^((w, "RR", epsilon.alt))(Z_k) - pi(v_k^((w, "RR", epsilon.alt)))$ then gives
$
|| U_R^("RR") ||_(L_p)
  <= C p^(1 slash 2) thin t_"mix"^(1 slash 2)
     lr((sum_(k=1)^(n-1) || v_k^((w, "RR", epsilon.alt)) ||_infinity^2))^(1 slash 2)
  <= C p^(1 slash 2) thin t_"mix"^(3 slash 2) thin || epsilon.alt ||_infinity thin sqrt(alpha) slash a^(3 slash 2).
$

*Step 2 ($U_M^("RR")$).* Project onto a unit vector $u$ and unfold the kernel
as a future-state sum,
$
cal(H)_(k+1)^("RR") u = sum_(l=1)^(n-k) F_l^("RR")(Z_(k+l)),
quad
F_l^("RR")(z) := 2 alpha thin [B_alpha^(a_l) tilde(A)(z) B_alpha^(b_l)
                                - B_(2 alpha)^(a_l) tilde(A)(z) B_(2 alpha)^(b_l)] thin u.
$
Each $F_l^("RR")$ is centered under $pi$ and the kernel-difference Lemma gives
the *$l$-uniform* bound
$
|| F_l^("RR") ||_infinity
  <= 2 alpha thin overline(C)_("RR") thin alpha thin (a_l + b_l) thin (1 - alpha a)^((n - k - 1) slash 2) thin || tilde(A) ||
  <= C thin alpha^2 thin (n - k) thin (1 - alpha a)^((n - k - 1) slash 2).
$
Squaring and summing over the $n - k$ values of $l$,
$
sum_(l = 1)^(n - k) || F_l^("RR") ||_infinity^2
  <= C^2 thin alpha^4 thin (n - k)^3 thin (1 - alpha a)^(n - k - 1).
$
Markov concentration on the future chain conditionally on $cal(F)_k$ then
yields
$
|| cal(H)_(k+1)^("RR") u - bb(E) [cal(H)_(k+1)^("RR") u thin | thin cal(F)_k] ||_(L_p, Z_k)
  <= C p^(1 slash 2) thin t_"mix"^(1 slash 2) thin alpha^2 thin (n - k)^(3 slash 2) thin (1 - alpha a)^((n - k - 1) slash 2),
$
and multiplying by $|| epsilon.alt(Z_k) || <= || epsilon.alt ||_infinity$
preserves the bound. Summing the squares over $k$, with
$sum_(m >= 1) m^3 (1 - alpha a)^(m - 1) <= 6 slash (alpha a)^4$,
$
sum_(k = 1)^(n - 1) || cal(H)_(k+1)^("RR") epsilon.alt(Z_k) - v_k^((w, "RR", epsilon.alt))(Z_k) ||_(L_p)^2
  <= C thin p thin t_"mix" thin || epsilon.alt ||_infinity^2 thin alpha^4 slash (alpha a)^4
  = C thin p thin t_"mix" thin || epsilon.alt ||_infinity^2 slash a^4.
$
The weighted Burkholder estimate for future-centered bilinear Markov sums
(Samsonov et al., 2025, Proposition 9; same instance as in the single-$alpha$
proof) closes the bound
$
|| U_M^("RR") ||_(L_p)
  <= C p lr((sum_(k=1)^(n-1) || cal(H)_(k+1)^("RR") epsilon.alt(Z_k) - v_k^((w, "RR", epsilon.alt))(Z_k) ||_(L_p)^2))^(1 slash 2)
  <= C p^(3 slash 2) thin t_"mix"^(1 slash 2) thin || epsilon.alt ||_infinity slash a^2.
$

*Assembly.* Adding the two bounds gives the claim. $square$

=== Two Approaches: Triangle Inequality vs. Kernel Difference

There are two natural ways to bound the centered RR-difference, and it is
instructive to lay them side by side.

*Approach A: triangle inequality on the single-$alpha$ lemma.* The
single-$alpha$ centered bound from the previous subsection,
$
|| S_n^((alpha)) - bb(E) S_n^((alpha)) ||_(L_p)
  <= C thin || epsilon.alt ||_infinity
     lr((p^(3 slash 2) thin t_"mix"^(1 slash 2) slash a + p^(1 slash 2) thin t_"mix"^(3 slash 2) sqrt(alpha slash a))),
$
applied separately at $alpha$ and $2 alpha$ and combined by
$|| 2 X - Y || <= 2 ||X|| + ||Y||$, yields
$
|| tilde(S)_n^("RR") - bb(E) tilde(S)_n^("RR") ||_(L_p)
  <= C thin || epsilon.alt ||_infinity
     lr((p^(3 slash 2) t_"mix"^(1 slash 2) slash a + p^(1 slash 2) t_"mix"^(3 slash 2) sqrt(alpha slash a))).
$
This bound is *blind to coupling*: the same-noise structure of $S_n^((alpha))$
and $S_n^((2 alpha))$ does not enter, and the resulting estimate is the one
that holds for a generic centred difference of two centred statistics.

*Approach B: kernel-difference identity.* Express $tilde(S)_n^("RR")$ as a
single noise-weighted sum with kernel $cal(H)^("RR")$ and feed the
kernel-difference Lemma through the conditional/stationary centring scheme.
The resulting bound, derived above, is
$
|| tilde(S)_n^("RR") - bb(E) tilde(S)_n^("RR") ||_(L_p)
  <= C thin || epsilon.alt ||_infinity
     lr((p^(3 slash 2) t_"mix"^(1 slash 2) slash a^2 + p^(1 slash 2) t_"mix"^(3 slash 2) sqrt(alpha) slash a^(3 slash 2))),
$
i.e. the same dependence on $alpha$ but an *additional* factor of $1 slash a$
in both terms. For $a < 1$ — the regime of interest — Approach B is
*strictly worse* than Approach A.

The mechanism is structural. The kernel-difference identity extracts a factor
of $alpha$ on each summand of $cal(H)^("RR")$, but at the price of a count
$a_l + b_l = n - k$ of summands carrying that gain *uniformly in $l$*.
Squaring and summing over $l$ in the Markov-Rosenthal/Burkholder variance
proxy, the $(n - k)$ factor combines with $sum (n-k)^q (1 - alpha a)^(n - k - 1)
<= q ! slash (alpha a)^(q + 1)$ to introduce one extra $1 slash (alpha a)$,
which is not absorbed by the $alpha^2$ kernel prefactor.

The same phenomenon, with weaker symptoms, was already visible in the
zeroth-order analysis of Chapter 2: the bound $|| H_j^((n)) || <=
overline(C)_A (1 - alpha a)^((n - j - 1) slash 2) thin 2 slash (alpha a)$
has the same telescope-times-geometric-tail structure, and the
$1 slash (alpha a)$ blow-up there is absorbed by *one* of the two
$alpha$-factors of the prefactor $alpha^2$ — leaving a residual $1 slash a$
in the variance proxy.

In short, the kernel-difference identity captures only the *deterministic*
part of the RR cancellation, through $B_alpha^m - B_(2 alpha)^m$. It does not
see the stochastic coupling through ${Z_k}$, and the latter is what the RR
construction is for. The kernel-difference route therefore replaces a
coupling-blind triangle estimate by an equally coupling-blind kernel
estimate, with no net gain on the centered $L_p$ norm of $tilde(S)_n^("RR")$.

=== Implication for the PR-Averaged Misadjustment

Both approaches feed into the misadjustment in the same way. After
$sqrt(n)$-scaled PR-averaging,
$
|| D_1^("mis, RR") ||_(L_p) = O(sqrt(n) thin alpha) = O(1) quad "at" quad alpha asymp n^(-1 slash 2),
$
which fails to vanish and does not match the martingale Berry--Esseen scale
$n^(-1 slash 4)$. The RR construction provides *no improvement* at this level
of analysis: the rate is the same as for the single-$alpha$ misadjustment.

A bound of order $n^(-1 slash 4)$ on the misadjustment is nevertheless
reachable, but through a *different* decomposition. The depth-two
perturbation expansion
$
H^((0, alpha)) = J^((1, alpha)) + J^((2, alpha)) + H^((2, alpha))
$
of Levin et al. (2025) splits the misadjustment into three independently
controlled pieces:

- the centered bilinear statistic $sum_t tilde(A)(Z_(t+1)) thin J_t^((0, alpha))$
  is bounded by Levin's Proposition 3 / Corollary 6 at order
  $sqrt(alpha r) + 1 slash sqrt(alpha)$ in $L_p$, contributing $O(n^(-1 slash 4))$
  after $sqrt(n)$-scaled PR-averaging;
- the high-order moments $|| J_n^((2, alpha)) ||_(L_p)$ and
  $|| H_n^((2, alpha)) ||_(L_p)$ are $O(alpha^(3 slash 2))$ by Levin's
  Propositions 8--9, giving the same $O(n^(-1 slash 4))$ contribution after
  PR-averaging and $sqrt(n)$-scaling;
- the bias of $J^((1, alpha))$ has the form $alpha Delta + O(alpha^2)$ by
  Levin's Proposition 2, and the leading $alpha Delta$ cancels under RR,
  leaving a stationary bias of order $alpha^2 = O(n^(-1))$, well below the
  target.

Stitching these pieces together yields $|| D_1^("mis, RR") ||_(L_p) = O(n^(-1 slash 4))$.
The kernel-difference Lemma proved earlier in this section is *not* a
load-bearing ingredient for that rate.

A *strict* improvement of the misadjustment to $O(n^(-1 slash 2))$,
subleading to martingale Berry--Esseen, would require a tool that exploits
the stochastic coupling of the two trajectories — for instance, Berbee-style
decoupling on the joint two-chain state space, or a depth-two RR cancellation
directly on $J^((2))$. We do not pursue this direction here.

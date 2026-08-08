#import "../defs.typ": *

== Centered Bound for the Shifted First-Order Perturbation

// This chapter is partly preliminary. The last-iterate RR discussion at the end
// is motivational and is not used in the final Berry--Esseen assembly; the
// downstream input retained by Chapter 4 is the centered shifted
// first-order bound proved below. The proof uses the deterministic-product
// perturbation expansion from Samsonov et al. (2025, Proposition 9), specialized
// to the constant-stepsize setting, and isolates the future-centered bilinear
// concentration input used in that argument.

Let
$
B = I - alpha overline(A).
$
Assume throughout that $0 < alpha <= alpha_infinity$. Then
$
||B^m||_Q <= (1 - alpha a)^(m slash 2),
quad m >= 0.
$
Set also
$
alpha_("inv") := frac(1, 2 || overline(A) ||).
$
If $w <= alpha_("inv")$, the Neumann series gives
$|| (I - w overline(A))^(-1) || <= 2$.
In Richardson--Romberg applications the single-stepsize estimates are applied
at $w in {alpha, 2 alpha}$, so the local assumptions are
$2 alpha <= alpha_infinity$ and $2 alpha <= alpha_("inv")$.
Constants denoted by $C$ may depend on fixed problem parameters, but not on
$alpha$, $n$, $p$, or $q$.

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
// &= alpha^2 sum_(1 <= k < j <= n)
//       B^(n-j) tilde(A)(Z_j) B^(j-1-k) epsilon.alt(Z_k) \
&= alpha^2 sum_(k=1)^(n-1) sum_(r=1)^(n-k)
      B^(n-k-r) tilde(A)(Z_(k+r)) B^(r-1) epsilon.alt(Z_k).
$
// The sign in the second line is positive because
// $J_(j-1)^((0, alpha))$ already contains the minus sign.

Use the shifted version
$
S_n
  = sum_(t=0)^(n-1)
      B^(n-t) tilde(A)(Z_(t+1)) J_t^((0, alpha)).
$
Then
$
S_n
  = sum_(j=1)^n B^(n-j+1) tilde(A)(Z_j) J_(j-1)^((0, alpha))
  = - frac(1, alpha) thin B thin J_n^((1, alpha)).
$
Thus the corresponding shifted first-order contribution is
$
T_n^((1, alpha)) = - alpha S_n = B thin J_n^((1, alpha)),
$
so passing from $T_n^((1, alpha))$ back to $J_n^((1, alpha))$ uses the local
bound on $B^(-1)$.

#lemma[
  *(Samsonov et al., 2025,
  Appendix D.2, Proposition 9)*
  Assume UGE 1 and stationarity. Let $cal(F)_k = sigma(Z_0, dots, Z_k)$.
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
    } ||_(L_p)
  &quad <= C p^(3 slash 2) t_"mix"^(1 slash 2) M_xi
     lr((sum_(k=1)^(n-1) sum_(l=1)^(n-k) beta_(k,l)^2))^(1 slash 2).
  $
  // This is the scalar, constant-stepsize form extracted from Samsonov et al.
  // (2025, Appendix D.2, Proposition 9). The centering is conditional with
  // respect to $cal(F)_k$.
] <lem:future-centered-bilinear-input>

#lemma[
  *(Centered shifted first-order bound.)*
  Assume the Markov chain is started from stationarity, that is, the law of
  $Z_1$ is $pi$, and $pi(tilde(A)) = 0$. For every deterministic direction
  $u in bb(R)^d$ and every $p >= 2$,
  $
  ||u^top (S_n - bb(E)S_n)||_(L_p)
    <= C ||u|| thin ||epsilon.alt||_infinity
      (p^(3 slash 2) t_"mix"^(1 slash 2) frac(1, a)
        + p^(1 slash 2) t_"mix"^(3 slash 2) sqrt(frac(alpha, a))).
  $
  // Consequently,
  // $
  // ||u^top (T_n^((1, alpha)) - bb(E)T_n^((1, alpha)))||_(L_p)
  //   <= C alpha ||u|| thin ||epsilon.alt||_infinity
  //     (p^(3 slash 2) t_"mix"^(1 slash 2) frac(1, a)
  //       + p^(1 slash 2) t_"mix"^(3 slash 2) sqrt(frac(alpha, a))).
  // $
] <lem:last-shifted-first-order>

_Proof._ Since $J_0^((0, alpha)) = 0$, the term $t = 0$ in $S_n$ vanishes.
Substituting the explicit formula for $J_t^((0, alpha))$ gives
$
S_n
&= - alpha sum_(t=1)^(n-1) sum_(k=1)^t
    B^(n-t) tilde(A)(Z_(t+1)) B^(t-k) epsilon.alt(Z_k)
&= - alpha sum_(k=1)^(n-1)
    H_(k+1)^((w)) epsilon.alt(Z_k),
$
where, after the change of summation index $l = t - k + 1$,
$
H_(k+1)^((w))
  = sum_(l=1)^(n-k)
      B^(n-k-l+1) tilde(A)(Z_(k+l)) B^(l-1).
$
// The kernel is future-dependent relative to $epsilon.alt(Z_k)$.

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

Let $cal(F)_k = sigma(Z_0, dots, Z_k)$. The Markov property gives
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
Here $sans(Q)$ is the Markov transition kernel, so
$
(sans(Q)^l tilde(A))(z)
  = integral tilde(A)(u) thin sans(Q)^l (z, thin d u)
  = bb(E) lr([tilde(A)(Z_(k+l)) | Z_k = z]).
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
// We bound $U_R$ and $U_M$ separately.

Fix a deterministic direction $u$, $||u|| = 1$; the general statement
follows by multiplying by $||u||$.

For $U_R$, since $pi(tilde(A)) = 0$, UGE gives
$
||(sans(Q)^l tilde(A))(z)||
  = lr(|| integral tilde(A)(y) {sans(Q)^l (z, thin d y) - pi(d y)} ||)
  <= 2 C_A Delta(sans(Q)^l)
  <= 2 C_A (1 slash 4)^(floor(l slash t_"mix")).
$
Hence
$
||v_k^((w))||_infinity
&<= C sum_(l=1)^(n-k)
    (1 - alpha a)^((n-k-l+1) slash 2)
    Delta(sans(Q)^l)
    (1 - alpha a)^((l-1) slash 2) 
// &<= C (1 - alpha a)^((n-k) slash 2)
//     sum_(l=1)^infinity (1 slash 4)^(floor(l slash t_"mix")) \
&<= C t_"mix" (1 - alpha a)^((n-k) slash 2).
$
Thus
$
||v_k^((w, epsilon))||_infinity
  <= C t_"mix" ||epsilon.alt||_infinity
      (1 - alpha a)^((n-k) slash 2).
$
Applying Markov concentration to the centered functions
$v_k^((w, epsilon)) - pi(v_k^((w, epsilon)))$ gives
$
||u^top U_R||_(L_p)
&<= C p^(1 slash 2) t_"mix"^(1 slash 2)
    lr((sum_(k=1)^(n-1) ||v_k^((w, epsilon))||_infinity^2))^(1 slash 2)
// &<= C p^(1 slash 2) t_"mix"^(3 slash 2) ||epsilon.alt||_infinity
//     lr((sum_(k=1)^(n-1) (1 - alpha a)^(n-k)))^(1 slash 2) \
&<= C p^(1 slash 2) t_"mix"^(3 slash 2)
    ||epsilon.alt||_infinity frac(1, sqrt(alpha a)).
$

For $U_M$, unfold the projected matrix kernel through
$(H_(k+1)^((w)))^top u$:
$
(H_(k+1)^((w)))^top u
  = sum_(l=1)^(n-k) g_(k,l)(Z_(k+l)),
quad
g_(k,l)(z)
  = (B^(l-1))^top thin tilde(A)(z)^top thin (B^(n-k-l+1))^top u.
$
Each $g_(k,l)$ is centered under $pi$, and
$
||g_(k,l)||_infinity
  <= C (1 - alpha a)^((n-k) slash 2),
$
$
sum_(l=1)^(n-k) ||g_(k,l)||_infinity^2
  <= C (n-k)(1 - alpha a)^(n-k).
$
// Unlike $U_R$, the conditional-centering estimate keeps the factor $n-k$.

The conditional expectation in the future-centered estimate is exactly
$
bb(E) lr([
  sum_(l=1)^(n-k) g_(k,l)(Z_(k+l))^top epsilon.alt(Z_k)
  thin | thin cal(F)_k
])
  = u^top v_k^((w, epsilon))(Z_k).
$
Apply that estimate with
$xi_k = epsilon.alt(Z_k)$ and
$beta_(k,l) = C (1 - alpha a)^((n-k) slash 2)$:
$
||u^top U_M||_(L_p)
&<= C p^(3 slash 2) t_"mix"^(1 slash 2) ||epsilon.alt||_infinity
    lr((sum_(k=1)^(n-1) sum_(l=1)^(n-k)
      (1 - alpha a)^(n-k)))^(1 slash 2)
// &= C p^(3 slash 2) t_"mix"^(1 slash 2) ||epsilon.alt||_infinity
//     lr((sum_(k=1)^(n-1) (n-k)(1 - alpha a)^(n-k)))^(1 slash 2) \
&<= C p^(3 slash 2) t_"mix"^(1 slash 2)
    ||epsilon.alt||_infinity frac(1, alpha a).
$
// The additional factor compared with $U_R$ comes from the conditional
// centering estimate.

// Finally, since $S_n - bb(E)S_n = -alpha (U_M + U_R)$,
$
||u^top (S_n - bb(E)S_n)||_(L_p)
&<= alpha thin (||u^top U_M||_(L_p) + ||u^top U_R||_(L_p))
&<= C ||epsilon.alt||_infinity
  lr((p^(3 slash 2) t_"mix"^(1 slash 2) frac(1, a)
    + p^(1 slash 2) t_"mix"^(3 slash 2) sqrt(frac(alpha, a)))).
$
// Restoring $||u||$ and using $T_n^((1, alpha)) = -alpha S_n$ proves the
// second display. #h(1fr) $square$

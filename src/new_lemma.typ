#import "defs.typ": *

== Study Asymptotic of $J_n^((1, alpha))$

Let's remember some formulas:
$ J_n^((0, alpha)) = (I - alpha overline(A)) J_(n-1)^((0, alpha)) - alpha epsilon.alt(Z_n), quad J_0^((0, alpha)) = 0 $

$ J_n^((1, alpha)) = (I - alpha overline(A)) J_(n-1)^((1, alpha)) - alpha tilde(A)(Z_n) J_(n-1)^((0, alpha)) $

$ J_n^((1, alpha)) = -alpha sum_(j=1)^n (I - alpha overline(A))^(n-j) tilde(A)(Z_j) $

$ J_n^((1, alpha)) = -alpha^2 sum_(j=1)^n sum_(l=1)^(j-1) (I - alpha overline(A))^(n-j) tilde(A)(Z_j) (I - alpha overline(A))^(j-1-l) epsilon.alt(Z_l) $

$Y_t = (J_t^((0, alpha)), Z_(t+1))$ is a Markov Chain with $Pi_(J, alpha)$ invariant distribution.

Let $psi(J, z) = tilde(A)(z) J$:
$ overline(psi)_t = psi(J_t^((0, alpha)), Z_(t+1)) - bb(E)_(Pi_(J, alpha)) [psi_0] $

We know that:
$ (bb(E)_(Pi_(J, alpha)) lr(||sum_(t=0)^(n-1) overline(psi)_t||)^p)^(1 slash p) lt.tilde
C_A kappa_Q^(1 slash 2) rho^(1 slash 2) t_"mix"^(1 slash 2) ||epsilon.alt||_infinity
(p a^(-1 slash 2) (alpha n)^(1 slash 2) + t_"mix"^(1 slash 2) alpha n^(1 slash 2) + a^(-1 slash 2) alpha^(1 slash 2))
lt.tilde C sqrt(alpha n) $

Let's study:
$ zeta_t (J, z) = (I - alpha overline(A))^(n-t) tilde(A)(Z_(t+1)) J_t^((0, alpha)) $
and:
$ (bb(E)_(Pi_(J, alpha)) lr(||sum_(t=0)^(n-1) overline(zeta)_t||)^p)^(1 slash p) $

$ B^k = (I - alpha A)^k quad ||B^k||_Q <= (1 - alpha a)^(k slash 2) ==>
sum_(k=0)^(n-1) ||B^k||_Q^2 <= sum_(k=0)^(n-1) (1 - alpha a)^k <= frac(1, alpha a) $

$
zeta_t = B^(n-t) tilde(A)(Z_(t+1)) J_t^((0, alpha))
= -alpha B^(n-t) tilde(A)(Z_(t+1)) (sum_(k=1)^t B^(t-k) epsilon.alt(Z_k))
= -alpha sum_(k=1)^t B^(n-t) tilde(A)(Z_(t+1)) B^(t-k) epsilon.alt(Z_k)
$

// ПРОВЕРИТЬ ЕЩЕ РАЗ (CHECK AGAIN)
$
S_n &= -alpha sum_(t=1)^(n-1) sum_(k=1)^t B^(n-t) tilde(A)(Z_(t+1)) B^(t-k) epsilon.alt(Z_k) \
&= -alpha sum_(k=1)^(n-1) (sum_(l=1)^(n-k) B^(n-(k+l-1)) tilde(A)(Z_(t+1)) B^(l-1)) epsilon.alt(Z_k) \
&= -alpha sum_(k=1)^(n-1) H_(k+1)^((w)) epsilon.alt(Z_k)
$

$ H_(k+1)^((w)) = sum_(l=1)^(n-k) B^(n-(k+l-1)) tilde(A)(Z_(k+l)) B^(l-1) $

$ mu_(t,k)^((w)) = bb(E)_pi [B^(n-t) tilde(A)(Z_(t+1)) B^(t-k) epsilon.alt(Z_k)] $

$ mu_k^((w)) = bb(E)_pi [sum_(l=1)^(n-k) B^(n-(k+l-1)) tilde(A)(Z_(k+l)) B^(l-1) epsilon.alt(Z_k)] $

$ S_n - bb(E) S_n = -alpha sum_(k=1)^(n-1) (H_(k+1)^((w)) epsilon.alt(Z_k) - mu_k^((w))) $

$
H_(k+1)^((w)) epsilon.alt(Z_k) - mu_k^((w))
&= H_(k+1)^((w)) epsilon.alt(Z_k) - bb(E)[H_(k+1)^((w)) epsilon.alt(Z_k) | cal(F)]
+ bb(E)[H_(k+1)^((w)) epsilon.alt(Z_k) | cal(F)] - mu_k^((w)) \
&quad "where" quad cal(F) = sigma(Z_1, dots, Z_k)
$

Decompose it to martingale + residual:

$ M_k^((w)) = H_(k+1)^((w)) epsilon.alt(Z_k) - bb(E)[H_(k+1)^((w)) epsilon.alt(Z_k) | cal(F)] $

$ R_k^((w)) = bb(E)[H_(k+1)^((w)) epsilon.alt(Z_k) | cal(F)] - mu_k^((w)) $

$ U_1^((w)) = H_2^((W)) epsilon.alt(Z_1) - mu_1^((w)) quad
U_2^((w)) = sum_(k=2)^(n-1) M_k^((w)) quad
U_3^((w)) = sum_(k=2)^(n-1) R_k^((w)) $

$ S_n - bb(E) S_n = -alpha (U_1^((w)) + U_2^((w)) + U_3^((w))) quad ||epsilon.alt(Z)|| <= ||epsilon.alt||_infinity $

$ ||M_k^((w))||_(L_p) <= 2 ||H_(k+1)^((w)) epsilon.alt(Z_k)||_(L_p) <=
2 ||epsilon.alt||_infinity sup_(||u|| = 1) ||H_(k+1)^((w)) u||_(L_p) $

$ H_(k+1)^((w)) u = sum_(l=1)^(n-k) g_(k,l)(Z_(k+l)) quad
g_(k,l)(Z) = B^(n-(n-k-1)) tilde(A)(Z) B^(l-1) u $

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

$ c_(k,l) = ||g_(k,l)||_infinity quad
u_k = 8 sqrt(t_"mix") (sum_(l=1)^(n-k) c_(k,l)^2)^(1 slash 2) ==>
||H_(k+1)^((w)) u - bb(E)[H_(k+1)^((w)) u]||_(L_p) <= p^(1 slash 2) u_k $

$ c_(k,l) = sup_Z ||B^(n-(n-k-1)) tilde(A)(Z) B^(l-1) u|| <= C_A dot ||B^(n-k-l+1)|| dot ||B^(l-1) u|| $

$ sum_(l=1)^(n-k) c_(k,l) <= C^2 sum_(l=1)^(n-k) (1 - alpha a)^(n-k-l+1+l-1) = C^2 (n-k)(1 - alpha a)^(n-k) $

$ ||H_(k+1)^((w)) u - bb(E)[H_(k+1)^((w)) u]||_(L_p) <=
C p^(1 slash 2) t_"mix"^(1 slash 2) sqrt((n-k)(1 - alpha a)^(n-k)) $

$
||U_2^((w))||_(L_p) &<= p (sum_(k=2)^(n-1) ||M_k^((w))||_(L_p)^2)^(1 slash 2) \
&lt.tilde C p^(3 slash 2) t_"mix"^(1 slash 2) ||epsilon.alt||_infinity
  (sum_(k=2)^(n-1) (n-k)(1 - alpha a)^(n-k))^(1 slash 2) \
&lt.tilde C p^(3 slash 2) t_"mix"^(1 slash 2) ||epsilon.alt||_infinity frac(1, alpha a)
$

$ v_k^((w))(z) = sum_(l=1)^(n-k) Q^l [B^(n-k-l+1) tilde(A)(z)] B^(l-1), quad
v_k^((w, epsilon.alt))(z) = v_k^((w))(z) epsilon.alt(z) $

$ bb(E)[H_(k+1)^((w)) epsilon.alt(Z_k) | cal(F)_k] =
(sum_(l=1)^(n-k) B^(n-k-l+1) (Q^l tilde(A))(z) B^(l-1)) epsilon.alt(Z_k) =
v_k^((w, epsilon.alt))(Z_k) $

$
bb(E)[H_(k+1)^((w)) epsilon.alt(Z_k) | cal(F)_(k-1)] &= (Q v_k^((w, epsilon.alt)))(Z_(k-1)) = phi_k (Z_(k-1)) = Q g_k \
overline(phi)_k (z) &= phi_k (z) - mu_k^((w))
$

$ U_3^((w)) = sum_(k=2)^(n-1) overline(phi)_k (Z_(k-1)) $

$ ||overline(phi)_k||_infinity <= sup_(z, z') ||phi_k (z) - phi_k (z')||
<= 2 Delta(Q) ||g||_infinity <= 2 ||g||_infinity <= 2 ||v_k^((w))||_infinity ||epsilon.alt||_infinity $

$ ||v_k^((w))||_infinity
<= sum_(l=1)^(n-k) ||B^(n-k-l+1)||_Q thin ||(Q^l tilde(A))(dot)||_infinity thin ||B^(l-1)||_Q $

Using $pi(tilde(A)) = 0$ and the definition of the Dobrushin coefficient,
$ ||(Q^l tilde(A))(z)|| = lr(||integral tilde(A)(u)(Q^l (z, d u) - pi(d u))||)
<= 2 C_A Delta(Q^l). $

Therefore,
$ ||v_k^((w))||_infinity
<= 2 C_A sum_(l=1)^(n-k) (1 - alpha a)^((n-k-l+1) slash 2)
(1 - alpha a)^((l-1) slash 2) Delta(Q^l)
= 2 C_A (1 - alpha a)^((n-k) slash 2) sum_(l=1)^(n-k) Delta(Q^l). $

By submultiplicativity of the Dobrushin coefficient,
$Delta(Q^l) <= Delta(Q)^l$, hence
$ sum_(l=1)^(n-k) Delta(Q^l) <= sum_(l=1)^infinity Delta(Q)^l = frac(Delta(Q), 1 - Delta(Q)) lt.tilde t_"mix". $

Consequently,
$ ||overline(phi)_k||_infinity <= C (1 - alpha a)^((n-k) slash 2) t_"mix" ||epsilon.alt||_infinity. $

Applying Lemma 11 to $U_3^((w)) = sum_(k=2)^(n-1) overline(phi)_k (Z_(k-1))$, we obtain
$ ||U_3^((w))||_(L_p) lt.tilde p^(1 slash 2) sqrt(t_"mix")
(sum_(k=2)^(n-1) ||overline(phi)_k||_infinity^2)^(1 slash 2)
lt.tilde p^(1 slash 2) t_"mix" ||epsilon.alt||_infinity
(sum_(k=2)^(n-1) (1 - alpha a)^(n-k))^(1 slash 2). $

Since
$ sum_(k=2)^(n-1) (1 - alpha a)^(n-k)
= sum_(j=1)^(n-2) (1 - alpha a)^j <= frac(1, alpha a), $
we conclude that
$ ||U_3^((w))||_(L_p) lt.tilde p^(1 slash 2) t_"mix" ||epsilon.alt||_infinity frac(1, sqrt(alpha a)). $

Combining the bounds for $U_1^((w)), U_2^((w)), U_3^((w))$, we finally obtain
$ (bb(E)_(Pi_(J, alpha)) ||S_n - bb(E) S_n||^p)^(1 slash p)
lt.tilde alpha (p^(3 slash 2) t_"mix"^(1 slash 2) frac(1, alpha a) + p^(1 slash 2) t_"mix" frac(1, sqrt(alpha a))) $

#import "defs.typ": *

== Linear Stochastic Approximation (LSA)

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

== Richardson-Romberg (RR) Step

Consider
$
theta_n^(("RR", alpha)) - theta^* &= 2(theta_n^((alpha)) - theta^*) - (theta_n^((2 alpha)) - theta^*) \
&= [2 Gamma_(1:n)^((alpha))(theta_0 - theta^*) - Gamma_(1:n)^((2 alpha))(theta_0 - theta^*)] \
&quad + [2 J_n^((0, alpha)) - J_n^((0, 2 alpha))]
  + [2 J_n^((1, alpha)) - J_n^((1, 2 alpha))]
  + [2 H_n^((1, alpha)) - H_n^((1, 2 alpha))].
$

Define
$ tilde(J)_n^((0, alpha)) = 2 J_n^((0, alpha)) - J_n^((0, 2 alpha)), quad
J_n^((0, alpha)) = -alpha sum_(j=1)^n (I - alpha overline(A))^(n-j) epsilon.alt(Z_j). $

Then
$
tilde(J)_n^((0, alpha))
&= -2 alpha sum_(j=1)^n [(I - alpha overline(A))^(n-j) - (I - 2 alpha overline(A))^(n-j)] epsilon.alt(Z_j) \
&= -2 alpha^2 overline(A) sum_(j=1)^n
  [sum_(i=1)^(n-j) (I - alpha overline(A))^(i-1) (I - 2 alpha overline(A))^(n-j-i)] epsilon.alt(Z_j) \
&= -2 alpha^2 overline(A) sum_(j=1)^n H_j^((n)) thin epsilon.alt(Z_j),
$
where
$ H_j^((n)) := sum_(i=1)^(n-j) (I - alpha overline(A))^(i-1) (I - 2 alpha overline(A))^(n-j-i). $

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

$
||H_j^((n))|| &= lr(||sum_(i=1)^(n-j) (I - alpha overline(A))^(i-1) (I - 2 alpha overline(A))^(n-j-i)||) \
&<= sum_(i=1)^(n-j) ||I - alpha overline(A)||^(i-1) ||I - 2 alpha overline(A)||^(n-j-i) \
&<= sum_(i=1)^(n-j) C_A (1 - alpha a)^(frac(i-1, 2)) (1 - 2 alpha a)^(frac(n-j-i, 2)) \
&= C_A sum_(k=0)^(n-j-1) (1 - 2 alpha a)^(k slash 2) (1 - alpha a)^((n-j-k-1) slash 2) \
&= C_A (1 - alpha a)^((n-j-1) slash 2) sum_(k=0)^(n-j-1) ((1 - 2 alpha a) / (1 - alpha a))^(k slash 2) \
&<= C_A (1 - alpha a)^((n-j-1) slash 2) frac(1, 1 - ((1 - 2 alpha a) / (1 - alpha a))^(1 slash 2))
$

$ ((1 - 2 alpha a) / (1 - alpha a))^(1 slash 2)
= (1 - frac(alpha a, 1 - alpha a))^(1 slash 2)
= 1 - frac(alpha a, 2(1 - alpha a)) + O((frac(alpha a, 1 - alpha a))^2)
= 1 - frac(1, 2) alpha a + O(alpha^2) $

$ 1 - ((1 - 2 alpha a) / (1 - alpha a))^(1 slash 2) = frac(1, 2) alpha a + O(alpha^2) $

$ frac(1, 1 - ((1 - 2 alpha a) / (1 - alpha a))^(1 slash 2)) = frac(2, alpha a) + O(1) $

$ ||H_j^((n))|| <= overline(C)_A (1 - alpha a)^((n-j-1) slash 2) frac(2, alpha a) $

To explore $tilde(J)_n^((0, alpha))$ we use the following statement:

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

$ tilde(J)_n^((0, alpha)) = -sum_(j=1)^n 2 alpha^2 overline(A) H_j^((n)) epsilon.alt(Z_j) $

$
||2 alpha^2 overline(A) H_j^((n))||_infinity
  &<= 2 alpha^2 tilde(C)_A (1 - alpha a)^((n-j-1) slash 2) frac(2, alpha a)
  = 4 alpha tilde(C)_A (1 - alpha a)^((n-j-1) slash 2) \
"where" quad tilde(C)_A &= ||A|| thin overline(C)_A
$

$
u_n^2
  &<= 64 dot 16 alpha^2 tilde(C)_A^2 sum_(j=1)^n (1 - alpha a)^(n-j-1) \
  &= hat(C)_A^2 alpha^2 frac(1, 1 - alpha a) sum_(k=0)^(n-1) (1 - alpha a)^k \
  &<= hat(C)_A^2 alpha^2 frac(1, 1 - alpha a) frac(1, 1 - 1 + alpha a) \
  &= hat(C)_A^2 alpha
$

Now use classic lemma:

#lemma[
  Let $X$ be an $bb(R)^d$-valued random variable satisfying
  $ bb(P)(||X|| >= t) <= 2 exp(-frac(t^2, 2 sigma^2)) quad "for all" t >= 0, $
  for some $sigma^2 > 0$. Then, for any $p >= 2$, it holds that
  $ bb(E)[||X||^p] <= 2 thin p^(p slash 2) thin sigma^p. $
]

$ frac(1, sqrt(alpha)) bb(E)^(1 slash 2) ||tilde(J)_n^((0, alpha))||^2 <= frac(1, sqrt(alpha)) dot 2 dot sqrt(alpha) hat(C)_A = 2 hat(C)_A $

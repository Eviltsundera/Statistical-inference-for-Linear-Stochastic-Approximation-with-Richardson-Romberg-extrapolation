#import "../defs.typ": *

== Deterministic Transient After Burn-in

For a generic step size $w in {alpha, 2 alpha}$ define the deterministic
transient
$
D_(op("tr"), n, n_0)^((w))
  := frac(1, sqrt(m)) sum_(k = n_0)^(n - 1)
      B_w^k (theta_0 - theta^*),
quad
B_w := I - w overline(A).
$ <eq:burn-single-transient>
The Richardson--Romberg transient entering @eq:burn-target is
$
D_(op("tr"), n, n_0)^("RR")(u)
  := u^top lr((
      2 D_(op("tr"), n, n_0)^((alpha))
      - D_(op("tr"), n, n_0)^((2 alpha))
    )).
$ <eq:burn-RR-transient>

#lemma[
  Assume $alpha, 2 alpha in (0, alpha_infinity]$ and the Lyapunov contraction
  @eq:contraction. Then, for each $w in {alpha, 2 alpha}$,
  $
  || D_(op("tr"), n, n_0)^((w)) ||
    <= frac(2 sqrt(kappa_Q), w a sqrt(m))
       (1 - w a)^(n_0 slash 2) || theta_0 - theta^* ||.
  $ <eq:burn-single-transient-bound>
  Consequently,
  $
  | D_(op("tr"), n, n_0)^("RR")(u) |
    <= frac(5 sqrt(kappa_Q) || u || || theta_0 - theta^* ||,
             alpha a sqrt(m))
       (1 - alpha a)^(n_0 slash 2).
  $ <eq:burn-RR-transient-bound>
] <lem:burn-deterministic-transient>

_Proof._ Let $r_w := sqrt(1 - w a)$. By @eq:contraction and equivalence of the
Euclidean and $Q$-norms,
$
|| B_w^k (theta_0 - theta^*) ||
  <= sqrt(kappa_Q) r_w^k || theta_0 - theta^* ||.
$
Therefore
$
|| D_(op("tr"), n, n_0)^((w)) ||
  <= frac(sqrt(kappa_Q) || theta_0 - theta^* ||, sqrt(m))
     sum_(k = n_0)^(n - 1) r_w^k
  <= frac(sqrt(kappa_Q) || theta_0 - theta^* ||, sqrt(m))
     frac(r_w^(n_0), 1 - r_w).
$
Since $1 - sqrt(1 - x) >= x slash 2$ for $0 <= x <= 1$, the last display gives
@eq:burn-single-transient-bound. Applying this estimate at $w = alpha$ and
$w = 2 alpha$, using $1 - 2 alpha a <= 1 - alpha a$, and taking the
triangle inequality in @eq:burn-RR-transient gives
$
| D_(op("tr"), n, n_0)^("RR")(u) |
  <= || u || lr((
       2 || D_(op("tr"), n, n_0)^((alpha)) ||
       + || D_(op("tr"), n, n_0)^((2 alpha)) ||
     ))
  <= frac(5 sqrt(kappa_Q) || u || || theta_0 - theta^* ||,
           alpha a sqrt(m))
       (1 - alpha a)^(n_0 slash 2).
$
$square$

#corollary[
  *(Mixing-scale burn-in removes the deterministic transient.)*
  If, for some $beta > 0$,
  $
  n_0 >= frac(2 beta, alpha a) log n,
  $ <eq:burn-log-condition>
  then
  $
  | D_(op("tr"), n, n_0)^("RR")(u) |
    <= frac(5 sqrt(kappa_Q) || u || || theta_0 - theta^* ||,
             alpha a sqrt(m)) n^(-beta).
  $ <eq:burn-log-transient-bound>
  In particular, at the balanced scale $alpha = c n^(-1 slash 2)$ and
  $m >= n slash 2$, this is $O(n^(-beta))$. Taking $beta = 1$ makes the
  deterministic transient negligible relative to the stationary
  $n^(-1 slash 4)$ Berry--Esseen rate.
] <cor:burn-log-transient>

_Proof._ The elementary inequality
$(1 - alpha a)^(n_0 slash 2) <= exp(-alpha a n_0 slash 2)$ and
@eq:burn-log-condition imply
$(1 - alpha a)^(n_0 slash 2) <= n^(-beta)$. Substitute this into
@eq:burn-RR-transient-bound. If $alpha = c n^(-1 slash 2)$ and $m >= n slash 2$,
then $(alpha sqrt(m))^(-1) <= sqrt(2) slash c$, so the displayed
$O(n^(-beta))$ bound follows. $square$

The deterministic transient is not the full initial-condition contribution:
the exact recursion contains the random product
$Gamma_(1:k)^((w)) := product_(j = 1)^k (I - w A(Z_j))$. The difference
between this product and $B_w^k$ is another finite-start term. We use the same
random-product stability input that appears in Levin et al. (2025,
Appendix D.1, Proposition 9). Let $alpha_("st")(p)$ denote the minimum of the
Levin depth-two startup ceiling and the product-stability ceiling at moment
order $2p$; the startup section below uses the same threshold.

#lemma[
  *(Imported random-product stability.)*
  Under the stability and bounded-noise assumptions used in Levin et al.
  (2025, Appendix D.1, Proposition 9), if $2 alpha <= alpha_("st")(p)$, then
  there exist constants $C_("prod") < infinity$ and $c_("prod") > 0$ such
  that, for every $p >= 2$, every $w in {alpha, 2 alpha}$, every $0 <= s < k$,
  and every $cal(F)_s$-measurable vector $V_s$,
  $
  || Gamma_(s + 1:k)^((w)) V_s ||_(L_p)
    <= C_("prod") exp(-c_("prod") w a (k - s) slash p)
       || V_s ||_(L_(2p)).
  $ <eq:burn-product-stability>
] <lem:burn-product-stability>

Define the accumulated RR random initial-product discrepancy by
$
cal(I)_(n,n_0)^("init,RR")(u)
  := frac(1, sqrt(m)) sum_(k = n_0)^(n - 1)
      u^top lr([
        2 lr((Gamma_(1:k)^((alpha)) - B_alpha^k))
        - lr((Gamma_(1:k)^((2 alpha)) - B_(2 alpha)^k))
      ]) (theta_0 - theta^*),
$ <eq:burn-random-init-discrepancy>
where the empty product at $k = 0$ is the identity.

#lemma[
  *(Burned-in random initial-product transient.)*
  Assume the hypotheses of @lem:burn-product-stability,
  $alpha, 2 alpha in (0, alpha_infinity]$, $alpha a <= 1 slash 4$, and
  the Lyapunov contraction @eq:contraction. Then, for every $p >= 2$,
  $
  || cal(I)_(n,n_0)^("init,RR")(u) ||_(L_p)
    <= frac(C_("init,RR") ||u|| ||theta_0 - theta^*|| p,
             alpha a sqrt(m))
       exp(-c_("init") alpha a n_0 slash p),
  $ <eq:burn-random-init-bound>
  where $C_("init,RR") < infinity$ and $c_("init") > 0$ depend only on
  $C_("prod")$, $c_("prod")$, and the Lyapunov norm-equivalence constant.
] <lem:burn-random-initial-product>

_Proof._ Let $e_0 := theta_0 - theta^*$. By @lem:burn-product-stability with
$s = 0$ and $V_0 = e_0$,
$
|| Gamma_(1:k)^((w)) e_0 ||_(L_p)
  <= C_("prod") exp(-c_("prod") w a k slash p) ||e_0||.
$
The deterministic Lyapunov contraction gives
$
|| B_w^k e_0 ||
  <= sqrt(kappa_Q) (1 - w a)^(k slash 2) ||e_0||
  <= sqrt(kappa_Q) exp(-w a k slash (2 p)) ||e_0||,
$
since $p >= 1$. Hence, after decreasing the exponential constant,
$
|| lr((Gamma_(1:k)^((w)) - B_w^k)) e_0 ||_(L_p)
  <= C thin exp(-c_("init") w a k slash p) ||e_0||.
$
Apply this estimate at $w = alpha$ and $w = 2 alpha$, use the triangle
inequality in @eq:burn-random-init-discrepancy, and extend the geometric sum
to infinity:
$
|| cal(I)_(n,n_0)^("init,RR")(u) ||_(L_p)
  <= frac(C ||u|| ||e_0||, sqrt(m))
     sum_(k = n_0)^infinity exp(-c_("init") alpha a k slash p).
$
Because $alpha a <= 1 slash 4$ and $p >= 2$,
$1 - exp(-c_("init") alpha a slash p) >= C^(-1) alpha a slash p$, which gives
@eq:burn-random-init-bound. $square$

#corollary[
  *(Mixing-scale burn-in removes the random initial-product discrepancy.)*
  If, for some $beta > 0$,
  $
  n_0 >= frac(beta p, c_("init") alpha a) log n,
  $ <eq:burn-log-init-condition>
  then
  $
  || cal(I)_(n,n_0)^("init,RR")(u) ||_(L_p)
    <= frac(C_("init,RR") ||u|| ||theta_0 - theta^*|| p,
             alpha a sqrt(m)) n^(-beta).
  $ <eq:burn-log-init-bound>
  At the balanced scale $alpha = c n^(-1 slash 2)$, with
  $m >= n slash 2$ and $p$ logarithmic in $n$, this is
  $"polylog"(n) thin n^(-beta)$.
] <cor:burn-log-initial-product>

_Proof._ Substitute @eq:burn-log-init-condition into
@eq:burn-random-init-bound. At the balanced scale,
$(alpha sqrt(m))^(-1) = O(1)$, so the remaining factor is logarithmic. $square$


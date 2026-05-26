#import "../defs.typ": *

== Richardson--Romberg extrapolation

To eliminate the leading $O(alpha)$ bias term, we employ the _Richardson--Romberg_ (RR) _extrapolation_ procedure.
Two LSA sequences are run _on the same Markov chain trajectory_ ${Z_k}$ with step sizes $alpha$ and $2 alpha$, and the RR iterate is formed as
$ overline(theta)_n^((alpha, "RR")) = 2 overline(theta)_n^((alpha)) - overline(theta)_n^((2 alpha)). $ <eq:rr-iterate>
Since both sequences share the same noise realization, the leading bias term $alpha Delta$ cancels, leaving a residual bias of order $O(alpha^(3\/2))$ or higher (Levin et al., 2025).

More generally, one can consider the multi-level extrapolation with $M$ step sizes $cal(A) = {alpha_1, dots, alpha_M}$ and coefficients ${h_m}$ determined by the Vandermonde system (Huo et al., 2024):
$ sum_(m=1)^M h_m = 1, quad sum_(m=1)^M h_m alpha_m^l = 0, quad l = 1, dots, M-1, $
which cancels successive powers in settings where such a power-series
expansion is available.

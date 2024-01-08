# APPENDIX
This section includes the proofs of *Lemma 1* and *Proposition 1*.  
The numbering cited in this section (e.g., (1)) corresponds to the numbered equations in the main text.

## Proof of *Lemma 1*

###### *Lemma 1:*  
Suppose $\mu$ is a positive regular Borel measure, and $f, e$ are Simple Functions that  $f(x) = \sum_{i=1}^n {{\alpha}}_i \chi_{I_{i}^{f}}(x)$ and $e(x) = \sum_{j=1}^m {{\beta}}_j \chi_{I_{j}^{e}}(x)$ then the distance $d(f,e)$ defined under (1) can be expressed as
$$ 
d(f,e) = \sum_{i=1}^n \sum_{j=1}^m \rho({{\alpha}}_i , {{\beta}}_j) \cdot \mu (I_{i}^{f} \cap I_{j}^{e}).
$$

###### *Proof:*
Follow (1), we have
$$
    d(f,e) = \int_{\mathbb X} \rho(\sum_{i=1}^n \alpha_i \cdot \chi_{I_{i}^{f}}(x),\sum_{j=1}^m {{\beta}}_j \cdot \chi_{I_{j}^{e}}(x)) \,  d\mu .
$$
Under the definition of Simple Function, $\{{I_{i}^{f}}\}_{i=1}^{n}$ and $\{{I_{i}^{e}}\}_{i=1}^{m}$ are both measurable partitions of $\mathbb X$. 
Therefor 
$$\sum_{i=1}^{n} \chi_{I_{i}^{f}}(x) \equiv 1 \, ,\, \sum_{j=1}^{m} \chi_{I_{j}^{e}}(x) \equiv 1.$$
Consequently,
$$
\sum_{i=1}^n \alpha_i \chi_{I_{i}^{f}}(x) = \sum_{i=1}^n \alpha_i \cdot (\chi_{I_{i}^{f}}(x)\cdot \sum_{j=1}^{m} \chi_{I_{i}^{e}}(x)) = \sum_{i=1}^n \sum_{j=1}^{m} \alpha_i \cdot \chi_{ {I_{i}^{f}} \cap {I_{i}^{e}}}(x).
$$
Similarly, 
$$\sum_{j=1}^m {{\beta}}_j \cdot \chi_{I_{j}^{e}}(x)= \sum_{j=1}^m \sum_{i=1}^n{{\beta}}_j \cdot \chi_{I_{j}^{e} \cap I_{i}^{f}}(x).$$ 
Hence, we obtain
$$   d(f,e) = \int_{\mathbb X} \sum_{i=1}^n \sum_{j=1}^{m} \rho(\alpha_i , {{\beta}}_j) \cdot \chi_{{I_{i}^{f}} \cap {I_{j}^{e}}}(x) \,  d\mu .
$$
Notably, $\{ {I_{i}^{f}} \cap {I_{j}^{e}} \}$ is also a measurable partition of $\mathbb X$, and $\mu$ is a positive regular Borel measure. Thus, we can further simplify the expression as
$$
\begin{aligned}
    d(f,e) 
    & = \sum_{i=1}^n \sum_{j=1}^{m} \int_{ {I_{i}^{f}} \cap {I_{j}^{e}}} \sum_{i=1}^n \sum_{j=1}^{m} \rho(\alpha_i , {{\beta}}_j) \cdot \chi_{{I_{i}^{f}} \cap {I_{j}^{e}}}(x) \,  d\mu \\
    & = \sum_{i=1}^n \sum_{j=1}^{m} \int_{{I_{i}^{f}} \cap {I_{j}^{e}}} \rho(\alpha_i , {{\beta}}_j) \cdot \chi_{{I_{i}^{f}} \cap {I_{j}^{e}}}(x) \,  d\mu \\
    &= \sum_{i=1}^n \sum_{j=1}^{m} \rho(\alpha_i , {{\beta}}_j) \, \cdot \mu(I_{i}^{f} \cap I_{j}^{e}).
\end{aligned}
$$
$\square$
 
## Proof of *Proposition 1*
###### *Proposition 1:* 
Under *Problem 2* and (4), (5) , let the function $\rho ({p}_{1} , {p}_{2})$ is linear depend on ${p}_{1}$ for a fixed ${p}_{2}$, then the $d(\cal F, \cal E)$ in (8) is equivalent to
$$
d({\cal F}, {\cal E}) = \sum_{k=1}^K \sum_{i=1}^{L^{(k)}} \sum_{j=1}^m \rho({p}_{i}^{(k)} , {p}_{j}^{\cal E}) \cdot \mu (I_{i}^{(k)} \cap I_{j}^{\cal E}).
$$
###### *Proof: *
If $\rho ({p}_{1} , {p}_{2})$ is linear depend on ${p}_{1}$ for a fixed ${p}_{2}$, under (5) and (6), we can rewrite the $\rho$ in (8) as:
$$
   \rho({p}_{i}^{\cal F} , {p}_{j}^{\cal E}) = \frac{1}{K}\sum_{k=1}^K\rho({p}_{i^k}^{(k)}, {p}_{j}^{\cal E}),
$$
where $I_{i}^{\cal F} \subseteq I_{i^k}^{(k)}$ for every $k$.
Therefor,
$$
\begin{aligned}
d({\cal F},{\cal E}) 
&= \sum_{i=1}^{L^{\cal F}} \sum_{j=1}^m (\frac{1}{K}\sum_{k=1}^K \rho({p}_{i^k}^{(k)}, {p}_{j}^{\cal E})) \cdot \mu (I_{i}^{\cal F} \cap I_{j}^{\cal E})\\
&=\frac{1}{K}\sum_{i=1}^{L^{\cal F}} \sum_{j=1}^m \sum_{k=1}^K \rho({p}_{i^k}^{(k)}, {p}_{j}^{\cal E}) \cdot \mu (I_{i}^{\cal F} \cap I_{i^k}^{(k)} \cap I_{j}^{\cal E}).
\end{aligned}
$$
Furthermore, for every $l \neq {i^k}$, we have $I_{i}^{\cal F} \cap I_{l}^{(k)} = \emptyset$, which implies:
$$
\begin{aligned}
d({\cal F},{\cal E}) 
&=\frac{1}{K} \sum_{i=1}^{L^{\cal F}} \sum_{j=1}^m\sum_{k=1}^K \sum_{l=1}^{L^{(k)}} \rho({p}_{l}^{(k)}, {p}_{j}^{\cal E}) \cdot \mu (I_{i}^{\cal F} \cap I_{l}^{(k)} \cap I_{j}^{\cal E})\\
&= \frac{1}{K} \sum_{k=1}^K \sum_{l=1}^{L^{(k)}} \sum_{j=1}^m \rho({p}_{l}^{(k)}, {p}_{j}^{\cal E}) \cdot \sum_{i=1}^{L^{\cal F}} \mu (I_{i}^{\cal F} \cap I_{l}^{(k)} \cap I_{j}^{\cal E}).
\end{aligned}
$$
As we defined in *Lemma 1*, $\mu$ is a positive regular Borel measure and $\{I_{i}^{\cal F} \}_{i=0}^{L^{\cal F}}$ is a measurable partition of $\mathbb X$. Hence, we obtain
$$
\sum_{i=1}^{L^{\cal F}} \mu (I_{i}^{\cal F} \cap I_{l}^{(k)} \cap I_{j}^{\cal E})
=\mu ((\bigcup_{i=1}^{L^{\cal F}} I_{i}^{\cal F}) \cap I_{l}^{(k)} \cap I_{j}^{\cal E})
=\mu (I_{l}^{(k)} \cap I_{j}^{\cal E}).
$$
Using these results, we can simplify the expression for $d(\cal{F}, \cal{E})$ as follows:
$$
    \begin{aligned}
    d({\cal F},{\cal E})  
    &= \frac{1}{K} \sum_{k=1}^K \sum_{l=1}^{L^{(k)}} \sum_{j=1}^m  \rho({p}_{l}^{(k)}, {p}_{j}^{\cal E}) \cdot \mu ( I_{l}^{(k)} \cap I_{j}^{\cal E})\\
    &= \frac{1}{K} \sum_{k=1}^K \sum_{i=1}^{L^{(k)}} \sum_{j=1}^m  \rho({p}_{i}^{(k)}, {p}_{j}^{\cal E}) \cdot \mu ( I_{i}^{(k)} \cap I_{j}^{\cal E}).
   \end{aligned}
$$
$\square$



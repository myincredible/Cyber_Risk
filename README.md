# Policy Improvement

Given control (updated during the $n^{th}$ iteration) $a^n(x):= (\eta^n,\rho^n)(x)$, solve the Bellman-Type ODE by 

```python
scipy.solve_bvp
```

$$
\frac{1}{2}\sigma^2 \cdot D_x^2 v^{n+1} + b^{a^n} \cdot D_x v^{n+1} + f^{a^n} - \delta v^{n+1} = 0,
\quad x \in [x_0, x_1]\subsetneq (0, 1)
$$

$$
v^{n+1}(x) = \mathbb E\left[\int_0^\infty e^{-\delta t} f^{a^n}(X_t)dt|X_0=x\right], \quad x \in \partial [x_0, x_1]
$$

3. Update the Markov control: 

$$
\eta^{n+1}(x) = 
\begin{cases}1 - \frac{(\alpha + \beta x)(1 - x) D_x v^{n+1}}{2 a_2} & \text{if}\; 1 - \frac{(\alpha + \beta x)(1 - x) D_x v^{n+1}}{2 a_2} \in [0, 1]\\
0 & \text{if}\; 1 - \frac{(\alpha + \beta x)(1 - x) D_x v^{n+1}}{2 a_2} < 0\\
1 & \text{Otherwise}
\end{cases}
$$

$$
\rho^{n+1}(x)=
\begin{cases}
\frac{D_x v^{n+1}}{2a_3x} &\text{if}\; D_x v^{n+1} > 0\\
0 & \text{Otherwise}
\end{cases}
$$

**Return:** $ v^n $ and $ a^{n+1} $


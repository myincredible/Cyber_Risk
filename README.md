# Policy Improvement

**Initialization:**  
Choose an initial (constant in $x$) control $a^0:=(\eta^0, \rho^0)\in \mathcal U_0$. 

**Repeat until** $\|v^{n+1} - v^n\|_{\mathbb L_2}<\epsilon$: 

1. Applying a Monte-Carlo simulation for obtaining boundary points: 

   **Repeat Until** $1.96\times \frac{\hat\sigma_N}{\sqrt N}<\epsilon$ or $\max_{i,j\in [N-M+1, N]}|v_i^{n+1}- v^{n+1}_j|<\delta$: 

   	1. Interpolate control $a^0$ so that one could find any control corresponding to $X_s\in [x_0, x_1]$ for any $s\in [0, \infty)$. 

   	2. Simulate one trajectory of controlled SDE by discretizing the sample path: 
       $$
       \Delta X_t:= X_{t+dt} - X_t = b^{a^n(X_t)}(X_t)\cdot dt + \sigma(X_t) \cdot \sqrt{dt}\cdot W,
       $$
       where $W\sim N(0, 1)$ is a standard normal random variable simulated by 

       ```python
       np.random.randn()
       ```

       Simulate one trajectory of objective function. **Repeat Until** $e^{-\delta t} f^{a^n(X_t)}{(X_t)}\leq \epsilon_{f}$: 

       **Update** the Riemann integral using the trapezoidal rule
       $$
       v^{n+1}_i(x):= \int_0^\infty e^{-\delta t} f^{a^n}(X_t)dt \leftarrow \frac12\left(e^{-\delta t} f^{a^n(X_t)}{(X_t)} + e^{-\delta (t+dt)} f^{a^n(X_t+dt)}{(X_t+dt)}\right) dt
       $$

   	3. **Update** objective function on boundary values $v^{n+1}$: 
       $$
       v^{n+1}(x):=\mathbb E\left[\int_0^\infty e^{-\delta t} f^{a^n}(X_t)dt|X_0=x\right]\leftarrow v^{n}(x) + \frac1n \left(v^{n+1}_i(x) -v^n(x)\right)
       $$

   	4. 

   

2. Given control (updated during the $n^{th}$ iteration) $a^n(x):= (\eta^n,\rho^n)(x)$, solve the Bellman-Type ODE by 

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


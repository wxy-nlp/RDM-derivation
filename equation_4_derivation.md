>**`Q3`**: The equation in 2.2 misses a "+" after const.? Also, how do we get the equation 4?

A3: Thank you for pointing it out! We rewrite the equation 2.2.
For equation 4, we follow [1] and derive this equation as below:

[1] introduces the reparameterized trick to discrete diffusion model, and the backward transition $q(\bm{x}^{(t-1)}|\bm{x}^{(t)}, \bm{x}^{(0)})$ can be rewritten as:
$$\begin{align}
   & q(\bm{x}^{(t-1)}|\bm{x}^{(t)}, \bm{x}^{(0)}) \nonumber \\
    &= \begin{cases}
\lambda_{t-1}^{(1)}\bm{x}^{(t)} + (1-\lambda_{t-1}^{(1)})\bm{q}_{\text{noise}}, &&\text{if } \bm{x}^{(t)}=\bm{x}^{(t)} \nonumber \\
\lambda_{t-1}^{(2)}\bm{x}^{(t)} + (1-\lambda_{t-1}^{(2)})\bm{q}_{\text{noise}}(\bm{x}^{(t)}), &&\text{if } \bm{x}^{(t)} \not=\bm{x}^{(0)}
\end{cases}
\end{align}$$
where $\bm{q}_{\text{noise}}(\bm{x}^{(t)}) = \beta_t\bm{x}^{(t)} + (1-\beta_t)\bm{q}_{\text{noise}}$, and both $\lambda_{t-1}^{(1)}$ and $\lambda_{t-1}^{(2)}$ are constants relating to $\beta_t$ and $\beta_{t-1}$.
Sampling from it is equivalent to first sampling from a Bernoulli distribution and then the corresponding component distribution:
$$\begin{aligned}
v_{t-1}^{(1)}\sim \text{Bernoulli}\left(\lambda_{t-1}^{(1)}\right),&&\bm{u}_t^{(1)}\sim \texttt{Cat}\left(\bm{u}; \bm{p}=\bm{q}_{\text{noise}}\right)\\
v_{t-1}^{(2)}\sim \text{Bernoulli}\left(\lambda_{t-1}^{(2)}\right),&& 
\bm{u}_t^{(2)}\sim \texttt{Cat}\left(\bm{u}; \bm{p}=\bm{q}_{\text{noise}}(\bm{x}_t) \right)
\end{aligned}\\$$
$$\bm{x}_{t-1}=\left\{
\begin{aligned}
v_{t-1}^{(1)}\bm{x}_t + \left(1-v_{t-1}^{(1)}\right)\bm{u}_t^{(1)},&&\text{if }\bm{x}_t=\bm{x}_0\\
v_{t-1}^{(2)}\bm{x}_t + \left(1-v_{t-1}^{(2)}\right)\bm{u}_t^{(2)},&&\text{if }\bm{x}_t\not=\bm{x}_0
\end{aligned}
\right.$$
This reparameterizes the backward transitions $q(\bm{x}^{(t-1)}|\bm{x}^{(t)}, \bm{x}^{(0)})$ and $p_{\theta}(\bm{x}^{(t-1)}|\bm{x}^{(t)})$ into $q(\bm{x}^{(t-1)},\bm{v}^{(t-1)}|\bm{x}^{(t)}, \bm{x}^{(0)})$ and $p_{\theta}(\bm{x}^{(t-1)},\bm{v}^{(t-1)}|\bm{x}^{(t)})$, respectively.

Since each token is modeled **conditionally independently**, so we consider the backward transition for **each token**, and sum the losses for them.
For i-th position, the backward transition is $q(\bm{x}_i^{(t-1)},\bm{v}_i^{(t-1)}|\bm{x}_i^{(t)}, \bm{x}_i^{(0)})$.

As shown in [1] (appendix C), the loss at i-th token can be written as below:
$$
\mathcal{J}_{t,i} = \mathbf{E}_{q(\bm{v}_i^{(t-1)})}\left[KL[q(\bm{x}_i^{(t-1)}|\bm{v}_i^{(t-1)},\bm{x}_i^{(t)}, \bm{x}_i^{(0)})||p_{\theta}(\bm{x}_i^{(t-1)}|\bm{v}_i^{(t-1)},\bm{x}_i^{(t)})]\right]
$$

Let $b_i{(t)}=\mathbf{1}_{x_i^{(t)}=x_i^{(0)}}$, $q(\bm{x}_i^{(t-1)}|\bm{v}_i^{(t-1)},\bm{x}_i^{(t)}, \bm{x}_i^{(0)})$ can be written as:
$$\begin{align}
   & q(\bm{x}_i^{(t-1)}|\bm{v}_i^{(t-1)},\bm{x}_i^{(t)}, \bm{x}_i^{(0)}) \nonumber \\
    &= \begin{cases}
v_{t-1,i}^{(1)}\bm{x}_i^{(t)} + (1-v_{t-1,i}^{(1)})\bm{q}_{\text{noise}}  &&\text{if } b_i{(t)}=0, \nonumber \\
v_{t-1,i}^{(2)}\bm{x}_i^{(0)} + (1-v_{t-1,i}^{(2)})\bm{q}_{\text{noise}}  &&\text{if } b_i{(t)}=1,
\end{cases}
\end{align}$$

And $p_{\theta}(\bm{x}_i^{(t-1)}|\bm{v}_i^{(t-1)},\bm{x}_i^{(t)})$ can be written as:
$$\begin{align}
   & p_{\theta}(\bm{x}_i^{(t-1)}|\bm{v}_i^{(t-1)},\bm{x}_i^{(t)}) \nonumber \\
    &= \begin{cases}
v_{t-1,i}^{(1)}\bm{x}_i^{(t)} + (1-v_{t-1,i}^{(1)})\bm{q}_{\text{noise}}  &&\text{if } b_i{(t)}=0, \nonumber \\
v_{t-1,i}^{(2)}p_{\theta}(\bm{x}_i^{(0)}|\bm{x}^{(t)}) + (1-v_{t-1,i}^{(2)})\bm{q}_{\text{noise}}  &&\text{if } b_i{(t)}=1,
\end{cases}
\end{align}$$

Therefore, the loss at i-th token can be computed by enumerating all cases with respect to $\bm{v}_i^{(t-1)}$ and $b_i(t)$. As noted in [1], the KL divergence is equal to $-\log p_{\theta}(x_i^{(0)}|x^{(t)})$ when $v_{t-1,i}^{(2)}=1$ and $b_i(t)=1$, while in other cases the KL divergence is 0. 

So we have:
$$\begin{align}
   & \mathcal{J}_t = \sum_{1 \leq i \leq L}\mathcal{J}_{t,i} \nonumber \\
   &= \sum_{1 \leq i \leq L} \mathbf{E}_{q(\bm{v}_i^{(t-1)})}\left[KL[q(\bm{x}_i^{(t-1)}|\bm{v}_i^{(t-1)},\bm{x}_i^{(t)}, \bm{x}_i^{(0)})||p_{\theta}(\bm{x}_i^{(t-1)}|\bm{v}_i^{(t-1)},\bm{x}_i^{(t)})]\right] \nonumber \\
    &= 
\sum_{1 \leq i \leq L} q(\bm{v}_i^{(t-1)}=1) (-\log p_{\theta}(x_i^{(0)}|x^{(t)})) \nonumber \\
&= 
-\lambda^{(t)}  {\sum_{1 \leq i \leq L}} b_i(t) \cdot \log p_{\theta}(\bm{x}_i^{(0)}|\bm{x}^{(t)}) \nonumber

\end{align}$$
[1] Zheng, L., Yuan, J., Yu, L., and Kong, L. A reparameterized discrete diffusion model for text generation. arXiv preprint arXiv:2302.05737
// Dixon-Coles bayesiano hierárquico com verossimilhança ponderada (decaimento temporal).
// Mesma parametrização de src/modelo.py. Com pesos, a posterior é uma pseudo-posterior.
data {
  int<lower=1> N;
  int<lower=2> T;
  array[N] int<lower=1, upper=T> m;
  array[N] int<lower=1, upper=T> v;
  array[N] int<lower=0> x;
  array[N] int<lower=0> y;
  vector<lower=0>[N] w;
}
parameters {
  real mu0;
  real casa;
  real<lower=-0.2, upper=0.2> rho;
  real<lower=0> sd_at;
  real<lower=0> sd_de;
  vector[T] z_at;
  vector[T] z_de;
}
transformed parameters {
  vector[T] ataque = sd_at * z_at;
  vector[T] defesa = sd_de * z_de;
}
model {
  mu0 ~ normal(0.2, 0.5);
  casa ~ normal(0.3, 0.3);
  sd_at ~ normal(0, 0.5);
  sd_de ~ normal(0, 0.5);
  z_at ~ std_normal();
  z_de ~ std_normal();
  for (i in 1:N) {
    real lam = exp(mu0 + casa + ataque[m[i]] + defesa[v[i]]);
    real mu = exp(mu0 + ataque[v[i]] + defesa[m[i]]);
    real tau = 1;
    if (x[i] == 0 && y[i] == 0) tau = 1 - lam * mu * rho;
    else if (x[i] == 0 && y[i] == 1) tau = 1 + lam * rho;
    else if (x[i] == 1 && y[i] == 0) tau = 1 + mu * rho;
    else if (x[i] == 1 && y[i] == 1) tau = 1 - rho;
    target += w[i] * (log(fmax(tau, 1e-10)) + poisson_lpmf(x[i] | lam) + poisson_lpmf(y[i] | mu));
  }
}

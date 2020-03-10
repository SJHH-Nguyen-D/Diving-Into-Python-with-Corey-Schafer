from scipy.optimize import anneal

result = anneal(func, x0, args=(), 
	schedule='fast', full_output=True, T0=None, 
	Tf=1e-12, maxeval=None, maxaccept=None, maxiter=400, 
	boltzmann=1.0, learn_rate=0.5, feps=1e-06, quench=1.0, 
	m=1.0, n=1.0, lower=-100, upper=100, dwell=50, disp=True)

print(result)


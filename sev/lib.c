#include <stdio.h>
#include <math.h>

double f(int n, double args[n]){
	float x, mu_p, sigma_p, tmrel, rr_mean, rr_max, mu, sigma, px, y;
	int flag;
	x = args[0];
	mu_p = args[1];
	sigma_p = args[2];
	tmrel = args[3];
	rr_mean = args[4];
	rr_max = args[5];
	flag = args[6];

	mu = log(mu_p / sqrt(1 + pow(sigma_p,2) / pow(mu_p,2))); 
	sigma = sqrt(log(1 + pow(sigma_p,2) / pow(mu_p,2))); 
	px = 1 / ( x * sigma * sqrt(2 * 3.14)) * exp(- pow((log(x) - mu),2) / (2 * pow(sigma,2)));

	if (flag == 1) {
		if (x < tmrel) {
			if ((pow(rr_mean,tmrel - x)) < rr_max) {
				y = (pow(rr_mean,tmrel - x)) * px;
			} else {
				y = rr_max * px;
			}
			
		} else {
			y = px;
		}
	} else {
		if (x > tmrel) {
			if (pow(rr_mean,(x - tmrel)) < rr_max) {
				y = (pow(rr_mean,x - tmrel)) * px;
			} else {
				y = rr_max * px;
			}
		} else {
			y = px;
		}
	}

    return y; 
}
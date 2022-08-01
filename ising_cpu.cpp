#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <cmath> 
#include <chrono>
#include <ctime>
#include <algorithm>
using namespace std;

int L; //length of the system
int N; // total number of sites
double temp=2.0; 
double J = 1.0; 
double Mtot; 
double Energy; 

int MC_sweep = 500; 
int Nmeasure = 1000; 
int Ndecorr = 10;

double de[3][9]; 
double w[3][9]; 


double random_number()
{
	return rand()/(double)RAND_MAX; 
}


void set_temp(double t)
{

	for(int i=-4;i<5;i++)
	{
		de[0][i+4] = -2*J*i; 
		de[2][i+4] = 2*J*i; 
		double tmp = exp(-de[0][i+4]/t);
		w[0][i+4] = min(tmp, 1.0); 
		w[2][i+4] = min(1.0/tmp, 1.0);  

	}

}

void Initialize_ising_model(int *spin, int **nn, double temp)
{

	srand(time(NULL)); 

	//initialize spin 
	for(int i=0;i<N;i++)
		{
			double tmp = random_number() - 0.5;
			if(tmp < 0 )
				spin[i] = -1; 
			else 
				spin[i] = 1;  
		}

	//periodic boundary condition
	for(int i=0;i<L;i++)
		for(int j=0;j<L;j++)
		{
			int num = i*L + j; 
			nn[num][0] = i*L + (j+1)%L; // right
			nn[num][1] = ((i-1+L)%L) * L + j; //top
			nn[num][2] = i*L + (j-1+L)%L; //left
			nn[num][3] = ((i+1)%L) * L + j; // bottom
		}

	Mtot = 0.0; 
	for(int i=0;i<N;i++)
		Mtot += spin[i]; 

	printf("Initial Magnetization: %.6lf\n", Mtot/N);

	Energy = 0.0; 
	for(int i=0;i<N;i++)
		Energy += -J*spin[i]*( spin[ nn[i][0] ] + spin[ nn[i][1] ] );

	printf("Initial Energy: %.6lf\n", Energy/N);

	set_temp(temp); 

}


/*
int metropolis(int *spin, int **nn)
{
	int nchanges = 0; 
	for(int i=0;i<N;i++)
	{
		int pos = (int)(random_number()*N); 
		int spin_sum = spin[nn[pos][0]] + spin[nn[pos][1]] + spin[nn[pos][2]] + spin[nn[pos][3]]; 
		int s = spin[pos]; 
		double deltaE = de[s+1][spin_sum + 4]; 

		if(deltaE <= 0.0 or random_number() < w[s+1][spin_sum + 4])
		{
			spin[pos] *= -1; 
			Mtot += -2*s; 
			Energy += deltaE; 
			nchanges += 1;
		}
			

	}
	return nchanges; 
}
*/



int main(void)
{

	L = 1024;
	N = L * L;
	temp = 2.0; 

	std::random_device rd;
	std::mt19937 mt(rd()); 
	std::uniform_real_distribution<double> gen(0.0, 1.0); 


	int *spin; 
	spin = (int *)malloc(sizeof(int) * N); 


	int **nn; 
	nn = (int**)malloc(sizeof(int*) * N);
	for(int i=0;i<N;i++)
		nn[i] = (int *)malloc(sizeof(int) * 4); 

	Initialize_ising_model(spin, nn, temp);


	auto t0 = chrono::high_resolution_clock::now();  

	for(int j=0;j<MC_sweep;j++)
	{

		double nchanges = 0.0; 

		for(int i=0;i<N;i++)
		{
			int pos = i; 
			int spin_sum = spin[nn[pos][0]] + spin[nn[pos][1]] + spin[nn[pos][2]] + spin[nn[pos][3]]; 
			int s = spin[pos]; 
			double deltaE = de[s+1][spin_sum + 4]; 

			if(deltaE <= 0.0 or gen(mt) < w[s+1][spin_sum + 4])
			{
				spin[pos] *= -1; 
				Mtot += -2*s; 
				Energy += deltaE; 
				nchanges += 1;
			}
				

		}

	}

	printf("Energy: %.6lf\n", Energy/N); 
	printf("Magnetization: %.6lf\n", Mtot/N);


	auto t1 = chrono::high_resolution_clock::now(); 
	auto elapsed  = t1 - t0; 
    double time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count(); 


    printf("MC sweep per ms: %f\n", (double) MC_sweep/time_ms );




	free(spin); 
	for(int i=0;i<N;i++)
		free(nn[i]); 
	free(nn); 

	return 0; 
}
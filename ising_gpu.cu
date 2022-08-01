#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <unistd.h>
#include <random>
#include <complex>
#include <curand.h>
#include <chrono>
#include <cuda.h>
// #include "cuda_marco.h"

using namespace std; 
double J1 = 1.0; 

const int THREAD = 128; 


__global__ void init_spins(signed char *lattice, const float* __restrict__ randval_d, const long long L )
{
    const long long tid = static_cast<long long> (blockDim.x) * blockIdx.x + threadIdx.x; 

    if(tid >= L*L/2) return ; 
 
    signed char val; 
    if(randval_d[tid] < 0.5f)
        val = -1;
    else 
        val = 1; 

    lattice[tid] = val; 

}


template<bool is_black>
__global__ void metropolis(signed char* lattice, const signed char* __restrict__ op_lattice, const float* __restrict__ randval_d,
                            const float beta, const long long nx, const long long ny)
{

    const long long tid = static_cast<long long>(blockDim.x)*blockIdx.x + threadIdx.x; 
    if(tid >= nx * ny) return ; 

    signed char spin = lattice[tid]; 

    const int i = tid / ny, j = tid % ny; 

    int down = ((i + 1)%nx) * ny + j; 
    int up = ((i-1+nx)%nx) * ny + j; 
    int left = i*ny + (j-1+ny)%ny; 
    int right = i*ny + (j+1)%ny; 
    int nearj; 

    if(is_black)
        nearj = i%2==1? right: left;
    else 
        nearj = i%2==1? left: right; 

    signed char nn_sum = op_lattice[down] + op_lattice[up] + op_lattice[tid] + op_lattice[nearj];
    float dE = 2.0f * nn_sum * spin;  
    float acceptance_ratio = exp(-beta * dE); 
    
    if(randval_d[tid] < acceptance_ratio)
        lattice[tid] = -spin; 

}



void update(signed char *lattice_black, signed char *lattice_white, float *randval_d, curandGenerator_t& gen, float temp, long long L)
{

	int blocks = (L*L + THREAD - 1)/THREAD; 

	curandGenerateUniform(gen, randval_d, L*L/2); 
	metropolis<true><<<blocks, THREAD>>>(lattice_black, lattice_white, randval_d, 1.0/temp, L, L/2); 

	curandGenerateUniform(gen, randval_d, L*L/2); 
    metropolis<false><<<blocks, THREAD>>>(lattice_white, lattice_black, randval_d, 1.0/temp, L, L/2);


}



void calculate_energy_mag(signed char* lattice_b_h, signed char* lattice_w_h, signed char* lattice_black, signed char* lattice_white, 
                        signed char* lattice_h, const long long L, double& energy, double& mag )
{
    
    cudaMemcpy(lattice_b_h, lattice_black, L*L/2*sizeof(*lattice_b_h), cudaMemcpyDeviceToHost); 
    cudaMemcpy(lattice_w_h, lattice_white, L*L/2*sizeof(*lattice_w_h), cudaMemcpyDeviceToHost);

    mag = 0.0; 
    energy = 0.0; 


    for(int i=0;i<L;i++)
        for(int  j=0;j<L/2;j++)
        {
            if(i%2)
            {
                lattice_h[i*L + 2*j+1] = lattice_b_h[i*L/2 + j]; 
                lattice_h[i*L + 2*j] = lattice_w_h[i*L/2 + j]; 
            }
            else 
            {
                lattice_h[i*L + 2*j] = lattice_b_h[i*L/2 + j]; 
                lattice_h[i*L + 2*j+1] = lattice_w_h[i*L/2 + j]; 
            }

            mag += (int)lattice_h[i*L+2*j]; 
            mag += (int)lattice_h[i*L+2*j+1]; 
        }

    mag /= (double)(L*L); 
    mag = fabs(mag); 

    int pos, up, down, left, right; 
    float nn_sum; 

    for(int i=0;i<L;i++)
        for(int j=0;j<L;j++)
        {
            pos = i*L + j; 
            up = ((i-1+L)%L)*L + j; 
            down = ((i+1)%L)*L + j; 
            left = i*L + (j-1+L)%L; 
            right = i*L + (j+1)&L; 

            nn_sum = lattice_h[up] +  lattice_h[down] + lattice_h[left] + lattice_h[right]; 

            energy += (float)lattice_h[pos] * nn_sum; 
        }

    energy *= -J1; 
    energy /= 2.0; 

    energy /= (double)(L*L); 

}


void save_spin_config(signed char* lattice_h, const long long L, string filename)
{

    ofstream fp; 
    fp.open(filename, ios::out); 

    for(int i=0;i<L;i++)
    {
        for(int j=0;j<L;j++)
            fp<<(int)lattice_h[i*L+j]<<" "; 
        fp<<endl; 
    }

    fp.close(); 
}



int main(int argc, char* argv[])
{

    std::random_device rd; 

    long long L = 1024;
    float temp = 2.0f;
    int MC_sweep = 200000;
    int MC_measure  = 100;  
    unsigned long long seed = (unsigned long long)rd(); 
    bool write_to_file = false; 


    char ch; 
    while((ch = getopt(argc, argv, "l:m:f:s:t:")) != EOF)
    {
        switch(ch)
        {
            case 'l' : L = atoi(optarg);
            break; 
            case 't' : temp = atof(optarg); 
            break; 
            case 's' : MC_sweep = atoi(optarg); 
            break;
            case 'm' : MC_measure = atoi(optarg); 
            break; 
            case 'f' : write_to_file = atoi(optarg); 
            break; 

        }
    }



    printf("\tlattice dimensions: %lld x %lld\n", L, L);
    printf("\tMC sweeps: %d\n", MC_sweep);


    curandGenerator_t gen; 
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10); 
    curandSetPseudoRandomGeneratorSeed(gen, seed); 

    float *randval_d;
    cudaMalloc(&randval_d, L*L/2*sizeof(*randval_d)); 

    signed char *lattice_black, *lattice_white; 

    cudaMalloc(&lattice_black, L*L/2*sizeof(*lattice_black)); 
    cudaMalloc(&lattice_white, L*L/2*sizeof(*lattice_white)); 

    signed char* lattice_w_h, *lattice_b_h, *lattice_h; 

    lattice_h = (signed char*)malloc(L*L*sizeof(*lattice_h) ); 
    lattice_b_h = (signed char*)malloc(L*L/2*sizeof(*lattice_b_h)); 
    lattice_w_h = (signed char*)malloc(L*L/2*sizeof(*lattice_w_h)); 


    int blocks = (L*L/2 + THREAD -1)/THREAD; 

    curandGenerateUniform(gen, randval_d, L*L/2);
    init_spins<<<blocks, THREAD>>>(lattice_black, randval_d, L); 

    curandGenerateUniform(gen, randval_d, L*L/2); 
    init_spins<<<blocks, THREAD>>>(lattice_white, randval_d, L); 

    cudaDeviceSynchronize(); 
    
    string filename, s_temp; 
    ofstream out; 

    if(write_to_file)
        out.open("energy_magnetization_" + to_string(L) + "x" + to_string(L) + "_" + to_string(MC_sweep) + ".txt"); 

    // for(int cnt = 300; cnt >= 150;)
    // {
        // temp = cnt * 0.01f; 
        
        // if(cnt > 250 or cnt <= 200) cnt -= 10; 
        // else cnt -= 5; 
        
        printf("Start equilibration for temperature %.2f\n", temp); 
          
        auto t0 = chrono::high_resolution_clock::now(); 

        //equilibrate the system to target temperature
        seed = (unsigned long long)rd(); 
        curandSetPseudoRandomGeneratorSeed(gen, seed); 


        for(int i=0;i<MC_sweep;i++)
        {
            update(lattice_black, lattice_white, randval_d, gen, temp, L); 

//             if(i%10000 == 0 or i == MC_sweep - 1)
//                 cout<<"Completed "<<i<<" MC sweep"<<endl; 
        }
        
        
        cudaDeviceSynchronize();
        printf("Equilibration for temperature %.2f is done!\n", temp);  


        auto t1 = chrono::high_resolution_clock::now(); 
        auto elapsed  = t1 - t0; 
        double time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count(); 


        printf("Elapsed time for equilibrium: %.2f ms\n", time_ms);
        printf("MC sweep per ms: %f\n", (double) MC_sweep/time_ms );


        double energy = 0.0, mag = 0.0; 
        double avg_ene = 0.0, avg_mag = 0.0, ene_sq = 0.0, mag_sq = 0.0; 

        //collect meaurements for energy and magnetization
        for(int i=1;i<=MC_measure;i++)
        {
            for(int j=0;j<100;j++)
                update(lattice_black, lattice_white, randval_d, gen, temp, L); 

            cudaDeviceSynchronize(); 
            calculate_energy_mag(lattice_b_h, lattice_w_h, lattice_black, lattice_white, lattice_h, L, energy, mag);
            avg_ene += energy; 
            avg_mag += mag; 
            ene_sq += norm(energy); 
            mag_sq += norm(mag);

        }

        avg_ene /= (double)MC_measure; 
        avg_mag /= (double)MC_measure; 
        ene_sq /= (double)MC_measure; 
        mag_sq /= (double)MC_measure; 
    
        printf("Energy: %.6f\n", avg_ene); 
        printf("Magnetization: %.2f\n\n", avg_mag);  
        
        if(write_to_file)
            out<<temp<<" "<<avg_ene<<" "<<ene_sq<<" "<<avg_mag<<" "<<mag_sq<<endl; 
    
        s_temp = to_string(temp); 
        while(s_temp.back() == '0') s_temp.pop_back(); 

        filename = "spin_config_" + to_string(L) + "x" + to_string(L) + "_" + to_string(MC_sweep) + "_temp_" + s_temp + ".txt";  


        if(write_to_file) save_spin_config(lattice_h, L, filename);
        
     
    // }
    

    out.close(); 

	return 0; 
}
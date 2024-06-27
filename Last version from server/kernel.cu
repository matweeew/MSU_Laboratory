#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>
#include <assert.h>
#include <cmath>

#define max_thread 256              //max thread for CUDA
#define vec_size 4                  //max degree of polynomial
#define dim 2*3                     //3 dimensions problem
#define eq_const 32.89868133696453  //2*10^(-3)/(mm*m/(ko))
#define tweezer_width 1.0           //mkm
#define translate_const 13.112635299027149 //in parrots
#define Zr 3.717861128508631        //mkm


__device__ double p(double t, double* vec)
{
    double res = 0;
    for (int i = 0; i < vec_size; i++) res += (vec[i] / ((double)(i + 1))) * pow(t, i + 1);
    return res;
}


__device__ double dp(double t, double* vec)
{
    double res = 0;
    for (int i = 0; i < vec_size; i++) res += vec[i] * pow(t, i);
    return res;
}


__device__ double P(double t, double* vec, double ramp_time)
{
    if (t < 0) return 0;
    if ((t >= 0) && (t < ramp_time)) return p(t, vec);
    else return 1.0;
}


__device__ double to_double(int a, int b)
{
    return ((double)(a) / (double)(b));
}


__device__ double width(double z)
{
    return tweezer_width*sqrt(1+pow(z/Zr,2));
}

__device__ double potential(double x, double y, double z, double Uresult, double U0, double ramp_time, double t, double* vec)
{
    return -(U0 + Uresult*P(t,vec,ramp_time)) * pow(tweezer_width/width(z),2) * exp(-2*(pow(x,2)+pow(y,2))/pow(width(z),2));
}


__device__ void f(double t, double* coord, double ramp_time, double Uresult, double U0, double* F, double* vec)
{
    F[0] = coord[1];
    F[1] = 4 * (coord[0] / pow(width(coord[4]),2)) * eq_const/2 * potential(coord[0],coord[2],coord[4],Uresult,U0,ramp_time,t,vec);
    F[2] = coord[3];
    F[3] = 4 * (coord[2] / pow(width(coord[4]),2)) * eq_const/2 * potential(coord[0],coord[2],coord[4],Uresult,U0,ramp_time,t,vec);
    F[4] = coord[5];
    F[5] = -(2 *coord[4] / pow(Zr,2))* pow(tweezer_width/width(coord[4]),2)*(2*((pow(coord[0],2)+pow(coord[2],2))/pow(width(coord[4]),2))-1)*eq_const/2*potential(coord[0],coord[2],coord[4],Uresult,U0,ramp_time,t,vec);
}


__device__ void increment(double t, double* atom_coordinates, double ramp_time, double Uresult, double U0, double tau, double* vec)
{
    double K[6][dim] = {0};
    double F[dim] = {0};
    double outXY[dim] = {0};

    //K1 calculating
    for (int i = 0; i < dim; i++) outXY[i] = atom_coordinates[i];
    f(t, outXY, ramp_time, Uresult, U0, F, vec);
    K[0][0] = tau * F[0];
    K[0][1] = tau * F[1];
    K[0][2] = tau * F[2];
    K[0][3] = tau * F[3];
    K[0][4] = tau * F[4];
    K[0][5] = tau * F[5];

    //K2 calculating
    for (int i = 0; i < dim; i++) outXY[i] = atom_coordinates[i] + to_double(1, 4) * K[0][i];
    f(t + to_double(1, 4) * tau, outXY, ramp_time, Uresult, U0, F, vec);
    K[1][0] = tau * F[0];
    K[1][1] = tau * F[1];
    K[1][2] = tau * F[2];
    K[1][3] = tau * F[3];
    K[1][4] = tau * F[4];
    K[1][5] = tau * F[5];

    //K3 calculating
    for (int i = 0; i < dim; i++) outXY[i] = atom_coordinates[i] + to_double(3, 32) * K[0][i] + to_double(9, 32) * K[1][i];
    f(t + to_double(3, 8) * tau, outXY, ramp_time, Uresult, U0, F, vec);
    K[2][0] = tau * F[0];
    K[2][1] = tau * F[1];
    K[2][2] = tau * F[2];
    K[2][3] = tau * F[3];
    K[2][4] = tau * F[4];
    K[2][5] = tau * F[5];

    //K4 calculating
    for (int i = 0; i < dim; i++) outXY[i] = atom_coordinates[i] + to_double(1932, 2197) * K[0][i] - to_double(7200, 2197) * K[1][i] + to_double(7296, 2197) * K[2][i];
    f(t + to_double(12, 13) * tau, outXY, ramp_time, Uresult, U0, F, vec);
    K[3][0] = tau * F[0];
    K[3][1] = tau * F[1];
    K[3][2] = tau * F[2];
    K[3][3] = tau * F[3];
    K[3][4] = tau * F[4];
    K[3][5] = tau * F[5];

    //K5 calculating
    for (int i = 0; i < dim; i++) outXY[i] = atom_coordinates[i] + to_double(439, 216) * K[0][i] - ((double)(8)) * K[1][i] + to_double(3680, 513) * K[2][i] - to_double(845, 4104) * K[3][i];
    f(t + tau, outXY, ramp_time, Uresult, U0, F, vec);
    K[4][0] = tau * F[0];
    K[4][1] = tau * F[1];
    K[4][2] = tau * F[2];
    K[4][3] = tau * F[3];
    K[4][4] = tau * F[4];
    K[4][5] = tau * F[5];

    //K6 calculating
    for (int i = 0; i < dim; i++) outXY[i] = atom_coordinates[i] - to_double(8, 27) * K[0][i] + ((double)(2)) * K[1][i] - to_double(3544, 2565) * K[2][i] + to_double(1859, 4104) * K[3][i] - to_double(11, 40) * K[4][i];
    f(t + to_double(1, 2) * tau, outXY, ramp_time, Uresult, U0, F, vec);
    K[5][0] = tau * F[0];
    K[5][1] = tau * F[1];
    K[5][2] = tau * F[2];
    K[5][3] = tau * F[3];
    K[5][4] = tau * F[4];
    K[5][5] = tau * F[5];

    //Result
    for (int i = 0; i < dim; i++) atom_coordinates[i] += to_double(16, 135) * K[0][i] + to_double(6656, 12825) * K[2][i] + to_double(28561, 56430) * K[3][i] - to_double(9, 50) * K[4][i] + to_double(2, 55) * K[5][i];
}


__device__ void rungeKutta(double Uresult, double U0, double ramp_time, double* atom_coordinates, double t0, double tau, double* vec)
{
    while (t0 < ramp_time)
    {
        if (tau >= (ramp_time - t0)) tau = ramp_time - t0;
        increment(t0, atom_coordinates, ramp_time, Uresult, U0, tau, vec);
        t0 += tau;
    }
}


__device__ double energy(double* coord, double Uresult, double U0, double ramp_time, double* vec)
{
    return (pow((coord[1]), 2) + pow(coord[3], 2) + pow(coord[5], 2)) + eq_const * potential(coord[0],coord[2],coord[4],Uresult,U0,ramp_time,ramp_time,vec);
}


__device__ double newton_search(double a, double b, double EPS, double* vec)
{
    double Xnn = 3 * EPS, Xn = b, X = EPS;
    while (fabs(Xnn - X) > EPS) {
        X = Xn;
        Xnn = Xn - ((p(Xn, vec) - 1.0) / dp(Xn, vec));
        Xn = Xnn;
    };
    return Xnn;
}


__device__ double get_tEnd(double Tmax, double tau, double EPS, double* vec)
{
    double ans, t = 0;
    while (t < Tmax)
    {
        if (tau >= (Tmax - t)) tau = Tmax - t;
        if ((1.0 - p(t, vec)) * (1.0 - p(t + tau, vec)) <= 0)
        {
            ans = newton_search(t, t + tau, EPS, vec);
            return ans;
        }
        if (p(t, vec) > p(t + tau, vec)) return -1;
        t += tau;
    }
    return -1;
}


__device__ double get_probability(double Uresult, double U0, double* sample, double ramp_time, double* vec, int sample_size, double* avarade_energy, double* stand_dev_energy, double* max_energy)
{
    double atom_coordinates[dim] = {0}, Ep, t0 = 0, tau = 0.01;
    double successes = 0;
    double* final_energies = new double[sample_size];
    *avarade_energy = 0;
    *stand_dev_energy = 0;
	*max_energy = -(U0+Uresult)*eq_const;
    for (int i = 0; i < sample_size; i++)
    {
        for (int j = 0; j < dim; j++) atom_coordinates[j] = sample[i * dim + j];
        rungeKutta(Uresult, U0, ramp_time, atom_coordinates, t0, tau, vec);
        Ep = energy(atom_coordinates, Uresult, U0, ramp_time, vec);
        if (Ep < 0)
        {
			if(Ep > *max_energy)
			{
				*max_energy = Ep;
			}
            final_energies[i] = Ep;
            *avarade_energy += Ep;
            successes++;
        }
    }
    if (!successes)
    {
        *avarade_energy = 0;
        *stand_dev_energy = 0;
        return 0;
    }
    else
    {
        *avarade_energy = *avarade_energy / successes;
        for (int i = 0; i < sample_size; i++)
        {
            *stand_dev_energy += pow(final_energies[i] - *avarade_energy,2.0);
        }
        *stand_dev_energy = sqrt(*stand_dev_energy/double(sample_size));
    }

    free(final_energies);

    return successes / double(sample_size);
}


__global__ void distributor_probability(double Uresult, double U0, double* data, double* sample, int sample_size, double* output, int cuda_thread_size)
{
    int threadLinearIdx = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("LOL IM HERE ALIVE OR NOT??? %d\n", threadLinearIdx);
    if (threadLinearIdx < cuda_thread_size)
    {
        //Проверка без начальных условий
        //if (threadLinearIdx % 10 == 0) printf("%.1lf mkm|| \t %.1f%% progress\n", s, 100 * float(threadLinearIdx) / float(N));
		//printf("Im alive %d\n", threadLinearIdx);
        double Tmax = 500, tau = 0.01;
        double ramp_time, avarade_energy = 0, stand_dev_energy = 0, probability, max_energy;
        double* vec = new double[vec_size];
        for (int i = 0; i < vec_size; i++) vec[i] = data[threadLinearIdx * vec_size + i];
		//for (int i = 0; i < vec_size; i++) printf("%lf ", vec[i]);
		//printf("it was vector\n");
        ramp_time = get_tEnd(Tmax, tau, 1e-6, vec);

        if (ramp_time != -1)
        {
            probability = get_probability(Uresult, U0, sample, ramp_time, vec, sample_size, &avarade_energy, &stand_dev_energy,&max_energy);
        }
        output[5 * threadLinearIdx + 0] = ramp_time;
        output[5 * threadLinearIdx + 1] = probability;
        output[5 * threadLinearIdx + 2] = avarade_energy;
        output[5 * threadLinearIdx + 3] = stand_dev_energy;
		output[5 * threadLinearIdx + 4] = max_energy;
        free(vec);
    }
}


int main()
{
    double Uresult = 5.0, U0 = 1.2;
    int allowed_vectors = 0;

    // create result file
    FILE* file;
    file = fopen("results.txt", "w");
    if (file == NULL)
    {
        printf("results.txt did't open\n");
        exit(2);
    }
    fclose(file);


    // loading random data from file
    printf("Start reading random data\n");
    file = fopen("rand.txt", "r");
    if (file == NULL)
    {
        printf("rand.txt did't open\n");
        exit(2);
    }
    int population, data_size;
    fscanf(file, "%d", &population);
    data_size = population * vec_size;
    double* data = (double*)malloc(data_size * sizeof(double));
    for (int i = 0; i < data_size; i++) fscanf(file, "%lf", &data[i]);
    fclose(file);
    printf("Data reading have done\n\n");

    // loading random initial conditions from file
    printf("Start reading initial conditions\n");
    file = fopen("conditions.txt", "r");
    if (file == NULL)
    {
        printf("conditions.txt did't open\n");
        exit(2);
    }
    int init_conditions_cnt, init_conditions_size;
    fscanf(file, "%d", &init_conditions_cnt);
    init_conditions_size = dim * init_conditions_cnt * sizeof(double);
    double* data_init = (double*)malloc(init_conditions_size);
    for (int i = 0; i < dim * init_conditions_cnt; i++) fscanf(file, "%lf", &data_init[i]);
    fclose(file);
    printf("Reading have done\n\n");

    //CUDA PART
    // initializing a size constants
    int send_size = vec_size * max_thread * sizeof(double);	 // size of the sent array in bytes
    int recive_size = 5 * max_thread * sizeof(double);		 // size of the received array in bytes


    //algorithm part for calculations
    int data_array_pointer = 0, cuda_array_pointer = 0;
	while (data_array_pointer < population)
	{


		// initialization of sent arrays (on host)
        double* inputDataOnHost = (double*)malloc(send_size); // input vectors array (on host)
        double* inputInitialDataOnHost = data_init;            // input initial conditions array (on host)

        // initialization of received arrays of arrays (on host)
        double* resultOnHost = (double*)malloc(recive_size);    // calculation result (received on host)

        // allocating memory on the device and binding pointers for arrays
        double* inputDataOnDevice = NULL;                       // input vectors array pointer (on device)
        double* inputInitialDataOnDevice = NULL;                // input initial conditions array pointer (on device)
        double* resultOnDevice = NULL;                          // result data array pointer (on device)
        cudaMalloc((void**)&inputDataOnDevice, send_size);                  // allocating memory
        cudaMalloc((void**)&inputInitialDataOnDevice, init_conditions_size);// on the device
        cudaMalloc((void**)&resultOnDevice, recive_size);                   // and binding pointers


		int era = 0; //for correct info input
        while(cuda_array_pointer < max_thread)
		{
		    for(int i = 0; i < vec_size; i++)
            {
                inputDataOnHost[vec_size*cuda_array_pointer + i] = data[vec_size*data_array_pointer + i];
            }
            cuda_array_pointer++;
            data_array_pointer++;
			if (data_array_pointer>=population) break;
		}

        // sending initial conditions data to the device
        cudaMemcpy(inputInitialDataOnDevice, inputInitialDataOnHost, init_conditions_size, cudaMemcpyHostToDevice);
		cudaMemcpy(inputDataOnDevice, inputDataOnHost, send_size, cudaMemcpyHostToDevice);

		distributor_probability << <256, 1 >> > (Uresult, U0, inputDataOnDevice, inputInitialDataOnDevice, init_conditions_cnt, resultOnDevice, cuda_array_pointer);

        // recive result
		cudaMemcpy(resultOnHost, resultOnDevice, recive_size, cudaMemcpyDeviceToHost);

		// free memory on device
		cudaFree(inputInitialDataOnDevice);
		cudaFree(inputDataOnDevice);
        cudaFree(resultOnDevice);
        //cudaMalloc((void**)&inputInitialDataOnDevice, init_conditions_size);
        //cudaMalloc((void**)&inputDataOnDevice, send_size);
        //cudaMalloc((void**)&resultOnDevice, recive_size);

        free(inputDataOnHost);
        free(resultOnHost);

        // information output in file and console
		file = fopen("results.txt", "a");
        if (file == NULL)
        {
            printf("results.txt did't open\n");
            exit(2);
        }
        for (int i = 0; i < cuda_array_pointer; i++)
        {
            if (resultOnHost[5 * i] != -1)
            {
                printf("%d \t time = %lf \t prob = %lf \t av_en = %lf \t stand_dev_en = %lf \t max_en = %lf\n",i,resultOnHost[5*i],resultOnHost[5*i+1],resultOnHost[5*i+2],resultOnHost[5*i+3],resultOnHost[5*i+4]);
                fprintf(file, "%lf %lf %lf %lf %lf ", resultOnHost[5 * i], resultOnHost[5 * i + 1], resultOnHost[5 * i + 2], resultOnHost[5 * i + 3], resultOnHost[5 * i + 4]);
                for (int j = 0; j < vec_size; j++) fprintf(file, "%lf ", data[vec_size * i + max_thread * era+ j]);//inputDataOnHost[i * vec_size + j]);
                fprintf(file, "\n ");
                allowed_vectors++;
            }
        }
        fclose(file);

        printf("\n %d \ %d done\n", data_array_pointer, population);
        printf("\nYou have %d allowed polynomials\n", allowed_vectors);

        cuda_array_pointer = 0;
	}

    //Освобождение памяти
    //cudaFree(inputDataOnDevice);
    //cudaFree(inputInitialDataOnDevice);
    //cudaFree(resultOnDevice);

    //free(inputDataOnHost);
    //free(resultOnHost);

    return 0;
}

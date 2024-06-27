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
#include <iostream>
#include <random>


#define dimension 6                             //размерность фазового пространства атома
#define vec 4                                   //полином какой степени ищется//НЕ ЗАБУДЬ ГЕНЕРАТОРЫ ПОПРАВИТЬ ТОГДА
#define grid_step 0.1                           //шаг сетки в пространстве полиномов
#define number_of_eras 10                       //кол-во циклов (шагов) по сетке от начальной точки
constexpr auto population = 10000;              //кол-во генерируемых/проверяемых точек
constexpr auto size_target = population * 100;  //забыл что это но работает так норм

#define w0 1.5                      //ширина перетяжки
#define Zr 8.31598055362004         //ширина по оси Z
#define CONST_N 0.09566840431136607 //размерная константа равная 10^(-3)*k/(m*mm) (используется в ураынении Ньютона)
#define U 4.0                       //глубина пинцета (нормированная на постоянную Больцмана, задается в мК)
#define CONST_E 5.22638590660173    //размерная константа равная  m*mm/(2*10^(-3)*k) (используется для вычисления энергии в мК)

#define tau 0.01            //шаг в Рунге-Кутта при решении уравнения Ньютона
#define EPS 1e-06           //точность в мкс, с которой определяется tEnd
#define lim 0.5			    //процент от начальной глубины, для предварительной фильтрации.
#define CONST_k 1.53846     //это две константы для более точного задания максимального времени Tmax перемещения атома,
#define CONST_b 9.53846		//т.к. время перемещения зависит от расстояния перемещения s (просто глянь где это используются)

#define Sinit 5.2			//поиск оптимального профиля начнется для расстояния перемещения Sinit
#define Smax 3.3			//максимальное расстояние перемещения, для которого будет произведенга оптимизация
#define dS 0.1				//шаг по S

#define init_point_1 0.635516//0.64//0.397023//0.64;//0.936040//0.64
#define init_point_2 0.288374//0.0//0.896440//0.0//0.0;//-0.011395
#define init_point_3 -0.253893//0.0//-0.459227//0.0//0.0;//-0.125110
#define init_point_4 0.040322//0.0//0.055564//0.0//0.0;//0.022031
#define init_point_5 0.0
#define init_point_6 0.0


__device__ double p(double t, double* vector, int n)
{
    double res = 0;
    for (int i = 0; i < n; i++) res += (vector[i] / ((double)(i + 1))) * pow(t, i + 1);
    return res;
}


__device__ double dp(double t, double* vector, int n)
{
    double res = 0;
    for (int i = 0; i < n; i++) res += vector[i] * pow(t, i);
    return res;
}


__device__ double P(double t, double* vector, int n, double tEnd, double s)
{
    if (t < tEnd / 2.0)   return s / 2.0 - p(tEnd / 2.0 - t, vector, n);
    else                  return s / 2.0 + p(t - tEnd / 2.0, vector, n);
}


__device__ double to_double(int a, int b)
{
    return ((double)(a) / (double)(b));
}


__device__ double w(double z)
{
    return w0*sqrt(1+pow(z/Zr, 2));
}


__device__ double exp_func(double x, double y, double z)
{
    return exp( -2*(pow(x,2)+pow(y,2))/(pow(w(z),2)));
}


__device__ double dU_dx(double x, double y, double z)
{
    return 4*(pow(w0,2)/pow(w(z),4))*x*exp_func(x,y,z);
}


__device__ double dU_dy(double x, double y, double z)
{
    return 4*(pow(w0,2)/pow(w(z),4))*y*exp_func(x,y,z);
}


__device__ double dU_dz(double x, double y, double z)
{
    return 2*pow((w0/w(z)),4)*(z/pow(Zr,2))*(1-2*((pow(x,2)+pow(y,2))/pow(w(z),2)))*exp_func(x,y,z);
}


__device__ void f(double t, double* XY, double tEnd, double s, double* F, double* vector, int n)
{
    F[0] = XY[1];
    F[1] = -CONST_N*U*dU_dx(XY[0]-P(t,vector,n,tEnd,s),XY[2],XY[4]);
    F[2] = XY[3];
    F[3] = -CONST_N*U*dU_dy(XY[0]-P(t,vector,n,tEnd,s),XY[2],XY[4]);
    F[4] = XY[5];
    F[5] = -CONST_N*U*dU_dz(XY[0]-P(t,vector,n,tEnd,s),XY[2],XY[4]);

}


__device__ void increment(double t, double* XY, double tEnd, double s, double step, double* vector, int n)
{
    double K[6][dimension] = { 0 };
    double F[dimension] = { 0 };
    double outXY[dimension] = { 0 };

    //K1 calculating
    for (int i = 0; i < dimension; i++) outXY[i] = XY[i];
    f(t, outXY, tEnd, s, F, vector, n);
    for(int i = 0; i < dimension; i++) K[0][i] = step * F[i];

    //K2 calculating
    for (int i = 0; i < dimension; i++) outXY[i] = XY[i] + to_double(1, 4) * K[0][i];
    f(t + to_double(1, 4) * step, outXY, tEnd, s, F, vector, n);
    for(int i = 0; i < dimension; i++) K[1][i] = step * F[i];

    //K3 calculating
    for (int i = 0; i < dimension; i++) outXY[i] = XY[i] + to_double(3, 32) * K[0][i] + to_double(9, 32) * K[1][i];
    f(t + to_double(3, 8) * step, outXY, tEnd, s, F, vector, n);
    for(int i = 0; i < dimension; i++) K[2][i] = step * F[i];

    //K4 calculating
    for (int i = 0; i < dimension; i++) outXY[i] = XY[i] + to_double(1932, 2197) * K[0][i] - to_double(7200, 2197) * K[1][i] + to_double(7296, 2197) * K[2][i];
    f(t + to_double(12, 13) * step, outXY, tEnd, s, F, vector, n);
    for(int i = 0; i < dimension; i++) K[3][i] = step * F[i];;

    //K5 calculating
    for (int i = 0; i < dimension; i++) outXY[i] = XY[i] + to_double(439, 216) * K[0][i] - ((double)(8)) * K[1][i] + to_double(3680, 513) * K[2][i] - to_double(845, 4104) * K[3][i];
    f(t + step, outXY, tEnd, s, F, vector, n);
    for(int i = 0; i < dimension; i++) K[4][i] = step * F[i];

    //K6 calculating
    for (int i = 0; i < dimension; i++) outXY[i] = XY[i] - to_double(8, 27) * K[0][i] + ((double)(2)) * K[1][i] - to_double(3544, 2565) * K[2][i] + to_double(1859, 4104) * K[3][i] - to_double(11, 40) * K[4][i];
    f(t + to_double(1, 2) * step, outXY, tEnd, s, F, vector, n);
    for(int i = 0; i < dimension; i++) K[5][i] = step * F[i];

    //Result
    for (int i = 0; i < dimension; i++) XY[i] += to_double(16, 135) * K[0][i] + to_double(6656, 12825) * K[2][i] + to_double(28561, 56430) * K[3][i] - to_double(9, 50) * K[4][i] + to_double(2, 55) * K[5][i];
}


__device__ void rungeKutta(double t0, double* XY, double tEnd, double s, double* vector, int n)
{
    while (t0 < tEnd)
    {
        double step = tau;
        if (step >= (tEnd - t0)) step = tEnd - t0;
        increment(t0, XY, tEnd, s, step, vector, n);
        t0 += step;
    }
}


__device__ double energy(double* XY, double tEnd, double s, double* vector, int n)
{
    return (pow((XY[1]),2) + pow(XY[3],2) + pow(XY[5],2))* CONST_E - U * pow((w0/w(XY[4])),2)*exp_func(XY[0]-P(tEnd, vector, n, tEnd, s),XY[2],XY[4]);
}


__device__ double newton_search(double a, double b, double s, double* vector, int n)
{
    double Xnn = 3 * EPS, Xn = b, X = EPS;
    while (fabs(Xnn - X) > EPS) {
        X = Xn;
        Xnn = Xn - ((p(Xn, vector, n) - s) / dp(Xn, vector, n));
        Xn = Xnn;
    };
    return Xnn;
}


__device__ double get_tEnd(double s, double Tmax, double* vector, int n)
{
    double ans, t = 0, step = tau;
    while (t < Tmax)
    {
        if (step >= (Tmax - t)) step = Tmax - t;
        if ((p(t, vector, n) - s / 2.0) * (p(t + step, vector, n) - s / 2.0) <= 0)
        {
            ans = newton_search(t, t + step, s / 2.0, vector, n);
            return 2 * ans;
        }
        if (p(t, vector, n) > p(t + step, vector, n)) return -1;
        t += step;
    }
    return -1;
}


__device__ double optimization_f(double s, double* E, double* vector, int n)
{
    double tEnd = 0, Tmax = CONST_k * s + CONST_b;
    tEnd = get_tEnd(s, Tmax, vector, n);
    if (tEnd != -1)
    {
        double XY[dimension] = { 0 }, t0 = 0;
        rungeKutta(t0, XY, tEnd, s, vector, n);
        *E = energy(XY, tEnd, s, vector, n);
        //printf("XY = %lf,%lf,%lf,%lf,%lf,%lf\tS = %lf \t Tmax = %lf\t  tEnd = %lf \t Energy = %lf\t vector = %lf, %lf, %lf, %lf, %lf\n",XY[0],XY[1],XY[2],XY[3],XY[4],XY[5],s,Tmax, tEnd, *E, vector[0], vector[1], vector[2], vector[3], vector[4]);
        if (*E < (-lim * U)) return tEnd;
    }
    return -1;
}


__global__ void distributor(double s, double* array, double* output, int n, int N)
{
    int threadLinearIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadLinearIdx < N)
    {
        //if (threadLinearIdx % 10000 == 0) printf("%.1f%% done\n", 100 * float(threadLinearIdx) / float(N));
        double tEnd, E;
        double* vector = (double*)malloc(n * sizeof(double));
        for (int i = 0; i < n; i++) vector[i] = array[threadLinearIdx * n + i];
        tEnd = optimization_f(s, &E, vector, n);
        output[2 * threadLinearIdx] = tEnd;
        output[2 * threadLinearIdx + 1] = E;
        free(vector);
    }
}


__device__ double get_P(double s, double* initials, int init_N, double tEnd, double* avarage_E, double* vector, int n)
{
    double XY[dimension] = { 0 }, Ep, t0 = 0;
    double successes = 0;
    *avarage_E = 0;
    for (int i = 0; i < init_N; i++)
    {
        for (int j = 0; j < dimension; j++) XY[j] = initials[i * dimension + j];
        rungeKutta(t0, XY, tEnd, s, vector, n);
        Ep = energy(XY, tEnd, s, vector, n);
        if (Ep < 0)
        {
            *avarage_E += Ep;
            successes++;
        }
    }
    if (!successes) *avarage_E = 0;
    else *avarage_E = *avarage_E / successes;
    return successes / double(init_N);
}


__global__ void distributor_P(double s, double* array, double* initials, double* output, int init_N, int n, int N)
{
    int threadLinearIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadLinearIdx < N)
    {
        //Проверка без начальных условий
        if (threadLinearIdx % 10 == 0) printf("%.1lf mkm|| \t %.1f%% progress\n", s, 100 * float(threadLinearIdx) / float(N));
        double tEnd, E, P = 0, avarage_E = 0;
        double* vector = (double*)malloc(n * sizeof(double));
        for (int i = 0; i < n; i++) vector[i] = array[threadLinearIdx * n + i];
        tEnd = optimization_f(s, &E, vector, n);


        if (tEnd != -1) P = get_P(s, initials, init_N, tEnd, &avarage_E, vector, n);

        output[4 * threadLinearIdx + 0] = tEnd;
        output[4 * threadLinearIdx + 1] = E;
        output[4 * threadLinearIdx + 2] = P;
        output[4 * threadLinearIdx + 3] = avarage_E;
        free(vector);
    }
}


double get_nearest_coordinate(double coordinate, double grid)
{
    return round(coordinate / grid) * grid;
}


double randnum_normal(double mu, double sigma)
{
    static std::default_random_engine generator;
    std::normal_distribution<double> distribution(mu, sigma);
    return distribution(generator);
}


int find(double** array, int N, double* vector, int n)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (array[i][j] != vector[j]) break;
            if ((j == (n - 1)) && (array[i][n - 1] == vector[n - 1])) return 1;
        }
    }
    return 0;
}


void calculate_p_e(double S, int alive_cnt)
{
    //ифнормация
    printf("============================================\nStart calculating P and <E>\n");

    //Загрузка случайных значений (полиномов) из файла
    printf("Start reading random population\n");
    FILE* file;
    file = fopen("data.txt", "r");
    if (file == NULL)
    {
        printf("data.txt did't open\n");
        exit(2);
    }
    int size;
    size = alive_cnt * vec;
    double* data = (double*)malloc(size * sizeof(double));
    for (int i = 0; i < size; i++) fscanf(file, "%lf", &data[i]);
    fclose(file);
    printf("Reading have done\n\n");

    //Загрузка случайных начальных условий из файла
    printf("Start reading initial conditions\n");
    file = fopen("conditions.txt", "r");
    if (file == NULL)
    {
        printf("conditions.txt did't open\n");
        exit(2);
    }
    int init_population;
    fscanf(file, "%d", &init_population);
    int Np = 6 * init_population * sizeof(double);
    double* data_init = (double*)malloc(Np);
    for (int i = 0; i < 6 * init_population; i++) fscanf(file, "%lf", &data_init[i]);
    fclose(file);
    printf("Reading have done\n");
    printf("============================================\n");

    //Часть CUDA
    int n = vec;       // степень полинома
    int N = alive_cnt;   // количество векторов
    int nb = size * sizeof(double);      // размер входного полиномов массива в байтах
    int Nb = 4 * N * sizeof(double);     // размер выходного массива полиномов в байтах

    // входные данные полиномов на хост
    double* inputDataOnHost = data;

    // входные начальные условия на хост
    double* inputInitialDataOnHost = data_init;

    //результат на хосте
    double* resultOnHost = (double*)malloc(Nb);

    //данные входные / выходные на девайсе
    double* inputDataOnDevice = NULL, * inputInitialDataOnDevice = NULL, * resultOnDevice = NULL;
    cudaMalloc((void**)&inputDataOnDevice, nb);
    cudaMalloc((void**)&inputInitialDataOnDevice, Np);
    cudaMalloc((void**)&resultOnDevice, Nb);

    //копирование данных на GPU и привязка указателю там к inputDataOnDevice
    cudaMemcpy(inputDataOnDevice, inputDataOnHost, nb, cudaMemcpyHostToDevice);
    cudaMemcpy(inputInitialDataOnDevice, inputInitialDataOnHost, Np, cudaMemcpyHostToDevice);

    //запуск ядра
    distributor_P << <4096, 256 >> > (S, inputDataOnDevice, inputInitialDataOnDevice, resultOnDevice, init_population, n, N);

    //копирование результатов на хост и привязка к указателю тут resultOnHost
    cudaMemcpy(resultOnHost, resultOnDevice, Nb, cudaMemcpyDeviceToHost);

    //освобождение памяти
    cudaFree(inputDataOnDevice);
    cudaFree(resultOnDevice);


    //вывод информации
    printf("===================results===================\n");
    for (int i = 0; i < N; i++)
    {
        if (resultOnHost[4 * i] != -1)
        {
            printf("%i\ttEnd = \t%.10f\tE = %.10f\tP = %lf\t<E> = %lf\n", i, resultOnHost[4 * i], resultOnHost[4 * i + 1], resultOnHost[4 * i + 2], resultOnHost[4 * i + 3]);
        }
    }
    printf("\n");


    //Запись результатов в файл
    printf("Start writing on file\n");
    file = fopen("data.txt", "w");
    if (file == NULL)
    {
        printf("data.txt did't open\n");
        exit(2);
    }
    int coun = 0;
    for (int i = 0; i < alive_cnt; i++)
    {
        if (resultOnHost[4 * i] != -1)
        {
            fprintf(file, "%lf %lf %lf %lf ", resultOnHost[4 * i], resultOnHost[4 * i + 1], resultOnHost[4 * i + 2], resultOnHost[4 * i + 3]);
            for (int j = 0; j < vec; j++) fprintf(file, "%lf ", inputDataOnHost[i * vec + j]);
            fprintf(file, "\n ");
            coun++;
        }
    }
    fclose(file);
    printf("Writing have done\n");
    printf("\nYou have %d allowed polynomials\n", coun);
    printf("================calculating done==================\n");

    //Освобождение памяти
    free(inputDataOnHost);
    free(resultOnHost);
}


int grid_method(double S, int era_cnt, double* init_vector)
{
    //инициализация файла для точек
    FILE* file;
    file = fopen("data.txt", "w");
    fclose(file);

    //часть CUDA
    const int nb = vec * population * sizeof(double); // размер входного массива в байтах
    const int Nb = 2 * population * sizeof(double);   // размер выходного массива в байтах

    //выходной массив на хосте
    double* inputDataOnHost = (double*)malloc(vec * population * sizeof(double));

    //результат на хосте
    double* resultOnHost = (double*)malloc(Nb);

    //данные входные / выходные на девайсе
    double* inputDataOnDevice = NULL, * resultOnDevice = NULL;
    cudaMalloc((void**)&inputDataOnDevice, nb);
    cudaMalloc((void**)&resultOnDevice, Nb);

    //часть сеточного метода
    //инициализация шага сетки
    double grid[vec];
    for (int i = 0; i < vec; i++) grid[i] = grid_step;

    //количесво живых точек вокруг одной точки сетки
    int alive_points = 0;
    //суммарное количество накопленных точек
    int alive_cnt = 0;

    //далее везде "указатели" cnt показывают текущее КОЛ-ВО элементов в массиве
    //точки сетки, которые уже были проверены
    double** checked_grid;
    int size = 1000 * population; //текущий размер массива на случай если он начнет переполняться               //ОНО ТАМ ВНИЗУ НЕ БУДЕТ РАБОТАТЬ ПЗДЦ
    checked_grid = (double**)malloc(size * sizeof(double**));
    for (int i = 0; i < size; i++) checked_grid[i] = (double*)malloc(vec * sizeof(double*));
    int checked_grid_cnt = 1;


    //точки для проверки в текущем цикле (массив заполняется в предыдущем цикле)
    double** target_grid = (double**)malloc(size_target * sizeof(double**));
    for (int i = 0; i < size_target; i++) target_grid[i] = (double*)malloc(vec * sizeof(double*));
    int target_grid_cnt = 1;


    //точки для проверки в следующем цикле (массив заполняется в текущем цикле)
    double** next_target_grid = (double**)malloc(size_target * sizeof(double**));
    for (int i = 0; i < size_target; i++) next_target_grid[i] = (double*)malloc(vec * sizeof(double*));
    int next_target_grid_cnt = 0;


    //вспомогательный массив для хранения ближайшей точки сетки для данного вектора
    double near_point[vec] = { 0 };

    //инициализация начального вектора
    //и одновременно добавляем его в массив проверенных
    for (int i = 0; i < vec; i++)
    {
        target_grid[0][i] = init_vector[i];
        checked_grid[0][i] = init_vector[i];
    }

    //цикл поиска точек фигуры на протяжении era_cnt
    for (int era = 0; era < era_cnt; era++)
    {
		if(!target_grid_cnt) break;
        //ифнормация
        printf("============================================\n\t\tEra %d\n", era + 1);
        printf("============================================\n");
        //цикл, пробегающий по точкам сетки, которые нужно проверить
        for (int q = 0; q < target_grid_cnt; q++)
        {
            //запись "случайной сферы" около точки target_grid[i]
            //цикл пробегает по всей популяции
            for (int j = 0; j < population; j++)
            {
                //цикл пробегает по одному вектору из популяции
                for (int k = 0; k < vec; k++)
                {
                    inputDataOnHost[j * vec + k] = randnum_normal(target_grid[q][k], grid[k]);
                }
            }

            //отправляем записанную "случайную сферу" на проверку на девайс
            //копирование данных на GPU и привязка указателю там к inputDataOnDevice
            cudaMemcpy(inputDataOnDevice, inputDataOnHost, nb, cudaMemcpyHostToDevice);

            //запуск ядра
            distributor << <4096, 256 >> > (S, inputDataOnDevice, resultOnDevice, vec, population);

            //копирование результатов на хост и привязка к указателю тут resultOnHost
            cudaMemcpy(resultOnHost, resultOnDevice, Nb, cudaMemcpyDeviceToHost);
            //ядро выполнило работу


            //смотрим результаты проверки
            for (int j = 0; j < population; j++) if (resultOnHost[2 * j] != -1) alive_points++;
            printf("Era %d / %d \t Grid-point %d / %d\t%d alive points around\n", era + 1, era_cnt, q, target_grid_cnt - 1, alive_points);
            alive_points = 0;


            //открываем файл для записи живых точек
            file = fopen("data.txt", "a");
            if (file == NULL)
            {
                printf("data.txt did't open\n");
                exit(2);
            }


            //в цикле ищем все живые точки: записываем их
            //в файл; вычисляем ближайшую точку сетки;
            //смотрим проверяли ли эту точку ранее; если
            //точка не проверяялась, то добовляем в очередь
            //на проверку
            for (int i = 0; i < population; i++)
            {
                if (resultOnHost[2 * i] != -1)
                {
                    //записываем живую точку в файл
                    fprintf(file, "%lf %lf ", resultOnHost[2 * i], resultOnHost[2 * i + 1]);
                    for (int j = 0; j < vec; j++) fprintf(file, "%lf ", inputDataOnHost[i * vec + j]);
                    fprintf(file, "\n");
                    alive_cnt++;

                    //находим ближайшую точку сетки для данного вектора
                    for (int j = 0; j < vec; j++) near_point[j] = get_nearest_coordinate(inputDataOnHost[i * vec + j], grid[j]);

                    //смотрим проверялась ли эта точка ранее
                    if (!find(checked_grid, checked_grid_cnt, near_point, vec))
                    {
                        //если точка не проверялась, то добавляем её
                        //в список на проверку для слудующего цикла,
                        //а также добавляем её в массив проверенных точек
                        for (int j = 0; j < vec; j++)
                        {
                            //список на следующую проверку
                            next_target_grid[next_target_grid_cnt][j] = near_point[j];

                            //массив провернных точек
                            checked_grid[checked_grid_cnt][j] = near_point[j];
                        }
                        next_target_grid_cnt++;
                        checked_grid_cnt++;
                        //printf("next_target_grid_cnt = %d\n", next_target_grid_cnt);
                    }

                    //увеличиваем массив проверенных точек на population,
                    //если он начинает переполняться
                    if (size - 100 < checked_grid_cnt) ///ЭТО НЕ РАБОТАЕТ
                    {
                        printf("\n\n\n%d\t%d\n\n\n", size - 100, checked_grid_cnt);
                        size += population;
                        checked_grid = (double**)realloc(checked_grid, size * sizeof(*checked_grid));
                    }
                }
            }
            fclose(file);
        }
        printf("========================results====================\n");
        //заполняем следующий массив для проверки и сдвигаем "указатели"
        target_grid_cnt = next_target_grid_cnt;
        next_target_grid_cnt = 0;
        for (int i = 0; i < target_grid_cnt; i++)
        {
            printf("Grid number %d \t vector:\t", i);
            for (int j = 0; j < vec; j++)
            {
                target_grid[i][j] = next_target_grid[i][j];
                printf("%lf\t", target_grid[i][j]);
            }
            printf("\n");

        }
        printf("============================================\n");
        printf("Number of nodes to check %d\n", target_grid_cnt);
        printf("Total live points  %d\n", alive_cnt);
        printf("============================================\n\n\n\n");
    }

    //освобождение памяти на девайсе
    cudaFree(inputDataOnDevice);
    cudaFree(resultOnDevice);

    //освобождение памяти на устройстве
    free(checked_grid);
    free(target_grid);
    free(next_target_grid);
    free(inputDataOnHost);
    free(resultOnHost);

    return alive_cnt;
}


void get_best_point(int alive_cnt, double* best_vector)
{
    //создаем массив для хранения всей информации о веекторах
    double** data;
    data = (double**)malloc(alive_cnt * sizeof(double**));
    for (int i = 0; i < alive_cnt; i++) data[i] = (double*)malloc((4 + vec) * sizeof(double*));

    //читаем данные из файла
    FILE* file;
    file = fopen("data.txt", "r");
    if (file == NULL)
    {
        printf("data.txt did't open\n");
        exit(2);
    }
    for (int i = 0; i < alive_cnt; i++)
    {
        for (int j = 0; j < 4 + vec; j++)
        {
            fscanf(file, "%lf", &data[i][j]);
        }
    }
    fclose(file);

    //фильтруем данные и отбираем лучшие точки

    //ищем минимум по <E> среди P = 1
    double E_min = 0;
    for (int i = 0; i < alive_cnt; i++)
    {
        if ((data[i][2] == 1.0) && (data[i][3] < E_min)) E_min = data[i][3];
    }

    //ищем среди P = 1 и менее 0.98<Emin> минимум по времени
    double T_min = 10;
    int number = 0;///////DELETE THAT SHIT
    for (int i = 0; i < alive_cnt; i++)
    {
        if ((data[i][2] == 1.0) && (data[i][3] < 0.98 * E_min) && (data[i][1] < T_min))
        {
            T_min = data[i][1];
            number = i;
        }
    }

    //записываем лучшую точку
    for (int i = 0; i < vec + 4; i++) best_vector[i] = data[number][i];

    //вывожу результат
    printf("Best vector:\n");
    for (int i = 0; i < vec + 4; i++) printf("%lf\t", data[number][i]);
    printf("============================================\n\n\n\n");
    number = 0;
}


int main()
{
    double init_vector[vec];
	int alive_cnt;
    double bestie[vec + 4];
    double S=Sinit;

    FILE* file;
    file = fopen("results.txt", "w");
    fclose(file);

    init_vector[0] = init_point_1;
    init_vector[1] = init_point_2;
    init_vector[2] = init_point_3;
    init_vector[3] = init_point_4;
    init_vector[4] = init_point_5;
    init_vector[5] = init_point_6;

    while (S > Smax)
    {
		printf("S = %lf\n", S);

        //Для S считаем лучший профиль
        alive_cnt = grid_method(S, number_of_eras, init_vector);
        calculate_p_e(S, alive_cnt);
        get_best_point(alive_cnt, bestie);

        //Записываем лучший профиль в файл
        file = fopen("results.txt", "a");
        if (file == NULL)
        {
            printf("results.txt did't open\n");
            exit(2);
        }
        fprintf(file, "%lf\t", S);
        for (int i = 0; i < vec + 4; i++) fprintf(file, "%lf ", bestie[i]);
        fprintf(file, "\n ");
        fclose(file);

        //Начальный вектор - лучший предыдущий
        for (int i = 0; i < vec; i++) init_vector[i] = bestie[i + 4];


        //Новое расстояние
        S = S - dS;
		if(S == Smax) break;
    }
    return 0;
}

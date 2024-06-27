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


#define vec 4                   //НЕ ЗАБУДЬ ГЕНЕРАТОРЫ ПОПРАВИТЬ ТОГДА
#define grid_step 5
constexpr auto population = 1000;
constexpr auto size_target = population * 100;


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


__device__ void f(double t, double* XY, double tEnd, double s, double* F, double* vector, int n, double C, double W)
{
    F[0] = XY[1];
    F[1] = -C * (XY[0] - P(t, vector, n, tEnd, s)) * exp(-(pow((XY[0] - P(t, vector, n, tEnd, s)), 2) + pow(XY[2], 2)) / (pow(W, 2)));
    F[2] = XY[3];
    F[3] = -C * XY[2] * exp(-(pow((XY[0] - P(t, vector, n, tEnd, s)), 2) + pow(XY[2], 2)) / (pow(W, 2)));
}


__device__ void increment(double t, double* XY, double tEnd, double s, double tau, double* vector, int n)
{
    double K[6][4] = { 0 };
    double F[4] = { 0 };
    double outXY[4] = { 0 };
    double C = 39.47841760435742, W = 1;

    //K1 calculating
    for (int i = 0; i < 4; i++) outXY[i] = XY[i];
    f(t, outXY, tEnd, s, F, vector, n, C, W);
    K[0][0] = tau * F[0];
    K[0][1] = tau * F[1];
    K[0][2] = tau * F[2];
    K[0][3] = tau * F[3];

    //K2 calculating
    for (int i = 0; i < 4; i++) outXY[i] = XY[i] + to_double(1, 4) * K[0][i];
    f(t + to_double(1, 4) * tau, outXY, tEnd, s, F, vector, n, C, W);
    K[1][0] = tau * F[0];
    K[1][1] = tau * F[1];
    K[1][2] = tau * F[2];
    K[1][3] = tau * F[3];

    //K3 calculating
    for (int i = 0; i < 4; i++) outXY[i] = XY[i] + to_double(3, 32) * K[0][i] + to_double(9, 32) * K[1][i];
    f(t + to_double(3, 8) * tau, outXY, tEnd, s, F, vector, n, C, W);
    K[2][0] = tau * F[0];
    K[2][1] = tau * F[1];
    K[2][2] = tau * F[2];
    K[2][3] = tau * F[3];

    //K4 calculating
    for (int i = 0; i < 4; i++) outXY[i] = XY[i] + to_double(1932, 2197) * K[0][i] - to_double(7200, 2197) * K[1][i] + to_double(7296, 2197) * K[2][i];
    f(t + to_double(12, 13) * tau, outXY, tEnd, s, F, vector, n, C, W);
    K[3][0] = tau * F[0];
    K[3][1] = tau * F[1];
    K[3][2] = tau * F[2];
    K[3][3] = tau * F[3];

    //K5 calculating
    for (int i = 0; i < 4; i++) outXY[i] = XY[i] + to_double(439, 216) * K[0][i] - ((double)(8)) * K[1][i] + to_double(3680, 513) * K[2][i] - to_double(845, 4104) * K[3][i];
    f(t + tau, outXY, tEnd, s, F, vector, n, C, W);
    K[4][0] = tau * F[0];
    K[4][1] = tau * F[1];
    K[4][2] = tau * F[2];
    K[4][3] = tau * F[3];

    //K6 calculating
    for (int i = 0; i < 4; i++) outXY[i] = XY[i] - to_double(8, 27) * K[0][i] + ((double)(2)) * K[1][i] - to_double(3544, 2565) * K[2][i] + to_double(1859, 4104) * K[3][i] - to_double(11, 40) * K[4][i];
    f(t + to_double(1, 2) * tau, outXY, tEnd, s, F, vector, n, C, W);
    K[5][0] = tau * F[0];
    K[5][1] = tau * F[1];
    K[5][2] = tau * F[2];
    K[5][3] = tau * F[3];

    //Result
    for (int i = 0; i < 4; i++) XY[i] += to_double(16, 135) * K[0][i] + to_double(6656, 12825) * K[2][i] + to_double(28561, 56430) * K[3][i] - to_double(9, 50) * K[4][i] + to_double(2, 55) * K[5][i];
}


__device__ void rungeKutta(double t0, double* XY, double tEnd, double s, double tau, double* vector, int n)
{
    while (t0 < tEnd)
    {
        if (tau >= (tEnd - t0)) tau = tEnd - t0;
        increment(t0, XY, tEnd, s, tau, vector, n);
        t0 += tau;
    }
}


__device__ double energy(double* XY, double tEnd, double s, double* vector, int n, double C, double W)
{
    return (pow((XY[1]), 2) + pow(XY[3], 2)) - C * exp(-(pow((XY[0] - P(tEnd, vector, n, tEnd, s)), 2) + pow(XY[2], 2)) / pow(W, 2));
}


__device__ double newton_search(double a, double b, double EPS, double s, double* vector, int n)
{
    double Xnn = 3 * EPS, Xn = b, X = EPS;
    while (fabs(Xnn - X) > EPS) {
        X = Xn;
        Xnn = Xn - ((p(Xn, vector, n) - s) / dp(Xn, vector, n));
        Xn = Xnn;
    };
    return Xnn;
}


__device__ double get_tEnd(double s, double Tmax, double tau, double EPS, double* vector, int n)
{
    double ans, t = 0;
    while (t < Tmax)
    {
        if (tau >= (Tmax - t)) tau = Tmax - t;
        if ((p(t, vector, n) - s / 2.0) * (p(t + tau, vector, n) - s / 2.0) <= 0)
        {
            ans = newton_search(t, t + tau, EPS, s / 2.0, vector, n);
            return 2 * ans;
        }
        if (p(t, vector, n) > p(t + tau, vector, n)) return -1;             //ЧТО С ЛИЦОМ? МБ НАДА?
        t += tau;
    }
    return -1;
}


__device__ double optimization_f(double s, double* E, double* vector, int n)
{
    double tEnd = 0, Tmax = 0.07218 * s + 1.46378, tau = 0.01, EPS = 1e-06, lim = 0.88664120885; ////////////////////LIM = -35 !!!!
    tEnd = get_tEnd(s, Tmax, tau, EPS, vector, n);
    if (tEnd != -1)
    {
        double XY[4] = { 0 }, t0 = 0, C = 39.47841760435742, W = 1;
        rungeKutta(t0, XY, tEnd, s, tau, vector, n);
        *E = energy(XY, tEnd, s, vector, n, C, W);
        if (*E < (-lim * C)) return tEnd;
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
    double XY[4] = { 0 }, Ep, t0 = 0, tau = 0.01, C = 39.47841760435742, W = 1;
    double successes = 0;
    *avarage_E = 0;
    for (int i = 0; i < init_N; i++)
    {
        for (int j = 0; j < 4; j++) XY[j] = initials[i * 4 + j];
        rungeKutta(t0, XY, tEnd, s, tau, vector, n);
        Ep = energy(XY, tEnd, s, vector, n, C, W);
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
    int Np = 4 * init_population * sizeof(double);
    double* data_init = (double*)malloc(Np);
    for (int i = 0; i < 4 * init_population; i++) fscanf(file, "%lf", &data_init[i]);
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
    distributor_P << <40960, 256 >> > (S, inputDataOnDevice, inputInitialDataOnDevice, resultOnDevice, init_population, n, N);

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


int grid_method(double S, double* init_vector)
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
    int size = 10000 * population; //текущий размер массива на случай если он начнет переполняться               //ОНО ТАМ ВНИЗУ НЕ БУДЕТ РАБОТАТЬ ПЗДЦ
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
    //счетчик популяций
    int era = 0;

    //цикл поиска всех точек фигуры
    while (target_grid_cnt)
    {
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

            //запуск ядра  2097152
            distributor << <40960, 256 >> > (S, inputDataOnDevice, resultOnDevice, vec, population);

            //копирование результатов на хост и привязка к указателю тут resultOnHost
            cudaMemcpy(resultOnHost, resultOnDevice, Nb, cudaMemcpyDeviceToHost);
            //ядро выполнило работу


            //смотрим результаты проверки 
            for (int j = 0; j < population; j++) if (resultOnHost[2 * j] != -1) alive_points++;
            printf("Era %d \t Grid-point %d / %d\t%d alive points around\n", era + 1, q, target_grid_cnt - 1, alive_points);
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
                    if (size - 100 < checked_grid_cnt)///ЭТО НЕ РАБОТАЕТ 
                    {
                        printf("Space is over sorry bro :(\n");
                        exit(2);
                    //    printf("\n\n\n%d\t%d\n\n\n", size - 100, checked_grid_cnt);
                    //    size += population;
                    //    checked_grid = (double**)realloc(checked_grid, size * sizeof(*checked_grid));
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
        era++;
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
    double init_vector[] = { 0.691200, 38.850753, -168.382466, 176.389313, 0, 0 }; //лучшее для 3.0 и 35К
    int alive_cnt;
    double bestie[vec + 4];
    double S = 3.0;

    FILE* file;
    file = fopen("results.txt", "w");
    fclose(file);

    alive_cnt = grid_method(S, init_vector);
    calculate_p_e(S, alive_cnt);
    get_best_point(alive_cnt, bestie);

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

    return 0;
}
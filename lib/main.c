#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <pthread.h>

#define NUM_THREADS 4
#define MSS_SIZE 2 // cantidad de puntos a seleccionar
#define THRESHOLD 0.01

float** observations; // Dynamic array to store observations
int num_obs = 0; // Number of observations

struct ThreadArgs {
    int start_iteration;
    int end_iteration;
    float max_inliers;
    float best_m;
    float best_c;
};

const char* getfield(char* line, int num)
{
    const char* tok;
    for (tok = strtok(line, ",");
            tok && *tok;
            tok = strtok(NULL, ",\n"))
    {
        if (!--num)
            return tok;
    }
    return NULL;
}

void readObservations(const char* input_file) {
    FILE* fp;
    if (!(fp = fopen(input_file, "r"))) {
        printf("Cannot open file\n");
        exit(1);
    }

    char line[1024];
    while (fgets(line, 1024, fp)) {
        char* tmp = strdup(line);
        const char* field2 = getfield(tmp, 2);
        const char* field1 = getfield(tmp, 1);

        float y = atof(field2);
        float x = atof(field1);
        // printf("%f - %f \n", x, y);
        free(tmp);

        // Allocate memory for a new observation
        float* observation = (float*)malloc(2 * sizeof(float));
        observation[0] = x;
        observation[1] = y;

        // Add the observation to the array
        observations = (float**)realloc(observations, (num_obs + 1) * sizeof(float*));
        observations[num_obs] = observation;

        num_obs++;
    }

    fclose(fp);
}

void fitModel(float mss_points[MSS_SIZE][2], float* m, float* c) {
    // Fit a model using the MSS
    *m = (mss_points[1][1] - mss_points[0][1]) / (mss_points[1][0] - mss_points[0][0]);
    *c = mss_points[0][1] - (*m * mss_points[0][0]);
}

int countInliers(float m, float c) {
    int num_inliers = 0;

    for (int i = 0; i < num_obs; i++) {
        float epsilon = (observations[i][1] - m * observations[i][0] - c) * (observations[i][1] - m * observations[i][0] - c);
        if (epsilon < THRESHOLD) {
            num_inliers++;
        }
    }

    return num_inliers;
}

// Función ejecutada por cada hilo
void* fitModelAndCountInliersThread(void* arg) {
    struct ThreadArgs* args = (struct ThreadArgs*)arg;

    for (int iteration = args->start_iteration; iteration < args->end_iteration; iteration++) {
        float mss_points[MSS_SIZE][2];
        int rand_index = rand() % num_obs;

        mss_points[0][0] = observations[rand_index][0];
        mss_points[0][1] = observations[rand_index][1];

        rand_index = rand() % num_obs;
        while (observations[rand_index][0] == mss_points[0][0]) {
            rand_index = rand() % num_obs;
        }

        mss_points[1][0] = observations[rand_index][0];
        mss_points[1][1] = observations[rand_index][1];

        float m, c;
        fitModel(mss_points, &m, &c);

        int num_inliers = countInliers(m, c);

        

        if (num_inliers > args->max_inliers) {
            args->max_inliers = num_inliers;
            args->best_m = m;
            args->best_c = c;
            printf("Iteration %d - Number of Inliers: %d, Best parameters values: m = %f, c = %f\n", iteration + 1, num_inliers, m, c);
        }
    }

    return NULL;
}

void fitModelAndCountInliers(int num_iterations, int num_threads) {
    srand(time(NULL));

    float max_inliers = 0.0f;
    float best_m = 0.0f;
    float best_c = 0.0f;

    // Crear hilos
    pthread_t threads[num_threads];
    struct ThreadArgs thread_args[num_threads];
    int iterations_per_thread = num_iterations / num_threads;
    int remaining_iterations = num_iterations % num_threads;

    for (int i = 0; i < num_threads; i++) {
        thread_args[i].start_iteration = i * iterations_per_thread;
        thread_args[i].end_iteration = (i + 1) * iterations_per_thread;
        thread_args[i].max_inliers = 0.0f;
        thread_args[i].best_m = 0.0f;
        thread_args[i].best_c = 0.0f;

        // El último hilo se hace cargo de las iteraciones restantes si no se dividen uniformemente
        if (i == num_threads - 1) {
            thread_args[i].end_iteration += remaining_iterations;
        }

        pthread_create(&threads[i], NULL, fitModelAndCountInliersThread, (void*)&thread_args[i]);
    }

    // Esperar a que todos los hilos terminen
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);

        if (thread_args[i].max_inliers > max_inliers) {
            max_inliers = thread_args[i].max_inliers;
            best_m = thread_args[i].best_m;
            best_c = thread_args[i].best_c;
        }
    }

    float outlier_ratio = (float)(num_obs - max_inliers) / num_obs;
    float inlier_ratio = (float)max_inliers / num_obs;


    printf("Outlier Ratio: %.2f%%\n", outlier_ratio * 100);
    printf("Inlier Ratio: %.2f%%\n", inlier_ratio * 100);
    printf("Number of Inliers: %f\n", max_inliers);
    printf("Best Parameters: m = %f, c = %f\n", best_m, best_c);
}

void cleanup() {
    // Free memory allocated for observations
    for (int i = 0; i < num_obs; i++) {
        free(observations[i]);
    }
    free(observations);
}


int main(int argc, char* argv[]) {
    if (argc < 2 || argc > 3) {
        printf("Usage: %s <num_iterations> [num_threads]\n", argv[0]);
        return 1;
    }
    printf("Bienvenido al programa de RANSAC\n");

    int num_iterations = atoi(argv[1]);
    if (num_iterations <= 0) {
        printf("Invalid number of iterations\n");
        return 1;
    }

    int num_threads = NUM_THREADS; // Valor predeterminado
    if (argc == 3) {
        num_threads = atoi(argv[2]);
        if (num_threads <= 0) {
            printf("Invalid number of threads\n");
            return 1;
        }
    }
    // int num_iterations = 20000;
    // int num_threads = NUM_THREADS; // Valor predeterminado

    printf("####### CARGANDO DATASET #######\n");
    const char* input_file = "../resources/boston.csv";
    readObservations(input_file);

    printf("####### INICIO #######\n");

    fitModelAndCountInliers(num_iterations, num_threads);
    
    printf("####### FIN #######\n");

    cleanup();

    return 0;
}


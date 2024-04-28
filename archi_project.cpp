#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace std::chrono;

int n = 48;

const int MAX_ITERATIONS = 200;
const int NUM_GLOWWORMS = 50;
const double GAMMA = 0.6;

vector<vector<double>> distanceMatrix; // Distance matrix representing distances between cities

struct Glowworm
{
    vector<int> tour;
    double brightness;
};

double calculateDistance(vector<int> &tour)
{
    double totalDistance = 0.0;
    for (int i = 0; i < tour.size() - 1; ++i)
    {
        totalDistance += distanceMatrix[tour[i]][tour[i + 1]];
    }
    totalDistance += distanceMatrix[tour.back()][tour[0]]; // Return to starting city
    return totalDistance;
}

void initializeGlowworms(vector<Glowworm> &glowworms)
{
    for (int i = 0; i < NUM_GLOWWORMS; ++i)
    {
        Glowworm newGlowworm;
        newGlowworm.tour.resize(distanceMatrix.size());

        // Initialize tour with cities in order
        for (int j = 0; j < newGlowworm.tour.size(); ++j)
        {
            newGlowworm.tour[j] = j;
        }

        random_shuffle(newGlowworm.tour.begin() + 1, newGlowworm.tour.end()); // Shuffle cities except starting city
        newGlowworm.brightness = 1.0 / calculateDistance(newGlowworm.tour);
        glowworms.push_back(newGlowworm);
    }
}

void swapTwoCities(vector<int> &tour, int city1, int city2)
{
    int temp = tour[city1];
    tour[city1] = tour[city2];
    tour[city2] = temp;
}

void glowwormSwarmOptimization()
{
    vector<Glowworm> glowworms;
    initializeGlowworms(glowworms);

    int num_threads = omp_get_max_threads();
    num_threads = 1;
    for (; num_threads <= 8; num_threads++)
    {
        cout << "Number of Threads: " << num_threads << endl;

        auto start = high_resolution_clock::now();

        for (int iter = 0; iter < MAX_ITERATIONS; ++iter)
        {
            // Update glowworm positions and brightness
            // Glowworm Movement and Local Search
#pragma omp parallel for num_threads(num_threads)
            for (int i = 0; i < NUM_GLOWWORMS; ++i)
            {
                // Glowworm Movement
                double totalBrightness = 0.0;
                for (const auto &gw : glowworms)
                {
                    totalBrightness += gw.brightness;
                }

                double decisionRange = GAMMA * glowworms[i].brightness / totalBrightness;
                vector<int> neighborIndices;
                for (int j = 0; j < NUM_GLOWWORMS; ++j)
                {
                    if (i != j && calculateDistance(glowworms[j].tour) < calculateDistance(glowworms[i].tour) + decisionRange)
                    {
                        neighborIndices.push_back(j);
                    }
                }

                if (!neighborIndices.empty())
                {
                    int selectedNeighborIndex = neighborIndices[rand() % neighborIndices.size()];
                    int city1 = rand() % (glowworms[i].tour.size() - 1) + 1; // Exclude the starting city
                    int city2 = rand() % (glowworms[selectedNeighborIndex].tour.size() - 1) + 1;
                    swapTwoCities(glowworms[i].tour, city1, city2);
                    glowworms[i].brightness = 1.0 / calculateDistance(glowworms[i].tour);
                }

                // Local Search (2-opt)
                vector<int> bestTour = glowworms[i].tour;
                double bestDistance = calculateDistance(bestTour);
                for (int j = 0; j < bestTour.size() - 1; ++j)
                {
                    for (int k = j + 1; k < bestTour.size(); ++k)
                    {
                        vector<int> newTour = bestTour;
                        reverse(newTour.begin() + j, newTour.begin() + k + 1);
                        double newDistance = calculateDistance(newTour);
                        if (newDistance < bestDistance)
                        {
                            bestTour = newTour;
                            bestDistance = newDistance;
                        }
                    }
                }
                glowworms[i].tour = bestTour;
                glowworms[i].brightness = 1.0 / bestDistance;
            }

            // Sort glowworms based on brightness
            sort(glowworms.begin(), glowworms.end(), [](const Glowworm &a, const Glowworm &b)
                 { return a.brightness > b.brightness; });

            // Update pheromone levels
            for (int i = 0; i < NUM_GLOWWORMS; ++i)
            {
                glowworms[i].brightness = 1.0 / calculateDistance(glowworms[i].tour);
            }
        }

        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start).count();
        cout << "Parallel Approach Time: " << duration << " ms" << endl;

        // Get the best solution
        Glowworm bestGlowworm = *max_element(glowworms.begin(), glowworms.end(), [](const Glowworm &a, const Glowworm &b)
                                             { return a.brightness < b.brightness; });

        // Output the best tour and distance
        cout << "Best Tour: ";
        for (int city : bestGlowworm.tour)
        {
            cout << city << " ";
        }
        cout << "\nBest Distance: " << calculateDistance(bestGlowworm.tour) << endl;
        cout << endl;
    }
}

int main()
{
    // Initialize distance matrix with actual distances between cities
    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);
    // #endif
    distanceMatrix.resize(n, vector<double>(n));
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cin >> distanceMatrix[i][j];
        }
    }
    // Call the Glowworm Swarm Optimization algorithm
    glowwormSwarmOptimization();

    return 0;
}
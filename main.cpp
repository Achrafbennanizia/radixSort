#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_scan.h>
#include <tbb/task_arena.h>
#include <tbb/spin_mutex.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>

using u64 = std::uint64_t;

int getMax(std::vector<int> vec, int n)
{
    int mx = vec[0];
    for (int i = 0; i < n; i++)
        if (vec[i] > mx)
            mx = vec[i];
    return mx;
}

int getnumDigits(int number)
{
    int digits = 0;
    while (number != 0)
    {
        number /= 10;
        digits++;
    }
    return digits;
}

static void CountingSort(std::vector<int> vec, int n, int exp)
{
    // Output array
    int output[n];
    int i, count[10] = {0};

    // Store count of occurrences
    // in count[]
    for (i = 0; i < n; i++)
        count[(vec[i] / exp) % 10]++;

    // Change count[i] so that count[i]
    // now contains actual position
    // of this digit in output[]
    for (i = 1; i < 10; i++)
        count[i] += count[i - 1];

    // Build the output array
    for (i = n - 1; i >= 0; i--)
    {
        output[count[(vec[i] / exp) % 10] - 1] = vec[i];
        count[(vec[i] / exp) % 10]--;
    }

    // Copy the output array to arr[],
    // so that arr[] now contains sorted
    // numbers according to current digit
    for (i = 0; i < n; i++)
        vec[i] = output[i];
}

void radixsort(std::vector<int> vec, int n)
{

    // Find the maximum number to
    // know number of digits
    int m = getMax(vec, n);

    // Do counting sort for every digit.
    // Note that instead of passing digit
    // number, exp is passed. exp is 10^i
    // where i is current digit number
    for (int exp = 1; m / exp > 0; exp *= 10)
        CountingSort(vec, n, exp);
}

static std::vector<int> sortVectorTBB(const std::vector<int> &vec)
{
    if (vec.empty())
        return vec;

    std::vector<int> sortedVec = vec;

    // Finde Maximum parallel
    u64 mx = tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0, sortedVec.size()),
        0ULL,
        [&](const tbb::blocked_range<size_t> &r, u64 local)
        {
            for (size_t i = r.begin(); i != r.end(); ++i)
                local = std::max(local, static_cast<u64>(sortedVec[i]));
            return local;
        },
        [](u64 a, u64 b)
        { return std::max(a, b); });

    int digitNumber = getnumDigits(mx);
    std::vector<int> output(sortedVec.size());

    for (int digit = 0; digit < digitNumber; ++digit)
    {
        int exp = static_cast<int>(pow(10, digit));
        const int RADIX = 10;

        // Parallele Histogram-Erstellung mit Reduction
        std::vector<int> count = tbb::parallel_reduce(
            tbb::blocked_range<size_t>(0, sortedVec.size()),
            std::vector<int>(RADIX, 0),
            [&](const tbb::blocked_range<size_t> &r, std::vector<int> localCount)
            {
                for (size_t i = r.begin(); i != r.end(); ++i)
                {
                    int digit_val = (sortedVec[i] / exp) % RADIX;
                    localCount[digit_val]++;
                }
                return localCount;
            },
            [](std::vector<int> a, const std::vector<int> &b)
            {
                for (size_t i = 0; i < a.size(); ++i)
                    a[i] += b[i];
                return a;
            });

        // Prefix Sum (seriell, nur 10 Elemente)
        for (int i = 1; i < RADIX; ++i)
            count[i] += count[i - 1];

        // Verteile rückwärts (für Stabilität) - SERIELL
        for (int i = static_cast<int>(sortedVec.size()) - 1; i >= 0; --i)
        {
            int digit_val = (sortedVec[i] / exp) % RADIX;
            output[count[digit_val] - 1] = sortedVec[i];
            count[digit_val]--;
        }

        sortedVec = output;
    }

    return sortedVec;
}

static std::vector<int> sortVectorSeq(const std::vector<int> &vec)
{
    std::vector<int> sortedVec = vec; // Create a copy to sort

    // std::sort(sortedVec.begin(), sortedVec.end());
    radixsort(sortedVec, sortedVec.size());

    return sortedVec;
}

std::vector<int> ramdomVector(int size)
{
    std::vector<int> randomVec;
    for (int i = 0; i < size; ++i)
    {
        randomVec.push_back(rand() % 1000); // Random numbers between 0 and 999
    }
    return randomVec;
}

struct Timed
{
    double seconds{};
};

static Timed Sort_seq(const std::vector<int> unsortVector)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    sortVectorSeq(unsortVector);
    auto t1 = std::chrono::high_resolution_clock::now();
    return {std::chrono::duration<double>(t1 - t0).count()};
}
static Timed Sort_par(const std::vector<int> unsortVector)
{
    auto t2 = std::chrono::high_resolution_clock::now();
    sortVectorTBB(unsortVector);
    auto t3 = std::chrono::high_resolution_clock::now();
    return {std::chrono::duration<double>(t3 - t2).count()};
}

int main(int argc, char **argv)
{
    try
    {
        int size = 1000000;
        std::vector<int> unsortVector = ramdomVector(size);

        auto seq = Sort_seq(unsortVector);
        auto par = Sort_par(unsortVector);

        const unsigned hw = tbb::this_task_arena::max_concurrency();

        const double speedup = (par.seconds > 0.0) ? (seq.seconds / par.seconds) : 0.0;
        const double efficiency = (hw > 0) ? (speedup / hw) : 0.0;

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "\n┌─────────────────────────────────────┐\n";
        std::cout << "│      Performance Analysis           │\n";
        std::cout << "├─────────────────────────────────────┤\n";
        std::cout << "│ Vector size:    " << std::setw(12) << size << "        │\n";
        std::cout << "│ Threads:        " << std::setw(12) << hw << "        │\n";
        std::cout << "├─────────────────────────────────────┤\n";
        std::cout << "│ Sequential:     " << std::setw(12) << seq.seconds << " s      │\n";
        std::cout << "│ Parallel:       " << std::setw(12) << par.seconds << " s      │\n";
        std::cout << "├─────────────────────────────────────┤\n";
        std::cout << "│ Speedup:        " << std::setw(12) << speedup << "x       │\n";
        std::cout << "│ Efficiency:     " << std::setw(12) << (efficiency * 100) << "%       │\n";
        std::cout << "└─────────────────────────────────────┘\n\n";
    }
    catch (const std::exception &ex)
    {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
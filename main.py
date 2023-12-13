import numpy as np
import random
import operator
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import shutil
import os
import imageio


from PIL import Image


class City:
    def __init__(self, name, x, y):
        self.name = name
        self.x = x
        self.y = y

    def distance(self, city):
        x_dis = abs(self.x - city.x)
        y_dis = abs(self.y - city.y)
        distance = np.sqrt((x_dis ** 2) + (y_dis ** 2))
        return distance

    def __repr__(self):
        return f"{self.name} - ({self.x}, {self.y})"

def convert_latitude_to_y(latitude):
    return (latitude + 90) * 10

def convert_longitude_to_x(longitude):
    return (longitude + 180) * 10

def createRoute(cityList):
    route = random.sample(cityList, len(cityList))
    return route

class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0

    def routeDistance(self):
        if self.distance == 0:
            pathDistance = 0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance

    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness

def initialPopulation(popSize, cityList):
    population = []
    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population

def rankRoutes(population):
    fitnessResults = {}
    for i in range(0, len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    return sorted(fitnessResults.items(), key=operator.itemgetter(1), reverse=True)

def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index", "Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()

    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100 * random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i, 3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults

def tournamentSelection(popRanked, eliteSize, tournamentSize):
    selectionResults = []

    # Select elite individuals directly
    for i in range(eliteSize):
        selectionResults.append(popRanked[i][0])

    # Run tournaments to fill the rest of the selection pool
    for i in range(len(popRanked) - eliteSize):
        tournamentContestants = random.sample(popRanked, tournamentSize)
        winner = max(tournamentContestants, key=lambda x: x[1])
        selectionResults.append(winner[0])

    return selectionResults

def elitismSelection(popRanked, eliteSize):
    selectionResults = []

    # rank best individuals
    popRanked = sorted(popRanked, key=lambda x: x[1], reverse=True)

    # Select elite individuals directly
    for i in range(eliteSize):
        selectionResults.append(popRanked[i][0])

    return selectionResults

def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool

def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent2))
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)
    for i in range(startGene, endGene):
        childP1.append(parent1[i])
    childP2 = [item for item in parent2 if item not in childP1]
    child = childP1 + childP2
    return child

def breedPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0, eliteSize):
        children.append(matingpool[i])
    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool) - i - 1])
        children.append(child)
    return children

def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if random.random() < mutationRate:
            swapWith = int(random.random() * len(individual))
            city1 = individual[swapped]
            city2 = individual[swapWith]
            individual[swapped] = city2
            individual[swapWith] = city1
    return individual

def mutatePopulation(population, mutationRate):
    mutatedPop = []
    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop

def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen)
    
    # selectionResults = selection(popRanked, eliteSize)
    
    tournamentSize = 10 
    selectionResults = tournamentSelection(popRanked, eliteSize, tournamentSize)
    
    # selectionResults = elitismSelection(popRanked, eliteSize)

    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration

# Update the plotRoute function to save the plot to an image in the specified folder
def plotRoute(bestRoute, generation, save_folder):
    x_values = [city.x for city in bestRoute]
    y_values = [city.y for city in bestRoute]

    plt.figure(figsize=(8, 6))
    plt.plot(x_values + [bestRoute[0].x], y_values + [bestRoute[0].y], marker='o')
    plt.title(f"Best Route => Generation: {generation} - Distance: {Fitness(bestRoute).routeDistance():.2f}")
    plt.xlabel("X")
    plt.ylabel("Y")

    for city in bestRoute:
        plt.annotate(city.name, (city.x, city.y), textcoords="offset points", xytext=(0, 5), ha='center')

    # Save the plot to an image file in the specified folder
    plt.savefig(os.path.join(save_folder, f"generation_{generation}.png"))
    plt.close()

def createGIF(image_folder, gif_name):
    images = []
    for filename in sorted(os.listdir(image_folder), key=lambda x: int(x.split('_')[1].split('.')[0])):
        if filename.endswith(".png"):
            images.append(Image.open(os.path.join(image_folder, filename)))

    # Save the GIF
    images[0].save(gif_name, save_all=True, append_images=images[1:], loop=0, duration=500)  # duration is in milliseconds


def create_video(image_folder, video_name, fps=24):
    images = []
    for filename in sorted(os.listdir(image_folder), key=lambda x: int(x.split('_')[1].split('.')[0])):
        if filename.endswith(".png"):
            images.append(imageio.imread(os.path.join(image_folder, filename)))

    # Save the video
    video_path = os.path.join(image_folder, video_name)
    imageio.mimsave(video_path, images, fps=fps)

    return video_path


    
def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))

    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    print("Initial best route:")
    print(bestRoute)
    
    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
        bestRouteIndex = rankRoutes(pop)[0][0]
        currentBestRoute = pop[bestRouteIndex]
        
        # Print only when there is an improvement
        if 1 / rankRoutes(pop)[0][1] < 1 / rankRoutes([bestRoute])[0][1]:
            print("rankRoutes(pop)[0][1]", rankRoutes(pop)[0][1])
            print(f"Generation {i + 1} - New best distance: {1 / rankRoutes(pop)[0][1]}")
            print(currentBestRoute)
            print()
            bestRoute = currentBestRoute

            save_folder = "foto_perkembangan_rute"
            plotRoute(bestRoute, i, save_folder)


    print("Final distance: " + str(1 / rankRoutes(pop)[0][1]))
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    return bestRoute


# Daftar 10 kota di Indonesia dengan lokasi latitude dan longitude yang dikonversi
city_data = {
    "Jakarta": (-6.2088, 106.8456),
    "Surabaya": (-7.2575, 112.7521),
    "Bandung": (-6.2088, 107.6029),
    "Medan": (3.5952, 98.6722),
    "Semarang": (-6.2088, 106.8456),
    "Makassar": (-5.1477, 119.4325),
    "Palembang": (-2.9888, 104.7569),
    "Balikpapan": (-1.2675, 116.8289),
    "Manado": (1.5016, 124.8440),
    "Denpasar": (-8.6705, 115.2126),
    "Bandar Lampung": (-5.429, 105.266),
    "Padang": (-0.95, 100.353),
    "Yogyakarta": (-7.801194, 110.364917),
    "Malang": (-7.977978, 112.561952),
    "Banjarmasin": (-3.318889, 114.591667),
    "Pekanbaru": (0.5091, 101.4477),
    "Tangerang": (-6.1786, 106.6297),
    "Surakarta": (-7.5582, 110.8318),
    "Cirebon": (-6.7321, 108.5519),
    "Tasikmalaya": (-7.3274, 108.2208),
    "Serang": (-6.1104, 106.1499),
    "Banjarbaru": (-3.4169, 114.5943),
    "Pontianak": (0.0227, 109.3425),
    "Manokwari": (-0.8615, 134.0620),
    "Ambon": (-3.6954, 128.1810),
    "Jayapura": (-2.5489, 140.7167),
    "Padang Panjang": (-0.4700, 100.4172),
    "Banda Aceh": (5.5480, 95.3191),
    "Mamuju": (-2.6774, 118.8770),
    "Samarinda": (-0.5025, 117.1534),
    "Mataram": (-8.5833, 116.1167),
    "Kupang": (-10.1787, 123.6070),
    "Sorong": (-0.8627, 131.2480),
    "Palangka Raya": (-2.2067, 113.9170),
    "Merauke": (-8.4932, 140.4012),
    "Ternate": (0.7924, 127.3630),
    "Langsa": (4.4682, 97.9681),
    "Sibolga": (1.7387, 98.7891),
    "Tidore": (0.6966, 127.4294),
    "Bontang": (0.1324, 117.4713),
    "Bitung": (1.4522, 125.1861),
}


cityList = [City(name, convert_longitude_to_x(longitude), convert_latitude_to_y(latitude)) for name, (latitude, longitude) in city_data.items()]

# Tes cityList
for city in cityList:
    print(city)

def main():
    save_folder = "foto_perkembangan_rute"
    # Delete all files in the save_folder
    if os.path.exists(save_folder):
        shutil.rmtree(save_folder)
    os.makedirs(save_folder, exist_ok=True)

    bestRoute = geneticAlgorithm(population=cityList, popSize=150, eliteSize=1, mutationRate=0.01, generations=500)

    createGIF(save_folder, "route_evolution.gif")
    # Example usage
    image_folder = "foto_perkembangan_rute"
    video_name = "video_perkembang_rute.mp4"
    create_video(image_folder, video_name)

if __name__ == '__main__':

    main()
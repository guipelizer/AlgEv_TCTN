import time
import random
import numpy as np
import matplotlib.pyplot as plt

from deap import base
from deap import creator
from deap import tools

from GNTN_sparse import *

import multiprocessing
import os

''' Parâmetros do algoritmo evolutivo
numExecutions -> número de execuções do algoritmo evolutivo
n_generations -> número máximo de gerações
n_pop -> número de indivíduos na população
tournamentSize -> tamanho do torneio para seleção dos pais
MUTPB -> probabilidade de mutação de um indivíduo
CROSSPB -> probabilidade de cruzamento dos pais
n_jobs -> número de núcleos que serão usados no paralelismo
'''
numExecutions = 5
n_generations = 50
n_pop = 20
n_folds = 10
tournamentSize = 3
MUTPB = 0.2
CROSSPB = 0.85
n_jobs = 4

def evalProblem(individual, datapath, lbl_examples, dataset_name):
    '''
        Função de aptidão -> média do F1 Weighted de cada fold
    '''
        k = int(individual[0])
        alpha = individual[1]

        result = np.zeros(n_folds)
        for id_data in range(1, n_folds+1, 1):
            result_fold = tctn(datapath, lbl_examples, id_data, dataset_name, None, alpha, 1000, k, piatetsky_shapiro)
            result[id_data - 1] = result_fold[3]

        fitness = result.mean(axis=0)
        return (fitness,)

def alg_ev(datapath, lbl_examples, dataset_name):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    # Definição do indivíduo
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    pool = multiprocessing.Pool(n_jobs)
    toolbox.register("map", pool.map)

    BOUND_LOW = (1, 0.01)
    BOUND_UP = (60, 1)
    sigma = ((BOUND_UP[0] - BOUND_LOW[0]) * 0.05, (BOUND_UP[1] - BOUND_LOW[1]) * 0.1)
    mu = (0, 0)
    indpb = 1

    def clipIndividual(individual, low, up):
        '''
            Função para manter os indivíduos dentro dos limites do problema
        '''
        individual[0] = int(np.clip(individual[0], low[0], up[0]))
        individual[1] = np.clip(individual[1], low[1], up[1])

    def create_attrs(low, up, size=None):
        '''
            Inicialização aleatória uniforme da população dentro dos limites do problema
        '''
        attrs = [np.random.randint(low[0],up[0]), np.random.rand()]
        clipIndividual(attrs, low, up)
        return attrs

    def cxMate(ind1, ind2, low, up, alpha):
        '''
            Procedimento de cruzamento de dois individuos
        '''
        tools.cxBlend(ind1, ind2, alpha)
        clipIndividual(ind1, low, up)
        clipIndividual(ind2, low, up)


    def mut(individual, low, up, mu, sigma, indpb):
        '''
            Procedimento de mutação de um indivíduo
        '''
        tools.mutGaussian(individual, mu, sigma, indpb)
        clipIndividual(individual, low, up)

    # Gerador de atributos dos indivíduos
    toolbox.register("attrs", create_attrs, BOUND_LOW, BOUND_UP)
    # Inicializador dos indivíduos
    toolbox.register("individual", tools.initIterate, creator.Individual,
                    toolbox.attrs)
    #Inicializador da população
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Registro da função de cruzamento na toolbox
    toolbox.register("mate", cxMate, low=BOUND_LOW, up=BOUND_UP, alpha=0.5)
    # Registro da função de mutação na toolbox
    toolbox.register("mutate", mut, low=BOUND_LOW, up=BOUND_UP, mu=mu, sigma=sigma, indpb=indpb)
    # Registro da função de seleção por torneio na toolbox
    toolbox.register("select", tools.selTournament, tournsize=tournamentSize)
    # Registro da função de seleção dos melhores indivíduos na toolbox
    toolbox.register("selectBest", tools.selBest, fit_attr='fitness')
    # Registro da função de aptidão na toolbox
    toolbox.register("evaluate", evalProblem, datapath=datapath, lbl_examples=lbl_examples, dataset_name=dataset_name)

    # Definição das estatísticas que serão salvas
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    
    # Definição do logbook
    algorithm_logbook = tools.Logbook()

    for n_exec in range(numExecutions):
        print('----------- Start {} execution ----------'.format(n_exec))
        start = time.time()

        logbook = tools.Logbook()
        hof = tools.HallOfFame(1, similar=np.array_equal)

        pop = toolbox.population(n=n_pop)

        fitness = list(toolbox.map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitness):
            ind.fitness.values = fit
        
        fits = [ind.fitness.values[0] for ind in pop]
        g = 0
        best_individual_per_generation = []

        record = stats.compile(pop)
        print(g, record)
        hof.update(pop)
        print('Best k: ', hof[0], hof[0].fitness.values)
        best_individual_per_generation.append(hof[0])
        same_fitness = 0
        logbook.record(gen=g, **record)

        while g < n_generations and same_fitness < 5:  
            g = g + 1 

            offspring = list()
            for i in range(int(n_pop / 2) - 1):
                parent1, parent2 = toolbox.select(pop, 2)
                # criação de clones pois a toolbox irá modificar os indivíduos in-place
                child1 = toolbox.clone(parent1)
                child2 = toolbox.clone(parent2)
                if random.random() < CROSSPB:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
                offspring.append(child1)
                offspring.append(child2)    

            for mutant in offspring:
                if random.random() < MUTPB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            pop = toolbox.selectBest(pop, 2) + offspring
            #print(pop)
            k_vec = np.zeros(n_pop)
            alpha_vec = np.zeros(n_pop)
            for i, ind in enumerate(pop):
                k_vec[i] = ind[0]
                alpha_vec[i] = ind[1]

            record = stats.compile(pop)
            print('------')
            print(g, record)
            print("K -\t\t avg: {0:.4f} \t std: {1:.4f} \t max: {2:.4f} \t min: {3:.4f}".format(k_vec.mean(), k_vec.std(), k_vec.max(), k_vec.min()))
            print("alpha -\t avg: {0:.4f} \t std: {1:.4f} \t max: {2:.4f} \t min: {3:.4f}".format(alpha_vec.mean(), alpha_vec.std(), alpha_vec.max(), alpha_vec.min()))
            hof.update(pop)
            print('Best k: ', hof[0], hof[0].fitness.values)
            best_clone = toolbox.clone(hof[0])
            logbook.record(gen=g, **record, best=best_clone, best_fitness=best_clone.fitness.values)


            best_individual_per_generation.append(hof[0])
            if(best_individual_per_generation[g].fitness.values == best_individual_per_generation[g-1].fitness.values):
                same_fitness = same_fitness + 1
                print("same fitness", same_fitness, " generations")
            else:
                same_fitness = 0


        #logbook.header = "gen", "avg", "max", "min", "std"
        

        end = time.time()
        time_elapsed = end - start
        best_individual = toolbox.clone(hof[0])
        best_fitness = best_individual.fitness.values
        algorithm_logbook.record(n_exec=n_exec, time=time_elapsed, best_individual=best_individual, best_fitness=best_fitness)

        gen = logbook.select('gen')
        fit_avgs = logbook.select('avg')
        fit_stds = logbook.select('std')
        
        if not os.path.exists("./results/"):
            os.makedirs("./results/")


        if not os.path.exists('./results/' + dataset_name + '/'):
                os.makedirs('./results/' + dataset_name + '/')

        if not os.path.exists('./results/' + dataset_name + '/' + str(lbl_examples) + '/'):
                os.makedirs('./results/' + dataset_name + '/' + str(lbl_examples) + '/')


        df = pd.DataFrame(logbook)
        df.to_csv('./results/' + dataset_name + '/' + str(lbl_examples) + '/' + dataset_name + '_' + str(lbl_examples) + '_lblexamples_' + 'exec_' + str(n_exec) + '.csv')

        print(hof[0], toolbox.evaluate(hof[0]))

    if not os.path.exists("./results/"):
        os.makedirs("./results/")

    if not os.path.exists('./results/' + dataset_name + '/'):
        os.makedirs('./results/' + dataset_name + '/')

    if not os.path.exists('./results/' + dataset_name + '/' + str(lbl_examples) + '/'):
        os.makedirs('./results/' + dataset_name + '/' + str(lbl_examples) + '/')


    algorithm_logbook_df = pd.DataFrame(algorithm_logbook)
    algorithm_logbook_df.to_csv('./results/' + dataset_name + '/' + str(lbl_examples) + '/' + dataset_name + '_' + str(lbl_examples) + '_lblexamples_execs_results.csv')


def main():
    data = './data_splited/'
    lbl_examples_list = [1]
    # Definição das bases a serem executadas
    dataset_name_list = ['oh0.wc', 'oh5.wc', 'oh15.wc']

    for lbl_examples in lbl_examples_list:
        for dataset_name in dataset_name_list:
            datapath = data + dataset_name + '/'
            print("=================", "Execution dataset ", dataset_name, " with ", lbl_examples, "examples", "=================")
            if len(os.listdir(datapath + str(lbl_examples) + '/')) == 0:
                print('skipped')
                continue

            alg_ev(datapath, lbl_examples, dataset_name)

if __name__ == '__main__':
    main()

